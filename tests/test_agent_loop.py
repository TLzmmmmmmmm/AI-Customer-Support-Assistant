import copy
import json
import unittest
from collections import deque
from types import MappingProxyType, SimpleNamespace

from agent.executor import ToolExecutor, ToolObservation
from agent.models import AgentDeadlineExceeded
from support_tools import ProductSearchResult, ToolError, ToolErrorCode


BASE_MESSAGES = [
    {"role": "system", "content": "existing RAG trust boundary"},
    {"role": "user", "content": "用户问题"},
]


def text_completion(content, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=content, tool_calls=None),
    )])


def tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def tool_completion(call_id, name, arguments, content=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="tool_calls",
        message=SimpleNamespace(
            content=content,
            tool_calls=[tool_call(call_id, name, arguments)],
        ),
    )])


class FakeCompleteChat:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    def __call__(self, messages, *, tools=None):
        self.calls.append({
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
        })
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


class RecordingDeadline:
    def __init__(self, fail_on=None):
        self.checks = 0
        self.fail_on = fail_on

    def ensure_active(self):
        self.checks += 1
        if self.checks == self.fail_on:
            raise AgentDeadlineExceeded("expired")


class RecordingExecutor:
    def __init__(self, observation=None):
        self.calls = []
        self.observation = observation or ToolObservation(
            content='{"ok":true,"result":{}}',
            success=True,
            reused=False,
            cache_key="key",
        )

    def execute(self, call, successful_observations):
        self.calls.append(call)
        return self.observation


class AgentLoopTests(unittest.TestCase):
    def test_direct_answer_preserves_input_and_adds_controlled_policy(self):
        from agent.loop import AGENT_TOOL_POLICY, AgentLoop

        original = copy.deepcopy(BASE_MESSAGES)
        complete = FakeCompleteChat([text_completion("直接回答")])
        executor = RecordingExecutor()
        deadline = RecordingDeadline()

        result = AgentLoop(
            executor=executor,
            complete_chat=complete,
        ).run(BASE_MESSAGES, deadline=deadline)

        self.assertEqual(result.answer, "直接回答")
        self.assertEqual(BASE_MESSAGES, original)
        self.assertEqual(executor.calls, [])
        self.assertEqual(len(complete.calls), 1)
        sent = complete.calls[0]["messages"]
        self.assertEqual(sent[0], BASE_MESSAGES[0])
        self.assertEqual(
            sent[1],
            {"role": "system", "content": AGENT_TOOL_POLICY},
        )
        self.assertEqual(sent[2], BASE_MESSAGES[1])
        self.assertEqual(len(complete.calls[0]["tools"]), 3)
        self.assertEqual(deadline.checks, 2)

    def test_tool_call_takes_precedence_and_builds_standard_history(self):
        from agent.loop import AgentLoop

        complete = FakeCompleteChat([
            tool_completion(
                "call-1",
                "search_products",
                '{"query":"酒店"}',
                content="不要把这段 action content 返回前端",
            ),
            text_completion("候选产品，请联系专业技术人员确认。"),
        ])
        executor = RecordingExecutor()
        deadline = RecordingDeadline()

        result = AgentLoop(
            executor=executor,
            complete_chat=complete,
        ).run(BASE_MESSAGES, deadline=deadline)

        self.assertEqual(result.answer, "候选产品，请联系专业技术人员确认。")
        self.assertNotIn("action content", result.answer)
        self.assertEqual(len(executor.calls), 1)
        second_messages = complete.calls[1]["messages"]
        self.assertEqual(
            second_messages[-2],
            {
                "role": "assistant",
                "content": "不要把这段 action content 返回前端",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "search_products",
                        "arguments": '{"query":"酒店"}',
                    },
                }],
            },
        )
        self.assertEqual(
            second_messages[-1],
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": '{"ok":true,"result":{}}',
            },
        )
        self.assertEqual(deadline.checks, 6)

    def test_policy_sequences_do_not_automatically_append_contact_tool(self):
        from agent.loop import AGENT_TOOL_POLICY, AgentLoop

        scenarios = (
            (
                "search_products",
                '{"query":"推荐适合酒店使用的产品"}',
                "候选型号如下，请联系专业技术人员确认最终选型。",
            ),
            (
                "search_products",
                '{"query":"推荐不存在场景的产品"}',
                "没有找到可靠候选，请联系专业技术人员进一步确认。",
            ),
            (
                "get_product_details",
                '{"product_id":"HP780"}',
                "HP780 的防护等级为 IP68。",
            ),
            (
                "get_contact_info",
                "{}",
                "您可以通过官方电话或邮箱联系我们。",
            ),
        )
        for name, arguments, answer in scenarios:
            with self.subTest(name=name, arguments=arguments):
                complete = FakeCompleteChat([
                    tool_completion("call-1", name, arguments),
                    text_completion(answer),
                ])
                executor = RecordingExecutor()

                result = AgentLoop(
                    executor=executor,
                    complete_chat=complete,
                ).run(BASE_MESSAGES, deadline=RecordingDeadline())

                self.assertEqual(result.answer, answer)
                self.assertEqual([call.name for call in executor.calls], [name])

        first_policy = complete.calls[0]["messages"][1]["content"]
        self.assertEqual(first_policy, AGENT_TOOL_POLICY)
        self.assertIn("require search_products", first_policy)
        self.assertIn("require get_product_details", first_policy)
        self.assertIn("require get_contact_info", first_policy)
        self.assertIn("does not require get_contact_info", first_policy)
        self.assertIn("professional technical staff", first_policy)

    def test_tool_error_is_observed_and_model_can_correct_arguments(self):
        from agent.loop import AgentLoop

        calls = []

        def details(product_id):
            calls.append(product_id)
            if product_id == "UNKNOWN":
                raise ToolError(
                    code=ToolErrorCode.PRODUCT_NOT_FOUND,
                    message="Product not found.",
                    tool_name="get_product_details",
                )
            return ProductSearchResult(products=[])

        executor = ToolExecutor(MappingProxyType({
            "get_product_details": details,
        }))
        complete = FakeCompleteChat([
            tool_completion(
                "call-1",
                "get_product_details",
                '{"product_id":"UNKNOWN"}',
            ),
            tool_completion(
                "call-2",
                "get_product_details",
                '{"product_id":"HP780"}',
            ),
            text_completion("无法可靠提供该型号详情。"),
        ])

        result = AgentLoop(
            executor=executor,
            complete_chat=complete,
        ).run(BASE_MESSAGES, deadline=RecordingDeadline())

        self.assertEqual(result.answer, "无法可靠提供该型号详情。")
        self.assertEqual(calls, ["UNKNOWN", "HP780"])
        first_observation = json.loads(complete.calls[1]["messages"][-1]["content"])
        second_observation = json.loads(complete.calls[2]["messages"][-1]["content"])
        self.assertEqual(first_observation["error"]["code"], "PRODUCT_NOT_FOUND")
        self.assertEqual(second_observation["error"]["code"], "TOOL_EXECUTION_ERROR")

    def test_duplicate_success_reuses_observation_but_consumes_call_slot(self):
        from agent.loop import AgentLoop

        calls = []

        def search_products(query):
            calls.append(query)
            return ProductSearchResult(products=[])

        complete = FakeCompleteChat([
            tool_completion(
                "call-1",
                "search_products",
                '{"query":" 酒店 "}',
            ),
            tool_completion(
                "call-2",
                "search_products",
                '{"query":"酒店"}',
            ),
            tool_completion(
                "call-3",
                "search_products",
                '{"query":"酒店"}',
            ),
            text_completion("没有可靠候选，请联系专业技术人员。"),
        ])
        executor = ToolExecutor(MappingProxyType({
            "search_products": search_products,
        }))

        result = AgentLoop(
            executor=executor,
            complete_chat=complete,
        ).run(BASE_MESSAGES, deadline=RecordingDeadline())

        self.assertEqual(result.answer, "没有可靠候选，请联系专业技术人员。")
        self.assertEqual(calls, ["酒店"])
        self.assertEqual(len(complete.calls), 4)
        self.assertIsNotNone(complete.calls[0]["tools"])
        self.assertIsNotNone(complete.calls[1]["tools"])
        self.assertIsNotNone(complete.calls[2]["tools"])
        self.assertIsNone(complete.calls[3]["tools"])
        observations = [
            complete.calls[index]["messages"][-1]["content"]
            for index in (1, 2, 3)
        ]
        self.assertEqual(observations[0], observations[1])
        self.assertEqual(observations[1], observations[2])
        self.assertEqual(
            [
                complete.calls[index]["messages"][-1]["tool_call_id"]
                for index in (1, 2, 3)
            ],
            ["call-1", "call-2", "call-3"],
        )

    def test_tools_disabled_completion_cannot_execute_fourth_call(self):
        from agent.loop import SAFE_AGENT_ANSWER, AgentLoop

        complete = FakeCompleteChat([
            tool_completion(f"call-{index}", "get_contact_info", "{}")
            for index in range(1, 5)
        ])
        executor = RecordingExecutor()

        result = AgentLoop(
            executor=executor,
            complete_chat=complete,
        ).run(BASE_MESSAGES, deadline=RecordingDeadline())

        self.assertEqual(result.answer, SAFE_AGENT_ANSWER)
        self.assertEqual(len(executor.calls), 3)
        self.assertEqual(len(complete.calls), 4)
        self.assertIsNone(complete.calls[-1]["tools"])

    def test_multiple_tool_calls_terminate_without_execution(self):
        from agent.loop import SAFE_AGENT_ANSWER, AgentLoop

        response = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="tool_calls",
            message=SimpleNamespace(
                content=None,
                tool_calls=[
                    tool_call("call-1", "get_contact_info", "{}"),
                    tool_call("call-2", "get_contact_info", "{}"),
                ],
            ),
        )])
        executor = RecordingExecutor()

        result = AgentLoop(
            executor=executor,
            complete_chat=FakeCompleteChat([response]),
        ).run(BASE_MESSAGES, deadline=RecordingDeadline())

        self.assertEqual(result.answer, SAFE_AGENT_ANSWER)
        self.assertEqual(executor.calls, [])

    def test_malformed_or_incomplete_provider_turn_terminates_safely(self):
        from agent.loop import SAFE_AGENT_ANSWER, AgentLoop

        cases = (
            SimpleNamespace(choices=[]),
            SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="   ", tool_calls=None),
            )]),
            text_completion("partial", finish_reason="length"),
            text_completion("blocked", finish_reason="content_filter"),
            tool_completion("", "get_contact_info", "{}"),
            tool_completion("call-1", "", "{}"),
        )
        for response in cases:
            with self.subTest(response=response):
                executor = RecordingExecutor()
                result = AgentLoop(
                    executor=executor,
                    complete_chat=FakeCompleteChat([response]),
                ).run(BASE_MESSAGES, deadline=RecordingDeadline())
                self.assertEqual(result.answer, SAFE_AGENT_ANSWER)
                self.assertEqual(executor.calls, [])

    def test_deadline_is_checked_around_provider_and_tool_operations(self):
        from agent.loop import AgentLoop

        complete = FakeCompleteChat([
            tool_completion("call-1", "get_contact_info", "{}"),
            text_completion("联系我们。"),
        ])
        deadline = RecordingDeadline()

        AgentLoop(
            executor=RecordingExecutor(),
            complete_chat=complete,
        ).run(BASE_MESSAGES, deadline=deadline)

        self.assertEqual(deadline.checks, 6)

    def test_deadline_expiry_after_provider_call_propagates_without_tool_execution(self):
        from agent.loop import AgentLoop

        executor = RecordingExecutor()
        deadline = RecordingDeadline(fail_on=2)

        with self.assertRaises(AgentDeadlineExceeded):
            AgentLoop(
                executor=executor,
                complete_chat=FakeCompleteChat([
                    tool_completion("call-1", "get_contact_info", "{}"),
                ]),
            ).run(BASE_MESSAGES, deadline=deadline)

        self.assertEqual(executor.calls, [])


if __name__ == "__main__":
    unittest.main()
