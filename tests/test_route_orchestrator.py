import json
import unittest
from collections import deque
from types import SimpleNamespace

from agent import AgentResult, ToolObservation
from models import ChatMessage
from routing import (
    FailureLayer,
    Route,
    RouteDecision,
    RoutingResult,
    ToolTrace,
)


def completion(content, *, finish_reason="stop", tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=content, tool_calls=tool_calls),
    )])


class FakeRouter:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def route(self, messages, *, deadline):
        self.calls.append((messages, deadline))
        return self.result


class FakeRetriever:
    def __init__(self, results=()):
        self.results = list(results)
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return list(self.results)


class FakeExecutor:
    def __init__(self, observation=None):
        self.observation = observation or ToolObservation(
            content='{"ok":true,"result":{}}',
            success=True,
            reused=False,
            cache_key="key",
            tool_name="get_contact_info",
        )
        self.calls = []

    def execute_named(self, name, arguments, cache):
        self.calls.append((name, arguments, cache))
        return self.observation


class FakeAgent:
    def __init__(self, result=None):
        self.result = result or AgentResult(answer="agent answer")
        self.calls = []

    def run(self, messages, *, deadline):
        self.calls.append((messages, deadline))
        return self.result


class FakeComplete:
    def __init__(self, responses=()):
        self.responses = deque(responses)
        self.calls = []

    def __call__(self, messages, *, tools=None):
        self.calls.append((messages, tools))
        return self.responses.popleft()


class Deadline:
    def __init__(self):
        self.checks = 0

    def ensure_active(self):
        self.checks += 1


def chat(question="用户问题"):
    return [ChatMessage(role="user", content=question)]


def retrieval_result(chunk_id="solution:emergency:content"):
    return SimpleNamespace(
        chunk_id=chunk_id,
        type="solution",
        section="应急通信",
        text="# 应急通信\n\n解决方案内容",
    )


def build_orchestrator(decision, *, router_failure=None, retriever=None,
                       executor=None, agent=None, complete=None):
    from routing import RouteOrchestrator

    return RouteOrchestrator(
        router=FakeRouter(RoutingResult(decision, router_failure)),
        retriever=retriever or FakeRetriever(),
        executor=executor or FakeExecutor(),
        agent_loop=agent or FakeAgent(),
        complete_chat=complete or FakeComplete(),
    )


class RouteOrchestratorTests(unittest.TestCase):
    def test_fallback_is_fixed_and_executes_no_other_capability(self):
        from routing import SAFE_FALLBACK_ANSWER

        retriever = FakeRetriever()
        executor = FakeExecutor()
        agent = FakeAgent()
        complete = FakeComplete()
        orchestrator = build_orchestrator(
            RouteDecision(Route.FALLBACK),
            retriever=retriever,
            executor=executor,
            agent=agent,
            complete=complete,
        )

        result = orchestrator.run(chat("LY198 今天库存多少？"), deadline=Deadline())

        self.assertEqual(result.answer, SAFE_FALLBACK_ANSWER)
        self.assertIsNone(result.trace.failure_layer)
        self.assertEqual(retriever.queries, [])
        self.assertEqual(executor.calls, [])
        self.assertEqual(agent.calls, [])
        self.assertEqual(complete.calls, [])

    def test_router_failure_uses_same_fixed_answer_with_routing_trace(self):
        from routing import SAFE_FALLBACK_ANSWER

        result = build_orchestrator(
            RouteDecision(Route.FALLBACK),
            router_failure=FailureLayer.ROUTING,
        ).run(chat("ambiguous"), deadline=Deadline())

        self.assertEqual(result.answer, SAFE_FALLBACK_ANSWER)
        self.assertEqual(result.trace.failure_layer, FailureLayer.ROUTING)

    def test_direct_uses_one_no_tool_generation_without_retrieval(self):
        retriever = FakeRetriever()
        complete = FakeComplete([completion("您好，请问有什么可以帮您？")])

        result = build_orchestrator(
            RouteDecision(Route.DIRECT),
            retriever=retriever,
            complete=complete,
        ).run(chat("你好"), deadline=Deadline())

        self.assertEqual(result.answer, "您好，请问有什么可以帮您？")
        self.assertEqual(retriever.queries, [])
        self.assertEqual(len(complete.calls), 1)
        self.assertIsNone(complete.calls[0][1])

    def test_knowledge_retrieves_latest_raw_query_then_generates_without_tools(self):
        retriever = FakeRetriever([retrieval_result()])
        complete = FakeComplete([completion("解决方案回答")])

        result = build_orchestrator(
            RouteDecision(Route.KNOWLEDGE),
            retriever=retriever,
            complete=complete,
        ).run(chat("你们有哪些解决方案？"), deadline=Deadline())

        self.assertEqual(result.answer, "解决方案回答")
        self.assertEqual(retriever.queries, ["你们有哪些解决方案？"])
        self.assertEqual(
            result.trace.retrieved_chunk_ids,
            ("solution:emergency:content",),
        )
        self.assertIsNone(complete.calls[0][1])

    def test_non_agentic_tool_routes_execute_known_tool_then_generate(self):
        cases = (
            (
                RouteDecision(Route.EXACT_PRODUCT, product_id="ly198"),
                "LY198 功率是多少？",
                "get_product_details",
                {"product_id": "ly198"},
            ),
            (
                RouteDecision(Route.PRODUCT_SEARCH),
                "推荐适合酒店的产品",
                "search_products",
                {"query": "推荐适合酒店的产品"},
            ),
            (
                RouteDecision(Route.CONTACT),
                "公司的电话是多少？",
                "get_contact_info",
                {},
            ),
        )
        for decision, question, tool_name, arguments in cases:
            with self.subTest(route=decision.route):
                observation = ToolObservation(
                    content=json.dumps({"ok": True, "result": {}}),
                    success=True,
                    reused=False,
                    cache_key="key",
                    tool_name=tool_name,
                )
                executor = FakeExecutor(observation)
                retriever = FakeRetriever()
                complete = FakeComplete([completion("最终回答")])

                result = build_orchestrator(
                    decision,
                    retriever=retriever,
                    executor=executor,
                    complete=complete,
                ).run(chat(question), deadline=Deadline())

                self.assertEqual(result.answer, "最终回答")
                self.assertEqual(executor.calls[0][:2], (tool_name, arguments))
                self.assertEqual(retriever.queries, [])
                self.assertEqual(result.trace.tool_calls, (
                    ToolTrace(name=tool_name, success=True),
                ))
                self.assertEqual(result.trace.tool_call_count, 1)

    def test_agentic_tool_route_skips_generic_rag_and_uses_agent(self):
        retriever = FakeRetriever()
        agent_result = AgentResult(
            answer="混合回答",
            tool_calls=(
                ToolTrace("get_product_details", True),
                ToolTrace("get_contact_info", True),
            ),
        )
        agent = FakeAgent(agent_result)

        result = build_orchestrator(
            RouteDecision(Route.EXACT_PRODUCT, agentic=True, product_id="ly198"),
            retriever=retriever,
            agent=agent,
        ).run(chat("LY198 功率是多少？另外公司电话是什么？"), deadline=Deadline())

        self.assertEqual(result.answer, "混合回答")
        self.assertEqual(retriever.queries, [])
        self.assertEqual(len(agent.calls), 1)
        self.assertEqual(result.trace.tool_call_count, 2)

    def test_agentic_knowledge_retrieves_before_agent(self):
        retriever = FakeRetriever([retrieval_result()])
        agent = FakeAgent(AgentResult(
            answer="方案和联系方式",
            tool_calls=(ToolTrace("get_contact_info", True),),
        ))

        result = build_orchestrator(
            RouteDecision(Route.KNOWLEDGE, agentic=True),
            retriever=retriever,
            agent=agent,
        ).run(
            chat("介绍应急通信解决方案，另外怎么联系你们？"),
            deadline=Deadline(),
        )

        self.assertEqual(result.answer, "方案和联系方式")
        self.assertEqual(
            retriever.queries,
            ["介绍应急通信解决方案，另外怎么联系你们？"],
        )
        self.assertIn("BEGIN_RAG_DATA", agent.calls[0][0][-1]["content"])
        self.assertEqual(result.trace.retrieved_chunk_ids, (
            "solution:emergency:content",
        ))

    def test_invalid_final_completion_returns_generation_failure(self):
        from agent import SAFE_AGENT_ANSWER

        result = build_orchestrator(
            RouteDecision(Route.DIRECT),
            complete=FakeComplete([completion("partial", finish_reason="length")]),
        ).run(chat("你好"), deadline=Deadline())

        self.assertEqual(result.answer, SAFE_AGENT_ANSWER)
        self.assertEqual(result.trace.failure_layer, FailureLayer.GENERATION)

    def test_direct_tool_domain_and_execution_errors_have_distinct_failures(self):
        cases = (
            ("PRODUCT_NOT_FOUND", None),
            ("TOOL_EXECUTION_ERROR", FailureLayer.TOOL_EXECUTION),
        )
        for error_code, failure in cases:
            with self.subTest(error_code=error_code):
                executor = FakeExecutor(ToolObservation(
                    content=json.dumps({
                        "ok": False,
                        "error": {"code": error_code},
                    }),
                    success=False,
                    reused=False,
                    cache_key=None,
                    tool_name="get_product_details",
                    error_code=error_code,
                ))
                result = build_orchestrator(
                    RouteDecision(Route.EXACT_PRODUCT, product_id="missing"),
                    executor=executor,
                    complete=FakeComplete([completion("安全回答")]),
                ).run(chat("missing 参数"), deadline=Deadline())

                self.assertEqual(result.trace.failure_layer, failure)
                self.assertEqual(result.trace.tool_calls[0].error_code, error_code)


if __name__ == "__main__":
    unittest.main()
