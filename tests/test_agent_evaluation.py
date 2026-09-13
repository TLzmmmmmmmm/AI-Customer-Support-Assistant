import unittest
from types import MappingProxyType

from agent import AgentDeadline, AgentToolCall, ToolExecutor
from models import ChatMessage
from routing import Route, RouteExecutionResult, RouteTrace
from support_tools import ProductSearchResult, ToolError, ToolErrorCode


class RecordingToolExecutorTests(unittest.TestCase):
    def test_records_normalized_valid_calls_and_preserves_duplicate_cache_behavior(self):
        from evaluation.agent import RecordingToolExecutor

        handler_calls = []

        def search_products(query):
            handler_calls.append(query)
            return ProductSearchResult(products=[])

        executor = RecordingToolExecutor(MappingProxyType({
            "search_products": search_products,
        }))
        cache = {}

        first = executor.execute_named(
            "search_products",
            {"query": "  LY198  "},
            cache,
        )
        second = executor.execute(
            AgentToolCall(
                id="call-2",
                name="search_products",
                arguments='{"query":"LY198"}',
            ),
            cache,
        )

        self.assertEqual(handler_calls, ["LY198"])
        self.assertTrue(first.success)
        self.assertFalse(first.reused)
        self.assertTrue(second.success)
        self.assertTrue(second.reused)
        self.assertEqual(first.content, second.content)
        self.assertEqual(
            executor.validated_tool_calls,
            (
                {"name": "search_products", "arguments": {"query": "LY198"}},
                {"name": "search_products", "arguments": {"query": "LY198"}},
            ),
        )

    def test_does_not_record_unknown_or_invalid_calls(self):
        from evaluation.agent import RecordingToolExecutor

        executor = RecordingToolExecutor(MappingProxyType({}))

        unknown = executor.execute_named("private_tool", {}, {})
        invalid = executor.execute_named(
            "search_products",
            {"query": "", "extra": True},
            {},
        )
        malformed = executor.execute(
            AgentToolCall("call-3", "search_products", "not-json"),
            {},
        )

        self.assertEqual(unknown.error_code, "INVALID_ARGUMENT")
        self.assertEqual(invalid.error_code, "INVALID_ARGUMENT")
        self.assertEqual(malformed.error_code, "INVALID_ARGUMENT")
        self.assertEqual(executor.validated_tool_calls, ())

    def test_valid_domain_failure_matches_production_and_is_recorded(self):
        from evaluation.agent import RecordingToolExecutor

        def missing(product_id):
            raise ToolError(
                code=ToolErrorCode.PRODUCT_NOT_FOUND,
                message="No product matches.",
                tool_name="get_product_details",
            )

        registry = MappingProxyType({"get_product_details": missing})
        production_cache = {}
        recording_cache = {}

        production = ToolExecutor(registry).execute_named(
            "get_product_details",
            {"product_id": "LY999"},
            production_cache,
        )
        recorder = RecordingToolExecutor(registry)
        observed = recorder.execute_named(
            "get_product_details",
            {"product_id": "LY999"},
            recording_cache,
        )

        self.assertEqual(observed, production)
        self.assertEqual(recording_cache, production_cache)
        self.assertEqual(
            recorder.validated_tool_calls,
            ({"name": "get_product_details", "arguments": {"product_id": "LY999"}},),
        )


class AgentEvaluationRunnerTests(unittest.TestCase):
    def test_returns_only_current_case_route_calls_and_final_answer(self):
        from evaluation.agent import AgentEvaluationRunner, RecordingToolExecutor

        executor = RecordingToolExecutor(MappingProxyType({
            "search_products": lambda query: ProductSearchResult(products=[]),
        }))

        class Orchestrator:
            def run(self, messages, *, deadline):
                executor.execute_named(
                    "search_products",
                    {"query": messages[-1].content},
                    {},
                )
                return RouteExecutionResult(
                    answer=f"answer:{messages[-1].content}",
                    trace=RouteTrace(route=Route.PRODUCT_SEARCH),
                )

        runner = AgentEvaluationRunner(Orchestrator(), executor)
        deadline = AgentDeadline.start(10, clock=lambda: 0)

        first = runner.run(
            (ChatMessage(role="user", content="  LY198  "),),
            deadline=deadline,
        )
        second = runner.run(
            (ChatMessage(role="user", content="LY298"),),
            deadline=deadline,
        )

        self.assertEqual(
            first,
            {
                "route": Route.PRODUCT_SEARCH,
                "tool_calls": (
                    {"name": "search_products", "arguments": {"query": "LY198"}},
                ),
                "final_answer": "answer:LY198",
            },
        )
        self.assertEqual(
            second,
            {
                "route": Route.PRODUCT_SEARCH,
                "tool_calls": (
                    {"name": "search_products", "arguments": {"query": "LY298"}},
                ),
                "final_answer": "answer:LY298",
            },
        )
        self.assertEqual(set(second), {"route", "tool_calls", "final_answer"})

    def test_propagates_the_original_orchestrator_exception(self):
        from evaluation.agent import AgentEvaluationRunner, RecordingToolExecutor

        expected = RuntimeError("route failed")

        class BrokenOrchestrator:
            def run(self, messages, *, deadline):
                raise expected

        runner = AgentEvaluationRunner(
            BrokenOrchestrator(),
            RecordingToolExecutor(MappingProxyType({})),
        )

        with self.assertRaises(RuntimeError) as raised:
            runner.run(
                (ChatMessage(role="user", content="hello"),),
                deadline=AgentDeadline.start(10, clock=lambda: 0),
            )

        self.assertIs(raised.exception, expected)


class ToolCallsMatchTests(unittest.TestCase):
    def test_finds_complete_matching_when_greedy_first_choice_would_fail(self):
        from evaluation.agent import tool_calls_match

        actual = (
            {"name": "search_products", "arguments": {"query": "radio", "limit": 3}},
            {"name": "search_products", "arguments": {"query": "radio"}},
        )
        expected = (
            {"name": "search_products", "arguments": {"query": "radio"}},
            {"name": "search_products", "arguments": {"query": "radio", "limit": 3}},
        )

        self.assertTrue(tool_calls_match(actual, expected))

    def test_requires_exact_multiplicity(self):
        from evaluation.agent import tool_calls_match

        call = {"name": "get_contact_info", "arguments": {}}

        self.assertFalse(tool_calls_match((call, call), (call,)))

    def test_requires_name_equality_and_expected_argument_subset(self):
        from evaluation.agent import tool_calls_match

        actual = ({
            "name": "get_product_details",
            "arguments": {"product_id": "ly198", "locale": "zh-CN"},
        },)

        self.assertTrue(tool_calls_match(actual, ({
            "name": "get_product_details",
            "arguments": {"product_id": "ly198"},
        },)))
        self.assertFalse(tool_calls_match(actual, ({
            "name": "get_product_details",
            "arguments": {"product_id": "ly298"},
        },)))
        self.assertFalse(tool_calls_match(actual, ({
            "name": "search_products",
            "arguments": {"product_id": "ly198"},
        },)))


if __name__ == "__main__":
    unittest.main()
