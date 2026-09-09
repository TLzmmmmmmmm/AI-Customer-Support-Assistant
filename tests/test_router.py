import unittest
from collections import deque
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

from knowledge_pipeline.retrieval.models import EntityMatch
from models import ChatMessage


class RoutingContractTests(unittest.TestCase):
    def test_route_decision_and_trace_keep_only_observable_bounded_state(self):
        from routing import (
            FailureLayer,
            Route,
            RouteDecision,
            RouteExecutionResult,
            RouteTrace,
            ToolTrace,
        )

        self.assertEqual(
            {route.value for route in Route},
            {
                "product_search",
                "exact_product",
                "contact",
                "knowledge",
                "direct",
                "fallback",
            },
        )
        decision = RouteDecision(
            route=Route.EXACT_PRODUCT,
            product_id="ly198",
        )
        tool_trace = ToolTrace(
            name="get_product_details",
            success=True,
        )
        trace = RouteTrace(
            route=Route.EXACT_PRODUCT,
            tool_calls=(tool_trace,),
            tool_call_count=1,
            retrieved_chunk_ids=(),
        )
        result = RouteExecutionResult(answer="2W", trace=trace)

        self.assertFalse(decision.agentic)
        self.assertFalse(hasattr(decision, "failure_layer"))
        self.assertEqual(result.trace.tool_calls, (tool_trace,))
        self.assertIsNone(result.trace.failure_layer)
        self.assertEqual(
            {layer.value for layer in FailureLayer},
            {
                "ROUTING",
                "TOOL_SELECTION",
                "ARGUMENT_GENERATION",
                "TOOL_EXECUTION",
                "RETRIEVAL",
                "GENERATION",
            },
        )
        with self.assertRaises(FrozenInstanceError):
            decision.agentic = True


def completion(content, *, finish_reason="stop", tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=content, tool_calls=tool_calls),
    )])


class RecordingRetriever:
    def __init__(self, products=None):
        self.products = products or {}
        self.queries = []

    def resolve_entities(self, query):
        self.queries.append(query)
        product_id = self.products.get(query)
        if product_id is None:
            return []
        return [EntityMatch(
            parent_document_id=f"product:{product_id}",
            alias=product_id,
            start=0,
        )]


class RecordingCompletion:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    def __call__(self, messages, *, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


class RecordingDeadline:
    def __init__(self):
        self.checks = 0

    def ensure_active(self):
        self.checks += 1


def messages(question, history=()):
    return [
        *(ChatMessage(role=role, content=content) for role, content in history),
        ChatMessage(role="user", content=question),
    ]


class HybridRouterTests(unittest.TestCase):
    def test_high_confidence_routes_do_not_call_llm(self):
        from routing import HybridRouter, Route

        cases = (
            ("推荐几款对讲机", Route.PRODUCT_SEARCH, None),
            ("推荐适合酒店使用的产品", Route.PRODUCT_SEARCH, None),
            ("地下停车场有什么产品推荐", Route.PRODUCT_SEARCH, None),
            ("公司的电话是多少？", Route.CONTACT, None),
            ("你好", Route.DIRECT, None),
            ("你们有哪些解决方案？", Route.KNOWLEDGE, None),
            ("LY198 今天还有多少库存？", Route.FALLBACK, "ly198"),
            ("LY198 功率是多少？", Route.EXACT_PRODUCT, "ly198"),
        )
        products = {
            question: "ly198"
            for question, _, product_id in cases
            if product_id == "ly198"
        }
        retriever = RecordingRetriever(products)
        complete = RecordingCompletion([])
        router = HybridRouter(retriever=retriever, complete_chat=complete)

        for question, expected_route, expected_product_id in cases:
            with self.subTest(question=question):
                result = router.route(messages(question), deadline=RecordingDeadline())
                self.assertEqual(result.decision.route, expected_route)
                self.assertFalse(result.decision.agentic)
                self.assertEqual(result.decision.product_id, expected_product_id)
                self.assertIsNone(result.failure_layer)

        self.assertEqual(complete.calls, [])

    def test_llm_classification_does_not_make_single_capability_agentic(self):
        from routing import HybridRouter, Route

        history = (
            ("user", "我们需要覆盖一个地下区域。"),
            ("assistant", "请说明人数和通信需求。"),
        )
        complete = RecordingCompletion([completion("product_search")])
        deadline = RecordingDeadline()

        result = HybridRouter(
            retriever=RecordingRetriever(),
            complete_chat=complete,
        ).route(
            messages("地下停车场几十个人通信，有什么建议？", history),
            deadline=deadline,
        )

        self.assertEqual(result.decision.route, Route.PRODUCT_SEARCH)
        self.assertFalse(result.decision.agentic)
        self.assertIsNone(result.failure_layer)
        sent = complete.calls[0]["messages"]
        self.assertEqual(
            sent[1:],
            [
                {"role": "user", "content": history[0][1]},
                {"role": "assistant", "content": history[1][1]},
                {
                    "role": "user",
                    "content": "地下停车场几十个人通信，有什么建议？",
                },
            ],
        )
        self.assertIsNone(complete.calls[0]["tools"])
        self.assertEqual(deadline.checks, 2)

    def test_backend_marks_mixed_and_observation_dependent_routes_agentic(self):
        from routing import HybridRouter, Route

        cases = (
            (
                "LY198 功率是多少？另外公司电话是什么？",
                "exact_product",
                Route.EXACT_PRODUCT,
                "ly198",
            ),
            (
                "介绍应急通信解决方案，另外怎么联系你们？",
                "knowledge",
                Route.KNOWLEDGE,
                None,
            ),
            (
                "推荐几个产品，如果有多个，再比较参数选一个。",
                "product_search",
                Route.PRODUCT_SEARCH,
                None,
            ),
            (
                "LY198 和适合酒店的产品有什么区别？",
                "product_search",
                Route.PRODUCT_SEARCH,
                "ly198",
            ),
        )
        products = {
            cases[0][0]: "ly198",
            cases[3][0]: "ly198",
        }
        complete = RecordingCompletion([
            completion(route) for _, route, _, _ in cases
        ])
        router = HybridRouter(
            retriever=RecordingRetriever(products),
            complete_chat=complete,
        )

        for question, _, expected_route, product_id in cases:
            with self.subTest(question=question):
                result = router.route(messages(question), deadline=RecordingDeadline())
                self.assertEqual(result.decision.route, expected_route)
                self.assertTrue(result.decision.agentic)
                self.assertEqual(result.decision.product_id, product_id)

    def test_unresolved_contextual_product_argument_is_agentic(self):
        from routing import HybridRouter, Route

        complete = RecordingCompletion([completion("exact_product")])
        result = HybridRouter(
            retriever=RecordingRetriever(),
            complete_chat=complete,
        ).route(
            messages(
                "第二个功率呢？",
                (
                    ("user", "推荐两款产品。"),
                    ("assistant", "可以考虑 LY198 和 HP780。"),
                ),
            ),
            deadline=RecordingDeadline(),
        )

        self.assertEqual(result.decision.route, Route.EXACT_PRODUCT)
        self.assertTrue(result.decision.agentic)
        self.assertIsNone(result.decision.product_id)

    def test_invalid_router_results_become_routing_failure_fallback(self):
        from routing import FailureLayer, HybridRouter, Route

        invalid = (
            completion("maybe_sales"),
            completion("product_search because products fit"),
            completion("   "),
            completion("product_search", tool_calls=[object()]),
            completion("product_search", finish_reason="length"),
            completion("knowledge", finish_reason="content_filter"),
            SimpleNamespace(choices=[]),
        )
        for response in invalid:
            with self.subTest(response=response):
                result = HybridRouter(
                    retriever=RecordingRetriever(),
                    complete_chat=RecordingCompletion([response]),
                ).route(
                    messages("地下停车场通信应该怎么解决？"),
                    deadline=RecordingDeadline(),
                )
                self.assertEqual(result.decision.route, Route.FALLBACK)
                self.assertFalse(result.decision.agentic)
                self.assertEqual(result.failure_layer, FailureLayer.ROUTING)

    def test_router_provider_error_propagates(self):
        error = RuntimeError("provider failure")
        with self.assertRaisesRegex(RuntimeError, "provider failure"):
            from routing import HybridRouter

            HybridRouter(
                retriever=RecordingRetriever(),
                complete_chat=RecordingCompletion([error]),
            ).route(
                messages("地下停车场通信应该怎么解决？"),
                deadline=RecordingDeadline(),
            )


if __name__ == "__main__":
    unittest.main()
