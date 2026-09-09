import unittest
from dataclasses import FrozenInstanceError


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


if __name__ == "__main__":
    unittest.main()
