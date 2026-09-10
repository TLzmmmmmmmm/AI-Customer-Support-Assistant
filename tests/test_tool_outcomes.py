import unittest

from agent import ToolObservation
from agent.tool_outcomes import (
    successful_tool_sources,
    tool_failure_from_trace,
    tool_trace_from_observation,
)
from knowledge_pipeline.models import SourceRef
from trace_models import FailureLayer, ToolTrace


class ToolOutcomeTests(unittest.TestCase):
    def test_trace_projects_only_trace_fields(self):
        source = SourceRef(title="产品资料", url="https://example.com/product")
        first = ToolObservation(
            content="first payload",
            success=True,
            reused=True,
            cache_key="first-key",
            tool_name="search_products",
            error_code="PRODUCT_NOT_FOUND",
            sources=(source,),
        )
        second = ToolObservation(
            content="different payload",
            success=True,
            reused=True,
            cache_key="different-key",
            tool_name="search_products",
            error_code="PRODUCT_NOT_FOUND",
            sources=(),
        )

        first_trace = tool_trace_from_observation(first)
        second_trace = tool_trace_from_observation(second)

        self.assertEqual(first_trace, ToolTrace(
            name="search_products",
            success=True,
            error_code="PRODUCT_NOT_FOUND",
            reused=True,
        ))
        self.assertEqual(second_trace, first_trace)

    def test_failure_classification_matches_existing_error_semantics(self):
        cases = (
            ("tool_executor", "TOOL_EXECUTION_ERROR", FailureLayer.TOOL_EXECUTION),
            ("get_contact_info", "TOOL_UNAVAILABLE", FailureLayer.TOOL_EXECUTION),
            ("tool_executor", "INVALID_ARGUMENT", FailureLayer.TOOL_SELECTION),
            ("search_products", "INVALID_ARGUMENT", FailureLayer.ARGUMENT_GENERATION),
            ("get_product_details", "PRODUCT_NOT_FOUND", None),
            ("get_contact_info", "UNKNOWN_CODE", None),
            ("get_contact_info", None, None),
        )

        for name, error_code, expected in cases:
            with self.subTest(name=name, error_code=error_code):
                trace = ToolTrace(
                    name=name,
                    success=False,
                    error_code=error_code,
                )
                self.assertEqual(
                    tool_failure_from_trace(trace),
                    expected,
                )

    def test_successful_observation_preserves_exact_sources(self):
        sources = (
            SourceRef(title="来源一", url="https://example.com/one"),
            SourceRef(title="来源二", url="https://example.com/two"),
        )
        observation = ToolObservation(
            content="payload",
            success=True,
            reused=False,
            cache_key="key",
            tool_name="search_products",
            sources=sources,
        )

        self.assertIs(
            successful_tool_sources(observation),
            sources,
        )

    def test_failed_observation_suppresses_sources_independently_of_error_code(self):
        source = SourceRef(title="来源", url="https://example.com/source")
        for error_code in ("PRODUCT_NOT_FOUND", "TOOL_EXECUTION_ERROR", None):
            with self.subTest(error_code=error_code):
                observation = ToolObservation(
                    content="payload",
                    success=False,
                    reused=False,
                    cache_key=None,
                    tool_name="get_product_details",
                    error_code=error_code,
                    sources=(source,),
                )
                self.assertEqual(
                    successful_tool_sources(observation),
                    (),
                )

    def test_inconsistent_success_keeps_failure_and_sources_independent(self):
        source = SourceRef(title="来源", url="https://example.com/source")
        observation = ToolObservation(
            content="payload",
            success=True,
            reused=False,
            cache_key="key",
            tool_name="get_contact_info",
            error_code="TOOL_EXECUTION_ERROR",
            sources=(source,),
        )
        trace = tool_trace_from_observation(observation)

        self.assertEqual(
            tool_failure_from_trace(trace),
            FailureLayer.TOOL_EXECUTION,
        )
        self.assertIs(
            successful_tool_sources(observation),
            observation.sources,
        )


if __name__ == "__main__":
    unittest.main()
