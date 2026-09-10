import unittest

from agent import SAFE_AGENT_ANSWER, ToolObservation
from knowledge_pipeline.models import SourceRef
from knowledge_pipeline.retrieval.models import RetrievalResult
from routing import (
    SAFE_FALLBACK_ANSWER,
    FailureLayer,
    Route,
    RouteExecutionResult,
    RouteTrace,
    ToolTrace,
)
from routing.finalization import finalize_non_agentic_route


def retrieval_result(chunk_id, sources):
    return RetrievalResult.model_validate({
        "rank": 1,
        "score": 0.9,
        "match_origin": "dense",
        "matched_entity_ids": [],
        "chunk_id": chunk_id,
        "parent_document_id": chunk_id.rsplit(":", 1)[0],
        "type": "solution",
        "section": "方案介绍",
        "text": "# 测试方案\n\n方案内容。",
        "content_hash": "a" * 64,
        "metadata": {
            "solution_id": "test",
            "slug": "test",
        },
        "source_url": sources[0].url,
        "source_files": ["src/content/solutions/test.md"],
        "sources": list(sources),
    })


class NonAgenticFinalizationTests(unittest.TestCase):
    def test_deterministic_success_projects_trace_and_sources(self):
        source = SourceRef(title="产品资料", url="https://example.com/product")
        observation = ToolObservation(
            content='{"ok":true}',
            success=True,
            reused=True,
            cache_key="key",
            tool_name="get_product_details",
            sources=(source,),
        )

        result = finalize_non_agentic_route(
            Route.EXACT_PRODUCT,
            answer="产品回答",
            tool_observation=observation,
        )

        self.assertEqual(result, RouteExecutionResult(
            answer="产品回答",
            trace=RouteTrace(
                route=Route.EXACT_PRODUCT,
                tool_calls=(ToolTrace(
                    name="get_product_details",
                    success=True,
                    reused=True,
                ),),
                tool_call_count=1,
            ),
            sources=(source,),
        ))

    def test_deterministic_tool_failure_precedes_generation_failure(self):
        observation = ToolObservation(
            content='{"ok":false}',
            success=False,
            reused=False,
            cache_key=None,
            tool_name="get_contact_info",
            error_code="TOOL_EXECUTION_ERROR",
        )

        result = finalize_non_agentic_route(
            Route.CONTACT,
            answer=SAFE_AGENT_ANSWER,
            tool_observation=observation,
            generation_failure=FailureLayer.GENERATION,
        )

        self.assertEqual(result.trace.failure_layer, FailureLayer.TOOL_EXECUTION)
        self.assertEqual(result.sources, ())

    def test_knowledge_preserves_hit_and_source_order_without_deduplication(self):
        first = SourceRef(title="A", url="https://example.com/a")
        second = SourceRef(title="B", url="https://example.com/b")
        results = (
            retrieval_result("solution:first:content", (first, second)),
            retrieval_result("solution:second:content", (first,)),
        )

        result = finalize_non_agentic_route(
            Route.KNOWLEDGE,
            answer="知识回答",
            retrieval_results=results,
            generation_failure=FailureLayer.GENERATION,
        )

        self.assertEqual(
            result.trace.retrieved_chunk_ids,
            ("solution:first:content", "solution:second:content"),
        )
        self.assertEqual(result.sources, (first, second, first))
        self.assertEqual(result.trace.failure_layer, FailureLayer.GENERATION)
        self.assertEqual(result.trace.tool_calls, ())
        self.assertEqual(result.trace.tool_call_count, 0)

    def test_safe_answer_suppression_uses_exact_string_equality(self):
        source = SourceRef(title="资料", url="https://example.com/source")
        results = (retrieval_result("solution:test:content", (source,)),)

        suppressed = finalize_non_agentic_route(
            Route.KNOWLEDGE,
            answer=SAFE_AGENT_ANSWER,
            retrieval_results=results,
        )
        retained = finalize_non_agentic_route(
            Route.KNOWLEDGE,
            answer=f"{SAFE_AGENT_ANSWER} ",
            retrieval_results=results,
            generation_failure=FailureLayer.GENERATION,
        )

        self.assertEqual(suppressed.sources, ())
        self.assertEqual(retained.sources, (source,))

    def test_direct_uses_generation_failure_and_empty_evidence_defaults(self):
        result = finalize_non_agentic_route(
            Route.DIRECT,
            answer="直接回答",
            generation_failure=FailureLayer.GENERATION,
        )

        self.assertEqual(result, RouteExecutionResult(
            answer="直接回答",
            trace=RouteTrace(
                route=Route.DIRECT,
                failure_layer=FailureLayer.GENERATION,
            ),
        ))

    def test_fallback_owns_fixed_answer_and_existing_trace_defaults(self):
        result = finalize_non_agentic_route(
            Route.FALLBACK,
            answer="must be ignored",
            routing_failure=FailureLayer.ROUTING,
            generation_failure=FailureLayer.GENERATION,
        )

        self.assertEqual(result, RouteExecutionResult(
            answer=SAFE_FALLBACK_ANSWER,
            trace=RouteTrace(
                route=Route.FALLBACK,
                failure_layer=FailureLayer.ROUTING,
            ),
        ))

    def test_non_fallback_requires_answer(self):
        with self.assertRaises(ValueError):
            finalize_non_agentic_route(Route.DIRECT)

    def test_deterministic_route_requires_tool_observation(self):
        with self.assertRaises(ValueError):
            finalize_non_agentic_route(
                Route.PRODUCT_SEARCH,
                answer="回答",
            )


if __name__ == "__main__":
    unittest.main()
