from __future__ import annotations

from collections.abc import Sequence

from agent import SAFE_AGENT_ANSWER, ToolObservation
from agent.tool_outcomes import (
    successful_tool_sources,
    tool_failure_from_trace,
    tool_trace_from_observation,
)
from knowledge_pipeline.retrieval.models import RetrievalResult
from trace_models import FailureLayer

from .deterministic import DETERMINISTIC_TOOL_ROUTES
from .knowledge import retrieval_sources
from .models import Route, RouteExecutionResult, RouteTrace


SAFE_FALLBACK_ANSWER = "目前无法根据现有资料可靠确认这项信息。"


def finalize_non_agentic_route(
    route: Route,
    *,
    answer: str | None = None,
    routing_failure: FailureLayer | None = None,
    retrieval_results: Sequence[RetrievalResult] = (),
    tool_observation: ToolObservation | None = None,
    generation_failure: FailureLayer | None = None,
) -> RouteExecutionResult:
    if route == Route.FALLBACK:
        return RouteExecutionResult(
            answer=SAFE_FALLBACK_ANSWER,
            trace=RouteTrace(
                route=route,
                failure_layer=routing_failure,
            ),
        )

    if answer is None:
        raise ValueError("answer is required for non-fallback routes")

    tool_calls = ()
    tool_call_count = 0
    retrieved_chunk_ids = ()
    failure_layer = generation_failure
    sources = ()

    if route in DETERMINISTIC_TOOL_ROUTES:
        if tool_observation is None:
            raise ValueError(
                "tool_observation is required for deterministic routes"
            )
        tool_trace = tool_trace_from_observation(tool_observation)
        tool_calls = (tool_trace,)
        tool_call_count = 1
        tool_failure = tool_failure_from_trace(tool_trace)
        failure_layer = tool_failure or generation_failure
        sources = successful_tool_sources(tool_observation)
    elif route == Route.KNOWLEDGE:
        retrieved_chunk_ids = tuple(
            result.chunk_id
            for result in retrieval_results
        )
        sources = retrieval_sources(retrieval_results)

    return RouteExecutionResult(
        answer=answer,
        trace=RouteTrace(
            route=route,
            tool_calls=tool_calls,
            tool_call_count=tool_call_count,
            retrieved_chunk_ids=retrieved_chunk_ids,
            failure_layer=failure_layer,
        ),
        sources=() if answer == SAFE_AGENT_ANSWER else sources,
    )


__all__ = ["SAFE_FALLBACK_ANSWER", "finalize_non_agentic_route"]
