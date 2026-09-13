from __future__ import annotations

from knowledge_pipeline.models import SourceRef
from trace_models import FailureLayer, ToolTrace

from .executor import ToolObservation


def tool_trace_from_observation(observation: ToolObservation) -> ToolTrace:
    return ToolTrace(
        name=observation.tool_name,
        success=observation.success,
        error_code=observation.error_code,
        reused=observation.reused,
    )


def tool_failure_from_trace(trace: ToolTrace) -> FailureLayer | None:
    if trace.error_code in {"TOOL_EXECUTION_ERROR", "TOOL_UNAVAILABLE"}:
        return FailureLayer.TOOL_EXECUTION
    if trace.error_code == "INVALID_ARGUMENT":
        if trace.name == "tool_executor":
            return FailureLayer.TOOL_SELECTION
        return FailureLayer.ARGUMENT_GENERATION
    return None


def successful_tool_sources(
    observation: ToolObservation,
) -> tuple[SourceRef, ...]:
    if observation.success:
        return observation.sources
    return ()


__all__ = [
    "successful_tool_sources",
    "tool_failure_from_trace",
    "tool_trace_from_observation",
]
