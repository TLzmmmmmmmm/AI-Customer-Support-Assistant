from __future__ import annotations

from collections.abc import Mapping, Sequence

from knowledge_pipeline.models import SourceRef
from trace_models import FailureLayer, ToolTrace

from .executor import ToolObservation
from .models import (
    AgentDeadline,
    AgentDeadlineExceeded,
    AgentResult,
    AgentToolCall,
    AgentTurn,
    TOOL_SPECS,
)
from .tool_outcomes import (
    successful_tool_sources,
    tool_failure_from_trace,
    tool_trace_from_observation,
)


AGENT_TOOL_POLICY = """
Use the provided deterministic tools under these rules:
- Exact-product factual answers require get_product_details. A factual answer requires a successful get_product_details observation for every product whose facts are stated. Conversation history may identify a product but is not factual evidence. Never answer product facts from previous assistant text, model memory, or general knowledge.
- Multiple products explicitly requested are intended targets, not ambiguous singular references. Retrieve every required product within the tool-call limit.
- A genuinely ambiguous singular reference with multiple plausible antecedents must be clarified. Do not guess or produce an exact-product factual claim unless that product has a successful get_product_details observation in the current execution.
- Explicit product discovery, candidate selection, and recommendation requests require search_products.
- Explicit phone, email, and contact-channel questions require get_contact_info.
- Recommendation and scenario-fit answers must frame products as candidates unless an observation explicitly proves suitability, and must direct the user to professional technical staff for final selection.
- When recommendation-oriented solution or support advice needs expert confirmation, add the same generic professional-staff guidance.
- If product search returns no candidates, state that no reliable candidate was found and still add generic professional-staff guidance.
- Generic guidance does not require get_contact_info. Never invent contact facts.
- Do not repeat a successful invocation merely because the original request still matches a must-use category.
- A failed invocation may be retried with corrected arguments or replaced with another appropriate tool.
- Propose at most one tool call per response.
""".strip()


def build_agent_messages(
    messages: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    provider_messages = [dict(message) for message in messages]
    provider_messages.insert(1, {
        "role": "system",
        "content": AGENT_TOOL_POLICY,
    })
    return provider_messages


def assistant_tool_message(turn: AgentTurn) -> dict[str, object]:
    call = turn.tool_calls[0]
    return {
        "role": "assistant",
        "content": turn.content,
        "tool_calls": [{
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": call.arguments,
            },
        }],
    }


def invalid_tool_call_traces(
    tool_calls: Sequence[AgentToolCall],
) -> tuple[ToolTrace, ...]:
    return tuple(
        ToolTrace(
            name=(
                call.name
                if call.name in TOOL_SPECS
                else "tool_executor"
            ),
            success=False,
            error_code="INVALID_ARGUMENT",
        )
        for call in tool_calls
    )


def attach_agent_error_metadata(
    error: Exception,
    *,
    layer: FailureLayer,
    tool_traces: Sequence[ToolTrace],
) -> None:
    try:
        error.failure_layer = layer
        error.tool_calls = tuple(tool_traces)
    except (AttributeError, TypeError):
        pass


def ensure_agent_active(
    deadline: AgentDeadline,
    *,
    layer: FailureLayer,
    tool_traces: Sequence[ToolTrace],
) -> None:
    try:
        deadline.ensure_active()
    except AgentDeadlineExceeded as error:
        attach_agent_error_metadata(
            error,
            layer=layer,
            tool_traces=tool_traces,
        )
        raise


def record_agent_observation(
    observation: ToolObservation,
    *,
    tool_traces: Sequence[ToolTrace],
    successful_sources: Sequence[SourceRef],
    pending_failures: Sequence[tuple[FailureLayer, str]],
) -> tuple[
    tuple[ToolTrace, ...],
    tuple[SourceRef, ...],
    tuple[tuple[FailureLayer, str], ...],
]:
    trace = tool_trace_from_observation(observation)
    updated_traces = (*tool_traces, trace)
    updated_sources = tuple(successful_sources)
    updated_failures = tuple(pending_failures)
    if trace.success:
        updated_sources = (
            *updated_sources,
            *successful_tool_sources(observation),
        )
        updated_failures = tuple(
            item
            for item in updated_failures
            if not (
                item[0] in {
                    FailureLayer.TOOL_SELECTION,
                    FailureLayer.ARGUMENT_GENERATION,
                }
                or (
                    item[0] == FailureLayer.TOOL_EXECUTION
                    and item[1] == trace.name
                )
            )
        )
    else:
        failure = tool_failure_from_trace(trace)
        if failure is not None:
            updated_failures = (*updated_failures, (failure, trace.name))
    return updated_traces, updated_sources, updated_failures


def finalize_agent_result(
    answer: str,
    *,
    tool_traces: Sequence[ToolTrace],
    successful_sources: Sequence[SourceRef],
    pending_failures: Sequence[tuple[FailureLayer, str]],
    failure: FailureLayer | None = None,
) -> AgentResult:
    unresolved = failure
    if unresolved is None and pending_failures:
        unresolved = pending_failures[0][0]
    return AgentResult(
        answer=answer,
        tool_calls=tuple(tool_traces),
        failure_layer=unresolved,
        sources=tuple(successful_sources),
    )


__all__ = [
    "AGENT_TOOL_POLICY",
    "assistant_tool_message",
    "attach_agent_error_metadata",
    "build_agent_messages",
    "ensure_agent_active",
    "finalize_agent_result",
    "invalid_tool_call_traces",
    "record_agent_observation",
]
