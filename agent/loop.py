from __future__ import annotations

from collections.abc import Mapping, Sequence

from .executor import ToolExecutor, ToolObservation
from .models import (
    AgentDeadline,
    AgentResult,
    llm_tool_schemas,
    normalize_agent_turn,
)
from .runtime import (
    AGENT_TOOL_POLICY,
    assistant_tool_message,
    attach_agent_error_metadata,
    build_agent_messages,
    ensure_agent_active,
    finalize_agent_result,
    invalid_tool_call_traces,
    record_agent_observation,
    tool_call_batch_rejection,
)
from trace_models import FailureLayer, ToolTrace


MAX_TOOL_CALLS = 3
SAFE_AGENT_ANSWER = "暂时无法完成本次咨询，请稍后重试。"


class AgentLoop:
    def __init__(self, *, executor: ToolExecutor, complete_chat) -> None:
        self._executor = executor
        self._complete_chat = complete_chat
        self._tool_schemas = llm_tool_schemas()

    def run(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        deadline: AgentDeadline,
    ) -> AgentResult:
        state = build_agent_messages(messages)
        processed_calls = 0
        successful_observations: dict[str, ToolObservation] = {}
        tool_traces: tuple[ToolTrace, ...] = ()
        successful_sources = ()
        pending_failures: tuple[tuple[FailureLayer, str], ...] = ()

        while True:
            tools_enabled = processed_calls < MAX_TOOL_CALLS
            completion_layer = (
                FailureLayer.TOOL_SELECTION
                if tools_enabled
                else FailureLayer.GENERATION
            )
            ensure_agent_active(
                deadline,
                layer=completion_layer,
                tool_traces=tool_traces,
            )
            try:
                if tools_enabled:
                    completion = self._complete_chat(
                        state,
                        tools=self._tool_schemas,
                    )
                else:
                    completion = self._complete_chat(state)
            except Exception as error:
                attach_agent_error_metadata(
                    error,
                    layer=completion_layer,
                    tool_traces=tool_traces,
                )
                raise
            ensure_agent_active(
                deadline,
                layer=completion_layer,
                tool_traces=tool_traces,
            )

            turn = normalize_agent_turn(completion)
            if turn is None:
                return finalize_agent_result(
                    SAFE_AGENT_ANSWER,
                    tool_traces=tool_traces,
                    successful_sources=successful_sources,
                    pending_failures=pending_failures,
                    failure=(
                        FailureLayer.TOOL_SELECTION
                        if tools_enabled
                        else FailureLayer.GENERATION
                    ),
                )

            if turn.tool_calls:
                rejection = tool_call_batch_rejection(
                    turn.tool_calls,
                    processed_calls=processed_calls,
                    max_tool_calls=MAX_TOOL_CALLS,
                )
                if not tools_enabled or rejection is not None:
                    tool_traces = (
                        *tool_traces,
                        *invalid_tool_call_traces(turn.tool_calls),
                    )
                    return finalize_agent_result(
                        SAFE_AGENT_ANSWER,
                        tool_traces=tool_traces,
                        successful_sources=successful_sources,
                        pending_failures=pending_failures,
                        failure=(
                            FailureLayer.GENERATION
                            if not tools_enabled
                            else FailureLayer.TOOL_SELECTION
                        ),
                    )

                processed_calls += len(turn.tool_calls)
                state.append(assistant_tool_message(turn))

                for call in turn.tool_calls:
                    ensure_agent_active(
                        deadline,
                        layer=FailureLayer.TOOL_EXECUTION,
                        tool_traces=tool_traces,
                    )
                    observation = self._executor.execute(
                        call,
                        successful_observations,
                    )
                    (
                        tool_traces,
                        successful_sources,
                        pending_failures,
                    ) = record_agent_observation(
                        observation,
                        tool_traces=tool_traces,
                        successful_sources=successful_sources,
                        pending_failures=pending_failures,
                    )
                    ensure_agent_active(
                        deadline,
                        layer=FailureLayer.TOOL_EXECUTION,
                        tool_traces=tool_traces,
                    )
                    state.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": observation.content,
                    })
                continue

            if turn.content is None or not turn.content.strip():
                return finalize_agent_result(
                    SAFE_AGENT_ANSWER,
                    tool_traces=tool_traces,
                    successful_sources=successful_sources,
                    pending_failures=pending_failures,
                    failure=(
                        FailureLayer.TOOL_SELECTION
                        if tools_enabled
                        else FailureLayer.GENERATION
                    ),
                )
            return finalize_agent_result(
                turn.content,
                tool_traces=tool_traces,
                successful_sources=successful_sources,
                pending_failures=pending_failures,
            )


__all__ = [
    "AGENT_TOOL_POLICY",
    "AgentLoop",
    "MAX_TOOL_CALLS",
    "SAFE_AGENT_ANSWER",
]
