from __future__ import annotations

from collections.abc import Mapping, Sequence

from .executor import ToolExecutor
from .models import (
    AgentDeadline,
    AgentResult,
    AgentToolCall,
    AgentTurn,
    TOOL_SPECS,
    llm_tool_schemas,
)
from trace_models import FailureLayer, ToolTrace


MAX_TOOL_CALLS = 3
SAFE_AGENT_ANSWER = "暂时无法完成本次咨询，请稍后重试。"

AGENT_TOOL_POLICY = """
Use the provided deterministic tools under these rules:
- Exact product model parameters, features, and details require get_product_details.
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


def _normalize_turn(completion) -> AgentTurn | None:
    try:
        choices = completion.choices
        if len(choices) != 1:
            return None

        choice = choices[0]
        finish_reason = choice.finish_reason
        if finish_reason in {"length", "content_filter"}:
            return None

        message = choice.message
        content = message.content
        if content is not None and not isinstance(content, str):
            return None

        calls = []
        for raw_call in message.tool_calls or ():
            if raw_call.type != "function":
                return None
            call_id = raw_call.id
            name = raw_call.function.name
            arguments = raw_call.function.arguments
            if (
                not isinstance(call_id, str)
                or not call_id.strip()
                or not isinstance(name, str)
                or not name.strip()
                or not isinstance(arguments, str)
            ):
                return None
            calls.append(AgentToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
            ))
    except (AttributeError, TypeError):
        return None

    return AgentTurn(
        content=content,
        tool_calls=tuple(calls),
        finish_reason=finish_reason,
    )


def _assistant_tool_message(turn: AgentTurn) -> dict[str, object]:
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
        state = [dict(message) for message in messages]
        state.insert(1, {
            "role": "system",
            "content": AGENT_TOOL_POLICY,
        })
        processed_calls = 0
        successful_observations: dict[str, str] = {}
        tool_traces: list[ToolTrace] = []
        pending_failures: list[tuple[FailureLayer, str]] = []

        def result(answer: str, failure: FailureLayer | None = None) -> AgentResult:
            unresolved = failure
            if unresolved is None and pending_failures:
                unresolved = pending_failures[0][0]
            return AgentResult(
                answer=answer,
                tool_calls=tuple(tool_traces),
                failure_layer=unresolved,
            )

        def record(observation) -> None:
            trace = ToolTrace(
                name=observation.tool_name,
                success=observation.success,
                error_code=observation.error_code,
                reused=observation.reused,
            )
            tool_traces.append(trace)
            if trace.success:
                pending_failures[:] = [
                    item
                    for item in pending_failures
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
                ]
            elif trace.error_code == "INVALID_ARGUMENT":
                layer = (
                    FailureLayer.TOOL_SELECTION
                    if trace.name == "tool_executor"
                    else FailureLayer.ARGUMENT_GENERATION
                )
                pending_failures.append((layer, trace.name))
            elif trace.error_code in {
                "TOOL_EXECUTION_ERROR",
                "TOOL_UNAVAILABLE",
            }:
                pending_failures.append((FailureLayer.TOOL_EXECUTION, trace.name))

        while True:
            tools_enabled = processed_calls < MAX_TOOL_CALLS
            deadline.ensure_active()
            try:
                if tools_enabled:
                    completion = self._complete_chat(
                        state,
                        tools=self._tool_schemas,
                    )
                else:
                    completion = self._complete_chat(state)
            except Exception as error:
                error.failure_layer = (
                    FailureLayer.TOOL_SELECTION
                    if tools_enabled
                    else FailureLayer.GENERATION
                )
                raise
            deadline.ensure_active()

            turn = _normalize_turn(completion)
            if turn is None:
                return result(
                    SAFE_AGENT_ANSWER,
                    FailureLayer.TOOL_SELECTION
                    if tools_enabled
                    else FailureLayer.GENERATION,
                )

            if turn.tool_calls:
                if not tools_enabled or len(turn.tool_calls) != 1:
                    for call in turn.tool_calls:
                        tool_traces.append(ToolTrace(
                            name=(
                                call.name
                                if call.name in TOOL_SPECS
                                else "tool_executor"
                            ),
                            success=False,
                            error_code="INVALID_ARGUMENT",
                        ))
                    return result(
                        SAFE_AGENT_ANSWER,
                        FailureLayer.GENERATION
                        if not tools_enabled
                        else FailureLayer.TOOL_SELECTION,
                    )

                processed_calls += 1
                state.append(_assistant_tool_message(turn))

                deadline.ensure_active()
                observation = self._executor.execute(
                    turn.tool_calls[0],
                    successful_observations,
                )
                record(observation)
                deadline.ensure_active()
                state.append({
                    "role": "tool",
                    "tool_call_id": turn.tool_calls[0].id,
                    "content": observation.content,
                })
                continue

            if turn.content is None or not turn.content.strip():
                return result(
                    SAFE_AGENT_ANSWER,
                    FailureLayer.TOOL_SELECTION
                    if tools_enabled
                    else FailureLayer.GENERATION,
                )
            return result(turn.content)


__all__ = [
    "AGENT_TOOL_POLICY",
    "AgentLoop",
    "MAX_TOOL_CALLS",
    "SAFE_AGENT_ANSWER",
]
