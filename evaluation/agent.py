"""Minimal offline observations for agent action evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, TypedDict

from agent import AgentDeadline, ToolExecutor, ToolObservation
from agent.models import TOOL_SPECS
from models import ChatMessage
from routing import Route, RouteExecutionResult


class RouteExecutionRunner(Protocol):
    def run(
        self,
        messages: Sequence[ChatMessage],
        *,
        deadline: AgentDeadline,
    ) -> RouteExecutionResult:
        ...


class ValidatedToolCall(TypedDict):
    name: str
    arguments: dict[str, object]


class AgentEvaluationObservation(TypedDict):
    route: Route
    tool_calls: tuple[ValidatedToolCall, ...]
    final_answer: str


class RecordingToolExecutor(ToolExecutor):
    """Observe validated calls without changing production execution."""

    def __init__(
        self,
        registry: Mapping[str, Callable[..., object]],
    ) -> None:
        super().__init__(registry)
        self._validated_tool_calls: list[ValidatedToolCall] = []

    @property
    def validated_tool_calls(self) -> tuple[ValidatedToolCall, ...]:
        return tuple(self._validated_tool_calls)

    def execute_named(
        self,
        name: str,
        arguments: Mapping[str, object],
        successful_observations: dict[str, ToolObservation],
    ) -> ToolObservation:
        observation = super().execute_named(
            name,
            arguments,
            successful_observations,
        )

        try:
            spec = TOOL_SPECS.get(name)
            if spec is not None:
                validated = spec.arguments_model.model_validate(arguments)
                self._validated_tool_calls.append({
                    "name": name,
                    "arguments": validated.model_dump(mode="json"),
                })
        except Exception:
            # Evaluation observation must never alter production behavior.
            pass

        return observation


class AgentEvaluationRunner:
    def __init__(
        self,
        orchestrator: RouteExecutionRunner,
        executor: RecordingToolExecutor,
    ) -> None:
        self._orchestrator = orchestrator
        self._executor = executor

    def run(
        self,
        messages: Sequence[ChatMessage],
        *,
        deadline: AgentDeadline,
    ) -> AgentEvaluationObservation:
        call_offset = len(self._executor.validated_tool_calls)
        result = self._orchestrator.run(messages, deadline=deadline)
        calls = self._executor.validated_tool_calls[call_offset:]
        return {
            "route": result.trace.route,
            "tool_calls": calls,
            "final_answer": result.answer,
        }


def _call_matches(
    actual: ValidatedToolCall,
    expected: ValidatedToolCall,
) -> bool:
    if actual["name"] != expected["name"]:
        return False
    return all(
        key in actual["arguments"] and actual["arguments"][key] == value
        for key, value in expected["arguments"].items()
    )


def tool_calls_match(
    actual: Sequence[ValidatedToolCall],
    expected: Sequence[ValidatedToolCall],
) -> bool:
    """Return whether a complete one-to-one valid call matching exists."""
    if len(actual) != len(expected):
        return False

    def match_remaining(
        expected_index: int,
        remaining_actual: tuple[ValidatedToolCall, ...],
    ) -> bool:
        if expected_index == len(expected):
            return True
        for index, actual_call in enumerate(remaining_actual):
            if _call_matches(actual_call, expected[expected_index]) and match_remaining(
                expected_index + 1,
                remaining_actual[:index] + remaining_actual[index + 1:],
            ):
                return True
        return False

    return match_remaining(0, tuple(actual))


__all__ = [
    "AgentEvaluationObservation",
    "AgentEvaluationRunner",
    "RecordingToolExecutor",
    "RouteExecutionRunner",
    "ValidatedToolCall",
    "tool_calls_match",
]
