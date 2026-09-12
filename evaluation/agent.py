"""Minimal offline observations for agent action evaluation."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypedDict

from agent import AgentDeadline, ToolExecutor, ToolObservation
from agent.models import TOOL_SPECS
from models import ChatMessage, ChatRequest
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
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


class ExpectedToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    required_args: dict[str, object]

    @field_validator("name")
    @classmethod
    def known_tool_name(cls, value: str) -> str:
        if value not in TOOL_SPECS:
            raise ValueError("unknown tool name")
        return value


class AgentEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    expected_route: Route
    expected_tools: tuple[ExpectedToolCall, ...]
    expected_answer: str

    @field_validator("case_id", "expected_answer")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def valid_conversation(self) -> AgentEvalCase:
        ChatRequest(messages=list(self.messages))
        return self


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


def load_agent_eval_cases(path: Path) -> tuple[AgentEvalCase, ...]:
    cases = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(f"blank JSONL line: {line_number}")
        try:
            cases.append(AgentEvalCase.model_validate(json.loads(line)))
        except Exception as error:
            raise ValueError(f"invalid agent evaluation case: {line_number}") from error
    if not cases:
        raise ValueError("agent evaluation dataset is empty")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate agent evaluation case_id")
    return tuple(cases)


def _expected_match_calls(
    expected_tools: Sequence[ExpectedToolCall],
) -> tuple[ValidatedToolCall, ...]:
    return tuple(
        {
            "name": tool.name,
            "arguments": dict(tool.required_args),
        }
        for tool in expected_tools
    )


def run_agent_evaluation(
    cases: Sequence[AgentEvalCase],
    runner: AgentEvaluationRunner,
    *,
    deadline_factory: Callable[[], AgentDeadline],
) -> dict[str, object]:
    if not cases:
        raise ValueError("agent evaluation requires at least one case")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate agent evaluation case_id")

    rows = []
    for case in cases:
        expected_tools = [
            tool.model_dump(mode="json")
            for tool in case.expected_tools
        ]
        try:
            observation = runner.run(
                case.messages,
                deadline=deadline_factory(),
            )
            actual_route = observation["route"].value
            actual_tool_calls = [
                {
                    "name": call["name"],
                    "arguments": dict(call["arguments"]),
                }
                for call in observation["tool_calls"]
            ]
            route_pass = observation["route"] == case.expected_route
            action_pass = tool_calls_match(
                observation["tool_calls"],
                _expected_match_calls(case.expected_tools),
            )
            row = {
                "case_id": case.case_id,
                "status": "completed",
                "expected_route": case.expected_route.value,
                "actual_route": actual_route,
                "expected_tools": expected_tools,
                "actual_tool_calls": actual_tool_calls,
                "route_pass": route_pass,
                "action_pass": action_pass,
                "final_answer": observation["final_answer"],
                "error_type": None,
            }
        except Exception as error:
            row = {
                "case_id": case.case_id,
                "status": "error",
                "expected_route": case.expected_route.value,
                "actual_route": None,
                "expected_tools": expected_tools,
                "actual_tool_calls": None,
                "route_pass": False,
                "action_pass": False,
                "final_answer": None,
                "error_type": type(error).__name__,
            }
        rows.append(row)

    total = len(rows)
    route_passed = sum(row["route_pass"] for row in rows)
    action_passed = sum(row["action_pass"] for row in rows)
    return {
        "summary": {
            "total_cases": total,
            "route_passed": route_passed,
            "route_accuracy": route_passed / total,
            "action_passed": action_passed,
            "action_accuracy": action_passed / total,
            "failed_case_ids": [
                row["case_id"]
                for row in rows
                if not row["route_pass"] or not row["action_pass"]
            ],
        },
        "cases": rows,
    }


__all__ = [
    "AgentEvalCase",
    "AgentEvaluationObservation",
    "AgentEvaluationRunner",
    "ExpectedToolCall",
    "RecordingToolExecutor",
    "RouteExecutionRunner",
    "ValidatedToolCall",
    "load_agent_eval_cases",
    "run_agent_evaluation",
    "tool_calls_match",
]
