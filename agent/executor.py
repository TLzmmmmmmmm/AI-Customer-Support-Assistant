from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pydantic import ValidationError

from support_tools import ToolError, ToolErrorCode

from .models import AgentToolCall, TOOL_SPECS


@dataclass(frozen=True)
class ToolObservation:
    content: str
    success: bool
    reused: bool
    cache_key: str | None
    tool_name: str = "tool_executor"
    error_code: str | None = None


def _serialize_error(error: ToolError) -> str:
    return json.dumps(
        {
            "ok": False,
            "error": {
                "code": error.code.value,
                "message": error.message,
                "tool_name": error.tool_name,
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _error_observation(error: ToolError) -> ToolObservation:
    return ToolObservation(
        content=_serialize_error(error),
        success=False,
        reused=False,
        cache_key=None,
        tool_name=error.tool_name,
        error_code=error.code.value,
    )


def _invalid_argument(tool_name: str) -> ToolError:
    return ToolError(
        code=ToolErrorCode.INVALID_ARGUMENT,
        message="The proposed tool arguments are invalid.",
        tool_name=tool_name,
    )


class ToolExecutor:
    def __init__(self, registry: Mapping[str, Callable[..., object]]) -> None:
        self._registry = registry

    def execute(
        self,
        call: AgentToolCall,
        successful_observations: dict[str, str],
    ) -> ToolObservation:
        if call.name not in TOOL_SPECS:
            return _error_observation(_invalid_argument("tool_executor"))

        try:
            decoded = json.loads(call.arguments)
            if not isinstance(decoded, dict):
                raise ValueError("tool arguments must be an object")
        except (json.JSONDecodeError, ValueError):
            return _error_observation(_invalid_argument(call.name))

        return self.execute_named(
            call.name,
            decoded,
            successful_observations,
        )

    def execute_named(
        self,
        name: str,
        arguments: Mapping[str, object],
        successful_observations: dict[str, str],
    ) -> ToolObservation:
        spec = TOOL_SPECS.get(name)
        if spec is None:
            return _error_observation(_invalid_argument("tool_executor"))

        try:
            validated = spec.arguments_model.model_validate(arguments)
        except ValidationError:
            return _error_observation(_invalid_argument(name))

        validated_arguments = validated.model_dump(mode="json")
        cache_key = json.dumps(
            {"name": name, "arguments": validated_arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        cached = successful_observations.get(cache_key)
        if cached is not None:
            return ToolObservation(
                content=cached,
                success=True,
                reused=True,
                cache_key=cache_key,
                tool_name=name,
            )

        tool = self._registry.get(name)
        if tool is None:
            return _error_observation(_invalid_argument("tool_executor"))

        try:
            result = tool(**validated_arguments)
            if not isinstance(result, spec.result_model):
                raise TypeError("unexpected tool result type")
        except ToolError as error:
            return _error_observation(error)
        except Exception:
            return _error_observation(ToolError(
                code=ToolErrorCode.TOOL_EXECUTION_ERROR,
                message="The tool could not complete the request.",
                tool_name=name,
            ))

        content = json.dumps(
            {
                "ok": True,
                "result": result.model_dump(mode="json"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        successful_observations[cache_key] = content
        return ToolObservation(
            content=content,
            success=True,
            reused=False,
            cache_key=cache_key,
            tool_name=name,
        )


__all__ = ["ToolExecutor", "ToolObservation"]
