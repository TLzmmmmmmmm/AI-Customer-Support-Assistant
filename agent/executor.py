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
        spec = TOOL_SPECS.get(call.name)
        if spec is None:
            return _error_observation(_invalid_argument("tool_executor"))

        try:
            decoded = json.loads(call.arguments)
            if not isinstance(decoded, dict):
                raise ValueError("tool arguments must be an object")
            arguments = spec.arguments_model.model_validate(decoded)
        except (json.JSONDecodeError, ValidationError, ValueError):
            return _error_observation(_invalid_argument(call.name))

        validated_arguments = arguments.model_dump(mode="json")
        cache_key = json.dumps(
            {"name": call.name, "arguments": validated_arguments},
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
            )

        tool = self._registry.get(call.name)
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
                tool_name=call.name,
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
        )


__all__ = ["ToolExecutor", "ToolObservation"]
