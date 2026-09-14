from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import Enum
from typing import Any


_current_request_state: ContextVar[object | None] = ContextVar(
    "current_request_state",
    default=None,
)

_REQUEST_TRACE_DEFAULTS: dict[str, object] = {
    "router_type": None,
    "retrieval_used": False,
    "retrieved_count": 0,
    "tool_execution_count": 0,
    "executed_tool_names": (),
    "tool_execution_success": None,
    "router_latency_ms": None,
    "retrieval_latency_ms": None,
    "tool_latency_ms": None,
    "model_latency_ms": None,
    "input_tokens": None,
    "output_tokens": None,
    "prompt_cache_hit_tokens": None,
    "prompt_cache_miss_tokens": None,
    "failure_code": None,
}


class FailureLayer(str, Enum):
    ROUTING = "ROUTING"
    TOOL_SELECTION = "TOOL_SELECTION"
    ARGUMENT_GENERATION = "ARGUMENT_GENERATION"
    TOOL_EXECUTION = "TOOL_EXECUTION"
    RETRIEVAL = "RETRIEVAL"
    GENERATION = "GENERATION"


@dataclass(frozen=True)
class ToolTrace:
    name: str
    success: bool
    error_code: str | None = None
    reused: bool = False


def initialize_request_trace(state: object) -> None:
    for name, value in _REQUEST_TRACE_DEFAULTS.items():
        setattr(state, name, value)
    state.input_tokens = 0
    state.output_tokens = 0
    state.prompt_cache_hit_tokens = 0
    state.prompt_cache_miss_tokens = 0
    state.model_usage_complete = True
    state.model_cache_usage_complete = True


def bind_request_state(state: object) -> Token:
    return _current_request_state.set(state)


def reset_request_state(token: Token) -> None:
    _current_request_state.reset(token)


def set_request_trace_field(name: str, value: object) -> None:
    state = _current_request_state.get()
    if state is None:
        return
    try:
        setattr(state, name, value)
    except Exception:
        pass


def add_request_duration(name: str, elapsed_ms: float) -> None:
    state = _current_request_state.get()
    if state is None:
        return
    try:
        current = getattr(state, name, None)
        setattr(
            state,
            name,
            elapsed_ms if current is None else current + elapsed_ms,
        )
    except Exception:
        pass


def record_model_response(completion: object) -> None:
    state = _current_request_state.get()
    if state is None:
        return
    try:
        usage = getattr(completion, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        cache_hit_tokens = getattr(
            usage,
            "prompt_cache_hit_tokens",
            None,
        )
        cache_miss_tokens = getattr(
            usage,
            "prompt_cache_miss_tokens",
            None,
        )
        valid = (
            isinstance(input_tokens, int)
            and not isinstance(input_tokens, bool)
            and isinstance(output_tokens, int)
            and not isinstance(output_tokens, bool)
        )
        if not valid:
            state.model_usage_complete = False
            state.input_tokens = None
            state.output_tokens = None
            state.model_cache_usage_complete = False
            state.prompt_cache_hit_tokens = None
            state.prompt_cache_miss_tokens = None
            return
        if getattr(state, "model_usage_complete", True):
            state.input_tokens += input_tokens
            state.output_tokens += output_tokens
        cache_valid = (
            isinstance(cache_hit_tokens, int)
            and not isinstance(cache_hit_tokens, bool)
            and cache_hit_tokens >= 0
            and isinstance(cache_miss_tokens, int)
            and not isinstance(cache_miss_tokens, bool)
            and cache_miss_tokens >= 0
            and cache_hit_tokens + cache_miss_tokens == input_tokens
        )
        if not cache_valid:
            state.model_cache_usage_complete = False
            state.prompt_cache_hit_tokens = None
            state.prompt_cache_miss_tokens = None
            return
        if getattr(state, "model_cache_usage_complete", True):
            state.prompt_cache_hit_tokens += cache_hit_tokens
            state.prompt_cache_miss_tokens += cache_miss_tokens
    except Exception:
        try:
            state.model_usage_complete = False
            state.input_tokens = None
            state.output_tokens = None
            state.model_cache_usage_complete = False
            state.prompt_cache_hit_tokens = None
            state.prompt_cache_miss_tokens = None
        except Exception:
            pass


def record_retrieval(count: int | None, elapsed_ms: float) -> None:
    state = _current_request_state.get()
    if state is None:
        return
    try:
        state.retrieval_used = True
        if count is not None:
            state.retrieved_count += count
        current = state.retrieval_latency_ms
        state.retrieval_latency_ms = (
            elapsed_ms if current is None else current + elapsed_ms
        )
    except Exception:
        pass


def record_tool_execution(
    name: str,
    success: bool,
    elapsed_ms: float,
) -> None:
    state = _current_request_state.get()
    if state is None:
        return
    try:
        state.tool_execution_count += 1
        state.executed_tool_names = (*state.executed_tool_names, name)
        current_success = state.tool_execution_success
        state.tool_execution_success = (
            success if current_success is None else current_success and success
        )
        current_latency = state.tool_latency_ms
        state.tool_latency_ms = (
            elapsed_ms
            if current_latency is None
            else current_latency + elapsed_ms
        )
    except Exception:
        pass


def request_trace_fields(state: object | None = None) -> dict[str, Any]:
    resolved = _current_request_state.get() if state is None else state
    return {
        name: (
            default
            if resolved is None
            else getattr(resolved, name, default)
        )
        for name, default in _REQUEST_TRACE_DEFAULTS.items()
    }


__all__ = [
    "FailureLayer",
    "ToolTrace",
    "add_request_duration",
    "bind_request_state",
    "initialize_request_trace",
    "record_model_response",
    "record_retrieval",
    "record_tool_execution",
    "request_trace_fields",
    "reset_request_state",
    "set_request_trace_field",
]
