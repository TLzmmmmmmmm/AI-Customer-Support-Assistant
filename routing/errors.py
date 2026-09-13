from __future__ import annotations

from trace_models import FailureLayer, ToolTrace

from .models import Route


def set_failure_layer(error: Exception, layer: FailureLayer) -> None:
    try:
        error.failure_layer = layer
    except (AttributeError, TypeError):
        pass


def set_error_context(
    error: Exception,
    *,
    route: Route,
    tool_calls: tuple[ToolTrace, ...] = (),
    retrieved_chunk_ids: tuple[str, ...] = (),
) -> None:
    for name, value in (
        ("route", route),
        ("tool_calls", tool_calls),
        ("retrieved_chunk_ids", retrieved_chunk_ids),
    ):
        try:
            setattr(error, name, value)
        except (AttributeError, TypeError):
            pass


__all__ = ["set_error_context", "set_failure_layer"]
