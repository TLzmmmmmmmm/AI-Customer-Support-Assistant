from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Route(str, Enum):
    PRODUCT_SEARCH = "product_search"
    EXACT_PRODUCT = "exact_product"
    CONTACT = "contact"
    KNOWLEDGE = "knowledge"
    DIRECT = "direct"
    FALLBACK = "fallback"


class FailureLayer(str, Enum):
    ROUTING = "ROUTING"
    TOOL_SELECTION = "TOOL_SELECTION"
    ARGUMENT_GENERATION = "ARGUMENT_GENERATION"
    TOOL_EXECUTION = "TOOL_EXECUTION"
    RETRIEVAL = "RETRIEVAL"
    GENERATION = "GENERATION"


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    agentic: bool = False
    product_id: str | None = None


@dataclass(frozen=True)
class ToolTrace:
    name: str
    success: bool
    error_code: str | None = None
    reused: bool = False


@dataclass(frozen=True)
class RouteTrace:
    route: Route
    tool_calls: tuple[ToolTrace, ...] = ()
    tool_call_count: int = 0
    retrieved_chunk_ids: tuple[str, ...] = ()
    failure_layer: FailureLayer | None = None


@dataclass(frozen=True)
class RouteExecutionResult:
    answer: str
    trace: RouteTrace


__all__ = [
    "FailureLayer",
    "Route",
    "RouteDecision",
    "RouteExecutionResult",
    "RouteTrace",
    "ToolTrace",
]
