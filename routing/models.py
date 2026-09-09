from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from trace_models import FailureLayer, ToolTrace


class Route(str, Enum):
    PRODUCT_SEARCH = "product_search"
    EXACT_PRODUCT = "exact_product"
    CONTACT = "contact"
    KNOWLEDGE = "knowledge"
    DIRECT = "direct"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    agentic: bool = False
    product_id: str | None = None


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


@dataclass(frozen=True)
class RoutingResult:
    decision: RouteDecision
    failure_layer: FailureLayer | None = None


__all__ = [
    "FailureLayer",
    "Route",
    "RouteDecision",
    "RouteExecutionResult",
    "RouteTrace",
    "RoutingResult",
    "ToolTrace",
]
