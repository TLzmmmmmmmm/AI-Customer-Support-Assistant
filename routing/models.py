from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from knowledge_pipeline.models import SourceRef
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
    citation_count: int = 0
    deduplicated_citation_count: int = 0
    invalid_source_count: int = 0
    citation_status: str = "none"
    answer_sanitized: bool = False


@dataclass(frozen=True)
class RouteExecutionResult:
    answer: str
    trace: RouteTrace
    sources: tuple[SourceRef, ...] = ()


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
