from __future__ import annotations

from collections.abc import Sequence

from agent import AgentDeadline, ToolExecutor, ToolObservation
from models import ChatMessage

from .models import Route, RouteDecision


_TOOL_NAMES = {
    Route.PRODUCT_SEARCH: "search_products",
    Route.EXACT_PRODUCT: "get_product_details",
    Route.CONTACT: "get_contact_info",
}
DETERMINISTIC_TOOL_ROUTES = frozenset(_TOOL_NAMES)


def execute_deterministic_route(
    decision: RouteDecision,
    messages: Sequence[ChatMessage],
    *,
    deadline: AgentDeadline,
    executor: ToolExecutor,
) -> ToolObservation:
    name = _TOOL_NAMES[decision.route]
    if decision.route == Route.PRODUCT_SEARCH:
        arguments = {"query": messages[-1].content}
    elif decision.route == Route.EXACT_PRODUCT:
        arguments = {"product_id": decision.product_id}
    else:
        arguments = {}

    deadline.ensure_active()
    observation = executor.execute_named(name, arguments, {})
    deadline.ensure_active()
    return observation


__all__ = [
    "DETERMINISTIC_TOOL_ROUTES",
    "execute_deterministic_route",
]
