from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

from routing import HybridRouter, Route

from .state import AgentState


AgentGraphNode: TypeAlias = Callable[[AgentState], dict[str, object]]


@dataclass(frozen=True)
class AgentGraphNodes:
    router: HybridRouter
    retrieve: AgentGraphNode
    rag_generate: AgentGraphNode
    agent_step: AgentGraphNode
    execute_tool: AgentGraphNode
    deterministic_tool: AgentGraphNode
    direct: AgentGraphNode
    fallback: AgentGraphNode
    finalize: AgentGraphNode


def route_node(
    state: AgentState,
    *,
    router: HybridRouter,
) -> dict[str, object]:
    routing = router.route(
        state["messages"],
        deadline=state["deadline"],
    )
    return {"route_decision": routing.decision}


def select_route_edge(
    state: AgentState,
) -> Literal[
    "agent_step",
    "deterministic_tool",
    "retrieve",
    "direct",
    "fallback",
]:
    decision = state["route_decision"]
    if decision.agentic:
        return "agent_step"
    if decision.route in {
        Route.PRODUCT_SEARCH,
        Route.EXACT_PRODUCT,
        Route.CONTACT,
    }:
        return "deterministic_tool"
    if decision.route == Route.KNOWLEDGE:
        return "retrieve"
    if decision.route == Route.DIRECT:
        return "direct"
    return "fallback"


def select_agent_step_edge(
    state: AgentState,
) -> Literal["execute_tool", "finalize"]:
    if state["tool_call"] is not None:
        return "execute_tool"
    return "finalize"


__all__ = [
    "AgentGraphNode",
    "AgentGraphNodes",
    "route_node",
    "select_agent_step_edge",
    "select_route_edge",
]
