from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

from agent import ToolExecutor
from knowledge_pipeline.retrieval import Retriever
from routing import HybridRouter, Route
from routing.deterministic import (
    DETERMINISTIC_TOOL_ROUTES,
    execute_deterministic_route,
)
from routing.knowledge import retrieval_sources, retrieve_knowledge

from .state import AgentState


AgentGraphNode: TypeAlias = Callable[[AgentState], dict[str, object]]


@dataclass(frozen=True)
class AgentGraphNodes:
    router: HybridRouter
    executor: ToolExecutor
    retriever: Retriever
    rag_generate: AgentGraphNode
    agent_step: AgentGraphNode
    execute_tool: AgentGraphNode
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


def deterministic_tool_node(
    state: AgentState,
    *,
    executor: ToolExecutor,
) -> dict[str, object]:
    observation = execute_deterministic_route(
        state["route_decision"],
        state["messages"],
        deadline=state["deadline"],
        executor=executor,
    )
    return {
        "tool_result": observation,
        "sources": observation.sources if observation.success else (),
    }


def retrieve_node(
    state: AgentState,
    *,
    retriever: Retriever,
) -> dict[str, object]:
    results = retrieve_knowledge(
        state["messages"],
        deadline=state["deadline"],
        retriever=retriever,
    )
    return {
        "retrieval_hits": tuple(results),
        "sources": retrieval_sources(results),
    }


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
    if decision.route in DETERMINISTIC_TOOL_ROUTES:
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
    "deterministic_tool_node",
    "retrieve_node",
    "route_node",
    "select_agent_step_edge",
    "select_route_edge",
]
