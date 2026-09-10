from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

from .state import AgentState, GraphBranch


AgentGraphNode: TypeAlias = Callable[[AgentState], dict[str, object]]


@dataclass(frozen=True)
class AgentGraphNodes:
    route: AgentGraphNode
    retrieve: AgentGraphNode
    rag_generate: AgentGraphNode
    agent_step: AgentGraphNode
    execute_tool: AgentGraphNode
    direct: AgentGraphNode
    fallback: AgentGraphNode
    finalize: AgentGraphNode


def select_graph_branch(state: AgentState) -> GraphBranch:
    return state["route"]


def select_agent_step_edge(
    state: AgentState,
) -> Literal["execute_tool", "finalize"]:
    if state["tool_call"] is not None:
        return "execute_tool"
    return "finalize"


__all__ = [
    "AgentGraphNode",
    "AgentGraphNodes",
    "select_agent_step_edge",
    "select_graph_branch",
]
