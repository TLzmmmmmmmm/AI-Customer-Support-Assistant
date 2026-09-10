from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from .nodes import (
    AgentGraphNodes,
    route_node,
    select_agent_step_edge,
    select_route_edge,
)
from .state import AgentState


def build_agent_graph(nodes: AgentGraphNodes):
    graph = StateGraph(AgentState)

    graph.add_node("route", partial(route_node, router=nodes.router))
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("rag_generate", nodes.rag_generate)
    graph.add_node("agent_step", nodes.agent_step)
    graph.add_node("execute_tool", nodes.execute_tool)
    graph.add_node("deterministic_tool", nodes.deterministic_tool)
    graph.add_node("direct", nodes.direct)
    graph.add_node("fallback", nodes.fallback)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        select_route_edge,
        {
            "agent_step": "agent_step",
            "deterministic_tool": "deterministic_tool",
            "retrieve": "retrieve",
            "direct": "direct",
            "fallback": "fallback",
        },
    )
    graph.add_edge("retrieve", "rag_generate")
    graph.add_edge("rag_generate", "finalize")
    graph.add_conditional_edges(
        "agent_step",
        select_agent_step_edge,
        {
            "execute_tool": "execute_tool",
            "finalize": "finalize",
        },
    )
    graph.add_edge("execute_tool", "agent_step")
    graph.add_edge("deterministic_tool", "finalize")
    graph.add_edge("direct", "finalize")
    graph.add_edge("fallback", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


__all__ = ["build_agent_graph"]
