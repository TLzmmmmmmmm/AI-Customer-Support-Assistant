from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from .nodes import (
    AgentGraphNodes,
    agent_finalize_node,
    agent_step_node,
    deterministic_generate_node,
    deterministic_tool_node,
    direct_node,
    fallback_node,
    finalize_node,
    execute_tool_node,
    rag_generate_node,
    retrieve_node,
    route_node,
    select_agent_step_edge,
    select_retrieve_edge,
    select_route_edge,
)
from .state import AgentState


def build_agent_graph(nodes: AgentGraphNodes):
    graph = StateGraph(AgentState)

    graph.add_node("route", partial(route_node, router=nodes.router))
    graph.add_node(
        "retrieve",
        partial(retrieve_node, retriever=nodes.retriever),
    )
    graph.add_node(
        "rag_generate",
        partial(rag_generate_node, complete_chat=nodes.complete_chat),
    )
    graph.add_node(
        "agent_step",
        partial(agent_step_node, complete_chat=nodes.complete_chat),
    )
    graph.add_node(
        "execute_tool",
        partial(execute_tool_node, executor=nodes.executor),
    )
    graph.add_node("agent_finalize", agent_finalize_node)
    graph.add_node(
        "deterministic_tool",
        partial(deterministic_tool_node, executor=nodes.executor),
    )
    graph.add_node(
        "deterministic_generate",
        partial(
            deterministic_generate_node,
            complete_chat=nodes.complete_chat,
        ),
    )
    graph.add_node(
        "direct",
        partial(direct_node, complete_chat=nodes.complete_chat),
    )
    graph.add_node("fallback", fallback_node)
    graph.add_node("finalize", finalize_node)

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
    graph.add_conditional_edges(
        "retrieve",
        select_retrieve_edge,
        {
            "agent_step": "agent_step",
            "rag_generate": "rag_generate",
        },
    )
    graph.add_edge("rag_generate", "finalize")
    graph.add_conditional_edges(
        "agent_step",
        select_agent_step_edge,
        {
            "execute_tool": "execute_tool",
            "agent_finalize": "agent_finalize",
        },
    )
    graph.add_edge("execute_tool", "agent_step")
    graph.add_edge("deterministic_tool", "deterministic_generate")
    graph.add_edge("deterministic_generate", "finalize")
    graph.add_edge("direct", "finalize")
    graph.add_edge("fallback", "finalize")
    graph.add_edge("finalize", END)
    graph.add_edge("agent_finalize", END)

    return graph.compile()


__all__ = ["build_agent_graph"]
