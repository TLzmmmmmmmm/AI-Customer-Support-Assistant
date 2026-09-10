from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from agent import ToolExecutor, ToolObservation
from agent.tool_outcomes import (
    successful_tool_sources,
    tool_failure_from_trace,
    tool_trace_from_observation,
)
from knowledge_pipeline.retrieval import Retriever
from prompts import build_direct_messages, build_rag_messages, build_tool_messages
from rag_context import build_retrieved_context
from routing import SAFE_FALLBACK_ANSWER, HybridRouter, Route
from routing.deterministic import (
    DETERMINISTIC_TOOL_ROUTES,
    execute_deterministic_route,
)
from routing.generation import generate_answer
from routing.finalization import finalize_non_agentic_route
from routing.knowledge import retrieval_sources, retrieve_knowledge

from .state import AgentState


AgentGraphNode: TypeAlias = Callable[[AgentState], dict[str, object]]
CompleteChat: TypeAlias = Callable[[Sequence[Mapping[str, object]]], object]


@dataclass(frozen=True)
class AgentGraphNodes:
    router: HybridRouter
    executor: ToolExecutor
    retriever: Retriever
    complete_chat: CompleteChat
    agent_step: AgentGraphNode
    execute_tool: AgentGraphNode


def route_node(
    state: AgentState,
    *,
    router: HybridRouter,
) -> dict[str, object]:
    routing = router.route(
        state["messages"],
        deadline=state["deadline"],
    )
    return {
        "route_decision": routing.decision,
        "routing_failure": routing.failure_layer,
    }


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
    trace = tool_trace_from_observation(observation)
    return {
        "tool_result": observation,
        "tool_failure": tool_failure_from_trace(trace),
        "sources": successful_tool_sources(observation),
    }


def deterministic_generate_node(
    state: AgentState,
    *,
    complete_chat: CompleteChat,
) -> dict[str, object]:
    decision = state["route_decision"]
    observation = cast(ToolObservation, state["tool_result"])
    provider_messages = build_tool_messages(
        state["messages"],
        route=decision.route.value,
        observation=observation.content,
    )
    answer, generation_failure = generate_answer(
        provider_messages,
        deadline=state["deadline"],
        complete_chat=complete_chat,
    )
    return {
        "answer": answer,
        "generation_failure": generation_failure,
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


def rag_generate_node(
    state: AgentState,
    *,
    complete_chat: CompleteChat,
) -> dict[str, object]:
    provider_messages = build_rag_messages(
        state["messages"],
        build_retrieved_context(state["retrieval_hits"]),
    )
    answer, generation_failure = generate_answer(
        provider_messages,
        deadline=state["deadline"],
        complete_chat=complete_chat,
    )
    return {
        "answer": answer,
        "generation_failure": generation_failure,
    }


def direct_node(
    state: AgentState,
    *,
    complete_chat: CompleteChat,
) -> dict[str, object]:
    answer, generation_failure = generate_answer(
        build_direct_messages(state["messages"]),
        deadline=state["deadline"],
        complete_chat=complete_chat,
    )
    return {
        "answer": answer,
        "generation_failure": generation_failure,
    }


def fallback_node(state: AgentState) -> dict[str, object]:
    return {"answer": SAFE_FALLBACK_ANSWER}


def finalize_node(state: AgentState) -> dict[str, object]:
    decision = state["route_decision"]
    result = finalize_non_agentic_route(
        decision.route,
        answer=state["answer"],
        routing_failure=state.get("routing_failure"),
        retrieval_results=state["retrieval_hits"],
        tool_observation=state["tool_result"],
        generation_failure=state.get("generation_failure"),
    )
    return {"result": result}


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
) -> Literal["execute_tool", "end"]:
    if state["tool_call"] is not None:
        return "execute_tool"
    return "end"


__all__ = [
    "AgentGraphNode",
    "AgentGraphNodes",
    "deterministic_generate_node",
    "deterministic_tool_node",
    "direct_node",
    "fallback_node",
    "finalize_node",
    "rag_generate_node",
    "retrieve_node",
    "route_node",
    "select_agent_step_edge",
    "select_route_edge",
]
