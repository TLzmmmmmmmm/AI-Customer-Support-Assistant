from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from agent import (
    MAX_TOOL_CALLS,
    SAFE_AGENT_ANSWER,
    AgentToolCall,
    ToolExecutor,
    ToolObservation,
    llm_tool_schemas,
    normalize_agent_turn,
)
from agent.runtime import (
    assistant_tool_message,
    attach_agent_error_metadata,
    build_agent_messages,
    ensure_agent_active,
    finalize_agent_result,
    invalid_tool_call_traces,
    record_agent_observation,
)
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
from routing.errors import set_error_context, set_failure_layer
from routing.generation import generate_answer
from routing.finalization import (
    finalize_agentic_route,
    finalize_non_agentic_route,
)
from routing.knowledge import retrieval_sources, retrieve_knowledge
from trace_models import FailureLayer

from .state import AgentState


class CompleteChat(Protocol):
    def __call__(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        tools: Sequence[Mapping[str, object]] | None = None,
    ) -> object:
        ...


@dataclass(frozen=True)
class AgentGraphNodes:
    router: HybridRouter
    executor: ToolExecutor
    retriever: Retriever
    complete_chat: CompleteChat


def route_node(
    state: AgentState,
    *,
    router: HybridRouter,
) -> dict[str, object]:
    try:
        routing = router.route(
            state["messages"],
            deadline=state["deadline"],
        )
    except Exception as error:
        set_failure_layer(error, FailureLayer.ROUTING)
        raise
    return {
        "route_decision": routing.decision,
        "routing_failure": routing.failure_layer,
    }


def deterministic_tool_node(
    state: AgentState,
    *,
    executor: ToolExecutor,
) -> dict[str, object]:
    decision = state["route_decision"]
    try:
        observation = execute_deterministic_route(
            decision,
            state["messages"],
            deadline=state["deadline"],
            executor=executor,
        )
    except Exception as error:
        set_failure_layer(error, FailureLayer.TOOL_EXECUTION)
        set_error_context(error, route=decision.route)
        raise
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
    try:
        answer, generation_failure = generate_answer(
            provider_messages,
            deadline=state["deadline"],
            complete_chat=complete_chat,
        )
    except Exception as error:
        set_failure_layer(error, FailureLayer.GENERATION)
        set_error_context(
            error,
            route=decision.route,
            tool_calls=(tool_trace_from_observation(observation),),
        )
        raise
    return {
        "answer": answer,
        "generation_failure": generation_failure,
    }


def retrieve_node(
    state: AgentState,
    *,
    retriever: Retriever,
) -> dict[str, object]:
    try:
        results = retrieve_knowledge(
            state["messages"],
            deadline=state["deadline"],
            retriever=retriever,
        )
    except Exception as error:
        set_failure_layer(error, FailureLayer.RETRIEVAL)
        set_error_context(error, route=Route.KNOWLEDGE)
        raise
    return {
        "retrieval_hits": tuple(results),
        "sources": retrieval_sources(results),
    }


def rag_generate_node(
    state: AgentState,
    *,
    complete_chat: CompleteChat,
) -> dict[str, object]:
    try:
        provider_messages = build_rag_messages(
            state["messages"],
            build_retrieved_context(state["retrieval_hits"]),
        )
    except Exception as error:
        set_failure_layer(error, FailureLayer.RETRIEVAL)
        set_error_context(error, route=Route.KNOWLEDGE)
        raise
    try:
        answer, generation_failure = generate_answer(
            provider_messages,
            deadline=state["deadline"],
            complete_chat=complete_chat,
        )
    except Exception as error:
        set_failure_layer(error, FailureLayer.GENERATION)
        set_error_context(
            error,
            route=Route.KNOWLEDGE,
            retrieved_chunk_ids=tuple(
                result.chunk_id for result in state["retrieval_hits"]
            ),
        )
        raise
    return {
        "answer": answer,
        "generation_failure": generation_failure,
    }


def direct_node(
    state: AgentState,
    *,
    complete_chat: CompleteChat,
) -> dict[str, object]:
    provider_messages = build_direct_messages(state["messages"])
    try:
        answer, generation_failure = generate_answer(
            provider_messages,
            deadline=state["deadline"],
            complete_chat=complete_chat,
        )
    except Exception as error:
        set_failure_layer(error, FailureLayer.GENERATION)
        set_error_context(error, route=Route.DIRECT)
        raise
    return {
        "answer": answer,
        "generation_failure": generation_failure,
    }


def fallback_node(state: AgentState) -> dict[str, object]:
    return {"answer": SAFE_FALLBACK_ANSWER}


def agent_step_node(
    state: AgentState,
    *,
    complete_chat: CompleteChat,
) -> dict[str, object]:
    agent_messages = state.get("agent_messages")
    if agent_messages is None:
        if state["route_decision"].route == Route.KNOWLEDGE:
            try:
                provider_messages = build_rag_messages(
                    state["messages"],
                    build_retrieved_context(state["retrieval_hits"]),
                )
            except Exception as error:
                set_failure_layer(error, FailureLayer.RETRIEVAL)
                set_error_context(error, route=Route.KNOWLEDGE)
                raise
        else:
            provider_messages = build_direct_messages(state["messages"])
    try:
        if agent_messages is None:
            agent_messages = build_agent_messages(provider_messages)

        processed_calls = state.get("agent_processed_calls", 0)
        successful_observations = state.get(
            "agent_successful_observations",
            {},
        )
        tool_traces = state.get("agent_tool_traces", ())
        pending_failures = state.get("agent_pending_failures", ())
        successful_sources = state.get("agent_successful_sources", ())
        tools_enabled = processed_calls < MAX_TOOL_CALLS
        completion_layer = (
            FailureLayer.TOOL_SELECTION
            if tools_enabled
            else FailureLayer.GENERATION
        )

        ensure_agent_active(
            state["deadline"],
            layer=completion_layer,
            tool_traces=tool_traces,
        )
        if tools_enabled:
            try:
                completion = complete_chat(
                    agent_messages,
                    tools=llm_tool_schemas(),
                )
            except Exception as error:
                attach_agent_error_metadata(
                    error,
                    layer=completion_layer,
                    tool_traces=tool_traces,
                )
                raise
        else:
            try:
                completion = complete_chat(agent_messages)
            except Exception as error:
                attach_agent_error_metadata(
                    error,
                    layer=completion_layer,
                    tool_traces=tool_traces,
                )
                raise
        ensure_agent_active(
            state["deadline"],
            layer=completion_layer,
            tool_traces=tool_traces,
        )

        update: dict[str, object] = {
            "agent_messages": agent_messages,
            "agent_processed_calls": processed_calls,
            "agent_successful_observations": successful_observations,
            "agent_tool_traces": tool_traces,
            "agent_pending_failures": pending_failures,
            "agent_successful_sources": successful_sources,
        }
        turn = normalize_agent_turn(completion)
        if turn is None:
            update.update({
                "tool_call": None,
                "answer": SAFE_AGENT_ANSWER,
                "agent_failure": completion_layer,
            })
            return update

        if turn.tool_calls:
            if not tools_enabled or len(turn.tool_calls) != 1:
                update.update({
                    "agent_tool_traces": (
                        *tool_traces,
                        *invalid_tool_call_traces(turn.tool_calls),
                    ),
                    "tool_call": None,
                    "answer": SAFE_AGENT_ANSWER,
                    "agent_failure": (
                        FailureLayer.GENERATION
                        if not tools_enabled
                        else FailureLayer.TOOL_SELECTION
                    ),
                })
                return update

            update.update({
                "agent_messages": [
                    *agent_messages,
                    assistant_tool_message(turn),
                ],
                "agent_processed_calls": processed_calls + 1,
                "tool_call": turn.tool_calls[0],
                "answer": None,
                "agent_failure": None,
            })
            return update

        if turn.content is None or not turn.content.strip():
            update.update({
                "tool_call": None,
                "answer": SAFE_AGENT_ANSWER,
                "agent_failure": completion_layer,
            })
            return update

        update.update({
            "tool_call": None,
            "answer": turn.content,
            "agent_failure": None,
        })
        return update
    except Exception as error:
        route = state["route_decision"].route
        set_error_context(
            error,
            route=route,
            tool_calls=tuple(getattr(error, "tool_calls", ())),
            retrieved_chunk_ids=(
                tuple(
                    result.chunk_id
                    for result in state["retrieval_hits"]
                )
                if route == Route.KNOWLEDGE
                else ()
            ),
        )
        raise


def execute_tool_node(
    state: AgentState,
    *,
    executor: ToolExecutor,
) -> dict[str, object]:
    try:
        call = cast(AgentToolCall, state["tool_call"])
        tool_traces = state["agent_tool_traces"]
        successful_observations = dict(
            state["agent_successful_observations"]
        )
        ensure_agent_active(
            state["deadline"],
            layer=FailureLayer.TOOL_EXECUTION,
            tool_traces=tool_traces,
        )
        observation = executor.execute(call, successful_observations)
        (
            updated_traces,
            updated_sources,
            updated_failures,
        ) = record_agent_observation(
            observation,
            tool_traces=tool_traces,
            successful_sources=state["agent_successful_sources"],
            pending_failures=state["agent_pending_failures"],
        )
        ensure_agent_active(
            state["deadline"],
            layer=FailureLayer.TOOL_EXECUTION,
            tool_traces=updated_traces,
        )
        return {
            "agent_messages": [
                *state["agent_messages"],
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": observation.content,
                },
            ],
            "agent_successful_observations": successful_observations,
            "agent_tool_traces": updated_traces,
            "agent_pending_failures": updated_failures,
            "agent_successful_sources": updated_sources,
            "tool_call": None,
            "tool_result": observation,
        }
    except Exception as error:
        route = state["route_decision"].route
        set_error_context(
            error,
            route=route,
            tool_calls=tuple(getattr(error, "tool_calls", ())),
            retrieved_chunk_ids=(
                tuple(
                    result.chunk_id
                    for result in state["retrieval_hits"]
                )
                if route == Route.KNOWLEDGE
                else ()
            ),
        )
        raise


def agent_finalize_node(state: AgentState) -> dict[str, object]:
    agent_result = finalize_agent_result(
        cast(str, state["answer"]),
        tool_traces=state["agent_tool_traces"],
        successful_sources=state["agent_successful_sources"],
        pending_failures=state["agent_pending_failures"],
        failure=state.get("agent_failure"),
    )
    return {
        "result": finalize_agentic_route(
            state["route_decision"].route,
            agent_result=agent_result,
            retrieval_results=state["retrieval_hits"],
        ),
    }


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
    if decision.route == Route.FALLBACK:
        return "fallback"
    if decision.route == Route.KNOWLEDGE:
        return "retrieve"
    if decision.agentic:
        return "agent_step"
    if decision.route in DETERMINISTIC_TOOL_ROUTES:
        return "deterministic_tool"
    if decision.route == Route.DIRECT:
        return "direct"
    return "fallback"


def select_retrieve_edge(
    state: AgentState,
) -> Literal["agent_step", "rag_generate"]:
    if state["route_decision"].agentic:
        return "agent_step"
    return "rag_generate"


def select_agent_step_edge(
    state: AgentState,
) -> Literal["execute_tool", "agent_finalize"]:
    if state["tool_call"] is not None:
        return "execute_tool"
    return "agent_finalize"


__all__ = [
    "AgentGraphNodes",
    "CompleteChat",
    "agent_finalize_node",
    "agent_step_node",
    "deterministic_generate_node",
    "deterministic_tool_node",
    "direct_node",
    "execute_tool_node",
    "fallback_node",
    "finalize_node",
    "rag_generate_node",
    "retrieve_node",
    "route_node",
    "select_agent_step_edge",
    "select_retrieve_edge",
    "select_route_edge",
]
