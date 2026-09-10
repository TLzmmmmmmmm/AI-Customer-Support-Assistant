from __future__ import annotations

from collections.abc import Sequence

from agent import (
    AgentDeadline,
    AgentLoop,
    SAFE_AGENT_ANSWER,
    ToolExecutor,
)
from agent.tool_outcomes import (
    tool_trace_from_observation,
)
from models import ChatMessage
from prompts import build_direct_messages, build_rag_messages, build_tool_messages
from rag_context import build_retrieved_context
from trace_models import FailureLayer, ToolTrace

from .deterministic import (
    DETERMINISTIC_TOOL_ROUTES,
    execute_deterministic_route,
)
from .errors import set_failure_layer as _set_failure_layer
from .finalization import SAFE_FALLBACK_ANSWER, finalize_non_agentic_route
from .generation import generate_answer
from .knowledge import retrieval_sources, retrieve_knowledge
from .models import Route, RouteExecutionResult, RouteTrace


def _set_error_context(
    error: Exception,
    *,
    route: Route,
    tool_calls: tuple[ToolTrace, ...] = (),
    retrieved_chunk_ids: tuple[str, ...] = (),
) -> None:
    for name, value in (
        ("route", route),
        ("tool_calls", tool_calls),
        ("retrieved_chunk_ids", retrieved_chunk_ids),
    ):
        try:
            setattr(error, name, value)
        except (AttributeError, TypeError):
            pass


class RouteOrchestrator:
    def __init__(
        self,
        *,
        router,
        retriever,
        executor: ToolExecutor,
        agent_loop: AgentLoop,
        complete_chat,
    ) -> None:
        self._router = router
        self._retriever = retriever
        self._executor = executor
        self._agent_loop = agent_loop
        self._complete_chat = complete_chat

    def run(
        self,
        messages: Sequence[ChatMessage],
        *,
        deadline: AgentDeadline,
    ) -> RouteExecutionResult:
        try:
            routing = self._router.route(messages, deadline=deadline)
        except Exception as error:
            _set_failure_layer(error, FailureLayer.ROUTING)
            raise

        decision = routing.decision
        if decision.route == Route.FALLBACK:
            return finalize_non_agentic_route(
                decision.route,
                routing_failure=routing.failure_layer,
            )

        retrieved = []
        if decision.route == Route.KNOWLEDGE:
            try:
                retrieved = retrieve_knowledge(
                    messages,
                    deadline=deadline,
                    retriever=self._retriever,
                )
                provider_messages = build_rag_messages(
                    messages,
                    build_retrieved_context(retrieved),
                )
            except Exception as error:
                _set_failure_layer(error, FailureLayer.RETRIEVAL)
                _set_error_context(error, route=decision.route)
                raise
        else:
            provider_messages = build_direct_messages(messages)

        retrieved_ids = tuple(item.chunk_id for item in retrieved)
        retrieved_sources = retrieval_sources(retrieved)
        if decision.agentic:
            try:
                result = self._agent_loop.run(provider_messages, deadline=deadline)
            except Exception as error:
                _set_error_context(
                    error,
                    route=decision.route,
                    tool_calls=tuple(getattr(error, "tool_calls", ())),
                    retrieved_chunk_ids=retrieved_ids,
                )
                raise
            return RouteExecutionResult(
                answer=result.answer,
                trace=RouteTrace(
                    route=decision.route,
                    tool_calls=result.tool_calls,
                    tool_call_count=len(result.tool_calls),
                    retrieved_chunk_ids=retrieved_ids,
                    failure_layer=result.failure_layer,
                ),
                sources=(
                    ()
                    if result.answer == SAFE_AGENT_ANSWER
                    else retrieved_sources + result.sources
                ),
            )

        observation = None
        if decision.route in DETERMINISTIC_TOOL_ROUTES:
            try:
                observation = execute_deterministic_route(
                    decision,
                    messages,
                    deadline=deadline,
                    executor=self._executor,
                )
            except Exception as error:
                _set_failure_layer(error, FailureLayer.TOOL_EXECUTION)
                _set_error_context(error, route=decision.route)
                raise
            provider_messages = build_tool_messages(
                messages,
                route=decision.route.value,
                observation=observation.content,
            )

        try:
            answer, generation_failure = generate_answer(
                provider_messages,
                deadline=deadline,
                complete_chat=self._complete_chat,
            )
        except Exception as error:
            _set_failure_layer(error, FailureLayer.GENERATION)
            _set_error_context(
                error,
                route=decision.route,
                tool_calls=(
                    ()
                    if observation is None
                    else (tool_trace_from_observation(observation),)
                ),
                retrieved_chunk_ids=retrieved_ids,
            )
            raise
        return finalize_non_agentic_route(
            decision.route,
            answer=answer,
            retrieval_results=retrieved,
            tool_observation=observation,
            generation_failure=generation_failure,
        )

__all__ = ["RouteOrchestrator", "SAFE_FALLBACK_ANSWER"]
