from __future__ import annotations

from collections.abc import Sequence

from agent import (
    AgentDeadline,
    AgentLoop,
    SAFE_AGENT_ANSWER,
    ToolExecutor,
    normalize_agent_turn,
)
from models import ChatMessage
from prompts import build_direct_messages, build_rag_messages, build_tool_messages
from rag_context import build_retrieved_context
from trace_models import FailureLayer, ToolTrace

from .models import Route, RouteExecutionResult, RouteTrace


SAFE_FALLBACK_ANSWER = "目前无法根据现有资料可靠确认这项信息。"


def _set_failure_layer(error: Exception, layer: FailureLayer) -> None:
    try:
        error.failure_layer = layer
    except (AttributeError, TypeError):
        pass


def _tool_trace(observation) -> ToolTrace:
    return ToolTrace(
        name=observation.tool_name,
        success=observation.success,
        error_code=observation.error_code,
        reused=observation.reused,
    )


def _tool_failure(trace: ToolTrace) -> FailureLayer | None:
    if trace.error_code in {"TOOL_EXECUTION_ERROR", "TOOL_UNAVAILABLE"}:
        return FailureLayer.TOOL_EXECUTION
    if trace.error_code == "INVALID_ARGUMENT":
        if trace.name == "tool_executor":
            return FailureLayer.TOOL_SELECTION
        return FailureLayer.ARGUMENT_GENERATION
    return None


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
            return RouteExecutionResult(
                answer=SAFE_FALLBACK_ANSWER,
                trace=RouteTrace(
                    route=decision.route,
                    failure_layer=routing.failure_layer,
                ),
            )

        retrieved = []
        if decision.route == Route.KNOWLEDGE:
            deadline.ensure_active()
            try:
                retrieved = self._retriever.retrieve(messages[-1].content)
            except Exception as error:
                _set_failure_layer(error, FailureLayer.RETRIEVAL)
                raise
            deadline.ensure_active()
            provider_messages = build_rag_messages(
                messages,
                build_retrieved_context(retrieved),
            )
        else:
            provider_messages = build_direct_messages(messages)

        retrieved_ids = tuple(item.chunk_id for item in retrieved)
        if decision.agentic:
            result = self._agent_loop.run(provider_messages, deadline=deadline)
            return RouteExecutionResult(
                answer=result.answer,
                trace=RouteTrace(
                    route=decision.route,
                    tool_calls=result.tool_calls,
                    tool_call_count=len(result.tool_calls),
                    retrieved_chunk_ids=retrieved_ids,
                    failure_layer=result.failure_layer,
                ),
            )

        tool_trace = None
        tool_failure = None
        if decision.route in {
            Route.EXACT_PRODUCT,
            Route.PRODUCT_SEARCH,
            Route.CONTACT,
        }:
            if decision.route == Route.EXACT_PRODUCT:
                name = "get_product_details"
                arguments = {"product_id": decision.product_id}
            elif decision.route == Route.PRODUCT_SEARCH:
                name = "search_products"
                arguments = {"query": messages[-1].content}
            else:
                name = "get_contact_info"
                arguments = {}

            deadline.ensure_active()
            observation = self._executor.execute_named(name, arguments, {})
            deadline.ensure_active()
            tool_trace = _tool_trace(observation)
            tool_failure = _tool_failure(tool_trace)
            provider_messages = build_tool_messages(
                messages,
                route=decision.route.value,
                observation=observation.content,
            )

        answer, generation_failure = self._generate(provider_messages, deadline)
        return RouteExecutionResult(
            answer=answer,
            trace=RouteTrace(
                route=decision.route,
                tool_calls=() if tool_trace is None else (tool_trace,),
                tool_call_count=0 if tool_trace is None else 1,
                retrieved_chunk_ids=retrieved_ids,
                failure_layer=tool_failure or generation_failure,
            ),
        )

    def _generate(self, messages, deadline):
        deadline.ensure_active()
        try:
            completion = self._complete_chat(messages)
        except Exception as error:
            _set_failure_layer(error, FailureLayer.GENERATION)
            raise
        deadline.ensure_active()
        turn = normalize_agent_turn(completion)
        if (
            turn is None
            or turn.tool_calls
            or turn.content is None
            or not turn.content.strip()
        ):
            return SAFE_AGENT_ANSWER, FailureLayer.GENERATION
        return turn.content, None


__all__ = ["RouteOrchestrator", "SAFE_FALLBACK_ANSWER"]
