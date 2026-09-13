from __future__ import annotations

from typing import NotRequired, TypedDict

from agent import AgentDeadline, AgentToolCall, ToolObservation
from knowledge_pipeline.models import SourceRef
from knowledge_pipeline.retrieval.models import RetrievalResult
from models import ChatMessage
from routing import RouteDecision, RouteExecutionResult
from trace_models import FailureLayer, ToolTrace


class AgentState(TypedDict):
    messages: tuple[ChatMessage, ...]
    last_user_message: ChatMessage
    deadline: AgentDeadline
    route_decision: NotRequired[RouteDecision]
    routing_failure: NotRequired[FailureLayer | None]
    agent_messages: NotRequired[list[dict[str, object]]]
    agent_processed_calls: NotRequired[int]
    agent_pending_tool_calls: NotRequired[tuple[AgentToolCall, ...]]
    agent_successful_observations: NotRequired[dict[str, ToolObservation]]
    agent_tool_traces: NotRequired[tuple[ToolTrace, ...]]
    agent_pending_failures: NotRequired[
        tuple[tuple[FailureLayer, str], ...]
    ]
    agent_successful_sources: NotRequired[tuple[SourceRef, ...]]
    agent_failure: NotRequired[FailureLayer | None]
    retrieval_hits: tuple[RetrievalResult, ...]
    tool_call: AgentToolCall | None
    tool_result: ToolObservation | None
    tool_failure: NotRequired[FailureLayer | None]
    generation_failure: NotRequired[FailureLayer | None]
    result: NotRequired[RouteExecutionResult]
    answer: str | None
    sources: tuple[SourceRef, ...]
    error: Exception | None


__all__ = ["AgentState"]
