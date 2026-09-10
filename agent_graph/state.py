from __future__ import annotations

from typing import NotRequired, TypedDict

from agent import AgentDeadline, AgentToolCall, ToolObservation
from knowledge_pipeline.models import SourceRef
from knowledge_pipeline.retrieval.models import RetrievalResult
from models import ChatMessage
from routing import RouteDecision, RouteExecutionResult
from trace_models import FailureLayer


class AgentState(TypedDict):
    messages: tuple[ChatMessage, ...]
    last_user_message: ChatMessage
    deadline: AgentDeadline
    route_decision: NotRequired[RouteDecision]
    routing_failure: NotRequired[FailureLayer | None]
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
