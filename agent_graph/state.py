from __future__ import annotations

from typing import Literal, TypedDict

from agent import AgentToolCall, ToolObservation
from knowledge_pipeline.models import SourceRef
from knowledge_pipeline.retrieval.models import RetrievalResult
from models import ChatMessage


GraphBranch = Literal["rag", "agent", "direct", "fallback"]


class AgentState(TypedDict):
    messages: tuple[ChatMessage, ...]
    last_user_message: ChatMessage
    route: GraphBranch
    retrieval_hits: tuple[RetrievalResult, ...]
    tool_call: AgentToolCall | None
    tool_result: ToolObservation | None
    answer: str | None
    sources: tuple[SourceRef, ...]
    error: Exception | None


__all__ = ["AgentState", "GraphBranch"]
