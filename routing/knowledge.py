from collections.abc import Sequence

from agent import AgentDeadline
from knowledge_pipeline.models import SourceRef
from knowledge_pipeline.retrieval import Retriever
from knowledge_pipeline.retrieval.models import RetrievalResult
from models import ChatMessage


def retrieve_knowledge(
    messages: Sequence[ChatMessage],
    *,
    deadline: AgentDeadline,
    retriever: Retriever,
) -> list[RetrievalResult]:
    deadline.ensure_active()
    results = retriever.retrieve(messages[-1].content)
    deadline.ensure_active()
    return results


def retrieval_sources(
    results: Sequence[RetrievalResult],
) -> tuple[SourceRef, ...]:
    return tuple(
        source
        for result in results
        for source in result.sources
    )


__all__ = ["retrieve_knowledge", "retrieval_sources"]
