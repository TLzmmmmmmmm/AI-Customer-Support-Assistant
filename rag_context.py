from collections.abc import Sequence

from knowledge_pipeline.retrieval.models import RetrievalResult


def build_retrieved_context(
    results: Sequence[RetrievalResult],
) -> list[dict[str, str]]:
    return [
        {
            "type": result.type,
            "section": result.section,
            "text": result.text,
        }
        for result in results
    ]
