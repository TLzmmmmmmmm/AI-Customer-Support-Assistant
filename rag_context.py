from collections.abc import Sequence

from citation import source_id
from knowledge_pipeline.retrieval.models import RetrievalResult


def build_retrieved_context(
    results: Sequence[RetrievalResult],
) -> list[dict[str, object]]:
    return [
        {
            "type": result.type,
            "section": result.section,
            "text": result.text,
            "source_ids": [source_id(source.url) for source in result.sources],
        }
        for result in results
    ]
