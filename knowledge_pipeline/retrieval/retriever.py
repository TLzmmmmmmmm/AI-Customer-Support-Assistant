from __future__ import annotations

from collections.abc import Collection, Sequence

from knowledge_pipeline.models import source_ref_from_text

from .embedding import EmbeddingProvider
from .entities import ExactEntityResolver
from .index import VectorIndex
from .models import (
    EmbeddingAPIError,
    RetrievalResult,
    SearchHit,
    VectorRecordValidationError,
)


class Retriever:
    """Compose exact entity handling with backend-neutral dense retrieval."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_index: VectorIndex,
        entity_resolver: ExactEntityResolver,
        default_top_k: int = 5,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be greater than zero")
        self._embedding_provider = embedding_provider
        self._vector_index = vector_index
        self._entity_resolver = entity_resolver
        self._default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        allowed_types: Collection[str] | None = None,
        unique_parent_documents: bool = False,
    ) -> list[RetrievalResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-blank string")
        resolved_top_k = self._default_top_k if top_k is None else top_k
        if resolved_top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        batch = self._embedding_provider.embed_query(query)
        if len(batch.vectors) != 1:
            raise EmbeddingAPIError(
                "query embedding provider returned an unexpected vector count"
            )
        query_vector = batch.vectors[0]
        entity_matches = self._entity_resolver.resolve(query)
        entity_ids = [match.parent_document_id for match in entity_matches]

        assembled: list[tuple[SearchHit, str, list[str]]] = []
        seen_chunk_ids: set[str] = set()
        seen_parent_document_ids: set[str] = set()

        def seen_result_count() -> int:
            if unique_parent_documents:
                return len(seen_parent_document_ids)
            return len(seen_chunk_ids)

        def append_hit(
            hit: SearchHit,
            origin: str,
            matched_entity_ids: list[str],
        ) -> None:
            if (
                len(assembled) >= resolved_top_k
                or hit.record.chunk_id in seen_chunk_ids
                or (
                    unique_parent_documents
                    and hit.record.parent_document_id
                    in seen_parent_document_ids
                )
            ):
                return
            seen_chunk_ids.add(hit.record.chunk_id)
            seen_parent_document_ids.add(hit.record.parent_document_id)
            assembled.append((hit, origin, matched_entity_ids))

        for entity_id in entity_ids:
            hits = self._vector_index.search(
                query_vector,
                top_k=1,
                parent_document_ids={entity_id},
                record_types=allowed_types,
                unique_parent_documents=unique_parent_documents,
            )
            if hits:
                append_hit(hits[0], "exact_entity", [entity_id])

        if entity_ids and len(assembled) < resolved_top_k:
            entity_hits = self._vector_index.search(
                query_vector,
                top_k=resolved_top_k + seen_result_count(),
                parent_document_ids=set(entity_ids),
                record_types=allowed_types,
                unique_parent_documents=unique_parent_documents,
            )
            for hit in entity_hits:
                append_hit(
                    hit,
                    "exact_entity",
                    [hit.record.parent_document_id],
                )

        if len(assembled) < resolved_top_k:
            dense_hits = self._vector_index.search(
                query_vector,
                top_k=resolved_top_k + seen_result_count(),
                record_types=allowed_types,
                unique_parent_documents=unique_parent_documents,
            )
            for hit in dense_hits:
                append_hit(hit, "dense", [])

        return [
            _to_result(
                hit,
                rank=rank,
                origin=origin,
                matched_entity_ids=matched_entity_ids,
            )
            for rank, (hit, origin, matched_entity_ids) in enumerate(
                assembled,
                start=1,
            )
        ]


def _to_result(
    hit: SearchHit,
    *,
    rank: int,
    origin: str,
    matched_entity_ids: list[str],
) -> RetrievalResult:
    record = hit.record
    try:
        source = source_ref_from_text(record.text, record.source_url)
    except ValueError as error:
        raise VectorRecordValidationError(
            f"record {record.chunk_id} has invalid source title provenance"
        ) from error
    return RetrievalResult.model_validate({
        "rank": rank,
        "score": hit.score,
        "match_origin": origin,
        "matched_entity_ids": matched_entity_ids,
        "chunk_id": record.chunk_id,
        "parent_document_id": record.parent_document_id,
        "type": record.type,
        "section": record.section,
        "text": record.text,
        "content_hash": record.content_hash,
        "metadata": record.metadata,
        "source_url": record.source_url,
        "source_files": list(record.source_files),
        "sources": [source],
    })


def format_debug_results(
    query: str,
    results: Sequence[RetrievalResult],
) -> str:
    lines = [f"Query: {query}"]
    if not results:
        lines.append("No results.")
        return "\n".join(lines)

    for result in results:
        lines.extend([
            "",
            f"Rank: {result.rank}",
            f"Score: {result.score:.6f}",
            f"Origin: {result.match_origin}",
            "Matched entities: "
            + (", ".join(result.matched_entity_ids) or "-"),
            f"Chunk ID: {result.chunk_id}",
            f"Parent ID: {result.parent_document_id}",
            f"Section: {result.section}",
            f"Source: {result.source_url}",
            "Source files: " + ", ".join(result.source_files),
            "Text:",
            result.text,
        ])
    return "\n".join(lines)


__all__ = ["Retriever", "format_debug_results"]
