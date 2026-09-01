from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Protocol

import numpy as np

from .models import SearchHit, VectorIndexNotReadyError, VectorRecord


class VectorIndex(Protocol):
    """Backend-neutral interface for exact or approximate vector indexes."""

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        parent_document_ids: Collection[str] | None = None,
    ) -> list[SearchHit]: ...


class NumpyExactVectorIndex:
    """In-memory exact cosine search for the small V1 corpus."""

    def __init__(self, records: Sequence[VectorRecord]) -> None:
        self._records = tuple(records)
        if not self._records:
            raise VectorIndexNotReadyError("vector index requires at least one record")

        chunk_ids = [record.chunk_id for record in self._records]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise VectorIndexNotReadyError("vector index contains duplicate chunk_id values")

        dimensions = {record.embedding_dimensions for record in self._records}
        if len(dimensions) != 1:
            raise VectorIndexNotReadyError("vector records have mixed dimensions")
        self._dimensions = dimensions.pop()

        matrix = np.asarray(
            [record.embedding for record in self._records],
            dtype=np.float32,
        ).copy()
        if matrix.shape != (len(self._records), self._dimensions):
            raise VectorIndexNotReadyError("vector record matrix has an invalid shape")
        if not np.all(np.isfinite(matrix)):
            raise VectorIndexNotReadyError("vector record matrix contains non-finite values")

        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms == 0.0):
            raise VectorIndexNotReadyError("vector record matrix contains a zero vector")
        self._normalized_matrix = matrix / norms[:, np.newaxis]

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        parent_document_ids: Collection[str] | None = None,
    ) -> list[SearchHit]:
        if top_k <= 0:
            raise VectorIndexNotReadyError("top_k must be greater than zero")

        query = np.asarray(query_vector, dtype=np.float32).copy()
        if query.shape != (self._dimensions,):
            raise VectorIndexNotReadyError(
                f"query vector must have {self._dimensions} dimensions"
            )
        if not np.all(np.isfinite(query)):
            raise VectorIndexNotReadyError("query vector contains non-finite values")
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            raise VectorIndexNotReadyError("query vector must have non-zero norm")
        normalized_query = query / norm

        if parent_document_ids is None:
            candidate_indexes = range(len(self._records))
        else:
            allowed = set(parent_document_ids)
            candidate_indexes = [
                index
                for index, record in enumerate(self._records)
                if record.parent_document_id in allowed
            ]
        candidate_indexes = list(candidate_indexes)
        if not candidate_indexes:
            return []

        scores = self._normalized_matrix[candidate_indexes] @ normalized_query
        ranked = sorted(
            zip(candidate_indexes, scores, strict=True),
            key=lambda item: (-float(item[1]), self._records[item[0]].chunk_id),
        )
        return [
            SearchHit(record=self._records[index], score=float(score))
            for index, score in ranked[:top_k]
        ]


__all__ = ["NumpyExactVectorIndex", "VectorIndex"]
