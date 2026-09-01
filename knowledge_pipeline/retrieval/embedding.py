from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI

from .config import DashScopeCredentials, EmbeddingConfig
from .models import EmbeddingAPIError, EmbeddingBatch


DOCUMENT_BATCH_SIZE = 20


class EmbeddingProvider(Protocol):
    @property
    def config(self) -> EmbeddingConfig:
        raise NotImplementedError

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        raise NotImplementedError

    def embed_query(self, text: str) -> EmbeddingBatch:
        raise NotImplementedError


class DashScopeEmbeddingProvider:
    def __init__(
        self,
        config: EmbeddingConfig,
        credentials: DashScopeCredentials,
        *,
        client=None,
    ):
        self._config = config
        self._client = client or OpenAI(
            api_key=credentials.api_key,
            base_url=credentials.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    @property
    def config(self) -> EmbeddingConfig:
        return self._config

    def _request(self, input_value: str | list[str]):
        try:
            return self._client.embeddings.create(
                model=self.config.model,
                input=input_value,
                dimensions=self.config.dimensions,
                encoding_format="float",
            )
        except Exception as error:
            raise EmbeddingAPIError(
                "embedding provider request failed"
            ) from error

    def _parse_response(
        self,
        response,
        expected_count: int,
    ) -> EmbeddingBatch:
        rows = list(getattr(response, "data", ()))
        if len(rows) != expected_count:
            raise EmbeddingAPIError(
                "embedding provider returned an unexpected row count"
            )

        by_index: dict[int, tuple[float, ...]] = {}
        for row in rows:
            index = getattr(row, "index", None)
            if not isinstance(index, int) or index in by_index:
                raise EmbeddingAPIError(
                    "embedding provider returned invalid row indices"
                )
            raw_vector = getattr(row, "embedding", None)
            if not isinstance(raw_vector, (list, tuple)):
                raise EmbeddingAPIError(
                    "embedding provider returned a non-vector row"
                )
            try:
                vector = tuple(float(value) for value in raw_vector)
            except (TypeError, ValueError) as error:
                raise EmbeddingAPIError(
                    "embedding provider returned non-numeric vector values"
                ) from error
            if len(vector) != self.config.dimensions:
                raise EmbeddingAPIError(
                    "embedding provider returned the wrong vector dimension"
                )
            if any(not math.isfinite(value) for value in vector):
                raise EmbeddingAPIError(
                    "embedding provider returned non-finite vector values"
                )
            if not any(value != 0.0 for value in vector):
                raise EmbeddingAPIError(
                    "embedding provider returned a zero vector"
                )
            by_index[index] = vector

        if set(by_index) != set(range(expected_count)):
            raise EmbeddingAPIError(
                "embedding provider returned invalid row indices"
            )

        usage = getattr(response, "usage", None)
        input_tokens = None
        if usage is not None:
            for attribute in ("prompt_tokens", "input_tokens", "total_tokens"):
                value = getattr(usage, attribute, None)
                if isinstance(value, int):
                    input_tokens = value
                    break
        return EmbeddingBatch(
            vectors=tuple(by_index[index] for index in range(expected_count)),
            input_tokens=input_tokens,
        )

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        normalized = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in normalized):
            raise EmbeddingAPIError("document embedding inputs must be non-blank strings")
        if not normalized:
            return EmbeddingBatch(vectors=(), input_tokens=0)

        vectors: list[Sequence[float]] = []
        token_counts: list[int | None] = []
        for start in range(0, len(normalized), DOCUMENT_BATCH_SIZE):
            batch = normalized[start:start + DOCUMENT_BATCH_SIZE]
            parsed = self._parse_response(
                self._request(batch),
                expected_count=len(batch),
            )
            vectors.extend(parsed.vectors)
            token_counts.append(parsed.input_tokens)
        input_tokens = (
            sum(value for value in token_counts if value is not None)
            if all(value is not None for value in token_counts)
            else None
        )
        return EmbeddingBatch(
            vectors=tuple(vectors),
            input_tokens=input_tokens,
        )

    def embed_query(self, text: str) -> EmbeddingBatch:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingAPIError("query embedding input must be non-blank")
        return self._parse_response(
            self._request(text),
            expected_count=1,
        )


__all__ = [
    "DashScopeEmbeddingProvider",
    "EmbeddingProvider",
]
