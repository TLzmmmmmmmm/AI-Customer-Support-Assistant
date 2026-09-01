from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from knowledge_pipeline.models import (
    CHUNK_ID,
    SHA256_HEX,
    CompanyMetadata,
    ContactMetadata,
    KnowledgeChunk,
    Metadata,
    ProductMetadata,
    SolutionMetadata,
    StrictModel,
    SupportMetadata,
)


class RetrievalError(Exception):
    """Base error for the retrieval subsystem."""


class EmbeddingConfigurationError(RetrievalError):
    """Embedding or retrieval configuration is invalid."""


class EmbeddingAPIError(RetrievalError):
    """The configured embedding provider failed or returned invalid data."""


class VectorRecordValidationError(RetrievalError):
    """Persisted vector records are missing, stale, or invalid."""


class VectorIndexNotReadyError(RetrievalError):
    """A usable local vector index is not available."""


class EntityCatalogError(RetrievalError):
    """The exact-entity catalog cannot be built or queried safely."""


class RetrievalEvaluationError(RetrievalError):
    """The fixed retrieval evaluation is invalid or cannot run."""


class VectorRecord(StrictModel):
    schema_version: Literal["1.0"]
    chunk_id: str = Field(min_length=1)
    parent_document_id: str = Field(min_length=1)
    type: Literal["product", "solution", "support", "company", "contact"]
    section: str = Field(min_length=1)
    text: str = Field(min_length=1)
    language: Literal["zh-CN"]
    content_hash: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    source_files: list[str] = Field(min_length=1)
    metadata: Metadata
    embedding_provider: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_dimensions: int = Field(gt=0)
    embedding_text_type: Literal["document"]
    embedding: list[float] = Field(min_length=1)

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        if not CHUNK_ID.fullmatch(value):
            raise ValueError(
                "must contain colon-separated lowercase ASCII segments"
            )
        return value

    @field_validator("content_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_HEX.fullmatch(value):
            raise ValueError("must be a lowercase SHA-256 hex digest")
        return value

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("must be an absolute HTTP(S) URL")
        return value

    @field_validator("source_files")
    @classmethod
    def validate_source_files(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("must be sorted and unique")
        if any(not item.strip() for item in value):
            raise ValueError("must not contain blank paths")
        return value

    @field_validator("text")
    @classmethod
    def validate_lf_text(cls, value: str) -> str:
        if "\r" in value:
            raise ValueError("must use LF line endings")
        return value

    @model_validator(mode="after")
    def validate_identity_metadata_and_embedding(self) -> "VectorRecord":
        if not self.parent_document_id.startswith(f"{self.type}:"):
            raise ValueError("parent_document_id must start with '<type>:'")
        if not self.chunk_id.startswith(f"{self.parent_document_id}:"):
            raise ValueError("chunk_id must extend parent_document_id")
        expected_metadata_type = {
            "product": ProductMetadata,
            "solution": SolutionMetadata,
            "support": SupportMetadata,
            "company": CompanyMetadata,
            "contact": ContactMetadata,
        }[self.type]
        if not isinstance(self.metadata, expected_metadata_type):
            raise ValueError(
                f"metadata does not match vector record type {self.type}"
            )
        if len(self.embedding) != self.embedding_dimensions:
            raise ValueError(
                "embedding length must equal embedding_dimensions"
            )
        if any(not math.isfinite(value) for value in self.embedding):
            raise ValueError("embedding values must be finite")
        if not any(value != 0.0 for value in self.embedding):
            raise ValueError("embedding must have non-zero norm")
        return self


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: Sequence[Sequence[float]]
    input_tokens: int | None


@dataclass(frozen=True)
class VectorBuildPlan:
    chunks: Sequence[KnowledgeChunk]
    reusable_embeddings: Mapping[str, Sequence[float]]
    to_embed: Sequence[KnowledgeChunk]
    deleted_chunk_ids: Sequence[str]
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    total_characters: int
    estimated_tokens: int
    estimated_cost_yuan: Decimal

    @property
    def reused_chunk_ids(self) -> Sequence[str]:
        return tuple(sorted(self.reusable_embeddings))


@dataclass(frozen=True)
class VectorBuildStats:
    records: Sequence[VectorRecord]
    reused_count: int
    embedded_count: int
    deleted_count: int
    actual_input_tokens: int | None


@dataclass(frozen=True)
class SearchHit:
    record: VectorRecord
    score: float


@dataclass(frozen=True)
class EntityMatch:
    parent_document_id: str
    alias: str
    start: int


class RetrievalResult(StrictModel):
    rank: int = Field(ge=1)
    score: float
    match_origin: Literal["exact_entity", "dense"]
    matched_entity_ids: list[str]
    chunk_id: str = Field(min_length=1)
    parent_document_id: str = Field(min_length=1)
    section: str = Field(min_length=1)
    text: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    metadata: Metadata
    source_url: str = Field(min_length=1)
    source_files: list[str] = Field(min_length=1)

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value

    @field_validator("matched_entity_ids")
    @classmethod
    def validate_entity_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("matched_entity_ids must be unique")
        return value


__all__ = [
    "EmbeddingAPIError",
    "EmbeddingBatch",
    "EmbeddingConfigurationError",
    "EntityCatalogError",
    "EntityMatch",
    "RetrievalError",
    "RetrievalEvaluationError",
    "RetrievalResult",
    "SearchHit",
    "VectorBuildPlan",
    "VectorBuildStats",
    "VectorIndexNotReadyError",
    "VectorRecord",
    "VectorRecordValidationError",
]
