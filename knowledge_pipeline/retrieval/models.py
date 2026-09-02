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
    CatalogMetadata,
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
    type: Literal[
        "catalog", "product", "solution", "support", "company", "contact"
    ]
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
            "catalog": CatalogMetadata,
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
    type: Literal[
        "catalog", "product", "solution", "support", "company", "contact"
    ]
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


class RelevanceGroup(StrictModel):
    id: str = Field(min_length=1)
    acceptable_chunk_ids: list[str] = Field(min_length=1)

    @field_validator("acceptable_chunk_ids")
    @classmethod
    def validate_chunk_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("acceptable_chunk_ids must be sorted and unique")
        return value


class RetrievalEvaluationCase(StrictModel):
    id: str = Field(min_length=1)
    source_case_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_chunk_ids: list[str] = Field(min_length=1)
    expected_parent_document_ids: list[str] = Field(min_length=1)
    expected_entity_ids: list[str]
    match_requirement: Literal["any", "all", "all_groups"]
    relevance_groups: list[RelevanceGroup]

    @field_validator(
        "expected_chunk_ids",
        "expected_parent_document_ids",
        "expected_entity_ids",
    )
    @classmethod
    def validate_unique_references(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("expected references must be unique")
        return value

    @model_validator(mode="after")
    def validate_match_requirement(self) -> "RetrievalEvaluationCase":
        if self.match_requirement == "all_groups":
            if not self.relevance_groups:
                raise ValueError("all_groups requires relevance_groups")
            grouped = {
                chunk_id
                for group in self.relevance_groups
                for chunk_id in group.acceptable_chunk_ids
            }
            if grouped != set(self.expected_chunk_ids):
                raise ValueError(
                    "relevance groups must exactly cover expected_chunk_ids"
                )
        elif self.relevance_groups:
            raise ValueError(
                "relevance_groups are only valid with all_groups"
            )
        return self


class RetrievalEvaluationSuite(StrictModel):
    schema_version: Literal["1.0"]
    suite_id: str = Field(min_length=1)
    chunk_snapshot_sha256: str = Field(min_length=1)
    default_top_k: int = Field(gt=0)
    cases: list[RetrievalEvaluationCase] = Field(min_length=1)

    @field_validator("chunk_snapshot_sha256")
    @classmethod
    def validate_snapshot_hash(cls, value: str) -> str:
        if not SHA256_HEX.fullmatch(value):
            raise ValueError("must be a lowercase SHA-256 hex digest")
        return value

    @model_validator(mode="after")
    def validate_case_ids(self) -> "RetrievalEvaluationSuite":
        ids = [case.id for case in self.cases]
        source_ids = [case.source_case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("evaluation case ids must be unique")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("evaluation source_case_ids must be unique")
        return self


class RetrievalMetrics(StrictModel):
    total_cases: int = Field(gt=0)
    hit_at_1: float = Field(ge=0.0, le=1.0)
    hit_at_3: float = Field(ge=0.0, le=1.0)
    hit_at_5: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    expected_chunk_recall_at_5: float = Field(ge=0.0, le=1.0)
    expected_parent_recall_at_5: float = Field(ge=0.0, le=1.0)
    entity_accuracy: float = Field(ge=0.0, le=1.0)
    complete_multi_source_recall: float = Field(ge=0.0, le=1.0)


class RetrievalFailure(StrictModel):
    case_id: str = Field(min_length=1)
    source_case_id: str = Field(min_length=1)
    kind: Literal[
        "missing_top_k",
        "incomplete_multi_source",
        "missed_entity",
        "late_rank",
    ]
    message: str = Field(min_length=1)


class RetrievalEvaluationResult(StrictModel):
    schema_version: Literal["1.0"]
    suite_id: str = Field(min_length=1)
    top_k: int = Field(gt=0)
    metrics: RetrievalMetrics
    failures: list[RetrievalFailure]


__all__ = [
    "EmbeddingAPIError",
    "EmbeddingBatch",
    "EmbeddingConfigurationError",
    "EntityCatalogError",
    "EntityMatch",
    "RetrievalError",
    "RetrievalEvaluationError",
    "RetrievalEvaluationCase",
    "RetrievalEvaluationResult",
    "RetrievalEvaluationSuite",
    "RetrievalFailure",
    "RetrievalMetrics",
    "RetrievalResult",
    "RelevanceGroup",
    "SearchHit",
    "VectorBuildPlan",
    "VectorBuildStats",
    "VectorIndexNotReadyError",
    "VectorRecord",
    "VectorRecordValidationError",
]
