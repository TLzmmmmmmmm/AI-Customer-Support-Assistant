from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import load_dotenv

from knowledge_pipeline.retrieval import (
    DashScopeCredentials,
    DashScopeEmbeddingProvider,
    EmbeddingConfig,
    ExactEntityResolver,
    NumpyExactVectorIndex,
    RetrievalConfig,
    Retriever,
    load_chunks,
    load_vector_records,
    validate_records_against_chunks,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS_PATH = REPOSITORY_ROOT / "knowledge" / "chunks.jsonl"
DEFAULT_VECTOR_RECORDS_PATH = (
    REPOSITORY_ROOT / "knowledge" / "vector_records.jsonl"
)


def create_embedding_provider(
    config: EmbeddingConfig,
    credentials: DashScopeCredentials,
) -> DashScopeEmbeddingProvider:
    return DashScopeEmbeddingProvider(config, credentials)


def build_retriever(
    *,
    values: Mapping[str, str] | None = None,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    vector_records_path: Path = DEFAULT_VECTOR_RECORDS_PATH,
) -> Retriever:
    if values is None:
        load_dotenv(REPOSITORY_ROOT / ".env")
        resolved_values = os.environ
    else:
        resolved_values = values
    embedding_config = EmbeddingConfig.from_mapping(resolved_values)
    retrieval_config = RetrievalConfig.from_mapping(resolved_values)
    chunks = load_chunks(chunks_path)
    records = load_vector_records(vector_records_path)
    validate_records_against_chunks(records, chunks, embedding_config)
    index = NumpyExactVectorIndex(records)
    resolver = ExactEntityResolver.from_records(records)
    credentials = DashScopeCredentials.from_mapping(resolved_values)
    provider = create_embedding_provider(embedding_config, credentials)
    return Retriever(
        embedding_provider=provider,
        vector_index=index,
        entity_resolver=resolver,
        default_top_k=retrieval_config.top_k,
    )


__all__ = ["build_retriever", "create_embedding_provider"]
