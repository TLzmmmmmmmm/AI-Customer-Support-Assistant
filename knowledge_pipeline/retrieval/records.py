from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from knowledge_pipeline.chunking import compute_chunk_content_hash
from knowledge_pipeline.models import KnowledgeChunk

from .config import EmbeddingConfig
from .embedding import EmbeddingProvider
from .models import (
    EmbeddingAPIError,
    EmbeddingConfigurationError,
    VectorBuildPlan,
    VectorBuildStats,
    VectorIndexNotReadyError,
    VectorRecord,
    VectorRecordValidationError,
)


def _read_jsonl(path: Path, *, artifact_name: str) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise VectorRecordValidationError(
            f"failed to read {artifact_name} {path}: {error}"
        ) from error
    if "\r" in text:
        raise VectorRecordValidationError(
            f"{artifact_name} {path} must use LF line endings"
        )
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    payloads: list[dict] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise VectorRecordValidationError(
                f"{artifact_name} {path} has an empty line at {line_number}"
            )
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise VectorRecordValidationError(
                f"invalid JSON in {artifact_name} {path} at line "
                f"{line_number}, column {error.colno}: {error.msg}"
            ) from error
        if not isinstance(payload, dict):
            raise VectorRecordValidationError(
                f"{artifact_name} {path} line {line_number} must be an object"
            )
        payloads.append(payload)
    return payloads


def load_chunks(path: Path) -> list[KnowledgeChunk]:
    if not path.is_file():
        raise VectorRecordValidationError(
            f"chunk artifact does not exist: {path}"
        )
    chunks: list[KnowledgeChunk] = []
    seen: set[str] = set()
    for line_number, payload in enumerate(
        _read_jsonl(path, artifact_name="chunk artifact"),
        start=1,
    ):
        try:
            chunk = KnowledgeChunk.model_validate(payload)
        except ValidationError as error:
            raise VectorRecordValidationError(
                f"invalid chunk in {path} at line {line_number}: {error}"
            ) from error
        if chunk.chunk_id in seen:
            raise VectorRecordValidationError(
                f"duplicate chunk_id in {path}: {chunk.chunk_id}"
            )
        seen.add(chunk.chunk_id)
        expected_hash = compute_chunk_content_hash(
            chunk.type,
            chunk.section,
            chunk.text,
            chunk.language,
            chunk.metadata,
        )
        if chunk.content_hash != expected_hash:
            raise VectorRecordValidationError(
                f"chunk {chunk.chunk_id} has a stale content_hash"
            )
        chunks.append(chunk)
    if not chunks:
        raise VectorRecordValidationError(
            f"chunk artifact contains no chunks: {path}"
        )
    return chunks


def _validate_record_collection(records: Sequence[VectorRecord]) -> None:
    seen: set[str] = set()
    for record in records:
        if record.chunk_id in seen:
            raise VectorRecordValidationError(
                f"duplicate vector record chunk_id: {record.chunk_id}"
            )
        seen.add(record.chunk_id)


def load_vector_records(
    path: Path,
    *,
    missing_ok: bool = False,
) -> list[VectorRecord]:
    if not path.is_file():
        if missing_ok:
            return []
        raise VectorIndexNotReadyError(
            f"vector records are missing: {path}; run "
            "scripts/build_embeddings.py --execute"
        )
    records: list[VectorRecord] = []
    for line_number, payload in enumerate(
        _read_jsonl(path, artifact_name="vector artifact"),
        start=1,
    ):
        try:
            records.append(VectorRecord.model_validate(payload))
        except ValidationError as error:
            raise VectorRecordValidationError(
                f"invalid vector record in {path} at line "
                f"{line_number}: {error}"
            ) from error
    _validate_record_collection(records)
    expected_order = sorted(record.chunk_id for record in records)
    if [record.chunk_id for record in records] != expected_order:
        raise VectorRecordValidationError(
            f"vector records are not sorted by chunk_id: {path}"
        )
    return records


def _copied_chunk_payload(chunk: KnowledgeChunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "parent_document_id": chunk.parent_document_id,
        "type": chunk.type,
        "section": chunk.section,
        "text": chunk.text,
        "language": chunk.language,
        "content_hash": chunk.content_hash,
        "source_url": chunk.source_url,
        "source_files": chunk.source_files,
        "metadata": chunk.metadata.model_dump(mode="json"),
    }


def _copied_record_payload(record: VectorRecord) -> dict:
    return {
        "chunk_id": record.chunk_id,
        "parent_document_id": record.parent_document_id,
        "type": record.type,
        "section": record.section,
        "text": record.text,
        "language": record.language,
        "content_hash": record.content_hash,
        "source_url": record.source_url,
        "source_files": record.source_files,
        "metadata": record.metadata.model_dump(mode="json"),
    }


def validate_records_against_chunks(
    records: Sequence[VectorRecord],
    chunks: Sequence[KnowledgeChunk],
    config: EmbeddingConfig,
) -> None:
    _validate_record_collection(records)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(chunk_by_id) != len(chunks):
        raise VectorRecordValidationError("chunks contain duplicate chunk_id values")
    record_by_id = {record.chunk_id: record for record in records}
    if set(record_by_id) != set(chunk_by_id):
        missing = sorted(set(chunk_by_id) - set(record_by_id))
        extra = sorted(set(record_by_id) - set(chunk_by_id))
        raise VectorRecordValidationError(
            f"vector record set is stale; missing={missing}, extra={extra}"
        )
    for chunk_id, chunk in chunk_by_id.items():
        record = record_by_id[chunk_id]
        if _copied_record_payload(record) != _copied_chunk_payload(chunk):
            raise VectorRecordValidationError(
                f"vector record fields are stale for {chunk_id}"
            )
        if (
            record.embedding_provider != config.provider
            or record.embedding_model != config.model
            or record.embedding_dimensions != config.dimensions
            or record.embedding_text_type != "document"
        ):
            raise VectorRecordValidationError(
                f"embedding configuration is stale for {chunk_id}"
            )


def _records_by_id(
    records: Sequence[VectorRecord],
) -> dict[str, VectorRecord]:
    _validate_record_collection(records)
    return {record.chunk_id: record for record in records}


def plan_vector_build(
    chunks: Sequence[KnowledgeChunk],
    existing: Sequence[VectorRecord],
    config: EmbeddingConfig,
) -> VectorBuildPlan:
    ordered_chunks = tuple(sorted(chunks, key=lambda item: item.chunk_id))
    chunk_ids = [chunk.chunk_id for chunk in ordered_chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise VectorRecordValidationError("chunks contain duplicate chunk_id values")
    old_by_id = _records_by_id(existing)
    reusable: dict[str, Sequence[float]] = {}
    to_embed: list[KnowledgeChunk] = []
    for chunk in ordered_chunks:
        old = old_by_id.get(chunk.chunk_id)
        if old is not None and (
            old.content_hash == chunk.content_hash
            and old.embedding_provider == config.provider
            and old.embedding_model == config.model
            and old.embedding_dimensions == config.dimensions
            and old.embedding_text_type == "document"
        ):
            reusable[chunk.chunk_id] = tuple(old.embedding)
        else:
            to_embed.append(chunk)
    deleted = tuple(sorted(set(old_by_id) - set(chunk_ids)))
    total_characters = sum(len(chunk.text) for chunk in to_embed)
    estimated_tokens = sum(max(1, len(chunk.text) * 2) for chunk in to_embed)
    estimated_cost = (
        Decimal(estimated_tokens)
        / Decimal(1000)
        * config.price_yuan_per_1k_tokens
    )
    return VectorBuildPlan(
        chunks=ordered_chunks,
        reusable_embeddings=reusable,
        to_embed=tuple(to_embed),
        deleted_chunk_ids=deleted,
        embedding_provider=config.provider,
        embedding_model=config.model,
        embedding_dimensions=config.dimensions,
        total_characters=total_characters,
        estimated_tokens=estimated_tokens,
        estimated_cost_yuan=estimated_cost,
    )


def _record_from_chunk(
    chunk: KnowledgeChunk,
    embedding: Sequence[float],
    *,
    provider: str,
    model: str,
    dimensions: int,
) -> VectorRecord:
    return VectorRecord.model_validate({
        "schema_version": "1.0",
        **_copied_chunk_payload(chunk),
        "embedding_provider": provider,
        "embedding_model": model,
        "embedding_dimensions": dimensions,
        "embedding_text_type": "document",
        "embedding": list(embedding),
    })


def serialize_vector_records(records: Sequence[VectorRecord]) -> bytes:
    _validate_record_collection(records)
    ordered = sorted(records, key=lambda item: item.chunk_id)
    lines = [
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for record in ordered
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def write_vector_records(
    path: Path,
    records: Sequence[VectorRecord],
) -> None:
    payload = serialize_vector_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        parsed = load_vector_records(temporary_path)
        if serialize_vector_records(parsed) != payload:
            raise VectorRecordValidationError(
                "temporary vector JSONL verification was not deterministic"
            )
        os.replace(temporary_path, path)
        temporary_path = None
    except VectorRecordValidationError:
        raise
    except (OSError, UnicodeError) as error:
        raise VectorRecordValidationError(
            f"failed to write vector records {path}: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def execute_vector_build(
    plan: VectorBuildPlan,
    provider: EmbeddingProvider,
    output_path: Path,
) -> VectorBuildStats:
    config = provider.config
    if (
        config.provider != plan.embedding_provider
        or config.model != plan.embedding_model
        or config.dimensions != plan.embedding_dimensions
    ):
        raise EmbeddingConfigurationError(
            "embedding provider configuration differs from the build plan"
        )

    embeddings = {
        chunk_id: tuple(vector)
        for chunk_id, vector in plan.reusable_embeddings.items()
    }
    actual_input_tokens: int | None = 0
    if plan.to_embed:
        batch = provider.embed_documents([chunk.text for chunk in plan.to_embed])
        if len(batch.vectors) != len(plan.to_embed):
            raise EmbeddingAPIError(
                "embedding provider returned an unexpected vector count"
            )
        for chunk, vector in zip(plan.to_embed, batch.vectors, strict=True):
            embeddings[chunk.chunk_id] = tuple(vector)
        actual_input_tokens = batch.input_tokens

    records = tuple(
        _record_from_chunk(
            chunk,
            embeddings[chunk.chunk_id],
            provider=plan.embedding_provider,
            model=plan.embedding_model,
            dimensions=plan.embedding_dimensions,
        )
        for chunk in plan.chunks
    )
    write_vector_records(output_path, records)
    return VectorBuildStats(
        records=records,
        reused_count=len(plan.reusable_embeddings),
        embedded_count=len(plan.to_embed),
        deleted_count=len(plan.deleted_chunk_ids),
        actual_input_tokens=actual_input_tokens,
    )


__all__ = [
    "execute_vector_build",
    "load_chunks",
    "load_vector_records",
    "plan_vector_build",
    "serialize_vector_records",
    "validate_records_against_chunks",
    "write_vector_records",
]
