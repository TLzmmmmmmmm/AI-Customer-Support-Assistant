import json
import tempfile
import unittest
from pathlib import Path

from knowledge_pipeline.chunking import compute_chunk_content_hash
from knowledge_pipeline.models import KnowledgeChunk
from knowledge_pipeline.retrieval.config import EmbeddingConfig
from knowledge_pipeline.retrieval.models import (
    EmbeddingAPIError,
    EmbeddingBatch,
    EmbeddingConfigurationError,
    VectorIndexNotReadyError,
    VectorRecord,
    VectorRecordValidationError,
)
from knowledge_pipeline.retrieval.records import (
    execute_vector_build,
    load_chunks,
    load_vector_records,
    plan_vector_build,
    serialize_vector_records,
    validate_records_against_chunks,
    write_vector_records,
)


def make_chunk(
    entity_id: str = "hp780",
    *,
    text: str | None = None,
    source_files: list[str] | None = None,
) -> KnowledgeChunk:
    section = f"产品 {entity_id}"
    chunk_text = text or f"# 产品 {entity_id}\n\n技术参数。"
    metadata = {
        "product_id": entity_id,
        "slug": entity_id,
        "category_id": "two-way-radio",
        "category_name": "对讲机通信",
    }
    return KnowledgeChunk.model_validate({
        "schema_version": "1.0",
        "chunk_id": f"product:{entity_id}:content",
        "parent_document_id": f"product:{entity_id}",
        "parent_document_hash": "b" * 64,
        "type": "product",
        "section": section,
        "text": chunk_text,
        "language": "zh-CN",
        "content_hash": compute_chunk_content_hash(
            "product",
            section,
            chunk_text,
            "zh-CN",
            metadata,
        ),
        "source_url": (
            f"https://www.shengborun.com/two-way-radio/{entity_id}/"
        ),
        "source_files": source_files or [
            f"src/content/products/two-way-radio/{entity_id}.json"
        ],
        "metadata": metadata,
    })


def embedding_config(
    *,
    model: str = "qwen3.7-text-embedding",
    dimensions: int = 3,
) -> EmbeddingConfig:
    return EmbeddingConfig.from_mapping({
        "EMBEDDING_MODEL": model,
        "EMBEDDING_DIMENSIONS": str(dimensions),
    })


def record_for(
    chunk: KnowledgeChunk,
    *,
    vector: list[float] | None = None,
    model: str = "qwen3.7-text-embedding",
) -> VectorRecord:
    embedding = vector or [1.0, 0.0, 0.0]
    return VectorRecord.model_validate({
        "schema_version": "1.0",
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
        "embedding_provider": "dashscope",
        "embedding_model": model,
        "embedding_dimensions": len(embedding),
        "embedding_text_type": "document",
        "embedding": embedding,
    })


class FakeProvider:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.document_calls: list[list[str]] = []

    def embed_documents(self, texts):
        inputs = list(texts)
        self.document_calls.append(inputs)
        vectors = tuple(
            [float(index + 1)] + [0.0] * (self.config.dimensions - 1)
            for index in range(len(inputs))
        )
        return EmbeddingBatch(vectors=vectors, input_tokens=len(inputs) * 10)


class FailingProvider(FakeProvider):
    def embed_documents(self, texts):
        raise EmbeddingAPIError("controlled failure")


class VectorRecordReuseTests(unittest.TestCase):
    def test_unchanged_identity_hash_and_embedding_config_are_reused(self):
        chunk = make_chunk()

        plan = plan_vector_build(
            [chunk],
            [record_for(chunk)],
            embedding_config(),
        )

        self.assertEqual(plan.reused_chunk_ids, (chunk.chunk_id,))
        self.assertEqual(plan.to_embed, ())

    def test_content_model_and_dimension_changes_invalidate_reuse(self):
        chunk = make_chunk()
        changed_chunk = make_chunk(text="# 产品 hp780\n\n更新后的技术参数。")
        cases = (
            (changed_chunk, record_for(chunk), embedding_config()),
            (chunk, record_for(chunk, model="old-model"), embedding_config()),
            (
                chunk,
                record_for(chunk, vector=[1.0, 0.0]),
                embedding_config(dimensions=3),
            ),
        )
        for current, old, config in cases:
            with self.subTest(old=old.embedding_model, dimension=len(old.embedding)):
                plan = plan_vector_build([current], [old], config)
                self.assertEqual(
                    [item.chunk_id for item in plan.to_embed],
                    [current.chunk_id],
                )

    def test_deleted_records_are_reported(self):
        current = make_chunk("hp780")
        deleted = make_chunk("hp790ex")

        plan = plan_vector_build(
            [current],
            [record_for(current), record_for(deleted)],
            embedding_config(),
        )

        self.assertEqual(plan.deleted_chunk_ids, (deleted.chunk_id,))

    def test_provenance_refreshes_without_reembedding(self):
        old_chunk = make_chunk(source_files=["src/old/hp780.json"])
        current = make_chunk(source_files=["src/new/hp780.json"])
        old_record = record_for(old_chunk)
        provider = FakeProvider(embedding_config())

        plan = plan_vector_build(
            [current],
            [old_record],
            embedding_config(),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "vectors.jsonl"
            stats = execute_vector_build(plan, provider, output)

        self.assertEqual(provider.document_calls, [])
        self.assertEqual(stats.records[0].source_files, ["src/new/hp780.json"])

    def test_estimate_is_conservative_and_uses_configured_price(self):
        chunk = make_chunk()

        plan = plan_vector_build([chunk], [], embedding_config())

        self.assertEqual(plan.total_characters, len(chunk.text))
        self.assertEqual(plan.estimated_tokens, len(chunk.text) * 2)
        self.assertGreater(plan.estimated_cost_yuan, 0)


class VectorRecordPersistenceTests(unittest.TestCase):
    def test_round_trip_is_sorted_utf8_lf_and_has_no_parent_hash(self):
        first = record_for(make_chunk("hp780"))
        second = record_for(make_chunk("hp790ex"), vector=[0.0, 1.0, 0.0])

        payload = serialize_vector_records([second, first])

        self.assertTrue(payload.endswith(b"\n"))
        lines = payload.decode("utf-8").splitlines()
        self.assertEqual(json.loads(lines[0])["chunk_id"], first.chunk_id)
        self.assertNotIn("parent_document_hash", json.loads(lines[0]))

    def test_missing_vector_file_is_optional_only_for_build_planning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.jsonl"

            self.assertEqual(load_vector_records(path, missing_ok=True), [])
            with self.assertRaises(VectorIndexNotReadyError):
                load_vector_records(path)

    def test_chunk_loader_recomputes_content_hash(self):
        chunk = make_chunk()
        payload = chunk.model_dump(mode="json")
        payload["content_hash"] = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.jsonl"
            path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

            with self.assertRaises(VectorRecordValidationError):
                load_chunks(path)

    def test_runtime_validation_rejects_stale_provenance(self):
        current = make_chunk(source_files=["src/new/hp780.json"])
        stale = record_for(make_chunk(source_files=["src/old/hp780.json"]))

        with self.assertRaises(VectorRecordValidationError):
            validate_records_against_chunks(
                [stale],
                [current],
                embedding_config(),
            )

    def test_failed_embedding_preserves_existing_output(self):
        chunk = make_chunk()
        plan = plan_vector_build([chunk], [], embedding_config())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "vectors.jsonl"
            output.write_bytes(b"original\n")

            with self.assertRaises(EmbeddingAPIError):
                execute_vector_build(
                    plan,
                    FailingProvider(embedding_config()),
                    output,
                )

            self.assertEqual(output.read_bytes(), b"original\n")

    def test_execute_rejects_provider_configuration_that_differs_from_plan(self):
        chunk = make_chunk()
        plan = plan_vector_build([chunk], [], embedding_config(dimensions=3))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "vectors.jsonl"

            with self.assertRaises(EmbeddingConfigurationError):
                execute_vector_build(
                    plan,
                    FakeProvider(embedding_config(dimensions=2)),
                    output,
                )

            self.assertFalse(output.exists())

    def test_execute_embeds_changed_chunks_and_writes_valid_records(self):
        chunks = [make_chunk("hp780"), make_chunk("hp790ex")]
        provider = FakeProvider(embedding_config())
        plan = plan_vector_build(chunks, [], embedding_config())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "vectors.jsonl"

            stats = execute_vector_build(plan, provider, output)
            loaded = load_vector_records(output)

        self.assertEqual(len(provider.document_calls), 1)
        self.assertEqual(len(provider.document_calls[0]), 2)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(stats.embedded_count, 2)
        self.assertEqual(stats.actual_input_tokens, 20)

    def test_atomic_writer_rejects_duplicate_ids(self):
        record = record_for(make_chunk())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "vectors.jsonl"

            with self.assertRaises(VectorRecordValidationError):
                write_vector_records(output, [record, record])


if __name__ == "__main__":
    unittest.main()
