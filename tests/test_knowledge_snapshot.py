from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from knowledge_pipeline.chunking import (
    load_documents,
    serialize_chunks,
)
from knowledge_pipeline.core import serialize_documents
from knowledge_pipeline.models import KnowledgeChunk
from knowledge_pipeline.retrieval.models import VectorRecord
from knowledge_pipeline.retrieval.records import (
    load_chunks,
    serialize_vector_records,
)


ROOT = Path(__file__).resolve().parents[1]


def _record_for(chunk: KnowledgeChunk, dimensions: int = 3) -> VectorRecord:
    return VectorRecord.model_validate({
        "schema_version": "1.0",
        **chunk.model_dump(mode="json", exclude={"parent_document_hash"}),
        "embedding_provider": "dashscope",
        "embedding_model": "qwen3.7-text-embedding",
        "embedding_dimensions": dimensions,
        "embedding_text_type": "document",
        "embedding": [1.0] + [0.0] * (dimensions - 1),
    })


def _write_valid_snapshot(directory: Path) -> tuple[Path, Path, Path]:
    documents = load_documents(ROOT / "knowledge" / "documents.jsonl")
    chunks = load_chunks(ROOT / "knowledge" / "chunks.jsonl")
    records = [_record_for(chunk) for chunk in chunks]
    documents_path = directory / "documents.jsonl"
    chunks_path = directory / "chunks.jsonl"
    vectors_path = directory / "vector_records.jsonl"
    documents_path.write_bytes(serialize_documents(documents))
    chunks_path.write_bytes(serialize_chunks(chunks))
    vectors_path.write_bytes(serialize_vector_records(records))
    return documents_path, chunks_path, vectors_path


class KnowledgeSnapshotCliTests(unittest.TestCase):
    def _run(self, documents: Path, chunks: Path, vectors: Path, *, dimensions=3):
        from scripts import validate_knowledge_snapshot

        stdout = io.StringIO()
        stderr = io.StringIO()
        environment = {
            "EMBEDDING_PROVIDER": "dashscope",
            "EMBEDDING_MODEL": "qwen3.7-text-embedding",
            "EMBEDDING_DIMENSIONS": str(dimensions),
        }
        with (
            patch.dict("os.environ", environment, clear=True),
            patch.object(validate_knowledge_snapshot, "load_dotenv"),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            code = validate_knowledge_snapshot.main([
                "--documents", str(documents),
                "--chunks", str(chunks),
                "--vectors", str(vectors),
            ])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_valid_snapshot_reports_counts_hashes_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory_name:
            paths = _write_valid_snapshot(Path(directory_name))
            code, stdout, stderr = self._run(*paths)

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("Knowledge snapshot validation: PASS", stdout)
        self.assertIn("Documents: 62", stdout)
        self.assertIn("Chunks: 74", stdout)
        self.assertIn("Vector records: 74", stdout)
        self.assertIn(
            "Embedding: dashscope / qwen3.7-text-embedding / 3",
            stdout,
        )
        self.assertEqual(stdout.count("SHA-256:"), 3)

    def test_missing_vectors_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            documents, chunks, _ = _write_valid_snapshot(directory)
            missing_vectors = directory / "missing.jsonl"
            code, stdout, stderr = self._run(
                documents,
                chunks,
                missing_vectors,
            )

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("validation: FAIL", stderr)
        self.assertIn("vector records are missing", stderr)

    def test_document_chunk_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory_name:
            paths = _write_valid_snapshot(Path(directory_name))
            chunks = load_chunks(paths[1])
            payload = chunks[0].model_dump(mode="json")
            payload["parent_document_hash"] = "a" * 64
            chunks[0] = KnowledgeChunk.model_validate(payload)
            paths[1].write_bytes(serialize_chunks(chunks))
            code, stdout, stderr = self._run(*paths)

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("validation: FAIL", stderr)
        self.assertIn("parent_document_hash", stderr)

    def test_stale_vector_content_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory_name:
            paths = _write_valid_snapshot(Path(directory_name))
            records = [
                _record_for(chunk)
                for chunk in load_chunks(paths[1])
            ]
            payload = records[0].model_dump(mode="json")
            payload["content_hash"] = "a" * 64
            records[0] = VectorRecord.model_validate(payload)
            paths[2].write_bytes(serialize_vector_records(records))
            code, stdout, stderr = self._run(*paths)

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("validation: FAIL", stderr)
        self.assertIn("vector record fields are stale", stderr)

    def test_embedding_configuration_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory_name:
            paths = _write_valid_snapshot(Path(directory_name))
            code, stdout, stderr = self._run(*paths, dimensions=1024)

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("validation: FAIL", stderr)
        self.assertIn("embedding configuration is stale", stderr)


if __name__ == "__main__":
    unittest.main()
