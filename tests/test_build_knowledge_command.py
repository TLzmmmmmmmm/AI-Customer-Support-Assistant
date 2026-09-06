from __future__ import annotations

import importlib
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from knowledge_pipeline.core import BuildError
from knowledge_pipeline.retrieval import (
    EmbeddingAPIError,
    EmbeddingBatch,
    EmbeddingConfig,
    load_chunks,
    load_vector_records,
    validate_records_against_chunks,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "knowledge" / "source"


class FakeProvider:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        inputs = list(texts)
        self.calls.append(inputs)
        vectors = tuple(
            [float(index + 1)] + [0.0] * (self.config.dimensions - 1)
            for index in range(len(inputs))
        )
        return EmbeddingBatch(vectors=vectors, input_tokens=len(inputs) * 10)


class FailingProvider(FakeProvider):
    def embed_documents(self, texts):
        raise EmbeddingAPIError("controlled provider failure")


class KnowledgeBuildCommandTests(unittest.TestCase):
    def setUp(self):
        try:
            self.command = importlib.import_module("scripts.build_knowledge")
        except ModuleNotFoundError:
            self.fail("scripts/build_knowledge.py unified command is missing")

    @staticmethod
    def _environment() -> dict[str, str]:
        return {
            "EMBEDDING_PROVIDER": "dashscope",
            "EMBEDDING_MODEL": "test-embedding-model",
            "EMBEDDING_DIMENSIONS": "3",
            "EMBEDDING_PRICE_YUAN_PER_1K_TOKENS": "0.001",
            "DASHSCOPE_API_KEY": "test-api-key",
            "DASHSCOPE_WORKSPACE_ID": "test-workspace",
        }

    @staticmethod
    def _arguments(root: Path, *, execute: bool = False) -> list[str]:
        arguments = [
            "--source-root",
            str(SOURCE_ROOT),
            "--documents",
            str(root / "documents.jsonl"),
            "--chunks",
            str(root / "chunks.jsonl"),
            "--vectors",
            str(root / "vector_records.jsonl"),
        ]
        if execute:
            arguments.append("--execute")
        return arguments

    def _run(self, arguments: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.dict(os.environ, self._environment(), clear=False):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = self.command.main(arguments)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_plan_only_builds_in_staging_without_api_or_live_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                self.command,
                "create_embedding_provider",
                side_effect=AssertionError("plan-only created a provider"),
            ):
                result, stdout, stderr = self._run(self._arguments(root))

            self.assertEqual(result, 0, stderr)
            self.assertIn("Mode: plan only", stdout)
            self.assertIn("Documents: 61", stdout)
            self.assertIn("Chunks: 73", stdout)
            self.assertIn("No API request was made", stdout)
            self.assertFalse((root / "documents.jsonl").exists())
            self.assertFalse((root / "chunks.jsonl").exists())
            self.assertFalse((root / "vector_records.jsonl").exists())

    def test_execute_validates_staged_snapshot_before_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = FakeProvider(
                EmbeddingConfig.from_mapping(self._environment())
            )
            with patch.object(
                self.command,
                "create_embedding_provider",
                return_value=provider,
            ):
                result, stdout, stderr = self._run(
                    self._arguments(root, execute=True)
                )

            self.assertEqual(result, 0, stderr)
            self.assertIn("Knowledge snapshot validation: PASS", stdout)
            self.assertIn("Snapshot activated", stdout)
            self.assertEqual(len(provider.calls), 1)
            chunks = load_chunks(root / "chunks.jsonl")
            vectors = load_vector_records(root / "vector_records.jsonl")
            validate_records_against_chunks(
                vectors,
                chunks,
                provider.config,
            )
            self.assertEqual(len(chunks), 73)
            self.assertEqual(len(vectors), 73)

    def test_provider_failure_preserves_all_live_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = (
                root / "documents.jsonl",
                root / "chunks.jsonl",
                root / "vector_records.jsonl",
            )
            originals = (b"old documents\n", b"old chunks\n", b"old vectors\n")
            for target, payload in zip(targets, originals, strict=True):
                target.write_bytes(payload)

            provider = FailingProvider(
                EmbeddingConfig.from_mapping(self._environment())
            )
            with patch.object(
                self.command,
                "load_vector_records",
                return_value=[],
            ), patch.object(
                self.command,
                "create_embedding_provider",
                return_value=provider,
            ):
                result, _stdout, stderr = self._run(
                    self._arguments(root, execute=True)
                )

            self.assertEqual(result, 1)
            self.assertIn("controlled provider failure", stderr)
            self.assertEqual(
                tuple(target.read_bytes() for target in targets),
                originals,
            )

    def test_validation_failure_preserves_all_live_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = (
                root / "documents.jsonl",
                root / "chunks.jsonl",
                root / "vector_records.jsonl",
            )
            originals = (b"old documents\n", b"old chunks\n", b"old vectors\n")
            for target, payload in zip(targets, originals, strict=True):
                target.write_bytes(payload)

            provider = FakeProvider(
                EmbeddingConfig.from_mapping(self._environment())
            )
            with patch.object(
                self.command,
                "load_vector_records",
                return_value=[],
            ), patch.object(
                self.command,
                "create_embedding_provider",
                return_value=provider,
            ), patch.object(
                self.command,
                "validate_snapshot",
                side_effect=BuildError("controlled validation failure"),
            ):
                result, _stdout, stderr = self._run(
                    self._arguments(root, execute=True)
                )

            self.assertEqual(result, 1)
            self.assertIn("controlled validation failure", stderr)
            self.assertEqual(
                tuple(target.read_bytes() for target in targets),
                originals,
            )

    def test_unchanged_execute_reuses_vectors_without_provider_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = FakeProvider(
                EmbeddingConfig.from_mapping(self._environment())
            )
            with patch.object(
                self.command,
                "create_embedding_provider",
                return_value=provider,
            ):
                first, _stdout, stderr = self._run(
                    self._arguments(root, execute=True)
                )
            self.assertEqual(first, 0, stderr)
            original_vectors = (root / "vector_records.jsonl").read_bytes()

            with patch.object(
                self.command,
                "create_embedding_provider",
                side_effect=AssertionError("unchanged build created a provider"),
            ):
                second, stdout, stderr = self._run(
                    self._arguments(root, execute=True)
                )

            self.assertEqual(second, 0, stderr)
            self.assertIn("Reused: 73", stdout)
            self.assertIn("To embed: 0", stdout)
            self.assertEqual(
                (root / "vector_records.jsonl").read_bytes(),
                original_vectors,
            )

    def test_tracked_chunk_artifact_is_lf_and_has_git_attribute(self):
        chunks = (REPOSITORY_ROOT / "knowledge" / "chunks.jsonl").read_bytes()
        attributes = (REPOSITORY_ROOT / ".gitattributes").read_text(
            encoding="utf-8"
        )

        self.assertNotIn(b"\r", chunks)
        self.assertIn("knowledge/chunks.jsonl text eol=lf", attributes)


if __name__ == "__main__":
    unittest.main()
