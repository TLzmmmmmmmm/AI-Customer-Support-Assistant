import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from knowledge_pipeline.chunking import compute_chunk_content_hash
from knowledge_pipeline.models import KnowledgeChunk
from knowledge_pipeline.retrieval.models import (
    EmbeddingBatch,
    EmbeddingConfigurationError,
    VectorIndexNotReadyError,
    VectorRecord,
)
from knowledge_pipeline.retrieval.records import serialize_vector_records


class FakeEmbeddingProvider:
    def __init__(self):
        self.query_calls: list[str] = []

    def embed_query(self, text: str) -> EmbeddingBatch:
        self.query_calls.append(text)
        return EmbeddingBatch(vectors=([1.0, 0.0],), input_tokens=1)


def write_valid_artifacts(directory: Path) -> tuple[Path, Path]:
    metadata = {
        "product_id": "hp780",
        "slug": "hp780",
        "category_id": "two-way-radio",
        "category_name": "对讲机通信",
    }
    text = "# HP780\n\n防护等级：IP68"
    section = "技术参数"
    chunk = KnowledgeChunk.model_validate({
        "schema_version": "1.0",
        "chunk_id": "product:hp780:specifications",
        "parent_document_id": "product:hp780",
        "parent_document_hash": "c" * 64,
        "type": "product",
        "section": section,
        "text": text,
        "language": "zh-CN",
        "content_hash": compute_chunk_content_hash(
            "product", section, text, "zh-CN", metadata
        ),
        "source_url": "https://example.com/hp780/",
        "source_files": ["src/content/products/hp780.json"],
        "metadata": metadata,
    })
    record = VectorRecord.model_validate({
        "schema_version": "1.0",
        **chunk.model_dump(
            mode="json",
            exclude={"parent_document_hash"},
        ),
        "embedding_provider": "dashscope",
        "embedding_model": "qwen3.7-text-embedding",
        "embedding_dimensions": 2,
        "embedding_text_type": "document",
        "embedding": [1.0, 0.0],
    })
    chunks_path = directory / "chunks.jsonl"
    vectors_path = directory / "vector_records.jsonl"
    chunks_path.write_text(
        json.dumps(
            chunk.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    vectors_path.write_bytes(serialize_vector_records([record]))
    return chunks_path, vectors_path


class RetrievalServiceTests(unittest.TestCase):
    def _service(self):
        try:
            return importlib.import_module("services.retrieval")
        except ModuleNotFoundError:
            self.fail("services.retrieval module is missing")

    def test_builds_real_retriever_from_validated_local_artifacts(self):
        service = self._service()
        provider = FakeEmbeddingProvider()
        values = {
            "EMBEDDING_DIMENSIONS": "2",
            "DASHSCOPE_API_KEY": "test-key",
            "DASHSCOPE_WORKSPACE_ID": "test-workspace",
        }
        with tempfile.TemporaryDirectory() as directory_name:
            chunks, vectors = write_valid_artifacts(Path(directory_name))
            with patch.object(
                service,
                "create_embedding_provider",
                return_value=provider,
            ):
                retriever = service.build_retriever(
                    values=values,
                    chunks_path=chunks,
                    vector_records_path=vectors,
                )
                results = retriever.retrieve("HP780 参数", top_k=1)

        self.assertEqual(provider.query_calls, ["HP780 参数"])
        self.assertEqual(
            [result.chunk_id for result in results],
            ["product:hp780:specifications"],
        )

    def test_missing_vectors_fail_before_provider_construction(self):
        service = self._service()
        values = {
            "EMBEDDING_DIMENSIONS": "2",
            "DASHSCOPE_API_KEY": "test-key",
            "DASHSCOPE_WORKSPACE_ID": "test-workspace",
        }
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            chunks, _ = write_valid_artifacts(directory)
            missing_vectors = directory / "missing.jsonl"
            with patch.object(service, "create_embedding_provider") as factory:
                with self.assertRaises(VectorIndexNotReadyError):
                    service.build_retriever(
                        values=values,
                        chunks_path=chunks,
                        vector_records_path=missing_vectors,
                    )

        factory.assert_not_called()

    def test_default_build_loads_repository_dotenv_before_configuration(self):
        service = self._service()
        provider = FakeEmbeddingProvider()

        def load_repository_dotenv(path: Path) -> None:
            os.environ.update({
                "DASHSCOPE_API_KEY": "dotenv-key",
                "DASHSCOPE_WORKSPACE_ID": "dotenv-workspace",
            })

        with tempfile.TemporaryDirectory() as directory_name:
            chunks, vectors = write_valid_artifacts(Path(directory_name))
            with (
                patch.dict(
                    os.environ,
                    {"EMBEDDING_DIMENSIONS": "2"},
                    clear=True,
                ),
                patch.object(
                    service,
                    "load_dotenv",
                    side_effect=load_repository_dotenv,
                    create=True,
                ) as load_dotenv,
                patch.object(
                    service,
                    "create_embedding_provider",
                    return_value=provider,
                ),
            ):
                try:
                    retriever = service.build_retriever(
                        chunks_path=chunks,
                        vector_records_path=vectors,
                    )
                except EmbeddingConfigurationError:
                    self.fail("build_retriever did not load the repository .env")

        load_dotenv.assert_called_once_with(
            service.REPOSITORY_ROOT / ".env"
        )
        results = retriever.retrieve("HP780", top_k=1)
        self.assertEqual(results[0].chunk_id, "product:hp780:specifications")


if __name__ == "__main__":
    unittest.main()
