import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from knowledge_pipeline.chunking import compute_chunk_content_hash
from knowledge_pipeline.models import KnowledgeChunk
from knowledge_pipeline.retrieval.embedding import EmbeddingBatch
from knowledge_pipeline.retrieval.entities import ExactEntityResolver
from knowledge_pipeline.retrieval.index import NumpyExactVectorIndex
from knowledge_pipeline.retrieval.models import VectorRecord
from knowledge_pipeline.retrieval.records import serialize_vector_records
from knowledge_pipeline.retrieval.retriever import Retriever, format_debug_results


def record(
    product_id: str,
    vector: list[float],
    *,
    chunk_suffix: str = "content",
    section: str | None = None,
) -> VectorRecord:
    resolved_section = section or f"{product_id} 产品介绍"
    return VectorRecord.model_validate({
        "schema_version": "1.0",
        "chunk_id": f"product:{product_id}:{chunk_suffix}",
        "parent_document_id": f"product:{product_id}",
        "type": "product",
        "section": resolved_section,
        "text": f"# {product_id}\n\n{resolved_section}全文",
        "language": "zh-CN",
        "content_hash": "b" * 64,
        "source_url": f"https://example.com/{product_id}/#{chunk_suffix}",
        "source_files": [f"src/content/products/{product_id}.json"],
        "metadata": {
            "product_id": product_id,
            "slug": product_id,
            "category_id": "two-way-radio",
            "category_name": "对讲机通信",
        },
        "embedding_provider": "dashscope",
        "embedding_model": "qwen3.7-text-embedding",
        "embedding_dimensions": len(vector),
        "embedding_text_type": "document",
        "embedding": vector,
    })


class FakeEmbeddingProvider:
    def __init__(self):
        self.query_calls: list[str] = []

    def embed_query(self, text: str) -> EmbeddingBatch:
        self.query_calls.append(text)
        return EmbeddingBatch(vectors=([1.0, 0.0],), input_tokens=4)


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            record("hp780", [0.80, 0.20]),
            record(
                "hp780",
                [1.00, 0.00],
                chunk_suffix="specifications",
                section="技术参数",
            ),
            record("hp790ex", [0.70, 0.30]),
            record("dense-one", [0.95, 0.05]),
            record("dense-two", [0.60, 0.40]),
            record("dense-three", [0.50, 0.50]),
        ]
        self.provider = FakeEmbeddingProvider()
        self.retriever = Retriever(
            embedding_provider=self.provider,
            vector_index=NumpyExactVectorIndex(self.records),
            entity_resolver=ExactEntityResolver.from_records(self.records),
            default_top_k=5,
        )

    def test_exact_entities_are_first_then_entity_and_dense_hits_fill(self):
        results = self.retriever.retrieve(
            "比较 HP790Ex 和 HP780",
            top_k=5,
        )

        self.assertEqual(
            [item.parent_document_id for item in results[:2]],
            ["product:hp790ex", "product:hp780"],
        )
        self.assertEqual(results[0].chunk_id, "product:hp790ex:content")
        self.assertEqual(results[1].chunk_id, "product:hp780:specifications")
        self.assertEqual(results[2].chunk_id, "product:hp780:content")
        self.assertEqual(
            [item.match_origin for item in results[:3]],
            ["exact_entity", "exact_entity", "exact_entity"],
        )
        self.assertEqual(len({item.chunk_id for item in results}), len(results))
        self.assertEqual(
            [item.rank for item in results],
            list(range(1, len(results) + 1)),
        )
        self.assertAlmostEqual(results[0].score, 0.919145, places=5)

    def test_no_entity_uses_global_dense_order(self):
        results = self.retriever.retrieve("如何进行应急协同指挥？", top_k=3)

        self.assertEqual(
            [item.chunk_id for item in results],
            [
                "product:hp780:specifications",
                "product:dense-one:content",
                "product:hp780:content",
            ],
        )
        self.assertTrue(all(item.match_origin == "dense" for item in results))

    def test_query_is_embedded_once_and_default_k_is_five(self):
        results = self.retriever.retrieve("HP780 参数")

        self.assertEqual(self.provider.query_calls, ["HP780 参数"])
        self.assertEqual(len(results), 5)

    def test_results_preserve_record_fields_and_entity_ids(self):
        result = self.retriever.retrieve("HP790Ex 参数", top_k=1)[0]
        source = next(
            item for item in self.records if item.chunk_id == result.chunk_id
        )

        self.assertEqual(result.matched_entity_ids, ["product:hp790ex"])
        self.assertEqual(result.model_dump(mode="json").get("type"), source.type)
        self.assertEqual(result.section, source.section)
        self.assertEqual(result.text, source.text)
        self.assertEqual(result.metadata, source.metadata)
        self.assertEqual(result.content_hash, source.content_hash)
        self.assertEqual(result.source_url, source.source_url)
        self.assertEqual(result.source_files, source.source_files)

    def test_blank_query_and_non_positive_k_are_rejected_before_embedding(self):
        for query, top_k in (("   ", 5), ("有效查询", 0)):
            with self.subTest(query=query, top_k=top_k):
                with self.assertRaises(ValueError):
                    self.retriever.retrieve(query, top_k=top_k)

        self.assertEqual(self.provider.query_calls, [])

    def test_debug_output_contains_required_readable_fields(self):
        query = "HP780 参数"
        output = format_debug_results(
            query,
            self.retriever.retrieve(query, top_k=1),
        )

        for label in (
            "Query:",
            "Rank:",
            "Score:",
            "Origin:",
            "Matched entities:",
            "Chunk ID:",
            "Parent ID:",
            "Section:",
            "Text:",
            "Source:",
            "Source files:",
        ):
            with self.subTest(label=label):
                self.assertIn(label, output)
        self.assertIn("Score: 1.000000", output)
        self.assertIn("技术参数全文", output)


class RetrieveCliTests(unittest.TestCase):
    def _write_artifacts(self, directory: Path) -> tuple[Path, Path]:
        metadata = {
            "product_id": "hp780",
            "slug": "hp780",
            "category_id": "two-way-radio",
            "category_name": "对讲机通信",
        }
        text = "# HP780\n\n技术参数全文"
        section = "技术参数"
        chunk = KnowledgeChunk.model_validate({
            "schema_version": "1.0",
            "chunk_id": "product:hp780:content",
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
        vector = VectorRecord.model_validate({
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
        vectors_path = directory / "vectors.jsonl"
        chunks_path.write_text(
            json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        vectors_path.write_bytes(serialize_vector_records([vector]))
        return chunks_path, vectors_path

    def test_missing_vector_file_fails_before_provider_construction(self):
        from scripts import retrieve

        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.jsonl"
            with patch.dict(os.environ, {}, clear=True):
                with patch.object(retrieve, "create_embedding_provider") as factory:
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        code = retrieve.main([
                            "HP780 参数",
                            "--vectors", str(missing),
                        ])

        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn(
            "run scripts/build_embeddings.py --execute",
            stderr.getvalue(),
        )
        factory.assert_not_called()

    def test_valid_local_artifacts_produce_compact_json_without_generation(self):
        from scripts import retrieve

        provider = FakeEmbeddingProvider()
        stdout = io.StringIO()
        stderr = io.StringIO()
        environment = {
            "EMBEDDING_DIMENSIONS": "2",
            "DASHSCOPE_API_KEY": "test-key",
            "DASHSCOPE_WORKSPACE_ID": "test-workspace",
        }
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            chunks, vectors = self._write_artifacts(directory)
            with patch.dict(os.environ, environment, clear=True):
                with patch.object(
                    retrieve,
                    "create_embedding_provider",
                    return_value=provider,
                ):
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        code = retrieve.main([
                            "HP780 参数",
                            "--chunks", str(chunks),
                            "--vectors", str(vectors),
                            "--top-k", "1",
                        ])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(provider.query_calls, ["HP780 参数"])
        self.assertEqual(payload[0]["chunk_id"], "product:hp780:content")
        self.assertEqual(payload[0]["type"], "product")
        self.assertEqual(payload[0]["parent_document_id"], "product:hp780")
        self.assertEqual(payload[0]["source_url"], "https://example.com/hp780/")
        self.assertNotIn("answer", payload[0])


if __name__ == "__main__":
    unittest.main()
