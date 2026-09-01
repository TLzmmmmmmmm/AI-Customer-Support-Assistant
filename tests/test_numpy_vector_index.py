import unittest

from knowledge_pipeline.retrieval.index import NumpyExactVectorIndex
from knowledge_pipeline.retrieval.models import (
    VectorIndexNotReadyError,
    VectorRecord,
)


def record(entity_id: str, vector: list[float]) -> VectorRecord:
    return VectorRecord.model_validate({
        "schema_version": "1.0",
        "chunk_id": f"product:{entity_id}:content",
        "parent_document_id": f"product:{entity_id}",
        "type": "product",
        "section": f"产品 {entity_id}",
        "text": f"# 产品 {entity_id}",
        "language": "zh-CN",
        "content_hash": (entity_id[0] if entity_id[0] in "abcdef" else "a") * 64,
        "source_url": f"https://example.com/{entity_id}/",
        "source_files": [f"src/content/products/{entity_id}.json"],
        "metadata": {
            "product_id": entity_id,
            "slug": entity_id,
            "category_id": "two-way-radio",
            "category_name": "对讲机通信",
        },
        "embedding_provider": "dashscope",
        "embedding_model": "qwen3.7-text-embedding",
        "embedding_dimensions": len(vector),
        "embedding_text_type": "document",
        "embedding": vector,
    })


class NumpyExactVectorIndexTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            record("a", [1.0, 0.0, 0.0]),
            record("b", [1.0, 1.0, 0.0]),
            record("c", [0.0, 1.0, 0.0]),
        ]
        self.index = NumpyExactVectorIndex(self.records)

    def test_cosine_search_returns_descending_exact_scores(self):
        hits = self.index.search([1.0, 0.0, 0.0], top_k=3)

        self.assertEqual(
            [hit.record.chunk_id for hit in hits],
            ["product:a:content", "product:b:content", "product:c:content"],
        )
        self.assertAlmostEqual(hits[0].score, 1.0, places=6)
        self.assertAlmostEqual(hits[1].score, 2 ** -0.5, places=6)
        self.assertAlmostEqual(hits[2].score, 0.0, places=6)

    def test_parent_filter_is_applied_before_top_k(self):
        hits = self.index.search(
            [1.0, 0.0, 0.0],
            top_k=5,
            parent_document_ids={"product:b", "product:c"},
        )

        self.assertEqual(
            [hit.record.parent_document_id for hit in hits],
            ["product:b", "product:c"],
        )

    def test_unknown_parent_filter_returns_no_hits(self):
        hits = self.index.search(
            [1.0, 0.0, 0.0],
            top_k=5,
            parent_document_ids={"product:missing"},
        )

        self.assertEqual(hits, [])

    def test_ties_are_broken_by_chunk_id(self):
        index = NumpyExactVectorIndex([
            record("b", [1.0, 0.0, 0.0]),
            record("a", [1.0, 0.0, 0.0]),
        ])

        hits = index.search([1.0, 0.0, 0.0], top_k=2)

        self.assertEqual(
            [hit.record.chunk_id for hit in hits],
            ["product:a:content", "product:b:content"],
        )

    def test_invalid_k_query_dimension_and_zero_query_fail(self):
        cases = (
            ([1.0, 0.0], 1),
            ([0.0, 0.0, 0.0], 1),
            ([1.0, 0.0, 0.0], 0),
        )
        for query, top_k in cases:
            with self.subTest(query=query, top_k=top_k):
                with self.assertRaises(VectorIndexNotReadyError):
                    self.index.search(query, top_k=top_k)

    def test_index_rejects_empty_duplicate_and_mixed_dimension_records(self):
        invalid_collections = (
            [],
            [self.records[0], self.records[0]],
            [self.records[0], record("d", [1.0, 0.0])],
        )
        for records in invalid_collections:
            with self.subTest(size=len(records)):
                with self.assertRaises(VectorIndexNotReadyError):
                    NumpyExactVectorIndex(records)

    def test_index_does_not_mutate_persisted_vectors(self):
        original = [item.embedding.copy() for item in self.records]

        self.index.search([1.0, 0.0, 0.0], top_k=2)

        self.assertEqual(
            [item.embedding for item in self.records],
            original,
        )


if __name__ == "__main__":
    unittest.main()
