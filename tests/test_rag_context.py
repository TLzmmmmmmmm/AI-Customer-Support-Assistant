import importlib
import unittest
from collections.abc import Callable, Sequence

from knowledge_pipeline.retrieval.models import RetrievalResult


def retrieval_result(
    rank: int,
    section: str,
    text: str,
) -> RetrievalResult:
    product_id = f"context-product-{rank}"
    return RetrievalResult.model_validate({
        "rank": rank,
        "score": 1.0 - (rank / 10),
        "match_origin": "exact_entity" if rank == 1 else "dense",
        "matched_entity_ids": (
            [f"product:{product_id}"] if rank == 1 else []
        ),
        "chunk_id": f"product:{product_id}:content",
        "parent_document_id": f"product:{product_id}",
        "type": "product",
        "section": section,
        "text": text,
        "content_hash": f"{rank}" * 64,
        "metadata": {
            "product_id": product_id,
            "slug": product_id,
            "category_id": "two-way-radio",
            "category_name": "对讲机通信",
        },
        "source_url": f"https://example.com/{product_id}/",
        "source_files": [f"src/content/products/{product_id}.json"],
    })


class RagContextBuilderTests(unittest.TestCase):
    def _builder(
        self,
    ) -> Callable[[Sequence[RetrievalResult]], list[dict[str, str]]]:
        try:
            module = importlib.import_module("rag_context")
        except ModuleNotFoundError:
            self.fail("rag_context module is missing")
        builder = getattr(module, "build_retrieved_context", None)
        self.assertIsNotNone(builder)
        return builder

    def test_preserves_all_five_results_in_order_with_only_llm_fields(self):
        results = [
            retrieval_result(1, "产品概述", "# 产品一\n\n完整概述文本。"),
            retrieval_result(2, "技术参数", "# 产品二\n\n功率：5W。"),
            retrieval_result(3, "产品功能", "# 产品三\n\n支持完整功能说明。"),
            retrieval_result(4, "应用场景", "# 产品四\n\n适用于应急通信。"),
            retrieval_result(
                5,
                "注意事项",
                "# 产品五\n\n不得截断的最后一条完整文本。",
            ),
        ]
        expected = [
            {
                "type": "product",
                "section": "产品概述",
                "text": "# 产品一\n\n完整概述文本。",
            },
            {
                "type": "product",
                "section": "技术参数",
                "text": "# 产品二\n\n功率：5W。",
            },
            {
                "type": "product",
                "section": "产品功能",
                "text": "# 产品三\n\n支持完整功能说明。",
            },
            {
                "type": "product",
                "section": "应用场景",
                "text": "# 产品四\n\n适用于应急通信。",
            },
            {
                "type": "product",
                "section": "注意事项",
                "text": "# 产品五\n\n不得截断的最后一条完整文本。",
            },
        ]

        first = self._builder()(results)
        second = self._builder()(results)

        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        for item in first:
            self.assertEqual(set(item), {"type", "section", "text"})

    def test_empty_results_produce_empty_context(self):
        self.assertEqual(self._builder()([]), [])


if __name__ == "__main__":
    unittest.main()
