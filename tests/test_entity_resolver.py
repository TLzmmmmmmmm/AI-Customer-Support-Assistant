import unittest

from knowledge_pipeline.models import ProductSource
from knowledge_pipeline.retrieval.entities import ExactEntityResolver
from knowledge_pipeline.retrieval.models import EntityCatalogError, VectorRecord


def product_record(
    product_id: str,
    *,
    chunk_suffix: str = "content",
    slug: str | None = None,
    heading: str | None = None,
) -> VectorRecord:
    resolved_slug = slug or product_id
    return VectorRecord.model_validate({
        "schema_version": "1.0",
        "chunk_id": f"product:{product_id}:{chunk_suffix}",
        "parent_document_id": f"product:{product_id}",
        "type": "product",
        "section": heading or product_id,
        "text": f"# {heading or product_id}\n\n产品说明",
        "language": "zh-CN",
        "content_hash": "a" * 64,
        "source_url": f"https://example.com/{product_id}/",
        "source_files": [f"src/content/products/{product_id}.json"],
        "metadata": {
            "product_id": product_id,
            "slug": resolved_slug,
            "category_id": "two-way-radio",
            "category_name": "对讲机通信",
        },
        "embedding_provider": "dashscope",
        "embedding_model": "qwen3.7-text-embedding",
        "embedding_dimensions": 3,
        "embedding_text_type": "document",
        "embedding": [1.0, 0.0, 0.0],
    })


def records() -> list[VectorRecord]:
    return [
        product_record("hp780", heading="海能达 HP780 数字对讲机"),
        product_record("hp790ex", heading="海能达 HP790Ex 防爆对讲机"),
        product_record("xir-p8668ex", heading="摩托罗拉 XiR P8668Ex"),
        product_record("ap", heading="无线 AP"),
    ]


def product_source(
    product_id: str,
    *,
    slug: str | None = None,
    name: str | None = None,
) -> ProductSource:
    return ProductSource.model_validate({
        "id": product_id,
        "name": name or f"产品 {product_id}",
        "slug": slug or product_id,
        "category_id": "two-way-radio",
        "key_features": ["特点"],
        "product_features": "产品说明",
        "technical_parameters": [{
            "group": "一般规格",
            "items": [{"name": "功率", "value": "2W"}],
        }],
        "source_path": f"/two-way-radio/{slug or product_id}/",
        "published": True,
        "provenance": [{
            "kind": "website_file",
            "reference": f"src/content/products/{product_id}.json",
        }],
    })


class EntityResolverTests(unittest.TestCase):
    def setUp(self):
        self.resolver = ExactEntityResolver.from_records(records())

    def test_case_space_and_hyphen_variants_resolve_the_same_model(self):
        for query in (
            "XiR P8668Ex 频段",
            "xir-p8668ex 频段",
            "XIR P8668EX 频段",
        ):
            with self.subTest(query=query):
                self.assertEqual(
                    [
                        match.parent_document_id
                        for match in self.resolver.resolve(query)
                    ],
                    ["product:xir-p8668ex"],
                )

    def test_multiple_models_follow_first_query_occurrence(self):
        matches = self.resolver.resolve("比较 HP790Ex 和 HP780")

        self.assertEqual(
            [match.parent_document_id for match in matches],
            ["product:hp790ex", "product:hp780"],
        )
        self.assertLess(matches[0].start, matches[1].start)

    def test_short_ap_alias_requires_full_alphanumeric_boundaries(self):
        self.assertEqual(self.resolver.resolve("capacity planning"), [])
        self.assertEqual(
            [
                match.parent_document_id
                for match in self.resolver.resolve("AP 的供电方式")
            ],
            ["product:ap"],
        )

    def test_ascii_model_alias_can_touch_chinese_grammar(self):
        self.assertEqual(
            [
                match.parent_document_id
                for match in self.resolver.resolve("HP780的频率范围")
            ],
            ["product:hp780"],
        )

    def test_chinese_product_heading_can_be_followed_without_space(self):
        resolver = ExactEntityResolver.from_records([
            product_record(
                "domestic-20w",
                heading="国产 20W短波电台",
            )
        ])

        self.assertEqual(
            [
                match.parent_document_id
                for match in resolver.resolve("国产 20W 短波电台搭配什么天线？")
            ],
            ["product:domestic-20w"],
        )

    def test_duplicate_aliases_for_same_parent_are_deduplicated(self):
        duplicate = product_record(
            "hp780",
            chunk_suffix="specifications",
            heading="HP780",
        )
        resolver = ExactEntityResolver.from_records([records()[0], duplicate])

        self.assertEqual(
            [match.parent_document_id for match in resolver.resolve("HP780")],
            ["product:hp780"],
        )

    def test_alias_collision_between_parents_fails_catalog_construction(self):
        colliding = [
            product_record("radio-one", slug="shared-model"),
            product_record("radio-two", slug="shared model"),
        ]

        with self.assertRaises(EntityCatalogError):
            ExactEntityResolver.from_records(colliding)

    def test_non_product_records_do_not_enter_catalog(self):
        non_product = records()[0].model_copy(update={
            "chunk_id": "support:hp780:service",
            "parent_document_id": "support:hp780",
            "type": "support",
            "metadata": {"service_id": "hp780"},
        })

        resolver = ExactEntityResolver.from_records([non_product])

        self.assertEqual(resolver.resolve("HP780"), [])

    def test_canonical_source_resolution_reuses_normalization_without_names(self):
        resolver = ExactEntityResolver.from_product_sources([
            product_source("ly198", name="润信达 LY198"),
            product_source("xir-p8668ex", name="摩托罗拉 XiR P8668Ex"),
        ])

        for value, expected in (
            ("LY198", ["product:ly198"]),
            (" ly198 ", ["product:ly198"]),
            ("XIR P8668EX", ["product:xir-p8668ex"]),
            ("润信达 LY198", []),
            ("LY19", []),
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    resolver.resolve_canonical_identifier(value),
                    expected,
                )

    def test_canonical_source_resolution_preserves_ambiguous_aliases(self):
        resolver = ExactEntityResolver.from_product_sources([
            product_source("radio-one", slug="shared-model"),
            product_source("radio-two", slug="shared-model"),
        ])

        self.assertEqual(
            resolver.resolve_canonical_identifier("SHARED MODEL"),
            ["product:radio-one", "product:radio-two"],
        )


if __name__ == "__main__":
    unittest.main()
