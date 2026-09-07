import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from knowledge_pipeline.core import SourceInventory
from knowledge_pipeline.models import (
    ContactSource,
    ProductCategorySource,
    ProductSource,
)
from knowledge_pipeline.retrieval.models import (
    RetrievalError,
    RetrievalResult,
)
from support_tools import (
    DeterministicTools,
    ToolError,
    ToolErrorCode,
    build_tool_registry,
)
from services.tools import build_deterministic_tools


def category() -> ProductCategorySource:
    return ProductCategorySource.model_validate({
        "id": "two-way-radio",
        "name": "测试对讲机分类",
        "slug": "two-way-radio",
        "short_description": "测试分类说明",
        "source_path": "/products/two-way-radio/",
        "published": True,
        "provenance": [{
            "kind": "website_file",
            "reference": "src/content/product-categories/two-way-radio.json",
        }],
    })


def product(
    product_id: str,
    *,
    slug: str | None = None,
    name: str | None = None,
) -> ProductSource:
    return ProductSource.model_validate({
        "id": product_id,
        "name": name or f"测试产品 {product_id}",
        "slug": slug or product_id,
        "category_id": "two-way-radio",
        "key_features": [f"{product_id} 特点"],
        "product_features": f"{product_id} 的权威产品说明",
        "technical_parameters": [{
            "group": "一般规格",
            "items": [{"name": "输出功率", "value": "3W"}],
        }],
        "source_path": f"/two-way-radio/{slug or product_id}/",
        "published": True,
        "provenance": [{
            "kind": "website_file",
            "reference": f"src/content/products/{product_id}.json",
        }],
    })


def contact(contact_id: str = "fixture-company") -> ContactSource:
    return ContactSource.model_validate({
        "id": contact_id,
        "company_name": f"测试公司 {contact_id}",
        "duty_phone": "13800138000",
        "email": f"{contact_id}@example.com",
        "source_path": "/about/#contact",
        "published": True,
        "provenance": [{
            "kind": "website_file",
            "reference": "src/pages/about.astro",
        }],
    })


def inventory(*products: ProductSource) -> SourceInventory:
    items = products or (product("ly198", name="测试品牌 LY198"),)
    return SourceInventory(
        products={item.id: item for item in items},
        categories={"two-way-radio": category()},
        contacts={"fixture-company": contact()},
    )


def retrieval_result(item: ProductSource, rank: int) -> RetrievalResult:
    return RetrievalResult.model_validate({
        "rank": rank,
        "score": 1.0 - rank / 100,
        "match_origin": "dense",
        "matched_entity_ids": [],
        "chunk_id": f"product:{item.id}:content",
        "parent_document_id": f"product:{item.id}",
        "type": "product",
        "section": "产品介绍",
        "text": f"# {item.name}\n\n{item.product_features}",
        "content_hash": "a" * 64,
        "metadata": {
            "product_id": item.id,
            "slug": item.slug,
            "category_id": item.category_id,
            "category_name": "测试对讲机分类",
        },
        "source_url": f"https://fixture.example/two-way-radio/{item.slug}/",
        "source_files": [f"src/content/products/{item.id}.json"],
    })


class FakeRetriever:
    def __init__(self, results=None, error: Exception | None = None):
        self.results = list(results or [])
        self.error = error
        self.calls: list[dict[str, object]] = []

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        allowed_types=None,
        unique_parent_documents: bool = False,
    ):
        self.calls.append({
            "query": query,
            "top_k": top_k,
            "allowed_types": allowed_types,
            "unique_parent_documents": unique_parent_documents,
        })
        if self.error is not None:
            raise self.error
        return self.results


class DeterministicToolTests(unittest.TestCase):
    def test_operational_builder_loads_the_authoritative_source_inventory(self):
        tools = build_deterministic_tools()

        result = tools.get_product_details("LY198")

        self.assertEqual(result.name, "润信达 LY198")
        self.assertEqual(result.product_features[:5], "LY198")
        self.assertEqual(
            result.sources[0].url,
            "https://www.shengborun.com/two-way-radio/ly198/",
        )

    def test_operational_builder_redacts_invalid_source_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaises(ToolError) as context:
                build_deterministic_tools(source_root=missing)

        self.assertEqual(
            context.exception.code,
            ToolErrorCode.TOOL_UNAVAILABLE,
        )
        self.assertNotIn(str(missing), context.exception.message)

    def test_product_details_uses_source_facts_and_canonical_resolution(self):
        tools = DeterministicTools(
            inventory=inventory(),
            site_base_url="https://fixture.example",
        )

        result = tools.get_product_details("  LY198  ")

        self.assertEqual(result.product_id, "ly198")
        self.assertEqual(result.name, "测试品牌 LY198")
        self.assertEqual(result.category_name, "测试对讲机分类")
        self.assertEqual(result.product_features, "ly198 的权威产品说明")
        self.assertEqual(result.technical_parameters[0].items[0].value, "3W")
        self.assertEqual(
            result.sources[0].model_dump(mode="json"),
            {
                "title": "测试品牌 LY198",
                "url": "https://fixture.example/two-way-radio/ly198/",
            },
        )

    def test_product_details_rejects_invalid_unknown_and_display_name_inputs(self):
        tools = DeterministicTools(
            inventory=inventory(),
            site_base_url="https://fixture.example",
        )
        cases = (
            (None, ToolErrorCode.INVALID_ARGUMENT),
            (" ", ToolErrorCode.INVALID_ARGUMENT),
            ("x" * 129, ToolErrorCode.INVALID_ARGUMENT),
            ("missing", ToolErrorCode.PRODUCT_NOT_FOUND),
            ("测试品牌 LY198", ToolErrorCode.PRODUCT_NOT_FOUND),
            ("LY19", ToolErrorCode.PRODUCT_NOT_FOUND),
        )

        for value, code in cases:
            with self.subTest(value=value):
                with self.assertRaises(ToolError) as context:
                    tools.get_product_details(value)
                self.assertEqual(context.exception.code, code)
                if isinstance(value, str) and value.strip():
                    self.assertNotIn(value, context.exception.message)

    def test_product_details_reports_ambiguous_canonical_slug(self):
        tools = DeterministicTools(
            inventory=inventory(
                product("radio-one", slug="shared-model"),
                product("radio-two", slug="shared-model"),
            ),
            site_base_url="https://fixture.example",
        )

        with self.assertRaises(ToolError) as context:
            tools.get_product_details("SHARED MODEL")

        self.assertEqual(
            context.exception.code,
            ToolErrorCode.AMBIGUOUS_PRODUCT,
        )

    def test_contact_returns_only_authoritative_supported_fields(self):
        tools = DeterministicTools(
            inventory=inventory(),
            site_base_url="https://fixture.example",
        )

        result = tools.get_contact_info()
        payload = result.model_dump(mode="json")

        self.assertEqual(payload["company_name"], "测试公司 fixture-company")
        self.assertEqual(payload["duty_phone"], "13800138000")
        self.assertEqual(payload["email"], "fixture-company@example.com")
        self.assertNotIn("address", payload)
        self.assertEqual(
            payload["sources"],
            [{
                "title": "联系我们",
                "url": "https://fixture.example/about/#contact",
            }],
        )

    def test_contact_requires_exactly_one_published_record(self):
        for contacts in ({}, {"one": contact("one"), "two": contact("two")}):
            source_inventory = inventory()
            source_inventory.contacts = contacts
            tools = DeterministicTools(
                inventory=source_inventory,
                site_base_url="https://fixture.example",
            )
            with self.subTest(count=len(contacts)):
                with self.assertRaises(ToolError) as context:
                    tools.get_contact_info()
                self.assertEqual(
                    context.exception.code,
                    ToolErrorCode.TOOL_UNAVAILABLE,
                )

    def test_product_search_passes_fixed_backend_controls_and_omits_score(self):
        products = tuple(product(f"radio-{index}") for index in range(1, 6))
        source_inventory = inventory(*products)
        retriever = FakeRetriever([
            retrieval_result(item, rank)
            for rank, item in enumerate(products, start=1)
        ])
        tools = DeterministicTools(
            inventory=source_inventory,
            site_base_url="https://fixture.example",
            retriever=retriever,
        )

        result = tools.search_products("  适合酒店使用的对讲机  ")

        self.assertEqual(retriever.calls, [{
            "query": "适合酒店使用的对讲机",
            "top_k": 5,
            "allowed_types": {"product"},
            "unique_parent_documents": True,
        }])
        self.assertEqual(len(result.products), 5)
        self.assertEqual(
            [item.product_id for item in result.products],
            [item.id for item in products],
        )
        self.assertNotIn("score", result.model_dump_json())
        self.assertEqual(
            result.products[0].sources[0].url,
            "https://fixture.example/two-way-radio/radio-1/",
        )

    def test_product_search_validates_before_retrieval(self):
        retriever = FakeRetriever()
        tools = DeterministicTools(
            inventory=inventory(),
            site_base_url="https://fixture.example",
            retriever=retriever,
        )
        for value in (None, " ", "x" * 4001):
            with self.subTest(value=value):
                with self.assertRaises(ToolError) as context:
                    tools.search_products(value)
                self.assertEqual(
                    context.exception.code,
                    ToolErrorCode.INVALID_ARGUMENT,
                )
        self.assertEqual(retriever.calls, [])

    def test_product_search_distinguishes_unavailable_and_unexpected_failures(self):
        cases = (
            (None, ToolErrorCode.TOOL_UNAVAILABLE),
            (FakeRetriever(error=RetrievalError("secret dependency detail")), ToolErrorCode.TOOL_UNAVAILABLE),
            (FakeRetriever(error=RuntimeError("secret internal detail")), ToolErrorCode.TOOL_EXECUTION_ERROR),
        )
        for retriever, code in cases:
            tools = DeterministicTools(
                inventory=inventory(),
                site_base_url="https://fixture.example",
                retriever=retriever,
            )
            with self.subTest(code=code):
                with self.assertRaises(ToolError) as context:
                    tools.search_products("酒店对讲机")
                self.assertEqual(context.exception.code, code)
                self.assertNotIn("secret", context.exception.message)

    def test_exact_tools_redact_unexpected_internal_failures(self):
        tools = DeterministicTools(
            inventory=inventory(),
            site_base_url="https://fixture.example",
        )
        with patch.object(
            tools,
            "_source_ref",
            side_effect=RuntimeError("private source failure"),
        ):
            for operation in (
                lambda: tools.get_product_details("LY198"),
                tools.get_contact_info,
            ):
                with self.subTest(operation=operation):
                    with self.assertRaises(ToolError) as context:
                        operation()
                    self.assertEqual(
                        context.exception.code,
                        ToolErrorCode.TOOL_EXECUTION_ERROR,
                    )
                    self.assertNotIn(
                        "private source failure",
                        context.exception.message,
                    )

    def test_registry_is_immutable_and_contains_only_approved_tools(self):
        tools = DeterministicTools(
            inventory=inventory(),
            site_base_url="https://fixture.example",
        )

        registry = build_tool_registry(tools)

        self.assertEqual(
            set(registry),
            {"search_products", "get_product_details", "get_contact_info"},
        )
        self.assertEqual(
            registry["get_product_details"]("ly198").product_id,
            "ly198",
        )
        with self.assertRaises(TypeError):
            registry["arbitrary"] = lambda: None


if __name__ == "__main__":
    unittest.main()
