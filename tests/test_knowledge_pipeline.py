import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from knowledge_pipeline.core import (
    BuildError,
    build_documents,
    load_sources,
    load_source_inventory,
    serialize_documents,
    validate_documents,
)
from knowledge_pipeline.models import CatalogMetadata, ProductSource


BASE_URL = "https://www.shengborun.com"


def provenance(reference: str) -> list[dict[str, str]]:
    return [{"kind": "website_file", "reference": reference}]


def valid_product() -> dict:
    return {
        "id": "xir-p8668ex",
        "name": "摩托罗拉 XiR P8668Ex",
        "slug": "xir-p8668ex",
        "category_id": "two-way-radio",
        "key_features": ["防爆机型", "音质清晰"],
        "product_features": "数字防爆对讲机，具备ATEX安全认证。",
        "technical_parameters": [{
            "group": "一般规格",
            "items": [{"name": "输出功率", "value": "1W"}],
        }],
        "source_path": "/two-way-radio/xir-p8668ex/",
        "published": True,
        "provenance": provenance(
            "src/content/products/two-way-radio/xir-p8668ex.json"
        ),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_complete_fixture(root: Path) -> None:
    write_json(root / "products" / "xir-p8668ex.json", valid_product())
    write_json(root / "product-categories" / "two-way-radio.json", {
        "id": "two-way-radio",
        "name": "对讲机通信",
        "slug": "two-way-radio",
        "short_description": "专业可靠的即时通信设备。",
        "source_path": "/two-way-radio/",
        "published": True,
        "provenance": provenance(
            "src/content/product-categories/two-way-radio.json"
        ),
    })
    write_json(root / "solutions" / "petrochemical.json", {
        "id": "petrochemical",
        "name": "石油石化行业无线对讲解决方案",
        "slug": "petrochemical",
        "summary": "解决厂区通信问题。",
        "core_needs": ["统一指挥调度"],
        "solution_design": "建设数字集群系统。",
        "features": ["动态信道分配"],
        "body_markdown": "## 行业背景\n\n正文。",
        "source_path": "/solutions/petrochemical/",
        "published": True,
        "provenance": provenance("src/content/solutions/petrochemical.md"),
    })
    write_json(root / "support" / "solution-design.json", {
        "id": "solution-design",
        "name": "方案设计",
        "summary": "确认客户实际需求。",
        "body": "围绕覆盖范围和系统容量提供网络规划方案。",
        "source_path": "/support/#solution-design",
        "published": True,
        "provenance": provenance("src/data/support-services.ts"),
    })
    write_json(root / "company.json", {
        "id": "shengborun",
        "name": "北京盛博润通信设备有限公司",
        "introduction": ["公司成立于2011年。", "公司专注通信行业需求。"],
        "source_path": "/about/#company",
        "published": True,
        "provenance": provenance("src/pages/about.astro"),
    })
    write_json(root / "contact.json", {
        "id": "shengborun",
        "company_name": "北京盛博润通信设备有限公司",
        "duty_phone": "13911733859",
        "email": "lsk777@sina.com",
        "source_path": "/about/#contact",
        "published": True,
        "provenance": provenance("src/pages/about.astro"),
    })


class KnowledgeSchemaTests(unittest.TestCase):
    def test_product_schema_rejects_solution_relationship_fields(self):
        payload = valid_product()
        payload["solution_ids"] = ["petrochemical"]

        with self.assertRaises(ValidationError):
            ProductSource.model_validate(payload)

    def test_catalog_metadata_requires_sorted_unique_category_ids(self):
        valid = CatalogMetadata(
            catalog_id="products",
            category_ids=["ict-integration", "two-way-radio"],
        )
        self.assertEqual(valid.catalog_id, "products")

        for category_ids in (
            ["two-way-radio", "ict-integration"],
            ["two-way-radio", "two-way-radio"],
            [],
        ):
            with self.subTest(category_ids=category_ids):
                with self.assertRaises(ValidationError):
                    CatalogMetadata(
                        catalog_id="products",
                        category_ids=category_ids,
                    )


class KnowledgePipelineTests(unittest.TestCase):
    def test_public_source_inventory_loader_applies_relationship_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_complete_fixture(root)

            inventory = load_source_inventory(root)
            self.assertIn("xir-p8668ex", inventory.products)

            payload = valid_product()
            payload["category_id"] = "missing-category"
            write_json(root / "products" / "xir-p8668ex.json", payload)
            with self.assertRaises(BuildError):
                load_source_inventory(root)

    def test_curated_snapshot_builds_61_normalized_documents(self):
        source_root = Path(__file__).resolve().parents[1] / "knowledge" / "source"

        inventory = load_sources(source_root)
        self.assertEqual(len(inventory.products), 49)
        self.assertEqual(len(inventory.categories), 4)
        self.assertEqual(len(inventory.solutions), 6)
        self.assertEqual(len(inventory.support), 3)
        self.assertEqual(len(inventory.companies), 1)
        self.assertEqual(len(inventory.contacts), 1)

        documents = build_documents(source_root, BASE_URL)
        counts = {
            type_: sum(document.type == type_ for document in documents)
            for type_ in (
                "catalog", "product", "solution", "support", "company", "contact"
            )
        }
        self.assertEqual(
            counts,
            {
                "catalog": 1,
                "product": 49,
                "solution": 6,
                "support": 3,
                "company": 1,
                "contact": 1,
            },
        )
        self.assertEqual(len(documents), 61)

    def test_build_normalizes_products_and_one_category_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_complete_fixture(root)

            documents = build_documents(root, BASE_URL)
            by_id = {document.document_id: document for document in documents}

            self.assertEqual(
                set(by_id),
                {
                    "catalog:products",
                    "product:xir-p8668ex",
                    "solution:petrochemical",
                    "support:solution-design",
                    "company:shengborun",
                    "contact:shengborun",
                },
            )
            catalog = by_id["catalog:products"]
            self.assertEqual(catalog.type, "catalog")
            self.assertEqual(catalog.title, "产品分类")
            self.assertEqual(catalog.source_path, "/products/")
            self.assertEqual(
                catalog.text,
                "# 产品分类\n\n"
                "网站目前展示以下 1 类产品。\n\n"
                "## 对讲机通信\n\n"
                "专业可靠的即时通信设备。",
            )
            self.assertEqual(
                catalog.metadata.model_dump(mode="json"),
                {
                    "catalog_id": "products",
                    "category_ids": ["two-way-radio"],
                },
            )
            self.assertEqual(
                catalog.source_files,
                ["src/content/product-categories/two-way-radio.json"],
            )
            product = by_id["product:xir-p8668ex"]
            self.assertEqual(
                product.text,
                "# 摩托罗拉 XiR P8668Ex\n\n"
                "产品分类：对讲机通信\n\n"
                "## 产品介绍\n\n"
                "数字防爆对讲机，具备ATEX安全认证。\n\n"
                "## 产品特点\n\n"
                "- 防爆机型\n"
                "- 音质清晰\n\n"
                "## 技术参数\n\n"
                "### 一般规格\n\n"
                "- 输出功率：1W",
            )
            self.assertEqual(
                product.model_dump(mode="json")["metadata"]["category_name"],
                "对讲机通信",
            )
            self.assertNotIn("solution", product.model_dump_json())

    def test_serialization_is_deterministic_sorted_utf8_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_complete_fixture(root)
            documents = list(reversed(build_documents(root, BASE_URL)))

            first = serialize_documents(documents)
            second = serialize_documents(list(reversed(documents)))

            self.assertEqual(first, second)
            self.assertNotIn(b"\r", first)
            self.assertTrue(first.endswith(b"\n"))
            self.assertFalse(first.endswith(b"\n\n"))
            self.assertIn("摩托罗拉".encode("utf-8"), first)
            ids = [
                json.loads(line)["document_id"]
                for line in first.decode("utf-8").splitlines()
            ]
            self.assertEqual(ids, sorted(ids))

    def test_duplicate_content_hash_is_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_complete_fixture(root)
            documents = build_documents(root, BASE_URL)
            duplicate = documents[0].model_copy(
                update={"document_id": "company:duplicate"}
            )

            with self.assertRaises(BuildError) as context:
                validate_documents([*documents, duplicate], BASE_URL)

            self.assertIn("duplicate content_hash", str(context.exception))

    def test_invalid_category_reference_reports_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_complete_fixture(root)
            payload = valid_product()
            payload["category_id"] = "missing-category"
            write_json(root / "products" / "xir-p8668ex.json", payload)

            with self.assertRaises(BuildError) as context:
                build_documents(root, BASE_URL)

            message = str(context.exception)
            self.assertIn("products/xir-p8668ex.json", message)
            self.assertIn("missing-category", message)


if __name__ == "__main__":
    unittest.main()
