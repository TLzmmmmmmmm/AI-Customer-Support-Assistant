import unittest

from pydantic import ValidationError

from knowledge_pipeline.chunking import (
    COMPANY_SECTION_SLUGS,
    MAX_CHUNK_CHARACTERS,
    PRODUCT_OPTIONAL_SECTION_SLUGS,
    PRODUCT_SPEC_SECTION_SLUGS,
    SOLUTION_BODY_SECTION_SLUGS,
)
from knowledge_pipeline.models import KnowledgeChunk


def valid_chunk_payload() -> dict:
    return {
        "schema_version": "1.0",
        "chunk_id": "product:xir-p8668ex:features",
        "parent_document_id": "product:xir-p8668ex",
        "parent_content_hash": "a" * 64,
        "type": "product",
        "section": "产品特点",
        "text": "# 摩托罗拉 XiR P8668Ex\n\n## 产品特点\n\n- 防爆机型",
        "language": "zh-CN",
        "source_url": "https://www.shengborun.com/two-way-radio/xir-p8668ex/",
        "source_files": ["src/content/products/two-way-radio/xir-p8668ex.json"],
        "metadata": {
            "product_id": "xir-p8668ex",
            "slug": "xir-p8668ex",
            "category_id": "two-way-radio",
            "category_name": "对讲机通信",
        },
    }


class KnowledgeChunkSchemaTests(unittest.TestCase):
    def test_valid_product_chunk(self):
        chunk = KnowledgeChunk.model_validate(valid_chunk_payload())
        self.assertEqual(chunk.section, "产品特点")

    def test_chunk_rejects_unknown_fields(self):
        payload = valid_chunk_payload()
        payload["unexpected"] = True
        with self.assertRaises(ValidationError):
            KnowledgeChunk.model_validate(payload)

    def test_chunk_rejects_non_ascii_id(self):
        payload = valid_chunk_payload()
        payload["chunk_id"] = "product:xir-p8668ex:产品特点"
        with self.assertRaises(ValidationError):
            KnowledgeChunk.model_validate(payload)

    def test_chunk_rejects_wrong_parent_type(self):
        payload = valid_chunk_payload()
        payload["parent_document_id"] = "solution:xir-p8668ex"
        with self.assertRaises(ValidationError):
            KnowledgeChunk.model_validate(payload)

    def test_chunk_rejects_non_hex_parent_hash(self):
        payload = valid_chunk_payload()
        payload["parent_content_hash"] = "A" * 64
        with self.assertRaises(ValidationError):
            KnowledgeChunk.model_validate(payload)

    def test_chunk_rejects_crlf_text(self):
        payload = valid_chunk_payload()
        payload["text"] = payload["text"].replace("\n", "\r\n")
        with self.assertRaises(ValidationError):
            KnowledgeChunk.model_validate(payload)

    def test_chunk_rejects_unsorted_source_files(self):
        payload = valid_chunk_payload()
        payload["source_files"] = ["src/z.json", "src/a.json"]
        with self.assertRaises(ValidationError):
            KnowledgeChunk.model_validate(payload)

    def test_chunk_rejects_wrong_metadata_type(self):
        payload = valid_chunk_payload()
        payload["type"] = "company"
        with self.assertRaises(ValidationError):
            KnowledgeChunk.model_validate(payload)


class HeadingRegistryTests(unittest.TestCase):
    def test_fixed_chunk_limit(self):
        self.assertEqual(MAX_CHUNK_CHARACTERS, 1000)

    def test_product_spec_registry(self):
        self.assertEqual(
            PRODUCT_SPEC_SECTION_SLUGS,
            {
                "安全防护": "security-protection", "安全上网": "secure-internet-access",
                "安全邮件": "secure-email", "负载均衡": "load-balancing",
                "高可靠性": "high-availability", "技术规格要求": "technical-requirements",
                "可选功能": "optional-features", "链路聚合": "link-aggregation",
                "绿色节能": "energy-efficiency", "设备接口": "device-interfaces",
                "数据分析": "data-analysis", "特色功能": "distinctive-features",
                "文件传输": "file-transfer", "物理指标": "physical-specifications",
                "系统参数": "system-parameters", "系统管理": "system-management",
                "一般规格": "general", "POE": "poe",
            },
        )

    def test_optional_solution_and_company_registries(self):
        self.assertEqual(PRODUCT_OPTIONAL_SECTION_SLUGS, {"应用场景": "applications"})
        self.assertEqual(
            SOLUTION_BODY_SECTION_SLUGS,
            {
                "方案概述": "overview", "应急救援解决方案": "emergency-rescue-solution",
                "行业背景": "industry-background", "解决方案": "solution",
                "系统功能": "system-functions", "行业通信现状": "industry-communications-status",
                "针对大型石油石化企业的数字集群系统解决方案": "digital-trunking-solution",
                "方案描述": "solution-description", "建设背景": "construction-background",
                "业务痛点": "business-pain-points", "客户需求": "customer-requirements",
            },
        )
        self.assertEqual(COMPANY_SECTION_SLUGS, {"公司简介": "company-profile"})
