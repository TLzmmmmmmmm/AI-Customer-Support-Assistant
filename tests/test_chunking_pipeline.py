import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from knowledge_pipeline.chunking import (
    COMPANY_SECTION_SLUGS,
    MAX_CHUNK_CHARACTERS,
    PRODUCT_OPTIONAL_SECTION_SLUGS,
    PRODUCT_SPEC_SECTION_SLUGS,
    SOLUTION_BODY_SECTION_SLUGS,
    _parse_markdown,
    _split_h3,
    load_documents,
)
from knowledge_pipeline.core import BuildError, compute_content_hash
from knowledge_pipeline.models import KnowledgeChunk, KnowledgeDocument


def make_document(
    *,
    type_: str,
    entity_id: str,
    title: str,
    text: str,
    source_url: str,
    source_files: list[str],
    metadata: dict,
) -> KnowledgeDocument:
    provisional = KnowledgeDocument.model_validate({
        "schema_version": "1.0",
        "document_id": f"{type_}:{entity_id}",
        "type": type_,
        "title": title,
        "text": text,
        "language": "zh-CN",
        "source_path": "/fixture/",
        "source_url": source_url,
        "source_files": source_files,
        "content_hash": "0" * 64,
        "metadata": metadata,
    })
    return provisional.model_copy(update={
        "content_hash": compute_content_hash(
            provisional.type,
            provisional.title,
            provisional.text,
            provisional.metadata,
        )
    })


def make_product_document() -> KnowledgeDocument:
    return make_document(
        type_="product",
        entity_id="xir-p8668ex",
        title="摩托罗拉 XiR P8668Ex",
        text=(
            "# 摩托罗拉 XiR P8668Ex\n\n"
            "产品分类：对讲机通信\n\n"
            "## 产品介绍\n\n数字防爆对讲机。\n\n"
            "## 产品特点\n\n- 防爆机型\n- 音质清晰\n\n"
            "## 技术参数\n\n### 一般规格\n\n- 输出功率：1W"
        ),
        source_url="https://www.shengborun.com/two-way-radio/xir-p8668ex/",
        source_files=["src/content/products/two-way-radio/xir-p8668ex.json"],
        metadata={
            "product_id": "xir-p8668ex",
            "slug": "xir-p8668ex",
            "category_id": "two-way-radio",
            "category_name": "对讲机通信",
        },
    )


def make_solution_document(text: str | None = None) -> KnowledgeDocument:
    normalized = text or (
        "# 智慧应急解决方案\n\n"
        "## 方案摘要\n\n提升应急管理能力。\n\n"
        "## 核心需求\n\n- 统一数据\n\n"
        "## 方案设计\n\n建设应急平台。\n\n"
        "## 方案特点\n\n- 协同指挥\n\n"
        "## 详细内容\n\n"
        "## 业务痛点\n\n"
        "### 风险感知滞后\n\n风险发现不及时。\n\n"
        "### 协同指挥低效\n\n跨部门协同困难。"
    )
    return make_document(
        type_="solution",
        entity_id="smart-emergency",
        title="智慧应急解决方案",
        text=normalized,
        source_url="https://www.shengborun.com/solutions/smart-emergency/",
        source_files=["src/content/solutions/smart-emergency.md"],
        metadata={"solution_id": "smart-emergency", "slug": "smart-emergency"},
    )


def long_feature_solution_text() -> str:
    return (
        "# 智慧应急解决方案\n\n"
        "## 方案摘要\n\n提升应急管理能力。\n\n"
        "## 核心需求\n\n- 统一数据\n\n"
        "## 方案设计\n\n建设应急平台。\n\n"
        "## 方案特点\n\n"
        "- 特点甲完整内容特点甲完整内容特点甲完整内容\n"
        "- 特点乙完整内容特点乙完整内容特点乙完整内容\n\n"
        "## 详细内容\n\n"
        "## 业务痛点\n\n风险发现不及时。"
    )


def make_support_document(body: str = "围绕覆盖范围提供网络规划方案。") -> KnowledgeDocument:
    return make_document(
        type_="support",
        entity_id="solution-design",
        title="方案设计",
        text=f"# 方案设计\n\n## 服务摘要\n\n确认客户需求。\n\n## 服务内容\n\n{body}",
        source_url="https://www.shengborun.com/support/#solution-design",
        source_files=["src/data/support-services.ts"],
        metadata={"service_id": "solution-design"},
    )


def make_company_document() -> KnowledgeDocument:
    return make_document(
        type_="company",
        entity_id="shengborun",
        title="北京盛博润通信设备有限公司",
        text="# 北京盛博润通信设备有限公司\n\n## 公司简介\n\n公司成立于2011年。",
        source_url="https://www.shengborun.com/about/#company",
        source_files=["src/pages/about.astro"],
        metadata={"company_id": "shengborun"},
    )


def make_contact_document() -> KnowledgeDocument:
    return make_document(
        type_="contact",
        entity_id="shengborun",
        title="联系我们",
        text="# 联系我们\n\n公司名称：北京盛博润通信设备有限公司\n值班电话：13911733859\n电子邮箱：lsk777@sina.com",
        source_url="https://www.shengborun.com/about/#contact",
        source_files=["src/pages/about.astro"],
        metadata={"contact_id": "shengborun"},
    )


@contextmanager
def temporary_document_file(
    documents: list[KnowledgeDocument],
) -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "documents.jsonl"
        lines = [
            json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
            for item in documents
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        yield path


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


class DocumentLoaderTests(unittest.TestCase):
    def test_loads_zero_byte_file_as_no_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "documents.jsonl"
            path.write_text("", encoding="utf-8")

            loaded = load_documents(path)

        self.assertEqual(loaded, [])

    def test_loads_u2028_and_u2029_as_text_within_one_jsonl_record(self):
        document = make_support_document("第一段\u2028第二段\u2029第三段")

        with temporary_document_file([document]) as path:
            loaded = load_documents(path)

        self.assertEqual(loaded, [document])

    def test_loads_valid_documents_in_input_order(self):
        documents = [make_solution_document(), make_product_document()]

        with temporary_document_file(documents) as path:
            loaded = load_documents(path)

        self.assertEqual(loaded, documents)

    def test_invalid_json_reports_file_and_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "documents.jsonl"
            path.write_text("{}\n{bad json}\n", encoding="utf-8")
            with self.assertRaises(BuildError) as context:
                load_documents(path)
            message = str(context.exception)
            self.assertIn(str(path), message)
            self.assertIn("line: 2", message)
            self.assertIn("invalid JSON", message)

    def test_schema_error_reports_document_id_field_and_reason(self):
        document = make_product_document().model_dump(mode="json")
        document["title"] = ""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "documents.jsonl"
            path.write_text(json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaises(BuildError) as context:
                load_documents(path)

        message = str(context.exception)
        self.assertIn("line: 1", message)
        self.assertIn("document_id: product:xir-p8668ex", message)
        self.assertIn("field: title", message)
        self.assertIn("reason:", message)

    def test_duplicate_document_id_reports_duplicate(self):
        document = make_product_document()
        with temporary_document_file([document, document]) as path:
            with self.assertRaises(BuildError) as context:
                load_documents(path)

        self.assertIn("duplicate document_id", str(context.exception))
        self.assertIn("product:xir-p8668ex", str(context.exception))

    def test_stale_hash_reports_document_id(self):
        document = make_product_document().model_copy(
            update={"content_hash": "b" * 64}
        )
        with temporary_document_file([document]) as path:
            with self.assertRaises(BuildError) as context:
                load_documents(path)
        self.assertIn("product:xir-p8668ex", str(context.exception))
        self.assertIn("content_hash", str(context.exception))

    def test_empty_line_is_rejected_at_its_original_line_number(self):
        document = make_product_document()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "documents.jsonl"
            encoded = json.dumps(document.model_dump(mode="json"), ensure_ascii=False)
            path.write_text(encoded + "\n\n", encoding="utf-8", newline="\n")
            with self.assertRaises(BuildError) as context:
                load_documents(path)

        self.assertIn("line: 2", str(context.exception))
        self.assertIn("empty line", str(context.exception))


class MarkdownParserTests(unittest.TestCase):
    def test_parser_preserves_section_raw_text_without_reformatting(self):
        document = make_support_document("第一段。\n\n第二段。")

        parsed = _parse_markdown(document)

        self.assertEqual(
            parsed.h2_sections[0].raw,
            "## 服务摘要\n\n确认客户需求。\n\n",
        )

    def test_parser_preserves_preamble_h2_and_nested_h3_text(self):
        parsed = _parse_markdown(make_product_document())
        self.assertEqual(parsed.title_line, "# 摩托罗拉 XiR P8668Ex")
        self.assertEqual(parsed.preamble, "产品分类：对讲机通信")
        self.assertEqual(
            [section.heading for section in parsed.h2_sections],
            ["产品介绍", "产品特点", "技术参数"],
        )
        groups = _split_h3(parsed.h2_sections[2])
        self.assertEqual([group.heading for group in groups], ["一般规格"])
        self.assertIn("- 输出功率：1W", groups[0].raw)

    def test_parser_rejects_missing_or_mismatched_h1(self):
        missing_h1 = make_product_document().model_copy(
            update={"text": "产品分类：对讲机通信"}
        )
        mismatched_h1 = make_product_document().model_copy(
            update={"text": "# 别名\n\n## 产品介绍\n\n数字防爆对讲机。"}
        )

        for document in (missing_h1, mismatched_h1):
            with self.subTest(document=document.document_id, text=document.text):
                with self.assertRaises(BuildError):
                    _parse_markdown(document)

    def test_parser_only_splits_exact_heading_lines(self):
        document = make_support_document(
            "事实中包含 ## 不应分割，且包含 ### 也不是独立标题。"
        )

        parsed = _parse_markdown(document)

        self.assertEqual([section.heading for section in parsed.h2_sections], ["服务摘要", "服务内容"])
        self.assertEqual(_split_h3(parsed.h2_sections[1]), [])
