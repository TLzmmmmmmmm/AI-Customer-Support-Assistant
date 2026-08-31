import json
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from knowledge_pipeline.chunking import (
    COMPANY_SECTION_SLUGS,
    MAX_CHUNK_CHARACTERS,
    PRODUCT_OPTIONAL_SECTION_SLUGS,
    PRODUCT_SPEC_SECTION_SLUGS,
    SOLUTION_BODY_SECTION_SLUGS,
    ChunkCandidate,
    SemanticUnit,
    _parse_markdown,
    _split_h3,
    _validate_coverage,
    build_and_write_chunks,
    build_chunks,
    calculate_chunk_statistics,
    compute_chunk_content_hash,
    load_documents,
    serialize_chunks,
    validate_chunks,
)
from knowledge_pipeline.core import BuildError, compute_content_hash
from knowledge_pipeline.models import KnowledgeChunk, KnowledgeDocument
from scripts.build_knowledge_chunks import main as build_chunks_main


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
    payload = {
        "schema_version": "1.0",
        "chunk_id": "product:xir-p8668ex:features",
        "parent_document_id": "product:xir-p8668ex",
        "parent_document_hash": "a" * 64,
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
    payload["content_hash"] = compute_chunk_content_hash(
        payload["type"],
        payload["section"],
        payload["text"],
        payload["language"],
        payload["metadata"],
    )
    return payload


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

    def test_chunk_rejects_non_hex_parent_document_hash(self):
        payload = valid_chunk_payload()
        payload["parent_document_hash"] = "A" * 64
        with self.assertRaises(ValidationError):
            KnowledgeChunk.model_validate(payload)

    def test_chunk_rejects_non_hex_content_hash(self):
        payload = valid_chunk_payload()
        payload["content_hash"] = "A" * 64
        with self.assertRaises(ValidationError):
            KnowledgeChunk.model_validate(payload)

    def test_chunk_requires_both_hash_fields(self):
        for field in ("parent_document_hash", "content_hash"):
            with self.subTest(field=field):
                payload = valid_chunk_payload()
                del payload[field]
                with self.assertRaises(ValidationError):
                    KnowledgeChunk.model_validate(payload)

    def test_chunk_content_hash_uses_the_canonical_semantic_payload(self):
        payload = valid_chunk_payload()
        expected = compute_chunk_content_hash(
            "product",
            "产品特点",
            "# 摩托罗拉 XiR P8668Ex\n\n## 产品特点\n\n- 防爆机型",
            "zh-CN",
            {
                "product_id": "xir-p8668ex",
                "slug": "xir-p8668ex",
                "category_id": "two-way-radio",
                "category_name": "对讲机通信",
            },
        )

        self.assertEqual(payload["content_hash"], expected)

    def test_chunk_content_hash_changes_when_semantic_fields_change(self):
        payload = valid_chunk_payload()
        original = payload["content_hash"]
        changed_payloads = []
        for field, value in (
            ("section", "产品介绍"),
            ("text", payload["text"] + "\n\n补充事实。"),
            ("metadata", {**payload["metadata"], "slug": "changed"}),
        ):
            changed = {**payload, field: value}
            changed_payloads.append(changed)

        for changed in changed_payloads:
            with self.subTest(changed=changed):
                self.assertNotEqual(
                    original,
                    compute_chunk_content_hash(
                        changed["type"],
                        changed["section"],
                        changed["text"],
                        changed["language"],
                        changed["metadata"],
                    ),
                )

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

    def test_invalid_utf8_reports_input_path_and_decode_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "documents.jsonl"
            path.write_bytes(b"\xff\n")

            with self.assertRaises(BuildError) as context:
                load_documents(path)

        message = str(context.exception)
        self.assertIn(str(path), message)
        self.assertIn("reason:", message)
        self.assertIn("utf-8", message)
        self.assertIn("invalid start byte", message)

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


class ProductChunkingTests(unittest.TestCase):
    def test_product_splits_overview_features_and_parameter_groups(self):
        document = make_product_document()

        chunks = build_chunks([document])

        self.assertEqual(
            [chunk.chunk_id for chunk in chunks],
            [
                "product:xir-p8668ex:overview",
                "product:xir-p8668ex:features",
                "product:xir-p8668ex:spec:general",
            ],
        )
        self.assertEqual(chunks[2].section, "一般规格")
        self.assertIn("# 摩托罗拉 XiR P8668Ex", chunks[2].text)
        self.assertIn("## 技术参数", chunks[2].text)
        self.assertIn("### 一般规格", chunks[2].text)
        self.assertNotIn("应用场景", "\n".join(chunk.text for chunk in chunks))
        self.assertEqual(chunks[0].metadata, document.metadata)
        self.assertEqual(chunks[0].source_files, document.source_files)
        for chunk in chunks:
            self.assertEqual(chunk.parent_document_hash, document.content_hash)
            self.assertEqual(
                chunk.content_hash,
                compute_chunk_content_hash(
                    chunk.type,
                    chunk.section,
                    chunk.text,
                    chunk.language,
                    chunk.metadata,
                ),
            )

    def test_product_rejects_unmapped_parameter_group(self):
        document = make_product_document().model_copy(update={
            "text": make_product_document().text.replace("一般规格", "未来规格"),
        })

        with self.assertRaises(BuildError) as context:
            build_chunks([document])

        message = str(context.exception)
        self.assertIn("product:xir-p8668ex", message)
        self.assertIn("未来规格", message)

    def test_product_rejects_duplicate_parameter_group_id(self):
        document = make_product_document().model_copy(update={
            "text": make_product_document().text
            + "\n\n### 一般规格\n\n- 工作电压：7.4V",
        })

        with self.assertRaises(BuildError) as context:
            build_chunks([document])

        message = str(context.exception)
        self.assertIn("product:xir-p8668ex", message)
        self.assertIn("duplicate", message)


class SolutionChunkingTests(unittest.TestCase):
    def test_solution_keeps_h3_children_in_their_real_h2_chunk(self):
        chunks = build_chunks([make_solution_document()])
        by_id = {chunk.chunk_id: chunk for chunk in chunks}

        self.assertEqual(
            list(by_id),
            [
                "solution:smart-emergency:summary",
                "solution:smart-emergency:core-needs",
                "solution:smart-emergency:design",
                "solution:smart-emergency:features",
                "solution:smart-emergency:body:business-pain-points",
            ],
        )
        pain_points = by_id[
            "solution:smart-emergency:body:business-pain-points"
        ]
        self.assertEqual(pain_points.section, "业务痛点")
        self.assertIn("### 风险感知滞后", pain_points.text)
        self.assertIn("### 协同指挥低效", pain_points.text)

    def test_solution_preserves_direct_details_prose_without_empty_wrapper_chunk(self):
        direct_prose = make_solution_document().text.replace(
            "## 详细内容\n\n## 业务痛点",
            "## 详细内容\n\n详细内容的直接事实。\n\n## 业务痛点",
        )
        with_details = build_chunks([make_solution_document(direct_prose)])
        without_details = build_chunks([make_solution_document()])

        self.assertIn(
            "solution:smart-emergency:details",
            [chunk.chunk_id for chunk in with_details],
        )
        details = next(
            chunk
            for chunk in with_details
            if chunk.chunk_id == "solution:smart-emergency:details"
        )
        self.assertIn("详细内容的直接事实。", details.text)
        self.assertNotIn(
            "solution:smart-emergency:details",
            [chunk.chunk_id for chunk in without_details],
        )

    def test_solution_rejects_unmapped_or_duplicate_body_heading(self):
        base = make_solution_document().text
        documents = (
            make_solution_document(base.replace("业务痛点", "未来章节", 1)),
            make_solution_document(base + "\n\n## 业务痛点\n\n重复章节。"),
        )

        for document, expected in zip(documents, ("未来章节", "duplicate")):
            with self.subTest(expected=expected):
                with self.assertRaises(BuildError) as context:
                    build_chunks([document])
                message = str(context.exception)
                self.assertIn(document.document_id, message)
                self.assertIn(expected, message)


class OtherTypeChunkingTests(unittest.TestCase):
    def test_short_support_company_and_contact_remain_whole(self):
        documents = [
            make_support_document(),
            make_company_document(),
            make_contact_document(),
        ]

        chunks = build_chunks(documents)

        self.assertEqual(
            [chunk.chunk_id for chunk in chunks],
            [
                "support:solution-design:content",
                "company:shengborun:company-profile",
                "contact:shengborun:contact",
            ],
        )
        self.assertEqual(chunks[0].section, "方案设计")
        self.assertEqual(chunks[0].text, documents[0].text)
        self.assertEqual(chunks[1].text, documents[1].text)
        self.assertEqual(chunks[2].text, documents[2].text)

    def test_company_rejects_unmapped_future_h2(self):
        document = make_company_document().model_copy(update={
            "text": make_company_document().text + "\n\n## 企业愿景\n\n成为行业伙伴。",
        })

        with self.assertRaises(BuildError) as context:
            build_chunks([document])

        message = str(context.exception)
        self.assertIn("company:shengborun", message)
        self.assertIn("企业愿景", message)

    def test_company_preamble_is_preserved_in_first_section(self):
        document = make_company_document().model_copy(update={
            "text": make_company_document().text.replace(
                "\n\n## 公司简介",
                "\n\n统一社会信用代码：测试值\n\n## 公司简介",
            ),
        })

        chunks = build_chunks([document])

        self.assertEqual(len(chunks), 1)
        self.assertIn("统一社会信用代码：测试值", chunks[0].text)

    def test_company_without_h2_falls_back_to_one_complete_overview_chunk(self):
        facts = ("公司成立于2011年。", "公司专注通信行业需求。")
        document = make_document(
            type_="company",
            entity_id="shengborun",
            title="北京盛博润通信设备有限公司",
            text=(
                "# 北京盛博润通信设备有限公司\n\n"
                f"{facts[0]}\n\n{facts[1]}"
            ),
            source_url="https://www.shengborun.com/about/#company",
            source_files=["src/pages/about.astro"],
            metadata={"company_id": "shengborun"},
        )

        chunks = build_chunks([document])

        self.assertEqual([chunk.chunk_id for chunk in chunks], [
            "company:shengborun:overview",
        ])
        self.assertEqual(chunks[0].section, document.title)
        self.assertEqual(chunks[0].text, document.text)
        for fact in facts:
            self.assertEqual(chunks[0].text.count(fact), 1)

    def test_company_without_h2_splits_at_natural_boundaries_with_stable_suffixes(self):
        facts = ("第一段完整公司事实" * 4, "第二段完整公司事实" * 4)
        document = make_document(
            type_="company",
            entity_id="shengborun",
            title="北京盛博润通信设备有限公司",
            text=(
                "# 北京盛博润通信设备有限公司\n\n"
                f"{facts[0]}\n\n{facts[1]}"
            ),
            source_url="https://www.shengborun.com/about/#company",
            source_files=["src/pages/about.astro"],
            metadata={"company_id": "shengborun"},
        )

        chunks = build_chunks([document], max_characters=70)

        self.assertEqual([chunk.chunk_id for chunk in chunks], [
            "company:shengborun:overview:1",
            "company:shengborun:overview:2",
        ])
        combined = "\n".join(chunk.text for chunk in chunks)
        for fact in facts:
            self.assertEqual(combined.count(fact), 1)


class NaturalSplitAndCoverageTests(unittest.TestCase):

    def test_nested_h3_ancestry_repeats_across_splits_without_repeating_facts(self):
        facts = (
            "第一段风险事实" * 4,
            "第二段风险事实" * 4,
            "- 第三项风险事实" * 4,
        )
        nested_body = "### 持续风险\n\n" + "\n\n".join(facts)
        document = make_solution_document(
            make_solution_document().text.replace(
                "### 风险感知滞后\n\n风险发现不及时。\n\n"
                "### 协同指挥低效\n\n跨部门协同困难。",
                nested_body,
            )
        )

        chunks = build_chunks([document], max_characters=85)
        body_chunks = [
            chunk
            for chunk in chunks
            if chunk.chunk_id.startswith(
                "solution:smart-emergency:body:business-pain-points:"
            )
        ]

        self.assertGreaterEqual(len(body_chunks), 2)
        for chunk in body_chunks:
            self.assertIn("## 业务痛点", chunk.text)
            self.assertIn("### 持续风险", chunk.text)
        combined = "\n".join(chunk.text for chunk in body_chunks)
        for fact in facts:
            self.assertEqual(combined.count(fact), 1)

    def test_leading_support_summary_paragraphs_participate_in_soft_limit_packing(self):
        first_summary = "摘要甲" * 10
        second_summary = "摘要乙" * 10
        document = make_support_document().model_copy(update={
            "text": (
                "# 方案设计\n\n"
                f"## 服务摘要\n\n{first_summary}\n\n{second_summary}\n\n"
                "## 服务内容\n\n短内容。"
            ),
        })

        chunks = build_chunks([document], max_characters=80)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 80 for chunk in chunks))
        combined = "\n".join(chunk.text for chunk in chunks)
        self.assertEqual(combined.count(first_summary), 1)
        self.assertEqual(combined.count(second_summary), 1)

    def test_list_item_keeps_its_indented_continuation_paragraph_indivisible(self):
        first_line = "- 主项首段主项首段"
        continuation = "  延续段落延续段落"
        feature_text = f"{first_line}\n\n{continuation}\n\n- 第二项内容第二项内容"
        document = make_solution_document(
            make_solution_document().text.replace("- 协同指挥", feature_text)
        )

        chunks = build_chunks([document], max_characters=35)

        feature_chunks = [
            chunk for chunk in chunks if ":features:" in chunk.chunk_id
        ]
        self.assertEqual(len(feature_chunks), 2)
        self.assertIn(first_line, feature_chunks[0].text)
        self.assertIn(continuation, feature_chunks[0].text)
        self.assertNotIn(continuation, feature_chunks[1].text)

    def test_oversized_section_splits_only_between_complete_list_items(self):
        document = make_solution_document(text=long_feature_solution_text())

        chunks = build_chunks([document], max_characters=50)

        feature_chunks = [
            chunk for chunk in chunks if ":features:" in chunk.chunk_id
        ]
        self.assertEqual(
            [chunk.chunk_id.rsplit(":", 1)[-1] for chunk in feature_chunks],
            ["1", "2"],
        )
        combined = "\n".join(chunk.text for chunk in feature_chunks)
        for item in ("- 特点甲完整内容", "- 特点乙完整内容"):
            self.assertEqual(combined.count(item), 1)

    def test_paragraphs_pack_greedily_without_truncation_or_repetition(self):
        paragraphs = [
            "第一段完整事实" * 3,
            "第二段完整事实" * 3,
            "第三段完整事实" * 3,
        ]
        document = make_support_document(body="\n\n".join(paragraphs))

        chunks = build_chunks([document], max_characters=90)

        self.assertEqual(
            [chunk.chunk_id for chunk in chunks],
            [
                "support:solution-design:content:1",
                "support:solution-design:content:2",
            ],
        )
        self.assertIn(paragraphs[0], chunks[0].text)
        self.assertIn(paragraphs[1], chunks[0].text)
        self.assertIn(paragraphs[2], chunks[1].text)
        combined = "\n".join(chunk.text for chunk in chunks)
        for paragraph in paragraphs:
            self.assertEqual(combined.count(paragraph), 1)

    def test_indivisible_paragraph_may_exceed_soft_limit(self):
        body = "连续事实" * 80
        document = make_support_document(body=body)

        chunks = build_chunks([document], max_characters=100)

        body_chunks = [chunk for chunk in chunks if body in chunk.text]
        self.assertEqual(len(body_chunks), 1)
        self.assertGreater(len(body_chunks[0].text), 100)
        self.assertEqual("\n".join(chunk.text for chunk in chunks).count(body), 1)

    def test_contact_never_enters_soft_limit_splitter(self):
        document = make_contact_document()

        chunks = build_chunks([document], max_characters=10)

        self.assertEqual([chunk.chunk_id for chunk in chunks], [
            "contact:shengborun:contact",
        ])
        self.assertEqual(chunks[0].text, document.text)

    def test_coverage_rejects_a_factual_unit_assigned_twice(self):
        document = make_support_document()
        chunk = build_chunks([document])[0]
        unit = SemanticUnit("fact", "确认客户需求。")
        candidate = ChunkCandidate(chunk, (unit.key,))

        with self.assertRaises(BuildError) as context:
            _validate_coverage(document, [unit], [candidate, candidate])

        self.assertIn("assigned more than once", str(context.exception))


class ChunkCollectionValidationTests(unittest.TestCase):
    def test_duplicate_chunk_id_is_fatal(self):
        documents = [make_product_document()]
        chunks = build_chunks(documents)

        with self.assertRaises(BuildError) as context:
            validate_chunks([chunks[0], chunks[0]], documents)

        self.assertIn("duplicate chunk_id", str(context.exception))

    def test_unknown_parent_id_is_fatal(self):
        documents = [make_product_document()]
        changed = build_chunks(documents)[0].model_copy(
            update={"parent_document_id": "product:unknown"}
        )

        with self.assertRaises(BuildError) as context:
            validate_chunks([changed], documents)

        self.assertIn("unknown parent_document_id", str(context.exception))

    def test_unknown_parent_and_stale_content_hash_report_both_errors(self):
        documents = [make_product_document()]
        changed = build_chunks(documents)[0].model_copy(update={
            "parent_document_id": "product:unknown",
            "content_hash": "b" * 64,
        })

        with self.assertRaises(BuildError) as context:
            validate_chunks([changed], documents)

        message = str(context.exception)
        self.assertIn("unknown parent_document_id", message)
        self.assertIn("field: content_hash", message)

    def test_inherited_source_url_must_match_parent(self):
        documents = [make_product_document()]
        chunks = build_chunks(documents)
        changed = chunks[0].model_copy(
            update={"source_url": "https://example.com/wrong"}
        )

        with self.assertRaises(BuildError) as context:
            validate_chunks([changed, *chunks[1:]], documents)

        self.assertIn("source_url", str(context.exception))

    def test_stale_chunk_content_hash_is_fatal_after_semantic_change(self):
        documents = [make_product_document()]
        chunks = build_chunks(documents)
        changed_metadata = chunks[0].metadata.model_copy(update={"slug": "changed"})

        for update in (
            {"content_hash": "b" * 64},
            {"section": "产品特点"},
            {"text": chunks[0].text + "\n\n补充事实。"},
            {"metadata": changed_metadata},
        ):
            with self.subTest(update=update):
                changed = chunks[0].model_copy(update=update)
                with self.assertRaises(BuildError) as context:
                    validate_chunks([changed, *chunks[1:]], documents)

                self.assertIn("field: content_hash", str(context.exception))

    def test_every_parent_must_have_a_chunk(self):
        documents = [make_product_document(), make_contact_document()]
        chunks = build_chunks(documents[:-1])

        with self.assertRaises(BuildError) as context:
            validate_chunks(chunks, documents)

        self.assertIn("has no chunks", str(context.exception))

    def test_chunk_order_must_match_the_deterministic_builder(self):
        documents = [make_product_document()]
        chunks = build_chunks(documents)

        with self.assertRaises(BuildError) as context:
            validate_chunks(list(reversed(chunks)), documents)

        self.assertIn("order or content", str(context.exception))


class ChunkSerializationTests(unittest.TestCase):

    def test_atomic_write_preserves_u2028_u2029_records_and_exact_bytes(self):
        document = make_support_document("第一段\u2028第二段\u2029第三段")
        expected = serialize_chunks(build_chunks([document]))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "documents.jsonl"
            output_path = root / "chunks.jsonl"
            with temporary_document_file([document]) as source:
                input_path.write_bytes(source.read_bytes())

            build_and_write_chunks(input_path, output_path)

            actual = output_path.read_bytes()

        self.assertEqual(actual, expected)
        self.assertIn("\u2028".encode("utf-8"), actual)
        self.assertIn("\u2029".encode("utf-8"), actual)

    def test_same_semantic_payload_has_same_hash_and_jsonl_bytes(self):
        first = KnowledgeChunk.model_validate(valid_chunk_payload())
        reordered = valid_chunk_payload()
        reordered["metadata"] = dict(reversed(list(reordered["metadata"].items())))
        reordered["content_hash"] = compute_chunk_content_hash(
            reordered["type"],
            reordered["section"],
            reordered["text"],
            reordered["language"],
            reordered["metadata"],
        )
        second = KnowledgeChunk.model_validate(reordered)

        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(serialize_chunks([first]), serialize_chunks([second]))

    def test_serialization_is_utf8_lf_and_byte_deterministic(self):
        documents = [make_product_document(), make_contact_document()]
        chunks = build_chunks(documents)

        first = serialize_chunks(chunks)
        second = serialize_chunks(build_chunks(documents))

        self.assertEqual(first, second)
        self.assertEqual(chunks[0].content_hash, build_chunks(documents)[0].content_hash)
        self.assertNotIn(b"\r", first)
        self.assertTrue(first.endswith(b"\n"))
        self.assertFalse(first.endswith(b"\n\n"))
        self.assertIn("产品特点".encode("utf-8"), first)
        lines = first.decode("utf-8").splitlines()
        self.assertEqual(len(lines), len(chunks))
        self.assertTrue(all(": " not in line for line in lines))
        self.assertEqual(
            list(json.loads(lines[0])),
            list(chunks[0].model_dump(mode="json")),
        )

    def test_failed_build_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "documents.jsonl"
            output_path = root / "chunks.jsonl"
            input_path.write_text("{invalid}\n", encoding="utf-8")
            output_path.write_bytes(b"previous-output\n")

            with self.assertRaises(BuildError):
                build_and_write_chunks(input_path, output_path)

            self.assertEqual(output_path.read_bytes(), b"previous-output\n")


class ChunkStatisticsTests(unittest.TestCase):
    def test_statistics_use_unicode_character_lengths(self):
        documents = [make_product_document(), make_contact_document()]
        chunks = build_chunks(documents)

        stats = calculate_chunk_statistics(documents, chunks)
        lengths = [len(chunk.text) for chunk in chunks]

        self.assertEqual(stats.total_documents, 2)
        self.assertEqual(stats.total_chunks, len(chunks))
        self.assertEqual(stats.average_characters, sum(lengths) / len(lengths))
        self.assertEqual(stats.median_characters, 48.5)
        self.assertEqual(stats.minimum_characters, min(lengths))
        self.assertEqual(stats.maximum_characters, max(lengths))
        self.assertEqual(stats.chunks_per_document, {
            "product:xir-p8668ex": 3,
            "contact:shengborun": 1,
        })
        self.assertEqual(stats.chunks_per_type, {"contact": 1, "product": 3})

    def test_statistics_handle_an_empty_collection(self):
        stats = calculate_chunk_statistics([], [])

        self.assertEqual(stats.total_documents, 0)
        self.assertEqual(stats.total_chunks, 0)
        self.assertEqual(stats.average_characters, 0.0)
        self.assertEqual(stats.median_characters, 0.0)
        self.assertEqual(stats.minimum_characters, 0)
        self.assertEqual(stats.maximum_characters, 0)


class BuildKnowledgeChunksCliTests(unittest.TestCase):
    def test_cli_writes_chunks_and_prints_labeled_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "documents.jsonl"
            output_path = root / "chunks.jsonl"
            with temporary_document_file([make_contact_document()]) as source:
                input_path.write_bytes(source.read_bytes())
            stdout = StringIO()

            with redirect_stdout(stdout):
                exit_code = build_chunks_main([
                    "--input", str(input_path), "--output", str(output_path),
                ])

        self.assertEqual(exit_code, 0)
        self.assertIn("Documents processed: 1", stdout.getvalue())
        self.assertIn("Chunks generated: 1", stdout.getvalue())
        self.assertIn("- contact:shengborun: 1", stdout.getvalue())
        self.assertIn("- contact: 1", stdout.getvalue())
        self.assertIn(f"Output: {output_path}", stdout.getvalue())

    def test_cli_reports_build_error_to_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "documents.jsonl"
            output_path = root / "chunks.jsonl"
            input_path.write_text("{invalid}\n", encoding="utf-8")
            stderr = StringIO()

            with redirect_stderr(stderr):
                exit_code = build_chunks_main([
                    "--input", str(input_path), "--output", str(output_path),
                ])

        self.assertEqual(exit_code, 1)
        self.assertIn("invalid JSON", stderr.getvalue())


class RealInventoryIntegrationTests(unittest.TestCase):
    def test_checked_in_headings_are_covered_by_explicit_registries(self):
        input_path = Path(__file__).resolve().parents[1] / "knowledge" / "documents.jsonl"
        documents = load_documents(input_path)

        for document in documents:
            parsed = _parse_markdown(document)
            headings = [section.heading for section in parsed.h2_sections]
            with self.subTest(document=document.document_id):
                if document.type == "product":
                    self.assertTrue(
                        set(headings).issubset(
                            {"产品介绍", "产品特点", "技术参数"}
                            | set(PRODUCT_OPTIONAL_SECTION_SLUGS)
                        )
                    )
                    technical_parameters = next(
                        section
                        for section in parsed.h2_sections
                        if section.heading == "技术参数"
                    )
                    self.assertTrue(
                        {
                            section.heading
                            for section in _split_h3(technical_parameters)
                        }.issubset(PRODUCT_SPEC_SECTION_SLUGS)
                    )
                elif document.type == "solution":
                    self.assertEqual(
                        headings[:5],
                        [
                            "方案摘要",
                            "核心需求",
                            "方案设计",
                            "方案特点",
                            "详细内容",
                        ],
                    )
                    self.assertTrue(
                        set(headings[5:]).issubset(SOLUTION_BODY_SECTION_SLUGS)
                    )
                elif document.type == "company":
                    self.assertTrue(set(headings).issubset(COMPANY_SECTION_SLUGS))

    def test_checked_in_documents_build_checked_in_chunks(self):
        repository_root = Path(__file__).resolve().parents[1]
        input_path = repository_root / "knowledge" / "documents.jsonl"
        output_path = repository_root / "knowledge" / "chunks.jsonl"
        documents = load_documents(input_path)
        chunks = build_chunks(documents)
        self.assertEqual(len(documents), 60)
        self.assertEqual(
            {chunk.parent_document_id for chunk in chunks},
            {document.document_id for document in documents},
        )
        artifact_records = [
            json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertTrue(artifact_records)
        self.assertTrue(all("parent_document_hash" in record for record in artifact_records))
        self.assertTrue(all("content_hash" in record for record in artifact_records))
        self.assertTrue(all("parent_content_hash" not in record for record in artifact_records))
        self.assertEqual(output_path.read_bytes(), serialize_chunks(chunks))
