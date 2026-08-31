from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from .core import BuildError, compute_content_hash
from .models import KnowledgeDocument


MAX_CHUNK_CHARACTERS = 1000


@dataclass(frozen=True)
class MarkdownSection:
    level: int
    heading: str
    raw: str
    body: str


@dataclass(frozen=True)
class ParsedDocument:
    title_line: str
    preamble: str
    h2_sections: tuple[MarkdownSection, ...]


def _document_error(
    input_path: Path,
    line_number: int,
    reason: str,
    document_id: str | None = None,
) -> str:
    lines = [f"[ERROR] {input_path}", f"line: {line_number}"]
    if document_id:
        lines.append(f"document_id: {document_id}")
    lines.append(f"reason: {reason}")
    return "\n".join(lines)


def _validation_field_path(location: tuple[object, ...]) -> str:
    result = ""
    for part in location:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            result += ("." if result else "") + str(part)
    return result


def load_documents(input_path: Path) -> list[KnowledgeDocument]:
    if not input_path.is_file():
        raise BuildError(f"[ERROR] {input_path}\nreason: input file does not exist")

    try:
        contents = input_path.read_text(encoding="utf-8")
    except OSError as error:
        raise BuildError(f"[ERROR] {input_path}\nreason: {error}") from error
    lines = [] if not contents else contents.split("\n")
    if lines and contents.endswith("\n"):
        lines.pop()

    documents: list[KnowledgeDocument] = []
    document_ids: set[str] = set()
    errors: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            errors.append(_document_error(input_path, line_number, "empty line"))
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(_document_error(
                input_path,
                line_number,
                f"invalid JSON: {error.msg}",
            ))
            continue

        document_id = payload.get("document_id") if isinstance(payload, dict) else None
        try:
            document = KnowledgeDocument.model_validate(payload)
        except ValidationError as error:
            messages = [
                _document_error(
                    input_path,
                    line_number,
                    f"field: {_validation_field_path(item['loc'])}\nreason: {item['msg']}",
                    document_id,
                )
                for item in error.errors(include_url=False)
            ]
            errors.extend(messages)
            continue

        expected_hash = compute_content_hash(
            document.type,
            document.title,
            document.text,
            document.metadata,
        )
        if document.content_hash != expected_hash:
            errors.append(_document_error(
                input_path,
                line_number,
                "field: content_hash\nreason: does not match normalized semantic content",
                document.document_id,
            ))
            continue
        if document.document_id in document_ids:
            errors.append(_document_error(
                input_path,
                line_number,
                "duplicate document_id",
                document.document_id,
            ))
            continue
        document_ids.add(document.document_id)
        documents.append(document)
    if errors:
        raise BuildError(errors)
    return documents


def _make_section(level: int, heading: str, raw: str) -> MarkdownSection:
    heading_line, _, body = raw.partition("\n")
    assert heading_line == "#" * level + " " + heading
    return MarkdownSection(level, heading, raw, body.lstrip("\n"))


def _parse_markdown(document: KnowledgeDocument) -> ParsedDocument:
    lines = document.text.split("\n")
    expected_title = f"# {document.title}"
    if not lines or lines[0] != expected_title:
        raise BuildError(
            f"[ERROR] {document.document_id}\nfield: text\n"
            f"reason: expected first line {expected_title}"
        )

    matches = list(re.finditer(r"(?m)^## (.+)$", document.text))
    preamble_end = matches[0].start() if matches else len(document.text)
    preamble = document.text[len(expected_title):preamble_end].strip("\n")
    sections = tuple(
        _make_section(
            2,
            match.group(1),
            document.text[match.start():matches[index + 1].start() if index + 1 < len(matches) else len(document.text)],
        )
        for index, match in enumerate(matches)
    )
    return ParsedDocument(expected_title, preamble, sections)


def _split_h3(section: MarkdownSection) -> list[MarkdownSection]:
    matches = list(re.finditer(r"(?m)^### (.+)$", section.raw))
    return [
        _make_section(
            3,
            match.group(1),
            section.raw[match.start():matches[index + 1].start() if index + 1 < len(matches) else len(section.raw)],
        )
        for index, match in enumerate(matches)
    ]

PRODUCT_SPEC_SECTION_SLUGS = {
    "安全防护": "security-protection", "安全上网": "secure-internet-access",
    "安全邮件": "secure-email", "负载均衡": "load-balancing",
    "高可靠性": "high-availability", "技术规格要求": "technical-requirements",
    "可选功能": "optional-features", "链路聚合": "link-aggregation",
    "绿色节能": "energy-efficiency", "设备接口": "device-interfaces",
    "数据分析": "data-analysis", "特色功能": "distinctive-features",
    "文件传输": "file-transfer", "物理指标": "physical-specifications",
    "系统参数": "system-parameters", "系统管理": "system-management",
    "一般规格": "general", "POE": "poe",
}

PRODUCT_OPTIONAL_SECTION_SLUGS = {"应用场景": "applications"}

SOLUTION_BODY_SECTION_SLUGS = {
    "方案概述": "overview", "应急救援解决方案": "emergency-rescue-solution",
    "行业背景": "industry-background", "解决方案": "solution",
    "系统功能": "system-functions", "行业通信现状": "industry-communications-status",
    "针对大型石油石化企业的数字集群系统解决方案": "digital-trunking-solution",
    "方案描述": "solution-description", "建设背景": "construction-background",
    "业务痛点": "business-pain-points", "客户需求": "customer-requirements",
}

COMPANY_SECTION_SLUGS = {"公司简介": "company-profile"}
