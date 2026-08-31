from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from .core import BuildError, compute_content_hash
from .models import KnowledgeChunk, KnowledgeDocument


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


@dataclass(frozen=True)
class SemanticUnit:
    key: str
    text: str


@dataclass(frozen=True)
class ChunkCandidate:
    chunk: KnowledgeChunk
    covered_unit_keys: tuple[str, ...]


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

SOLUTION_FIXED_SECTIONS = (
    ("方案摘要", "summary"),
    ("核心需求", "core-needs"),
    ("方案设计", "design"),
    ("方案特点", "features"),
)


def _chunk(
    parent: KnowledgeDocument,
    *,
    chunk_id: str,
    section: str,
    text: str,
) -> KnowledgeChunk:
    return KnowledgeChunk.model_validate({
        "schema_version": "1.0",
        "chunk_id": chunk_id,
        "parent_document_id": parent.document_id,
        "parent_content_hash": parent.content_hash,
        "type": parent.type,
        "section": section,
        "text": text,
        "language": parent.language,
        "source_url": parent.source_url,
        "source_files": parent.source_files,
        "metadata": parent.metadata.model_dump(mode="json"),
    })


def _error(parent: KnowledgeDocument, reason: str) -> BuildError:
    return BuildError(f"[ERROR] {parent.document_id}\nreason: {reason}")


def _join_context(*parts: str) -> str:
    return "\n\n".join(part.strip("\n") for part in parts if part.strip("\n"))


_LIST_ITEM = re.compile(r"^(?:[-+*]|\d+[.)])\s+")


def _natural_units(text: str) -> list[str]:
    stripped = text.strip("\n")
    if not stripped:
        return []

    units: list[str] = []
    for block in re.split(r"\n{2,}", stripped):
        current: list[str] = []
        current_is_list = False
        for line in block.split("\n"):
            is_list_item = bool(_LIST_ITEM.match(line))
            if is_list_item and current:
                units.append("\n".join(current))
                current = []
            elif not is_list_item and current and not current_is_list:
                current.append(line)
                continue
            current.append(line)
            current_is_list = is_list_item or current_is_list
        if current:
            units.append("\n".join(current))

    grouped: list[str] = []
    index = 0
    while index < len(units):
        unit = units[index]
        if re.fullmatch(r"#{3,6} .+", unit) and index + 1 < len(units):
            grouped.append(f"{unit}\n\n{units[index + 1]}")
            index += 2
        else:
            grouped.append(unit)
            index += 1
    return grouped


def _join_natural_units(units: list[str]) -> str:
    result = ""
    previous_was_list = False
    for unit in units:
        is_list = bool(_LIST_ITEM.match(unit))
        if result:
            result += "\n" if previous_was_list and is_list else "\n\n"
        result += unit
        previous_was_list = is_list
    return result


def _render_chunk_text(prefix: str, unit_texts: list[str]) -> str:
    return _join_context(prefix, _join_natural_units(unit_texts))


def _split_section(
    parent: KnowledgeDocument,
    *,
    base_chunk_id: str,
    section: str,
    first_prefix: str,
    repeat_prefix: str,
    content: str,
    full_text: str,
    unit_key_prefix: str,
    max_characters: int,
    leading_units: tuple[SemanticUnit, ...] = (),
) -> tuple[list[SemanticUnit], list[ChunkCandidate]]:
    content_units = [
        SemanticUnit(f"{unit_key_prefix}:{index}", unit)
        for index, unit in enumerate(_natural_units(content), start=1)
    ]
    if not content_units:
        raise _error(parent, f"section {section} has no factual content")

    groups: list[list[SemanticUnit]] = []
    current: list[SemanticUnit] = []
    for unit in content_units:
        prefix = first_prefix if not groups else repeat_prefix
        proposed = current + [unit]
        proposed_text = _render_chunk_text(
            prefix,
            [item.text for item in proposed],
        )
        if current and len(proposed_text) > max_characters:
            groups.append(current)
            current = [unit]
        else:
            current = proposed
    if current:
        groups.append(current)

    multiple = len(groups) > 1
    candidates: list[ChunkCandidate] = []
    for index, group in enumerate(groups, start=1):
        chunk_id = f"{base_chunk_id}:{index}" if multiple else base_chunk_id
        if multiple:
            prefix = first_prefix if index == 1 else repeat_prefix
            text = _render_chunk_text(prefix, [unit.text for unit in group])
        else:
            text = full_text.strip("\n")
        covered_keys = [unit.key for unit in group]
        if index == 1:
            covered_keys = [unit.key for unit in leading_units] + covered_keys
        candidates.append(ChunkCandidate(
            _chunk(
                parent,
                chunk_id=chunk_id,
                section=section,
                text=text,
            ),
            tuple(covered_keys),
        ))
    return [*leading_units, *content_units], candidates


def _validate_coverage(
    parent: KnowledgeDocument,
    units: list[SemanticUnit],
    candidates: list[ChunkCandidate],
) -> None:
    unit_keys = [unit.key for unit in units]
    duplicate_definitions = sorted(
        key for key, count in Counter(unit_keys).items() if count > 1
    )
    if duplicate_definitions:
        raise _error(
            parent,
            f"duplicate semantic unit keys: {duplicate_definitions}",
        )

    expected = set(unit_keys)
    assignment_counts = Counter(
        key
        for candidate in candidates
        for key in candidate.covered_unit_keys
    )
    assigned = set(assignment_counts)
    if assigned != expected:
        missing = sorted(expected - assigned)
        unexpected = sorted(assigned - expected)
        raise _error(
            parent,
            f"semantic coverage mismatch; missing={missing}, unexpected={unexpected}",
        )
    duplicated = sorted(
        key
        for key, count in assignment_counts.items()
        if count > 1 and not key.startswith("identity:")
    )
    if duplicated:
        raise _error(
            parent,
            f"factual semantic unit assigned more than once: {duplicated}",
        )

    unit_by_key = {unit.key: unit for unit in units}
    for candidate in candidates:
        for key in candidate.covered_unit_keys:
            if unit_by_key[key].text not in candidate.chunk.text:
                raise _error(
                    parent,
                    f"source text changed or missing for semantic unit {key}",
                )


def _product_candidates(
    parent: KnowledgeDocument,
    max_characters: int,
) -> tuple[list[SemanticUnit], list[ChunkCandidate]]:
    parsed = _parse_markdown(parent)
    required = ("产品介绍", "产品特点", "技术参数")
    actual_required = tuple(section.heading for section in parsed.h2_sections[:3])
    if actual_required != required:
        raise _error(
            parent,
            f"expected Product H2 sequence {list(required)}, got {list(actual_required)}",
        )

    introduction, features, specifications = parsed.h2_sections[:3]
    optional_sections = parsed.h2_sections[3:]
    for section in optional_sections:
        if section.heading not in PRODUCT_OPTIONAL_SECTION_SLUGS:
            raise _error(parent, f"unmapped Product H2 heading: {section.heading}")

    identity = SemanticUnit("identity:title", parsed.title_line)
    overview_leading = [identity]
    overview_first_prefix_parts = [parsed.title_line]
    if parsed.preamble:
        preamble_units = tuple(
            SemanticUnit(f"overview:preamble:{index}", unit)
            for index, unit in enumerate(_natural_units(parsed.preamble), start=1)
        )
        overview_leading.extend(preamble_units)
        overview_first_prefix_parts.append(parsed.preamble)
    overview_heading = f"## {introduction.heading}"
    overview_first_prefix_parts.append(overview_heading)
    units, candidates = _split_section(
        parent,
        base_chunk_id=f"{parent.document_id}:overview",
        section=introduction.heading,
        first_prefix=_join_context(*overview_first_prefix_parts),
        repeat_prefix=_join_context(parsed.title_line, overview_heading),
        content=introduction.body,
        full_text=_join_context(parsed.title_line, parsed.preamble, introduction.raw),
        unit_key_prefix="overview:content",
        max_characters=max_characters,
        leading_units=tuple(overview_leading),
    )
    feature_units, feature_candidates = _split_section(
        parent,
        base_chunk_id=f"{parent.document_id}:features",
        section=features.heading,
        first_prefix=_join_context(parsed.title_line, f"## {features.heading}"),
        repeat_prefix=_join_context(parsed.title_line, f"## {features.heading}"),
        content=features.body,
        full_text=_join_context(parsed.title_line, features.raw),
        unit_key_prefix="features:content",
        max_characters=max_characters,
    )
    units.extend(feature_units)
    candidates.extend(feature_candidates)
    generated_ids = {
        f"{parent.document_id}:overview",
        f"{parent.document_id}:features",
    }

    groups = _split_h3(specifications)
    if not groups:
        raise _error(parent, "Product 技术参数 must contain at least one H3 group")
    specifications_prefix = f"## {specifications.heading}"
    prefix_body = specifications.raw[len(specifications_prefix):]
    first_group_offset = prefix_body.find("### ")
    if first_group_offset >= 0 and prefix_body[:first_group_offset].strip():
        raise _error(parent, "Product 技术参数 contains unassigned prose before its first H3")
    for index, group in enumerate(groups):
        slug = PRODUCT_SPEC_SECTION_SLUGS.get(group.heading)
        if slug is None:
            raise _error(parent, f"unmapped Product H3 heading: {group.heading}")
        base_chunk_id = f"{parent.document_id}:spec:{slug}"
        if base_chunk_id in generated_ids:
            raise _error(parent, f"duplicate generated chunk_id: {base_chunk_id}")
        generated_ids.add(base_chunk_id)
        prefix = _join_context(
            parsed.title_line,
            specifications_prefix,
            f"### {group.heading}",
        )
        group_units, group_candidates = _split_section(
            parent,
            base_chunk_id=base_chunk_id,
            section=group.heading,
            first_prefix=prefix,
            repeat_prefix=prefix,
            content=group.body,
            full_text=_join_context(parsed.title_line, specifications_prefix, group.raw),
            unit_key_prefix=f"spec:{index}",
            max_characters=max_characters,
        )
        units.extend(group_units)
        candidates.extend(group_candidates)

    for index, section in enumerate(optional_sections):
        slug = PRODUCT_OPTIONAL_SECTION_SLUGS[section.heading]
        base_chunk_id = f"{parent.document_id}:{slug}"
        if base_chunk_id in generated_ids:
            raise _error(parent, f"duplicate generated chunk_id: {base_chunk_id}")
        generated_ids.add(base_chunk_id)
        prefix = _join_context(parsed.title_line, f"## {section.heading}")
        section_units, section_candidates = _split_section(
            parent,
            base_chunk_id=base_chunk_id,
            section=section.heading,
            first_prefix=prefix,
            repeat_prefix=prefix,
            content=section.body,
            full_text=_join_context(parsed.title_line, section.raw),
            unit_key_prefix=f"optional:{index}",
            max_characters=max_characters,
        )
        units.extend(section_units)
        candidates.extend(section_candidates)
    return units, candidates


def _solution_candidates(
    parent: KnowledgeDocument,
    max_characters: int,
) -> tuple[list[SemanticUnit], list[ChunkCandidate]]:
    parsed = _parse_markdown(parent)
    details_positions = [
        index
        for index, section in enumerate(parsed.h2_sections)
        if section.heading == "详细内容"
    ]
    if details_positions != [len(SOLUTION_FIXED_SECTIONS)]:
        raise _error(
            parent,
            "expected one Solution 详细内容 boundary after the four fixed sections",
        )
    details_index = details_positions[0]
    fixed_sections = parsed.h2_sections[:details_index]
    expected_headings = tuple(heading for heading, _ in SOLUTION_FIXED_SECTIONS)
    actual_headings = tuple(section.heading for section in fixed_sections)
    if actual_headings != expected_headings:
        raise _error(
            parent,
            f"expected Solution H2 sequence {list(expected_headings)}, got {list(actual_headings)}",
        )

    identity = SemanticUnit("identity:title", parsed.title_line)
    units: list[SemanticUnit] = []
    candidates: list[ChunkCandidate] = []
    for index, (section, (_, slug)) in enumerate(
        zip(fixed_sections, SOLUTION_FIXED_SECTIONS)
    ):
        heading = f"## {section.heading}"
        first_prefix_parts = [parsed.title_line]
        leading_units: list[SemanticUnit] = []
        repeat_prefix = _join_context(parsed.title_line, heading)
        if index == 0 and parsed.preamble:
            leading_units.append(identity)
            leading_units.extend(
                SemanticUnit(f"summary:preamble:{unit_index}", unit)
                for unit_index, unit in enumerate(
                    _natural_units(parsed.preamble), start=1
                )
            )
            first_prefix_parts.append(parsed.preamble)
        elif index == 0:
            leading_units.append(identity)
        first_prefix_parts.append(heading)
        section_units, section_candidates = _split_section(
            parent,
            base_chunk_id=f"{parent.document_id}:{slug}",
            section=section.heading,
            first_prefix=_join_context(*first_prefix_parts),
            repeat_prefix=repeat_prefix,
            content=section.body,
            full_text=_join_context(parsed.title_line, parsed.preamble if index == 0 else "", section.raw),
            unit_key_prefix=f"fixed:{index}",
            max_characters=max_characters,
            leading_units=tuple(leading_units),
        )
        units.extend(section_units)
        candidates.extend(section_candidates)

    details = parsed.h2_sections[details_index]
    if details.body.strip():
        prefix = _join_context(parsed.title_line, f"## {details.heading}")
        details_units, details_candidates = _split_section(
            parent,
            base_chunk_id=f"{parent.document_id}:details",
            section=details.heading,
            first_prefix=prefix,
            repeat_prefix=prefix,
            content=details.body,
            full_text=_join_context(parsed.title_line, details.raw),
            unit_key_prefix="details",
            max_characters=max_characters,
        )
        units.extend(details_units)
        candidates.extend(details_candidates)

    generated_ids = {candidate.chunk.chunk_id for candidate in candidates}
    for index, section in enumerate(parsed.h2_sections[details_index + 1:]):
        slug = SOLUTION_BODY_SECTION_SLUGS.get(section.heading)
        if slug is None:
            raise _error(parent, f"unmapped Solution H2 heading: {section.heading}")
        chunk_id = f"{parent.document_id}:body:{slug}"
        if chunk_id in generated_ids:
            raise _error(parent, f"duplicate generated chunk_id: {chunk_id}")
        generated_ids.add(chunk_id)
        prefix = _join_context(parsed.title_line, f"## {section.heading}")
        section_units, section_candidates = _split_section(
            parent,
            base_chunk_id=chunk_id,
            section=section.heading,
            first_prefix=prefix,
            repeat_prefix=prefix,
            content=section.body,
            full_text=_join_context(parsed.title_line, section.raw),
            unit_key_prefix=f"body:{index}",
            max_characters=max_characters,
        )
        units.extend(section_units)
        candidates.extend(section_candidates)
    return units, candidates


def _support_candidates(
    parent: KnowledgeDocument,
    max_characters: int,
) -> tuple[list[SemanticUnit], list[ChunkCandidate]]:
    parsed = _parse_markdown(parent)
    headings = tuple(section.heading for section in parsed.h2_sections)
    if headings != ("服务摘要", "服务内容"):
        raise _error(
            parent,
            f"expected Support H2 sequence ['服务摘要', '服务内容'], got {list(headings)}",
        )
    summary, content = parsed.h2_sections
    leading_units = [SemanticUnit("identity:title", parsed.title_line)]
    leading_units.extend(
        SemanticUnit(f"support:preamble:{index}", unit)
        for index, unit in enumerate(_natural_units(parsed.preamble), start=1)
    )
    leading_units.extend(
        SemanticUnit(f"support:summary:{index}", unit)
        for index, unit in enumerate(_natural_units(summary.body), start=1)
    )
    return _split_section(
        parent,
        base_chunk_id=f"{parent.document_id}:content",
        section=parent.title,
        first_prefix=_join_context(
            parsed.title_line,
            parsed.preamble,
            summary.raw,
            f"## {content.heading}",
        ),
        repeat_prefix=_join_context(parsed.title_line, f"## {content.heading}"),
        content=content.body,
        full_text=parent.text,
        unit_key_prefix="support:content",
        max_characters=max_characters,
        leading_units=tuple(leading_units),
    )


def _company_candidates(
    parent: KnowledgeDocument,
    max_characters: int,
) -> tuple[list[SemanticUnit], list[ChunkCandidate]]:
    parsed = _parse_markdown(parent)
    if not parsed.h2_sections:
        raise _error(parent, "Company document must contain at least one H2 section")

    identity = SemanticUnit("identity:title", parsed.title_line)
    units: list[SemanticUnit] = []
    candidates: list[ChunkCandidate] = []
    generated_ids: set[str] = set()
    for index, section in enumerate(parsed.h2_sections):
        slug = COMPANY_SECTION_SLUGS.get(section.heading)
        if slug is None:
            raise _error(parent, f"unmapped Company H2 heading: {section.heading}")
        chunk_id = f"{parent.document_id}:{slug}"
        if chunk_id in generated_ids:
            raise _error(parent, f"duplicate generated chunk_id: {chunk_id}")
        generated_ids.add(chunk_id)
        heading = f"## {section.heading}"
        first_prefix_parts = [parsed.title_line]
        leading_units: list[SemanticUnit] = []
        if index == 0:
            leading_units.append(identity)
            if parsed.preamble:
                leading_units.extend(
                    SemanticUnit(f"company:preamble:{unit_index}", unit)
                    for unit_index, unit in enumerate(
                        _natural_units(parsed.preamble), start=1
                    )
                )
                first_prefix_parts.append(parsed.preamble)
        first_prefix_parts.append(heading)
        section_units, section_candidates = _split_section(
            parent,
            base_chunk_id=chunk_id,
            section=section.heading,
            first_prefix=_join_context(*first_prefix_parts),
            repeat_prefix=_join_context(parsed.title_line, heading),
            content=section.body,
            full_text=_join_context(
                parsed.title_line,
                parsed.preamble if index == 0 else "",
                section.raw,
            ),
            unit_key_prefix=f"company:section:{index}",
            max_characters=max_characters,
            leading_units=tuple(leading_units),
        )
        units.extend(section_units)
        candidates.extend(section_candidates)
    return units, candidates


def _contact_candidates(
    parent: KnowledgeDocument,
) -> tuple[list[SemanticUnit], list[ChunkCandidate]]:
    unit = SemanticUnit("content", parent.text)
    candidate = ChunkCandidate(
        _chunk(
            parent,
            chunk_id=f"{parent.document_id}:contact",
            section=parent.title,
            text=parent.text,
        ),
        (unit.key,),
    )
    return [unit], [candidate]


def build_chunks(
    documents: list[KnowledgeDocument],
    max_characters: int = MAX_CHUNK_CHARACTERS,
) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        if document.type == "product":
            units, candidates = _product_candidates(document, max_characters)
        elif document.type == "solution":
            units, candidates = _solution_candidates(document, max_characters)
        elif document.type == "support":
            units, candidates = _support_candidates(document, max_characters)
        elif document.type == "company":
            units, candidates = _company_candidates(document, max_characters)
        elif document.type == "contact":
            units, candidates = _contact_candidates(document)
        else:
            raise _error(document, f"unsupported document type: {document.type}")
        _validate_coverage(document, units, candidates)
        chunks.extend(candidate.chunk for candidate in candidates)
    return chunks
