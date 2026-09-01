from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median

from pydantic import BaseModel, ValidationError

from .core import BuildError, compute_content_hash
from .models import KnowledgeChunk, KnowledgeDocument


MAX_CHUNK_CHARACTERS = 1000


def compute_chunk_content_hash(
    type_: str,
    section: str,
    text: str,
    language: str,
    metadata: BaseModel | dict[str, object],
) -> str:
    """Return the canonical semantic SHA-256 for one Chunk."""
    metadata_payload = (
        metadata.model_dump(mode="json")
        if isinstance(metadata, BaseModel)
        else metadata
    )
    payload = {
        "type": type_,
        "section": section,
        "text": text,
        "language": language,
        "metadata": metadata_payload,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


@dataclass(frozen=True)
class SemanticBlock:
    unit: SemanticUnit
    ancestor_headings: tuple[str, ...]


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


def _split_jsonl_records(contents: str) -> list[str]:
    """Split JSONL only on its LF record delimiter."""
    if not contents:
        return []
    records = contents.split("\n")
    if contents.endswith("\n"):
        records.pop()
    return records


def load_documents(input_path: Path) -> list[KnowledgeDocument]:
    if not input_path.is_file():
        raise BuildError(f"[ERROR] {input_path}\nreason: input file does not exist")

    try:
        contents = input_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise BuildError(f"[ERROR] {input_path}\nreason: {error}") from error
    lines = _split_jsonl_records(contents)

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
    content_hash = compute_chunk_content_hash(
        parent.type,
        section,
        text,
        parent.language,
        parent.metadata,
    )
    return KnowledgeChunk.model_validate({
        "schema_version": "1.0",
        "chunk_id": chunk_id,
        "parent_document_id": parent.document_id,
        "parent_document_hash": parent.content_hash,
        "type": parent.type,
        "section": section,
        "text": text,
        "language": parent.language,
        "content_hash": content_hash,
        "source_url": parent.source_url,
        "source_files": parent.source_files,
        "metadata": parent.metadata.model_dump(mode="json"),
    })


def _error(parent: KnowledgeDocument, reason: str) -> BuildError:
    return BuildError(f"[ERROR] {parent.document_id}\nreason: {reason}")


def _join_context(*parts: str) -> str:
    return "\n\n".join(part.strip("\n") for part in parts if part.strip("\n"))


_LIST_ITEM = re.compile(r"^(?:[-+*]|\d+[.)])\s+")
_MARKDOWN_SUBHEADING = re.compile(r"^(#{3,6}) .+$")
_FENCED_CODE_OPEN = re.compile(
    r"^(?:(?P<backtick>`{3,})[^`]*|(?P<tilde>~{3,}).*)$"
)


def _fenced_code_end(line: str, opening_fence: str) -> bool:
    marker = re.escape(opening_fence[0])
    return bool(re.fullmatch(
        rf" {{0,3}}{marker}{{{len(opening_fence)},}}[ \t]*",
        line,
    ))


def _natural_units(text: str) -> list[str]:
    stripped = text.strip("\n")
    if not stripped:
        return []

    units: list[str] = []
    lines = stripped.split("\n")
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue

        fence_match = _FENCED_CODE_OPEN.fullmatch(lines[index])
        if fence_match:
            opening_fence = (
                fence_match.group("backtick") or fence_match.group("tilde")
            )
            fenced_lines = [lines[index]]
            index += 1
            while index < len(lines):
                fenced_lines.append(lines[index])
                index += 1
                if _fenced_code_end(fenced_lines[-1], opening_fence):
                    break
            units.append("\n".join(fenced_lines))
            continue

        if _MARKDOWN_SUBHEADING.fullmatch(lines[index]):
            units.append(lines[index])
            index += 1
            continue

        if _LIST_ITEM.match(lines[index]):
            item_lines = [lines[index]]
            index += 1
            while index < len(lines):
                if _LIST_ITEM.match(lines[index]):
                    break
                if (
                    _MARKDOWN_SUBHEADING.fullmatch(lines[index])
                    or _FENCED_CODE_OPEN.fullmatch(lines[index])
                ):
                    break
                if lines[index].strip():
                    item_lines.append(lines[index])
                    index += 1
                    continue

                continuation = index
                while continuation < len(lines) and not lines[continuation].strip():
                    continuation += 1
                if (
                    continuation < len(lines)
                    and lines[continuation].startswith((" ", "\t"))
                ):
                    item_lines.extend(lines[index:continuation])
                    index = continuation
                    continue
                break
            units.append("\n".join(item_lines))
            continue

        paragraph = [lines[index]]
        index += 1
        while (
            index < len(lines)
            and lines[index].strip()
            and not _LIST_ITEM.match(lines[index])
            and not _MARKDOWN_SUBHEADING.fullmatch(lines[index])
            and not _FENCED_CODE_OPEN.fullmatch(lines[index])
        ):
            paragraph.append(lines[index])
            index += 1
        units.append("\n".join(paragraph))

    return units


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


def _semantic_blocks(
    text: str,
    *,
    key_prefix: str,
    ancestor_headings: tuple[str, ...],
) -> list[SemanticBlock]:
    blocks: list[SemanticBlock] = []
    active_nested_headings: list[tuple[int, str]] = []
    for index, unit in enumerate(_natural_units(text), start=1):
        heading_match = _MARKDOWN_SUBHEADING.fullmatch(unit)
        if heading_match:
            level = len(heading_match.group(1))
            active_nested_headings = [
                heading
                for heading in active_nested_headings
                if heading[0] < level
            ]
            active_nested_headings.append((level, unit))
            continue
        blocks.append(SemanticBlock(
            SemanticUnit(f"{key_prefix}:{index}", unit),
            ancestor_headings + tuple(
                heading for _, heading in active_nested_headings
            ),
        ))
    return blocks


def _render_blocks(identity_text: str, blocks: list[SemanticBlock]) -> str:
    parts = [identity_text]
    index = 0
    while index < len(blocks):
        ancestor_headings = blocks[index].ancestor_headings
        parts.extend(ancestor_headings)
        texts: list[str] = []
        while (
            index < len(blocks)
            and blocks[index].ancestor_headings == ancestor_headings
        ):
            texts.append(blocks[index].unit.text)
            index += 1
        parts.append(_join_natural_units(texts))
    return _join_context(*parts)


def _split_section(
    parent: KnowledgeDocument,
    *,
    base_chunk_id: str,
    section: str,
    identity: SemanticUnit,
    blocks: list[SemanticBlock],
    full_text: str,
    max_characters: int,
    include_identity_unit: bool = False,
) -> tuple[list[SemanticUnit], list[ChunkCandidate]]:
    if not blocks:
        raise _error(parent, f"section {section} has no factual content")

    groups: list[list[SemanticBlock]] = []
    current: list[SemanticBlock] = []
    for block in blocks:
        proposed = current + [block]
        proposed_text = _render_blocks(identity.text, proposed)
        if current and len(proposed_text) > max_characters:
            groups.append(current)
            current = [block]
        else:
            current = proposed
    if current:
        groups.append(current)

    multiple = len(groups) > 1
    candidates: list[ChunkCandidate] = []
    for index, group in enumerate(groups, start=1):
        chunk_id = f"{base_chunk_id}:{index}" if multiple else base_chunk_id
        if multiple:
            text = _render_blocks(identity.text, group)
        else:
            text = full_text.strip("\n")
        covered_keys = [identity.key]
        covered_keys.extend(block.unit.key for block in group)
        candidates.append(ChunkCandidate(
            _chunk(
                parent,
                chunk_id=chunk_id,
                section=section,
                text=text,
            ),
            tuple(covered_keys),
        ))
    units = [block.unit for block in blocks]
    if include_identity_unit:
        units.insert(0, identity)
    return units, candidates


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
    overview_blocks: list[SemanticBlock] = []
    if parsed.preamble:
        overview_blocks.extend(
            _semantic_blocks(
                parsed.preamble,
                key_prefix="overview:preamble",
                ancestor_headings=(),
            )
        )
    overview_heading = f"## {introduction.heading}"
    overview_blocks.extend(_semantic_blocks(
        introduction.body,
        key_prefix="overview:content",
        ancestor_headings=(overview_heading,),
    ))
    units, candidates = _split_section(
        parent,
        base_chunk_id=f"{parent.document_id}:overview",
        section=introduction.heading,
        identity=identity,
        blocks=overview_blocks,
        full_text=_join_context(parsed.title_line, parsed.preamble, introduction.raw),
        max_characters=max_characters,
        include_identity_unit=True,
    )
    feature_heading = f"## {features.heading}"
    feature_units, feature_candidates = _split_section(
        parent,
        base_chunk_id=f"{parent.document_id}:features",
        section=features.heading,
        identity=identity,
        blocks=_semantic_blocks(
            features.body,
            key_prefix="features:content",
            ancestor_headings=(feature_heading,),
        ),
        full_text=_join_context(parsed.title_line, features.raw),
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
        group_heading = f"### {group.heading}"
        group_units, group_candidates = _split_section(
            parent,
            base_chunk_id=base_chunk_id,
            section=group.heading,
            identity=identity,
            blocks=_semantic_blocks(
                group.body,
                key_prefix=f"spec:{index}",
                ancestor_headings=(specifications_prefix, group_heading),
            ),
            full_text=_join_context(parsed.title_line, specifications_prefix, group.raw),
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
        optional_heading = f"## {section.heading}"
        section_units, section_candidates = _split_section(
            parent,
            base_chunk_id=base_chunk_id,
            section=section.heading,
            identity=identity,
            blocks=_semantic_blocks(
                section.body,
                key_prefix=f"optional:{index}",
                ancestor_headings=(optional_heading,),
            ),
            full_text=_join_context(parsed.title_line, section.raw),
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
        blocks: list[SemanticBlock] = []
        if index == 0 and parsed.preamble:
            blocks.extend(_semantic_blocks(
                parsed.preamble,
                key_prefix="summary:preamble",
                ancestor_headings=(),
            ))
        blocks.extend(_semantic_blocks(
            section.body,
            key_prefix=f"fixed:{index}",
            ancestor_headings=(heading,),
        ))
        section_units, section_candidates = _split_section(
            parent,
            base_chunk_id=f"{parent.document_id}:{slug}",
            section=section.heading,
            identity=identity,
            blocks=blocks,
            full_text=_join_context(parsed.title_line, parsed.preamble if index == 0 else "", section.raw),
            max_characters=max_characters,
            include_identity_unit=index == 0,
        )
        units.extend(section_units)
        candidates.extend(section_candidates)

    details = parsed.h2_sections[details_index]
    if details.body.strip():
        details_heading = f"## {details.heading}"
        details_units, details_candidates = _split_section(
            parent,
            base_chunk_id=f"{parent.document_id}:details",
            section=details.heading,
            identity=identity,
            blocks=_semantic_blocks(
                details.body,
                key_prefix="details",
                ancestor_headings=(details_heading,),
            ),
            full_text=_join_context(parsed.title_line, details.raw),
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
        body_heading = f"## {section.heading}"
        section_units, section_candidates = _split_section(
            parent,
            base_chunk_id=chunk_id,
            section=section.heading,
            identity=identity,
            blocks=_semantic_blocks(
                section.body,
                key_prefix=f"body:{index}",
                ancestor_headings=(body_heading,),
            ),
            full_text=_join_context(parsed.title_line, section.raw),
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
    identity = SemanticUnit("identity:title", parsed.title_line)
    summary_heading = f"## {summary.heading}"
    content_heading = f"## {content.heading}"
    blocks = _semantic_blocks(
        parsed.preamble,
        key_prefix="support:preamble",
        ancestor_headings=(),
    )
    blocks.extend(_semantic_blocks(
        summary.body,
        key_prefix="support:summary",
        ancestor_headings=(summary_heading,),
    ))
    blocks.extend(_semantic_blocks(
        content.body,
        key_prefix="support:content",
        ancestor_headings=(content_heading,),
    ))
    return _split_section(
        parent,
        base_chunk_id=f"{parent.document_id}:content",
        section=parent.title,
        identity=identity,
        blocks=blocks,
        full_text=parent.text,
        max_characters=max_characters,
        include_identity_unit=True,
    )


def _company_candidates(
    parent: KnowledgeDocument,
    max_characters: int,
) -> tuple[list[SemanticUnit], list[ChunkCandidate]]:
    parsed = _parse_markdown(parent)
    if not parsed.h2_sections:
        identity = SemanticUnit("identity:title", parsed.title_line)
        return _split_section(
            parent,
            base_chunk_id=f"{parent.document_id}:overview",
            section=parent.title,
            identity=identity,
            blocks=_semantic_blocks(
                parsed.preamble,
                key_prefix="company:overview",
                ancestor_headings=(),
            ),
            full_text=parent.text,
            max_characters=max_characters,
            include_identity_unit=True,
        )

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
        blocks: list[SemanticBlock] = []
        if index == 0:
            if parsed.preamble:
                blocks.extend(_semantic_blocks(
                    parsed.preamble,
                    key_prefix="company:preamble",
                    ancestor_headings=(),
                ))
        blocks.extend(_semantic_blocks(
            section.body,
            key_prefix=f"company:section:{index}",
            ancestor_headings=(heading,),
        ))
        section_units, section_candidates = _split_section(
            parent,
            base_chunk_id=chunk_id,
            section=section.heading,
            identity=identity,
            blocks=blocks,
            full_text=_join_context(
                parsed.title_line,
                parsed.preamble if index == 0 else "",
                section.raw,
            ),
            max_characters=max_characters,
            include_identity_unit=index == 0,
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


def validate_chunks(
    chunks: list[KnowledgeChunk],
    documents: list[KnowledgeDocument],
) -> None:
    parent_by_id = {document.document_id: document for document in documents}
    chunk_ids: set[str] = set()
    chunks_by_parent: Counter[str] = Counter()
    errors: list[str] = []

    for chunk in chunks:
        if chunk.chunk_id in chunk_ids:
            errors.append(f"[ERROR] duplicate chunk_id: {chunk.chunk_id}")
        chunk_ids.add(chunk.chunk_id)

        expected_content_hash = compute_chunk_content_hash(
            chunk.type,
            chunk.section,
            chunk.text,
            chunk.language,
            chunk.metadata,
        )
        if chunk.content_hash != expected_content_hash:
            errors.append(
                f"[ERROR] {chunk.chunk_id}\nfield: content_hash\n"
                "reason: does not match canonical semantic content"
            )

        parent = parent_by_id.get(chunk.parent_document_id)
        if parent is None:
            errors.append(
                f"[ERROR] {chunk.chunk_id}\nfield: parent_document_id\n"
                f"reason: unknown parent_document_id {chunk.parent_document_id}"
            )
            continue

        chunks_by_parent[parent.document_id] += 1
        for field in (
            "parent_document_hash",
            "type",
            "language",
            "source_url",
            "source_files",
            "metadata",
        ):
            expected = (
                parent.content_hash
                if field == "parent_document_hash"
                else getattr(parent, field)
            )
            if getattr(chunk, field) != expected:
                errors.append(
                    f"[ERROR] {chunk.chunk_id}\nfield: {field}\n"
                    f"reason: does not match parent document {parent.document_id}"
                )

    for document in documents:
        if chunks_by_parent[document.document_id] == 0:
            errors.append(
                f"[ERROR] {document.document_id}\nreason: parent document has no chunks"
            )

    if errors:
        raise BuildError(errors)

    expected_chunks = build_chunks(documents)
    actual_payloads = [chunk.model_dump(mode="json") for chunk in chunks]
    expected_payloads = [
        chunk.model_dump(mode="json") for chunk in expected_chunks
    ]
    if actual_payloads != expected_payloads:
        raise BuildError(
            "[ERROR] chunk order or content does not match fresh deterministic build"
        )


def serialize_chunks(chunks: list[KnowledgeChunk]) -> bytes:
    lines = [
        json.dumps(
            chunk.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for chunk in chunks
    ]
    serialized = "\n".join(lines)
    if lines:
        serialized += "\n"
    return serialized.encode("utf-8")


@dataclass(frozen=True)
class ChunkStatistics:
    total_documents: int
    total_chunks: int
    average_characters: float
    median_characters: float
    minimum_characters: int
    maximum_characters: int
    chunks_per_document: dict[str, int]
    chunks_per_type: dict[str, int]


def calculate_chunk_statistics(
    documents: list[KnowledgeDocument],
    chunks: list[KnowledgeChunk],
) -> ChunkStatistics:
    validate_chunks(chunks, documents)
    lengths = [len(chunk.text) for chunk in chunks]
    chunks_per_document = Counter(
        chunk.parent_document_id for chunk in chunks
    )
    chunks_per_type = Counter(chunk.type for chunk in chunks)
    return ChunkStatistics(
        total_documents=len(documents),
        total_chunks=len(chunks),
        average_characters=fmean(lengths) if lengths else 0.0,
        median_characters=float(median(lengths)) if lengths else 0.0,
        minimum_characters=min(lengths) if lengths else 0,
        maximum_characters=max(lengths) if lengths else 0,
        chunks_per_document={
            document.document_id: chunks_per_document[document.document_id]
            for document in documents
        },
        chunks_per_type={
            type_: chunks_per_type[type_]
            for type_ in sorted(chunks_per_type)
        },
    )


def write_chunks(
    output_path: Path,
    chunks: list[KnowledgeChunk],
    documents: list[KnowledgeDocument],
) -> None:
    validate_chunks(chunks, documents)
    payload = serialize_chunks(chunks)
    temp_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        temporary_lines = _split_jsonl_records(
            temp_path.read_text(encoding="utf-8")
        )
        parsed = [
            KnowledgeChunk.model_validate(json.loads(line))
            for line in temporary_lines
        ]
        validate_chunks(parsed, documents)
        if serialize_chunks(parsed) != payload:
            raise BuildError(
                "temporary JSONL verification was not byte deterministic"
            )
        os.replace(temp_path, output_path)
        temp_path = None
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as error:
        raise BuildError(f"failed to write {output_path}: {error}") from error
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def build_and_write_chunks(
    input_path: Path,
    output_path: Path,
) -> "ChunkStatistics":
    documents = load_documents(input_path)
    chunks = build_chunks(documents)
    validate_chunks(chunks, documents)
    write_chunks(output_path, chunks, documents)
    return calculate_chunk_statistics(documents, chunks)
