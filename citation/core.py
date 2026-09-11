from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from pydantic import ValidationError

from knowledge_pipeline.models import SourceRef


_REFERENCE_HEADING = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?"
    r"(?:参考资料|引用|来源|references?|citations?|sources?)"
    r"[ \t]*[:：]?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_MARKDOWN_LINK = re.compile(
    r"\[([^\]\r\n]+)\]\(\s*(?:https?://|www\.)[^)\r\n]+\s*\)",
    re.IGNORECASE,
)
_ANGLE_URL = re.compile(
    r"<(?:https?://|www\.)[^<>\s]+>",
    re.IGNORECASE,
)
_PLAIN_URL = re.compile(
    r"(?:https?://|www\.)"
    r"[^\s<>\[\]()\"'，。！？；：]+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CitationCollection:
    sources: tuple[SourceRef, ...]
    input_count: int
    invalid_source_count: int
    deduplicated_count: int


@dataclass(frozen=True)
class CitationRenderResult:
    text: str
    sources: tuple[SourceRef, ...]
    citation_heading: str | None
    invalid_source_count: int
    deduplicated_count: int
    answer_sanitized: bool
    used_fallback: bool


def _normalized_url_identity(url: str) -> str:
    parsed = urlsplit(url.strip())
    return urlunsplit((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path,
        parsed.query,
        parsed.fragment,
    ))


def collect_sources(sources: Iterable[object]) -> CitationCollection:
    collected: list[SourceRef] = []
    seen: set[str] = set()
    input_count = 0
    invalid_count = 0
    duplicate_count = 0

    for candidate in sources:
        input_count += 1
        try:
            source = SourceRef.model_validate(candidate)
        except (TypeError, ValidationError, ValueError):
            invalid_count += 1
            continue

        identity = _normalized_url_identity(source.url)
        if identity in seen:
            duplicate_count += 1
            continue
        seen.add(identity)
        collected.append(source)

    return CitationCollection(
        sources=tuple(collected),
        input_count=input_count,
        invalid_source_count=invalid_count,
        deduplicated_count=duplicate_count,
    )


def sanitize_generated_answer(answer: str) -> str:
    heading = _REFERENCE_HEADING.search(answer)
    without_references = answer if heading is None else answer[:heading.start()]
    cleaned = _MARKDOWN_LINK.sub(r"\1", without_references)
    cleaned = _ANGLE_URL.sub("", cleaned)
    cleaned = _PLAIN_URL.sub("", cleaned)
    return cleaned.strip()


def _answer_language(text: str) -> str | None:
    cjk_count = sum("\u4e00" <= character <= "\u9fff" for character in text)
    english_words = re.findall(
        r"(?<![A-Za-z0-9])[A-Za-z]{2,}(?![A-Za-z0-9])",
        text,
    )
    english_count = sum(len(word) for word in english_words)
    if cjk_count and not english_count:
        return "zh"
    if english_count and not cjk_count:
        return "en"
    if cjk_count and english_count:
        return "zh" if cjk_count * 2 >= english_count else "en"
    return None


def citation_heading(
    *,
    language_text: str,
    language_hint: str = "",
) -> str:
    language = _answer_language(language_text) or _answer_language(language_hint)
    if language == "zh":
        return "参考资料："
    return "References:"


def render_answer(
    answer: str,
    sources: Iterable[object],
    *,
    fallback: str,
    language_hint: str,
) -> CitationRenderResult:
    stripped = answer.strip()
    cleaned = sanitize_generated_answer(answer)
    sanitized = cleaned != stripped
    if not cleaned:
        return CitationRenderResult(
            text=fallback,
            sources=(),
            citation_heading=None,
            invalid_source_count=0,
            deduplicated_count=0,
            answer_sanitized=sanitized,
            used_fallback=True,
        )

    collection = collect_sources(sources)
    heading = None
    if collection.sources:
        heading = citation_heading(
            language_text=cleaned,
            language_hint=language_hint,
        )
    return CitationRenderResult(
        text=cleaned,
        sources=collection.sources,
        citation_heading=heading,
        invalid_source_count=collection.invalid_source_count,
        deduplicated_count=collection.deduplicated_count,
        answer_sanitized=sanitized,
        used_fallback=False,
    )
