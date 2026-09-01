from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from knowledge_pipeline.models import ProductMetadata

from .models import EntityCatalogError, EntityMatch, VectorRecord


_SEPARATORS = re.compile(r"[\s\-‐‑‒–—―]+")
_FIRST_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return _SEPARATORS.sub("-", normalized)


def _has_alphanumeric_boundaries(text: str, start: int, length: int) -> bool:
    before = text[start - 1] if start > 0 else None
    end = start + length
    after = text[end] if end < len(text) else None
    return not (
        (before is not None and before.isalnum())
        or (after is not None and after.isalnum())
    )


class ExactEntityResolver:
    """Resolve explicit product/model identifiers without fuzzy matching."""

    def __init__(self, aliases: dict[str, str]) -> None:
        self._aliases = dict(aliases)

    @classmethod
    def from_records(
        cls,
        records: Sequence[VectorRecord],
    ) -> "ExactEntityResolver":
        aliases: dict[str, str] = {}
        for record in records:
            if record.type != "product":
                continue
            if not isinstance(record.metadata, ProductMetadata):
                raise EntityCatalogError(
                    f"product record {record.chunk_id} has invalid metadata"
                )

            heading_match = _FIRST_H1.search(record.text)
            candidates = (
                record.parent_document_id.removeprefix("product:"),
                record.metadata.product_id,
                record.metadata.slug,
                heading_match.group(1) if heading_match else "",
            )
            for candidate in candidates:
                alias = _normalize(candidate)
                if not alias:
                    continue
                existing_parent = aliases.get(alias)
                if (
                    existing_parent is not None
                    and existing_parent != record.parent_document_id
                ):
                    raise EntityCatalogError(
                        f"entity alias {alias!r} maps to both "
                        f"{existing_parent!r} and {record.parent_document_id!r}"
                    )
                aliases[alias] = record.parent_document_id
        return cls(aliases)

    def resolve(self, query: str) -> list[EntityMatch]:
        normalized_query = _normalize(query)
        if not normalized_query:
            return []

        best_by_parent: dict[str, tuple[int, str]] = {}
        for alias, parent_document_id in self._aliases.items():
            start = normalized_query.find(alias)
            while start >= 0:
                if _has_alphanumeric_boundaries(
                    normalized_query,
                    start,
                    len(alias),
                ):
                    candidate = (start, alias)
                    existing = best_by_parent.get(parent_document_id)
                    if existing is None or candidate < existing:
                        best_by_parent[parent_document_id] = candidate
                    break
                start = normalized_query.find(alias, start + 1)

        matches = [
            EntityMatch(
                parent_document_id=parent_document_id,
                alias=alias,
                start=start,
            )
            for parent_document_id, (start, alias) in best_by_parent.items()
        ]
        return sorted(
            matches,
            key=lambda match: (match.start, match.parent_document_id),
        )


__all__ = ["ExactEntityResolver"]
