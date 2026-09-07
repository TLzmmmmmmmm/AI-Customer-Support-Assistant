from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from .service import DeterministicTools


def build_tool_registry(
    tools: DeterministicTools,
) -> Mapping[str, Callable[..., object]]:
    """Return the complete immutable allowlist for deterministic tools."""

    return MappingProxyType({
        "search_products": tools.search_products,
        "get_product_details": tools.get_product_details,
        "get_contact_info": tools.get_contact_info,
    })


__all__ = ["build_tool_registry"]
