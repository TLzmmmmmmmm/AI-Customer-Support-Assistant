from __future__ import annotations

from pathlib import Path

from knowledge_pipeline import BuildError, load_source_inventory
from support_tools import DeterministicTools, ToolError, ToolErrorCode
from support_tools.service import ProductRetriever


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPOSITORY_ROOT / "knowledge" / "source"
DEFAULT_SITE_BASE_URL = "https://www.shengborun.com"


def build_deterministic_tools(
    *,
    retriever: ProductRetriever | None = None,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    site_base_url: str = DEFAULT_SITE_BASE_URL,
) -> DeterministicTools:
    try:
        inventory = load_source_inventory(source_root)
        return DeterministicTools(
            inventory=inventory,
            site_base_url=site_base_url,
            retriever=retriever,
        )
    except ToolError:
        raise
    except (BuildError, OSError, ValueError) as error:
        raise ToolError(
            code=ToolErrorCode.TOOL_UNAVAILABLE,
            message="The authoritative tool data is unavailable.",
            tool_name="tool_layer",
        ) from error


__all__ = [
    "DEFAULT_SITE_BASE_URL",
    "DEFAULT_SOURCE_ROOT",
    "build_deterministic_tools",
]
