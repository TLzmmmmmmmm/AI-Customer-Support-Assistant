from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from support_tools import (
    ContactInfoResult,
    ProductDetailsResult,
    ProductSearchResult,
)
from support_tools.service import (
    MAX_PRODUCT_ID_CHARACTERS,
    MAX_PRODUCT_QUERY_CHARACTERS,
)


class ToolArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class SearchProductsArguments(ToolArguments):
    query: str = Field(
        min_length=1,
        max_length=MAX_PRODUCT_QUERY_CHARACTERS,
    )


class ProductDetailsArguments(ToolArguments):
    product_id: str = Field(
        min_length=1,
        max_length=MAX_PRODUCT_ID_CHARACTERS,
    )


class ContactInfoArguments(ToolArguments):
    pass


@dataclass(frozen=True)
class ToolSpec:
    description: str
    arguments_model: type[ToolArguments]
    result_model: type[BaseModel]


TOOL_SPECS: Mapping[str, ToolSpec] = MappingProxyType({
    "search_products": ToolSpec(
        description=(
            "Find product candidates for an explicit discovery, selection, "
            "or recommendation request using the user's unmodified query."
        ),
        arguments_model=SearchProductsArguments,
        result_model=ProductSearchResult,
    ),
    "get_product_details": ToolSpec(
        description=(
            "Retrieve authoritative details for one exact canonical product "
            "ID or slug; do not pass a display name or fuzzy description."
        ),
        arguments_model=ProductDetailsArguments,
        result_model=ProductDetailsResult,
    ),
    "get_contact_info": ToolSpec(
        description=(
            "Retrieve official phone and email contact channels when the user "
            "explicitly asks how to contact the company; no address is returned."
        ),
        arguments_model=ContactInfoArguments,
        result_model=ContactInfoResult,
    ),
})


@dataclass(frozen=True)
class AgentToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class AgentTurn:
    content: str | None
    tool_calls: tuple[AgentToolCall, ...]
    finish_reason: str | None


@dataclass(frozen=True)
class AgentResult:
    answer: str


class AgentDeadlineExceeded(Exception):
    pass


@dataclass(frozen=True)
class AgentDeadline:
    expires_at: float
    clock: Callable[[], float] = field(repr=False, compare=False)

    @classmethod
    def start(
        cls,
        timeout_seconds: float,
        *,
        clock: Callable[[], float],
    ) -> AgentDeadline:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        return cls(
            expires_at=clock() + timeout_seconds,
            clock=clock,
        )

    def ensure_active(self) -> None:
        if self.clock() >= self.expires_at:
            raise AgentDeadlineExceeded("agent deadline exceeded")


def llm_tool_schemas() -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    for name, spec in TOOL_SPECS.items():
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": spec.description,
                "parameters": spec.arguments_model.model_json_schema(),
            },
        })
    return schemas


__all__ = [
    "AgentDeadline",
    "AgentDeadlineExceeded",
    "AgentResult",
    "AgentToolCall",
    "AgentTurn",
    "ContactInfoArguments",
    "ProductDetailsArguments",
    "SearchProductsArguments",
    "TOOL_SPECS",
    "ToolArguments",
    "ToolSpec",
    "llm_tool_schemas",
]
