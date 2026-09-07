from __future__ import annotations

from enum import Enum

from pydantic import Field, field_validator

from knowledge_pipeline.models import (
    SourceRef,
    StrictModel,
    TechnicalParameterGroup,
)


class ToolErrorCode(str, Enum):
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    PRODUCT_NOT_FOUND = "PRODUCT_NOT_FOUND"
    AMBIGUOUS_PRODUCT = "AMBIGUOUS_PRODUCT"
    TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"
    TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"


class ToolError(Exception):
    """Safe structured failure for future tool orchestration."""

    def __init__(
        self,
        *,
        code: ToolErrorCode,
        message: str,
        tool_name: str,
    ) -> None:
        self.code = code
        self.message = message
        self.tool_name = tool_name
        super().__init__(message)


class SourcedResult(StrictModel):
    sources: list[SourceRef] = Field(min_length=1)

    @field_validator("sources")
    @classmethod
    def validate_unique_source_urls(
        cls,
        value: list[SourceRef],
    ) -> list[SourceRef]:
        urls = [source.url for source in value]
        if len(urls) != len(set(urls)):
            raise ValueError("sources must have unique URLs")
        return value


class ProductDetailsResult(SourcedResult):
    product_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category_id: str = Field(min_length=1)
    category_name: str = Field(min_length=1)
    key_features: list[str] = Field(min_length=1)
    product_features: str = Field(min_length=1)
    technical_parameters: list[TechnicalParameterGroup] = Field(min_length=1)


class ContactInfoResult(SourcedResult):
    company_name: str = Field(min_length=1)
    duty_phone: str = Field(min_length=1)
    email: str = Field(min_length=1)


class ProductSearchItem(SourcedResult):
    product_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category_id: str = Field(min_length=1)
    category_name: str = Field(min_length=1)
    relevant_content: str = Field(min_length=1)


class ProductSearchResult(StrictModel):
    products: list[ProductSearchItem]


ToolResult = ProductDetailsResult | ContactInfoResult | ProductSearchResult


__all__ = [
    "ContactInfoResult",
    "ProductDetailsResult",
    "ProductSearchItem",
    "ProductSearchResult",
    "ToolError",
    "ToolErrorCode",
    "ToolResult",
]
