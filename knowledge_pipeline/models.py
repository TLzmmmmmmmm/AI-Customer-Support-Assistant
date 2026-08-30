from __future__ import annotations

import re
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


KEBAB_CASE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE = re.compile(r"^[0-9]+$")
NonEmptyStr = Annotated[str, Field(min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class WebsiteFileProvenance(StrictModel):
    kind: Literal["website_file"]
    reference: NonEmptyStr

    @field_validator("reference")
    @classmethod
    def validate_reference(cls, value: str) -> str:
        parts = value.split("/")
        if (
            value.startswith(("/", "\\"))
            or "\\" in value
            or ":" in value
            or ".." in parts
            or not value.startswith("src/")
        ):
            raise ValueError("must be a Shengborun repository-relative src/ path")
        return value


class TechnicalParameterItem(StrictModel):
    name: NonEmptyStr
    value: NonEmptyStr


class TechnicalParameterGroup(StrictModel):
    group: NonEmptyStr
    items: list[TechnicalParameterItem] = Field(min_length=1)


class SourceBase(StrictModel):
    id: NonEmptyStr
    source_path: NonEmptyStr
    published: bool
    provenance: list[WebsiteFileProvenance] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not KEBAB_CASE.fullmatch(value):
            raise ValueError("must be lowercase kebab-case")
        return value

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            not value.startswith("/")
            or parsed.scheme
            or parsed.netloc
            or parsed.query
            or "\\" in value
            or ".." in parsed.path.split("/")
            or ("#" in value and not parsed.fragment)
        ):
            raise ValueError("must be a safe website-relative path without query")
        return value


class SluggedSource(SourceBase):
    slug: NonEmptyStr

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, value: str) -> str:
        if not KEBAB_CASE.fullmatch(value):
            raise ValueError("must be lowercase kebab-case")
        return value


class ProductSource(SluggedSource):
    name: NonEmptyStr
    category_id: NonEmptyStr
    key_features: list[NonEmptyStr] = Field(min_length=1)
    product_features: NonEmptyStr
    technical_parameters: list[TechnicalParameterGroup] = Field(min_length=1)

    @field_validator("category_id")
    @classmethod
    def validate_category_id(cls, value: str) -> str:
        if not KEBAB_CASE.fullmatch(value):
            raise ValueError("must be lowercase kebab-case")
        return value


class ProductCategorySource(SluggedSource):
    name: NonEmptyStr
    short_description: NonEmptyStr


class SolutionSource(SluggedSource):
    name: NonEmptyStr
    summary: NonEmptyStr
    core_needs: list[NonEmptyStr] = Field(min_length=1)
    solution_design: NonEmptyStr
    features: list[NonEmptyStr] = Field(min_length=1)
    body_markdown: NonEmptyStr


class SupportSource(SourceBase):
    name: NonEmptyStr
    summary: NonEmptyStr
    body: NonEmptyStr


class CompanySource(SourceBase):
    name: NonEmptyStr
    introduction: list[NonEmptyStr] = Field(min_length=1)


class ContactSource(SourceBase):
    company_name: NonEmptyStr
    duty_phone: NonEmptyStr
    email: NonEmptyStr

    @field_validator("duty_phone")
    @classmethod
    def validate_phone(cls, value: str) -> str:
        if not PHONE.fullmatch(value):
            raise ValueError("must contain digits only")
        return value

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        if not EMAIL.fullmatch(value):
            raise ValueError("must be a valid email address")
        return value


class ProductMetadata(StrictModel):
    product_id: NonEmptyStr
    slug: NonEmptyStr
    category_id: NonEmptyStr
    category_name: NonEmptyStr


class SolutionMetadata(StrictModel):
    solution_id: NonEmptyStr
    slug: NonEmptyStr


class SupportMetadata(StrictModel):
    service_id: NonEmptyStr


class CompanyMetadata(StrictModel):
    company_id: NonEmptyStr


class ContactMetadata(StrictModel):
    contact_id: NonEmptyStr


Metadata = (
    ProductMetadata
    | SolutionMetadata
    | SupportMetadata
    | CompanyMetadata
    | ContactMetadata
)


class KnowledgeDocument(StrictModel):
    schema_version: Literal["1.0"]
    document_id: NonEmptyStr
    type: Literal["product", "solution", "support", "company", "contact"]
    title: NonEmptyStr
    text: NonEmptyStr
    language: Literal["zh-CN"]
    source_path: NonEmptyStr
    source_url: NonEmptyStr
    source_files: list[NonEmptyStr] = Field(min_length=1)
    content_hash: NonEmptyStr
    metadata: Metadata

    @field_validator("content_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_HEX.fullmatch(value):
            raise ValueError("must be a lowercase SHA-256 hex digest")
        return value

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("must be an absolute HTTP(S) URL")
        return value

    @field_validator("source_files")
    @classmethod
    def validate_source_files(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("must be sorted and unique")
        return value

    @field_validator("text")
    @classmethod
    def validate_lf_text(cls, value: str) -> str:
        if "\r" in value:
            raise ValueError("must use LF line endings")
        return value

    @model_validator(mode="after")
    def validate_identity_and_metadata(self) -> "KnowledgeDocument":
        if not self.document_id.startswith(f"{self.type}:"):
            raise ValueError("document_id must start with '<type>:'")
        expected = {
            "product": ProductMetadata,
            "solution": SolutionMetadata,
            "support": SupportMetadata,
            "company": CompanyMetadata,
            "contact": ContactMetadata,
        }[self.type]
        if not isinstance(self.metadata, expected):
            raise ValueError(f"metadata does not match document type {self.type}")
        return self

