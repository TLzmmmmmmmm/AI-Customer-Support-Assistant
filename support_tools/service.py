from __future__ import annotations

from collections.abc import Collection
from typing import Protocol
from urllib.parse import urlsplit

from knowledge_pipeline.core import SourceInventory
from knowledge_pipeline.models import ProductMetadata, SourceBase, SourceRef
from knowledge_pipeline.retrieval.entities import ExactEntityResolver
from knowledge_pipeline.retrieval.models import RetrievalError, RetrievalResult

from .models import (
    ContactInfoResult,
    ProductDetailsResult,
    ProductSearchItem,
    ProductSearchResult,
    ToolError,
    ToolErrorCode,
)


PRODUCT_SEARCH_LIMIT = 5
MAX_PRODUCT_ID_CHARACTERS = 128
MAX_PRODUCT_QUERY_CHARACTERS = 4000


class ProductRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        allowed_types: Collection[str] | None = None,
        unique_parent_documents: bool = False,
    ) -> list[RetrievalResult]: ...


class DeterministicTools:
    """Validated deterministic capabilities for future orchestration."""

    def __init__(
        self,
        *,
        inventory: SourceInventory,
        site_base_url: str,
        retriever: ProductRetriever | None = None,
    ) -> None:
        parsed = urlsplit(site_base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "site_base_url must contain only HTTP(S) scheme and authority"
            )
        self._inventory = inventory
        self._site_base_url = site_base_url.rstrip("/")
        self._retriever = retriever
        self._product_resolver = ExactEntityResolver.from_product_sources(
            inventory.products.values()
        )

    def get_product_details(self, product_id: str) -> ProductDetailsResult:
        tool_name = "get_product_details"
        try:
            normalized_input = self._validate_string(
                product_id,
                maximum=MAX_PRODUCT_ID_CHARACTERS,
                tool_name=tool_name,
            )
            matches = self._product_resolver.resolve_canonical_identifier(
                normalized_input
            )
            if not matches:
                raise ToolError(
                    code=ToolErrorCode.PRODUCT_NOT_FOUND,
                    message="No product matches the supplied canonical identifier.",
                    tool_name=tool_name,
                )
            if len(matches) > 1:
                raise ToolError(
                    code=ToolErrorCode.AMBIGUOUS_PRODUCT,
                    message="The supplied canonical identifier is ambiguous.",
                    tool_name=tool_name,
                )

            product_id = matches[0].removeprefix("product:")
            product = self._inventory.products.get(product_id)
            if product is None or not product.published:
                raise self._unavailable(tool_name)
            category = self._inventory.categories.get(product.category_id)
            if category is None or not category.published:
                raise self._unavailable(tool_name)

            return ProductDetailsResult(
                product_id=product.id,
                name=product.name,
                category_id=category.id,
                category_name=category.name,
                key_features=list(product.key_features),
                product_features=product.product_features,
                technical_parameters=list(product.technical_parameters),
                sources=[self._source_ref(product.name, product)],
            )
        except ToolError:
            raise
        except Exception as error:
            raise self._execution_error(tool_name) from error

    def get_contact_info(self) -> ContactInfoResult:
        tool_name = "get_contact_info"
        try:
            contacts = [
                contact
                for contact in self._inventory.contacts.values()
                if contact.published
            ]
            if len(contacts) != 1:
                raise self._unavailable(tool_name)
            contact = contacts[0]
            return ContactInfoResult(
                company_name=contact.company_name,
                duty_phone=contact.duty_phone,
                email=contact.email,
                sources=[self._source_ref("联系我们", contact)],
            )
        except ToolError:
            raise
        except Exception as error:
            raise self._execution_error(tool_name) from error

    def search_products(self, query: str) -> ProductSearchResult:
        tool_name = "search_products"
        normalized_query = self._validate_string(
            query,
            maximum=MAX_PRODUCT_QUERY_CHARACTERS,
            tool_name=tool_name,
        )
        if self._retriever is None:
            raise self._unavailable(tool_name)

        try:
            results = self._retriever.retrieve(
                normalized_query,
                top_k=PRODUCT_SEARCH_LIMIT,
                allowed_types={"product"},
                unique_parent_documents=True,
            )
            products: list[ProductSearchItem] = []
            seen_product_ids: set[str] = set()
            for result in results:
                if result.type != "product" or not isinstance(
                    result.metadata,
                    ProductMetadata,
                ):
                    raise RuntimeError("product search returned a non-product result")
                product_id = result.metadata.product_id
                if product_id in seen_product_ids:
                    continue
                product = self._inventory.products.get(product_id)
                category = self._inventory.categories.get(
                    result.metadata.category_id
                )
                if (
                    product is None
                    or not product.published
                    or category is None
                    or not category.published
                ):
                    raise RetrievalError("product search result is stale")
                seen_product_ids.add(product_id)
                products.append(ProductSearchItem(
                    product_id=product.id,
                    name=product.name,
                    category_id=category.id,
                    category_name=category.name,
                    relevant_content=result.text,
                    sources=list(result.sources),
                ))
                if len(products) >= PRODUCT_SEARCH_LIMIT:
                    break
            return ProductSearchResult(products=products)
        except ToolError:
            raise
        except RetrievalError as error:
            raise self._unavailable(tool_name) from error
        except Exception as error:
            raise ToolError(
                code=ToolErrorCode.TOOL_EXECUTION_ERROR,
                message="The tool could not complete safely.",
                tool_name=tool_name,
            ) from error

    def _source_ref(self, title: str, source: SourceBase) -> SourceRef:
        return SourceRef(
            title=title,
            url=self._site_base_url + source.source_path,
        )

    @staticmethod
    def _validate_string(
        value: object,
        *,
        maximum: int,
        tool_name: str,
    ) -> str:
        if not isinstance(value, str):
            raise ToolError(
                code=ToolErrorCode.INVALID_ARGUMENT,
                message="The tool argument must be a string.",
                tool_name=tool_name,
            )
        normalized = value.strip()
        if not normalized or len(normalized) > maximum:
            raise ToolError(
                code=ToolErrorCode.INVALID_ARGUMENT,
                message="The tool argument is blank or exceeds its size limit.",
                tool_name=tool_name,
            )
        return normalized

    @staticmethod
    def _unavailable(tool_name: str) -> ToolError:
        return ToolError(
            code=ToolErrorCode.TOOL_UNAVAILABLE,
            message="The tool dependency is unavailable.",
            tool_name=tool_name,
        )

    @staticmethod
    def _execution_error(tool_name: str) -> ToolError:
        return ToolError(
            code=ToolErrorCode.TOOL_EXECUTION_ERROR,
            message="The tool could not complete safely.",
            tool_name=tool_name,
        )


__all__ = [
    "DeterministicTools",
    "MAX_PRODUCT_ID_CHARACTERS",
    "MAX_PRODUCT_QUERY_CHARACTERS",
    "PRODUCT_SEARCH_LIMIT",
    "ProductRetriever",
]
