from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from .models import EmbeddingConfigurationError


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    dimensions: int
    timeout_seconds: float
    max_retries: int
    price_yuan_per_1k_tokens: Decimal

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> "EmbeddingConfig":
        provider = values.get("EMBEDDING_PROVIDER", "dashscope").strip()
        model = values.get(
            "EMBEDDING_MODEL",
            "qwen3.7-text-embedding",
        ).strip()
        if provider != "dashscope":
            raise EmbeddingConfigurationError(
                f"unsupported embedding provider: {provider}"
            )
        if not model:
            raise EmbeddingConfigurationError("embedding model is required")
        try:
            dimensions = int(values.get("EMBEDDING_DIMENSIONS", "1024"))
            timeout_seconds = float(
                values.get("EMBEDDING_TIMEOUT_SECONDS", "30")
            )
            max_retries = int(values.get("EMBEDDING_MAX_RETRIES", "2"))
            price = Decimal(
                values.get(
                    "EMBEDDING_PRICE_YUAN_PER_1K_TOKENS",
                    "0.0005",
                )
            )
        except (ValueError, InvalidOperation) as error:
            raise EmbeddingConfigurationError(
                "invalid numeric embedding configuration"
            ) from error
        if dimensions <= 0:
            raise EmbeddingConfigurationError(
                "EMBEDDING_DIMENSIONS must be positive"
            )
        if timeout_seconds <= 0:
            raise EmbeddingConfigurationError(
                "EMBEDDING_TIMEOUT_SECONDS must be positive"
            )
        if max_retries < 0:
            raise EmbeddingConfigurationError(
                "EMBEDDING_MAX_RETRIES must not be negative"
            )
        if price < 0:
            raise EmbeddingConfigurationError(
                "embedding price must not be negative"
            )
        return cls(
            provider=provider,
            model=model,
            dimensions=dimensions,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            price_yuan_per_1k_tokens=price,
        )


@dataclass(frozen=True)
class DashScopeCredentials:
    api_key: str
    workspace_id: str
    base_url: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> "DashScopeCredentials":
        api_key = values.get("DASHSCOPE_API_KEY", "").strip()
        workspace_id = values.get("DASHSCOPE_WORKSPACE_ID", "").strip()
        if not api_key or not workspace_id:
            raise EmbeddingConfigurationError(
                "DASHSCOPE_API_KEY and DASHSCOPE_WORKSPACE_ID are required"
            )
        return cls(
            api_key=api_key,
            workspace_id=workspace_id,
            base_url=(
                f"https://{workspace_id}."
                "cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
            ),
        )


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> "RetrievalConfig":
        try:
            top_k = int(values.get("RETRIEVAL_TOP_K", "5"))
        except ValueError as error:
            raise EmbeddingConfigurationError(
                "RETRIEVAL_TOP_K must be an integer"
            ) from error
        if top_k <= 0:
            raise EmbeddingConfigurationError(
                "RETRIEVAL_TOP_K must be positive"
            )
        return cls(top_k=top_k)
