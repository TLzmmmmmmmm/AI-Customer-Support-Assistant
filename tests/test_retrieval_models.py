import math
import unittest

from pydantic import ValidationError

from knowledge_pipeline.retrieval.config import (
    DashScopeCredentials,
    EmbeddingConfig,
    EmbeddingConfigurationError,
    RetrievalConfig,
)
from knowledge_pipeline.retrieval.models import VectorRecord


def record_payload() -> dict:
    return {
        "schema_version": "1.0",
        "chunk_id": "product:hp780:content",
        "parent_document_id": "product:hp780",
        "type": "product",
        "section": "海能达 HP780",
        "text": "# 海能达 HP780",
        "language": "zh-CN",
        "content_hash": "a" * 64,
        "source_url": "https://www.shengborun.com/two-way-radio/hp780/",
        "source_files": [
            "src/content/products/two-way-radio/hp780.json",
        ],
        "metadata": {
            "product_id": "hp780",
            "slug": "hp780",
            "category_id": "two-way-radio",
            "category_name": "对讲机通信",
        },
        "embedding_provider": "dashscope",
        "embedding_model": "qwen3.7-text-embedding",
        "embedding_dimensions": 3,
        "embedding_text_type": "document",
        "embedding": [1.0, 0.0, 0.0],
    }


class RetrievalConfigurationTests(unittest.TestCase):
    def test_embedding_config_has_expected_defaults_without_secrets(self):
        config = EmbeddingConfig.from_mapping({})

        self.assertEqual(config.provider, "dashscope")
        self.assertEqual(config.model, "qwen3.7-text-embedding")
        self.assertEqual(config.dimensions, 1024)
        self.assertEqual(config.timeout_seconds, 30.0)
        self.assertEqual(config.max_retries, 2)
        self.assertFalse(hasattr(config, "api_key"))

    def test_credentials_build_confirmed_beijing_url(self):
        credentials = DashScopeCredentials.from_mapping({
            "DASHSCOPE_API_KEY": "secret",
            "DASHSCOPE_WORKSPACE_ID": "workspace",
        })

        self.assertEqual(credentials.api_key, "secret")
        self.assertEqual(
            credentials.base_url,
            "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        )

    def test_plan_configuration_does_not_require_credentials(self):
        config = EmbeddingConfig.from_mapping({
            "EMBEDDING_DIMENSIONS": "768",
        })

        self.assertEqual(config.dimensions, 768)

    def test_credentials_require_key_and_workspace(self):
        for values in ({}, {"DASHSCOPE_API_KEY": "key"}):
            with self.subTest(values=values):
                with self.assertRaises(EmbeddingConfigurationError):
                    DashScopeCredentials.from_mapping(values)

    def test_embedding_config_rejects_invalid_values(self):
        invalid = (
            {"EMBEDDING_PROVIDER": "local"},
            {"EMBEDDING_MODEL": " "},
            {"EMBEDDING_DIMENSIONS": "0"},
            {"EMBEDDING_TIMEOUT_SECONDS": "0"},
            {"EMBEDDING_MAX_RETRIES": "-1"},
            {"EMBEDDING_PRICE_YUAN_PER_1K_TOKENS": "not-a-number"},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(EmbeddingConfigurationError):
                    EmbeddingConfig.from_mapping(values)

    def test_retrieval_config_defaults_to_top_five_and_rejects_zero(self):
        self.assertEqual(RetrievalConfig.from_mapping({}).top_k, 5)
        with self.assertRaises(EmbeddingConfigurationError):
            RetrievalConfig.from_mapping({"RETRIEVAL_TOP_K": "0"})


class VectorRecordTests(unittest.TestCase):
    def test_valid_vector_record_preserves_semantic_section(self):
        record = VectorRecord.model_validate(record_payload())

        self.assertEqual(record.section, "海能达 HP780")
        self.assertEqual(record.parent_document_id, "product:hp780")
        self.assertEqual(record.embedding, [1.0, 0.0, 0.0])

    def test_vector_record_rejects_wrong_dimension(self):
        payload = record_payload()
        payload["embedding_dimensions"] = 1024

        with self.assertRaises(ValidationError):
            VectorRecord.model_validate(payload)

    def test_vector_record_rejects_non_finite_and_zero_vectors(self):
        for vector in ([math.nan, 0.0, 0.0], [0.0, 0.0, 0.0]):
            payload = record_payload()
            payload["embedding"] = vector
            with self.subTest(vector=vector):
                with self.assertRaises(ValidationError):
                    VectorRecord.model_validate(payload)

    def test_vector_record_does_not_accept_parent_document_hash(self):
        payload = record_payload()
        payload["parent_document_hash"] = "b" * 64

        with self.assertRaises(ValidationError):
            VectorRecord.model_validate(payload)

    def test_vector_record_rejects_wrong_metadata_type(self):
        payload = record_payload()
        payload["metadata"] = {"solution_id": "hp780", "slug": "hp780"}

        with self.assertRaises(ValidationError):
            VectorRecord.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
