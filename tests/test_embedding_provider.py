import unittest
from types import SimpleNamespace
from unittest.mock import patch

from knowledge_pipeline.retrieval.config import (
    DashScopeCredentials,
    EmbeddingConfig,
)
from knowledge_pipeline.retrieval.embedding import DashScopeEmbeddingProvider
from knowledge_pipeline.retrieval.models import EmbeddingAPIError


class FakeEmbeddingsEndpoint:
    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.mode == "api-error":
            raise RuntimeError("secret provider detail")

        inputs = kwargs["input"]
        count = 1 if isinstance(inputs, str) else len(inputs)
        dimensions = kwargs["dimensions"]
        rows = [
            SimpleNamespace(
                index=index,
                embedding=[float(index + 1)] + [0.0] * (dimensions - 1),
            )
            for index in range(count)
        ]
        if self.mode == "missing-row":
            rows = rows[:-1]
        elif self.mode == "wrong-dimension" and rows:
            rows[0] = SimpleNamespace(index=0, embedding=[1.0])
        elif self.mode == "duplicate-index" and len(rows) > 1:
            rows[1] = SimpleNamespace(index=0, embedding=rows[1].embedding)
        elif self.mode == "reverse":
            rows.reverse()
        return SimpleNamespace(
            data=rows,
            usage=SimpleNamespace(prompt_tokens=count),
        )


class FakeClient:
    def __init__(self, mode: str = "ok"):
        self.embeddings = FakeEmbeddingsEndpoint(mode)


def config() -> EmbeddingConfig:
    return EmbeddingConfig.from_mapping({
        "EMBEDDING_DIMENSIONS": "1024",
    })


def credentials() -> DashScopeCredentials:
    return DashScopeCredentials.from_mapping({
        "DASHSCOPE_API_KEY": "secret-key",
        "DASHSCOPE_WORKSPACE_ID": "workspace-id",
    })


def make_provider(mode: str = "ok"):
    client = FakeClient(mode)
    provider = DashScopeEmbeddingProvider(
        config(),
        credentials(),
        client=client,
    )
    return provider, client.embeddings.calls


class EmbeddingProviderTests(unittest.TestCase):
    def test_documents_are_batched_at_twenty_and_usage_is_summed(self):
        provider, calls = make_provider()

        result = provider.embed_documents([
            f"text-{index}" for index in range(21)
        ])

        self.assertEqual([len(call["input"]) for call in calls], [20, 1])
        self.assertEqual(result.input_tokens, 21)
        self.assertEqual(len(result.vectors), 21)

    def test_query_uses_configured_model_dimension_and_float_output(self):
        provider, calls = make_provider()

        result = provider.embed_query("HP780 的防护等级")

        self.assertEqual(calls[0]["model"], "qwen3.7-text-embedding")
        self.assertEqual(calls[0]["dimensions"], 1024)
        self.assertEqual(calls[0]["encoding_format"], "float")
        self.assertEqual(calls[0]["input"], "HP780 的防护等级")
        self.assertEqual(len(result.vectors), 1)

    def test_response_rows_are_restored_to_input_order(self):
        provider, calls = make_provider("reverse")

        result = provider.embed_documents(["first", "second"])

        self.assertEqual(result.vectors[0][0], 1.0)
        self.assertEqual(result.vectors[1][0], 2.0)

    def test_malformed_output_fails_as_embedding_api_error(self):
        for mode in ("missing-row", "wrong-dimension", "duplicate-index"):
            provider, calls = make_provider(mode)
            with self.subTest(mode=mode):
                with self.assertRaises(EmbeddingAPIError):
                    provider.embed_documents(["one", "two"])

    def test_empty_document_batch_returns_without_api_call(self):
        provider, calls = make_provider()

        result = provider.embed_documents([])

        self.assertEqual(result.vectors, ())
        self.assertEqual(result.input_tokens, 0)
        self.assertEqual(calls, [])

    def test_blank_inputs_fail_without_api_call(self):
        provider, calls = make_provider()

        with self.assertRaises(EmbeddingAPIError):
            provider.embed_query("   ")

        self.assertEqual(calls, [])

    def test_provider_errors_are_redacted(self):
        provider, calls = make_provider("api-error")

        with self.assertRaises(EmbeddingAPIError) as context:
            provider.embed_query("safe query")

        self.assertNotIn("secret provider detail", str(context.exception))
        self.assertIn("embedding provider request failed", str(context.exception))

    def test_real_client_uses_credentials_and_transport_configuration(self):
        sentinel = object()
        with patch(
            "knowledge_pipeline.retrieval.embedding.OpenAI",
            return_value=sentinel,
        ) as openai:
            DashScopeEmbeddingProvider(config(), credentials())

        openai.assert_called_once_with(
            api_key="secret-key",
            base_url=(
                "https://workspace-id.cn-beijing.maas.aliyuncs.com/"
                "compatible-mode/v1"
            ),
            timeout=30.0,
            max_retries=2,
        )


if __name__ == "__main__":
    unittest.main()
