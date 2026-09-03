"""HTTP acceptance with real RAG components and controlled external calls."""

import json
import unittest
from collections import defaultdict, deque
from threading import BoundedSemaphore
from types import SimpleNamespace
from unittest.mock import patch

import httpx2 as httpx
from fastapi.testclient import TestClient
from openai import APIConnectionError, APIStatusError, APITimeoutError

import concurrency
import main
import rate_limit
from knowledge_pipeline.retrieval import (
    ExactEntityResolver,
    NumpyExactVectorIndex,
    Retriever,
)
from knowledge_pipeline.retrieval.models import (
    EmbeddingAPIError,
    EmbeddingBatch,
    VectorRecord,
)
from routes import chat
from services import llm


def record(product_id, vector, text):
    return VectorRecord.model_validate({
        "schema_version": "1.0",
        "chunk_id": f"product:{product_id}:content",
        "parent_document_id": f"product:{product_id}",
        "type": "product",
        "section": "技术参数",
        "text": text,
        "language": "zh-CN",
        "content_hash": "a" * 64,
        "source_url": f"https://example.com/{product_id}/",
        "source_files": [f"src/content/products/{product_id}.json"],
        "metadata": {
            "product_id": product_id,
            "slug": product_id,
            "category_id": "two-way-radio",
            "category_name": "对讲机通信",
        },
        "embedding_provider": "dashscope",
        "embedding_model": "qwen3.7-text-embedding",
        "embedding_dimensions": 2,
        "embedding_text_type": "document",
        "embedding": vector,
    })


class ControlledEmbedding:
    def __init__(self):
        self.queries = []
        self.error = None

    def embed_query(self, query):
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return EmbeddingBatch(vectors=([1.0, 0.0],), input_tokens=1)


def delta(content):
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=content),
    )])


def provider_error(status):
    return APIStatusError(
        "private provider detail",
        response=httpx.Response(
            status,
            request=httpx.Request("POST", "https://example.com/completions"),
        ),
        body=None,
    )


class ChatHttpAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.provider = ControlledEmbedding()
        records = [
            record("hp780", [0.0, 1.0], "# HP780\n\n防护等级：IP68"),
            record("hp790ex", [1.0, 0.0], "# HP790EX\n\n防爆机型"),
        ]
        retriever = Retriever(
            embedding_provider=self.provider,
            vector_index=NumpyExactVectorIndex(records),
            entity_resolver=ExactEntityResolver.from_records(records),
            default_top_k=5,
        )
        self.slot = BoundedSemaphore(1)
        self.enterContext(patch.object(concurrency, "llm_semaphore", self.slot))
        self.enterContext(patch.object(
            rate_limit, "request_history", defaultdict(deque),
        ))
        self.enterContext(patch.object(main, "build_retriever", return_value=retriever))
        self.create = self.enterContext(patch.object(
            llm.client.chat.completions,
            "create",
            side_effect=lambda **kwargs: iter([delta("IP"), delta("68")]),
        ))
        self.enterContext(patch.object(llm.time, "sleep"))
        self.client = self.enterContext(TestClient(main.app, raise_server_exceptions=False))
        self.messages = [{"role": "user", "content": "HP780 的防护等级是什么？"}]

    def post(self, messages=None):
        return self.client.post(
            "/api/chat-stream",
            json={"messages": self.messages if messages is None else messages},
        )

    def assert_slot_available(self):
        available = self.slot.acquire(blocking=False)
        try:
            self.assertTrue(available, "request leaked its concurrency slot")
        finally:
            if available:
                self.slot.release()

    def assert_http_error(self, response, status, code):
        self.assertEqual(response.status_code, status)
        error = response.json()["error"]
        self.assertEqual(set(error), {"code", "message", "request_id"})
        self.assertEqual(error["code"], code)
        self.assertEqual(error["request_id"], response.headers["X-Request-ID"])
        self.assertNotIn("private", response.text)

    def test_real_rag_route_preserves_history_stream_and_minimal_log(self):
        history = [
            {"role": "user", "content": "介绍对讲机"},
            {"role": "assistant", "content": "请告诉我型号。"},
        ]
        with self.assertLogs("ai_customer_support", level="INFO") as logs:
            response = self.post(history + self.messages)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("application/x-ndjson"))
        self.assertEqual([json.loads(line) for line in response.text.splitlines()], [
            {"type": "delta", "content": "IP"},
            {"type": "delta", "content": "68"},
            {"type": "done"},
        ])
        self.assertEqual(self.provider.queries, [self.messages[0]["content"]])
        self.assertEqual(self.create.call_count, 1)
        messages = self.create.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1:3], history)
        payload = json.loads(messages[-1]["content"].split("BEGIN_RAG_DATA\n", 1)[1].rsplit("\nEND_RAG_DATA", 1)[0])
        self.assertEqual(payload, {
            "user_question": "HP780 的防护等级是什么？",
            "retrieved_context": [
                {"type": "product", "section": "技术参数", "text": "# HP780\n\n防护等级：IP68"},
                {"type": "product", "section": "技术参数", "text": "# HP790EX\n\n防爆机型"},
            ],
        })
        self.assertEqual(len(logs.records), 1)
        self.assertRegex(logs.records[0].getMessage(), (
            r"^request_id=" + response.headers["X-Request-ID"]
            + r" http_status=200 outcome=success latency_ms=\d+\.\d error=-$"
        ))
        self.assert_slot_available()

    def test_invalid_inputs_do_not_call_providers(self):
        invalid = [
            [{"role": "user", "content": "   "}],
            [{"role": "user", "content": "x" * 4001}],
            [{"role": "assistant", "content": "invalid start"}],
            [{"role": "user" if i % 2 == 0 else "assistant", "content": "x"} for i in range(21)],
            [{"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 3000} for i in range(7)],
        ]
        for messages in invalid:
            with self.subTest(length=len(messages)):
                self.assert_http_error(self.post(messages), 422, "validation_error")
        self.assertEqual(self.provider.queries, [])
        self.create.assert_not_called()
        self.assert_slot_available()

    def test_busy_slot_rejects_before_retrieval(self):
        self.slot.acquire()
        try:
            self.assert_http_error(self.post(), 503, "concurrency_limit")
        finally:
            self.slot.release()
        self.assertEqual(self.provider.queries, [])
        self.create.assert_not_called()

    def test_rate_limit_rejects_second_request_before_retrieval(self):
        with patch.object(rate_limit, "RATE_LIMIT_REQUESTS", 1):
            self.assertEqual(self.post().status_code, 200)
            response = self.post()
        self.assert_http_error(response, 429, "rate_limit")
        self.assertGreater(int(response.headers["Retry-After"]), 0)
        self.assertEqual(len(self.provider.queries), 1)
        self.assertEqual(self.create.call_count, 1)
        self.assert_slot_available()

    def test_retrieval_failure_is_safe_http_503_without_generation(self):
        self.provider.error = EmbeddingAPIError("private embedding detail")
        with self.assertLogs("ai_customer_support", level="INFO") as logs:
            response = self.post()
        self.assert_http_error(response, 503, "retrieval_unavailable")
        self.create.assert_not_called()
        self.assertEqual(len(logs.records), 1)
        self.assertRegex(logs.records[0].getMessage(), (
            r"^request_id=" + response.headers["X-Request-ID"]
            + r" http_status=503 outcome=retrieval_unavailable latency_ms=\d+\.\d error=EmbeddingAPIError$"
        ))
        self.assert_slot_available()

    def test_pre_stream_errors_preserve_status_retry_and_slot_contract(self):
        request = httpx.Request("POST", "https://example.com/completions")
        cases = [
            (APITimeoutError(request=request), 504, "timeout", 1),
            (APIConnectionError(request=request), 503, "provider_unavailable", 2),
            (provider_error(401), 502, "provider_error", 1),
            (provider_error(429), 502, "provider_error", 2),
            (provider_error(500), 502, "provider_error", 2),
        ]
        for error, status, code, attempts in cases:
            with self.subTest(error=type(error).__name__, status=getattr(error, "status_code", None)):
                self.create.reset_mock()
                self.create.side_effect = error
                self.assert_http_error(self.post(), status, code)
                self.assertEqual(self.create.call_count, attempts)
                self.assert_slot_available()
        self.assertEqual(len(self.provider.queries), len(cases))

    def test_connection_retry_does_not_repeat_retrieval(self):
        self.create.side_effect = [
            APIConnectionError(request=httpx.Request("POST", "https://example.com")),
            iter([delta("恢复成功")]),
        ]
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual([json.loads(line) for line in response.text.splitlines()], [
            {"type": "delta", "content": "恢复成功"}, {"type": "done"},
        ])
        self.assertEqual(self.create.call_count, 2)
        self.assertEqual(len(self.provider.queries), 1)
        self.assert_slot_available()

    def test_stream_errors_have_request_id_and_no_done_or_retry(self):
        request = httpx.Request("POST", "https://example.com/completions")
        for error, code in [
            (APITimeoutError(request=request), "timeout"),
            (APIConnectionError(request=request), "connection_error"),
            (provider_error(500), "upstream_error"),
        ]:
            with self.subTest(code=code):
                def broken_stream():
                    yield delta("部分内容")
                    raise error

                self.create.reset_mock()
                self.create.side_effect = lambda **kwargs: broken_stream()
                response = self.post()
                self.assertEqual(response.status_code, 200)
                events = [json.loads(line) for line in response.text.splitlines()]
                self.assertEqual([event["type"] for event in events], ["delta", "error"])
                self.assertEqual(events[0], {"type": "delta", "content": "部分内容"})
                self.assertEqual(set(events[1]), {"type", "code", "message", "request_id"})
                self.assertEqual(events[1]["code"], code)
                self.assertEqual(events[1]["request_id"], response.headers["X-Request-ID"])
                self.assertNotIn("private", response.text)
                self.assertEqual(self.create.call_count, 1)
                self.assert_slot_available()

    def test_unexpected_context_defect_remains_safe_500(self):
        with patch.object(chat, "build_retrieved_context", side_effect=RuntimeError("private defect")):
            response = self.post()
        self.assert_http_error(response, 500, "internal_error")
        self.create.assert_not_called()
        self.assert_slot_available()


if __name__ == "__main__":
    unittest.main()
