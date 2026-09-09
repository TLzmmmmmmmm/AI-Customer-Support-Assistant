"""HTTP acceptance for Routing V1 with controlled external calls."""

import json
import unittest
from collections import defaultdict, deque
from threading import BoundedSemaphore
from types import SimpleNamespace
from unittest.mock import patch

import httpx2 as httpx
from fastapi.testclient import TestClient
from openai import APIConnectionError, APITimeoutError

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
from routing import SAFE_FALLBACK_ANSWER
from services import llm


def record(product_id, vector, text, *, record_type="product"):
    return VectorRecord.model_validate({
        "schema_version": "1.0",
        "chunk_id": f"{record_type}:{product_id}:content",
        "parent_document_id": f"{record_type}:{product_id}",
        "type": record_type,
        "section": "技术参数" if record_type == "product" else "解决方案",
        "text": text,
        "language": "zh-CN",
        "content_hash": "a" * 64,
        "source_url": f"https://example.com/{product_id}/",
        "source_files": [f"src/content/{record_type}/{product_id}.json"],
        "metadata": ({
            "product_id": product_id,
            "slug": product_id,
            "category_id": "two-way-radio",
            "category_name": "对讲机通信",
        } if record_type == "product" else {
            "solution_id": product_id,
            "slug": product_id,
        }),
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


def text_completion(content, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=content, tool_calls=None),
    )])


def tool_completion(call_id, name, arguments):
    call = SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="tool_calls",
        message=SimpleNamespace(content=None, tool_calls=[call]),
    )])


class ChatHttpRoutingAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.embedding = ControlledEmbedding()
        records = [
            record("ly198", [0.0, 1.0], "# 润信达 LY198\n\n功率：2W"),
            record("hp780", [0.2, 0.8], "# HP780\n\n防护等级：IP68"),
            record(
                "hotel",
                [1.0, 0.0],
                "# 酒店通信\n\n酒店通信解决方案内容",
                record_type="solution",
            ),
        ]
        retriever = Retriever(
            embedding_provider=self.embedding,
            vector_index=NumpyExactVectorIndex(records),
            entity_resolver=ExactEntityResolver.from_records(records),
            default_top_k=5,
        )
        self.slot = BoundedSemaphore(1)
        self.enterContext(patch.object(concurrency, "llm_semaphore", self.slot))
        self.enterContext(patch.object(
            rate_limit,
            "request_history",
            defaultdict(deque),
        ))
        self.enterContext(patch.object(main, "build_retriever", return_value=retriever))
        self.create = self.enterContext(patch.object(
            llm.client.chat.completions,
            "create",
            side_effect=lambda **kwargs: text_completion("受控回答"),
        ))
        self.enterContext(patch.object(llm.time, "sleep"))
        self.client = self.enterContext(
            TestClient(main.app, raise_server_exceptions=False)
        )

    def post(self, question, history=()):
        messages = [*history, {"role": "user", "content": question}]
        return self.client.post("/api/chat-stream", json={"messages": messages})

    def assert_ndjson(self, response, answer="受控回答"):
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [json.loads(line) for line in response.text.splitlines()],
            [{"type": "delta", "content": answer}, {"type": "done"}],
        )

    def assert_slot_available(self):
        acquired = self.slot.acquire(blocking=False)
        try:
            self.assertTrue(acquired)
        finally:
            if acquired:
                self.slot.release()

    def test_six_canonical_routes_preserve_http_and_skip_unneeded_retrieval(self):
        cases = (
            ("推荐几款对讲机", "product_search", "search_products", 1, 1),
            ("LY198 功率是多少？", "exact_product", "get_product_details", 0, 1),
            ("公司的电话是多少？", "contact", "get_contact_info", 0, 1),
            ("你们有哪些解决方案？", "knowledge", None, 1, 1),
            ("你好", "direct", None, 0, 1),
            ("LY198 今天还有多少库存？", "fallback", None, 0, 0),
        )
        for question, route, tool, embedding_count, completion_count in cases:
            with self.subTest(question=question):
                self.embedding.queries.clear()
                self.create.reset_mock()
                with self.assertLogs("ai_customer_support", level="INFO") as logs:
                    response = self.post(question)
                answer = SAFE_FALLBACK_ANSWER if route == "fallback" else "受控回答"
                self.assert_ndjson(response, answer)
                self.assertEqual(len(self.embedding.queries), embedding_count)
                self.assertEqual(self.create.call_count, completion_count)
                message = logs.records[0].getMessage()
                self.assertIn(f"route={route}", message)
                expected_count = 1 if tool else 0
                self.assertIn(f"tool_call_count={expected_count}", message)
                if tool:
                    self.assertIn(f'"name":"{tool}"', message)
                self.assert_slot_available()

    def test_explicit_scenario_product_requests_are_deterministic_searches(self):
        for question in (
            "推荐适合酒店使用的产品",
            "地下停车场有什么产品推荐",
        ):
            with self.subTest(question=question):
                self.embedding.queries.clear()
                self.create.reset_mock()
                with self.assertLogs("ai_customer_support", level="INFO") as logs:
                    response = self.post(question)
                self.assert_ndjson(response)
                self.assertEqual(self.embedding.queries, [question])
                self.assertEqual(self.create.call_count, 1)
                self.assertNotIn("tools", self.create.call_args.kwargs)
                self.assertIn("route=product_search", logs.records[0].getMessage())
                self.assertIn('"name":"search_products"', logs.records[0].getMessage())

    def test_scenario_only_uses_router_then_non_agentic_product_search(self):
        question = "地下停车场几十个人通信，有什么建议？"
        self.create.side_effect = [
            text_completion("product_search"),
            text_completion("候选产品，请联系专业技术人员确认选型。"),
        ]
        with self.assertLogs("ai_customer_support", level="INFO") as logs:
            response = self.post(question)

        self.assert_ndjson(response, "候选产品，请联系专业技术人员确认选型。")
        self.assertEqual(self.embedding.queries, [question])
        self.assertEqual(self.create.call_count, 2)
        self.assertNotIn("tools", self.create.call_args_list[1].kwargs)
        self.assertIn("route=product_search", logs.records[0].getMessage())
        self.assertIn("tool_call_count=1", logs.records[0].getMessage())

    def test_mixed_knowledge_and_contact_is_agentic_after_one_retrieval(self):
        question = "介绍应急通信解决方案，另外怎么联系你们？"
        self.create.side_effect = [
            tool_completion("contact-1", "get_contact_info", "{}"),
            text_completion("方案说明和联系渠道。"),
        ]
        with self.assertLogs("ai_customer_support", level="INFO") as logs:
            response = self.post(question)

        self.assert_ndjson(response, "方案说明和联系渠道。")
        self.assertEqual(self.embedding.queries, [question])
        self.assertEqual(self.create.call_count, 2)
        self.assertIn("tools", self.create.call_args_list[0].kwargs)
        message = logs.records[0].getMessage()
        self.assertIn("route=knowledge", message)
        self.assertIn('"name":"get_contact_info"', message)

    def test_contextual_followup_uses_router_and_agent_without_generic_retrieval(self):
        self.create.side_effect = [
            text_completion("exact_product"),
            tool_completion(
                "details-1",
                "get_product_details",
                '{"product_id":"hp780"}',
            ),
            text_completion("第二款 HP780 的防护等级是 IP68。"),
        ]
        history = (
            {"role": "user", "content": "推荐两款产品。"},
            {"role": "assistant", "content": "可考虑 LY198 和 HP780。"},
        )
        with self.assertLogs("ai_customer_support", level="INFO") as logs:
            response = self.post("第二个的防护等级呢？", history)

        self.assert_ndjson(response, "第二款 HP780 的防护等级是 IP68。")
        self.assertEqual(self.embedding.queries, [])
        self.assertEqual(self.create.call_count, 3)
        self.assertIn("route=exact_product", logs.records[0].getMessage())

    def test_unknown_product_domain_error_is_observed_without_system_failure(self):
        self.create.side_effect = [
            text_completion("exact_product"),
            tool_completion(
                "missing-1",
                "get_product_details",
                '{"product_id":"unknown-77"}',
            ),
            text_completion("没有找到该型号，请核对后重试。"),
        ]
        with self.assertLogs("ai_customer_support", level="INFO") as logs:
            response = self.post("UNKNOWN-77 的功率是多少？")

        self.assert_ndjson(response, "没有找到该型号，请核对后重试。")
        message = logs.records[0].getMessage()
        self.assertIn('"error_code":"PRODUCT_NOT_FOUND"', message)
        self.assertIn("failure_layer=-", message)
        self.assertEqual(self.embedding.queries, [])

    def test_invalid_router_output_returns_fixed_fallback_with_routing_failure(self):
        self.create.side_effect = lambda **kwargs: text_completion(
            "product_search because maybe"
        )
        with self.assertLogs("ai_customer_support", level="INFO") as logs:
            response = self.post("地下停车场通信应该怎么解决？")

        self.assert_ndjson(response, SAFE_FALLBACK_ANSWER)
        self.assertEqual(self.embedding.queries, [])
        message = logs.records[0].getMessage()
        self.assertIn("route=fallback", message)
        self.assertIn("failure_layer=ROUTING", message)

    def test_trace_log_never_contains_conversation_observation_or_contact_facts(self):
        private_question = "怎么联系你们？private-question-marker"
        with self.assertLogs("ai_customer_support", level="INFO") as logs:
            response = self.post(private_question)
        self.assert_ndjson(response)
        message = logs.records[0].getMessage()
        for forbidden in (
            private_question,
            "BEGIN_TOOL_DATA",
            "company_name",
            "duty_phone",
            "email",
            "address",
            "@",
        ):
            self.assertNotIn(forbidden, message)

    def test_router_provider_and_retrieval_failures_keep_safe_http_contract(self):
        request = httpx.Request("POST", "https://example.com/chat")
        self.create.side_effect = APITimeoutError(request=request)
        with self.assertLogs("ai_customer_support", level="INFO") as logs:
            response = self.post("地下停车场通信应该怎么解决？")
        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["error"]["code"], "timeout")
        self.assertIn("failure_layer=ROUTING", logs.records[0].getMessage())
        self.assert_slot_available()

        self.create.side_effect = lambda **kwargs: text_completion("unused")
        self.embedding.error = EmbeddingAPIError("private embedding detail")
        with self.assertLogs("ai_customer_support", level="INFO") as logs:
            response = self.post("你们有哪些解决方案？")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "retrieval_unavailable")
        message = logs.records[0].getMessage()
        self.assertIn("route=knowledge", message)
        self.assertIn("failure_layer=RETRIEVAL", message)
        self.assertNotIn("private embedding detail", message)
        self.assert_slot_available()

    def test_validation_rate_and_concurrency_fail_before_orchestration(self):
        invalid = self.client.post(
            "/api/chat-stream",
            json={"messages": [{"role": "user", "content": "   "}]},
        )
        self.assertEqual(invalid.status_code, 422)

        self.slot.acquire()
        try:
            busy = self.post("你好")
        finally:
            self.slot.release()
        self.assertEqual(busy.status_code, 503)
        self.assertEqual(busy.json()["error"]["code"], "concurrency_limit")

        rate_limit.request_history.clear()
        with patch.object(rate_limit, "RATE_LIMIT_REQUESTS", 1):
            self.assertEqual(self.post("你好").status_code, 200)
            limited = self.post("你好")
        self.assertEqual(limited.status_code, 429)
        self.assertEqual(limited.json()["error"]["code"], "rate_limit")
        self.assert_slot_available()


if __name__ == "__main__":
    unittest.main()
