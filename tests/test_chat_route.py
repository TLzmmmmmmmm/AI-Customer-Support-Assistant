import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx2 as httpx
from fastapi import HTTPException
from openai import APIConnectionError, APIStatusError, APITimeoutError

from agent import AgentDeadlineExceeded
from knowledge_pipeline.models import SourceRef
from knowledge_pipeline.retrieval.models import (
    EmbeddingAPIError,
    EntityCatalogError,
    VectorIndexNotReadyError,
)
from models import ChatMessage, ChatRequest
from routes import chat
from routing import Route, RouteExecutionResult, RouteTrace
from trace_models import FailureLayer, ToolTrace


def status_error(status_code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://example.com/chat")
    return APIStatusError(
        "private provider error",
        response=httpx.Response(status_code, request=request),
        body=None,
    )


class FakeOrchestrator:
    def __init__(self, *, events, answer="完整回答", error=None, sources=()):
        self.events = events
        self.answer = answer
        self.error = error
        self.sources = sources
        self.calls = []

    def run(self, messages, *, deadline):
        self.events.append("orchestrate")
        self.calls.append({"messages": messages, "deadline": deadline})
        if self.error is not None:
            raise self.error
        return RouteExecutionResult(
            answer=self.answer,
            trace=RouteTrace(
                route=Route.PRODUCT_SEARCH,
                tool_calls=(ToolTrace(name="search_products", success=True),),
                tool_call_count=1,
            ),
            sources=tuple(self.sources),
        )


async def consume_response(response) -> str:
    parts = []
    async for part in response.body_iterator:
        parts.append(part.decode("utf-8") if isinstance(part, bytes) else part)
    return "".join(parts)


class ChatRouteOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.request = SimpleNamespace(state=SimpleNamespace(
            started_at=1.0,
            request_id="request-1",
            route_trace=None,
            failure_layer=None,
        ))
        self.payload = ChatRequest(messages=[
            ChatMessage(role="user", content="介绍 HP780。"),
            ChatMessage(role="assistant", content="HP780 是一款对讲机。"),
            ChatMessage(role="user", content="它的防护等级是什么？"),
        ])

    def test_one_slot_covers_orchestration_and_complete_ndjson_response(self):
        events = []
        orchestrator = FakeOrchestrator(events=events)

        def acquire():
            events.append("acquire")
            return True

        def clock():
            events.append("clock")
            return 10.0

        def release():
            events.append("release")

        with (
            patch.object(chat, "try_acquire_llm_slot", side_effect=acquire),
            patch.object(chat, "release_llm_slot", side_effect=release),
            patch.object(chat.time, "monotonic", side_effect=clock),
            patch.object(chat, "log_request") as logged,
        ):
            response = chat.chat_stream(
                self.payload,
                self.request,
                None,
                orchestrator,
            )
            self.assertEqual(
                events,
                ["acquire", "clock", "clock", "orchestrate"],
            )
            body = asyncio.run(consume_response(response))

        self.assertEqual(events.count("release"), 1)
        self.assertEqual(orchestrator.calls[0]["messages"], self.payload.messages)
        self.assertEqual(
            [json.loads(line) for line in body.splitlines()],
            [
                {"type": "delta", "content": "完整回答"},
                {"type": "done"},
            ],
        )
        trace = self.request.state.route_trace
        self.assertEqual(trace.route, Route.PRODUCT_SEARCH)
        logged.assert_called_once_with(
            request_id="request-1",
            http_status=200,
            outcome="success",
            started_at=1.0,
            trace=trace,
        )

    def test_final_delta_removes_model_urls_and_appends_only_trusted_citations(self):
        trusted = SourceRef(
            title="LY198 产品详情",
            url="https://trusted.example/products/ly198/",
        )
        orchestrator = FakeOrchestrator(
            events=[],
            answer=(
                "LY198 功率信息。详情见 "
                "[产品页](https://fake.example/ly198)。\n\n"
                "参考资料：\nFake：https://fake.example/source"
            ),
            sources=(trusted,),
        )

        with (
            patch.object(chat, "try_acquire_llm_slot", return_value=True),
            patch.object(chat, "release_llm_slot"),
            patch.object(chat, "log_request"),
        ):
            response = chat.chat_stream(
                self.payload,
                self.request,
                None,
                orchestrator,
            )
            body = asyncio.run(consume_response(response))

        events = [json.loads(line) for line in body.splitlines()]
        self.assertEqual(events, [
            {
                "type": "delta",
                "content": "LY198 功率信息。详情见 产品页。",
            },
            {
                "type": "citations",
                "heading": "参考资料：",
                "items": [{
                    "title": "LY198 产品详情",
                    "url": "https://trusted.example/products/ly198/",
                }],
            },
            {"type": "done"},
        ])
        self.assertNotIn("fake.example", body)
        self.assertEqual(self.request.state.route_trace.citation_count, 1)
        self.assertTrue(self.request.state.route_trace.answer_sanitized)

    def test_empty_sanitized_answer_uses_safe_answer_without_references(self):
        from agent import SAFE_AGENT_ANSWER

        orchestrator = FakeOrchestrator(
            events=[],
            answer="References:\nhttps://fake.example",
            sources=(SourceRef(
                title="Trusted",
                url="https://trusted.example/source",
            ),),
        )

        with (
            patch.object(chat, "try_acquire_llm_slot", return_value=True),
            patch.object(chat, "release_llm_slot"),
            patch.object(chat, "log_request"),
        ):
            response = chat.chat_stream(
                self.payload, self.request, None, orchestrator
            )
            body = asyncio.run(consume_response(response))

        events = [json.loads(line) for line in body.splitlines()]
        self.assertEqual(events[0], {
            "type": "delta",
            "content": SAFE_AGENT_ANSWER,
        })
        self.assertNotIn("References:", body)
        self.assertEqual(self.request.state.route_trace.citation_count, 0)

    def test_known_errors_keep_existing_http_mapping_and_release_slot(self):
        request = httpx.Request("POST", "https://example.com/chat")
        cases = (
            (EmbeddingAPIError("provider failed"), 503, "retrieval_unavailable"),
            (VectorIndexNotReadyError("index failed"), 503, "retrieval_unavailable"),
            (EntityCatalogError("resolver failed"), 503, "retrieval_unavailable"),
            (AgentDeadlineExceeded("expired"), 504, "timeout"),
            (APITimeoutError(request=request), 504, "timeout"),
            (APIConnectionError(request=request), 503, "provider_unavailable"),
            (status_error(400), 502, "provider_error"),
        )
        for error, status, code in cases:
            with self.subTest(error=type(error).__name__):
                error.route = Route.KNOWLEDGE
                error.failure_layer = (
                    FailureLayer.RETRIEVAL
                    if isinstance(error, (
                        EmbeddingAPIError,
                        VectorIndexNotReadyError,
                        EntityCatalogError,
                    ))
                    else FailureLayer.GENERATION
                )
                orchestrator = FakeOrchestrator(events=[], error=error)
                with (
                    patch.object(chat, "try_acquire_llm_slot", return_value=True),
                    patch.object(chat, "release_llm_slot") as release,
                ):
                    with self.assertRaises(HTTPException) as caught:
                        chat.chat_stream(
                            self.payload,
                            self.request,
                            None,
                            orchestrator,
                        )

                self.assertEqual(caught.exception.status_code, status)
                self.assertEqual(caught.exception.detail["code"], code)
                self.assertEqual(
                    self.request.state.route_trace.failure_layer,
                    error.failure_layer,
                )
                release.assert_called_once_with()

    def test_unexpected_error_propagates_records_safe_trace_and_releases_slot(self):
        error = RuntimeError("private defect")
        error.route = Route.DIRECT
        error.failure_layer = FailureLayer.GENERATION
        orchestrator = FakeOrchestrator(events=[], error=error)

        with (
            patch.object(chat, "try_acquire_llm_slot", return_value=True),
            patch.object(chat, "release_llm_slot") as release,
        ):
            with self.assertRaises(RuntimeError):
                chat.chat_stream(
                    self.payload,
                    self.request,
                    None,
                    orchestrator,
                )

        self.assertEqual(self.request.state.route_trace.route, Route.DIRECT)
        release.assert_called_once_with()

    def test_busy_slot_rejects_before_deadline_or_orchestration(self):
        orchestrator = FakeOrchestrator(events=[])
        with (
            patch.object(chat, "try_acquire_llm_slot", return_value=False),
            patch.object(chat, "release_llm_slot") as release,
            patch.object(chat.time, "monotonic") as clock,
        ):
            with self.assertRaises(HTTPException) as caught:
                chat.chat_stream(
                    self.payload,
                    self.request,
                    None,
                    orchestrator,
                )

        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(orchestrator.calls, [])
        self.assertIsNone(self.request.state.route_trace)
        clock.assert_not_called()
        release.assert_not_called()


if __name__ == "__main__":
    unittest.main()
