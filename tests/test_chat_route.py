import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx2 as httpx
from fastapi import HTTPException
from openai import APIConnectionError, APIStatusError, APITimeoutError
from starlette.requests import ClientDisconnect

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
from trace_models import (
    FailureLayer,
    ToolTrace,
    bind_request_state,
    initialize_request_trace,
    request_trace_fields,
    reset_request_state,
)


def status_error(status_code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://example.com/chat")
    return APIStatusError(
        "private provider error",
        response=httpx.Response(status_code, request=request),
        body=None,
    )


class FakeOrchestrator:
    def __init__(
        self,
        *,
        events,
        answer="完整回答",
        error=None,
        sources=(),
        chunks=(),
    ):
        self.events = events
        self.answer = answer
        self.error = error
        self.sources = sources
        self.chunks = tuple(chunks)
        self.calls = []
        self.closed = False

    def _result(self):
        return RouteExecutionResult(
            answer=self.answer,
            trace=RouteTrace(
                route=Route.PRODUCT_SEARCH,
                tool_calls=(ToolTrace(name="search_products", success=True),),
                tool_call_count=1,
            ),
            sources=tuple(self.sources),
        )

    def run(self, messages, *, deadline):
        self.events.append("orchestrate")
        self.calls.append({"messages": messages, "deadline": deadline})
        if self.error is not None:
            raise self.error
        return self._result()

    def stream(self, messages, *, deadline):
        self.events.append("orchestrate")
        self.calls.append({"messages": messages, "deadline": deadline})
        try:
            if self.error is not None and not self.chunks:
                raise self.error
            yield from self.chunks
            if self.error is not None:
                raise self.error
            yield self._result()
        finally:
            self.closed = True


async def consume_response(response, observed_events=None) -> str:
    parts = []
    async for part in response.body_iterator:
        text = part.decode("utf-8") if isinstance(part, bytes) else part
        parts.append(text)
        if observed_events is not None:
            observed_events.append(json.loads(text)["type"])
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

        def release():
            events.append("release")

        with (
            patch.object(chat, "try_acquire_llm_slot", side_effect=acquire),
            patch.object(chat, "release_llm_slot", side_effect=release),
            patch.object(chat.time, "monotonic", return_value=10.0),
            patch.object(
                chat,
                "log_request",
                side_effect=lambda **_: events.append("summary"),
            ) as logged,
        ):
            response = chat.chat_stream(
                self.payload,
                self.request,
                None,
                orchestrator,
            )
            self.assertEqual(
                events,
                ["acquire", "orchestrate"],
            )
            body = asyncio.run(consume_response(response, events))

        self.assertEqual(events.count("release"), 1)
        self.assertEqual(events[-4:], ["delta", "done", "summary", "release"])
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
            completed_at=10.0,
        )
        self.assertEqual(trace.first_delta_latency_ms, 9000.0)
        self.assertEqual(trace.buffering_saved_ms, 0.0)

    def test_safe_chunks_stream_in_order_before_done(self):
        events = []
        orchestrator = FakeOrchestrator(
            events=events,
            answer="第一段第二段",
            chunks=("第一段", "第二段"),
        )

        with (
            patch.object(chat, "try_acquire_llm_slot", return_value=True),
            patch.object(
                chat,
                "release_llm_slot",
                side_effect=lambda: events.append("release"),
            ) as release,
            patch.object(
                chat,
                "log_request",
                side_effect=lambda **_: events.append("summary"),
            ),
        ):
            response = chat.chat_stream(
                self.payload, self.request, None, orchestrator
            )
            body = asyncio.run(consume_response(response, events))

        self.assertEqual(
            [json.loads(line) for line in body.splitlines()],
            [
                {"type": "delta", "content": "第一段"},
                {"type": "delta", "content": "第二段"},
                {"type": "done"},
            ],
        )
        self.assertEqual(events[-4:], ["delta", "done", "summary", "release"])
        self.assertTrue(orchestrator.closed)
        release.assert_called_once_with()

    def test_summary_occurs_only_after_done_iterator_resumes(self):
        events = []
        orchestrator = FakeOrchestrator(events=events)

        async def verify_order(response):
            iterator = response.body_iterator.__aiter__()
            self.assertEqual(json.loads(await anext(iterator))["type"], "delta")
            self.assertNotIn("summary", events)
            self.assertEqual(json.loads(await anext(iterator))["type"], "done")
            self.assertNotIn("summary", events)
            with self.assertRaises(StopAsyncIteration):
                await anext(iterator)
            self.assertEqual(events[-2:], ["summary", "release"])

        with (
            patch.object(chat, "try_acquire_llm_slot", return_value=True),
            patch.object(
                chat,
                "release_llm_slot",
                side_effect=lambda: events.append("release"),
            ),
            patch.object(
                chat,
                "log_request",
                side_effect=lambda **_: events.append("summary"),
            ),
        ):
            response = chat.chat_stream(
                self.payload, self.request, None, orchestrator
            )
            asyncio.run(verify_order(response))

    def test_post_delta_failures_use_existing_safe_error_taxonomy(self):
        request = httpx.Request("POST", "https://example.com/chat")
        cases = (
            (EmbeddingAPIError("private"), "retrieval_unavailable"),
            (AgentDeadlineExceeded("expired"), "timeout"),
            (APIConnectionError(request=request), "provider_unavailable"),
            (status_error(500), "provider_error"),
        )

        for error, expected_code in cases:
            with self.subTest(error=type(error).__name__):
                error.route = Route.DIRECT
                error.failure_layer = FailureLayer.GENERATION
                self.request.state.route_trace = None
                orchestrator = FakeOrchestrator(
                    events=[],
                    answer="不应完成",
                    chunks=("部分回答",),
                    error=error,
                )

                with (
                    patch.object(chat, "try_acquire_llm_slot", return_value=True),
                    patch.object(chat, "release_llm_slot") as release,
                    patch.object(chat, "log_request") as logged,
                ):
                    response = chat.chat_stream(
                        self.payload, self.request, None, orchestrator
                    )
                    body = asyncio.run(consume_response(response))

                body_events = [json.loads(line) for line in body.splitlines()]
                self.assertEqual(
                    body_events[0],
                    {"type": "delta", "content": "部分回答"},
                )
                self.assertEqual(body_events[1]["type"], "error")
                self.assertEqual(body_events[1]["code"], expected_code)
                self.assertEqual(body_events[1]["request_id"], "request-1")
                event_types = [event["type"] for event in body_events]
                self.assertNotIn("done", event_types)
                self.assertNotIn("citations", event_types)
                self.assertEqual(logged.call_args.kwargs["http_status"], 200)
                self.assertEqual(
                    logged.call_args.kwargs["outcome"],
                    expected_code,
                )
                self.assertEqual(
                    logged.call_args.kwargs["failure_code"],
                    expected_code,
                )
                self.assertIsNotNone(
                    logged.call_args.kwargs["trace"].first_delta_latency_ms
                )
                self.assertIsNone(
                    logged.call_args.kwargs["trace"].buffering_saved_ms
                )
                release.assert_called_once_with()

    def test_streamed_final_invariant_mismatch_fails_closed(self):
        orchestrator = FakeOrchestrator(
            events=[],
            answer="不同的最终回答",
            chunks=("已经发送",),
        )

        with (
            patch.object(chat, "try_acquire_llm_slot", return_value=True),
            patch.object(chat, "release_llm_slot"),
            patch.object(chat, "log_request") as logged,
        ):
            response = chat.chat_stream(
                self.payload, self.request, None, orchestrator
            )
            body = asyncio.run(consume_response(response))

        body_events = [json.loads(line) for line in body.splitlines()]
        self.assertEqual([item["type"] for item in body_events], ["delta", "error"])
        self.assertEqual(body_events[-1]["code"], "internal_error")
        self.assertEqual(logged.call_args.kwargs["outcome"], "internal_error")

    def test_disconnect_after_delta_closes_graph_and_logs_interruption(self):
        events = []
        orchestrator = FakeOrchestrator(
            events=events,
            answer="第一段第二段",
            chunks=("第一段", "第二段"),
        )

        async def consume_one_and_close(response):
            iterator = response.body_iterator.__aiter__()
            self.assertEqual(json.loads(await anext(iterator))["type"], "delta")
            await iterator.aclose()

        with (
            patch.object(chat, "try_acquire_llm_slot", return_value=True),
            patch.object(chat, "release_llm_slot") as release,
            patch.object(chat, "log_request") as logged,
        ):
            response = chat.chat_stream(
                self.payload, self.request, None, orchestrator
            )
            asyncio.run(consume_one_and_close(response))

        self.assertTrue(orchestrator.closed)
        release.assert_called_once_with()
        self.assertEqual(logged.call_count, 1)
        kwargs = logged.call_args.kwargs
        self.assertEqual(kwargs["http_status"], 200)
        self.assertEqual(kwargs["outcome"], "stream_interrupted")
        self.assertEqual(kwargs["failure_code"], "stream_interrupted")
        self.assertIsNone(kwargs["failure_layer"])
        self.assertIsNotNone(self.request.state.first_delta_latency_ms)
        self.assertIsNone(self.request.state.buffering_saved_ms)

    def test_asgi_send_disconnect_closes_response_owner(self):
        orchestrator = FakeOrchestrator(
            events=[],
            answer="第一段第二段",
            chunks=("第一段", "第二段"),
        )

        async def disconnect(response):
            async def receive():
                return {"type": "http.disconnect"}

            async def send(message):
                if (
                    message["type"] == "http.response.body"
                    and message.get("more_body")
                ):
                    raise OSError("client disconnected")

            with self.assertRaises(ClientDisconnect):
                await response(
                    {"type": "http", "asgi": {"spec_version": "2.4"}},
                    receive,
                    send,
                )

        with (
            patch.object(chat, "try_acquire_llm_slot", return_value=True),
            patch.object(chat, "release_llm_slot") as release,
            patch.object(chat, "log_request") as logged,
        ):
            response = chat.chat_stream(
                self.payload, self.request, None, orchestrator
            )
            asyncio.run(disconnect(response))

        self.assertTrue(orchestrator.closed)
        release.assert_called_once_with()
        self.assertEqual(logged.call_args.kwargs["outcome"], "stream_interrupted")

    def test_close_before_first_body_delta_releases_prefetched_stream(self):
        orchestrator = FakeOrchestrator(
            events=[],
            answer="第一段第二段",
            chunks=("第一段", "第二段"),
        )

        async def close_without_reading(response):
            await response.body_iterator.aclose()

        outer_state = SimpleNamespace()
        initialize_request_trace(outer_state)
        outer_state.router_type = "outer-context"
        outer_token = bind_request_state(outer_state)
        try:
            with (
                patch.object(chat, "try_acquire_llm_slot", return_value=True),
                patch.object(chat, "release_llm_slot") as release,
                patch.object(chat, "log_request") as logged,
            ):
                response = chat.chat_stream(
                    self.payload, self.request, None, orchestrator
                )
                self.assertEqual(
                    request_trace_fields()["router_type"],
                    "outer-context",
                )
                asyncio.run(close_without_reading(response))
                self.assertEqual(
                    request_trace_fields()["router_type"],
                    "outer-context",
                )
        finally:
            reset_request_state(outer_token)

        self.assertTrue(orchestrator.closed)
        release.assert_called_once_with()
        self.assertEqual(logged.call_args.kwargs["outcome"], "stream_interrupted")
        self.assertIsNone(self.request.state.first_delta_latency_ms)

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

    def test_rendering_failure_preserves_completed_route_trace(self):
        orchestrator = FakeOrchestrator(events=[])

        with (
            patch.object(chat, "try_acquire_llm_slot", return_value=True),
            patch.object(chat, "release_llm_slot") as release,
            patch.object(
                chat,
                "render_answer",
                side_effect=RuntimeError("render failure"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "render failure"):
                chat.chat_stream(
                    self.payload,
                    self.request,
                    None,
                    orchestrator,
                )

        trace = self.request.state.route_trace
        self.assertIsNotNone(trace)
        self.assertEqual(trace.route, Route.PRODUCT_SEARCH)
        self.assertEqual(trace.tool_call_count, 1)
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
