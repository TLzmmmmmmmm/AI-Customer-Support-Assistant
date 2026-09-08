import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx2 as httpx
from fastapi import HTTPException
from openai import APIConnectionError, APIStatusError, APITimeoutError

from agent import AgentDeadlineExceeded, AgentResult
from knowledge_pipeline.retrieval.models import (
    EmbeddingAPIError,
    EntityCatalogError,
    RetrievalResult,
    VectorIndexNotReadyError,
)
from models import ChatMessage, ChatRequest
from routes import chat


def retrieval_result() -> RetrievalResult:
    return RetrievalResult.model_validate({
        "rank": 1,
        "score": -0.25,
        "match_origin": "exact_entity",
        "matched_entity_ids": ["product:hp780"],
        "chunk_id": "product:hp780:specifications",
        "parent_document_id": "product:hp780",
        "type": "product",
        "section": "技术参数",
        "text": "# HP780\n\n防护等级：IP68",
        "content_hash": "a" * 64,
        "metadata": {
            "product_id": "hp780",
            "slug": "hp780",
            "category_id": "two-way-radio",
            "category_name": "对讲机通信",
        },
        "source_url": "https://example.com/hp780/",
        "source_files": ["src/content/products/hp780.json"],
    })


def status_error(status_code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://example.com/chat")
    return APIStatusError(
        "private provider error",
        response=httpx.Response(status_code, request=request),
        body=None,
    )


class FakeRetriever:
    def __init__(self, *, events, error=None, results=None):
        self.events = events
        self.error = error
        self.results = results if results is not None else [retrieval_result()]
        self.queries = []

    def retrieve(self, query):
        self.events.append("retrieve")
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.results


class FakeAgentLoop:
    def __init__(self, *, events, answer="完整回答", error=None):
        self.events = events
        self.answer = answer
        self.error = error
        self.calls = []

    def run(self, messages, *, deadline):
        self.events.append("agent")
        self.calls.append({"messages": messages, "deadline": deadline})
        if self.error is not None:
            raise self.error
        return AgentResult(answer=self.answer)


async def consume_response(response) -> str:
    parts = []
    async for part in response.body_iterator:
        parts.append(part.decode("utf-8") if isinstance(part, bytes) else part)
    return "".join(parts)


def parse_rag_payload(content: str) -> dict:
    begin = "BEGIN_RAG_DATA\n"
    end = "\nEND_RAG_DATA"
    start = content.index(begin) + len(begin)
    finish = content.rindex(end)
    return json.loads(content[start:finish])


class ChatRouteAgentOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.request = SimpleNamespace(state=SimpleNamespace(
            started_at=1.0,
            request_id="request-1",
        ))
        self.payload = ChatRequest(messages=[
            ChatMessage(role="user", content="介绍 HP780。"),
            ChatMessage(role="assistant", content="HP780 是一款对讲机。"),
            ChatMessage(role="user", content="它的防护等级是什么？"),
        ])

    def test_one_slot_covers_initial_rag_agent_and_complete_ndjson_response(self):
        events = []
        retriever = FakeRetriever(events=events)
        agent_loop = FakeAgentLoop(events=events, answer="完整回答")
        clock_value = 10.0

        def acquire():
            events.append("acquire")
            return True

        def clock():
            events.append("clock")
            return clock_value

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
                retriever,
                agent_loop,
            )
            self.assertEqual(
                events,
                ["acquire", "clock", "clock", "retrieve", "clock", "agent"],
            )
            body = asyncio.run(consume_response(response))

        self.assertEqual(events.count("release"), 1)
        self.assertGreater(events.index("release"), events.index("agent"))
        self.assertEqual(retriever.queries, ["它的防护等级是什么？"])
        self.assertEqual(
            [json.loads(line) for line in body.splitlines()],
            [
                {"type": "delta", "content": "完整回答"},
                {"type": "done"},
            ],
        )
        provider_messages = agent_loop.calls[0]["messages"]
        self.assertEqual(
            provider_messages[1:3],
            [
                {"role": "user", "content": "介绍 HP780。"},
                {"role": "assistant", "content": "HP780 是一款对讲机。"},
            ],
        )
        self.assertEqual(
            parse_rag_payload(provider_messages[-1]["content"]),
            {
                "retrieved_context": [{
                    "type": "product",
                    "section": "技术参数",
                    "text": "# HP780\n\n防护等级：IP68",
                }],
                "user_question": "它的防护等级是什么？",
            },
        )
        logged.assert_called_once_with(
            request_id="request-1",
            http_status=200,
            outcome="success",
            started_at=1.0,
        )

    def test_deadline_expiry_after_retrieval_returns_504_before_agent(self):
        events = []
        retriever = FakeRetriever(events=events)
        agent_loop = FakeAgentLoop(events=events)
        times = iter([10.0, 10.1, 130.0])

        with (
            patch.object(chat, "try_acquire_llm_slot", return_value=True),
            patch.object(chat, "release_llm_slot") as release,
            patch.object(
                chat.time,
                "monotonic",
                side_effect=lambda: next(times),
            ),
        ):
            with self.assertRaises(HTTPException) as caught:
                chat.chat_stream(
                    self.payload,
                    self.request,
                    None,
                    retriever,
                    agent_loop,
                )

        self.assertEqual(caught.exception.status_code, 504)
        self.assertEqual(caught.exception.detail["code"], "timeout")
        self.assertEqual(agent_loop.calls, [])
        release.assert_called_once_with()

    def test_known_retrieval_errors_return_503_and_release_slot(self):
        for error in (
            EmbeddingAPIError("provider failed"),
            VectorIndexNotReadyError("index failed"),
            EntityCatalogError("resolver failed"),
        ):
            with self.subTest(error=type(error).__name__):
                events = []
                retriever = FakeRetriever(events=events, error=error)
                agent_loop = FakeAgentLoop(events=events)
                with (
                    patch.object(chat, "try_acquire_llm_slot", return_value=True),
                    patch.object(chat, "release_llm_slot") as release,
                ):
                    with self.assertRaises(HTTPException) as caught:
                        chat.chat_stream(
                            self.payload,
                            self.request,
                            None,
                            retriever,
                            agent_loop,
                        )

                self.assertEqual(caught.exception.status_code, 503)
                self.assertEqual(
                    caught.exception.detail["code"],
                    "retrieval_unavailable",
                )
                self.assertEqual(agent_loop.calls, [])
                release.assert_called_once_with()

    def test_agent_errors_keep_existing_http_mapping_and_release_slot(self):
        request = httpx.Request("POST", "https://example.com/chat")
        cases = (
            (AgentDeadlineExceeded("expired"), 504, "timeout"),
            (APITimeoutError(request=request), 504, "timeout"),
            (APIConnectionError(request=request), 503, "provider_unavailable"),
            (status_error(400), 502, "provider_error"),
        )
        for error, status, code in cases:
            with self.subTest(error=type(error).__name__):
                events = []
                retriever = FakeRetriever(events=events)
                agent_loop = FakeAgentLoop(events=events, error=error)
                with (
                    patch.object(chat, "try_acquire_llm_slot", return_value=True),
                    patch.object(chat, "release_llm_slot") as release,
                ):
                    with self.assertRaises(HTTPException) as caught:
                        chat.chat_stream(
                            self.payload,
                            self.request,
                            None,
                            retriever,
                            agent_loop,
                        )

                self.assertEqual(caught.exception.status_code, status)
                self.assertEqual(caught.exception.detail["code"], code)
                release.assert_called_once_with()

    def test_unexpected_agent_error_propagates_and_releases_slot(self):
        events = []
        retriever = FakeRetriever(events=events)
        agent_loop = FakeAgentLoop(
            events=events,
            error=RuntimeError("private defect"),
        )

        with (
            patch.object(chat, "try_acquire_llm_slot", return_value=True),
            patch.object(chat, "release_llm_slot") as release,
        ):
            with self.assertRaises(RuntimeError):
                chat.chat_stream(
                    self.payload,
                    self.request,
                    None,
                    retriever,
                    agent_loop,
                )

        release.assert_called_once_with()

    def test_busy_slot_rejects_before_deadline_retrieval_or_agent(self):
        events = []
        retriever = FakeRetriever(events=events)
        agent_loop = FakeAgentLoop(events=events)

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
                    retriever,
                    agent_loop,
                )

        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(events, [])
        clock.assert_not_called()
        release.assert_not_called()


if __name__ == "__main__":
    unittest.main()
