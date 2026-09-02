import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from knowledge_pipeline.retrieval.models import (
    EmbeddingAPIError,
    EntityCatalogError,
    RetrievalError,
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


class FakeRetriever:
    def __init__(
        self,
        *,
        events: list[str],
        error: Exception | None = None,
        results: list[RetrievalResult] | None = None,
    ):
        self.events = events
        self.error = error
        self.results = results if results is not None else [retrieval_result()]
        self.queries: list[str] = []

    def retrieve(self, query: str):
        self.events.append("retrieve")
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.results


async def consume_response(response) -> str:
    parts: list[str] = []
    async for part in response.body_iterator:
        parts.append(part.decode("utf-8") if isinstance(part, bytes) else part)
    return "".join(parts)


def parse_rag_payload(content: str) -> dict:
    begin = "BEGIN_RAG_DATA\n"
    end = "\nEND_RAG_DATA"
    start = content.index(begin) + len(begin)
    finish = content.rindex(end)
    return json.loads(content[start:finish])


class ChatRouteRagOrchestrationTests(unittest.TestCase):
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

    def test_retrieves_latest_question_and_preserves_ndjson_stream(self):
        events: list[str] = []
        result = retrieval_result()
        retriever = FakeRetriever(events=events, results=[result])
        captured_messages: list[dict[str, str]] = []
        captured_results: list[RetrievalResult] = []
        real_context_builder = chat.build_retrieved_context

        def capture_context(results):
            captured_results.extend(results)
            return real_context_builder(results)

        def acquire() -> bool:
            events.append("acquire")
            return True

        def open_stream(messages):
            events.append("open")
            captured_messages.extend(messages)
            return []

        def release() -> None:
            events.append("release")

        with (
            patch.object(chat, "try_acquire_llm_slot", side_effect=acquire),
            patch.object(chat, "open_chat_stream", side_effect=open_stream),
            patch.object(chat, "release_llm_slot", side_effect=release),
            patch.object(
                chat,
                "build_retrieved_context",
                side_effect=capture_context,
            ),
            patch.object(chat, "log_request") as logged,
        ):
            try:
                response = chat.chat_stream(
                    self.payload,
                    self.request,
                    None,
                    retriever,
                )
            except TypeError:
                self.fail("chat_stream has no Retriever orchestration input")
            body = asyncio.run(consume_response(response))

        self.assertEqual(
            retriever.queries,
            ["它的防护等级是什么？"],
        )
        self.assertEqual(events, ["acquire", "retrieve", "open", "release"])
        self.assertEqual(len(captured_results), 1)
        self.assertIs(captured_results[0], result)
        self.assertEqual(captured_results[0].rank, 1)
        self.assertEqual(captured_results[0].score, -0.25)
        self.assertEqual(captured_results[0].match_origin, "exact_entity")
        self.assertEqual(captured_results[0].matched_entity_ids, ["product:hp780"])
        self.assertEqual(captured_results[0].chunk_id, "product:hp780:specifications")
        self.assertEqual(captured_results[0].parent_document_id, "product:hp780")
        self.assertEqual(captured_results[0].content_hash, "a" * 64)
        self.assertEqual(captured_results[0].metadata.product_id, "hp780")
        self.assertEqual(captured_results[0].source_url, "https://example.com/hp780/")
        self.assertEqual(
            captured_results[0].source_files,
            ["src/content/products/hp780.json"],
        )
        logged.assert_called_once_with(
            request_id="request-1",
            http_status=200,
            outcome="success",
            started_at=1.0,
        )
        self.assertEqual(response.media_type, "application/x-ndjson")
        self.assertEqual(
            [json.loads(line) for line in body.splitlines()],
            [{"type": "done"}],
        )
        self.assertEqual(
            captured_messages[1:3],
            [
                {"role": "user", "content": "介绍 HP780。"},
                {
                    "role": "assistant",
                    "content": "HP780 是一款对讲机。",
                },
            ],
        )
        self.assertEqual(
            parse_rag_payload(captured_messages[-1]["content"]),
            {
                "retrieved_context": [{
                    "type": "product",
                    "section": "技术参数",
                    "text": "# HP780\n\n防护等级：IP68",
                }],
                "user_question": "它的防护等级是什么？",
            },
        )

    def test_pre_stream_orchestration_error_releases_slot_without_generation(self):
        events: list[str] = []
        retriever = FakeRetriever(
            events=events,
            error=RuntimeError("context unavailable"),
        )

        def acquire() -> bool:
            events.append("acquire")
            return True

        def release() -> None:
            events.append("release")

        with (
            patch.object(chat, "try_acquire_llm_slot", side_effect=acquire),
            patch.object(chat, "release_llm_slot", side_effect=release) as released,
            patch.object(chat, "open_chat_stream") as open_stream,
        ):
            try:
                with self.assertRaises(RuntimeError):
                    chat.chat_stream(
                        self.payload,
                        self.request,
                        None,
                        retriever,
                    )
            except TypeError:
                self.fail("chat_stream has no Retriever orchestration input")

        self.assertEqual(events, ["acquire", "retrieve", "release"])
        released.assert_called_once_with()
        open_stream.assert_not_called()

    def test_known_retrieval_failures_return_503_without_generation(self):
        for retrieval_error in (
            EmbeddingAPIError("provider failed"),
            VectorIndexNotReadyError("index failed"),
            EntityCatalogError("entity resolver failed"),
        ):
            with self.subTest(error_type=type(retrieval_error).__name__):
                events: list[str] = []
                retriever = FakeRetriever(events=events, error=retrieval_error)

                def acquire() -> bool:
                    events.append("acquire")
                    return True

                def release() -> None:
                    events.append("release")

                with (
                    patch.object(chat, "try_acquire_llm_slot", side_effect=acquire),
                    patch.object(
                        chat,
                        "release_llm_slot",
                        side_effect=release,
                    ) as released,
                    patch.object(chat, "open_chat_stream") as open_stream,
                ):
                    try:
                        with self.assertRaises(HTTPException) as caught:
                            chat.chat_stream(
                                self.payload,
                                self.request,
                                None,
                                retriever,
                            )
                    except RetrievalError:
                        self.fail(
                            "known retrieval failure was not mapped to HTTP 503"
                        )

                self.assertEqual(caught.exception.status_code, 503)
                self.assertEqual(
                    caught.exception.detail,
                    {
                        "code": "retrieval_unavailable",
                        "message": "服务暂时不可用，请稍后再试。",
                        "internal_error": type(retrieval_error).__name__,
                    },
                )
                self.assertEqual(events, ["acquire", "retrieve", "release"])
                released.assert_called_once_with()
                open_stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
