import importlib
import unittest
from types import SimpleNamespace

from knowledge_pipeline.models import SourceRef
from models import ChatMessage


class RecordingDeadline:
    def __init__(self, events):
        self.events = events

    def ensure_active(self):
        self.events.append("deadline")


class RecordingRetriever:
    def __init__(self, results, events, error=None):
        self.results = results
        self.events = events
        self.error = error

    def retrieve(self, query):
        self.events.append(("retrieve", query))
        if self.error is not None:
            raise self.error
        return self.results


class KnowledgeRoutingTests(unittest.TestCase):
    def _module(self):
        try:
            return importlib.import_module("routing.knowledge")
        except ModuleNotFoundError:
            self.fail("routing.knowledge module is missing")

    def test_retrieves_latest_message_between_two_deadline_checks(self):
        events = []
        expected = [SimpleNamespace(chunk_id="knowledge:one")]
        messages = (
            ChatMessage(role="user", content="先前问题"),
            ChatMessage(role="assistant", content="先前回答"),
            ChatMessage(role="user", content="最新原始问题"),
        )

        results = self._module().retrieve_knowledge(
            messages,
            deadline=RecordingDeadline(events),
            retriever=RecordingRetriever(expected, events),
        )

        self.assertIs(results, expected)
        self.assertEqual(events, [
            "deadline",
            ("retrieve", "最新原始问题"),
            "deadline",
        ])

    def test_retrieval_failure_propagates_without_annotation(self):
        events = []
        expected = RuntimeError("retrieval unavailable")

        with self.assertRaises(RuntimeError) as caught:
            self._module().retrieve_knowledge(
                (ChatMessage(role="user", content="问题"),),
                deadline=RecordingDeadline(events),
                retriever=RecordingRetriever([], events, error=expected),
            )

        self.assertIs(caught.exception, expected)
        self.assertFalse(hasattr(caught.exception, "failure_layer"))
        self.assertEqual(events, ["deadline", ("retrieve", "问题")])

    def test_sources_are_flattened_in_hit_and_source_order(self):
        first = SourceRef(title="A", url="https://example.com/a")
        second = SourceRef(title="B", url="https://example.com/b")
        results = [
            SimpleNamespace(sources=[first, second]),
            SimpleNamespace(sources=[first]),
        ]

        sources = self._module().retrieval_sources(results)

        self.assertEqual(sources, (first, second, first))

    def test_zero_hits_return_an_empty_list_and_empty_sources(self):
        events = []
        results = self._module().retrieve_knowledge(
            (ChatMessage(role="user", content="没有命中"),),
            deadline=RecordingDeadline(events),
            retriever=RecordingRetriever([], events),
        )

        self.assertEqual(results, [])
        self.assertEqual(self._module().retrieval_sources(results), ())
        self.assertEqual(events, [
            "deadline",
            ("retrieve", "没有命中"),
            "deadline",
        ])


if __name__ == "__main__":
    unittest.main()
