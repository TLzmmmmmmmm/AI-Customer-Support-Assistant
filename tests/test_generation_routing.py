import importlib
import unittest
from types import SimpleNamespace

from agent import AgentDeadlineExceeded, SAFE_AGENT_ANSWER
from trace_models import FailureLayer


def completion(content, *, choices=1, finish_reason="stop", tool_calls=None):
    choice = SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=content, tool_calls=tool_calls),
    )
    return SimpleNamespace(choices=[choice] * choices)


def tool_call():
    return SimpleNamespace(
        type="function",
        id="call-1",
        function=SimpleNamespace(name="get_contact_info", arguments="{}"),
    )


class RecordingDeadline:
    def __init__(self, events, *, fail_on=None, error=None):
        self.events = events
        self.fail_on = fail_on
        self.error = error or AgentDeadlineExceeded("expired")
        self.checks = 0

    def ensure_active(self):
        self.checks += 1
        self.events.append("deadline")
        if self.checks == self.fail_on:
            raise self.error


class RecordingCompleteChat:
    def __init__(self, response, events):
        self.response = response
        self.events = events
        self.calls = []

    def __call__(self, provider_messages):
        self.events.append("provider")
        self.calls.append(provider_messages)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class GenerationRoutingTests(unittest.TestCase):
    def _generate_answer(self):
        try:
            module = importlib.import_module("routing.generation")
        except ModuleNotFoundError:
            self.fail("routing.generation module is missing")
        helper = getattr(module, "generate_answer", None)
        self.assertIsNotNone(helper)
        return helper

    def _generate(self, response):
        events = []
        provider = RecordingCompleteChat(response, events)
        result = self._generate_answer()(
            [{"role": "user", "content": "问题"}],
            deadline=RecordingDeadline(events),
            complete_chat=provider,
        )
        return result, events, provider

    def test_valid_completion_returns_content_without_failure(self):
        result, events, _ = self._generate(completion("有效回答"))

        self.assertEqual(result, ("有效回答", None))
        self.assertEqual(events, ["deadline", "provider", "deadline"])

    def test_normalization_failure_returns_safe_generation_failure(self):
        result, _, _ = self._generate(completion("unused", choices=0))

        self.assertEqual(
            result,
            (SAFE_AGENT_ANSWER, FailureLayer.GENERATION),
        )

    def test_unexpected_tool_call_returns_safe_generation_failure(self):
        result, _, _ = self._generate(
            completion(None, tool_calls=[tool_call()]),
        )

        self.assertEqual(
            result,
            (SAFE_AGENT_ANSWER, FailureLayer.GENERATION),
        )

    def test_none_content_returns_safe_generation_failure(self):
        result, _, _ = self._generate(completion(None))

        self.assertEqual(
            result,
            (SAFE_AGENT_ANSWER, FailureLayer.GENERATION),
        )

    def test_whitespace_content_returns_safe_generation_failure(self):
        result, _, _ = self._generate(completion("  \n\t"))

        self.assertEqual(
            result,
            (SAFE_AGENT_ANSWER, FailureLayer.GENERATION),
        )

    def test_deadline_failure_before_provider_propagates_without_annotation(self):
        events = []
        expected = AgentDeadlineExceeded("before provider")
        provider = RecordingCompleteChat(completion("unused"), events)

        with self.assertRaises(AgentDeadlineExceeded) as caught:
            self._generate_answer()(
                [{"role": "user", "content": "问题"}],
                deadline=RecordingDeadline(
                    events,
                    fail_on=1,
                    error=expected,
                ),
                complete_chat=provider,
            )

        self.assertIs(caught.exception, expected)
        self.assertFalse(hasattr(caught.exception, "failure_layer"))
        self.assertEqual(events, ["deadline"])
        self.assertEqual(provider.calls, [])

    def test_provider_exception_is_annotated_as_generation(self):
        events = []
        expected = RuntimeError("provider failed")

        with self.assertRaises(RuntimeError) as caught:
            self._generate_answer()(
                [{"role": "user", "content": "问题"}],
                deadline=RecordingDeadline(events),
                complete_chat=RecordingCompleteChat(expected, events),
            )

        self.assertEqual(caught.exception.failure_layer, FailureLayer.GENERATION)
        self.assertEqual(events, ["deadline", "provider"])

    def test_provider_exception_is_reraised_unchanged(self):
        events = []
        expected = RuntimeError("provider failed")

        with self.assertRaises(RuntimeError) as caught:
            self._generate_answer()(
                [{"role": "user", "content": "问题"}],
                deadline=RecordingDeadline(events),
                complete_chat=RecordingCompleteChat(expected, events),
            )

        self.assertIs(caught.exception, expected)

    def test_deadline_failure_after_provider_propagates_without_annotation(self):
        events = []
        expected = AgentDeadlineExceeded("after provider")

        with self.assertRaises(AgentDeadlineExceeded) as caught:
            self._generate_answer()(
                [{"role": "user", "content": "问题"}],
                deadline=RecordingDeadline(
                    events,
                    fail_on=2,
                    error=expected,
                ),
                complete_chat=RecordingCompleteChat(
                    completion("unused"),
                    events,
                ),
            )

        self.assertIs(caught.exception, expected)
        self.assertFalse(hasattr(caught.exception, "failure_layer"))
        self.assertEqual(events, ["deadline", "provider", "deadline"])

    def test_provider_receives_messages_as_its_only_argument(self):
        events = []
        provider_messages = [{"role": "user", "content": "问题"}]
        provider = RecordingCompleteChat(completion("回答"), events)

        self._generate_answer()(
            provider_messages,
            deadline=RecordingDeadline(events),
            complete_chat=provider,
        )

        self.assertEqual(provider.calls, [provider_messages])
        self.assertIs(provider.calls[0], provider_messages)


if __name__ == "__main__":
    unittest.main()
