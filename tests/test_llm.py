import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx2 as httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError

from services import llm
from trace_models import (
    bind_request_state,
    initialize_request_trace,
    request_trace_fields,
    reset_request_state,
)


PROVIDER_MESSAGES = [
    {"role": "system", "content": "应用程序控制的系统规则"},
    {"role": "user", "content": "介绍 HP780。"},
    {"role": "assistant", "content": "HP780 是一款对讲机。"},
    {
        "role": "user",
        "content": (
            "BEGIN_RAG_DATA\n"
            '{"retrieved_context":[],"user_question":"参数？"}\n'
            "END_RAG_DATA"
        ),
    },
]


def status_error(status_code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://example.com/chat")
    return APIStatusError(
        "provider error",
        response=httpx.Response(status_code, request=request),
        body=None,
    )


class LlmProviderMessageTests(unittest.TestCase):
    def _assert_request(self, create, *, usage_stream: bool = False) -> None:
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["model"], llm.DEEPSEEK_MODEL)
        self.assertEqual(kwargs["messages"], PROVIDER_MESSAGES)
        self.assertTrue(kwargs["stream"])
        if usage_stream:
            self.assertEqual(
                kwargs["stream_options"],
                {"include_usage": True},
            )
        else:
            self.assertNotIn("stream_options", kwargs)
        self.assertEqual(
            kwargs["extra_body"],
            {"thinking": {"type": "disabled"}},
        )

    def test_open_chat_stream_sends_provider_messages_unchanged(self):
        sentinel = object()
        with patch.object(
            llm.client.chat.completions,
            "create",
            return_value=sentinel,
        ) as create:
            try:
                result = llm.open_chat_stream(PROVIDER_MESSAGES)
            except (AttributeError, TypeError):
                self.fail("open_chat_stream does not accept provider messages")

        self.assertIs(result, sentinel)
        self._assert_request(create)

    def test_stream_chat_retains_provider_messages_and_content_streaming(self):
        usage = SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=3,
            prompt_cache_hit_tokens=8,
            prompt_cache_miss_tokens=4,
        )
        stream = [
            SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content="第一段"),
            )], usage=None),
            SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content="第二段"),
            )], usage=None),
            SimpleNamespace(choices=[], usage=usage),
        ]
        state = SimpleNamespace()
        initialize_request_trace(state)
        telemetry_token = bind_request_state(state)
        try:
            with (
                patch.object(
                    llm.client.chat.completions,
                    "create",
                    return_value=stream,
                ) as create,
                patch.object(llm.time, "monotonic", side_effect=[1.0, 1.25]),
            ):
                content = list(llm.stream_chat(PROVIDER_MESSAGES))
        finally:
            reset_request_state(telemetry_token)

        self.assertEqual(content, ["第一段", "第二段"])
        self._assert_request(create, usage_stream=True)
        telemetry = request_trace_fields(state)
        self.assertEqual(telemetry["input_tokens"], 12)
        self.assertEqual(telemetry["output_tokens"], 3)
        self.assertEqual(telemetry["prompt_cache_hit_tokens"], 8)
        self.assertEqual(telemetry["prompt_cache_miss_tokens"], 4)
        self.assertEqual(telemetry["model_latency_ms"], 250.0)

    def test_stream_chat_retries_connection_status_failures_before_content(self):
        retryable_errors = (
            APIConnectionError(
                request=httpx.Request("POST", "https://example.com/chat"),
            ),
            status_error(429),
            status_error(500),
        )
        for error in retryable_errors:
            with self.subTest(error=type(error).__name__):
                usage = SimpleNamespace(
                    prompt_tokens=1,
                    completion_tokens=1,
                    prompt_cache_hit_tokens=0,
                    prompt_cache_miss_tokens=1,
                )
                successful = [SimpleNamespace(choices=[], usage=usage)]
                with (
                    patch.object(
                        llm.client.chat.completions,
                        "create",
                        side_effect=[error, successful],
                    ) as create,
                    patch.object(llm.time, "sleep") as sleep,
                ):
                    self.assertEqual(list(llm.stream_chat(PROVIDER_MESSAGES)), [])

                self.assertEqual(create.call_count, 2)
                sleep.assert_called_once_with(llm.LLM_RETRY_DELAY_SECONDS)

    def test_stream_chat_does_not_retry_or_duplicate_after_content(self):
        error = APIConnectionError(
            request=httpx.Request("POST", "https://example.com/chat"),
        )

        def interrupted():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="唯一段"))],
                usage=None,
            )
            raise error

        with (
            patch.object(
                llm.client.chat.completions,
                "create",
                return_value=interrupted(),
            ) as create,
            patch.object(llm.time, "sleep") as sleep,
        ):
            stream = llm.stream_chat(PROVIDER_MESSAGES)
            self.assertEqual(next(stream), "唯一段")
            with self.assertRaises(APIConnectionError):
                next(stream)

        self.assertEqual(create.call_count, 1)
        sleep.assert_not_called()

    def test_stream_chat_timeout_is_never_retried(self):
        error = APITimeoutError(
            request=httpx.Request("POST", "https://example.com/chat"),
        )
        with (
            patch.object(
                llm.client.chat.completions,
                "create",
                side_effect=error,
            ) as create,
            patch.object(llm.time, "sleep") as sleep,
        ):
            with self.assertRaises(APITimeoutError):
                list(llm.stream_chat(PROVIDER_MESSAGES))

        self.assertEqual(create.call_count, 1)
        sleep.assert_not_called()


class LlmCompleteChatTests(unittest.TestCase):
    def test_complete_chat_forwards_non_null_max_tokens(self):
        with patch.object(
            llm.client.chat.completions,
            "create",
            return_value=SimpleNamespace(usage=None),
        ) as create:
            llm.complete_chat(PROVIDER_MESSAGES, max_tokens=16)

        self.assertEqual(create.call_args.kwargs["max_tokens"], 16)

    def test_complete_chat_omits_null_max_tokens(self):
        with patch.object(
            llm.client.chat.completions,
            "create",
            return_value=SimpleNamespace(usage=None),
        ) as create:
            llm.complete_chat(PROVIDER_MESSAGES)

        self.assertNotIn("max_tokens", create.call_args.kwargs)

    def test_complete_chat_records_provider_cache_usage(self):
        state = SimpleNamespace()
        initialize_request_trace(state)
        completion = SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=3,
            prompt_cache_hit_tokens=8,
            prompt_cache_miss_tokens=4,
        ))

        telemetry_token = bind_request_state(state)
        try:
            with patch.object(
                llm.client.chat.completions,
                "create",
                return_value=completion,
            ):
                llm.complete_chat(PROVIDER_MESSAGES)
        finally:
            reset_request_state(telemetry_token)

        telemetry = request_trace_fields(state)
        self.assertEqual(telemetry["prompt_cache_hit_tokens"], 8)
        self.assertEqual(telemetry["prompt_cache_miss_tokens"], 4)

    def test_missing_usage_nulls_totals_after_prior_usage(self):
        state = SimpleNamespace()
        initialize_request_trace(state)
        first = SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=8,
            completion_tokens=2,
        ))
        second = SimpleNamespace(usage=None)

        telemetry_token = bind_request_state(state)
        try:
            with patch.object(
                llm.client.chat.completions,
                "create",
                side_effect=[first, second],
            ):
                llm.complete_chat(PROVIDER_MESSAGES)
                llm.complete_chat(PROVIDER_MESSAGES)
        finally:
            reset_request_state(telemetry_token)

        telemetry = request_trace_fields(state)
        self.assertIsNone(telemetry["input_tokens"])
        self.assertIsNone(telemetry["output_tokens"])
        self.assertIsNotNone(telemetry["model_latency_ms"])

    def test_complete_chat_sends_native_tools_without_mutating_inputs(self):
        messages = [dict(message) for message in PROVIDER_MESSAGES]
        tools = [{
            "type": "function",
            "function": {
                "name": "get_contact_info",
                "description": "Get contact channels.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        }]
        expected_messages = [dict(message) for message in messages]
        expected_tools = [dict(tool) for tool in tools]
        response = object()

        with patch.object(
            llm.client.chat.completions,
            "create",
            return_value=response,
        ) as create:
            result = llm.complete_chat(messages, tools=tools)

        self.assertIs(result, response)
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["messages"], expected_messages)
        self.assertEqual(kwargs["tools"], expected_tools)
        self.assertIsNot(kwargs["messages"], messages)
        self.assertIsNot(kwargs["messages"][0], messages[0])
        self.assertIsNot(kwargs["tools"], tools)
        self.assertIsNot(kwargs["tools"][0], tools[0])
        self.assertFalse(kwargs["stream"])
        self.assertEqual(
            kwargs["extra_body"],
            {"thinking": {"type": "disabled"}},
        )
        self.assertEqual(messages, expected_messages)
        self.assertEqual(tools, expected_tools)

    def test_complete_chat_omits_tools_for_finalization(self):
        with patch.object(
            llm.client.chat.completions,
            "create",
            return_value=object(),
        ) as create:
            llm.complete_chat(PROVIDER_MESSAGES)

        self.assertNotIn("tools", create.call_args.kwargs)
        self.assertFalse(create.call_args.kwargs["stream"])

    def test_complete_chat_timeout_is_not_retried(self):
        error = APITimeoutError(
            request=httpx.Request("POST", "https://example.com/chat"),
        )
        with (
            patch.object(
                llm.client.chat.completions,
                "create",
                side_effect=error,
            ) as create,
            patch.object(llm.time, "sleep") as sleep,
        ):
            with self.assertRaises(APITimeoutError):
                llm.complete_chat(PROVIDER_MESSAGES)

        self.assertEqual(create.call_count, 1)
        sleep.assert_not_called()

    def test_complete_chat_connection_error_uses_existing_retry_limit(self):
        error = APIConnectionError(
            request=httpx.Request("POST", "https://example.com/chat"),
        )
        with (
            patch.object(
                llm.client.chat.completions,
                "create",
                side_effect=error,
            ) as create,
            patch.object(llm.time, "sleep") as sleep,
        ):
            with self.assertRaises(APIConnectionError):
                llm.complete_chat(PROVIDER_MESSAGES)

        self.assertEqual(create.call_count, llm.LLM_APP_MAX_RETRIES + 1)
        self.assertEqual(sleep.call_count, llm.LLM_APP_MAX_RETRIES)

    def test_complete_chat_retries_429_and_server_errors(self):
        for status_code in (429, 500):
            with self.subTest(status_code=status_code):
                response = object()
                with (
                    patch.object(
                        llm.client.chat.completions,
                        "create",
                        side_effect=[status_error(status_code), response],
                    ) as create,
                    patch.object(llm.time, "sleep") as sleep,
                ):
                    result = llm.complete_chat(PROVIDER_MESSAGES)

                self.assertIs(result, response)
                self.assertEqual(create.call_count, 2)
                sleep.assert_called_once_with(llm.LLM_RETRY_DELAY_SECONDS)

    def test_complete_chat_does_not_retry_non_retryable_status(self):
        error = status_error(400)
        with (
            patch.object(
                llm.client.chat.completions,
                "create",
                side_effect=error,
            ) as create,
            patch.object(llm.time, "sleep") as sleep,
        ):
            with self.assertRaises(APIStatusError):
                llm.complete_chat(PROVIDER_MESSAGES)

        self.assertEqual(create.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
