import time
from collections.abc import Iterator, Mapping, Sequence

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_TIMEOUT_SECONDS,
    DEEPSEEK_MAX_RETRIES,
    LLM_APP_MAX_RETRIES,
    LLM_RETRY_DELAY_SECONDS,
)
from trace_models import (
    add_request_duration,
    record_model_response,
    record_model_usage,
)


client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    timeout=DEEPSEEK_TIMEOUT_SECONDS,
    max_retries=DEEPSEEK_MAX_RETRIES,
)


class IncompleteModelStreamError(RuntimeError):
    pass

def is_retryable_status(error: APIStatusError) -> bool:
    return error.status_code == 429 or error.status_code >= 500


def _copy_messages(
    messages: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [dict(message) for message in messages]


def complete_chat(
    messages: Sequence[Mapping[str, object]],
    *,
    tools: Sequence[Mapping[str, object]] | None = None,
    max_tokens: int | None = None,
):
    started_at = time.monotonic()
    provider_messages = _copy_messages(messages)
    provider_tools = None if tools is None else [dict(tool) for tool in tools]
    attempt = 0

    try:
        while True:
            try:
                request: dict[str, object] = {
                    "model": DEEPSEEK_MODEL,
                    "messages": provider_messages,
                    "stream": False,
                    "extra_body": {
                        "thinking": {
                            "type": "disabled",
                        },
                    },
                }
                if provider_tools is not None:
                    request["tools"] = provider_tools
                if max_tokens is not None:
                    request["max_tokens"] = max_tokens
                completion = client.chat.completions.create(**request)
                record_model_response(completion)
                return completion

            except APITimeoutError:
                raise

            except APIConnectionError:
                if attempt >= LLM_APP_MAX_RETRIES:
                    raise

            except APIStatusError as error:
                if (
                    not is_retryable_status(error)
                    or attempt >= LLM_APP_MAX_RETRIES
                ):
                    raise

            attempt += 1
            time.sleep(LLM_RETRY_DELAY_SECONDS)
    finally:
        add_request_duration(
            "model_latency_ms",
            (time.monotonic() - started_at) * 1000,
        )


def open_chat_stream(messages: Sequence[Mapping[str, str]]):
    provider_messages = _copy_messages(messages)

    attempt = 0

    while True:
        try:
            return client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=provider_messages,
                stream=True,
                extra_body={
                    "thinking": {
                        "type": "disabled",
                    }
                },
            )

        except APITimeoutError:
            raise

        except APIConnectionError:
            if attempt >= LLM_APP_MAX_RETRIES:
                raise

        except APIStatusError as error:
            if (
                not is_retryable_status(error)
                or attempt >= LLM_APP_MAX_RETRIES
            ):
                raise

        attempt += 1
        time.sleep(LLM_RETRY_DELAY_SECONDS)

def iter_chat_content(stream) -> Iterator[str]:
    for chunk in stream:
        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content

        if content:
            yield content

def stream_chat(
    messages: Sequence[Mapping[str, str]],
) -> Iterator[str]:
    started_at = time.monotonic()
    provider_messages = _copy_messages(messages)

    attempt = 0

    try:
        while True:
            has_yielded_content = False
            final_usage = None
            finish_reason = None

            try:
                stream = client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=provider_messages,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body={
                        "thinking": {
                            "type": "disabled",
                        }
                    },
                )

                for chunk in stream:
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        final_usage = usage
                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    content = choice.delta.content

                    if content:
                        has_yielded_content = True
                        yield content

                    if getattr(choice, "finish_reason", None) is not None:
                        finish_reason = choice.finish_reason

                record_model_usage(final_usage)
                if finish_reason != "stop" and has_yielded_content:
                    raise IncompleteModelStreamError(
                        "provider stream ended without a complete answer"
                    )
                return

            except APITimeoutError:
                raise

            except APIConnectionError:
                if has_yielded_content or attempt >= LLM_APP_MAX_RETRIES:
                    raise

            except APIStatusError as error:
                if (
                    has_yielded_content
                    or not is_retryable_status(error)
                    or attempt >= LLM_APP_MAX_RETRIES
                ):
                    raise

            attempt += 1
            time.sleep(LLM_RETRY_DELAY_SECONDS)
    finally:
        add_request_duration(
            "model_latency_ms",
            (time.monotonic() - started_at) * 1000,
        )
