from collections.abc import Iterator

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)
import time

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_TIMEOUT_SECONDS,
    DEEPSEEK_MAX_RETRIES,
    LLM_APP_MAX_RETRIES,
    LLM_RETRY_DELAY_SECONDS,
)
from models import ChatMessage
from prompts import SYSTEM_PROMPT


client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    timeout=DEEPSEEK_TIMEOUT_SECONDS,
    max_retries=DEEPSEEK_MAX_RETRIES,
)

def is_retryable_status(error: APIStatusError) -> bool:
    return error.status_code == 429 or error.status_code >= 500

def stream_chat(messages: list[ChatMessage]) -> Iterator[str]:
    conversation = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in messages
    ]

    attempt = 0

    while True:
        has_yielded_content = False

        try:
            stream = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    *conversation,
                ],
                stream=True,
                extra_body={
                    "thinking": {
                        "type": "disabled",
                    }
                },
            )

            for chunk in stream:
                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content

                if content:
                    has_yielded_content = True
                    yield content

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