from collections.abc import Iterator

from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
)
from models import ChatMessage
from prompts import SYSTEM_PROMPT


client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)


def stream_chat(messages: list[ChatMessage]) -> Iterator[str]:
    conversation = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in messages
    ]

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
            yield content