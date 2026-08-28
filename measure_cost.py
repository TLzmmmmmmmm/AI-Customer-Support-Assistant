from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
)
from prompts import SYSTEM_PROMPT


client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)


messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    },
    {
        "role": "user",
        "content": "用十句话介绍你自己",
    },
]


stream = client.chat.completions.create(
    model=DEEPSEEK_MODEL,
    messages=messages,
    stream=True,
    stream_options={
        "include_usage": True,
    },
    extra_body={
        "thinking": {
            "type": "disabled",
        }
    },
)


usage = None

for chunk in stream:
    if chunk.usage is not None:
        usage = chunk.usage

    if not chunk.choices:
        continue

    content = chunk.choices[0].delta.content

    if content:
        print(content, end="", flush=True)


print("\n\n--- Usage ---")

if usage is None:
    print("Usage data unavailable")
else:
    print(
        "Prompt tokens:",
        usage.prompt_tokens,
    )

    print(
        "Cache hit tokens:",
        usage.prompt_cache_hit_tokens,
    )

    print(
        "Cache miss tokens:",
        usage.prompt_cache_miss_tokens,
    )

    print(
        "Completion tokens:",
        usage.completion_tokens,
    )

    print(
        "Total tokens:",
        usage.total_tokens,
    )