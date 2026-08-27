import json

from collections.abc import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)

from models import ChatRequest
from services.llm import (
    iter_chat_content,
    open_chat_stream,
)


router = APIRouter()

def encode_event(event: dict[str, object]) -> str:
    return json.dumps(
        event,
        ensure_ascii=False,
    ) + "\n"

def stream_events(stream) -> Iterator[str]:
    try:
        for content in iter_chat_content(stream):
            yield encode_event({
                "type": "delta",
                "content": content,
            })

        yield encode_event({
            "type": "done",
        })

    except APITimeoutError:
        yield encode_event({
            "type": "error",
            "code": "timeout",
            "message": "AI service timed out",
        })

    except APIConnectionError:
        yield encode_event({
            "type": "error",
            "code": "connection_error",
            "message": "AI service is unavailable",
        })

    except APIStatusError:
        yield encode_event({
            "type": "error",
            "code": "upstream_error",
            "message": "AI service returned an error",
        })

@router.post("/api/chat-stream")
def chat_stream(request: ChatRequest):
    try:
        stream = open_chat_stream(request.messages)

    except APITimeoutError:
        raise HTTPException(
            status_code=504,
            detail="AI service timed out",
        )

    except APIConnectionError:
        raise HTTPException(
            status_code=503,
            detail="AI service is unavailable",
        )

    except APIStatusError:
        raise HTTPException(
            status_code=502,
            detail="AI service returned an error",
        )

    return StreamingResponse(
        stream_events(stream),
        media_type="application/x-ndjson",
    )