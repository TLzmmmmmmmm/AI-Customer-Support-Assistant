import json

from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)
from rate_limit import enforce_rate_limit
from models import ChatRequest
from services.llm import (
    iter_chat_content,
    open_chat_stream,
)
from concurrency import (
    release_llm_slot,
    try_acquire_llm_slot,
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

def stream_events_with_slot(stream) -> Iterator[str]:
    try:
        yield from stream_events(stream)
    finally:
        release_llm_slot()

@router.post("/api/chat-stream")
def chat_stream(
    payload: ChatRequest,
    _: None = Depends(enforce_rate_limit),
):
    if not try_acquire_llm_slot():
        raise HTTPException(
            status_code=503,
            detail="AI service is busy",
        )
    
    try:
        stream = open_chat_stream(payload.messages)

    except APITimeoutError:
        release_llm_slot()
        raise HTTPException(
            status_code=504,
            detail="AI service timed out",
        )

    except APIConnectionError:
        release_llm_slot()
        raise HTTPException(
            status_code=503,
            detail="AI service is unavailable",
        )

    except APIStatusError:
        release_llm_slot()
        raise HTTPException(
            status_code=502,
            detail="AI service returned an error",
        )

    return StreamingResponse(
        stream_events_with_slot(stream),
        media_type="application/x-ndjson",
    )

