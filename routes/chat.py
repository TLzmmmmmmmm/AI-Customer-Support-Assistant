import json
import time
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Request
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
from app_logging import log_request

router = APIRouter()

def encode_event(event: dict[str, object]) -> str:
    return json.dumps(
        event,
        ensure_ascii=False,
    ) + "\n"

def stream_events(
    stream,
    request_id: str,
    started_at: float,
) -> Iterator[str]:
    try:
        for content in iter_chat_content(stream):
            yield encode_event({
                "type": "delta",
                "content": content,
            })

        log_request(
            request_id=request_id,
            http_status=200,
            outcome="success",
            started_at=started_at,
        )

        yield encode_event({
            "type": "done",
        })

    except APITimeoutError:
        log_request(
            request_id=request_id,
            http_status=200,
            outcome="stream_timeout",
            started_at=started_at,
            error="APITimeoutError",
        )
        yield encode_event({
            "type": "error",
            "code": "timeout",
            "message": "AI service timed out",
        })

    except APIConnectionError:
        log_request(
            request_id=request_id,
            http_status=200,
            outcome="stream_connection_error",
            started_at=started_at,
            error="APIConnectionError",
        )
        yield encode_event({
            "type": "error",
            "code": "connection_error",
            "message": "AI service is unavailable",
        })

    except APIStatusError as error:
        log_request(
            request_id=request_id,
            http_status=200,
            outcome="provider_error",
            started_at=started_at,
            error=type(error).__name__,
        )

        yield encode_event({
            "type": "error",
            "code": "upstream_error",
            "message": "AI service returned an error",
        })

def stream_events_with_slot(
    stream,
    request_id: str,
    started_at: float,
) -> Iterator[str]:
    try:
        yield from stream_events(stream, request_id, started_at)
    finally:
        release_llm_slot()

@router.post("/api/chat-stream")
def chat_stream(
    payload: ChatRequest,
    request: Request,
    _: None = Depends(enforce_rate_limit),
):
    started_at = time.monotonic()
    request_id = request.state.request_id
    
    if not try_acquire_llm_slot():
        log_request(
            request_id=request_id,
            http_status=503,
            outcome="concurrency_limit",
            started_at=started_at,
            error=None,
        )

        raise HTTPException(
            status_code=503,
            detail="AI service is busy",
        )
    
    try:
        stream = open_chat_stream(payload.messages)

    except APITimeoutError:
        release_llm_slot()

        log_request(
            request_id=request_id,
            http_status=504,
            outcome="timeout",
            started_at=started_at,
            error="APITimeoutError",
        )
        
        raise HTTPException(
            status_code=504,
            detail="AI service timed out",
        )

    except APIConnectionError:
        release_llm_slot()

        log_request(
            request_id=request_id,
            http_status=503,
            outcome="provider_unavailable",
            started_at=started_at,
            error="APIConnectionError",
        )

        raise HTTPException(
            status_code=503,
            detail="AI service is unavailable",
        )

    except APIStatusError as error:
        release_llm_slot()

        log_request(
            request_id=request_id,
            http_status=502,
            outcome="provider_error",
            started_at=started_at,
            error=type(error).__name__,
        )

        raise HTTPException(
            status_code=502,
            detail="AI service returned an error",
        )

    return StreamingResponse(
        stream_events_with_slot(
            stream,
            request_id,
            started_at,
        ),
        media_type="application/x-ndjson",
    )

