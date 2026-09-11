import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import Protocol

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from knowledge_pipeline.retrieval import RetrievalError
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)
from rate_limit import enforce_rate_limit
from models import ChatMessage, ChatRequest
from concurrency import (
    release_llm_slot,
    try_acquire_llm_slot,
)
from app_logging import log_request
from agent import AgentDeadline, AgentDeadlineExceeded, SAFE_AGENT_ANSWER
from citation import render_answer
from config import AGENT_TIMEOUT_SECONDS
from routing import RouteExecutionResult, RouteTrace

router = APIRouter()


class _RouteRunner(Protocol):
    def run(
        self,
        messages: Sequence[ChatMessage],
        *,
        deadline: AgentDeadline,
    ) -> RouteExecutionResult:
        ...


def get_route_orchestrator(request: Request) -> _RouteRunner:
    return request.app.state.route_orchestrator


def encode_event(event: dict[str, object]) -> str:
    return json.dumps(
        event,
        ensure_ascii=False,
    ) + "\n"

def answer_events_with_slot(
    answer: str,
    request_id: str,
    started_at: float,
    trace: RouteTrace,
) -> Iterator[str]:
    try:
        yield encode_event({
            "type": "delta",
            "content": answer,
        })
        log_request(
            request_id=request_id,
            http_status=200,
            outcome="success",
            started_at=started_at,
            trace=trace,
        )
        yield encode_event({
            "type": "done",
        })
    finally:
        release_llm_slot()

@router.post("/api/chat-stream")
def chat_stream(
    payload: ChatRequest,
    request: Request,
    _: None = Depends(enforce_rate_limit),
    orchestrator: _RouteRunner = Depends(get_route_orchestrator),
):
    started_at = request.state.started_at
    request_id = request.state.request_id

    if not try_acquire_llm_slot():
        raise HTTPException(
            status_code=503,
            detail={
                "code": "concurrency_limit",
                "message": "当前咨询人数较多，请稍后再试。",
            },
        )
    
    try:
        deadline = AgentDeadline.start(
            AGENT_TIMEOUT_SECONDS,
            clock=time.monotonic,
        )
        deadline.ensure_active()
        route_result = orchestrator.run(
            payload.messages,
            deadline=deadline,
        )
        rendered = render_answer(
            route_result.answer,
            route_result.sources,
            fallback=SAFE_AGENT_ANSWER,
            language_hint=payload.messages[-1].content,
        )
        citation_status = (
            "degraded"
            if rendered.invalid_source_count
            else "rendered"
            if rendered.sources
            else "none"
        )
        trace = replace(
            route_result.trace,
            citation_count=len(rendered.sources),
            deduplicated_citation_count=rendered.deduplicated_count,
            invalid_source_count=rendered.invalid_source_count,
            citation_status=citation_status,
            answer_sanitized=rendered.answer_sanitized,
        )
        request.state.route_trace = trace

    except RetrievalError as error:
        _remember_error_trace(request, error)
        release_llm_slot()

        raise HTTPException(
            status_code=503,
            detail={
                "code": "retrieval_unavailable",
                "message": "服务暂时不可用，请稍后再试。",
                "internal_error": type(error).__name__,
            },
        )

    except (AgentDeadlineExceeded, APITimeoutError) as error:
        _remember_error_trace(request, error)
        release_llm_slot()

        raise HTTPException(
            status_code=504,
            detail={
                "code": "timeout",
                "message": "服务响应超时，请重新尝试。",
                "internal_error": type(error).__name__,
            },
        )

    except APIConnectionError as error:
        _remember_error_trace(request, error)
        release_llm_slot()

        raise HTTPException(
            status_code=503,
            detail={
                "code": "provider_unavailable",
                "message": "服务暂时不可用，请稍后再试。",
                "internal_error": type(error).__name__,
            },
        )

    except APIStatusError as error:
        _remember_error_trace(request, error)
        release_llm_slot()

        raise HTTPException(
            status_code=502,
            detail={
                "code": "provider_error",
                "message": "服务暂时出现异常，请稍后再试。",
                "internal_error": type(error).__name__,
            },
        )

    except Exception as error:
        _remember_error_trace(request, error)
        release_llm_slot()
        raise

    return StreamingResponse(
        answer_events_with_slot(
            rendered.text,
            request_id,
            started_at,
            trace,
        ),
        media_type="application/x-ndjson",
    )


def _remember_error_trace(request: Request, error: Exception) -> None:
    route = getattr(error, "route", None)
    failure_layer = getattr(error, "failure_layer", None)
    request.state.failure_layer = failure_layer
    if route is None:
        return
    tool_calls = tuple(getattr(error, "tool_calls", ()))
    retrieved_chunk_ids = tuple(getattr(error, "retrieved_chunk_ids", ()))
    request.state.route_trace = RouteTrace(
        route=route,
        tool_calls=tool_calls,
        tool_call_count=len(tool_calls),
        retrieved_chunk_ids=retrieved_chunk_ids,
        failure_layer=failure_layer,
    )

