import json
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import replace
from typing import Protocol

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from knowledge_pipeline.retrieval import RetrievalError
from openai import APIConnectionError, APIStatusError, APITimeoutError
from starlette.concurrency import run_in_threadpool

from agent import AgentDeadline, AgentDeadlineExceeded, SAFE_AGENT_ANSWER
from app_logging import log_request
from citation import CitationRenderResult, render_answer
from concurrency import release_llm_slot, try_acquire_llm_slot
from config import AGENT_TIMEOUT_SECONDS
from models import ChatMessage, ChatRequest
from rate_limit import enforce_rate_limit
from routing import RouteExecutionResult, RouteTrace
from trace_models import (
    bind_request_state,
    request_trace_fields,
    reset_request_state,
)


router = APIRouter()


class _RouteRunner(Protocol):
    def run(
        self,
        messages: Sequence[ChatMessage],
        *,
        deadline: AgentDeadline,
    ) -> RouteExecutionResult:
        ...

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        deadline: AgentDeadline,
    ) -> Iterator[str | RouteExecutionResult]:
        ...


def get_route_orchestrator(request: Request) -> _RouteRunner:
    return request.app.state.route_orchestrator


def encode_event(event: dict[str, object]) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


def _advance_with_request_state(
    iterator: Iterator[str | RouteExecutionResult],
    state: object,
) -> str | RouteExecutionResult:
    token = bind_request_state(state)
    try:
        return next(iterator)
    finally:
        reset_request_state(token)


def _close_with_request_state(iterator: Iterator[object], state: object) -> None:
    token = bind_request_state(state)
    try:
        iterator.close()
    finally:
        reset_request_state(token)


def _prepare_result(
    route_result: RouteExecutionResult,
    *,
    request_state: object,
    language_hint: str,
) -> tuple[CitationRenderResult, RouteTrace]:
    request_state.route_trace = route_result.trace
    rendered = render_answer(
        route_result.answer,
        route_result.sources,
        fallback=SAFE_AGENT_ANSWER,
        language_hint=language_hint,
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
    request_state.route_trace = trace
    return rendered, trace


def _stream_error(error: Exception) -> tuple[str, str]:
    contract = _error_contract(error)
    if contract is None:
        return "internal_error", "服务暂时出现异常，请稍后再试。"
    _, code, message = contract
    return code, message


def _error_contract(error: Exception) -> tuple[int, str, str] | None:
    if isinstance(error, RetrievalError):
        return (
            503,
            "retrieval_unavailable",
            "服务暂时不可用，请稍后再试。",
        )
    if isinstance(error, (AgentDeadlineExceeded, APITimeoutError)):
        return 504, "timeout", "服务响应超时，请重新尝试。"
    if isinstance(error, APIConnectionError):
        return 503, "provider_unavailable", "服务暂时不可用，请稍后再试。"
    if isinstance(error, APIStatusError):
        return 502, "provider_error", "服务暂时出现异常，请稍后再试。"
    return None


def _prefetch_http_exception(error: Exception) -> HTTPException | None:
    contract = _error_contract(error)
    if contract is None:
        return None
    status, code, message = contract
    return HTTPException(
        status_code=status,
        detail={
            "code": code,
            "message": message,
            "internal_error": type(error).__name__,
        },
    )


class _ResponseLifecycle:
    def __init__(
        self,
        *,
        route_iterator: Iterator[str | RouteExecutionResult],
        request_id: str,
        started_at: float,
        request_state: object,
    ) -> None:
        self.route_iterator = route_iterator
        self.request_id = request_id
        self.started_at = started_at
        self.request_state = request_state
        self.trace: RouteTrace | None = getattr(
            request_state,
            "route_trace",
            None,
        )
        self.first_delta_at: float | None = None
        self.done_yielded = False
        self.summary_logged = False
        self.closed = False
        if not hasattr(request_state, "first_delta_latency_ms"):
            request_state.first_delta_latency_ms = None
        if not hasattr(request_state, "buffering_saved_ms"):
            request_state.buffering_saved_ms = None

    def set_trace(self, trace: RouteTrace | None) -> None:
        self.trace = trace
        self.request_state.route_trace = trace

    def mark_first_delta(self) -> None:
        if self.first_delta_at is not None:
            return
        self.first_delta_at = time.monotonic()
        latency_ms = (self.first_delta_at - self.started_at) * 1000
        self.request_state.first_delta_latency_ms = latency_ms
        if self.trace is not None:
            self.set_trace(replace(
                self.trace,
                first_delta_latency_ms=latency_ms,
            ))

    def mark_done(self) -> None:
        self.done_yielded = True

    def log_failure(self, error: Exception, *, code: str) -> None:
        completed_at = time.monotonic()
        failure_layer = getattr(error, "failure_layer", None)
        self.request_state.failure_code = code
        self.request_state.failure_layer = failure_layer
        self.request_state.buffering_saved_ms = None
        if self.trace is not None:
            self.set_trace(replace(
                self.trace,
                failure_code=code,
                buffering_saved_ms=None,
            ))
        self.summary_logged = True
        log_request(
            request_id=self.request_id,
            http_status=200,
            outcome=code,
            started_at=self.started_at,
            error=type(error).__name__,
            trace=self.trace,
            failure_layer=failure_layer,
            failure_code=code,
            request_state=self.request_state,
            completed_at=completed_at,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        completed_at = time.monotonic()
        try:
            try:
                _close_with_request_state(
                    self.route_iterator,
                    self.request_state,
                )
            except Exception:
                pass
            if self.summary_logged:
                return
            self.summary_logged = True
            if self.done_yielded:
                if self.first_delta_at is not None:
                    buffering_saved_ms = (
                        completed_at - self.first_delta_at
                    ) * 1000
                    self.request_state.buffering_saved_ms = buffering_saved_ms
                    if self.trace is not None:
                        self.set_trace(replace(
                            self.trace,
                            buffering_saved_ms=buffering_saved_ms,
                        ))
                log_request(
                    request_id=self.request_id,
                    http_status=200,
                    outcome="success",
                    started_at=self.started_at,
                    trace=self.trace,
                    completed_at=completed_at,
                )
                return

            code = "stream_interrupted"
            self.request_state.failure_code = code
            self.request_state.failure_layer = None
            self.request_state.buffering_saved_ms = None
            if self.trace is not None:
                self.set_trace(replace(
                    self.trace,
                    failure_code=code,
                    buffering_saved_ms=None,
                ))
            log_request(
                request_id=self.request_id,
                http_status=200,
                outcome=code,
                started_at=self.started_at,
                trace=self.trace,
                failure_layer=None,
                failure_code=code,
                request_state=self.request_state,
                completed_at=completed_at,
            )
        finally:
            release_llm_slot()


_ITERATOR_DONE = object()


def _next_or_done(iterator: Iterator[str]) -> str | object:
    try:
        return next(iterator)
    except StopIteration:
        return _ITERATOR_DONE


class _AsyncClosingIterator(AsyncIterator[str]):
    def __init__(
        self,
        iterator: Iterator[str],
        lifecycle: _ResponseLifecycle,
    ) -> None:
        self._iterator = iterator
        self._lifecycle = lifecycle
        self._closed = False

    def __aiter__(self) -> "_AsyncClosingIterator":
        return self

    async def __anext__(self) -> str:
        if self._closed:
            raise StopAsyncIteration
        try:
            item = await run_in_threadpool(_next_or_done, self._iterator)
        except BaseException:
            await self.aclose()
            raise
        if item is _ITERATOR_DONE:
            await self.aclose()
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True

        def close() -> None:
            try:
                self._iterator.close()
            finally:
                self._lifecycle.close()

        await run_in_threadpool(close)


class _OwnedStreamingResponse(StreamingResponse):
    async def stream_response(self, send) -> None:
        try:
            await super().stream_response(send)
        finally:
            await self.body_iterator.aclose()


def answer_events_with_slot(
    first_item: str | RouteExecutionResult,
    route_iterator: Iterator[str | RouteExecutionResult],
    prepared_result: tuple[CitationRenderResult, RouteTrace] | None,
    request: Request,
    language_hint: str,
    lifecycle: _ResponseLifecycle,
) -> Iterator[str]:
    visible_deltas: list[str] = []
    try:
        if prepared_result is None:
            current = first_item
            while isinstance(current, str):
                lifecycle.mark_first_delta()
                visible_deltas.append(current)
                yield encode_event({"type": "delta", "content": current})
                current = _advance_with_request_state(
                    route_iterator,
                    request.state,
                )
            rendered, trace = _prepare_result(
                current,
                request_state=request.state,
                language_hint=language_hint,
            )
            lifecycle.set_trace(trace)
            if "".join(visible_deltas) != rendered.text:
                raise RuntimeError("streamed answer differs from rendered answer")
        else:
            rendered, trace = prepared_result
            lifecycle.set_trace(trace)

        if not visible_deltas:
            lifecycle.mark_first_delta()
            yield encode_event({"type": "delta", "content": rendered.text})
        if rendered.sources and rendered.citation_heading:
            yield encode_event({
                "type": "citations",
                "heading": rendered.citation_heading,
                "items": [
                    {"title": source.title, "url": source.url}
                    for source in rendered.sources
                ],
            })
        yield encode_event({"type": "done"})
        lifecycle.mark_done()
    except Exception as error:
        _remember_error_trace(request, error)
        lifecycle.set_trace(getattr(request.state, "route_trace", None))
        code, message = _stream_error(error)
        lifecycle.log_failure(error, code=code)
        yield encode_event({
            "type": "error",
            "code": code,
            "message": message,
            "request_id": lifecycle.request_id,
        })


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

    route_iterator = None
    try:
        deadline = AgentDeadline.start(
            AGENT_TIMEOUT_SECONDS,
            clock=time.monotonic,
        )
        deadline.ensure_active()
        route_iterator = iter(orchestrator.stream(
            payload.messages,
            deadline=deadline,
        ))
        first_item = _advance_with_request_state(route_iterator, request.state)
        prepared_result = (
            _prepare_result(
                first_item,
                request_state=request.state,
                language_hint=payload.messages[-1].content,
            )
            if isinstance(first_item, RouteExecutionResult)
            else None
        )
    except StopIteration as error:
        if route_iterator is not None:
            _close_with_request_state(route_iterator, request.state)
        release_llm_slot()
        raise RuntimeError("route stream ended without a result") from error
    except Exception as error:
        _remember_error_trace(request, error)
        if route_iterator is not None:
            _close_with_request_state(route_iterator, request.state)
        release_llm_slot()
        mapped = _prefetch_http_exception(error)
        if mapped is not None:
            raise mapped from error
        raise

    lifecycle = _ResponseLifecycle(
        route_iterator=route_iterator,
        request_id=request_id,
        started_at=started_at,
        request_state=request.state,
    )
    body_iterator = answer_events_with_slot(
        first_item,
        route_iterator,
        prepared_result,
        request,
        payload.messages[-1].content,
        lifecycle,
    )
    return _OwnedStreamingResponse(
        _AsyncClosingIterator(body_iterator, lifecycle),
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
        **request_trace_fields(request.state),
    )
