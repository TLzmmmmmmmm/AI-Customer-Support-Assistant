import json
import time
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from knowledge_pipeline.retrieval import RetrievalError, Retriever
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)
from rate_limit import enforce_rate_limit
from models import ChatRequest
from prompts import build_rag_messages
from rag_context import build_retrieved_context
from concurrency import (
    release_llm_slot,
    try_acquire_llm_slot,
)
from app_logging import log_request
from agent import AgentDeadline, AgentDeadlineExceeded, AgentLoop
from config import AGENT_TIMEOUT_SECONDS

router = APIRouter()


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_agent_loop(request: Request) -> AgentLoop:
    return request.app.state.agent_loop


def encode_event(event: dict[str, object]) -> str:
    return json.dumps(
        event,
        ensure_ascii=False,
    ) + "\n"

def answer_events_with_slot(
    answer: str,
    request_id: str,
    started_at: float,
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
    retriever: Retriever = Depends(get_retriever),
    agent_loop: AgentLoop = Depends(get_agent_loop),
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
        results = retriever.retrieve(payload.messages[-1].content)
        deadline.ensure_active()
        retrieved_context = build_retrieved_context(results)
        provider_messages = build_rag_messages(
            payload.messages,
            retrieved_context,
        )
        agent_result = agent_loop.run(
            provider_messages,
            deadline=deadline,
        )

    except RetrievalError as error:
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
        release_llm_slot()

        raise HTTPException(
            status_code=502,
            detail={
                "code": "provider_error",
                "message": "服务暂时出现异常，请稍后再试。",
                "internal_error": type(error).__name__,
            },
        )

    except Exception:
        release_llm_slot()
        raise

    return StreamingResponse(
        answer_events_with_slot(
            agent_result.answer,
            request_id,
            started_at,
        ),
        media_type="application/x-ndjson",
    )

