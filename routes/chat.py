from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)

from models import ChatRequest
from services.llm import stream_chat


router = APIRouter()


@router.post("/api/chat-stream")
def chat_stream(request: ChatRequest):
    try:
        return StreamingResponse(
            stream_chat(request.messages),
            media_type="text/plain",
        )

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

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        )