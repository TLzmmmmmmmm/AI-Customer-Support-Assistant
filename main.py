import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)
from pydantic import BaseModel
from typing import Literal

load_dotenv()

api_key = os.environ.get("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY is not set")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4321",
        "http://127.0.0.1:4321",
    ],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/chat-stream")
def chat_stream(request: ChatRequest):
    try:
        conversation = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ]

        stream = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful customer support assistant.",
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

        def generate():
            for chunk in stream:
                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content

                if content:
                    yield content

        return StreamingResponse(
            generate(),
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