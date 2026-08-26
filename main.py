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


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/stream-test")
def stream_test():

    def generate():
        print("generator started")

        for i in range(1, 6):
            print(f"before yield {i}")

            yield f"chunk {i}\n"

            print(f"after yield {i}")

            time.sleep(1)

        print("generator finished")

    return StreamingResponse(
        generate(),
        media_type="text/plain",
    )

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>AI Customer Support</title>
    </head>

    <body>
        <h1>AI Customer Support</h1>

        <input
            id="message"
            type="text"
            placeholder="请输入问题"
        >

        <button onclick="sendMessage()">
            Send
        </button>

        <p id="answer"></p>

        <script>
            async function sendMessage() {
                const input = document.getElementById("message");
                const answerElement = document.getElementById("answer");

                const response = await fetch("/api/chat-stream", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        message: input.value
                    })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                while (true) {
                    const { value, done } = await reader.read();

                    if (done) {
                        break;
                    }

                    const text = decoder.decode(value, {
                        stream: true
                    });

                    answerElement.textContent += text;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        stream = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful customer support assistant.",
                },
                {
                    "role": "user",
                    "content": request.message,
                },
            ],
            stream=True,
            extra_body={
                "thinking": {
                    "type": "disabled",
                }
            },
        )
       
        answer_parts = []

        for chunk in stream:
            content = chunk.choices[0].delta.content

            if content:
                print(repr(content), flush=True)
                answer_parts.append(content)

        answer = "".join(answer_parts)

        return {
            "answer": answer
        }

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

@app.post("/api/chat-stream")
def chat_stream(request: ChatRequest):
    try:
        stream = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful customer support assistant.",
                },
                {
                    "role": "user",
                    "content": request.message,
                },
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