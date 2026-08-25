import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}

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

                const response = await fetch("/api/chat", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        message: input.value
                    })
                });

                const data = await response.json();

                answerElement.textContent = data.answer;
            }
        </script>
    </body>
    </html>
    """

@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
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
            stream=False,
            extra_body={
                "thinking": {
                    "type": "disabled",
                }
            },
        )

        answer = response.choices[0].message.content

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