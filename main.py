import uuid
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from routes.chat import router as chat_router
from starlette.exceptions import HTTPException as StarletteHTTPException

from error_handling import (
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from services.retrieval import build_retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = build_retriever()
    try:
        yield
    finally:
        del app.state.retriever


app = FastAPI(lifespan=lifespan)

app.add_exception_handler(
    RequestValidationError,
    validation_exception_handler,
)

app.add_exception_handler(
    StarletteHTTPException,
    http_exception_handler,
)

app.add_exception_handler(
    Exception,
    unhandled_exception_handler,
)

app.include_router(chat_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4321",
        "http://127.0.0.1:4321",
    ],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

@app.middleware("http")
async def add_request_id(
    request: Request,
    call_next,
):
    request_id = uuid.uuid4().hex

    request.state.request_id = request_id
    request.state.started_at = time.monotonic()

    response = await call_next(request)

    response.headers["X-Request-ID"] = request_id

    return response

@app.get("/health")
def health():
    return {"status": "ok"}
