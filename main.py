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
from services.llm import complete_chat
from services.tools import build_deterministic_tools
from support_tools import build_tool_registry
from agent import ToolExecutor
from agent_graph import (
    AgentGraphNodes,
    GraphRouteOrchestrator,
    build_agent_graph,
)
from routing import HybridRouter


@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever = build_retriever()
    tools = build_deterministic_tools(retriever=retriever)
    registry = build_tool_registry(tools)
    executor = ToolExecutor(registry)
    router = HybridRouter(
        retriever=retriever,
        complete_chat=complete_chat,
    )
    graph = build_agent_graph(AgentGraphNodes(
        router=router,
        executor=executor,
        retriever=retriever,
        complete_chat=complete_chat,
    ))
    app.state.route_orchestrator = GraphRouteOrchestrator(
        graph,
    )
    try:
        yield
    finally:
        del app.state.route_orchestrator


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
    request.state.route_trace = None
    request.state.failure_layer = None

    response = await call_next(request)

    response.headers["X-Request-ID"] = request_id

    return response

@app.get("/health")
def health():
    return {"status": "ok"}
