import json
import logging
from collections import defaultdict, deque
from types import MappingProxyType

import pytest
from fastapi.testclient import TestClient

import app_logging
import main
import rate_limit
from agent import AgentToolCall, ToolExecutor
from routes.chat import get_route_orchestrator
from routing import Route, RouteExecutionResult, RouteTrace


SECURITY_CANARY = "SECURITY_CANARY_DO_NOT_EXPOSE_9F31"

# D06-D09 deliberately reuse the named existing regressions instead of
# duplicating executor/schema/graph mechanics in this production-boundary file.
CASE_MAP = (
    ("D01", "HTTP message length", "S8", "PASS_PREVENTED"),
    ("D02", "HTTP message count", "S8", "PASS_PREVENTED"),
    ("D03", "HTTP conversation length", "S8", "PASS_PREVENTED"),
    ("D04", "HTTP JSON parsing", "S8/S11", "PASS_PREVENTED"),
    ("D05", "HTTP request schema", "S8/S11", "PASS_PREVENTED"),
    ("D06", "ToolExecutor allowlist", "S3/S4/S12", "PASS_PREVENTED"),
    ("D07", "ToolExecutor JSON shape", "S4/S12", "PASS_PREVENTED"),
    ("D08", "ToolExecutor argument schema", "S4/S12", "PASS_PREVENTED"),
    ("D09", "Agent tool-call batch budget", "S3/S4/S12", "PASS_PREVENTED"),
    ("D10", "HTTP rate limit", "S9/S12", "PASS_PREVENTED"),
    ("D11", "HTTP exception boundary", "S11", "PASS_PREVENTED"),
    ("D12", "Tool exception boundary", "S11/S12", "PASS_PREVENTED"),
)

EXISTING_CASE_TESTS = {
    "D06": (
        "tests/test_agent_executor.py::"
        "AgentExecutorTests::"
        "test_unknown_tool_is_redacted_and_never_executes_registry"
    ),
    "D07": (
        "tests/test_agent_executor.py::"
        "AgentExecutorTests::"
        "test_invalid_argument_shapes_do_not_execute_tool"
    ),
    "D08": (
        "tests/test_agent_executor.py::"
        "AgentExecutorTests::"
        "test_named_invalid_arguments_do_not_execute_handler"
    ),
    "D09": (
        "tests/test_agent_graph.py::AgentGraphTests::"
        "test_agent_oversized_batch_is_rejected_atomically"
    ),
}


class RecordingOrchestrator:
    def __init__(self) -> None:
        self.calls = 0
        self.error: Exception | None = None

    def run(self, messages, *, deadline):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return RouteExecutionResult(
            answer="受控回答",
            trace=RouteTrace(route=Route.DIRECT),
        )


@pytest.fixture
def http_boundary(monkeypatch):
    orchestrator = RecordingOrchestrator()
    monkeypatch.setattr(
        rate_limit,
        "request_history",
        defaultdict(deque),
    )
    main.app.dependency_overrides[get_route_orchestrator] = (
        lambda: orchestrator
    )
    client = TestClient(main.app, raise_server_exceptions=False)
    try:
        yield client, orchestrator
    finally:
        client.close()
        main.app.dependency_overrides.pop(get_route_orchestrator, None)


def assert_safe_validation_error(response, orchestrator) -> None:
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["request_id"]
    lowered = response.text.casefold()
    assert "traceback" not in lowered
    assert ".py" not in lowered
    assert SECURITY_CANARY.casefold() not in lowered
    assert orchestrator.calls == 0


def test_d01_oversized_message_is_rejected_before_orchestration(
    http_boundary,
):
    client, orchestrator = http_boundary

    response = client.post(
        "/api/chat-stream",
        json={"messages": [{"role": "user", "content": "x" * 4001}]},
    )

    assert_safe_validation_error(response, orchestrator)


def test_d02_too_many_messages_are_rejected_before_orchestration(
    http_boundary,
):
    client, orchestrator = http_boundary
    messages = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"message-{index}",
        }
        for index in range(21)
    ]

    response = client.post(
        "/api/chat-stream",
        json={"messages": messages},
    )

    assert_safe_validation_error(response, orchestrator)


def test_d03_oversized_conversation_is_rejected_before_orchestration(
    http_boundary,
):
    client, orchestrator = http_boundary
    messages = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": str(index) + "x" * 2999,
        }
        for index in range(7)
    ]

    response = client.post(
        "/api/chat-stream",
        json={"messages": messages},
    )

    assert_safe_validation_error(response, orchestrator)


def test_d04_malformed_json_is_safely_rejected_before_orchestration(
    http_boundary,
):
    client, orchestrator = http_boundary

    response = client.post(
        "/api/chat-stream",
        content=f'{{"messages":["{SECURITY_CANARY}"',
        headers={"Content-Type": "application/json"},
    )

    assert_safe_validation_error(response, orchestrator)


@pytest.mark.parametrize(
    "payload",
    (
        {"messages": "not-a-list"},
        {},
        {"messages": [{"role": "system", "content": SECURITY_CANARY}]},
        {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
            ]
        },
    ),
    ids=("wrong-field-type", "missing-messages", "invalid-role", "role-order"),
)
def test_d05_invalid_request_shapes_are_safely_rejected(
    http_boundary,
    payload,
):
    client, orchestrator = http_boundary

    response = client.post("/api/chat-stream", json=payload)

    assert_safe_validation_error(response, orchestrator)


def test_d10_rate_limit_rejects_only_request_beyond_threshold(
    http_boundary,
    monkeypatch,
):
    client, orchestrator = http_boundary
    monkeypatch.setattr(rate_limit, "RATE_LIMIT_REQUESTS", 2)
    request = {
        "messages": [{"role": "user", "content": "你好"}],
    }

    admitted = [
        client.post("/api/chat-stream", json=request)
        for _ in range(2)
    ]
    rejected = client.post("/api/chat-stream", json=request)

    assert [response.status_code for response in admitted] == [200, 200]
    assert orchestrator.calls == 2
    assert rejected.status_code == 429
    assert rejected.json()["error"]["code"] == "rate_limit"
    assert rejected.json()["error"]["request_id"]
    assert rejected.headers["Retry-After"].isdigit()
    assert int(rejected.headers["Retry-After"]) > 0
    assert "traceback" not in rejected.text.casefold()


def test_d11_internal_exception_canary_is_absent_from_response_and_logs(
    http_boundary,
    caplog,
):
    client, orchestrator = http_boundary
    orchestrator.error = RuntimeError(
        f"{SECURITY_CANARY} traceback C:\\private\\server.py"
    )
    app_logging.logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="ai_customer_support")
    try:
        response = client.post(
            "/api/chat-stream",
            json={"messages": [{"role": "user", "content": "你好"}]},
        )
    finally:
        app_logging.logger.removeHandler(caplog.handler)

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert response.json()["error"]["request_id"]
    public = response.text.casefold()
    logs = caplog.text.casefold()
    for forbidden in (
        SECURITY_CANARY.casefold(),
        "traceback",
        "server.py",
        "c:\\private",
    ):
        assert forbidden not in public
        assert forbidden not in logs
    assert orchestrator.calls == 1


def test_d12_tool_exception_canary_becomes_safe_structured_observation():
    calls = 0

    def failing_search(query):
        nonlocal calls
        calls += 1
        raise RuntimeError(
            f"{SECURITY_CANARY} traceback C:\\private\\tool.py"
        )

    executor = ToolExecutor(MappingProxyType({
        "search_products": failing_search,
    }))

    observation = executor.execute(
        AgentToolCall(
            "call-1",
            "search_products",
            '{"query":"酒店对讲机"}',
        ),
        {},
    )
    payload = json.loads(observation.content)

    assert calls == 1
    assert observation.success is False
    assert payload["error"]["code"] == "TOOL_EXECUTION_ERROR"
    assert payload["error"]["tool_name"] == "search_products"
    lowered = observation.content.casefold()
    assert SECURITY_CANARY.casefold() not in lowered
    assert "traceback" not in lowered
    assert ".py" not in lowered
