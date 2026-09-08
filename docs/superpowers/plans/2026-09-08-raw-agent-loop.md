# Raw Agent Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded native-tool-calling Agent V0 behind the existing `/api/chat-stream` endpoint while reusing the Day 1 deterministic tool contracts and preserving initial RAG behavior.

**Architecture:** Initial RAG continues to retrieve only the latest raw user message and produces the existing provider messages. A small synchronous `AgentLoop` then performs non-streaming native Chat Completions, executes at most three single tool calls through the immutable Day 1 registry, feeds complete observations back to the model, and returns one complete answer. The route adapts that answer to the unchanged NDJSON schema and owns one concurrency slot across initial RAG, the full Agent run, and response cleanup.

**Tech Stack:** Python 3.12, FastAPI, Pydantic 2, OpenAI Python SDK 3.3.1 against DeepSeek Chat Completions, standard-library `json`, `dataclasses`, `MappingProxyType`, `time.monotonic`, and `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-08-raw-agent-loop-design.md`

## Global Constraints

- Continue on the existing `week3-tool` branch; do not create another branch or worktree.
- Reuse Day 1 `ToolError`, `ToolErrorCode`, `SourceRef`, tool result models, `DeterministicTools`, and immutable registry. Do not duplicate them in `agent/`.
- Do not modify deterministic tool behavior, Retriever behavior, the knowledge schema, or the initial RAG query.
- Do not add LangChain, LangGraph, MCP, query rewriting, server memory, citation rendering, frontend changes, or a deterministic router.
- Keep `POST /api/chat-stream`, `ChatRequest`, and the NDJSON event schema unchanged.
- Agent LLM calls are non-streaming. A successful response emits one complete `delta` event and one `done` event.
- Accept at most one native tool call per LLM response. A valid tool call takes precedence over accompanying assistant content.
- `MAX_TOOL_CALLS = 3` counts successful, failed, and cached duplicate proposals.
- After the third observation, make exactly one tools-disabled final completion and never execute a fourth tool.
- `AGENT_TIMEOUT_SECONDS = 120.0` is a lightweight monotonic deadline checked before and after blocking operations. It does not preempt a synchronous operation already in progress.
- A recommendation or scenario-fit answer uses generic professional-technical-staff guidance without requiring `get_contact_info`. Explicit phone, email, or contact-channel questions must use `get_contact_info`.
- Tool observations never expose raw exceptions, tracebacks, internal paths, arbitrary unknown tool names, or hidden implementation details.
- Automated tests use controlled fakes and make no paid LLM or embedding calls.
- Run project tests with `.venv\Scripts\python.exe`; this interpreter requires the user's sandbox approval.

---

## File Structure

### New production files

- `agent/__init__.py` — public Agent V0 exports only.
- `agent/models.py` — Agent-owned input schemas, normalized call/turn/result models, tool specifications, and deadline value object.
- `agent/executor.py` — immutable-registry execution boundary and controlled observation serialization.
- `agent/loop.py` — request-local message state, tool budget, cache, policy, response normalization, and finalization.

### Modified production files

- `services/llm.py` — add a non-streaming Chat Completions adapter without removing legacy streaming functions.
- `config.py` — define the accepted 120-second lightweight Agent deadline.
- `main.py` — construct one Retriever and share it with the Day 1 tools and application-scoped Agent Loop.
- `routes/chat.py` — preserve initial RAG, run the Agent under one request slot/deadline, and adapt its complete answer to NDJSON.

### New tests

- `tests/test_agent_models.py` — schemas, tool definitions, strict inputs, and deadline behavior.
- `tests/test_agent_executor.py` — registry boundary, validation, errors, serialization, and request-local success reuse.
- `tests/test_agent_loop.py` — native message history, policy paths, duplicate accounting, malformed responses, and max-call finalization.

### Modified tests

- `tests/test_llm.py` — non-streaming adapter request and retry contract.
- `tests/test_main.py` — single Retriever sharing and Agent lifecycle cleanup.
- `tests/test_chat_route.py` — direct route orchestration and deadline/slot behavior.
- `tests/test_chat_http.py` — public endpoint, complete-answer NDJSON, provider failures, and no internal tool leakage.

---

### Task 1: Agent-Owned Models and One Source of Tool Schemas

**Files:**
- Create: `agent/__init__.py`
- Create: `agent/models.py`
- Create: `tests/test_agent_models.py`

**Interfaces:**
- Consumes: `MAX_PRODUCT_ID_CHARACTERS`, `MAX_PRODUCT_QUERY_CHARACTERS` from `support_tools.service`; `ProductDetailsResult`, `ContactInfoResult`, and `ProductSearchResult` from `support_tools`.
- Produces: `AgentToolCall`, `AgentTurn`, `AgentResult`, `AgentDeadline`, `AgentDeadlineExceeded`, `ToolSpec`, `TOOL_SPECS`, and `llm_tool_schemas()`.

- [x] **Step 1: Write failing schema and deadline tests**

Create `tests/test_agent_models.py` with focused tests equivalent to:

```python
import unittest

from pydantic import ValidationError


class AgentModelTests(unittest.TestCase):
    def test_tool_schemas_have_exact_names_arguments_and_no_extra_properties(self):
        from agent.models import llm_tool_schemas

        schemas = llm_tool_schemas()
        by_name = {
            item["function"]["name"]: item["function"]["parameters"]
            for item in schemas
        }
        self.assertEqual(set(by_name), {
            "search_products",
            "get_product_details",
            "get_contact_info",
        })
        self.assertEqual(by_name["search_products"]["required"], ["query"])
        self.assertEqual(
            by_name["get_product_details"]["required"],
            ["product_id"],
        )
        self.assertEqual(by_name["get_contact_info"]["properties"], {})
        self.assertTrue(all(
            parameters["additionalProperties"] is False
            for parameters in by_name.values()
        ))

    def test_argument_models_reuse_day_one_limits_and_reject_extra_fields(self):
        from agent.models import ProductDetailsArguments, SearchProductsArguments

        self.assertEqual(
            SearchProductsArguments(query="  酒店对讲机  ").query,
            "酒店对讲机",
        )
        self.assertEqual(
            ProductDetailsArguments(product_id=" LY198 ").product_id,
            "LY198",
        )
        for model, payload in (
            (SearchProductsArguments, {"query": "x" * 4001}),
            (ProductDetailsArguments, {"product_id": "x" * 129}),
            (ProductDetailsArguments, {"product_id": "LY198", "extra": 1}),
        ):
            with self.subTest(model=model.__name__, payload=payload):
                with self.assertRaises(ValidationError):
                    model.model_validate(payload)

    def test_deadline_uses_injected_monotonic_clock(self):
        from agent.models import AgentDeadline, AgentDeadlineExceeded

        times = iter([10.0, 129.9, 130.0])
        deadline = AgentDeadline.start(120.0, clock=lambda: next(times))
        deadline.ensure_active()
        with self.assertRaises(AgentDeadlineExceeded):
            deadline.ensure_active()
```

- [x] **Step 2: Run the new tests and verify they fail because `agent.models` does not exist**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_agent_models -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agent'`.

- [x] **Step 3: Implement strict Agent models and explicit tool specifications**

Create `agent/models.py` with these concrete shapes:

```python
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from support_tools import (
    ContactInfoResult,
    ProductDetailsResult,
    ProductSearchResult,
)
from support_tools.service import (
    MAX_PRODUCT_ID_CHARACTERS,
    MAX_PRODUCT_QUERY_CHARACTERS,
)


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SearchProductsArguments(ToolArguments):
    query: str = Field(min_length=1, max_length=MAX_PRODUCT_QUERY_CHARACTERS)


class ProductDetailsArguments(ToolArguments):
    product_id: str = Field(min_length=1, max_length=MAX_PRODUCT_ID_CHARACTERS)


class ContactInfoArguments(ToolArguments):
    pass


@dataclass(frozen=True)
class ToolSpec:
    description: str
    arguments_model: type[ToolArguments]
    result_model: type[BaseModel]


TOOL_SPECS: Mapping[str, ToolSpec] = MappingProxyType({
    "search_products": ToolSpec(
        description=(
            "Find product candidates for an explicit discovery, selection, "
            "or recommendation request using the user's unmodified query."
        ),
        arguments_model=SearchProductsArguments,
        result_model=ProductSearchResult,
    ),
    "get_product_details": ToolSpec(
        description=(
            "Retrieve authoritative details for one exact canonical product "
            "ID or slug; do not pass a display name or fuzzy description."
        ),
        arguments_model=ProductDetailsArguments,
        result_model=ProductDetailsResult,
    ),
    "get_contact_info": ToolSpec(
        description=(
            "Retrieve official phone and email contact channels when the user "
            "explicitly asks how to contact the company; no address is returned."
        ),
        arguments_model=ContactInfoArguments,
        result_model=ContactInfoResult,
    ),
})


@dataclass(frozen=True)
class AgentToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class AgentTurn:
    content: str | None
    tool_calls: tuple[AgentToolCall, ...]
    finish_reason: str | None


@dataclass(frozen=True)
class AgentResult:
    answer: str


class AgentDeadlineExceeded(Exception):
    pass


@dataclass(frozen=True)
class AgentDeadline:
    expires_at: float
    clock: Callable[[], float] = field(repr=False, compare=False)

    @classmethod
    def start(
        cls,
        timeout_seconds: float,
        *,
        clock: Callable[[], float],
    ) -> "AgentDeadline":
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        return cls(expires_at=clock() + timeout_seconds, clock=clock)

    def ensure_active(self) -> None:
        if self.clock() >= self.expires_at:
            raise AgentDeadlineExceeded("agent deadline exceeded")


def llm_tool_schemas() -> list[dict[str, Any]]:
    schemas = []
    for name, spec in TOOL_SPECS.items():
        parameters = spec.arguments_model.model_json_schema()
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": spec.description,
                "parameters": parameters,
            },
        })
    return schemas
```

Ensure `llm_tool_schemas()` returns fresh nested dictionaries so callers cannot
mutate `TOOL_SPECS`. Do not put executable callables in this mapping.

- [x] **Step 4: Export only public Agent model contracts**

Create `agent/__init__.py` exporting the concrete names that later tasks import.
Do not export Day 1 result types under new aliases.

- [x] **Step 5: Run model tests**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_agent_models -v
```

Expected: all tests PASS.

- [x] **Step 6: Commit Task 1**

```powershell
git add agent\__init__.py agent\models.py tests\test_agent_models.py
git commit -m "feat: define native agent tool schemas"
```

---

### Task 2: Immutable-Registry Tool Executor and Controlled Observations

**Files:**
- Create: `agent/executor.py`
- Create: `tests/test_agent_executor.py`
- Modify: `agent/__init__.py`

**Interfaces:**
- Consumes: `AgentToolCall`, `TOOL_SPECS`, Day 1 `ToolError`, `ToolErrorCode`, result models, and `Mapping[str, Callable[..., object]]` from `build_tool_registry()`.
- Produces: `ToolObservation` and `ToolExecutor.execute(call, successful_observations)`.

- [x] **Step 1: Write failing executor tests for success, safe failures, and caching**

Create `tests/test_agent_executor.py`. Use real Day 1 result model instances and
a `MappingProxyType` registry. Cover these exact assertions:

```python
import json
import unittest
from types import MappingProxyType

from agent.models import AgentToolCall
from support_tools import ProductSearchResult, ToolError, ToolErrorCode


class AgentExecutorTests(unittest.TestCase):
    def test_success_serializes_existing_result_and_reuses_request_cache(self):
        from agent.executor import ToolExecutor

        calls = []

        def search_products(query):
            calls.append(query)
            return ProductSearchResult(products=[])

        executor = ToolExecutor(MappingProxyType({
            "search_products": search_products,
        }))
        cache = {}
        first = executor.execute(
            AgentToolCall("call-1", "search_products", '{"query":" 酒店 "}'),
            cache,
        )
        second = executor.execute(
            AgentToolCall("call-2", "search_products", '{"query":"酒店"}'),
            cache,
        )

        self.assertEqual(calls, ["酒店"])
        self.assertTrue(first.success)
        self.assertFalse(first.reused)
        self.assertTrue(second.reused)
        self.assertEqual(first.content, second.content)
        self.assertEqual(json.loads(first.content), {
            "ok": True,
            "result": {"products": []},
        })

    def test_day_one_error_is_a_safe_observation_and_not_cached(self):
        from agent.executor import ToolExecutor

        calls = []

        def details(product_id):
            calls.append(product_id)
            raise ToolError(
                code=ToolErrorCode.PRODUCT_NOT_FOUND,
                message="No product matches the supplied canonical identifier.",
                tool_name="get_product_details",
            )

        executor = ToolExecutor(MappingProxyType({
            "get_product_details": details,
        }))
        cache = {}
        result = executor.execute(
            AgentToolCall(
                "call-1",
                "get_product_details",
                '{"product_id":"LY999"}',
            ),
            cache,
        )
        self.assertFalse(result.success)
        self.assertEqual(cache, {})
        self.assertEqual(
            json.loads(result.content)["error"]["code"],
            "PRODUCT_NOT_FOUND",
        )

    def test_unexpected_callable_error_is_redacted(self):
        from agent.executor import ToolExecutor

        def broken(query):
            raise RuntimeError("private path and traceback")

        executor = ToolExecutor(MappingProxyType({"search_products": broken}))
        result = executor.execute(
            AgentToolCall("call-1", "search_products", '{"query":"酒店"}'),
            {},
        )
        self.assertEqual(
            json.loads(result.content)["error"]["code"],
            "TOOL_EXECUTION_ERROR",
        )
        self.assertNotIn("private", result.content)
```

Also add table-driven tests for:

- unknown name → `INVALID_ARGUMENT`, `tool_name="tool_executor"`, no arbitrary
  name echoed;
- malformed JSON;
- JSON array/string/null instead of object;
- missing, additional, or wrong-type arguments;
- registry/spec mismatch;
- callable returning the wrong Day 1 result model;
- failure followed by corrected arguments executes again;
- same tool with different validated arguments executes both calls.

- [x] **Step 2: Run the executor tests and verify the missing module failure**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_agent_executor -v
```

Expected: FAIL because `agent.executor` does not exist.

- [x] **Step 3: Implement the execution result and serializer**

Create `agent/executor.py` around these exact interfaces:

```python
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pydantic import ValidationError

from support_tools import ToolError, ToolErrorCode

from .models import AgentToolCall, TOOL_SPECS


@dataclass(frozen=True)
class ToolObservation:
    content: str
    success: bool
    reused: bool
    cache_key: str | None


class ToolExecutor:
    def __init__(self, registry: Mapping[str, Callable[..., object]]) -> None:
        self._registry = registry

    def execute(
        self,
        call: AgentToolCall,
        successful_observations: dict[str, str],
    ) -> ToolObservation:
        spec = TOOL_SPECS.get(call.name)
        if spec is None:
            return _error_observation(_invalid_argument("tool_executor"))

        try:
            decoded = json.loads(call.arguments)
            if not isinstance(decoded, dict):
                raise ValueError("tool arguments must be an object")
            arguments = spec.arguments_model.model_validate(decoded)
        except (json.JSONDecodeError, ValidationError, ValueError):
            return _error_observation(_invalid_argument(call.name))

        validated_arguments = arguments.model_dump(mode="json")
        cache_key = json.dumps(
            {"name": call.name, "arguments": validated_arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        cached = successful_observations.get(cache_key)
        if cached is not None:
            return ToolObservation(cached, True, True, cache_key)

        tool = self._registry.get(call.name)
        if tool is None:
            return _error_observation(_invalid_argument("tool_executor"))

        try:
            result = tool(**validated_arguments)
            if not isinstance(result, spec.result_model):
                raise TypeError("unexpected tool result type")
        except ToolError as error:
            return _error_observation(error)
        except Exception:
            return _error_observation(ToolError(
                code=ToolErrorCode.TOOL_EXECUTION_ERROR,
                message="The tool could not complete the request.",
                tool_name=call.name,
            ))

        content = json.dumps(
            {"ok": True, "result": result.model_dump(mode="json")},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        successful_observations[cache_key] = content
        return ToolObservation(content, True, False, cache_key)
```

Use `json.loads()` and the selected Pydantic input model. Serialize canonical
arguments with:

```python
validated_arguments = arguments_model.model_dump(mode="json")
cache_key = json.dumps(
    {"name": call.name, "arguments": validated_arguments},
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
)
```

On a cached success, return the cached observation without registry execution.
On a new success, require `isinstance(result, spec.result_model)`, serialize its
existing `model_dump(mode="json")`, and store only the final compact observation
JSON in the caller-provided cache.

Use one helper that creates the existing error type, not a new error model:

```python
def _invalid_argument(tool_name: str) -> ToolError:
    return ToolError(
        code=ToolErrorCode.INVALID_ARGUMENT,
        message="The proposed tool arguments are invalid.",
        tool_name=tool_name,
    )
```

Implement `_error_observation()` to serialize `ToolError` to the confirmed
`{"ok":false,"error":{"code":string,"message":string,"tool_name":string}}`
envelope with `success=False`, `reused=False`, and `cache_key=None`. Catch
`ValidationError`, `json.JSONDecodeError`, and wrong decoded types as invalid
arguments. Catch existing `ToolError` separately. Convert every other callable
exception or wrong result type to `TOOL_EXECUTION_ERROR`. Never include `str(error)`.

Do not use `getattr`, `globals`, `eval`, `exec`, dynamic imports, signature
inspection, or a second callable registry.

- [x] **Step 4: Export executor contracts and run focused tests**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_agent_models tests.test_agent_executor -v
```

Expected: all tests PASS.

- [x] **Step 5: Commit Task 2**

```powershell
git add agent\__init__.py agent\executor.py tests\test_agent_executor.py
git commit -m "feat: add safe agent tool executor"
```

---

### Task 3: Non-Streaming Native Tool-Calling LLM Adapter

**Files:**
- Modify: `services/llm.py`
- Modify: `tests/test_llm.py`

**Interfaces:**
- Consumes: provider-compatible `messages` and optional native function-tool schema dictionaries.
- Produces: `complete_chat(messages, *, tools=None) -> object`, returning one complete SDK `ChatCompletion` response.

- [ ] **Step 1: Add failing non-streaming request tests**

Extend `tests/test_llm.py` with tests equivalent to:

```python
    def test_complete_chat_sends_native_tools_without_mutating_inputs(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "get_contact_info",
                "description": "Get contact channels.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        }]
        response = object()
        with patch.object(
            llm.client.chat.completions,
            "create",
            return_value=response,
        ) as create:
            result = llm.complete_chat(PROVIDER_MESSAGES, tools=tools)

        self.assertIs(result, response)
        self.assertEqual(create.call_args.kwargs["messages"], PROVIDER_MESSAGES)
        self.assertEqual(create.call_args.kwargs["tools"], tools)
        self.assertFalse(create.call_args.kwargs["stream"])
        self.assertEqual(
            create.call_args.kwargs["extra_body"],
            {"thinking": {"type": "disabled"}},
        )

    def test_complete_chat_omits_tools_for_finalization(self):
        with patch.object(
            llm.client.chat.completions,
            "create",
            return_value=object(),
        ) as create:
            llm.complete_chat(PROVIDER_MESSAGES)
        self.assertNotIn("tools", create.call_args.kwargs)
        self.assertFalse(create.call_args.kwargs["stream"])
```

Add retry tests proving:

- timeout makes one attempt and propagates `APITimeoutError`;
- connection error makes at most `LLM_APP_MAX_RETRIES + 1` attempts;
- HTTP 429 and 5xx retry according to the existing policy;
- non-retryable 4xx propagates after one attempt;
- inputs are shallow-copied and not mutated.

- [ ] **Step 2: Run the focused tests and verify `complete_chat` is missing**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_llm -v
```

Expected: FAIL with `AttributeError: module 'services.llm' has no attribute 'complete_chat'`.

- [ ] **Step 3: Implement `complete_chat` using the existing retry contract**

Add to `services/llm.py`:

```python
def complete_chat(
    messages: Sequence[Mapping[str, object]],
    *,
    tools: Sequence[Mapping[str, object]] | None = None,
):
    provider_messages = [dict(message) for message in messages]
    provider_tools = None if tools is None else [dict(tool) for tool in tools]
    attempt = 0

    while True:
        try:
            request = {
                "model": DEEPSEEK_MODEL,
                "messages": provider_messages,
                "stream": False,
                "extra_body": {"thinking": {"type": "disabled"}},
            }
            if provider_tools is not None:
                request["tools"] = provider_tools
            return client.chat.completions.create(**request)
        except APITimeoutError:
            raise
        except APIConnectionError:
            if attempt >= LLM_APP_MAX_RETRIES:
                raise
        except APIStatusError as error:
            if not is_retryable_status(error) or attempt >= LLM_APP_MAX_RETRIES:
                raise
        attempt += 1
        time.sleep(LLM_RETRY_DELAY_SECONDS)
```

Keep `open_chat_stream`, `iter_chat_content`, and `stream_chat` intact. If a
small private request-copy helper is shared, ensure legacy tests still prove
the existing streaming request bytes and retry behavior.

- [ ] **Step 4: Run LLM and existing route provider tests**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_llm tests.test_chat_route tests.test_chat_http -v
```

Expected: all currently applicable tests PASS before route migration.

- [ ] **Step 5: Commit Task 3**

```powershell
git add services\llm.py tests\test_llm.py
git commit -m "feat: add non-streaming tool completion"
```

---

### Task 4: Bounded Agent Loop and Provider-Compatible Message State

**Files:**
- Create: `agent/loop.py`
- Create: `tests/test_agent_loop.py`
- Modify: `agent/__init__.py`

**Interfaces:**
- Consumes: `ToolExecutor`, `AgentDeadline`, `llm_tool_schemas()`, and a callable matching `complete_chat(messages, *, tools=None)`.
- Produces: `AgentLoop.run(messages, *, deadline) -> AgentResult`, `MAX_TOOL_CALLS = 3`, `SAFE_AGENT_ANSWER`, and `AGENT_TOOL_POLICY`.

- [ ] **Step 1: Write failing loop tests for no-tool, one-tool, and action precedence**

Create SDK-shaped fakes without importing private SDK internals:

```python
from types import SimpleNamespace


def text_completion(content, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=content, tool_calls=None),
    )])


def tool_completion(call_id, name, arguments, content=None):
    call = SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="tool_calls",
        message=SimpleNamespace(content=content, tool_calls=[call]),
    )])
```

Write tests proving:

- direct company-name completion returns final text, makes one completion, and
  never executes a tool;
- a `content + tool_call` response executes the call, never returns the action
  content, and returns only the later final answer;
- the appended assistant message contains the exact call ID/name/raw arguments;
- the next `role="tool"` message has the matching `tool_call_id` and complete
  observation JSON;
- the original input message list is not mutated;
- the policy system message is inserted without modifying the existing RAG
  system message.

- [ ] **Step 2: Add failing loop tests for the must-use behavior sequences**

Use controlled completion sequences to verify:

```text
explicit product recommendation
  → search_products
  → final candidate answer with generic professional guidance
  → no get_contact_info execution

empty product search
  → final no-reliable-candidate answer with generic professional guidance
  → no get_contact_info execution

exact model parameter question
  → get_product_details
  → final answer

explicit contact-channel question
  → get_contact_info
  → final answer
```

Inspect the model-visible `AGENT_TOOL_POLICY` in the first completion request and
assert it states these must-use and generic-handoff rules. The fake model
controls decisions; do not add Python intent matching or a deterministic router.

- [ ] **Step 3: Add failing loop tests for errors, duplicates, and maximum calls**

Cover:

- `PRODUCT_NOT_FOUND` observation followed by a corrected invocation and final
  answer;
- unexpected callable failure appears as `TOOL_EXECUTION_ERROR` and contains no
  private text;
- the same successful normalized invocation proposed twice executes once but
  consumes two slots and pairs the cached result with the new call ID;
- the same tool with two different arguments executes twice;
- three tool proposals produce three observations followed by exactly one
  completion whose request omits `tools`;
- a tools-disabled response containing another tool call returns
  `SAFE_AGENT_ANSWER` and executes no fourth tool;
- multiple tool calls in one enabled response execute none and return
  `SAFE_AGENT_ANSWER`;
- no choices, blank content, missing call ID/function, and `finish_reason` equal
  to `length` or `content_filter` terminate safely;
- every enabled and disabled LLM call checks the supplied deadline before and
  after the call;
- every tool execution checks the deadline before and after execution.

- [ ] **Step 4: Run the loop tests and verify the missing implementation failure**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_agent_loop -v
```

Expected: FAIL because `agent.loop` does not exist.

- [ ] **Step 5: Implement the explicit loop and safe normalizer**

Create `agent/loop.py` with:

```python
MAX_TOOL_CALLS = 3
SAFE_AGENT_ANSWER = "暂时无法完成本次咨询，请稍后重试。"

AGENT_TOOL_POLICY = """
Use the provided deterministic tools under these rules:
- Exact product model parameters, features, and details require get_product_details.
- Explicit product discovery, candidate selection, and recommendation requests require search_products.
- Explicit phone, email, and contact-channel questions require get_contact_info.
- Recommendation and scenario-fit answers must frame products as candidates unless an observation explicitly proves suitability, and must direct the user to professional technical staff for final selection.
- When recommendation-oriented solution or support advice needs expert confirmation, add the same generic professional-staff guidance.
- If product search returns no candidates, state that no reliable candidate was found and still add generic professional-staff guidance.
- Generic guidance does not require get_contact_info. Never invent contact facts.
- Do not repeat a successful invocation merely because the original request still matches a must-use category.
- Propose at most one tool call per response.
""".strip()
```

Use this constructor and run signature:

```python
class AgentLoop:
    def __init__(self, *, executor: ToolExecutor, complete_chat) -> None:
        self._executor = executor
        self._complete_chat = complete_chat
        self._tool_schemas = llm_tool_schemas()
```

Add `run(self, messages: Sequence[Mapping[str, object]], *, deadline:
AgentDeadline) -> AgentResult` and implement the following state transitions.

Construct a fresh request-local list, counter, and `dict[str, str]` success
cache. Insert the Agent policy as a second controlled system message after the
existing RAG system message. Do not mutate input mappings.

For every completion:

1. `deadline.ensure_active()` before the provider call.
2. Call `complete_chat(state, tools=self._tool_schemas)` while the call count is
   below three; omit `tools` only for the final tools-disabled completion.
3. `deadline.ensure_active()` after the provider call.
4. Require exactly one choice and reject `length`/`content_filter`.
5. Normalize direct SDK attributes (`choice.message.content`,
   `choice.message.tool_calls`, `tool_call.id`, `tool_call.function.name`, and
   `tool_call.function.arguments`) into `AgentTurn`; do not use dynamic tool
   lookup.
6. If tool calls exist, require exactly one valid function call. Treat it as the
   action even when content is present.
7. Increment the tool-call count before executor handling.
8. Append a controlled provider-compatible assistant tool-call dictionary.
9. Check the deadline, execute or reuse through `ToolExecutor`, check the
   deadline again, and append the matching tool observation.
10. If this was call three, make one final completion without `tools`; accept
    only non-blank content with no tool calls.
11. Otherwise continue the enabled loop.
12. A non-blank content response with no tool calls returns `AgentResult`.

Malformed provider structure returns `SAFE_AGENT_ANSWER`; provider API errors
and `AgentDeadlineExceeded` propagate to the route. Never catch those as tool
errors.

- [ ] **Step 6: Run Agent unit tests**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_agent_models tests.test_agent_executor tests.test_agent_loop -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit Task 4**

```powershell
git add agent\__init__.py agent\loop.py tests\test_agent_loop.py
git commit -m "feat: implement bounded raw agent loop"
```

---

### Task 5: Application Lifespan, Request Deadline, and Route Integration

**Files:**
- Modify: `config.py`
- Modify: `main.py`
- Modify: `routes/chat.py`
- Modify: `tests/test_main.py`
- Modify: `tests/test_chat_route.py`

**Interfaces:**
- Consumes: `build_retriever()`, `build_deterministic_tools(retriever=...)`, `build_tool_registry()`, `ToolExecutor`, `AgentLoop`, `complete_chat`, `AgentDeadline`, and `AGENT_TIMEOUT_SECONDS`.
- Produces: `app.state.agent_loop`, `get_agent_loop(request)`, and the migrated `/api/chat-stream` orchestration.

- [ ] **Step 1: Write failing lifespan tests for one shared Retriever**

Extend `tests/test_main.py` so a lifespan run asserts:

```python
with (
    patch.object(main, "build_retriever", return_value=retriever) as build_retriever,
    patch.object(main, "build_deterministic_tools", return_value=tools) as build_tools,
    patch.object(main, "build_tool_registry", return_value=registry) as build_registry,
):
    async with main.app.router.lifespan_context(main.app):
        self.assertIs(main.app.state.retriever, retriever)
        self.assertIsInstance(main.app.state.agent_loop, AgentLoop)

build_retriever.assert_called_once_with()
build_tools.assert_called_once_with(retriever=retriever)
build_registry.assert_called_once_with(tools)
self.assertFalse(hasattr(main.app.state, "retriever"))
self.assertFalse(hasattr(main.app.state, "agent_loop"))
```

Do not construct a second Retriever in either builder.

- [ ] **Step 2: Write failing direct-route tests for Agent orchestration and deadline**

Replace the old direct route's streaming-provider assumptions in
`tests/test_chat_route.py` with a fake Agent Loop whose `run()` records provider
messages and deadline. Assert:

- the slot is acquired before initial retrieval;
- the deadline starts immediately after acquisition;
- the Retriever receives only `payload.messages[-1].content` unchanged;
- existing `build_rag_messages()` receives the complete public history and RAG
  context;
- the Agent receives those provider messages and the deadline;
- the response contains one complete delta and one done;
- tool-call history from the fake Agent is never exposed;
- `AgentDeadlineExceeded` before Agent start maps to HTTP 504 with the existing
  safe timeout message;
- initial `RetrievalError` still maps to HTTP 503;
- provider timeout/connection/status errors from Agent still map to existing
  504/503/502 responses;
- unexpected errors still reach the safe 500 handler;
- every error path releases the slot exactly once.

Use an injected/patched monotonic sequence rather than sleeping:

```python
times = iter([10.0, 10.1, 130.0])
with patch.object(chat.time, "monotonic", side_effect=lambda: next(times)):
    with self.assertRaises(HTTPException) as caught:
        chat.chat_stream(
            payload,
            request,
            None,
            retriever,
            agent_loop,
        )

self.assertEqual(caught.exception.status_code, 504)
self.assertEqual(agent_loop.calls, [])
```

- [ ] **Step 3: Run focused tests and verify the missing Agent dependencies**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_main tests.test_chat_route -v
```

Expected: FAIL because lifespan and route do not yet construct or accept the
Agent Loop.

- [ ] **Step 4: Add the accepted deadline configuration**

Add to `config.py`:

```python
AGENT_TIMEOUT_SECONDS = 120.0
```

Do not add a second environment-loading path or timeout framework.

- [ ] **Step 5: Wire the shared application-scoped Agent**

Update `main.py` lifespan in this order:

```python
retriever = build_retriever()
tools = build_deterministic_tools(retriever=retriever)
registry = build_tool_registry(tools)
app.state.retriever = retriever
app.state.agent_loop = AgentLoop(
    executor=ToolExecutor(registry),
    complete_chat=complete_chat,
)
try:
    yield
finally:
    del app.state.agent_loop
    del app.state.retriever
```

Import existing Day 1 builders; do not recreate product/contact facts or tool
callables in `main.py`.

- [ ] **Step 6: Migrate the route while preserving its public protocol**

Add:

```python
def get_agent_loop(request: Request) -> AgentLoop:
    return request.app.state.agent_loop
```

After `try_acquire_llm_slot()` succeeds:

```python
deadline = AgentDeadline.start(
    AGENT_TIMEOUT_SECONDS,
    clock=time.monotonic,
)
deadline.ensure_active()
results = retriever.retrieve(payload.messages[-1].content)
deadline.ensure_active()
retrieved_context = build_retrieved_context(results)
provider_messages = build_rag_messages(payload.messages, retrieved_context)
agent_result = agent_loop.run(provider_messages, deadline=deadline)
```

Map `AgentDeadlineExceeded` to the existing HTTP 504 timeout code/message.
Preserve the existing mappings for `RetrievalError`, `APITimeoutError`,
`APIConnectionError`, and `APIStatusError`. Release the slot before raising any
pre-response error.

Replace route usage of the provider stream with a small generator that:

```python
try:
    yield encode_event({"type": "delta", "content": agent_result.answer})
    log_request(
        request_id=request_id,
        http_status=200,
        outcome="success",
        started_at=started_at,
    )
    yield encode_event({"type": "done"})
finally:
    release_llm_slot()
```

The slot therefore remains held from acquisition through initial RAG, the
entire Agent Loop, and response body cleanup. Do not acquire inside
`AgentLoop.run()` or per completion.

Keep legacy streaming helpers if existing evaluation or unit tests still import
them; remove only route-specific dead code after confirming no caller remains.

- [ ] **Step 7: Run application integration tests**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_main tests.test_chat_route tests.test_llm -v
```

Expected: all tests PASS.

- [ ] **Step 8: Commit Task 5**

```powershell
git add config.py main.py routes\chat.py tests\test_main.py tests\test_chat_route.py
git commit -m "feat: connect agent loop to chat route"
```

---

### Task 6: HTTP Acceptance, Privacy Regression, and Final Verification

**Files:**
- Modify: `tests/test_chat_http.py`
- If verification exposes a defect, modify only the production or test files
  already listed in Tasks 1–5 and add a focused regression test first.

**Interfaces:**
- Consumes: the complete application path from `POST /api/chat-stream` through initial RAG, Agent Loop, tool registry, and NDJSON response.
- Produces: release evidence for Day 2; no new production interface.

- [ ] **Step 1: Update HTTP fakes to return complete non-streaming completions**

Replace iterator/delta fakes used by the route with SDK-shaped complete
responses. Preserve separate legacy streaming tests in `tests/test_llm.py`.

For a normal answer, assert exactly:

```python
self.assertEqual(
    [json.loads(line) for line in response.text.splitlines()],
    [
        {"type": "delta", "content": "IP68"},
        {"type": "done"},
    ],
)
```

Assert the provider request uses `stream=False`, includes all three native tool
schemas on enabled turns, preserves the original RAG message and prior public
history, and never sends a rewritten query to initial retrieval.

- [ ] **Step 2: Add end-to-end controlled tool-call cases**

Add HTTP acceptance cases with completion side effects for:

- `get_product_details("HP780")` followed by a final answer;
- `search_products("推荐适合酒店使用的产品")` followed by a candidate-framed
  final answer and generic professional guidance, with no contact tool call;
- explicit `get_contact_info()` followed by an answer containing only company
  name, phone, and email facts from its observation, never an address;
- `PRODUCT_NOT_FOUND` followed by a safe final answer;
- three repeated/failed proposals followed by one tools-disabled final answer;
- a malicious “keep calling tools forever” request that executes no more than
  three tool proposals;
- internal assistant tool calls and tool observation JSON absent from the
  frontend NDJSON body.

Use the real app-scoped `DeterministicTools` with controlled Retriever and
provider fakes where practical. Do not copy product/contact facts into Agent
fixtures when the authoritative local source builder can provide them.

- [ ] **Step 3: Replace obsolete route-stream failure expectations**

Because Day 2 intentionally finishes all non-streaming Agent completions before
creating the response body, provider failures are pre-response HTTP errors.
Update the old `test_stream_errors_have_request_id_and_no_done_or_retry` route
case to assert the existing 504/503/502 JSON error contract instead of partial
NDJSON. Keep direct unit coverage for legacy `open_chat_stream` and
`iter_chat_content`; do not claim the route still emits token-level deltas.

- [ ] **Step 4: Run the focused Day 2 suite**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_agent_models tests.test_agent_executor tests.test_agent_loop tests.test_llm tests.test_main tests.test_chat_route tests.test_chat_http tests.test_support_tools tests.test_retriever -v
```

Expected: all tests PASS with no real external request.

- [ ] **Step 5: Run source and contract safety checks**

Run:

```powershell
rg -n "getattr\(|globals\(|eval\(|exec\(" agent
rg -n -i "address|地址" agent routes\chat.py
git diff --check
```

Expected:

- no dynamic execution or arbitrary callable lookup in `agent/`;
- no Agent address field or hard-coded address fact;
- no whitespace errors.

Review the changed-file list and confirm there are no changes to knowledge
sources, schemas, Retriever behavior, frontend files, or dependency manifests.

- [ ] **Step 6: Run the complete repository regression suite**

Run:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: all tests PASS. The pre-Day-2 baseline is 306 tests; the final count
must be greater than 306 because the new Agent tests are included.

- [ ] **Step 7: Perform offline manual acceptance**

Run the existing Day 1 checks to prove tool compatibility remains intact:

```powershell
.venv\Scripts\python.exe scripts\verify_tools.py product LY198
.venv\Scripts\python.exe scripts\verify_tools.py contact
```

Expected: both commands exit 0, return valid complete JSON, and contact output
contains no address field. Do not run paid semantic search or a live LLM smoke
test without separate explicit authorization.

- [ ] **Step 8: Commit acceptance-test updates**

```powershell
git add tests\test_chat_http.py
git commit -m "test: verify raw agent HTTP behavior"
```

- [ ] **Step 9: Confirm final branch state**

Run:

```powershell
git status --short --branch
git log --oneline --decorate -12
```

Expected: branch `week3-tool`, clean working tree, no new worktree, and all Day 2
commits local until the user explicitly requests push or integration.
