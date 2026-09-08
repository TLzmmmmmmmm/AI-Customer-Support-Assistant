# Week 3 Day 2 — Raw Agent Loop Design

## Objective

Implement the first production-oriented Agent V0 on top of the existing
deterministic tool layer without LangChain, LangGraph, MCP, or another agent
framework. The implementation teaches and preserves the primitive loop:

```text
User → LLM → tool proposal → backend execution → tool observation → LLM → answer
```

Day 2 reuses the Day 1 `ToolError`, `SourceRef`, tool result models,
`DeterministicTools`, and immutable registry. It does not duplicate those
contracts or move product and contact facts into the Agent layer.

## Scope Boundaries

Day 2 keeps the following behavior unchanged:

- the public `POST /api/chat-stream` endpoint;
- the existing `ChatRequest` shape;
- the NDJSON event schema;
- the initial retrieval of only the latest raw user message;
- the existing RAG prompt and trust boundary;
- rate limiting, application-level LLM concurrency, and safe HTTP errors;
- the Retriever, embedding, ingestion, chunking, vector index, and knowledge
  schemas;
- the deterministic tool implementations and their authoritative
  `knowledge/source` data path.

The following work is explicitly deferred:

- deterministic routing and skipping unnecessary initial retrieval (Day 3);
- citation rendering and citation events (Day 4);
- streaming tool-call delta assembly;
- server-side conversation memory;
- query rewriting;
- frontend changes;
- framework-level agent abstractions.

## Architecture

The request flow is:

```text
ChatRequest
  → retrieve(latest raw user message)
  → build_rag_messages(existing conversation, retrieved context)
  → AgentLoop.run(provider messages)
  → complete final answer
  → existing NDJSON adapter
      → one complete delta event
      → one done event
```

The Agent Loop uses non-streaming Chat Completions with native `tools` and
`tool_calls`. Tool results are complete structured observations. The Agent
finishes before the HTTP response body is emitted; Day 2 prioritizes loop
correctness over token-level streaming.

The application lifespan constructs one Retriever. The same Retriever serves
both initial RAG and `search_products`; no second embedding provider or vector
index is created. The lifespan also constructs `DeterministicTools`, the Day 1
immutable registry, a tool executor, and a stateless application-scoped Agent
Loop. All per-request messages, counters, and caches remain local to `run()`.

## Modules and Responsibilities

### `agent/models.py`

Defines only Agent-owned transport and state models:

- one normalized native tool call with its ID, function name, and raw JSON
  arguments;
- one normalized LLM turn containing either final text or one tool call;
- the final Agent result;
- the three tool input models used to generate JSON Schema and validate
  proposed arguments.

Tool input models forbid additional properties. They import the Day 1 input
length constants so schema and execution validation do not create independent
limits. The deterministic tools remain the final input-validation boundary.

This module does not redefine `ToolError`, `SourceRef`, or any Day 1 result
model.

### `agent/executor.py`

Accepts a complete tool call and a per-run successful-observation cache. Its
flow is:

```text
tool call
  → validate call envelope and JSON object arguments
  → validate the selected input model
  → immutable Day 1 registry lookup
  → reuse cached success or execute callable
  → serialize a controlled observation
```

The executor never uses `getattr()`, `globals()`, `eval()`, `exec()`, dynamic
imports, or an unregistered callable. Tool schemas and argument validators come
from one explicit specification so the model-visible schema and backend
validation cannot drift independently. The Day 1 registry remains the only
source of executable callables.

Successful observations serialize the existing Day 1 result model directly:

```json
{"ok":true,"result":{"product_id":"ly198","sources":[{"title":"润信达 LY198","url":"https://www.shengborun.com/two-way-radio/ly198/"}]}}
```

Failure observations serialize the existing `ToolError`:

```json
{"ok":false,"error":{"code":"PRODUCT_NOT_FOUND","message":"No product matches the supplied canonical identifier.","tool_name":"get_product_details"}}
```

Raw arguments, Python exceptions, tracebacks, internal paths, and implementation
details are never included in an observation.

### `agent/loop.py`

Maintains one request's internal provider message list, processed tool-call
count, and successful-observation cache. It calls the non-streaming LLM adapter,
normalizes the result, executes at most one proposed tool, appends the standard
assistant tool-call message plus its matching `role="tool"` observation, and
continues until it receives final text.

The file also contains the small Agent tool-use system instruction. A separate
policy framework or router is not introduced.

## Native Tool-Calling Protocol

The LLM receives three function schemas:

```text
search_products(query: string)
get_product_details(product_id: string)
get_contact_info()
```

Every schema uses `additionalProperties: false`. The model proposes a native
tool call; it never executes a tool. The backend accepts at most one tool call
per LLM response.

The Agent's internal history follows the standard pairing:

```text
assistant: tool_calls=[{id, function: {name, arguments}}]
tool:      tool_call_id=id, content={complete observation JSON}
assistant: final answer or next tool call
```

Tool-call and tool-result messages exist only during one HTTP request. They are
not sent to the frontend, accepted in `ChatRequest`, persisted, or restored on
the next request. Multi-turn references such as “它” are resolved from the
existing user/assistant history supplied by the client. Initial RAG still
receives only the latest raw user message and never receives a rewritten query.

## Constrained Tool-Selection Policy

The LLM receives an explicit must-use policy:

- exact product model parameters, features, or details require
  `get_product_details`;
- explicit product discovery, candidate selection, or recommendation requests
  require `search_products`;
- phone, email, or contact-channel requests require `get_contact_info`;
- after successful candidate discovery for a usage scenario, the model should
  call `get_contact_info`, describe products only as candidates, and direct the
  user to professional sales staff for final selection;
- once a successful observation satisfies a requirement, the model must not
  repeat the same invocation merely because the original request still belongs
  to a must-use category;
- a failed invocation may be retried with corrected arguments or replaced with
  another appropriate tool;
- the model must never invent facts missing from observations;
- each response may propose at most one tool call.

Initial RAG remains available as background, but it does not replace a required
deterministic tool. Questions answerable from system-confirmed facts, such as
the company name, may be answered without a tool.

This is constrained LLM selection, not a deterministic router. The backend
does not classify intent, automatically choose tools, or automatically append
`get_contact_info`. Consequently, Day 2 cannot guarantee that a noncompliant
model always follows the must-use policy. Day 3 owns deterministic routing.

## Duplicate Invocations

A successful invocation is cached by function name plus canonical JSON for its
validated arguments. JSON keys are sorted and insignificant input whitespace is
removed for the cache key.

- Repeating the same successful invocation does not execute the callable again.
- The cached observation is paired with the new `tool_call_id` so the provider
  history remains valid.
- The repeated proposal still consumes one of the three tool-call slots.
- Failures are not cached.
- The same tool with different validated arguments executes normally, enabling
  requests such as comparing two product IDs.

This separates the external side effect or retrieval count from the safety
budget and prevents repeated proposals from bypassing the loop limit.

## Tool-Call Limit

`MAX_TOOL_CALLS = 3` limits processed model proposals, not successful tool
executions. Successful, failed, and cached duplicate proposals all consume one
slot. A normal text completion consumes no tool slot.

After the third observation is appended, the backend makes one final
non-streaming LLM completion with tools disabled. That completion receives the
complete history, including the third observation, and can only produce the
final natural-language answer. It does not consume another tool slot.

If contact information was not successfully obtained within the limit, the
final answer may retain the professional-sales recommendation but must not
invent a phone number, email address, or other contact fact.

## Error Handling

### Controlled tool observations

The following model-proposal defects become
`ToolError(code=INVALID_ARGUMENT)` observations when a valid tool-call ID is
available:

- an unregistered tool name;
- arguments that are not valid JSON;
- arguments whose decoded value is not an object;
- missing, additional, or incorrectly typed arguments.

An unregistered tool does not echo the arbitrary proposed name and uses
`tool_name="tool_executor"`. Existing Day 1 tool errors are serialized without
changing their code or safe message. Any unexpected exception from a registered
callable becomes the existing `TOOL_EXECUTION_ERROR`.

### Safe Agent termination

The following invalid LLM responses terminate the loop without tool execution:

- no completion choice;
- multiple tool calls in one response;
- a tool call without a valid ID or function payload;
- neither non-blank final text nor a tool call;
- a truncated or otherwise incomplete completion;
- a tools-disabled final completion that still contains a tool call.

The safe final answer is:

```text
暂时无法完成本次咨询，请稍后重试。
```

### Provider and HTTP errors

Existing provider exceptions continue through the current route mapping:

- timeout → HTTP 504;
- connection failure → HTTP 503;
- provider status error → HTTP 502.

Rate limiting and the application-level LLM slot retain their current behavior.
The slot is released on every success and failure path. Successful and safely
terminated Agent results use the normal NDJSON response:

```json
{"type":"delta","content":"complete final answer"}
{"type":"done"}
```

No internal tool message, traceback, path, or raw provider structure is sent to
the frontend.

## Provenance and Contact Privacy

Every successful tool observation retains the complete Day 1 `sources` list.
The LLM may use these trusted `SourceRef` values as factual provenance. Day 2
does not automatically render source titles or URLs, add citation events, or
change the NDJSON contract. When a user explicitly asks for a source, the model
may use a trusted observed `SourceRef`; it must not construct a URL.

`get_contact_info` remains the only tool source for structured contact facts and
continues to exclude the company address. The Agent layer does not add an
address field or maintain a second copy of contact data.

## LLM Service Integration

`services/llm.py` retains its existing streaming functions for compatibility
with current tests and evaluation utilities. It adds one non-streaming Chat
Completions function that:

- uses the existing client, model, timeout, and retry configuration;
- keeps thinking disabled;
- sends native tool schemas when tools are enabled;
- omits tools or sets `tool_choice="none"` for the final tools-disabled call;
- returns the complete assistant message for Agent normalization;
- retains the existing application retry behavior for connection, rate-limit,
  and retryable provider failures.

No streaming tool-call assembly is implemented.

## Planned File Changes

New files:

```text
agent/__init__.py
agent/models.py
agent/executor.py
agent/loop.py
tests/test_agent_models.py
tests/test_agent_executor.py
tests/test_agent_loop.py
```

Modified files:

```text
services/llm.py
main.py
routes/chat.py
tests/test_llm.py
tests/test_main.py
tests/test_chat_route.py
tests/test_chat_http.py
```

No frontend, knowledge source, knowledge schema, Retriever, prompt baseline, or
deterministic tool implementation is modified.

## Testing Strategy

All Agent tests use fake completions and fake tools; automated tests make no
paid LLM or embedding calls.

Required behavioral coverage includes:

1. A direct company-name answer performs no deterministic tool call.
2. An explicit recommendation request calls `search_products`.
3. An exact model parameter question calls `get_product_details`.
4. A contact request calls `get_contact_info`.
5. `PRODUCT_NOT_FOUND` becomes a controlled observation and safe answer.
6. An unexpected callable exception becomes `TOOL_EXECUTION_ERROR` without a
   traceback or internal message.
7. Three processed calls trigger one tools-disabled final completion and never
   a fourth tool execution.
8. An identical successful invocation reuses its observation, does not execute
   twice, and still consumes a call slot.
9. The same tool with different arguments executes for each argument set.
10. Unknown tools, malformed JSON, non-object JSON, and invalid argument shapes
    become controlled observations.
11. Multiple tool calls in one response terminate safely without executing any
    of them.
12. Every assistant tool call is paired with the correct `tool_call_id`.
13. Internal tool history never appears in the frontend response.
14. Initial RAG receives exactly the latest unmodified user message.
15. Successful HTTP output contains exactly one complete `delta` and one `done`.
16. Existing rate-limit, concurrency-release, and provider HTTP mappings remain
    valid.
17. Application startup constructs one Retriever and shares it with initial RAG
    and `search_products`.
18. Existing RAG, retrieval, deterministic-tool, and full repository regression
    suites remain green.

## Completion Criteria

Day 2 is complete when the existing endpoint can run the non-streaming native
Agent Loop, safely execute only registered Day 1 tools, return complete tool
observations to the LLM, perform multiple reasoning steps, enforce the
three-call safety boundary, and emit the complete final answer through the
unchanged NDJSON event schema without changing existing RAG retrieval behavior.
