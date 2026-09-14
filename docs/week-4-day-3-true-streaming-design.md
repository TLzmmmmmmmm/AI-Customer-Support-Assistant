# Week 4 Day 3 True Streaming and Router Guardrail Design

Date: 2026-09-14

## Goal

Add a 16-token hard output guardrail to the LLM fallback router and implement
safe provider-to-HTTP streaming for eligible final-answer paths. Preserve the
existing routing, final-answer, citation, error, and telemetry semantics while
adding end-to-end server-side time-to-first-visible-delta and per-request
buffering-saved measurements.

Day 3 is a controlled experiment as well as an implementation task. The
streaming path remains enabled only if measured TTFT improvement clears the
approved threshold without reliability, finalization, or material
total-latency regression.

## Scope and non-goals

The implementation will:

- enforce `ROUTER_MAX_TOKENS = 16` for the LLM fallback classifier;
- replace buffered final generation with true provider streaming for direct,
  non-agentic knowledge, and non-agentic deterministic-tool routes;
- use LangGraph native custom-stream events to carry only prefix-safe final
  answer text;
- preserve agentic, router-classification, and tool-selection completions as
  buffered internal operations;
- add `first_delta_latency_ms` and `buffering_saved_ms` to the existing
  request-local telemetry and single request summary;
- collect a controlled buffered baseline and streaming-after comparison.

It will not optimize retrieval, tools, prompts, context size, answer length,
or the agent loop. It will not change the public NDJSON vocabulary, add a
second model/client, upgrade LangGraph, add an event bus, worker pool,
WebSocket, Redis, background task, or telemetry framework, or modify
`evaluation/` and `eval/`.

## Verified current architecture

The installed LangGraph version is 1.2.11. Its compiled graph exposes native
`stream(..., stream_mode=...)`, and synchronous graph nodes can publish custom
events through `langgraph.config.get_stream_writer()`.

The graph is compiled during FastAPI lifespan and is currently invoked through
`GraphRouteOrchestrator.run()`, which calls `compiled_graph.invoke()` before
the route returns a `StreamingResponse`. The response currently contains one
buffered answer delta, optional citations, and done.

The current final HTTP boundary calls `render_answer()` after graph completion.
That function strips leading/trailing whitespace, removes raw and angle URLs,
rewrites Markdown links to their labels, truncates reference sections,
validates and deduplicates trusted citations, and substitutes a safe fallback
when sanitization removes the whole answer. Citations are a separate NDJSON
event. The frontend in `D:\Shengborun` already concatenates multiple delta
events and rejects streams that end without done.

The request summary is currently emitted from the response iterator, but graph
execution and request-local ContextVar cleanup occur before that iterator
starts. Day 3 must move ownership so streaming telemetry remains bound until
the graph and HTTP stream reach a terminal state.

## Router guardrail

`routing/router.py` will define:

```python
ROUTER_MAX_TOKENS = 16
```

The router instruction becomes:

```text
Return exactly one label from this list and nothing else:
```

`complete_chat()` gains `max_tokens: int | None = None` and adds the provider
argument only when it is non-null. The fallback classifier calls
`complete_chat(..., max_tokens=ROUTER_MAX_TOKENS)`.

Deterministic routing continues to bypass the LLM classifier. The existing
strict `Route(...)` parsing, six-label vocabulary, finish-reason checks, and
invalid or truncated output to fallback behavior remain unchanged.

## Streaming execution architecture

`GraphRouteOrchestrator.run()` remains available for existing callers. A new
lazy route-iterator interface builds the same initial graph state and consumes
the compiled graph with custom and value stream modes.

Only these nodes publish custom final-answer chunks:

- `direct_node`;
- `rag_generate_node` for non-agentic knowledge;
- `deterministic_generate_node` after a deterministic tool result.

`route_node`, `agent_step_node`, `execute_tool_node`, and deterministic tool
execution never publish content. Agentic requests and static fallback complete
during prefetch and retain their existing single buffered answer delta.

The HTTP route creates the lazy graph iterator and advances it before creating
the `StreamingResponse`:

```text
request setup
  -> create graph iterator
  -> prefetch first safe custom answer chunk or graph terminal result
     -> exception before first chunk: existing non-2xx HTTP handler
     -> graph completed without chunks: buffered response
     -> safe chunk available: create 200 streaming response
  -> advance graph through response iterator
  -> terminal RouteExecutionResult
  -> full render and invariant verification
  -> optional citations
  -> done
  -> one request summary
  -> cleanup
```

Prefetch is required because Starlette sends response headers before it starts
iterating a `StreamingResponse`. Without prefetch, an error before the first
answer chunk would incorrectly become an HTTP 200 stream error instead of the
existing 502/503/504 response.

The graph and lower modules do not receive a FastAPI `Request`. Each advance
of the lazy graph iterator temporarily binds the existing request state to the
current execution context and resets it before returning. Telemetry values
continue to accumulate on request state. This avoids relying on a ContextVar
token surviving across FastAPI/Starlette thread boundaries.

## Shared prefix-safe answer sanitization

The current buffered sanitizer and incremental path will use one
implementation. The existing text rules will be extracted into shared grammar
and state transitions rather than duplicated as an approximate streaming
sanitizer.

The incremental interface is conceptually:

```python
sanitizer.feed(raw_chunk) -> tuple[str, ...]
sanitizer.finish() -> str
```

`sanitize_generated_answer(answer)` feeds the full answer through the same
state machine and joins its safe output. `render_answer()` continues to call
that buffered API, so citation and fallback policy stay unchanged.

The state machine emits only prefix-safe characters: after text is emitted,
no future continuation can cause `render_answer()` to remove or rewrite it.
It provides these guarantees:

- leading whitespace remains pending until safe body text is confirmed and is
  then discarded;
- potential trailing whitespace remains pending, is emitted only when a later
  safe non-whitespace character proves it internal, and is discarded at EOF;
- `http://`, `https://`, and `www.` candidates remain pending through their
  complete raw URL token and are discarded when confirmed;
- angle-URL candidates remain pending until confirmed or disproved;
- Markdown links use explicit parsing state from `[` through label, `](`,
  optional whitespace, scheme, URL, and closing `)`. No fixed tail length is
  assumed. A disproved candidate is reprocessed under the ordinary text and
  raw-URL rules;
- at each line start, possible reference-heading prefixes remain pending. A
  completed heading causes that line and all subsequent raw text to be
  discarded;
- no visible delta is emitted until at least one safe body character exists;
- if sanitization yields no body, graph execution produces no custom delta and
  the existing final `render_answer()` supplies the same safe fallback.

The complete raw provider answer is retained in graph state. At graph
completion, the HTTP boundary calls the existing `render_answer()` on that raw
answer and checks:

```text
concatenated_user_visible_deltas == rendered.text
```

This is an invariant check, not a repair mechanism. A mismatch after streaming
begins terminates with the existing safe `internal_error` stream semantics;
citations and done are not sent.

If implementation reveals a current final-answer transformation that cannot
be represented prefix-safely, work stops before an approximation is shipped
and the Day 3 closeout selects C.

## Provider streaming and usage

The existing DeepSeek/OpenAI-compatible client remains the only provider
client. The existing `stream_chat()` path will be consolidated for final
generation rather than adding another streaming client.

Final streaming requests use:

```python
stream=True
stream_options={"include_usage": True}
extra_body={"thinking": {"type": "disabled"}}
```

The helper yields only non-empty provider content. It retains the current
retryable status policy. A connection or retryable status failure may retry
before any content is yielded. Once any content is yielded, every provider
failure is terminal and the model call is never retried.

`record_model_usage(usage)` will become the shared usage accumulator;
`record_model_response(completion)` delegates to it. Streaming records only
the final provider usage object and records it once. Existing input/output and
cache hit/miss completeness semantics remain unchanged, and missing values are
never estimated. Total elapsed provider time, including pre-content retries,
continues to accumulate in `model_latency_ms`.

## Streaming latency telemetry

The existing request state, `RouteTrace`, request-summary formatter, and Day 2
summary parser gain two optional fields:

```text
first_delta_latency_ms
buffering_saved_ms
```

It measures monotonic elapsed time from `request.state.started_at`, set by the
existing middleware, to immediately before the first non-empty user-visible
NDJSON delta is yielded. It is recorded once. It remains null when no visible
delta is emitted.

This is server-side time to the first application-visible delta. It is not
browser rendering latency.

`buffering_saved_ms` is a per-request derived measurement for a successfully
completed response that emitted a visible delta and then successfully yielded
`done`:

```text
buffering_saved_ms = done-yield elapsed time - first-delta elapsed time
```

Both elapsed values use the same monotonic request start and are subtracted
before rounding. Equivalently, it measures the server-side interval for which
the pre-Day-3 buffered response would still have withheld already-safe visible
answer content. It remains null if there is no visible delta or no successfully
yielded `done`. Buffered paths therefore normally record a value near zero.
This is not a stage-latency gap and makes no claim about uninstrumented
overhead or the additivity of router, retrieval, tool, and model timings.

Existing `total_latency_ms` retains its separate request-terminal meaning. On
successful streams, its terminal timestamp and the buffering-saved terminal
timestamp are taken immediately after the response iterator resumes from the
successful `done` yield and before the summary is emitted.

Streamed paths normally snapshot this value when the graph finalizer builds
`RouteTrace`. Buffered agentic and fallback paths complete the graph before
their first delta, so the HTTP boundary updates the frozen trace with the value
recorded immediately before that delta. After `done` is yielded successfully,
the HTTP boundary similarly updates the frozen trace with
`buffering_saved_ms` before logging it. No new telemetry service is introduced.

## Terminal states, logging, and cleanup

Before the first delta, existing exception-to-HTTP mappings remain intact.
After the first delta, headers have committed to HTTP 200. Provider or internal
failures emit the existing NDJSON error shape with a safe public code and
message, emit neither citations nor done, and log the real `http_status=200`
with the corresponding `outcome` and `failure_code`.

On client disconnect, no further body event is attempted. The summary records
`http_status=200`, `outcome=stream_interrupted`,
`failure_code=stream_interrupted`, and `failure_layer=null`.

The response owner maintains explicit terminal and logged flags. Its single
`finally` path closes the graph/provider iterator, resets any request-local
binding in the same context that created it, releases the concurrency slot
once, and emits a request summary if a terminal path has not already done so.
Successful order is:

```text
last delta -> optional citations -> done successfully yielded
           -> one complete summary -> cleanup
```

The success summary must not be emitted before `done`. The response iterator
marks the successful terminal state at the `done` boundary, yields `done`, and
only after that yield completes emits the one summary. Its `finally` path still
guarantees one summary for error or interruption paths, but a stream that never
yields `done` is never logged as successful.

The summary therefore includes the final route, router type, retrieval/tool
metrics, model latency, provider tokens, cache tokens, TTFT, citation status,
buffering saved, and failure fields. No second summary logger or telemetry
framework is added.

## Test strategy

Router tests prove the constant, provider argument, deterministic bypass,
unchanged valid six-label parsing, and invalid/truncated fallback.

Provider tests use fake SDK chunks and cover ordered content, final usage once,
cache completeness, accumulated duration, thinking disabled, retry before
content, no retry after content, and partial failure without duplicated text.

Sanitizer tests compare incremental concatenation with buffered
`render_answer()` for ordinary text, leading/trailing whitespace, raw and angle
URLs, Markdown links, reference headings, and fully removed answers. Each case
runs as one chunk, at every single split position, and under deterministic
random multi-split patterns, including splits inside schemes, labels, `](`,
headings, and whitespace.

Graph tests prove custom final-answer events for the three eligible paths and
the absence of events for router output, tool calls and observations, agent
internal turns, and agentic final generation. Existing route selection remains
unchanged.

HTTP tests cover multi-delta ordering, concatenated-answer equivalence,
citations then done, unchanged request ID, no internal leakage, non-null TTFT,
null TTFT when no delta exists, per-request buffering-saved semantics, pre- and
post-first-delta errors, disconnect, one terminal summary strictly after a
successful `done` yield, and request-context and slot cleanup.

The existing frontend unit and E2E contract tests in `D:\Shengborun` will run
to verify multiple-delta concatenation and incomplete-stream behavior. One
E2E regression case is added for a response that emits at least one delta and
then reaches EOF without `done`. It must show the interruption error, remove
the partial assistant bubble, and prove through the next request body that
neither the failed user turn nor partial assistant text was committed to API
conversation history. This is distinct from the existing post-delta explicit
error-event rollback test. The complete backend suite remains the final
regression gate.

## Controlled performance comparison

The experiment uses six tracked single-turn prompts: two each for actual
`product_search`, `knowledge`, and `direct`. Each prompt runs three times with
fixed ordering, concurrency one, a 6.5-second interval, and no runner retry.

The baseline phase first adds the router guardrail and TTFT telemetry while
retaining the current buffered one-delta behavior, then executes 18 requests.
The after phase enables true streaming and repeats the identical 18 requests.
No full Day 2 workload is repeated.

The maximum new paid workload is 36 requests under a CNY 0.50 ceiling. Work
stops if the projection approaches the ceiling or a balance/configuration
failure occurs.

The measurement is valid only if all 36 requests succeed: 18/18 baseline and
18/18 after. Each phase must contain six successful observations for each
actual route (`product_search`, `knowledge`, and `direct`); a route mismatch,
missing request, failed request, missing `done`, or finalization-invariant
failure means the 36/36 gate did not pass.

For each phase and actual route, the report includes successful/failed counts,
TTFT P50/P95, `buffering_saved_ms` P50/P95, total-latency P50/P95, model
latency, and output tokens. The two slow-path TTFT gates require both
`product_search` and `knowledge` to:

- reduce TTFT P50 by at least 30%;
- reduce TTFT P50 by at least 500 ms.

Separately, the all-route total-latency gate applies to every route, including
the `direct` control:

```text
after total_latency_ms P50 <= baseline total_latency_ms P50 * 1.15
```

Conclusion A therefore requires all of the following explicitly: 36/36
successful requests, six observations per actual route per phase, preserved
answer/finalization invariants, both slow-path TTFT gates, and the <=15%
total-latency gate on all three routes. A working implementation that misses
any materiality or regression gate selects B; unsafe finalization semantics
select C.

`direct` is the control route. The comparison does not claim that streaming
reduces model-generation or total latency unless the measurements show it.
Repeated identical prompts can increase provider cache reuse and are reported
as a limitation rather than treated as causal evidence.

Tracked artifacts are:

- `performance/day3_streaming_workload_v1.json`;
- `performance/results/week4-day3-streaming-analysis.json`;
- `docs/week-4-day-3-streaming-closeout.md`.

Raw baseline/after mappings and server logs remain under ignored
`performance/runs/` and contain no prompts or answers.

## Closeout decision

The report ends with exactly one conclusion selected after measurement:

- A. True streaming materially improves perceived latency and should remain
  enabled.
- B. Streaming works but measured TTFT benefit is too small to justify the
  complexity.
- C. Streaming cannot safely preserve current finalization/citation semantics
  without disproportionate architectural change.

No conclusion is chosen before the after measurement completes.

## Success criteria

- The router provider call enforces the 16-token maximum without changing
  routing semantics.
- Eligible routes expose only prefix-safe final answer deltas from the real
  provider stream.
- Agent, router, tools, retrieval context, system prompts, and arguments never
  reach the client.
- Incremental and buffered finalization are identical under adversarial chunk
  boundaries.
- Provider token/cache and model-duration telemetry remain complete.
- `first_delta_latency_ms` follows its server-side definition.
- `buffering_saved_ms` is recorded per request only after a visible delta and
  successful `done`, and is null for every no-done path.
- Before-first-delta HTTP behavior and post-first-delta NDJSON errors follow
  their approved semantics.
- A successful request summary occurs exactly once and only after `done` has
  been yielded successfully; error and interruption paths also log once.
- A delta followed by EOF without `done` cannot commit the failed turn or its
  partial assistant text to frontend API conversation history.
- Context and concurrency ownership clean up on success, failure, and client
  interruption.
- The performance conclusion enforces 36/36 success and the <=15%
  total-latency P50 gate separately for all three actual routes.
- Both controlled phases stay within 36 calls and CNY 0.50.
- Backend and frontend regression suites pass.
- No unnecessary dependency, infrastructure, evaluation change, or unrelated
  refactor is introduced.
