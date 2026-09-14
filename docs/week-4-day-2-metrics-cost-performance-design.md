# Week 4 Day 2 Metrics, Cost, and Performance Analysis Design

Date: 2026-09-14

## Goal

Run a small, reproducible production-style workload through the real local
`POST /api/chat-stream` path, join each attempt to the existing Day 1 request
summary by `request_id`, and produce machine-readable analysis plus a Markdown
report that supports exactly one Day 3 recommendation.

This is offline performance analysis, not route-accuracy evaluation and not a
production observability redesign.

## Scope and non-goals

The implementation will:

- make one small Day 1 telemetry extension for provider-reported DeepSeek cache
  usage;
- add a standalone HTTP workload runner with a configurable base URL;
- add an offline request-summary parser, joiner, analyzer, and Markdown report
  renderer;
- execute a sequential local production-style run using real model, retrieval,
  and tool paths;
- preserve missing, failed, and incomplete observations in coverage reporting.

It will not modify `evaluation/`, `eval/`, evaluation data or runners, routing
or prompt behavior, the production API or NDJSON schema, true streaming,
retrieval behavior, tool behavior, dashboards, notebooks, telemetry services,
pricing services, or load-testing infrastructure.

## Approved architecture

The local application server, workload runner, and analyzer remain separate.

1. A local uvicorn process serves the existing application and writes stderr to
   a run-specific log file.
2. The runner sends real HTTP requests to a configurable base URL, validates
   the existing NDJSON contract, captures `X-Request-ID`, and writes local
   request metadata as JSONL.
3. The analyzer reads the workload JSONL and either the raw production log or
   exported JSONL summary records, joins on `request_id`, computes diagnostics,
   and writes the final JSON and Markdown artifacts.

The runner does not read telemetry from the HTTP response and production does
not expose a telemetry endpoint.

## Minimal production telemetry extension

DeepSeek's successful Chat Completion usage contains the top-level fields
`prompt_cache_hit_tokens` and `prompt_cache_miss_tokens`. The installed OpenAI
SDK accepts and exposes these provider-specific fields because its usage model
allows extra fields.

The existing request-local accumulation in `trace_models.py` will add optional
totals with the same names. They will be copied into the existing `RouteTrace`
and emitted by the existing single request-summary log.

For each successful Chat Completion:

- both fields must be non-negative integers;
- their sum must equal `prompt_tokens`;
- valid values are accumulated across all successful model calls;
- if either value is missing or invalid, or the equality fails, both
  request-level cache totals become `null` and remain `null`.

Cache completeness is independent from the existing input/output token
completeness. No cache split is inferred from total input tokens. Logging stays
best-effort, and the lower-level LLM boundary receives no FastAPI `Request`.

Sources:

- https://api-docs.deepseek.com/api/create-chat-completion/
- https://api-docs.deepseek.com/zh-cn/quick_start/pricing/

## Workload manifest

`performance/workload_v1.json` will contain:

- 24 single-turn cases: four distinct prompts for each of the six target routes
  `direct`, `knowledge`, `exact_product`, `product_search`, `contact`, and
  `fallback`;
- a repeat count of five for each single-turn case, producing 120 planned
  single-turn requests;
- six independent calibration prompts, one per target route, that are not
  reused by the measured single-turn cases;
- four three-turn synthetic conversations, producing 12 planned multi-turn
  requests.

The four conversations cover:

1. product need clarification, candidates, and a specific-model follow-up;
2. solution consultation, scenario refinement, and a product suggestion;
3. a known-model query, comparison follow-up, and sales contact;
4. general consultation, a capability-boundary follow-up, and an unsupported
   requested action.

Target routes construct a representative workload only. They are not asserted,
scored, or used for aggregation. All route analysis uses the actual production
route from telemetry.

The manifest loader validates only reusable structural invariants: unique IDs,
supported routes, non-empty text, a positive repeat count, and valid non-empty
conversations. The experiment-specific counts of 24 measured prompts, five
repeats, six calibration prompts, and four three-turn conversations are tested
as properties of `workload_v1.json`, not hard-coded as generic loader rules.

Single-turn executions are interleaved with a fixed random seed. Conversations
are interleaved with each other while preserving turn order inside each
conversation. A later turn includes all prior user messages and the real prior
assistant answers using the existing alternating history schema.

## Workload runner

`scripts/run_week4_day2_workload.py` will:

- default to `http://127.0.0.1:8000` and accept a different base URL;
- require explicit `--execute` for paid requests;
- refuse to overwrite an existing run output;
- run with concurrency one and a default interval of at least 6.5 seconds to
  respect the current 10-request rolling 60-second application limit;
- use the standard library HTTP client rather than add a dependency;
- accept HTTP error responses as recorded workload outcomes;
- require and record `X-Request-ID` when the server returned one;
- parse only valid `delta`, optional `citations`, and terminal `done` events for
  successful NDJSON responses;
- keep assistant answer text in memory only for conversation history;
- never copy prompts, answers, retrieved text, system prompts, or tool arguments
  into raw result rows;
- perform no runner-level retries.

A single-turn failure is recorded and execution continues. If a conversation
turn fails, the remaining turns in that conversation are recorded as skipped
because no assistant response may be fabricated.

Each attempted row records only run/case identifiers, target route,
single-turn or multi-turn kind, optional conversation ID and turn index,
timestamps, HTTP status, transport/NDJSON outcome, and request ID. A run header
records manifest identity, base URL, seed, timing configuration, and start time.

## Joining telemetry

The raw log parser identifies the existing request-summary line by its required
keys and parses its logger timestamp plus compact key/value fields. It ignores
unrelated uvicorn lines. The exported-record reader accepts one JSON object per
line with equivalent summary fields.

The join is a left join from attempted workload rows to summaries. The analyzer
reports:

- attempts without a request ID;
- request IDs with no matching summary;
- duplicate summaries for a request ID;
- malformed candidate summary lines;
- summaries in the server log that are unrelated to the workload.

No missing or duplicate observation is silently selected or discarded.

## Analysis populations and percentile definition

All primary request-level latency, token, tool, and cost metrics include both
single-turn and multi-turn successful requests. The report also shows their
separate sample counts.

Success means a joined production summary with HTTP 200 and outcome `success`.
Failures and unmatched attempts are reported separately and never mixed into
primary successful-request latency percentiles.

P50 and P95 use nearest rank for every metric:

```text
rank = ceil(p / 100 * N)
```

after ascending sort, using one-based ranks. Every percentile is accompanied
by its contributing sample count. Route P95 values are diagnostics, not SLOs.

## Latency and stage diagnostics

The analyzer reports overall successful-request total-latency count, P50, and
P95, then the same values by actual route.

For router, retrieval, tool, and model latency, each actual route reports
non-null count, mean, P50, and P95. Router-type groups report count and total
latency P50/P95. Router-type comparison is descriptive rather than causal
because route mix may differ.

For each request, `dominant_stage` is the non-null measured stage with the
largest latency, and its diagnostic ratio is:

```text
dominant_stage_ratio = max(non-null stage latency) / total_latency_ms
```

The ratio is used only to determine whether a single measured stage dominates.
Because stage metrics can be sequential and can overlap, a low ratio means only
that no single measured stage dominates. Stage metrics are never summed,
presented as mutually exclusive percentages, or used to infer unexplained or
uninstrumented latency.

## Token and tool diagnostics

Primary token averages use successful requests whose `input_tokens` and
`output_tokens` are both non-null. Nulls are excluded rather than converted to
zero. Coverage is reported as complete records over eligible successful
requests, including the excluded count.

The analyzer reports overall average input/output/total Chat Completion tokens
and the same averages by actual route. Embedding tokens are excluded.

Tool analysis uses `tool_execution_count`, never legacy `tool_call_count`. It
reports overall and route averages, the distribution of execution counts, and
common ordered `executed_tool_names` sequences for agentic/product-search
traffic. The report retains legacy ToolTrace counts only as existing telemetry,
not as the requested tool-call metric.

## Failures

The report includes attempted, successful, failed, skipped, and unmatched
counts; failure rate over attempted requests; and distributions of existing
`failure_layer` and `failure_code`. It does not invent failure categories.

Measured tokens and cost from failed requests are reported separately when
complete. They do not enter the main successful-request token or cost averages.

## Pricing and cost

The offline pricing snapshot uses DeepSeek's official Chinese documentation as
of 2026-09-14, currency CNY, for configured model `deepseek-v4-flash`:

| Period | Cache-hit input / 1M | Cache-miss input / 1M | Output / 1M |
| --- | ---: | ---: | ---: |
| Off-peak | CNY 0.05 | CNY 1.50 | CNY 4.50 |
| Peak | CNY 0.10 | CNY 3.00 | CNY 9.00 |

Peak time is Monday-Friday, 09:00-12:00 and 14:00-18:00 Asia/Shanghai. The
implementation treats these as half-open intervals `[09:00, 12:00)` and
`[14:00, 18:00)`; all other times are off-peak. Each request's logger timestamp
chooses its rate.

Per-request estimated cost requires non-null cache-hit, cache-miss, and
output token totals plus a valid timestamp:

```text
hit_tokens * hit_rate / 1_000_000
+ miss_tokens * miss_rate / 1_000_000
+ output_tokens * output_rate / 1_000_000
```

The field name is `estimated_cost_cny`. The report calls this an estimate,
never actual or exact billed cost. It reports cost
coverage, mean/P50/P95 cost per successful request, and route-level cost. If
the real run reveals incomplete cache telemetry, those records are excluded
from estimated cost averages and the limitation is explicit; no input split is
inferred.

Each complete synthetic conversation's `estimated_cost_cny` is the sum of its
request-level estimates. The report lists scenario estimates, conversation
count, average, and nearest-rank median. Interrupted conversations are listed
with `partial_estimated_cost_cny` and excluded from complete-conversation
averages.

## Slow requests and bottleneck conclusion

The ten slowest joined attempts are shown using only request ID, actual route,
router type, total and stage latencies, token totals, tool execution count and
names, and failure status. No prompt or response content appears.

The report directly answers all requested bottleneck questions. There is no
pre-existing production latency SLO, so it makes diagnostic comparisons rather
than pass/fail claims. The final recommendation is an explicit engineering
judgment stored with supporting measurements in both artifacts and must be
exactly one of:

- A. True streaming is justified
- B. Tool/agent execution optimization is justified
- C. Another specific measured bottleneck should be addressed
- D. No material optimization is currently justified

No hidden score or retrospective SLO chooses the category. Ambiguous evidence
results in D. Lack of TTFT telemetry is stated as a limitation when considering
true streaming.

## Execution safety

The approved paid-call ceiling is CNY 5, with no recharge or resource purchase.
Before the full run, a six-request calibration uses the manifest's six
independent calibration prompts, one per target route. These prompts never
appear in the formal 24-prompt measured workload and therefore do not warm an
identical measured prompt before the run. Average measured calibration cost
multiplied by 132 and calibration tokens repriced at peak rates are labeled only
as a `budget screening projection`. They determine whether expected spend is
clearly below CNY 5; they are not an upper bound. Execution stops for renewed
approval if either projection approaches CNY 5 or if balance/configuration
errors occur.

The full run is sequential and has no automatic retries. Calibration request
IDs are not present in the full workload mapping, so their log lines cannot
enter the formal analysis.

## Artifacts and repository layout

- `performance/workload_v1.json`: tracked manifest.
- `performance/analysis.py`: pure parsing, joining, aggregation, cost, and
  report-rendering logic.
- `scripts/run_week4_day2_workload.py`: HTTP runner CLI.
- `scripts/analyze_week4_day2.py`: analyzer CLI.
- `performance/runs/`: ignored raw mappings and server logs.
- `performance/results/week4-day2-analysis.json`: tracked machine-readable
  final result.
- `docs/week-4-day-2-performance-report.md`: tracked human report.

The `.gitignore` will explicitly ignore raw run files and allow only the agreed
manifest, compact analysis result, and final report.

## Tests and verification

Focused offline tests cover:

- nearest-rank P50/P95;
- raw log and exported JSONL parsing;
- route and stage aggregation;
- null token exclusion and coverage;
- `tool_execution_count` and ordered tool sequences;
- complete and interrupted conversation grouping;
- peak/off-peak boundaries;
- estimated cache-hit/cache-miss/output cost;
- missing and duplicate summary joins;
- runner manifest ordering, NDJSON validation, metadata privacy, and failure
  behavior without real network calls.

Focused production tests cover cache-token accumulation, missing/invalid split
propagation, and summary emission. Relevant production regression tests and the
full test suite run after the optional telemetry extension.

After tests pass, execution proceeds in this order:

1. start the local production app with a run-specific stderr log;
2. run and analyze the six-request calibration;
3. verify the budget screening projection is clearly below CNY 5;
4. run the full workload;
5. join and analyze telemetry;
6. generate and inspect both final artifacts;
7. stop the local server and report results and limitations.

## Success criteria

- Planned workload requests use the real local production path at concurrency
  one and stay within the approved cost ceiling.
- Every attempt is represented as success, failure, skipped, missing telemetry,
  or another explicit outcome.
- All requested latency, stage, token, tool, cost, conversation, failure, and
  slow-request sections are present with counts and coverage.
- Aggregation uses actual routes and nearest-rank percentiles.
- The final report ends with exactly one supported A/B/C/D recommendation.
- No evaluation semantics or unrelated production behavior changes.

The final limitations state: "Repeated test cases may increase prompt cache
reuse; observed estimated cost reflects the measured cache behavior of this
representative synthetic workload."
