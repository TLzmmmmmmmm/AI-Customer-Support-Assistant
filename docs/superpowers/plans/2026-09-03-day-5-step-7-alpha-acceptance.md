# Day 5 Step 7 Alpha Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task in the current session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the existing production HTTP route and Astro experience, then request separate approval for paid semantic acceptance.

**Architecture:** Exercise the real FastAPI application, Retriever, NumPy index, entity resolver, context/prompt builders, generation adapter, middleware, logging, and reliability controls using deterministic external-provider responses. Run the existing frontend suite unchanged. Keep live semantic evaluation separate from deterministic protocol/reliability evidence.

**Tech Stack:** unittest, FastAPI TestClient, httpx2 (installed SDK transport), NumPy, Vitest, Astro, Playwright.

**Spec:** `docs/superpowers/specs/2026-09-02-day-5-full-rag-pipeline-design.md`

## Global Constraints

- Preserve `POST /api/chat-stream`, `application/x-ndjson`, and `delta` / `done` / `error` events.
- No frontend redesign, new endpoint, generation provider, logging schema, score threshold, or vector rebuild.
- Keep Top-K=5, latest-message retrieval and bounded full-history generation.
- Do not modify `.env`, curated knowledge, or production vector records.
- No paid API call without a fresh cost report and explicit user approval.
- No merge, push, or unrelated branch cleanup.

## Task 1: HTTP Integration and Reliability Acceptance

**Files:** Create `tests/test_chat_http.py` only. Inspect `main.py`, `routes/chat.py`, `services/llm.py`, `concurrency.py`, `rate_limit.py`, `models.py`, and existing retrieval/context tests.

**Interfaces:** TestClient runs `main.app` with lifespan. Substitute startup's provider composition with a real fixture `Retriever`; use a two-dimensional controlled query embedding and two real `VectorRecord` objects. Patch only the generation client's `client.chat.completions.create` call, keeping its application retries and stream decoder real.

- [ ] Build fixtures with these hand-checked identities and vectors:

```python
records = [
    make_record("hp780", [0.0, 1.0], "# HP780\n\n防护等级：IP68"),
    make_record("hp790ex", [1.0, 0.0], "# HP790EX\n\n防爆机型"),
]
retriever = Retriever(
    embedding_provider=controlled_provider,
    vector_index=NumpyExactVectorIndex(records),
    entity_resolver=ExactEntityResolver.from_records(records),
    default_top_k=5,
)
```

`make_record` returns a validated VectorRecord with product metadata, deterministic hash, HTTP source URL, source file, and explicit embedding model/dimension fields. `controlled_provider.embed_query` records the query and returns `[1.0, 0.0]`, or raises an injected `EmbeddingAPIError`; it never calls a service. This intentionally makes the wrong product denser so explicit HP780 entity handling is exercised.

- [ ] Add TestClient setup with isolated `BoundedSemaphore(1)` and `defaultdict(deque)` rate history. Use `addCleanup`/context managers to restore globals, patches and lifespan even on failure. The app-scoped retriever and SDK boundary are injected before startup; no real external call is possible.

- [ ] Add the following executable HTTP cases with literal expectations:

| Test | Request/fault | Required outcome |
| --- | --- | --- |
| success | bounded history ending in `HP780 的防护等级是什么？` | one embedding query, HP780 evidence first despite adverse dense score, history preserved, only evidence fields in prompt, two delta events then done, request ID header, minimal success log |
| invalid input | blank, 4001 chars, assistant-first, 21 messages, >20000 aggregate chars | 422, safe validation JSON, no embedding/generation |
| concurrency | hold the single real test slot before POST | 503 concurrency_limit, no providers |
| rate limit | set request limit to one and POST twice | first 200, second 429 rate_limit with Retry-After, one embedding |
| retrieval failure | injected EmbeddingAPIError with private detail | 503 retrieval_unavailable, no generation, redacted response/log |
| pre-stream failures | timeout, connection, 401, 429, 500 provider errors | HTTP 504/503/502; timeout/401 once, connection/429/500 twice; slot reusable |
| retry success | one connection error then normal provider stream | 200 and normal NDJSON, two generation attempts, one retrieval |
| in-stream failures | delta then timeout/connection/provider error | HTTP 200, delta then error with request ID, no done and no retry; slot reusable |
| unexpected context defect | RuntimeError while building context | 500 internal_error, no generation, no private detail |

Use provider chunks shaped as:

```python
SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="IP68"))])
```

Check slot reuse using real nonblocking acquire/release after each request. Capture the actual application logger with `assertLogs` for privacy cases; do not replace logging with a fake. These tests characterize already-required behavior, so an initially passing result needs no production mutation.

- [ ] Run focused tests:

```powershell
$env:PYTHONPATH=(Resolve-Path '.venv\Lib\site-packages').Path
$day5Python='C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $day5Python -m unittest tests.test_chat_http -v
```

If a test exposes a defect, diagnose it before changing production behavior and keep any repair within the approved Day 5 contract.

## Task 2: Existing Backend and Frontend Suites

**Files:** No frontend source changes planned. Existing frontend project is `D:/Shengborun`.

- [ ] Run complete backend suite with the same runtime:

```powershell
& $day5Python -m unittest discover -s tests -v
```

- [ ] In `D:/Shengborun`, run the existing commands in order:

```powershell
pnpm run test:unit
pnpm run build
pnpm run test:e2e
```

The Playwright config starts a localhost Astro preview automatically. Chat tests intercept `/api/chat-stream`, so they incur no API costs. Do not substitute these tests for live semantic acceptance. If a sandbox restriction blocks a required command, request execution approval instead of bypassing the restriction. Do not download new dependencies without need and permission.

- [ ] Validate current local vector artifacts with `services.retrieval.build_retriever()` without calling `retrieve`; this constructs clients but makes no embedding request. Record hashes of knowledge/vector artifacts before and after acceptance to confirm no regeneration.

## Task 3: Separate Live Semantic Approval Gate

**Files:** Record evidence in `docs/day-5-alpha-acceptance.md`. No automatically running live test is added to CI.

- [ ] Verify current prices from official pages:
  - `https://api-docs.deepseek.com/zh-cn/quick_start/pricing/`
  - `https://help.aliyun.com/zh/model-studio/embedding`
- [ ] Propose four cases: HP780 known fact; an unsupported company fact; HP780 versus HP790EX cross-product risk; a synthetic instruction-injection RetrievalResult. First three use the real Retriever + DashScope + DeepSeek, the fourth real DeepSeek with controlled evidence, without editing the corpus.
- [ ] Report the expected three query embeddings plus four generation calls, retry allowance, input/output assumptions, and billing method. Existing `deepseek-v4-flash` is authoritative; use high-peak uncached prices for a conservative estimate. Request approval for test-only output limits if needed to control cost; do not change production limits.
- [ ] Stop before calling either paid provider until explicitly approved. Record live acceptance as NOT RUN, not passed. Account balances are not inspected or topped up.

## Task 4: Evidence and Handoff

- [ ] Record test counts, commands, limitations, actual changed files, and remaining live gate in `docs/day-5-alpha-acceptance.md`.
- [ ] Check both repositories with `git status --short`; review backend `git diff --check` and diff. Stage only this step's explicit files, then commit as `test: verify day 5 http alpha acceptance` after fresh test evidence.
- [ ] Report deterministic acceptance separately from semantic acceptance. Do not claim Day 5 complete while paid semantic tests remain unapproved or unexecuted.
