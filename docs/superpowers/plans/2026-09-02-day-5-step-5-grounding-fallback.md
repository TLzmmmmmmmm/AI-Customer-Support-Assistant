# Day 5 Step 5 Grounding and Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give known retrieval infrastructure failures a deterministic HTTP 503 response with no ungrounded generation, while verifying the existing semantic-insufficiency and no-threshold grounding rules.

**Architecture:** Keep retrieval, context construction, prompt construction, and generation unchanged. Add one narrow `RetrievalError` branch inside the existing pre-stream route exception boundary, before the generic 500 branch; reuse the existing HTTP exception handler for response shaping and production logging.

**Tech Stack:** Python 3.12, FastAPI, existing Day 4 retrieval exceptions, existing RAG pipeline, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-02-day-5-full-rag-pipeline-design.md`

## Global Constraints

- Execute only Day 5 Step 5 and stop after its report.
- A known query embedding or retrieval failure returns HTTP 503 with code `retrieval_unavailable`.
- Retrieval failure releases the acquired concurrency slot exactly once and never calls DeepSeek.
- Unexpected Context/Prompt Builder defects remain on the existing 500 path.
- Existing DeepSeek 502/503/504 pre-stream behavior and in-stream NDJSON error behavior remain unchanged.
- Do not add score thresholds, score calibration, intent routing, fallback generation, a fallback model, reranking, query rewriting, or retries.
- Keep the fixed insufficiency sentence exactly `目前公司的资料中没有找到足够信息确认这一点。`
- Preserve complete ordered Top-K evidence even when cosine similarity is low or negative.
- Do not change production logging fields, HTTP/NDJSON provenance, frontend code, retrieval code, prompts, knowledge artifacts, or vector artifacts.
- Do not call DashScope, DeepSeek, or any paid API.

## File Structure

- Modify `routes/chat.py`: classify `RetrievalError` before the generic exception branch.
- Modify `tests/test_chat_route.py`: cover representative retrieval exception subclasses, 503 details, no generation, exact slot release, and low-score evidence preservation.
- Re-run existing `tests/test_prompts.py`: verifies empty evidence retains the fixed semantic-insufficiency rule.
- Re-run existing `tests/test_rag_context.py`: verifies all ordered evidence is projected without thresholding or deletion.

---

### Task 1: Map known retrieval failures without generation

**Files:**
- Modify: `routes/chat.py`
- Modify: `tests/test_chat_route.py`

**Interfaces:**
- Consumes: any `RetrievalError` raised before `open_chat_stream()` by query embedding, vector search, or entity resolution.
- Produces: FastAPI `HTTPException(status_code=503)` with `detail.code == "retrieval_unavailable"`, a stable public message, and backend-only `internal_error` for the existing logging handler.

- [ ] **Step 1: Write the failing failure-classification test**

Import `EmbeddingAPIError`, `VectorIndexNotReadyError`, `EntityCatalogError`, and FastAPI `HTTPException`. Add a table-driven test using the existing `FakeRetriever`:

```python
for retrieval_error in (
    EmbeddingAPIError("provider failed"),
    VectorIndexNotReadyError("index failed"),
    EntityCatalogError("entity resolver failed"),
):
    with self.subTest(error_type=type(retrieval_error).__name__):
        with self.assertRaises(HTTPException) as caught:
            chat.chat_stream(
                self.payload,
                self.request,
                None,
                FakeRetriever(events=events, error=retrieval_error),
            )
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.detail, {
            "code": "retrieval_unavailable",
            "message": "服务暂时不可用，请稍后再试。",
            "internal_error": type(retrieval_error).__name__,
        })
```

For each subcase, assert acquisition precedes retrieval, `release_llm_slot()` is called exactly once, and `open_chat_stream()` is never called. Keep the existing `RuntimeError` test unchanged to prove unexpected orchestration defects are still re-raised for the global 500 handler.

- [ ] **Step 2: Run the new test and verify RED**

```powershell
$env:PYTHONPATH = (Resolve-Path -LiteralPath '.venv\Lib\site-packages').Path
$day5Python = 'C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $day5Python -m unittest tests.test_chat_route -v
```

Expected: the new subcases fail because the route currently re-raises the original retrieval exceptions through its generic branch. Existing success and unexpected-error tests remain green.

- [ ] **Step 3: Implement the minimal exception branch**

Import `RetrievalError` beside `Retriever` and add this branch after the pre-stream `try` body but before OpenAI and generic exception branches:

```python
except RetrievalError as error:
    release_llm_slot()
    raise HTTPException(
        status_code=503,
        detail={
            "code": "retrieval_unavailable",
            "message": "服务暂时不可用，请稍后再试。",
            "internal_error": type(error).__name__,
        },
    )
```

Do not call `log_request()` directly; the existing `http_exception_handler()` already records request ID, status, outcome, latency, and error type while excluding `internal_error` from the public response.

- [ ] **Step 4: Run route tests and verify GREEN**

Run `tests.test_chat_route` again. Expected: all success, known retrieval failure, and unexpected defect tests pass.

### Task 2: Verify semantic insufficiency and absence of thresholds

**Files:**
- Modify: `tests/test_chat_route.py`
- Verify: `tests/test_prompts.py`
- Verify: `tests/test_rag_context.py`

**Interfaces:**
- Consumes: low/negative finite retrieval scores and an empty evidence list.
- Produces: unchanged evidence projection and unchanged fixed insufficiency instruction; no new production code.

- [ ] **Step 1: Make the route success fixture exercise a negative score**

Change only the controlled `retrieval_result()` score from `0.97` to `-0.25`. Keep the expected final RAG payload unchanged. The existing success test then proves a low score does not remove the hit before DeepSeek.

- [ ] **Step 2: Run focused grounding tests**

```powershell
& $day5Python -m unittest tests.test_chat_route tests.test_rag_context tests.test_prompts -v
```

Expected: all tests pass. Specifically, the route still passes the negative-score hit as `type/section/text`; Context Builder still preserves all five ordered hits; Prompt Builder with empty evidence still contains `目前公司的资料中没有找到足够信息确认这一点。`

- [ ] **Step 3: Re-run startup-failure and generation compatibility tests**

```powershell
& $day5Python -m unittest tests.test_retrieval_service tests.test_vector_records tests.test_llm tests.test_main -v
```

Expected: local artifact startup validation and DeepSeek provider-message behavior remain unchanged.

### Task 3: Full verification and Step 5 commit

**Files:**
- Verify only `routes/chat.py` and `tests/test_chat_route.py` changed in implementation.

**Interfaces:**
- Consumes: completed Step 5 behavior.
- Produces: one tested Step 5 commit; no Step 6 observability changes.

- [ ] **Step 1: Run full backend regression**

```powershell
& $day5Python -m unittest discover -s tests -v
```

Expected: all tests pass without external API access.

- [ ] **Step 2: Audit exact scope**

```powershell
git diff --check
git status --short
git diff --stat
git diff --name-only
```

Expected implementation changes are exactly `routes/chat.py` and `tests/test_chat_route.py`. Confirm logging, error response shaping, prompts, context builder, Retriever, embedding provider, LLM client, startup, configuration, frontend, knowledge, vectors, and evaluation artifacts remain unchanged.

- [ ] **Step 3: Commit Step 5**

```powershell
git add -- routes/chat.py tests/test_chat_route.py
git diff --cached --check
git commit -m "feat: fail closed when retrieval is unavailable"
```

- [ ] **Step 4: Report and stop**

Report inspected files, pre-existing generic failure behavior, exact 503 mapping, no-generation evidence, fixed insufficiency/no-threshold evidence, RED/GREEN results, final regression count, unchanged protocol/logging, and the Step 6 concern: verify request-lifecycle provenance remains available while production logs and responses retain their data-minimized schema. Stop without implementing Step 6.
