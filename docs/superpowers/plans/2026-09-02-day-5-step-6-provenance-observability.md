# Day 5 Step 6 Provenance and Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lock in the existing backend provenance and privacy boundaries without adding persistent retrieval logs or changing the chat response protocol.

**Architecture:** The route continues to hold complete `RetrievalResult` objects until the Context Builder boundary, while only `type`, `section`, and `text` enter the LLM prompt. Focused tests will prove full provenance survives to that boundary, success and failure logs keep the existing minimal schema, and HTTP/NDJSON responses do not expose internal retrieval details.

**Tech Stack:** Python 3, FastAPI/Starlette exceptions and responses, Pydantic models, `unittest`, `unittest.mock`.

**Spec:** `docs/superpowers/specs/2026-09-02-day-5-full-rag-pipeline-design.md`

## Global Constraints

- Preserve `POST /api/chat-stream`, `application/x-ndjson`, and the existing `delta`, `done`, and `error` event protocol.
- Preserve complete backend `RetrievalResult` provenance during the request lifecycle.
- Keep LLM evidence limited to `type`, `section`, and `text`.
- Keep production logging limited to request ID, HTTP status, outcome/result, latency, and error type.
- Do not persist or log raw queries, query hashes or lengths, chunk IDs, parent IDs, ranks, scores, match origins, source URLs, source files, or chunk text.
- Do not add source, citation, or provenance response events.
- Do not modify production code unless a focused test exposes a genuine contract gap.
- Do not call DashScope or DeepSeek.

---

### Task 1: Lock Full Provenance at the Context Boundary

**Files:**
- Modify: `tests/test_chat_route.py`

**Interfaces:**
- Consumes: `chat.chat_stream(...)`, `RetrievalResult`, `build_retrieved_context(results)`.
- Produces: a route-level regression test proving the exact result object, including all backend provenance fields, reaches the Context Builder while the provider payload and NDJSON response remain minimized.

- [ ] **Step 1: Make the route fake return a caller-owned result**

Extend only the test fake:

```python
class FakeRetriever:
    def __init__(
        self,
        *,
        events: list[str],
        error: Exception | None = None,
        results: list[RetrievalResult] | None = None,
    ):
        self.events = events
        self.error = error
        self.results = results if results is not None else [retrieval_result()]
        self.queries: list[str] = []

    def retrieve(self, query: str):
        self.events.append("retrieve")
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.results
```

- [ ] **Step 2: Strengthen the existing successful orchestration test**

Create one `RetrievalResult`, inject it into `FakeRetriever`, wrap the real Context Builder, and capture the objects passed across the route boundary:

```python
result = retrieval_result()
retriever = FakeRetriever(events=events, results=[result])
captured_results: list[RetrievalResult] = []
real_context_builder = chat.build_retrieved_context

def capture_context(results):
    captured_results.extend(results)
    return real_context_builder(results)
```

Patch `chat.build_retrieved_context` with `capture_context`, retain the existing provider-message and exact NDJSON assertions, and assert:

```python
self.assertIs(captured_results[0], result)
self.assertEqual(captured_results[0].rank, 1)
self.assertEqual(captured_results[0].score, -0.25)
self.assertEqual(captured_results[0].match_origin, "exact_entity")
self.assertEqual(captured_results[0].matched_entity_ids, ["product:hp780"])
self.assertEqual(captured_results[0].chunk_id, "product:hp780:specifications")
self.assertEqual(captured_results[0].parent_document_id, "product:hp780")
self.assertEqual(captured_results[0].content_hash, "a" * 64)
self.assertEqual(captured_results[0].metadata.product_id, "hp780")
self.assertEqual(captured_results[0].source_url, "https://example.com/hp780/")
self.assertEqual(
    captured_results[0].source_files,
    ["src/content/products/hp780.json"],
)
```

These assertions catch any future route change that converts results into evidence-only dictionaries before the Context Builder boundary.

- [ ] **Step 3: Lock the success logging call to the minimal schema**

Keep the patched `chat.log_request` mock and assert the exact call:

```python
logged.assert_called_once_with(
    request_id="request-1",
    http_status=200,
    outcome="success",
    started_at=1.0,
)
```

The exact keyword set ensures the route does not send query or retrieval provenance to production logging.

- [ ] **Step 4: Run the focused route tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '.venv\Lib\site-packages').Path
$day5Python='C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $day5Python -m unittest tests.test_chat_route -v
```

Expected: all route tests pass. This is a characterization task; a passing first run proves the already-implemented boundary, while the assertions provide future regression protection.

### Task 2: Lock HTTP Error Redaction and Failure Logging

**Files:**
- Create: `tests/test_error_handling.py`

**Interfaces:**
- Consumes: `error_handling.http_exception_handler(request, exc)` and the existing `log_request(...)` boundary.
- Produces: a regression test proving internal retrieval details may exist in an in-memory exception but are excluded from both the public response and the production logging call.

- [ ] **Step 1: Add the focused privacy-boundary test**

Create `tests/test_error_handling.py` with this controlled exception payload:

```python
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

import error_handling


class ErrorHandlingPrivacyTests(unittest.TestCase):
    def test_retrieval_error_redacts_provenance_and_logs_only_error_type(self):
        request = SimpleNamespace(state=SimpleNamespace(
            request_id="request-1",
            started_at=1.0,
        ))
        exception = HTTPException(
            status_code=503,
            detail={
                "code": "retrieval_unavailable",
                "message": "服务暂时不可用，请稍后再试。",
                "internal_error": "EmbeddingAPIError",
                "query": "private question",
                "chunk_id": "product:hp780:specifications",
                "parent_document_id": "product:hp780",
                "score": 0.91,
                "source_url": "https://example.com/hp780/",
            },
        )

        with patch.object(error_handling, "log_request") as logged:
            response = asyncio.run(
                error_handling.http_exception_handler(request, exception)
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["X-Request-ID"], "request-1")
        self.assertEqual(
            json.loads(response.body),
            {
                "error": {
                    "code": "retrieval_unavailable",
                    "message": "服务暂时不可用，请稍后再试。",
                    "request_id": "request-1",
                }
            },
        )
        logged.assert_called_once_with(
            request_id="request-1",
            http_status=503,
            outcome="retrieval_unavailable",
            started_at=1.0,
            error="EmbeddingAPIError",
        )
```

The input deliberately carries forbidden fields so the test proves the handler is an active redaction boundary rather than merely observing today's route payload.

- [ ] **Step 2: Run the focused privacy tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '.venv\Lib\site-packages').Path
$day5Python='C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $day5Python -m unittest tests.test_error_handling tests.test_chat_route -v
```

Expected: all tests pass without production code changes.

### Task 3: Verify the Step 6 Contract and Commit

**Files:**
- Verify: `tests/test_chat_route.py`
- Verify: `tests/test_error_handling.py`
- Verify unchanged: `routes/chat.py`, `error_handling.py`, `app_logging.py`, `rag_context.py`

**Interfaces:**
- Consumes: the complete backend test suite and Git diff.
- Produces: evidence that Step 6 adds tests only and preserves production behavior.

- [ ] **Step 1: Run the complete backend test suite**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path '.venv\Lib\site-packages').Path
$day5Python='C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $day5Python -m unittest discover -s tests -v
```

Expected: 174 tests pass with zero failures or errors.

- [ ] **Step 2: Audit formatting and scope**

Run:

```powershell
git diff --check
git status --short
git diff --stat
git diff --name-only
```

Expected implementation scope after the plan commit:

```text
tests/test_chat_route.py
tests/test_error_handling.py
```

No production source, frontend source, knowledge artifact, vector artifact, or environment file may change.

- [ ] **Step 3: Commit the Step 6 tests**

Run:

```powershell
git add -- tests/test_chat_route.py tests/test_error_handling.py
git diff --cached --check
git commit -m "test: lock rag provenance privacy boundary"
```

- [ ] **Step 4: Verify the committed state**

Run:

```powershell
git status --short
git log -2 --oneline
git show --name-only --format= HEAD
```

Expected: clean worktree and a commit containing only the two test files. Report Step 6 and stop before Step 7.
