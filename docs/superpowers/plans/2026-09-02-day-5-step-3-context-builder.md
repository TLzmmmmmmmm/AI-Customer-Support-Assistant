# Day 5 Step 3 Context Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure Context Builder that converts ordered Day 4 retrieval results into complete, LLM-ready evidence containing only `type`, `section`, and `text`.

**Architecture:** Create one dependency-light module at the application boundary. The builder performs a direct order-preserving projection and leaves Top-K selection, ranking, JSON envelope construction, and generation to their existing owners.

**Tech Stack:** Python 3.12, existing Pydantic `RetrievalResult`, standard-library type annotations, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-02-day-5-full-rag-pipeline-design.md`

## Global Constraints

- Execute only Day 5 Step 3 and stop after its report.
- Preserve every supplied result and its order; do not truncate, sort, summarize, deduplicate, threshold, or mutate retrieval results.
- LLM evidence contains exactly `type`, `section`, and `text`.
- Exclude rank, score, match origin, matched entity IDs, chunk and parent IDs, hashes, metadata, URLs, source paths, and vector details.
- Do not change retrieval, prompts, DeepSeek integration, routes, startup, logging, frontend, knowledge artifacts, or vector artifacts.
- Do not call DashScope, DeepSeek, or any paid API.

## File Structure

- Create `rag_context.py`: pure order-preserving projection from `RetrievalResult` objects to evidence dictionaries.
- Create `tests/test_rag_context.py`: Top-5 preservation, full-text preservation, exact field allowlist, determinism, and empty-input behavior.

---

### Task 1: Implement the pure ordered Context Builder

**Files:**
- Create: `rag_context.py`
- Create: `tests/test_rag_context.py`

**Interfaces:**
- Consumes: `results: Sequence[RetrievalResult]` in Retriever order.
- Produces: `build_retrieved_context(results) -> list[dict[str, str]]` suitable for Step 2 `build_rag_messages(..., retrieved_context)`.

- [ ] **Step 1: Write failing projection tests**

Create a `retrieval_result(rank, section, text)` fixture with complete valid product provenance. Add a test with five differently ranked results and assert the returned list is exactly:

```python
[
    {"type": result.type, "section": result.section, "text": result.text}
    for result in results
]
```

Also assert each item has exactly the key set `{"type", "section", "text"}` and that a second call returns the same value. Because exact dictionary equality is used, any leaked `rank`, `score`, `match_origin`, `matched_entity_ids`, `chunk_id`, `parent_document_id`, `content_hash`, `metadata`, `source_url`, or `source_files` fails the test.

Add an empty-input test:

```python
self.assertEqual(build_retrieved_context([]), [])
```

- [ ] **Step 2: Run the tests and verify RED**

```powershell
$env:PYTHONPATH = (Resolve-Path -LiteralPath '.venv\Lib\site-packages').Path
$day5Python = 'C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $day5Python -m unittest tests.test_rag_context -v
```

Expected: import failure because `rag_context.py` does not exist. This is the only acceptable RED cause.

- [ ] **Step 3: Implement the minimal builder**

Create `rag_context.py`:

```python
from collections.abc import Sequence

from knowledge_pipeline.retrieval.models import RetrievalResult


def build_retrieved_context(
    results: Sequence[RetrievalResult],
) -> list[dict[str, str]]:
    return [
        {
            "type": result.type,
            "section": result.section,
            "text": result.text,
        }
        for result in results
    ]
```

Do not add validation, Top-K logic, serialization, logging, or integration code.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run `python -m unittest tests.test_rag_context -v` through the configured command above. Expected: both tests pass without API access.

- [ ] **Step 5: Run compatibility and complete regression tests**

```powershell
& $day5Python -m unittest tests.test_rag_context tests.test_prompts tests.test_retriever -v
& $day5Python -m unittest discover -s tests -v
```

Expected: all tests pass; the existing Retriever and Prompt Builder remain unchanged.

- [ ] **Step 6: Verify scope and commit Step 3 only**

```powershell
git diff --check
git status --short
git add -- rag_context.py tests/test_rag_context.py
git diff --cached --check
git commit -m "feat: add ordered rag context builder"
```

Expected implementation files are exactly `rag_context.py` and `tests/test_rag_context.py`. The plan document may be committed separately; no other source or artifact changes are allowed.

- [ ] **Step 7: Report and stop**

Report inspected files, existing boundaries, exact projection behavior, RED/GREEN evidence, final regression count, compatibility observations, and the Step 4 interface: route orchestration will pass the Retriever's ordered results into `build_retrieved_context()`, then pass that output into `build_rag_messages()`. Stop without implementing Step 4.
