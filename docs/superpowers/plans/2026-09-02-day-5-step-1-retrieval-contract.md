# Day 5 Step 1 Retrieval Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Minimally extend the existing Day 4 `RetrievalResult` contract with an explicit knowledge `type` and prove that every required evidence/provenance field survives retrieval.

**Architecture:** Reuse the existing strict `RetrievalResult` model rather than creating a Day 5 schema. Add the same six-value knowledge-type literal already used by `VectorRecord`, copy `VectorRecord.type` in the existing `_to_result()` projection, and update only direct result fixtures and focused contract tests.

**Tech Stack:** Python 3.12, Pydantic 2, `unittest`, existing Day 4 retrieval package.

**Spec:** `docs/superpowers/specs/2026-09-02-day-5-full-rag-pipeline-design.md`

## Global Constraints

- Execute only Day 5 Step 1 and stop after its required report.
- Do not implement context construction, prompt injection, route integration, application startup, generation, grounding, or logging.
- Do not change ranking, Top-K=5, exact-entity behavior, embeddings, vectors, vector-record persistence, or evaluation semantics.
- Do not create a parallel retrieval-result schema.
- Preserve `POST /api/chat-stream`, `application/x-ndjson`, and existing `delta`/`done`/`error` behavior without touching those files.
- Make no external API calls and incur no usage charges.
- Keep production logs and frontend files unchanged.

## File Structure

- Modify `knowledge_pipeline/retrieval/models.py`: require `RetrievalResult.type` with values `catalog`, `product`, `solution`, `support`, `company`, or `contact`.
- Modify `knowledge_pipeline/retrieval/retriever.py`: copy `record.type` into the existing result projection.
- Modify `tests/test_retriever.py`: prove the Retriever preserves `type` and all existing evidence/provenance fields; prove CLI JSON exposes the stable contract.
- Modify `tests/test_retrieval_models.py`: prove `type` is required by direct `RetrievalResult` validation.
- Modify `tests/test_retrieval_evaluation.py`: update its direct `RetrievalResult` fixture with the newly required field; do not change evaluation cases or expected metrics.

---

### Task 1: Add the explicit retrieval knowledge type

**Files:**
- Modify: `knowledge_pipeline/retrieval/models.py:190-211`
- Modify: `knowledge_pipeline/retrieval/retriever.py:109-131`
- Modify: `tests/test_retriever.py:128-142`
- Modify: `tests/test_retriever.py:260-290`
- Modify: `tests/test_retrieval_models.py:1-180`
- Modify: `tests/test_retrieval_evaluation.py:61-78`

**Interfaces:**
- Consumes: existing validated `VectorRecord.type` and existing `_to_result(hit, rank, origin, matched_entity_ids) -> RetrievalResult`.
- Produces: `RetrievalResult.type: Literal["catalog", "product", "solution", "support", "company", "contact"]` on every retrieval result and serialized CLI result.

- [ ] **Step 1: Add a focused failing projection assertion**

In `tests/test_retriever.py`, extend
`test_results_preserve_record_fields_and_entity_ids` so the first copied field
checked is the explicit type:

```python
self.assertEqual(result.matched_entity_ids, ["product:hp790ex"])
self.assertEqual(result.type, source.type)
self.assertEqual(result.section, source.section)
```

This uses the current real `Retriever`, fake query embedding, exact NumPy index,
and valid `VectorRecord`; it tests the public contract rather than calling the
private projection directly.

- [ ] **Step 2: Add a focused failing model-contract test**

In `tests/test_retrieval_models.py`, import `RetrievalResult` alongside
`VectorRecord` and add this helper below `record_payload()`:

```python
def retrieval_result_payload() -> dict:
    payload = record_payload()
    for field in (
        "schema_version",
        "language",
        "embedding_provider",
        "embedding_model",
        "embedding_dimensions",
        "embedding_text_type",
        "embedding",
    ):
        payload.pop(field)
    payload.update({
        "rank": 1,
        "score": 0.95,
        "match_origin": "exact_entity",
        "matched_entity_ids": ["product:hp780"],
    })
    return payload
```

Add a new `RetrievalResultTests` class immediately after
`RetrievalConfigurationTests` and place this test in it:

```python
def test_retrieval_result_requires_explicit_knowledge_type(self):
    payload = retrieval_result_payload()
    result = RetrievalResult.model_validate(payload)

    self.assertEqual(result.type, "product")

    payload.pop("type")
    with self.assertRaises(ValidationError):
        RetrievalResult.model_validate(payload)
```

The first validation currently fails because `type` is an extra field on the
strict model. After implementation, the second validation proves downstream
code can rely on the field rather than infer it from metadata or IDs.

- [ ] **Step 3: Run the focused tests and verify the intended RED failures**

From the repository root in the current Windows environment:

```powershell
$env:PYTHONPATH = (Resolve-Path -LiteralPath '.venv\Lib\site-packages').Path
$day5Python = 'C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $day5Python -m unittest tests.test_retriever tests.test_retrieval_models -v
```

Expected result: failures limited to the missing/forbidden
`RetrievalResult.type`; existing retrieval behavior continues to pass.

- [ ] **Step 4: Implement the minimal strict field and projection**

In `knowledge_pipeline/retrieval/models.py`, add the field immediately after
`parent_document_id` so identity fields remain grouped:

```python
class RetrievalResult(StrictModel):
    rank: int = Field(ge=1)
    score: float
    match_origin: Literal["exact_entity", "dense"]
    matched_entity_ids: list[str]
    chunk_id: str = Field(min_length=1)
    parent_document_id: str = Field(min_length=1)
    type: Literal[
        "catalog", "product", "solution", "support", "company", "contact"
    ]
    section: str = Field(min_length=1)
```

Do not add another metadata model or change the existing `Metadata` union.
Every production result originates from an already validated `VectorRecord`,
which already enforces type-specific metadata.

In `knowledge_pipeline/retrieval/retriever.py`, add one value to the existing
dictionary passed to `RetrievalResult.model_validate`:

```python
"chunk_id": record.chunk_id,
"parent_document_id": record.parent_document_id,
"type": record.type,
"section": record.section,
```

Do not change result assembly, scores, ordering, entity handling, or Top-K.

- [ ] **Step 5: Update direct result fixtures for the required contract**

In `tests/test_retrieval_evaluation.py`, update only the `result()` helper:

```python
"chunk_id": source.chunk_id,
"parent_document_id": source.parent_document_id,
"type": source.type,
"section": source.section,
```

This is a compatibility update for a direct model fixture. Do not edit
`eval/retrieval_v1.json`, `eval/retrieval_v1_1.json`, either results file, or
any expected metric.

- [ ] **Step 6: Strengthen the existing CLI contract assertion**

In `tests/test_retriever.py`, extend
`test_valid_local_artifacts_produce_compact_json_without_generation`:

```python
self.assertEqual(payload[0]["chunk_id"], "product:hp780:content")
self.assertEqual(payload[0]["type"], "product")
self.assertEqual(payload[0]["parent_document_id"], "product:hp780")
self.assertEqual(payload[0]["source_url"], "https://example.com/hp780/")
self.assertNotIn("answer", payload[0])
```

This proves the existing public retrieval command carries both evidence type
and backend provenance without adding generation.

- [ ] **Step 7: Run all focused Day 4 contract/evaluation tests**

```powershell
$env:PYTHONPATH = (Resolve-Path -LiteralPath '.venv\Lib\site-packages').Path
$day5Python = 'C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $day5Python -m unittest tests.test_retrieval_models tests.test_retriever tests.test_retrieval_evaluation -v
```

Expected result: every focused test passes; the frozen v1/v1.1 suites and
metrics remain unchanged; no API request occurs.

- [ ] **Step 8: Run the complete backend regression suite**

```powershell
$env:PYTHONPATH = (Resolve-Path -LiteralPath '.venv\Lib\site-packages').Path
$day5Python = 'C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $day5Python -m unittest discover -s tests -v
```

Expected result: all existing backend tests pass. No knowledge artifact or
vector file changes.

- [ ] **Step 9: Verify the exact Step 1 scope**

```powershell
git diff --check
git status --short
git diff --stat
```

Expected changed files are exactly:

```text
knowledge_pipeline/retrieval/models.py
knowledge_pipeline/retrieval/retriever.py
tests/test_retrieval_models.py
tests/test_retriever.py
tests/test_retrieval_evaluation.py
```

Confirm that `prompts.py`, `rag_context.py`, `services/llm.py`,
`services/retrieval.py`, `routes/chat.py`, `main.py`, frontend files, knowledge
JSONL, vector records, evaluation fixtures, and evaluation results are
unchanged.

- [ ] **Step 10: Commit Step 1 only**

```powershell
git add -- knowledge_pipeline/retrieval/models.py knowledge_pipeline/retrieval/retriever.py tests/test_retrieval_models.py tests/test_retriever.py tests/test_retrieval_evaluation.py
git diff --cached --check
git commit -m "feat: expose retrieval result knowledge type"
```

- [ ] **Step 11: Report and stop**

Report the nine required Step 1 items:

1. files inspected;
2. the pre-existing `RetrievalResult` structure;
3. its exact pre-existing fields;
4. the missing explicit `type`;
5. the two production-code changes and why;
6. focused tests added/updated;
7. commands run;
8. focused and full-suite results;
9. Step 2 concern: Prompt Builder must consume `type/section/text` without
   exposing provenance.

Then stop. Do not begin Step 2.
