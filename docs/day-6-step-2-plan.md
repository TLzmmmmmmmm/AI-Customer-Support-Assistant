# Day 6 Step 2 implementation plan

> Execution: inline in the current session, as explicitly requested. No new task, worktree, commit, provider call or later Day 6 step.

**Goal:** Build 60 curated cases and offline validation without changing production behavior.

**Architecture:** Separate evaluation-only case models from production retrieval models. Keep 50 frozen/development cases and 10 holdout cases in separate files, with separate authoring evidence reviews. A manifest pins dataset, fixtures and knowledge snapshots; the validator only reads local files.

**Tech stack:** Existing Python/Pydantic/unittest; JSON and Markdown; no new dependencies.

**Spec:** `docs/day-6-evaluation-contract.md` plus owner-approved 20/30/10 split, 0/1/2 rubric, binary hallucination, N/A exclusions, proposed scores pending owner review, and separate engineering/quality acceptance.

## Task 1 — Offline case contract

- [x] Create `tests/test_rag_dataset.py`: valid system-only answer and unknown case accepted; invalid enum/blank/duplicate question/unknown reference/wrong parent/review evidence/fixture rejected.
- [x] Run `python -m unittest tests.test_rag_dataset` and observe missing implementation failure.
- [x] Create `evaluation/__init__.py` and `evaluation/dataset.py`: `EvaluationCase`, `load_cases(path)`, `validate_cases(cases, documents, chunks, reviews, fixtures, system_rule_ids)`; no retrieval or generation imports.
- [x] Run focused tests, then refactor only within the new module.

## Task 2 — Curated dataset and provenance

- [x] Create `eval/rag_v1.json` (20 frozen + 30 dev), `eval/rag_v1_holdout.json` (10 new questions), and corresponding `*_authoring.json` files. Each authoring row gives six-dimensional review, diagnostic purpose, exact evidence excerpts and missing-information rationale where applicable.
- [x] Create separate development/holdout attack fixtures. Synthetic text stays outside the KB; clean chunk labels are not altered by attacks.
- [x] Keep baseline question bytes as decoded strings and IDs unchanged. Revise current annotations only: radio rules, minimally necessary facts, unsupported causal requirements. Document deviations from historical expectations.
- [x] Verify supporting chunks against normalized documents and local authoritative content. Review absence against current KB, owner rules and application capabilities, not a retrieval miss.
- [x] Create `eval/rag_v1_manifest.json` containing SHA-256 seals for curated artifacts, historical baseline, KB snapshots, and prior-question source. Record rules/rubric and no holdout execution.

## Task 3 — Bundle validation and handoff

- [x] Add tests for baseline mutation, cross-file duplicates, bad seals, wrong split allocation and prior holdout reuse; implement `validate_bundle(root)` and `scripts/validate_rag_dataset.py`.
- [x] Run `python scripts/validate_rag_dataset.py`, focused tests, full `python -m unittest discover -s tests`, and `git diff --check`.
- [x] Record actual counts, evidence caveats, changed files and results in `docs/day-6-step-2-dataset.md`; append approved decisions to the evaluation contract.
- [x] STOP before Step 3. No metrics, runs or model-quality claims.

## Test examples / independent expectations

```python
case = EvaluationCase.model_validate({
    'id': 'test-1', 'category': 'company', 'split': 'dev',
    'question': 'What is your brand?', 'expected_behavior': 'answer',
    'expected_answer': 'Shengborun Communications',
    'expected_document_ids': [], 'expected_chunk_ids': [],
    'expected_system_rule_ids': ['company_identity_v1'],
})
assert case.question == 'What is your brand?'
# A case pointing to chunk c1 in document d1 must reject expected_document_ids=['d2'].
# A review quote 'Battery: 9999mAh' must fail against text 'Battery: 1500mAh'.
# A held-out question equal to an old development question after NFKC/case/space
# normalization must fail even if its new ID differs.
```

User-approved runtime: bundled Python with `.venv/Lib/site-packages` on PYTHONPATH. No `.env` reads are needed.
