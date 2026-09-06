# Bounded Prompt Rule Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve baseline-007 by replacing ambiguous product-existence and commercial-relationship prompt rules with the owner-approved canonical predicate model.

**Architecture:** Keep the existing RAG pipeline unchanged and modify only the system-prompt policy family. Deterministic tests lock the prompt contract and provider-message structure; real production-path calls are gated behind a complete offline pass.

**Tech Stack:** Python, unittest, FastAPI production route, existing DashScope retriever and DeepSeek streaming provider.

**Spec:** `docs/day-7-step-4-rule-conflict-map.md` plus the owner-approved bounded consolidation instruction attached on 2026-09-06.

## Global Constraints

- Keep retrieval, Top-K=5, embeddings, knowledge, context serialization, providers/models, API, NDJSON, production logging and deployment unchanged.
- Preserve closed-world product-feature behavior, open-world commercial facts, Ex/CQST/防爆 classification, English-only answers to fully English questions, and partial answers.
- Do not run the unseen V1.1 holdout.
- Preserve historical raw artifacts and identify all prompt states truthfully.

---

### Task 1: Lock the canonical prompt and provider-message contract

**Files:**
- Modify: `tests/test_prompts.py`
- Inspect: `prompts.py`, `rag_context.py`

**Interfaces:**
- Consumes: `prompts.SYSTEM_PROMPT`, `prompts.build_rag_messages()`.
- Produces: focused deterministic regression tests for the seven approved semantics and unchanged model-visible context.

- [ ] Add literal contract assertions for user mention versus existence, retrieved `type=product` existence, relationship independence, answer-scope routing, explicit-commercial abstention, provenance visibility, and closed/open-world separation.
- [ ] Run `python -m unittest tests.test_prompts -v` and confirm RED because the positive retrieved-product rule and consolidated distinctions are absent.

### Task 2: Consolidate the conflicting policy family

**Files:**
- Modify: `prompts.py`

**Interfaces:**
- Consumes: the Task 1 contract.
- Produces: a semantically consolidated `SYSTEM_PROMPT`; no application interface changes.

- [ ] Rewrite the user-mention clause so its applicability is explicitly limited to user wording alone.
- [ ] Replace ambiguous portfolio/ownership checks with predicate-specific evidence rules.
- [ ] Add the positive `type=product` published-knowledge existence rule.
- [ ] Rewrite the final pre-answer ownership check so unasked unknowns are omitted and asked unknowns are explicitly unresolved.
- [ ] Consolidate the late appended rules into the canonical section instead of stacking them.
- [ ] Run focused prompt tests and confirm GREEN.

### Task 3: Verify all offline behavior and forensic state

**Files:**
- Inspect: all existing tests and the actual baseline-007 provider-message builder path.

**Interfaces:**
- Consumes: revised prompt and unchanged builders.
- Produces: offline test evidence, hashes, character counts, and builder inspection result.

- [ ] Run focused prompt and route/context tests.
- [ ] Run the complete repository unittest suite.
- [ ] Run `git diff --check`.
- [ ] Record historical 50-case, failed append, and canonical prompt hashes/character counts without rewriting old artifacts.

### Task 4: Run gated seen regressions

**Files:**
- Create: new timestamped raw results through `scripts/evaluate_rag_generation.py`.
- Create: `eval/results/day7-step4-canonical-consolidation-review.json`.
- Modify: `docs/day-7-step-4-regression-latency.md`.

**Interfaces:**
- Consumes: unchanged production `/api/chat-stream` path and approved six-case set.
- Produces: targeted quality decision; only on pass, one Dev 30 + Frozen 20 seen-regression decision.

- [ ] Run exactly the six approved targeted seen cases through the real application path.
- [ ] Review every output against its owner-approved expectation.
- [ ] If any blocker exists, stop without appending another rule or running the full suite.
- [ ] If targeted passes, run Dev 30 + Frozen 20 once, review regressions, and determine Step 5 freeze eligibility.

### Task 5: Report and stop

**Files:**
- Modify: `docs/day-7-step-4-regression-latency.md`
- Create: `eval/results/day7-step4-canonical-consolidation-review.json`

**Interfaces:**
- Consumes: Tasks 1–4 evidence.
- Produces: the required 18-point final report and a clear acceptance decision.

- [ ] Record exact clauses changed, canonical rules, counts/hashes, tests, case outcomes, invariants, regressions, freeze eligibility, and non-deployment/non-holdout confirmation.
- [ ] Stop before Day 7 Step 5.
