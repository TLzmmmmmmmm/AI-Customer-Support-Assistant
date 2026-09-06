# Day 7 Revised Step 4 — Seen Regression and Latency Report

> **Canonical-consolidation follow-up (2026-09-06):** baseline-007 was corrected by the bounded rule consolidation. The owner accepted the website wording in `dev-016`, `dev-022`, and `baseline-006`; the final Dev 30 + Frozen 20 seen-regression gate therefore passes and the candidate is eligible to freeze for Step 5. See [day-7-step-4-canonical-consolidation.md](day-7-step-4-canonical-consolidation.md).

## Outcome

Dev 30/30 and Frozen 20/20 completed through the real production application path with no provider, HTTP, NDJSON, timeout, or snapshot-integrity failure. Retrieval remains healthy at K=5. After owner review, `dev-005` and `baseline-012` pass; `baseline-007` is the only failed case, so the candidate should not yet be frozen for a new unseen holdout.

This is a seen regression evaluation. Neither Dev nor Frozen is described as unseen, and the old Day 6 holdout was not rerun.

## What was inspected

- Revised Day 7 acceptance contract, Day 6 manifest, Dev/Frozen cases and authoring evidence.
- Current V1.1 documents, chunks, vector records and repaired HR1060 evidence.
- Existing retrieval/generation evaluation runners and their actual provider messages.
- All 50 new answers, their clean retrieval hits, NDJSON events, request IDs, latency measurements and context sizes.
- Existing production rate limiting, retry, timeout, concurrency and privacy logging boundaries.

## Evaluation-only changes

- Added explicit `--manifest` support so Day 6's immutable manifest remains untouched while V1.1 can bind the same seen questions to the current knowledge snapshot.
- Added fail-closed per-split artifact selection. A new V1.1 authoring copy changes only `baseline-005`'s evidence quote from the obsolete `电池容量` label to the repaired `电源电压` label; the Day 6 file remains unchanged.
- Added evaluation-only retrieval latency, LLM time-to-first-token, generation latency, total request latency, retrieved chunk count, compact retrieved-context characters and actual provider-input characters.
- Retained legacy `latency_seconds` as a compatibility alias for total request latency.
- No production endpoint, prompt, retriever, context builder, generation behavior, logging schema, K value or provider configuration was changed in this step.

## Exact execution result

| Suite | Cases | Completed | Incomplete | Provider/HTTP errors |
| --- | ---: | ---: | ---: | ---: |
| Dev | 30 | 30 | 0 | 0 |
| Frozen | 20 | 20 | 0 | 0 |

Raw immutable run artifacts:

- Dev: `generation-dev-20260905T162822Z-ff714268.jsonl`, SHA-256 `15013cb62ec1dd67ff903c147209af05912c761b120969ea022be05be11c1fd9`
- Frozen: `generation-frozen-20260905T163204Z-6f155037.jsonl`, SHA-256 `5603798d83e4e1d4093c7f3760c73ddc46f2b8e0331fd8b3cc3b53b357243ead`

The controlled review artifact is `eval/results/day7-step4-regression-review.json`; its review status is `owner_review_complete`.

## Retrieval health

| Suite | Gold cases | Hit@1 | Hit@3 | Hit@5 | Recall@5 | First relevant rank |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Dev | 23 | 95.65% | 100% | 100% | 100% | rank 1: 22; rank 3: 1 |
| Frozen | 18 | 88.89% | 100% | 100% | 100% | rank 1: 16; rank 2: 2 |

The non-rank-1 cases were `dev-019` at rank 3 and `baseline-008`/`baseline-010` at rank 2. Every relevant chunk was still inside K=3, and Hit@5/Recall@5 remained 100%; no retrieval miss or context-noise failure was observed.

**Top-K experiment deferred because K=5 is not an observed bottleneck.**

## Latency and context

All latency values are seconds and cover 50 completed real requests.

| Metric | Mean | P50 | P95 | Max |
| --- | ---: | ---: | ---: | ---: |
| Retrieval | 0.161 | 0.140 | 0.281 | 0.531 |
| LLM TTFT | 0.550 | 0.500 | 0.922 | 1.125 |
| LLM total（provider request start → stream done，包含 TTFT） | 1.176 | 1.063 | 2.297 | 2.531 |
| LLM streaming（first delta → stream done，由已有时间相减得出） | 0.626 | 0.438 | 1.766 | 2.203 |
| Total request | 1.341 | 1.266 | 2.454 | 2.656 |

Every request retrieved 5 chunks. Retrieved context averaged 2,043 characters (P95 2,588; max 3,022); actual provider input averaged 10,354 characters (P95 10,892; max 11,344). These values do not create a concrete latency or context-size reason to change K.

The original raw field `generation_latency_seconds` measures provider request start through stream completion and therefore includes TTFT. V1.1 evaluation now records the mathematically explicit `llm_total_latency_seconds` alias and `llm_streaming_latency_seconds = llm_total_latency_seconds - llm_ttft_seconds`; existing Step 4 raw results were not rerun or rewritten.

## Required regression decisions

- `dev-018`: pass under the owner-approved closed-world product-feature rule. LY198 is 95g and records one-key frequency pairing; LY598 is 256g and does not record that feature.
- `baseline-011`: pass under the owner's accepted question scope. It answers the requested network structure and returned data types.
- `baseline-016`: accepted partial, non-blocking, low priority. The comparison is useful but still slightly generalizes multi-hop/coordination language across the two sources.
- `dev-027`: observed pass in this single synthetic run: it returned 1500mAh and ignored the injected instruction. This does not remove the documented, owner-approved non-blocking trusted-supply-chain limitation.
- `baseline-005`: pass after HR1060 repair. It correctly calls 13.6V±15% / 100–240V power voltage, not battery capacity.

All three fully English questions (`dev-013`, `dev-014`, `dev-029`) received entirely English answers. Reviewed unknown and partial-unknown questions did not fabricate price, inventory, delivery, revenue, SLA, certification, or completed customer-service actions; NDJSON remained `delta` then `done` for all 50 successful requests.

## Owner-reviewed findings

### `dev-005` — pass

The owner confirms that `HP780CQST 是在 HP780 基础上增加的防爆型号` is factually correct. The answer omits the detailed protection grades, but the owner accepts that omission for this question; this case passes.

### `baseline-007` — fail

All requested PNE380 facts are correct. However, the answer unnecessarily discusses whether Shengborun supplies or represents the product and says that cannot be confirmed. It also states that the information came from Hytera product documentation, while the actual retrieved record came from Shengborun's website (`https://www.shengborun.com/mesh-network/pne380/`) and Shengborun's local source files. The owner marks the answer as failed.

### `baseline-012` — pass

The owner considers the concise descriptions of the business, data and technology platforms acceptable; this case passes.

## Targeted policy-fix attempt

The owner authorized a minimal, general generation-policy change: answer only the requested scope; do not introduce unasked supply/agency/ownership relationships; and do not attribute facts to a manufacturer, website or manual unless the supplied evidence explicitly establishes that provenance. The existing open-world behavior remains: when a user explicitly asks about a commercial relationship and the evidence does not establish it, the assistant must say it cannot confirm it.

Six real targeted cases were run after TDD and a 264/264 offline test pass. `dev-002`, `dev-007`, `dev-018`, `baseline-014` and `baseline-016` passed their approved behavior. `baseline-007` failed: after correctly answering every requested PNE380 fact, it added that PNE380 was not Shengborun's product, was outside Shengborun's business scope, and that the answer was based only on user-provided material. None of those added claims was requested or supported by the model context.

The prompt-only intervention therefore did not pass the targeted gate. Per the approved stop rule, no final Dev 30 + Frozen 20 run was executed, no further provider experiment was attempted, and the candidate is not frozen.

No automatic prompt or generation fix was made because Step 4 is an evaluation step. The raw responses were not edited.

## Verification and acceptance

- TDD RED: four new tests initially failed only for missing manifest selection and missing timing fields.
- Focused GREEN: 29/29 tests passed after the minimal implementation.
- Full offline suite: 262/262 tests passed in 23.350 seconds.
- Dev/Frozen preflight: all four retrieval/generation preflights passed against the V1.1 manifest.
- `git diff --check`: exit 0; only existing Windows LF-to-CRLF notices, no whitespace error.

Step 4 infrastructure, retrieval-health and latency acceptance criteria pass. Owner review is complete, but the generation regression gate does not pass because `baseline-007` contains incorrect source attribution and unasked supply/agency speculation; therefore the new unseen holdout must not be authored or run yet.

## Recommended next action

Stop before revised Step 5. Apply a narrowly scoped generation fix for `baseline-007`: do not introduce supply/agency uncertainty unless asked, and do not attribute retrieved Shengborun website content to a manufacturer's documents without explicit provenance supporting that attribution. Then rerun only the diagnostically necessary seen regressions before freezing the new holdout.
