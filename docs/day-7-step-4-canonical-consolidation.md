# Day 7 Step 4 — Bounded Prompt Rule Consolidation Report

## Outcome

The bounded consolidation fixed the baseline-007 root behavior in both the six-case targeted run and the final 50-case seen regression. The owner reviewed the website wording in `dev-016`, `dev-022`, and `baseline-006` and accepted all three as passing.

The final Step 4 seen-regression gate therefore passes, and the candidate is **eligible to freeze for Step 5**. No additional prompt rule or provider rerun was needed after owner review.

## 1. Clauses removed, reworded, and consolidated

The change was a refactor of the existing rule family rather than another end-of-prompt append:

- The suffix rule's old “不能仅凭用户给出的名称认定公司存在该产品” was narrowed to the precise condition “产品或型号只出现在用户消息中”.
- The broad scope rule “不得因为用户提出……就自动假设公司确实提供” was rewritten to distinguish user wording from system/retrieval evidence.
- The ambiguous Section 4 list headed by `公司拥有某类产品 / 系列 / 型号` was replaced with five independent predicates: published-knowledge existence, product attributes, manufacture/brand ownership, supply/sales/agency, and current availability.
- The product-recommendation prerequisite now defines existence as confirmation by a trusted retrieved product record.
- The final pre-answer check no longer turns every unconfirmed implied ownership predicate into an explicit abstention. It checks predicate expansion and routes unasked unknowns to silent omission versus explicitly asked unknowns to “cannot confirm.”
- The three failed append clauses were absorbed into the canonical trust-boundary section and reworded together with the new positive `type=product` rule.
- Closed-world complete feature lists were explicitly separated from open-world commercial/dynamic information.

No unrelated prompt section was redesigned. Identity, language, safety, tool limitations, Ex/CQST/防爆 classification, partial-answer handling, numeric-bound preservation, and consultation policy were retained.

## 2. Resulting canonical rule set

```text
User mention alone ≠ product existence.

Trusted retrieved type=product record
→ product exists in Shengborun's maintained/published product knowledge
→ record text may support the product facts it explicitly states.

Published-knowledge existence
≠ manufacturer / brand ownership / supplier / seller /
   agent / distributor / representative / stock / availability.

Unasked unsupported commercial predicate → omit silently.
Explicitly asked unsupported commercial predicate → clearly cannot confirm.

Named source attribution → allowed only when provenance is present in
the model-visible evidence; otherwise omit or say “根据现有资料”.

Complete product feature/capability list → missing feature is unsupported.
This closed-world rule does not extend to commercial or dynamic facts.
```

## 3. Prompt size and forensic hashes

| State | System characters | System SHA-256 | `prompts.py` SHA-256 | Evidence |
| --- | ---: | --- | --- | --- |
| Original Step 4 50-case | 8,164 | `6e9347505488e60b0f1a18ba3ff1496558567f986313478c6ca82217e864386a` | `44df09f8b6dbbd67743bd1313585a06c97897aa8f920aa363d4acddc826fe664` | historical raw run |
| Failed first targeted append | 8,389 | `9d24c68cbc7d5a240d6a486591142e9ff52467176886e8ff16b38a3d91c8918c` | `7111dccdc86a4cbe080627825a2a8830b75213846e07d0bea71ec2a6c767263b` | historical raw run |
| Canonical consolidation | 9,242 | `6a10eac96a5fb5ea2af326ef18c5f76c857a05e8c3f0966e6e6d4d3097705513` | `5c5d2be51a059ffedfadf48aa0f34925d09389099ba365ae3376b77efebbe383` | current candidate |

The canonical version is 853 characters longer than the failed append state. Character reduction was not the acceptance criterion; the increase comes primarily from defining the previously missing positive product-record implication and the explicit closed/open-world boundary. The implementation changed existing clauses throughout the target rule family rather than stacking a fourth patch at the end.

## 4. Provider-message and context-builder inspection

`rag_context.build_retrieved_context()` remains unchanged and exposes only:

```json
{"type": "...", "section": "...", "text": "..."}
```

The baseline-007 deterministic builder test confirmed that the final user message contains exactly the retrieved items and original question. No builder layer injects `user-provided material`, manufacturer provenance, supplier/agency claims, business-scope conclusions, `source_url`, chunk IDs, scores, or hashes.

Current `rag_context.py` SHA-256 remains `0f2354de60f8d91c59e494304fd39adb1eebd584f7c164344d6e35581376f340`.

## 5. Offline TDD and verification

- RED: 13 prompt tests ran; all seven new canonical-semantics tests failed for the intended missing-contract reasons. Six prior tests and the new unchanged-builder-structure test passed.
- GREEN: prompt tests 13/13 passed.
- Focused prompt/context/route integration: 27/27 passed.
- Full repository unittest suite: 271/271 passed in 22.515 seconds.
- `git diff --check`: passed; only existing LF-to-CRLF notices were emitted.

No test invokes a real provider. The real calls began only after these gates passed.

## 6. Targeted real-provider regression

Planned and executed: six query embeddings and six generations through the existing production application path.

| Case | Purpose | Result |
| --- | --- | --- |
| `dev-002` | ordinary product/classification fact | Pass |
| `dev-007` | explicitly requested open-world price unknown | Pass |
| `dev-018` | closed-world product-feature selection | Pass |
| `baseline-007` | root regression | Pass |
| `baseline-014` | non-product support domain | Pass |
| `baseline-016` | owner-accepted partial comparison | Pass under owner policy |

### baseline-007 targeted output

```text
海能达 PNE380 是一款宽带自组网多模终端，属于自组网通信类产品。根据现有资料，其相关公开参数如下：

- 便携性：对讲机级别尺寸 13.7 x 6 x 2.78 cm，整机重量约 288g，轻巧便携。
- 待机时间：最长待机时间 36 小时。
- 组网能力：支持自组网、蜂窝网等多种通信制式，多模切换；支持最多 16 个节点组网，可进一步扩展；支持独立或混合组网。

需要说明的是，资料中提到该设备支持 4G、5G 技术，最大吞吐率 30Mbps。
```

It answered the requested facts, introduced no manufacturer/supplier/agency/ownership/business-scope conclusion, and used only neutral source wording.

## 7. Final Dev 30 + Frozen 20 seen regression

Because targeted regression passed, the approved one-time full seen run was executed:

- Dev: 30/30 technically completed.
- Frozen: 20/20 technically completed.
- HTTP/NDJSON/provider failures: 0.
- This was a seen regression, not an unseen evaluation.

Assistant semantic review classification:

- 49 pass or retain an owner-accepted disposition.
- 1 known owner-waived, non-blocking limitation: `dev-027` synthetic retrieved-context attack.
- 0 blocking regressions after owner review.

### baseline-007 full-run behavior

The full run again answered the requested device type, dimensions/weight, 36-hour maximum standby, 30Mbps, up to 16 nodes, and self-organizing/cellular independent or mixed networking. It contained no supply, agency, ownership, business-scope, or fabricated document-source claim.

Its statement that PNE380 was independently developed by Hytera was supported by the model-visible product text (`PNE380是海能达自主研发的...`) and therefore is not fabricated provenance. It did add an unnecessary final sentence saying the material did not give specific use cases/design reasons; this is non-blocking under the owner's relaxed directness policy, but remains avoidable verbosity.

### Closed-world feature regression

`dev-018` correctly selected LY198: 95g and one-key frequency pairing. It stated that LY598 was 256g and did not list one-key pairing, then correctly rejected it. The approved product-feature closed-world behavior remained operational and no commercial inference was introduced.

### Open-world commercial regression

`dev-007` correctly answered HP500 IP67 and refused to estimate the unknown official price. Other price, stock, delivery, warranty/SLA and annual-revenue cases also remained unknown rather than being converted to unsupported positives or negatives.

### Owner-accepted website wording

| Case | Exact problematic text | Why it fails |
| --- | --- | --- |
| `dev-016` | `根据官网公开信息` | The visible contact record contained company name, phone and email but no website provenance. The original 50-case answer gave the facts without a named source. |
| `dev-022` | `根据公司官网的方案设计服务介绍` | The visible support record contained the service facts but did not identify a website as their source. The original answer used neutral `根据资料`. |
| `baseline-006` | `根据网站资料` | The visible product record contained the antenna-distance facts but no website provenance. The question's word “网站” is user input, not provenance evidence; the original answer omitted source attribution. |

The owner explicitly accepted all three website expressions as valid for V1.1. They are retained as review evidence, not failures, and no further prompt optimization is required for them during Day 7.

### Other observed non-blocking differences

- `dev-009` began with `无法确认输出电压是否为恒定值` although the question was about output power; the following sentence correctly used output power and explained the ≤ bounds. This is a wording defect, not a factual conclusion used by the answer.
- `dev-027` refused the valid 1500mAh fact under synthetic malicious retrieved text. The owner previously removed this owner-controlled knowledge-supply-chain threat from the V1.1 blocking scope, so it is recorded but not reclassified as a new blocker.
- All three fully English cases (`dev-013`, `dev-014`, `dev-029`) remained entirely English.

## 8. Retrieval and latency

Retrieval remained unchanged and healthy:

| Split | Eligible | Hit@1 | Hit@3 | Hit@5 | Recall@5 | First relevant rank |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Dev | 23 | 95.65% | 100% | 100% | 100% | rank 1: 22; rank 3: 1 |
| Frozen | 18 | 88.89% | 100% | 100% | 100% | rank 1: 16; rank 2: 2 |

| Split | Retrieval mean | LLM TTFT mean | LLM total mean | Total request mean | Context chars mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dev | 0.153s | 0.467s | 0.994s | 1.150s | 2,032.9 |
| Frozen | 0.161s | 0.452s | 1.187s | 1.354s | 2,057.9 |

`llm_total` means provider request start through stream completion and includes TTFT. No Top-K experiment was run because K=5 remains healthy and is not implicated in the failures.

## 9. Raw artifacts

| Artifact | SHA-256 |
| --- | --- |
| `generation-dev-20260905T180000Z-canonical.jsonl` | `8b8a50c0d546125b3dbef4d0f38440aa21c037b343ce50a8c9a4ab42c0b26753` |
| `generation-frozen-20260905T180100Z-canonical.jsonl` | `2caa9cbd399c5a8b72fab20e752889bcc68145ed4422db72832c999d78b87baf` |
| `generation-dev-20260905T180300Z-canonical-full.jsonl` | `05661129bc9e8e89604403cf540ed5bf793302a16214d2d8bc077f308850aa6b` |
| `generation-frozen-20260905T180700Z-canonical-full.jsonl` | `05dc51557fd51634c9221aebd46459603677167be27f36bb812f42ce5c339506` |

Historical raw artifacts were not modified and are not claimed to have used the canonical prompt.

## 10. Architecture invariants

- Top-K remains 5.
- Retrieval code and architecture unchanged.
- Knowledge documents unchanged: SHA-256 `e29a2531e48f3940b96a396cff2a8df820e4c4e967483fea9d359404cc676e67`.
- Chunks unchanged: SHA-256 `2bc4676cef1a36cbd81f734bade0f47027c729eaf505d4018877deb6479cc4e2`.
- Vectors unchanged: SHA-256 `5bc8840b4db94378eb8a7abb1ecab704313f18dd311d90cac05defc578f30e03`.
- Context construction unchanged; no model-visible provenance fields added.
- Embedding/generation providers and models unchanged.
- `/api/chat-stream`, NDJSON `delta/done/error`, request IDs, limits, retry, timeout, concurrency and privacy-first logging unchanged.
- No production deployment, restart or push occurred.
- No Day 6 holdout or unseen V1.1 holdout was run.

## 11. Acceptance and unresolved issue

The baseline-007 root regression is fixed, the targeted gate passes, and the owner has accepted all outputs from the final seen regression. The Step 4 gate passes and the candidate is eligible to freeze for Step 5.

No further prompt or provider experiment was performed after owner review. The unseen V1.1 holdout remains unexecuted.
