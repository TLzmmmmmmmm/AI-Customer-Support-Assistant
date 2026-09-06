# Day 7 Step 5B — V1.1 New Holdout Report

## Outcome

`Step 5B — Business Acceptance PASS after owner review`

The sealed holdout ran exactly once from clean commit `191aae805856b7b3aabffdaefe5c8e01e306437c`. The pre-sealed rubric result was **13/15 pass, 2/15 initially flagged**. After reviewing the actual customer impact, the owner accepted cases 006 and 007 as non-blocking; no system change and no rerun occurred.

## Execution integrity

- Query embedding calls: 15
- Generation calls: 15
- Completed: 15/15
- Incomplete: 0
- Automatic reruns: 0
- Synthetic fixtures: 0
- Raw SHA-256: `bf9c2ccc158acbedc0537a1be0b265e41bd0e01a2dcc4c00cae0e9cc9a3d70a1`
- Freeze SHA-256: `650c72e8631fcc6cc203276e16a98aa5cf7e4efc49f4598010617d9f94972efc`
- Snapshot verification: unchanged

## Initially flagged cases

### v1.1-holdout-006 — closed-world capability policy not applied

The expected gateway chunk ranked first and listed `doc`, `pdf`, and `mht` as the supported report formats. Under the owner-approved closed-world product-capability rule, the answer should say XLSX export is not supported; instead the model said it could not confirm XLSX support. Retrieval supplied sufficient evidence, so this is a generation/policy-application failure.

### v1.1-holdout-007 — invalid numeric-bound inference

The near- and remote-end repeater chunks ranked first and second. The remote unit is only constrained as `<20kg`, which does not establish whether it is below 10kg, but the answer declared that it did not satisfy the `<10kg` requirement. This is an unsupported negative inference from an insufficient numerical bound.

## Other cases

The assistant review marks 13/15 cases as passing. The fully English firewall case stayed entirely in English, the static package-inclusion and dynamic-price cases abstained clearly, and the less-leading combiner recommendation selected the correct product without retrieval difficulty.

## Retrieval

- Eligible cases: 14
- Hit@1: 85.71% (12/14)
- Hit@3: 100%
- Hit@5: 100%
- Recall@5: 100%
- First relevant rank: #1 for 12 cases, #2 for one, #3 for one

Both blocking failures had complete relevant evidence in the context. There is no evidence that Top-K, embeddings, NumPy retrieval, or the knowledge index caused either failure.

## Latency

| Metric | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| Retrieval | 0.174s | 0.125s | 0.469s |
| LLM TTFT | 0.448s | 0.234s | 0.735s |
| LLM total | 0.957s | 0.454s | 2.110s |
| Streaming after first delta | 0.509s | 0.219s | 1.703s |
| Total request | 1.135s | 0.579s | 2.375s |

Every case used five retrieved chunks. Mean retrieved context was 2,268.7 characters and mean total provider input was 11,668.6 characters.

## Owner disposition

Cases 006 and 007 are both accepted as usable and non-blocking. Step 5B therefore passes business acceptance and the candidate may proceed to Step 6; it is not yet classified as a production candidate until the Step 6 smoke gate passes.

This holdout is now seen evidence and must never be run or described as unseen again. The original raw output, pre-sealed rubric scores, and initial flags remain preserved as an audit trail.
