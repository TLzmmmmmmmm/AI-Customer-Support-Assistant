# Day 7 Step 5A — New V1.1 Holdout Preparation

## Status

`assistant_evidence_review_complete / owner_review_pending / not_sealed / not_executed`

This step prepares the new RAG V1.1 holdout but deliberately does not run retrieval, query embeddings, or generation. The holdout may be sealed only after the owner reviews the questions, expected answers, required facts, forbidden claims, and cited evidence.

## Candidate snapshot

- Candidate: `rag-v1.1`
- Production endpoint: `POST /api/chat-stream`
- Streaming protocol: `application/x-ndjson` with `delta`, `done`, and `error`
- Retrieval: NumPy exact cosine, Top-K 5
- Embedding: DashScope `qwen3.7-text-embedding`, 1024 dimensions
- Generation: DeepSeek `deepseek-v4-flash`
- Knowledge: 61 normalized documents, 73 chunks
- Prompt SHA-256: `5c5d2be51a059ffedfadf48aa0f34925d09389099ba365ae3376b77efebbe383`
- Documents SHA-256: `e29a2531e48f3940b96a396cff2a8df820e4c4e967483fea9d359404cc676e67`
- Chunks SHA-256: `2bc4676cef1a36cbd81f734bade0f47027c729eaf505d4018877deb6479cc4e2`
- Vector records SHA-256: `5bc8840b4db94378eb8a7abb1ecab704313f18dd311d90cac05defc578f30e03`
- Preparation base commit: `849957cd4719ce2bee41aa5f366b8a4908c23901`
- The worktree was dirty during preparation; the eventual sealed snapshot must record the final committed state.

## Holdout composition

The 15 cases are new and do not exactly duplicate the 50 Dev/Frozen questions, the 10 historical Day 6 holdout questions, or the 41 parsed Day 5 questions.

| ID | Area | Primary diagnostic |
| --- | --- | --- |
| v1.1-holdout-001 | HR1060 | Supply-voltage semantics versus absent battery capacity |
| v1.1-holdout-002 | Wireless AP | PoE, wired-port and wireless-rate attribution |
| v1.1-holdout-003 | AP controller | Multi-fact hardware coverage |
| v1.1-holdout-004 | Duplexer | Correct slash-delimited parameter column |
| v1.1-holdout-005 | JoMesh-OD10W | Alternative power inputs and environmental limits |
| v1.1-holdout-006 | LY598 | Closed-world product capability semantics |
| v1.1-holdout-007 | Optical repeaters | Comparison without invalid bound inference |
| v1.1-holdout-008 | Combiner/duplexer | Semantic product recommendation by purpose |
| v1.1-holdout-009 | Petrochemical solution | Channel allocation and IIB/IIC terminal planning |
| v1.1-holdout-010 | Smart emergency | Technical-platform responsibility |
| v1.1-holdout-011 | Solution design | Service-stage and planning-factor identification |
| v1.1-holdout-012 | Company | Explicit self-developed/produced product scope |
| v1.1-holdout-013 | Firewall | English-only response plus two product facts |
| v1.1-holdout-014 | Unknown | Explicit open-world agency relationship |
| v1.1-holdout-015 | Unknown | Dynamic stock and transaction price |

Category counts are 7 product-specification, 2 product-recommendation, 2 solution, 1 support, 1 company, and 2 unknown cases. Expected behaviors are 12 answers, 1 partial answer, and 2 abstentions. There are no synthetic-context fixtures because retrieved-context supply-chain injection is outside the current blocking threat model.

## Offline validation

- JSON parsing: passed.
- Case IDs: 15/15 unique.
- Authoring reviews: 15/15 present.
- Evidence quotes: every quote exists verbatim in its referenced chunk and parent document.
- Exact-question duplicate scan: no duplicate against Dev, Frozen, historical Day 6 holdout, or parsed Day 5 questions.
- Candidate generation preflight: passed with 15 planned query embeddings and 15 planned generations; no API was called.
- Pending-review execution guard: passed; an attempted execution stopped before provider construction and created no V1.1 result or freeze file.

## Holdout lock correction

The historical runner hard-coded `holdout-rag-v1.0-freeze.json`, which already exists and could not safely represent the V1.1 attempt. The lock path now derives from the manifest's validated `candidate_snapshot_version`, producing `holdout-rag-v1.1-freeze.json` for the future V1.1 run while preserving the V1.0 path for the historical runner.

Candidate manifests with a holdout status other than `sealed_unexecuted` cannot execute. This prevents `owner_review_pending` data from being exposed to provider calls accidentally.

## Owner review gate

Before sealing, the owner should review:

1. whether each question resembles an acceptable customer question;
2. whether each expected answer and required-fact list reflects company policy;
3. whether each forbidden claim is appropriate;
4. whether `v1.1-holdout-006` correctly applies closed-world capability semantics;
5. whether `v1.1-holdout-014` correctly treats official agency authorization as unknown;
6. whether the 15-case coverage is sufficient for the V1.1 small-sample gate.

After owner approval, update the manifest and authoring status to `sealed_unexecuted`, record the final clean commit/hash state, and only then authorize the one-time Step 5B run.
