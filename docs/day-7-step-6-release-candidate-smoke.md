# Day 7 Step 6 — Release-Candidate Smoke Verification

## Outcome

`PASS — AI Customer Support RAG V1.1 — Production Candidate`

This was a local/development verification only. No production server, Nginx configuration, deployment, restart, push, retrieval settings, prompt, knowledge, chunks, vectors, provider/model configuration, API contract, or production logging policy was changed.

## Step 5B owner disposition carried forward

- Step 5B status: `Business Acceptance PASS after owner review`
- Pre-sealed rubric result: 13/15 pass, 2/15 initially flagged
- Owner disposition: cases 006 and 007 are acceptable and non-blocking
- No system change and no holdout rerun occurred

## Offline engineering gates

- Full suite: 276/276 tests passed in 25.016 seconds before the live smoke run.
- Knowledge snapshot: PASS; 61 documents, 73 chunks, and 73 vector records.
- Embedding snapshot: DashScope `qwen3.7-text-embedding`, 1024 dimensions.
- Health: `GET /health` returned HTTP 200 with `{"status":"ok"}` and an `X-Request-ID`.
- HTTP contract tests cover `POST /api/chat-stream`, `application/x-ndjson`, `delta`, `done`, practical streamed and pre-stream error paths, request IDs, timeouts, connection retry, rate limiting, concurrency limits, input limits, bounded history, latest-message-only retrieval, and slot release.
- Logging tests confirm the production log remains limited to request ID, HTTP status, outcome, latency, and error type; retrieval/query contents are not persisted.

Knowledge snapshot hashes:

- Documents: `e29a2531e48f3940b96a396cff2a8df820e4c4e967483fea9d359404cc676e67`
- Chunks: `2bc4676cef1a36cbd81f734bade0f47027c729eaf505d4018877deb6479cc4e2`
- Vectors: `5bc8840b4db94378eb8a7abb1ecab704313f18dd311d90cac05defc578f30e03`

## Real local application-path smoke

All six required business behaviors used the real local FastAPI route, current local vector index, DashScope query embedding, and DeepSeek generation. Each successful response returned HTTP 200, `application/x-ndjson`, one or more `delta` events, a terminal `done` event, no stream error, and an `X-Request-ID`.

| Case | Retrieval evidence | Answer result | Status |
| --- | --- | --- | --- |
| Chinese product fact | HP780 ranked #1 | Correctly answered `IP68` | PASS |
| Chinese product capability | 网闸 ranked #1 | Correctly listed POP3/SMTP access and four supported mail filters/audits | PASS |
| Product recommendation | 接收机分路器 ranked #1 | Correctly recommended the receiver splitter for sharing one RF line | PASS |
| Solution | 应急自组网方案 ranked #1 | Correctly described self-powered terminals, ground/air relay, and image/voice/data communication | PASS |
| Unknown company fact | No supporting company revenue evidence | Clearly stated that 2026 revenue cannot be confirmed | PASS |
| HR1060 corrected semantics | HR1060 ranked #1 | Correctly identified `13.6V±15%` DC and `100-240V` AC as power-supply voltage | PASS |

Successful-case provider usage was 170 embedding input tokens, 38,638 DeepSeek prompt tokens (33,024 cached), and 386 completion tokens. Mean end-to-end TestClient latency was 1.151 seconds, with a 0.640-second minimum and 1.735-second maximum.

## Interrupted harness attempt

The first smoke attempt completed the HP780 case, then the old Day 5 evaluation harness rejected the second provider message before generation because its fixed 30,000-byte test budget was smaller than that valid Top-K context. The application returned a safe HTTP 500, the runner stopped immediately, and all knowledge artifacts remained unchanged; this was a smoke-harness constraint, not a production pipeline or answer failure.

The harness budget was made explicitly configurable with a tested default of 30,000 bytes, preserving Day 5 behavior. Step 6 selected 50,000 bytes and resumed only the five cases that had not completed, so the successful HP780 generation was not repeated. Across both attempts there were seven embedding calls and six generation calls; the extra embedding was the interrupted second case, which never reached DeepSeek.

Raw audit hashes:

- Attempt 1: `5c2405baebebcc2408ed4cce16f322a46cbaae59ca15ff5cba298f22bab5712d`
- Attempt 2: `7c3b2982721aa0a0947bab57b97b058c7178317fdd5664738c4aae6b6035808d`

## Acceptance

All revised Day 7 gates are satisfied: known regressions passed, the owner accepted the new holdout, HR1060 data and vectors are aligned, the knowledge rebuild workflow is documented, and the local release-candidate smoke passed without production changes. The correct classification is:

`AI Customer Support RAG V1.1 — Production Candidate`

It is not classified as Production. Deployment remains a separately authorized activity.
