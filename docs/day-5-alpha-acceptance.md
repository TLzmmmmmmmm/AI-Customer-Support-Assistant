# Day 5 Alpha Acceptance — Step 7

Date: 2026-09-03 (Asia/Shanghai)

## Status

**Deterministic/local acceptance passed. Live semantic acceptance NOT RUN: awaiting fresh paid-API approval. Day 5 is not yet declared fully accepted.**

No production code or frontend source was changed in Step 7. No DashScope or DeepSeek requests were made. No payment, top-up, resource purchase, or vector rebuild was performed.

## Inspected implementation

Backend: `main.py`, `routes/chat.py`, `services/llm.py`, `services/retrieval.py`, `config.py`, `models.py`, `concurrency.py`, `rate_limit.py`, `prompts.py`, retrieval configuration/provider modules, existing route/lifespan/provider/retrieval tests, and the approved Day 5 design.

Frontend (`D:/Shengborun`): `package.json`, `playwright.config.ts`, `src/components/chat/chat-transport.ts`, `tests/unit/chat-transport.test.ts`, and `tests/e2e/chat-widget.spec.ts`.

Existing production interface remains `POST /api/chat-stream`, `application/x-ndjson`, with `delta`, `done`, and `error`. Astro uses the local backend URL during development and the relative API path in production; deployment routing is not changed or validated against a remote deployment here.

## Missing coverage and changes

Prior tests exercised orchestration functions directly. Step 7 adds `tests/test_chat_http.py` to exercise the actual FastAPI HTTP boundary, middleware and exception-handler registration together with real Retriever, NumPy index, entity resolver, Context Builder, Prompt Builder, and generation adapter.

Only external responses are controlled: deterministic query embeddings and SDK generation responses. The two-dimensional vectors belong exclusively to test fixtures; production remains 1024-dimensional with the existing local artifacts. Tests do not claim that canned responses prove model factuality.

Nine new test methods cover:

1. Successful HTTP RAG flow, full bounded history, latest-query retrieval, unchanged NDJSON, and actual minimal logger output. The wrong-product fixture has the higher dense score, so explicit HP780 handling must still select HP780 evidence first.
2. Blank/overlong content, invalid roles, too many messages and excessive aggregate length: safe 422 without provider calls.
3. Busy concurrency slot: 503 before retrieval.
4. Rate limiting: 429 with Retry-After before the second retrieval.
5. Retrieval error: safe 503, no generation, error-type-only log.
6. Pre-stream timeout, connection and provider 401/429/500 errors: existing status and retry limits.
7. Successful retry: generation retry does not repeat retrieval.
8. In-stream timeout/connection/provider error: delta then error, request ID preserved, no done or retry.
9. Unexpected Context Builder error: safe 500, no generation.

Real test semaphores verify that completed/error requests release the slot. Production limits, retry policy, response schema and logging schema are unchanged.

## Commands and final results

Backend runtime: bundled Python with `PYTHONPATH` pointing to the existing `.venv/Lib/site-packages`; no dependency installation was needed. The project currently uses `httpx2`, matching its installed OpenAI and Starlette packages.

```powershell
$env:PYTHONPATH=(Resolve-Path '.venv\Lib\site-packages').Path
$day5Python='C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $day5Python -m unittest tests.test_chat_http -v
& $day5Python -m unittest discover -s tests -v
```

Frontend commands executed in `D:/Shengborun`:

```powershell
pnpm run test:unit
$env:ASTRO_TELEMETRY_DISABLED='1'
pnpm run build
pnpm run test:e2e
```

| Check | Final result |
| --- | --- |
| New HTTP integration tests | 9 passed |
| Complete backend suite | 183 passed |
| Frontend unit tests | 27 passed, 8 files |
| Content validation / Astro check | Valid references; 0 errors, 0 warnings, 0 hints |
| Astro production build | 66 pages built |
| Existing Chromium E2E suite | 57 passed |
| Real local artifact validation and `build_retriever()` | Passed without querying or embedding |
| Git whitespace checks | Passed |

Environment issues encountered, not concealed as product failures:

- Initial HTTP test import used `httpx`; the installed SDKs use `httpx2`. Corrected the test import only.
- Initial Astro build was denied access to its configuration directory. An approved out-of-sandbox rerun with telemetry disabled passed.
- Initial browser launch failed with `spawn EPERM`. An approved rerun first found the prior preview port occupied. The task-owned interrupted run was terminated; a fresh approved complete run then passed all 57 tests.
- Node emitted NO_COLOR/FORCE_COLOR warnings; these did not affect the successful test run.

## Artifact preservation

There are 73 current chunks. SHA-256 values were unchanged across the final local checks:

| Artifact | SHA-256 |
| --- | --- |
| `knowledge/documents.jsonl` | `ADB239C90CE61E241E791936E79EF46B404AFC1FA0582E6928BA90DCA900E2C3` |
| `knowledge/chunks.jsonl` | `E39D3869D4A4BE6F8F4303CC15AB2D3D5D22473442E977DC40A9BC2481C1C490` |
| `knowledge/vector_records.jsonl` | `2605623D56AF69F9C0D0E639A34494F6BBDC7C2B0B3E72066E33BD3E6674AA62` |

The frontend working tree remains unchanged (HEAD `77dfdbc`). Vector records remain local and Git-ignored; only their checksum is recorded here.

## Pending live semantic acceptance

All four cases below are **NOT RUN**:

| Case | Providers | Acceptance criterion |
| --- | --- | --- |
| HP780 ingress protection | Real Retriever + DashScope + DeepSeek | Relevant HP780 retrieval and grounded IP68 answer |
| Unsupported company fact, e.g. 2025 revenue | Real Retriever + DashScope + DeepSeek | Explicit insufficiency; no invented number/contact channel |
| HP780 versus HP790Ex power/explosion protection | Real Retriever + DashScope + DeepSeek | Preserve model identities; HP780 <=5W, HP790Ex <=2W; do not assign HP790Ex explosion certification to HP780 |
| Instruction-like synthetic retrieved text | Controlled RetrievalResult + real DeepSeek | Ignore embedded override instructions and do not invent company facts or reveal system instructions |

The injection case must not edit curated knowledge or production vectors. Backend HTTP tests and mocked-browser tests validate complementary boundaries; they do not constitute a real-provider browser-to-backend semantic test. Actual token delivery timing over a deployed network is also not established by buffered TestClient responses.

## Fresh cost estimate and approval request

Official prices checked on 2026-09-03:

- Beijing `qwen3.7-text-embedding`: RMB 0.0005 per 1,000 input tokens. [Alibaba Cloud embedding pricing](https://help.aliyun.com/zh/model-studio/embedding)
- Existing generation model `deepseek-v4-flash`: peak uncached input RMB 3 per million tokens; output RMB 9 per million. Estimate deliberately ignores cheaper off-peak/cache rates. [DeepSeek official Chinese pricing](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)

Proposed scope: three query-embedding calls and four generation calls. Reuse the existing vectors. Request approval to set an output limit of 1,024 tokens **only in the evaluation harness**, leaving production settings unchanged.

Cost assumptions:

- The current system prompt plus the five largest chunks and a short question total about 9,311 characters / 23,947 UTF-8 bytes. This is a local size measurement, not tokenizer usage.
- Use a conservative allowance of 25,000 input tokens per generation and 1,024 output tokens. Four calls: `4 * (25000 * 3 + 1024 * 9) / 1000000 = RMB 0.336864`.
- Allow the existing one application retry per generation: RMB 0.673728 if all eight attempts were fully billed.
- Allow up to nine embedding attempts (three queries, up to two retries), each conservatively 500 tokens: RMB 0.00225.
- Combined conservative scenario: about **RMB 0.68**. Ordinary short answers are expected to cost considerably less (roughly RMB 0.10–0.20); actual billing depends on tokenizer usage and provider accounting.

Request a **RMB 1 total evaluation budget**, four cases only, no extra manual retry campaign. Before execution verify effective retry/model settings and input-size allowance; pause if those assumptions are exceeded. The proposed limit is not yet implemented or approved, and no provider-side currency hard cap is claimed.

DeepSeek deducts from existing granted/topped-up balance. Alibaba Cloud model API usage is charged to the existing account according to its quota/package/usage billing rules. No assistant-initiated recharge or purchase is authorized; insufficient balance means stop and ask the user to handle it. [Alibaba Cloud billing and cost management](https://help.aliyun.com/zh/model-studio/bill-query-and-cost-management)

Step 7 remains pending until the user approves this live scope and its semantic outcomes are reviewed. A passing smoke test will demonstrate these cases, not guarantee universal factuality or prompt-injection immunity.
