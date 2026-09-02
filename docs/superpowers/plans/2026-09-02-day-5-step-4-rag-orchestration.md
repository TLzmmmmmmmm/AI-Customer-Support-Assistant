# Day 5 Step 4 RAG Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the validated Day 4 Retriever, Step 3 Context Builder, Step 2 Prompt Builder, and existing DeepSeek NDJSON stream while preserving the current endpoint and reliability controls.

**Architecture:** Build one application-scoped Retriever from repository-local artifacts during FastAPI lifespan startup. Keep route orchestration explicit: acquire the existing slot, retrieve with only the final user message, project evidence, build provider-ready messages, open the existing DeepSeek stream, and release the slot through the existing stream wrapper or any pre-stream exception path.

**Tech Stack:** Python 3.12, FastAPI/Starlette, OpenAI-compatible DeepSeek client, existing NumPy/DashScope retrieval components, Pydantic, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-02-day-5-full-rag-pipeline-design.md`

## Global Constraints

- Execute only Day 5 Step 4 and stop after its report.
- Preserve `POST /api/chat-stream`, `application/x-ndjson`, and the existing `delta`, `done`, and `error` event protocol.
- Retrieve exactly once for every validated request using only `payload.messages[-1].content`; do not concatenate or rewrite history.
- Preserve all bounded history for generation through `build_rag_messages()`.
- Keep one concurrency slot from before retrieval until the stream ends or a pre-stream path fails.
- Keep existing rate limiting, request IDs, logging schema, DeepSeek retry/timeouts, CORS, validation limits, and frontend behavior unchanged.
- Do not expose provenance in LLM evidence, logs, HTTP responses, or NDJSON.
- Do not add the Step 5 `retrieval_unavailable` mapping yet; retrieval faults may use the existing 500 path in this step, but every pre-stream exception must release the slot exactly once.
- Do not add score thresholds, fallback generation, reranking, query rewriting, visible citations, or frontend changes.
- Do not generate embeddings and do not call DashScope, DeepSeek, or another paid API during implementation or tests.

## File Structure

- Create `services/retrieval.py`: repository-root artifact resolution and construction of one validated production Retriever; no embedding generation.
- Create `tests/test_retrieval_service.py`: real temporary artifact assembly plus controlled provider; startup validation precedes provider construction.
- Modify `services/llm.py`: accept complete provider-ready message mappings instead of constructing system/history prompts.
- Create `tests/test_llm.py`: prove the exact message list reaches the existing DeepSeek request unchanged.
- Modify `routes/chat.py`: orchestrate retrieval, context projection, prompt construction, and existing streaming within one slot.
- Create `tests/test_chat_route.py`: prove latest-message-only retrieval, full-history generation input, unchanged NDJSON/media type, and slot cleanup.
- Modify `main.py`: FastAPI lifespan constructs and stores one application-scoped Retriever.
- Create `tests/test_main.py`: prove one startup construction populates `app.state.retriever` for the lifespan.

---

### Task 1: Construct the production Retriever without network work

**Files:**
- Create: `services/retrieval.py`
- Create: `tests/test_retrieval_service.py`

**Interfaces:**
- Consumes: repository-local `knowledge/chunks.jsonl`, `knowledge/vector_records.jsonl`, `os.environ`, and existing Day 4 configuration/classes.
- Produces: `build_retriever(...) -> Retriever` and `create_embedding_provider(config, credentials)` as the provider-specific construction seam.

- [ ] **Step 1: Write failing service-construction tests**

Create a valid two-dimensional temporary `KnowledgeChunk` and matching `VectorRecord`, serialize them with existing production serializers, and use this controlled query provider:

```python
class FakeEmbeddingProvider:
    def __init__(self):
        self.query_calls = []

    def embed_query(self, text):
        self.query_calls.append(text)
        return EmbeddingBatch(vectors=([1.0, 0.0],), input_tokens=1)
```

Patch only `services.retrieval.create_embedding_provider`, call:

```python
retriever = build_retriever(
    values={
        "EMBEDDING_DIMENSIONS": "2",
        "DASHSCOPE_API_KEY": "test-key",
        "DASHSCOPE_WORKSPACE_ID": "test-workspace",
    },
    chunks_path=chunks_path,
    vector_records_path=vectors_path,
)
results = retriever.retrieve("HP780 参数", top_k=1)
```

Assert the fake saw exactly one query and the real Retriever returned the matching chunk. Add a missing-vector test asserting `VectorIndexNotReadyError` and that the provider seam was not called.

- [ ] **Step 2: Run service tests and verify RED**

```powershell
$env:PYTHONPATH = (Resolve-Path -LiteralPath '.venv\Lib\site-packages').Path
$day5Python = 'C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $day5Python -m unittest tests.test_retrieval_service -v
```

Expected: assertion failures identifying the missing `services.retrieval` module, not an external-provider error.

- [ ] **Step 3: Implement minimal production assembly**

Create `services/retrieval.py` with repository-root defaults and this flow:

```python
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS_PATH = REPOSITORY_ROOT / "knowledge" / "chunks.jsonl"
DEFAULT_VECTOR_RECORDS_PATH = (
    REPOSITORY_ROOT / "knowledge" / "vector_records.jsonl"
)


def create_embedding_provider(config, credentials):
    return DashScopeEmbeddingProvider(config, credentials)


def build_retriever(
    *,
    values: Mapping[str, str] | None = None,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    vector_records_path: Path = DEFAULT_VECTOR_RECORDS_PATH,
) -> Retriever:
    resolved_values = os.environ if values is None else values
    embedding_config = EmbeddingConfig.from_mapping(resolved_values)
    retrieval_config = RetrievalConfig.from_mapping(resolved_values)
    chunks = load_chunks(chunks_path)
    records = load_vector_records(vector_records_path)
    validate_records_against_chunks(records, chunks, embedding_config)
    index = NumpyExactVectorIndex(records)
    resolver = ExactEntityResolver.from_records(records)
    credentials = DashScopeCredentials.from_mapping(resolved_values)
    provider = create_embedding_provider(embedding_config, credentials)
    return Retriever(
        embedding_provider=provider,
        vector_index=index,
        entity_resolver=resolver,
        default_top_k=retrieval_config.top_k,
    )
```

Provider construction may create an SDK client but must not make an API request. Artifact validation must occur before provider construction.

- [ ] **Step 4: Run service tests and verify GREEN**

Run the Task 1 test command again. Expected: both tests pass with zero network calls.

### Task 2: Make DeepSeek accept provider-ready messages

**Files:**
- Modify: `services/llm.py`
- Create: `tests/test_llm.py`

**Interfaces:**
- Consumes: `messages: Sequence[Mapping[str, str]]` already containing the system message, bounded history, and final RAG user envelope.
- Produces: unchanged DeepSeek streaming response from `open_chat_stream()` and unchanged compatibility behavior from `stream_chat()`.

- [ ] **Step 1: Write a failing provider-message test**

Patch `services.llm.client.chat.completions.create`, call `open_chat_stream()` with a literal provider-ready list containing `system`, earlier `user`/`assistant`, and the final RAG `user` message. Assert the request's `messages` value equals that list exactly and that model, streaming, disabled thinking, retry settings, and returned stream behavior remain unchanged.

- [ ] **Step 2: Run LLM tests and verify RED**

```powershell
& $day5Python -m unittest tests.test_llm -v
```

Expected: failure because the current function tries to read `.role` and `.content` from message dictionaries and prepends `SYSTEM_PROMPT` itself.

- [ ] **Step 3: Move all prompt ownership out of the client**

In `services/llm.py`:

```python
from collections.abc import Iterator, Mapping, Sequence


def _copy_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    return [dict(message) for message in messages]
```

Change both `open_chat_stream()` and the retained `stream_chat()` compatibility function to accept this input and pass `_copy_messages(messages)` directly as the SDK `messages` argument. Remove imports of `ChatMessage` and `SYSTEM_PROMPT`; do not modify retry, timeout, model, `stream=True`, or disabled-thinking logic.

- [ ] **Step 4: Run LLM tests and verify GREEN**

Run `tests.test_llm` plus `tests.test_prompts`. Expected: provider-message and Prompt Builder tests pass.

### Task 3: Orchestrate RAG inside the existing route and slot

**Files:**
- Modify: `routes/chat.py`
- Create: `tests/test_chat_route.py`

**Interfaces:**
- Consumes: validated `ChatRequest`, request-scoped access to `request.app.state.retriever`, Step 3 `build_retrieved_context()`, and Step 2 `build_rag_messages()`.
- Produces: the same `StreamingResponse` and NDJSON event protocol as before.

- [ ] **Step 1: Write failing orchestration and cleanup tests**

Use a fake Retriever that records queries and returns one complete `RetrievalResult`. Patch only the slot functions, `open_chat_stream`, and request logging. Call `chat_stream()` directly with a three-message conversation and the fake Retriever dependency, consume `response.body_iterator`, and assert:

```python
fake_retriever.queries == ["它的防护等级是什么？"]
response.media_type == "application/x-ndjson"
decoded_events == [{"type": "done"}]
release_llm_slot.assert_called_once_with()
```

Assert the captured DeepSeek messages preserve the first user/assistant messages and contain the final JSON envelope with exactly the returned hit's `type`, `section`, and `text`. Add a second test whose Retriever raises `RuntimeError`; assert the error propagates for the existing 500 handler, DeepSeek is not called, and the acquired slot is released exactly once.

- [ ] **Step 2: Run route tests and verify RED**

```powershell
& $day5Python -m unittest tests.test_chat_route -v
```

Expected: failures because the current route has no Retriever dependency and sends raw `ChatMessage` objects directly to DeepSeek.

- [ ] **Step 3: Add minimal explicit orchestration**

Add:

```python
def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever
```

Inject it into `chat_stream()`. Immediately after slot acquisition and inside the existing pre-stream `try` block:

```python
results = retriever.retrieve(payload.messages[-1].content)
retrieved_context = build_retrieved_context(results)
provider_messages = build_rag_messages(
    payload.messages,
    retrieved_context,
)
stream = open_chat_stream(provider_messages)
```

Keep existing OpenAI exception mappings unchanged. Add one final `except Exception` that releases the slot then re-raises, allowing the existing global 500 handler to respond. Do not catch `RetrievalError` specially until Step 5.

- [ ] **Step 4: Run route tests and verify GREEN**

Run `tests.test_chat_route`, `tests.test_prompts`, and `tests.test_rag_context`. Expected: all pass and no real API is called.

### Task 4: Store one Retriever for the FastAPI lifespan

**Files:**
- Modify: `main.py`
- Create: `tests/test_main.py`

**Interfaces:**
- Consumes: `services.retrieval.build_retriever()`.
- Produces: `app.state.retriever` available to the route for the complete application lifespan.

- [ ] **Step 1: Write a failing lifespan test**

Patch `main.build_retriever` to return a sentinel. Enter `main.app.router.lifespan_context(main.app)` and assert the sentinel is present as `app.state.retriever` and the builder was called exactly once. This test never loads real artifacts or creates a real provider.

- [ ] **Step 2: Run lifespan test and verify RED**

```powershell
& $day5Python -m unittest tests.test_main -v
```

Expected: failure because the current FastAPI application has no custom lifespan and never creates a Retriever.

- [ ] **Step 3: Add the application lifespan**

Define before application construction:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = build_retriever()
    try:
        yield
    finally:
        del app.state.retriever


app = FastAPI(lifespan=lifespan)
```

Do not catch startup retrieval/configuration/artifact exceptions: a broken production index must fail startup clearly. Do not generate embeddings at startup.

- [ ] **Step 4: Run lifespan and orchestration tests and verify GREEN**

Run `tests.test_main`, `tests.test_retrieval_service`, `tests.test_chat_route`, and `tests.test_llm`. Expected: all pass without secrets leaving process memory and without network calls.

### Task 5: Full verification and Step 4 commit

**Files:**
- Verify only the files listed above.

**Interfaces:**
- Consumes: all four completed Task interfaces.
- Produces: a tested Step 4 commit; no Step 5 behavior.

- [ ] **Step 1: Run full backend regression**

```powershell
& $day5Python -m unittest discover -s tests -v
```

Expected: all existing and new backend tests pass; no external API request occurs.

- [ ] **Step 2: Audit compatibility and scope**

```powershell
git diff --check
git status --short
git diff --stat
git diff --name-only
```

Expected implementation changes are exactly `services/retrieval.py`, `services/llm.py`, `routes/chat.py`, `main.py`, and four matching test files. Confirm `config.py`, concurrency/rate-limit/logging/error modules, frontend, knowledge artifacts, vector artifacts, and Day 4 retrieval code remain unchanged.

- [ ] **Step 3: Commit Step 4**

```powershell
git add -- services/retrieval.py services/llm.py routes/chat.py main.py tests/test_retrieval_service.py tests/test_llm.py tests/test_chat_route.py tests/test_main.py
git diff --cached --check
git commit -m "feat: connect retrieval to chat streaming"
```

- [ ] **Step 4: Report and stop**

Report inspected files, prior boundaries, missing orchestration, exact changes, RED/GREEN evidence, final regression count, protocol/reliability compatibility, absence of API calls, and the Step 5 concern: known retrieval failures still require their approved 503 `retrieval_unavailable` mapping and no-generation assertion. Stop without implementing Step 5.
