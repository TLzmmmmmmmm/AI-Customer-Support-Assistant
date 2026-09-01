# Day 4 Embedding and Semantic Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production V1 pipeline that embeds validated Day 3 chunks with configurable DashScope embeddings, persists reusable local vector records, and returns debuggable exact-cosine Top-K results with minimal exact-entity handling and a fixed retrieval evaluation.

**Architecture:** Keep provider API calls, record persistence, NumPy search, entity resolution, retrieval orchestration, and evaluation in separate modules. The runtime loads a validated immutable local index, creates one query embedding, prioritizes exact entity candidates, and fills the remaining configurable Top-K positions from global dense search.

**Tech Stack:** Python 3.12, Pydantic 2.13.4, OpenAI Python SDK 3.3.1 against the Alibaba Cloud compatible endpoint, NumPy 2.5.2, `unittest`, JSONL.

**Spec:** `docs/superpowers/specs/2026-09-01-day-4-embedding-semantic-retrieval-design.md`

## Global Constraints

- Corpus input is only the validated `knowledge/chunks.jsonl` Day 3 artifact.
- V1 provider is configurable; confirmed defaults are `dashscope`, `qwen3.7-text-embedding`, 1024 dimensions, and China North 2 (Beijing).
- Read secrets only from `DASHSCOPE_API_KEY` and `DASHSCOPE_WORKSPACE_ID`; never persist or print them.
- Do not import or depend on the DeepSeek generation provider from retrieval modules.
- Persist one local `knowledge/vector_records.jsonl`; add it to `.gitignore` and never commit it.
- Reuse vectors only when `chunk_id`, `content_hash`, provider, model, dimensions, and document text type match.
- Use exact NumPy cosine search; do not add FAISS, Chroma, another vector database, BM25, hybrid retrieval, or reranking.
- Keep Top-K configurable with default K=5.
- Exact entity resolution is deterministic and isolated from dense scoring; never manufacture a hybrid score.
- Do not modify `/api/chat-stream` or add generation, RAG prompts, DeepSeek calls, or agents.
- Default tests must be offline and must not call a paid API.
- Before any live embedding build or evaluation, show estimated cost and obtain explicit user approval.

## File Map

| Path | Responsibility |
| --- | --- |
| `requirements.txt` | Add the exact NumPy runtime dependency. |
| `.gitignore` | Ignore only the local vector-record artifact and temporary siblings. |
| `knowledge_pipeline/retrieval/__init__.py` | Export the stable public retrieval API. |
| `knowledge_pipeline/retrieval/config.py` | Parse and validate retrieval configuration without import-time secret checks. |
| `knowledge_pipeline/retrieval/models.py` | Define strict vector-record, build-plan, search-hit, retrieval-result, and evaluation models. |
| `knowledge_pipeline/retrieval/embedding.py` | Define the provider protocol and DashScope-compatible implementation. |
| `knowledge_pipeline/retrieval/records.py` | Load chunks/records, plan reuse, execute document embedding, validate, serialize, and atomically write JSONL. |
| `knowledge_pipeline/retrieval/index.py` | Define `VectorIndex` and implement immutable NumPy exact cosine search. |
| `knowledge_pipeline/retrieval/entities.py` | Build and query the exact product/model entity catalog. |
| `knowledge_pipeline/retrieval/retriever.py` | Orchestrate entity-first retrieval, dense fill, de-duplication, and debug formatting. |
| `knowledge_pipeline/retrieval/evaluation.py` | Load the fixed suite, calculate metrics, and classify important failures. |
| `scripts/build_embeddings.py` | Print a dry-run plan by default and execute only with `--execute`. |
| `scripts/retrieve.py` | Load the local index and run one query with optional debug output. |
| `scripts/evaluate_retrieval.py` | Run the fixed 18-query suite and write the machine-readable result. |
| `eval/retrieval_v1.json` | Commit the fixed 18-case retrieval suite. |
| `eval/retrieval_v1_results.json` | Commit the first approved live result baseline. |
| `docs/retrieval-schema-v1.md` | Document vector-record schema, validation, and rebuild workflow. |
| `docs/retrieval-architecture-v1.md` | Document runtime flow, interfaces, entity priority, and scope boundary. |
| `tests/test_retrieval_models.py` | Configuration and Pydantic model validation. |
| `tests/test_embedding_provider.py` | Provider construction, batching, request shape, usage, and errors. |
| `tests/test_vector_records.py` | Record copying, reuse, invalidation, atomic writes, and failure safety. |
| `tests/test_numpy_vector_index.py` | Exact cosine, filters, K behavior, and deterministic ties. |
| `tests/test_entity_resolver.py` | Normalization, multi-entity order, short aliases, and collisions. |
| `tests/test_retriever.py` | Entity-first ranking, dense fill, de-duplication, and debug fields. |
| `tests/test_retrieval_evaluation.py` | Suite integrity, metrics, snapshot hash, and failure classification. |

---

### Task 1: Retrieval Configuration and Strict Core Models

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Create: `knowledge_pipeline/retrieval/__init__.py`
- Create: `knowledge_pipeline/retrieval/config.py`
- Create: `knowledge_pipeline/retrieval/models.py`
- Create: `tests/test_retrieval_models.py`

**Interfaces:**
- Consumes: `knowledge_pipeline.models.Metadata`, `StrictModel`, and the existing typed metadata models.
- Produces: `EmbeddingConfig`, `DashScopeCredentials`, `RetrievalConfig`, `VectorRecord`, `EmbeddingBatch`, `VectorBuildPlan`, `VectorBuildStats`, `SearchHit`, `EntityMatch`, and `RetrievalResult`.

- [ ] **Step 1: Add failing configuration and vector-record tests**

Create `tests/test_retrieval_models.py` with focused fixtures and these tests:

```python
import math
import unittest

from pydantic import ValidationError

from knowledge_pipeline.retrieval.config import (
    DashScopeCredentials,
    EmbeddingConfig,
    RetrievalConfig,
)
from knowledge_pipeline.retrieval.models import VectorRecord


def record_payload() -> dict:
    return {
        "schema_version": "1.0",
        "chunk_id": "product:hp780:content",
        "parent_document_id": "product:hp780",
        "type": "product",
        "section": "海能达 HP780",
        "text": "# 海能达 HP780",
        "language": "zh-CN",
        "content_hash": "a" * 64,
        "source_url": "https://www.shengborun.com/two-way-radio/hp780/",
        "source_files": ["src/content/products/two-way-radio/hp780.json"],
        "metadata": {
            "product_id": "hp780",
            "slug": "hp780",
            "category_id": "two-way-radio",
            "category_name": "对讲机通信",
        },
        "embedding_provider": "dashscope",
        "embedding_model": "qwen3.7-text-embedding",
        "embedding_dimensions": 3,
        "embedding_text_type": "document",
        "embedding": [1.0, 0.0, 0.0],
    }


class RetrievalModelTests(unittest.TestCase):
    def test_embedding_config_reads_expected_values_without_deepseek(self):
        config = EmbeddingConfig.from_mapping({
            "EMBEDDING_PROVIDER": "dashscope",
            "EMBEDDING_MODEL": "qwen3.7-text-embedding",
            "EMBEDDING_DIMENSIONS": "1024",
        })
        self.assertEqual(config.dimensions, 1024)
        self.assertFalse(hasattr(config, "api_key"))

    def test_credentials_build_the_confirmed_beijing_url(self):
        credentials = DashScopeCredentials.from_mapping({
            "DASHSCOPE_API_KEY": "secret",
            "DASHSCOPE_WORKSPACE_ID": "workspace",
        })
        self.assertEqual(
            credentials.base_url,
            "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        )

    def test_plan_configuration_does_not_require_credentials(self):
        self.assertEqual(EmbeddingConfig.from_mapping({}).dimensions, 1024)

    def test_retrieval_config_defaults_to_top_five(self):
        self.assertEqual(RetrievalConfig.from_mapping({}).top_k, 5)

    def test_vector_record_rejects_wrong_dimension(self):
        payload = record_payload()
        payload["embedding_dimensions"] = 1024
        with self.assertRaises(ValidationError):
            VectorRecord.model_validate(payload)

    def test_vector_record_rejects_non_finite_and_zero_vectors(self):
        for vector in ([math.nan, 0.0, 0.0], [0.0, 0.0, 0.0]):
            payload = record_payload()
            payload["embedding"] = vector
            with self.subTest(vector=vector), self.assertRaises(ValidationError):
                VectorRecord.model_validate(payload)

    def test_vector_record_does_not_accept_parent_document_hash(self):
        payload = record_payload()
        payload["parent_document_hash"] = "b" * 64
        with self.assertRaises(ValidationError):
            VectorRecord.model_validate(payload)
```

- [ ] **Step 2: Run the tests and confirm the module is absent**

Run:

```powershell
python -m unittest tests.test_retrieval_models -v
```

Expected: import failure for `knowledge_pipeline.retrieval`.

- [ ] **Step 3: Add NumPy and local-artifact ignore rules**

Append to `requirements.txt`:

```text
numpy==2.5.2
```

Add under a new `# Local retrieval artifacts` heading in `.gitignore`:

```text
knowledge/vector_records.jsonl
knowledge/.vector_records.jsonl.*.tmp
```

- [ ] **Step 4: Implement configuration without import-time secret validation**

In `config.py`, use frozen dataclasses and `from_mapping()` constructors. Use
these exact public fields:

```python
@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    dimensions: int
    timeout_seconds: float
    max_retries: int
    price_yuan_per_1k_tokens: Decimal

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> "EmbeddingConfig":
        provider = values.get("EMBEDDING_PROVIDER", "dashscope").strip()
        model = values.get("EMBEDDING_MODEL", "qwen3.7-text-embedding").strip()
        if provider != "dashscope":
            raise EmbeddingConfigurationError(f"unsupported provider: {provider}")
        if not model:
            raise EmbeddingConfigurationError("embedding model is required")
        try:
            dimensions = int(values.get("EMBEDDING_DIMENSIONS", "1024"))
            timeout = float(values.get("EMBEDDING_TIMEOUT_SECONDS", "30"))
            retries = int(values.get("EMBEDDING_MAX_RETRIES", "2"))
            price = Decimal(values.get("EMBEDDING_PRICE_YUAN_PER_1K_TOKENS", "0.0005"))
        except (ValueError, InvalidOperation) as error:
            raise EmbeddingConfigurationError("invalid embedding numeric configuration") from error
        if dimensions <= 0 or timeout <= 0 or retries < 0 or price < 0:
            raise EmbeddingConfigurationError("embedding numeric configuration is out of range")
        return cls(
            provider=provider,
            model=model,
            dimensions=dimensions,
            timeout_seconds=timeout,
            max_retries=retries,
            price_yuan_per_1k_tokens=price,
        )


@dataclass(frozen=True)
class DashScopeCredentials:
    api_key: str
    workspace_id: str
    base_url: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> "DashScopeCredentials":
        api_key = values.get("DASHSCOPE_API_KEY", "").strip()
        workspace_id = values.get("DASHSCOPE_WORKSPACE_ID", "").strip()
        if not api_key or not workspace_id:
            raise EmbeddingConfigurationError(
                "DASHSCOPE_API_KEY and DASHSCOPE_WORKSPACE_ID are required"
            )
        return cls(
            api_key=api_key,
            workspace_id=workspace_id,
            base_url=(
                f"https://{workspace_id}."
                "cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
            ),
        )


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> "RetrievalConfig":
        try:
            top_k = int(values.get("RETRIEVAL_TOP_K", "5"))
        except ValueError as error:
            raise EmbeddingConfigurationError("RETRIEVAL_TOP_K must be an integer") from error
        if top_k <= 0:
            raise EmbeddingConfigurationError("RETRIEVAL_TOP_K must be positive")
        return cls(top_k=top_k)
```

Replace the two protocol-body markers above with concrete parsing code. Defaults
must be provider `dashscope`, model `qwen3.7-text-embedding`, dimensions 1024,
timeout 30 seconds, max retries 2, price `Decimal("0.0005")`, and Top-K 5.
Reject unsupported providers, blank models, non-positive dimensions/K/timeout,
and negative retry counts in `EmbeddingConfig`. Reject blank workspace IDs and
keys only in `DashScopeCredentials.from_mapping()`, so plan-only commands do
not require secrets. Use `EmbeddingConfigurationError` and do not call
`load_dotenv()` at module import.

- [ ] **Step 5: Implement strict retrieval models**

In `models.py`, define `RetrievalError` subclasses and Pydantic/dataclass models.
`VectorRecord` must contain exactly the schema fields approved in the spec and
must validate SHA-256, source URL/files, metadata type, dimension equality,
finite values, and non-zero norm. Define these stable result signatures:

```python
@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: Sequence[Sequence[float]]
    input_tokens: int | None


@dataclass(frozen=True)
class VectorBuildPlan:
    chunks: Sequence[KnowledgeChunk]
    reusable_embeddings: Mapping[str, Sequence[float]]
    to_embed: Sequence[KnowledgeChunk]
    deleted_chunk_ids: Sequence[str]
    total_characters: int
    estimated_tokens: int
    estimated_cost_yuan: Decimal

    @property
    def reused_chunk_ids(self) -> Sequence[str]:
        return tuple(sorted(self.reusable_embeddings))


@dataclass(frozen=True)
class VectorBuildStats:
    records: Sequence[VectorRecord]
    reused_count: int
    embedded_count: int
    deleted_count: int
    actual_input_tokens: int | None


@dataclass(frozen=True)
class SearchHit:
    record: VectorRecord
    score: float


@dataclass(frozen=True)
class EntityMatch:
    parent_document_id: str
    alias: str
    start: int


class RetrievalResult(StrictModel):
    rank: int
    score: float
    match_origin: Literal["exact_entity", "dense"]
    matched_entity_ids: list[str]
    chunk_id: str
    parent_document_id: str
    section: str
    text: str
    content_hash: str
    metadata: Metadata
    source_url: str
    source_files: list[str]
```

- [ ] **Step 6: Run focused and existing model tests**

Run:

```powershell
python -m unittest tests.test_retrieval_models tests.test_models -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit the foundations**

```powershell
git add requirements.txt .gitignore knowledge_pipeline/retrieval tests/test_retrieval_models.py
git commit -m "feat: define retrieval configuration and models"
```

---

### Task 2: Configurable DashScope Embedding Provider

**Files:**
- Create: `knowledge_pipeline/retrieval/embedding.py`
- Create: `tests/test_embedding_provider.py`
- Modify: `knowledge_pipeline/retrieval/__init__.py`

**Interfaces:**
- Consumes: `EmbeddingConfig`, `DashScopeCredentials`, `EmbeddingBatch`, and `EmbeddingAPIError` from Task 1.
- Produces: `EmbeddingProvider` protocol and `DashScopeEmbeddingProvider` with `embed_documents(texts)` and `embed_query(text)`.

- [ ] **Step 1: Write failing provider tests with a fake SDK client**

Create a fake client whose `embeddings.create()` records keyword arguments and
returns `data[index].embedding` plus `usage.prompt_tokens`. Test:

```python
class EmbeddingProviderTests(unittest.TestCase):
    def test_documents_are_batched_at_twenty_and_usage_is_summed(self):
        provider, calls = make_provider()
        result = provider.embed_documents([f"text-{index}" for index in range(21)])
        self.assertEqual([len(call["input"]) for call in calls], [20, 1])
        self.assertEqual(result.input_tokens, 21)
        self.assertEqual(len(result.vectors), 21)

    def test_query_uses_one_input_and_configured_model_dimension(self):
        provider, calls = make_provider()
        result = provider.embed_query("HP780 的防护等级")
        self.assertEqual(calls[0]["model"], "qwen3.7-text-embedding")
        self.assertEqual(calls[0]["dimensions"], 1024)
        self.assertEqual(calls[0]["encoding_format"], "float")
        self.assertEqual(calls[0]["input"], "HP780 的防护等级")
        self.assertEqual(len(result.vectors), 1)

    def test_output_count_and_dimension_mismatches_fail(self):
        for mode in ("missing-row", "wrong-dimension"):
            provider, calls = make_provider(mode=mode)
            with self.subTest(mode=mode), self.assertRaises(EmbeddingAPIError):
                provider.embed_documents(["one", "two"])

    def test_empty_document_batch_returns_without_an_api_call(self):
        provider, calls = make_provider()
        result = provider.embed_documents([])
        self.assertEqual(result.vectors, ())
        self.assertEqual(calls, [])
```

Also mock `OpenAI` construction and assert the API key and confirmed Beijing
base URL come only from `DashScopeCredentials`, while timeout and max retries
come only from `EmbeddingConfig`.

- [ ] **Step 2: Run the provider tests and confirm failure**

```powershell
python -m unittest tests.test_embedding_provider -v
```

Expected: import failure for `EmbeddingProvider`.

- [ ] **Step 3: Implement the protocol and provider**

Use this protocol:

```python
class EmbeddingProvider(Protocol):
    @property
    def config(self) -> EmbeddingConfig:
        raise NotImplementedError

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        raise NotImplementedError

    def embed_query(self, text: str) -> EmbeddingBatch:
        raise NotImplementedError
```

Implement `DashScopeEmbeddingProvider(config, credentials, client=None)`.
Construct the OpenAI client only when no client is injected. Batch documents in groups of 20,
preserve input order using response `index`, validate row count and dimensions,
reject blank inputs, and convert SDK exceptions to a redacted
`EmbeddingAPIError`. Rely on the configured SDK retry count; do not add a
second unbounded retry loop. Keep document/query methods separate even though
the compatible endpoint currently receives the same model parameters.

- [ ] **Step 4: Run focused tests**

```powershell
python -m unittest tests.test_embedding_provider tests.test_retrieval_models -v
```

Expected: all tests pass without network access.

- [ ] **Step 5: Commit the provider**

```powershell
git add knowledge_pipeline/retrieval/embedding.py knowledge_pipeline/retrieval/__init__.py tests/test_embedding_provider.py
git commit -m "feat: add configurable DashScope embeddings"
```

---

### Task 3: Vector Record Reuse and Atomic Persistence

**Files:**
- Create: `knowledge_pipeline/retrieval/records.py`
- Create: `tests/test_vector_records.py`
- Modify: `knowledge_pipeline/retrieval/__init__.py`

**Interfaces:**
- Consumes: validated `KnowledgeChunk`, `EmbeddingProvider`, `EmbeddingConfig`, `VectorRecord`, `VectorBuildPlan`, and `VectorBuildStats`.
- Produces: `load_chunks(path)`, `load_vector_records(path, missing_ok=False)`, `validate_records_against_chunks(records, chunks, config)`, `plan_vector_build(chunks, existing, config)`, `execute_vector_build(plan, provider, output_path)`, and `write_vector_records(path, records)`.

- [ ] **Step 1: Write failing reuse and persistence tests**

Use temporary paths, two product chunks, and a fake provider. Add exact tests:

```python
class VectorRecordTests(unittest.TestCase):
    def test_unchanged_identity_hash_and_embedding_config_are_reused(self):
        plan = plan_vector_build([chunk_a], [record_for(chunk_a)], config())
        self.assertEqual(plan.reused_chunk_ids, (chunk_a.chunk_id,))
        self.assertEqual(plan.to_embed, ())

    def test_content_model_and_dimension_changes_invalidate_reuse(self):
        cases = (
            {"content_hash": "b" * 64},
            {"embedding_model": "different-model"},
            {"embedding_dimensions": 768},
        )
        for updates in cases:
            old = record_for(chunk_a).model_copy(update=updates)
            with self.subTest(updates=updates):
                plan = plan_vector_build([chunk_a], [old], config())
                self.assertEqual([item.chunk_id for item in plan.to_embed], [chunk_a.chunk_id])

    def test_deleted_records_are_reported_and_removed(self):
        plan = plan_vector_build([chunk_a], [record_for(chunk_a), record_for(chunk_b)], config())
        self.assertEqual(plan.deleted_chunk_ids, (chunk_b.chunk_id,))

    def test_non_embedding_fields_refresh_without_reembedding(self):
        stale = record_for(chunk_a).model_copy(update={"section": "错误章节"})
        plan = plan_vector_build([chunk_a], [stale], config())
        self.assertEqual(plan.reused_chunk_ids, (chunk_a.chunk_id,))
        provider = FakeProvider()
        records = execute_vector_build(plan, provider, output).records
        self.assertEqual(records[0].section, chunk_a.section)
        self.assertEqual(provider.document_calls, [])

    def test_failed_embedding_does_not_replace_the_existing_file(self):
        output.write_text("original\n", encoding="utf-8")
        with self.assertRaises(EmbeddingAPIError):
            execute_vector_build(plan, FailingProvider(), output)
        self.assertEqual(output.read_text(encoding="utf-8"), "original\n")
```

Add round-trip tests for deterministic chunk-ID ordering, a final LF, duplicate
IDs, invalid JSON, 1024 dimensions, and no `parent_document_hash` field. Test
that `load_vector_records(missing_path, missing_ok=True)` returns an empty list,
while the default call raises `VectorIndexNotReadyError`. Test that
`validate_records_against_chunks()` rejects stale copied fields at runtime even
though the explicit build path can refresh those fields without re-embedding.

- [ ] **Step 2: Run the record tests and confirm failure**

```powershell
python -m unittest tests.test_vector_records -v
```

Expected: import failure for `knowledge_pipeline.retrieval.records`.

- [ ] **Step 3: Implement load, validation, and reuse planning**

`load_chunks()` must parse each non-empty JSONL line as `KnowledgeChunk` and
raise `VectorRecordValidationError` with path and line number. The build plan
must use maps keyed by chunk ID, reject duplicates, use only the six approved
reuse fields to decide whether an embedding is reusable, and refresh every
other copied field from the current chunk when it constructs output records.
It must compute:

```python
estimated_tokens = sum(max(1, len(chunk.text) * 2) for chunk in to_embed)
estimated_cost_yuan = (
    Decimal(estimated_tokens)
    / Decimal(1000)
    * config.price_yuan_per_1k_tokens
)
```

Label this token count as a conservative estimate; never call it exact.

- [ ] **Step 4: Implement execution and atomic write**

Call the provider once through `embed_documents()` for the ordered
`to_embed` texts. Construct new `VectorRecord` instances by copying every
approved Day 3 field except `parent_document_hash`. Merge with reused records,
sort by chunk ID, serialize compact UTF-8 JSON with `ensure_ascii=False`, write
and `fsync()` a same-directory temporary file, re-read it, compare deterministic
bytes, and use `os.replace()` only after full validation. Always remove the
temporary file in `finally`.

- [ ] **Step 5: Run record and Day 3 regression tests**

```powershell
python -m unittest tests.test_vector_records tests.test_chunking_pipeline -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit persistence and reuse**

```powershell
git add knowledge_pipeline/retrieval/records.py knowledge_pipeline/retrieval/__init__.py tests/test_vector_records.py
git commit -m "feat: persist reusable vector records"
```

---

### Task 4: Immutable NumPy Exact Vector Index

**Files:**
- Create: `knowledge_pipeline/retrieval/index.py`
- Create: `tests/test_numpy_vector_index.py`
- Modify: `knowledge_pipeline/retrieval/__init__.py`

**Interfaces:**
- Consumes: ordered `Sequence[VectorRecord]` and a query vector.
- Produces: `VectorIndex.search(query_vector, top_k, parent_document_ids=None) -> list[SearchHit]` and `NumpyExactVectorIndex`.

- [ ] **Step 1: Write failing exact-search tests**

Create orthogonal three-dimensional records and assert:

```python
class NumpyVectorIndexTests(unittest.TestCase):
    def test_cosine_search_returns_descending_exact_scores(self):
        index = NumpyExactVectorIndex([record("a", [1, 0, 0]), record("b", [1, 1, 0])])
        hits = index.search([1, 0, 0], top_k=2)
        self.assertEqual([hit.record.chunk_id for hit in hits], ["product:a:content", "product:b:content"])
        self.assertAlmostEqual(hits[0].score, 1.0, places=6)

    def test_parent_filter_is_applied_before_top_k(self):
        hits = index.search([1, 0, 0], top_k=5, parent_document_ids={"product:b"})
        self.assertEqual([hit.record.parent_document_id for hit in hits], ["product:b"])

    def test_ties_are_broken_by_chunk_id(self):
        hits = index.search([1, 0, 0], top_k=2)
        self.assertEqual([hit.record.chunk_id for hit in hits], sorted(hit.record.chunk_id for hit in hits))

    def test_invalid_k_query_dimension_and_zero_query_fail(self):
        for query, k in (([1, 0], 1), ([0, 0, 0], 1), ([1, 0, 0], 0)):
            with self.subTest(query=query, k=k), self.assertRaises(VectorIndexNotReadyError):
                index.search(query, top_k=k)
```

Also assert the input record vectors remain unchanged after index construction.

- [ ] **Step 2: Run the index tests and confirm failure**

```powershell
python -m unittest tests.test_numpy_vector_index -v
```

Expected: import failure for `NumpyExactVectorIndex`.

- [ ] **Step 3: Implement normalization and deterministic search**

Build a copied `numpy.float32` matrix, validate shape and finite non-zero norms,
and normalize rows once. For each query, normalize a copied float32 vector,
calculate `matrix @ query`, filter candidate indices when parent IDs are given,
and order with the key `(-score, chunk_id)`. Return at most K hits and never
mutate persisted models.

- [ ] **Step 4: Run focused tests**

```powershell
python -m unittest tests.test_numpy_vector_index tests.test_retrieval_models -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit exact search**

```powershell
git add knowledge_pipeline/retrieval/index.py knowledge_pipeline/retrieval/__init__.py tests/test_numpy_vector_index.py
git commit -m "feat: add NumPy exact vector search"
```

---

### Task 5: Deterministic Product and Model Entity Resolution

**Files:**
- Create: `knowledge_pipeline/retrieval/entities.py`
- Create: `tests/test_entity_resolver.py`
- Modify: `knowledge_pipeline/retrieval/__init__.py`

**Interfaces:**
- Consumes: `Sequence[VectorRecord]`.
- Produces: `ExactEntityResolver.from_records(records)` and `resolve(query) -> list[EntityMatch]`.

- [ ] **Step 1: Write failing normalization and collision tests**

Use HP780, HP790Ex, XiR P8668Ex, and AP records. Test:

```python
class EntityResolverTests(unittest.TestCase):
    def test_case_space_and_hyphen_variants_resolve_the_same_model(self):
        resolver = ExactEntityResolver.from_records(records())
        for query in ("XiR P8668Ex 频段", "xir-p8668ex 频段", "XIR P8668EX 频段"):
            with self.subTest(query=query):
                self.assertEqual(
                    [match.parent_document_id for match in resolver.resolve(query)],
                    ["product:xir-p8668ex"],
                )

    def test_multiple_models_follow_first_query_occurrence(self):
        matches = resolver.resolve("比较 HP780 和 HP790Ex")
        self.assertEqual(
            [match.parent_document_id for match in matches],
            ["product:hp780", "product:hp790ex"],
        )

    def test_short_ap_alias_does_not_match_inside_unrelated_text(self):
        self.assertEqual(resolver.resolve("capacity planning"), [])

    def test_alias_collision_fails_catalog_construction(self):
        with self.assertRaises(EntityCatalogError):
            ExactEntityResolver.from_records(colliding_records())
```

Also assert duplicate aliases for multiple chunks of the same parent are legal
and de-duplicated.

- [ ] **Step 2: Run the resolver tests and confirm failure**

```powershell
python -m unittest tests.test_entity_resolver -v
```

Expected: import failure for `ExactEntityResolver`.

- [ ] **Step 3: Implement catalog construction**

For product records only, derive aliases from the parent ID suffix,
`metadata.product_id`, `metadata.slug`, and the first Markdown H1. Normalize
with `unicodedata.normalize("NFKC", value).casefold()`, treat spaces and hyphens
as equivalent separators, preserve Chinese characters, and require full
alphanumeric boundaries. Do not implement edit distance or fuzzy search.

For aliases shorter than three alphanumeric characters, require a standalone
token boundary on both sides. Sort resolved entities by their first occurrence
in the original query, then parent ID. Raise `EntityCatalogError` if one
normalized alias maps to different parents.

- [ ] **Step 4: Run entity and model tests**

```powershell
python -m unittest tests.test_entity_resolver tests.test_retrieval_models -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit entity resolution**

```powershell
git add knowledge_pipeline/retrieval/entities.py knowledge_pipeline/retrieval/__init__.py tests/test_entity_resolver.py
git commit -m "feat: resolve exact product entities"
```

---

### Task 6: Entity-First Retriever and Readable Debug Output

**Files:**
- Create: `knowledge_pipeline/retrieval/retriever.py`
- Create: `tests/test_retriever.py`
- Modify: `knowledge_pipeline/retrieval/__init__.py`

**Interfaces:**
- Consumes: `EmbeddingProvider`, `VectorIndex`, `ExactEntityResolver`, and default K.
- Produces: `Retriever.retrieve(query: str, top_k: int | None = None) -> list[RetrievalResult]` and `format_debug_results(query, results) -> str`.

- [ ] **Step 1: Write failing orchestration tests**

Use a fake provider that records query calls and a fake index with controlled
hits. Test:

```python
class RetrieverTests(unittest.TestCase):
    def test_exact_entities_are_first_and_dense_results_fill_without_duplicates(self):
        results = retriever().retrieve("比较 HP780 和 HP790Ex", top_k=5)
        self.assertEqual(
            [item.parent_document_id for item in results[:2]],
            ["product:hp780", "product:hp790ex"],
        )
        self.assertEqual(len({item.chunk_id for item in results}), len(results))
        self.assertEqual([item.rank for item in results], list(range(1, len(results) + 1)))
        self.assertTrue(all(item.match_origin == "exact_entity" for item in results[:2]))

    def test_no_entity_uses_global_dense_order(self):
        results = retriever().retrieve("如何进行应急协同指挥？", top_k=3)
        self.assertEqual([item.chunk_id for item in results], dense_chunk_ids[:3])
        self.assertTrue(all(item.match_origin == "dense" for item in results))

    def test_query_is_embedded_once_and_default_k_is_five(self):
        results = retriever().retrieve("HP780 参数")
        self.assertEqual(fake_provider.query_calls, ["HP780 参数"])
        self.assertEqual(len(results), 5)

    def test_debug_output_contains_required_readable_fields(self):
        output = format_debug_results("HP780 参数", retriever().retrieve("HP780 参数"))
        for label in ("Rank:", "Score:", "Chunk ID:", "Section:", "Text:", "Source:"):
            self.assertIn(label, output)
```

- [ ] **Step 2: Run the retriever tests and confirm failure**

```powershell
python -m unittest tests.test_retriever -v
```

Expected: import failure for `Retriever`.

- [ ] **Step 3: Implement entity-first assembly**

Reject blank queries and non-positive K. Create exactly one query embedding.
For each resolved parent in query order, request that parent's best hit and add
it if unseen. Then request remaining hits across the union of resolved parents,
ordered by raw cosine and chunk ID. Finally request global dense hits and fill
until K, always de-duplicating by chunk ID. Do not alter scores. Populate all
copied metadata and source fields in `RetrievalResult`.

- [ ] **Step 4: Implement debug formatting**

Render one block per result with rank, score to six decimal places, origin,
matched entities, chunk ID, parent ID, section, source URL, source files, and
full chunk text. Ensure no config object or provider exception content appears.

- [ ] **Step 5: Run focused retrieval tests**

```powershell
python -m unittest tests.test_retriever tests.test_numpy_vector_index tests.test_entity_resolver -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit retrieval orchestration**

```powershell
git add knowledge_pipeline/retrieval/retriever.py knowledge_pipeline/retrieval/__init__.py tests/test_retriever.py
git commit -m "feat: retrieve entity-aware semantic results"
```

---

### Task 7: Explicit Build and Query CLIs

**Files:**
- Create: `scripts/build_embeddings.py`
- Create: `scripts/retrieve.py`
- Modify: `tests/test_vector_records.py`
- Modify: `tests/test_retriever.py`

**Interfaces:**
- Consumes: configuration, record planning/execution, index, resolver, retriever, and debug formatter.
- Produces: plan-only build command, explicit `--execute` build, and one-query retrieval command.

- [ ] **Step 1: Add failing CLI tests**

Patch configuration and provider factories so no network is used. Add tests
that call each `main(argv)` directly:

```python
def test_build_cli_defaults_to_plan_only(self):
    code, stdout, stderr = run_build_cli([])
    self.assertEqual(code, 0)
    self.assertIn("Mode: plan only", stdout)
    self.assertIn("To embed:", stdout)
    self.assertIn("Estimated cost (CNY):", stdout)
    self.assertEqual(fake_provider.document_calls, [])


def test_build_cli_execute_calls_provider_and_writes_output(self):
    code, stdout, stderr = run_build_cli(["--execute", "--output", str(output)])
    self.assertEqual(code, 0)
    self.assertTrue(output.is_file())
    self.assertIn("Actual input tokens:", stdout)


def test_retrieve_cli_fails_fast_when_vector_file_is_missing(self):
    code, stdout, stderr = run_retrieve_cli(["HP780 参数", "--vectors", str(missing)])
    self.assertEqual(code, 1)
    self.assertIn("run scripts/build_embeddings.py --execute", stderr)
```

- [ ] **Step 2: Run CLI tests and confirm failure**

```powershell
python -m unittest tests.test_vector_records tests.test_retriever -v
```

Expected: CLI imports fail.

- [ ] **Step 3: Implement `build_embeddings.py`**

Use defaults:

```text
--chunks knowledge/chunks.jsonl
--output knowledge/vector_records.jsonl
--execute false
```

Call `load_dotenv()` only inside `main()`. Always print total, reused, to embed,
deleted, characters, conservative estimated tokens, and estimated CNY cost.
Return before provider construction unless `--execute` is present. With
`--execute`, construct `DashScopeEmbeddingProvider`, execute the plan, and print
actual usage when available. Catch only contextual retrieval errors and return
1 with a concise stderr message.

- [ ] **Step 4: Implement `retrieve.py`**

Accept a positional query plus `--top-k`, `--vectors`, and `--debug`. Load
dotenv, configuration, records, immutable index, entity resolver, and provider.
Validate the vector artifact against current chunks before creating the index.
Print compact JSON results normally and readable blocks with `--debug`. A
missing/stale artifact must instruct the user to run the explicit build command
and must not instantiate the provider.

- [ ] **Step 5: Run CLI and full offline tests**

```powershell
python -m unittest discover -s tests -v
```

Expected: all tests pass and no network request occurs.

- [ ] **Step 6: Commit CLIs**

```powershell
git add scripts/build_embeddings.py scripts/retrieve.py tests/test_vector_records.py tests/test_retriever.py
git commit -m "feat: add explicit embedding and retrieval commands"
```

---

### Task 8: Fixed 18-Query Evaluation and Failure Classification

**Files:**
- Create: `eval/retrieval_v1.json`
- Create: `knowledge_pipeline/retrieval/evaluation.py`
- Create: `scripts/evaluate_retrieval.py`
- Create: `tests/test_retrieval_evaluation.py`
- Modify: `knowledge_pipeline/retrieval/models.py`
- Modify: `knowledge_pipeline/retrieval/__init__.py`

**Interfaces:**
- Consumes: `Retriever`, current chunks, and the fixed evaluation suite.
- Produces: `load_evaluation_suite(path)`, `evaluate_retrieval(suite, retriever, chunks, top_k=5)`, `serialize_evaluation_result(result)`, and the evaluation CLI.

- [ ] **Step 1: Define the fixed case mapping**

Create cases for baseline IDs 001–017 and 019. Use these direct mappings:

```text
002 -> product:xir-p8668ex:content
003 -> product:hp780:content
004 -> product:hp780:content + product:hp790ex:content
005 -> product:hr1060:content
006 -> product:domestic-20w:content
007 -> product:pne380:content
008 -> solution:hotel:body:system-functions
009 -> solution:enterprise:body:solution
010 -> solution:petrochemical:body:solution
011 -> solution:civil-defense:content
012 -> solution:smart-emergency:body:solution-overview
013 -> support:solution-design:content
014 -> support:project-implementation:content
015 -> support:delivery-training:content
016 -> solution:emergency-mesh:content + solution:civil-defense:content
017 -> product:hp780:content + product:hp790ex:content
019 -> product:domestic-20w:content
```

For baseline-001, define four relevance groups, each satisfied by any chunk
whose metadata has one of these category IDs:

```text
two-way-radio
shortwave-radio
mesh-network
ict-integration
```

Materialize each group as its sorted explicit list of acceptable chunk IDs when
the fixture is created; do not select one arbitrary representative product.
The case still stores the union in `expected_chunk_ids`, and sets
`match_requirement` to `all_groups`. This is required because Day 3 contains
category provenance in product metadata but no dedicated category chunk.
For this case, Hit and recall calculations use the four relevance groups rather
than treating every product in the union as independently required.

Set `expected_entity_ids` for cases 002–007, 017, and 019. Case 004 and 017
must contain both expected product entities. Copy each query and category
verbatim from `eval/baseline_v0.json`.

- [ ] **Step 2: Write failing suite and metric tests**

Add tests:

```python
class RetrievalEvaluationTests(unittest.TestCase):
    def test_suite_contains_exactly_the_approved_baseline_cases(self):
        suite = load_evaluation_suite(EVAL_PATH)
        self.assertEqual(len(suite.cases), 18)
        self.assertEqual(
            {case.source_case_id for case in suite.cases},
            {f"baseline-{index:03d}" for index in range(1, 18)} | {"baseline-019"},
        )

    def test_every_expected_chunk_and_parent_exists(self):
        validate_suite_against_chunks(load_evaluation_suite(EVAL_PATH), load_chunks(CHUNKS_PATH))

    def test_metrics_include_hit_mrr_recall_and_entity_accuracy(self):
        result = evaluate_retrieval(suite_fixture(), scripted_retriever(), chunks_fixture(), top_k=5)
        self.assertEqual(result.metrics.hit_at_1, 0.5)
        self.assertEqual(result.metrics.hit_at_5, 1.0)
        self.assertEqual(result.metrics.mean_reciprocal_rank, 0.75)
        self.assertEqual(result.metrics.entity_accuracy, 1.0)

    def test_important_failures_are_classified(self):
        failures = evaluate_retrieval(failing_suite(), failing_retriever(), chunks_fixture()).failures
        self.assertEqual(
            {failure.kind for failure in failures},
            {"missing_top_k", "incomplete_multi_source", "missed_entity", "late_rank"},
        )
```

Define `suite_fixture()` with two cases: the first has its relevant chunk at
rank one and the second at rank two. `scripted_retriever()` returns those exact
rankings and the expected entity IDs for both cases, producing Hit@1 0.5,
Hit@5 1.0, MRR 0.75, and entity accuracy 1.0. Define `failing_suite()` with four
cases, each constructed to trigger exactly one of the four asserted failure
kinds.

Add a deterministic chunk snapshot hash test. Compute SHA-256 over the exact
bytes of `knowledge/chunks.jsonl`, not file metadata.

- [ ] **Step 3: Run evaluation tests and confirm failure**

```powershell
python -m unittest tests.test_retrieval_evaluation -v
```

Expected: import or missing-fixture failure.

- [ ] **Step 4: Implement strict evaluation models and metrics**

Add strict Pydantic models for `RelevanceGroup`, `RetrievalEvaluationCase`,
`RetrievalEvaluationSuite`, `RetrievalMetrics`, `RetrievalFailure`, and
`RetrievalEvaluationResult`. Validate unique IDs, K=5 default, non-empty
expected evidence, valid match requirement, and all expected references against
the current chunk artifact.

Calculate Hit@1/3/5 from the first relevant chunk, MRR from the first relevant
rank, group recall for baseline-001, expected-chunk and parent Recall@5, entity
accuracy, and complete multi-source recall. Classify the failures named in the
spec, including a warning when the first relevant result is rank four or five.

- [ ] **Step 5: Implement the evaluation CLI**

Use defaults:

```text
--suite eval/retrieval_v1.json
--chunks knowledge/chunks.jsonl
--vectors knowledge/vector_records.jsonl
--output eval/retrieval_v1_results.json
--top-k 5
```

Load the real provider only after the local vector artifact and suite validate.
Print query count and a conservative query-token/cost estimate, then require
`--execute` before making query API calls. Without `--execute`, exit 0 after the
plan. With it, print aggregate metrics and every important failure, and write
compact deterministic JSON with no embeddings or secrets.

- [ ] **Step 6: Run evaluation and full offline tests**

```powershell
python -m unittest tests.test_retrieval_evaluation -v
python -m unittest discover -s tests -v
```

Expected: all tests pass without network access.

- [ ] **Step 7: Commit the suite and evaluator**

Because `eval/` is currently ignored, force-add only the two intended tracked
fixture paths; do not force-add other evaluation files.

```powershell
git add knowledge_pipeline/retrieval/evaluation.py knowledge_pipeline/retrieval/models.py knowledge_pipeline/retrieval/__init__.py scripts/evaluate_retrieval.py tests/test_retrieval_evaluation.py
git add -f eval/retrieval_v1.json
git commit -m "feat: add fixed semantic retrieval evaluation"
```

---

### Task 9: Operator Documentation and Offline Release Verification

**Files:**
- Create: `docs/retrieval-schema-v1.md`
- Create: `docs/retrieval-architecture-v1.md`
- Modify: `knowledge_pipeline/retrieval/__init__.py`

**Interfaces:**
- Consumes: all public interfaces and CLIs implemented in Tasks 1–8.
- Produces: stable documented public imports and operator workflows.

- [ ] **Step 1: Write schema documentation**

Document every vector-record field, explicitly state that
`parent_document_hash` is absent, state that `section` exactly inherits the Day
3 semantic label, define the six-part reuse key, document all validation
failures, and show plan-only and explicit build commands.

- [ ] **Step 2: Write architecture documentation**

Document provider separation, local JSONL persistence, immutable NumPy cosine
search, exact-entity candidate priority, dense fill, Top-K=5 default, debug
output, evaluation commands, fee behavior, and the strict Day 4 exclusions.
Include this operator sequence:

```powershell
python scripts/build_embeddings.py
python scripts/build_embeddings.py --execute
python scripts/retrieve.py "海能达 HP780 的防护等级是什么？" --debug
python scripts/evaluate_retrieval.py
python scripts/evaluate_retrieval.py --execute
```

State that the two `--execute` commands call the paid external provider.

- [ ] **Step 3: Finalize package exports**

Export only the stable consumer surface from
`knowledge_pipeline/retrieval/__init__.py`: configuration, provider protocol and
DashScope provider, vector record, repository build/load functions, index
interface and NumPy implementation, resolver, retriever, retrieval result, and
evaluation entry points. Do not export internal normalization helpers.

- [ ] **Step 4: Run offline release verification**

```powershell
python -m unittest discover -s tests -v
python scripts/build_embeddings.py
python scripts/evaluate_retrieval.py
git diff --check
git status --short
```

Expected: all tests pass; both scripts remain plan-only and make no API calls;
diff check is clean; only intended files and the user's existing untracked
`scripts/test_embedding.py` appear.

- [ ] **Step 5: Commit documentation**

```powershell
git add docs/retrieval-schema-v1.md docs/retrieval-architecture-v1.md knowledge_pipeline/retrieval/__init__.py
git commit -m "docs: document semantic retrieval operations"
```

---

### Task 10: Approved Live Build, Evaluation Baseline, and Final Verification

**Files:**
- Local ignored output: `knowledge/vector_records.jsonl`
- Create after approved live run: `eval/retrieval_v1_results.json`

**Interfaces:**
- Consumes: the completed build and evaluation CLIs.
- Produces: a validated local vector artifact and committed first retrieval baseline.

- [ ] **Step 1: Show dry-run cost estimates**

```powershell
python scripts/build_embeddings.py
python scripts/evaluate_retrieval.py
```

Capture total/reused/to-embed counts, conservative token estimates, and CNY
estimates. Confirm `knowledge/vector_records.jsonl` remains ignored.

- [ ] **Step 2: Stop and obtain explicit user approval**

Report both estimates and ask permission to run the two `--execute` commands.
Do not continue until the user explicitly approves the paid API calls.

- [ ] **Step 3: Build the real local vector artifact after approval**

```powershell
python scripts/build_embeddings.py --execute
```

Expected: 72 validated 1024-dimensional records, reported actual usage, and a
successful atomic write to the ignored local artifact.

- [ ] **Step 4: Prove unchanged chunk vectors are reused**

```powershell
python scripts/build_embeddings.py
```

Expected: `To embed: 0`, `Reused: 72`, estimated cost zero, and no API call.

- [ ] **Step 5: Run representative debug queries**

```powershell
python scripts/retrieve.py "摩托罗拉 XiR P8668Ex 支持什么频段？" --debug
python scripts/retrieve.py "海能达 HP780 和 HP790Ex 有什么区别？" --debug
python scripts/retrieve.py "智慧应急方案如何支持协同指挥？" --debug
```

Expected: rank/score/chunk ID/section/text/source are readable; the two-model
query contains both exact entities before dense fill.

- [ ] **Step 6: Run and inspect the approved 18-query evaluation**

```powershell
python scripts/evaluate_retrieval.py --execute
```

Read every important failure in the console and
`eval/retrieval_v1_results.json`. Do not hide quality failures by changing K,
adding reranking, or changing the evaluation set. Record the failures in the
final handoff.

- [ ] **Step 7: Force-add only the approved result baseline and commit**

```powershell
git add -f eval/retrieval_v1_results.json
git commit -m "test: record day 4 retrieval baseline"
```

- [ ] **Step 8: Run final verification**

```powershell
python -m unittest discover -s tests -v
python scripts/build_embeddings.py
git diff --check
git status --short
git log --oneline -10
```

Expected: all offline tests pass; build plan reports complete reuse and zero
new document-embedding cost; no tracked secrets or vector artifact; the only
pre-existing unrelated untracked file remains `scripts/test_embedding.py`.
