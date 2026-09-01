# Day 4 Embedding and Semantic Retrieval Design

Date: 2026-09-01
Status: Approved in chat; awaiting written-spec review

## 1. Scope

Day 4 implements the production V1 retrieval pipeline:

```text
validated Day 3 chunks
  -> document embeddings
  -> local persisted vector records
  -> query embedding
  -> exact cosine Top-K retrieval
```

The corpus currently contains 72 Chinese chunks. V1 prioritizes simplicity,
exactness, debuggability, and low operational overhead.

Day 4 explicitly excludes generation, DeepSeek calls, RAG prompts, reranking,
BM25 or full hybrid search, FAISS, Chroma, other vector databases, and agents.
It does not modify `/api/chat-stream`.

## 2. Confirmed Decisions

- Embedding provider: configurable external provider.
- V1 provider: Alibaba Cloud Model Studio (DashScope), China North 2
  (Beijing).
- V1 model: `qwen3.7-text-embedding`.
- Vector dimension: 1024.
- Similarity: cosine similarity.
- Retrieval backend: NumPy exact search.
- Default Top-K: 5, configurable at runtime.
- Persistence: one local JSONL vector-record file.
- Vector records are ignored by Git and built separately in each environment.
- Chunk vectors are reused using stable `chunk_id` and `content_hash`, with
  embedding configuration included in reuse eligibility.
- Exact entity handling uses entity-first candidates followed by dense fill.
- Evaluation reuses 18 knowledge-grounded questions from the existing V0
  baseline.
- API-backed index construction is always an explicit action. Missing or stale
  indexes fail fast and never trigger implicit paid rebuilds.

## 3. Current Corpus Assessment

The 72 Day 3 chunks contain 26,104 characters in total. Their length
distribution is:

| Statistic | Characters |
| --- | ---: |
| Mean | 362.6 |
| Median | 323 |
| P90 | 610 |
| P95 | 645 |
| Maximum | 739 |

Solution chunks average 496.4 characters. These lengths are appropriate for
the current semantic retrieval design because the chunks remain organized by
product or semantic section. Day 4 will preserve the Day 3 chunk boundaries.
Retrieval evaluation, rather than speculative rechunking, will identify any
semantic dilution.

## 4. Architecture

### 4.1 Components

`EmbeddingProvider` is the only embedding dependency visible to higher-level
code. It exposes separate document and query operations:

```python
class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError
```

The V1 `DashScopeEmbeddingProvider` hides provider-specific authentication,
the workspace URL, request construction, batching, timeouts, retries, and
error mapping.

`VectorRecordRepository` owns vector-record validation, reuse planning,
JSONL loading, and atomic persistence.

`VectorIndex` is the replaceable retrieval-backend interface. V1 implements
`NumpyExactVectorIndex`. A future FAISS or vector-database implementation can
replace it without changing the retriever or later RAG logic.

`ExactEntityResolver` performs deterministic product/model resolution without
using embedding similarity, BM25, or fuzzy full-text search.

`Retriever` coordinates entity resolution, query embedding, exact vector
search, entity-first candidate selection, dense fill, de-duplication, and
debug output.

### 4.2 Build Data Flow

1. Read and strictly validate `knowledge/chunks.jsonl` using the Day 3 model.
2. Read the existing local vector records when present.
3. Produce a build plan containing reuse, embed, delete, and invalid counts.
4. Reuse a record only when its identity, content hash, and embedding
   configuration are compatible.
5. Batch only new or changed chunk texts through
   `EmbeddingProvider.embed_documents()`.
6. Validate every returned vector before combining it with reused records.
7. Write the complete result to a same-directory temporary file.
8. Re-read and validate the temporary artifact.
9. Atomically replace the prior vector-record file.

A failed build never replaces the last valid artifact.

### 4.3 Query Data Flow

```text
query
  -> ExactEntityResolver
  -> EmbeddingProvider.embed_query()
  -> NumpyExactVectorIndex cosine scores
  -> exact-entity candidates
  -> global dense de-duplicated fill
  -> RetrievalResult[Top-K]
```

If no exact entity is resolved, the retriever returns the global cosine
Top-K. The default K is 5.

## 5. Embedding Configuration

The existing secrets are read from `.env`:

```text
DASHSCOPE_API_KEY
DASHSCOPE_WORKSPACE_ID
```

The provider base URL for the confirmed region is:

```text
https://{DASHSCOPE_WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

Non-secret retrieval configuration includes:

```text
EMBEDDING_PROVIDER=dashscope
EMBEDDING_MODEL=qwen3.7-text-embedding
EMBEDDING_DIMENSIONS=1024
EMBEDDING_TIMEOUT_SECONDS=30
EMBEDDING_MAX_RETRIES=2
RETRIEVAL_TOP_K=5
```

Defaults may be supplied in code, but provider and model values remain
configurable. Embedding configuration must not import or depend on the
DeepSeek generation client or its credentials.

The provider interface distinguishes document and query embeddings even when
the selected compatible API uses the same endpoint and request shape. This
preserves the correct boundary for a future provider that supports distinct
query/document parameters or instructions.

## 6. Vector Record Schema

The default local artifact is `knowledge/vector_records.jsonl`. Each line is a
complete record:

```json
{
  "schema_version": "1.0",
  "chunk_id": "product:hp780:content",
  "parent_document_id": "product:hp780",
  "type": "product",
  "section": "海能达 HP780",
  "text": "# 海能达 HP780",
  "language": "zh-CN",
  "content_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "source_url": "https://www.shengborun.com/two-way-radio/hp780/",
  "source_files": ["src/content/products/two-way-radio/hp780.json"],
  "metadata": {
    "product_id": "hp780",
    "slug": "hp780",
    "category_id": "two-way-radio",
    "category_name": "对讲机通信"
  },
  "embedding_provider": "dashscope",
  "embedding_model": "qwen3.7-text-embedding",
  "embedding_dimensions": 1024,
  "embedding_text_type": "document",
  "embedding": [0.0299418177, 0.0270466171]
}
```

The actual `embedding` array contains exactly 1024 finite floats.
`parent_document_hash` is deliberately excluded. `section` is copied exactly
from the Day 3 semantic section label.

The record preserves the required chunk identity, parent identity, text,
embedding, typed metadata, source/provenance, and content hash. It also keeps
type and language for validation and future filtering compatibility.

### 6.1 Reuse Eligibility

An old record is reusable only when all of the following match the requested
build:

- `chunk_id`
- `content_hash`
- embedding provider
- embedding model
- embedding dimensions
- embedding text type (`document`)

Changing the model, provider, or dimension invalidates all vectors. Changing
chunk order does not. Deleted chunk IDs are removed, new IDs are embedded,
and a stable ID with a new content hash is re-embedded.

When `chunk_id`, `content_hash`, and embedding configuration still match but a
non-embedded copied field such as provenance changes, the build reuses the
vector and refreshes the record from the current Day 3 chunk. Runtime loading
still rejects such a stale artifact until the explicit build refreshes it.

### 6.2 Validation

Output and runtime-load validation requires:

- unique chunk IDs;
- a one-to-one set match between current chunks and vector records;
- exact equality for copied chunk fields, including `section`, text,
  metadata, source, and content hash;
- matching provider configuration;
- exactly 1024 finite vector values;
- non-zero vector norm;
- no secrets or request headers in the artifact.

Any mismatch fails clearly instead of silently loading a stale index.

## 7. NumPy Exact Retrieval

The JSON vectors are loaded as a `float32` matrix. The in-memory matrix and
query vector are L2-normalized, and cosine similarity is calculated with one
matrix multiplication:

```text
scores = normalized_matrix @ normalized_query
```

The persisted record retains the provider's float output; normalization is an
in-memory index concern. Scores must be finite. Ties are ordered by
`chunk_id`, making output deterministic.

The index is immutable after load. Rebuilding records requires explicitly
creating and loading a new index.

## 8. Exact Entity Handling

The entity resolver builds a deterministic product catalog from:

- `parent_document_id`;
- `metadata.product_id`;
- `metadata.slug`;
- the first H1 product name in chunk text;
- a small explicit alias override file only when required.

Normalization uses Unicode NFKC, lowercase English letters, normalized spaces
and separators, and full identifier boundaries. Short or generic aliases use
stricter matching to prevent terms such as `AP` from producing incidental
matches. Alias collisions fail catalog construction unless explicitly
resolved.

For a query containing one or more exact entities:

1. Score chunks belonging to the resolved parent document IDs.
2. Ensure each explicitly mentioned entity contributes its best chunk when K
   permits.
3. Add remaining entity candidates by cosine score.
4. Fill unused positions from the global dense ranking.
5. De-duplicate by chunk ID.

Cosine scores are never boosted or replaced by a synthetic hybrid score.
Results therefore expose their origin so entity priority remains visible.

## 9. Retrieval Results and Debug Output

Each `RetrievalResult` contains:

```text
rank
score
match_origin          exact_entity | dense
matched_entity_ids
chunk_id
parent_document_id
section
text
content_hash
metadata
source_url
source_files
```

The debug CLI prints readable result blocks containing at minimum rank, raw
cosine score, match origin, chunk ID, section, full text, source URL, and
source files. It never prints credentials, authorization headers, or a full
provider request.

## 10. Cost and Execution Safety

The provider's official price at design time is RMB 0.0005 per 1,000 input
tokens, with a stated one-million-token free allowance valid for 90 days after
Model Studio activation. Code must not assume that free allowance remains.
See the [Alibaba Cloud text embedding API documentation](https://help.aliyun.com/zh/model-studio/developer-reference/text-embedding-synchronous-api).

The build command is plan-only by default. It reports:

- total, reusable, new/changed, deleted, and invalid record counts;
- characters requiring embedding;
- a clearly labeled conservative token estimate;
- an estimated charge using configurable pricing metadata.

A separate explicit execution flag is required to make API calls. After the
call, actual usage is reported when the provider returns it. The user funds and
controls the Alibaba Cloud account; the application never creates billing
resources or payment methods.

Runtime query embedding necessarily consumes provider tokens. Chunk vectors
are reused, but arbitrary user-query embeddings are not persisted. Unit tests
never call the real API. Before this implementation performs its first real
build or 18-query evaluation, it must show an estimate and obtain explicit
user approval.

## 11. Evaluation

`eval/retrieval_v1.json` freezes 18 knowledge-grounded questions selected from
the existing V0 baseline: the 17 answerable cases plus the partially
answerable 20 W shortwave-distance case. The current-price/inventory/delivery
case and the tool-capability case are excluded because they do not have a
relevant knowledge chunk and Day 4 does not define an abstention threshold.

Each evaluation case records:

- stable ID and query;
- category;
- expected parent document IDs;
- expected chunk IDs;
- expected entity IDs where applicable;
- whether all or any expected items are required.

The product-category overview case is represented by four explicit relevance
groups, one per category, because Day 3 preserves category identity in product
chunk metadata but has no dedicated category chunk. Each group contains the
sorted acceptable chunk IDs for that category; evaluation requires at least
one retrieved chunk from every group instead of choosing arbitrary
representative products.

The evaluator calls the production retriever with K=5 and reports:

- Hit@1, Hit@3, and Hit@5;
- mean reciprocal rank;
- expected chunk Recall@5;
- expected parent Recall@5;
- exact-entity accuracy;
- complete recall for multi-document questions.

Important per-query failures include missing expected Top-5 chunks, incomplete
multi-source recall, missed or incorrect entity resolution, expected results
only at ranks four or five, a competing similar model ahead of the exact
entity, duplicate chunks, non-finite scores, and unexpected result counts.

`eval/retrieval_v1.json` and the first generated
`eval/retrieval_v1_results.json` are committed. The result artifact records
model, dimension, K, chunk snapshot hash, run time, metrics, actual rankings,
and important failures, but no embeddings or secrets.

The first evaluation establishes an observable baseline rather than hiding
failures behind a permissive threshold. Structural failures return a non-zero
exit status. Quality failures remain explicit in the report and can become
regression gates after the baseline is reviewed.

## 12. Errors

The retrieval subsystem provides contextual errors for configuration, API,
record validation, index readiness, entity catalog construction, and
evaluation. In particular:

- missing credentials fail only when a real provider is constructed;
- a missing or stale vector file gives the explicit rebuild instruction;
- authentication, model, workspace, and region errors fail immediately;
- rate limits, timeouts, and transient server errors receive bounded retries;
- input/output count mismatches and invalid vectors fail the whole batch;
- partial builds never replace a valid artifact.

## 13. Proposed File Boundaries

```text
knowledge_pipeline/retrieval/
  __init__.py
  config.py
  models.py
  embedding.py
  records.py
  index.py
  entities.py
  retriever.py
  evaluation.py

scripts/
  build_embeddings.py
  retrieve.py
  evaluate_retrieval.py

tests/
  test_embedding_provider.py
  test_vector_records.py
  test_numpy_vector_index.py
  test_entity_resolver.py
  test_retriever.py
  test_retrieval_evaluation.py

docs/
  retrieval-schema-v1.md
  retrieval-architecture-v1.md

eval/
  retrieval_v1.json
  retrieval_v1_results.json
```

The existing untracked `scripts/test_embedding.py` remains unchanged as the
user's manual connectivity probe.

## 14. Verification Strategy

Default tests use fake providers, mock API responses, and deterministic vectors
to cover:

- provider request construction, batching, retry boundaries, and dimension
  checks;
- independent embedding/generation configuration;
- record schema and exact copied-field preservation;
- incremental reuse, additions, updates, deletions, and full invalidation;
- atomic-write failure protection;
- exact cosine ordering, tie-breaking, and configurable K;
- single-entity, multi-entity, short-alias, and alias-collision behavior;
- entity-first retrieval, dense fill, and de-duplication;
- required debug fields;
- evaluation-fixture references, metric calculations, and failure categories.

No default test requires network access or incurs provider charges. Real build
and evaluation are explicit final verification steps performed only after
separate cost approval.
