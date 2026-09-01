# Embedding and Semantic Retrieval Architecture V1

## Production data flow

```text
knowledge/chunks.jsonl (Day 3, 72 chunks)
        │
        │ stable chunk_id + content_hash reuse check
        ▼
EmbeddingProvider interface
        │
        ├─ V1: DashScope qwen3.7-text-embedding, 1024 dimensions
        └─ later: replaceable local/remote provider without retrieval changes
        │
        ▼
knowledge/vector_records.jsonl (local, validated, Git-ignored)
        │
        ▼
NumpyExactVectorIndex (immutable normalized matrix)
        ▲
        │ query → same configurable embedding provider/model → 1024-d vector
        │
ExactEntityResolver ── explicit product/model candidates
        │
        ▼
entity-best hits → remaining entity hits → global dense fill → Top-K
        │
        ▼
ranked chunks + raw cosine scores + metadata + provenance
```

This is the full Day 4 boundary. It stops at retrieval results and does not
send retrieved text to DeepSeek or any other generation model.

## Why NumPy exact search

The current corpus has fewer than 100 chunks. A normalized NumPy matrix and
exact matrix-vector multiplication provide deterministic cosine scores with no
index server, migration, background process, or database schema. Equal scores
are ordered by `chunk_id`, which makes tests and debugging repeatable.

`VectorIndex` isolates the backend. A future FAISS, Chroma, or other index can
implement the same search interface without changing `Retriever` or later RAG
logic. V1 intentionally does not introduce those dependencies because the
current scale does not demonstrate a persistence, filtering, latency, or
operational benefit.

## Provider separation and configuration

`EmbeddingProvider` owns only document/query embedding. DashScope-specific
client construction and response parsing stay in
`DashScopeEmbeddingProvider`. Retrieval does not know about DashScope, and
embedding configuration is separate from any future generation-LLM provider.
DeepSeek is not imported or called by Day 4.

Non-secret defaults and overrides:

- `EMBEDDING_PROVIDER=dashscope`
- `EMBEDDING_MODEL=qwen3.7-text-embedding`
- `EMBEDDING_DIMENSIONS=1024`
- `EMBEDDING_TIMEOUT_SECONDS=30`
- `EMBEDDING_MAX_RETRIES=2`
- `EMBEDDING_PRICE_YUAN_PER_1K_TOKENS=0.0005`
- `RETRIEVAL_TOP_K=5`

Secrets remain in `.env`:

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_WORKSPACE_ID`

The workspace ID selects the confirmed Beijing-compatible endpoint. Secrets
are read only for explicit API execution and are never persisted in vector or
evaluation artifacts.

## Retrieval behavior

The default is configurable Top-K=5. Every request creates exactly one query
embedding. The exact-entity resolver independently recognizes explicit product
and model identifiers with Unicode normalization, case folding, and equivalent
space/hyphen separators. It performs no fuzzy matching.

When entities are present, retrieval adds the best chunk for each entity in
query order, adds remaining chunks belonging to those entities by raw cosine
score, then fills unused positions from global dense search. `chunk_id`
deduplication is applied throughout. Without an entity, results use global
dense order directly. Scores are not boosted or rewritten.

Debug mode prints rank, six-decimal cosine score, origin, matched entity IDs,
chunk ID, parent ID, inherited section, full text, source URL, and source files:

```powershell
python scripts/retrieve.py "海能达 HP780 的防护等级是什么？" --debug
```

This query command calls the paid external embedding provider once.

## Fixed retrieval evaluation

`eval/retrieval_v1.json` freezes 18 answerable or partially answerable queries
from baseline cases 001–017 and 019. It pins the exact Day 3 chunk-file SHA-256,
explicit expected chunks/parents/entities, and four relevance groups for the
catalog-overview question.

Reported metrics are Hit@1/3/5, mean reciprocal rank, expected chunk and parent
Recall@5, exact-entity accuracy, and complete multi-source recall. Important
failures remain visible as `missing_top_k`, `incomplete_multi_source`,
`missed_entity`, or `late_rank`; the evaluator does not silently change K or
the frozen relevance mapping.

## Operator sequence and fee boundary

```powershell
python scripts/build_embeddings.py
python scripts/build_embeddings.py --execute
python scripts/retrieve.py "海能达 HP780 的防护等级是什么？" --debug
python scripts/evaluate_retrieval.py
python scripts/evaluate_retrieval.py --execute
```

The first and fourth commands are plan-only and make no API requests. Both
`--execute` commands call the paid external provider. The retrieval query also
calls it once. Review the printed conservative CNY estimate before either
`--execute` command. Alibaba Cloud bills usage to the account behind the
configured DashScope credentials; these commands do not initiate a separate
payment or purchase.

## Explicit Day 4 exclusions

V1 contains no generation LLM, RAG prompt, answer generation, reranking,
BM25/full hybrid search, FAISS, Chroma/vector database, agent, GPU, local
embedding model, or model-serving infrastructure. The existing chat endpoint
is unchanged.

