# Retrieval Vector Record Schema V1

Day 4 persists one validated JSON object per Day 3 chunk in
`knowledge/vector_records.jsonl`. The file is local and Git-ignored because it
contains reproducible 1024-dimensional vectors. Records are sorted by
`chunk_id`, encoded as UTF-8, and use LF line endings.

## Record fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Always `1.0`. |
| `chunk_id` | Stable Day 3 chunk identifier and vector-record identity. |
| `parent_document_id` | Stable parent knowledge-document identifier. |
| `type` | `catalog`, `product`, `solution`, `support`, `company`, or `contact`. |
| `section` | The Day 3 semantic section label, inherited exactly without renaming. |
| `text` | Full chunk text used as the document embedding input. |
| `language` | Currently `zh-CN`. |
| `content_hash` | SHA-256 of the Day 3 semantic chunk payload. |
| `source_url` | Public source/provenance URL. |
| `source_files` | Sorted, unique source/provenance file paths. |
| `metadata` | Type-specific Day 3 metadata. |
| `embedding_provider` | Provider identity, currently `dashscope`. |
| `embedding_model` | Configured embedding model, currently `qwen3.7-text-embedding`. |
| `embedding_dimensions` | Configured vector size, currently `1024`. |
| `embedding_text_type` | Always `document` for persisted chunks. |
| `embedding` | Finite, non-zero floating-point vector. |

`parent_document_hash` is deliberately absent. Day 4 reuse is based on chunk
identity and semantic content, while provenance fields are refreshed from the
current Day 3 artifact even when an embedding can be reused.

## Embedding reuse key

An existing vector is reusable only when all six values match:

1. `chunk_id`
2. `content_hash`
3. `embedding_provider`
4. `embedding_model`
5. `embedding_dimensions`
6. `embedding_text_type` (`document`)

This means edits to source paths or URLs can update the record without another
embedding call, while changed semantic text, model, provider, or dimensions
forces re-embedding. Deleted Day 3 chunks are removed on the next successful
build.

## Validation and failure behavior

The pipeline rejects:

- missing, empty, invalid-UTF-8, invalid-JSON, CRLF, or unsorted artifacts;
- duplicate or malformed `chunk_id` values;
- stale Day 3 `content_hash` values;
- missing, extra, or stale vector records relative to current chunks;
- mismatched copied text, section, metadata, or provenance;
- stale provider, model, dimension, or text-type configuration;
- non-numeric, non-finite, zero, or wrong-dimension vectors;
- provider responses with missing, duplicate, or out-of-range row indices.

Writes use a temporary file, re-read and validate the complete serialized
artifact, then replace the destination. A provider or validation failure leaves
the previous vector file intact. Provider errors are redacted so credentials or
raw response content are not printed.

## Build commands

Planning is the default and never calls the embedding API:

```powershell
python scripts/build_embeddings.py
```

It prints total, reused, to-embed, deleted, character, conservative-token, and
estimated CNY cost counts. After reviewing that output, explicit execution is:

```powershell
python scripts/build_embeddings.py --execute
```

Only the command with `--execute` creates the external provider and may incur
DashScope usage charges. Charges are billed through the Alibaba Cloud account
associated with `DASHSCOPE_API_KEY` and `DASHSCOPE_WORKSPACE_ID`; the script
does not perform a separate payment operation.
