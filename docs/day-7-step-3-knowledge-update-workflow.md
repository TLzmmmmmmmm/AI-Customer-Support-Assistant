# Day 7 Step 3 — Knowledge Update Workflow

## Purpose

This is the V1.1 operator workflow for producing a complete, validated local
knowledge snapshot. It reuses the repository's existing builders and keeps the
application stopped from activating an invalid or partial snapshot.

Day 7 does not deploy this snapshot.

## Architecture and data flow

```text
Authoritative Astro content
        ↓ owner-reviewed synchronization
Backend knowledge/source
        ↓ source schema + relationship validation
knowledge/documents.jsonl
        ↓ document hash + deterministic normalization validation
knowledge/chunks.jsonl
        ↓ document/chunk reference + hash validation
DashScope document embeddings
        ↓ chunk/vector/config validation
knowledge/vector_records.jsonl
        ↓ complete snapshot validation
Application restart / candidate activation
```

The application continues to use exact entity handling plus NumPy exact cosine
retrieval with Top-K 5. This workflow does not add a vector database, reranker,
hybrid search, query rewriting, or another framework.

## V1.1 rebuild policy

V1.1 performs a full deterministic rebuild of documents and chunks because the
corpus is small. It does **not add incremental embedding architecture**; it
preserves and formally records the existing `content_hash`-based
unchanged-vector reuse.

The embedding planner reuses an existing vector only when all of the following
still match:

```text
chunk_id
content_hash
embedding provider
embedding model
embedding dimensions
embedding text type
```

Changed chunks are re-embedded, unchanged vectors are reused, and deleted chunk
IDs are removed from the rebuilt vector artifact. This is existing Day 4
behavior, not a new incremental knowledge architecture.

## Preconditions

Before rebuilding:

1. The owner has reviewed the authoritative content change.
2. Astro content and `knowledge/source` agree.
3. The target checkout contains the intended code and source revision.
4. The project virtual environment and locked requirements are installed.
5. The server-local `.env` contains the approved DashScope configuration and
   credentials; secrets must not be committed or copied into logs.
6. The running service is not restarted until the entire workflow passes.

The embedding execution is a paid external API operation. Always run the plan
first, report its estimated calls/tokens/cost, and obtain owner approval before
using `--execute`.

## Windows development/candidate sequence

Run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\build_knowledge_documents.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

.\.venv\Scripts\python.exe scripts\build_knowledge_chunks.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Plan only: no API request.
.\.venv\Scripts\python.exe scripts\build_embeddings.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Stop here for owner cost approval when To embed is greater than zero.
.\.venv\Scripts\python.exe scripts\build_embeddings.py --execute
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

.\.venv\Scripts\python.exe scripts\validate_knowledge_snapshot.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Do not start/restart the application unless the last command prints:

```text
Knowledge snapshot validation: PASS
```

## Linux production-server sequence

`knowledge/vector_records.jsonl` is intentionally Git-ignored. Consequently,
deploying tracked code and source files is not sufficient: the production
server must build its own matching vector artifact using its local `.env` and
DashScope credentials.

After an independently authorized code/source deployment, run from the server
checkout without restarting the service:

```bash
set -euo pipefail

.venv/bin/python scripts/build_knowledge_documents.py
.venv/bin/python scripts/build_knowledge_chunks.py

# Plan only. Record and obtain approval for non-zero paid work.
.venv/bin/python scripts/build_embeddings.py

# Run only after approval.
.venv/bin/python scripts/build_embeddings.py --execute
.venv/bin/python scripts/validate_knowledge_snapshot.py
```

Only after every command succeeds may the separately authorized deployment
procedure restart the service and perform a health/smoke check. This document
does not authorize `git pull`, service restart, Nginx changes, push, or any
production action.

## Fail-closed behavior

The workflow fails closed at several boundaries:

- source loading rejects malformed JSON, invalid schemas, broken relationships,
  unpublished references, and invalid source paths;
- document and chunk builders validate deterministic content and hashes before
  replacing their output files;
- vector writing uses a temporary file, validates it, then atomically replaces
  the previous vector artifact;
- a failed embedding request preserves the previous vector artifact;
- `validate_knowledge_snapshot.py` rejects missing or invalid artifacts,
  document/chunk mismatch, stale vectors, and embedding-configuration mismatch;
- application startup independently loads chunks and vectors and refuses to
  start when they are missing or inconsistent.

If any build or validation command fails:

```text
STOP
→ do not restart the service
→ preserve the running in-memory retriever
→ diagnose/fix the source or environment
→ rerun the complete workflow
```

Do not manually patch documents, chunks, vector records, hashes, or embeddings.
If rollback is required, restore a known source/code revision and rebuild the
complete snapshot through the same workflow.

## Snapshot validation output

The validator is read-only and makes no external API request. A successful run
reports:

```text
document count
chunk count
vector-record count
embedding provider / model / dimensions
documents SHA-256
chunks SHA-256
vectors SHA-256
```

Record these values with the release candidate so the deployed vector artifact
can be matched to the tracked knowledge snapshot without storing vectors in Git.

## Future upgrade path

If corpus size or embedding cost later justifies a broader incremental pipeline,
the existing hashes provide the migration path:

```text
content_hash
→ changed documents
→ changed chunks
→ re-embed only changed chunks
```

That architecture is intentionally not added for V1.1.
