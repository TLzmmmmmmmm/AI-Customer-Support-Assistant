# Day 7 Step 3 — Knowledge Update Workflow

## Purpose

This is the V1.1 operator workflow for producing a complete, validated local
knowledge snapshot. `scripts/build_knowledge.py` orchestrates the repository's
existing builders and does not activate an invalid or partial snapshot.

Day 7 does not deploy this snapshot.

## Architecture and data flow

```text
Authoritative website content
        ↓ owner-reviewed synchronization (outside this command)
Backend knowledge/source (build input)
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

The unified command builds documents, chunks, and vectors in a temporary staging
directory. In plan-only mode it reports the complete rebuild and embedding cost
without changing live artifacts. With `--execute`, it validates the complete
staged snapshot before replacing the three live artifacts.

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
2. Astro content and `knowledge/source` agree. Synchronizing those two source
   representations remains an explicit owner-reviewed prerequisite; the build
   command does not scrape, infer, or overwrite either one.
3. The target checkout contains the intended code and source revision.
4. The project virtual environment and locked requirements are installed.
5. The server-local `.env` contains the approved DashScope configuration and
   credentials; secrets must not be committed or copied into logs.
6. The running service is not restarted until the entire workflow passes.

The embedding execution is a paid external API operation. Always run the plan
first, report its estimated calls/tokens/cost, and obtain owner approval before
using `--execute`.

## Windows development/candidate sequence

Run from the repository root in PowerShell. The first command is always safe to
run: it validates source/documents/chunks in staging, compares staged chunks to
the current vectors, reports reuse and estimated paid work, then deletes the
staging directory without changing the live snapshot.

```powershell
# Plan only: no API request.
.\.venv\Scripts\python.exe scripts\build_knowledge.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Run only after owner approval of any reported embedding cost.
# Changed chunks may call DashScope; unchanged vectors are reused.
.\.venv\Scripts\python.exe scripts\build_knowledge.py --execute
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

# Plan only. Record and obtain approval for non-zero paid work.
.venv/bin/python scripts/build_knowledge.py

# Run only after owner approval.
.venv/bin/python scripts/build_knowledge.py --execute
```

Only after every command succeeds may the separately authorized deployment
procedure restart the service and perform a health/smoke check. This document
does not authorize `git pull`, service restart, Nginx changes, push, or any
production action.

## Fail-closed behavior

The workflow fails closed at several boundaries:

- source loading rejects malformed JSON, invalid schemas, broken relationships,
  unpublished references, and invalid source paths;
- document and chunk builders validate deterministic content and hashes inside
  a temporary staging directory;
- vector generation writes and validates its artifact in the same staging
  directory;
- a failed embedding request or staged validation preserves all three previous
  live artifacts;
- complete snapshot validation rejects missing or invalid artifacts,
  document/chunk mismatch, stale vectors, and embedding-configuration mismatch;
- only a fully valid staged snapshot reaches activation; an activation I/O
  failure triggers best-effort rollback of files already replaced;
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

The three-file replacement is process-controlled and rollback-protected, but no
ordinary filesystem can make three independent file replacements one hardware-
atomic operation. Keep the running service on its existing in-memory snapshot
until the command succeeds, and retain startup validation before restart.

Do not manually patch documents, chunks, vector records, hashes, or embeddings.
If rollback is required, restore a known source/code revision and rebuild the
complete snapshot through the same workflow.

## Snapshot validation output

The validation stage makes no external API request. A successful execution
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
