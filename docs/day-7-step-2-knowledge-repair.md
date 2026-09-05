# Day 7 Step 2 — Knowledge Schema Repair and V1.1 Rebuild

## Scope

This step repaired a semantic source-data defect and propagated the repair
through the existing production knowledge pipeline. It did not modify the
retrieval architecture, generation behavior, production API, logging policy,
or any Day 6 raw evaluation result.

## Root cause

The normalizer renders `technical_parameters[].name` and `.value` exactly as
provided by the authoritative source. Three repeater products used voltage
values under the semantically incorrect field name `电池容量`, so the incorrect
label propagated unchanged into normalized documents, chunks, and vectors.

The owner updated both the Astro content and the backend knowledge source. The
two representations were inspected and agreed on the following corrections:

| Product | Correct field | Value |
| --- | --- | --- |
| HR1060 | 电源电压 | 直流：13.6V±15% 交流：100-240V |
| SLR1000 | 电源电压 | 12V |
| SLR5300 | 电源电压 | 直流：11-14V 交流：100-240V |

The related product inventory was searched for the same unit/label mismatch.
These were the three affected records; ordinary battery-capacity values remain
labelled `电池容量`, and the existing receiver-splitter voltage field was
already labelled `电源电压`.

## Rebuild

The generated artifacts were not patched manually. They were rebuilt through
the repository's existing commands in this order:

```text
scripts/build_knowledge_documents.py
scripts/build_knowledge_chunks.py
scripts/build_embeddings.py --execute
```

Build result:

```text
documents: 61
chunks: 73
vectors: 73
average chunk characters: 359.66
median chunk characters: 323.00
minimum chunk characters: 64
maximum chunk characters: 739
```

Embedding reuse and paid-call result:

```text
reused vectors: 70
new embeddings: 3
deleted vectors: 0
characters submitted: 1054
conservative token estimate: 2108
estimated cost: CNY 0.00105400
actual input tokens: 730
provider: DashScope
model: qwen3.7-text-embedding
dimensions: 1024
```

A final plan-only run reported 73 reused, 0 to embed, 0 deleted, and zero
estimated additional cost.

## V1.1 knowledge snapshot hashes

```text
knowledge/source/products/hr1060.json
37551d105e161c494ea12ffb78ac1cc5e61418805a88dffbec18e25769820cca

knowledge/source/products/slr1000.json
553b2a83ca173488af301a872042511b260be42bd00630724e80da69915e71e3

knowledge/source/products/slr5300.json
19ece39e65e58ca24defdadeb906e431c2a42662167fa249c1fe2e9b1138f9ab

knowledge/documents.jsonl
e29a2531e48f3940b96a396cff2a8df820e4c4e967483fea9d359404cc676e67

knowledge/chunks.jsonl
2bc4676cef1a36cbd81f734bade0f47027c729eaf505d4018877deb6479cc4e2

knowledge/vector_records.jsonl
5bc8840b4db94378eb8a7abb1ecab704313f18dd311d90cac05defc578f30e03
```

The vector artifact remains local and Git-ignored under the established Day 4
policy. Its production rebuild/load workflow is the subject of revised Day 7
Step 3.

## Derived-data validation

For all three affected products:

- normalized text contains `电源电压` with the expected value;
- no affected derived record labels that value as `电池容量`;
- each chunk's `parent_document_hash` matches its document `content_hash`;
- each vector record's `content_hash` matches its chunk;
- each vector has exactly 1024 dimensions;
- provider/model remain DashScope / `qwen3.7-text-embedding`;
- source URL and Astro source-file provenance are preserved.

The fixed retrieval suite contains the same 18 questions and ground truth. Its
snapshot pin was updated from the Day 6 chunk hash to the new V1.1 chunk hash so
the default evaluation CLI fails closed against mismatched knowledge while
remaining usable with the current candidate snapshot.

## Verification

```text
Focused knowledge/chunk/vector/retriever tests:
114 tests, OK

Snapshot regression test before pin update:
10 tests, 2 failures (expected RED: old chunk snapshot hash)

Snapshot regression test after pin update:
10 tests, OK

Complete offline unittest suite after repair:
255 tests, OK
exit code: 0
```

The HTTP error/status lines emitted during the full suite are intentional
simulations from existing error-path tests.

## Day 6 preservation check

The following historical raw artifacts retain their Step 1 hashes:

```text
generation dev raw
6ace9eecd192baa1e4e0567d1223232489fc2667c153161de5935b97c2826fb4

generation frozen raw
32910b6d47daf7bf997ce0904c148fc896ebcceb9abcefa6bd25991b2cd43b8b

Day 6 holdout raw
c0882d8c378806792e6f78272dae89255d3b417a299e6ec1c354fbb1413bacef

Day 6 holdout freeze
098397ef5838ba559a0a7ceb6820aa90e52bd475610132d0c6cee689ddc3afbf
```

## Acceptance

The source/schema defect is corrected in the authoritative backend source and
in every current derived artifact. Counts, references, hashes, vector reuse,
and the complete offline test suite validate successfully. Revised Day 7 Step
2 therefore meets its acceptance criteria.
