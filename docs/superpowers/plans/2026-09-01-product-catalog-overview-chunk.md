# Product Catalog Overview Chunk Implementation Plan

**Goal:** Add one authoritative product-category overview document/chunk so
catalog breadth queries can retrieve all four published categories as one
grounded result.

**Architecture:** Day 2 deterministically aggregates the four validated,
published `ProductCategorySource` records into a formal `catalog:products`
document. Day 3 preserves it as one `catalog:products:overview` chunk. Day 4
embeds only that new chunk and reuses the existing 72 vectors.

## Task 1: Extend normalized knowledge models

- Add strict `CatalogMetadata(catalog_id, category_ids)`.
- Add `catalog` to document/chunk type unions and metadata validation.
- Normalize one products catalog from the fixed published category registry.
- Preserve `/products/` URL and all four category source-file provenance paths.
- Update document counts from 60 to 61.

## Task 2: Add deterministic catalog chunking

- Add one whole-document candidate with ID `catalog:products:overview`.
- Preserve the semantic section label `产品分类`.
- Update real-artifact integration tests and regenerate deterministic Day 2/3
  JSONL artifacts.

## Task 3: Extend retrieval persistence

- Allow validated catalog metadata in vector records and retrieval results.
- Keep exact product/model entity handling unchanged.
- Prove the build plan reports 72 reused vectors and one new vector.

## Task 4: Version the evaluation

- Preserve `retrieval_v1.json` and its recorded result as historical baseline.
- Add `retrieval_v1_1.json` pinned to the new chunk snapshot.
- Point baseline-001 to `catalog:products:overview`; keep the other 17 mappings
  and queries unchanged.
- Make the evaluation CLI default to the latest suite.

## Task 5: Verify and incrementally embed

- Run the full offline suite and deterministic artifact checks.
- Print the one-chunk embedding and 18-query evaluation cost estimates.
- Obtain explicit fee confirmation before the incremental `--execute` calls.
- Persist 73 local vectors, run the frozen V1.1 evaluation, record failures,
  and commit only the non-secret result artifact.

## Strict exclusions

No generation, prompt changes, reranking, BM25/hybrid retrieval, K change,
FAISS/Chroma, or agent behavior is added.
