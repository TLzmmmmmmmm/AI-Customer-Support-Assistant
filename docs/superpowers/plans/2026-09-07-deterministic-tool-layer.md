# Deterministic Tool Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build three directly testable deterministic support tools with trusted provenance, structured errors, product-only semantic search, and an immutable allowlist.

**Architecture:** Add reusable provenance to the existing knowledge/retrieval models, extend the existing index and Retriever with opt-in product filtering and unique-parent ranking, and implement a `DeterministicTools` service over the validated authoritative source inventory. Exact lookup reuses `ExactEntityResolver`; semantic search reuses the injected Retriever; no chat or LLM orchestration changes are made.

**Tech Stack:** Python 3.12, Pydantic 2.13, NumPy 2.5, `unittest`

**Spec:** `docs/superpowers/specs/2026-09-06-deterministic-tool-layer-design.md`

## Global Constraints

- Work on branch `week3-tool` in the current checkout; do not create another worktree.
- Keep `POST /api/chat-stream`, NDJSON events, latest-user-message retrieval, conversation history, prompts, and frontend behavior unchanged.
- Use `knowledge/source` as the only source of exact product and contact facts.
- Reuse `ExactEntityResolver`; do not duplicate entity normalization.
- `search_products` returns at most five final unique products after product filtering and grouping.
- Do not expose retrieval score in tool results.
- Do not return the company street address.
- Do not add LLM tool selection, an agent loop, dynamic execution, query rewriting, a new index, or a new embedding model.

---

### Task 1: Reusable SourceRef and Retrieval Provenance

**Files:**
- Modify: `knowledge_pipeline/models.py`
- Modify: `knowledge_pipeline/retrieval/models.py`
- Modify: `knowledge_pipeline/retrieval/retriever.py`
- Modify: `knowledge_pipeline/retrieval/__init__.py`
- Test: `tests/test_retrieval_models.py`
- Test: `tests/test_retriever.py`

**Interfaces:**
- Produces: `SourceRef(title: str, url: str)`.
- Produces: `source_ref_from_text(text: str, source_url: str) -> SourceRef`.
- Extends: `RetrievalResult.sources: list[SourceRef]` while retaining `source_url` and `source_files`.

- [ ] **Step 1: Write failing provenance tests**

Add tests using literal values that require absolute HTTP(S) URLs, derive
`{"title": "润信达 LY198", "url": "https://www.shengborun.com/two-way-radio/ly198/"}`
from `# 润信达 LY198\n\n产品说明`, reject missing H1, and ensure a Retriever result exposes one
SourceRef without changing existing source fields.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m unittest tests.test_retrieval_models tests.test_retriever -v
```

Expected: import/model failures because `SourceRef` and `sources` do not yet exist.

- [ ] **Step 3: Implement the minimal provenance contract**

Add a strict model and deterministic parser equivalent to:

```python
class SourceRef(StrictModel):
    title: NonEmptyStr
    url: NonEmptyStr

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("must be an absolute HTTP(S) URL")
        return value


def source_ref_from_text(text: str, source_url: str) -> SourceRef:
    first_line = text.split("\n", 1)[0]
    if not first_line.startswith("# ") or not first_line[2:].strip():
        raise ValueError("text must start with a non-empty H1 source title")
    return SourceRef(title=first_line[2:].strip(), url=source_url)
```

Add required `sources` to RetrievalResult and have `_to_result` supply exactly one source.
Convert provenance construction failures to `VectorRecordValidationError` so retrieval fails closed.

- [ ] **Step 4: Run focused and context regression tests and verify GREEN**

```powershell
python -m unittest tests.test_retrieval_models tests.test_retriever tests.test_rag_context -v
```

Expected: all pass; RAG context remains unchanged.

- [ ] **Step 5: Commit**

```powershell
git add knowledge_pipeline/models.py knowledge_pipeline/retrieval/models.py knowledge_pipeline/retrieval/retriever.py knowledge_pipeline/retrieval/__init__.py tests/test_retrieval_models.py tests/test_retriever.py
git commit -m "feat: add reusable retrieval provenance"
```

### Task 2: Product Filtering and Final-Unique Top-K

**Files:**
- Modify: `knowledge_pipeline/retrieval/index.py`
- Modify: `knowledge_pipeline/retrieval/retriever.py`
- Test: `tests/test_numpy_vector_index.py`
- Test: `tests/test_retriever.py`

**Interfaces:**
- Extends: `VectorIndex.search(..., record_types: Collection[str] | None = None, unique_parent_documents: bool = False)`.
- Extends: `Retriever.retrieve(..., allowed_types: Collection[str] | None = None, unique_parent_documents: bool = False)`.
- Preserves: all omitted-option behavior.

- [ ] **Step 1: Write failing index and Retriever tests**

Create literal vector records containing product and solution types, including two
high-scoring chunks for one product and at least five distinct products. Assert that
type filtering occurs before limiting and that `top_k=5` with unique parents returns
five distinct `parent_document_id` values, keeping the highest-scoring chunk for each.
Also assert the existing default call still returns raw chunk-level results.

- [ ] **Step 2: Run the focused tests and verify RED**

```powershell
python -m unittest tests.test_numpy_vector_index tests.test_retriever -v
```

Expected: unexpected keyword argument failures for the new options.

- [ ] **Step 3: Implement filtering before grouping before limiting**

In the NumPy index:

```text
candidate record indexes
→ filter record.type when record_types is supplied
→ compute scores
→ sort by descending score and chunk_id tie-break
→ if unique_parent_documents, keep first hit per parent
→ slice to top_k
```

Thread both options through every Retriever index call. When unique parents are
requested, track both seen chunk IDs and seen parent IDs across exact-entity and dense
assembly, and over-fetch by the number of already selected parents so exact hits do not
reduce the requested final unique count.

- [ ] **Step 4: Run focused tests and verify GREEN**

```powershell
python -m unittest tests.test_numpy_vector_index tests.test_retriever -v
```

- [ ] **Step 5: Commit**

```powershell
git add knowledge_pipeline/retrieval/index.py knowledge_pipeline/retrieval/retriever.py tests/test_numpy_vector_index.py tests/test_retriever.py
git commit -m "feat: add filtered unique-parent retrieval"
```

### Task 3: Canonical Product Resolution from Structured Sources

**Files:**
- Modify: `knowledge_pipeline/core.py`
- Modify: `knowledge_pipeline/__init__.py`
- Modify: `knowledge_pipeline/retrieval/entities.py`
- Test: `tests/test_knowledge_pipeline.py`
- Test: `tests/test_entity_resolver.py`

**Interfaces:**
- Produces: `load_source_inventory(source_root: Path) -> SourceInventory`, including existing relationship validation.
- Produces: `ExactEntityResolver.from_product_sources(products: Iterable[ProductSource]) -> ExactEntityResolver`.
- Produces: `ExactEntityResolver.resolve_canonical_identifier(value: str) -> list[str]` returning canonical parent document IDs.
- Preserves: `ExactEntityResolver.from_records()` and `resolve(query)` behavior.

- [ ] **Step 1: Write failing source-inventory and canonical resolver tests**

Require the public source loader to reject invalid category relationships and require
canonical resolution to accept `LY198`, `ly198`, and separator/case variants of IDs or
slugs. Assert that a full display name and a fuzzy near-match return no match. Build a
fixture whose normalized canonical slug collides across two products and assert two
targets are returned for the tool layer to classify as ambiguous.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
python -m unittest tests.test_knowledge_pipeline tests.test_entity_resolver -v
```

- [ ] **Step 3: Refactor the existing resolver without duplicating normalization**

Keep `_normalize` as the single implementation. Store query aliases separately from a
canonical alias-to-parent-ID collection. `from_records` retains current collision-fatal
query behavior. `from_product_sources` records only published `id` and `slug` aliases,
deduplicates aliases for the same parent, and preserves multiple parents for an
ambiguous canonical alias. Exact canonical lookup normalizes the complete input once and
performs a dictionary equality lookup.

Expose a validated source loader and make `build_documents` use it so tools and the
knowledge build share the same relationship validation path.

- [ ] **Step 4: Run focused tests and verify GREEN**

```powershell
python -m unittest tests.test_knowledge_pipeline tests.test_entity_resolver -v
```

- [ ] **Step 5: Commit**

```powershell
git add knowledge_pipeline/core.py knowledge_pipeline/__init__.py knowledge_pipeline/retrieval/entities.py tests/test_knowledge_pipeline.py tests/test_entity_resolver.py
git commit -m "feat: resolve canonical products from source data"
```

### Task 4: Deterministic Tool Contracts, Service, and Registry

**Files:**
- Create: `support_tools/__init__.py`
- Create: `support_tools/models.py`
- Create: `support_tools/service.py`
- Create: `support_tools/registry.py`
- Create: `services/tools.py`
- Create: `tests/test_support_tools.py`

**Interfaces:**
- Produces: `ToolErrorCode` with all five stable codes.
- Produces: `ToolError(code, message, tool_name)`.
- Produces: `ProductDetailsResult`, `ContactInfoResult`, `ProductSearchItem`, `ProductSearchResult`.
- Produces: `DeterministicTools(inventory, site_base_url, retriever=None)` and its three public methods.
- Produces: `build_deterministic_tools(...) -> DeterministicTools`.
- Produces: `build_tool_registry(tools) -> Mapping[str, Callable[..., object]]`.

- [ ] **Step 1: Write failing tool contract and behavior tests**

Use temporary real source JSON fixtures and a narrow fake Retriever that records its
arguments. Cover:

```text
get_product_details: known, unknown, blank, non-string, >128, normalized ID, full-name rejection, ambiguous
get_contact_info: stable fields, no address, source, zero/multiple published contacts
search_products: blank, non-string, >4000, exact stripped query, top_k=5, product-only, unique parents, provenance, no score
errors: known unavailable dependency versus unexpected execution failure
registry: exactly three names and immutable
```

Expected product literals must come from fixture values that differ from production so
the tests fail if facts are hardcoded in the tool layer.

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m unittest tests.test_support_tools -v
```

Expected: module import failure because `support_tools` does not exist.

- [ ] **Step 3: Implement strict models and ToolError**

Use `StrictModel`, reuse `TechnicalParameterGroup`, and validate non-empty unique source
URLs. Define:

```python
class ToolErrorCode(str, Enum):
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    PRODUCT_NOT_FOUND = "PRODUCT_NOT_FOUND"
    AMBIGUOUS_PRODUCT = "AMBIGUOUS_PRODUCT"
    TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"
    TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
```

ToolError stores a typed code, safe message, and tool name. Never place raw exception
text in its message.

- [ ] **Step 4: Implement deterministic methods over injected dependencies**

Validate strings explicitly. Build product details and contact results only from
SourceInventory models. Resolve product IDs only through
`resolve_canonical_identifier`. Build trusted URLs using a validated base URL plus the
source model's already validated `source_path`.

Call search exactly as:

```python
self._retriever.retrieve(
    normalized_query,
    top_k=5,
    allowed_types={"product"},
    unique_parent_documents=True,
)
```

Require every returned hit to be a product with `ProductMetadata`, join identity fields
back to SourceInventory, preserve hit order and relevant text, and expose its SourceRef
without score. Missing/stale dependencies fail closed.

- [ ] **Step 5: Implement the immutable explicit registry and operational builder**

Use `MappingProxyType` over a literal mapping of the three approved names to bound
methods. The builder loads the validated inventory and accepts an optional injected
Retriever. Do not add a generic dispatcher or arbitrary attribute lookup.

- [ ] **Step 6: Run tool tests and verify GREEN**

```powershell
python -m unittest tests.test_support_tools -v
```

- [ ] **Step 7: Commit**

```powershell
git add support_tools services/tools.py tests/test_support_tools.py
git commit -m "feat: add deterministic support tools"
```

### Task 5: Manual Verification, Contract Documentation, and Full Regression

**Files:**
- Create: `scripts/verify_tools.py`
- Create: `tests/test_verify_tools_cli.py`
- Create: `docs/tool-contract-v1.md`

**Interfaces:**
- Produces: offline `product` and `contact` CLI commands.
- Produces: explicit online `search` CLI command using the existing Retriever builder.

- [ ] **Step 1: Write failing CLI behavior tests**

Call the CLI entry function with controlled arguments and capture stdout/stderr. Require
product and contact commands to construct tools without a Retriever, print validated
UTF-8 JSON, and return zero. Inject a fake Retriever builder for search and assert the
query is not rewritten. Require ToolError output to be structured and return nonzero.

- [ ] **Step 2: Run CLI tests and verify RED**

```powershell
python -m unittest tests.test_verify_tools_cli -v
```

- [ ] **Step 3: Implement the internal verification CLI**

Use `argparse` subcommands `product PRODUCT_ID`, `contact`, and `search QUERY`. Build the
real Retriever only inside the search branch. Print `model_dump(mode="json")` with
`ensure_ascii=False`; print only safe ToolError fields on failure. Do not add an HTTP
route.

- [ ] **Step 4: Write the engineering contract documentation**

Document exact input/output/error/data-source/provenance contracts, fixed limits,
offline versus external manual commands, allowlist ownership, and all Day 2+ deferrals.
Explicitly state that scenario searches return candidates and future orchestration must
also use contact information and advise consultation with professional sales.

- [ ] **Step 5: Run CLI, focused, and full regression verification**

```powershell
python -m unittest tests.test_verify_tools_cli tests.test_support_tools tests.test_retriever tests.test_numpy_vector_index tests.test_rag_context tests.test_chat_route tests.test_chat_http -v
python -m unittest discover -s tests -v
python scripts/verify_tools.py product LY198
python scripts/verify_tools.py contact
git diff --check
```

Expected: all tests pass; offline commands return canonical JSON; no search provider call
is made during automated verification.

- [ ] **Step 6: Review scope and contracts**

Compare every completion criterion in the design spec to the changed files and test
output. Confirm no prompt, route, streaming protocol, frontend, vector artifact,
embedding model, or evaluation fixture changed.

- [ ] **Step 7: Commit**

```powershell
git add scripts/verify_tools.py tests/test_verify_tools_cli.py docs/tool-contract-v1.md
git commit -m "docs: add tool verification and contract"
```
