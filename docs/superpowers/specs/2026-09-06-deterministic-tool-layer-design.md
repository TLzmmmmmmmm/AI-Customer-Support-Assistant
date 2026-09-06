# Deterministic Tool Layer and Provenance Contract Design

## Purpose

Week 3 Day 1 adds three deterministic backend tools and the contracts needed
to expose their results safely to a future orchestration layer:

- `search_products(query)`
- `get_product_details(product_id)`
- `get_contact_info()`

The tools must be directly testable without any LLM tool-selection logic. This
work does not add an agent loop, function-calling integration, citation
rendering, query rewriting, new retrieval infrastructure, new public endpoint,
or frontend behavior.

## Existing Behavior That Must Remain Stable

- The external endpoint remains `POST /api/chat-stream`.
- Streaming remains NDJSON with `delta`, `done`, and `error` events.
- The RAG flow remains Retriever → Context Builder → Prompt Builder →
  `open_chat_stream()`.
- Retrieval continues to use only the latest user message.
- Ordinary RAG retrieval remains chunk-oriented and searches all supported
  knowledge types unless its caller explicitly supplies a backend filter.
- Day 1 does not connect the tool registry to the chat route.

## Architectural Approach

Create a focused deterministic tool domain layer. A `DeterministicTools`
object owns a validated source inventory and receives the existing `Retriever`
as an optional injected dependency. Its three public methods expose only the
LLM-controlled arguments in the tool contracts.

The tool layer reuses the current authoritative structured data under
`knowledge/source`. It does not contain or maintain a second copy of product
specifications, category names, company details, phone numbers, or email
addresses. Product search delegates to the existing embedding and vector-index
stack.

The conceptual flow is:

```text
knowledge/source
→ validated SourceInventory
→ ExactEntityResolver (canonical product ID/slug resolution)
→ DeterministicTools
   ├─ get_product_details()
   ├─ get_contact_info()
   └─ search_products() → existing Retriever
→ immutable ToolRegistry
```

## Source of Truth and Dependency Construction

The existing structured source path, `knowledge/source`, is the sole runtime
source of exact product and contact facts. The tool builder loads and validates
that inventory once and passes it to `DeterministicTools`.

Published product records are joined to their existing product-category
records through `category_id`. Published contact information is read from the
existing contact record. The tool layer never parses product specifications or
contact facts back out of Markdown or retrieval chunks.

The Retriever dependency is optional at object construction so exact product
and contact tools can be exercised offline. Calling `search_products` without
a Retriever produces `TOOL_UNAVAILABLE`. Production or manual search
construction injects the existing Retriever instance; it does not create a
second vector index or embedding implementation.

## Provenance Contract

Use the project's strict Pydantic modeling convention:

```python
class SourceRef(StrictModel):
    title: str
    url: str
```

Both fields are non-empty. `url` must be an absolute HTTP(S) URL. Collections
of sources must not contain duplicate URLs so later citation code can
deduplicate by URL.

For deterministic tools, titles come from existing backend-owned structured
names or normalized knowledge titles. URLs are deterministically constructed
from the trusted public-site base URL and the source record's validated
`source_path`. The LLM never provides a title or URL.

For RAG retrieval, every current knowledge chunk has a pipeline-enforced first
line in the form `# <normalized document title>`. `RetrievalResult` derives a
`SourceRef` from that validated H1 and the persisted canonical `source_url`.
Failure to find a valid H1 is a backend data-contract failure; no title is
guessed or generated. Existing `source_url` and `source_files` fields remain on
`RetrievalResult` for compatibility, while a new `sources` field exposes the
reusable provenance contract.

This approach avoids changing the persisted KnowledgeChunk and VectorRecord
schemas, embedding reuse keys, or sealed evaluation snapshot hashes.

## Tool Result Contracts

All result models use `StrictModel`, reject unknown fields, and validate that
their source lists contain unique URLs.

### ProductDetailsResult

```text
product_id: str
name: str
category_id: str
category_name: str
key_features: list[str]
product_features: str
technical_parameters: list[TechnicalParameterGroup]
sources: list[SourceRef]
```

`TechnicalParameterGroup` and its item model are reused from the current
knowledge models. The result does not add an application field because the
authoritative product records do not contain product-to-scenario relationships.

### ContactInfoResult

```text
company_name: str
duty_phone: str
email: str
sources: list[SourceRef]
```

The result deliberately excludes the company's street address. Although the
website currently displays an address, the AI knowledge contract intentionally
excludes it and the customer-support AI must not return it.

### ProductSearchResult

```text
products: list[ProductSearchItem]

ProductSearchItem:
    product_id: str
    name: str
    category_id: str
    category_name: str
    relevant_content: str
    sources: list[SourceRef]
```

Each `product_id` appears at most once. `relevant_content` is the highest-ranked
matching source chunk returned by the Retriever and is not rewritten or
summarized by the tool. The public tool result does not expose cosine score;
ordering expresses relative retrieval rank, while the underlying
`RetrievalResult` retains its score for diagnostics.

## Error Contract

Tool failures use one structured exception type:

```python
ToolError(
    code: ToolErrorCode,
    message: str,
    tool_name: str,
)
```

The stable error codes are:

- `INVALID_ARGUMENT`: an argument is not a string, becomes blank after
  trimming/normalization, or exceeds its length limit.
- `PRODUCT_NOT_FOUND`: no canonical product ID or slug matches.
- `AMBIGUOUS_PRODUCT`: a normalized canonical ID or slug maps to more than one
  product.
- `TOOL_UNAVAILABLE`: required authoritative data, Retriever, embedding
  provider, or vector index is unavailable or invalid.
- `TOOL_EXECUTION_ERROR`: an unexpected internal execution failure occurs.

Messages are safe and stable. An original Python exception may be retained
through exception chaining for backend diagnosis, but raw exception details
are not included in the future LLM-facing contract.

## Canonical Product Resolution

Canonical product lookup must reuse `ExactEntityResolver`; the tool layer must
not implement a second entity-normalization function.

Extend the resolver with a structured-product catalog constructor and an exact
canonical-resolution method. The new catalog uses only `ProductSource.id` and
`ProductSource.slug`. It uses the resolver's existing NFKC, case-folding,
whitespace, and hyphen/separator normalization. Resolution requires equality
with the complete normalized input and does not use substring or fuzzy
matching.

Consequently, `LY198` and `ly198` resolve to the same canonical product, while
`润信达 LY198` is not accepted by `get_product_details`. The existing
query-oriented `ExactEntityResolver.resolve()` continues to recognize display
names for ordinary RAG retrieval and retains its current semantics.

A canonical alias with no target yields `PRODUCT_NOT_FOUND`; a canonical alias
with multiple targets yields `AMBIGUOUS_PRODUCT`. After resolution, all returned
facts are read from the same `SourceInventory` used to build the resolver.

## Tool Behavior

### get_product_details

1. Require `product_id` to be a string.
2. Strip surrounding whitespace and require 1–128 characters.
3. Resolve the complete identifier with the canonical ExactEntityResolver API.
4. Return `PRODUCT_NOT_FOUND` or `AMBIGUOUS_PRODUCT` without guessing.
5. Read the resolved published product and category from SourceInventory.
6. Return `ProductDetailsResult` with trusted provenance.

This lookup does not depend on semantic retrieval, embeddings, or a vector
artifact.

### get_contact_info

1. Accept no LLM-controlled arguments.
2. Require exactly one usable published ContactSource in the validated
   inventory.
3. Return only company name, duty phone, email, and trusted provenance.
4. Treat absent or ambiguous authoritative contact data as
   `TOOL_UNAVAILABLE`.

### search_products

1. Require `query` to be a string.
2. Strip only surrounding whitespace and require 1–4000 characters.
3. Pass that stripped query directly to the Retriever without rewriting or
   conversation history.
4. Call the Retriever with server-owned settings equivalent to
   `top_k=5`, `allowed_types={"product"}`, and
   `unique_parent_documents=True`.
5. Return at most five unique products in Retriever order.
6. Map each hit to structured source-owned identity/category fields, its raw
   relevant chunk, and `SourceRef`.

The tool guarantees product-only, bounded, deduplicated candidates. It does not
claim that a candidate is proven suitable for a specific industry scenario,
because the source schema deliberately contains no product-to-solution
relationships.

## Retriever and Vector Index Extension

Add optional backend-only filters to the existing retrieval interfaces:

```text
allowed_types: collection of knowledge record types or None
unique_parent_documents: bool = False
```

Type restriction is applied while selecting vector-index candidates, not by
discarding an already truncated mixed-type result. Unique-parent selection
keeps the highest-scoring chunk for each parent document so a product cannot
consume more than one of the five result slots.

Both options default to the current behavior. Calls from `/api/chat-stream`
continue to omit them, preserving general RAG retrieval, entity precedence,
and chunk-level results. The options are never part of the LLM-visible
`search_products` input schema.

## Explicit Tool Registry

Build an immutable mapping from exactly these names to the bound methods on one
`DeterministicTools` instance:

```text
search_products
get_product_details
get_contact_info
```

No registry operation uses `globals()`, `eval()`, `exec()`, dynamic import, or
arbitrary attribute lookup. Day 1 does not add tool-call JSON parsing,
dispatch recursion, retries, or an agent loop.

## Testing Strategy

Use the existing `unittest` framework and test doubles. Tests must not call a
real embedding provider.

### Contract tests

- Require non-empty SourceRef fields and absolute HTTP(S) URLs.
- Reject duplicate source URLs in result models.
- Distinguish all five ToolError codes.
- Build RAG SourceRef values from a valid H1 and canonical source URL.
- Fail safely when a retrieval record lacks a valid H1.

### ExactEntityResolver tests

- Resolve `LY198` and `ly198` to the same canonical product.
- Accept canonical ID and slug variants.
- Reject full display names for canonical detail lookup.
- Perform no fuzzy guessing.
- Represent normalized canonical collisions as ambiguous results.
- Preserve all current query-oriented resolver behavior.

### get_product_details tests

- Return a known product with exact structured fields.
- Reject unknown, blank, non-string, and overlength input.
- Test supported canonical normalization.
- Reject an invalid near-match instead of guessing.
- Include a canonical trusted SourceRef.
- Prove returned facts originate from the injected source fixture rather than a
  tool-layer constant.

### get_contact_info tests

- Return deterministic company name, duty phone, and email.
- Include canonical trusted provenance.
- Expose no address field.
- Fail safely for missing or ambiguous contact data.

### search_products tests

- Pass the stripped query directly to the fake Retriever once.
- Reject blank, non-string, and overlength queries before retrieval.
- Enforce product filtering, unique parents, and a maximum of five products.
- Preserve ranking while removing duplicate products.
- Include provenance and omit score.
- Map a missing Retriever and known retrieval availability failures to
  `TOOL_UNAVAILABLE`.
- Map unexpected failures to `TOOL_EXECUTION_ERROR` without exposing raw
  details.

### Registry and regressions

- Assert that the immutable registry contains exactly the three approved
  names and cannot be modified.
- Verify ordinary Retriever defaults remain unfiltered and chunk-oriented.
- Run existing Retriever, RAG context, chat route, and streaming regressions.
- Run the complete unit-test suite.

## Manual Verification

Provide one internal script with explicit subcommands:

```powershell
python scripts/verify_tools.py product LY198
python scripts/verify_tools.py contact
python scripts/verify_tools.py search "适合酒店使用的对讲机"
```

Product and contact commands run offline and do not require embedding
credentials. Only the explicit search command constructs the existing real
Retriever and sends one query to the configured embedding provider. The
documentation must state the credential requirement and external-request
implication. The script prints validated model JSON and does not expose a new
HTTP route.

## Documentation

Add `docs/tool-contract-v1.md` describing each tool's name, purpose, input,
output, error codes, data source, provenance behavior, and fixed bounds. It
also records these trust rules:

- A future LLM may choose tools; the backend executes only allowlisted tools.
- Exact facts use deterministic structured lookup.
- Semantic discovery reuses the Retriever.
- Provenance titles and URLs are backend-owned.
- Scenario searches provide candidates, not proven fit claims.

For a future scenario-specific product request, orchestration should combine
`search_products` with `get_contact_info` and advise the user to contact a
professional sales representative for final selection. Automatic multi-tool
selection and response wording are explicitly deferred beyond Day 1.

## Deferred Work

- DeepSeek or OpenAI-compatible tool calling
- LLM tool selection and tool-result messages
- Agent loop, recursion, and maximum tool-call controls
- Automatic combination of scenario search and contact information
- Citation rendering or citation streaming
- Query rewriting, reranking, new vector storage, or new embedding models
- Product-to-scenario suitability data or inferred recommendations
- Prompt, HTTP API, streaming protocol, and frontend changes

## Completion Criteria

Day 1 is complete when all three tools and the immutable allowlist exist; exact
lookup uses authoritative structured data and canonical resolution; search
reuses the existing product-filtered Retriever; results expose trusted
SourceRef values; all error categories are distinguishable; manual verification
and contract documentation exist; the full test suite passes; and no deferred
orchestration behavior has been introduced.
