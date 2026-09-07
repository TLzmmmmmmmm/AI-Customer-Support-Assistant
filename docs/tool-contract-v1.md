# Tool Contract V1

## Boundary

Week 3 Day 1 exposes three deterministic Python tools. They are backend-owned,
validated, explicitly allowlisted, and directly callable without an LLM.

```text
search_products
get_product_details
get_contact_info
```

The immutable registry contains only these names. It does not use dynamic
imports, arbitrary attribute lookup, `globals()`, `eval()`, or `exec()`.

An LLM may choose tools in a later phase. The backend remains responsible for
validating arguments, executing only allowlisted capabilities, and owning every
source title and URL. Day 1 does not connect tools to `/api/chat-stream`.

## Shared Provenance

Successful tool results expose one or more sources:

```json
{
  "title": "润信达 LY198",
  "url": "https://www.shengborun.com/two-way-radio/ly198/"
}
```

`title` is human-readable. `url` is an absolute trusted HTTP(S) URL built from
backend data, never from LLM output. Source lists reject duplicate URLs so a
future citation layer can deduplicate them.

Retrieval results use the same SourceRef contract. The source title comes from
the normalized H1 invariant preserved in each knowledge chunk; the URL comes
from the persisted `source_url`. Existing internal `source_url` and
`source_files` fields remain available.

## Shared Errors

Tools raise `ToolError` with `code`, a safe `message`, and `tool_name`.

| Code | Meaning |
| --- | --- |
| `INVALID_ARGUMENT` | Argument type, blank input, or size limit is invalid. |
| `PRODUCT_NOT_FOUND` | No canonical product ID or slug matches. |
| `AMBIGUOUS_PRODUCT` | A canonical ID or slug identifies multiple products. |
| `TOOL_UNAVAILABLE` | Authoritative data or a required retrieval dependency is unavailable. |
| `TOOL_EXECUTION_ERROR` | An unexpected failure prevented safe execution. |

Raw internal exception messages are not part of the tool contract.

## get_product_details

**Purpose:** Return exact structured facts for one product.

**Input:** `product_id: str`, trimmed length 1–128.

The input is resolved as a complete canonical product ID or slug through the
existing ExactEntityResolver normalization. Case, Unicode compatibility,
spaces, and supported separator variants normalize consistently. Full product
display names and fuzzy near-matches are not accepted.

**Output:**

```text
product_id
name
category_id
category_name
key_features[]
product_features
technical_parameters[]
sources[]
```

**Errors:** `INVALID_ARGUMENT`, `PRODUCT_NOT_FOUND`, `AMBIGUOUS_PRODUCT`,
`TOOL_UNAVAILABLE`, `TOOL_EXECUTION_ERROR`.

**Data source:** Published product and category models loaded from
`knowledge/source`. Semantic retrieval is not used.

## get_contact_info

**Purpose:** Return the supported public contact channels.

**Input:** None.

**Output:**

```text
company_name
duty_phone
email
sources[]
```

The address is deliberately excluded and must not be returned by this tool.

**Errors:** `TOOL_UNAVAILABLE`, `TOOL_EXECUTION_ERROR`.

**Data source:** The single published ContactSource under `knowledge/source`.
The tool does not maintain a second copy of any contact fact.

## search_products

**Purpose:** Return product candidates for semantic discovery.

**Input:** `query: str`, trimmed length 1–4000. The trimmed query is passed
directly to the existing Retriever without rewriting or conversation history.

The backend fixes all retrieval controls:

```text
allowed type = product
unique parent documents = true
top_k = 5 final unique products
```

Filtering occurs before scoring-group truncation. The index scores product
chunks, groups them by parent product, retains each product's highest-scoring
chunk, ranks those unique products, and then applies `top_k=5`. Five means five
final unique products when at least five product parents are available, not
five raw chunks before deduplication.

**Output:**

```text
products[]
  product_id
  name
  category_id
  category_name
  relevant_content
  sources[]
```

`relevant_content` is an unmodified retrieved chunk. Raw cosine score remains
internal to RetrievalResult and is not exposed in the tool result.

**Errors:** `INVALID_ARGUMENT`, `TOOL_UNAVAILABLE`, `TOOL_EXECUTION_ERROR`.

**Data source:** The existing Retriever/embedding/vector-index stack plus the
authoritative structured product inventory for exact identity fields.

Search results are candidates, not proof that a product fits a particular
industry scenario. Future orchestration for scenario-specific selection should
also call `get_contact_info` and advise the customer to contact professional
sales staff for final product selection.

## Manual Verification

Exact product and contact checks are offline:

```powershell
python scripts/verify_tools.py product LY198
python scripts/verify_tools.py contact
```

Semantic search is explicit:

```powershell
python scripts/verify_tools.py search "适合酒店使用的对讲机"
```

The search command loads the repository `.env`, requires the existing
DashScope credentials and vector artifact, and makes one embedding request.
The offline commands do not construct a Retriever or call an external service.

## Deferred Beyond Day 1

- LLM tool schemas, selection, parsing, and tool-result messages
- Agent loops, recursion, retries, and maximum tool-call controls
- Automatic combination of scenario search with contact information
- Citation rendering and streaming
- Query rewriting, reranking, new vector stores, and new embeddings
- Product-to-scenario suitability facts or inferred recommendations
- Prompt, public HTTP API, streaming protocol, and frontend changes
