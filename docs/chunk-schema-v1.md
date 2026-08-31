# Chunk Schema v1

## Purpose and pipeline boundary

Day 2 produces the checked-in, normalized input `knowledge/documents.jsonl`. Day 3 reads only that artifact, validates every Document, creates structure-aware Chunks, and writes the checked-in output `knowledge/chunks.jsonl`.

The Day 3 boundary ends at deterministic Chunk production. It does not include experimental chunkers, an evaluation framework, embeddings, a vector database, retrieval, reranking, an LLM, or RAG generation.

## Chunk Schema v1

Every line in `knowledge/chunks.jsonl` is one compact UTF-8 JSON object with LF line endings. Unknown fields are rejected. A real Product Chunk is:

~~~json
{
  "schema_version": "1.0",
  "chunk_id": "product:xir-p8668ex:overview",
  "parent_document_id": "product:xir-p8668ex",
  "parent_document_hash": "6875c042efd4d39175b763fc443eec62e87f3fca8e6651b68b8ac9055dad8379",
  "type": "product",
  "section": "产品介绍",
  "text": "# 摩托罗拉 XiR P8668Ex\n\n产品分类：对讲机通信\n\n## 产品介绍\n\nMOTOTRBO XiR P8668Ex，数字防爆对讲机沿用工业化设计，具备ATEX安全认证，防水防尘等级：IP68，防爆等级：ATEX/IECEx Ex ib IIC T4 Gb；Ex ib IIIC T130°C Db",
  "language": "zh-CN",
  "content_hash": "fb40b4f1f11b2360d64d48c692ac3ba3b832dcb540a05de6b52e94411ec9f96c",
  "source_url": "https://www.shengborun.com/two-way-radio/xir-p8668ex/",
  "source_files": [
    "src/content/product-categories/two-way-radio.json",
    "src/content/products/two-way-radio/xir-p8668ex.json"
  ],
  "metadata": {
    "product_id": "xir-p8668ex",
    "slug": "xir-p8668ex",
    "category_id": "two-way-radio",
    "category_name": "对讲机通信"
  }
}
~~~

| Field | Contract |
|---|---|
| `schema_version` | Always `1.0`. |
| `chunk_id` | Globally unique, colon-separated lowercase ASCII/kebab-case segments. It extends `parent_document_id`. Split parts add deterministic numeric suffixes such as `:1`. |
| `parent_document_id` | Exact Day 2 `document_id`; it begins with the same `type`. |
| `parent_document_hash` | Exact lowercase SHA-256 `content_hash` of the parent Document. |
| `type` | One of `product`, `solution`, `support`, `company`, or `contact`. |
| `section` | Human-readable Chinese source section or entity title. It is not the ID slug. |
| `text` | Exact factual parent text selected for this Chunk, with enough repeated Markdown identity and heading context to stand alone. LF only. |
| `language` | Always `zh-CN`. |
| `content_hash` | Lowercase SHA-256 of this Chunk's canonical semantic payload: exactly `type`, `section`, `text`, `language`, and typed `metadata`, encoded as compact sorted-key UTF-8 JSON. It deliberately excludes IDs, parent hash, provenance, schema version, timestamps, and paths. |
| `source_url` | Exact absolute HTTP(S) URL inherited from the parent. |
| `source_files` | Exact sorted, unique, non-empty provenance list inherited from the parent. |
| `metadata` | Exact type-specific metadata inherited from the parent. |

## Text, identity, provenance, and metadata

`text` is the eventual context-bearing content. Product names, category names, headings, descriptions, features, parameters, solution facts, service facts, company facts, and contact values stay in readable text; factual content is not hidden only in metadata.

Top-level fields provide stable identity and provenance. `chunk_id` identifies the Chunk, while `parent_document_id` and `parent_document_hash` identify the precise Day 2 parent. `content_hash` identifies the canonical semantic Chunk payload independently of IDs and provenance. `type`, `language`, `source_url`, and `source_files` are copied exactly from that parent.

`metadata` is also inherited exactly and remains type-specific:

| Type | Metadata fields |
|---|---|
| Product | `product_id`, `slug`, `category_id`, `category_name` |
| Solution | `solution_id`, `slug` |
| Support | `service_id` |
| Company | `company_id` |
| Contact | `contact_id` |

## Production chunking rules

All Chunks repeat the parent H1 title. When a section must be split, its relevant ancestor headings are also repeated so every part remains understandable. Repeated title and heading context is allowed; factual semantic units are assigned exactly once.

### Product

- `产品介绍` becomes `product:<product-id>:overview` and includes the Product title, category preamble, and introduction.
- `产品特点` becomes `product:<product-id>:features`.
- Every `技术参数` H3 group becomes `product:<product-id>:spec:<registered-slug>`. A group remains separate from every other parameter group.
- A registered optional `应用场景` H2 becomes `product:<product-id>:applications` only when that heading and factual text already exist in the Day 2 Document. The build never infers an application or a Product–Solution association.

### Solution

- The required leading H2 sequence is `方案摘要`, `核心需求`, `方案设计`, `方案特点`; it becomes `:summary`, `:core-needs`, `:design`, and `:features` respectively.
- Exactly one `详细内容` boundary must follow those four sections. Non-empty direct content under that boundary becomes `:details`.
- Each following registered H2 becomes `solution:<solution-id>:body:<registered-slug>`.
- Natural units beneath a body section are packed together up to the soft maximum. An H3 heading is indivisible from its following paragraph or list item, so related child context is not orphaned.

### Support

- The required H2 sequence is `服务摘要`, `服务内容`.
- Both sections form `support:<service-id>:content`, splitting only at natural-unit boundaries if necessary.

### Company

- Each registered H2 becomes `company:<company-id>:<registered-slug>`.
- The first section retains any parent preamble. Current `公司简介` maps to `:company-profile`.
- If a Company Document has no H2, it falls back to `company:<company-id>:overview`, uses the parent title as `section`, and preserves the complete parent text. If natural paragraph or list-item boundaries require secondary splitting, the parts use deterministic `:1`, `:2`, and later suffixes; otherwise it remains one exact whole-document Chunk.

### Contact

- The entire Document becomes `contact:<contact-id>:contact` without splitting. Company name, duty phone, and email remain together.

## Chinese heading to ASCII ID registries

Human-readable `section` values preserve the original Chinese headings. Machine identity never transliterates or guesses: every variable source heading must have an explicit reviewed mapping to a lowercase ASCII slug. An unregistered Product H2 or H3, Solution body H2, or Company H2 is a fatal build error. Duplicate generated IDs are also fatal. This fail-closed policy makes a new heading a schema decision rather than a silent ID change.

Product technical-parameter H3 registry:

| Chinese heading | ASCII slug | Chinese heading | ASCII slug |
|---|---|---|---|
| 安全防护 | `security-protection` | 安全上网 | `secure-internet-access` |
| 安全邮件 | `secure-email` | 负载均衡 | `load-balancing` |
| 高可靠性 | `high-availability` | 技术规格要求 | `technical-requirements` |
| 可选功能 | `optional-features` | 链路聚合 | `link-aggregation` |
| 绿色节能 | `energy-efficiency` | 设备接口 | `device-interfaces` |
| 数据分析 | `data-analysis` | 特色功能 | `distinctive-features` |
| 文件传输 | `file-transfer` | 物理指标 | `physical-specifications` |
| 系统参数 | `system-parameters` | 系统管理 | `system-management` |
| 一般规格 | `general` | POE | `poe` |

Other variable-heading registries:

| Scope | Chinese heading | ASCII slug |
|---|---|---|
| Product optional H2 | 应用场景 | `applications` |
| Solution body H2 | 方案概述 | `overview` |
| Solution body H2 | 应急救援解决方案 | `emergency-rescue-solution` |
| Solution body H2 | 行业背景 | `industry-background` |
| Solution body H2 | 解决方案 | `solution` |
| Solution body H2 | 系统功能 | `system-functions` |
| Solution body H2 | 行业通信现状 | `industry-communications-status` |
| Solution body H2 | 针对大型石油石化企业的数字集群系统解决方案 | `digital-trunking-solution` |
| Solution body H2 | 方案描述 | `solution-description` |
| Solution body H2 | 建设背景 | `construction-background` |
| Solution body H2 | 业务痛点 | `business-pain-points` |
| Solution body H2 | 客户需求 | `customer-requirements` |
| Company H2 | 公司简介 | `company-profile` |

The fixed Product, Solution, Support, and Contact IDs described above are also literal reviewed mappings; they are not derived from Chinese text.

## Size, natural units, and factual fidelity

The production soft maximum is 1000 Unicode characters measured with Python character length, not UTF-8 bytes or model tokens. The splitter packs natural units in source order. A paragraph is a unit and each Markdown list item is a unit. H3–H6 headings establish active ancestry through the next heading of the same or shallower nested level; every secondary-split part repeats the H1, H2, and active nested ancestry for its factual units.

The maximum is intentionally soft. If one indivisible unit plus required title/heading context exceeds 1000 characters, that Chunk remains over the limit rather than rewriting or cutting the fact. Contact is an indivisible whole-document exception. These cases are accepted and reported by statistics, not silently truncated.

There is no sliding-window overlap. Title and ancestor headings may repeat as context, but every factual semantic unit must be covered exactly once. The builder preserves exact source spelling, punctuation, numbers, units, list items, and paragraph text. It does not summarize, paraphrase, infer applications, or create Product–Solution links.

## Validation and deterministic output

Before publication, the build verifies:

- all input JSONL records parse as strict Day 2 Documents and their content hashes recalculate correctly;
- every Document has at least one Chunk and every Chunk references a known parent;
- `parent_document_hash`, type, language, URL, files, and metadata match the parent exactly;
- every `content_hash` recalculates from canonical compact JSON with `ensure_ascii=False`, sorted keys, `(",", ":")` separators, and UTF-8 SHA-256;
- every defined semantic unit is covered, no factual unit is assigned more than once, and exact unit text occurs in its Chunk;
- `chunk_id` values are unique and satisfy the ASCII identity contract;
- Chunk content and order equal a fresh deterministic build from the same Documents;
- JSONL serialization is compact UTF-8, LF-only, in deterministic Document/section/unit order, with one final newline. Record parsing uses LF only, so valid U+2028 and U+2029 characters inside JSON strings remain factual text rather than becoming record boundaries.

Output replacement is atomic. The builder writes a temporary file beside the destination, parses and validates every temporary record, confirms byte-deterministic reserialization, then replaces `knowledge/chunks.jsonl`. A failed build leaves the previous artifact unchanged.

## Statistics

The CLI reports total Documents, total Chunks, average, median, minimum, and maximum `text` length, plus Chunk counts for every Document and type. All lengths are Unicode character counts. They are descriptive production statistics, not token estimates.

## Build and synchronization responsibility

From the repository root:

~~~powershell
python scripts/build_knowledge_chunks.py
~~~

`knowledge/documents.jsonl` and `knowledge/chunks.jsonl` are synchronized artifacts. When authoritative Shengborun website knowledge changes, first refresh the curated Day 2 source snapshot and rebuild and review `knowledge/documents.jsonl` under the Day 2 contract. Only then rebuild and review `knowledge/chunks.jsonl`. Never edit Chunk facts by hand to bypass the parent Document.

The checked-in artifact deliberately stops before embeddings and retrieval.
