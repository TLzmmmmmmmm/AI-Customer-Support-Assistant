# Knowledge Schema v1

## Purpose and boundary

Knowledge Ingestion V1 answers:

> What knowledge do we have, and how should it be represented?

It converts a curated snapshot of the Shengborun website's authoritative content into validated Normalized Documents. The pipeline stops at knowledge/documents.jsonl.

It does not implement chunking, embeddings, a vector store, retrieval, or LLM logic.

## Source of truth and synchronization responsibility

The Shengborun website remains the authoritative source of public company knowledge.

knowledge/source/ is a curated RAG snapshot of that website. It is intentionally stored in the AI project so the knowledge build is deterministic and does not depend on the Astro repository at runtime.

When authoritative website content is published, changed, unpublished, or removed, the maintainer making or accepting that website change is responsible for updating the corresponding file under knowledge/source/, preserving its provenance, rebuilding knowledge/documents.jsonl, and running the Core tests.

The build does not automatically detect drift between the website and the snapshot.

## Current inventory

| Source type | Source records | Normalized Documents |
|---|---:|---:|
| Product | 49 | 49 |
| Product Category | 4 | 0 |
| Solution | 6 | 6 |
| Support Service | 3 | 3 |
| Company | 1 | 1 |
| Contact | 1 | 1 |
| Total | 64 | 60 |

Product Categories validate and enrich Products but do not become V1 documents.

## Included knowledge

- Product identity, category, features, description, and technical parameters.
- Solution identity, summary, core needs, design, features, and full Markdown body.
- Each Support Service's name, summary, and body.
- The four Company introduction paragraphs.
- Company name, duty phone, and email in Contact.

## Excluded knowledge

- Navigation, breadcrumbs, footer, copyright, and page furniture.
- Homepage promotional or duplicate summaries.
- Images, image paths, alt text, icons, and visual configuration.
- SEO fields and sort order.
- Company street address and page CTA text.
- Product application recommendations.
- Product–Solution or Solution–Product relationships.

## Curated Source JSON

Each file contains one strict JSON object. Unknown fields fail validation. Required strings and arrays cannot be empty. Every record explicitly declares published and non-empty provenance.

Common provenance:

~~~json
{
  "kind": "website_file",
  "reference": "src/content/products/two-way-radio/xir-p8668ex.json"
}
~~~

The reference is relative to the Shengborun repository root. The Core build validates its format but does not require the website repository to exist.

### Source schemas

Product:

~~~text
id
name
slug
category_id
key_features[]
product_features
technical_parameters[].group
technical_parameters[].items[].name
technical_parameters[].items[].value
source_path
published
provenance[]
~~~

Product Category:

~~~text
id
name
slug
short_description
source_path
published
provenance[]
~~~

Solution:

~~~text
id
name
slug
summary
core_needs[]
solution_design
features[]
body_markdown
source_path
published
provenance[]
~~~

Support Service:

~~~text
id
name
summary
body
source_path
published
provenance[]
~~~

Company:

~~~text
id
name
introduction[]
source_path
published
provenance[]
~~~

Contact:

~~~text
id
company_name
duty_phone
email
source_path
published
provenance[]
~~~

Contact deliberately excludes the address and any product-selection instruction.

## Normalized Document Schema v1

Every line in knowledge/documents.jsonl is one UTF-8 JSON object:

~~~json
{
  "schema_version": "1.0",
  "document_id": "product:xir-p8668ex",
  "type": "product",
  "title": "摩托罗拉 XiR P8668Ex",
  "text": "# 摩托罗拉 XiR P8668Ex\n\n...",
  "language": "zh-CN",
  "source_path": "/two-way-radio/xir-p8668ex/",
  "source_url": "https://www.shengborun.com/two-way-radio/xir-p8668ex/",
  "source_files": [
    "src/content/product-categories/two-way-radio.json",
    "src/content/products/two-way-radio/xir-p8668ex.json"
  ],
  "content_hash": "64-character lowercase SHA-256",
  "metadata": {
    "product_id": "xir-p8668ex",
    "slug": "xir-p8668ex",
    "category_id": "two-way-radio",
    "category_name": "对讲机通信"
  }
}
~~~

Allowed document types are product, solution, support, company, and contact.

Document IDs use the pattern type:stable-source-id. They do not contain a URL, timestamp, file path, or content hash.

language is always zh-CN. English model names and technical abbreviations may remain inside Chinese documents.

## Text versus metadata

text is future embedding and LLM-context content. Retrieval-critical identity appears directly in it.

metadata is small, type-specific filtering and debugging information:

| Type | Metadata |
|---|---|
| Product | product_id, slug, category_id, category_name |
| Solution | solution_id, slug |
| Support | service_id |
| Company | company_id |
| Contact | contact_id |

Descriptions, parameters, solution details, and contact values are not hidden only in metadata.

## Source → Document transformation

Product:

~~~text
name + resolved category name
+ product_features
+ key_features
+ grouped technical_parameters
→ one Product document
~~~

The Category name is resolved through category_id and included in Product text and metadata. No use case or Solution link is inferred.

Solution:

~~~text
name + summary + core_needs
+ solution_design + features + body_markdown
→ one Solution document
~~~

Markdown structure is retained. It is not split in V1.

Support:

~~~text
name + summary + body
→ one Support document per service
~~~

Company:

~~~text
name + ordered introduction paragraphs
→ one Company document
~~~

Contact:

~~~text
company_name + duty_phone + email
→ one Contact document
~~~

Normalization only trims outer/trailing whitespace, standardizes line endings to LF, and applies deterministic Markdown templates. It does not rewrite facts, punctuation, numbers, models, or units.

## Content hash

content_hash is SHA-256 over canonical JSON containing:

~~~text
type
title
text
language
metadata
~~~

It excludes schema_version, document_id, source_path, source_url, source_files, timestamps, and absolute filesystem paths.

## Core validation rules

- JSON must parse and match the strict type-specific Source Schema.
- Source IDs and slugs use lowercase kebab-case.
- Entity filename and ID must agree.
- Required values and lists must be non-empty.
- Provenance is a safe src/... repository-relative path.
- Published Products reference an existing published Category.
- Source paths match the approved website route for their type.
- Product and Solution schemas reject relationship fields.
- Every Normalized Document validates against Schema v1.
- document_id is globally unique.
- content_hash is unique across documents and matches recalculation.
- Every required document type produces at least one document.
- Source URL equals the configured base URL plus source_path.
- A failed build does not replace the existing output.

## Deterministic output

Documents are sorted by document_id. JSON uses readable UTF-8 Chinese, compact one-object-per-line serialization, LF line endings, and one final newline. No timestamps are emitted.

Identical source data and base URL produce byte-for-byte identical documents.jsonl.

## Build and test

~~~powershell
python scripts/build_knowledge_documents.py
python -m unittest discover -s tests -v
~~~

Minimal Core CLI overrides:

~~~powershell
python scripts/build_knowledge_documents.py --source-root knowledge/source --output knowledge/documents.jsonl --base-url https://www.shengborun.com
~~~

Online URL checking is deferred engineering hardening and is not part of the V1 Core gate.

## Updating the snapshot

1. Confirm the website change is authoritative and published.
2. Update only the corresponding curated Source JSON.
3. Preserve the original wording and update provenance if the source file changes.
4. Do not add display, SEO, image, recommendation, or Product–Solution fields.
5. Run the build.
6. Review the count report and changed JSONL records.
7. Run all tests.
8. Commit Source JSON and regenerated documents.jsonl together.

