# RAG Knowledge Documents V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, strict, offline-first pipeline that converts curated website-knowledge JSON into validated Normalized Documents and stops before chunking.

**Architecture:** A focused `knowledge_pipeline` Python package owns type-specific Pydantic source schemas, pure normalizers, collection validation, deterministic hashing/serialization, atomic output, and optional URL checking. A thin CLI reads only `knowledge/source/**/*.json`; it never parses or depends on the adjacent Astro repository.

**Tech Stack:** Python 3, Pydantic 2.13.4, standard-library `unittest`, `json`, `hashlib`, `tempfile`, `os.replace`, `urllib.request`.

**Spec:** `docs/superpowers/specs/2026-08-30-rag-knowledge-documents-v1-design.md`

## Core-first execution priority

Execute Source Schema → Loading → Normalization → Validation → curated snapshot → 60 Normalized Documents → deterministic JSONL → Core tests as one delivery path. Do not pause Core for teaching checkpoints.

The optional URL checker, elaborate aggregate diagnostics, advanced CLI polish, and excessive edge-case tests are deferred engineering hardening. They are not Core completion gates. Atomic replacement and deterministic output remain required but intentionally simple.

`knowledge/source/` is the curated RAG snapshot of authoritative website content. The website maintainer is responsible for synchronizing this snapshot whenever authoritative public knowledge changes.

## Global Constraints

- The runtime build must not read `D:\Shengborun` or require that repository to exist.
- All source JSON is manually curated and uses strict `snake_case` schemas.
- Normalize by restructuring only; do not rewrite prose, punctuation, technical values, models, or units.
- Product documents contain no application recommendations and no Product–Solution relationships.
- Product Categories validate/enrich Products but do not produce V1 documents.
- Document types are exactly `product`, `solution`, `support`, `company`, and `contact`.
- `language` is exactly `zh-CN` and `schema_version` is exactly `1.0`.
- Default citation origin is `https://www.shengborun.com`.
- Duplicate document IDs and duplicate content hashes are fatal.
- Any failure leaves the previous `knowledge/documents.jsonl` unchanged.
- Serialization is UTF-8, readable Chinese, LF-only, one compact object per line, sorted by `document_id`, with one final newline.
- Do not add timestamps.
- Stop before chunking; add no embedding, vector database, retrieval, or LLM logic.
- Tests use the repository's existing `unittest` style.
- The existing local `.venv` currently points to a missing Python installation. Before execution, use an available Python interpreter and install `requirements.txt`; do not encode a machine-specific interpreter path in project files.

## File Structure

Create the following focused implementation units:

- `knowledge_pipeline/__init__.py` — public package exports.
- `knowledge_pipeline/models.py` — strict source and normalized-document Pydantic models.
- `knowledge_pipeline/diagnostics.py` — structured, sortable build diagnostics and aggregate failure type.
- `knowledge_pipeline/loading.py` — source discovery, JSON parsing, per-type validation, and source inventory.
- `knowledge_pipeline/normalizers.py` — pure per-type Source → Document transformations and canonical hash.
- `knowledge_pipeline/validation.py` — cross-source and collection-wide invariants.
- `knowledge_pipeline/output.py` — deterministic JSONL encoding, temporary-file verification, atomic replacement.
- `knowledge_pipeline/url_check.py` — optional online page checks with fragment removal.
- `knowledge_pipeline/pipeline.py` — orchestration and build report.
- `scripts/build_knowledge_documents.py` — thin command-line entry point.
- `tests/knowledge_pipeline/` — unit and integration tests, isolated from chat API tests.
- `knowledge/source/` — curated records described by the spec.
- `docs/knowledge-schema-v1.md` — human-facing schema and operating guide.

No production ingestion logic belongs in `main.py`, `routes/`, or `services/` because this is an offline build subsystem.

---

### Task 1: Strict Domain Models

**Files:**
- Create: `knowledge_pipeline/__init__.py`
- Create: `knowledge_pipeline/models.py`
- Create: `tests/knowledge_pipeline/__init__.py`
- Create: `tests/knowledge_pipeline/test_models.py`

**Interfaces:**
- Produces: `ProductSource`, `ProductCategorySource`, `SolutionSource`, `SupportSource`, `CompanySource`, `ContactSource`.
- Produces: `ProductDocument`, `SolutionDocument`, `SupportDocument`, `CompanyDocument`, `ContactDocument` and `DocumentAdapter`.
- Produces: `SourceInventory`, a dataclass containing dictionaries keyed by source ID.
- Consumes: only Pydantic and the standard library.

- [ ] **Step 1: Write failing strict-schema tests**

Create tests that prove trimming, required non-empty lists, unknown-field rejection, provenance path validation, ID/slug rules, and exact metadata/document discrimination.

```python
import unittest

from pydantic import ValidationError

from knowledge_pipeline.models import (
    ContactSource,
    ProductMetadata,
    ProductSource,
    ProductDocument,
)


class SourceModelTests(unittest.TestCase):
    def test_product_source_trims_outer_whitespace_and_rejects_unknown_fields(self):
        payload = {
            "id": "xir-p8668ex",
            "name": "  摩托罗拉 XiR P8668Ex  ",
            "slug": "xir-p8668ex",
            "category_id": "two-way-radio",
            "key_features": ["防爆机型"],
            "product_features": "产品介绍",
            "technical_parameters": [{
                "group": "一般规格",
                "items": [{"name": "输出功率", "value": "1W"}],
            }],
            "source_path": "/two-way-radio/xir-p8668ex/",
            "published": True,
            "provenance": [{
                "kind": "website_file",
                "reference": "src/content/products/two-way-radio/xir-p8668ex.json",
            }],
        }

        product = ProductSource.model_validate(payload)
        self.assertEqual(product.name, "摩托罗拉 XiR P8668Ex")

        with self.assertRaises(ValidationError):
            ProductSource.model_validate({**payload, "solution_ids": ["petrochemical"]})

    def test_product_source_rejects_empty_required_list(self):
        payload = valid_product_payload()
        payload["key_features"] = []
        with self.assertRaises(ValidationError):
            ProductSource.model_validate(payload)

    def test_provenance_rejects_absolute_or_parent_paths(self):
        for reference in ("D:/Shengborun/src/file.json", "../src/file.json", "/src/file.json"):
            payload = valid_product_payload()
            payload["provenance"][0]["reference"] = reference
            with self.subTest(reference=reference):
                with self.assertRaises(ValidationError):
                    ProductSource.model_validate(payload)

    def test_contact_schema_has_no_address_or_selection_notice(self):
        payload = valid_contact_payload()
        for forbidden in ("address", "product_selection_notice"):
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(ValidationError):
                    ContactSource.model_validate({**payload, forbidden: "not allowed"})

    def test_product_document_requires_product_metadata(self):
        document = valid_product_document_payload()
        document["metadata"] = {"solution_id": "petrochemical", "slug": "petrochemical"}
        with self.assertRaises(ValidationError):
            ProductDocument.model_validate(document)
```

Include helpers in the same test module returning complete valid payloads, so each negative test changes exactly one property.

- [ ] **Step 2: Run the model tests and verify failure**

Run:

```powershell
python -m unittest tests.knowledge_pipeline.test_models -v
```

Expected: import failure because `knowledge_pipeline.models` does not exist.

- [ ] **Step 3: Implement strict source primitives and models**

Use a shared strict base and explicit validators.

```python
from dataclasses import dataclass, field
from typing import Annotated, Literal
import re

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator


KEBAB_CASE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class WebsiteFileProvenance(StrictModel):
    kind: Literal["website_file"]
    reference: str

    @field_validator("reference")
    @classmethod
    def validate_reference(cls, value: str) -> str:
        if (
            not value
            or value.startswith(("/", "\\"))
            or "\\" in value
            or ":" in value
            or ".." in value.split("/")
            or not value.startswith("src/")
        ):
            raise ValueError("must be a Shengborun repository-relative src/ path")
        return value


class TechnicalParameterItem(StrictModel):
    name: str = Field(min_length=1)
    value: str = Field(min_length=1)


class TechnicalParameterGroup(StrictModel):
    group: str = Field(min_length=1)
    items: list[TechnicalParameterItem] = Field(min_length=1)


class SourceBase(StrictModel):
    id: str
    source_path: str
    published: bool
    provenance: list[WebsiteFileProvenance] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not KEBAB_CASE.fullmatch(value):
            raise ValueError("must be lowercase kebab-case")
        return value


class SluggedSource(SourceBase):
    slug: str

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, value: str) -> str:
        if not KEBAB_CASE.fullmatch(value):
            raise ValueError("must be lowercase kebab-case")
        return value


class ProductSource(SluggedSource):
    name: str = Field(min_length=1)
    category_id: str
    key_features: list[str] = Field(min_length=1)
    product_features: str = Field(min_length=1)
    technical_parameters: list[TechnicalParameterGroup] = Field(min_length=1)


class ProductCategorySource(SluggedSource):
    name: str = Field(min_length=1)
    short_description: str = Field(min_length=1)


class SolutionSource(SluggedSource):
    name: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    core_needs: list[str] = Field(min_length=1)
    solution_design: str = Field(min_length=1)
    features: list[str] = Field(min_length=1)
    body_markdown: str = Field(min_length=1)


class SupportSource(SourceBase):
    name: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    body: str = Field(min_length=1)


class CompanySource(SourceBase):
    name: str = Field(min_length=1)
    introduction: list[str] = Field(min_length=1)


class ContactSource(SourceBase):
    company_name: str = Field(min_length=1)
    duty_phone: str = Field(min_length=1)
    email: str = Field(min_length=1)
```

Add collection-item validators so every string in lists is non-empty after trimming. Validate `source_path` centrally: it begins with `/`, contains no scheme, query, backslash, or `..` segment, and its fragment—when present—is non-empty.

Define strict metadata and document subclasses:

```python
class ProductMetadata(StrictModel):
    product_id: str
    slug: str
    category_id: str
    category_name: str


class SolutionMetadata(StrictModel):
    solution_id: str
    slug: str


class SupportMetadata(StrictModel):
    service_id: str


class CompanyMetadata(StrictModel):
    company_id: str


class ContactMetadata(StrictModel):
    contact_id: str


class DocumentBase(StrictModel):
    schema_version: Literal["1.0"]
    document_id: str
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    language: Literal["zh-CN"]
    source_path: str
    source_url: str
    source_files: list[str] = Field(min_length=1)
    content_hash: str


class ProductDocument(DocumentBase):
    type: Literal["product"]
    metadata: ProductMetadata


class SolutionDocument(DocumentBase):
    type: Literal["solution"]
    metadata: SolutionMetadata


class SupportDocument(DocumentBase):
    type: Literal["support"]
    metadata: SupportMetadata


class CompanyDocument(DocumentBase):
    type: Literal["company"]
    metadata: CompanyMetadata


class ContactDocument(DocumentBase):
    type: Literal["contact"]
    metadata: ContactMetadata


KnowledgeDocument = Annotated[
    ProductDocument | SolutionDocument | SupportDocument | CompanyDocument | ContactDocument,
    Field(discriminator="type"),
]
DocumentAdapter = TypeAdapter(KnowledgeDocument)
```

Add field validators for document ID shape, lowercase SHA-256, absolute HTTP(S) source URL, sorted unique relative `source_files`, and LF-only text.

Define `SourceInventory`:

```python
@dataclass(frozen=True)
class SourceInventory:
    products: dict[str, ProductSource] = field(default_factory=dict)
    product_categories: dict[str, ProductCategorySource] = field(default_factory=dict)
    solutions: dict[str, SolutionSource] = field(default_factory=dict)
    support: dict[str, SupportSource] = field(default_factory=dict)
    companies: dict[str, CompanySource] = field(default_factory=dict)
    contacts: dict[str, ContactSource] = field(default_factory=dict)
```

- [ ] **Step 4: Run model tests**

Run:

```powershell
python -m unittest tests.knowledge_pipeline.test_models -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit domain models**

```powershell
git add knowledge_pipeline/__init__.py knowledge_pipeline/models.py tests/knowledge_pipeline/__init__.py tests/knowledge_pipeline/test_models.py
git commit -m "feat: define strict knowledge schemas"
```

---

### Task 2: Source Discovery, Parsing, and Aggregate Diagnostics

**Files:**
- Create: `knowledge_pipeline/diagnostics.py`
- Create: `knowledge_pipeline/loading.py`
- Create: `tests/knowledge_pipeline/test_loading.py`

**Interfaces:**
- Consumes: source model classes and `SourceInventory` from Task 1.
- Produces: `Diagnostic(file: str, entity: str | None, field: str | None, reason: str)`.
- Produces: `BuildFailure(diagnostics: list[Diagnostic])`.
- Produces: `load_source_inventory(source_root: Path) -> SourceInventory`.
- Later tasks consume a fully typed inventory or catch one aggregate `BuildFailure`.

- [ ] **Step 1: Write failing loader and diagnostic tests**

Cover malformed JSON with line/column, unknown fields with a precise Pydantic path, filename/ID mismatch, duplicate IDs, all errors aggregated across files, and deterministic diagnostic ordering.

```python
class SourceLoadingTests(unittest.TestCase):
    def test_loader_aggregates_json_schema_and_filename_errors(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_minimal_source_tree(root)
            (root / "products" / "broken.json").write_text('{"id":', encoding="utf-8")
            write_json(
                root / "support" / "wrong-file-name.json",
                {**valid_support_payload(), "id": "solution-design", "unexpected": True},
            )

            with self.assertRaises(BuildFailure) as context:
                load_source_inventory(root)

            rendered = "\n".join(item.render() for item in context.exception.diagnostics)
            self.assertIn("products/broken.json", rendered)
            self.assertIn("line 1", rendered)
            self.assertIn("support/wrong-file-name.json", rendered)
            self.assertIn("unexpected", rendered)
            self.assertIn("filename must equal entity id", rendered)
```

Use test helpers to create the six required source locations, including single-file Company and Contact records.

- [ ] **Step 2: Run the loader test and verify failure**

Run:

```powershell
python -m unittest tests.knowledge_pipeline.test_loading -v
```

Expected: import failure for `knowledge_pipeline.loading`.

- [ ] **Step 3: Implement structured diagnostics**

```python
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Diagnostic:
    file: str
    entity: str = ""
    field: str = ""
    reason: str = ""

    def render(self) -> str:
        lines = [f"[ERROR] {self.file}"]
        if self.entity:
            lines.append(f"entity: {self.entity}")
        if self.field:
            lines.append(f"field: {self.field}")
        lines.append(f"reason: {self.reason}")
        return "\n".join(lines)


class BuildFailure(Exception):
    def __init__(self, diagnostics: list[Diagnostic]):
        self.diagnostics = sorted(diagnostics)
        super().__init__(f"knowledge build failed with {len(self.diagnostics)} error(s)")
```

- [ ] **Step 4: Implement source discovery and validation**

Map locations to models explicitly; do not infer arbitrary directories:

```python
SOURCE_LAYOUT = {
    "products": ProductSource,
    "product-categories": ProductCategorySource,
    "solutions": SolutionSource,
    "support": SupportSource,
}
SINGLE_FILES = {
    "company.json": CompanySource,
    "contact.json": ContactSource,
}
```

For each JSON file:

1. Read UTF-8.
2. Decode with `json.loads`.
3. Report `JSONDecodeError.lineno` and `colno`.
4. Reject a non-object top level.
5. Validate with the exact model.
6. Convert each Pydantic error location to dot/index syntax such as `technical_parameters[0].items[2].value`.
7. Verify directory filename stem equals validated ID.
8. Detect duplicate IDs even if separate paths somehow supply them.
9. Continue scanning independent files and raise one sorted `BuildFailure` at the end.

Ignore no unexpected files: a JSON file in an unknown nested source location is an error rather than silently skipped.

- [ ] **Step 5: Run loading tests and all existing tests**

Run:

```powershell
python -m unittest tests.knowledge_pipeline.test_loading -v
python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit loading and diagnostics**

```powershell
git add knowledge_pipeline/diagnostics.py knowledge_pipeline/loading.py tests/knowledge_pipeline/test_loading.py
git commit -m "feat: load and validate knowledge sources"
```

---

### Task 3: Pure Normalizers and Canonical Content Hashes

**Files:**
- Create: `knowledge_pipeline/normalizers.py`
- Create: `tests/knowledge_pipeline/test_normalizers.py`

**Interfaces:**
- Consumes: typed source models from Task 1.
- Produces:
  - `normalize_product(product, category, base_url) -> ProductDocument`
  - `normalize_solution(solution, base_url) -> SolutionDocument`
  - `normalize_support(service, base_url) -> SupportDocument`
  - `normalize_company(company, base_url) -> CompanyDocument`
  - `normalize_contact(contact, base_url) -> ContactDocument`
  - `compute_content_hash(type_, title, text, language, metadata) -> str`
- Normalizers are pure: no filesystem, network, clock, or environment access.

- [ ] **Step 1: Write exact-output failing tests**

For each type, assert the full text template, document ID, metadata, URL, provenance, and hash. The Product test must use the original parameter value unchanged.

```python
class ProductNormalizerTests(unittest.TestCase):
    def test_product_text_preserves_identity_category_and_parameter_value(self):
        product = ProductSource.model_validate(valid_product_payload())
        category = ProductCategorySource.model_validate(valid_category_payload())

        document = normalize_product(product, category, "https://www.shengborun.com")

        self.assertEqual(document.document_id, "product:xir-p8668ex")
        self.assertEqual(document.type, "product")
        self.assertEqual(document.title, "摩托罗拉 XiR P8668Ex")
        self.assertEqual(
            document.text,
            "# 摩托罗拉 XiR P8668Ex\n\n"
            "产品分类：对讲机通信\n\n"
            "## 产品介绍\n\n产品介绍\n\n"
            "## 产品特点\n\n- 防爆机型\n\n"
            "## 技术参数\n\n"
            "### 一般规格\n\n- 输出功率：1W",
        )
        self.assertNotIn("solution", document.model_dump_json())
        self.assertEqual(
            document.source_files,
            [
                "src/content/product-categories/two-way-radio.json",
                "src/content/products/two-way-radio/xir-p8668ex.json",
            ],
        )
```

Add:

- Solution test retaining nested Markdown body without splitting.
- Support test with exactly two sections.
- Company test preserving paragraph order.
- Contact test proving address and selection guidance are absent.
- Hash test proving source URL and source files do not affect the hash.
- Hash test proving metadata or text changes do affect it.
- Base URL test that removes a trailing slash before joining.
- CRLF and trailing-space normalization test.

- [ ] **Step 2: Run normalizer tests and verify failure**

Run:

```powershell
python -m unittest tests.knowledge_pipeline.test_normalizers -v
```

Expected: import failure for `knowledge_pipeline.normalizers`.

- [ ] **Step 3: Implement canonical text helpers and hash**

```python
def normalize_lines(value: str) -> str:
    lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def canonical_metadata(metadata: BaseModel) -> dict[str, object]:
    return metadata.model_dump(mode="json")


def compute_content_hash(
    type_: str,
    title: str,
    text: str,
    language: str,
    metadata: BaseModel,
) -> str:
    payload = {
        "type": type_,
        "title": title,
        "text": text,
        "language": language,
        "metadata": canonical_metadata(metadata),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

Implement `join_source_url` with a validated base URL and preserve a `source_path` fragment in the resulting citation URL.

- [ ] **Step 4: Implement all five normalizers**

Build exact Markdown templates from the spec. Create metadata first, compute the hash from the five approved semantic fields, then validate the finished payload through the concrete Document model.

Never derive product use cases. Never scan Solution content from Product normalization.

- [ ] **Step 5: Run normalizer and full tests**

Run:

```powershell
python -m unittest tests.knowledge_pipeline.test_normalizers -v
python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit normalizers**

```powershell
git add knowledge_pipeline/normalizers.py tests/knowledge_pipeline/test_normalizers.py
git commit -m "feat: normalize curated knowledge documents"
```

---

### Task 4: Cross-Source and Document-Collection Validation

**Files:**
- Create: `knowledge_pipeline/validation.py`
- Create: `tests/knowledge_pipeline/test_validation.py`

**Interfaces:**
- Consumes: `SourceInventory` and `KnowledgeDocument` objects.
- Produces:
  - `validate_source_relationships(inventory: SourceInventory) -> None`
  - `validate_document_collection(documents: list[KnowledgeDocument], base_url: str) -> None`
- Raises: aggregate `BuildFailure` with all safely discoverable errors.

- [ ] **Step 1: Write failing relationship tests**

```python
class RelationshipValidationTests(unittest.TestCase):
    def test_published_product_requires_published_existing_category(self):
        inventory = inventory_with_one_product_and_category()
        inventory.product_categories["two-way-radio"].published = False

        with self.assertRaises(BuildFailure) as context:
            validate_source_relationships(inventory)

        rendered = "\n".join(item.render() for item in context.exception.diagnostics)
        self.assertIn("product:xir-p8668ex", rendered)
        self.assertIn("category_id", rendered)
        self.assertIn("published category", rendered)

    def test_product_path_must_match_category_and_product_slugs(self):
        inventory = inventory_with_one_product_and_category()
        inventory.products["xir-p8668ex"].source_path = "/wrong/xir-p8668ex/"
        with self.assertRaises(BuildFailure):
            validate_source_relationships(inventory)
```

Because Pydantic models are frozen only if configured that way, create modified copies with `model_copy(update=...)` rather than mutating if needed.

Add exact route tests:

- Product: `/{category.slug}/{product.slug}/`.
- Solution: `/solutions/{solution.slug}/`.
- Support: `/support/#{service.id}`.
- Company: `/about/#company`.
- Contact: `/about/#contact`.

- [ ] **Step 2: Write failing document-collection tests**

Test:

- Duplicate `document_id`.
- Duplicate `content_hash` across different IDs.
- Missing one required type.
- Tampered hash.
- Wrong ID/type relationship.
- Wrong generated URL.
- Successful collection with all five types.
- A `source_url` whose origin differs from the supplied `base_url`.

- [ ] **Step 3: Run validation tests and verify failure**

Run:

```powershell
python -m unittest tests.knowledge_pipeline.test_validation -v
```

Expected: import failure for `knowledge_pipeline.validation`.

- [ ] **Step 4: Implement source relationship validation**

Aggregate errors rather than stopping at the first Product. Validate unpublished records structurally but apply route/reference inclusion checks to published records that can produce or enrich documents.

Do not validate provenance file existence because the website repository is intentionally absent from runtime.

- [ ] **Step 5: Implement document collection validation**

Reparse each document through `DocumentAdapter`, recalculate hashes, verify every `source_url` equals `join_source_url(base_url, source_path)`, collect IDs and hashes, and require at least one of every type. Pass `base_url` explicitly; do not read an environment variable or module-global value inside validation.

Expose a helper returning deterministic counts:

```python
def count_documents_by_type(documents: list[KnowledgeDocument]) -> dict[str, int]:
    return {
        type_: sum(document.type == type_ for document in documents)
        for type_ in ("product", "solution", "support", "company", "contact")
    }
```

- [ ] **Step 6: Run validation and full tests**

Run:

```powershell
python -m unittest tests.knowledge_pipeline.test_validation -v
python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit relationship validation**

```powershell
git add knowledge_pipeline/validation.py tests/knowledge_pipeline/test_validation.py
git commit -m "feat: validate knowledge relationships"
```

---

### Task 5: Deterministic JSONL and Atomic Replacement

**Files:**
- Create: `knowledge_pipeline/output.py`
- Create: `tests/knowledge_pipeline/test_output.py`

**Interfaces:**
- Consumes: validated `list[KnowledgeDocument]`.
- Produces:
  - `serialize_jsonl(documents, base_url: str) -> bytes`
  - `verify_jsonl_bytes(payload: bytes, base_url: str) -> list[KnowledgeDocument]`
  - `write_jsonl_atomically(output_path: Path, documents, base_url: str) -> None`
- Calls `validate_document_collection(documents, base_url)` before and after serialization.

- [ ] **Step 1: Write failing deterministic-output tests**

```python
class JsonlOutputTests(unittest.TestCase):
    def test_serialization_is_sorted_utf8_lf_and_has_one_final_newline(self):
        documents = valid_documents_in_reverse_order()
        payload = serialize_jsonl(documents, "https://www.shengborun.com")

        self.assertNotIn(b"\\r", payload)
        self.assertTrue(payload.endswith(b"\\n"))
        self.assertFalse(payload.endswith(b"\\n\\n"))
        text = payload.decode("utf-8")
        self.assertIn("摩托罗拉", text)
        ids = [json.loads(line)["document_id"] for line in text.splitlines()]
        self.assertEqual(ids, sorted(ids))

    def test_failed_write_preserves_existing_output(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "documents.jsonl"
            output.write_bytes(b"previous\\n")
            with self.assertRaises(BuildFailure):
                write_jsonl_atomically(
                    output,
                    documents_with_duplicate_hash(),
                    "https://www.shengborun.com",
                )
            self.assertEqual(output.read_bytes(), b"previous\\n")
```

Add a test that monkey-patches temporary verification to fail after writing and proves the previous output remains unchanged and no temp file is left behind.

- [ ] **Step 2: Run output tests and verify failure**

Run:

```powershell
python -m unittest tests.knowledge_pipeline.test_output -v
```

Expected: import failure for `knowledge_pipeline.output`.

- [ ] **Step 3: Implement stable serialization**

Dump models with fixed model field order, `ensure_ascii=False`, and compact separators. Sort by `document_id` and append exactly one LF.

```python
def serialize_jsonl(documents: list[KnowledgeDocument], base_url: str) -> bytes:
    validate_document_collection(documents, base_url)
    lines = [
        json.dumps(
            document.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for document in sorted(documents, key=lambda item: item.document_id)
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")
```

- [ ] **Step 4: Implement temporary-file verification and atomic replace**

Create the temporary file in `output_path.parent`, flush and `os.fsync` it, re-read bytes, validate every non-empty line through `DocumentAdapter`, run `validate_document_collection(parsed_documents, base_url)`, compare the reserialized bytes, then call `os.replace`.

Use `try/finally` to remove an uncommitted temp file. Do not unlink or truncate the existing output before `os.replace` succeeds.

- [ ] **Step 5: Run output and full tests**

Run:

```powershell
python -m unittest tests.knowledge_pipeline.test_output -v
python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit deterministic output**

```powershell
git add knowledge_pipeline/output.py tests/knowledge_pipeline/test_output.py
git commit -m "feat: write knowledge JSONL atomically"
```

---

### Task 6: Pipeline Orchestration, CLI, and Optional URL Checks

**Files:**
- Create: `knowledge_pipeline/pipeline.py`
- Create: `knowledge_pipeline/url_check.py`
- Create: `scripts/build_knowledge_documents.py`
- Create: `tests/knowledge_pipeline/test_pipeline.py`
- Create: `tests/knowledge_pipeline/test_url_check.py`
- Create: `tests/knowledge_pipeline/test_cli.py`

**Interfaces:**
- Produces:
  - `build_documents(source_root: Path, base_url: str) -> list[KnowledgeDocument]`
  - `run_build(source_root: Path, output_path: Path, base_url: str, check_urls: bool) -> BuildReport`
  - `check_source_urls(documents, opener=urlopen) -> None`
  - `BuildReport(output_path: Path, counts: dict[str, int])`
  - CLI options `--source-root`, `--output`, `--base-url`, and `--check-urls`.

- [ ] **Step 1: Write failing pipeline integration tests with temporary sources**

Test that the pipeline:

- Loads one published record of every document type plus one Category.
- Excludes a structurally valid unpublished Product.
- Produces six documents when the fixture has two published Products and one of each other type.
- Performs URL checking before output replacement when requested.
- Preserves prior output on any source, normalization, collection, or URL error.
- Returns deterministic type counts.

- [ ] **Step 2: Write failing URL-check tests**

Inject a fake opener. Assert unique page URLs are requested once, fragments are removed, redirects/final success are accepted, HTTP 404 becomes “page unavailable,” and `URLError` becomes “network failure.”

```python
def test_url_check_removes_fragments_and_deduplicates_pages(self):
    requested = []

    def fake_opener(request, timeout):
        requested.append(request.full_url)
        return FakeResponse(status=200)

    check_source_urls(company_and_contact_documents(), opener=fake_opener)

    self.assertEqual(requested, ["https://www.shengborun.com/about/"])
```

Use GET requests with a short explicit timeout and a descriptive User-Agent. Do not use HEAD because some deployments reject HEAD despite serving GET.

- [ ] **Step 3: Write failing CLI tests**

Call `main([...])` directly rather than spawning a subprocess. Assert:

- Defaults point to `knowledge/source` and `knowledge/documents.jsonl` relative to repository root.
- Default base URL is `https://www.shengborun.com`.
- Success prints total and counts by type.
- Failure prints all diagnostics to stderr and returns non-zero.
- `--check-urls` is opt-in.

- [ ] **Step 4: Run pipeline/URL/CLI tests and verify failure**

Run:

```powershell
python -m unittest \
  tests.knowledge_pipeline.test_pipeline \
  tests.knowledge_pipeline.test_url_check \
  tests.knowledge_pipeline.test_cli -v
```

Expected: imports fail for the not-yet-created modules.

- [ ] **Step 5: Implement pipeline orchestration**

```python
@dataclass(frozen=True)
class BuildReport:
    output_path: Path
    counts: dict[str, int]


def build_documents(source_root: Path, base_url: str) -> list[KnowledgeDocument]:
    inventory = load_source_inventory(source_root)
    validate_source_relationships(inventory)
    documents = normalize_published_inventory(inventory, base_url)
    validate_document_collection(documents, base_url)
    return documents


def run_build(
    source_root: Path,
    output_path: Path,
    base_url: str,
    check_urls: bool = False,
) -> BuildReport:
    documents = build_documents(source_root, base_url)
    if check_urls:
        check_source_urls(documents)
    write_jsonl_atomically(output_path, documents, base_url)
    return BuildReport(output_path, count_documents_by_type(documents))
```

`normalize_published_inventory` iterates IDs deterministically, skips `published: false`, passes each Product its Category, and creates no Category Document.

- [ ] **Step 6: Implement optional URL checking**

Strip fragments with `urllib.parse.urldefrag`, sort and deduplicate page URLs, issue GET requests, follow standard redirects, and aggregate diagnostics. Treat final 2xx as success. Classify HTTP errors separately from `URLError` and timeout/transport errors.

- [ ] **Step 7: Implement the thin CLI**

The script adds the repository root to `sys.path` only if required by direct script execution, parses options, calls `run_build`, prints the report, renders aggregate diagnostics, and returns an integer exit code. Keep logic out of the CLI.

- [ ] **Step 8: Run pipeline tests and full regression**

Run:

```powershell
python -m unittest \
  tests.knowledge_pipeline.test_pipeline \
  tests.knowledge_pipeline.test_url_check \
  tests.knowledge_pipeline.test_cli -v
python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit pipeline and CLI**

```powershell
git add knowledge_pipeline/pipeline.py knowledge_pipeline/url_check.py scripts/build_knowledge_documents.py tests/knowledge_pipeline/test_pipeline.py tests/knowledge_pipeline/test_url_check.py tests/knowledge_pipeline/test_cli.py
git commit -m "feat: orchestrate knowledge document builds"
```

---

### Task 7: Curate Product Categories and Products

**Files:**
- Create: `knowledge/source/product-categories/*.json` (4 files)
- Create: `knowledge/source/products/*.json` (49 files)
- Create: `tests/knowledge_pipeline/test_curated_sources.py`

**Interfaces:**
- Consumes: the exact schemas and loader from Tasks 1–2.
- Produces: validated Category lookup records and Product records for the real inventory.
- Source references: `D:\Shengborun\src\content\product-categories` and `D:\Shengborun\src\content\products`, read-only during curation.

- [ ] **Step 1: Write a failing real-inventory test**

```python
class CuratedSourceInventoryTests(unittest.TestCase):
    def test_curated_products_and_categories_load(self):
        root = repository_root() / "knowledge" / "source"
        inventory = load_source_inventory(root)

        self.assertEqual(len(inventory.products), 49)
        self.assertEqual(len(inventory.product_categories), 4)
        self.assertTrue(all(item.published for item in inventory.products.values()))
        self.assertTrue(all(item.published for item in inventory.product_categories.values()))
```

This test records the approved initial extraction inventory; runtime validation still does not permanently require 49 or 4.

- [ ] **Step 2: Run the inventory test and verify failure**

Run:

```powershell
python -m unittest tests.knowledge_pipeline.test_curated_sources -v
```

Expected: failure because curated files do not exist.

- [ ] **Step 3: Create four Category Source JSON files**

Create IDs:

    ict-integration
    mesh-network
    shortwave-radio
    two-way-radio

For each, copy `id`, `name`, `slug`, `shortDescription → short_description`, and `published` exactly. Add:

- `source_path` as `/{slug}/`.
- provenance reference to `src/content/product-categories/<id>.json`.

Do not copy sort order or UI data.

- [ ] **Step 4: Create 49 Product Source JSON files**

Create one file per existing Product ID:

    ap
    ap-controller
    band-pass-duplexer
    cavity-combiner
    domestic-5w
    domestic-20w
    domestic-125w
    domestic-500w
    e-center
    e-mesh580p
    e-pack200
    envoy-x
    firewall
    gateway
    hp500
    hp500cqst
    hp710ex
    hp780
    hp780cqst
    hp790ex
    hr1060
    industrial-switch
    jm-ua4wl
    jomesh-bp2w
    jomesh-hs
    jomesh-od10w
    ly198
    ly598
    ly598-keyboard
    network-isolation-gateway
    optic-repeater-near
    optic-repeater-remote
    pne380
    receiver-splitter
    router
    sentry-h-6110-mp
    sentry-h-6120
    server
    slr1000
    slr5300
    switch
    x5s
    x6s
    x7
    x15
    xir-gp328d-plus-ex
    xir-gp338d-plus-ex
    xir-p6600i-ex
    xir-p8668ex

For every Product:

- Copy `id`, `name`, `slug`, `categoryId → category_id`, `keyFeatures → key_features`, `productFeatures → product_features`, `technicalParameters → technical_parameters`, and `published` exactly.
- Rename nested keys `items` unchanged; keep `group`, `name`, and `value` exact.
- Set `source_path` to `/{category-slug}/{product-slug}/`.
- Add provenance for the exact original Product file under `src/content/products/<category-id>/<id>.json`.
- Do not copy image, gallery, sort order, SEO, or inferred fields.

Use apply_patch for curated files. Do not create a permanent Astro extraction script.

- [ ] **Step 5: Run real source loading and relationship tests**

Run:

```powershell
python -m unittest tests.knowledge_pipeline.test_curated_sources -v
python scripts/build_knowledge_documents.py --source-root knowledge/source --output knowledge/documents.jsonl
```

Expected at this intermediate point: the inventory test passes its Product/Category assertions, while a full build fails clearly because Solution, Support, Company, and Contact are not populated yet. Verify the error names the missing required types; do not weaken validation to make the partial build pass.

- [ ] **Step 6: Commit curated Product data**

```powershell
git add knowledge/source/product-categories knowledge/source/products tests/knowledge_pipeline/test_curated_sources.py
git commit -m "data: curate product knowledge sources"
```

---

### Task 8: Curate Solutions, Support, Company, and Contact

**Files:**
- Create: `knowledge/source/solutions/*.json` (6 files)
- Create: `knowledge/source/support/*.json` (3 files)
- Create: `knowledge/source/company.json`
- Create: `knowledge/source/contact.json`
- Modify: `tests/knowledge_pipeline/test_curated_sources.py`

**Interfaces:**
- Consumes: exact Source Schemas and real website files as read-only references.
- Produces: the remaining real inventory needed for a complete build.

- [ ] **Step 1: Extend the failing inventory test**

Assert:

```python
self.assertEqual(len(inventory.solutions), 6)
self.assertEqual(len(inventory.support), 3)
self.assertEqual(len(inventory.companies), 1)
self.assertEqual(len(inventory.contacts), 1)
self.assertEqual(set(inventory.companies), {"shengborun"})
self.assertEqual(set(inventory.contacts), {"shengborun"})
```

Run the test and confirm it fails on the missing records.

- [ ] **Step 2: Create six Solution JSON files**

Create IDs:

    civil-defense
    emergency-mesh
    enterprise
    hotel
    petrochemical
    smart-emergency

For each Markdown source:

- Add explicit `id` equal to the current slug.
- Copy `name`, `slug`, `summary`, `coreNeeds → core_needs`, `solutionDesign → solution_design`, `features`, and `published` exactly.
- Copy all Markdown after the closing frontmatter delimiter into `body_markdown`, retaining headings, paragraphs, ordered lists, and unordered lists.
- Set `source_path` to `/solutions/{slug}/`.
- Add provenance `src/content/solutions/{id}.md`.
- Exclude every image, image alt, sort, and SEO field.

- [ ] **Step 3: Create three Support Service JSON files**

Create IDs:

    solution-design
    project-implementation
    delivery-training

Copy `id`, `name`, `summary`, and `body` exactly from `src/data/support-services.ts`. Use each existing `href` as `source_path`. Set `published` to true and provenance to `src/data/support-services.ts`.

Do not copy icons, page Hero, repeated visual copy, or CTA.

- [ ] **Step 4: Create Company JSON**

Create `knowledge/source/company.json` with:

- `id` = `shengborun`.
- Exact company name.
- The four About-page company-introduction paragraphs as four ordered `introduction` strings.
- `source_path` = `/about/#company`.
- `published` = true.
- provenance = `src/pages/about.astro`.

Exclude the hero subtitle and contact content.

- [ ] **Step 5: Create Contact JSON**

Create `knowledge/source/contact.json` with:

- `id` = `shengborun`.
- `company_name` exactly as displayed.
- `duty_phone` exactly as displayed.
- `email` exactly as displayed.
- `source_path` = `/about/#contact`.
- `published` = true.
- provenance = `src/pages/about.astro`.

Do not include address, CTA, tel/mailto links, or product-selection guidance.

- [ ] **Step 6: Run source, relationship, and complete build tests**

Run:

```powershell
python -m unittest tests.knowledge_pipeline.test_curated_sources -v
python scripts/build_knowledge_documents.py
```

Expected report:

    Documents generated: 60
    - product: 49
    - solution: 6
    - support: 3
    - company: 1
    - contact: 1
    Output: knowledge/documents.jsonl

- [ ] **Step 7: Commit remaining curated sources**

```powershell
git add knowledge/source/solutions knowledge/source/support knowledge/source/company.json knowledge/source/contact.json tests/knowledge_pipeline/test_curated_sources.py
git commit -m "data: curate company knowledge sources"
```

---

### Task 9: End-to-End Verification and Human Documentation

**Files:**
- Create: `docs/knowledge-schema-v1.md`
- Modify: `tests/knowledge_pipeline/test_pipeline.py`
- Generate: `knowledge/documents.jsonl`

**Interfaces:**
- Consumes: all implementation and curated source data.
- Produces: final machine artifact, human guide, and an end-to-end regression test.

- [ ] **Step 1: Add a failing end-to-end determinism test**

Use the real curated source directory and two temporary output paths:

```python
def test_real_inventory_build_is_byte_deterministic(self):
    root = repository_root()
    first = self.temp_dir / "first.jsonl"
    second = self.temp_dir / "second.jsonl"

    first_report = run_build(root / "knowledge" / "source", first, DEFAULT_BASE_URL, False)
    second_report = run_build(root / "knowledge" / "source", second, DEFAULT_BASE_URL, False)

    self.assertEqual(first.read_bytes(), second.read_bytes())
    self.assertEqual(sum(first_report.counts.values()), 60)
    self.assertEqual(first_report.counts["product"], 49)
    self.assertEqual(first_report.counts["solution"], 6)
```

The exact-count assertions here are a regression check for the approved initial dataset, not a runtime schema invariant.

- [ ] **Step 2: Run the end-to-end test**

Run:

```powershell
python -m unittest tests.knowledge_pipeline.test_pipeline -v
```

Expected: fail until any integration discrepancy is resolved.

- [ ] **Step 3: Fix only integration defects exposed by the test**

Examples of allowed fixes are inconsistent path construction, provenance ordering, Markdown newline normalization, or report formatting. Do not loosen strict schema rules or add excluded fields to make the data pass.

Re-run the focused test after each minimal correction.

- [ ] **Step 4: Write the human-facing schema guide**

Create `docs/knowledge-schema-v1.md` containing:

1. Scope and explicit pre-chunk stopping point.
2. Current knowledge inventory.
3. Include/exclude table.
4. Source directory contract.
5. Complete schema and one valid JSON example for every source type.
6. Complete Knowledge Document Schema v1 example.
7. Text-versus-metadata explanation.
8. Per-type transformation templates.
9. ID, language, URL, provenance, and hash rules.
10. Validation and atomic failure behavior.
11. Default build command and `--check-urls` command.
12. Safe manual update workflow for adding, editing, unpublishing, and removing knowledge.
13. Expected build report and troubleshooting examples.

Do not copy the design spec verbatim; write it as an operator-facing reference.

- [ ] **Step 5: Generate the committed machine artifact**

Run:

```powershell
python scripts/build_knowledge_documents.py
```

Expected: 60 documents with the approved type counts.

Run it a second time and verify no Git diff is produced for `knowledge/documents.jsonl`:

```powershell
python scripts/build_knowledge_documents.py
git diff --exit-code -- knowledge/documents.jsonl
```

Expected: exit code 0 and no diff after the second identical build.

- [ ] **Step 6: Run all automated verification**

Run:

```powershell
python -m unittest discover -s tests -v
python scripts/build_knowledge_documents.py
git diff --check
```

Expected: all tests pass, build succeeds, and no whitespace errors are reported.

Do not run the networked `--check-urls` mode unless network access is available and explicitly appropriate. The offline build is the required acceptance gate.

- [ ] **Step 7: Perform the required manual sample inspection**

Inspect complete JSONL records for:

- A Product with multiple technical-parameter groups.
- `solution:smart-emergency` because it has nested headings and a long body.
- `support:solution-design`.
- `company:shengborun`.
- `contact:shengborun`.

Verify:

- Original technical values are unchanged.
- Product name and category appear in text.
- Solution Markdown hierarchy is retained.
- Contact has no address or product-selection rule.
- No Product or Solution metadata contains relationships to the other type.
- No image, alt, SEO, homepage promotion, navigation, breadcrumb, footer, or CTA appears.
- Source path, absolute URL, and source files are accurate.
- Hashes are unique and lowercase SHA-256.
- Each JSONL line is independently parseable.

Record the inspected IDs and result in the final implementation handoff; do not add timestamps to the artifact.

- [ ] **Step 8: Commit docs, tests, and generated output**

```powershell
git add docs/knowledge-schema-v1.md tests/knowledge_pipeline/test_pipeline.py knowledge/documents.jsonl
git commit -m "docs: publish knowledge schema v1"
```

- [ ] **Step 9: Final scope audit**

Run:

```powershell
rg -n "chunk|embedding|vector|similarity|retrieval|openai|llm" knowledge_pipeline scripts/build_knowledge_documents.py
```

Expected: no implementation of any post-normalization capability. Mentions in comments or error text are unnecessary and should be removed; the operator documentation may state the explicit exclusions.

Run `git status --short` and require a clean worktree before reporting completion.
