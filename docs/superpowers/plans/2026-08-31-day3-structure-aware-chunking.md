# Day 3 Structure-Aware Chunking V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the 60 validated Day 2 documents in `knowledge/documents.jsonl` into deterministic, validated, structure-aware chunks in `knowledge/chunks.jsonl`, then stop before embeddings or retrieval.

**Architecture:** Add a strict `KnowledgeChunk` Pydantic contract and a dedicated `knowledge_pipeline.chunking` module. The module reloads and validates Day 2 JSONL, parses its controlled Markdown templates into registered semantic units, applies one type-aware production strategy, validates provenance and evidence coverage, and atomically serializes deterministic JSONL. A small CLI generates the artifact and prints character-based statistics.

**Tech Stack:** Python 3, Pydantic 2, standard-library `json`, `pathlib`, `tempfile`, `statistics`, `unittest`; no tokenizer, embedding, vector, retrieval, or LLM dependencies.

**Spec:** `docs/superpowers/specs/2026-08-31-day3-structure-aware-chunking-design.md`

## Global Constraints

- Day 3 reads only `knowledge/documents.jsonl`; it must not read `knowledge/source`, Astro Content, or `D:\Shengborun`.
- Implement only the production `structure_aware` strategy; do not add `whole_document`, `fixed_size`, or an experiment framework.
- Use a 1000 Unicode-character soft maximum for Product, Solution, Support, and Company; Contact always remains one chunk.
- Split oversized semantic sections only at paragraph or complete-list-item boundaries; an indivisible unit may exceed 1000 characters.
- Do not mechanically overlap, summarize, rewrite, infer, translate factual text, or create Product–Solution relationships.
- Chunk IDs are deterministic ASCII identifiers backed by explicit mappings; new unmapped real headings fail the build.
- `section` and `text` retain the real Chinese heading and factual text.
- Every input Document produces at least one Chunk and every registered factual semantic unit is covered by original text.
- Output is UTF-8 JSONL, LF-only, fixed field order, no timestamps, atomic, and byte deterministic.
- Preserve every Day 2 test and stop before embedding, vector storage, retrieval, reranking, LLM calls, or RAG generation.

## File Structure

- Modify `knowledge_pipeline/models.py`: define strict Chunk Schema v1 and Chunk ID validation.
- Modify `knowledge_pipeline/core.py`: expose the existing canonical Document hash calculation for safe Day 3 input verification without duplicating hash logic.
- Create `knowledge_pipeline/chunking.py`: own JSONL loading, controlled Markdown parsing, title registries, semantic chunking, coverage validation, collection validation, deterministic serialization, atomic writing, and statistics.
- Modify `knowledge_pipeline/__init__.py`: export the stable Day 3 public interfaces.
- Create `scripts/build_knowledge_chunks.py`: minimal command-line entry point with default input/output and required statistics.
- Create `tests/test_chunking_pipeline.py`: focused schema, parser, chunker, validation, atomic-output, real-inventory, and deterministic-output tests.
- Create `docs/chunk-schema-v1.md`: human-readable contract, rules, provenance, synchronization boundary, build command, and scope exclusions.
- Generate `knowledge/chunks.jsonl`: checked-in deterministic production artifact built from the checked-in Day 2 documents.

---

### Task 1: Define the strict Chunk Schema and stable heading registries

**Files:**
- Modify: `knowledge_pipeline/models.py`
- Create: `knowledge_pipeline/chunking.py`
- Create: `tests/test_chunking_pipeline.py`

**Interfaces:**
- Consumes: existing `Metadata`, `StrictModel`, `SHA256_HEX`, and type-specific metadata models from `knowledge_pipeline.models`.
- Produces: `KnowledgeChunk`; `PRODUCT_SPEC_SECTION_SLUGS`; `PRODUCT_OPTIONAL_SECTION_SLUGS`; `SOLUTION_BODY_SECTION_SLUGS`; `COMPANY_SECTION_SLUGS`; `MAX_CHUNK_CHARACTERS = 1000`.

- [ ] **Step 1: Write failing schema tests**

Create `tests/test_chunking_pipeline.py` with a valid Product chunk payload and tests for strict fields, ASCII identity, parent identity, hash format, LF text, sorted provenance, and type-specific metadata:

```python
import unittest

from pydantic import ValidationError

from knowledge_pipeline.models import KnowledgeChunk


def valid_chunk_payload() -> dict:
    return {
        "schema_version": "1.0",
        "chunk_id": "product:xir-p8668ex:features",
        "parent_document_id": "product:xir-p8668ex",
        "parent_content_hash": "a" * 64,
        "type": "product",
        "section": "产品特点",
        "text": "# 摩托罗拉 XiR P8668Ex\n\n## 产品特点\n\n- 防爆机型",
        "language": "zh-CN",
        "source_url": "https://www.shengborun.com/two-way-radio/xir-p8668ex/",
        "source_files": ["src/content/products/two-way-radio/xir-p8668ex.json"],
        "metadata": {
            "product_id": "xir-p8668ex",
            "slug": "xir-p8668ex",
            "category_id": "two-way-radio",
            "category_name": "对讲机通信",
        },
    }


class KnowledgeChunkSchemaTests(unittest.TestCase):
    def test_valid_product_chunk(self):
        chunk = KnowledgeChunk.model_validate(valid_chunk_payload())
        self.assertEqual(chunk.section, "产品特点")

    def test_chunk_rejects_non_ascii_id(self):
        payload = valid_chunk_payload()
        payload["chunk_id"] = "product:xir-p8668ex:产品特点"
        with self.assertRaises(ValidationError):
            KnowledgeChunk.model_validate(payload)

    def test_chunk_rejects_wrong_parent_type(self):
        payload = valid_chunk_payload()
        payload["parent_document_id"] = "solution:xir-p8668ex"
        with self.assertRaises(ValidationError):
            KnowledgeChunk.model_validate(payload)
```

- [ ] **Step 2: Run the schema tests and verify the missing model failure**

Run: `python -m unittest tests.test_chunking_pipeline.KnowledgeChunkSchemaTests -v`

Expected: FAIL because `KnowledgeChunk` does not exist.

- [ ] **Step 3: Add the minimal strict Chunk model**

In `knowledge_pipeline/models.py`, add a colon-segmented ASCII pattern and `KnowledgeChunk`. Reuse the existing `Metadata` union rather than creating a second metadata contract:

```python
CHUNK_ID = re.compile(
    r"^[a-z0-9]+(?:-[a-z0-9]+)*(?::[a-z0-9]+(?:-[a-z0-9]+)*)+$"
)


class KnowledgeChunk(StrictModel):
    schema_version: Literal["1.0"]
    chunk_id: NonEmptyStr
    parent_document_id: NonEmptyStr
    parent_content_hash: NonEmptyStr
    type: Literal["product", "solution", "support", "company", "contact"]
    section: NonEmptyStr
    text: NonEmptyStr
    language: Literal["zh-CN"]
    source_url: NonEmptyStr
    source_files: list[NonEmptyStr] = Field(min_length=1)
    metadata: Metadata

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        if not CHUNK_ID.fullmatch(value):
            raise ValueError("must contain colon-separated lowercase ASCII segments")
        return value

    @field_validator("parent_content_hash")
    @classmethod
    def validate_parent_hash(cls, value: str) -> str:
        if not SHA256_HEX.fullmatch(value):
            raise ValueError("must be a lowercase SHA-256 hex digest")
        return value

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("must be an absolute HTTP(S) URL")
        return value

    @field_validator("source_files")
    @classmethod
    def validate_source_files(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("must be sorted and unique")
        return value

    @field_validator("text")
    @classmethod
    def validate_lf_text(cls, value: str) -> str:
        if "\r" in value:
            raise ValueError("must use LF line endings")
        return value

    @model_validator(mode="after")
    def validate_identity_and_metadata(self) -> "KnowledgeChunk":
        if not self.parent_document_id.startswith(f"{self.type}:"):
            raise ValueError("parent_document_id must start with '<type>:'")
        if not self.chunk_id.startswith(f"{self.parent_document_id}:"):
            raise ValueError("chunk_id must extend parent_document_id")
        expected = {
            "product": ProductMetadata,
            "solution": SolutionMetadata,
            "support": SupportMetadata,
            "company": CompanyMetadata,
            "contact": ContactMetadata,
        }[self.type]
        if not isinstance(self.metadata, expected):
            raise ValueError(f"metadata does not match chunk type {self.type}")
        return self
```

- [ ] **Step 4: Add explicit heading registries**

Create `knowledge_pipeline/chunking.py` with the fixed soft limit and every real heading currently present in `documents.jsonl`:

```python
MAX_CHUNK_CHARACTERS = 1000

PRODUCT_SPEC_SECTION_SLUGS = {
    "安全防护": "security-protection",
    "安全上网": "secure-internet-access",
    "安全邮件": "secure-email",
    "负载均衡": "load-balancing",
    "高可靠性": "high-availability",
    "技术规格要求": "technical-requirements",
    "可选功能": "optional-features",
    "链路聚合": "link-aggregation",
    "绿色节能": "energy-efficiency",
    "设备接口": "device-interfaces",
    "数据分析": "data-analysis",
    "特色功能": "distinctive-features",
    "文件传输": "file-transfer",
    "物理指标": "physical-specifications",
    "系统参数": "system-parameters",
    "系统管理": "system-management",
    "一般规格": "general",
    "POE": "poe",
}

PRODUCT_OPTIONAL_SECTION_SLUGS = {
    "应用场景": "applications",
}

SOLUTION_BODY_SECTION_SLUGS = {
    "方案概述": "overview",
    "应急救援解决方案": "emergency-rescue-solution",
    "行业背景": "industry-background",
    "解决方案": "solution",
    "系统功能": "system-functions",
    "行业通信现状": "industry-communications-status",
    "针对大型石油石化企业的数字集群系统解决方案": "digital-trunking-solution",
    "方案描述": "solution-description",
    "建设背景": "construction-background",
    "业务痛点": "business-pain-points",
    "客户需求": "customer-requirements",
}

COMPANY_SECTION_SLUGS = {
    "公司简介": "company-profile",
}
```

Add `HeadingRegistryTests` that asserts the registries equal the explicit sets above. The real-inventory test in Task 5 will additionally scan the checked-in 60 Documents and prove that every current Product H3, optional Product H2, Solution body H2, and Company H2 is mapped. Exclude fixed template headings and the `详细内容` boundary marker from the body-heading assertion.

- [ ] **Step 5: Run the schema and registry tests**

Run: `python -m unittest tests.test_chunking_pipeline.KnowledgeChunkSchemaTests tests.test_chunking_pipeline.HeadingRegistryTests -v`

Expected: PASS.

- [ ] **Step 6: Commit the schema contract**

```bash
git add knowledge_pipeline/models.py knowledge_pipeline/chunking.py tests/test_chunking_pipeline.py
git commit -m "feat: define chunk schema and heading registry"
```

---

### Task 2: Load validated Day 2 JSONL and parse controlled Markdown

**Files:**
- Modify: `knowledge_pipeline/core.py`
- Modify: `knowledge_pipeline/__init__.py`
- Modify: `knowledge_pipeline/chunking.py`
- Modify: `tests/test_chunking_pipeline.py`

**Interfaces:**
- Consumes: `KnowledgeDocument`, `BuildError`, and the Day 2 canonical hash payload.
- Produces: `compute_content_hash(type_: str, title: str, text: str, metadata: BaseModel) -> str`; `load_documents(input_path: Path) -> list[KnowledgeDocument]`; `_parse_markdown(document: KnowledgeDocument) -> ParsedDocument`; `_split_h3(section: MarkdownSection) -> list[MarkdownSection]`.

- [ ] **Step 1: Add reusable exact Document factories to the Day 3 test module**

After `compute_content_hash` is exposed in Step 4, use one strict factory throughout the test module. Until then, importing it is the expected red state for the loader tests:

```python
from contextlib import contextmanager
from typing import Iterator

from knowledge_pipeline.core import compute_content_hash
from knowledge_pipeline.models import KnowledgeDocument


def make_document(
    *,
    type_: str,
    entity_id: str,
    title: str,
    text: str,
    source_url: str,
    source_files: list[str],
    metadata: dict,
) -> KnowledgeDocument:
    provisional = KnowledgeDocument.model_validate({
        "schema_version": "1.0",
        "document_id": f"{type_}:{entity_id}",
        "type": type_,
        "title": title,
        "text": text,
        "language": "zh-CN",
        "source_path": "/fixture/",
        "source_url": source_url,
        "source_files": source_files,
        "content_hash": "0" * 64,
        "metadata": metadata,
    })
    return provisional.model_copy(update={
        "content_hash": compute_content_hash(
            provisional.type,
            provisional.title,
            provisional.text,
            provisional.metadata,
        )
    })


def make_product_document() -> KnowledgeDocument:
    return make_document(
        type_="product",
        entity_id="xir-p8668ex",
        title="摩托罗拉 XiR P8668Ex",
        text=(
            "# 摩托罗拉 XiR P8668Ex\n\n"
            "产品分类：对讲机通信\n\n"
            "## 产品介绍\n\n数字防爆对讲机。\n\n"
            "## 产品特点\n\n- 防爆机型\n- 音质清晰\n\n"
            "## 技术参数\n\n### 一般规格\n\n- 输出功率：1W"
        ),
        source_url="https://www.shengborun.com/two-way-radio/xir-p8668ex/",
        source_files=["src/content/products/two-way-radio/xir-p8668ex.json"],
        metadata={
            "product_id": "xir-p8668ex",
            "slug": "xir-p8668ex",
            "category_id": "two-way-radio",
            "category_name": "对讲机通信",
        },
    )


def make_solution_document(text: str | None = None) -> KnowledgeDocument:
    normalized = text or (
        "# 智慧应急解决方案\n\n"
        "## 方案摘要\n\n提升应急管理能力。\n\n"
        "## 核心需求\n\n- 统一数据\n\n"
        "## 方案设计\n\n建设应急平台。\n\n"
        "## 方案特点\n\n- 协同指挥\n\n"
        "## 详细内容\n\n"
        "## 业务痛点\n\n"
        "### 风险感知滞后\n\n风险发现不及时。\n\n"
        "### 协同指挥低效\n\n跨部门协同困难。"
    )
    return make_document(
        type_="solution",
        entity_id="smart-emergency",
        title="智慧应急解决方案",
        text=normalized,
        source_url="https://www.shengborun.com/solutions/smart-emergency/",
        source_files=["src/content/solutions/smart-emergency.md"],
        metadata={"solution_id": "smart-emergency", "slug": "smart-emergency"},
    )


def long_feature_solution_text() -> str:
    return (
        "# 智慧应急解决方案\n\n"
        "## 方案摘要\n\n提升应急管理能力。\n\n"
        "## 核心需求\n\n- 统一数据\n\n"
        "## 方案设计\n\n建设应急平台。\n\n"
        "## 方案特点\n\n"
        "- 特点甲完整内容特点甲完整内容特点甲完整内容\n"
        "- 特点乙完整内容特点乙完整内容特点乙完整内容\n\n"
        "## 详细内容\n\n"
        "## 业务痛点\n\n风险发现不及时。"
    )


def make_support_document(body: str = "围绕覆盖范围提供网络规划方案。") -> KnowledgeDocument:
    return make_document(
        type_="support",
        entity_id="solution-design",
        title="方案设计",
        text=f"# 方案设计\n\n## 服务摘要\n\n确认客户需求。\n\n## 服务内容\n\n{body}",
        source_url="https://www.shengborun.com/support/#solution-design",
        source_files=["src/data/support-services.ts"],
        metadata={"service_id": "solution-design"},
    )


def make_company_document() -> KnowledgeDocument:
    return make_document(
        type_="company",
        entity_id="shengborun",
        title="北京盛博润通信设备有限公司",
        text="# 北京盛博润通信设备有限公司\n\n## 公司简介\n\n公司成立于2011年。",
        source_url="https://www.shengborun.com/about/#company",
        source_files=["src/pages/about.astro"],
        metadata={"company_id": "shengborun"},
    )


def make_contact_document() -> KnowledgeDocument:
    return make_document(
        type_="contact",
        entity_id="shengborun",
        title="联系我们",
        text="# 联系我们\n\n公司名称：北京盛博润通信设备有限公司\n值班电话：13911733859\n电子邮箱：lsk777@sina.com",
        source_url="https://www.shengborun.com/about/#contact",
        source_files=["src/pages/about.astro"],
        metadata={"contact_id": "shengborun"},
    )


@contextmanager
def temporary_document_file(
    documents: list[KnowledgeDocument],
) -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "documents.jsonl"
        lines = [
            json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
            for item in documents
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        yield path
```

- [ ] **Step 2: Write failing loader tests**

Add a helper that writes supplied `KnowledgeDocument` objects as JSONL, then test valid loading, precise invalid JSON diagnostics, Pydantic diagnostics, duplicate IDs, and stale content hashes:

```python
class DocumentLoaderTests(unittest.TestCase):
    def test_invalid_json_reports_file_and_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "documents.jsonl"
            path.write_text("{}\n{bad json}\n", encoding="utf-8")
            with self.assertRaises(BuildError) as context:
                load_documents(path)
            message = str(context.exception)
            self.assertIn(str(path), message)
            self.assertIn("line: 2", message)
            self.assertIn("invalid JSON", message)

    def test_stale_hash_reports_document_id(self):
        document = make_product_document().model_copy(
            update={"content_hash": "b" * 64}
        )
        with temporary_document_file([document]) as path:
            with self.assertRaises(BuildError) as context:
                load_documents(path)
        self.assertIn("product:xir-p8668ex", str(context.exception))
        self.assertIn("content_hash", str(context.exception))
```

- [ ] **Step 3: Run the loader tests and verify they fail**

Run: `python -m unittest tests.test_chunking_pipeline.DocumentLoaderTests -v`

Expected: FAIL because `load_documents` and the public hash helper do not exist.

- [ ] **Step 4: Expose the canonical Day 2 content hash helper**

In `knowledge_pipeline/core.py`, rename `_hash_payload` to `compute_content_hash`, update `_document` and `validate_documents` to call the public name, and export it from `knowledge_pipeline/__init__.py`. Do not change the hash payload or generated Day 2 hashes:

```python
def compute_content_hash(
    type_: str,
    title: str,
    text: str,
    metadata: BaseModel,
) -> str:
    payload = {
        "type": type_,
        "title": title,
        "text": text,
        "language": "zh-CN",
        "metadata": metadata.model_dump(mode="json"),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

- [ ] **Step 5: Implement strict JSONL loading**

In `knowledge_pipeline/chunking.py`, implement `load_documents` so it:

1. Requires a real file.
2. Rejects empty lines instead of silently changing line numbering.
3. Reports JSON errors with file and one-based line number.
4. Reports Pydantic errors with file, line, recognizable `document_id`, field path, and reason.
5. Recomputes `content_hash` with `compute_content_hash`.
6. Rejects duplicate `document_id`.
7. Preserves input order.

Use this diagnostic form:

```python
def _document_error(
    input_path: Path,
    line_number: int,
    reason: str,
    document_id: str | None = None,
) -> str:
    lines = [f"[ERROR] {input_path}", f"line: {line_number}"]
    if document_id:
        lines.append(f"document_id: {document_id}")
    lines.append(f"reason: {reason}")
    return "\n".join(lines)
```

- [ ] **Step 6: Write failing Markdown parser tests**

Test the normalized template shape explicitly:

```python
class MarkdownParserTests(unittest.TestCase):
    def test_parser_preserves_preamble_h2_and_nested_h3_text(self):
        parsed = _parse_markdown(make_product_document())
        self.assertEqual(parsed.title_line, "# 摩托罗拉 XiR P8668Ex")
        self.assertEqual(parsed.preamble, "产品分类：对讲机通信")
        self.assertEqual(
            [section.heading for section in parsed.h2_sections],
            ["产品介绍", "产品特点", "技术参数"],
        )
        groups = _split_h3(parsed.h2_sections[2])
        self.assertEqual([group.heading for group in groups], ["一般规格"])
        self.assertIn("- 输出功率：1W", groups[0].raw)
```

- [ ] **Step 7: Implement the controlled Markdown parser**

Add immutable internal records and parsing functions. They must preserve normalized text exactly, reject missing or mismatched H1, and retain H3 content inside its parent H2 until a type-specific chunker asks to split it:

```python
@dataclass(frozen=True)
class MarkdownSection:
    level: int
    heading: str
    raw: str
    body: str


@dataclass(frozen=True)
class ParsedDocument:
    title_line: str
    preamble: str
    h2_sections: tuple[MarkdownSection, ...]
```

`_parse_markdown` must split only on lines matching `^## `; `_split_h3` must split only on lines matching `^### `. Lines that merely contain `#` remain ordinary facts.

- [ ] **Step 8: Run loader, parser, and all Day 2 tests**

Run: `python -m unittest tests.test_chunking_pipeline.DocumentLoaderTests tests.test_chunking_pipeline.MarkdownParserTests tests.test_knowledge_pipeline -v`

Expected: PASS, including unchanged Day 2 hashes and serialization tests.

- [ ] **Step 9: Commit the validated input boundary**

```bash
git add knowledge_pipeline/core.py knowledge_pipeline/__init__.py knowledge_pipeline/chunking.py tests/test_chunking_pipeline.py
git commit -m "feat: load and parse normalized documents"
```

---

### Task 3: Implement the type-aware production chunker with evidence coverage

**Files:**
- Modify: `knowledge_pipeline/chunking.py`
- Modify: `tests/test_chunking_pipeline.py`

**Interfaces:**
- Consumes: `ParsedDocument`, `MarkdownSection`, heading registries, and `KnowledgeDocument`.
- Produces: `build_chunks(documents: list[KnowledgeDocument], max_characters: int = 1000) -> list[KnowledgeChunk]`; internal `SemanticUnit` and `ChunkCandidate` coverage records.

- [ ] **Step 1: Write failing Product chunking tests**

Assert exact IDs, real Chinese section names, inherited fields, identity context, and absence of inferred knowledge:

```python
class ProductChunkingTests(unittest.TestCase):
    def test_product_splits_overview_features_and_parameter_groups(self):
        document = make_product_document()
        chunks = build_chunks([document])
        self.assertEqual(
            [chunk.chunk_id for chunk in chunks],
            [
                "product:xir-p8668ex:overview",
                "product:xir-p8668ex:features",
                "product:xir-p8668ex:spec:general",
            ],
        )
        self.assertEqual(chunks[2].section, "一般规格")
        self.assertIn("# 摩托罗拉 XiR P8668Ex", chunks[2].text)
        self.assertIn("## 技术参数", chunks[2].text)
        self.assertIn("### 一般规格", chunks[2].text)
        self.assertNotIn("应用场景", "\n".join(chunk.text for chunk in chunks))
        self.assertEqual(chunks[0].metadata, document.metadata)
        self.assertEqual(chunks[0].source_files, document.source_files)
```

Add a second Product with an unmapped H3 and assert `BuildError` contains the parent ID and real heading.

- [ ] **Step 2: Run Product tests and verify they fail**

Run: `python -m unittest tests.test_chunking_pipeline.ProductChunkingTests -v`

Expected: FAIL because `build_chunks` does not exist.

- [ ] **Step 3: Implement semantic units, Chunk construction, and Product rules**

Use internal coverage records so every factual source block has an identity and an assignment:

```python
@dataclass(frozen=True)
class SemanticUnit:
    key: str
    text: str


@dataclass(frozen=True)
class ChunkCandidate:
    chunk: KnowledgeChunk
    covered_unit_keys: tuple[str, ...]
```

Create chunks only through this interface, which copies `parent_content_hash`, `type`, `language`, `source_url`, `source_files`, and `metadata` from the parent:

```python
def _chunk(
    parent: KnowledgeDocument,
    *,
    chunk_id: str,
    section: str,
    text: str,
) -> KnowledgeChunk:
    return KnowledgeChunk.model_validate({
        "schema_version": "1.0",
        "chunk_id": chunk_id,
        "parent_document_id": parent.document_id,
        "parent_content_hash": parent.content_hash,
        "type": parent.type,
        "section": section,
        "text": text,
        "language": parent.language,
        "source_url": parent.source_url,
        "source_files": parent.source_files,
        "metadata": parent.metadata.model_dump(mode="json"),
    })
```

Product parsing must require the fixed `产品介绍`, `产品特点`, and `技术参数` structure, combine H1 + preamble + introduction for `overview`, and use the explicit registry for every H3 group. Recognize only explicitly mapped optional H2 sections such as `应用场景`; any other Product H2 fails instead of being ignored.

- [ ] **Step 4: Write failing Solution tests**

Use a Solution with the four fixed sections, the `详细内容` marker, an H2 `业务痛点`, and multiple H3 children. Assert the H3 sections stay together:

```python
class SolutionChunkingTests(unittest.TestCase):
    def test_solution_keeps_h3_children_in_their_real_h2_chunk(self):
        chunks = build_chunks([make_solution_document()])
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.assertEqual(
            list(by_id),
            [
                "solution:smart-emergency:summary",
                "solution:smart-emergency:core-needs",
                "solution:smart-emergency:design",
                "solution:smart-emergency:features",
                "solution:smart-emergency:body:business-pain-points",
            ],
        )
        pain_points = by_id[
            "solution:smart-emergency:body:business-pain-points"
        ]
        self.assertEqual(pain_points.section, "业务痛点")
        self.assertIn("### 风险感知滞后", pain_points.text)
        self.assertIn("### 协同指挥低效", pain_points.text)
```

Also test that `详细内容` with direct prose before the first real body H2 produces a deterministic `solution:<id>:details` Chunk, so that prose cannot be silently lost. An empty wrapper marker produces no empty Chunk.

- [ ] **Step 5: Implement Solution rules**

Split the fixed prefix by position before `## 详细内容`, not merely by heading name. Require the fixed sequence:

```python
SOLUTION_FIXED_SECTIONS = (
    ("方案摘要", "summary"),
    ("核心需求", "core-needs"),
    ("方案设计", "design"),
    ("方案特点", "features"),
)
```

After the `详细内容` boundary, treat each real H2 and all of its H3 descendants as one semantic section. Fail on unmapped headings or duplicate generated IDs. Never create Architecture or Benefits sections.

- [ ] **Step 6: Write failing Support, Company, and Contact tests**

Assert Support remains one chunk under the soft limit, Company remains one chunk for the current real `公司简介` section, and Contact always remains one Chunk. Separately assert that an unmapped future Company H2 fails explicitly rather than being silently discarded; after maintainers add its explicit registry mapping, the generic Company logic will split it as a real section.

```python
class OtherTypeChunkingTests(unittest.TestCase):
    def test_short_support_company_and_contact_remain_whole(self):
        documents = [
            make_support_document(),
            make_company_document(),
            make_contact_document(),
        ]
        chunks = build_chunks(documents)
        self.assertEqual(
            [chunk.chunk_id for chunk in chunks],
            [
                "support:solution-design:content",
                "company:shengborun:company-profile",
                "contact:shengborun:contact",
            ],
        )
```

- [ ] **Step 7: Implement Support, Company, and Contact rules**

- Support uses its complete normalized text and `section` equal to its service title; oversized Support content uses the common natural splitter.
- Company with one real H2 uses one Chunk; multiple real H2 sections use an explicit Company heading registry and repeat H1 identity.
- Contact uses the complete normalized text and never enters the soft-limit splitter.
- Any preamble facts must be assigned to the first appropriate Chunk or a deterministic overview Chunk; they may not be dropped.

- [ ] **Step 8: Write failing natural-boundary and coverage tests**

Add tests with a small injected maximum such as 90 characters so the fixtures stay readable. Verify paragraph/list packing, deterministic `:1` and `:2`, no sentence truncation, no repeated factual unit, and an indivisible over-limit paragraph:

```python
class NaturalSplitAndCoverageTests(unittest.TestCase):
    def test_oversized_section_splits_only_between_complete_list_items(self):
        document = make_solution_document(text=long_feature_solution_text())
        chunks = build_chunks([document], max_characters=90)
        feature_chunks = [
            chunk for chunk in chunks if ":features:" in chunk.chunk_id
        ]
        self.assertEqual(
            [chunk.chunk_id.rsplit(":", 1)[-1] for chunk in feature_chunks],
            ["1", "2"],
        )
        combined = "\n".join(chunk.text for chunk in feature_chunks)
        for item in ("- 特点甲完整内容", "- 特点乙完整内容"):
            self.assertEqual(combined.count(item), 1)

    def test_indivisible_paragraph_may_exceed_soft_limit(self):
        document = make_support_document(body="连续事实" * 80)
        chunks = build_chunks([document], max_characters=100)
        self.assertEqual(len(chunks), 1)
        self.assertGreater(len(chunks[0].text), 100)
```

- [ ] **Step 9: Implement natural splitting and coverage validation**

Implement `_natural_units` to separate blank-line paragraphs and recognize complete Markdown list items. Implement a greedy packer that counts the repeated identity/heading prefix in each Chunk, never splits a unit, and adds numeric suffixes only when more than one output Chunk is produced.

Before returning from `build_chunks`, validate coverage:

```python
def _validate_coverage(
    parent: KnowledgeDocument,
    units: list[SemanticUnit],
    candidates: list[ChunkCandidate],
) -> None:
    expected = {unit.key for unit in units}
    assigned = {
        key
        for candidate in candidates
        for key in candidate.covered_unit_keys
    }
    if assigned != expected:
        missing = sorted(expected - assigned)
        unexpected = sorted(assigned - expected)
        raise BuildError(
            f"[ERROR] {parent.document_id}\n"
            f"reason: semantic coverage mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    unit_by_key = {unit.key: unit for unit in units}
    for candidate in candidates:
        for key in candidate.covered_unit_keys:
            if unit_by_key[key].text not in candidate.chunk.text:
                raise BuildError(
                    f"[ERROR] {parent.document_id}\n"
                    f"reason: source text changed or missing for semantic unit {key}"
                )
```

Additionally reject a semantic unit assigned more than once unless it is explicitly marked as repeatable identity context. Factual units are never repeatable.

- [ ] **Step 10: Run all type-aware and coverage tests**

Run: `python -m unittest tests.test_chunking_pipeline.ProductChunkingTests tests.test_chunking_pipeline.SolutionChunkingTests tests.test_chunking_pipeline.OtherTypeChunkingTests tests.test_chunking_pipeline.NaturalSplitAndCoverageTests -v`

Expected: PASS.

- [ ] **Step 11: Commit the production chunker**

```bash
git add knowledge_pipeline/chunking.py tests/test_chunking_pipeline.py
git commit -m "feat: add structure-aware document chunking"
```

---

### Task 4: Validate collections, write deterministic JSONL atomically, and expose statistics

**Files:**
- Modify: `knowledge_pipeline/chunking.py`
- Modify: `knowledge_pipeline/__init__.py`
- Create: `scripts/build_knowledge_chunks.py`
- Modify: `tests/test_chunking_pipeline.py`

**Interfaces:**
- Consumes: `load_documents`, `build_chunks`, `KnowledgeChunk`, and parent `KnowledgeDocument` objects.
- Produces: `validate_chunks(chunks, documents) -> None`; `serialize_chunks(chunks) -> bytes`; `write_chunks(output_path, chunks, documents) -> None`; `ChunkStatistics`; `calculate_chunk_statistics(documents, chunks) -> ChunkStatistics`; `build_and_write_chunks(input_path, output_path) -> ChunkStatistics`.

- [ ] **Step 1: Write failing collection-validation tests**

Cover duplicate Chunk IDs, unknown parent IDs, inherited-field drift, empty parent coverage, and output order drift:

```python
class ChunkCollectionValidationTests(unittest.TestCase):
    def test_duplicate_chunk_id_is_fatal(self):
        documents = [make_product_document()]
        chunks = build_chunks(documents)
        with self.assertRaises(BuildError) as context:
            validate_chunks([chunks[0], chunks[0]], documents)
        self.assertIn("duplicate chunk_id", str(context.exception))

    def test_inherited_source_url_must_match_parent(self):
        documents = [make_product_document()]
        chunks = build_chunks(documents)
        changed = chunks[0].model_copy(
            update={"source_url": "https://example.com/wrong"}
        )
        with self.assertRaises(BuildError) as context:
            validate_chunks([changed, *chunks[1:]], documents)
        self.assertIn("source_url", str(context.exception))
```

- [ ] **Step 2: Run validation tests and verify they fail**

Run: `python -m unittest tests.test_chunking_pipeline.ChunkCollectionValidationTests -v`

Expected: FAIL because `validate_chunks` does not exist.

- [ ] **Step 3: Implement collection validation**

Build a parent map and validate:

- globally unique Chunk IDs;
- every parent reference exists;
- parent hash and inherited fields match exactly;
- section/type metadata are valid;
- every parent has at least one Chunk;
- Chunk order equals a fresh deterministic `build_chunks(documents)` result.

Compare full `model_dump(mode="json")` payloads against the freshly built expected sequence so a re-read temporary file cannot silently reorder or mutate content.

- [ ] **Step 4: Write failing serialization and atomic-output tests**

Test UTF-8 Chinese, LF-only output, one compact object per line, fixed field order, final newline, two identical builds, and preservation of an existing output file on invalid input:

```python
class ChunkSerializationTests(unittest.TestCase):
    def test_serialization_is_utf8_lf_and_byte_deterministic(self):
        documents = [make_product_document(), make_contact_document()]
        chunks = build_chunks(documents)
        first = serialize_chunks(chunks)
        second = serialize_chunks(build_chunks(documents))
        self.assertEqual(first, second)
        self.assertNotIn(b"\r", first)
        self.assertTrue(first.endswith(b"\n"))
        self.assertFalse(first.endswith(b"\n\n"))
        self.assertIn("产品特点".encode("utf-8"), first)

    def test_failed_build_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "documents.jsonl"
            output_path = root / "chunks.jsonl"
            input_path.write_text("{invalid}\n", encoding="utf-8")
            output_path.write_bytes(b"previous-output\n")
            with self.assertRaises(BuildError):
                build_and_write_chunks(input_path, output_path)
            self.assertEqual(output_path.read_bytes(), b"previous-output\n")
```

- [ ] **Step 5: Implement deterministic serialization and atomic writing**

`serialize_chunks` must serialize model fields in declaration order with `ensure_ascii=False` and compact separators, preserving the already validated list order:

```python
def serialize_chunks(chunks: list[KnowledgeChunk]) -> bytes:
    lines = [
        json.dumps(
            chunk.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for chunk in chunks
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")
```

Follow the Day 2 `NamedTemporaryFile` + flush + `os.fsync` + re-read + `os.replace` pattern. Re-parse every temporary line as `KnowledgeChunk`, call `validate_chunks`, compare serialized bytes, and only then replace the destination.

- [ ] **Step 6: Write failing statistics tests**

Use exact Unicode character lengths and check total Documents, total Chunks, arithmetic mean, median, min, max, per-parent counts, and per-type counts:

```python
class ChunkStatisticsTests(unittest.TestCase):
    def test_statistics_use_unicode_character_lengths(self):
        documents = [make_product_document(), make_contact_document()]
        chunks = build_chunks(documents)
        stats = calculate_chunk_statistics(documents, chunks)
        lengths = [len(chunk.text) for chunk in chunks]
        self.assertEqual(stats.total_documents, 2)
        self.assertEqual(stats.total_chunks, len(chunks))
        self.assertEqual(stats.minimum_characters, min(lengths))
        self.assertEqual(stats.maximum_characters, max(lengths))
        self.assertEqual(stats.chunks_per_document["contact:shengborun"], 1)
```

- [ ] **Step 7: Implement statistics and orchestration**

Add an immutable result model:

```python
@dataclass(frozen=True)
class ChunkStatistics:
    total_documents: int
    total_chunks: int
    average_characters: float
    median_characters: float
    minimum_characters: int
    maximum_characters: int
    chunks_per_document: dict[str, int]
    chunks_per_type: dict[str, int]
```

Use `statistics.fmean` and `statistics.median`. `build_and_write_chunks` must load, build, validate, atomically write, and return statistics only after successful replacement.

- [ ] **Step 8: Create the minimal CLI**

Create `scripts/build_knowledge_chunks.py` with only `--input` and `--output` options, defaulting to the checked-in knowledge files. Print stable, labeled statistics and every per-document/type count. On `BuildError`, print the detailed diagnostic to stderr and return exit code 1:

```python
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        stats = build_and_write_chunks(args.input, args.output)
    except BuildError as error:
        print(str(error), file=sys.stderr)
        return 1

    print(f"Documents processed: {stats.total_documents}")
    print(f"Chunks generated: {stats.total_chunks}")
    print(f"Average chunk length (characters): {stats.average_characters:.2f}")
    print(f"Median chunk length (characters): {stats.median_characters:.2f}")
    print(f"Minimum chunk length (characters): {stats.minimum_characters}")
    print(f"Maximum chunk length (characters): {stats.maximum_characters}")
    for document_id, count in stats.chunks_per_document.items():
        print(f"- {document_id}: {count}")
    for type_, count in stats.chunks_per_type.items():
        print(f"- {type_}: {count}")
    print(f"Output: {args.output}")
    return 0
```

- [ ] **Step 9: Export public Day 3 interfaces and run Task 4 tests**

Export `KnowledgeChunk`, `ChunkStatistics`, `build_and_write_chunks`, `build_chunks`, `load_documents`, `serialize_chunks`, and `validate_chunks` from `knowledge_pipeline/__init__.py`.

Run: `python -m unittest tests.test_chunking_pipeline.ChunkCollectionValidationTests tests.test_chunking_pipeline.ChunkSerializationTests tests.test_chunking_pipeline.ChunkStatisticsTests -v`

Expected: PASS.

- [ ] **Step 10: Commit deterministic output infrastructure**

```bash
git add knowledge_pipeline/chunking.py knowledge_pipeline/__init__.py scripts/build_knowledge_chunks.py tests/test_chunking_pipeline.py
git commit -m "feat: validate and serialize knowledge chunks"
```

---

### Task 5: Generate and inspect the production artifact, document the contract, and run full verification

**Files:**
- Create: `knowledge/chunks.jsonl`
- Create: `docs/chunk-schema-v1.md`
- Modify: `tests/test_chunking_pipeline.py`

**Interfaces:**
- Consumes: checked-in `knowledge/documents.jsonl` and the completed Day 3 CLI.
- Produces: checked-in deterministic `knowledge/chunks.jsonl`, final schema documentation, and real-inventory regression coverage.

- [ ] **Step 1: Write the failing real-inventory integration test**

Add a test that loads the real 60 Documents, builds Chunks, validates every parent is represented, checks all current registries, and compares serialization with the checked-in artifact:

```python
class RealInventoryIntegrationTests(unittest.TestCase):
    def test_checked_in_documents_build_checked_in_chunks(self):
        repository_root = Path(__file__).resolve().parents[1]
        input_path = repository_root / "knowledge" / "documents.jsonl"
        output_path = repository_root / "knowledge" / "chunks.jsonl"
        documents = load_documents(input_path)
        chunks = build_chunks(documents)
        self.assertEqual(len(documents), 60)
        self.assertEqual(
            {chunk.parent_document_id for chunk in chunks},
            {document.document_id for document in documents},
        )
        self.assertEqual(output_path.read_bytes(), serialize_chunks(chunks))
```

The count 60 is a regression assertion for the current curated snapshot, not a production validation constant.

- [ ] **Step 2: Run the integration test and verify the artifact-missing failure**

Run: `python -m unittest tests.test_chunking_pipeline.RealInventoryIntegrationTests -v`

Expected: FAIL because `knowledge/chunks.jsonl` has not been generated.

- [ ] **Step 3: Generate the production Chunk artifact**

Run: `python scripts/build_knowledge_chunks.py`

Expected: exit code 0, 60 Documents processed, a positive Chunk count, all five type counts printed, and `knowledge/chunks.jsonl` created.

- [ ] **Step 4: Inspect representative output manually**

Inspect at least:

- `product:xir-p8668ex:overview`, `:features`, and every `:spec:*` Chunk.
- A long Product with multiple parameter groups.
- `solution:smart-emergency:body:business-pain-points`, confirming all H3 children remain together.
- One Support Chunk.
- `company:shengborun:company-profile`.
- `contact:shengborun:contact`.

For each sample, verify parent identity, exact factual text, Chinese `section`, ASCII ID, inherited URL/files/metadata, and absence of inferred Product applications or Product–Solution associations.

- [ ] **Step 5: Write the Chunk Schema V1 documentation**

Create `docs/chunk-schema-v1.md` covering:

- Day 2 input and Day 3 output boundary.
- Full Chunk Schema with one real Product example.
- Text versus top-level identity/provenance fields versus inherited metadata.
- Per-type production chunking rules.
- Explicit Chinese-heading-to-ASCII registry policy and failure behavior.
- 1000-character soft maximum and indivisible-unit exception.
- No-overlap and exact-factual-text rules.
- Coverage, parent, uniqueness, deterministic-order, and atomic-output validation.
- Character-based statistics.
- Build command: `python scripts/build_knowledge_chunks.py`.
- Curated snapshot synchronization responsibility: refresh Day 2 first when authoritative website knowledge changes, then rebuild Day 3.
- Strict exclusions: experimental chunkers, evaluation framework, embeddings, vector database, retrieval, reranking, LLM, and RAG generation.

- [ ] **Step 6: Run the Day 3 integration test again**

Run: `python -m unittest tests.test_chunking_pipeline.RealInventoryIntegrationTests -v`

Expected: PASS and checked-in artifact bytes exactly equal a fresh in-memory build.

- [ ] **Step 7: Run the complete repository test suite**

Run: `python -m unittest discover -s tests -v`

Expected: all existing Day 2 and new Day 3 tests PASS.

- [ ] **Step 8: Verify deterministic rebuild and capture the artifact hash**

Run the builder twice and calculate the SHA-256 after each run:

```powershell
python scripts/build_knowledge_chunks.py
$firstHash = (Get-FileHash -Algorithm SHA256 -LiteralPath 'knowledge/chunks.jsonl').Hash
python scripts/build_knowledge_chunks.py
$secondHash = (Get-FileHash -Algorithm SHA256 -LiteralPath 'knowledge/chunks.jsonl').Hash
if ($firstHash -ne $secondHash) { throw 'chunks.jsonl is not deterministic' }
$firstHash
```

Expected: the two hashes are identical.

- [ ] **Step 9: Scan the implementation for forbidden scope**

Run:

```powershell
rg -n -i "embedding|vector.?store|cosine|similarity search|semantic retrieval|rerank|llm|openai" knowledge_pipeline/chunking.py scripts/build_knowledge_chunks.py tests/test_chunking_pipeline.py
```

Expected: no implementation references. Documentation assertions that explicitly state exclusions may be reviewed separately.

- [ ] **Step 10: Review the final diff and commit the artifact and documentation**

Run: `git diff --check`

Expected: no whitespace errors.

```bash
git add knowledge/chunks.jsonl docs/chunk-schema-v1.md tests/test_chunking_pipeline.py
git commit -m "docs: publish chunk schema and deterministic artifact"
```

Record for the final handoff:

- final Chunk Schema;
- per-type rules;
- total Documents and Chunks;
- average, median, min, and max character length;
- Chunks per Document and per type;
- deterministic artifact SHA-256;
- full test command and passing count;
- assumptions and edge cases, especially unmapped headings and indivisible over-limit units;
- explicit confirmation that implementation stops before embeddings and retrieval.
