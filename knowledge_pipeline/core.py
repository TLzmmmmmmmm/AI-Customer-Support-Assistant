from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

from pydantic import BaseModel, ValidationError

from .models import (
    CatalogMetadata,
    CompanyMetadata,
    CompanySource,
    ContactMetadata,
    ContactSource,
    KnowledgeDocument,
    ProductCategorySource,
    ProductMetadata,
    ProductSource,
    SolutionMetadata,
    SolutionSource,
    SupportMetadata,
    SupportSource,
)


DOCUMENT_TYPES = (
    "catalog", "product", "solution", "support", "company", "contact"
)
PRODUCT_CATEGORY_DISPLAY_ORDER = (
    "two-way-radio",
    "shortwave-radio",
    "mesh-network",
    "ict-integration",
)
MODEL_BY_DIRECTORY = {
    "products": ProductSource,
    "product-categories": ProductCategorySource,
    "solutions": SolutionSource,
    "support": SupportSource,
}
MODEL_BY_FILE = {
    "company.json": CompanySource,
    "contact.json": ContactSource,
}


class BuildError(Exception):
    def __init__(self, errors: str | Iterable[str]):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("\n\n".join(self.errors))


@dataclass
class SourceInventory:
    products: dict[str, ProductSource] = field(default_factory=dict)
    categories: dict[str, ProductCategorySource] = field(default_factory=dict)
    solutions: dict[str, SolutionSource] = field(default_factory=dict)
    support: dict[str, SupportSource] = field(default_factory=dict)
    companies: dict[str, CompanySource] = field(default_factory=dict)
    contacts: dict[str, ContactSource] = field(default_factory=dict)


def _field_path(location: tuple[object, ...]) -> str:
    result = ""
    for part in location:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            result += ("." if result else "") + str(part)
    return result


def _schema_errors(relative_file: str, error: ValidationError) -> list[str]:
    messages = []
    for item in error.errors(include_url=False):
        field_name = _field_path(item["loc"])
        messages.append(
            f"[ERROR] knowledge/source/{relative_file}\n"
            f"field: {field_name}\n"
            f"reason: {item['msg']}"
        )
    return messages


def _read_model(path: Path, root: Path, model_type: type[BaseModel]):
    relative = path.relative_to(root).as_posix()
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BuildError(
            f"[ERROR] knowledge/source/{relative}\n"
            f"reason: invalid JSON at line {error.lineno}, column {error.colno}: "
            f"{error.msg}"
        ) from error
    except OSError as error:
        raise BuildError(
            f"[ERROR] knowledge/source/{relative}\nreason: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise BuildError(
            f"[ERROR] knowledge/source/{relative}\n"
            "reason: top-level JSON value must be an object"
        )
    try:
        model = model_type.model_validate(payload)
    except ValidationError as error:
        raise BuildError(_schema_errors(relative, error)) from error
    if path.stem != model.id and path.name not in MODEL_BY_FILE:
        raise BuildError(
            f"[ERROR] knowledge/source/{relative}\n"
            f"entity: {model.id}\n"
            "reason: filename must equal entity id"
        )
    return model


def load_sources(source_root: Path) -> SourceInventory:
    if not source_root.is_dir():
        raise BuildError(f"[ERROR] {source_root}\nreason: source directory does not exist")

    inventory = SourceInventory()
    errors: list[str] = []
    destinations = {
        "products": inventory.products,
        "product-categories": inventory.categories,
        "solutions": inventory.solutions,
        "support": inventory.support,
    }

    for directory_name, model_type in MODEL_BY_DIRECTORY.items():
        directory = source_root / directory_name
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                model = _read_model(path, source_root, model_type)
                destination = destinations[directory_name]
                if model.id in destination:
                    raise BuildError(
                        f"[ERROR] knowledge/source/{path.relative_to(source_root).as_posix()}\n"
                        f"entity: {model.id}\nreason: duplicate source id"
                    )
                destination[model.id] = model
            except BuildError as error:
                errors.extend(error.errors)

    for file_name, model_type in MODEL_BY_FILE.items():
        path = source_root / file_name
        if not path.exists():
            continue
        try:
            model = _read_model(path, source_root, model_type)
            destination = inventory.companies if file_name == "company.json" else inventory.contacts
            destination[model.id] = model
        except BuildError as error:
            errors.extend(error.errors)

    if errors:
        raise BuildError(sorted(errors))
    return inventory


def _normalize_text(value: str) -> str:
    lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def _source_files(*sources) -> list[str]:
    return sorted({
        item.reference
        for source in sources
        for item in source.provenance
    })


def _source_url(base_url: str, source_path: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise BuildError(f"invalid base URL: {base_url}")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise BuildError("base URL must contain only scheme and authority")
    return base_url.rstrip("/") + source_path


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


def _document(
    *,
    type_: str,
    entity_id: str,
    title: str,
    text: str,
    source_path: str,
    source_files: list[str],
    metadata: BaseModel,
    base_url: str,
) -> KnowledgeDocument:
    normalized_text = _normalize_text(text)
    return KnowledgeDocument.model_validate({
        "schema_version": "1.0",
        "document_id": f"{type_}:{entity_id}",
        "type": type_,
        "title": title,
        "text": normalized_text,
        "language": "zh-CN",
        "source_path": source_path,
        "source_url": _source_url(base_url, source_path),
        "source_files": source_files,
        "content_hash": compute_content_hash(type_, title, normalized_text, metadata),
        "metadata": metadata.model_dump(mode="json"),
    })


def normalize_product(
    source: ProductSource,
    category: ProductCategorySource,
    base_url: str,
) -> KnowledgeDocument:
    feature_lines = "\n".join(f"- {item}" for item in source.key_features)
    parameter_groups = []
    for group in source.technical_parameters:
        items = "\n".join(f"- {item.name}：{item.value}" for item in group.items)
        parameter_groups.append(f"### {group.group}\n\n{items}")
    text = (
        f"# {source.name}\n\n"
        f"产品分类：{category.name}\n\n"
        f"## 产品介绍\n\n{source.product_features}\n\n"
        f"## 产品特点\n\n{feature_lines}\n\n"
        f"## 技术参数\n\n" + "\n\n".join(parameter_groups)
    )
    metadata = ProductMetadata(
        product_id=source.id,
        slug=source.slug,
        category_id=category.id,
        category_name=category.name,
    )
    return _document(
        type_="product",
        entity_id=source.id,
        title=source.name,
        text=text,
        source_path=source.source_path,
        source_files=_source_files(source, category),
        metadata=metadata,
        base_url=base_url,
    )


def normalize_product_catalog(
    categories: Iterable[ProductCategorySource],
    base_url: str,
) -> KnowledgeDocument:
    order = {
        category_id: index
        for index, category_id in enumerate(PRODUCT_CATEGORY_DISPLAY_ORDER)
    }
    published = sorted(
        (category for category in categories if category.published),
        key=lambda category: (order.get(category.id, len(order)), category.id),
    )
    if not published:
        raise BuildError("cannot build product catalog without published categories")
    sections = "\n\n".join(
        f"## {category.name}\n\n{category.short_description}"
        for category in published
    )
    text = (
        "# 产品分类\n\n"
        f"网站目前展示以下 {len(published)} 类产品。\n\n"
        f"{sections}"
    )
    metadata = CatalogMetadata(
        catalog_id="products",
        category_ids=sorted(category.id for category in published),
    )
    return _document(
        type_="catalog",
        entity_id="products",
        title="产品分类",
        text=text,
        source_path="/products/",
        source_files=_source_files(*published),
        metadata=metadata,
        base_url=base_url,
    )


def normalize_solution(source: SolutionSource, base_url: str) -> KnowledgeDocument:
    needs = "\n".join(f"- {item}" for item in source.core_needs)
    features = "\n".join(f"- {item}" for item in source.features)
    text = (
        f"# {source.name}\n\n"
        f"## 方案摘要\n\n{source.summary}\n\n"
        f"## 核心需求\n\n{needs}\n\n"
        f"## 方案设计\n\n{source.solution_design}\n\n"
        f"## 方案特点\n\n{features}\n\n"
        f"## 详细内容\n\n{source.body_markdown}"
    )
    metadata = SolutionMetadata(solution_id=source.id, slug=source.slug)
    return _document(
        type_="solution",
        entity_id=source.id,
        title=source.name,
        text=text,
        source_path=source.source_path,
        source_files=_source_files(source),
        metadata=metadata,
        base_url=base_url,
    )


def normalize_support(source: SupportSource, base_url: str) -> KnowledgeDocument:
    text = (
        f"# {source.name}\n\n"
        f"## 服务摘要\n\n{source.summary}\n\n"
        f"## 服务内容\n\n{source.body}"
    )
    metadata = SupportMetadata(service_id=source.id)
    return _document(
        type_="support",
        entity_id=source.id,
        title=source.name,
        text=text,
        source_path=source.source_path,
        source_files=_source_files(source),
        metadata=metadata,
        base_url=base_url,
    )


def normalize_company(source: CompanySource, base_url: str) -> KnowledgeDocument:
    text = f"# {source.name}\n\n## 公司简介\n\n" + "\n\n".join(source.introduction)
    metadata = CompanyMetadata(company_id=source.id)
    return _document(
        type_="company",
        entity_id=source.id,
        title=source.name,
        text=text,
        source_path=source.source_path,
        source_files=_source_files(source),
        metadata=metadata,
        base_url=base_url,
    )


def normalize_contact(source: ContactSource, base_url: str) -> KnowledgeDocument:
    text = (
        "# 联系我们\n\n"
        f"公司名称：{source.company_name}\n"
        f"值班电话：{source.duty_phone}\n"
        f"电子邮箱：{source.email}"
    )
    metadata = ContactMetadata(contact_id=source.id)
    return _document(
        type_="contact",
        entity_id=source.id,
        title="联系我们",
        text=text,
        source_path=source.source_path,
        source_files=_source_files(source),
        metadata=metadata,
        base_url=base_url,
    )


def _validate_source_relationships(inventory: SourceInventory) -> None:
    errors: list[str] = []
    for source in inventory.products.values():
        if not source.published:
            continue
        category = inventory.categories.get(source.category_id)
        if category is None or not category.published:
            errors.append(
                f"[ERROR] knowledge/source/products/{source.id}.json\n"
                f"entity: product:{source.id}\nfield: category_id\n"
                f"reason: referenced published category not found: {source.category_id}"
            )
            continue
        expected = f"/{category.slug}/{source.slug}/"
        if source.source_path != expected:
            errors.append(
                f"[ERROR] product:{source.id}\nfield: source_path\n"
                f"reason: expected {expected}, got {source.source_path}"
            )
    for source in inventory.solutions.values():
        expected = f"/solutions/{source.slug}/"
        if source.published and source.source_path != expected:
            errors.append(
                f"[ERROR] solution:{source.id}\nfield: source_path\n"
                f"reason: expected {expected}, got {source.source_path}"
            )
    for source in inventory.support.values():
        expected = f"/support/#{source.id}"
        if source.published and source.source_path != expected:
            errors.append(
                f"[ERROR] support:{source.id}\nfield: source_path\n"
                f"reason: expected {expected}, got {source.source_path}"
            )
    for source in inventory.companies.values():
        if source.published and source.source_path != "/about/#company":
            errors.append(
                f"[ERROR] company:{source.id}\nfield: source_path\n"
                "reason: expected /about/#company"
            )
    for source in inventory.contacts.values():
        if source.published and source.source_path != "/about/#contact":
            errors.append(
                f"[ERROR] contact:{source.id}\nfield: source_path\n"
                "reason: expected /about/#contact"
            )
    if errors:
        raise BuildError(sorted(errors))


def load_source_inventory(source_root: Path) -> SourceInventory:
    """Load authoritative structured sources with relationship validation."""

    inventory = load_sources(source_root)
    _validate_source_relationships(inventory)
    return inventory


def build_documents(source_root: Path, base_url: str) -> list[KnowledgeDocument]:
    inventory = load_source_inventory(source_root)
    documents: list[KnowledgeDocument] = []
    documents.append(normalize_product_catalog(inventory.categories.values(), base_url))
    for source in inventory.products.values():
        if source.published:
            documents.append(
                normalize_product(source, inventory.categories[source.category_id], base_url)
            )
    documents.extend(
        normalize_solution(source, base_url)
        for source in inventory.solutions.values()
        if source.published
    )
    documents.extend(
        normalize_support(source, base_url)
        for source in inventory.support.values()
        if source.published
    )
    documents.extend(
        normalize_company(source, base_url)
        for source in inventory.companies.values()
        if source.published
    )
    documents.extend(
        normalize_contact(source, base_url)
        for source in inventory.contacts.values()
        if source.published
    )
    validate_documents(documents, base_url)
    return sorted(documents, key=lambda item: item.document_id)


def validate_documents(
    documents: list[KnowledgeDocument],
    base_url: str,
) -> None:
    errors: list[str] = []
    ids: dict[str, list[str]] = {}
    hashes: dict[str, list[str]] = {}
    counts = {item: 0 for item in DOCUMENT_TYPES}
    for document in documents:
        ids.setdefault(document.document_id, []).append(document.document_id)
        hashes.setdefault(document.content_hash, []).append(document.document_id)
        counts[document.type] += 1
        expected_url = _source_url(base_url, document.source_path)
        if document.source_url != expected_url:
            errors.append(
                f"[ERROR] {document.document_id}\nfield: source_url\n"
                f"reason: expected {expected_url}, got {document.source_url}"
            )
        expected_hash = compute_content_hash(
            document.type,
            document.title,
            document.text,
            document.metadata,
        )
        if document.content_hash != expected_hash:
            errors.append(
                f"[ERROR] {document.document_id}\nfield: content_hash\n"
                "reason: does not match normalized semantic content"
            )
    for document_id, matches in ids.items():
        if len(matches) > 1:
            errors.append(f"[ERROR] duplicate document_id: {document_id}")
    for content_hash, document_ids in hashes.items():
        if len(document_ids) > 1:
            errors.append(
                f"[ERROR] duplicate content_hash {content_hash}: "
                + ", ".join(sorted(document_ids))
            )
    for type_, count in counts.items():
        if count == 0:
            errors.append(f"[ERROR] required document type has no documents: {type_}")
    if errors:
        raise BuildError(sorted(errors))


def serialize_documents(documents: list[KnowledgeDocument]) -> bytes:
    lines = [
        json.dumps(
            document.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for document in sorted(documents, key=lambda item: item.document_id)
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_documents(
    output_path: Path,
    documents: list[KnowledgeDocument],
    base_url: str,
) -> None:
    validate_documents(documents, base_url)
    payload = serialize_documents(documents)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        parsed = [
            KnowledgeDocument.model_validate(json.loads(line))
            for line in temp_path.read_text(encoding="utf-8").splitlines()
        ]
        validate_documents(parsed, base_url)
        if serialize_documents(parsed) != payload:
            raise BuildError("temporary JSONL verification was not byte deterministic")
        os.replace(temp_path, output_path)
        temp_path = None
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise BuildError(f"failed to write {output_path}: {error}") from error
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def build_and_write(
    source_root: Path,
    output_path: Path,
    base_url: str,
) -> dict[str, int]:
    documents = build_documents(source_root, base_url)
    write_documents(output_path, documents, base_url)
    return {
        type_: sum(document.type == type_ for document in documents)
        for type_ in DOCUMENT_TYPES
    }
