"""Day 6 case contract and offline integrity checks (no model calls or metrics).

Evidence excerpts are checked for existence, not semantic entailment. The
separate authoring notes explain why they support each expected answer.
"""

from __future__ import annotations

import ast
import hashlib
import json
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    category: Literal["product_spec", "product_recommendation", "solution",
                      "support", "company", "unknown", "adversarial"]
    split: Literal["frozen", "dev", "holdout"]
    question: str
    expected_behavior: Literal["answer", "partial_answer", "abstain", "clarify"]
    expected_answer: str
    expected_document_ids: list[str]
    expected_chunk_ids: list[str]
    expected_system_rule_ids: list[str]
    required_facts: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    fixture_id: str | None = None

    @field_validator("id", "question", "expected_answer", "fixture_id")
    @classmethod
    def nonblank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        # Never rewrite historical questions, even their whitespace.
        return value

    @field_validator("expected_document_ids", "expected_chunk_ids",
                     "expected_system_rule_ids", "required_facts", "forbidden_claims")
    @classmethod
    def distinct_nonblank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values) or len(set(values)) != len(values):
            raise ValueError("list entries must be nonblank and unique")
        return values


class CaseFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["1.0"]
    dataset_version: Literal["rag-v1.0"]
    cases: list[EvaluationCase] = Field(min_length=1)


DIMENSIONS = {
    "naturalness": {"很人工", "尚可", "很像真实用户"},
    "ground_truth": {"模糊", "部分明确", "非常明确"},
    "retrieval_value": {"纯关键词", "有少量改写", "真正语义改写"},
    "evidence_alignment": {"不支持", "部分支持", "明确支持"},
    "diagnostic_value": {"很难归因", "尚可", "明确测某能力"},
    "ambiguity": {"很歧义", "有一点", "几乎无歧义"},
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases(path: Path) -> list[EvaluationCase]:
    return CaseFile.model_validate(read_json(path)).cases


def normalized_question(question: str) -> str:
    return "".join(unicodedata.normalize("NFKC", question).casefold().split())


def unique_index(rows: Iterable[dict], key: str) -> dict[str, dict]:
    index = {}
    for row in rows:
        identity = row[key]
        if identity in index:
            raise ValueError(f"duplicate {key}: {identity}")
        index[identity] = row
    return index


def validate_cases(
    cases: list[EvaluationCase],
    documents: list[dict],
    chunks: list[dict],
    reviews: list[dict],
    fixtures: list[dict],
    system_rule_ids: set[str],
) -> None:
    """Validate labels and evidence links, without executing fixtures/retrieval."""
    unique_index([case.model_dump() for case in cases], "id")
    questions = [normalized_question(case.question) for case in cases]
    if len(set(questions)) != len(questions):
        raise ValueError("duplicate normalized question")
    docs = unique_index(documents, "document_id")
    chunk_map = unique_index(chunks, "chunk_id")
    review_map = unique_index(reviews, "case_id")
    fixture_map = unique_index(fixtures, "id")
    if set(review_map) != {case.id for case in cases}:
        raise ValueError("review coverage must match case IDs exactly")
    if set(fixture_map) != {case.fixture_id for case in cases if case.fixture_id}:
        raise ValueError("fixture coverage must match referenced fixtures")

    for case in cases:
        for ids, index, kind in (
            (case.expected_document_ids, docs, "document"),
            (case.expected_chunk_ids, chunk_map, "chunk"),
            (case.expected_system_rule_ids, system_rule_ids, "system rule"),
        ):
            if missing := set(ids) - set(index):
                raise ValueError(f"{case.id}: missing {kind}: {sorted(missing)}")
        parents = {chunk_map[c]["parent_document_id"] for c in case.expected_chunk_ids}
        if parents != set(case.expected_document_ids):
            raise ValueError(f"{case.id}: expected document IDs must match chunk parents")
        if case.expected_behavior in {"answer", "partial_answer"} and not (
            case.expected_chunk_ids or case.expected_system_rule_ids
        ):
            raise ValueError(f"{case.id}: answer needs explicit grounding")

        review = review_map[case.id]
        dimensions = review.get("dimensions", {})
        if set(dimensions) != set(DIMENSIONS) or any(
            value not in DIMENSIONS[name] for name, value in dimensions.items()
        ):
            raise ValueError(f"{case.id}: invalid review dimension")
        for field in ("diagnostic_note", "origin", "annotation_note"):
            if not isinstance(review.get(field), str) or not review[field].strip():
                raise ValueError(f"{case.id}: review needs {field}")
        if case.expected_behavior != "answer" and not review.get("unavailable_reason", "").strip():
            raise ValueError(f"{case.id}: needs unavailable_reason")
        evidence = review.get("evidence", [])
        if {item["chunk_id"] for item in evidence} != set(case.expected_chunk_ids):
            raise ValueError(f"{case.id}: evidence must cover exactly the expected chunks")
        for item in evidence:
            source = chunk_map[item["chunk_id"]]
            quote = item["quote"]
            if not isinstance(quote, str) or not quote.strip() or quote not in source["text"]:
                raise ValueError(f"{case.id}: evidence quote absent from chunk")
            if quote not in docs[source["parent_document_id"]]["text"]:
                raise ValueError(f"{case.id}: evidence quote absent from parent document")

        if case.fixture_id:
            fixture = fixture_map[case.fixture_id]
            if (case.category != "adversarial" or fixture.get("synthetic") is not True
                or fixture.get("mode") != "append_to_chunk_text"
                or fixture.get("split") != case.split
                or fixture.get("target_chunk_id") not in case.expected_chunk_ids
                or not isinstance(fixture.get("text"), str) or not fixture["text"].strip()):
                raise ValueError(f"{case.id}: invalid synthetic fixture")


ARTIFACT_FILES = {
    "eval/rag_v1.json", "eval/rag_v1_holdout.json",
    "eval/rag_v1_authoring.json", "eval/rag_v1_holdout_authoring.json",
    "eval/fixtures/rag_v1_attacks.json", "eval/fixtures/rag_v1_holdout_attacks.json",
}
PROTECTED_FILES = {
    "eval/baseline_v0.json", "eval/baseline_v0_results.json",
    "knowledge/documents.jsonl", "knowledge/chunks.jsonl",
    "knowledge/vector_records.jsonl", "scripts/day5_abstention_eval.py",
}


def validate_bundle(root: Path) -> dict:
    """Read and validate the entire authored bundle, including holdout labels.

    This is NOT a retrieval/holdout run. Future runners should load only their
    explicitly requested case file/split; this function never invokes providers.
    SHA seals detect accidental drift, not deliberate tampering with the seal.
    """
    manifest = read_json(root / "eval/rag_v1_manifest.json")
    if (manifest.get("schema_version") != "1.0"
        or manifest.get("dataset_version") != "rag-v1.0"
        or set(manifest.get("artifact_sha256", {})) != ARTIFACT_FILES
        or set(manifest.get("protected_sha256", {})) != PROTECTED_FILES
        or manifest.get("prior_question_sources") != ["scripts/day5_abstention_eval.py"]):
        raise ValueError("invalid or incomplete dataset manifest")
    for relative, expected in {
        **manifest["artifact_sha256"], **manifest["protected_sha256"]
    }.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"snapshot mismatch: {relative}")

    main = load_cases(root / "eval/rag_v1.json")
    holdout = load_cases(root / "eval/rag_v1_holdout.json")
    if any(case.split == "holdout" for case in main) or any(
        case.split != "holdout" for case in holdout
    ):
        raise ValueError("incorrect split placement in case files")
    cases = main + holdout
    counts = dict(Counter(case.split for case in cases))
    if counts != {"frozen": 20, "dev": 30, "holdout": 10} or counts != manifest["split_counts"]:
        raise ValueError("split counts must be frozen=20, dev=30, holdout=10")
    categories = Counter(case.category for case in cases)
    if set(categories) != {"product_spec", "product_recommendation", "solution",
                          "support", "company", "unknown", "adversarial"}:
        raise ValueError("dataset must cover all seven approved categories")

    baseline = read_json(root / "eval/baseline_v0.json")["cases"]
    frozen = {case.id: case.question for case in cases if case.split == "frozen"}
    if len(baseline) != 20 or frozen != {row["id"]: row["question"] for row in baseline}:
        raise ValueError("frozen question IDs/text must match historical V0 exactly")
    if any(row["v0"]["status"] != "completed" or not row["v0"]["answer"] for row in baseline):
        raise ValueError("historical V0 raw answers must be preserved")

    def jsonl(relative):
        return [json.loads(line) for line in (root / relative).read_text(encoding="utf-8").splitlines() if line.strip()]

    reviews = read_json(root / "eval/rag_v1_authoring.json") + read_json(root / "eval/rag_v1_holdout_authoring.json")
    fixtures = read_json(root / "eval/fixtures/rag_v1_attacks.json") + read_json(root / "eval/fixtures/rag_v1_holdout_attacks.json")
    validate_cases(cases, jsonl("knowledge/documents.jsonl"), jsonl("knowledge/chunks.jsonl"),
                   reviews, fixtures, set(manifest["system_rules"]))

    # Parse old literal (case_name, question) tuples without importing live scripts.
    # This avoids .env loading, provider construction and accidental paid calls.
    tree = ast.parse((root / "scripts/day5_abstention_eval.py").read_text(encoding="utf-8"))
    previous = {
        normalized_question(node.elts[1].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Tuple) and len(node.elts) == 2
        and all(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in node.elts)
    }
    if any(normalized_question(case.question) in previous for case in holdout):
        raise ValueError("holdout contains a previously used Day 5 question")

    return {
        "dataset_version": manifest["dataset_version"],
        "total_cases": len(cases), "splits": counts,
        "categories": dict(sorted(categories.items())),
        "behaviors": dict(sorted(Counter(case.expected_behavior for case in cases).items())),
        "retrieval_labeled_cases": sum(bool(case.expected_chunk_ids) for case in cases),
        "no_retrieval_gold_cases": sum(not case.expected_chunk_ids for case in cases),
        "synthetic_fixture_cases": len(fixtures),
        "authoring_reviews": len(reviews), "historical_raw_answers": len(baseline),
        "prior_day5_unique_questions_checked": len(previous),
        "semantic_review": "assistant evidence review; owner review pending",
        "execution": "offline validation only; no retrieval, generation or holdout run",
    }
