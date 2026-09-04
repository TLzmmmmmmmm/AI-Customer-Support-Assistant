"""Day 6 retrieval evaluation: preflight by default, paid queries only with --execute."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.dataset import load_cases, read_json, validate_cases
from evaluation.retrieval import run_retrieval_evaluation
from knowledge_pipeline.retrieval import (
    EmbeddingConfig, RetrievalConfig, RetrievalError, load_chunks,
    load_vector_records, validate_records_against_chunks,
)
from services.retrieval import build_retriever


class EvaluationInputError(ValueError):
    """Safe local validation messages, never provider response bodies."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_selected(root: Path, split: str):
    """Read only the requested split's file/review/fixtures, never all holdout data."""
    manifest = read_json(root / "eval/rag_v1_manifest.json")
    if manifest.get("schema_version") != "1.0" or manifest.get("dataset_version") != "rag-v1.0":
        raise EvaluationInputError("unsupported dataset manifest version")
    stem = "rag_v1_holdout" if split == "holdout" else "rag_v1"
    dataset_path = f"eval/{stem}.json"
    review_path = f"eval/{stem}_authoring.json"
    fixture_path = f"eval/fixtures/{stem}_attacks.json"
    protected = ["knowledge/documents.jsonl", "knowledge/chunks.jsonl", "knowledge/vector_records.jsonl"]
    snapshots = {}
    for relative, group in [(p, "artifact_sha256") for p in (dataset_path, review_path, fixture_path)] + [
        (p, "protected_sha256") for p in protected
    ]:
        digest = _sha256(root / relative)
        if digest != manifest[group].get(relative):
            raise EvaluationInputError(f"snapshot mismatch: {relative}")
        snapshots[relative] = digest
    cases = [case for case in load_cases(root / dataset_path) if case.split == split]
    if not cases:
        raise EvaluationInputError("selected split contains no cases")
    ids = {case.id for case in cases}
    reviews = [row for row in read_json(root / review_path) if row["case_id"] in ids]
    fixture_ids = {case.fixture_id for case in cases if case.fixture_id}
    fixtures = [row for row in read_json(root / fixture_path) if row["id"] in fixture_ids]
    documents = [json.loads(line) for line in (root / protected[0]).read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks = load_chunks(root / "knowledge/chunks.jsonl")
    validate_cases(cases, documents, [chunk.model_dump(mode="json") for chunk in chunks],
                   reviews, fixtures, set(manifest["system_rules"]))
    return cases, chunks, manifest, snapshots


def _git_state() -> dict:
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain", "--untracked-files=normal")
        return {"git_commit": commit, "git_dirty": None if status is None else bool(status)}
    except OSError:
        return {"git_commit": None, "git_dirty": None}


def _output_path(root: Path, requested: Path | None, split: str) -> Path:
    directory = (root / "eval/results").resolve()
    if requested is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = directory / f"retrieval-{split}-{timestamp}-{uuid4().hex[:8]}.json"
    else:
        path = (requested if requested.is_absolute() else root / requested).resolve()
    if not path.is_relative_to(directory) or path.suffix != ".json":
        raise EvaluationInputError("output must be a .json file inside eval/results")
    if path.exists():
        raise EvaluationInputError("output already exists; choose a new run filename")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split", choices=("dev", "frozen", "holdout"), default="dev")
    parser.add_argument("--top-k", type=int, help="Override RETRIEVAL_TOP_K (default 5)")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-holdout", action="store_true", help="Only use after Step 8 configuration freeze")
    parser.add_argument("--output", type=Path, help="New path under eval/results; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        if args.split == "holdout" and not args.allow_holdout:
            raise EvaluationInputError("holdout is locked; --allow-holdout is reserved for Step 8")
        root = args.root.resolve()
        # Read config without exporting values into process-wide production state.
        values = {key: value for key, value in dotenv_values(root / ".env").items() if value is not None}
        values.update(os.environ)
        config = EmbeddingConfig.from_mapping(values)
        top_k = RetrievalConfig.from_mapping(values).top_k if args.top_k is None else args.top_k
        if top_k <= 0:
            raise EvaluationInputError("top_k must be positive")
        cases, chunks, manifest, snapshots = _load_selected(root, args.split)
        records = load_vector_records(root / "knowledge/vector_records.jsonl")
        validate_records_against_chunks(records, chunks, config)
        calls = sum(bool(case.expected_chunk_ids) for case in cases)
        plan = {"mode": "execute" if args.execute else "preflight", "split": args.split,
                "selected_cases": len(cases), "query_embedding_calls": calls,
                "excluded_no_gold_cases": len(cases) - calls, "top_k": top_k,
                "embedding_provider": config.provider, "embedding_model": config.model,
                "embedding_dimensions": config.dimensions, "generation_called": False}
        if not args.execute:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0

        destination = _output_path(root, args.output, args.split)
        metadata = {
            "run_id": uuid4().hex, "dataset_version": manifest["dataset_version"],
            "split": args.split, "embedding_provider": config.provider,
            "embedding_model": config.model, "embedding_dimensions": config.dimensions,
            "embedding_timeout_seconds": config.timeout_seconds,
            "embedding_max_retries": config.max_retries,
            "retrieval_backend": "numpy_exact_cosine", "snapshot_sha256": snapshots,
            "manifest_sha256": _sha256(root / "eval/rag_v1_manifest.json"),
            "generation_called": False, "fixture_application": "none_retrieval_only",
            "prompt_sha256_not_used_for_retrieval": _sha256(ROOT / "prompts.py"),
            "code_sha256": {relative: _sha256(ROOT / relative) for relative in (
                "evaluation/retrieval.py", "evaluation/dataset.py", "scripts/evaluate_rag_retrieval.py",
                "services/retrieval.py", "knowledge_pipeline/retrieval/retriever.py",
                "knowledge_pipeline/retrieval/entities.py", "knowledge_pipeline/retrieval/index.py",
                "knowledge_pipeline/retrieval/embedding.py", "knowledge_pipeline/retrieval/config.py",
            )},
            **_git_state(),
        }
        retriever = build_retriever(values=values, chunks_path=root / "knowledge/chunks.jsonl",
                                    vector_records_path=root / "knowledge/vector_records.jsonl")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Reserve the destination before paid calls; x also protects against races.
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            report = run_retrieval_evaluation(cases, retriever, top_k=top_k, metadata=metadata)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": report["status"], "top_k": top_k,
                          "summary": report["summary"], "output": str(destination)},
                         ensure_ascii=False, indent=2))
        return 1 if report["status"] == "incomplete" else 0
    except EvaluationInputError as error:
        print(f"Retrieval evaluation stopped: {error}", file=sys.stderr)
        return 1
    except (RetrievalError, OSError, ValueError, KeyError, TypeError) as error:
        # Never echo raw configuration/provider exception text, even in eval.
        print(f"Retrieval evaluation stopped: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
