"""Day 6 generation evaluation: preflight by default, holdout explicitly locked."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from uuid import uuid4

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.dataset import read_json
from evaluation.generation import run_generation_cases
from knowledge_pipeline.retrieval import EmbeddingConfig, RetrievalConfig, load_vector_records, validate_records_against_chunks
from scripts.evaluate_rag_retrieval import (
    _git_state, _load_selected, _manifest_label, _resolve_manifest, _sha256, _split_paths,
    EvaluationInputError,
)
from services.retrieval import build_retriever


def _holdout_freeze_path(root: Path, manifest: dict) -> Path:
    candidate = manifest.get("candidate_snapshot_version", manifest.get("dataset_version"))
    if not isinstance(candidate, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", candidate) is None:
        raise EvaluationInputError("invalid candidate snapshot identifier")
    return root / "eval/results" / f"holdout-{candidate}-freeze.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split", choices=("dev", "frozen", "holdout"), default="dev")
    parser.add_argument("--allow-holdout", action="store_true", help="Step 8 only; freeze and run holdout once")
    parser.add_argument("--manifest", type=Path, help="Manifest .json inside eval (defaults to Day 6 manifest)")
    parser.add_argument("--case-id", action="append", dest="case_ids", help="Run only this case ID; repeatable")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, help="New .jsonl under eval/results")
    args = parser.parse_args(argv)
    try:
        if args.split == "holdout" and not args.allow_holdout:
            raise EvaluationInputError("holdout requires --allow-holdout")
        root = args.root.resolve()
        manifest_path = _resolve_manifest(root, args.manifest)
        cases, chunks, manifest, snapshots = _load_selected(root, args.split, manifest_path)
        if (args.split == "holdout" and args.execute
            and manifest.get("candidate_snapshot_version")
            and manifest.get("holdout_policy", {}).get("status") != "sealed_unexecuted"):
            raise EvaluationInputError("candidate holdout must be owner-reviewed and sealed before execution")
        if args.case_ids:
            if len(args.case_ids) != len(set(args.case_ids)):
                raise EvaluationInputError("case IDs must not be duplicated")
            available = {case.id for case in cases}
            missing = [case_id for case_id in args.case_ids if case_id not in available]
            if missing:
                raise EvaluationInputError("selected case ID is absent from the requested split")
            requested = set(args.case_ids)
            cases = [case for case in cases if case.id in requested]
        values = {key: value for key, value in dotenv_values(root / ".env").items() if value is not None}
        values.update(os.environ)
        embedding = EmbeddingConfig.from_mapping(values)
        retrieval = RetrievalConfig.from_mapping(values)
        validate_records_against_chunks(load_vector_records(root / "knowledge/vector_records.jsonl"), chunks, embedding)
        selected_fixtures = {case.fixture_id for case in cases if case.fixture_id}
        fixture_path = _split_paths(manifest, args.split)[2]
        fixtures = [row for row in read_json(root / fixture_path) if row["id"] in selected_fixtures]
        # Production configuration, never a separate generation model or parameter set.
        import config
        plan = {"mode": "execute" if args.execute else "preflight", "split": args.split,
                "case_count": len(cases), "planned_query_embeddings": len(cases),
                "planned_generations": len(cases), "synthetic_context_cases": len(selected_fixtures),
                "top_k": retrieval.top_k, "embedding_model": embedding.model,
                "embedding_dimensions": embedding.dimensions, "generation_model": config.DEEPSEEK_MODEL,
                "generation_parameters": "unchanged_production", "history_mode": "single_turn",
                "request_interval_seconds": config.RATE_LIMIT_WINDOW_SECONDS / config.RATE_LIMIT_REQUESTS + 0.1,
                "manifest": _manifest_label(root, manifest_path),
                "selected_case_ids": [case.id for case in cases]}
        if not args.execute:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0

        directory = (root / "eval/results").resolve()
        if args.output:
            destination = (args.output if args.output.is_absolute() else root / args.output).resolve()
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            destination = directory / f"generation-{args.split}-{stamp}-{uuid4().hex[:8]}.jsonl"
        if not destination.is_relative_to(directory) or destination.suffix != ".jsonl" or destination.exists():
            raise EvaluationInputError("output must be a new .jsonl file inside eval/results")
        code_paths = (
            "evaluation/generation.py", "scripts/evaluate_rag_generation.py", "scripts/evaluate_rag_retrieval.py",
            "evaluation/dataset.py", "main.py", "routes/chat.py", "services/llm.py", "services/retrieval.py",
            "prompts.py", "rag_context.py", "models.py", "config.py", "rate_limit.py", "concurrency.py",
            "app_logging.py", "error_handling.py", "knowledge_pipeline/retrieval/retriever.py",
            "knowledge_pipeline/retrieval/index.py", "knowledge_pipeline/retrieval/entities.py",
            "knowledge_pipeline/retrieval/embedding.py", "knowledge_pipeline/retrieval/config.py",
        )
        code_hashes = {path: _sha256(ROOT / path) for path in code_paths}
        manifest_hash = _sha256(manifest_path)
        metadata = {**plan, "run_id": uuid4().hex, "dataset_version": manifest["dataset_version"],
                    "rubric_version": manifest.get("rubric_version"), "snapshot_sha256": snapshots,
                    "manifest_sha256": manifest_hash, "system_rules": manifest["system_rules"],
                    "code_sha256": code_hashes, "embedding_provider": embedding.provider,
                    "embedding_timeout_seconds": embedding.timeout_seconds, "embedding_max_retries": embedding.max_retries,
                    "generation_timeout_seconds": config.DEEPSEEK_TIMEOUT_SECONDS,
                    "generation_sdk_max_retries": config.DEEPSEEK_MAX_RETRIES,
                    "generation_app_max_retries": config.LLM_APP_MAX_RETRIES,
                    "review_status": "pending_review", **_git_state()}
        directory.mkdir(parents=True, exist_ok=True)
        if args.split == "holdout":
            # Exclusive creation also blocks concurrent/repeated attempts, even
            # after interruption. Never delete this record to call results unseen.
            freeze_path = _holdout_freeze_path(root, manifest)
            if freeze_path.exists():
                raise EvaluationInputError("holdout attempt already reserved; inspect its saved results, do not rerun")
            freeze = {"schema_version": "1.0", "status": "reserved_before_api",
                      "frozen_at": datetime.now(timezone.utc).isoformat(),
                      "output": str(destination.relative_to(root)), "metadata": metadata,
                      "acceptance_policy": manifest.get("acceptance_policy"),
                      "rerun_policy": "Stop after incomplete attempt; no automatic restart or retuning"}
            with freeze_path.open("x", encoding="utf-8", newline="\n") as record:
                record.write(json.dumps(freeze, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        with destination.open("x", encoding="utf-8", newline="\n") as output:
            counts = {"selected_cases": len(cases), "attempted_cases": 0, "completed_cases": 0}
            def emit(row):
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                if row["event"] == "case":
                    counts["attempted_cases"] += 1
                    counts["completed_cases"] += int(row["status"] == "completed")
                # Evidence stays in curated evaluation storage, not terminal logs.
                print(json.dumps({key: row[key] for key in ("event", "case_id", "status", "error_type") if key in row}), flush=True)

            emit({"event": "start", "utc": datetime.now(timezone.utc).isoformat(), "metadata": metadata})
            summary = {"status": "incomplete", "error_type": None}
            try:
                summary = run_generation_cases(cases, fixtures, emit, allow_holdout=args.allow_holdout, retriever_factory=lambda: build_retriever(
                    values=values, chunks_path=root / "knowledge/chunks.jsonl",
                    vector_records_path=root / "knowledge/vector_records.jsonl"))
            except Exception as error:
                summary = {"status": "incomplete", "error_type": type(error).__name__}
            except KeyboardInterrupt:
                summary = {"status": "incomplete", "error_type": "KeyboardInterrupt"}
            finally:
                summary.update(counts)
                summary["incomplete_cases"] = counts["attempted_cases"] - counts["completed_cases"]
                summary["not_attempted_cases"] = len(cases) - counts["attempted_cases"]
                try:
                    unchanged = (all(_sha256(root / path) == digest for path, digest in snapshots.items())
                                 and _sha256(manifest_path) == manifest_hash
                                 and all(_sha256(ROOT / path) == digest for path, digest in code_hashes.items()))
                except Exception:
                    unchanged = None
                if not unchanged:
                    summary["status"] = "incomplete"
                    summary["error_type"] = "SnapshotVerificationError" if unchanged is None else "SnapshotChanged"
                emit({"event": "end", "utc": datetime.now(timezone.utc).isoformat(),
                      **summary, "snapshots_unchanged": unchanged})
        print(json.dumps({"output": str(destination), **summary}, ensure_ascii=False), flush=True)
        return 0 if summary["status"] == "completed" else 1
    except Exception as error:
        message = str(error) if isinstance(error, EvaluationInputError) else type(error).__name__
        print(f"Generation evaluation stopped: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
