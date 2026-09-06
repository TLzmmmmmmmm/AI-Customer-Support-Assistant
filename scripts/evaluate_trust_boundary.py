"""Day 7 paired trust-boundary experiment; preflight unless --execute."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.dataset import read_json
from evaluation.generation import run_generation_cases
from evaluation.trust_boundary import (
    REQUIRED_ATTACK_CLASSES,
    build_experiment_cases,
    complete_experiment_run,
    reserve_experiment_run,
    validate_approved_baseline,
    validate_scenarios,
)
from knowledge_pipeline.retrieval import EmbeddingConfig, RetrievalConfig, load_chunks
from scripts.evaluate_rag_retrieval import EvaluationInputError, _git_state, _sha256
from services.retrieval import build_retriever


SNAPSHOT_PATHS = (
    "prompts.py",
    "routes/chat.py",
    "rag_context.py",
    "services/llm.py",
    "services/retrieval.py",
    "knowledge/documents.jsonl",
    "knowledge/chunks.jsonl",
    "knowledge/vector_records.jsonl",
    "eval/trust_boundary_v1.json",
    "eval/trust_boundary_v1_baseline.json",
    "evaluation/generation.py",
    "evaluation/trust_boundary.py",
    "scripts/evaluate_trust_boundary.py",
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeat-scenario", action="append", default=[])
    parser.add_argument("--additional-repeats", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        design = read_json(root / "eval/trust_boundary_v1.json")
        approved_baseline = read_json(root / "eval/trust_boundary_v1_baseline.json")
        validate_approved_baseline(root, approved_baseline)
        snapshot_paths = tuple(dict.fromkeys(
            (*SNAPSHOT_PATHS, *approved_baseline["sha256"].keys())
        ))
        scenarios = design.get("scenarios", [])
        chunks = load_chunks(root / "knowledge/chunks.jsonl")
        validate_scenarios(scenarios, {chunk.chunk_id for chunk in chunks})
        classes = {row["attack_class"] for row in scenarios}
        if classes != REQUIRED_ATTACK_CLASSES:
            raise EvaluationInputError("experiment must cover the six approved attack classes exactly")
        repeats = set(args.repeat_scenario) if args.repeat_scenario else None
        cases, fixtures = build_experiment_cases(
            scenarios,
            repeat_attack_ids=repeats,
            attack_repeats=args.additional_repeats,
        )
        if repeats is None and len(cases) != design["approved_initial_generation_calls"]:
            raise EvaluationInputError("initial call count differs from owner-approved plan")
        if len(cases) > design["approved_maximum_generation_calls"]:
            raise EvaluationInputError("planned calls exceed owner-approved maximum")

        values = {key: value for key, value in dotenv_values(root / ".env").items() if value is not None}
        values.update(os.environ)
        embedding = EmbeddingConfig.from_mapping(values)
        retrieval = RetrievalConfig.from_mapping(values)
        import config
        plan = {
            "mode": "execute" if args.execute else "preflight",
            "experiment_version": design["experiment_version"],
            "phase": "initial_pairs" if repeats is None else "diagnostic_repeats",
            "case_count": len(cases),
            "planned_query_embeddings": len(cases),
            "planned_generations": len(cases),
            "clean_cases": sum(case.fixture_id is None for case in cases),
            "attacked_cases": sum(case.fixture_id is not None for case in cases),
            "attack_classes": sorted(classes if repeats is None else {
                row["attack_class"] for row in scenarios if row["id"] in repeats
            }),
            "top_k": retrieval.top_k,
            "embedding_model": embedding.model,
            "embedding_dimensions": embedding.dimensions,
            "generation_model": config.DEEPSEEK_MODEL,
            "generation_parameters": "unchanged_production",
            "request_interval_seconds": config.RATE_LIMIT_WINDOW_SECONDS / config.RATE_LIMIT_REQUESTS + 0.1,
        }
        if not args.execute:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0

        directory = (root / "eval/results").resolve()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = ((args.output if args.output and args.output.is_absolute()
                        else root / args.output) if args.output else
                       directory / f"trust-boundary-{plan['phase']}-{stamp}-{uuid4().hex[:8]}.jsonl").resolve()
        if not destination.is_relative_to(directory) or destination.suffix != ".jsonl" or destination.exists():
            raise EvaluationInputError("output must be a new .jsonl inside eval/results")
        ledger_path = directory / "trust-boundary-day7-step2-ledger.json"
        reservation_id = reserve_experiment_run(
            ledger_path,
            experiment_version=design["experiment_version"],
            phase=plan["phase"],
            case_ids=[case.id for case in cases],
            max_provider_calls=design["approved_maximum_generation_calls"],
            app_max_retries=config.LLM_APP_MAX_RETRIES,
        )
        snapshots = {path: _sha256(root / path) for path in snapshot_paths}
        metadata = {
            **plan,
            "run_id": uuid4().hex,
            "review_status": "pending_review",
            "scenario_contract": scenarios,
            "snapshot_sha256": snapshots,
            "embedding_provider": embedding.provider,
            "generation_timeout_seconds": config.DEEPSEEK_TIMEOUT_SECONDS,
            "generation_sdk_max_retries": config.DEEPSEEK_MAX_RETRIES,
            "generation_app_max_retries": config.LLM_APP_MAX_RETRIES,
            **_git_state(),
        }
        directory.mkdir(parents=True, exist_ok=True)
        actual_provider_calls = 0
        with destination.open("x", encoding="utf-8", newline="\n") as output:
            counts = {"selected_cases": len(cases), "attempted_cases": 0, "completed_cases": 0}

            def emit(row):
                nonlocal actual_provider_calls
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                if row["event"] == "case":
                    actual_provider_calls += len(row["provider_requests"])
                    counts["attempted_cases"] += 1
                    counts["completed_cases"] += int(row["status"] == "completed")
                print(json.dumps({key: row[key] for key in ("event", "case_id", "status", "error_type") if key in row}), flush=True)

            emit({"event": "start", "utc": datetime.now(timezone.utc).isoformat(), "metadata": metadata})
            summary = {"status": "incomplete", "error_type": None}
            try:
                summary = run_generation_cases(
                    cases,
                    fixtures,
                    emit,
                    retriever_factory=lambda: build_retriever(
                        values=values,
                        chunks_path=root / "knowledge/chunks.jsonl",
                        vector_records_path=root / "knowledge/vector_records.jsonl",
                    ),
                )
            except Exception as error:
                summary = {"status": "incomplete", "error_type": type(error).__name__}
            except KeyboardInterrupt:
                summary = {"status": "incomplete", "error_type": "KeyboardInterrupt"}
            finally:
                summary.update(counts)
                summary["incomplete_cases"] = counts["attempted_cases"] - counts["completed_cases"]
                summary["not_attempted_cases"] = len(cases) - counts["attempted_cases"]
                try:
                    unchanged = all(_sha256(root / path) == digest for path, digest in snapshots.items())
                except Exception:
                    unchanged = None
                if not unchanged:
                    summary["status"] = "incomplete"
                    summary["error_type"] = "SnapshotVerificationError" if unchanged is None else "SnapshotChanged"
                emit({"event": "end", "utc": datetime.now(timezone.utc).isoformat(), **summary,
                      "snapshots_unchanged": unchanged})
        complete_experiment_run(
            ledger_path,
            reservation_id,
            actual_provider_calls=actual_provider_calls,
            output=str(destination.relative_to(root)).replace("\\", "/"),
            output_sha256=_sha256(destination),
            status=summary["status"],
        )
        print(json.dumps({"output": str(destination), **summary}, ensure_ascii=False), flush=True)
        return 0 if summary["status"] == "completed" else 1
    except Exception as error:
        message = str(error) if isinstance(error, (EvaluationInputError, ValueError)) else type(error).__name__
        print(f"Trust-boundary evaluation stopped: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
