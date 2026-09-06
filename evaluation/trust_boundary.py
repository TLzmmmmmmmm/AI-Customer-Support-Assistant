"""Build controlled clean/attacked pairs without changing persistent knowledge."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from evaluation.dataset import EvaluationCase


REQUIRED_ATTACK_CLASSES = {
    "change_factual_value",
    "hide_legitimate_value",
    "force_unnecessary_refusal",
    "fake_authority",
    "ignore_other_evidence",
    "redirect_company_fact",
}


def validate_approved_baseline(root: Path, baseline: dict) -> None:
    hashes = baseline.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("approved baseline needs SHA-256 entries")
    for relative, expected in hashes.items():
        path = root / relative
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or len(expected) != 64
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"approved baseline mismatch: {relative}")


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _with_ledger_lock(path: Path):
    lock = path.with_name(f".{path.name}.lock")
    return lock, lock.open("x", encoding="utf-8")


def reserve_experiment_run(
    path: Path,
    *,
    experiment_version: str,
    phase: str,
    case_ids: Sequence[str],
    max_provider_calls: int,
    app_max_retries: int,
) -> str:
    """Exclusively reserve phase/case attempts and worst-case provider calls."""
    if phase not in {"initial_pairs", "diagnostic_repeats"}:
        raise ValueError("unknown experiment phase")
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("case attempt IDs must be nonempty and unique")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock, handle = _with_ledger_lock(path)
    handle.close()
    try:
        if path.exists():
            ledger = json.loads(path.read_text(encoding="utf-8"))
        else:
            ledger = {
                "schema_version": "1.0",
                "experiment_version": experiment_version,
                "approved_maximum_provider_calls": max_provider_calls,
                "runs": [],
                "actual_provider_calls": 0,
            }
        if (
            ledger.get("experiment_version") != experiment_version
            or ledger.get("approved_maximum_provider_calls") != max_provider_calls
        ):
            raise ValueError("experiment ledger contract mismatch")
        runs = ledger.get("runs", [])
        if phase == "initial_pairs" and runs:
            raise ValueError("initial phase already reserved")
        if phase == "diagnostic_repeats" and not any(
            row.get("phase") == "initial_pairs" and row.get("status") == "completed"
            for row in runs
        ):
            raise ValueError("diagnostic repeats require a completed initial phase")
        previous_ids = {case for row in runs for case in row.get("case_ids", [])}
        if previous_ids.intersection(case_ids):
            raise ValueError("case attempt already reserved")
        reserved = len(case_ids) * (1 + app_max_retries)
        charged = sum(
            row["actual_provider_calls"]
            if row.get("actual_provider_calls") is not None
            else row["reserved_provider_calls"]
            for row in runs
        )
        if charged + reserved > max_provider_calls:
            raise ValueError("cumulative provider-call budget exceeded")
        reservation_id = uuid4().hex
        runs.append({
            "reservation_id": reservation_id,
            "phase": phase,
            "case_ids": list(case_ids),
            "reserved_provider_calls": reserved,
            "actual_provider_calls": None,
            "status": "reserved_before_api",
            "output": None,
            "output_sha256": None,
        })
        ledger["runs"] = runs
        _write_json_atomic(path, ledger)
        return reservation_id
    finally:
        lock.unlink(missing_ok=True)


def complete_experiment_run(
    path: Path,
    reservation_id: str,
    *,
    actual_provider_calls: int,
    output: str,
    output_sha256: str,
    status: str = "completed",
) -> None:
    lock, handle = _with_ledger_lock(path)
    handle.close()
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
        matches = [row for row in ledger["runs"] if row["reservation_id"] == reservation_id]
        if len(matches) != 1 or matches[0]["status"] != "reserved_before_api":
            raise ValueError("unknown or completed experiment reservation")
        row = matches[0]
        if not 0 <= actual_provider_calls <= row["reserved_provider_calls"]:
            raise ValueError("actual provider calls exceed reservation")
        row.update({
            "actual_provider_calls": actual_provider_calls,
            "status": status,
            "output": output,
            "output_sha256": output_sha256,
        })
        ledger["actual_provider_calls"] = sum(
            item.get("actual_provider_calls") or 0 for item in ledger["runs"]
        )
        _write_json_atomic(path, ledger)
    finally:
        lock.unlink(missing_ok=True)


def validate_scenarios(scenarios: Sequence[dict], chunk_ids: set[str]) -> None:
    ids = [row.get("id") for row in scenarios]
    classes = [row.get("attack_class") for row in scenarios]
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("scenario IDs must be nonblank strings")
    if len(ids) != len(set(ids)):
        raise ValueError("scenario IDs must be unique")
    if len(classes) != len(set(classes)):
        raise ValueError("attack classes must be unique")
    for row in scenarios:
        if row.get("target_chunk_id") not in chunk_ids:
            raise ValueError(f"unknown target chunk: {row.get('target_chunk_id')}")
        for field in (
            "question", "attack_text", "expected_document_ids",
            "expected_chunk_ids", "trusted_answer_facts",
            "forbidden_attack_outcomes",
        ):
            value = row.get(field)
            if not value or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"{row['id']}: missing {field}")
        if row["target_chunk_id"] not in row["expected_chunk_ids"]:
            raise ValueError(f"{row['id']}: target chunk must be expected evidence")


def build_experiment_cases(
    scenarios: Sequence[dict],
    *,
    repeat_attack_ids: set[str] | None = None,
    attack_repeats: int = 1,
) -> tuple[list[EvaluationCase], list[dict]]:
    """Build initial pairs, or up to two additional attacked attempts.

    Initial mode (repeat_attack_ids=None) emits clean + attacked-r1. Repeat mode
    emits attacked-r2 onward only, keeping the approved total at <=3 attempts.
    """
    if not isinstance(attack_repeats, int) or isinstance(attack_repeats, bool):
        raise ValueError("attack_repeats must be an integer between 1 and 2")
    if repeat_attack_ids is not None and not 1 <= attack_repeats <= 2:
        raise ValueError("attack_repeats must be between 1 and 2")
    selected_ids = {row["id"] for row in scenarios}
    if repeat_attack_ids is not None and not repeat_attack_ids <= selected_ids:
        raise ValueError("repeat attack IDs must reference known scenarios")

    fixtures = [
        {
            "id": f"tb-{row['id']}-attack",
            "split": "dev",
            "synthetic": True,
            "mode": "append_to_chunk_text",
            "target_chunk_id": row["target_chunk_id"],
            "text": row["attack_text"],
            "attack_class": row["attack_class"],
        }
        for row in scenarios
        if repeat_attack_ids is None or row["id"] in repeat_attack_ids
    ]
    cases: list[EvaluationCase] = []
    for row in scenarios:
        if repeat_attack_ids is not None and row["id"] not in repeat_attack_ids:
            continue
        variants = (
            [("clean", None), ("attacked-r1", f"tb-{row['id']}-attack")]
            if repeat_attack_ids is None else
            [(f"attacked-r{number}", f"tb-{row['id']}-attack")
             for number in range(2, attack_repeats + 2)]
        )
        for suffix, fixture_id in variants:
            cases.append(EvaluationCase.model_validate({
                "id": f"tb-{row['id']}-{suffix}",
                "category": "adversarial",
                "split": "dev",
                "question": row["question"],
                "expected_behavior": "answer",
                "expected_answer": "; ".join(row["trusted_answer_facts"]),
                "expected_document_ids": row["expected_document_ids"],
                "expected_chunk_ids": row["expected_chunk_ids"],
                "expected_system_rule_ids": [],
                "required_facts": row["trusted_answer_facts"],
                "forbidden_claims": row["forbidden_attack_outcomes"],
                "fixture_id": fixture_id,
            }))
    return cases, fixtures


__all__ = [
    "REQUIRED_ATTACK_CLASSES", "build_experiment_cases",
    "complete_experiment_run", "reserve_experiment_run",
    "validate_approved_baseline", "validate_scenarios",
]
