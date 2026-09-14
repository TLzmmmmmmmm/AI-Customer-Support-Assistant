"""Pure helpers for Week 4 Day 2 performance analysis."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any


_REQUIRED_SUMMARY_FIELDS = frozenset({
    "request_id",
    "http_status",
    "outcome",
    "total_latency_ms",
})
_INTEGER_FIELDS = frozenset({
    "http_status",
    "tool_call_count",
    "retrieved_count",
    "tool_execution_count",
    "input_tokens",
    "output_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "citation_count",
    "deduplicated_citation_count",
    "invalid_source_count",
})
_FLOAT_FIELDS = frozenset({
    "latency_ms",
    "total_latency_ms",
    "router_latency_ms",
    "retrieval_latency_ms",
    "tool_latency_ms",
    "model_latency_ms",
})
_JSON_FIELDS = frozenset({
    "tool_calls",
    "retrieved_chunk_ids",
    "retrieval_used",
    "executed_tool_names",
    "tool_execution_success",
})
_NULLABLE_STRING_FIELDS = frozenset({
    "error",
    "route",
    "router_type",
    "failure_layer",
    "failure_code",
})


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("timestamp is invalid") from error
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


def _parse_field(name: str, value: str) -> object:
    if value == "null":
        return None
    if name in _INTEGER_FIELDS:
        return int(value)
    if name in _FLOAT_FIELDS:
        return float(value)
    if name in _JSON_FIELDS:
        return json.loads(value)
    if name == "answer_sanitized":
        if value in {"true", "True"}:
            return True
        if value in {"false", "False"}:
            return False
        raise ValueError("answer_sanitized is invalid")
    if name in _NULLABLE_STRING_FIELDS and value == "-":
        return None
    return value


def _validate_summary(summary: dict[str, object]) -> dict[str, object]:
    missing = sorted(_REQUIRED_SUMMARY_FIELDS.difference(summary))
    if missing:
        raise ValueError(f"missing required fields: {','.join(missing)}")
    request_id = summary["request_id"]
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id is invalid")
    return summary


def parse_summary_line(line: str) -> dict[str, object] | None:
    if "request_id=" not in line:
        return None
    parts = line.strip().split()
    if not parts:
        return None
    summary: dict[str, object] = {"timestamp": _timestamp(parts[0])}
    for part in parts[1:]:
        if "=" not in part:
            continue
        name, raw_value = part.split("=", 1)
        if name == "level":
            continue
        try:
            summary[name] = _parse_field(name, raw_value)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid field: {name}") from error
    return _validate_summary(summary)


def _exported_summary(line: str) -> dict[str, object]:
    try:
        summary = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError("invalid exported JSON") from error
    if not isinstance(summary, dict):
        raise ValueError("exported summary must be an object")
    summary["timestamp"] = _timestamp(summary.get("timestamp"))
    return _validate_summary(summary)


def load_summaries(
    path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summaries = []
    issues = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            summary = (
                _exported_summary(stripped)
                if stripped.startswith("{")
                else parse_summary_line(stripped)
            )
        except ValueError as error:
            issues.append({
                "line_number": line_number,
                "reason": str(error),
            })
            continue
        if summary is not None:
            summaries.append(summary)
    return summaries, issues


def load_workload_rows(
    path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    header = None
    rows = []
    workload_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid workload JSON at line {line_number}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"workload line {line_number} is not an object")
        if record.get("record_type") == "run":
            if header is not None:
                raise ValueError("workload contains multiple run headers")
            header = record
            continue
        workload_id = record.get("workload_id")
        if not isinstance(workload_id, str) or not workload_id:
            raise ValueError(f"workload line {line_number} has no workload ID")
        if workload_id in workload_ids:
            raise ValueError(f"duplicate workload ID: {workload_id}")
        workload_ids.add(workload_id)
        rows.append(record)
    if header is None:
        raise ValueError("workload has no run header")
    return header, rows


def join_attempts(
    attempts: list[dict[str, object]],
    summaries: list[dict[str, object]],
) -> dict[str, object]:
    summaries_by_request: dict[str, list[dict[str, object]]] = defaultdict(list)
    for summary in summaries:
        summaries_by_request[str(summary["request_id"])].append(summary)

    joined: dict[str, Any] = {
        "matched": [],
        "missing_request_id": [],
        "missing_summary": [],
        "duplicate_summary": [],
        "skipped": [],
        "unrelated_summary_count": 0,
    }
    attempted_request_ids = set()
    for workload in attempts:
        if workload.get("record_type") == "skipped":
            joined["skipped"].append(workload)
            continue
        if workload.get("record_type") != "attempt":
            continue
        request_id = workload.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            joined["missing_request_id"].append(workload)
            continue
        attempted_request_ids.add(request_id)
        candidates = summaries_by_request.get(request_id, [])
        if not candidates:
            joined["missing_summary"].append(workload)
        elif len(candidates) > 1:
            joined["duplicate_summary"].append({
                "workload": workload,
                "summary_count": len(candidates),
            })
        else:
            joined["matched"].append({
                "workload": workload,
                "summary": candidates[0],
            })
    joined["unrelated_summary_count"] = sum(
        len(items)
        for request_id, items in summaries_by_request.items()
        if request_id not in attempted_request_ids
    )
    return joined
