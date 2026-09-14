"""Pure helpers for Week 4 Day 2 performance analysis."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


PRICING_SNAPSHOT = {
    "provider": "DeepSeek",
    "configured_model": "deepseek-v4-flash",
    "currency": "CNY",
    "unit_tokens": 1_000_000,
    "source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
    "snapshot_date": "2026-09-14",
    "timezone": "Asia/Shanghai",
    "peak_windows": ["weekday 09:00-12:00", "weekday 14:00-18:00"],
    "off_peak": {
        "cache_hit_input": 0.05,
        "cache_miss_input": 1.50,
        "output": 4.50,
    },
    "peak": {
        "cache_hit_input": 0.10,
        "cache_miss_input": 3.00,
        "output": 9.00,
    },
}


def _shanghai_timezone():
    try:
        return ZoneInfo("Asia/Shanghai")
    except ZoneInfoNotFoundError:
        return timezone(timedelta(hours=8), "Asia/Shanghai")


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


def nearest_rank(
    values: Sequence[float],
    percentile: int,
) -> float | None:
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[rank - 1]


def numeric_stats(values: Sequence[float]) -> dict[str, object]:
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    return {
        "count": len(numeric_values),
        "mean": sum(numeric_values) / len(numeric_values),
        "p50": nearest_rank(numeric_values, 50),
        "p95": nearest_rank(numeric_values, 95),
    }


def is_peak(timestamp: datetime) -> bool:
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    local = timestamp.astimezone(_shanghai_timezone())
    if local.weekday() >= 5:
        return False
    minute = local.hour * 60 + local.minute
    return 9 * 60 <= minute < 12 * 60 or 14 * 60 <= minute < 18 * 60


def estimate_request_cost_cny(
    summary: Mapping[str, object],
    pricing: Mapping[str, object] = PRICING_SNAPSHOT,
) -> float | None:
    timestamp = summary.get("timestamp")
    token_fields = (
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "output_tokens",
    )
    tokens = [summary.get(field) for field in token_fields]
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
        return None
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in tokens
    ):
        return None
    period = "peak" if is_peak(timestamp) else "off_peak"
    rates = pricing[period]
    unit = pricing["unit_tokens"]
    return (
        tokens[0] * rates["cache_hit_input"]
        + tokens[1] * rates["cache_miss_input"]
        + tokens[2] * rates["output"]
    ) / unit


def _is_success(pair: Mapping[str, object]) -> bool:
    summary = pair["summary"]
    return (
        summary.get("http_status") == 200
        and summary.get("outcome") == "success"
    )


def _group_pairs(
    pairs: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for pair in pairs:
        value = pair["summary"].get(field)
        grouped[str(value) if value is not None else "unknown"].append(pair)
    return dict(grouped)


def _latency_stats(
    pairs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return numeric_stats([
        pair["summary"]["total_latency_ms"]
        for pair in pairs
        if isinstance(pair["summary"].get("total_latency_ms"), (int, float))
    ])


def _stage_latency(
    pairs: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    return {
        stage: numeric_stats([
            pair["summary"][field]
            for pair in pairs
            if isinstance(pair["summary"].get(field), (int, float))
        ])
        for stage, field in (
            ("router", "router_latency_ms"),
            ("retrieval", "retrieval_latency_ms"),
            ("tool", "tool_latency_ms"),
            ("model", "model_latency_ms"),
        )
    }


def _token_metrics(
    pairs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    complete = [
        pair
        for pair in pairs
        if isinstance(pair["summary"].get("input_tokens"), int)
        and isinstance(pair["summary"].get("output_tokens"), int)
    ]
    count = len(complete)
    if not count:
        return {
            "count": 0,
            "average_input": None,
            "average_output": None,
            "average_total": None,
        }
    inputs = [pair["summary"]["input_tokens"] for pair in complete]
    outputs = [pair["summary"]["output_tokens"] for pair in complete]
    return {
        "count": count,
        "average_input": sum(inputs) / count,
        "average_output": sum(outputs) / count,
        "average_total": sum(inputs + outputs) / count,
    }


def _conversation_metrics(
    matched: Sequence[Mapping[str, object]],
    skipped: Sequence[Mapping[str, object]],
    pricing: Mapping[str, object],
) -> dict[str, object]:
    grouped: dict[str, dict[str, list[Mapping[str, object]]]] = defaultdict(
        lambda: {"attempts": [], "skipped": []}
    )
    for pair in matched:
        workload = pair["workload"]
        conversation_id = workload.get("conversation_id")
        if workload.get("kind") == "multi_turn" and conversation_id:
            grouped[str(conversation_id)]["attempts"].append(pair)
    for row in skipped:
        conversation_id = row.get("conversation_id")
        if row.get("kind") == "multi_turn" and conversation_id:
            grouped[str(conversation_id)]["skipped"].append(row)

    items = []
    for conversation_id in sorted(grouped):
        group = grouped[conversation_id]
        attempts = sorted(
            group["attempts"],
            key=lambda pair: pair["workload"].get("turn_index") or 0,
        )
        skipped_rows = sorted(
            group["skipped"],
            key=lambda row: row.get("turn_index") or 0,
        )
        complete = bool(attempts) and not skipped_rows and all(
            _is_success(pair) for pair in attempts
        )
        costs = [
            estimate_request_cost_cny(pair["summary"], pricing)
            for pair in attempts
        ]
        item = {
            "conversation_id": conversation_id,
            "complete": complete,
            "turn_indices": [
                pair["workload"].get("turn_index") for pair in attempts
            ],
            "skipped_turn_indices": [
                row.get("turn_index") for row in skipped_rows
            ],
        }
        available_cost = sum(cost for cost in costs if cost is not None)
        if complete and costs and all(cost is not None for cost in costs):
            item["estimated_cost_cny"] = available_cost
        else:
            item["partial_estimated_cost_cny"] = available_cost
        items.append(item)
    complete_costs = [
        item["estimated_cost_cny"]
        for item in items
        if "estimated_cost_cny" in item
    ]
    return {
        "complete_count": sum(item["complete"] for item in items),
        "partial_count": sum(not item["complete"] for item in items),
        "estimated_cost_cny": numeric_stats(complete_costs),
        "items": items,
    }


def analyze_join(
    joined: Mapping[str, object],
    pricing: Mapping[str, object] | None = None,
) -> dict[str, object]:
    selected_pricing = pricing or PRICING_SNAPSHOT
    matched = list(joined["matched"])
    successful = [pair for pair in matched if _is_success(pair)]
    unsuccessful = [pair for pair in matched if not _is_success(pair)]
    missing_request_id = list(joined["missing_request_id"])
    missing_summary = list(joined["missing_summary"])
    duplicate_summary = list(joined["duplicate_summary"])
    skipped = list(joined["skipped"])
    attempted = (
        len(matched)
        + len(missing_request_id)
        + len(missing_summary)
        + len(duplicate_summary)
    )
    failed = attempted - len(successful)

    by_route = _group_pairs(successful, "route")
    by_router_type = _group_pairs(successful, "router_type")

    dominant_rows = []
    for pair in successful:
        summary = pair["summary"]
        stages = {
            stage: summary.get(field)
            for stage, field in (
                ("router", "router_latency_ms"),
                ("retrieval", "retrieval_latency_ms"),
                ("tool", "tool_latency_ms"),
                ("model", "model_latency_ms"),
            )
            if isinstance(summary.get(field), (int, float))
        }
        total = summary.get("total_latency_ms")
        if not stages or not isinstance(total, (int, float)) or total <= 0:
            continue
        stage, latency = max(stages.items(), key=lambda item: item[1])
        dominant_rows.append({
            "request_id": summary["request_id"],
            "dominant_stage": stage,
            "dominant_stage_ratio": latency / total,
            "model_to_total_ratio": (
                summary.get("model_latency_ms") / total
                if isinstance(summary.get("model_latency_ms"), (int, float))
                else None
            ),
        })

    token_overall = _token_metrics(successful)
    execution_counts = [
        int(pair["summary"].get("tool_execution_count", 0))
        for pair in successful
    ]
    sequence_counts = Counter(
        tuple(pair["summary"].get("executed_tool_names") or [])
        for pair in successful
        if pair["summary"].get("executed_tool_names")
    )
    failure_layers = Counter(
        pair["summary"].get("failure_layer")
        for pair in unsuccessful
        if pair["summary"].get("failure_layer") is not None
    )
    failure_codes = Counter(
        pair["summary"].get("failure_code")
        for pair in unsuccessful
        if pair["summary"].get("failure_code") is not None
    )
    successful_cost_rows = [
        (pair, estimate_request_cost_cny(pair["summary"], selected_pricing))
        for pair in successful
    ]
    complete_successful_cost_rows = [
        (pair, cost)
        for pair, cost in successful_cost_rows
        if cost is not None
    ]
    cost_by_route: dict[str, list[float]] = defaultdict(list)
    for pair, cost in complete_successful_cost_rows:
        route = pair["summary"].get("route")
        cost_by_route[str(route) if route is not None else "unknown"].append(cost)
    failed_costs = [
        cost
        for pair in unsuccessful
        if (cost := estimate_request_cost_cny(
            pair["summary"], selected_pricing
        )) is not None
    ]

    return {
        "samples": {
            "attempted": attempted,
            "successful": len(successful),
            "failed": failed,
            "skipped": len(skipped),
            "single_turn_successful": sum(
                pair["workload"].get("kind") == "single_turn"
                for pair in successful
            ),
            "multi_turn_successful": sum(
                pair["workload"].get("kind") == "multi_turn"
                for pair in successful
            ),
        },
        "telemetry_join": {
            "matched": len(matched),
            "missing_request_id": len(missing_request_id),
            "missing_summary": len(missing_summary),
            "duplicate_summary": len(duplicate_summary),
            "unrelated_summary_count": joined["unrelated_summary_count"],
        },
        "latency": {
            "overall": _latency_stats(successful),
            "by_route": {
                route: _latency_stats(pairs)
                for route, pairs in sorted(by_route.items())
            },
            "by_router_type": {
                router_type: _latency_stats(pairs)
                for router_type, pairs in sorted(by_router_type.items())
            },
        },
        "stage_latency": {
            "by_route": {
                route: _stage_latency(pairs)
                for route, pairs in sorted(by_route.items())
            },
            "dominant_stage_distribution": dict(sorted(Counter(
                row["dominant_stage"] for row in dominant_rows
            ).items())),
            "dominant_stage_ratio": numeric_stats([
                row["dominant_stage_ratio"] for row in dominant_rows
            ]),
            "request_diagnostics": dominant_rows,
        },
        "tokens": {
            "coverage": {
                "complete": token_overall["count"],
                "eligible": len(successful),
                "excluded": len(successful) - token_overall["count"],
            },
            "overall": token_overall,
            "by_route": {
                route: _token_metrics(pairs)
                for route, pairs in sorted(by_route.items())
            },
        },
        "tools": {
            "overall_average_execution_count": (
                sum(execution_counts) / len(execution_counts)
                if execution_counts else None
            ),
            "average_execution_count_by_route": {
                route: sum(
                    int(pair["summary"].get("tool_execution_count", 0))
                    for pair in pairs
                ) / len(pairs)
                for route, pairs in sorted(by_route.items())
            },
            "execution_count_distribution": {
                str(count): frequency
                for count, frequency in sorted(Counter(execution_counts).items())
            },
            "ordered_name_sequences": [
                {"tool_names": list(names), "count": count}
                for names, count in sorted(
                    sequence_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
        },
        "failures": {
            "attempted": attempted,
            "failed": failed,
            "failure_rate": failed / attempted if attempted else None,
            "by_layer": dict(sorted(failure_layers.items())),
            "by_code": dict(sorted(failure_codes.items())),
        },
        "estimated_cost_cny": {
            "pricing_snapshot": dict(selected_pricing),
            "coverage": {
                "complete": len(complete_successful_cost_rows),
                "eligible": len(successful),
                "excluded": len(successful) - len(complete_successful_cost_rows),
            },
            "overall": numeric_stats([
                cost for _, cost in complete_successful_cost_rows
            ]),
            "by_route": {
                route: numeric_stats(costs)
                for route, costs in sorted(cost_by_route.items())
            },
            "requests": [
                {
                    "request_id": pair["summary"]["request_id"],
                    "route": pair["summary"].get("route"),
                    "estimated_cost_cny": cost,
                }
                for pair, cost in complete_successful_cost_rows
            ],
            "failed_requests": {
                "count": len(failed_costs),
                "total": sum(failed_costs),
            },
        },
        "conversations": _conversation_metrics(
            matched,
            skipped,
            selected_pricing,
        ),
    }
