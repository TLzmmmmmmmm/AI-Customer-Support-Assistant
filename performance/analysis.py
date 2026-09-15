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
        "cache_hit_input": 0.02,
        "cache_miss_input": 1.00,
        "output": 4.00,
    },
    "peak": {
        "cache_hit_input": 0.04,
        "cache_miss_input": 2.00,
        "output": 8.00,
    },
}

APPROVED_LABELS = {
    "A": "A. True streaming is justified",
    "B": "B. Tool/agent execution optimization is justified",
    "C": "C. Another specific measured bottleneck should be addressed",
    "D": "D. No material optimization is currently justified",
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
    "first_delta_latency_ms",
    "buffering_saved_ms",
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
    repeated_product_search_executions = sum(
        pair["summary"].get("route") == "product_search"
        and (pair["summary"].get("tool_execution_count") or 0) > 1
        for pair in successful
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
    overall_costs = [cost for _, cost in complete_successful_cost_rows]
    overall_cost_stats = numeric_stats(overall_costs)
    overall_cost_stats["total"] = sum(overall_costs)
    route_cost_stats = {}
    for route, route_costs in sorted(cost_by_route.items()):
        metrics = numeric_stats(route_costs)
        metrics["total"] = sum(route_costs)
        route_cost_stats[route] = metrics
    diagnostics_by_request = {
        row["request_id"]: row for row in dominant_rows
    }
    slow_requests = []
    for pair in sorted(
        matched,
        key=lambda item: item["summary"].get("total_latency_ms") or 0,
        reverse=True,
    )[:10]:
        summary = pair["summary"]
        diagnostic = diagnostics_by_request.get(summary["request_id"], {})
        slow_requests.append({
            "request_id": summary["request_id"],
            "route": summary.get("route"),
            "router_type": summary.get("router_type"),
            "total_latency_ms": summary.get("total_latency_ms"),
            "router_latency_ms": summary.get("router_latency_ms"),
            "retrieval_latency_ms": summary.get("retrieval_latency_ms"),
            "tool_latency_ms": summary.get("tool_latency_ms"),
            "model_latency_ms": summary.get("model_latency_ms"),
            "dominant_stage": diagnostic.get("dominant_stage"),
            "dominant_stage_ratio": diagnostic.get("dominant_stage_ratio"),
            "input_tokens": summary.get("input_tokens"),
            "output_tokens": summary.get("output_tokens"),
            "tool_execution_count": summary.get("tool_execution_count"),
            "executed_tool_names": summary.get("executed_tool_names"),
            "outcome": summary.get("outcome"),
            "failure_layer": summary.get("failure_layer"),
            "failure_code": summary.get("failure_code"),
        })

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
            "product_search_requests_with_multiple_executions": (
                repeated_product_search_executions
            ),
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
            "overall": overall_cost_stats,
            "by_route": route_cost_stats,
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
        "slowest_requests": slow_requests,
    }


def _display(value: object, decimals: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def _route_with_highest(
    groups: Mapping[str, Mapping[str, object]],
    field: str,
    decimals: int = 1,
) -> str:
    eligible = [
        (name, metrics.get(field))
        for name, metrics in groups.items()
        if isinstance(metrics.get(field), (int, float))
    ]
    if not eligible:
        return "No observations"
    name, value = max(eligible, key=lambda item: item[1])
    return f"{name} ({_display(value, decimals)})"


def render_markdown(analysis: Mapping[str, object]) -> str:
    recommendation = analysis.get("recommendation")
    if not isinstance(recommendation, Mapping):
        raise ValueError("recommendation is required")
    category = recommendation.get("category")
    if category not in APPROVED_LABELS:
        raise ValueError("recommendation category is invalid")
    rationale = recommendation.get("rationale")
    if (
        not isinstance(rationale, list)
        or not rationale
        or any(not isinstance(item, str) or not item.strip() for item in rationale)
    ):
        raise ValueError("recommendation rationale must be non-empty")
    if recommendation.get("label") != APPROVED_LABELS[category]:
        raise ValueError("recommendation label does not match category")

    samples = analysis["samples"]
    telemetry = analysis["telemetry_join"]
    latency = analysis["latency"]
    stages = analysis["stage_latency"]
    tokens = analysis["tokens"]
    tools = analysis["tools"]
    costs = analysis["estimated_cost_cny"]
    pricing = costs["pricing_snapshot"]
    conversations = analysis["conversations"]
    failures = analysis["failures"]
    workload = analysis.get("workload", {})
    lines = [
        "# Week 4 Day 2 Metrics, Cost, and Performance Analysis",
        "",
        "## Workload methodology",
        "",
        (
            "Sequential production-style HTTP workload with no runner retries. "
            f"Manifest seed: `{workload.get('seed', 'unknown')}`; repeat count: "
            f"`{workload.get('repeat_count', 'unknown')}`. Target routes shape "
            "the workload only; every grouping below uses the actual production route."
        ),
        "",
        "## Sample counts",
        "",
        "| Attempted | Successful | Failed | Skipped | Single-turn success | Multi-turn success |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {samples['attempted']} | {samples['successful']} | "
            f"{samples['failed']} | {samples['skipped']} | "
            f"{samples['single_turn_successful']} | "
            f"{samples['multi_turn_successful']} |"
        ),
        "",
        "## Metric definitions",
        "",
        (
            "Success requires a uniquely joined production summary with HTTP 200 "
            "and `outcome=success`. P50 and P95 use the nearest-rank definition; "
            "each table includes its contributing sample count. Stage measurements "
            "may overlap and are not added as percentages of total latency."
        ),
        "",
        "## Telemetry coverage",
        "",
        "| Matched | Missing request ID | Missing summary | Duplicate summary | Unrelated summaries |",
        "| ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {telemetry['matched']} | {telemetry['missing_request_id']} | "
            f"{telemetry['missing_summary']} | {telemetry['duplicate_summary']} | "
            f"{telemetry['unrelated_summary_count']} |"
        ),
        "",
        "## P50/P95 latency",
        "",
        "| Sample count | Mean ms | P50 ms | P95 ms |",
        "| ---: | ---: | ---: | ---: |",
        (
            f"| {latency['overall']['count']} | {_display(latency['overall']['mean'])} | "
            f"{_display(latency['overall']['p50'])} | "
            f"{_display(latency['overall']['p95'])} |"
        ),
        "",
        "## Latency by actual route",
        "",
        "| Route | Sample count | Mean ms | P50 ms | P95 ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for route, metrics in latency["by_route"].items():
        lines.append(
            f"| {route} | {metrics['count']} | {_display(metrics['mean'])} | "
            f"{_display(metrics['p50'])} | {_display(metrics['p95'])} |"
        )
    if not latency["by_route"]:
        lines.append("| No observations | 0 | — | — | — |")

    lines.extend((
        "",
        "## Stage diagnosis",
        "",
        (
            "`dominant_stage_ratio` is the largest non-null measured stage divided "
            "by positive total latency. It identifies whether one measured stage "
            "dominates; stage timings may be sequential or overlap."
        ),
        "",
        "| Route | Stage | Sample count | Mean ms | P50 ms | P95 ms |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ))
    for route, route_stages in stages["by_route"].items():
        for stage, metrics in route_stages.items():
            lines.append(
                f"| {route} | {stage} | {metrics['count']} | "
                f"{_display(metrics['mean'])} | {_display(metrics['p50'])} | "
                f"{_display(metrics['p95'])} |"
            )
    ratio = stages["dominant_stage_ratio"]
    if isinstance(ratio.get("p50"), (int, float)) and ratio["p50"] < 0.5:
        lines.extend(("", "No single measured stage dominates at the median ratio."))

    token_coverage = tokens["coverage"]
    token_overall = tokens["overall"]
    lines.extend((
        "",
        "## Tokens/request",
        "",
        (
            f"Complete token sample count: {token_coverage['complete']} of "
            f"{token_coverage['eligible']}; excluded: {token_coverage['excluded']}."
        ),
        "",
        "| Average input | Average output | Average total |",
        "| ---: | ---: | ---: |",
        (
            f"| {_display(token_overall['average_input'])} | "
            f"{_display(token_overall['average_output'])} | "
            f"{_display(token_overall['average_total'])} |"
        ),
        "",
        "## Tokens by route",
        "",
        "| Route | Sample count | Average input | Average output | Average total |",
        "| --- | ---: | ---: | ---: | ---: |",
    ))
    for route, metrics in tokens["by_route"].items():
        lines.append(
            f"| {route} | {metrics['count']} | {_display(metrics['average_input'])} | "
            f"{_display(metrics['average_output'])} | "
            f"{_display(metrics['average_total'])} |"
        )

    lines.extend((
        "",
        "## Tool execution",
        "",
        f"Average execution count: {_display(tools['overall_average_execution_count'])}.",
        "",
        "Execution-count distribution: "
        + json.dumps(tools["execution_count_distribution"], ensure_ascii=False),
        "",
        "Ordered executed-tool sequences: "
        + json.dumps(tools["ordered_name_sequences"], ensure_ascii=False),
        "",
        (
            "Product-search requests with more than one execution: "
            f"{tools['product_search_requests_with_multiple_executions']}."
        ),
        "",
        "## Estimated cost/request",
        "",
        (
            f"Coverage: {costs['coverage']['complete']} of "
            f"{costs['coverage']['eligible']} successful requests; excluded: "
            f"{costs['coverage']['excluded']}. Currency: CNY."
        ),
        "",
        (
            f"Pricing snapshot for `{pricing['configured_model']}` per "
            f"{pricing['unit_tokens']:,} tokens:"
        ),
        "",
        "| Period | Cache-hit input | Cache-miss input | Output |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Off-peak | {pricing['off_peak']['cache_hit_input']:.2f} | "
            f"{pricing['off_peak']['cache_miss_input']:.2f} | "
            f"{pricing['off_peak']['output']:.2f} |"
        ),
        (
            f"| Peak | {pricing['peak']['cache_hit_input']:.2f} | "
            f"{pricing['peak']['cache_miss_input']:.2f} | "
            f"{pricing['peak']['output']:.2f} |"
        ),
        "",
        "| Sample count | Total estimated CNY | Mean CNY | P50 CNY | P95 CNY |",
        "| ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {costs['overall']['count']} | "
            f"{_display(costs['overall']['total'], 6)} | "
            f"{_display(costs['overall']['mean'], 6)} | "
            f"{_display(costs['overall']['p50'], 6)} | "
            f"{_display(costs['overall']['p95'], 6)} |"
        ),
        "",
        "| Actual route | Sample count | Total estimated CNY | Mean CNY | P50 CNY | P95 CNY |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ))
    for route, metrics in costs["by_route"].items():
        lines.append(
            f"| {route} | {metrics['count']} | {_display(metrics['total'], 6)} | "
            f"{_display(metrics['mean'], 6)} | {_display(metrics['p50'], 6)} | "
            f"{_display(metrics['p95'], 6)} |"
        )
    lines.extend((
        "",
        "## Estimated cost/conversation",
        "",
        (
            f"Complete cost sample count: {conversations['estimated_cost_cny']['count']}; "
            f"mean CNY: {_display(conversations['estimated_cost_cny']['mean'], 6)}; "
            f"P50 CNY: {_display(conversations['estimated_cost_cny']['p50'], 6)}."
        ),
        "",
        "| conversation_id | Complete | Estimated or partial CNY |",
        "| --- | --- | ---: |",
    ))
    for item in conversations["items"]:
        item_cost = item.get(
            "estimated_cost_cny",
            item.get("partial_estimated_cost_cny"),
        )
        lines.append(
            f"| {item['conversation_id']} | {str(item['complete']).lower()} | "
            f"{_display(item_cost, 6)} |"
        )
    lines.extend((
        "",
        "## Failures",
        "",
        (
            f"Failed {failures['failed']} of {failures['attempted']} attempts "
            f"(rate {_display(failures['failure_rate'], 4)}). Layers: "
            f"{json.dumps(failures['by_layer'], ensure_ascii=False)}. Codes: "
            f"{json.dumps(failures['by_code'], ensure_ascii=False)}."
        ),
        "",
        "## Slowest requests",
        "",
        "| Request ID | Route | Router type | Total ms | Model ms | Dominant stage | Ratio | Tool executions | Outcome |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
    ))
    for row in analysis["slowest_requests"]:
        lines.append(
            f"| {row['request_id']} | {_display(row['route'])} | "
            f"{_display(row['router_type'])} | {_display(row['total_latency_ms'])} | "
            f"{_display(row['model_latency_ms'])} | {_display(row['dominant_stage'])} | "
            f"{_display(row['dominant_stage_ratio'], 3)} | "
            f"{_display(row['tool_execution_count'])} | {_display(row['outcome'])} |"
        )
    if not analysis["slowest_requests"]:
        lines.append("| No observations | — | — | — | — | — | — | — | — |")

    slow_model_bound = sum(
        row.get("dominant_stage") == "model"
        for row in analysis["slowest_requests"]
    )
    product_repeats = sum(
        row.get("route") == "product_search"
        and (row.get("tool_execution_count") or 0) > 1
        for row in analysis["slowest_requests"]
    )
    lines.extend((
        "",
        "## Bottleneck conclusion",
        "",
        (
            "- Latency concentration: dominant measured stages were "
            f"{json.dumps(stages['dominant_stage_distribution'], ensure_ascii=False)}; "
            f"median dominant-stage ratio was {_display(ratio['p50'], 3)}."
        ),
        (
            "- Slowest actual routes: highest route P50 was "
            f"{_route_with_highest(latency['by_route'], 'p50')}; highest route P95 "
            f"was {_route_with_highest(latency['by_route'], 'p95')}."
        ),
        (
            f"- Slow-row model diagnosis: {slow_model_bound} of "
            f"{len(analysis['slowest_requests'])} listed rows were model-dominant."
        ),
        (
            "- Router-type comparison is descriptive, not causal: "
            + json.dumps(latency["by_router_type"], ensure_ascii=False)
            + "."
        ),
        (
            "- Retrieval and tool materiality are represented by their route-level "
            "non-null sample counts and latency percentiles in the stage table."
        ),
        (
            "- Repeated product-search execution: "
            f"{tools['product_search_requests_with_multiple_executions']} successful "
            "product-search requests executed more than one tool; "
            f"{product_repeats} were in the listed slow rows."
        ),
        (
            "- Highest-token route by average total tokens: "
            f"{_route_with_highest(tokens['by_route'], 'average_total')}; "
            "highest-cost route by mean estimated CNY: "
            f"{_route_with_highest(costs['by_route'], 'mean', 6)}."
        ),
        "",
        "## Day 3 recommendation",
        "",
    ))
    lines.extend(f"- {item}" for item in rationale)
    lines.extend((
        "",
        "## Limitations",
        "",
        (
            "This representative synthetic workload is not a production SLO or a "
            "universal conversation average. Router-type comparisons may have "
            "different route mixes. TTFT is not instrumented, so this evidence "
            "cannot establish a true-streaming TTFT improvement."
        ),
        "",
        (
            "Repeated test cases may increase prompt cache reuse; observed estimated "
            "cost reflects the measured cache behavior of this representative "
            "synthetic workload."
        ),
        "",
        APPROVED_LABELS[category],
    ))
    return "\n".join(lines) + "\n"
