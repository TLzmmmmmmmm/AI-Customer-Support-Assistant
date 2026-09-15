"""Pure helpers for the Week 4 Day 3 streaming comparison."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from performance.analysis import estimate_request_cost_cny, numeric_stats


DAY3_ROUTES = ("product_search", "knowledge", "direct")
DAY3_TOTAL_REQUESTS = 36
DAY3_PHASE_REQUESTS = 18
DAY3_COST_CEILING_CNY = 0.50


def _non_blank(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def load_day3_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("workload manifest must be an object")
    _non_blank(manifest.get("schema_version"), "schema_version")
    seed = manifest.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    repeat_count = manifest.get("repeat_count")
    if (
        not isinstance(repeat_count, int)
        or isinstance(repeat_count, bool)
        or repeat_count <= 0
    ):
        raise ValueError("repeat_count must be positive")
    interval = manifest.get("interval_seconds")
    if (
        not isinstance(interval, (int, float))
        or isinstance(interval, bool)
        or interval <= 0
    ):
        raise ValueError("interval_seconds must be positive")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")

    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be an object")
        case_id = _non_blank(case.get("case_id"), f"case {index} case_id")
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        route = _non_blank(
            case.get("target_route"),
            f"case {case_id} target_route",
        )
        if route not in DAY3_ROUTES:
            raise ValueError(f"unsupported target route: {route}")
        messages = case.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"case {case_id} messages must be non-empty")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(f"case {case_id} message must be an object")
            if message.get("role") not in {"user", "assistant"}:
                raise ValueError(f"case {case_id} has invalid message role")
            _non_blank(message.get("content"), f"case {case_id} content")
    return manifest


def build_day3_schedule(
    manifest: Mapping[str, object],
    phase: str,
) -> list[dict[str, object]]:
    if phase not in {"baseline", "after"}:
        raise ValueError("phase must be baseline or after")
    schedule = []
    for case in manifest["cases"]:
        for execution_index in range(1, int(manifest["repeat_count"]) + 1):
            schedule.append({
                "workload_id": f"{phase}:{case['case_id']}:{execution_index}",
                "phase": phase,
                "case_id": case["case_id"],
                "target_route": case["target_route"],
                "execution_index": execution_index,
                "messages": [dict(message) for message in case["messages"]],
            })
    random.Random(int(manifest["seed"])).shuffle(schedule)
    return schedule


def parse_success_ndjson(body: str) -> dict[str, object]:
    try:
        events = [
            json.loads(line)
            for line in body.splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as error:
        raise ValueError("response contains invalid NDJSON") from error
    if not events or not all(isinstance(event, dict) for event in events):
        raise ValueError("response has an invalid event sequence")
    if events[-1].get("type") != "done":
        raise ValueError("response does not end with done")
    if len(events) < 2:
        raise ValueError("response has an invalid event sequence")
    if sum(event.get("type") == "done" for event in events) != 1:
        raise ValueError("response contains an invalid done event")

    parts = []
    citations_seen = False
    for event in events[:-1]:
        event_type = event.get("type")
        if event_type == "delta" and not citations_seen:
            content = event.get("content")
            if not isinstance(content, str) or not content:
                raise ValueError("delta content is empty or invalid")
            parts.append(content)
        elif event_type == "citations" and parts and not citations_seen:
            citations_seen = True
        else:
            raise ValueError("response has an invalid event sequence")
    if not parts:
        raise ValueError("response has no answer delta")
    return {"answer": "".join(parts), "delta_count": len(parts)}


def day2_budget_screening_projection(path: Path) -> float:
    analysis = json.loads(path.read_text(encoding="utf-8"))
    by_route = analysis["estimated_cost_cny"]["by_route"]
    means = [by_route[route]["mean"] for route in DAY3_ROUTES]
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in means
    ):
        raise ValueError("Day 2 route cost means are incomplete")
    return 12 * sum(float(value) for value in means)


def _is_success(pair: Mapping[str, object]) -> bool:
    workload = pair["workload"]
    summary = pair["summary"]
    return (
        workload.get("outcome") == "success"
        and workload.get("terminal_done") is True
        and summary.get("http_status") == 200
        and summary.get("outcome") == "success"
    )


def _numeric_summary(
    pairs: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, object]:
    return numeric_stats([
        float(pair["summary"][field])
        for pair in pairs
        if isinstance(pair["summary"].get(field), (int, float))
        and not isinstance(pair["summary"].get(field), bool)
    ])


def _route_metrics(
    pairs: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    metrics = {}
    for route in DAY3_ROUTES:
        route_pairs = [
            pair for pair in pairs
            if pair["summary"].get("route") == route and _is_success(pair)
        ]
        metrics[route] = {
            "successful": len(route_pairs),
            "first_delta_latency_ms": _numeric_summary(
                route_pairs, "first_delta_latency_ms"
            ),
            "buffering_saved_ms": _numeric_summary(
                route_pairs, "buffering_saved_ms"
            ),
            "total_latency_ms": _numeric_summary(
                route_pairs, "total_latency_ms"
            ),
            "model_latency_ms": _numeric_summary(
                route_pairs, "model_latency_ms"
            ),
            "output_tokens": _numeric_summary(route_pairs, "output_tokens"),
        }
    return metrics


def _p50(metrics: Mapping[str, object], route: str, field: str) -> float | None:
    value = metrics[route][field]["p50"]
    return float(value) if isinstance(value, (int, float)) else None


def analyze_day3_phases(
    baseline_pairs: Sequence[Mapping[str, object]],
    after_pairs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    phase_pairs = {"baseline": list(baseline_pairs), "after": list(after_pairs)}
    metrics = {
        phase: _route_metrics(pairs)
        for phase, pairs in phase_pairs.items()
    }
    successful = {
        phase: sum(_is_success(pair) for pair in pairs)
        for phase, pairs in phase_pairs.items()
    }
    route_counts = {
        phase: dict(Counter(
            str(pair["summary"].get("route"))
            for pair in pairs
            if _is_success(pair)
            and pair["summary"].get("route") in DAY3_ROUTES
        ))
        for phase, pairs in phase_pairs.items()
    }
    for counts in route_counts.values():
        for route in DAY3_ROUTES:
            counts.setdefault(route, 0)
        ordered = {route: counts[route] for route in DAY3_ROUTES}
        counts.clear()
        counts.update(ordered)

    success_gate = (
        len(baseline_pairs) == DAY3_PHASE_REQUESTS
        and len(after_pairs) == DAY3_PHASE_REQUESTS
        and successful == {"baseline": 18, "after": 18}
    )
    route_counts_pass = all(
        route_counts[phase][route] == 6
        for phase in phase_pairs
        for route in DAY3_ROUTES
    )
    terminal_gate = all(
        pair["workload"].get("terminal_done") is True
        for pairs in phase_pairs.values()
        for pair in pairs
    )

    ttft: dict[str, dict[str, object]] = {}
    for route in ("product_search", "knowledge"):
        before = _p50(metrics["baseline"], route, "first_delta_latency_ms")
        after = _p50(metrics["after"], route, "first_delta_latency_ms")
        absolute = None if before is None or after is None else before - after
        relative = (
            None
            if before is None or after is None or before <= 0
            else absolute / before
        )
        ttft[route] = {
            "baseline_p50_ms": before,
            "after_p50_ms": after,
            "absolute_reduction_ms": absolute,
            "relative_reduction": relative,
            "pass": (
                absolute is not None
                and relative is not None
                and absolute >= 500
                and relative >= 0.30
            ),
        }

    total_gate = {}
    for route in DAY3_ROUTES:
        before = _p50(metrics["baseline"], route, "total_latency_ms")
        after = _p50(metrics["after"], route, "total_latency_ms")
        total_gate[route] = (
            before is not None
            and after is not None
            and after <= before * 1.15
        )

    all_pairs = [*baseline_pairs, *after_pairs]
    request_costs = []
    request_rows = []
    for pair in all_pairs:
        summary = pair["summary"]
        workload = pair["workload"]
        cost = estimate_request_cost_cny(summary)
        if cost is not None:
            request_costs.append(cost)
        request_rows.append({
            "request_id": summary.get("request_id"),
            "phase": workload.get("phase"),
            "case_id": workload.get("case_id"),
            "target_route": workload.get("target_route"),
            "route": summary.get("route"),
            "success": _is_success(pair),
            "terminal_done": workload.get("terminal_done") is True,
            "first_delta_latency_ms": summary.get("first_delta_latency_ms"),
            "buffering_saved_ms": summary.get("buffering_saved_ms"),
            "total_latency_ms": summary.get("total_latency_ms"),
            "model_latency_ms": summary.get("model_latency_ms"),
            "output_tokens": summary.get("output_tokens"),
            "estimated_cost_cny": cost,
        })

    all_pass = (
        success_gate
        and route_counts_pass
        and terminal_gate
        and all(item["pass"] for item in ttft.values())
        and all(total_gate.values())
    )
    return {
        "schema_version": "1.0",
        "samples": {
            "baseline": len(baseline_pairs),
            "after": len(after_pairs),
            "total": len(all_pairs),
            "successful": successful,
        },
        "by_phase_route": metrics,
        "gates": {
            "success_36_of_36": success_gate,
            "route_counts": route_counts,
            "route_counts_pass": route_counts_pass,
            "terminal_and_finalization": terminal_gate,
            "ttft": ttft,
            "total_latency_p50_within_15_percent": total_gate,
            "all_pass": all_pass,
        },
        "estimated_cost_cny": {
            "coverage": len(request_costs),
            "total": sum(request_costs),
        },
        "requests": request_rows,
        "conclusion": "A" if all_pass else "B",
    }


def render_day3_markdown(analysis: Mapping[str, object]) -> str:
    gates = analysis["gates"]
    metrics = analysis["by_phase_route"]
    lines = [
        "# Week 4 Day 3 Streaming Closeout",
        "",
        "## Result",
        "",
        f"Conclusion: **{analysis['conclusion']}**.",
        "",
        "## Explicit gates",
        "",
        f"- 36/36 success: `{gates['success_36_of_36']}`",
        f"- Exact route counts: `{gates['route_counts_pass']}`",
        f"- Terminal/finalization invariant: `{gates['terminal_and_finalization']}`",
    ]
    for route in ("product_search", "knowledge"):
        item = gates["ttft"][route]
        lines.append(
            f"- {route} TTFT >=30% and >=500 ms: `{item['pass']}` "
            f"({item['absolute_reduction_ms']} ms, {item['relative_reduction']})"
        )
    for route in DAY3_ROUTES:
        lines.append(
            f"- {route} total-latency P50 <=15% regression: "
            f"`{gates['total_latency_p50_within_15_percent'][route]}`"
        )
    lines.extend((
        "",
        "## Route metrics",
        "",
        "| Phase | Route | Success | TTFT P50/P95 ms | Buffering saved P50/P95 ms | Total P50/P95 ms |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ))
    for phase in ("baseline", "after"):
        for route in DAY3_ROUTES:
            item = metrics[phase][route]
            first = item["first_delta_latency_ms"]
            saved = item["buffering_saved_ms"]
            total = item["total_latency_ms"]
            lines.append(
                f"| {phase} | {route} | {item['successful']} | "
                f"{first['p50']} / {first['p95']} | "
                f"{saved['p50']} / {saved['p95']} | "
                f"{total['p50']} / {total['p95']} |"
            )
    lines.extend((
        "",
        "## Estimated cost and limitations",
        "",
        f"Estimated measured cost: CNY {analysis['estimated_cost_cny']['total']:.6f}.",
        "",
        "Repeated test cases may increase prompt cache reuse; observed estimated cost reflects the measured cache behavior of this representative synthetic workload.",
        "",
        "This is synthetic server-side TTFT, not browser rendering latency or a production SLO.",
        "",
    ))
    return "\n".join(lines)


__all__ = [
    "DAY3_COST_CEILING_CNY",
    "DAY3_PHASE_REQUESTS",
    "DAY3_ROUTES",
    "DAY3_TOTAL_REQUESTS",
    "analyze_day3_phases",
    "build_day3_schedule",
    "day2_budget_screening_projection",
    "load_day3_manifest",
    "parse_success_ndjson",
    "render_day3_markdown",
]
