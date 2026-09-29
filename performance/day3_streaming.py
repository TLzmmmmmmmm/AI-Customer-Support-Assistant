"""Pure helpers for the Week 4 Day 3 streaming comparison."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from performance.analysis import estimate_request_cost_cny, numeric_stats


DAY3_ROUTES = (
    "product_search",
    "knowledge",
    "exact_product",
    "contact",
    "direct",
)
DAY3_ROUTE_REQUESTS = 10
DAY3_PHASE_REQUESTS = 50
DAY3_TOTAL_REQUESTS = 100
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


def day2_budget_screening_projection(
    path: Path,
    manifest: Mapping[str, object],
) -> float:
    analysis = json.loads(path.read_text(encoding="utf-8"))
    by_route = analysis["estimated_cost_cny"]["by_route"]
    means = [
        by_route[case["target_route"]]["mean"]
        for case in manifest["cases"]
    ]
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in means
    ):
        raise ValueError("Day 2 route cost means are incomplete")
    return 2 * int(manifest["repeat_count"]) * sum(
        float(value) for value in means
    )


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


def _workload_numeric_summary(
    pairs: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, object]:
    return numeric_stats([
        float(pair["workload"][field])
        for pair in pairs
        if isinstance(pair["workload"].get(field), (int, float))
        and not isinstance(pair["workload"].get(field), bool)
    ])


def _route_metrics(
    pairs: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    metrics = {}
    for route in DAY3_ROUTES:
        observed_pairs = [
            pair for pair in pairs
            if pair["summary"].get("route") == route
        ]
        route_pairs = [
            pair for pair in pairs
            if pair["summary"].get("route") == route and _is_success(pair)
        ]
        costs = [
            cost
            for pair in observed_pairs
            if (cost := estimate_request_cost_cny(pair["summary"])) is not None
        ]
        cost_stats = numeric_stats(costs)
        cost_stats["total"] = sum(costs)
        metrics[route] = {
            "observed": len(observed_pairs),
            "successful": len(route_pairs),
            "failed": len(observed_pairs) - len(route_pairs),
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
            "client_first_delta_latency_ms": _workload_numeric_summary(
                route_pairs, "client_first_delta_latency_ms"
            ),
            "client_total_latency_ms": _workload_numeric_summary(
                route_pairs, "client_total_latency_ms"
            ),
            "answer_chars": _workload_numeric_summary(
                route_pairs, "answer_chars"
            ),
            "estimated_cost_cny": cost_stats,
        }
    return metrics


def _paired_differences(
    baseline_pairs: Sequence[Mapping[str, object]],
    after_pairs: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    baseline_by_id = {
        pair["workload"].get("pair_id"): pair
        for pair in baseline_pairs
        if pair["workload"].get("pair_id")
    }
    after_by_id = {
        pair["workload"].get("pair_id"): pair
        for pair in after_pairs
        if pair["workload"].get("pair_id")
    }
    metrics = {}
    for route in DAY3_ROUTES:
        pairs = [
            (baseline_by_id[pair_id], after_by_id[pair_id])
            for pair_id in baseline_by_id.keys() & after_by_id.keys()
            if baseline_by_id[pair_id]["summary"].get("route") == route
            and after_by_id[pair_id]["summary"].get("route") == route
            and _is_success(baseline_by_id[pair_id])
            and _is_success(after_by_id[pair_id])
        ]

        def differences(source: str, field: str, *, reverse: bool = False):
            values = []
            for baseline, after in pairs:
                before = baseline[source].get(field)
                current = after[source].get(field)
                if (
                    isinstance(before, (int, float))
                    and not isinstance(before, bool)
                    and isinstance(current, (int, float))
                    and not isinstance(current, bool)
                ):
                    delta = float(current) - float(before)
                    values.append(delta if reverse else -delta)
            return numeric_stats(values)

        metrics[route] = {
            "matched": len(pairs),
            "server_first_delta_reduction_ms": differences(
                "summary", "first_delta_latency_ms"
            ),
            "client_first_delta_reduction_ms": differences(
                "workload", "client_first_delta_latency_ms"
            ),
            "total_latency_change_ms": differences(
                "summary", "total_latency_ms", reverse=True
            ),
            "output_tokens_change": differences(
                "summary", "output_tokens", reverse=True
            ),
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
    paired_differences = _paired_differences(baseline_pairs, after_pairs)
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
        and successful == {
            "baseline": DAY3_PHASE_REQUESTS,
            "after": DAY3_PHASE_REQUESTS,
        }
    )
    success_by_phase = {
        "baseline_50_of_50": (
            len(baseline_pairs) == DAY3_PHASE_REQUESTS
            and successful["baseline"] == DAY3_PHASE_REQUESTS
        ),
        "after_50_of_50": (
            len(after_pairs) == DAY3_PHASE_REQUESTS
            and successful["after"] == DAY3_PHASE_REQUESTS
        ),
    }
    route_counts_pass = all(
        route_counts[phase][route] == DAY3_ROUTE_REQUESTS
        for phase in phase_pairs
        for route in DAY3_ROUTES
    )
    terminal_gate = all(
        pair["workload"].get("terminal_done") is True
        for pairs in phase_pairs.values()
        for pair in pairs
    )
    baseline_pair_ids = {
        pair["workload"].get("pair_id") for pair in baseline_pairs
    }
    after_pair_ids = {
        pair["workload"].get("pair_id") for pair in after_pairs
    }
    pair_gate = (
        None not in baseline_pair_ids
        and baseline_pair_ids == after_pair_ids
        and len(baseline_pair_ids) == DAY3_PHASE_REQUESTS
    )
    baseline_mode_gate = all(
        pair["workload"].get("mode") == "buffered"
        and pair["workload"].get("delta_count") == 1
        for pair in baseline_pairs
    )
    after_mode_gate = all(
        pair["workload"].get("mode") == "streaming"
        and isinstance(pair["workload"].get("delta_count"), int)
        and pair["workload"]["delta_count"] > 1
        for pair in after_pairs
    )
    streaming_mode_gate = baseline_mode_gate and after_mode_gate
    target_routes_match = all(
        pair["summary"].get("route") == pair["workload"].get("target_route")
        for pairs in phase_pairs.values()
        for pair in pairs
    )
    client_metrics_complete = all(
        all(
            isinstance(pair["workload"].get(field), (int, float))
            and not isinstance(pair["workload"].get(field), bool)
            for field in (
                "client_first_delta_latency_ms",
                "client_total_latency_ms",
            )
        )
        for pairs in phase_pairs.values()
        for pair in pairs
    )

    ttft: dict[str, dict[str, object]] = {}
    client_ttft: dict[str, dict[str, object]] = {}
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
        client_before = _p50(
            metrics["baseline"], route, "client_first_delta_latency_ms"
        )
        client_after = _p50(
            metrics["after"], route, "client_first_delta_latency_ms"
        )
        client_absolute = (
            None
            if client_before is None or client_after is None
            else client_before - client_after
        )
        client_relative = (
            None
            if client_absolute is None or client_before is None or client_before <= 0
            else client_absolute / client_before
        )
        client_ttft[route] = {
            "baseline_p50_ms": client_before,
            "after_p50_ms": client_after,
            "absolute_reduction_ms": client_absolute,
            "relative_reduction": client_relative,
            "pass": (
                client_absolute is not None
                and client_relative is not None
                and client_absolute >= 500
                and client_relative >= 0.30
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
    all_routes_total_gate = all(total_gate.values())

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
            "pair_id": workload.get("pair_id"),
            "mode": workload.get("mode"),
            "route": summary.get("route"),
            "success": _is_success(pair),
            "terminal_done": workload.get("terminal_done") is True,
            "first_delta_latency_ms": summary.get("first_delta_latency_ms"),
            "buffering_saved_ms": summary.get("buffering_saved_ms"),
            "total_latency_ms": summary.get("total_latency_ms"),
            "model_latency_ms": summary.get("model_latency_ms"),
            "output_tokens": summary.get("output_tokens"),
            "delta_count": workload.get("delta_count"),
            "client_first_delta_latency_ms": workload.get(
                "client_first_delta_latency_ms"
            ),
            "client_total_latency_ms": workload.get("client_total_latency_ms"),
            "answer_chars": workload.get("answer_chars"),
            "estimated_cost_cny": cost,
        })

    all_pass = (
        success_gate
        and route_counts_pass
        and terminal_gate
        and pair_gate
        and streaming_mode_gate
        and target_routes_match
        and client_metrics_complete
        and all(item["pass"] for item in ttft.values())
        and all(item["pass"] for item in client_ttft.values())
        and all_routes_total_gate
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
            "success_by_phase": success_by_phase,
            "success_100_of_100": success_gate,
            "route_counts": route_counts,
            "route_counts_pass": route_counts_pass,
            "terminal_and_finalization": terminal_gate,
            "pairs_complete": pair_gate,
            "baseline_mode_verified": baseline_mode_gate,
            "after_mode_verified": after_mode_gate,
            "streaming_mode_verified": streaming_mode_gate,
            "target_routes_match": target_routes_match,
            "client_metrics_complete": client_metrics_complete,
            "ttft": ttft,
            "client_ttft": client_ttft,
            "total_latency_p50_within_15_percent": total_gate,
            "all_routes_total_latency_p50_within_15_percent": (
                all_routes_total_gate
            ),
            "all_pass": all_pass,
        },
        "paired_differences": paired_differences,
        "estimated_cost_cny": {
            "coverage": len(request_costs),
            "total": sum(request_costs),
            "by_phase_route": {
                phase: {
                    route: metrics[phase][route]["estimated_cost_cny"]["total"]
                    for route in DAY3_ROUTES
                }
                for phase in ("baseline", "after")
            },
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
        (
            "Conclusion: **A**. Every required safety and performance gate passed."
            if analysis["conclusion"] == "A"
            else "Conclusion: **B**. At least one comparison gate failed."
        ),
        "",
        "## Explicit gates",
        "",
        f"- 50/50 baseline success: `{gates['success_by_phase']['baseline_50_of_50']}`",
        f"- 50/50 after success: `{gates['success_by_phase']['after_50_of_50']}`",
        f"- 100/100 success: `{gates['success_100_of_100']}`",
        f"- Exact route counts: `{gates['route_counts_pass']}`",
        f"- Complete baseline/after pairs: `{gates['pairs_complete']}`",
        f"- Streaming mode verified: `{gates['streaming_mode_verified']}`",
        f"- Target routes match: `{gates['target_routes_match']}`",
        f"- Client metrics complete: `{gates['client_metrics_complete']}`",
        "- Every measured success ended with `done` after the final render "
        f"invariant: `{gates['terminal_and_finalization']}`",
    ]
    for route in ("product_search", "knowledge"):
        item = gates["ttft"][route]
        lines.append(
            f"- {route} TTFT >=30% and >=500 ms: `{item['pass']}` "
            f"({item['absolute_reduction_ms']} ms, {item['relative_reduction']})"
        )
        client = gates["client_ttft"][route]
        lines.append(
            f"- {route} client TTFT >=30% and >=500 ms: "
            f"`{client['pass']}` ({client['absolute_reduction_ms']} ms, "
            f"{client['relative_reduction']})"
        )
    for route in DAY3_ROUTES:
        lines.append(
            f"- {route} total-latency P50 <=15% regression: "
            f"`{gates['total_latency_p50_within_15_percent'][route]}`"
        )
    lines.append(
        "- All routes total-latency P50 <=15% regression: "
        f"`{gates['all_routes_total_latency_p50_within_15_percent']}`"
    )
    lines.extend((
        "",
        "## Route metrics",
        "",
        "| Phase | Route | Success | Failure | Server TTFT P50/P95 ms | Client TTFT P50/P95 ms | Client total P50/P95 ms | Total P50/P95 ms | Model P50/P95 ms | Output tokens P50/P95 | Estimated cost CNY |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ))
    for phase in ("baseline", "after"):
        for route in DAY3_ROUTES:
            item = metrics[phase][route]
            first = item["first_delta_latency_ms"]
            client_first = item["client_first_delta_latency_ms"]
            client_total = item["client_total_latency_ms"]
            total = item["total_latency_ms"]
            model = item["model_latency_ms"]
            output = item["output_tokens"]
            cost = item["estimated_cost_cny"]
            lines.append(
                f"| {phase} | {route} | {item['successful']} | "
                f"{item['failed']} | "
                f"{first['p50']} / {first['p95']} | "
                f"{client_first['p50']} / {client_first['p95']} | "
                f"{client_total['p50']} / {client_total['p95']} | "
                f"{total['p50']} / {total['p95']} | "
                f"{model['p50']} / {model['p95']} | "
                f"{output['p50']} / {output['p95']} | "
                f"{cost['total']:.6f} |"
            )
    lines.extend((
        "",
        "## Paired latency differences",
        "",
        "Positive reductions mean streaming was faster; positive total/token changes mean streaming was larger or slower.",
        "",
        "| Route | Matched | Server TTFT reduction P50/P95 ms | Client TTFT reduction P50/P95 ms | Total change P50/P95 ms | Output token change P50/P95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ))
    for route in DAY3_ROUTES:
        item = analysis["paired_differences"][route]
        server = item["server_first_delta_reduction_ms"]
        client = item["client_first_delta_reduction_ms"]
        total = item["total_latency_change_ms"]
        tokens = item["output_tokens_change"]
        lines.append(
            f"| {route} | {item['matched']} | "
            f"{server['p50']} / {server['p95']} | "
            f"{client['p50']} / {client['p95']} | "
            f"{total['p50']} / {total['p95']} | "
            f"{tokens['p50']} / {tokens['p95']} |"
        )
    lines.extend((
        "",
        "## Estimated cost and limitations",
        "",
        f"Estimated measured cost: CNY {analysis['estimated_cost_cny']['total']:.6f}.",
        "",
        "Repeated test cases may increase prompt cache reuse; observed estimated cost reflects the measured cache behavior of this representative synthetic workload.",
        "",
        "Client timing measures receipt of NDJSON events by this local HTTP test client; it is not browser rendering latency or a production SLO.",
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
