"""Run a paired buffered-versus-streaming latency comparison."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEEPSEEK_MODEL
from performance.day3_streaming import (
    DAY3_COST_CEILING_CNY,
    DAY3_PHASE_REQUESTS,
    DAY3_TOTAL_REQUESTS,
    build_day3_schedule,
    day2_budget_screening_projection,
    load_day3_manifest,
    parse_success_ndjson,
)
from scripts.run_week4_day2_workload import check_health


DEFAULT_MANIFEST = ROOT / "performance" / "day3_streaming_workload_v1.json"
DEFAULT_DAY2_ANALYSIS = (
    ROOT / "performance" / "results" / "week4-day2-analysis.json"
)


@dataclass(frozen=True)
class TimedHttpResult:
    status_code: int
    headers: Mapping[str, str]
    body: str
    headers_latency_ms: float
    first_delta_latency_ms: float | None
    total_latency_ms: float


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def post_chat_streaming(
    base_url: str,
    messages: list[dict[str, str]],
    timeout_seconds: float,
) -> TimedHttpResult:
    request = Request(
        base_url.rstrip("/") + "/api/chat-stream",
        data=json.dumps({"messages": messages}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = time.monotonic()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            headers_latency_ms = (time.monotonic() - started_at) * 1000
            lines = []
            first_delta_latency_ms = None
            for raw_line in response:
                line = raw_line.decode("utf-8")
                lines.append(line)
                if first_delta_latency_ms is None and line.strip():
                    event = json.loads(line)
                    if event.get("type") == "delta":
                        first_delta_latency_ms = (
                            time.monotonic() - started_at
                        ) * 1000
            total_latency_ms = (time.monotonic() - started_at) * 1000
            return TimedHttpResult(
                status_code=response.status,
                headers=dict(response.headers.items()),
                body="".join(lines),
                headers_latency_ms=headers_latency_ms,
                first_delta_latency_ms=first_delta_latency_ms,
                total_latency_ms=total_latency_ms,
            )
    except HTTPError as error:
        headers_latency_ms = (time.monotonic() - started_at) * 1000
        body = error.read().decode("utf-8")
        return TimedHttpResult(
            status_code=error.code,
            headers=dict(error.headers.items()),
            body=body,
            headers_latency_ms=headers_latency_ms,
            first_delta_latency_ms=None,
            total_latency_ms=(time.monotonic() - started_at) * 1000,
        )


def _comparison_schedule(
    manifest: dict[str, object],
) -> list[tuple[str, tuple[str, str], dict[str, dict[str, object]]]]:
    by_phase = {
        phase: {
            (row["case_id"], row["execution_index"]): row
            for row in build_day3_schedule(manifest, phase)
        }
        for phase in ("baseline", "after")
    }
    keys = list(by_phase["baseline"])
    randomizer = random.Random(int(manifest["seed"]))
    randomizer.shuffle(keys)
    schedule = []
    for case_id, execution_index in keys:
        pair_id = f"{case_id}:{execution_index}"
        order = (
            ("baseline", "after")
            if randomizer.randrange(2) == 0
            else ("after", "baseline")
        )
        schedule.append((pair_id, order, {
            phase: by_phase[phase][(case_id, execution_index)]
            for phase in ("baseline", "after")
        }))
    return schedule


def _run_header(
    manifest: dict[str, object],
    *,
    phase: str,
    mode: str,
    base_url: str,
) -> dict[str, object]:
    return {
        "record_type": "run",
        "schema_version": manifest["schema_version"],
        "phase": phase,
        "mode": mode,
        "started_at": _now(),
        "base_url": base_url.rstrip("/"),
        "model": DEEPSEEK_MODEL,
        "seed": manifest["seed"],
        "repeat_count": manifest["repeat_count"],
        "interval_seconds": manifest["interval_seconds"],
        "planned_requests": DAY3_PHASE_REQUESTS,
    }


def _attempt(
    item: dict[str, object],
    *,
    pair_id: str,
    phase: str,
    mode: str,
    base_url: str,
    timeout_seconds: float,
) -> dict[str, object]:
    row = {
        "record_type": "attempt",
        "workload_id": item["workload_id"],
        "phase": phase,
        "mode": mode,
        "pair_id": pair_id,
        "case_id": item["case_id"],
        "target_route": item["target_route"],
        "execution_index": item["execution_index"],
        "started_at": _now(),
        "completed_at": None,
        "http_status": None,
        "request_id": None,
        "outcome": "transport_error",
        "terminal_done": False,
        "delta_count": 0,
        "answer_chars": None,
        "client_headers_latency_ms": None,
        "client_first_delta_latency_ms": None,
        "client_total_latency_ms": None,
    }
    try:
        result = post_chat_streaming(
            base_url,
            [dict(message) for message in item["messages"]],
            timeout_seconds,
        )
        row["http_status"] = result.status_code
        row["request_id"] = next((
            value for name, value in result.headers.items()
            if name.casefold() == "x-request-id"
        ), None)
        row["client_headers_latency_ms"] = result.headers_latency_ms
        row["client_first_delta_latency_ms"] = result.first_delta_latency_ms
        row["client_total_latency_ms"] = result.total_latency_ms
        if result.status_code == 200:
            parsed = parse_success_ndjson(result.body)
            row["terminal_done"] = True
            row["delta_count"] = parsed["delta_count"]
            row["answer_chars"] = len(parsed["answer"])
            row["outcome"] = "success"
        else:
            row["outcome"] = "http_error"
    except ValueError:
        row["outcome"] = "ndjson_error"
    except Exception as error:
        row["error_type"] = type(error).__name__
    row["completed_at"] = _now()
    return row


def run_comparison(
    manifest: dict[str, object],
    *,
    baseline_base_url: str,
    after_base_url: str,
    baseline_output: Path,
    after_output: Path,
    timeout_seconds: float,
) -> dict[str, int]:
    if baseline_base_url.rstrip("/") == after_base_url.rstrip("/"):
        raise ValueError("baseline and after must use different endpoints")
    for output in (baseline_output, after_output):
        if output.exists():
            raise FileExistsError(f"workload output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    schedule = _comparison_schedule(manifest)
    if len(schedule) != DAY3_PHASE_REQUESTS:
        raise ValueError(
            f"Day 3 comparison must contain {DAY3_PHASE_REQUESTS} pairs"
        )

    endpoints = {
        "baseline": ("buffered", baseline_base_url),
        "after": ("streaming", after_base_url),
    }
    counts = {"attempted": 0, "successful": 0, "failed": 0}
    interval = float(manifest["interval_seconds"])
    request_index = 0
    with (
        baseline_output.open("x", encoding="utf-8", newline="\n") as baseline,
        after_output.open("x", encoding="utf-8", newline="\n") as after,
    ):
        streams = {"baseline": baseline, "after": after}
        for phase in ("baseline", "after"):
            mode, base_url = endpoints[phase]
            streams[phase].write(json.dumps(_run_header(
                manifest,
                phase=phase,
                mode=mode,
                base_url=base_url,
            ), ensure_ascii=False) + "\n")
            streams[phase].flush()

        stop = False
        for pair_id, order, items in schedule:
            for phase in order:
                if request_index:
                    time.sleep(interval)
                mode, base_url = endpoints[phase]
                row = _attempt(
                    items[phase],
                    pair_id=pair_id,
                    phase=phase,
                    mode=mode,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                )
                request_index += 1
                counts["attempted"] += 1
                if row["outcome"] == "success":
                    counts["successful"] += 1
                else:
                    counts["failed"] += 1
                    stop = True
                streams[phase].write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )
                streams[phase].flush()
                if stop:
                    break
            if stop:
                break
    return counts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--baseline-base-url", default="http://127.0.0.1:8001"
    )
    parser.add_argument("--after-base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--after-output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--day2-analysis", type=Path, default=DEFAULT_DAY2_ANALYSIS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-paid-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = load_day3_manifest(args.manifest)
    projection = day2_budget_screening_projection(args.day2_analysis, manifest)
    preflight = {
        "planned_requests": DAY3_TOTAL_REQUESTS,
        "planned_pairs": DAY3_PHASE_REQUESTS,
        "budget_screening_projection_cny": projection,
        "cost_ceiling_cny": DAY3_COST_CEILING_CNY,
    }
    if projection >= DAY3_COST_CEILING_CNY:
        raise RuntimeError("budget screening projection reaches cost ceiling")
    if args.dry_run:
        print(json.dumps(preflight))
        return 0
    if not args.confirm_paid_run:
        raise ValueError("--confirm-paid-run is required")
    check_health(args.baseline_base_url, args.timeout_seconds)
    check_health(args.after_base_url, args.timeout_seconds)
    counts = run_comparison(
        manifest,
        baseline_base_url=args.baseline_base_url,
        after_base_url=args.after_base_url,
        baseline_output=args.baseline_output,
        after_output=args.after_output,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps({**preflight, **counts}))
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
