"""Run one controlled Week 4 Day 3 HTTP phase without retries."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from performance.day3_streaming import (
    DAY3_COST_CEILING_CNY,
    DAY3_PHASE_REQUESTS,
    build_day3_schedule,
    day2_budget_screening_projection,
    load_day3_manifest,
    parse_success_ndjson,
)
from scripts.run_week4_day2_workload import check_health, post_chat


DEFAULT_MANIFEST = ROOT / "performance" / "day3_streaming_workload_v1.json"
DEFAULT_DAY2_ANALYSIS = (
    ROOT / "performance" / "results" / "week4-day2-analysis.json"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_phase(
    manifest: dict[str, object],
    *,
    phase: str,
    base_url: str,
    output: Path,
    timeout_seconds: float,
) -> dict[str, int]:
    if output.exists():
        raise FileExistsError(f"workload output already exists: {output}")
    schedule = build_day3_schedule(manifest, phase)
    if len(schedule) != DAY3_PHASE_REQUESTS:
        raise ValueError(
            f"Day 3 phase must contain exactly {DAY3_PHASE_REQUESTS} requests"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"attempted": 0, "successful": 0, "failed": 0}
    interval = float(manifest["interval_seconds"])
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({
            "record_type": "run",
            "schema_version": manifest["schema_version"],
            "phase": phase,
            "started_at": _now(),
            "base_url": base_url.rstrip("/"),
            "seed": manifest["seed"],
            "repeat_count": manifest["repeat_count"],
            "interval_seconds": interval,
        }, ensure_ascii=False) + "\n")
        stream.flush()
        for index, item in enumerate(schedule):
            if index:
                time.sleep(interval)
            started_at = _now()
            row = {
                "record_type": "attempt",
                "workload_id": item["workload_id"],
                "phase": phase,
                "case_id": item["case_id"],
                "target_route": item["target_route"],
                "execution_index": item["execution_index"],
                "started_at": started_at,
                "completed_at": None,
                "http_status": None,
                "request_id": None,
                "outcome": "transport_error",
                "terminal_done": False,
                "delta_count": 0,
            }
            try:
                result = post_chat(
                    base_url,
                    [dict(message) for message in item["messages"]],
                    timeout_seconds,
                )
                row["http_status"] = result.status_code
                row["request_id"] = next((
                    value for name, value in result.headers.items()
                    if name.casefold() == "x-request-id"
                ), None)
                if result.status_code == 200:
                    parsed = parse_success_ndjson(result.body)
                    row["terminal_done"] = True
                    row["delta_count"] = parsed["delta_count"]
                    row["outcome"] = "success"
                else:
                    row["outcome"] = "http_error"
            except ValueError:
                row["outcome"] = "ndjson_error"
            except Exception as error:
                row["error_type"] = type(error).__name__
            row["completed_at"] = _now()
            counts["attempted"] += 1
            if row["outcome"] == "success":
                counts["successful"] += 1
            else:
                counts["failed"] += 1
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if row["outcome"] != "success":
                break
    return counts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--phase", choices=("baseline", "after"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--day2-analysis", type=Path, default=DEFAULT_DAY2_ANALYSIS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-paid-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = load_day3_manifest(args.manifest)
    schedule = build_day3_schedule(manifest, args.phase)
    projection = day2_budget_screening_projection(args.day2_analysis)
    preflight = {
        "phase": args.phase,
        "planned_requests": len(schedule),
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
    check_health(args.base_url, args.timeout_seconds)
    counts = run_phase(
        manifest,
        phase=args.phase,
        base_url=args.base_url,
        output=args.output,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps({**preflight, **counts}))
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
