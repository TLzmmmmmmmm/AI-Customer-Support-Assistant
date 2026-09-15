"""Join and analyze the two Week 4 Day 3 streaming phases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from performance.analysis import join_attempts, load_summaries, load_workload_rows
from performance.day3_streaming import analyze_day3_phases, render_day3_markdown


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--baseline-summaries", type=Path, required=True)
    parser.add_argument("--after-results", type=Path, required=True)
    parser.add_argument("--after-summaries", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _phase(results: Path, summaries: Path, expected: str):
    header, attempts = load_workload_rows(results)
    if header.get("phase") != expected:
        raise ValueError(f"expected {expected} phase")
    summary_rows, issues = load_summaries(summaries)
    joined = join_attempts(attempts, summary_rows)
    if issues:
        raise ValueError(f"{expected} summaries contain malformed lines")
    for key in ("missing_request_id", "missing_summary", "duplicate_summary"):
        if joined[key]:
            raise ValueError(f"{expected} join has {key}")
    return joined["matched"], {
        "attempts": len(attempts),
        "matched": len(joined["matched"]),
        "unrelated_summary_count": joined["unrelated_summary_count"],
    }


def _write(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    baseline, baseline_join = _phase(
        args.baseline_results, args.baseline_summaries, "baseline"
    )
    after, after_join = _phase(args.after_results, args.after_summaries, "after")
    analysis = analyze_day3_phases(baseline, after)
    analysis["telemetry_join"] = {
        "baseline": baseline_join,
        "after": after_join,
    }
    _write(
        args.json_output,
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        overwrite=args.overwrite,
    )
    _write(
        args.markdown_output,
        render_day3_markdown(analysis),
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "conclusion": analysis["conclusion"],
        "all_gates_pass": analysis["gates"]["all_pass"],
        "successful": analysis["samples"]["successful"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
