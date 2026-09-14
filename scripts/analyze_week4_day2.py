"""Join Week 4 Day 2 workload rows to production summaries and report them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from performance.analysis import (
    APPROVED_LABELS,
    analyze_join,
    join_attempts,
    load_summaries,
    load_workload_rows,
    render_markdown,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-results", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument(
        "--recommendation",
        choices=sorted(APPROVED_LABELS),
        required=True,
    )
    parser.add_argument("--rationale", action="append", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _write_outputs(
    json_output: Path,
    markdown_output: Path,
    json_text: str,
    markdown_text: str,
    *,
    overwrite: bool,
) -> None:
    outputs = (json_output, markdown_output)
    if not overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"output already exists: {existing[0]}")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = (
        json_output.with_name(json_output.name + ".tmp"),
        markdown_output.with_name(markdown_output.name + ".tmp"),
    )
    try:
        temporary[0].write_text(json_text, encoding="utf-8", newline="\n")
        temporary[1].write_text(markdown_text, encoding="utf-8", newline="\n")
        temporary[0].replace(json_output)
        temporary[1].replace(markdown_output)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if any(not item.strip() for item in args.rationale):
        raise ValueError("rationale must be non-empty")

    header, attempts = load_workload_rows(args.workload_results)
    summaries, issues = load_summaries(args.summaries)
    joined = join_attempts(attempts, summaries)
    analysis = analyze_join(joined)
    analysis["workload"] = header
    analysis["telemetry_join"]["malformed_summary_lines"] = len(issues)
    analysis["recommendation"] = {
        "category": args.recommendation,
        "label": APPROVED_LABELS[args.recommendation],
        "rationale": args.rationale,
    }
    json_text = json.dumps(
        analysis,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + "\n"
    markdown_text = render_markdown(analysis)
    _write_outputs(
        args.json_output,
        args.markdown_output,
        json_text,
        markdown_text,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "attempted": analysis["samples"]["attempted"],
        "successful": analysis["samples"]["successful"],
        "failed": analysis["samples"]["failed"],
        "recommendation": args.recommendation,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
