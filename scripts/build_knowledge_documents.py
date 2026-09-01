from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from knowledge_pipeline.core import BuildError, build_and_write


DEFAULT_BASE_URL = "https://www.shengborun.com"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build validated pre-chunk knowledge documents."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPOSITORY_ROOT / "knowledge" / "source",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "knowledge" / "documents.jsonl",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        counts = build_and_write(args.source_root, args.output, args.base_url)
    except BuildError as error:
        print(str(error), file=sys.stderr)
        return 1

    total = sum(counts.values())
    print(f"Documents generated: {total}")
    for type_ in (
        "catalog", "product", "solution", "support", "company", "contact"
    ):
        print(f"- {type_}: {counts[type_]}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
