from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from knowledge_pipeline.chunking import build_and_write_chunks
from knowledge_pipeline.core import BuildError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build validated structure-aware knowledge chunks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPOSITORY_ROOT / "knowledge" / "documents.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "knowledge" / "chunks.jsonl",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        stats = build_and_write_chunks(args.input, args.output)
    except BuildError as error:
        print(str(error), file=sys.stderr)
        return 1

    print(f"Documents processed: {stats.total_documents}")
    print(f"Chunks generated: {stats.total_chunks}")
    print(f"Average chunk length (characters): {stats.average_characters:.2f}")
    print(f"Median chunk length (characters): {stats.median_characters:.2f}")
    print(f"Minimum chunk length (characters): {stats.minimum_characters}")
    print(f"Maximum chunk length (characters): {stats.maximum_characters}")
    for document_id, count in stats.chunks_per_document.items():
        print(f"- {document_id}: {count}")
    for type_, count in stats.chunks_per_type.items():
        print(f"- {type_}: {count}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
