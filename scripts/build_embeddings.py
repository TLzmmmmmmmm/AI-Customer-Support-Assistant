from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from knowledge_pipeline.retrieval import (
    DashScopeCredentials,
    DashScopeEmbeddingProvider,
    EmbeddingConfig,
    RetrievalError,
    execute_vector_build,
    load_chunks,
    load_vector_records,
    plan_vector_build,
)


def create_embedding_provider(
    config: EmbeddingConfig,
    credentials: DashScopeCredentials,
) -> DashScopeEmbeddingProvider:
    return DashScopeEmbeddingProvider(config, credentials)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or explicitly execute the local embedding build."
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=REPOSITORY_ROOT / "knowledge" / "chunks.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "knowledge" / "vector_records.jsonl",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Call the configured paid embedding API and persist vectors.",
    )
    return parser.parse_args(argv)


def _print_plan(plan, *, execute: bool) -> None:
    print(f"Mode: {'execute' if execute else 'plan only'}")
    print(f"Total chunks: {len(plan.chunks)}")
    print(f"Reused: {len(plan.reused_chunk_ids)}")
    print(f"To embed: {len(plan.to_embed)}")
    print(f"Deleted: {len(plan.deleted_chunk_ids)}")
    print(f"Characters to embed: {plan.total_characters}")
    print(f"Estimated tokens (conservative): {plan.estimated_tokens}")
    print(f"Estimated cost (CNY): {plan.estimated_cost_yuan:.8f}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(REPOSITORY_ROOT / ".env")
    try:
        config = EmbeddingConfig.from_mapping(os.environ)
        chunks = load_chunks(args.chunks)
        existing = load_vector_records(args.output, missing_ok=True)
        plan = plan_vector_build(chunks, existing, config)
        _print_plan(plan, execute=args.execute)
        if not args.execute:
            print("No API request was made. Add --execute after reviewing cost.")
            return 0

        credentials = DashScopeCredentials.from_mapping(os.environ)
        provider = create_embedding_provider(config, credentials)
        stats = execute_vector_build(plan, provider, args.output)
    except RetrievalError as error:
        print(f"Embedding build failed: {error}", file=sys.stderr)
        return 1

    usage = (
        str(stats.actual_input_tokens)
        if stats.actual_input_tokens is not None
        else "unavailable"
    )
    print(f"Actual input tokens: {usage}")
    print(f"Embedded: {stats.embedded_count}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
