from __future__ import annotations

import argparse
import hashlib
import os
import sys
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from knowledge_pipeline.retrieval import (
    DashScopeCredentials,
    DashScopeEmbeddingProvider,
    EmbeddingConfig,
    ExactEntityResolver,
    NumpyExactVectorIndex,
    RetrievalError,
    RetrievalEvaluationError,
    Retriever,
    evaluate_retrieval,
    load_chunks,
    load_evaluation_suite,
    load_vector_records,
    serialize_evaluation_result,
    validate_records_against_chunks,
    validate_suite_against_chunks,
)


def create_embedding_provider(
    config: EmbeddingConfig,
    credentials: DashScopeCredentials,
) -> DashScopeEmbeddingProvider:
    return DashScopeEmbeddingProvider(config, credentials)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or explicitly run the fixed retrieval evaluation."
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=REPOSITORY_ROOT / "eval" / "retrieval_v1_1.json",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=REPOSITORY_ROOT / "knowledge" / "chunks.jsonl",
    )
    parser.add_argument(
        "--vectors",
        type=Path,
        default=REPOSITORY_ROOT / "knowledge" / "vector_records.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "eval" / "retrieval_v1_1_results.json",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Call the paid query embedding API for every fixed query.",
    )
    return parser.parse_args(argv)


def _validate_snapshot(path: Path, expected_hash: str) -> None:
    try:
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise RetrievalEvaluationError(
            f"failed to hash chunk artifact {path}: {error}"
        ) from error
    if actual_hash != expected_hash:
        raise RetrievalEvaluationError(
            "evaluation suite was created for a different chunk snapshot"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(REPOSITORY_ROOT / ".env")
    try:
        if args.top_k <= 0:
            raise RetrievalEvaluationError("top_k must be positive")
        config = EmbeddingConfig.from_mapping(os.environ)
        suite = load_evaluation_suite(args.suite)
        chunks = load_chunks(args.chunks)
        validate_suite_against_chunks(suite, chunks)
        _validate_snapshot(args.chunks, suite.chunk_snapshot_sha256)

        estimated_tokens = sum(
            max(1, len(case.query) * 2) for case in suite.cases
        )
        estimated_cost = (
            Decimal(estimated_tokens)
            / Decimal(1000)
            * config.price_yuan_per_1k_tokens
        )
        print(f"Mode: {'execute' if args.execute else 'plan only'}")
        print(f"Queries: {len(suite.cases)}")
        print(f"Top-K: {args.top_k}")
        print(f"Estimated query tokens (conservative): {estimated_tokens}")
        print(f"Estimated query cost (CNY): {estimated_cost:.8f}")
        if not args.execute:
            print("No API request was made. Add --execute after reviewing cost.")
            return 0

        records = load_vector_records(args.vectors)
        validate_records_against_chunks(records, chunks, config)
        index = NumpyExactVectorIndex(records)
        resolver = ExactEntityResolver.from_records(records)
        credentials = DashScopeCredentials.from_mapping(os.environ)
        provider = create_embedding_provider(config, credentials)
        retriever = Retriever(
            embedding_provider=provider,
            vector_index=index,
            entity_resolver=resolver,
            default_top_k=args.top_k,
        )
        result = evaluate_retrieval(
            suite,
            retriever,
            chunks,
            top_k=args.top_k,
        )
        payload = serialize_evaluation_result(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    except RetrievalError as error:
        print(f"Retrieval evaluation failed: {error}", file=sys.stderr)
        return 1

    metrics = result.metrics
    print(f"Hit@1: {metrics.hit_at_1:.6f}")
    print(f"Hit@3: {metrics.hit_at_3:.6f}")
    print(f"Hit@5: {metrics.hit_at_5:.6f}")
    print(f"MRR: {metrics.mean_reciprocal_rank:.6f}")
    print(
        "Expected chunk Recall@5: "
        f"{metrics.expected_chunk_recall_at_5:.6f}"
    )
    print(
        "Expected parent Recall@5: "
        f"{metrics.expected_parent_recall_at_5:.6f}"
    )
    print(f"Entity accuracy: {metrics.entity_accuracy:.6f}")
    print(
        "Complete multi-source recall: "
        f"{metrics.complete_multi_source_recall:.6f}"
    )
    for failure in result.failures:
        print(
            f"Failure: {failure.case_id} [{failure.kind}] "
            f"{failure.message}"
        )
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
