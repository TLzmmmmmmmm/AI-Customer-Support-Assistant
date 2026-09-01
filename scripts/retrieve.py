from __future__ import annotations

import argparse
import json
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
    EmbeddingConfigurationError,
    ExactEntityResolver,
    NumpyExactVectorIndex,
    RetrievalConfig,
    RetrievalError,
    Retriever,
    format_debug_results,
    load_chunks,
    load_vector_records,
    validate_records_against_chunks,
)


def create_embedding_provider(
    config: EmbeddingConfig,
    credentials: DashScopeCredentials,
) -> DashScopeEmbeddingProvider:
    return DashScopeEmbeddingProvider(config, credentials)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve Top-K chunks without generation."
    )
    parser.add_argument("query", help="One retrieval query.")
    parser.add_argument("--top-k", type=int)
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
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(REPOSITORY_ROOT / ".env")
    try:
        embedding_config = EmbeddingConfig.from_mapping(os.environ)
        retrieval_config = RetrievalConfig.from_mapping(os.environ)
        top_k = retrieval_config.top_k if args.top_k is None else args.top_k
        if top_k <= 0:
            raise EmbeddingConfigurationError("top_k must be positive")
        if not args.query.strip():
            raise EmbeddingConfigurationError("query must be non-blank")

        chunks = load_chunks(args.chunks)
        records = load_vector_records(args.vectors)
        validate_records_against_chunks(records, chunks, embedding_config)

        index = NumpyExactVectorIndex(records)
        resolver = ExactEntityResolver.from_records(records)
        credentials = DashScopeCredentials.from_mapping(os.environ)
        provider = create_embedding_provider(embedding_config, credentials)
        retriever = Retriever(
            embedding_provider=provider,
            vector_index=index,
            entity_resolver=resolver,
            default_top_k=retrieval_config.top_k,
        )
        results = retriever.retrieve(args.query, top_k=top_k)
    except RetrievalError as error:
        print(f"Retrieval failed: {error}", file=sys.stderr)
        return 1

    if args.debug:
        print(format_debug_results(args.query, results))
    else:
        print(json.dumps(
            [result.model_dump(mode="json") for result in results],
            ensure_ascii=False,
            separators=(",", ":"),
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
