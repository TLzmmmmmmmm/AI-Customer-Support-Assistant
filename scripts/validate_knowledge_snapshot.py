from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from knowledge_pipeline.chunking import load_documents, validate_chunks
from knowledge_pipeline.core import BuildError, validate_documents
from knowledge_pipeline.retrieval import (
    EmbeddingConfig,
    RetrievalError,
    load_chunks,
    load_vector_records,
    validate_records_against_chunks,
)


DEFAULT_BASE_URL = "https://www.shengborun.com"


@dataclass(frozen=True)
class SnapshotValidation:
    document_count: int
    chunk_count: int
    vector_count: int
    document_sha256: str
    chunk_sha256: str
    vector_sha256: str
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_snapshot(
    *,
    documents_path: Path,
    chunks_path: Path,
    vectors_path: Path,
    base_url: str,
    embedding_config: EmbeddingConfig,
) -> SnapshotValidation:
    documents = load_documents(documents_path)
    validate_documents(documents, base_url)
    chunks = load_chunks(chunks_path)
    validate_chunks(chunks, documents)
    vectors = load_vector_records(vectors_path)
    validate_records_against_chunks(vectors, chunks, embedding_config)
    return SnapshotValidation(
        document_count=len(documents),
        chunk_count=len(chunks),
        vector_count=len(vectors),
        document_sha256=_sha256(documents_path),
        chunk_sha256=_sha256(chunks_path),
        vector_sha256=_sha256(vectors_path),
        embedding_provider=embedding_config.provider,
        embedding_model=embedding_config.model,
        embedding_dimensions=embedding_config.dimensions,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a complete local knowledge/vector snapshot."
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=REPOSITORY_ROOT / "knowledge" / "documents.jsonl",
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
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(REPOSITORY_ROOT / ".env")
    try:
        config = EmbeddingConfig.from_mapping(os.environ)
        result = validate_snapshot(
            documents_path=args.documents,
            chunks_path=args.chunks,
            vectors_path=args.vectors,
            base_url=args.base_url,
            embedding_config=config,
        )
    except (BuildError, RetrievalError, OSError, UnicodeError) as error:
        print(f"Knowledge snapshot validation: FAIL\n{error}", file=sys.stderr)
        return 1

    print("Knowledge snapshot validation: PASS")
    print(f"Documents: {result.document_count}")
    print(f"Chunks: {result.chunk_count}")
    print(f"Vector records: {result.vector_count}")
    print(
        "Embedding: "
        f"{result.embedding_provider} / {result.embedding_model} / "
        f"{result.embedding_dimensions}"
    )
    print(f"Documents SHA-256: {result.document_sha256}")
    print(f"Chunks SHA-256: {result.chunk_sha256}")
    print(f"Vectors SHA-256: {result.vector_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
