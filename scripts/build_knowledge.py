from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from knowledge_pipeline.chunking import build_and_write_chunks
from knowledge_pipeline.core import BuildError, build_and_write
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
from scripts.validate_knowledge_snapshot import validate_snapshot


DEFAULT_BASE_URL = "https://www.shengborun.com"


class _ReuseOnlyProvider:
    """Supplies config when a complete build needs no external API call."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def embed_documents(self, texts):
        raise RuntimeError("reuse-only provider cannot create embeddings")


def create_embedding_provider(
    config: EmbeddingConfig,
    credentials: DashScopeCredentials,
) -> DashScopeEmbeddingProvider:
    return DashScopeEmbeddingProvider(config, credentials)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a complete validated knowledge snapshot in staging, then "
            "activate it only with --execute."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPOSITORY_ROOT / "knowledge" / "source",
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
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Call the paid embedding API for changed chunks when needed, "
            "validate, and activate the staged snapshot."
        ),
    )
    return parser.parse_args(argv)


def _print_plan(*, document_count: int, chunk_count: int, plan, execute: bool) -> None:
    print(f"Mode: {'execute' if execute else 'plan only'}")
    print(f"Documents: {document_count}")
    print(f"Chunks: {chunk_count}")
    print(f"Reused: {len(plan.reused_chunk_ids)}")
    print(f"To embed: {len(plan.to_embed)}")
    print(f"Deleted: {len(plan.deleted_chunk_ids)}")
    print(f"Characters to embed: {plan.total_characters}")
    print(f"Estimated tokens (conservative): {plan.estimated_tokens}")
    print(f"Estimated cost (CNY): {plan.estimated_cost_yuan:.8f}")


def _write_replacement(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".activate.tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _activate_snapshot(
    staged_and_live_paths: tuple[tuple[Path, Path], ...],
) -> None:
    previous = {
        live: live.read_bytes() if live.is_file() else None
        for _staged, live in staged_and_live_paths
    }
    activated: list[Path] = []
    try:
        for staged, live in staged_and_live_paths:
            _write_replacement(live, staged.read_bytes())
            activated.append(live)
    except OSError as activation_error:
        rollback_errors: list[str] = []
        for live in reversed(activated):
            try:
                old_payload = previous[live]
                if old_payload is None:
                    live.unlink(missing_ok=True)
                else:
                    _write_replacement(live, old_payload)
            except OSError as rollback_error:
                rollback_errors.append(f"{live}: {rollback_error}")
        details = (
            f"; rollback errors: {', '.join(rollback_errors)}"
            if rollback_errors
            else ""
        )
        raise OSError(f"snapshot activation failed: {activation_error}{details}") from activation_error


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(REPOSITORY_ROOT / ".env")

    try:
        config = EmbeddingConfig.from_mapping(os.environ)
        with tempfile.TemporaryDirectory(prefix="knowledge-build-") as directory:
            staging_root = Path(directory)
            staged_documents = staging_root / "documents.jsonl"
            staged_chunks = staging_root / "chunks.jsonl"
            staged_vectors = staging_root / "vector_records.jsonl"

            document_counts = build_and_write(
                args.source_root,
                staged_documents,
                args.base_url,
            )
            chunk_stats = build_and_write_chunks(
                staged_documents,
                staged_chunks,
            )
            chunks = load_chunks(staged_chunks)
            existing_vectors = load_vector_records(args.vectors, missing_ok=True)
            plan = plan_vector_build(chunks, existing_vectors, config)
            _print_plan(
                document_count=sum(document_counts.values()),
                chunk_count=chunk_stats.total_chunks,
                plan=plan,
                execute=args.execute,
            )

            if not args.execute:
                print("No API request was made. Add --execute after reviewing cost.")
                print("No live artifact was changed.")
                return 0

            if plan.to_embed:
                credentials = DashScopeCredentials.from_mapping(os.environ)
                provider = create_embedding_provider(config, credentials)
            else:
                provider = _ReuseOnlyProvider(config)
            build_stats = execute_vector_build(plan, provider, staged_vectors)

            validation = validate_snapshot(
                documents_path=staged_documents,
                chunks_path=staged_chunks,
                vectors_path=staged_vectors,
                base_url=args.base_url,
                embedding_config=config,
            )
            print("Knowledge snapshot validation: PASS")
            print(f"Documents SHA-256: {validation.document_sha256}")
            print(f"Chunks SHA-256: {validation.chunk_sha256}")
            print(f"Vectors SHA-256: {validation.vector_sha256}")

            _activate_snapshot((
                (staged_documents, args.documents),
                (staged_chunks, args.chunks),
                (staged_vectors, args.vectors),
            ))
            print(f"Embedded: {build_stats.embedded_count}")
            print("Snapshot activated")
            print(f"Documents output: {args.documents}")
            print(f"Chunks output: {args.chunks}")
            print(f"Vectors output: {args.vectors}")
    except (BuildError, RetrievalError, OSError, UnicodeError) as error:
        print(f"Knowledge build failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
