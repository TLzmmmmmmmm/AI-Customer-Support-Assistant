from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from knowledge_pipeline.models import KnowledgeChunk

from .models import (
    RetrievalEvaluationError,
    RetrievalEvaluationResult,
    RetrievalEvaluationSuite,
    RetrievalFailure,
    RetrievalMetrics,
)
from .retriever import Retriever


def load_evaluation_suite(path: Path) -> RetrievalEvaluationSuite:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return RetrievalEvaluationSuite.model_validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as error:
        raise RetrievalEvaluationError(
            f"failed to load retrieval evaluation suite {path}: {error}"
        ) from error


def validate_suite_against_chunks(
    suite: RetrievalEvaluationSuite,
    chunks: Sequence[KnowledgeChunk],
) -> None:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(chunk_by_id) != len(chunks):
        raise RetrievalEvaluationError("chunk collection contains duplicate ids")
    parent_ids = {chunk.parent_document_id for chunk in chunks}
    for case in suite.cases:
        missing_chunks = sorted(set(case.expected_chunk_ids) - set(chunk_by_id))
        missing_parents = sorted(
            set(case.expected_parent_document_ids) - parent_ids
        )
        missing_entities = sorted(set(case.expected_entity_ids) - parent_ids)
        if missing_chunks or missing_parents or missing_entities:
            raise RetrievalEvaluationError(
                f"evaluation case {case.id} has missing references: "
                f"chunks={missing_chunks}, parents={missing_parents}, "
                f"entities={missing_entities}"
            )
        expected_chunk_parents = {
            chunk_by_id[chunk_id].parent_document_id
            for chunk_id in case.expected_chunk_ids
        }
        if not expected_chunk_parents.issubset(
            set(case.expected_parent_document_ids)
        ):
            raise RetrievalEvaluationError(
                f"evaluation case {case.id} omits an expected chunk parent"
            )


def _mean(values: Sequence[float], *, empty: float = 1.0) -> float:
    return sum(values) / len(values) if values else empty


def evaluate_retrieval(
    suite: RetrievalEvaluationSuite,
    retriever: Retriever,
    chunks: Sequence[KnowledgeChunk],
    top_k: int = 5,
) -> RetrievalEvaluationResult:
    if top_k <= 0:
        raise RetrievalEvaluationError("evaluation top_k must be positive")
    validate_suite_against_chunks(suite, chunks)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    first_ranks: list[int | None] = []
    chunk_recalls: list[float] = []
    parent_recalls: list[float] = []
    entity_scores: list[float] = []
    multi_source_scores: list[float] = []
    failures: list[RetrievalFailure] = []

    for case in suite.cases:
        results = retriever.retrieve(case.query, top_k=top_k)
        retrieved_chunk_ids = [item.chunk_id for item in results[:top_k]]
        retrieved_parent_ids = {
            item.parent_document_id for item in results[:top_k]
        }
        expected_chunks = set(case.expected_chunk_ids)
        relevant_ranks = [
            rank
            for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1)
            if chunk_id in expected_chunks
        ]
        first_rank = min(relevant_ranks) if relevant_ranks else None
        first_ranks.append(first_rank)

        if case.match_requirement == "all_groups":
            group_hits: list[float] = []
            parent_group_hits: list[float] = []
            for group in case.relevance_groups:
                group_chunk_ids = set(group.acceptable_chunk_ids)
                group_hits.append(
                    float(bool(group_chunk_ids & set(retrieved_chunk_ids)))
                )
                group_parent_ids = {
                    chunk_by_id[chunk_id].parent_document_id
                    for chunk_id in group.acceptable_chunk_ids
                }
                parent_group_hits.append(
                    float(bool(group_parent_ids & retrieved_parent_ids))
                )
            chunk_recall = _mean(group_hits, empty=0.0)
            parent_recall = _mean(parent_group_hits, empty=0.0)
            complete = all(value == 1.0 for value in group_hits)
        else:
            chunk_recall = len(
                expected_chunks & set(retrieved_chunk_ids)
            ) / len(expected_chunks)
            expected_parents = set(case.expected_parent_document_ids)
            parent_recall = len(
                expected_parents & retrieved_parent_ids
            ) / len(expected_parents)
            complete = expected_chunks.issubset(set(retrieved_chunk_ids))
        chunk_recalls.append(chunk_recall)
        parent_recalls.append(parent_recall)

        if first_rank is None:
            failures.append(RetrievalFailure(
                case_id=case.id,
                source_case_id=case.source_case_id,
                kind="missing_top_k",
                message=f"No expected evidence appeared in Top-{top_k}.",
            ))
        elif first_rank in {4, 5}:
            failures.append(RetrievalFailure(
                case_id=case.id,
                source_case_id=case.source_case_id,
                kind="late_rank",
                message=f"First expected evidence appeared at rank {first_rank}.",
            ))

        if case.match_requirement in {"all", "all_groups"}:
            multi_source_scores.append(float(complete))
            if not complete:
                failures.append(RetrievalFailure(
                    case_id=case.id,
                    source_case_id=case.source_case_id,
                    kind="incomplete_multi_source",
                    message="Not all required evidence sources/groups were retrieved.",
                ))

        if case.expected_entity_ids:
            matched_entities = {
                entity_id
                for item in results[:top_k]
                for entity_id in item.matched_entity_ids
            }
            entity_complete = set(case.expected_entity_ids).issubset(
                matched_entities
            )
            entity_scores.append(float(entity_complete))
            if not entity_complete:
                failures.append(RetrievalFailure(
                    case_id=case.id,
                    source_case_id=case.source_case_id,
                    kind="missed_entity",
                    message="One or more expected product entities were not resolved.",
                ))

    total = len(suite.cases)
    reciprocal_ranks = [
        (1.0 / rank) if rank is not None else 0.0 for rank in first_ranks
    ]
    metrics = RetrievalMetrics(
        total_cases=total,
        hit_at_1=sum(rank == 1 for rank in first_ranks) / total,
        hit_at_3=sum(
            rank is not None and rank <= 3 for rank in first_ranks
        ) / total,
        hit_at_5=sum(
            rank is not None and rank <= 5 for rank in first_ranks
        ) / total,
        mean_reciprocal_rank=_mean(reciprocal_ranks, empty=0.0),
        expected_chunk_recall_at_5=_mean(chunk_recalls, empty=0.0),
        expected_parent_recall_at_5=_mean(parent_recalls, empty=0.0),
        entity_accuracy=_mean(entity_scores),
        complete_multi_source_recall=_mean(multi_source_scores),
    )
    return RetrievalEvaluationResult(
        schema_version="1.0",
        suite_id=suite.suite_id,
        top_k=top_k,
        metrics=metrics,
        failures=failures,
    )


def serialize_evaluation_result(result: RetrievalEvaluationResult) -> bytes:
    return (
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


__all__ = [
    "evaluate_retrieval",
    "load_evaluation_suite",
    "serialize_evaluation_result",
    "validate_suite_against_chunks",
]
