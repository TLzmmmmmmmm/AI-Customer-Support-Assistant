"""Simple Day 6 retrieval-only metrics over the existing production Retriever."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
import time

from evaluation.dataset import EvaluationCase
from knowledge_pipeline.retrieval.models import RetrievalResult
from knowledge_pipeline.retrieval.retriever import Retriever


def _check_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


def score_hits(
    expected_chunk_ids: Sequence[str],
    hits: Sequence[RetrievalResult],
    *,
    top_k: int,
) -> dict:
    """Use final retrieval order, not cosine re-sorting or document-level credit."""
    _check_top_k(top_k)
    selected = list(hits[:top_k])
    ids = [hit.chunk_id for hit in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate retrieved chunk IDs")
    if [hit.rank for hit in selected] != list(range(1, len(selected) + 1)):
        raise ValueError("retrieved ranks must match final result order")
    expected = set(expected_chunk_ids)
    if not expected:
        return {"hit_at_k": None, "recall_at_k": None, "first_relevant_rank": None}
    matched = expected.intersection(ids)
    first = next((hit.rank for hit in selected if hit.chunk_id in expected), None)
    return {"hit_at_k": int(bool(matched)), "recall_at_k": len(matched) / len(expected),
            "first_relevant_rank": first}


def run_retrieval_evaluation(
    cases: Sequence[EvaluationCase],
    retriever: Retriever,
    *,
    top_k: int = 5,
    metadata: dict,
) -> dict:
    """Query only cases with retrieval gold; retain sanitized per-case errors.

    Caller owns explicit split selection, snapshot checks and result persistence.
    No fixture is applied here: injection resistance belongs to generation eval.
    This function does not print queries/hits or call production logging.
    """
    _check_top_k(top_k)
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    empty_metrics = {"hit_at_k": None, "recall_at_k": None, "first_relevant_rank": None}
    for case in cases:
        row = {
            "case_id": case.id, "category": case.category, "split": case.split,
            "query": case.question, "expected_chunk_ids": list(case.expected_chunk_ids),
            "expected_document_ids": list(case.expected_document_ids),
            "expected_system_rule_ids": list(case.expected_system_rule_ids),
            "fixture_id": case.fixture_id, "fixture_applied": False,
            "retrieval_executed": False, "retrieved_chunk_ids": [], "hits": [],
            "retrieval_latency_seconds": None,
            "metrics": dict(empty_metrics), "error_type": None,
        }
        if not case.expected_chunk_ids:
            row["status"] = "excluded_no_gold"
        else:
            row["retrieval_executed"] = True
            retrieval_started = time.monotonic()
            try:
                hits = retriever.retrieve(case.question, top_k=top_k)[:top_k]
                row["metrics"] = score_hits(case.expected_chunk_ids, hits, top_k=top_k)
                row["retrieved_chunk_ids"] = [hit.chunk_id for hit in hits]
                row["hits"] = [hit.model_dump(mode="json") for hit in hits]
                row["status"] = "evaluated"
            except Exception as error:
                # Isolate each case, including malformed provider responses.
                # Process interrupts (BaseException) deliberately still propagate.
                # No exception bodies: upstream errors can contain credentials.
                row["status"] = "error"
                row["metrics"] = dict(empty_metrics)
                row["retrieved_chunk_ids"] = []
                row["hits"] = []
                row["error_type"] = type(error).__name__
            finally:
                row["retrieval_latency_seconds"] = round(time.monotonic() - retrieval_started, 6)
        rows.append(row)

    scored = [row for row in rows if row["status"] == "evaluated"]
    errors = sum(row["status"] == "error" for row in rows)
    excluded = sum(row["status"] == "excluded_no_gold" for row in rows)
    rank_counts = Counter(
        str(row["metrics"]["first_relevant_rank"])
        if row["metrics"]["first_relevant_rank"] is not None else "not_found"
        for row in scored
    )
    summary = {
        "selected_cases": len(rows), "attempted_cases": len(rows) - excluded,
        "scored_cases": len(scored), "excluded_no_gold_cases": excluded,
        "error_cases": errors, "metric_scope": "successfully_evaluated_cases_only",
        "hit_at_k": sum(row["metrics"]["hit_at_k"] for row in scored) / len(scored) if scored else None,
        "recall_at_k": sum(row["metrics"]["recall_at_k"] for row in scored) / len(scored) if scored else None,
        "first_relevant_rank_counts": dict(rank_counts),
    }
    return {
        "schema_version": "1.0", "evaluation_type": "retrieval_only",
        "status": "incomplete" if errors else "completed", "top_k": top_k,
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "metadata": dict(metadata), "summary": summary, "cases": rows,
    }
