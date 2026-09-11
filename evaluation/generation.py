"""Controlled eval observers over the real HTTP/RAG path; never production imports.

Observers call the original functions without changing generation parameters.
Only explicitly selected attack fixtures change a deep copy of retrieved text.
"""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import time
from unittest.mock import patch


def apply_fixture(hits, fixture):
    """Keep clean provenance/hash; modified text is explicitly synthetic evidence."""
    copies = [hit.model_copy(deep=True) for hit in hits]
    applied = False
    if fixture:
        for hit in copies:
            if hit.chunk_id == fixture["target_chunk_id"]:
                hit.text += fixture["text"]
                applied = True
    return copies, applied


def run_generation_cases(cases, fixtures, emit, *, retriever_factory=None, request_interval=None, allow_holdout=False):
    """Emit each completed attempt before starting the next, with no manual retry.

    Run in a dedicated process: observers temporarily wrap module references.
    Production rate/concurrency controls remain active. Failed cases are saved
    and end the batch; no repeated billing after an ambiguous provider failure.
    """
    allowed = ({"dev"}, {"frozen"}, {"holdout"}) if allow_holdout else ({"dev"}, {"frozen"})
    if not cases or {case.split for case in cases} not in allowed:
        raise ValueError("Accepts one nonempty split; holdout requires explicit unlock")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    fixture_map = {fixture["id"]: fixture for fixture in fixtures}
    if len(fixture_map) != len(fixtures) or any(
        case.fixture_id and case.fixture_id not in fixture_map for case in cases
    ):
        raise ValueError("invalid fixture references")

    from fastapi.testclient import TestClient
    from agent_graph import nodes as agent_graph_nodes
    import config
    import main
    from routing import Route, RouteDecision, RoutingResult
    from services import llm

    interval = (config.RATE_LIMIT_WINDOW_SECONDS / config.RATE_LIMIT_REQUESTS + 0.1
                if request_interval is None else request_interval)
    if interval < 0:
        raise ValueError("request interval cannot be negative")
    real_create = llm.client.chat.completions.create
    real_context = agent_graph_nodes.build_retrieved_context
    real_build_retriever = main.build_retriever
    active = {}

    class EvaluationKnowledgeRouter:
        def route(self, messages, *, deadline):
            return RoutingResult(RouteDecision(Route.KNOWLEDGE))

    built_retrievers = []

    def build_evaluation_retriever():
        factory = retriever_factory or real_build_retriever
        retriever = factory()
        built_retrievers.append(retriever)
        return retriever

    def observe_create(**kwargs):
        # Capture exactly what is sent; no max_tokens, usage options or judge.
        active["provider_requests"].append(deepcopy(kwargs))
        active["provider_input_characters"] += sum(
            len(message.get("content", ""))
            for message in kwargs.get("messages", [])
            if isinstance(message.get("content", ""), str)
        )
        generation_started = time.monotonic()
        try:
            completion = real_create(**kwargs)
        except Exception as error:
            elapsed = round(time.monotonic() - generation_started, 6)
            total = round((active["llm_total_latency_seconds"] or 0.0) + elapsed, 6)
            active["generation_latency_seconds"] = total
            active["llm_total_latency_seconds"] = total
            active["provider_errors"].append(type(error).__name__)
            raise

        elapsed = round(time.monotonic() - generation_started, 6)
        total = round((active["llm_total_latency_seconds"] or 0.0) + elapsed, 6)
        active["generation_latency_seconds"] = total
        active["llm_total_latency_seconds"] = total
        if active["llm_ttft_seconds"] is None:
            active["llm_ttft_seconds"] = elapsed
        active["llm_streaming_latency_seconds"] = 0.0
        active["provider_model"] = getattr(completion, "model", None)
        for choice in completion.choices:
            content = choice.message.content
            if isinstance(content, str):
                active["provider_partial_answer"] += content
            if choice.finish_reason:
                active["finish_reasons"].append(choice.finish_reason)
        return completion

    def observe_context(hits):
        context = real_context(hits)
        active["context"] = deepcopy(context)
        active["retrieved_context_characters"] = len(json.dumps(
            context, ensure_ascii=False, separators=(",", ":")
        ))
        return context

    completed = attempted = 0
    last_start = None
    with ExitStack() as stack:
        stack.enter_context(patch.object(
            main,
            "build_retriever",
            build_evaluation_retriever,
        ))
        stack.enter_context(patch.object(
            main,
            "HybridRouter",
            return_value=EvaluationKnowledgeRouter(),
        ))
        stack.enter_context(patch.object(llm.client.chat.completions, "create", observe_create))
        stack.enter_context(patch.object(
            agent_graph_nodes,
            "build_retrieved_context",
            observe_context,
        ))
        client = stack.enter_context(TestClient(main.app, raise_server_exceptions=False))
        retriever = built_retrievers[0]
        real_retrieve = retriever.retrieve

        def observe_retrieve(query):
            retrieval_started = time.monotonic()
            try:
                clean = real_retrieve(query)
            except Exception as error:
                active["retrieval_error_type"] = type(error).__name__
                raise
            finally:
                active["retrieval_latency_seconds"] = round(time.monotonic() - retrieval_started, 6)
            active["clean_hits"] = [hit.model_dump(mode="json") for hit in clean]
            active["retrieved_chunk_count"] = len(clean)
            fixture = fixture_map.get(active["fixture_id"])
            effective, applied = apply_fixture(clean, fixture)
            active["fixture_applied"] = applied
            active["fixture_status"] = "applied" if applied else ("not_exercised" if fixture else "not_requested")
            active["evaluation_mode"] = "synthetic_context" if applied else "unmodified"
            active["effective_hits"] = [hit.model_dump(mode="json") for hit in effective]
            active["effective_text_sha256"] = {
                hit.chunk_id: hashlib.sha256(hit.text.encode("utf-8")).hexdigest() for hit in effective
            }
            return effective

        stack.enter_context(patch.object(retriever, "retrieve", observe_retrieve))
        for case in cases:
            if last_start is not None:
                delay = max(0.0, interval - (time.monotonic() - last_start))
                while delay > 0:
                    # Short sleeps keep operator interruption responsive.
                    time.sleep(min(delay, 1.0))
                    delay = max(0.0, interval - (time.monotonic() - last_start))
            last_start = time.monotonic()
            active = {
                "event": "case", "case_id": case.id, "category": case.category,
                "split": case.split, "query": case.question,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "fixture_id": case.fixture_id, "fixture_applied": False,
                "fixture_status": "not_exercised" if case.fixture_id else "not_requested",
                "evaluation_mode": "unmodified", "clean_hits": [], "effective_hits": [],
                "context": [], "provider_requests": [], "provider_errors": [],
                "finish_reasons": [], "provider_model": None, "retrieval_error_type": None,
                "answer": "", "provider_partial_answer": "", "event_types": [], "http_status": None,
                "status": "incomplete", "error_type": None, "review_status": "pending_review",
                "retrieval_latency_seconds": None, "llm_ttft_seconds": None,
                "generation_latency_seconds": None, "llm_total_latency_seconds": None,
                "llm_streaming_latency_seconds": None, "total_request_latency_seconds": None,
                "retrieved_chunk_count": 0, "retrieved_context_characters": 0,
                "provider_input_characters": 0,
            }
            attempted += 1
            request_started = time.monotonic()
            try:
                response = client.post("/api/chat-stream", json={
                    "messages": [{"role": "user", "content": case.question}],
                })
                active["http_status"] = response.status_code
                active["request_id"] = response.headers.get("X-Request-ID")
                active["content_type"] = response.headers.get("content-type", "")
                if response.status_code == 200:
                    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
                    active["event_types"] = [event["type"] for event in events]
                    active["answer"] = "".join(event.get("content", "") for event in events if event["type"] == "delta")
                    valid = (
                        bool(active["answer"].strip()) and bool(events)
                        and active["event_types"][-1] == "done"
                        and active["event_types"].count("done") == 1
                        and all(kind in ("delta", "done") for kind in active["event_types"])
                        and active["content_type"].startswith("application/x-ndjson")
                        and bool(active["finish_reasons"])
                        and active["finish_reasons"][-1] == "stop"
                        and all(
                            reason in ("tool_calls", "stop")
                            for reason in active["finish_reasons"]
                        )
                    )
                    if valid:
                        active["status"] = "completed"
                        completed += 1
                if active["status"] != "completed":
                    active["error_type"] = (active["retrieval_error_type"] or
                        next(iter(reversed(active["provider_errors"])), None) or "GenerationNotComplete")
            except Exception as error:
                active["error_type"] = type(error).__name__
            except KeyboardInterrupt:
                active["error_type"] = "KeyboardInterrupt"
                raise
            finally:
                active["total_request_latency_seconds"] = round(time.monotonic() - request_started, 6)
                # Compatibility alias retained for existing Day 6 result readers.
                active["latency_seconds"] = active["total_request_latency_seconds"]
                emit(deepcopy(active))
            if active["status"] != "completed":
                break
    return {"selected_cases": len(cases), "attempted_cases": attempted,
            "completed_cases": completed, "incomplete_cases": attempted - completed,
            "not_attempted_cases": len(cases) - attempted,
            "status": "completed" if completed == len(cases) else "incomplete"}
