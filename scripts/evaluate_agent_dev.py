"""Run the frozen Dev24 agent evaluation through production LangGraph."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import AgentDeadline
from evaluation.agent import (
    AgentEvaluationRunner,
    RecordingToolExecutor,
    load_agent_eval_cases,
    run_agent_evaluation,
)
from routing import Route


def _validate_dev24(cases) -> None:
    counts = Counter(case.expected_route for case in cases)
    if len(cases) != 24 or counts != Counter({route: 4 for route in Route}):
        raise ValueError("Dev24 must contain exactly four cases per route")


def _build_production_runner() -> AgentEvaluationRunner:
    from agent_graph import AgentGraphNodes, GraphRouteOrchestrator, build_agent_graph
    from routing import HybridRouter
    from services.llm import complete_chat
    from services.retrieval import build_retriever
    from services.tools import build_deterministic_tools
    from support_tools import build_tool_registry

    retriever = build_retriever()
    tools = build_deterministic_tools(retriever=retriever)
    executor = RecordingToolExecutor(build_tool_registry(tools))
    router = HybridRouter(
        retriever=retriever,
        complete_chat=complete_chat,
    )
    graph = build_agent_graph(AgentGraphNodes(
        router=router,
        executor=executor,
        retriever=retriever,
        complete_chat=complete_chat,
    ))
    return AgentEvaluationRunner(GraphRouteOrchestrator(graph), executor)


def _configured_model_metadata(root: Path) -> dict[str, str]:
    import config
    from knowledge_pipeline.retrieval import EmbeddingConfig

    values = {
        key: value
        for key, value in dotenv_values(root / ".env").items()
        if value is not None
    }
    values.update(os.environ)
    embedding = EmbeddingConfig.from_mapping(values)
    return {
        "generation_model": config.DEEPSEEK_MODEL,
        "embedding_provider": embedding.provider,
        "embedding_model": embedding.model,
    }


def _new_deadline() -> AgentDeadline:
    import config

    return AgentDeadline.start(
        config.AGENT_TIMEOUT_SECONDS,
        clock=time.monotonic,
    )


def _output_path(root: Path, requested: Path | None) -> Path:
    directory = (root / "eval/results").resolve()
    if requested is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = directory / f"agent-dev-v1-{stamp}-{uuid4().hex[:8]}.json"
    else:
        path = (requested if requested.is_absolute() else root / requested).resolve()
    if (
        not path.is_relative_to(directory)
        or path.suffix != ".json"
        or path.exists()
    ):
        raise ValueError("output must be a new .json file inside eval/results")
    return path


def _print_summary(summary: dict[str, object], destination: Path) -> None:
    total = summary["total_cases"]
    route_passed = summary["route_passed"]
    action_passed = summary["action_passed"]
    failed = summary["failed_case_ids"]
    print(f"Total cases: {total}")
    print(
        f"Route Accuracy: {route_passed}/{total} "
        f"({summary['route_accuracy']:.2%})"
    )
    print(
        f"Action Accuracy: {action_passed}/{total} "
        f"({summary['action_accuracy']:.2%})"
    )
    print("Failed case IDs: " + (", ".join(failed) if failed else "-"))
    print(f"Output result path: {destination}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        root = args.root.resolve()
        dataset = root / "evaluation/agent_dev_v1.jsonl"
        cases = load_agent_eval_cases(dataset)
        _validate_dev24(cases)
        if not args.execute:
            print(f"Total cases: {len(cases)}")
            print("Mode: preflight (no provider calls)")
            return 0

        destination = _output_path(root, args.output)
        runner = _build_production_runner()
        report = run_agent_evaluation(
            cases,
            runner,
            deadline_factory=_new_deadline,
        )
        payload = {
            "schema_version": "1.0",
            "evaluation_type": "agent_dev",
            "metadata": {
                "dataset_path": "evaluation/agent_dev_v1.jsonl",
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                **_configured_model_metadata(root),
            },
            **report,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        _print_summary(report["summary"], destination)
        return 0
    except Exception as error:
        print(
            f"Agent evaluation stopped: {type(error).__name__}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
