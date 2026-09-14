"""Run the Week 4 Day 2 production-style workload over real HTTP."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "performance" / "workload_v1.json"
SUPPORTED_ROUTES = frozenset({
    "direct",
    "knowledge",
    "exact_product",
    "product_search",
    "contact",
    "fallback",
})


@dataclass(frozen=True)
class HttpResult:
    status_code: int
    headers: Mapping[str, str]
    body: str


PostChat = Callable[[str, list[dict[str, str]], float], HttpResult]


def _non_blank(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    return value.strip()


def _route(value: object) -> str:
    route = _non_blank(value, "target_route")
    if route not in SUPPORTED_ROUTES:
        raise ValueError(f"unsupported target route: {route}")
    return route


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return value


def load_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("workload manifest must be an object")

    _non_blank(manifest.get("schema_version"), "schema_version")
    seed = manifest.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    repeat_count = manifest.get("repeat_count")
    if (
        not isinstance(repeat_count, int)
        or isinstance(repeat_count, bool)
        or repeat_count <= 0
    ):
        raise ValueError("repeat_count must be a positive integer")

    case_ids: set[str] = set()
    for case in _list(manifest.get("single_turn_cases"), "single_turn_cases"):
        if not isinstance(case, dict):
            raise ValueError("single-turn case must be an object")
        case_id = _non_blank(case.get("case_id"), "case_id")
        if case_id in case_ids:
            raise ValueError(f"duplicate case ID: {case_id}")
        case_ids.add(case_id)
        _route(case.get("target_route"))
        messages = _list(case.get("messages"), "messages")
        if len(messages) != 1 or not isinstance(messages[0], dict):
            raise ValueError("single-turn case must contain one user message")
        if messages[0].get("role") != "user":
            raise ValueError("single-turn message role must be user")
        _non_blank(messages[0].get("content"), "message content")

    for case in _list(manifest.get("calibration_cases"), "calibration_cases"):
        if not isinstance(case, dict):
            raise ValueError("calibration case must be an object")
        case_id = _non_blank(case.get("case_id"), "case_id")
        if case_id in case_ids:
            raise ValueError(f"duplicate case ID: {case_id}")
        case_ids.add(case_id)
        _route(case.get("target_route"))
        _non_blank(case.get("user_content"), "user_content")

    conversation_ids: set[str] = set()
    turn_ids: set[str] = set()
    for conversation in _list(
        manifest.get("conversations"),
        "conversations",
    ):
        if not isinstance(conversation, dict):
            raise ValueError("conversation must be an object")
        conversation_id = _non_blank(
            conversation.get("conversation_id"),
            "conversation_id",
        )
        if conversation_id in conversation_ids:
            raise ValueError(f"duplicate conversation ID: {conversation_id}")
        conversation_ids.add(conversation_id)
        for turn in _list(conversation.get("turns"), "turns"):
            if not isinstance(turn, dict):
                raise ValueError("conversation turn must be an object")
            turn_id = _non_blank(turn.get("turn_id"), "turn_id")
            scoped_turn_id = f"{conversation_id}:{turn_id}"
            if scoped_turn_id in turn_ids:
                raise ValueError(f"duplicate turn ID: {scoped_turn_id}")
            turn_ids.add(scoped_turn_id)
            _route(turn.get("target_route"))
            _non_blank(turn.get("user_content"), "user_content")

    return manifest


def build_single_turn_schedule(
    manifest: Mapping[str, object],
    *,
    calibration: bool = False,
) -> list[dict[str, object]]:
    if calibration:
        return [
            {
                "workload_id": f"calibration:{case['case_id']}",
                "kind": "calibration",
                "case_id": case["case_id"],
                "target_route": case["target_route"],
                "messages": [{
                    "role": "user",
                    "content": case["user_content"],
                }],
            }
            for case in manifest["calibration_cases"]
        ]

    schedule = []
    for case in manifest["single_turn_cases"]:
        for execution_index in range(1, int(manifest["repeat_count"]) + 1):
            schedule.append({
                "workload_id": (
                    f"single:{case['case_id']}:{execution_index}"
                ),
                "kind": "single_turn",
                "case_id": case["case_id"],
                "target_route": case["target_route"],
                "execution_index": execution_index,
                "messages": [dict(message) for message in case["messages"]],
            })
    random.Random(int(manifest["seed"])).shuffle(schedule)
    return schedule


def parse_success_ndjson(body: str) -> str:
    try:
        events = [
            json.loads(line)
            for line in body.splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as error:
        raise ValueError("response contains invalid NDJSON") from error
    if len(events) < 2 or not all(isinstance(event, dict) for event in events):
        raise ValueError("response has an invalid event sequence")
    if events[-1].get("type") != "done":
        raise ValueError("response does not end with done")
    if sum(event.get("type") == "done" for event in events) != 1:
        raise ValueError("response contains an invalid done event")

    content_parts = []
    citations_seen = False
    for event in events[:-1]:
        event_type = event.get("type")
        if event_type == "delta" and not citations_seen:
            content = event.get("content")
            if not isinstance(content, str) or not content:
                raise ValueError("delta content is empty or invalid")
            content_parts.append(content)
        elif event_type == "citations" and content_parts and not citations_seen:
            citations_seen = True
        else:
            raise ValueError("response has an invalid event sequence")
    if not content_parts:
        raise ValueError("response has no answer delta")
    return "".join(content_parts)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _header(headers: Mapping[str, str], name: str) -> str | None:
    lowered = name.casefold()
    for key, value in headers.items():
        if key.casefold() == lowered:
            return value
    return None


def _attempt_row(
    item: Mapping[str, object],
    *,
    base_url: str,
    timeout_seconds: float,
    post_chat_fn: PostChat,
) -> tuple[dict[str, object], str | None]:
    started_at = _now()
    try:
        result = post_chat_fn(
            base_url,
            [dict(message) for message in item["messages"]],
            timeout_seconds,
        )
    except Exception as error:
        return ({
            "record_type": "attempt",
            "workload_id": item["workload_id"],
            "kind": item["kind"],
            "case_id": item.get("case_id"),
            "target_route": item["target_route"],
            "conversation_id": item.get("conversation_id"),
            "turn_index": item.get("turn_index"),
            "started_at": started_at,
            "completed_at": _now(),
            "http_status": None,
            "outcome": "transport_error",
            "request_id": None,
            "error_type": type(error).__name__,
        }, None)

    request_id = _header(result.headers, "X-Request-ID")
    row = {
        "record_type": "attempt",
        "workload_id": item["workload_id"],
        "kind": item["kind"],
        "case_id": item.get("case_id"),
        "target_route": item["target_route"],
        "conversation_id": item.get("conversation_id"),
        "turn_index": item.get("turn_index"),
        "started_at": started_at,
        "completed_at": _now(),
        "http_status": result.status_code,
        "request_id": request_id,
    }
    if result.status_code != 200:
        row["outcome"] = "http_error"
        return row, None
    content_type = _header(result.headers, "content-type") or ""
    if not content_type.startswith("application/x-ndjson"):
        row["outcome"] = "ndjson_error"
        row["error_type"] = "UnexpectedContentType"
        return row, None
    if not request_id:
        row["outcome"] = "missing_request_id"
        return row, None
    try:
        answer = parse_success_ndjson(result.body)
    except ValueError:
        row["outcome"] = "ndjson_error"
        row["error_type"] = "InvalidNdjson"
        return row, None
    row["outcome"] = "success"
    return row, answer


def run_workload(
    manifest: Mapping[str, object],
    *,
    base_url: str,
    destination: Path,
    interval_seconds: float,
    timeout_seconds: float,
    calibration: bool,
    post_chat_fn: PostChat,
    sleep_fn: Callable[[float], None],
) -> dict[str, int]:
    if destination.exists():
        raise FileExistsError(f"workload output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts = {"attempted": 0, "successful": 0, "failed": 0, "skipped": 0}
    first_attempt = True

    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        header = {
            "record_type": "run",
            "schema_version": manifest["schema_version"],
            "started_at": _now(),
            "base_url": base_url.rstrip("/"),
            "seed": manifest["seed"],
            "repeat_count": manifest["repeat_count"],
            "calibration": calibration,
        }
        stream.write(json.dumps(header, ensure_ascii=False) + "\n")
        stream.flush()

        def execute(item: Mapping[str, object]) -> str | None:
            nonlocal first_attempt
            if not first_attempt:
                sleep_fn(interval_seconds)
            first_attempt = False
            row, answer = _attempt_row(
                item,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                post_chat_fn=post_chat_fn,
            )
            counts["attempted"] += 1
            if row["outcome"] == "success":
                counts["successful"] += 1
            else:
                counts["failed"] += 1
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            return answer

        for item in build_single_turn_schedule(
            manifest,
            calibration=calibration,
        ):
            execute(item)

        if not calibration:
            histories = {
                conversation["conversation_id"]: []
                for conversation in manifest["conversations"]
            }
            failed_conversations: set[str] = set()
            maximum_turns = max(
                len(conversation["turns"])
                for conversation in manifest["conversations"]
            )
            for turn_index in range(maximum_turns):
                for conversation in manifest["conversations"]:
                    conversation_id = conversation["conversation_id"]
                    turns = conversation["turns"]
                    if turn_index >= len(turns):
                        continue
                    turn = turns[turn_index]
                    workload_id = (
                        f"conversation:{conversation_id}:{turn['turn_id']}"
                    )
                    if conversation_id in failed_conversations:
                        skipped = {
                            "record_type": "skipped",
                            "workload_id": workload_id,
                            "kind": "multi_turn",
                            "case_id": None,
                            "target_route": turn["target_route"],
                            "conversation_id": conversation_id,
                            "turn_index": turn_index + 1,
                            "reason": "prior_turn_failed",
                        }
                        stream.write(
                            json.dumps(skipped, ensure_ascii=False) + "\n"
                        )
                        stream.flush()
                        counts["skipped"] += 1
                        continue
                    history = histories[conversation_id]
                    user_message = {
                        "role": "user",
                        "content": turn["user_content"],
                    }
                    item = {
                        "workload_id": workload_id,
                        "kind": "multi_turn",
                        "case_id": None,
                        "target_route": turn["target_route"],
                        "conversation_id": conversation_id,
                        "turn_index": turn_index + 1,
                        "messages": [*history, user_message],
                    }
                    answer = execute(item)
                    if answer is None:
                        failed_conversations.add(conversation_id)
                    else:
                        history.extend((
                            user_message,
                            {"role": "assistant", "content": answer},
                        ))
    return counts


def post_chat(
    base_url: str,
    messages: list[dict[str, str]],
    timeout_seconds: float,
) -> HttpResult:
    request = Request(
        base_url.rstrip("/") + "/api/chat-stream",
        data=json.dumps({"messages": messages}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return HttpResult(
                status_code=response.status,
                headers=dict(response.headers.items()),
                body=response.read().decode("utf-8"),
            )
    except HTTPError as error:
        return HttpResult(
            status_code=error.code,
            headers=dict(error.headers.items()),
            body=error.read().decode("utf-8"),
        )


def check_health(base_url: str, timeout_seconds: float) -> None:
    with urlopen(base_url.rstrip("/") + "/health", timeout=timeout_seconds) as response:
        if response.status != 200:
            raise RuntimeError(f"health check returned HTTP {response.status}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--interval-seconds", type=float, default=6.5)
    parser.add_argument("--timeout-seconds", type=float, default=150.0)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    planned_single = len(build_single_turn_schedule(manifest))
    planned_conversation = sum(
        len(conversation["turns"])
        for conversation in manifest["conversations"]
    )
    plan = {
        "base_url": args.base_url,
        "calibration_requests": len(build_single_turn_schedule(
            manifest,
            calibration=True,
        )),
        "single_turn_requests": planned_single,
        "multi_turn_requests": planned_conversation,
        "execute": args.execute,
    }
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False))
        return 0
    if args.output is None:
        raise SystemExit("--output is required with --execute")
    if args.interval_seconds < 0 or args.timeout_seconds <= 0:
        raise SystemExit("interval must be non-negative and timeout positive")
    check_health(args.base_url, args.timeout_seconds)
    summary = run_workload(
        manifest,
        base_url=args.base_url,
        destination=args.output,
        interval_seconds=args.interval_seconds,
        timeout_seconds=args.timeout_seconds,
        calibration=args.calibration,
        post_chat_fn=post_chat,
        sleep_fn=time.sleep,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
