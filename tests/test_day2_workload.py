import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_week4_day2_workload import (
    HttpResult,
    build_single_turn_schedule,
    load_manifest,
    parse_success_ndjson,
    run_workload,
)


ROOT = Path(__file__).resolve().parents[1]
ROUTES = {
    "direct",
    "knowledge",
    "exact_product",
    "product_search",
    "contact",
    "fallback",
}


def success_result(answer="受控回答"):
    return HttpResult(
        status_code=200,
        headers={
            "content-type": "application/x-ndjson",
            "x-request-id": "request-1",
        },
        body=(
            json.dumps({"type": "delta", "content": answer})
            + "\n"
            + json.dumps({"type": "done"})
            + "\n"
        ),
    )


def small_manifest():
    return {
        "schema_version": "1.0",
        "seed": 7,
        "repeat_count": 1,
        "single_turn_cases": [{
            "case_id": "single-1",
            "target_route": "direct",
            "messages": [{"role": "user", "content": "私密单轮问题"}],
        }],
        "calibration_cases": [{
            "case_id": "calibration-1",
            "target_route": "direct",
            "user_content": "独立校准问题",
        }],
        "conversations": [{
            "conversation_id": "conversation-1",
            "turns": [
                {
                    "turn_id": "turn-1",
                    "target_route": "direct",
                    "user_content": "私密多轮问题一",
                },
                {
                    "turn_id": "turn-2",
                    "target_route": "contact",
                    "user_content": "私密多轮问题二",
                },
            ],
        }],
    }


class WorkloadManifestTests(unittest.TestCase):
    def test_v1_manifest_has_approved_experiment_composition(self):
        manifest = load_manifest(ROOT / "performance/workload_v1.json")

        self.assertEqual(len(manifest["single_turn_cases"]), 24)
        self.assertEqual(
            {case["target_route"] for case in manifest["single_turn_cases"]},
            ROUTES,
        )
        self.assertEqual(len(manifest["conversations"]), 4)
        self.assertTrue(all(
            len(item["turns"]) == 3
            for item in manifest["conversations"]
        ))
        self.assertEqual(len(manifest["calibration_cases"]), 6)
        calibration_prompts = {
            case["user_content"]
            for case in manifest["calibration_cases"]
        }
        measured_prompts = {
            case["messages"][0]["content"]
            for case in manifest["single_turn_cases"]
        }
        self.assertTrue(calibration_prompts.isdisjoint(measured_prompts))
        self.assertEqual(len(build_single_turn_schedule(manifest)), 120)
        self.assertEqual(
            len(build_single_turn_schedule(manifest, calibration=True)),
            6,
        )

    def test_loader_accepts_other_positive_experiment_sizes(self):
        manifest = small_manifest()
        manifest["repeat_count"] = 2
        manifest["conversations"][0]["turns"] = [
            manifest["conversations"][0]["turns"][0]
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = load_manifest(path)

        self.assertEqual(loaded["repeat_count"], 2)
        self.assertEqual(len(loaded["conversations"]), 1)
        self.assertEqual(len(loaded["conversations"][0]["turns"]), 1)

    def test_schedule_is_reproducible_and_interleaved(self):
        manifest = load_manifest(ROOT / "performance/workload_v1.json")

        first = build_single_turn_schedule(manifest)
        second = build_single_turn_schedule(manifest)

        self.assertEqual(first, second)
        routes = [row["target_route"] for row in first]
        for route in ROUTES:
            positions = [
                index
                for index, value in enumerate(routes)
                if value == route
            ]
            self.assertNotEqual(
                positions,
                list(range(positions[0], positions[0] + len(positions))),
            )


class NdjsonParsingTests(unittest.TestCase):
    def test_accepts_delta_with_optional_citations_and_done(self):
        lines = [
            {"type": "delta", "content": "第一段"},
            {"type": "delta", "content": "第二段"},
            {"type": "citations", "heading": "来源", "items": []},
            {"type": "done"},
        ]
        body = "\n".join(json.dumps(item) for item in lines) + "\n"

        self.assertEqual(parse_success_ndjson(body), "第一段第二段")

    def test_rejects_invalid_success_event_sequences(self):
        invalid_bodies = (
            "not-json\n",
            json.dumps({"type": "delta", "content": ""}) + "\n",
            json.dumps({"type": "delta", "content": "answer"}) + "\n",
            "\n".join((
                json.dumps({"type": "delta", "content": "answer"}),
                json.dumps({"type": "done"}),
                json.dumps({"type": "done"}),
            )),
            "\n".join((
                json.dumps({"type": "delta", "content": "answer"}),
                json.dumps({"type": "done"}),
                json.dumps({"type": "citations", "items": []}),
            )),
            "\n".join((
                json.dumps({"type": "error", "code": "internal_error"}),
                json.dumps({"type": "done"}),
            )),
        )

        for body in invalid_bodies:
            with self.subTest(body=body):
                with self.assertRaises(ValueError):
                    parse_success_ndjson(body)


class WorkloadExecutionTests(unittest.TestCase):
    def test_result_rows_exclude_prompt_and_answer_content(self):
        calls = []

        def post(base_url, messages, timeout):
            calls.append(messages)
            if len(calls) == 2:
                return HttpResult(
                    status_code=500,
                    headers={"x-request-id": "request-2"},
                    body='{"error":{"code":"internal_error"}}',
                )
            result = success_result("不得持久化的回答")
            return HttpResult(
                result.status_code,
                {**result.headers, "x-request-id": f"request-{len(calls)}"},
                result.body,
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.jsonl"
            summary = run_workload(
                small_manifest(),
                base_url="http://127.0.0.1:8000",
                destination=output,
                interval_seconds=0,
                timeout_seconds=1,
                calibration=False,
                post_chat_fn=post,
                sleep_fn=lambda _: None,
            )
            serialized = output.read_text(encoding="utf-8")
            rows = [json.loads(line) for line in serialized.splitlines()]

        self.assertEqual(summary, {
            "attempted": 2,
            "successful": 1,
            "failed": 1,
            "skipped": 1,
        })
        self.assertEqual(len(calls), 2)
        self.assertEqual([row["record_type"] for row in rows], [
            "run",
            "attempt",
            "attempt",
            "skipped",
        ])
        for private_text in (
            "私密单轮问题",
            "私密多轮问题一",
            "私密多轮问题二",
            "不得持久化的回答",
        ):
            self.assertNotIn(private_text, serialized)

    def test_successful_conversation_preserves_real_history(self):
        calls = []

        def post(base_url, messages, timeout):
            calls.append([dict(message) for message in messages])
            result = success_result(f"回答-{len(calls)}")
            return HttpResult(
                result.status_code,
                {**result.headers, "x-request-id": f"request-{len(calls)}"},
                result.body,
            )

        with tempfile.TemporaryDirectory() as directory:
            summary = run_workload(
                small_manifest(),
                base_url="http://127.0.0.1:8000",
                destination=Path(directory) / "run.jsonl",
                interval_seconds=0,
                timeout_seconds=1,
                calibration=False,
                post_chat_fn=post,
                sleep_fn=lambda _: None,
            )

        self.assertEqual(summary["attempted"], 3)
        self.assertEqual(calls[2], [
            {"role": "user", "content": "私密多轮问题一"},
            {"role": "assistant", "content": "回答-2"},
            {"role": "user", "content": "私密多轮问题二"},
        ])


if __name__ == "__main__":
    unittest.main()
