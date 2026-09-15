import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from performance.day3_streaming import (
    analyze_day3_phases,
    build_day3_schedule,
    day2_budget_screening_projection,
    load_day3_manifest,
    parse_success_ndjson,
    render_day3_markdown,
)
from scripts import run_week4_day3_streaming as runner


def manifest_data():
    return {
        "schema_version": "1.0",
        "seed": 20260914,
        "repeat_count": 3,
        "interval_seconds": 6.5,
        "cases": [
            {
                "case_id": f"{route}-{index}",
                "target_route": route,
                "messages": [{"role": "user", "content": f"{route} {index}"}],
            }
            for route in ("direct", "knowledge", "product_search")
            for index in (1, 2)
        ],
    }


def pair(phase, route, index, *, success=True, total=1000.0, ttft=2000.0):
    request_id = f"{phase}-{route}-{index}"
    return {
        "workload": {
            "record_type": "attempt",
            "workload_id": request_id,
            "phase": phase,
            "target_route": route,
            "request_id": request_id,
            "outcome": "success" if success else "ndjson_error",
            "terminal_done": success,
        },
        "summary": {
            "timestamp": datetime(2026, 9, 15, 1, tzinfo=timezone.utc),
            "request_id": request_id,
            "http_status": 200,
            "outcome": "success" if success else "internal_error",
            "route": route,
            "first_delta_latency_ms": ttft,
            "buffering_saved_ms": max(0.0, total - ttft),
            "total_latency_ms": total,
            "model_latency_ms": 800.0,
            "output_tokens": 20,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
        },
    }


def passing_pairs():
    baseline = []
    after = []
    for route in ("direct", "knowledge", "product_search"):
        for index in range(6):
            baseline.append(pair(
                "baseline",
                route,
                index,
                total=1000.0,
                ttft=2000.0,
            ))
            after.append(pair(
                "after",
                route,
                index,
                total=1150.0,
                ttft=1000.0 if route != "direct" else 1800.0,
            ))
    return baseline, after


class Day3ManifestTests(unittest.TestCase):
    def test_loader_accepts_required_structure_and_schedule_is_6_per_route(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest_data()), encoding="utf-8")
            manifest = load_day3_manifest(path)

        first = build_day3_schedule(manifest, "baseline")
        second = build_day3_schedule(manifest, "baseline")

        self.assertEqual(first, second)
        self.assertEqual(len(first), 18)
        self.assertEqual(len({row["workload_id"] for row in first}), 18)
        self.assertEqual(
            {route: sum(row["target_route"] == route for row in first)
             for route in ("direct", "knowledge", "product_search")},
            {"direct": 6, "knowledge": 6, "product_search": 6},
        )
        self.assertTrue(all(row["phase"] == "baseline" for row in first))

    def test_loader_rejects_invalid_required_structure(self):
        invalid_manifests = []
        duplicate = manifest_data()
        duplicate["cases"][1]["case_id"] = duplicate["cases"][0]["case_id"]
        invalid_manifests.append(duplicate)
        invalid_manifests.extend((
            {**manifest_data(), "repeat_count": 0},
            {**manifest_data(), "cases": []},
            {**manifest_data(), "cases": [{
                "case_id": "bad-route",
                "target_route": "fallback",
                "messages": [{"role": "user", "content": "hello"}],
            }]},
            {**manifest_data(), "cases": [{
                "case_id": "blank",
                "target_route": "direct",
                "messages": [{"role": "user", "content": "   "}],
            }]},
        ))

        for manifest in invalid_manifests:
            with self.subTest(manifest=manifest):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "manifest.json"
                    path.write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_day3_manifest(path)

    def test_success_parser_accepts_multiple_deltas_and_requires_final_done(self):
        body = "\n".join((
            json.dumps({"type": "delta", "content": "第一段"}),
            json.dumps({"type": "delta", "content": "第二段"}),
            json.dumps({"type": "done"}),
        ))

        parsed = parse_success_ndjson(body)

        self.assertEqual(parsed, {"answer": "第一段第二段", "delta_count": 2})
        with self.assertRaisesRegex(ValueError, "done"):
            parse_success_ndjson(json.dumps({
                "type": "delta",
                "content": "partial",
            }))

    def test_budget_screening_projection_uses_three_route_means_for_36_calls(self):
        analysis = {
            "estimated_cost_cny": {
                "by_route": {
                    "direct": {"mean": 0.001},
                    "knowledge": {"mean": 0.002},
                    "product_search": {"mean": 0.004},
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.json"
            path.write_text(json.dumps(analysis), encoding="utf-8")

            projection = day2_budget_screening_projection(path)

        self.assertAlmostEqual(projection, 12 * (0.001 + 0.002 + 0.004))

    def test_phase_runner_stops_after_first_failure_without_retry(self):
        manifest = manifest_data()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "phase.jsonl"
            with (
                patch.object(
                    runner,
                    "post_chat",
                    side_effect=RuntimeError("configuration failed"),
                ) as post,
                patch.object(runner.time, "sleep") as sleep,
            ):
                counts = runner.run_phase(
                    manifest,
                    phase="baseline",
                    base_url="http://127.0.0.1:8000",
                    output=output,
                    timeout_seconds=1.0,
                )

        self.assertEqual(counts, {"attempted": 1, "successful": 0, "failed": 1})
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_phase_runner_writes_18_privacy_bounded_rows_and_sleeps_17_times(self):
        manifest = manifest_data()
        response = SimpleNamespace(
            status_code=200,
            headers={"X-Request-ID": "request-1"},
            body="\n".join((
                json.dumps({"type": "delta", "content": "answer"}),
                json.dumps({"type": "done"}),
            )),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "phase.jsonl"
            with (
                patch.object(runner, "post_chat", return_value=response) as post,
                patch.object(runner.time, "sleep") as sleep,
            ):
                counts = runner.run_phase(
                    manifest,
                    phase="baseline",
                    base_url="http://127.0.0.1:8000",
                    output=output,
                    timeout_seconds=1.0,
                )
            rows = [json.loads(line) for line in output.read_text(
                encoding="utf-8"
            ).splitlines()]

        self.assertEqual(counts, {"attempted": 18, "successful": 18, "failed": 0})
        self.assertEqual(post.call_count, 18)
        self.assertEqual(sleep.call_count, 17)
        self.assertTrue(all(call.args == (6.5,) for call in sleep.call_args_list))
        self.assertEqual(len(rows), 19)
        self.assertTrue(all("messages" not in row for row in rows[1:]))
        self.assertTrue(all("answer" not in row for row in rows[1:]))

    def test_phase_runner_refuses_to_overwrite_an_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "phase.jsonl"
            output.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                runner.run_phase(
                    manifest_data(),
                    phase="baseline",
                    base_url="http://127.0.0.1:8000",
                    output=output,
                    timeout_seconds=1.0,
                )


class Day3GateTests(unittest.TestCase):
    def test_all_gates_pass_at_exact_total_latency_boundary(self):
        baseline, after = passing_pairs()

        analysis = analyze_day3_phases(baseline, after)

        self.assertTrue(analysis["gates"]["success_36_of_36"])
        self.assertEqual(
            analysis["gates"]["success_by_phase"],
            {"baseline_18_of_18": True, "after_18_of_18": True},
        )
        self.assertEqual(analysis["gates"]["route_counts"], {
            "baseline": {"product_search": 6, "knowledge": 6, "direct": 6},
            "after": {"product_search": 6, "knowledge": 6, "direct": 6},
        })
        self.assertTrue(analysis["gates"]["route_counts_pass"])
        self.assertTrue(analysis["gates"]["ttft"]["product_search"]["pass"])
        self.assertTrue(analysis["gates"]["ttft"]["knowledge"]["pass"])
        self.assertEqual(
            analysis["gates"]["total_latency_p50_within_15_percent"],
            {"product_search": True, "knowledge": True, "direct": True},
        )
        self.assertTrue(
            analysis["gates"][
                "all_routes_total_latency_p50_within_15_percent"
            ]
        )
        self.assertTrue(analysis["gates"]["all_pass"])
        self.assertEqual(analysis["conclusion"], "A")

    def test_analysis_and_report_include_route_failures_tokens_latency_and_cost(self):
        baseline, after = passing_pairs()
        after[-1] = pair(
            "after",
            "product_search",
            5,
            success=False,
            total=1150.0,
            ttft=1000.0,
        )

        analysis = analyze_day3_phases(baseline, after)
        route = analysis["by_phase_route"]["after"]["product_search"]
        report = render_day3_markdown(analysis)

        self.assertEqual(route["observed"], 6)
        self.assertEqual(route["successful"], 5)
        self.assertEqual(route["failed"], 1)
        self.assertEqual(route["model_latency_ms"]["count"], 5)
        self.assertEqual(route["output_tokens"]["count"], 5)
        self.assertEqual(route["estimated_cost_cny"]["count"], 6)
        self.assertIn("18/18 baseline success", report)
        self.assertIn("18/18 after success", report)
        self.assertIn("All routes total-latency P50 <=15% regression", report)
        self.assertIn("Model P50/P95 ms", report)
        self.assertIn("Output tokens P50/P95", report)
        self.assertIn("Estimated cost CNY", report)
        self.assertIn("| after | product_search | 5 | 1 |", report)

    def test_one_failed_request_prevents_36_of_36_and_conclusion_a(self):
        baseline, after = passing_pairs()
        after[-1] = pair(
            "after",
            "product_search",
            5,
            success=False,
            total=1150.0,
            ttft=1000.0,
        )

        analysis = analyze_day3_phases(baseline, after)

        self.assertFalse(analysis["gates"]["success_36_of_36"])
        self.assertFalse(analysis["gates"]["all_pass"])
        self.assertEqual(analysis["conclusion"], "B")

    def test_direct_regression_over_15_percent_prevents_conclusion_a(self):
        baseline, after = passing_pairs()
        for candidate in after:
            if candidate["summary"]["route"] == "direct":
                candidate["summary"]["total_latency_ms"] = 1151.0

        analysis = analyze_day3_phases(baseline, after)

        self.assertFalse(
            analysis["gates"]["total_latency_p50_within_15_percent"]["direct"]
        )
        self.assertFalse(analysis["gates"]["all_pass"])
        self.assertEqual(analysis["conclusion"], "B")


if __name__ == "__main__":
    unittest.main()
