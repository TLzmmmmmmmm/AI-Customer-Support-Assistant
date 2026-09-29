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
        "repeat_count": 5,
        "interval_seconds": 6.5,
        "cases": [
            {
                "case_id": f"{route}-{index}",
                "target_route": route,
                "messages": [{"role": "user", "content": f"{route} {index}"}],
            }
            for route in (
                "direct",
                "knowledge",
                "product_search",
                "exact_product",
                "contact",
            )
            for index in (1, 2)
        ],
    }


def pair(phase, route, index, *, success=True, total=2500.0, ttft=2000.0):
    request_id = f"{phase}-{route}-{index}"
    case_id = f"{route}-{index // 5}"
    execution_index = index % 5 + 1
    return {
        "workload": {
            "record_type": "attempt",
            "workload_id": request_id,
            "phase": phase,
            "mode": "buffered" if phase == "baseline" else "streaming",
            "pair_id": f"{case_id}:{execution_index}",
            "case_id": case_id,
            "execution_index": execution_index,
            "target_route": route,
            "request_id": request_id,
            "outcome": "success" if success else "ndjson_error",
            "terminal_done": success,
            "delta_count": 1 if phase == "baseline" else 2,
            "client_first_delta_latency_ms": ttft + 50.0,
            "client_total_latency_ms": total + 50.0,
            "answer_chars": 40,
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
    for route in (
        "product_search",
        "knowledge",
        "exact_product",
        "contact",
        "direct",
    ):
        for index in range(10):
            baseline.append(pair(
                "baseline",
                route,
                index,
                total=2500.0,
                ttft=2000.0,
            ))
            after.append(pair(
                "after",
                route,
                index,
                total=2875.0,
                ttft=1000.0 if route != "direct" else 1800.0,
            ))
    return baseline, after


class Day3ManifestTests(unittest.TestCase):
    def test_loader_accepts_required_structure_and_schedule_is_10_per_route(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest_data()), encoding="utf-8")
            manifest = load_day3_manifest(path)

        first = build_day3_schedule(manifest, "baseline")
        second = build_day3_schedule(manifest, "baseline")

        self.assertEqual(first, second)
        self.assertEqual(len(first), 50)
        self.assertEqual(len({row["workload_id"] for row in first}), 50)
        self.assertEqual(
            {route: sum(row["target_route"] == route for row in first)
             for route in (
                 "direct",
                 "knowledge",
                 "product_search",
                 "exact_product",
                 "contact",
             )},
            {
                "direct": 10,
                "knowledge": 10,
                "product_search": 10,
                "exact_product": 10,
                "contact": 10,
            },
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

    def test_budget_projection_uses_manifest_route_counts_for_both_modes(self):
        analysis = {
            "estimated_cost_cny": {
                "by_route": {
                    "direct": {"mean": 0.001},
                    "knowledge": {"mean": 0.002},
                    "product_search": {"mean": 0.004},
                    "exact_product": {"mean": 0.003},
                    "contact": {"mean": 0.0015},
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.json"
            path.write_text(json.dumps(analysis), encoding="utf-8")

            projection = day2_budget_screening_projection(path, manifest_data())

        self.assertAlmostEqual(
            projection,
            20 * (0.001 + 0.002 + 0.004 + 0.003 + 0.0015),
        )

    def test_streaming_http_client_records_first_delta_and_total_latency(self):
        response = SimpleNamespace(
            status=200,
            headers={"X-Request-ID": "request-1"},
            __enter__=lambda self: self,
            __exit__=lambda self, *args: None,
        )
        response.__iter__ = lambda self: iter((
            b'{"type":"delta","content":"first"}\n',
            b'{"type":"delta","content":"second"}\n',
            b'{"type":"done"}\n',
        ))

        class Response:
            status = response.status
            headers = response.headers

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def __iter__(self):
                return response.__iter__(response)

        with (
            patch.object(runner, "urlopen", return_value=Response()),
            patch.object(
                runner.time,
                "monotonic",
                side_effect=(10.0, 10.1, 10.25, 10.8),
            ),
        ):
            result = runner.post_chat_streaming(
                "http://127.0.0.1:8002",
                [{"role": "user", "content": "hello"}],
                1.0,
            )

        self.assertEqual(result.status_code, 200)
        self.assertAlmostEqual(result.headers_latency_ms, 100.0)
        self.assertAlmostEqual(result.first_delta_latency_ms, 250.0)
        self.assertAlmostEqual(result.total_latency_ms, 800.0)
        self.assertIn('"type":"done"', result.body)

    def test_comparison_runner_stops_after_first_failure_without_retry(self):
        manifest = manifest_data()
        with tempfile.TemporaryDirectory() as directory:
            baseline_output = Path(directory) / "baseline.jsonl"
            after_output = Path(directory) / "after.jsonl"
            with (
                patch.object(
                    runner,
                    "post_chat_streaming",
                    side_effect=RuntimeError("configuration failed"),
                ) as post,
                patch.object(runner.time, "sleep") as sleep,
            ):
                counts = runner.run_comparison(
                    manifest,
                    baseline_base_url="http://127.0.0.1:8001",
                    after_base_url="http://127.0.0.1:8002",
                    baseline_output=baseline_output,
                    after_output=after_output,
                    timeout_seconds=1.0,
                )

        self.assertEqual(counts, {"attempted": 1, "successful": 0, "failed": 1})
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_comparison_runner_writes_paired_privacy_bounded_rows(self):
        manifest = manifest_data()

        def response(base_url, messages, timeout_seconds):
            del messages, timeout_seconds
            deltas = ("answer",) if base_url.endswith("8001") else ("an", "swer")
            return SimpleNamespace(
                status_code=200,
                headers={"X-Request-ID": f"request-{base_url[-1]}"},
                body="\n".join((
                    *(json.dumps({"type": "delta", "content": delta})
                      for delta in deltas),
                    json.dumps({"type": "done"}),
                )),
                headers_latency_ms=100.0,
                first_delta_latency_ms=(500.0 if len(deltas) == 1 else 200.0),
                total_latency_ms=600.0,
            )

        with tempfile.TemporaryDirectory() as directory:
            baseline_output = Path(directory) / "baseline.jsonl"
            after_output = Path(directory) / "after.jsonl"
            with (
                patch.object(
                    runner,
                    "post_chat_streaming",
                    side_effect=response,
                ) as post,
                patch.object(runner.time, "sleep") as sleep,
            ):
                counts = runner.run_comparison(
                    manifest,
                    baseline_base_url="http://127.0.0.1:8001",
                    after_base_url="http://127.0.0.1:8002",
                    baseline_output=baseline_output,
                    after_output=after_output,
                    timeout_seconds=1.0,
                )
            baseline_rows = [json.loads(line) for line in baseline_output.read_text(
                encoding="utf-8"
            ).splitlines()]
            after_rows = [json.loads(line) for line in after_output.read_text(
                encoding="utf-8"
            ).splitlines()]

        self.assertEqual(counts, {"attempted": 100, "successful": 100, "failed": 0})
        self.assertEqual(post.call_count, 100)
        self.assertEqual(sleep.call_count, 99)
        self.assertTrue(all(call.args == (6.5,) for call in sleep.call_args_list))
        self.assertEqual(len(baseline_rows), 51)
        self.assertEqual(len(after_rows), 51)
        self.assertEqual(baseline_rows[0]["mode"], "buffered")
        self.assertEqual(after_rows[0]["mode"], "streaming")
        self.assertEqual(
            {row["pair_id"] for row in baseline_rows[1:]},
            {row["pair_id"] for row in after_rows[1:]},
        )
        for row in [*baseline_rows[1:], *after_rows[1:]]:
            self.assertNotIn("messages", row)
            self.assertNotIn("answer", row)
            self.assertIn("answer_chars", row)
            self.assertIn("client_first_delta_latency_ms", row)
            self.assertIn("client_total_latency_ms", row)

    def test_comparison_runner_rejects_same_endpoint_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline_output = Path(directory) / "baseline.jsonl"
            after_output = Path(directory) / "after.jsonl"
            baseline_output.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                runner.run_comparison(
                    manifest_data(),
                    baseline_base_url="http://127.0.0.1:8001",
                    after_base_url="http://127.0.0.1:8002",
                    baseline_output=baseline_output,
                    after_output=after_output,
                    timeout_seconds=1.0,
                )
            with self.assertRaisesRegex(ValueError, "different endpoints"):
                runner.run_comparison(
                    manifest_data(),
                    baseline_base_url="http://127.0.0.1:8001",
                    after_base_url="http://127.0.0.1:8001/",
                    baseline_output=Path(directory) / "new-baseline.jsonl",
                    after_output=after_output,
                    timeout_seconds=1.0,
                )


class Day3GateTests(unittest.TestCase):
    def test_all_gates_pass_at_exact_total_latency_boundary(self):
        baseline, after = passing_pairs()

        analysis = analyze_day3_phases(baseline, after)

        self.assertTrue(analysis["gates"]["success_100_of_100"])
        self.assertEqual(
            analysis["gates"]["success_by_phase"],
            {"baseline_50_of_50": True, "after_50_of_50": True},
        )
        self.assertEqual(analysis["gates"]["route_counts"], {
            "baseline": {
                "product_search": 10,
                "knowledge": 10,
                "exact_product": 10,
                "contact": 10,
                "direct": 10,
            },
            "after": {
                "product_search": 10,
                "knowledge": 10,
                "exact_product": 10,
                "contact": 10,
                "direct": 10,
            },
        })
        self.assertTrue(analysis["gates"]["route_counts_pass"])
        self.assertTrue(analysis["gates"]["ttft"]["product_search"]["pass"])
        self.assertTrue(analysis["gates"]["ttft"]["knowledge"]["pass"])
        self.assertTrue(analysis["gates"]["client_ttft"]["product_search"]["pass"])
        self.assertTrue(analysis["gates"]["streaming_mode_verified"])
        self.assertEqual(
            analysis["paired_differences"]["product_search"]
            ["server_first_delta_reduction_ms"]["p50"],
            1000.0,
        )
        self.assertEqual(
            analysis["gates"]["total_latency_p50_within_15_percent"],
            {
                "product_search": True,
                "knowledge": True,
                "exact_product": True,
                "contact": True,
                "direct": True,
            },
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
        after[9] = pair(
            "after",
            "product_search",
            9,
            success=False,
            total=2875.0,
            ttft=1000.0,
        )

        analysis = analyze_day3_phases(baseline, after)
        route = analysis["by_phase_route"]["after"]["product_search"]
        report = render_day3_markdown(analysis)

        self.assertEqual(route["observed"], 10)
        self.assertEqual(route["successful"], 9)
        self.assertEqual(route["failed"], 1)
        self.assertEqual(route["model_latency_ms"]["count"], 9)
        self.assertEqual(route["output_tokens"]["count"], 9)
        self.assertEqual(route["estimated_cost_cny"]["count"], 10)
        self.assertIn("50/50 baseline success", report)
        self.assertIn("50/50 after success", report)
        self.assertIn("Streaming mode verified", report)
        self.assertIn("Client TTFT P50/P95 ms", report)
        self.assertIn("Paired latency differences", report)
        self.assertIn("All routes total-latency P50 <=15% regression", report)
        self.assertIn("Model P50/P95 ms", report)
        self.assertIn("Output tokens P50/P95", report)
        self.assertIn("Estimated cost CNY", report)
        self.assertIn("| after | product_search | 9 | 1 |", report)
        self.assertNotIn("Production deployment:", report)

    def test_one_failed_request_prevents_100_of_100_and_conclusion_a(self):
        baseline, after = passing_pairs()
        after[9] = pair(
            "after",
            "product_search",
            9,
            success=False,
            total=2875.0,
            ttft=1000.0,
        )

        analysis = analyze_day3_phases(baseline, after)

        self.assertFalse(analysis["gates"]["success_100_of_100"])
        self.assertFalse(analysis["gates"]["all_pass"])
        self.assertEqual(analysis["conclusion"], "B")

    def test_direct_regression_over_15_percent_prevents_conclusion_a(self):
        baseline, after = passing_pairs()
        for candidate in after:
            if candidate["summary"]["route"] == "direct":
                candidate["summary"]["total_latency_ms"] = 2876.0

        analysis = analyze_day3_phases(baseline, after)

        self.assertFalse(
            analysis["gates"]["total_latency_p50_within_15_percent"]["direct"]
        )
        self.assertFalse(analysis["gates"]["all_pass"])
        self.assertEqual(analysis["conclusion"], "B")

    def test_single_delta_after_responses_fail_mode_verification(self):
        baseline, after = passing_pairs()
        for candidate in after:
            candidate["workload"]["delta_count"] = 1

        analysis = analyze_day3_phases(baseline, after)

        self.assertFalse(analysis["gates"]["streaming_mode_verified"])
        self.assertFalse(analysis["gates"]["all_pass"])
        self.assertEqual(analysis["conclusion"], "B")

    def test_route_mismatch_prevents_conclusion_a(self):
        baseline, after = passing_pairs()
        baseline[0]["summary"]["route"] = "knowledge"

        analysis = analyze_day3_phases(baseline, after)

        self.assertFalse(analysis["gates"]["target_routes_match"])
        self.assertFalse(analysis["gates"]["all_pass"])

    def test_missing_client_timing_prevents_conclusion_a(self):
        baseline, after = passing_pairs()
        after[0]["workload"]["client_first_delta_latency_ms"] = None

        analysis = analyze_day3_phases(baseline, after)

        self.assertFalse(analysis["gates"]["client_metrics_complete"])
        self.assertFalse(analysis["gates"]["all_pass"])


if __name__ == "__main__":
    unittest.main()
