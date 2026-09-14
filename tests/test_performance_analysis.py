import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from performance.analysis import (
    PRICING_SNAPSHOT,
    analyze_join,
    estimate_request_cost_cny,
    is_peak,
    join_attempts,
    load_summaries,
    load_workload_rows,
    nearest_rank,
    numeric_stats,
    parse_summary_line,
)


SUMMARY_LINE = (
    "2026-09-14T11:30:00+0800 level=INFO "
    "request_id=req-1 http_status=200 outcome=success "
    "latency_ms=1200.0 total_latency_ms=1200.0 error=- "
    "route=knowledge router_type=llm tool_calls=[] tool_call_count=0 "
    'retrieved_chunk_ids=[\"chunk-1\"] retrieval_used=true '
    "retrieved_count=1 executed_tool_names=[] tool_execution_count=0 "
    "tool_execution_success=null router_latency_ms=300.0 "
    "retrieval_latency_ms=20.0 tool_latency_ms=null "
    "model_latency_ms=1100.0 input_tokens=100 output_tokens=20 "
    "prompt_cache_hit_tokens=80 prompt_cache_miss_tokens=20 "
    "failure_layer=- failure_code=null citation_count=1 "
    "deduplicated_citation_count=0 invalid_source_count=0 "
    "citation_status=rendered answer_sanitized=False"
)


class SummaryParsingTests(unittest.TestCase):
    def test_parses_existing_key_value_summary(self):
        summary = parse_summary_line(SUMMARY_LINE)

        self.assertEqual(summary["request_id"], "req-1")
        self.assertEqual(summary["http_status"], 200)
        self.assertEqual(summary["total_latency_ms"], 1200.0)
        self.assertEqual(summary["retrieved_chunk_ids"], ["chunk-1"])
        self.assertIs(summary["retrieval_used"], True)
        self.assertIsNone(summary["tool_execution_success"])
        self.assertIsNone(summary["tool_latency_ms"])
        self.assertIsNone(summary["failure_layer"])
        self.assertIs(summary["answer_sanitized"], False)
        self.assertIsInstance(summary["timestamp"], datetime)
        self.assertIsNotNone(summary["timestamp"].tzinfo)

    def test_ignores_unrelated_server_log_lines(self):
        self.assertIsNone(parse_summary_line(
            "INFO: 127.0.0.1:50000 - POST /api/chat-stream HTTP/1.1 200 OK"
        ))

    def test_loader_accepts_exported_jsonl_and_reports_malformed_candidates(self):
        exported = {
            "timestamp": "2026-09-14T12:30:00+08:00",
            "request_id": "req-json",
            "http_status": 503,
            "outcome": "provider_unavailable",
            "total_latency_ms": 900.0,
            "route": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summaries.log"
            path.write_text(
                "\n".join((
                    json.dumps(exported),
                    "2026-09-14T12:31:00+0800 level=INFO request_id=broken",
                    "INFO: unrelated",
                )),
                encoding="utf-8",
            )
            summaries, issues = load_summaries(path)

        self.assertEqual([row["request_id"] for row in summaries], [
            "req-json",
        ])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["line_number"], 2)
        self.assertIn("missing required", issues[0]["reason"])
        self.assertNotIn("request_id=broken", json.dumps(issues))


class WorkloadJoinTests(unittest.TestCase):
    def test_loads_one_run_header_and_rows(self):
        records = [
            {"record_type": "run", "schema_version": "1.0"},
            {
                "record_type": "attempt",
                "workload_id": "single:one:1",
                "request_id": "req-1",
            },
            {
                "record_type": "skipped",
                "workload_id": "conversation:one:third",
                "request_id": None,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in records),
                encoding="utf-8",
            )
            header, rows = load_workload_rows(path)

        self.assertEqual(header["schema_version"], "1.0")
        self.assertEqual(len(rows), 2)

    def test_left_join_preserves_missing_duplicate_and_skipped_rows(self):
        attempts = [
            {
                "record_type": "attempt",
                "workload_id": "matched",
                "request_id": "req-1",
            },
            {
                "record_type": "attempt",
                "workload_id": "no-id",
                "request_id": None,
            },
            {
                "record_type": "attempt",
                "workload_id": "missing",
                "request_id": "req-missing",
            },
            {
                "record_type": "attempt",
                "workload_id": "duplicate",
                "request_id": "req-duplicate",
            },
            {
                "record_type": "skipped",
                "workload_id": "skipped",
                "request_id": None,
            },
        ]
        summaries = [
            {"request_id": "req-1", "outcome": "success"},
            {"request_id": "req-duplicate", "outcome": "success"},
            {"request_id": "req-duplicate", "outcome": "success"},
            {"request_id": "unrelated", "outcome": "success"},
        ]

        joined = join_attempts(attempts, summaries)

        self.assertEqual(len(joined["matched"]), 1)
        self.assertEqual(
            joined["matched"][0]["workload"]["workload_id"],
            "matched",
        )
        self.assertEqual(len(joined["missing_request_id"]), 1)
        self.assertEqual(len(joined["missing_summary"]), 1)
        self.assertEqual(len(joined["duplicate_summary"]), 1)
        self.assertEqual(len(joined["skipped"]), 1)
        self.assertEqual(joined["unrelated_summary_count"], 1)


class AggregationTests(unittest.TestCase):
    def test_nearest_rank_uses_one_based_ceiling(self):
        values = [1, 2, 3, 4, 100]

        self.assertEqual(nearest_rank(values, 50), 3)
        self.assertEqual(nearest_rank(values, 95), 100)
        self.assertIsNone(nearest_rank([], 50))
        with self.assertRaises(ValueError):
            nearest_rank(values, -1)
        with self.assertRaises(ValueError):
            nearest_rank(values, 101)

    def test_numeric_stats_reports_nulls_for_no_samples(self):
        self.assertEqual(numeric_stats([]), {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
        })

    def test_aggregate_uses_successful_joined_actual_routes(self):
        joined = self._joined_fixture()

        analysis = analyze_join(joined)

        self.assertEqual(analysis["samples"], {
            "attempted": 4,
            "successful": 3,
            "failed": 1,
            "skipped": 1,
            "single_turn_successful": 1,
            "multi_turn_successful": 2,
        })
        self.assertEqual(analysis["telemetry_join"]["matched"], 4)
        self.assertEqual(analysis["latency"]["overall"]["count"], 3)
        self.assertEqual(analysis["latency"]["overall"]["p50"], 200.0)
        self.assertEqual(
            set(analysis["latency"]["by_route"]),
            {"direct", "product_search"},
        )
        self.assertNotIn("target-only", analysis["latency"]["by_route"])
        self.assertEqual(
            analysis["latency"]["by_router_type"]["deterministic"]["count"],
            1,
        )

        product_stages = analysis["stage_latency"]["by_route"][
            "product_search"
        ]
        self.assertEqual(product_stages["retrieval"]["count"], 1)
        self.assertEqual(product_stages["tool"]["count"], 2)
        self.assertEqual(
            analysis["stage_latency"]["dominant_stage_distribution"],
            {"model": 3},
        )
        self.assertEqual(
            analysis["stage_latency"]["dominant_stage_ratio"]["count"],
            3,
        )
        self.assertNotIn("gap", json.dumps(analysis["stage_latency"]))
        self.assertNotIn("uninstrumented", json.dumps(analysis))

        self.assertEqual(analysis["tokens"]["coverage"], {
            "complete": 2,
            "eligible": 3,
            "excluded": 1,
        })
        self.assertEqual(analysis["tokens"]["overall"]["average_input"], 15.0)
        self.assertEqual(analysis["tokens"]["overall"]["average_output"], 3.0)
        self.assertEqual(analysis["tokens"]["overall"]["average_total"], 18.0)

        self.assertEqual(analysis["tools"]["overall_average_execution_count"], 1.0)
        self.assertEqual(analysis["tools"]["execution_count_distribution"], {
            "0": 1,
            "1": 1,
            "2": 1,
        })
        self.assertEqual(analysis["tools"]["ordered_name_sequences"], [
            {"tool_names": ["search_products"], "count": 1},
            {
                "tool_names": ["search_products", "search_products"],
                "count": 1,
            },
        ])

        self.assertEqual(analysis["failures"]["failure_rate"], 0.25)
        self.assertEqual(analysis["failures"]["by_layer"], {"model": 1})
        self.assertEqual(analysis["failures"]["by_code"], {
            "provider_timeout": 1,
        })

    def test_conversation_grouping_distinguishes_complete_and_partial(self):
        joined = self._joined_fixture()

        conversations = analyze_join(joined)["conversations"]

        self.assertEqual(conversations["complete_count"], 1)
        self.assertEqual(conversations["partial_count"], 1)
        self.assertEqual(
            conversations["items"][0]["turn_indices"],
            [1, 2],
        )
        self.assertIs(conversations["items"][0]["complete"], True)
        self.assertIs(conversations["items"][1]["complete"], False)
        self.assertEqual(conversations["items"][1]["skipped_turn_indices"], [3])

    def test_peak_windows_use_shanghai_time_and_half_open_boundaries(self):
        cases = (
            ("2026-09-14T08:59:59+08:00", False),
            ("2026-09-14T09:00:00+08:00", True),
            ("2026-09-14T11:59:59+08:00", True),
            ("2026-09-14T12:00:00+08:00", False),
            ("2026-09-14T14:00:00+08:00", True),
            ("2026-09-14T17:59:59+08:00", True),
            ("2026-09-14T18:00:00+08:00", False),
            ("2026-09-19T10:00:00+08:00", False),
        )
        for timestamp, expected in cases:
            with self.subTest(timestamp=timestamp):
                self.assertIs(is_peak(datetime.fromisoformat(timestamp)), expected)
        self.assertIs(
            is_peak(datetime(2026, 9, 14, 1, tzinfo=timezone.utc)),
            True,
        )
        with self.assertRaises(ValueError):
            is_peak(datetime(2026, 9, 14, 9))

    def test_estimated_cost_requires_complete_provider_usage(self):
        summary = {
            "timestamp": datetime.fromisoformat("2026-09-14T09:00:00+08:00"),
            "prompt_cache_hit_tokens": 800_000,
            "prompt_cache_miss_tokens": 200_000,
            "output_tokens": 100_000,
        }

        self.assertAlmostEqual(
            estimate_request_cost_cny(summary),
            0.8 * 0.10 + 0.2 * 3.00 + 0.1 * 9.00,
        )
        self.assertEqual(estimate_request_cost_cny({
            **summary,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "output_tokens": 0,
        }), 0.0)
        for field in (
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
            "output_tokens",
        ):
            self.assertIsNone(estimate_request_cost_cny({
                **summary,
                field: None,
            }))
        self.assertIsNone(estimate_request_cost_cny({
            **summary,
            "timestamp": "not-a-datetime",
        }))

    def test_cost_aggregation_is_estimated_and_separates_failed_requests(self):
        analysis = analyze_join(self._joined_fixture())
        cost = analysis["estimated_cost_cny"]

        self.assertEqual(PRICING_SNAPSHOT["configured_model"], "deepseek-v4-flash")
        self.assertNotIn("version", PRICING_SNAPSHOT)
        self.assertEqual(PRICING_SNAPSHOT["off_peak"], {
            "cache_hit_input": 0.05,
            "cache_miss_input": 1.50,
            "output": 4.50,
        })
        self.assertEqual(PRICING_SNAPSHOT["peak"], {
            "cache_hit_input": 0.10,
            "cache_miss_input": 3.00,
            "output": 9.00,
        })
        self.assertEqual(cost["coverage"], {
            "complete": 2,
            "eligible": 3,
            "excluded": 1,
        })
        self.assertEqual(cost["overall"]["count"], 2)
        self.assertEqual(cost["failed_requests"]["count"], 1)
        self.assertGreater(cost["failed_requests"]["total"], 0)
        self.assertNotIn("actual", json.dumps(cost).lower())
        self.assertNotIn("exact", json.dumps(cost).lower())
        self.assertNotIn("billed", json.dumps(cost).lower())

        conversations = analysis["conversations"]
        self.assertEqual(conversations["estimated_cost_cny"]["count"], 1)
        self.assertGreater(
            conversations["items"][0]["estimated_cost_cny"],
            0,
        )
        self.assertGreater(
            conversations["items"][1]["partial_estimated_cost_cny"],
            0,
        )
        self.assertNotIn("estimated_cost_cny", conversations["items"][1])

    @staticmethod
    def _joined_fixture():
        def pair(
            workload_id,
            request_id,
            *,
            kind,
            route,
            total,
            input_tokens,
            output_tokens,
            tool_count,
            tool_names,
            conversation_id=None,
            turn_index=None,
            outcome="success",
            http_status=200,
            failure_layer=None,
            failure_code=None,
            retrieval_latency=None,
            cache_complete=True,
        ):
            return {
                "workload": {
                    "record_type": "attempt",
                    "workload_id": workload_id,
                    "request_id": request_id,
                    "kind": kind,
                    "target_route": "target-only",
                    "conversation_id": conversation_id,
                    "turn_index": turn_index,
                },
                "summary": {
                    "timestamp": datetime.fromisoformat(
                        "2026-09-14T13:00:00+08:00"
                    ),
                    "request_id": request_id,
                    "http_status": http_status,
                    "outcome": outcome,
                    "route": route,
                    "router_type": (
                        "deterministic" if route == "direct" else "llm"
                    ),
                    "total_latency_ms": total,
                    "router_latency_ms": 10.0,
                    "retrieval_latency_ms": retrieval_latency,
                    "tool_latency_ms": 40.0 if tool_count else None,
                    "model_latency_ms": total * 0.6,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "prompt_cache_hit_tokens": (
                        (input_tokens or 0) // 2 if cache_complete else None
                    ),
                    "prompt_cache_miss_tokens": (
                        (input_tokens or 0) - (input_tokens or 0) // 2
                        if cache_complete else None
                    ),
                    "tool_call_count": 99,
                    "tool_execution_count": tool_count,
                    "executed_tool_names": tool_names,
                    "failure_layer": failure_layer,
                    "failure_code": failure_code,
                },
            }

        return {
            "matched": [
                pair(
                    "single", "r1", kind="single_turn", route="direct",
                    total=100.0, input_tokens=10, output_tokens=2,
                    tool_count=0, tool_names=[], cache_complete=False,
                ),
                pair(
                    "c1-1", "r2", kind="multi_turn",
                    route="product_search", total=200.0, input_tokens=20,
                    output_tokens=4, tool_count=1,
                    tool_names=["search_products"], conversation_id="c1",
                    turn_index=1, retrieval_latency=20.0,
                ),
                pair(
                    "c1-2", "r3", kind="multi_turn",
                    route="product_search", total=300.0, input_tokens=None,
                    output_tokens=6, tool_count=2,
                    tool_names=["search_products", "search_products"],
                    conversation_id="c1", turn_index=2,
                ),
                pair(
                    "c2-2", "r4", kind="multi_turn", route="knowledge",
                    total=400.0, input_tokens=30, output_tokens=5,
                    tool_count=0, tool_names=[], conversation_id="c2",
                    turn_index=2, outcome="error", http_status=504,
                    failure_layer="model", failure_code="provider_timeout",
                ),
            ],
            "missing_request_id": [],
            "missing_summary": [],
            "duplicate_summary": [],
            "skipped": [{
                "record_type": "skipped",
                "workload_id": "c2-3",
                "kind": "multi_turn",
                "conversation_id": "c2",
                "turn_index": 3,
            }],
            "unrelated_summary_count": 0,
        }


if __name__ == "__main__":
    unittest.main()
