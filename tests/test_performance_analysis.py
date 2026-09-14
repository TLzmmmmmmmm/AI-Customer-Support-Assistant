import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from performance.analysis import (
    join_attempts,
    load_summaries,
    load_workload_rows,
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


if __name__ == "__main__":
    unittest.main()
