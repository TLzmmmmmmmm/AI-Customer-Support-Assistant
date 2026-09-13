import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from routing import Route, RouteTrace
from trace_models import ToolTrace

from scripts import day7_production_smoke as smoke


class FakeResponse:
    def __init__(self, *, events=(), status_code=200, request_id="request-1"):
        self.status_code = status_code
        self.headers = {
            "content-type": (
                "application/x-ndjson"
                if status_code == 200
                else "application/json"
            ),
            "X-Request-ID": request_id,
        }
        self.text = "".join(
            json.dumps(event, ensure_ascii=False) + "\n"
            for event in events
        )

    def json(self):
        return {"error": {"code": "provider_error"}}


class FakeClient:
    def __init__(self, responses, traces):
        self.responses = iter(responses)
        self.traces = iter(traces)
        self.payloads = []
        self.trace_box = None

    def post(self, path, *, json):
        self.payloads.append((path, json))
        self.trace_box["trace"] = next(self.traces)
        return next(self.responses)


def success_events(answer="回答", *, citations=False):
    events = [{"type": "delta", "content": answer}]
    if citations:
        events.append({
            "type": "citations",
            "heading": "参考资料：",
            "items": [{"title": "资料", "url": "https://example.com"}],
        })
    events.append({"type": "done"})
    return events


class Day7ProductionSmokeTests(unittest.TestCase):
    def test_fixed_matrix_has_eight_unique_cases_and_six_routes(self):
        self.assertEqual(len(smoke.CASES), 8)
        self.assertEqual(len({case.case_id for case in smoke.CASES}), 8)
        self.assertEqual(
            {case.expected_route for case in smoke.CASES},
            set(Route),
        )

    def test_preflight_validates_without_creating_client_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def validate_dependencies():
                calls.append("validated")

            def forbidden_client(_):
                self.fail("preflight must not construct the production client")

            result = smoke.run(
                root=root,
                dependency_validator=validate_dependencies,
                client_context_factory=forbidden_client,
            )

            self.assertEqual(result, 0)
            self.assertEqual(calls, ["validated"])
            self.assertFalse((root / smoke.RESULT_PATH).exists())

    def test_ndjson_accepts_delta_optional_citations_then_done(self):
        for citations in (False, True):
            with self.subTest(citations=citations):
                answer = smoke.parse_success_response(FakeResponse(
                    events=success_events("完整回答", citations=citations),
                ))
                self.assertEqual(answer, "完整回答")

    def test_ndjson_rejects_missing_request_id_and_events_after_done(self):
        missing_id = FakeResponse(events=success_events(), request_id="")
        after_done = FakeResponse(events=[
            {"type": "delta", "content": "回答"},
            {"type": "done"},
            {"type": "citations", "heading": "参考资料：", "items": []},
        ])

        with self.assertRaises(smoke.SmokeFailure) as missing:
            smoke.parse_success_response(missing_id)
        with self.assertRaises(smoke.SmokeFailure) as trailing:
            smoke.parse_success_response(after_done)

        self.assertEqual(missing.exception.layer, "transport")
        self.assertEqual(trailing.exception.layer, "ndjson")

    def test_case_validation_checks_route_tools_and_answer(self):
        case = next(case for case in smoke.CASES if case.case_id == "exact_product")
        trace = RouteTrace(
            route=Route.EXACT_PRODUCT,
            tool_calls=(ToolTrace("get_product_details", True),),
            tool_call_count=1,
        )

        smoke.validate_case(case, trace, "直流 13.6V±15%，交流 100-240V。")

        with self.assertRaises(smoke.SmokeFailure) as wrong_route:
            smoke.validate_case(
                case,
                RouteTrace(route=Route.DIRECT),
                "直流 13.6V±15%，交流 100-240V。",
            )
        with self.assertRaises(smoke.SmokeFailure) as missing_fact:
            smoke.validate_case(case, trace, "电源参数请参考产品资料。")

        self.assertEqual(wrong_route.exception.layer, "route")
        self.assertEqual(missing_fact.exception.layer, "answer")

    def test_execute_writes_only_redacted_case_results_and_uses_http_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contact = root / "knowledge/source/contact.json"
            contact.parent.mkdir(parents=True)
            contact.write_text(json.dumps({
                "duty_phone": "13911733859",
                "email": "lsk777@sina.com",
            }), encoding="utf-8")
            traces = [
                RouteTrace(
                    route=case.expected_route,
                    tool_calls=tuple(
                        ToolTrace(name, True) for name in case.required_tools
                    ),
                    tool_call_count=len(case.required_tools),
                    retrieved_chunk_ids=(
                        ("support:project-implementation:content",)
                        if case.case_id == "knowledge"
                        else ()
                    ),
                )
                for case in smoke.CASES
            ]
            answers = {
                "direct": "您好，请问有什么可以帮您？",
                "product_search": "候选产品如下，请由专业技术人员确认最终选型。",
                "exact_product": "直流 13.6V±15%，交流 100-240V。",
                "follow_up": "第二个产品 HP780 的防护等级是 IP68。",
                "knowledge": "主要包括现场勘测、频率备案、安装施工和系统调试。",
                "contact": "值班电话 13911733859，邮箱 lsk777@sina.com。",
                "unsupported_operation": smoke.SAFE_FALLBACK_ANSWER,
                "prompt_injection": "LY198 的输出功率是 ≤2W。",
            }
            responses = [
                FakeResponse(events=success_events(answers[case.case_id]))
                for case in smoke.CASES
            ]
            client = FakeClient(responses, traces)

            @contextmanager
            def client_context(trace_box):
                client.trace_box = trace_box
                yield client

            result = smoke.run(
                execute=True,
                root=root,
                dependency_validator=lambda: None,
                client_context_factory=client_context,
            )

            self.assertEqual(result, 0)
            rows = [
                json.loads(line)
                for line in (root / smoke.RESULT_PATH).read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(rows, [
                {"case_id": case.case_id, "status": "PASS"}
                for case in smoke.CASES
            ])
            self.assertEqual(
                [payload[0] for payload in client.payloads],
                ["/api/chat-stream"] * 8,
            )
            self.assertEqual(
                [payload[1]["messages"] for payload in client.payloads],
                [list(case.messages) for case in smoke.CASES],
            )
            serialized = (root / smoke.RESULT_PATH).read_text(encoding="utf-8")
            self.assertNotIn("13.6V", serialized)
            self.assertNotIn("13911733859", serialized)

    def test_execute_never_overwrites_existing_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / smoke.RESULT_PATH
            destination.parent.mkdir(parents=True)
            destination.write_text("existing\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                smoke.run(
                    execute=True,
                    root=root,
                    dependency_validator=lambda: None,
                )

            self.assertEqual(destination.read_text(encoding="utf-8"), "existing\n")

    def test_execute_records_failure_boundary_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contact = root / "knowledge/source/contact.json"
            contact.parent.mkdir(parents=True)
            contact.write_text(json.dumps({
                "duty_phone": "13911733859",
                "email": "lsk777@sina.com",
            }), encoding="utf-8")
            traces = [RouteTrace(route=case.expected_route) for case in smoke.CASES]
            responses = [
                FakeResponse(events=[{"type": "done"}]),
                *(FakeResponse(events=success_events("回答")) for _ in range(7)),
            ]
            client = FakeClient(responses, traces)

            @contextmanager
            def client_context(trace_box):
                client.trace_box = trace_box
                yield client

            result = smoke.run(
                execute=True,
                root=root,
                cases=(
                    smoke.CASES[0],
                    *(
                        smoke.SmokeCase(
                            case.case_id,
                            case.messages,
                            case.expected_route,
                        )
                        for case in smoke.CASES[1:]
                    ),
                ),
                dependency_validator=lambda: None,
                client_context_factory=client_context,
            )

            rows = [
                json.loads(line)
                for line in (root / smoke.RESULT_PATH).read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(result, 1)
            self.assertEqual(rows[0], {
                "case_id": "direct",
                "status": "FAIL",
                "failure_layer": "ndjson",
            })
            self.assertEqual(len(rows), 8)


if __name__ == "__main__":
    unittest.main()
