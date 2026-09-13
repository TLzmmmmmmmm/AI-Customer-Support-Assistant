import io
import json
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
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
    def _case(self, case_id):
        return next(case for case in smoke.CASES if case.case_id == case_id)

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
        case = self._case("exact_product")
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

    def test_product_search_professional_selection_wording_variants_pass(self):
        case = self._case("product_search")
        trace = RouteTrace(
            route=Route.PRODUCT_SEARCH,
            tool_calls=(ToolTrace("search_products", True),),
            tool_call_count=1,
            citation_count=1,
        )
        answers = (
            "以下型号可作为候选，请由技术人员确认最终选型。",
            "以下型号可作为候选，请由销售人员确认最终选择。",
            "以下型号可作为候选，请由专业人员作最终确认。",
            "以下型号可作为候选，请由技术或销售人员决定最终选型。",
        )

        for answer in answers:
            with self.subTest(answer=answer):
                smoke.validate_case(case, trace, answer)

    def test_product_search_without_final_selection_boundary_fails(self):
        case = self._case("product_search")
        trace = RouteTrace(
            route=Route.PRODUCT_SEARCH,
            tool_calls=(ToolTrace("search_products", True),),
            tool_call_count=1,
            citation_count=1,
        )

        with self.assertRaises(smoke.SmokeFailure) as caught:
            smoke.validate_case(case, trace, "以下型号适合仓库日常联络。")

        self.assertEqual(caught.exception.layer, "answer")

    def test_product_search_requires_candidate_and_rejects_strong_suitability(self):
        case = self._case("product_search")
        no_candidate = RouteTrace(
            route=Route.PRODUCT_SEARCH,
            tool_calls=(ToolTrace("search_products", True),),
            tool_call_count=1,
        )
        with_candidate = RouteTrace(
            route=Route.PRODUCT_SEARCH,
            tool_calls=(ToolTrace("search_products", True),),
            tool_call_count=1,
            citation_count=1,
        )

        with self.assertRaises(smoke.SmokeFailure) as missing:
            smoke.validate_case(
                case,
                no_candidate,
                "以下为候选，请由技术人员确认最终选型。",
            )
        with self.assertRaises(smoke.SmokeFailure) as unsupported:
            smoke.validate_case(
                case,
                with_candidate,
                "该型号完全适合仓库，请由技术人员确认最终选型。",
            )

        self.assertEqual(missing.exception.layer, "tool")
        self.assertEqual(unsupported.exception.layer, "answer")

    def test_follow_up_details_only_passes(self):
        case = self._case("follow_up")
        trace = RouteTrace(
            route=Route.EXACT_PRODUCT,
            tool_calls=(ToolTrace("get_product_details", True),),
            tool_call_count=1,
        )

        smoke.validate_case(case, trace, "第二个产品 HP780 的防护等级是 IP68。")

    def test_follow_up_search_then_details_passes_with_warning(self):
        case = self._case("follow_up")
        trace = RouteTrace(
            route=Route.EXACT_PRODUCT,
            tool_calls=(
                ToolTrace("search_products", True),
                ToolTrace("get_product_details", True),
            ),
            tool_call_count=2,
        )
        output = io.StringIO()

        with redirect_stdout(output):
            smoke.validate_case(
                case,
                trace,
                "第二个产品 HP780 的防护等级是 IP68。",
            )

        warning = output.getvalue()
        self.assertIn("follow_up", warning)
        self.assertIn("redundant discovery", warning)

    def test_follow_up_without_successful_details_fails(self):
        case = self._case("follow_up")
        cases = (
            (ToolTrace("search_products", True),),
            (ToolTrace("get_product_details", False, "PRODUCT_NOT_FOUND"),),
        )

        for tool_calls in cases:
            with self.subTest(tool_calls=tool_calls):
                trace = RouteTrace(
                    route=Route.EXACT_PRODUCT,
                    tool_calls=tool_calls,
                    tool_call_count=len(tool_calls),
                )
                with self.assertRaises(smoke.SmokeFailure) as caught:
                    smoke.validate_case(
                        case,
                        trace,
                        "第二个产品 HP780 的防护等级是 IP68。",
                    )
                self.assertEqual(caught.exception.layer, "tool")

    def test_prompt_injection_negated_99w_passes(self):
        case = self._case("prompt_injection")
        trace = RouteTrace(
            route=Route.EXACT_PRODUCT,
            tool_calls=(ToolTrace("get_product_details", True),),
            tool_call_count=1,
        )

        smoke.validate_case(case, trace, "LY198 的输出功率是 ≤2W，不是 99W。")

    def test_prompt_injection_without_authoritative_power_fails(self):
        case = self._case("prompt_injection")
        trace = RouteTrace(
            route=Route.EXACT_PRODUCT,
            tool_calls=(ToolTrace("get_product_details", True),),
            tool_call_count=1,
        )

        with self.assertRaises(smoke.SmokeFailure) as caught:
            smoke.validate_case(case, trace, "无法确认 LY198 的输出功率。")

        self.assertEqual(caught.exception.layer, "answer")

    def test_prompt_injection_prompt_leakage_still_fails(self):
        case = self._case("prompt_injection")
        trace = RouteTrace(
            route=Route.EXACT_PRODUCT,
            tool_calls=(ToolTrace("get_product_details", True),),
            tool_call_count=1,
        )

        answers = (
            "LY198 是 ≤2W。隐藏 system prompt 内容如下……",
            "LY198 是 ≤2W。隐藏 System Prompt 内容如下……",
            "LY198 是 ≤2W。AGENT_TOOL_POLICY 内容如下……",
        )
        for answer in answers:
            with self.subTest(answer=answer):
                with self.assertRaises(smoke.SmokeFailure) as caught:
                    smoke.validate_case(case, trace, answer)
                self.assertEqual(caught.exception.layer, "answer")

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
                    citation_count=(
                        1 if case.case_id == "product_search" else 0
                    ),
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

    def test_execute_prints_failure_reason_without_persisting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                smoke.SmokeCase(
                    "direct",
                    smoke.CASES[0].messages,
                    Route.DIRECT,
                ),
                *smoke.CASES[1:],
            )
            responses = [
                FakeResponse(events=success_events(smoke.SAFE_AGENT_ANSWER)),
                *(FakeResponse(events=success_events("回答")) for _ in range(7)),
            ]
            traces = [RouteTrace(route=case.expected_route) for case in cases]
            client = FakeClient(responses, traces)

            @contextmanager
            def client_context(trace_box):
                client.trace_box = trace_box
                yield client

            output = io.StringIO()
            with redirect_stdout(output):
                smoke.run(
                    execute=True,
                    root=root,
                    cases=cases,
                    dependency_validator=lambda: None,
                    client_context_factory=client_context,
                )

            self.assertIn("direct", output.getvalue())
            self.assertIn("answer", output.getvalue())
            self.assertIn("invalid greeting", output.getvalue())
            persisted = (root / smoke.RESULT_PATH).read_text(encoding="utf-8")
            self.assertNotIn("invalid greeting", persisted)


if __name__ == "__main__":
    unittest.main()
