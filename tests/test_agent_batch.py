import importlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent import AgentDeadline
from routing import Route


def case_payload(case_id, route, *, tools=(), messages=None):
    return {
        "case_id": case_id,
        "messages": messages or [{"role": "user", "content": case_id}],
        "expected_route": route,
        "expected_tools": list(tools),
        "expected_answer": "2分：正确。1分：部分正确。0分：错误。",
    }


class RecordingObservationRunner:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.questions = []

    def run(self, messages, *, deadline):
        question = messages[-1].content
        self.questions.append(question)
        outcome = self.outcomes[question]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class AgentCaseLoaderTests(unittest.TestCase):
    def test_loads_exact_case_contract_and_reuses_chat_validation(self):
        from evaluation.agent import load_agent_eval_cases

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            payload = case_payload(
                "dev_exact_01",
                "exact_product",
                tools=({
                    "name": "get_product_details",
                    "required_args": {"product_id": "ly198"},
                },),
                messages=[
                    {"role": "user", "content": "LY198 的功率是多少？"},
                    {"role": "assistant", "content": "不超过 2W。"},
                    {"role": "user", "content": "频段呢？"},
                ],
            )
            path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

            cases = load_agent_eval_cases(path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "dev_exact_01")
        self.assertEqual(cases[0].expected_route, Route.EXACT_PRODUCT)
        self.assertEqual(cases[0].messages[-1].content, "频段呢？")
        self.assertEqual(
            cases[0].expected_tools[0].required_args,
            {"product_id": "ly198"},
        )

    def test_rejects_duplicate_ids_extra_fields_and_invalid_conversations(self):
        from evaluation.agent import load_agent_eval_cases

        invalid_payloads = (
            [case_payload("duplicate", "direct"), case_payload("duplicate", "direct")],
            [{**case_payload("extra", "direct"), "unexpected": True}],
            [case_payload(
                "roles",
                "direct",
                messages=[
                    {"role": "user", "content": "one"},
                    {"role": "user", "content": "two"},
                ],
            )],
        )
        for index, payloads in enumerate(invalid_payloads):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "cases.jsonl"
                path.write_text(
                    "".join(json.dumps(payload) + "\n" for payload in payloads),
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    load_agent_eval_cases(path)


class AgentBatchTests(unittest.TestCase):
    def test_scores_every_case_from_json_expectations_and_preserves_answers(self):
        from evaluation.agent import AgentEvalCase, run_agent_evaluation

        cases = tuple(AgentEvalCase.model_validate(payload) for payload in (
            case_payload("direct-pass", "direct"),
            case_payload(
                "exact-pass",
                "exact_product",
                tools=({
                    "name": "get_product_details",
                    "required_args": {"product_id": "ly198"},
                },),
            ),
            case_payload(
                "contact-error",
                "contact",
                tools=({"name": "get_contact_info", "required_args": {}},),
            ),
            case_payload("route-fail-action-pass", "knowledge"),
        ))
        outcomes = {
            "direct-pass": {
                "route": Route.DIRECT,
                "tool_calls": (),
                "final_answer": "direct answer",
            },
            "exact-pass": {
                "route": Route.EXACT_PRODUCT,
                "tool_calls": ({
                    "name": "get_product_details",
                    "arguments": {"product_id": "ly198", "locale": "zh-CN"},
                },),
                "final_answer": "exact answer",
            },
            "contact-error": RuntimeError("secret provider body"),
            "route-fail-action-pass": {
                "route": Route.DIRECT,
                "tool_calls": (),
                "final_answer": "wrong route answer",
            },
        }
        runner = RecordingObservationRunner(outcomes)

        report = run_agent_evaluation(
            cases,
            runner,
            deadline_factory=lambda: AgentDeadline.start(10, clock=lambda: 0),
        )

        self.assertEqual(
            runner.questions,
            ["direct-pass", "exact-pass", "contact-error", "route-fail-action-pass"],
        )
        self.assertEqual(report["summary"], {
            "total_cases": 4,
            "route_passed": 2,
            "route_accuracy": 0.5,
            "action_passed": 3,
            "action_accuracy": 0.75,
            "failed_case_ids": ["contact-error", "route-fail-action-pass"],
        })
        self.assertEqual(report["cases"][0]["final_answer"], "direct answer")
        self.assertEqual(report["cases"][1]["actual_tool_calls"][0]["arguments"], {
            "product_id": "ly198",
            "locale": "zh-CN",
        })
        error = report["cases"][2]
        self.assertEqual(error["status"], "error")
        self.assertIsNone(error["actual_route"])
        self.assertIsNone(error["actual_tool_calls"])
        self.assertIsNone(error["final_answer"])
        self.assertFalse(error["route_pass"])
        self.assertFalse(error["action_pass"])
        self.assertEqual(error["error_type"], "RuntimeError")
        self.assertNotIn("secret provider body", json.dumps(report))
        self.assertTrue(report["cases"][3]["action_pass"])

    def test_does_not_catch_base_exception(self):
        from evaluation.agent import AgentEvalCase, run_agent_evaluation

        case = AgentEvalCase.model_validate(case_payload("interrupt", "direct"))
        runner = RecordingObservationRunner({"interrupt": KeyboardInterrupt()})

        with self.assertRaises(KeyboardInterrupt):
            run_agent_evaluation(
                (case,),
                runner,
                deadline_factory=lambda: AgentDeadline.start(10, clock=lambda: 0),
            )


class AgentBatchCliTests(unittest.TestCase):
    def setUp(self):
        try:
            self.command = importlib.import_module("scripts.evaluate_agent_dev")
        except ModuleNotFoundError:
            self.fail("agent Dev evaluation CLI is not implemented")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "evaluation").mkdir()
        routes = (
            "direct",
            "product_search",
            "exact_product",
            "contact",
            "knowledge",
            "fallback",
        )
        payloads = [
            case_payload(f"dev_{route}_{index}", route)
            for route in routes
            for index in range(1, 5)
        ]
        (self.root / "evaluation/agent_dev_v1.jsonl").write_text(
            "".join(json.dumps(payload) + "\n" for payload in payloads),
            encoding="utf-8",
        )
        holdout_payloads = [
            case_payload(f"holdout_{route}_01", route)
            for route in routes
        ]
        (self.root / "evaluation/agent_holdout_v1.jsonl").write_text(
            "".join(json.dumps(payload) + "\n" for payload in holdout_payloads),
            encoding="utf-8",
        )

    def invoke(self, *arguments):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = self.command.main(["--root", str(self.root), *arguments])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_preflight_validates_dev24_without_building_production_dependencies(self):
        with patch.object(
            self.command,
            "_build_production_runner",
            side_effect=AssertionError("provider path constructed"),
        ):
            code, output, error = self.invoke()

        self.assertEqual(code, 0, error)
        self.assertEqual(output.splitlines(), [
            "Total cases: 24",
            "Mode: preflight (no provider calls)",
        ])
        self.assertFalse((self.root / "eval/results").exists())

    def test_holdout_requires_explicit_unlock(self):
        with patch.object(
            self.command,
            "_build_production_runner",
            side_effect=AssertionError("provider path constructed"),
        ):
            code, output, error = self.invoke("--split", "holdout")

        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        self.assertIn("Agent evaluation stopped: ValueError", error)
        self.assertFalse((self.root / "eval/results").exists())

    def test_holdout_preflight_validates_six_cases_without_building_dependencies(self):
        with patch.object(
            self.command,
            "_build_production_runner",
            side_effect=AssertionError("provider path constructed"),
        ):
            code, output, error = self.invoke(
                "--split",
                "holdout",
                "--allow-holdout",
            )

        self.assertEqual(code, 0, error)
        self.assertEqual(output.splitlines(), [
            "Total cases: 6",
            "Mode: preflight (no provider calls)",
        ])
        self.assertFalse((self.root / "eval/results").exists())

    def test_holdout_execute_writes_holdout_report(self):
        outcomes = {
            f"holdout_{route}_01": {
                "route": Route(route),
                "tool_calls": (),
                "final_answer": f"answer:{route}",
            }
            for route in (
                "direct",
                "product_search",
                "exact_product",
                "contact",
                "knowledge",
                "fallback",
            )
        }
        runner = RecordingObservationRunner(outcomes)
        with (
            patch.object(self.command, "_build_production_runner", return_value=runner),
            patch.object(self.command, "_configured_model_metadata", return_value={}),
        ):
            code, output, error = self.invoke(
                "--split",
                "holdout",
                "--allow-holdout",
                "--execute",
            )

        self.assertEqual(code, 0, error)
        self.assertIn("Total cases: 6", output)
        result_files = list(
            (self.root / "eval/results").glob("agent-holdout-v1-*.json")
        )
        self.assertEqual(len(result_files), 1)
        report = json.loads(result_files[0].read_text(encoding="utf-8"))
        self.assertEqual(report["evaluation_type"], "agent_holdout")
        self.assertEqual(
            report["metadata"]["dataset_path"],
            "evaluation/agent_holdout_v1.jsonl",
        )

    def test_execute_writes_report_and_prints_only_the_five_summary_lines(self):
        outcomes = {
            f"dev_{route}_{index}": {
                "route": Route(route),
                "tool_calls": (),
                "final_answer": f"answer:{route}:{index}",
            }
            for route in (
                "direct",
                "product_search",
                "exact_product",
                "contact",
                "knowledge",
                "fallback",
            )
            for index in range(1, 5)
        }
        runner = RecordingObservationRunner(outcomes)
        with (
            patch.object(self.command, "_build_production_runner", return_value=runner),
            patch.object(
                self.command,
                "_configured_model_metadata",
                return_value={
                    "generation_model": "fake-generation",
                    "embedding_provider": "fake-embedding-provider",
                    "embedding_model": "fake-embedding",
                },
            ),
        ):
            code, output, error = self.invoke("--execute")

        self.assertEqual(code, 0, error)
        lines = output.splitlines()
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[:4], [
            "Total cases: 24",
            "Route Accuracy: 24/24 (100.00%)",
            "Action Accuracy: 24/24 (100.00%)",
            "Failed case IDs: -",
        ])
        self.assertTrue(lines[4].startswith("Output result path: "))
        result_files = list((self.root / "eval/results").glob("agent-dev-v1-*.json"))
        self.assertEqual(len(result_files), 1)
        report = json.loads(result_files[0].read_text(encoding="utf-8"))
        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(report["evaluation_type"], "agent_dev")
        self.assertEqual(report["summary"]["route_accuracy"], 1.0)
        self.assertEqual(report["metadata"]["generation_model"], "fake-generation")
        self.assertEqual(report["cases"][-1]["final_answer"], "answer:fallback:4")

    def test_production_builder_injects_one_recording_executor_into_graph(self):
        from agent_graph import GraphRouteOrchestrator
        from evaluation.agent import AgentEvaluationRunner, RecordingToolExecutor

        retriever = object()
        tools = object()
        registry = {}
        router = object()
        compiled_graph = object()
        complete_chat = object()
        with (
            patch("services.retrieval.build_retriever", return_value=retriever),
            patch("services.tools.build_deterministic_tools", return_value=tools),
            patch("support_tools.build_tool_registry", return_value=registry),
            patch("routing.HybridRouter", return_value=router),
            patch("agent_graph.build_agent_graph", return_value=compiled_graph) as build_graph,
            patch("services.llm.complete_chat", complete_chat),
        ):
            runner = self.command._build_production_runner()

        self.assertIsInstance(runner, AgentEvaluationRunner)
        self.assertIsInstance(runner._executor, RecordingToolExecutor)
        self.assertIsInstance(runner._orchestrator, GraphRouteOrchestrator)
        self.assertIs(runner._orchestrator._compiled_graph, compiled_graph)
        nodes = build_graph.call_args.args[0]
        self.assertIs(nodes.executor, runner._executor)
        self.assertIs(nodes.retriever, retriever)
        self.assertIs(nodes.router, router)
        self.assertIs(nodes.complete_chat, complete_chat)


if __name__ == "__main__":
    unittest.main()
