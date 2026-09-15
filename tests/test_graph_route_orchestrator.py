import io
import unittest
from contextlib import redirect_stdout

from agent import AgentDeadline
from agent_graph import GraphRouteOrchestrator
from models import ChatMessage
from routing import Route, RouteExecutionResult, RouteTrace


class RecordingCompiledGraph:
    def __init__(self, final_state):
        self.final_state = final_state
        self.states = []

    def invoke(self, state):
        self.states.append(state)
        return self.final_state


class StreamingCompiledGraph(RecordingCompiledGraph):
    def __init__(self, events):
        super().__init__(None)
        self.events = events
        self.stream_kwargs = None

    def stream(self, state, **kwargs):
        self.states.append(state)
        self.stream_kwargs = kwargs
        yield from self.events


class GraphRouteOrchestratorTests(unittest.TestCase):
    def test_stream_yields_only_custom_strings_then_the_final_result(self):
        result = RouteExecutionResult(
            answer="第一段第二段",
            trace=RouteTrace(route=Route.DIRECT),
        )
        graph = StreamingCompiledGraph([
            ("values", {"answer": None}),
            ("custom", "第一段"),
            ("values", {"answer": "第一段"}),
            ("custom", "第二段"),
            ("values", {"answer": "第一段第二段", "result": result}),
        ])
        orchestrator = GraphRouteOrchestrator(graph)
        messages = [ChatMessage(role="user", content="你好")]
        deadline = AgentDeadline(expires_at=10.0, clock=lambda: 0.0)

        items = list(orchestrator.stream(messages, deadline=deadline))

        self.assertEqual(items[:-1], ["第一段", "第二段"])
        self.assertIs(items[-1], result)
        self.assertEqual(
            graph.stream_kwargs["stream_mode"],
            ["custom", "values"],
        )

    def test_stream_fails_closed_without_a_final_result(self):
        graph = StreamingCompiledGraph([("values", {"answer": "完成"})])
        orchestrator = GraphRouteOrchestrator(graph)
        messages = [ChatMessage(role="user", content="你好")]
        deadline = AgentDeadline(expires_at=10.0, clock=lambda: 0.0)

        with self.assertRaises(RuntimeError):
            list(orchestrator.stream(messages, deadline=deadline))

    def test_stream_rejects_invalid_custom_payloads(self):
        messages = [ChatMessage(role="user", content="你好")]
        deadline = AgentDeadline(expires_at=10.0, clock=lambda: 0.0)
        invalid_payloads = ("", None, 1, {"answer": "leak"})

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                graph = StreamingCompiledGraph([("custom", payload)])
                orchestrator = GraphRouteOrchestrator(graph)
                with self.assertRaises((TypeError, ValueError)):
                    list(orchestrator.stream(messages, deadline=deadline))

    def test_run_builds_exact_request_state_and_returns_graph_result(self):
        result = RouteExecutionResult(
            answer="完成",
            trace=RouteTrace(route=Route.DIRECT),
        )
        graph = RecordingCompiledGraph({"result": result})
        orchestrator = GraphRouteOrchestrator(graph)
        messages = [
            ChatMessage(role="user", content="你好"),
            ChatMessage(role="assistant", content="您好"),
            ChatMessage(role="user", content="继续"),
        ]
        deadline = AgentDeadline(expires_at=10.0, clock=lambda: 0.0)

        output = io.StringIO()
        with redirect_stdout(output):
            actual = orchestrator.run(messages, deadline=deadline)

        self.assertIs(actual, result)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(len(graph.states), 1)
        state = graph.states[0]
        self.assertEqual(set(state), {
            "messages",
            "last_user_message",
            "deadline",
            "retrieval_hits",
            "tool_call",
            "tool_result",
            "answer",
            "sources",
            "error",
        })
        self.assertEqual(state["messages"], tuple(messages))
        self.assertIsInstance(state["messages"], tuple)
        self.assertIs(state["last_user_message"], messages[-1])
        self.assertIs(state["deadline"], deadline)
        self.assertEqual(state["retrieval_hits"], ())
        self.assertIsNone(state["tool_call"])
        self.assertIsNone(state["tool_result"])
        self.assertIsNone(state["answer"])
        self.assertEqual(state["sources"], ())
        self.assertIsNone(state["error"])

    def test_missing_graph_result_surfaces_key_error(self):
        orchestrator = GraphRouteOrchestrator(
            RecordingCompiledGraph({}),
        )
        messages = [ChatMessage(role="user", content="你好")]
        deadline = AgentDeadline(expires_at=10.0, clock=lambda: 0.0)

        with self.assertRaises(KeyError):
            orchestrator.run(messages, deadline=deadline)


if __name__ == "__main__":
    unittest.main()
