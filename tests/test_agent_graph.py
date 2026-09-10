import unittest
from collections import deque
from types import SimpleNamespace

from agent import AgentToolCall, ToolObservation
from agent_graph import AgentGraphNodes, build_agent_graph
from models import ChatMessage
from routing import HybridRouter, Route, RouteDecision, RoutingResult


class RecordingRouter:
    def __init__(self, decision, visited):
        self.decision = decision
        self.visited = visited

    def route(self, messages, *, deadline):
        self.visited.append("route")
        return RoutingResult(self.decision)


class RecordingRetriever:
    def __init__(self):
        self.queries = []

    def resolve_entities(self, query):
        self.queries.append(query)
        return []


class RecordingCompletion:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    def __call__(self, messages, *, tools=None):
        self.calls.append((messages, tools))
        return self.responses.popleft()


class RecordingDeadline:
    def __init__(self):
        self.checks = 0

    def ensure_active(self):
        self.checks += 1


def _completion(content):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content=content, tool_calls=None),
    )])


def _initial_state(*, deadline=None):
    message = ChatMessage(role="user", content="测试问题")
    return {
        "messages": (message,),
        "last_user_message": message,
        "deadline": deadline or RecordingDeadline(),
        "retrieval_hits": (),
        "tool_call": None,
        "tool_result": None,
        "answer": None,
        "sources": (),
        "error": None,
    }


def _recording_node(visited, name, update=None):
    def node(state):
        visited.append(name)
        if update is None:
            return {}
        return update(state)

    return node


def _nodes(visited, *, decision=None, router=None, **overrides):
    if router is None:
        router = RecordingRouter(
            decision or RouteDecision(Route.DIRECT),
            visited,
        )
    names = (
        "retrieve",
        "rag_generate",
        "agent_step",
        "execute_tool",
        "deterministic_tool",
        "direct",
        "fallback",
        "finalize",
    )
    return AgentGraphNodes(
        router=router,
        **{
            name: overrides.get(name, _recording_node(visited, name))
            for name in names
        },
    )


class AgentGraphTests(unittest.TestCase):
    def test_rag_path_retrieves_before_generation_and_finalization(self):
        visited = []
        graph = build_agent_graph(_nodes(
            visited,
            decision=RouteDecision(Route.KNOWLEDGE),
        ))

        graph.invoke(_initial_state())

        self.assertEqual(
            visited,
            ["route", "retrieve", "rag_generate", "finalize"],
        )

    def test_direct_path_skips_retrieval_and_tools(self):
        visited = []
        graph = build_agent_graph(_nodes(
            visited,
            decision=RouteDecision(Route.DIRECT),
        ))

        graph.invoke(_initial_state())

        self.assertEqual(visited, ["route", "direct", "finalize"])

    def test_fallback_path_goes_directly_to_finalization(self):
        visited = []
        graph = build_agent_graph(_nodes(
            visited,
            decision=RouteDecision(Route.FALLBACK),
        ))

        graph.invoke(_initial_state())

        self.assertEqual(visited, ["route", "fallback", "finalize"])

    def test_agent_path_executes_one_tool_then_finalizes(self):
        visited = []
        call = AgentToolCall(
            id="call-1",
            name="get_contact_info",
            arguments="{}",
        )
        observation = ToolObservation(
            content='{"ok":true}',
            success=True,
            reused=False,
            cache_key="get_contact_info:{}",
            tool_name="get_contact_info",
        )
        agent_visits = 0

        def agent_step(state):
            nonlocal agent_visits
            visited.append("agent_step")
            agent_visits += 1
            if agent_visits == 1:
                return {"tool_call": call}
            return {"tool_call": None, "answer": "agent answer"}

        def execute_tool(state):
            visited.append("execute_tool")
            self.assertEqual(state["tool_call"], call)
            return {"tool_result": observation}

        graph = build_agent_graph(_nodes(
            visited,
            decision=RouteDecision(Route.KNOWLEDGE, agentic=True),
            agent_step=agent_step,
            execute_tool=execute_tool,
        ))

        result = graph.invoke(
            _initial_state(),
            config={"recursion_limit": 10},
        )

        self.assertEqual(
            visited,
            [
                "route",
                "agent_step",
                "execute_tool",
                "agent_step",
                "finalize",
            ],
        )
        self.assertEqual(result["answer"], "agent answer")
        self.assertEqual(result["tool_result"], observation)

    def test_non_agentic_routes_follow_production_mapping(self):
        cases = (
            (Route.PRODUCT_SEARCH, ["route", "deterministic_tool", "finalize"]),
            (Route.EXACT_PRODUCT, ["route", "deterministic_tool", "finalize"]),
            (Route.CONTACT, ["route", "deterministic_tool", "finalize"]),
            (Route.KNOWLEDGE, ["route", "retrieve", "rag_generate", "finalize"]),
            (Route.DIRECT, ["route", "direct", "finalize"]),
            (Route.FALLBACK, ["route", "fallback", "finalize"]),
        )

        for route, expected in cases:
            with self.subTest(route=route):
                visited = []
                graph = build_agent_graph(_nodes(
                    visited,
                    decision=RouteDecision(route, agentic=False),
                ))

                graph.invoke(_initial_state())

                self.assertEqual(visited, expected)

    def test_agentic_precedes_every_production_route(self):
        for route in Route:
            with self.subTest(route=route):
                visited = []
                graph = build_agent_graph(_nodes(
                    visited,
                    decision=RouteDecision(route, agentic=True),
                ))

                graph.invoke(_initial_state())

                self.assertEqual(
                    visited,
                    ["route", "agent_step", "finalize"],
                )

    def test_route_node_uses_real_hybrid_router_contract(self):
        visited = []
        retriever = RecordingRetriever()
        complete = RecordingCompletion([_completion("direct")])
        router = HybridRouter(
            retriever=retriever,
            complete_chat=complete,
        )
        deadline = RecordingDeadline()
        first = ChatMessage(role="user", content="先前的问题")
        answer = ChatMessage(role="assistant", content="先前的回答")
        question = ChatMessage(role="user", content="请说明相关情况")
        state = _initial_state(deadline=deadline)
        state["messages"] = (first, answer, question)
        state["last_user_message"] = question
        graph = build_agent_graph(_nodes(
            visited,
            router=router,
        ))

        result = graph.invoke(state)

        self.assertEqual(
            result["route_decision"],
            RouteDecision(Route.DIRECT),
        )
        self.assertEqual(retriever.queries, ["请说明相关情况"])
        self.assertEqual(deadline.checks, 2)
        self.assertEqual(len(complete.calls), 1)
        provider_messages, tools = complete.calls[0]
        self.assertIsNone(tools)
        self.assertEqual(
            provider_messages[1:],
            [
                {"role": "user", "content": "先前的问题"},
                {"role": "assistant", "content": "先前的回答"},
                {"role": "user", "content": "请说明相关情况"},
            ],
        )
        self.assertEqual(visited, ["direct", "finalize"])


if __name__ == "__main__":
    unittest.main()
