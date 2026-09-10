import unittest

from agent import AgentToolCall, ToolObservation
from agent_graph import AgentGraphNodes, build_agent_graph
from models import ChatMessage


def _initial_state(route):
    message = ChatMessage(role="user", content="测试问题")
    return {
        "messages": (message,),
        "last_user_message": message,
        "route": route,
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


def _nodes(visited, **overrides):
    names = (
        "route",
        "retrieve",
        "rag_generate",
        "agent_step",
        "execute_tool",
        "direct",
        "fallback",
        "finalize",
    )
    return AgentGraphNodes(**{
        name: overrides.get(name, _recording_node(visited, name))
        for name in names
    })


class AgentGraphTests(unittest.TestCase):
    def test_rag_path_retrieves_before_generation_and_finalization(self):
        visited = []
        graph = build_agent_graph(_nodes(visited))

        graph.invoke(_initial_state("rag"))

        self.assertEqual(
            visited,
            ["route", "retrieve", "rag_generate", "finalize"],
        )

    def test_direct_path_skips_retrieval_and_tools(self):
        visited = []
        graph = build_agent_graph(_nodes(visited))

        graph.invoke(_initial_state("direct"))

        self.assertEqual(visited, ["route", "direct", "finalize"])

    def test_fallback_path_goes_directly_to_finalization(self):
        visited = []
        graph = build_agent_graph(_nodes(visited))

        graph.invoke(_initial_state("fallback"))

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
            agent_step=agent_step,
            execute_tool=execute_tool,
        ))

        result = graph.invoke(
            _initial_state("agent"),
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


if __name__ == "__main__":
    unittest.main()
