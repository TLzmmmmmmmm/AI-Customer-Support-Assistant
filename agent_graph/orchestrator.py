from collections.abc import Sequence

from agent import AgentDeadline
from models import ChatMessage
from routing import RouteExecutionResult

from .state import AgentState


class GraphRouteOrchestrator:
    def __init__(self, compiled_graph) -> None:
        self._compiled_graph = compiled_graph

    def run(
        self,
        messages: Sequence[ChatMessage],
        *,
        deadline: AgentDeadline,
    ) -> RouteExecutionResult:
        initial_state: AgentState = {
            "messages": tuple(messages),
            "last_user_message": messages[-1],
            "deadline": deadline,
            "retrieval_hits": (),
            "tool_call": None,
            "tool_result": None,
            "answer": None,
            "sources": (),
            "error": None,
        }
        final_state = self._compiled_graph.invoke(initial_state)

        print("=== AI GRAPH DEBUG ===")
        print("route_decision:", final_state.get("route_decision"))
        print("tool_result:", final_state.get("tool_result"))
        print("agent_tool_traces:", final_state.get("agent_tool_traces"))
        print("retrieval_hits:", final_state.get("retrieval_hits"))
        print("agent_successful_sources:", final_state.get("agent_successful_sources"))
        print("sources:", final_state.get("sources"))
        print("result:", final_state.get("result"))
        print("======================")

        return final_state["result"]


__all__ = ["GraphRouteOrchestrator"]
