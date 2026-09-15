from collections.abc import Sequence

from agent import AgentDeadline
from models import ChatMessage
from routing import RouteExecutionResult

from .state import AgentState


class GraphRouteOrchestrator:
    def __init__(self, compiled_graph) -> None:
        self._compiled_graph = compiled_graph

    @staticmethod
    def _initial_state(
        messages: Sequence[ChatMessage],
        *,
        deadline: AgentDeadline,
    ) -> AgentState:
        return {
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

    def run(
        self,
        messages: Sequence[ChatMessage],
        *,
        deadline: AgentDeadline,
    ) -> RouteExecutionResult:
        final_state = self._compiled_graph.invoke(
            self._initial_state(messages, deadline=deadline),
        )
        return final_state["result"]

__all__ = ["GraphRouteOrchestrator"]
