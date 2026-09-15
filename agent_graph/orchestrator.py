from collections.abc import Iterator, Sequence

from agent import AgentDeadline
from models import ChatMessage
from routing import RouteExecutionResult

from .nodes import (
    _enable_final_answer_streaming,
    _reset_final_answer_streaming,
)
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

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        deadline: AgentDeadline,
    ) -> Iterator[str | RouteExecutionResult]:
        final_state = None
        graph_stream = self._compiled_graph.stream(
            self._initial_state(messages, deadline=deadline),
            stream_mode=["custom", "values"],
        )
        try:
            while True:
                token = _enable_final_answer_streaming()
                try:
                    mode, value = next(graph_stream)
                except StopIteration:
                    break
                finally:
                    _reset_final_answer_streaming(token)
                if mode == "custom":
                    if not isinstance(value, str):
                        raise TypeError(
                            "graph custom stream payload must be a string"
                        )
                    if not value:
                        raise ValueError(
                            "graph custom stream payload must not be empty"
                        )
                    yield value
                elif mode == "values":
                    final_state = value
        finally:
            close = getattr(graph_stream, "close", None)
            if close is not None:
                token = _enable_final_answer_streaming()
                try:
                    close()
                finally:
                    _reset_final_answer_streaming(token)
        if not isinstance(final_state, dict) or "result" not in final_state:
            raise RuntimeError("graph stream ended without a final result")
        yield final_state["result"]


__all__ = ["GraphRouteOrchestrator"]
