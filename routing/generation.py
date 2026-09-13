from collections.abc import Callable, Mapping, Sequence

from agent import AgentDeadline, SAFE_AGENT_ANSWER, normalize_agent_turn
from trace_models import FailureLayer

from .errors import set_failure_layer


def generate_answer(
    provider_messages: Sequence[Mapping[str, object]],
    *,
    deadline: AgentDeadline,
    complete_chat: Callable[[Sequence[Mapping[str, object]]], object],
) -> tuple[str, FailureLayer | None]:
    deadline.ensure_active()
    try:
        completion = complete_chat(provider_messages)
    except Exception as error:
        set_failure_layer(error, FailureLayer.GENERATION)
        raise
    deadline.ensure_active()
    turn = normalize_agent_turn(completion)
    if (
        turn is None
        or turn.tool_calls
        or turn.content is None
        or not turn.content.strip()
    ):
        return SAFE_AGENT_ANSWER, FailureLayer.GENERATION
    return turn.content, None


__all__ = ["generate_answer"]
