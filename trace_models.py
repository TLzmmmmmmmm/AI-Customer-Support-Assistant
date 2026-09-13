from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureLayer(str, Enum):
    ROUTING = "ROUTING"
    TOOL_SELECTION = "TOOL_SELECTION"
    ARGUMENT_GENERATION = "ARGUMENT_GENERATION"
    TOOL_EXECUTION = "TOOL_EXECUTION"
    RETRIEVAL = "RETRIEVAL"
    GENERATION = "GENERATION"


@dataclass(frozen=True)
class ToolTrace:
    name: str
    success: bool
    error_code: str | None = None
    reused: bool = False


__all__ = ["FailureLayer", "ToolTrace"]
