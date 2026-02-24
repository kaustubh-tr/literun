"""Constants and Enums for the literun package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any

Role = Literal["system", "user", "assistant"]

ContentType = Literal["text", "tool_call", "tool_output", "reasoning"]

DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 60.0  # seconds
DEFAULT_MAX_TOOL_CALLS_LIMIT = 10
DEFAULT_MAX_ITERATIONS_LIMIT = 20

@dataclass
class ToolCall:
    """Structured data for tool call information."""

    call_id: str
    name: str
    arguments: dict[str, Any] | str


Message = dict[str, Any]
