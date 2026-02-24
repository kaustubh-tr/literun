"""Run items for tracking agent execution steps."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias
from dataclasses import dataclass, field

from .usage import TokenUsage


@dataclass
class BaseRunItem:
    """Base contract for all run items."""
    type: str = field(init=False)
    id: str | None = None
    role: str | None = None
    raw_item: Any | None = None
    token_usage: TokenUsage | None = None


@dataclass
class MessageOutputItem(BaseRunItem):
    """Message output from the LLM."""
    type: Literal["message.output.item"] = "message.output.item"
    role: Literal["assistant"] = "assistant"
    content: str | dict[str, Any] | None = None


@dataclass
class ToolCallItem(BaseRunItem):
    """Tool call requested by the LLM."""
    type: Literal["tool.call.item"] = "tool.call.item"
    role: Literal["tool_call"] = "tool_call"
    call_id: str | None = None
    name: str | None = None
    arguments: dict[str, Any] | None = None


@dataclass
class ToolCallOutputItem(BaseRunItem):
    """Output from a tool call execution."""
    type: Literal["tool.output.item"] = "tool.output.item"
    role: Literal["tool_output"] = "tool_output"
    call_id: str | None = None
    name: str | None = None
    result: str | dict[str, Any] | None = None


@dataclass
class ReasoningItem(BaseRunItem):
    """Reasoning content from the LLM."""
    type: Literal["reasoning.item"] = "reasoning.item"
    role: Literal["assistant"] = "assistant"
    signature: str | None = None
    summary: str | None = None


RunItem: TypeAlias = (
    MessageOutputItem | ToolCallItem | ToolCallOutputItem | ReasoningItem
)
