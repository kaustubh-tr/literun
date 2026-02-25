"""Streaming events for agent execution."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias
from dataclasses import dataclass, field

from .usage import TokenUsage


@dataclass
class BaseStreamEvent:
    """Base contract for all stream events."""
    type: str = field(init=False)
    id: str | None = None
    raw_event: Any | None = None
    token_usage: TokenUsage | None = None


@dataclass
class MessageOutputStreamDelta(BaseStreamEvent):
    """Delta event for streaming message output."""
    type: Literal["message.output.delta"] = "message.output.delta"
    delta: str | dict[str, Any] | None = None


@dataclass
class MessageOutputStreamDone(BaseStreamEvent):
    """Completion event for message output."""
    type: Literal["message.output.done"] = "message.output.done"
    output: str | dict[str, Any] | None = None


@dataclass
class ToolCallStreamDelta(BaseStreamEvent):
    """Delta event for streaming tool call."""
    type: Literal["tool.call.delta"] = "tool.call.delta"
    name: str | None = None
    call_id: str | None = None
    delta: str | dict[str, Any] | None = None


@dataclass
class ToolCallStreamDone(BaseStreamEvent):
    """Completion event for tool call."""
    type: Literal["tool.call.done"] = "tool.call.done"
    name: str | None = None
    call_id: str | None = None
    output: str | dict[str, Any] | None = None


@dataclass
class ToolCallOutputStreamDone(BaseStreamEvent):
    """Completion event for tool call output."""
    type: Literal["tool.output.done"] = "tool.output.done"
    name: str | None = None
    output: str | dict[str, Any] | None = None


@dataclass
class ReasoningStreamDelta(BaseStreamEvent):
    """Delta event for streaming reasoning."""
    type: Literal["reasoning.delta"] = "reasoning.delta"
    delta: str | dict[str, Any] | None = None


@dataclass
class ReasoningStreamDone(BaseStreamEvent):
    """Completion event for reasoning."""
    type: Literal["reasoning.done"] = "reasoning.done"
    output: str | dict[str, Any] | None = None


@dataclass
class OtherStreamEvent(BaseStreamEvent):
    """Generic event for other stream events."""
    type: Literal["other.event"] = "other.event"
    

@dataclass
class StreamErrorEvent(BaseStreamEvent):
    """Event for stream errors."""
    type: Literal["stream.error"] = "stream.error"
    error: str | None = None


@dataclass
class StreamStartEvent(BaseStreamEvent):
    """Event indicating the start of a stream."""
    type: Literal["stream.start"] = "stream.start"


@dataclass
class StreamEndEvent(BaseStreamEvent):
    """Event indicating the end of a stream."""
    type: Literal["stream.end"] = "stream.end"


StreamEvent: TypeAlias = (
    MessageOutputStreamDelta
    | MessageOutputStreamDone
    | ToolCallStreamDelta
    | ToolCallStreamDone
    | ToolCallOutputStreamDone
    | ReasoningStreamDelta
    | ReasoningStreamDone
    | OtherStreamEvent
    | StreamErrorEvent
    | StreamStartEvent
    | StreamEndEvent
    | None
)
