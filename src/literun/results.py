"""Results for agent runs."""

from __future__ import annotations

from dataclasses import dataclass

from .items import RunItem
from .events import StreamEvent
from .usage import TokenUsage, Timing


@dataclass
class RunResult:
    """Result of a non-streaming agent run."""

    output: str
    """Final text output produced by the agent."""
    new_items: list[RunItem]
    """All run items (messages, tool calls, tool outputs) generated during the run."""
    token_usage: TokenUsage | None
    """Aggregated token usage across all turns, or None if unavailable."""
    timing: Timing
    """Wall-clock timing for the entire run."""

    def dict(self):
        return {
            "output": self.output,
            "new_items": self.new_items,
            "token_usage": self.token_usage.dict() if self.token_usage else None,
            "timing": self.timing.dict(),
        }


@dataclass
class RunStreamEvent:
    """Event from a streaming agent run."""

    output: str
    """Cumulative text output produced by the agent up to this event."""
    event: StreamEvent
    """The underlying stream event emitted by the provider adapter."""
    token_usage: TokenUsage | None
    """Cumulative token usage up to this event when model usage is available."""
    timing: Timing
    """Wall-clock timing snapshot at the point this event was emitted."""

    def dict(self):
        return {
            "output": self.output,
            "event": self.event,
            "token_usage": self.token_usage.dict() if self.token_usage else None,
            "timing": self.timing.dict(),
        }
