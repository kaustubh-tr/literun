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
    new_items: list[RunItem]
    token_usage: TokenUsage | None
    timing: Timing

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
    event: StreamEvent
    token_usage: TokenUsage | None
    timing: Timing
    """Cumulative token usage up to this event when model usage is available."""

    def dict(self):
        return {
            "output": self.output,
            "event": self.event,
            "token_usage": self.token_usage.dict() if self.token_usage else None,
            "timing": self.timing.dict(),
        }
