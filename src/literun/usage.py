"""Token usage and timing tracking for agent runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenUsage:
    """Contains the token usage information for a given step or run."""

    input_tokens: int
    output_tokens: int
    cached_read_tokens: int | None = None
    cached_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    tool_use_tokens: int | None = None
    total_tokens: int | None = None

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Merge two usages; used when accumulating across agent turns."""

        def _add_opt(a: int | None, b: int | None) -> int | None:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_read_tokens=_add_opt(self.cached_read_tokens, other.cached_read_tokens),
            cached_write_tokens=_add_opt(self.cached_write_tokens, other.cached_write_tokens),
            reasoning_tokens=_add_opt(self.reasoning_tokens, other.reasoning_tokens),
            tool_use_tokens=_add_opt(self.tool_use_tokens, other.tool_use_tokens),
            total_tokens=self.resolved_total_tokens + other.resolved_total_tokens,
        )

    @property
    def resolved_total_tokens(self) -> int:
        """Resolve total tokens from provider value or fallback computation."""
        if self.total_tokens is not None:
            return self.total_tokens
        return (
            self.input_tokens
            + self.output_tokens
            + (self.cached_read_tokens or 0)
            + (self.cached_write_tokens or 0)
            + (self.reasoning_tokens or 0)
            + (self.tool_use_tokens or 0)
        )

    def dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_read_tokens": self.cached_read_tokens,
            "cached_write_tokens": self.cached_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "tool_use_tokens": self.tool_use_tokens,
            "total_tokens": self.resolved_total_tokens,
        }

    def __repr__(self) -> str:
        return (
            f"TokenUsage(input_tokens={self.input_tokens}, "
            f"output_tokens={self.output_tokens}, "
            f"cached_read_tokens={self.cached_read_tokens}, "
            f"cached_write_tokens={self.cached_write_tokens}, "
            f"reasoning_tokens={self.reasoning_tokens}, "
            f"tool_use_tokens={self.tool_use_tokens}, "
            f"total_tokens={self.total_tokens})"
        )


@dataclass
class Timing:
    """Contains the timing information for a given step or run."""

    start_time: float
    end_time: float | None = None

    @property
    def duration(self):
        return None if self.end_time is None else self.end_time - self.start_time

    def dict(self):
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
        }

    def __repr__(self) -> str:
        return (
            f"Timing(start_time={self.start_time}, "
            f"end_time={self.end_time}, "
            f"duration={self.duration})"
        )
