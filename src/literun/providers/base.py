"""Base adapter classes for provider-specific implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, AsyncIterator
import json

from ..items import RunItem
from ..constants import ToolCall
from ..usage import TokenUsage
from ..events import StreamEvent


class AdapterMixin:
    """Shared utilities for all adapters."""
    
    @staticmethod
    def normalize_arguments(raw_arguments: Any) -> dict[str, Any] | str:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return raw_arguments
            return raw_arguments
        return {}


class ResponseAdapter(ABC):
    """Base class for provider-specific response adapters."""

    @abstractmethod
    def extract_text(self, response: Any) -> str:
        """Extract text content from provider response."""
        ...

    @abstractmethod
    def extract_tool_calls(self, response: Any) -> list[ToolCall]:
        """Extract tool calls from provider response.

        Returns list of dicts with keys: call_id, name, arguments
        """
        ...

    @abstractmethod
    def extract_token_usage(self, response: Any) -> TokenUsage | None:
        """Extract token usage from provider response."""
        ...

    @abstractmethod
    def build_tool_call_message(self, response: Any) -> list[dict[str, Any]]:
        """Build tool call message for this provider.

        This returns the raw response content that should be appended to
        message history when tool calls are made.

        Args:
            response: Provider-specific response object

        Returns:
            Canonical messages to append to conversation history
        """
        ...

    @abstractmethod
    def build_tool_output_message(
        self, call_id: str, tool_name: str, tool_output: str
    ) -> list[dict[str, Any]]:
        """Build tool result message for this provider.

        Args:
            call_id: Tool call ID
            tool_name: Name of the tool
            tool_output: Output from tool execution

        Returns:
            Canonical messages
        """
        ...

    @abstractmethod
    def to_run_items(self, response: Any) -> list[RunItem]:
        """Convert provider response to normalized RunItems."""
        ...


class StreamAdapter(ABC):
    """Base class for streaming response adapters."""

    supports_streaming: bool = True
    """Whether this adapter supports live stream processing."""

    def process_stream(self, stream: Any) -> Iterator[StreamEvent]:
        """Process a streaming response and yield StreamEvents."""
        for chunk in stream:
            event = self._process_chunk(chunk)
            if event is not None:
                yield event

    async def aprocess_stream(self, stream: Any) -> AsyncIterator[StreamEvent]:
        """Process a streaming response asynchronously and yield StreamEvents."""
        async for chunk in stream:
            event = self._process_chunk(chunk)
            if event is not None:
                yield event

    def _process_chunk(self, chunk: Any) -> StreamEvent | None:
        """Process a single raw chunk into a StreamEvent. Override in each provider."""
        return None

    @abstractmethod
    def extract_token_usage(self, response: Any) -> TokenUsage | None:
        """Extract token usage from provider response."""
        ...

    @abstractmethod
    def build_tool_call_message(
        self,
        text: str,
        tool_calls: list[ToolCall],
    ) -> list[dict[str, Any]]:
        """Build tool-call continuation message from streamed message data.

        Args:
            text: Final text accumulated for the streamed message.
            tool_calls: Tool calls emitted in the streamed message.
        """
        ...

    @abstractmethod
    def build_tool_output_message(
        self, call_id: str, tool_name: str, tool_output: str
    ) -> list[dict[str, Any]]:
        """Build tool result message for this provider.

        Args:
            call_id: Tool call ID
            tool_name: Name of the tool
            tool_output: Output from tool execution

        Returns:
            Canonical messages
        """
        ...
