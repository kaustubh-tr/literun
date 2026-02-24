"""OpenAI stream adapter for processing streaming responses."""

from __future__ import annotations

import json
from typing import Any

from ...usage import TokenUsage
from ...events import (
    StreamEvent,
    MessageOutputStreamDelta,
    MessageOutputStreamDone,
    ToolCallStreamDelta,
    ToolCallStreamDone,
    ReasoningStreamDelta,
    ReasoningStreamDone,
    OtherStreamEvent,
    StreamStartEvent,
    StreamEndEvent,
)
from ...constants import ToolCall
from ..base import StreamAdapter, AdapterMixin


class OpenAIStreamAdapter(StreamAdapter, AdapterMixin):
    """Adapter for OpenAI streaming responses."""

    supports_streaming: bool = True

    def __init__(self):
        self._tool_call_registry: dict[str, dict[str, Any]] = {}
        self._stream_started: bool = False
        self._stream_end: bool = False
        self._turn_has_tool_calls: bool = False
        self._turn_text_done: bool = False

    def _register_tool_call_from_output_item(self, item: Any) -> None:
        item_id = getattr(item, "id", None)
        if not item_id:
            return

        self._tool_call_registry[item_id] = {
            "item_id": item_id,
            "call_id": getattr(item, "call_id", None),
            "name": getattr(item, "name", None),
            "arguments_buffer": "",
        }

    def _process_chunk(self, chunk: Any) -> StreamEvent:
        """Process a single chunk and return the appropriate StreamEvent."""
        if not hasattr(chunk, "type"):
            return None

        event_type = chunk.type

        if event_type == "response.created":
            self._tool_call_registry.clear()
            self._stream_end = False
            self._turn_has_tool_calls = False
            self._turn_text_done = False

            if self._stream_started is False:
                self._stream_started = True
                response = getattr(chunk, "response", None)
                return StreamStartEvent(
                    id=getattr(response, "id", None),
                    raw_event=chunk,
                    token_usage=self.extract_token_usage(chunk),
                )
            return None

        if event_type == "response.output_text.delta":
            return MessageOutputStreamDelta(
                id=getattr(chunk, "item_id", None),
                delta=getattr(chunk, "delta", None),
                raw_event=chunk,
            )

        if event_type == "response.output_text.done":
            self._stream_end = True
            self._turn_text_done = True
            return MessageOutputStreamDone(
                id=getattr(chunk, "item_id", None),
                output=getattr(chunk, "text", None),
                raw_event=chunk,
            )

        if event_type == "response.function_call_arguments.delta":
            self._turn_has_tool_calls = True
            item_id = getattr(chunk, "item_id", None)
            if not item_id:
                return OtherStreamEvent(
                    id=None,
                    raw_event=chunk,
                    token_usage=None,
                )

            if item_id not in self._tool_call_registry:
                return OtherStreamEvent(
                    id=None,
                    raw_event=chunk,
                    token_usage=None,
                )

            meta = self._tool_call_registry[item_id]
            delta = getattr(chunk, "delta", None) or ""
            meta["arguments_buffer"] = f"{meta.get('arguments_buffer', '')}{delta}"
            call_id = meta.get("call_id")
            name = meta.get("name")
            if not isinstance(call_id, str) or not call_id:
                return OtherStreamEvent(
                    id=None,
                    raw_event=chunk,
                    token_usage=None,
                )
            if not isinstance(name, str) or not name:
                return OtherStreamEvent(
                    id=None,
                    raw_event=chunk,
                    token_usage=None,
                )

            return ToolCallStreamDelta(
                id=item_id,
                name=name,
                call_id=call_id,
                delta=delta,
                raw_event=chunk,
            )

        if event_type == "response.function_call_arguments.done":
            self._turn_has_tool_calls = True
            item_id = getattr(chunk, "item_id", None)
            if not item_id:
                return OtherStreamEvent(
                    id=None,
                    raw_event=chunk,
                    token_usage=None,
                )

            if item_id not in self._tool_call_registry:
                return OtherStreamEvent(
                    id=None,
                    raw_event=chunk,
                    token_usage=None,
                )

            meta = self._tool_call_registry[item_id]
            raw_arguments = getattr(chunk, "arguments", None)
            if raw_arguments is None:
                raw_arguments = meta.get("arguments_buffer", "")
            normalized_arguments = self.normalize_arguments(raw_arguments)

            call_id = meta.get("call_id")
            name = meta.get("name")
            if not isinstance(call_id, str) or not call_id:
                return OtherStreamEvent(
                    id=None,
                    raw_event=chunk,
                    token_usage=None,
                )
            if not isinstance(name, str) or not name:
                return OtherStreamEvent(
                    id=None,
                    raw_event=chunk,
                    token_usage=None,
                )

            return ToolCallStreamDone(
                id=item_id,
                name=name,
                call_id=call_id,
                output=normalized_arguments,
                raw_event=chunk,
            )

        if event_type == "response.reasoning_summary_text.delta":
            return ReasoningStreamDelta(
                id=getattr(chunk, "item_id", None),
                delta=getattr(chunk, "delta", None),
                raw_event=chunk,
            )

        if event_type == "response.reasoning_summary_text.done":
            return ReasoningStreamDone(
                id=getattr(chunk, "item_id", None),
                output=getattr(chunk, "text", None),
                raw_event=chunk,
            )

        if event_type == "response.output_item.added":
            item = getattr(chunk, "item", None)
            if getattr(item, "type", None) == "function_call":
                self._turn_has_tool_calls = True
                self._register_tool_call_from_output_item(item)
            return None

        if event_type == "response.completed":
            response = getattr(chunk, "response", None)
            usage = self.extract_token_usage(chunk)
            response_id = getattr(response, "id", None)
            if self._turn_has_tool_calls is True:
                return OtherStreamEvent(
                    id=response_id,
                    raw_event=chunk,
                    token_usage=usage,
                )
            if self._turn_text_done is True:
                return StreamEndEvent(
                    id=response_id,
                    raw_event=chunk,
                    token_usage=usage,
                )
            return OtherStreamEvent(
                id=response_id,
                raw_event=chunk,
                token_usage=usage,
            )

        return OtherStreamEvent(
            id=None,
            raw_event=chunk,
            token_usage=self.extract_token_usage(chunk),
        )

    def extract_token_usage(self, chunk) -> TokenUsage | None:
        """Extract token usage from OpenAI response."""
        if getattr(chunk, "type", None) != "response.completed":
            return None

        response = getattr(chunk, "response", None)
        usage = getattr(response, "usage", None) if response is not None else None
        if usage is None:
            return None

        cached_tokens = (
            getattr(usage.input_tokens_details, "cached_tokens", 0)
            if hasattr(usage, "input_tokens_details") and usage.input_tokens_details
            else 0
        )
        reasoning_tokens = (
            getattr(usage.output_tokens_details, "reasoning_tokens", 0)
            if hasattr(usage, "output_tokens_details") and usage.output_tokens_details
            else 0
        )
        return TokenUsage(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cached_read_tokens=int(cached_tokens or 0),
            reasoning_tokens=int(reasoning_tokens or 0),
            total_tokens=getattr(usage, "total_tokens", None),
        )

    def build_tool_call_message(
        self, text: str, tool_calls: list[ToolCall]
    ) -> list[dict[str, Any]]:
        """Build OpenAI continuation messages for streamed tool turns."""
        messages: list[dict[str, Any]] = []
        if text:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            )

        for tool_call in tool_calls:
            arguments = (
                tool_call.arguments
                if isinstance(tool_call.arguments, dict)
                else self.normalize_arguments(tool_call.arguments)
            )
            if not isinstance(arguments, dict):
                arguments = {}
            messages.append(
                {
                    "type": "function_call",
                    "call_id": tool_call.call_id,
                    "name": tool_call.name,
                    "arguments": json.dumps(arguments),
                }
            )
        return messages

    def build_tool_output_message(
        self, call_id: str, tool_name: str, tool_output: str
    ) -> list[dict[str, Any]]:
        """Build OpenAI tool result message."""
        return [
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": tool_output,
            }
        ]
