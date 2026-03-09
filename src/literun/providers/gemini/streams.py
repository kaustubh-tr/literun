"""Gemini stream adapter for processing streaming responses."""

from __future__ import annotations

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


class GeminiStreamAdapter(StreamAdapter, AdapterMixin):
    """Adapter for Gemini streaming responses."""

    supports_streaming: bool = True

    _PART_TEXT = "text"
    _PART_TOOL = "tool_call"
    _PART_REASONING = "reasoning"

    def __init__(self):
        self._part_kind_by_index: dict[int, str] = {}
        self._tool_call_state_by_index: dict[int, dict[str, Any]] = {}
        self._text_buffers: dict[int, str] = {}
        self._reasoning_buffers: dict[int, str] = {}
        self._reasoning_signatures: dict[int, str] = {}

        self._stream_started: bool = False
        self._stream_end: bool = False
        self._turn_has_tool_calls: bool = False
        self._turn_text_done: bool = False

    def _finalize_thought_part(self, index: int) -> str:
        summary = self._reasoning_buffers.get(index, "")
        return summary

    def _process_chunk(self, chunk: Any) -> StreamEvent:
        """Process a single chunk and yield StreamEvents."""
        if not hasattr(chunk, "event_type"):
            return None

        event_type = chunk.event_type

        if event_type == "interaction.start":
            self._part_kind_by_index.clear()
            self._tool_call_state_by_index.clear()
            self._text_buffers.clear()
            self._reasoning_buffers.clear()
            self._reasoning_signatures.clear()
            self._stream_end = False
            self._turn_has_tool_calls = False
            self._turn_text_done = False

            if self._stream_started is False:
                self._stream_started = True
                return StreamStartEvent(
                    id=getattr(chunk, "interaction_id", None),
                    raw_event=chunk,
                    token_usage=self.extract_token_usage(chunk),
                )
            return None

        if event_type == "content.start":
            content = getattr(chunk, "content", None)
            index = int(getattr(chunk, "index", 0) or 0)
            content_type = getattr(content, "type", None)

            if content_type == "thought":
                self._part_kind_by_index[index] = self._PART_REASONING
                if index not in self._reasoning_buffers:
                    self._reasoning_buffers[index] = ""
                return None

            if content_type == "function_call":
                self._part_kind_by_index[index] = self._PART_TOOL
                self._turn_has_tool_calls = True
                if content is not None:
                    state = self._tool_call_state_by_index.get(index)
                    if state is None:
                        state = {"call_id": None, "name": None, "arguments": {}}
                        self._tool_call_state_by_index[index] = state

                    call_id = getattr(content, "id", None)
                    name = getattr(content, "name", None)
                    if isinstance(call_id, str) and call_id:
                        state["call_id"] = call_id
                    if isinstance(name, str) and name:
                        state["name"] = name

                    raw_arguments = getattr(content, "arguments", None)
                    if raw_arguments is not None:
                        normalized_arguments = self.normalize_arguments(raw_arguments)
                        state["arguments"] = normalized_arguments
                return None

            if content_type == "text":
                self._part_kind_by_index[index] = self._PART_TEXT
                if index not in self._text_buffers:
                    self._text_buffers[index] = ""
                return None

        if event_type == "content.delta" and hasattr(chunk, "delta"):
            delta = chunk.delta
            index = int(getattr(chunk, "index", 0) or 0)
            delta_type = getattr(delta, "type", None)
            event_id = getattr(chunk, "event_id", None)

            if delta_type == "thought_summary":
                self._part_kind_by_index[index] = self._PART_REASONING
                thought_delta = (
                    getattr(getattr(delta, "content", None), "text", None)
                    or getattr(delta, "text", None)
                    or ""
                )
                self._reasoning_buffers[index] = (
                    self._reasoning_buffers.get(index, "") + thought_delta
                )
                return ReasoningStreamDelta(
                    id=event_id,
                    delta=thought_delta,
                    raw_event=chunk,
                )

            if delta_type == "thought_signature":
                self._part_kind_by_index[index] = self._PART_REASONING
                signature = getattr(delta, "signature", None)
                if isinstance(signature, str):
                    self._reasoning_signatures[index] = signature
                return OtherStreamEvent(
                    id=event_id,
                    raw_event=chunk,
                    token_usage=None,
                )

            if delta_type == "function_call":
                self._part_kind_by_index[index] = self._PART_TOOL
                self._turn_has_tool_calls = True
                state = self._tool_call_state_by_index.get(index)
                if state is None:
                    state = {"call_id": None, "name": None, "arguments": {}}
                    self._tool_call_state_by_index[index] = state

                provider_call_id = getattr(delta, "id", None)
                name = getattr(delta, "name", None)
                if isinstance(provider_call_id, str) and provider_call_id:
                    state["call_id"] = provider_call_id
                if isinstance(name, str) and name:
                    state["name"] = name

                raw_arguments = getattr(delta, "arguments", None)
                if raw_arguments is not None:
                    normalized_arguments = self.normalize_arguments(raw_arguments)
                    state["arguments"] = normalized_arguments

                call_id = state.get("call_id")
                if not isinstance(call_id, str) or not call_id:
                    return OtherStreamEvent(
                        id=event_id,
                        raw_event=chunk,
                        token_usage=None,
                    )
                return ToolCallStreamDelta(
                    id=event_id,
                    name=state.get("name"),
                    call_id=state.get("call_id"),
                    delta=state.get("arguments"),
                    raw_event=chunk,
                )

            if delta_type == "text":
                self._part_kind_by_index[index] = self._PART_TEXT
                text_delta = getattr(delta, "text", "") or ""
                self._text_buffers[index] = self._text_buffers.get(index, "") + text_delta
                return MessageOutputStreamDelta(
                    id=event_id,
                    delta=text_delta,
                    raw_event=chunk,
                )

        if event_type == "content.stop":
            index = int(getattr(chunk, "index", 0) or 0)
            part_kind = self._part_kind_by_index.get(index)
            event_id = getattr(chunk, "event_id", None)

            if part_kind == self._PART_REASONING:
                thought = self._finalize_thought_part(index)
                return ReasoningStreamDone(
                    id=event_id,
                    output=thought,
                    raw_event=chunk,
                )

            if part_kind == self._PART_TOOL:
                state = self._tool_call_state_by_index.get(index, {})
                call_id = state.get("call_id")
                name = state.get("name")
                arguments = state.get("arguments", {})
                if not isinstance(call_id, str) or not call_id:
                    return OtherStreamEvent(
                        id=event_id,
                        raw_event=chunk,
                        token_usage=None,
                    )
                if not isinstance(name, str) or not name:
                    return OtherStreamEvent(
                        id=event_id,
                        raw_event=chunk,
                        token_usage=None,
                    )

                return ToolCallStreamDone(
                    id=event_id,
                    name=name,
                    call_id=call_id,
                    output=arguments,
                    raw_event=chunk,
                )

            if part_kind == self._PART_TEXT:
                text = self._text_buffers.get(index, "")
                self._stream_end = True
                self._turn_text_done = True
                return MessageOutputStreamDone(
                    id=event_id,
                    output=text,
                    raw_event=chunk,
                )

        if event_type == "interaction.complete":
            usage = self.extract_token_usage(chunk)
            interaction_id = getattr(chunk, "interaction_id", None)
            if self._turn_has_tool_calls is True:
                return OtherStreamEvent(
                    id=interaction_id,
                    raw_event=chunk,
                    token_usage=usage,
                )
            if self._turn_text_done is True:
                return StreamEndEvent(
                    id=interaction_id,
                    raw_event=chunk,
                    token_usage=usage,
                )
            return OtherStreamEvent(
                id=interaction_id,
                raw_event=chunk,
                token_usage=usage,
            )

        return OtherStreamEvent(
            id=None,
            raw_event=chunk,
            token_usage=self.extract_token_usage(chunk),
        )

    def extract_token_usage(self, chunk) -> TokenUsage | None:
        """Extract final turn token usage from interaction.complete only."""
        if getattr(chunk, "event_type", None) != "interaction.complete":
            return None

        interaction = getattr(chunk, "interaction", None)
        usage = getattr(interaction, "usage", None) if interaction is not None else None
        if usage is None:
            return None

        total_input_tokens = int(getattr(usage, "total_input_tokens", 0) or 0)
        cached_tokens = int(getattr(usage, "total_cached_tokens", 0) or 0)
        return TokenUsage(
            input_tokens=max(total_input_tokens - cached_tokens, 0),
            output_tokens=int(getattr(usage, "total_output_tokens", 0) or 0),
            cached_read_tokens=cached_tokens,
            reasoning_tokens=int(getattr(usage, "total_thought_tokens", 0) or 0),
            tool_use_tokens=int(getattr(usage, "total_tool_use_tokens", 0) or 0),
            total_tokens=getattr(usage, "total_tokens", None),
        )

    def build_tool_call_message(
        self, text: str, tool_calls: list[ToolCall]
    ) -> list[dict[str, Any]]:
        """Build Gemini continuation messages for streamed tool turns."""
        content: list[dict[str, Any]] = []

        thought_indices = sorted(
            set(self._reasoning_buffers.keys()) | set(self._reasoning_signatures.keys())
        )
        for index in thought_indices:
            summary = self._reasoning_buffers.get(index, "")
            signature = self._reasoning_signatures.get(index)
            if not summary and not signature:
                continue
            thought_payload: dict[str, Any] = {"type": "thought"}
            if summary:
                thought_payload["summary"] = [{"type": "text", "text": summary}]
            if signature:
                thought_payload["signature"] = signature
            content.append(thought_payload)

        for tool_call in tool_calls:
            arguments = (
                tool_call.arguments
                if isinstance(tool_call.arguments, dict)
                else self.normalize_arguments(tool_call.arguments)
            )
            if not isinstance(arguments, dict):
                arguments = {}
            content.append(
                {
                    "type": "function_call",
                    "id": tool_call.call_id,
                    "name": tool_call.name,
                    "arguments": arguments,
                }
            )

        if text:
            content.append({"type": "text", "text": text})

        if not content:
            return []
        return [{"role": "model", "content": content}]

    def build_tool_output_message(
        self, call_id: str, tool_name: str, tool_output: str
    ) -> list[dict[str, Any]]:
        """Build Gemini tool result message."""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "function_result",
                        "call_id": call_id,
                        "name": tool_name,
                        "result": tool_output,
                    }
                ],
            }
        ]
