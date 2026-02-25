from __future__ import annotations

from typing import Any, AsyncIterator

from pydantic import Field

from literun.constants import ToolCall
from literun.events import StreamEvent
from literun.items import RunItem
from literun.llm import BaseLLM
from literun.prompt import PromptTemplate
from literun.providers.base import ResponseAdapter, StreamAdapter
from literun.usage import TokenUsage


class FakeResponseAdapter(ResponseAdapter):
    """Minimal deterministic adapter for Runner/Agent unit tests."""

    def extract_text(self, response: dict[str, Any]) -> str:
        return str(response.get("text", ""))

    def extract_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        return list(response.get("tool_calls", []))

    def extract_token_usage(self, response: dict[str, Any]) -> TokenUsage | None:
        return response.get("usage")

    def build_tool_call_message(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        return list(response.get("tool_call_messages", []))

    def build_tool_output_message(
        self, call_id: str, tool_name: str, tool_output: str
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "name": tool_name,
                        "content": tool_output,
                    }
                ],
            }
        ]

    def to_run_items(self, response: dict[str, Any]) -> list[RunItem]:
        return list(response.get("items", []))


class FakeStreamAdapter(StreamAdapter):
    """Passthrough stream adapter for feeding pre-built StreamEvent objects."""

    supports_streaming: bool = True

    def process_stream(self, stream: Any):
        for event in stream:
            yield event

    async def aprocess_stream(self, stream: Any) -> AsyncIterator[StreamEvent]:
        async for event in stream:
            yield event

    def extract_token_usage(self, response: Any) -> TokenUsage | None:
        return None

    def build_tool_call_message(
        self,
        text: str,
        tool_calls: list[ToolCall],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if text:
            messages.append({"role": "assistant", "content": text})
        for tc in tool_calls:
            messages.append(
                {
                    "type": "function_call",
                    "call_id": tc.call_id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
            )
        return messages

    def build_tool_output_message(
        self, call_id: str, tool_name: str, tool_output: str
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": tool_output,
            }
        ]


class _AsyncListStream:
    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = events

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self._gen()

    async def _gen(self) -> AsyncIterator[StreamEvent]:
        for event in self._events:
            yield event


class FakeLLM(BaseLLM):
    """Test LLM that returns scripted responses/streams."""

    scripted_responses: list[dict[str, Any]] = Field(default_factory=list)
    scripted_streams: list[list[StreamEvent]] = Field(default_factory=list)

    def generate(
        self,
        messages: list[dict[str, Any]],
        system_instruction: str | None = None,
        stream: bool = False,
        tools: list[Any] | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> Any:
        if stream:
            if not self.scripted_streams:
                return []
            return self.scripted_streams.pop(0)
        if not self.scripted_responses:
            return {"text": "", "tool_calls": [], "usage": None, "items": []}
        return self.scripted_responses.pop(0)

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        system_instruction: str | None = None,
        stream: bool = False,
        tools: list[Any] | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> Any:
        if stream:
            if not self.scripted_streams:
                return _AsyncListStream([])
            return _AsyncListStream(self.scripted_streams.pop(0))
        if not self.scripted_responses:
            return {"text": "", "tool_calls": [], "usage": None, "items": []}
        return self.scripted_responses.pop(0)

    def normalize_messages(
        self,
        messages: str | list[dict[str, Any]] | PromptTemplate,
    ) -> list[dict[str, Any]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, list):
            return list(messages)
        if isinstance(messages, PromptTemplate):
            normalized: list[dict[str, Any]] = []
            for msg in messages.to_messages():
                normalized.append(
                    {
                        "role": msg.role,
                        "content": [block.model_dump() for block in msg.content],
                    }
                )
            return normalized
        raise TypeError(f"Unsupported messages type: {type(messages).__name__}")

    def get_response_adapter(self) -> ResponseAdapter:
        return FakeResponseAdapter()

    def get_stream_adapter(self) -> StreamAdapter:
        return FakeStreamAdapter()

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None
