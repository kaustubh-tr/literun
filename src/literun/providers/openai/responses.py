"""OpenAI response adapter for normalizing responses to RunItems."""

from __future__ import annotations

from typing import Any
import json

from ...items import (
    RunItem,
    MessageOutputItem,
    ToolCallItem,
    ReasoningItem,
)
from ...constants import ToolCall
from ...usage import TokenUsage
from ..base import ResponseAdapter, AdapterMixin


class OpenAIResponseAdapter(ResponseAdapter, AdapterMixin):
    """Adapter for OpenAI Responses API objects."""

    def _normalize_reasoning_summary(self, summary: Any) -> str | None:
        if isinstance(summary, str):
            return summary

        if isinstance(summary, list):
            parts: list[str] = []
            for entry in summary:
                if isinstance(entry, str):
                    parts.append(entry)
                elif isinstance(entry, dict) and isinstance(entry.get("text"), str):
                    parts.append(entry["text"])
                else:
                    text = getattr(entry, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
            combined = "".join(parts).strip()
            return combined or None

        return None

    def extract_text(self, response) -> str:
        """Extract text from OpenAI response.

        OpenAI Responses API: response.output contains ResponseOutputMessage
        Types: 'output_text' (has .text) or 'message' (has .content as list)
        """
        texts: list[str] = []
        if not response.output:
            return ""

        for output in response.output:
            if output.type == "message":
                for content in output.content:
                    if content.type == "output_text":
                        texts.append(content.text)

        return "".join(texts)

    def extract_tool_calls(self, response) -> list[ToolCall]:
        """Extract tool calls from OpenAI response.

        OpenAI Responses API structure: response.output contains ResponseFunctionToolCall objects directly
        """
        tool_calls: list[ToolCall] = []
        if not response.output:
            return tool_calls

        for output in response.output:
            if hasattr(output, "type") and output.type == "function_call":
                arguments = self.normalize_arguments(getattr(output, "arguments", {}))
                tool_calls.append(
                    ToolCall(
                        call_id=output.call_id,
                        name=output.name,
                        arguments=arguments,
                    )
                )

        return tool_calls

    def extract_token_usage(self, response) -> TokenUsage:
        """Extract token usage from OpenAI response."""
        usage = getattr(response, "usage", None)
        if usage:
            cached_tokens = (
                getattr(usage.input_tokens_details, "cached_tokens", 0)
                if hasattr(usage, "input_tokens_details") and usage.input_tokens_details
                else 0
            )
            reasoning_tokens = (
                getattr(usage.output_tokens_details, "reasoning_tokens", 0)
                if hasattr(usage, "output_tokens_details")
                and usage.output_tokens_details
                else 0
            )
            return TokenUsage(
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                cached_read_tokens=int(cached_tokens or 0),
                reasoning_tokens=int(reasoning_tokens or 0),
                total_tokens=getattr(usage, "total_tokens", None),
            )
        return None

    def build_tool_call_message(self, response: Any) -> list[dict[str, Any]]:
        """Build OpenAI continuation messages from response."""
        messages: list[dict[str, Any]] = []
        outputs = getattr(response, "output", None)
        if not outputs:
            return messages

        for item in outputs:
            if item.type == "message":
                text_parts = [c.text for c in item.content if c.type == "output_text"]
                text = "".join(text_parts)
                if text:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        }
                    )
                continue

            if item.type == "function_call":
                arguments = self.normalize_arguments(getattr(item, "arguments", {}))
                messages.append(
                    {
                        "type": "function_call",
                        "call_id": getattr(item, "call_id", ""),
                        "name": getattr(item, "name", ""),
                        "arguments": (
                            json.dumps(arguments)
                            if isinstance(arguments, dict)
                            else arguments
                        ),
                    }
                )
                continue

            if item.type == "reasoning":
                summary = self._normalize_reasoning_summary(getattr(item, "summary", None))
                payload: dict[str, Any] = {"type": "reasoning"}
                if summary is not None:
                    payload["summary"] = [{"type": "summary_text", "text": summary}]
                signature = getattr(item, "encrypted_content", None)
                if signature is not None:
                    payload["encrypted_content"] = signature
                messages.append(payload)

        return messages

    def build_tool_output_message(
        self, call_id: str, tool_name: str, tool_output: str
    ) -> list[dict[str, Any]]:
        """Build OpenAI tool result messages."""
        return [
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": tool_output,
            }
        ]

    def to_run_items(self, response) -> list[RunItem]:
        """Convert OpenAI response to RunItems."""
        items: list[RunItem] = []
        if not hasattr(response, "output") or not response.output:
            return items

        for item in response.output:
            if item.type == "message":
                text_parts = [c.text for c in item.content if c.type == "output_text"]
                final_output_text = "".join(text_parts)
                items.append(
                    MessageOutputItem(
                        id=getattr(item, "id", None),
                        content=final_output_text,
                        raw_item=item,
                    )
                )
            elif item.type == "function_call":
                arguments = self.normalize_arguments(getattr(item, "arguments", {}))
                items.append(
                    ToolCallItem(
                        id=getattr(item, "id", None),
                        call_id=getattr(item, "call_id", None),
                        name=getattr(item, "name", None),
                        arguments=arguments if isinstance(arguments, dict) else None,
                        raw_item=item,
                    )
                )
            elif item.type == "reasoning":
                items.append(
                    ReasoningItem(
                        id=getattr(item, "id", None),
                        signature=getattr(item, "encrypted_content", None),
                        summary=getattr(item, "summary", None),
                        raw_item=item,
                    )
                )
        return items
