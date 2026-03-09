"""Gemini response adapter for normalizing responses to RunItems."""

from __future__ import annotations

from typing import Any

from ...items import (
    RunItem,
    MessageOutputItem,
    ToolCallItem,
    ReasoningItem,
)
from ...constants import ToolCall
from ...usage import TokenUsage
from ..base import ResponseAdapter, AdapterMixin


class GeminiResponseAdapter(ResponseAdapter, AdapterMixin):
    """Adapter for Gemini Interaction API objects."""

    def _normalize_thought_summary(self, raw_summary: Any) -> str | None:
        if isinstance(raw_summary, str):
            return raw_summary

        if isinstance(raw_summary, list):
            parts: list[str] = []
            for item in raw_summary:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                else:
                    text = getattr(item, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
            joined = "".join(parts).strip()
            return joined or None

        return None

    def extract_text(self, response) -> str:
        """Extract text from Gemini response.

        Gemini Interactions API structure: response.outputs (list of content items)
        """
        if not hasattr(response, "outputs") or not response.outputs:
            return ""

        text_parts: list[str] = []
        for output in response.outputs:
            if hasattr(output, "type") and output.type == "text":
                text_parts.append(getattr(output, "text", "") or "")
        return "".join(text_parts)

    def extract_tool_calls(self, response) -> list[ToolCall]:
        """Extract tool calls from Gemini response.

        Gemini Interactions API structure: response.outputs (filter type='function_call')
        """
        tool_calls: list[ToolCall] = []
        if not hasattr(response, "outputs") or not response.outputs:
            return tool_calls

        for output in response.outputs:
            if hasattr(output, "type") and output.type == "function_call":
                tool_calls.append(
                    ToolCall(
                        call_id=output.id,
                        name=output.name,
                        arguments=self.normalize_arguments(getattr(output, "arguments", {})),
                    )
                )
        return tool_calls

    def extract_token_usage(self, response) -> TokenUsage | None:
        """Extract token usage from Gemini response."""
        usage = getattr(response, "usage", None)
        if usage:
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
        return None

    def build_tool_call_message(self, response: Any) -> list[dict[str, Any]]:
        """Build Gemini continuation messages from response."""
        outputs = getattr(response, "outputs", None)
        if not outputs:
            return []

        content: list[dict[str, Any]] = []
        for item in outputs:
            if item.type == "text":
                text = getattr(item, "text", "") or ""
                if text:
                    content.append({"type": "text", "text": text})
                continue

            if item.type == "function_call":
                arguments = self.normalize_arguments(getattr(item, "arguments", {}))
                content.append(
                    {
                        "type": "function_call",
                        "id": getattr(item, "id", ""),
                        "name": getattr(item, "name", ""),
                        "arguments": arguments if isinstance(arguments, dict) else {},
                    }
                )
                continue

            if item.type == "thought":
                thought_payload: dict[str, Any] = {"type": "thought"}
                summary = self._normalize_thought_summary(getattr(item, "summary", None))
                if summary is not None:
                    thought_payload["summary"] = [{"type": "text", "text": summary}]
                signature = getattr(item, "signature", None)
                if signature is not None:
                    thought_payload["signature"] = signature
                content.append(thought_payload)

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

    def to_run_items(self, response) -> list[RunItem]:
        """Convert Gemini response to RunItems."""
        items: list[RunItem] = []
        if not hasattr(response, "outputs") or not response.outputs:
            return items

        for item in response.outputs:
            if item.type == "text":
                items.append(
                    MessageOutputItem(
                        id=getattr(item, "id", None),
                        content=item.text,
                        raw_item=item,
                    )
                )
            elif item.type == "function_call":
                arguments = self.normalize_arguments(getattr(item, "arguments", {}))
                items.append(
                    ToolCallItem(
                        id=getattr(item, "id", None),
                        call_id=getattr(item, "id", None),
                        name=getattr(item, "name", None),
                        arguments=arguments if isinstance(arguments, dict) else None,
                        raw_item=item,
                    )
                )
            elif item.type == "thought":
                items.append(
                    ReasoningItem(
                        id=getattr(item, "id", None),
                        signature=getattr(item, "signature", None),
                        summary=self._normalize_thought_summary(getattr(item, "summary", None)),
                        raw_item=item,
                    )
                )
        return items
