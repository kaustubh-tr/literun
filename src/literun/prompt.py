"""Canonical prompt container for conversation messages."""

from __future__ import annotations

from typing import Iterable, Any
import json

from pydantic import BaseModel, ConfigDict, PrivateAttr

from .message import (
    PromptMessage,
    TextBlock,
    ToolCallBlock,
    ToolOutputBlock,
    ReasoningBlock,
)
from .errors import AgentInputError, AgentSerializationError


class PromptTemplate(BaseModel):
    """Container for canonical conversation messages."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _messages: list[PromptMessage] = PrivateAttr(default_factory=list)

    @property
    def messages(self) -> list[PromptMessage]:
        """Return stored messages."""
        return self._messages

    def add_message(self, message: PromptMessage) -> PromptTemplate:
        """Append one canonical message."""
        if not isinstance(message, PromptMessage):
            raise AgentInputError("Expected PromptMessage")
        self._messages.append(message)
        return self

    def add_messages(self, messages: Iterable[PromptMessage]) -> PromptTemplate:
        """Append multiple canonical messages."""
        for message in messages:
            self.add_message(message)
        return self

    def add_system(self, text: str) -> PromptTemplate:
        """Append a system text message."""
        return self.add_message(
            PromptMessage(role="system", content=[TextBlock(text=text)])
        )

    def add_user(self, text: str) -> PromptTemplate:
        """Append a user text message."""
        return self.add_message(
            PromptMessage(role="user", content=[TextBlock(text=text)])
        )

    def add_assistant(self, text: str) -> PromptTemplate:
        """Append an assistant text message."""
        return self.add_message(
            PromptMessage(role="assistant", content=[TextBlock(text=text)])
        )

    def add_tool_call(
        self,
        *,
        name: str,
        arguments: dict[str, Any] | str,
        call_id: str,
    ) -> PromptTemplate:
        """Append an assistant tool-call message."""
        parsed_arguments: dict[str, Any]
        if isinstance(arguments, dict):
            parsed_arguments = dict(arguments)
        elif isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise AgentSerializationError(
                    "arguments JSON must be valid and decode to an object"
                ) from exc
            if not isinstance(parsed, dict):
                raise AgentSerializationError(
                    "arguments JSON must decode to an object"
                )
            parsed_arguments = parsed
        else:
            raise AgentInputError("arguments must be dict or JSON string")

        return self.add_message(
            PromptMessage(
                role="assistant",
                content=[
                    ToolCallBlock(
                        name=name,
                        arguments=parsed_arguments,
                        call_id=call_id,
                    )
                ],
            )
        )

    def add_tool_output(
        self,
        *,
        call_id: str,
        output: str | dict[str, object],
        name: str | None = None,
        is_error: bool | None = None,
    ) -> PromptTemplate:
        """Append a user tool-output message."""
        return self.add_message(
            PromptMessage(
                role="user",
                content=[
                    ToolOutputBlock(
                        call_id=call_id,
                        name=name,
                        output=output,
                        is_error=is_error,
                    )
                ],
            )
        )

    def add_reasoning(
        self,
        *,
        summary: str | None = None,
        signature: str | None = None,
        reasoning_id: str | None = None,
        provider_meta: dict[str, object] | None = None,
    ) -> PromptTemplate:
        """Append an assistant reasoning message."""
        return self.add_message(
            PromptMessage(
                role="assistant",
                content=[
                    ReasoningBlock(
                        summary=summary,
                        signature=signature,
                        reasoning_id=reasoning_id,
                        provider_meta=provider_meta,
                    )
                ],
            )
        )

    def copy(self) -> PromptTemplate:
        """Create a shallow copy of this prompt template."""
        new = PromptTemplate()
        new._messages = list(self._messages)
        return new

    def to_messages(self) -> list[PromptMessage]:
        """Return a shallow copy of canonical messages."""
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)
