"""Canonical message and content-block models for conversation state."""

from __future__ import annotations

from typing import Any, Annotated, Literal, TypeAlias
from pydantic import BaseModel, ConfigDict, Field, model_validator


MessageRole: TypeAlias = Literal["system", "user", "assistant"]


class TextBlock(BaseModel):
    """Plain text content."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str


class ToolCallBlock(BaseModel):
    """Assistant-emitted tool call block."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    arguments: dict[str, Any]


class ToolOutputBlock(BaseModel):
    """User-emitted tool output block."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_output"] = "tool_output"
    call_id: str
    name: str | None = None
    output: str | dict[str, Any]
    is_error: bool | None = None


class ReasoningBlock(BaseModel):
    """Provider-agnostic reasoning metadata block."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["reasoning"] = "reasoning"
    summary: str | None = None
    signature: str | None = None
    reasoning_id: str | None = None
    provider_meta: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_reasoning_payload(self) -> ReasoningBlock:
        if (
            self.summary is None
            and self.signature is None
            and self.reasoning_id is None
            and self.provider_meta is None
        ):
            raise ValueError(
                "reasoning block requires at least one of summary/signature/reasoning_id/provider_meta"
            )
        return self


MessageContentBlock: TypeAlias = Annotated[
    TextBlock | ToolCallBlock | ToolOutputBlock | ReasoningBlock,
    Field(discriminator="type"),
]


class PromptMessage(BaseModel):
    """A canonical conversation message with strict role/block invariants."""

    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: list[MessageContentBlock]

    @model_validator(mode="after")
    def _validate_message_invariants(self) -> PromptMessage:
        if not self.content:
            raise ValueError("message content cannot be empty")

        allowed_by_role: dict[str, set[str]] = {
            "system": {"text"},
            "assistant": {"text", "tool_call", "reasoning"},
            "user": {"text", "tool_output"},
        }

        allowed = allowed_by_role[self.role]
        for block in self.content:
            if block.type not in allowed:
                raise ValueError(
                    f"block type '{block.type}' is not valid for role '{self.role}'"
                )

        return self

