"""Message structures for prompts."""

from __future__ import annotations

from typing import Any, Dict
from pydantic import BaseModel, model_validator

from .constants import Role, ContentType


class PromptMessage(BaseModel):
    """Domain representation of a single semantic message in a conversation.

    This class is the only place that knows how to convert a semantic
    message into an OpenAI-compatible message dictionary. It enforces
    invariants depending on the message type.

    Args:
        role: The role of the message sender. Required for text messages.
            Options: "system", "user", "assistant", "developer", "tool"
        text: The text content of the message (required for text messages).
        name: The name of the tool (for function calls).
        arguments: The arguments for the tool as a JSON string (for function calls).
        call_id: The ID of the tool call.
        output: The output of the tool execution (for function call output messages).
        content_type: The type of content.
            Options: "input_text", "output_text", "message", "function_call", "function_call_output"
    """

    role: Role | None = None
    text: str | None = None
    name: str | None = None
    arguments: str | None = None
    call_id: str | None = None
    output: str | None = None
    content_type: ContentType

    @model_validator(mode="after")
    def _validate_invariants(self) -> PromptMessage:
        """Enforce invariants so that invalid messages are never constructed.

        Raises:
            ValueError: If required fields are missing for the given content_type.
        """
        # Text messages (system / user / assistant)
        if self.content_type in ("input_text", "output_text"):
            if self.role is None:
                raise ValueError("role is required for text messages")
            if not isinstance(self.text, str):
                raise ValueError("text is required for text messages")

        # Tool call (model -> agent)
        elif self.content_type == "function_call":
            if not self.name:
                raise ValueError("name is required for FUNCTION_CALL")
            if not isinstance(self.arguments, str):
                raise ValueError("arguments must be a JSON string")
            if not self.call_id:
                raise ValueError("call_id is required for FUNCTION_CALL")

        # Tool output (agent -> model)
        elif self.content_type == "function_call_output":
            if not self.call_id:
                raise ValueError("call_id is required for FUNCTION_CALL_OUTPUT")
            if not isinstance(self.output, str):
                raise ValueError("output must be a string")
        else:
            raise ValueError(f"Unsupported content_type: {self.content_type}")
        
        return self

    def to_openai_message(self) -> Dict[str, Any]:
        """Convert the PromptMessage to an OpenAI-compatible message dictionary.

        Returns:
            Dict[str, Any]: The formatted message dictionary.

        Raises:
            ValueError: If required fields are missing for the specified content_type.
            RuntimeError: If the message state is invalid (should not occur).
        """
        # System / User / Assistant messages
        if self.content_type in ("input_text", "output_text"):
            return {
                "role": self.role,
                "content": [
                    {
                        "type": self.content_type,
                        "text": self.text,
                    }
                ],
            }

        # Tool call (model -> agent)
        if self.content_type == "function_call":
            return {
                "type": self.content_type,
                "name": self.name,
                "arguments": self.arguments,
                "call_id": self.call_id,
            }

        # Tool output (agent -> model)
        if self.content_type == "function_call_output":
            return {
                "type": self.content_type,
                "call_id": self.call_id,
                "output": self.output,
            }

        # Should never reach here due to validation
        raise RuntimeError("Invalid PromptMessage state")

    def __repr__(self) -> str:
        """Return a concise representation of the message for debugging."""
        return (
            f"PromptMessage("
            f"content_type={self.content_type}, "
            f"role={self.role}, "
            f"name={self.name}, "
            f"call_id={self.call_id}"
            f")"
        )
