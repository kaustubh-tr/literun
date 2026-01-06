from typing import Any, Dict, Optional
from .constants import Role, ContentType


class PromptMessage:
    """
    Domain representation of a single semantic message in the conversation.
    
    This class is the ONLY place that knows how to convert a semantic
    message into an OpenAI-compatible message dictionary.
    """
    
    def __init__(
        self,
        *,
        role: Optional[Role] = None,
        content_type: ContentType,
        text: Optional[str] = None,
        name: Optional[str] = None,
        arguments: Optional[str] = None,
        call_id: Optional[str] = None,
        output: Optional[str] = None,
    ) -> None:
        """
        Initialize a PromptMessage.
        Args:
            role (Optional[Role]): The role of the message sender (e.g., USER, ASSISTANT).
            content_type (ContentType): The type of content (e.g., INPUT_TEXT, FUNCTION_CALL).
            text (Optional[str]): The text content of the message.
            name (Optional[str]): The name of the tool (for function calls).
            arguments (Optional[str]): The arguments for the tool (for function calls).
            call_id (Optional[str]): The ID of the tool call.
            output (Optional[str]): The output of the tool execution.
        """
        self.role = role
        self.content_type = content_type
        self.text = text
        self.name = name
        self.arguments = arguments
        self.call_id = call_id
        self.output = output
        
        self._validate()
        
    def _validate(self) -> None:
        """
        Enforce invariants so invalid messages are never constructed.
        """
        # Text messages (system / user / assistant)
        if self.content_type in (ContentType.INPUT_TEXT, ContentType.OUTPUT_TEXT):
            if self.role is None:
                raise ValueError("role is required for text messages")
            if not isinstance(self.text, str):
                raise ValueError("text is required for text messages")

        # Tool call (model -> agent)
        elif self.content_type == ContentType.FUNCTION_CALL:
            if not self.name:
                raise ValueError("name is required for FUNCTION_CALL")
            if not isinstance(self.arguments, str):
                raise ValueError("arguments must be a JSON string")
            if not self.call_id:
                raise ValueError("call_id is required for FUNCTION_CALL")

        # Tool output (agent -> model)
        elif self.content_type == ContentType.FUNCTION_CALL_OUTPUT:
            if not self.call_id:
                raise ValueError("call_id is required for FUNCTION_CALL_OUTPUT")
            if not isinstance(self.output, str):
                raise ValueError("output must be a string")

        else:
            raise ValueError(f"Unsupported content_type: {self.content_type}")
        
    def to_openai_message(self) -> Dict[str, Any]:
        """
        Convert the message to an OpenAI message dictionary.
        Returns:
            Dict[str, Any]: The formatted message.
        Raises:
            ValueError: If required fields are missing for the specified content_type.
        """
        # System / User / Assistant messages
        if self.content_type in (ContentType.INPUT_TEXT, ContentType.OUTPUT_TEXT):
            return {
                "role": self.role.value,
                "content": [
                    {
                        "type": self.content_type.value,
                        "text": self.text,
                    }
                ],
            }

        # Tool call (model -> agent)
        if self.content_type == ContentType.FUNCTION_CALL:
            return {
                "type": self.content_type.value,
                "name": self.name,
                "arguments": self.arguments,
                "call_id": self.call_id,
            }

        # Tool output (agent -> model)
        if self.content_type == ContentType.FUNCTION_CALL_OUTPUT:
            return {
                "type": self.content_type.value,
                "call_id": self.call_id,
                "output": self.output,
            }

        # Should never reach here due to validation
        raise RuntimeError("Invalid PromptMessage state")

    def __repr__(self) -> str:
        return (
            f"PromptMessage("
            f"content_type={self.content_type}, "
            f"role={self.role}, "
            f"name={self.name}, "
            f"call_id={self.call_id}"
            f")"
        )