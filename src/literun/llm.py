"""LLM client wrapper and configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict

from .tool import Tool
from .constants import DEFAULT_TIMEOUT

if TYPE_CHECKING:
    from .prompt import PromptTemplate
    from .providers.base import ResponseAdapter, StreamAdapter


class BaseLLM(ABC, BaseModel):
    """Base class for all LLM providers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str
    """Model name or identifier to use for generation."""
    temperature: float | None = None
    """Sampling temperature for generation. Higher values mean more randomness."""
    max_output_tokens: int | None = None
    """Maximum number of tokens to generate in the output."""
    timeout: float | None = DEFAULT_TIMEOUT
    """Timeout for LLM requests in seconds."""
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Additional keyword arguments to pass to the LLM model."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        system_instruction: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> Any:
        """Generate a response from the LLM."""
        ...

    @abstractmethod
    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        system_instruction: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> Any:
        """Generate a response from the LLM asynchronously."""
        ...

    @abstractmethod
    def normalize_messages(
        self,
        messages: str | list[dict[str, Any]] | PromptTemplate,
    ) -> list[dict[str, Any]]:
        """Normalize messages for provider usage."""
        ...

    @abstractmethod
    def get_response_adapter(self) -> ResponseAdapter:
        """Get the appropriate response adapter for this provider.

        Returns:
            ResponseAdapter instance for parsing non-streaming responses
        """
        ...

    @abstractmethod
    def get_stream_adapter(self) -> StreamAdapter:
        """Get the appropriate stream adapter for this provider.

        Returns:
            StreamAdapter instance for parsing streaming responses
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the client connection."""
        ...

    @abstractmethod
    async def aclose(self) -> None:
        """Close the client connection asynchronously."""
        ...
