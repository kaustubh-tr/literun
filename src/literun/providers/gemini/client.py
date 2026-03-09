"""Gemini provider for literun."""

from __future__ import annotations

from typing import Any, Literal
import os
import json
from pydantic import Field, ConfigDict, model_validator

from ...llm import BaseLLM
from ...tool import Tool
from ...constants import DEFAULT_MAX_RETRIES
from ...errors import (
    AgentInputError,
    AgentSerializationError,
    LLMError,
    APIConnectionError as LiteAPIConnectionError,
    APIStatusError as LiteAPIStatusError,
    InvalidRequestError as LiteInvalidRequestError,
    AuthenticationError as LiteAuthenticationError,
    RateLimitError as LiteRateLimitError,
)
from ...logger import AgentLogger
from ...message import PromptMessage
from ...prompt import PromptTemplate
from .responses import GeminiResponseAdapter
from .streams import GeminiStreamAdapter

_LOGGER = AgentLogger(name="literun.providers.gemini")
_DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"


class ChatGemini(BaseLLM):
    """Gemini provider using Interactions API (beta)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    model: str = _DEFAULT_GEMINI_MODEL
    """Gemini Interactions model id (for example: ``gemini-3-flash-preview``)."""

    api_key: str | None = None
    """Google API key; if omitted, reads ``GOOGLE_API_KEY`` from environment."""

    project: str | None = None
    """Optional Google Cloud project id passed to the SDK client."""

    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    """Thinking level mapped to ``generation_config.thinking_level``."""

    reasoning_summary: Literal["auto", "none"] | None = None
    """Thinking summary mode mapped to ``generation_config.thinking_summaries``."""

    response_format: object | None = None
    """Optional structured response format object passed to API."""

    response_mime_type: str | None = None
    """Optional MIME type for response content (for example, JSON MIME types)."""

    tool_choice: Literal["auto", "any", "none", "validated"] = "auto"
    """Default tool policy mapped to ``generation_config.tool_choice``."""

    max_retries: int = DEFAULT_MAX_RETRIES
    """Configured retry count for SDK-level transient failures."""

    store: bool = False
    """Whether to set ``store`` in Interactions API requests."""

    client: Any = Field(default=None, exclude=True)
    """Internal sync Gemini SDK client instance (not serialized)."""

    aclient: Any = Field(default=None, exclude=True)
    """Internal async Gemini SDK client instance (not serialized)."""

    @property
    def provider(self) -> str:
        """Gemini provider name."""
        return "gemini"

    @model_validator(mode="after")
    def _init_client(self) -> Any:
        """Initialize the Gemini clients."""
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "Google GenAI is not installed. Please install it using `pip install literun[gemini]`"
            )

        self.api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise LiteAuthenticationError(
                "Missing API key for Gemini provider.",
                context={
                    "provider": "gemini",
                    "hint": "Set the 'GOOGLE_API_KEY' environment variable or pass 'api_key' explicitly."
                },
            )
        client_kwargs = {
            "api_key": self.api_key,
            "project": self.project,
        }
        if not self.client:
            self.client = genai.Client(**client_kwargs)
        if not self.aclient:
            self.aclient = genai.Client(**client_kwargs).aio
        return self

    def get_response_adapter(self) -> GeminiResponseAdapter:
        """Get the Gemini response adapter."""
        return GeminiResponseAdapter()

    def get_stream_adapter(self) -> GeminiStreamAdapter:
        """Get the Gemini stream adapter."""
        return GeminiStreamAdapter()

    def normalize_messages(
        self,
        messages: str | list[dict[str, Any]] | PromptTemplate,
    ) -> list[dict[str, Any]]:
        """Normalize messages into Gemini Interactions format."""
        if isinstance(messages, PromptTemplate):
            return self._serialize_prompt(messages.to_messages())

        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        if isinstance(messages, list):
            return list(messages)

        raise AgentInputError(
            "Invalid messages input. Provide `str`, provider-native `list[dict]` "
            "(Gemini Interactions schema), or `PromptTemplate`.",
            context={
                "provider": "gemini",
                "received_type": type(messages).__name__,
                "expected": "str | list[dict] | PromptTemplate",
                "hint": "See Gemini Interactions input schema or use PromptTemplate.",
            },
        )

    def _serialize_prompt(
        self,
        messages: list[PromptMessage],
    ) -> list[dict[str, Any]]:
        """Serialize canonical messages into Gemini Interactions turns."""
        serialized_turns: list[dict[str, Any]] = []

        for message in messages:
            if message.role == "system":
                raise AgentSerializationError(
                    "Gemini PromptTemplate system messages are not supported. "
                    "Use Agent(system_instruction=...) or provider-native messages.",
                    context={
                        "provider": "gemini",
                        "role": message.role,
                        "hint": "Gemini system instruction is a separate top-level field.",
                    },
                )

            if message.role not in {"user", "assistant"}:
                raise AgentSerializationError(
                    f"Unsupported role '{message.role}' for Gemini messages",
                    context={
                        "provider": "gemini",
                        "role": message.role,
                        "hint": "Gemini messages support user/assistant (mapped to user/model).",
                    },
                )

            gemini_role = "model" if message.role == "assistant" else "user"
            content: list[dict[str, Any]] = []
            for block in message.content:
                if block.type == "text":
                    content.append({"type": "text", "text": block.text})
                    continue

                if block.type == "tool_call":
                    content.append(
                        {
                            "type": "function_call",
                            "id": block.call_id,
                            "name": block.name,
                            "arguments": block.arguments,
                        }
                    )
                    continue

                if block.type == "tool_output":
                    content.append(
                        {
                            "type": "function_result",
                            "call_id": block.call_id,
                            "name": block.name,
                            "result": block.output,
                        }
                    )
                    continue

                if block.type == "reasoning":
                    thought: dict[str, Any] = {"type": "thought"}
                    if block.summary is not None:
                        thought["summary"] = [{"type": "text", "text": block.summary}]
                    if block.signature is not None:
                        thought["signature"] = block.signature
                    if "signature" not in thought:
                        raise AgentSerializationError(
                            "Gemini thought replay requires signature",
                            context={
                                "provider": "gemini",
                                "role": message.role,
                                "block_type": block.type,
                                "hint": "Include signature in ReasoningBlock for Gemini thought replay.",
                            },
                        )
                    content.append(thought)
                    continue

                raise AgentSerializationError(
                    f"Unsupported canonical block type '{block.type}' for Gemini",
                    context={
                        "provider": "gemini",
                        "role": message.role,
                        "block_type": block.type,
                        "hint": "Use text/tool_call/tool_output/reasoning blocks supported by Gemini serializer.",
                    },
                )

            if content:
                serialized_turns.append({"role": gemini_role, "content": content})

        return serialized_turns

    def _prepare_request_params(
        self,
        messages: list[dict[str, Any]],
        system_instruction: str | None,
        stream: bool,
        tools: list[Tool] | None,
        tool_choice: str | None,
        parallel_tool_calls: bool | None,  # Not supported by Gemini
    ) -> dict[str, Any]:
        """Prepare request parameters for Gemini Interactions API."""
        if parallel_tool_calls is False:
            import warnings
            warnings.warn(
                "Gemini Interactions API currently does not support disabling parallel tool calls. "
                "The model may still return multiple tool calls.",
                UserWarning,
                stacklevel=2
            )
        params = {
            "model": self.model,
            "input": messages,
            "system_instruction": system_instruction,
            "stream": stream,
            "store": self.store,
            "timeout": self.timeout,
            **self.model_kwargs,
        }
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        if tools:
            params["tools"] = [tool.to_gemini_tool() for tool in tools]
            generation_config["tool_choice"] = tool_choice or self.tool_choice
        if self.reasoning_effort is not None:
            generation_config["thinking_level"] = self.reasoning_effort
        if self.reasoning_summary is not None:
            generation_config["thinking_summaries"] = self.reasoning_summary
        if self.response_format is not None:
            params["response_format"] = self.response_format
        if self.response_mime_type is not None:
            params["response_mime_type"] = self.response_mime_type
        params["generation_config"] = generation_config
        return params

    def _map_provider_exception(self, exc: Exception) -> LLMError:
        """Map Gemini SDK exceptions to literun LLMError types."""
        context = {"provider": "gemini", "model": self.model}
        error_message = str(exc)
        from google.genai import _interactions as interactions
        from google.genai import errors as genai_errors

        if isinstance(exc, (interactions.AuthenticationError, interactions.PermissionDeniedError)):
            return LiteAuthenticationError(error_message, context=context, cause=exc)
        if isinstance(exc, interactions.RateLimitError):
            return LiteRateLimitError(error_message, context=context, cause=exc)
        if isinstance(exc, (interactions.BadRequestError, interactions.UnprocessableEntityError)):
            return LiteInvalidRequestError(error_message, context=context, cause=exc)
        if isinstance(exc, (interactions.APIConnectionError, interactions.APITimeoutError)):
            return LiteAPIConnectionError(error_message, context=context, cause=exc)
        if isinstance(exc, interactions.APIStatusError):
            return LiteAPIStatusError(error_message, context=context, cause=exc)

        if genai_errors is not None:
            if isinstance(exc, genai_errors.ClientError):
                return LiteInvalidRequestError(error_message, context=context, cause=exc)
            if isinstance(exc, genai_errors.ServerError):
                return LiteAPIStatusError(error_message, context=context, cause=exc)
            if isinstance(exc, genai_errors.APIError):
                return LLMError(error_message, context=context, cause=exc)

        return LLMError(error_message, context=context, cause=exc)

    def generate(
        self,
        messages: list[dict[str, Any]],
        system_instruction: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,  # Not supported by Gemini
    ) -> Any:
        """Generate a response from Gemini."""
        params = self._prepare_request_params(
            messages=messages,
            system_instruction=system_instruction,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        from google.genai._interactions import APIError
        try:
            return self.client.interactions.create(**params)
        except APIError as exc:
            mapped = self._map_provider_exception(exc)
            _LOGGER.log_exception(mapped, context={"method": "generate"})
            raise mapped from exc

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        system_instruction: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,  # Not supported by Gemini
    ) -> Any:
        """Generate a response from Gemini asynchronously."""
        params = self._prepare_request_params(
            messages=messages,
            system_instruction=system_instruction,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        from google.genai._interactions import APIError
        try:
            return await self.aclient.interactions.create(**params)
        except APIError as exc:
            mapped = self._map_provider_exception(exc)
            _LOGGER.log_exception(mapped, context={"method": "agenerate"})
            raise mapped from exc

    def close(self) -> None:
        """Close the Gemini client connection."""
        self.client.close()

    async def aclose(self) -> None:
        """Close the async Gemini client connection."""
        await self.aclient.aclose()

    def __enter__(self) -> ChatGemini:
        """Enter context and return self."""
        return self

    def __exit__(self, *exc) -> None:
        """Exit context and close client connection."""
        self.close()

    async def __aenter__(self) -> ChatGemini:
        """Enter async context and return self."""
        return self

    async def __aexit__(self, *exc) -> None:
        """Exit async context and close async client connection."""
        await self.aclose()
