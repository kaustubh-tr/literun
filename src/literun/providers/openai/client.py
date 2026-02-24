"""OpenAI provider for literun."""

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
from .responses import OpenAIResponseAdapter
from .streams import OpenAIStreamAdapter

_LOGGER = AgentLogger(name="literun.providers.openai")
_DEFAULT_OPENAI_MODEL = "gpt-5-nano"


class ChatOpenAI(BaseLLM):
    """OpenAI provider using Responses API."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    model: str = _DEFAULT_OPENAI_MODEL
    api_key: str | None = None
    organization: str | None = None
    project: str | None = None
    base_url: str | None = None
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None
    ) = None
    reasoning_summary: Literal["auto", "concise", "detailed"] | None = None
    verbosity: Literal["low", "medium", "high"] | None = None
    text_format: Literal["text", "json_object", "json_schema"] = "text"
    response_format: object | None = None
    tool_choice: Literal["auto", "any", "none", "required"] = "auto"
    max_retries: int = DEFAULT_MAX_RETRIES
    store: bool = False
    client: Any = Field(default=None, exclude=True)
    aclient: Any = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _init_client(self) -> Any:
        """Initialize the OpenAI clients."""
        try:
            from openai import OpenAI, AsyncOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI is not installed. Please install it using `pip install literun[openai]`"
            )

        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LiteAuthenticationError(
                "Missing API key for OpenAI provider.",
                context={
                    "provider": "openai",
                    "hint": "Set the 'OPENAI_API_KEY' environment variable or pass 'api_key' explicitly."
                },
            )
        client_kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "organization": self.organization,
            "project": self.project,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if not self.client:
            self.client = OpenAI(**client_kwargs)
        if not self.aclient:
            self.aclient = AsyncOpenAI(**client_kwargs)
        return self

    def get_response_adapter(self) -> OpenAIResponseAdapter:
        """Get the OpenAI response adapter."""
        return OpenAIResponseAdapter()

    def get_stream_adapter(self) -> OpenAIStreamAdapter:
        """Get the OpenAI stream adapter."""
        return OpenAIStreamAdapter()

    def normalize_messages(
        self,
        messages: str | list[dict[str, Any]] | PromptTemplate,
    ) -> list[dict[str, Any]]:
        if isinstance(messages, PromptTemplate):
            return self._serialize_prompt(messages.to_messages())

        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        if isinstance(messages, list):
            return list(messages)

        raise AgentInputError(
            "Invalid messages input. Provide `str`, provider-native `list[dict]` "
            "(OpenAI Responses schema), or `PromptTemplate`.",
            context={
                "provider": "openai",
                "received_type": type(messages).__name__,
                "expected": "str | list[dict] | PromptTemplate",
                "hint": "See OpenAI Responses API input schema or use PromptTemplate.",
            },
        )

    def _serialize_prompt(self, messages: list[PromptMessage]) -> list[dict[str, Any]]:
        """Serialize canonical messages into OpenAI Responses input items."""
        serialized: list[dict[str, Any]] = []

        for message in messages:
            for block in message.content:
                if block.type == "text":
                    if message.role in {"system", "user"}:
                        serialized.append(
                            {
                                "role": message.role,
                                "content": [{"type": "input_text", "text": block.text}],
                            }
                        )
                    elif message.role == "assistant":
                        serialized.append(
                            {
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": block.text}],
                            }
                        )
                    continue

                if block.type == "tool_call":
                    serialized.append(
                        {
                            "type": "function_call",
                            "call_id": block.call_id,
                            "name": block.name,
                            "arguments": json.dumps(block.arguments),
                        }
                    )
                    continue

                if block.type == "tool_output":
                    output = (
                        block.output
                        if isinstance(block.output, str)
                        else json.dumps(block.output)
                    )
                    serialized.append(
                        {
                            "type": "function_call_output",
                            "call_id": block.call_id,
                            "output": output,
                        }
                    )
                    continue

                if block.type == "reasoning":
                    if not block.reasoning_id or not block.summary:
                        raise AgentSerializationError(
                            "OpenAI reasoning replay requires reasoning_id and summary",
                            context={
                                "provider": "openai",
                                "role": message.role,
                                "block_type": block.type,
                                "hint": "Provide both reasoning_id and summary in ReasoningBlock.",
                            },
                        )
                    item: dict[str, Any] = {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": block.summary}],
                    }
                    if block.signature is not None:
                        item["encrypted_content"] = block.signature
                    serialized.append(item)
                    continue

                raise AgentSerializationError(
                    f"Unsupported canonical block type '{block.type}' for OpenAI",
                    context={
                        "provider": "openai",
                        "role": message.role,
                        "block_type": block.type,
                        "hint": "Use text/tool_call/tool_output/reasoning blocks supported by OpenAI serializer.",
                    },
                )

        return serialized

    def _prepare_request_params(
        self,
        messages: list[dict[str, Any]],
        system_instruction: str | None,
        stream: bool,
        tools: list[Tool] | None,
        tool_choice: str | None,
        parallel_tool_calls: bool | None,
    ) -> dict[str, Any]:
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "instructions": system_instruction,
            "input": messages,
            "stream": stream,
            "store": self.store,
            "max_output_tokens": self.max_output_tokens,
            **self.model_kwargs,
        }
        reasoning_config = {}
        text_config = {}
        if self.reasoning_effort is not None:
            reasoning_config["effort"] = self.reasoning_effort
        if self.reasoning_summary is not None:
            reasoning_config["summary"] = self.reasoning_summary
        if reasoning_config:
            params["reasoning"] = reasoning_config
        if self.verbosity is not None:
            text_config["verbosity"] = self.verbosity
        if self.text_format == "json_schema":
            text_config["format"] = self.response_format
        if self.text_format in ("text", "json_object"):
            text_config["format"] = {"type": self.text_format}
        if text_config:
            params["text"] = text_config
        if tools:
            params["tools"] = [tool.to_openai_tool() for tool in tools]
            params["tool_choice"] = tool_choice or "auto"
            params["parallel_tool_calls"] = (
                parallel_tool_calls if parallel_tool_calls is not None else True
            )
        return params

    def _map_provider_exception(self, exc: Exception) -> LLMError:
        context = {"provider": "openai", "model": self.model}
        error_message = str(exc)
        import openai

        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return LiteAuthenticationError(error_message, context=context, cause=exc)
        if isinstance(exc, openai.RateLimitError):
            return LiteRateLimitError(error_message, context=context, cause=exc)
        if isinstance(exc, (openai.BadRequestError, openai.UnprocessableEntityError)):
            return LiteInvalidRequestError(error_message, context=context, cause=exc)
        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
            return LiteAPIConnectionError(error_message, context=context, cause=exc)
        if isinstance(exc, openai.APIStatusError):
            return LiteAPIStatusError(error_message, context=context, cause=exc)

        return LLMError(error_message, context=context, cause=exc)

    def generate(
        self,
        messages: list[dict[str, Any]],
        system_instruction: str | None,
        stream: bool,
        tools: list[Tool] | None,
        tool_choice: str | None,
        parallel_tool_calls: bool | None,
    ) -> Any:
        params = self._prepare_request_params(
            messages=messages,
            system_instruction=system_instruction,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        from openai import OpenAIError
        try:
            return self.client.responses.create(**params)
        except OpenAIError as exc:
            mapped = self._map_provider_exception(exc)
            _LOGGER.log_exception(mapped, context={"method": "generate"})
            raise mapped from exc

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        system_instruction: str | None,
        stream: bool,
        tools: list[Tool] | None,
        tool_choice: str | None,
        parallel_tool_calls: bool | None,
    ) -> Any:
        params = self._prepare_request_params(
            messages=messages,
            system_instruction=system_instruction,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        from openai import OpenAIError
        try:
            return await self.aclient.responses.create(**params)
        except OpenAIError as exc:
            mapped = self._map_provider_exception(exc)
            _LOGGER.log_exception(mapped, context={"method": "agenerate"})
            raise mapped from exc

    def close(self):
        self.client.close()

    async def aclose(self):
        return await self.aclient.close()

    def __enter__(self) -> ChatOpenAI:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    async def __aenter__(self) -> ChatOpenAI:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()
