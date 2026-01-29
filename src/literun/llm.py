"""LLM client wrapper and configuration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Iterator
from openai import OpenAI

from .tool import Tool


class ChatOpenAI:
    """Stateless wrapper for a configured OpenAI model.

    Provides a unified interface to call the OpenAI Responses API, optionally
    binding tools and streaming outputs.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4.1-mini",
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
        store: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the ChatOpenAI instance.

        Args:
            model: The model name to use.
            temperature: Sampling temperature.
            api_key: OpenAI API key.
            base_url: Custom base URL for OpenAI API.
            max_output_tokens: Maximum number of tokens in the output.
            tools: Optional list of Tool instances to bind.
            tool_choice: Optional tool selection strategy.
            parallel_tool_calls: Whether to allow parallel tool calls.
            store: Whether to store model responses.
            **kwargs: Additional model parameters.
        """
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.model_kwargs = kwargs
        self.client = (
            OpenAI(api_key=api_key, base_url=base_url)
            if base_url
            else OpenAI(api_key=api_key)
        )
        self.store = store
        self._tools = tools
        self._tool_choice = tool_choice
        self._parallel_tool_calls = parallel_tool_calls

    def bind_tools(
        self,
        *,
        tools: List[Tool],
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> ChatOpenAI:
        """Bind tools to the LLM instance.

        Args:
            tools: List of Tool instances to bind.
            tool_choice: Optional tool selection strategy.
            parallel_tool_calls: Whether to allow parallel tool calls.

        Returns:
            ``ChatOpenAI``: The updated instance with tools bound.
        """
        self._tools = tools
        self._tool_choice = tool_choice
        self._parallel_tool_calls = parallel_tool_calls
        return self

    def chat(
        self,
        *,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Any:
        """Call the model with the given messages.

        Args:
            messages: List of messages in OpenAI format.
            stream: Whether to stream the output.
            tools: Optional list of OpenAI tool definitions.
            tool_choice: Optional tool selection strategy.
            parallel_tool_calls: Whether to allow parallel tool calls.

        Returns:
            Any: The OpenAI Responses API response object (or stream).
        """
        params = {
            "model": self.model,
            "input": messages,
            "stream": stream,
            "store": self.store,
            **self.model_kwargs,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            params["max_output_tokens"] = self.max_output_tokens

        # Tools resolution
        current_tools = tools or (
            [tool.to_openai_tool() for tool in self._tools] if self._tools else None
        )
        if current_tools:
            params["tools"] = current_tools
            params["tool_choice"] = tool_choice or self._tool_choice
            params["parallel_tool_calls"] = (
                parallel_tool_calls
                if parallel_tool_calls is not None
                else self._parallel_tool_calls
            )

        return self.client.responses.create(**params)

    def invoke(self, messages: List[Dict[str, Any]]) -> Any:
        """Synchronously call the model.

        Args:
            messages: List of messages in OpenAI format.

        Returns:
            Any: The OpenAI Responses API response object.
        """
        return self.chat(messages=messages, stream=False)

    def stream(
        self,
        *,
        messages: List[Dict[str, Any]],
    ) -> Iterator[Any]:
        """Stream the model response.

        Args:
            messages: List of messages in OpenAI format.

        Yields:
            Any: Streamed response events from the OpenAI Responses API.
        """
        response = self.chat(messages=messages, stream=True)
        for event in response:
            yield event
