"""Agent definition and configuration."""

from __future__ import annotations

from pydantic import BaseModel, model_validator, ConfigDict
from typing import Any, Iterator, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import RunResult, RunStreamEvent

from .tool import Tool
from .llm import BaseLLM
from .runner import Runner
from .constants import DEFAULT_MAX_ITERATIONS_LIMIT
from .prompt import PromptTemplate


class Agent(BaseModel):
    """Agent orchestrator that manages LLM, tools, and execution flow."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str | None = None
    """Optional name for the agent."""
    description: str | None = None
    """Optional description of the agent."""
    llm: BaseLLM
    """The language model to use for generation."""
    system_instruction: str | None = None
    """System instruction to guide the agent's behavior."""
    tools: list[Tool] | None = None
    """List of tools available to the agent."""
    tool_choice: str = "auto"
    """Tool choice strategy.
    OpenAI: 'auto', 'any', 'none', 'required'
    Gemini: 'auto', 'any', 'none', 'validated'
    Anthropic: 'auto', 'any', 'none', 'tool'
    """
    parallel_tool_calls: bool = True
    """Whether to allow parallel tool calls (OpenAI only)."""
    max_iterations: int = DEFAULT_MAX_ITERATIONS_LIMIT
    """Maximum number of agent loop iterations."""

    @model_validator(mode="after")
    def _validate_config(self) -> Agent:
        """Validate configuration and initialize tools."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        return self

    def run(
        self,
        messages: str | list[dict[str, Any]] | PromptTemplate,
        *,
        runtime_context: dict[str, Any] | None = None,
    ) -> RunResult:
        """Run the agent synchronously.

        Args:
            messages: User input (string or list of messages)
            runtime_context: Runtime context to pass to tools

        Returns:
            RunResult with output, token usage, timing, and new items
        """
        return Runner.run(
            agent=self,
            messages=messages,
            runtime_context=runtime_context,
        )

    async def arun(
        self,
        messages: str | list[dict[str, Any]] | PromptTemplate,
        *,
        runtime_context: dict[str, Any] | None = None,
    ) -> RunResult:
        """Run the agent asynchronously.

        Args:
            messages: User input (string or list of messages)
            runtime_context: Runtime context to pass to tools

        Returns:
            RunResult with output, token usage, timing, and new items
        """
        return await Runner.arun(
            agent=self,
            messages=messages,
            runtime_context=runtime_context,
        )

    def stream(
        self,
        messages: str | list[dict[str, Any]] | PromptTemplate,
        *,
        runtime_context: dict[str, Any] | None = None,
    ) -> Iterator[RunStreamEvent]:
        """Run the agent synchronously with streaming.

        Args:
            messages: User input (string or list of messages)
            runtime_context: Runtime context to pass to tools

        Yields:
            RunStreamEvent objects containing streaming events
        """
        return Runner.stream(
            agent=self,
            messages=messages,
            runtime_context=runtime_context,
        )

    async def astream(
        self,
        messages: str | list[dict[str, Any]] | PromptTemplate,
        *,
        runtime_context: dict[str, Any] | None = None,
    ) -> AsyncIterator[RunStreamEvent]:
        """Run the agent asynchronously with streaming.

        Args:
            messages: User input (string or list of messages)
            runtime_context: Runtime context to pass to tools

        Yields:
            RunStreamEvent objects containing streaming events
        """
        return Runner.astream(
            agent=self,
            messages=messages,
            runtime_context=runtime_context,
        )
