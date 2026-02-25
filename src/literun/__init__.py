"""Literun package initialization."""

from __future__ import annotations

from .agent import Agent
from .providers import ChatOpenAI
from .tool import Tool, ToolRuntime, tool
from .prompt import PromptTemplate
from .message import (
    PromptMessage,
    TextBlock,
    ToolCallBlock,
    ToolOutputBlock,
    ReasoningBlock,
    MessageRole,
)
from .constants import Role, ContentType
from .items import (
    RunItem,
    MessageOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
    ReasoningItem,
)
from .events import StreamEvent
from .results import RunResult, RunStreamEvent
from .usage import TokenUsage, Timing
from .logger import AgentLogger
from .errors import (
    ErrorCode,
    LiteRunError,
    AgentError,
    AgentInputError,
    AgentSerializationError,
    AgentParsingError,
    AgentExecutionError,
    AgentToolExecutionError,
    AgentToolCallError,
    AgentMaxIterationsError,
    LLMError,
    APIConnectionError,
    APIStatusError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
)


__all__ = [
    "Agent",
    "ChatOpenAI",
    "Tool",
    "ToolRuntime",
    "tool",
    "PromptTemplate",
    "PromptMessage",
    "TextBlock",
    "ToolCallBlock",
    "ToolOutputBlock",
    "ReasoningBlock",
    "MessageRole",
    "Role",
    "ContentType",
    "RunItem",
    "MessageOutputItem",
    "ToolCallItem",
    "ToolCallOutputItem",
    "ReasoningItem",
    "StreamEvent",
    "RunResult",
    "RunStreamEvent",
    "TokenUsage",
    "Timing",
    "AgentLogger",
    "ErrorCode",
    "LiteRunError",
    "AgentError",
    "AgentInputError",
    "AgentSerializationError",
    "AgentParsingError",
    "AgentExecutionError",
    "AgentToolExecutionError",
    "AgentToolCallError",
    "AgentMaxIterationsError",
    "LLMError",
    "APIConnectionError",
    "APIStatusError",
    "RateLimitError",
    "AuthenticationError",
    "InvalidRequestError",
]

__version__ = "0.2.0"
