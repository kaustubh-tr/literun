"""Error hierarchy for literun."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Stable error codes for programmatic handling and logging."""

    UNKNOWN = "unknown"
    
    # Agent errors
    AGENT_INPUT_INVALID = "agent.input.invalid"
    AGENT_SERIALIZATION_FAILED = "agent.serialization.failed"
    AGENT_PARSING_FAILED = "agent.parsing.failed"
    AGENT_EXECUTION_FAILED = "agent.execution.failed"
    AGENT_MAX_ITERATIONS = "agent.max_iterations"

    # Tool errors
    TOOL_CALL_INVALID = "tool.call.invalid"
    TOOL_EXECUTION_FAILED = "tool.execution.failed"

    # Provider/API errors
    API_CONNECTION_FAILED = "api.connection.failed"
    API_STATUS_ERROR = "api.status.error"
    API_INVALID_REQUEST = "api.invalid_request"
    API_AUTH_FAILED = "api.auth.failed"
    API_RATE_LIMITED = "api.rate_limited"


class LiteRunError(Exception):
    """Base exception for literun with structured error metadata."""

    default_code: ErrorCode = ErrorCode.UNKNOWN
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        error_code: ErrorCode | None = None,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code: ErrorCode = error_code or self.default_code
        self.context: dict[str, Any] = context or {}
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause
            
    @property
    def retryable_error(self) -> bool:
        return self.retryable


class AgentError(LiteRunError):
    """Base exception for agent/runtime errors."""


class AgentInputError(AgentError):
    """Invalid input payload for agent APIs."""

    default_code = ErrorCode.AGENT_INPUT_INVALID


class AgentSerializationError(AgentError):
    """Failed to serialize prompt/messages for provider request payloads."""

    default_code = ErrorCode.AGENT_SERIALIZATION_FAILED


class AgentParsingError(AgentError):
    """Failed to parse provider outputs/events into normalized structures."""

    default_code = ErrorCode.AGENT_PARSING_FAILED


class AgentExecutionError(AgentError):
    """Error during agent loop execution/orchestration."""

    default_code = ErrorCode.AGENT_EXECUTION_FAILED


class AgentToolCallError(AgentError):
    """Invalid tool call format or argument payload."""

    default_code = ErrorCode.TOOL_CALL_INVALID


class AgentToolExecutionError(AgentError):
    """Tool execution failure."""

    default_code = ErrorCode.TOOL_EXECUTION_FAILED


class AgentMaxIterationsError(AgentError):
    """Agent loop exceeded configured max iterations."""

    default_code = ErrorCode.AGENT_MAX_ITERATIONS


class LLMError(LiteRunError):
    """Base exception for provider/LLM errors."""

    default_code = ErrorCode.API_STATUS_ERROR


class APIConnectionError(LLMError):
    """Network/timeout/connection failure when calling a provider API."""

    default_code = ErrorCode.API_CONNECTION_FAILED
    retryable = True


class APIStatusError(LLMError):
    """Generic provider API status failure."""

    default_code = ErrorCode.API_STATUS_ERROR


class InvalidRequestError(APIStatusError):
    """Invalid request parameters sent to provider."""

    default_code = ErrorCode.API_INVALID_REQUEST


class AuthenticationError(APIStatusError):
    """Authentication/authorization failure."""

    default_code = ErrorCode.API_AUTH_FAILED


class RateLimitError(APIStatusError):
    """Rate-limit failure."""

    default_code = ErrorCode.API_RATE_LIMITED
    retryable = True
