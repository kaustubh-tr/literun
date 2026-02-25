"""Structured logger helpers for agent runs."""

from __future__ import annotations

import logging
from typing import Any

from .errors import ErrorCode, LiteRunError


class AgentLogger:
    """Thin wrapper for structured literun logging."""

    def __init__(self, name: str = "literun") -> None:
        """Initialize the logger service.

        Args:
            name: The namespace for the Python logging module.
        """
        self.name = name

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(self.name)

    def error_payload(
        self,
        exc: Exception,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a normalized error payload for logs/telemetry."""
        payload: dict[str, Any] = {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "error_code": ErrorCode.UNKNOWN.value,
            "retryable": False,
            "context": context or {},
        }

        if isinstance(exc, LiteRunError):
            payload["error_code"] = exc.error_code.value
            payload["retryable"] = exc.retryable_error
            
            combined_context = dict(exc.context)
            if context:
                combined_context.update(context)
            payload["context"] = combined_context
            if exc.cause is not None:
                payload["cause_type"] = type(exc.cause).__name__
                payload["cause_message"] = str(exc.cause)

        if exc.__cause__ is not None:
            payload["cause_type"] = type(exc.__cause__).__name__
            payload["cause_message"] = str(exc.__cause__)

        return payload

    def log_exception(
        self,
        exc: Exception,
        *,
        context: dict[str, Any] | None = None,
        level: int = logging.ERROR,
        include_traceback: bool = False,
    ) -> dict[str, Any]:
        """Emit a structured error log and return payload."""
        payload = self.error_payload(exc, context=context)
        self.logger.log(
            level,
            f"[{payload['error_code']}] {payload['message']}",
            exc_info=include_traceback,
            extra={"literun_error": payload},
        )
        return payload
