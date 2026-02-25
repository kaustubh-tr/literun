"""Tool definition and runtime context."""

from __future__ import annotations

import inspect
import logging
import asyncio
from typing import Any, get_type_hints
from collections.abc import Awaitable, Callable
from pydantic import BaseModel, ConfigDict, model_validator

from .errors import AgentToolExecutionError, AgentToolCallError

logger = logging.getLogger(__name__)


class ToolRuntime(BaseModel):
    """Runtime context container for tools."""

    model_config = ConfigDict(extra="allow")


class Tool(BaseModel):
    """Represents a callable tool that can be invoked by an agent or LLM."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    func: Callable[..., str] | None = None
    """Synchronous function to execute"""
    coroutine: Callable[..., Awaitable[str]] | None = None
    """Asynchronous coroutine to execute"""
    name: str | None = None
    """Name of the tool (auto-derived from function if not provided)"""
    description: str = ""
    """Description of what the tool does"""
    input_schema: type[BaseModel] | None = None
    """Pydantic model for input validation"""
    output_schema: type[BaseModel] | None = None
    """Pydantic model for output validation"""
    strict: bool | None = None
    """Whether to enforce strict schema validation (OpenAI/Anthropic)"""

    @model_validator(mode="after")
    def _validate_callable(self) -> Tool:
        """Ensure correct usage of func (sync) vs coroutine (async)."""
        if self.func is None and self.coroutine is None:
            raise ValueError("One of `func` or `coroutine` must be provided.")

        # func must be synchronous (not a coroutine)
        if self.func and inspect.iscoroutinefunction(self.func):
            raise ValueError("`func` should be a synchronous function, not async.")

        # coroutine must be an async function
        if self.coroutine and not inspect.iscoroutinefunction(self.coroutine):
            raise ValueError("`coroutine` should be an async function.")

        return self

    @model_validator(mode="after")
    def _validate_name(self) -> Tool:
        """Ensure tool has a valid name."""
        if not self.name:
            if self.func:
                self.name = self.func.__name__
            elif self.coroutine:
                self.name = self.coroutine.__name__
            else:
                raise ValueError("Tool must have a name or a callable with a name.")
        return self

    def _validate_input(self, args: dict[str, Any]) -> dict[str, Any]:
        """Validate input arguments against schema."""
        if self.input_schema:
            try:
                validated = self.input_schema(**args)
                return validated.model_dump()
            except Exception as e:
                raise AgentToolCallError(
                    f"Invalid arguments for tool {self.name}: {e}"
                ) from e
        return args

    def _validate_output(self, output: Any) -> Any:
        """Validate output against schema if defined."""
        if self.output_schema:
            try:
                validated = self.output_schema(**{"result": output})
                return validated.result
            except Exception as e:
                raise AgentToolExecutionError(
                    f"Invalid output from tool {self.name}: {e}"
                ) from e
        return output

    def _inject_runtime(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None,
        target_callable: Callable[..., Any],
    ) -> dict[str, Any]:
        """Inject `ToolRuntime` if the callable requests it."""
        final_args = dict(args)
        try:
            type_hints = get_type_hints(target_callable)
        except Exception:
            sig = inspect.signature(target_callable)
            type_hints = {
                name: param.annotation
                for name, param in sig.parameters.items()
                if param.annotation != inspect.Parameter.empty
            }

        for pname, ptype in type_hints.items():
            if ptype is ToolRuntime:
                final_args[pname] = ToolRuntime(**(runtime_context or {}))

        return final_args
    
    def _python_type_to_json_schema(self, py_type: type) -> str:
        """Map a Python type annotation to its JSON Schema type string.

        Supports ``str``, ``int``, ``float``, ``bool``, ``list``, and ``dict``.
        Any other type falls back to ``"string"`` and emits a ``WARNING`` so that
        unexpected annotations are visible during development.
        """
        PYTHON_TO_JSON_SCHEMA = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        json_type = PYTHON_TO_JSON_SCHEMA.get(py_type)
        if json_type is None:
            logger.warning(
                "Unsupported type annotation %r in tool %r; falling back to 'string'.",
                py_type,
                self.name,
            )
            return "string"
        return json_type

    def _generate_parameters_schema(self) -> dict[str, Any]:
        """Helper to dynamically generate JSON schema while hiding framework artifacts."""
        if self.input_schema:
            return self.input_schema.model_json_schema()

        target_callable = self.func or self.coroutine
        sig = inspect.signature(target_callable)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            # Hide the framework runtime parameter from the LLM
            if param.annotation is ToolRuntime:
                continue

            # Variadic parameters do not map to named JSON properties.
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            annotation = (
                param.annotation
                if param.annotation is not inspect.Parameter.empty
                else str
            )
            properties[name] = {"type": self._python_type_to_json_schema(annotation)}
            if param.default is inspect.Parameter.empty:
                required.append(name)

        schema = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI tool schema format."""
        schema = {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self._generate_parameters_schema()
        }

        # Apply OpenAI-specific additional constraints
        if not self.input_schema:
            schema["parameters"]["additionalProperties"] = False

        if self.strict is not None:
            schema["strict"] = self.strict

        return schema

    def to_gemini_tool(self) -> dict[str, Any]:
        """Convert the tool into Google Gemini's required JSON schema format."""
        schema = {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self._generate_parameters_schema()
        }
        return schema

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Convert the tool into Anthropic Claude's required JSON schema format."""
        schema = {
            "name": self.name,
            "description": self.description,
            "input_schema": self._generate_parameters_schema()
        }

        # Apply Anthropic-specific strict constraints
        if self.strict is not None:
            schema["strict"] = self.strict

        return schema

    def run(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute the tool synchronously.

        Args:
            args: Arguments from the LLM
            runtime_context: Runtime context to inject into tool function

        Returns:
            Tool execution result

        Raises:
            AgentToolExecutionError: If tool execution fails
            AgentToolCallError: If arguments are invalid
        """
        if not self.func:
            raise AgentToolExecutionError(
                f"Tool {self.name} has no synchronous implementation"
            )

        # Validate input
        validated_args = self._validate_input(args)

        # Inject runtime context
        final_args = self._inject_runtime(validated_args, runtime_context, self.func)

        # Execute
        try:
            result = self.func(**final_args)
        except Exception as e:
            raise AgentToolExecutionError(
                f"Tool {self.name} execution failed: {e}"
            ) from e

        # Validate output
        return self._validate_output(result)

    async def arun(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute the tool asynchronously.

        Args:
            args: Arguments from the LLM
            runtime_context: Runtime context to inject into tool function

        Returns:
            Tool execution result

        Raises:
            AgentToolExecutionError: If tool execution fails
            AgentToolCallError: If arguments are invalid
        """
        # Validate input
        validated_args = self._validate_input(args)

        if self.coroutine:
            # Use async coroutine
            final_args = self._inject_runtime(
                validated_args, runtime_context, self.coroutine
            )
            try:
                result = await self.coroutine(**final_args)
            except Exception as e:
                raise AgentToolExecutionError(
                    f"Tool {self.name} execution failed: {e}"
                ) from e
        else:
            # Fallback to sync function in thread pool
            final_args = self._inject_runtime(
                validated_args, runtime_context, self.func
            )
            try:
                result = await asyncio.to_thread(self.func, **final_args)
            except Exception as e:
                raise AgentToolExecutionError(
                    f"Tool {self.name} execution failed: {e}"
                ) from e

        # Validate output
        return self._validate_output(result)
    
    @classmethod
    def from_callable(
        cls,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: type[BaseModel] | None = None,
        output_schema: type[BaseModel] | None = None,
        strict: bool | None = None,
    ) -> Tool:
        """Create a Tool from a sync or async callable."""
        if not callable(fn):
            raise TypeError("from_callable expects a callable object")

        actual_name = name or getattr(fn, "__name__", None)
        if not actual_name:
            raise ValueError("Tool name could not be inferred from callable")

        actual_desc = description or (inspect.getdoc(fn) or "")

        kwargs = {
            "name": actual_name,
            "description": actual_desc,
            "input_schema": input_schema,
            "output_schema": output_schema,
            "strict": strict,
        }

        if inspect.iscoroutinefunction(fn):
            return cls(coroutine=fn, **kwargs)
        return cls(func=fn, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        """Direct call support for sync execution."""
        if self.func is not None:
            return self.func(*args, **kwargs)
        raise NotImplementedError(
            "This tool does not have a synchronous function implementation."
        )

    def __str__(self) -> str:
        return f"Tool(name={self.name}, description={self.description})"


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    input_schema: type[BaseModel] | None = None,
    output_schema: type[BaseModel] | None = None,
    strict: bool | None = None,
) -> Callable[[Callable[..., Any]], Tool] | Tool:
    """Decorator/helper to build a Tool from a callable."""

    def decorate(fn: Callable[..., Any]) -> Tool:
        return Tool.from_callable(
            fn,
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            strict=strict,
        )

    if fn is not None:
        return decorate(fn)
    return decorate
