"""Agent execution runner."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator

if TYPE_CHECKING:
    from .agent import Agent

from .constants import ToolCall
from .errors import (
    AgentExecutionError,
    AgentMaxIterationsError,
    AgentToolCallError,
    AgentToolExecutionError,
)
from .events import StreamEvent, ToolCallOutputStreamDone
from .items import RunItem, ToolCallOutputItem
from .logger import AgentLogger
from .providers.base import ResponseAdapter, StreamAdapter
from .prompt import PromptTemplate
from .results import RunResult, RunStreamEvent
from .tool import Tool
from .usage import Timing, TokenUsage

_LOGGER = AgentLogger(name="literun.runner")


class Runner:
    """Core execution engine for agent loops.

    Handles the agentic loop: LLM call -> tool detection -> tool execution -> repeat.
    """
    @classmethod
    def run(
        cls,
        agent: Agent,
        messages: str | list[dict[str, Any]] | PromptTemplate,
        *,
        runtime_context: dict[str, Any] | None = None,
    ) -> RunResult:
        """Run the agent loop synchronously."""
        return cls._run_nonstream_sync(
            agent=agent,
            messages=messages,
            runtime_context=runtime_context,
        )

    @classmethod
    async def arun(
        cls,
        agent: Agent,
        messages: str | list[dict[str, Any]] | PromptTemplate,
        *,
        runtime_context: dict[str, Any] | None = None,
    ) -> RunResult:
        """Run the agent loop asynchronously."""
        return await cls._run_nonstream_async(
            agent=agent,
            messages=messages,
            runtime_context=runtime_context,
        )

    @classmethod
    def stream(
        cls,
        agent: Agent,
        messages: str | list[dict[str, Any]] | PromptTemplate,
        *,
        runtime_context: dict[str, Any] | None = None,
    ) -> Iterator[RunStreamEvent]:
        """Run the agent loop synchronously with streaming."""
        for event in cls._stream_sync(
            agent=agent,
            messages=messages,
            runtime_context=runtime_context,
        ):
            yield event

    @classmethod
    async def astream(
        cls,
        agent: Agent,
        messages: str | list[dict[str, Any]] | PromptTemplate,
        *,
        runtime_context: dict[str, Any] | None = None,
    ) -> AsyncIterator[RunStreamEvent]:
        """Run the agent loop asynchronously with streaming."""
        async for event in cls._stream_async(
            agent=agent,
            messages=messages,
            runtime_context=runtime_context,
        ):
            yield event

    @classmethod
    def _run_nonstream_sync(
        cls,
        agent: Agent,
        messages: str | list[dict[str, Any]] | PromptTemplate,
        *,
        runtime_context: dict[str, Any] | None = None,
    ) -> RunResult:
        start_time, messages, total_token_usage = cls._initialize_run_state(
            agent=agent, messages=messages
        )
        all_items: list[RunItem] = []
        adapter: ResponseAdapter = agent.llm.get_response_adapter()

        iteration = 0
        while iteration < agent.max_iterations:
            response = agent.llm.generate(
                messages=messages,
                system_instruction=agent.system_instruction,
                stream=False,
                tools=agent.tools,
                tool_choice=agent.tool_choice,
                parallel_tool_calls=agent.parallel_tool_calls,
            )

            text, tool_calls = cls._process_llm_response(
                response=response,
                adapter=adapter,
                all_items=all_items,
                total_token_usage=total_token_usage,
            )

            if not tool_calls:
                return cls._build_run_result(
                    output=text,
                    all_items=all_items,
                    total_token_usage=total_token_usage,
                    start_time=start_time,
                )

            messages.extend(adapter.build_tool_call_message(response))
            cls._execute_nonstream_tool_calls_sync(
                agent=agent,
                tool_calls=tool_calls,
                runtime_context=runtime_context,
                adapter=adapter,
                all_items=all_items,
                messages=messages,
            )
            iteration += 1

        raise AgentMaxIterationsError(
            f"Agent exceeded max iterations ({agent.max_iterations}) without completing",
            context={
                "provider": type(agent.llm).__name__,
                "max_iterations": agent.max_iterations,
                "mode": "run_sync",
            },
        )

    @classmethod
    async def _run_nonstream_async(
        cls,
        agent: Agent,
        messages: str | list[dict[str, Any]] | PromptTemplate,
        *,
        runtime_context: dict[str, Any] | None = None,
    ) -> RunResult:
        start_time, messages, total_token_usage = cls._initialize_run_state(
            agent=agent, messages=messages
        )
        all_items: list[RunItem] = []
        adapter: ResponseAdapter = agent.llm.get_response_adapter()

        iteration = 0
        while iteration < agent.max_iterations:
            response = await agent.llm.agenerate(
                messages=messages,
                system_instruction=agent.system_instruction,
                stream=False,
                tools=agent.tools,
                tool_choice=agent.tool_choice,
                parallel_tool_calls=agent.parallel_tool_calls,
            )

            text, tool_calls = cls._process_llm_response(
                response=response,
                adapter=adapter,
                all_items=all_items,
                total_token_usage=total_token_usage,
            )

            if not tool_calls:
                return cls._build_run_result(
                    output=text,
                    all_items=all_items,
                    total_token_usage=total_token_usage,
                    start_time=start_time,
                )

            messages.extend(adapter.build_tool_call_message(response))
            await cls._execute_nonstream_tool_calls_async(
                agent=agent,
                tool_calls=tool_calls,
                runtime_context=runtime_context,
                adapter=adapter,
                all_items=all_items,
                messages=messages,
            )
            iteration += 1

        raise AgentMaxIterationsError(
            f"Agent exceeded max iterations ({agent.max_iterations}) without completing",
            context={
                "provider": type(agent.llm).__name__,
                "max_iterations": agent.max_iterations,
                "mode": "run_async",
            },
        )

    @classmethod
    def _stream_sync(
        cls,
        agent: Agent,
        messages: str | list[dict[str, Any]] | PromptTemplate,
        *,
        runtime_context: dict[str, Any] | None,
    ) -> Iterator[RunStreamEvent]:
        start_time, messages, total_token_usage = cls._initialize_run_state(
            agent=agent, messages=messages
        )
        stream_adapter: StreamAdapter = agent.llm.get_stream_adapter()
        if not stream_adapter.supports_streaming:
            raise AgentExecutionError(
                f"Streaming is not supported for {type(agent.llm).__name__}. "
                "Use run() or arun() instead.",
                context={"provider": type(agent.llm).__name__, "mode": "stream_sync"},
            )

        final_output = ""
        iteration = 0

        while iteration < agent.max_iterations:
            stream = agent.llm.generate(
                messages=messages,
                system_instruction=agent.system_instruction,
                stream=True,
                tools=agent.tools,
                tool_choice=agent.tool_choice,
                parallel_tool_calls=agent.parallel_tool_calls,
            )

            streamed_tool_calls: list[ToolCall] = []
            turn_text = ""

            for event in stream_adapter.process_stream(stream):
                turn_text += cls._stream_text_fragment(event, allow_done=not turn_text)
                final_output, yielded_event = cls._process_stream_event(
                    event=event,
                    final_output=final_output,
                    total_token_usage=total_token_usage,
                    start_time=start_time,
                )
                if yielded_event is not None:
                    yield yielded_event
                stream_tool_call = cls._extract_stream_tool_call(event)
                if stream_tool_call is not None:
                    streamed_tool_calls.append(stream_tool_call)

            if not streamed_tool_calls:
                return

            messages.extend(
                stream_adapter.build_tool_call_message(
                    text=turn_text,
                    tool_calls=streamed_tool_calls,
                )
            )

            for tc in streamed_tool_calls:
                tool_output = cls._run_tool(
                    agent=agent,
                    tool_name=tc.name,
                    tool_args=tc.arguments,
                    runtime_context=runtime_context,
                )

                yield cls._build_stream_event(
                    output=final_output,
                    event=ToolCallOutputStreamDone(
                        id=tc.call_id,
                        name=tc.name,
                        output=str(tool_output),
                    ),
                    token_usage=None,
                    start_time=start_time,
                )

                messages.extend(
                    stream_adapter.build_tool_output_message(
                        tc.call_id,
                        tc.name,
                        str(tool_output),
                    )
                )

            iteration += 1

        raise AgentMaxIterationsError(
            f"Agent exceeded max iterations ({agent.max_iterations}) without completing",
            context={
                "provider": type(agent.llm).__name__,
                "max_iterations": agent.max_iterations,
                "mode": "stream_sync",
            },
        )

    @classmethod
    async def _stream_async(
        cls,
        agent: Agent,
        messages: str | list[dict[str, Any]] | PromptTemplate,
        *,
        runtime_context: dict[str, Any] | None,
    ) -> AsyncIterator[RunStreamEvent]:
        start_time, messages, total_token_usage = cls._initialize_run_state(
            agent=agent, messages=messages
        )
        stream_adapter: StreamAdapter = agent.llm.get_stream_adapter()
        if not stream_adapter.supports_streaming:
            raise AgentExecutionError(
                f"Streaming is not supported for {type(agent.llm).__name__}. "
                "Use run() or arun() instead.",
                context={"provider": type(agent.llm).__name__, "mode": "stream_async"},
            )

        final_output = ""
        iteration = 0

        while iteration < agent.max_iterations:
            stream = await agent.llm.agenerate(
                messages=messages,
                system_instruction=agent.system_instruction,
                stream=True,
                tools=agent.tools,
                tool_choice=agent.tool_choice,
                parallel_tool_calls=agent.parallel_tool_calls,
            )

            streamed_tool_calls: list[ToolCall] = []
            turn_text = ""

            async for event in stream_adapter.aprocess_stream(stream):
                turn_text += cls._stream_text_fragment(event, allow_done=not turn_text)
                final_output, yielded_event = cls._process_stream_event(
                    event=event,
                    final_output=final_output,
                    total_token_usage=total_token_usage,
                    start_time=start_time,
                )
                if yielded_event is not None:
                    yield yielded_event
                stream_tool_call = cls._extract_stream_tool_call(event)
                if stream_tool_call is not None:
                    streamed_tool_calls.append(stream_tool_call)

            if not streamed_tool_calls:
                return

            messages.extend(
                stream_adapter.build_tool_call_message(
                    text=turn_text,
                    tool_calls=streamed_tool_calls,
                )
            )

            for tc in streamed_tool_calls:
                tool_output = await cls._arun_tool(
                    agent=agent,
                    tool_name=tc.name,
                    tool_args=tc.arguments,
                    runtime_context=runtime_context,
                )

                yield cls._build_stream_event(
                    output=final_output,
                    event=ToolCallOutputStreamDone(
                        id=tc.call_id,
                        name=tc.name,
                        output=str(tool_output),
                    ),
                    token_usage=None,
                    start_time=start_time,
                )

                messages.extend(
                    stream_adapter.build_tool_output_message(
                        tc.call_id,
                        tc.name,
                        str(tool_output),
                    )
                )

            iteration += 1

        raise AgentMaxIterationsError(
            f"Agent exceeded max iterations ({agent.max_iterations}) without completing",
            context={
                "provider": type(agent.llm).__name__,
                "max_iterations": agent.max_iterations,
                "mode": "stream_async",
            },
        )

    @staticmethod
    def _initialize_messages(
        agent: Agent,
        messages: str | list[dict[str, Any]] | PromptTemplate,
    ) -> list[dict[str, Any]]:
        """Initialize provider-native conversation history via provider normalization."""
        return agent.llm.normalize_messages(messages)

    @staticmethod
    def _new_token_usage() -> TokenUsage:
        """Create a fresh zeroed token usage object."""
        return TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)

    @classmethod
    def _initialize_run_state(
        cls,
        agent: Agent,
        messages: str | list[dict[str, Any]] | PromptTemplate,
    ) -> tuple[float, list[dict[str, Any]], TokenUsage]:
        """Initialize common state shared by run/stream entrypoints."""
        return (
            time.perf_counter(),
            cls._initialize_messages(agent, messages),
            cls._new_token_usage(),
        )

    @staticmethod
    def _accumulate_token_usage(total_usage: TokenUsage, new_usage: TokenUsage) -> None:
        """Accumulate token usage in-place using __add__."""
        result = total_usage + new_usage
        for field_name, value in vars(result).items():
            setattr(total_usage, field_name, value)

    @staticmethod
    def _copy_token_usage(token_usage: TokenUsage) -> TokenUsage:
        """Return a value snapshot for event payloads."""
        return TokenUsage(
            input_tokens=token_usage.input_tokens,
            output_tokens=token_usage.output_tokens,
            cached_read_tokens=token_usage.cached_read_tokens,
            cached_write_tokens=token_usage.cached_write_tokens,
            reasoning_tokens=token_usage.reasoning_tokens,
            tool_use_tokens=token_usage.tool_use_tokens,
            total_tokens=token_usage.total_tokens,
        )

    @classmethod
    def _build_run_result(
        cls,
        output: str,
        all_items: list[RunItem],
        total_token_usage: TokenUsage,
        start_time: float,
    ) -> RunResult:
        """Build a finalized non-streaming run result."""
        return RunResult(
            output=output,
            new_items=all_items,
            token_usage=total_token_usage,
            timing=Timing(
                start_time=start_time, end_time=time.perf_counter()
            ),
        )

    @classmethod
    def _build_stream_event(
        cls,
        *,
        output: str,
        event: StreamEvent,
        token_usage: TokenUsage | None,
        start_time: float,
    ) -> RunStreamEvent:
        """Build a streaming event with updated timing snapshot."""
        return RunStreamEvent(
            output=output,
            event=event,
            token_usage=token_usage,
            timing=Timing(
                start_time=start_time, end_time=time.perf_counter()
            ),
        )

    @classmethod
    def _process_stream_event(
        cls,
        *,
        event: StreamEvent,
        final_output: str,
        total_token_usage: TokenUsage,
        start_time: float,
    ) -> tuple[str, RunStreamEvent | None]:
        """Process one stream event and return updated state + yielded event."""
        if event is None:
            return final_output, None

        if getattr(event, "type", "") == "message.output.delta":
            delta = getattr(event, "delta", None)
            if isinstance(delta, str):
                final_output += delta
            elif isinstance(delta, dict) and "text" in delta:
                text_delta = delta["text"]
                final_output += text_delta

        event_token_usage = getattr(event, "token_usage", None)
        cumulative_usage: TokenUsage | None = None
        if event_token_usage is not None:
            cls._accumulate_token_usage(total_token_usage, event_token_usage)
            cumulative_usage = cls._copy_token_usage(total_token_usage)

        run_stream_event = cls._build_stream_event(
            output=final_output,
            event=event,
            token_usage=cumulative_usage,
            start_time=start_time,
        )
        return final_output, run_stream_event

    @staticmethod
    def _stream_text_fragment(event: StreamEvent, allow_done: bool) -> str:
        """Extract text contribution from a stream event for per-turn reconstruction.

        Delta events (``message.output.delta``) are always used when present.
        Done events (``message.output.done``) are used as a fallback only when
        no delta text has been accumulated yet for the current turn, preventing
        double-counting when a provider emits both delta and done events.
        """
        if event is None:
            return ""

        event_type = getattr(event, "type", "")
        if event_type == "message.output.delta":
            delta = getattr(event, "delta", None)
            if isinstance(delta, str):
                return delta
            if isinstance(delta, dict) and isinstance(delta.get("text"), str):
                return delta["text"]
            return ""

        if allow_done and event_type == "message.output.done":
            output = getattr(event, "output", None)
            if isinstance(output, str):
                return output
            if isinstance(output, dict) and isinstance(output.get("text"), str):
                return output["text"]

        return ""

    @classmethod
    def _execute_nonstream_tool_calls_sync(
        cls,
        *,
        agent: Agent,
        tool_calls: list[ToolCall],
        runtime_context: dict[str, Any] | None = None,
        adapter: ResponseAdapter,
        all_items: list[RunItem],
        messages: list[dict[str, Any]],
    ) -> None:
        """Execute non-stream tool calls synchronously and update state."""
        for tool_call in tool_calls:
            tool_output = cls._run_tool(
                agent=agent,
                tool_name=tool_call.name,
                tool_args=tool_call.arguments,
                runtime_context=runtime_context,
            )
            cls._handle_tool_execution_result(
                tool_call=tool_call,
                tool_output=tool_output,
                adapter=adapter,
                all_items=all_items,
                messages=messages,
            )

    @classmethod
    async def _execute_nonstream_tool_calls_async(
        cls,
        *,
        agent: Agent,
        tool_calls: list[ToolCall],
        runtime_context: dict[str, Any] | None = None,
        adapter: ResponseAdapter,
        all_items: list[RunItem],
        messages: list[dict[str, Any]],
    ) -> None:
        """Execute non-stream tool calls asynchronously and update state."""
        for tool_call in tool_calls:
            tool_output = await cls._arun_tool(
                agent=agent,
                tool_name=tool_call.name,
                tool_args=tool_call.arguments,
                runtime_context=runtime_context,
            )
            cls._handle_tool_execution_result(
                tool_call=tool_call,
                tool_output=tool_output,
                adapter=adapter,
                all_items=all_items,
                messages=messages,
            )

    @staticmethod
    def _validate_and_prepare_tool(
        agent: Agent,
        tool_name: str,
        tool_args: dict[str, Any] | str,
    ) -> tuple[Tool, dict[str, Any]]:
        """Validate tool exists and prepare arguments for execution."""
        tool = None
        if agent.tools:
            for candidate in agent.tools:
                if candidate.name == tool_name:
                    tool = candidate
                    break

        if tool is None:
            raise AgentToolCallError(
                f"Tool '{tool_name}' not found",
                context={"tool_name": tool_name},
            )

        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError as exc:
                raise AgentToolCallError(
                    f"Invalid JSON arguments for tool '{tool_name}'",
                    context={"tool_name": tool_name},
                    cause=exc,
                ) from exc

        if not isinstance(tool_args, dict):
            raise AgentToolCallError(
                f"Tool '{tool_name}' arguments must be a JSON object",
                context={
                    "tool_name": tool_name,
                    "received_type": type(tool_args).__name__,
                },
            )

        return tool, tool_args

    @classmethod
    def _process_llm_response(
        cls,
        response: Any,
        adapter: ResponseAdapter,
        all_items: list[RunItem],
        total_token_usage: TokenUsage,
    ) -> tuple[str, list[ToolCall]]:
        """Process LLM response and extract normalized text, tool calls, and usage."""
        text = adapter.extract_text(response)
        tool_calls = adapter.extract_tool_calls(response)

        response_usage = adapter.extract_token_usage(response)
        if response_usage is not None:
            cls._accumulate_token_usage(total_token_usage, response_usage)

        response_items = adapter.to_run_items(response)
        all_items.extend(response_items)

        return text, tool_calls

    @classmethod
    def _extract_stream_tool_call(cls, event: StreamEvent) -> ToolCall | None:
        """Extract a normalized ToolCall from a stream event if available."""
        if event is None or getattr(event, "type", None) != "tool.call.done":
            return None

        call_id = getattr(event, "call_id", None) or getattr(event, "id", None)
        if not isinstance(call_id, str) or not call_id:
            return None

        name = getattr(event, "name", None)
        if not isinstance(name, str) or not name:
            return None

        raw_output = getattr(event, "output", None)
        arguments: dict[str, Any] | str
        if isinstance(raw_output, dict):
            arguments = raw_output
        elif isinstance(raw_output, str):
            try:
                parsed = json.loads(raw_output)
                arguments = parsed if isinstance(parsed, dict) else raw_output
            except json.JSONDecodeError:
                arguments = raw_output
        else:
            arguments = {}

        return ToolCall(call_id=call_id, name=name, arguments=arguments)

    @classmethod
    def _handle_tool_execution_result(
        cls,
        tool_call: ToolCall,
        tool_output: str,
        adapter: ResponseAdapter,
        all_items: list[RunItem],
        messages: list[dict[str, Any]],
    ) -> None:
        """Handle tool execution result by creating output item and updating messages."""
        all_items.append(
            ToolCallOutputItem(
                call_id=tool_call.call_id,
                name=tool_call.name,
                result=str(tool_output),
            )
        )
        messages.extend(
            adapter.build_tool_output_message(
                tool_call.call_id,
                tool_call.name,
                str(tool_output),
            )
        )

    @classmethod
    def _run_tool(
        cls,
        agent: Agent,
        tool_name: str,
        tool_args: dict[str, Any] | str,
        runtime_context: dict[str, Any] | None = None,
    ) -> str:
        """Execute a tool synchronously."""
        try:
            tool, validated_args = cls._validate_and_prepare_tool(
                agent=agent,
                tool_name=tool_name,
                tool_args=tool_args,
            )
            result = tool.run(validated_args, runtime_context)
            return str(result)
        except (AgentToolCallError, AgentToolExecutionError) as exc:
            _LOGGER.log_exception(
                exc,
                context={
                    "tool_name": tool_name,
                    "execution_mode": "sync",
                },
            )
            return f"Error executing tool '{tool_name}': {exc}"
        except Exception as exc:
            _LOGGER.log_exception(
                exc,
                context={
                    "tool_name": tool_name,
                    "execution_mode": "sync",
                },
                include_traceback=True,
            )
            raise AgentExecutionError(
                "Unexpected internal error during tool execution",
                context={
                    "tool_name": tool_name,
                    "execution_mode": "sync",
                },
                cause=exc,
            ) from exc

    @classmethod
    async def _arun_tool(
        cls,
        agent: Agent,
        tool_name: str,
        tool_args: dict[str, Any] | str,
        runtime_context: dict[str, Any] | None = None,
    ) -> str:
        """Execute a tool asynchronously."""
        try:
            tool, validated_args = cls._validate_and_prepare_tool(
                agent=agent,
                tool_name=tool_name,
                tool_args=tool_args,
            )
            result = await tool.arun(validated_args, runtime_context)
            return str(result)
        except (AgentToolCallError, AgentToolExecutionError) as exc:
            _LOGGER.log_exception(
                exc,
                context={
                    "tool_name": tool_name,
                    "execution_mode": "async",
                },
            )
            return f"Error executing tool '{tool_name}': {exc}"
        except Exception as exc:
            _LOGGER.log_exception(
                exc,
                context={
                    "tool_name": tool_name,
                    "execution_mode": "async",
                },
                include_traceback=True,
            )
            raise AgentExecutionError(
                "Unexpected internal error during tool execution",
                context={
                    "tool_name": tool_name,
                    "execution_mode": "async",
                },
                cause=exc,
            ) from exc
