import os
import sys
import unittest
from unittest.mock import patch

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun import Agent, Tool
from literun.constants import ToolCall
from literun.errors import AgentExecutionError, AgentToolCallError
from literun.events import ToolCallStreamDone
from literun.runner import Runner

from tests.helpers import FakeLLM


class TestRunnerErrorPaths(unittest.TestCase):
    def setUp(self):
        self.agent = Agent(llm=FakeLLM(model="fake-model"), tools=[])

    def test_validate_and_prepare_tool_missing_tool(self):
        with self.assertRaises(AgentToolCallError):
            Runner._validate_and_prepare_tool(
                agent=self.agent,
                tool_name="unknown_tool",
                tool_args={},
            )

    def test_validate_and_prepare_tool_invalid_json_arguments(self):
        echo = Tool.from_callable(lambda text: text, name="echo")
        self.agent.tools = [echo]
        with self.assertRaises(AgentToolCallError):
            Runner._validate_and_prepare_tool(
                agent=self.agent,
                tool_name="echo",
                tool_args='{"bad_json"',
            )

    def test_validate_and_prepare_tool_non_object_arguments(self):
        echo = Tool.from_callable(lambda text: text, name="echo")
        self.agent.tools = [echo]
        with self.assertRaises(AgentToolCallError):
            Runner._validate_and_prepare_tool(
                agent=self.agent,
                tool_name="echo",
                tool_args='["not", "an", "object"]',
            )

    def test_run_tool_returns_error_string_for_known_tool_errors(self):
        tool = Tool.from_callable(lambda text: text, name="echo")
        agent = Agent(llm=FakeLLM(model="fake-model"), tools=[tool])
        result = Runner._run_tool(
            agent=agent,
            tool_name="echo",
            tool_args='{"bad_json"',
            runtime_context=None,
        )
        self.assertIn("Error executing tool 'echo'", result)

    def test_arun_tool_unexpected_exception_raises_agent_execution_error(self):
        tool = Tool.from_callable(lambda x: x, name="echo")
        agent = Agent(llm=FakeLLM(model="fake-model"), tools=[tool])

        with patch.object(
            Runner,
            "_validate_and_prepare_tool",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaises(AgentExecutionError):
                self.async_run(
                    Runner._arun_tool(
                        agent=agent,
                        tool_name="echo",
                        tool_args={},
                        runtime_context=None,
                    )
                )

    def test_extract_stream_tool_call_parses_dict_output(self):
        event = ToolCallStreamDone(
            id="item1",
            call_id="call1",
            name="calc",
            output={"a": 1},
        )
        tc = Runner._extract_stream_tool_call(event)
        self.assertEqual(tc, ToolCall(call_id="call1", name="calc", arguments={"a": 1}))

    def test_extract_stream_tool_call_parses_json_string_output(self):
        event = ToolCallStreamDone(
            id="item1",
            call_id="call1",
            name="calc",
            output='{"a": 1}',
        )
        tc = Runner._extract_stream_tool_call(event)
        self.assertEqual(tc, ToolCall(call_id="call1", name="calc", arguments={"a": 1}))

    def test_extract_stream_tool_call_missing_name_returns_none(self):
        event = ToolCallStreamDone(
            id="item1",
            call_id="call1",
            name=None,
            output={"a": 1},
        )
        tc = Runner._extract_stream_tool_call(event)
        self.assertIsNone(tc)

    @staticmethod
    def async_run(awaitable):
        import asyncio

        return asyncio.run(awaitable)


if __name__ == "__main__":
    unittest.main()
