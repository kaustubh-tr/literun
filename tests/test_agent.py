import os
import sys
import unittest

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun import Agent, Tool
from literun.constants import ToolCall
from literun.errors import AgentMaxIterationsError
from literun.events import (
    MessageOutputStreamDelta,
    OtherStreamEvent,
    StreamEndEvent,
    ToolCallStreamDone,
)
from literun.results import RunResult, RunStreamEvent
from literun.usage import TokenUsage

from tests.helpers import FakeLLM


class TestAgentSync(unittest.TestCase):
    def test_run_text_only(self):
        llm = FakeLLM(
            model="fake-model",
            scripted_responses=[
                {
                    "text": "hello world",
                    "tool_calls": [],
                    "usage": TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
                    "items": [],
                }
            ],
        )
        agent = Agent(llm=llm)
        result = agent.run("say hello")

        self.assertIsInstance(result, RunResult)
        self.assertEqual(result.output, "hello world")
        self.assertEqual(result.token_usage.total_tokens, 15)

    def test_run_with_tool_loop(self):
        def add(a: int, b: int) -> int:
            return a + b

        add_tool = Tool.from_callable(add, name="add")

        llm = FakeLLM(
            model="fake-model",
            scripted_responses=[
                {
                    "text": "",
                    "tool_calls": [ToolCall(call_id="c1", name="add", arguments={"a": 2, "b": 3})],
                    "usage": TokenUsage(input_tokens=5, output_tokens=2, total_tokens=7),
                    "items": [],
                    "tool_call_messages": [
                        {
                            "type": "function_call",
                            "call_id": "c1",
                            "name": "add",
                            "arguments": {"a": 2, "b": 3},
                        }
                    ],
                },
                {
                    "text": "result is 5",
                    "tool_calls": [],
                    "usage": TokenUsage(input_tokens=8, output_tokens=4, total_tokens=12),
                    "items": [],
                },
            ],
        )
        agent = Agent(llm=llm, tools=[add_tool], max_iterations=3)
        result = agent.run("add 2 and 3")

        self.assertEqual(result.output, "result is 5")
        self.assertEqual(result.token_usage.total_tokens, 19)
        self.assertTrue(any(item.type == "tool.output.item" for item in result.new_items))

    def test_run_max_iterations_error(self):
        def noop() -> str:
            return "ok"

        llm = FakeLLM(
            model="fake-model",
            scripted_responses=[
                {
                    "text": "",
                    "tool_calls": [ToolCall(call_id="c1", name="noop", arguments={})],
                    "usage": TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                    "items": [],
                }
            ],
        )
        agent = Agent(llm=llm, tools=[Tool.from_callable(noop, name="noop")], max_iterations=1)

        with self.assertRaises(AgentMaxIterationsError):
            agent.run("loop forever")

    def test_stream_with_tool_then_final_text(self):
        def add(a: int, b: int) -> int:
            return a + b

        llm = FakeLLM(
            model="fake-model",
            scripted_streams=[
                [
                    ToolCallStreamDone(
                        id="c1",
                        call_id="c1",
                        name="add",
                        output={"a": 1, "b": 2},
                    ),
                    OtherStreamEvent(id="turn1"),
                ],
                [
                    MessageOutputStreamDelta(id="m1", delta="3"),
                    StreamEndEvent(
                        id="turn2",
                        token_usage=TokenUsage(input_tokens=3, output_tokens=1, total_tokens=4),
                    ),
                ],
            ],
        )
        agent = Agent(llm=llm, tools=[Tool.from_callable(add, name="add")], max_iterations=3)

        events = list(agent.stream("add 1 and 2"))
        self.assertTrue(events)
        self.assertIsInstance(events[0], RunStreamEvent)
        self.assertTrue(any(e.event.type == "tool.output.done" for e in events))
        self.assertTrue(any(e.output.endswith("3") for e in events))


if __name__ == "__main__":
    unittest.main()
