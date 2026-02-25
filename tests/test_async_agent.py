import os
import sys
import unittest

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun import Agent, Tool
from literun.constants import ToolCall
from literun.events import MessageOutputStreamDelta, StreamEndEvent, ToolCallStreamDone
from literun.results import RunResult, RunStreamEvent
from literun.usage import TokenUsage

from tests.helpers import FakeLLM


class TestAgentAsync(unittest.IsolatedAsyncioTestCase):
    async def test_arun_text_only(self):
        llm = FakeLLM(
            model="fake-model",
            scripted_responses=[
                {
                    "text": "hello async",
                    "tool_calls": [],
                    "usage": TokenUsage(input_tokens=4, output_tokens=3, total_tokens=7),
                    "items": [],
                }
            ],
        )
        agent = Agent(llm=llm)
        result = await agent.arun("hello?")

        self.assertIsInstance(result, RunResult)
        self.assertEqual(result.output, "hello async")
        self.assertEqual(result.token_usage.total_tokens, 7)

    async def test_arun_with_tool(self):
        async def mul(x: int, y: int) -> int:
            return x * y

        llm = FakeLLM(
            model="fake-model",
            scripted_responses=[
                {
                    "text": "",
                    "tool_calls": [ToolCall(call_id="c1", name="mul", arguments={"x": 3, "y": 4})],
                    "usage": TokenUsage(input_tokens=5, output_tokens=2, total_tokens=7),
                    "items": [],
                },
                {
                    "text": "12",
                    "tool_calls": [],
                    "usage": TokenUsage(input_tokens=3, output_tokens=1, total_tokens=4),
                    "items": [],
                },
            ],
        )
        agent = Agent(llm=llm, tools=[Tool.from_callable(mul, name="mul")], max_iterations=3)
        result = await agent.arun("multiply")
        self.assertEqual(result.output, "12")
        self.assertEqual(result.token_usage.total_tokens, 11)

    async def test_astream(self):
        llm = FakeLLM(
            model="fake-model",
            scripted_streams=[
                [
                    MessageOutputStreamDelta(id="m1", delta="hello"),
                    StreamEndEvent(
                        id="end1",
                        token_usage=TokenUsage(input_tokens=2, output_tokens=1, total_tokens=3),
                    ),
                ]
            ],
        )
        agent = Agent(llm=llm)

        outputs: list[str] = []
        events: list[RunStreamEvent] = []
        async for event in agent.astream("say hello"):
            events.append(event)
            outputs.append(event.output)

        self.assertTrue(events)
        self.assertEqual(outputs[-1], "hello")

    async def test_astream_tool_loop(self):
        def add(a: int, b: int) -> int:
            return a + b

        llm = FakeLLM(
            model="fake-model",
            scripted_streams=[
                [ToolCallStreamDone(id="c1", call_id="c1", name="add", output={"a": 10, "b": 5})],
                [MessageOutputStreamDelta(id="m2", delta="15"), StreamEndEvent(id="end2")],
            ],
        )
        agent = Agent(llm=llm, tools=[Tool.from_callable(add, name="add")], max_iterations=3)

        saw_tool_output_done = False
        last_output = ""
        async for event in agent.astream("add"):
            if event.event.type == "tool.output.done":
                saw_tool_output_done = True
            last_output = event.output

        self.assertTrue(saw_tool_output_done)
        self.assertEqual(last_output, "15")


if __name__ == "__main__":
    unittest.main()
