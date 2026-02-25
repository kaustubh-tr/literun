import os
import sys
import unittest

from pydantic import ValidationError

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun import (
    PromptMessage,
    PromptTemplate,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    ToolOutputBlock,
)
from literun.errors import AgentInputError, AgentSerializationError


class TestPromptMessage(unittest.TestCase):
    def test_message_role_invariants(self):
        PromptMessage(role="system", content=[TextBlock(text="rules")])
        PromptMessage(
            role="assistant",
            content=[ToolCallBlock(call_id="c1", name="search", arguments={"q": "x"})],
        )
        PromptMessage(
            role="user",
            content=[ToolOutputBlock(call_id="c1", output="ok")],
        )

        with self.assertRaises(ValidationError):
            PromptMessage(role="system", content=[])

        with self.assertRaises(ValidationError):
            PromptMessage(
                role="system",
                content=[ToolCallBlock(call_id="c1", name="x", arguments={})],
            )


class TestPromptTemplate(unittest.TestCase):
    def test_builder_methods(self):
        prompt = PromptTemplate()
        prompt.add_system("S")
        prompt.add_user("U")
        prompt.add_assistant("A")
        prompt.add_tool_call(name="weather", arguments={"city": "Tokyo"}, call_id="tc1")
        prompt.add_tool_output(call_id="tc1", output="22C", name="weather")
        prompt.add_reasoning(summary="short summary", reasoning_id="r1")

        messages = prompt.to_messages()
        self.assertEqual(len(messages), 6)
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[1].role, "user")
        self.assertEqual(messages[2].role, "assistant")
        self.assertEqual(messages[3].content[0].type, "tool_call")
        self.assertEqual(messages[4].content[0].type, "tool_output")
        self.assertIsInstance(messages[5].content[0], ReasoningBlock)

    def test_add_tool_call_json_string(self):
        prompt = PromptTemplate()
        prompt.add_tool_call(name="search", arguments='{"q":"hello"}', call_id="c1")
        msg = prompt.to_messages()[0]
        block = msg.content[0]
        self.assertEqual(block.type, "tool_call")
        self.assertEqual(block.arguments, {"q": "hello"})

    def test_add_tool_call_invalid_json(self):
        prompt = PromptTemplate()
        with self.assertRaises(AgentSerializationError):
            prompt.add_tool_call(name="search", arguments="{bad json", call_id="c1")

        with self.assertRaises(AgentInputError):
            prompt.add_tool_call(name="search", arguments=123, call_id="c1")  # type: ignore[arg-type]

    def test_copy(self):
        prompt = PromptTemplate().add_user("hello")
        clone = prompt.copy()
        self.assertEqual(len(prompt), len(clone))
        self.assertIsNot(prompt.messages, clone.messages)


if __name__ == "__main__":
    unittest.main()
