import os
import sys
import unittest

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun import ChatOpenAI, PromptTemplate
from literun.errors import AgentInputError, AgentSerializationError

try:
    import openai  # noqa: F401

    HAS_OPENAI_SDK = True
except Exception:
    HAS_OPENAI_SDK = False


@unittest.skipUnless(HAS_OPENAI_SDK, "openai sdk not installed")
class TestChatOpenAINormalization(unittest.TestCase):
    def setUp(self):
        self.llm = ChatOpenAI(model="gpt-5-nano", api_key="test-key")

    def test_normalize_messages_str(self):
        normalized = self.llm.normalize_messages("Hello")
        self.assertEqual(normalized, [{"role": "user", "content": "Hello"}])

    def test_normalize_messages_list_copy(self):
        source = [{"role": "user", "content": "Hi"}]
        normalized = self.llm.normalize_messages(source)
        self.assertEqual(normalized, source)
        self.assertIsNot(normalized, source)

    def test_normalize_messages_prompt_template(self):
        prompt = PromptTemplate()
        prompt.add_system("System rule")
        prompt.add_user("Question")
        prompt.add_assistant("Answer draft")
        prompt.add_tool_call(
            name="get_weather",
            call_id="call_1",
            arguments={"city": "Tokyo"},
        )
        prompt.add_tool_output(
            call_id="call_1",
            name="get_weather",
            output="22C",
        )
        prompt.add_reasoning(summary="Reasoning summary", reasoning_id="rs_1")

        normalized = self.llm.normalize_messages(prompt)
        self.assertTrue(any(item.get("role") == "system" for item in normalized))
        self.assertTrue(any(item.get("type") == "function_call" for item in normalized))
        self.assertTrue(
            any(item.get("type") == "function_call_output" for item in normalized)
        )
        self.assertTrue(any(item.get("type") == "reasoning" for item in normalized))

    def test_normalize_messages_invalid_input(self):
        with self.assertRaises(AgentInputError):
            self.llm.normalize_messages(123)  # type: ignore[arg-type]

    def test_reasoning_block_requires_id_and_summary_for_openai_serializer(self):
        prompt = PromptTemplate()
        prompt.add_reasoning(summary="Only summary, no id")
        with self.assertRaises(AgentSerializationError):
            self.llm.normalize_messages(prompt)


if __name__ == "__main__":
    unittest.main()
