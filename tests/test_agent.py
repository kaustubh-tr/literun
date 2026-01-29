import sys
import os
import unittest

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun import Agent, Tool, ArgsSchema, ChatOpenAI
from literun.results import RunResult

# Check if API key is set
SKIP_REAL_API_TESTS = os.getenv("OPENAI_API_KEY") is None


# Tests that don't require API key
class TestAgentConstructor(unittest.TestCase):
    """Tests for agent initialization and tool registration that don't require API calls"""

    def test_tools_constructor(self):
        tool = Tool(
            func=lambda: None,
            name="test_tool",
            description="desc",
            args_schema=[],
        )
        llm = ChatOpenAI(model="gpt-4o", api_key="fake-api-key")
        agent = Agent(llm=llm, tools=[tool])
        self.assertIn("test_tool", agent.tools)

        # Test duplicate registration
        with self.assertRaises(ValueError):
            # Try to initialize with duplicate tools in list
            Agent(llm=llm, tools=[tool, tool])


@unittest.skipIf(SKIP_REAL_API_TESTS, "OPENAI_API_KEY not set")
class TestAgent(unittest.TestCase):
    def test_initialization(self):
        llm = ChatOpenAI(model="gpt-4o")
        agent = Agent(llm=llm, system_prompt="Test system prompt")
        self.assertEqual(agent.llm.model, "gpt-4o")
        self.assertEqual(agent.system_prompt, "Test system prompt")
        self.assertEqual(agent.tools, {})

    def test_invoke_simple_response(self):
        llm = ChatOpenAI(model="gpt-4o")
        agent = Agent(llm=llm, system_prompt="You are a helpful assistant.")
        response = agent.invoke(user_input="Say 'Hello world' and nothing else.")
        self.assertIsInstance(response, RunResult)
        self.assertIn("Hello world", response.final_output)

    def test_invoke_with_tool_call(self):
        # Setup tool
        def echo(msg: str) -> str:
            return f"Echo: {msg}"

        tool = Tool(
            name="echo",
            description="Echoes the message back to the user",
            args_schema=[
                ArgsSchema(name="msg", type=str, description="The message to echo")
            ],
            func=echo,
        )

        llm = ChatOpenAI(model="gpt-4o")
        agent = Agent(
            llm=llm,
            system_prompt="You are a helpful assistant. Use the echo tool when asked.",
            tools=[tool],
        )

        response = agent.invoke(user_input="Please use the echo tool to say 'hello'")
        self.assertIn("Echo: hello", response.final_output)

    def test_stream_simple_response(self):
        llm = ChatOpenAI(model="gpt-4o")
        agent = Agent(llm=llm, system_prompt="You are a helpful assistant.")
        chunks = []
        for result in agent.stream(user_input="Say 'Hello world' and nothing else."):
            event = result.event
            if event.type == "response.output_text.delta":
                chunks.append(event.delta)
        response = "".join(chunks)
        self.assertIn("Hello world", response)

    def test_stream_with_tool_call(self):
        """Test streaming with tool calls"""

        # Setup tool
        def get_info(topic: str) -> str:
            return f"Information about {topic}"

        tool = Tool(
            name="get_info",
            description="Get information about a topic",
            args_schema=[
                ArgsSchema(
                    name="topic",
                    type=str,
                    description="The topic to get information about",
                )
            ],
            func=get_info,
        )

        llm = ChatOpenAI(model="gpt-4o")
        agent = Agent(
            llm=llm,
            system_prompt="You are a helpful assistant. Use get_info tool when asked for information.",
            tools=[tool],
        )

        # Track events
        tool_call_detected = False
        text_content = ""

        for result in agent.stream(
            user_input="Get information about Python and tell me about it"
        ):
            event = result.event

            if event.type == "response.function_call_arguments.done":
                tool_call_detected = True
            elif event.type == "response.output_text.delta":
                text_content += event.delta

        # Verify tool calls were made and completed
        self.assertTrue(tool_call_detected, "Should detect completed tool call")
        self.assertIn("Python", text_content)

    def test_stream_empty_input(self):
        """Test that streaming with empty input raises ValueError"""
        llm = ChatOpenAI(model="gpt-4o")
        agent = Agent(llm=llm, system_prompt="You are a helpful assistant.")

        with self.assertRaises(ValueError) as context:
            list(agent.stream(user_input=""))

        self.assertIn("cannot be empty", str(context.exception))


if __name__ == "__main__":
    unittest.main()
