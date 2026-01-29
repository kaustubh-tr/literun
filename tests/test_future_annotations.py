"""Test that ToolRuntime injection works with future annotations."""

from __future__ import annotations

import unittest
from literun import Agent, ChatOpenAI, Tool, ArgsSchema, ToolRuntime


class TestFutureAnnotations(unittest.TestCase):
    """Test that tools work correctly when using future annotations."""

    def test_toolruntime_with_future_annotations(self):
        """Verify ToolRuntime injection works when using from __future__ import annotations."""

        # Define a tool that uses ToolRuntime with future annotations enabled
        def get_config_value(key: str, ctx: ToolRuntime) -> str:
            value = getattr(ctx, key, "not found")
            return f"Config {key} = {value}"

        tool = Tool(
            name="get_config",
            description="Get a config value",
            func=get_config_value,
            args_schema=[ArgsSchema(name="key", type=str, description="Config key")],
        )

        agent = Agent(llm=ChatOpenAI(api_key="fake"), tools=[tool])

        # Execute the tool with runtime context
        result = agent._execute_tool(
            "get_config", {"key": "db_host"}, runtime_context={"db_host": "localhost"}
        )

        self.assertEqual(result, "Config db_host = localhost")

    def test_no_toolruntime_with_future_annotations(self):
        """Verify tools without ToolRuntime still work with future annotations."""

        def simple_echo(msg: str) -> str:
            return f"Echo: {msg}"

        tool = Tool(
            name="echo",
            description="Echo a message",
            func=simple_echo,
            args_schema=[
                ArgsSchema(name="msg", type=str, description="Message to echo")
            ],
        )

        agent = Agent(llm=ChatOpenAI(api_key="fake"), tools=[tool])

        result = agent._execute_tool("echo", {"msg": "hello"})
        self.assertEqual(result, "Echo: hello")


if __name__ == "__main__":
    unittest.main()
