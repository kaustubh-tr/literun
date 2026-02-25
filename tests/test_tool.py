import asyncio
import os
import sys
import unittest

from pydantic import BaseModel

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun import Tool, ToolRuntime, tool
from literun.errors import AgentToolCallError


class InputSchema(BaseModel):
    x: int


class OutputSchema(BaseModel):
    result: int


class TestToolDefinition(unittest.TestCase):
    def test_from_callable_sync(self):
        def add(x: int) -> int:
            return x + 1

        t = Tool.from_callable(add, description="increment")
        self.assertEqual(t.name, "add")
        self.assertEqual(t.description, "increment")
        self.assertIsNotNone(t.func)
        self.assertIsNone(t.coroutine)

    def test_from_callable_async(self):
        async def add_async(x: int) -> int:
            return x + 1

        t = Tool.from_callable(add_async, name="add_async")
        self.assertEqual(t.name, "add_async")
        self.assertIsNone(t.func)
        self.assertIsNotNone(t.coroutine)

    def test_from_callable_invalid(self):
        with self.assertRaises(TypeError):
            Tool.from_callable(123)  # type: ignore[arg-type]

    def test_openai_schema_generation_excludes_runtime(self):
        def search(query: str, runtime: ToolRuntime) -> str:
            return f"{query}-{getattr(runtime, 'request_id', 'na')}"

        t = Tool.from_callable(search, name="search")
        schema = t.to_openai_tool()
        params = schema["parameters"]["properties"]
        self.assertIn("query", params)
        self.assertNotIn("runtime", params)

    def test_sync_run_with_runtime_and_validation(self):
        def multiply(x: int, runtime: ToolRuntime) -> int:
            return x * int(getattr(runtime, "factor", 1))

        t = Tool.from_callable(
            multiply,
            input_schema=InputSchema,
            output_schema=OutputSchema,
        )
        result = t.run({"x": "3"}, runtime_context={"factor": 4})
        self.assertEqual(result, 12)

    def test_invalid_input_raises_agent_tool_call_error(self):
        def multiply(x: int) -> int:
            return x * 2

        t = Tool.from_callable(multiply, input_schema=InputSchema)
        with self.assertRaises(AgentToolCallError):
            t.run({"x": "bad-int"})


class TestToolAsync(unittest.IsolatedAsyncioTestCase):
    async def test_async_run_with_coroutine(self):
        async def mul_async(x: int) -> int:
            return x * 2

        t = Tool.from_callable(mul_async, input_schema=InputSchema)
        result = await t.arun({"x": 7})
        self.assertEqual(result, 14)

    async def test_async_run_falls_back_to_sync(self):
        def mul_sync(x: int) -> int:
            return x * 3

        t = Tool.from_callable(mul_sync, input_schema=InputSchema)
        result = await t.arun({"x": 5})
        self.assertEqual(result, 15)


class TestToolDecorator(unittest.TestCase):
    def test_tool_decorator(self):
        @tool(name="echo")
        def echo(text: str) -> str:
            return text

        self.assertIsInstance(echo, Tool)
        self.assertEqual(echo.name, "echo")
        self.assertEqual(echo.run({"text": "hi"}), "hi")


if __name__ == "__main__":
    unittest.main()
