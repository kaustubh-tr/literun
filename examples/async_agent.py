import asyncio
import os
import sys
from pathlib import Path

# Make local package importable when running this file directly.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from literun import Agent, Tool, tool
from literun.providers import ChatOpenAI
# from literun.providers import ChatGemini  # uncomment for Gemini models


@tool(name="get_weather")
async def get_weather(location: str) -> str:
    """Async weather tool."""
    await asyncio.sleep(0)
    return f"The weather in {location} is 25C and sunny."


def calculator(a: int, b: int) -> int:
    """Sync calculator tool."""
    return a + b


async def main() -> None:
    # Toggle provider by commenting/uncommenting one of the blocks below.

    # --- OpenAI provider ---
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY.")
        return
    llm = ChatOpenAI(model="gpt-5-nano")

    # --- Gemini provider (uncomment below and comment out the OpenAI block above) ---
    # if not os.getenv("GOOGLE_API_KEY"):
    #     print("Please set GOOGLE_API_KEY.")
    #     return
    # llm = ChatGemini(model="gemini-3-flash-preview")

    calc_tool = Tool.from_callable(calculator, name="calculator")

    agent = Agent(
        llm=llm,
        system_instruction="You are a helpful assistant.",
        tools=[get_weather, calc_tool],
        tool_choice="auto",
        parallel_tool_calls=True,
        max_iterations=10,
    )

    print(f"\n=== Async run (weather tool) ({llm.provider}) ===")
    result = await agent.arun("What is the weather in Mumbai?")
    print("Output:", result.output)
    print("\nUsage:", result.token_usage)

    print(f"\n=== Async run (calculator tool) ({llm.provider}) ===")
    result = await agent.arun("Calculate 50 + 100")
    print("Output:", result.output)
    print("\nUsage:", result.token_usage)

    print(f"\n=== Async stream ({llm.provider}) ===")
    async for event in agent.astream("Tell me a short joke about async programming."):
        if event.event.type == "message.output.delta" and isinstance(
            getattr(event.event, "delta", None), str
        ):
            print(event.event.delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
