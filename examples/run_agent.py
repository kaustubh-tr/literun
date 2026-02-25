import os
import sys
from pathlib import Path

# Make local package importable when running this file directly.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from literun import Agent, ChatOpenAI, PromptTemplate, Tool, ToolRuntime


def get_weather(location: str, units: str = "celsius") -> str:
    """Return a fake weather reading."""
    if units == "celsius":
        return f"{location}: 22C"
    return f"{location}: 72F"


def search_database(query: str, limit: int, runtime: ToolRuntime) -> str:
    """Demo tool showing runtime context injection."""
    user_id = getattr(runtime, "user_id", "unknown")
    request_id = getattr(runtime, "request_id", "unknown")
    return (
        f"Found {limit} rows for '{query}'. "
        f"(user_id={user_id}, request_id={request_id})"
    )


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY.")
        return

    llm = ChatOpenAI(model="gpt-5-nano")  # reasoning models don't support temperature.
    weather_tool = Tool.from_callable(get_weather, name="get_weather")
    search_tool = Tool.from_callable(search_database, name="search_database")

    agent = Agent(
        llm=llm,
        system_instruction=("You are a helpful assistant. Use tools when needed."),
        tools=[weather_tool, search_tool],
        tool_choice="auto",
        parallel_tool_calls=True,
        max_iterations=10,
    )

    prompt = PromptTemplate()
    # Prior conversation history pattern:
    # User -> Tool Call -> Tool Output -> Assistant -> User -> (Model produces final assistant)
    prompt.add_user("Find the weather in Paris.")
    prompt.add_tool_call(
        name="get_weather",
        call_id="call_123",
        arguments={"location": "Paris", "units": "celsius"},
    )
    prompt.add_tool_output(
        call_id="call_123",
        name="get_weather",
        output="Paris: 22C",
    )
    prompt.add_assistant("The weather in Paris is 22C.")
    prompt.add_user("Now find records for John Doe with limit 2 and weather in Tokyo.")

    runtime_context = {
        "user_id": "user-123",
        "request_id": "req-456",
    }

    print("\n=== Agent Non-streaming ===")
    result = agent.run(prompt, runtime_context=runtime_context)
    print("Output:", result.output)
    print("\nUsage:", result.token_usage)

    print("\n=== Agent Streaming ===")
    for event in agent.stream(prompt, runtime_context=runtime_context):
        if event.event.type == "message.output.delta" and isinstance(
            getattr(event.event, "delta", None), str
        ):
            print(event.event.delta, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
