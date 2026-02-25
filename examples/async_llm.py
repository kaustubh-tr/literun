import asyncio
import os
import sys
from pathlib import Path

# Make local package importable when running this file directly.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from literun import ChatOpenAI


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY.")
        return

    llm = ChatOpenAI(model="gpt-5-nano")  # reasoning models doesn't support temperature.

    messages = llm.normalize_messages(
        [{"role": "user", "content": "Write a 2-line poem about recursion."}]
    )

    print("\n=== Async Non-streaming ===")
    response = await llm.agenerate(
        messages=messages,
        system_instruction="You are concise.",
        stream=False,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=None,
    )
    response_adapter = llm.get_response_adapter()
    print("Assistant:", response_adapter.extract_text(response))
    print("\nUsage:", response_adapter.extract_token_usage(response))

    print("\n=== Async Streaming ===")
    stream = await llm.agenerate(
        messages=messages,
        system_instruction="You are a helpful assistant.",
        stream=True,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=None,
    )
    stream_adapter = llm.get_stream_adapter()
    print("Assistant: ", end="", flush=True)
    async for event in stream_adapter.aprocess_stream(stream):
        if event.type == "message.output.delta" and isinstance(event.delta, str):
            print(event.delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
