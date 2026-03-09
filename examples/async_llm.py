import asyncio
import os
import sys
from pathlib import Path

# Make local package importable when running this file directly.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from literun.providers import ChatOpenAI
# from literun.providers import ChatGemini  # uncomment for Gemini models


async def main() -> None:
    # Toggle provider by commenting/uncommenting the block below.
    llm = ChatOpenAI(model="gpt-5-nano")  # for OpenAI models
    # llm = ChatGemini(model="gemini-3-flash-preview")  # for Gemini models

    if llm.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY.")
        return
    if llm.provider == "gemini" and not os.getenv("GOOGLE_API_KEY"):
        print("Please set GOOGLE_API_KEY.")
        return

    messages = llm.normalize_messages(
        [{"role": "user", "content": "Write a 2-line poem about recursion."}]
    )

    print(f"\n=== Async Non-streaming ({llm.provider}) ===")
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

    print(f"\n=== Async Streaming ({llm.provider}) ===")
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
