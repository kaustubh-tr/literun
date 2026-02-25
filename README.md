# LiteRun 🚀

[![PyPI - Version](https://img.shields.io/pypi/v/literun)](https://pypi.org/project/literun/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/literun)](https://pypi.org/project/literun/)
[![PyPI - License](https://img.shields.io/pypi/l/literun)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-DOCS-blue)](https://github.com/kaustubh-tr/literun/blob/main/DOCS.md)

A lightweight, production-grade Python framework for building predictable, multi-turn AI agents. LiteRun standardizes the chaotic mechanics of modern LLM APIs (like tool-loop continuation, JSON stream assembly, and token accounting) while giving you absolute control over execution and state.

*Currently supports **OpenAI Responses API**.*

## Key Features

- **Standardized Execution**: A symmetric API for `run()`, `arun()`, `stream()`, and `astream()` with normalized result/event schemas for supported provider behavior.
- **Structured Tooling Runtime**: Pydantic-powered schema generation, execution routing, and output validation.
- **Secure Context Injection**: Safely pass ephemeral app state (like DB connections or Tenant IDs) into tools via `ToolRuntime` without exposing it to the LLM.
- **OpenAI-Focused Token Accounting**: Exposes explicit `cached_read`, `reasoning`, and standard token buckets when usage data is available.
- **Canonical Prompting**: A strictly typed `PromptTemplate` builder that enforces message invariants before network execution.

## Requirements

- Python 3.10+

> **Note**: Core dependencies like `openai` and `pydantic` are automatically installed when you install `literun`.

## Installation

Install `literun` directly from PyPI. 

```bash
pip install literun
```

Set your API key in your environment:

```bash
export OPENAI_API_KEY="sk-proj-..."
```

## Quick Start

Here is a simple example demonstrating how to initialize an Agent, register a Tool using Pydantic schemas, and execute a synchronous run.

```python
from literun import Agent, ChatOpenAI, Tool
from pydantic import BaseModel, Field

# 1. Define the tool's input schema for strict validation
class WeatherInput(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    unit: str = Field(default="celsius", description="The unit of temperature")

# 2. Define the Python logic
def get_weather(location: str, unit: str = "celsius") -> str:
    return f"The weather in {location} is 22 degrees {unit}."

# 3. Wrap it in a LiteRun Tool
weather_tool = Tool(
    func=get_weather,
    name="get_weather",
    description="Get the current weather for a specific location.",
    input_schema=WeatherInput,
    strict=True # Enforces stricter OpenAI schema adherence (model/provider dependent)
)

# 4. Initialize the Agent Orchestrator
agent = Agent(
    llm=ChatOpenAI(model="gpt-5-nano"),
    system_instruction="You are a helpful and concise weather assistant.",
    tools=[weather_tool],
)

# 5. Execute the Run
result = agent.run("What is the weather in Tokyo?")

print(f"Response: {result.output}")
print(f"Usage: {result.token_usage}")
print(f"Execution Time: {result.timing.duration:.2f}s")

```

### Advanced Usage & Examples

LiteRun supports sync/async execution in both non-streaming and streaming modes, plus runtime context injection and direct LLM client usage.

👉 Check out the [Documentation](https://github.com/kaustubh-tr/literun/blob/main/DOCS.md) and [Examples](https://github.com/kaustubh-tr/literun/blob/main/examples/) for more details.

## Testing

This project uses `pytest` as the primary test runner, but supports `unittest` as well.

```bash
# Run all tests
python -m pytest
```

or using unittest:

```bash
python -m unittest discover tests
```

> **Note**: Some integration tests may require the `OPENAI_API_KEY` environment variable. They are automatically skipped if it is missing.

## License

Copyright (c) 2026 Kaustubh Trivedi.

Distributed under the terms of the [MIT](https://github.com/kaustubh-tr/literun/blob/main/LICENSE) license, LiteRun is free and open source software.
