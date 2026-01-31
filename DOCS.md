# LiteRun Documentation 📚

LiteRun is a lightweight, flexible Python framework for building custom OpenAI agents. It provides a robust abstraction over the OpenAI Chat Completions API, adding tool management, structured prompt handling, and event-driven execution without the bloat of larger frameworks.

## Table of Contents

- [Core Architecture](#core-architecture)
- [Agent Execution](#agent-execution)
- [Tool Management](#tool-management)
- [Runtime Context Injection](#runtime-context-injection)
- [Prompt Templates](#prompt-templates)
- [Streaming](#streaming)
- [Direct LLM Usage](#direct-llm-usage)

---

## Core Architecture

LiteRun is built around three main components:

1.  **Agent**: The orchestrator that manages the interaction loop between the user, the LLM, and the tools.
2.  **Tool**: A wrapper around Python functions that handles argument validation (via Pydantic logic) and schema generation for OpenAI.
3.  **ChatOpenAI**: A wrapper around the `openai` client that handles API communication, including `bind_tools`.

### Design Philosophy

- **Type Safety**: Heavily relies on Python type hints and Pydantic for validation.
- **Transparency**: Exposes raw OpenAI events and responses where possible.
- **Simplicity**: Minimal abstractions; "it's just Python functions".

---

## Agent Execution

The `Agent` runs a loop:

1.  Appends user input to the history.
2.  Calls the LLM.
3.  If the LLM calls a tool:
    - Executes the tool.
    - Appends the tool result to the history.
    - Repeats Step 2.
4.  If the LLM responds with text, returns the final result.

The `invoke` method returns a `RunResult` object containing the final output and a list of all items (messages, tool calls) generated in this run.

```python
agent = Agent(llm=llm, tools=[...])
result = agent.invoke("Hello")

# result is a RunResult object
print(result.final_output)  # The text string
print(result.new_items)     # List of all items (msgs, tool calls) generated in this run
```

---

## Tool Management

Tools are defined using the `Tool` class. You must provide:

- `name`: Unique identifier.
- `description`: Used by the LLM to understand when to call it.
- `func`: The actual Python function.
- `args_schema`: A definition of arguments for the LLM.

### Using `ArgsSchema`

The `ArgsSchema` maps argument names to types and descriptions. This generates the JSON Schema sent to OpenAI.

```python
from literun import Tool, ArgsSchema

def my_func(x: int):
    return x * 2

tool = Tool(
    name="doubler",
    description="Doubles a number",
    func=my_func,
    args_schema=[
        ArgsSchema(name="x", type=int, description="Number to double")
    ]
)
```

---

## Runtime Context Injection

Sometimes tools need access to data that shouldn't be visible to the LLM (e.g., database connections, User IDs, API keys). LiteRun supports **runtime context injection**.

1.  Annotate an argument in your tool function with `ToolRuntime`.
2.  Pass a dictionary to `agent.invoke(..., runtime_context={...})`.
3.  The agent will automatically strip this argument from the LLM schema and inject the context object at execution time.

```python
from literun import ToolRuntime

def sensitive_tool(data: str, ctx: ToolRuntime) -> str:
    # 'data' comes from LLM
    # 'ctx' comes from your application
    api_key = getattr(ctx, "api_key", None)
    return f"Processed {data} with {api_key}"

# ... Initialize tool & agent ...

agent.invoke(
    "process data",
    runtime_context={"api_key": "secret_123"}
)
```

---

## Prompt Templates

The `PromptTemplate` class helps structure conversation history. It replaces simple list-of-dict management with a type-safe builder.

```python
from literun import PromptTemplate

template = PromptTemplate()
template.add_system("You are a helpful assistant.")
template.add_user("Hello")
template.add_assistant("Hi there")

agent.invoke(user_input="How are you?", prompt_template=template)
```

You can also simulate tool interactions for testing or history restoration:

```python
# Add a tool call and its output
template.add_tool_call(
    name="get_weather",
    arguments='{"location": "Tokyo"}',
    call_id="call_123"
)
template.add_tool_output(
    call_id="call_123",
    output="Sunny, 25C"
)
```

This is especially useful for managing long-term memory or restoring chat sessions.

---

## Streaming

LiteRun supports real-time streaming of both text generation and tool execution status usage `agent.stream()`.

The stream yields `RunResultStreaming` objects containing an `event`.

Key Events:

- `response.output_text.delta`: A chunk of text content.
- `response.output_text.done`: Sent when text generation is complete.
- `response.function_call_arguments.delta`: A chunk of tool arguments (JSON).
- `response.function_call_arguments.done`: Sent when the LLM finishes generating arguments for a tool call.

```python
for result in agent.stream("Hello"):
    event = result.event

    # Text Streaming
    if event.type == "response.output_text.delta":
        print(event.delta, end="")

    # Tool Argument Streaming
    elif event.type == "response.function_call_arguments.delta":
        print(event.delta, end="")

    # Completion Events
    elif event.type == "response.output_text.done":
        print("\nText generation complete")
    elif event.type == "response.function_call_arguments.done":
        print(f"\nTool call complete: {event.name}({event.arguments})")
```

---

## Direct LLM Usage

If you don't need the agent loop (e.g. for a simple chat or classification task without tools), you can use `ChatOpenAI` directly.

```python
from literun import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
response = llm.invoke([{"role": "user", "content": "Hi"}])
print(response.output_text)
```

You can also bind tools manually if you want to handle execution yourself:

```python
llm.bind_tools([my_tool])
response = llm.invoke(...)
# Check response.output for tool calls
```

### Streaming with ChatOpenAI

```python
stream = llm.stream([{"role": "user", "content": "Tell me a joke."}])
for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="")
```

---

## Examples

For complete, runnable code examples covering these concepts, please visit the [**examples**](https://github.com/kaustubh-tr/literun/blob/main/examples/) directory in the repository.
