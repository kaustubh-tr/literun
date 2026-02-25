# LiteRun Documentation 📚

Welcome to the official documentation for **LiteRun**, a lightweight, production-grade agent framework focused on clarity, control, and predictable runtime behavior.

This document is written as a comprehensive narrative guide. It is designed to be read front-to-back, equipping you with the deep architectural understanding required to design, execute, debug, and productionize LiteRun-based workflows.

> **Current Scope Notice:** This edition of the documentation focuses exclusively on the **OpenAI Responses API** as the primary execution engine. While LiteRun's internal architecture is fundamentally multi-provider, specific details regarding Gemini and Anthropic implementations are reserved for future work and are omitted from this phase.

> **Documentation Quality Notice:** This documentation was drafted with AI assistance and reviewed by maintainers. It may still contain mistakes or stale details.  
> If you find an issue, please open a documentation issue at:  
> [Open a documentation issue](https://github.com/kaustubh-tr/literun/issues/new)  
> Please include:  
> (1) section heading,  
> (2) what is incorrect,  
> (3) expected behavior, and  
> (4) code/example if relevant.

---

## Table of Contents

**Part 1: Foundations & Quick Start**

1. [Why LiteRun Exists](#1-why-literun-exists)
2. [Core Concepts at a Glance](#2-core-concepts-at-a-glance)
   * [2.1 The Agent](#21-the-agent)
   * [2.2 The LLM Provider Client (ChatOpenAI)](#22-the-llm-provider-client-chatopenai)
   * [2.3 The Tool](#23-the-tool)
   * [2.4 PromptTemplate and Canonical Messages](#24-prompttemplate-and-canonical-messages)
   * [2.5 The Run Schemas](#25-the-run-schemas)
3. [Quick Start](#3-quick-start)
   * [3.1 Installation and Environment Setup](#31-installation-and-environment-setup)
   * [3.2 Minimal Non-Stream Run](#32-minimal-non-stream-run)
   * [3.3 Minimal Stream Run](#33-minimal-stream-run)
   * [3.4 Full Executable Examples](#34-full-executable-examples)

**Part 2: The Core Public API**

4. [Public API: Agent](#4-public-api-agent)
   * [4.1 Constructor Parameters](#41-constructor-parameters)
   * [4.2 Execution Methods](#42-execution-methods)
   * [4.3 The messages Input Argument](#43-the-messages-input-argument)
   * [4.4 The runtime_context Dependency Injection](#44-the-runtime_context-dependency-injection)
5. [Public API: ChatOpenAI](#5-public-api-chatopenai)
   * [5.1 Standard Configuration Parameters](#51-standard-configuration-parameters)
   * [5.2 Advanced Responses API Parameters](#52-advanced-responses-api-parameters)
   * [5.3 Internal Method Lifecycle](#53-internal-method-lifecycle)
6. [Public API: Tool and @tool](#6-public-api-tool-and-tool)
   * [6.1 The Tool Object Model](#61-the-tool-object-model)
   * [6.2 Preferred Creation Styles](#62-preferred-creation-styles)
   * [6.3 Secure Context Injection via ToolRuntime](#63-secure-context-injection-via-toolruntime)
   * [6.4 Input and Output Validation](#64-input-and-output-validation)
   * [6.5 Design Guidance: Correct vs. Incorrect Tool Usage](#65-design-guidance-correct-vs-incorrect-tool-usage)
7. [Public API: PromptTemplate and Canonical Messages](#7-public-api-prompttemplate-and-canonical-messages)
   * [7.1 Why Use Templates Instead of Raw Dicts?](#71-why-use-templates-instead-of-raw-dicts)
   * [7.2 The Canonical Message Schema](#72-the-canonical-message-schema)
   * [7.3 Block Variants](#73-block-variants)
   * [7.4 Role Invariants](#74-role-invariants)
   * [7.5 The Builder API Reference](#75-the-builder-api-reference)

**Part 3: Inputs, Standardization, & Schemas**

8. [Input Modes in Practice](#8-input-modes-in-practice)
   * [Mode A: The str Input (Simplicity)](#mode-a-the-str-input-simplicity)
   * [Mode B: The list[dict] Input (Absolute Control)](#mode-b-the-listdict-input-absolute-control)
   * [Mode C: PromptTemplate (Strict Safety)](#mode-c-prompttemplate-strict-safety)
9. [The Standardization Model](#9-the-standardization-model)
   * [9.1 Run-Level Standardization](#91-run-level-standardization)
   * [9.2 Event-Level Standardization](#92-event-level-standardization)
   * [9.3 Usage-Level Standardization](#93-usage-level-standardization)
   * [9.4 Why This Contract Matters for Enterprise Integration](#94-why-this-contract-matters-for-enterprise-integration)
10. [Run and RunItem Schemas in Detail](#10-run-and-runitem-schemas-in-detail)
    * [10.1 The RunResult Schema](#101-the-runresult-schema)
    * [10.2 The RunStreamEvent Schema](#102-the-runstreamevent-schema)
    * [10.3 The RunItem (Trace Schema) in Depth](#103-the-runitem-trace-schema-in-depth)

**Part 4: Streaming, Accounting & Internals**

11. [Streaming Event Schemas in Detail](#11-streaming-event-schemas-in-detail)
    * [11.1 Shared Stream Event Fields](#111-shared-stream-event-fields)
    * [11.2 Text Message Events](#112-text-message-events)
    * [11.3 Tool Execution Events](#113-tool-execution-events)
    * [11.4 Reasoning Events](#114-reasoning-events)
    * [11.5 Lifecycle and Fallback Events](#115-lifecycle-and-fallback-events)
    * [11.6 End Semantics and Multi-Turn Loops](#116-end-semantics-and-multi-turn-loops)
12. [Token Usage Model in Detail](#12-token-usage-model-in-detail)
    * [12.1 Conceptual Meaning of Each Field](#121-conceptual-meaning-of-each-field)
    * [12.2 The OpenAI Mapping Rationale (Unblending)](#122-the-openai-mapping-rationale-unblending)
    * [12.3 The Timing Schema](#123-the-timing-schema)
13. [OpenAI-Specific Behavior and Mapping](#13-openai-specific-behavior-and-mapping)
    * [13.1 Message Normalization](#131-message-normalization)
    * [13.2 Reasoning Replay Expectations](#132-reasoning-replay-expectations)
    * [13.3 Stream Tool-Call Correlation](#133-stream-tool-call-correlation)

**Part 5: Observability & Best Practices**

14. [Error Model: What Fails, Why, and Where to Fix](#14-error-model-what-fails-why-and-where-to-fix)
    * [14.1 Agent-Level Errors](#141-agent-level-errors)
    * [14.2 Tool Errors](#142-tool-errors)
    * [14.3 Provider Errors](#143-provider-errors)
    * [14.4 Triage Order: Where to Look First](#144-triage-order-where-to-look-first)
15. [Logging Model: How to Observe and Debug Runs](#15-logging-model-how-to-observe-and-debug-runs)
    * [15.1 Structured Payload Fields](#151-structured-payload-fields)
    * [15.2 Why Structured Logging Matters](#152-why-structured-logging-matters)
    * [15.3 Practical Usage at the Application Boundary](#153-practical-usage-at-the-application-boundary)
16. [Design Guidance: Correct vs Incorrect Usage Patterns](#16-design-guidance-correct-vs-incorrect-usage-patterns)
    * [16.1 Message Input Choice](#161-message-input-choice)
    * [16.2 Tool Design](#162-tool-design)
    * [16.3 Streaming UI Design](#163-streaming-ui-design)
    * [16.4 Usage Analytics](#164-usage-analytics)
17. [Future Work](#17-future-work)
18. [Closing Notes](#18-closing-notes)

---

## 1. Why LiteRun Exists

When engineering agentic systems against modern provider APIs, the true difficulty is rarely found in the basic act of sending a prompt and printing a text response. A standard HTTP client can achieve that.

The real complexity emerges when you attempt to build robust, multi-turn, tool-using systems. Engineers consistently hit a wall when dealing with:

* **Tool-Call Continuation Behavior:** When an LLM decides to use a tool, it suspends its textual response, emits a JSON payload, and waits. The framework must execute the local function, capture the result, format it exactly to the provider's highly specific requirements, append it to the conversation history, and re-trigger the model. Doing this safely across parallel tool calls is mathematically and logically complex.
* **Streamed Tool Argument Assembly:** During a streaming response, an LLM does not return a complete JSON object for a tool call. It streams fragmented text deltas (e.g., `{"loc`, `ation":`, ` "Pari`, `s"}`). Assembling these fragments into valid JSON, mapping them to the correct tool execution ID, and handling interruptions requires a rigorous internal state registry.
* **Context Preservation:** Passing ephemeral state (like a database connection pool or a user's API token) to a tool function without accidentally leaking that context into the LLM's visible prompt is a critical security requirement.
* **Token Usage Accounting:** Modern models separate tokens into complex buckets—Standard Input, Cached Read Context, Cached Write Context, Output Text, and Internal Reasoning. Aggregating these accurately across a multi-turn agent loop is tedious but financially critical.
* **Practical Observability:** When a production system fails, knowing *why* it failed (Did the network time out? Did the LLM hallucinate invalid JSON? Did the Python tool function throw a `KeyError`?) dictates your recovery strategy.

LiteRun exists to make these specific mechanics explicit, reliable, and reusable without forcing you into a massive, opaque framework abstraction.

**Our Philosophy: A Predictable Runtime Contract**
The goal of LiteRun is not to hide the realities of the underlying provider APIs. The goal is to provide a reliable runtime contract. LiteRun standardizes the execution loop, the streaming events, and the token math, but it intentionally preserves the `raw_event` and allows you to inject provider-native payloads when expert-level control is required.

---

## 2. Core Concepts at a Glance

The LiteRun architecture is built upon five foundational pillars. Understanding how these pieces interact is the key to mastering the framework.

### 2.1 The `Agent`

The `Agent` is your user-facing orchestrator. It acts as the supervisor for the entire execution lifecycle. It is responsible for holding the configuration (such as the system prompt and the list of available tools) and managing the "while-loop" that drives multi-turn interactions. It exposes the primary execution methods for synchronous, asynchronous, and streaming workflows.

### 2.2 The LLM Provider Client (`ChatOpenAI`)

The provider client serves as the translation layer between LiteRun's normalized engine and the external world. Rather than hardcoding OpenAI logic into the Agent, the Agent delegates network calls to `ChatOpenAI`. This client is responsible for taking your conversation history, serializing it into the exact JSON schema required by the OpenAI Responses API, managing retries, and returning normalized response adapters.

### 2.3 The `Tool`

A `Tool` is a strictly defined wrapper around a standard Python function (synchronous) or coroutine (asynchronous). The `Tool` class automatically inspects your Python type hints and Pydantic models to generate the JSON Schema that the LLM needs to understand the function. Furthermore, it manages the secure injection of runtime context during execution, acting as a firewall between the LLM and your application's internal state.

### 2.4 `PromptTemplate` and Canonical Messages

While LiteRun allows you to pass raw dictionaries as conversation history, doing so is error-prone. The `PromptTemplate` provides a strongly typed, Pydantic-backed builder for conversation history. It enforces structural invariants (e.g., ensuring a `system` message does not accidentally contain a `tool_call` block), preventing silent API rejections before the network request is even made.

### 2.5 The Run Schemas

This is the standardization contract of LiteRun. Regardless of how complex the execution loop gets, the framework will always yield data conforming to these immutable structures:

* **`RunResult`**: The final artifact of a non-streaming run. It contains the complete textual response, a full historical trace of all actions, and aggregated token usage.
* **`RunStreamEvent`**: The high-level wrapper yielded during streaming. It contains the granular delta event, a cumulative text buffer, and point-in-time token and timing snapshots.
* **`StreamEvent`**: The underlying, normalized state payload (e.g., `message.output.delta` or `tool.call.done`) carried inside the `RunStreamEvent`. This dictates exactly what piece of the network chunk just arrived.
* **`RunItem`**: A historical trace artifact representing a single completed action in the loop (e.g., the model speaking, a tool being requested, or a tool returning data). Stored as a list inside the `RunResult`.
* **`TokenUsage`**: The mathematical accounting of all computational cost incurred during the run, strictly isolated into distinct buckets (e.g., cached vs. reasoning vs. standard).

---

## 3. Quick Start

Let's put the core concepts into practice. These minimal examples demonstrate how to instantiate an Agent and execute both non-streaming and streaming workflows.

### 3.1 Installation and Environment Setup

LiteRun is distributed as a standard Python package. Install it via `pip`.

```bash
pip install literun
```

Because LiteRun acts as a client to external APIs, you must provide authentication credentials. The most secure and frictionless way to do this is by setting an environment variable in your terminal or loading it via a `.env` file. LiteRun will automatically detect this variable upon initialization.

```bash
export OPENAI_API_KEY="sk-proj-your-api-key-here"
```

### 3.2 Minimal Non-Stream Run

A non-stream run (`.run()`) blocks the current thread until the LLM has completely finished its generation (including any necessary tool executions along the way) and returned a final textual response.

This is the standard approach for backend API endpoints, batch processing scripts, or CLI tools.

```python
from literun import Agent, ChatOpenAI

# 1. Initialize the Provider Client and the Agent
agent = Agent(
    # We instantiate ChatOpenAI targeting a specific model.
    llm=ChatOpenAI(model="gpt-5-nano"),
    
    # The system instruction defines the fundamental persona and rules.
    system_instruction="You are a concise, highly technical AI assistant.",
)

# 2. Execute the run synchronously
# We pass a simple string. LiteRun automatically wraps this into a standard user message.
result = agent.run("Explain the concept of LiteRun in one sentence.")

# 3. Inspect the RunResult object
# result.output contains the final, concatenated text generated by the model.
print("--- Output ---")
print(result.output)

# result.token_usage contains the exact, normalized token accounting for the entire call.
print("\n--- Telemetry ---")
print(f"Total Tokens: {result.token_usage.total_tokens}")

# result.timing contains the high-precision duration of the network and execution loop.
print(f"Execution Time: {result.timing.duration:.2f} seconds")
```

**What happens under the hood?**
When you call `agent.run()`, the Agent delegates the string to `ChatOpenAI.normalize_messages()`. The client translates the system instruction and user prompt into an OpenAI Responses payload. The framework then enters a `while` loop, fires the network request, receives the response, parses the textual output, calculates the tokens used, packs it all into a `RunResult` dataclass, and exits the loop.

### 3.3 Minimal Stream Run

For user-facing applications (like chat interfaces or real-time dashboards), waiting for a full response introduces unacceptable latency. The `.stream()` method allows you to process the LLM's output character-by-character as it is generated on the provider's servers.

```python
from literun import Agent, ChatOpenAI

# Initialize the Agent
agent = Agent(llm=ChatOpenAI(model="gpt-5-nano"))

print("Assistant: ", end="")

# agent.stream() returns a Python Iterator that yields RunStreamEvent objects.
for stream_event in agent.stream("Write a short, one-line joke about Python programming."):
    
    # We extract the inner, normalized event from the wrapper.
    event = stream_event.event
    
    # LiteRun normalizes complex SSE chunks into a flat hierarchy of event types.
    # We check if this specific event represents a chunk of textual output.
    if event.type == "message.output.delta" and isinstance(event.delta, str):
        # Flush the delta directly to standard out to create a typing effect.
        print(event.delta, end="", flush=True)

print("\n[Stream Complete]")
```

**What happens under the hood?**
When you call `agent.stream()`, the underlying `ChatOpenAI` client opens an HTTP connection with `stream=True`. The OpenAI API begins firing raw Server-Sent Events (SSE). The `OpenAIStreamAdapter` catches these chunks, identifies if they belong to text generation, tool argument assembly, or reasoning summaries, translates them into LiteRun's standard `StreamEvent` classes, and yields them sequentially to your `for` loop.

### 3.4 Full Executable Examples

While the snippets above demonstrate the core mechanics, we provide complete, runnable scripts in the repository's `examples/` directory. These files include proper imports, error handling, and environment setup.

* **Agent Orchestration (Multi-Turn & Tools):**
  * [examples/run_agent.py](https://github.com/kaustubh-tr/literun/blob/main/examples/run_agent.py): A complete synchronous agent loop with tool execution.
  * [examples/async_agent.py](https://github.com/kaustubh-tr/literun/blob/main/examples/async_agent.py): An asynchronous agent implementation (`arun` / `astream`), ideal for high-concurrency environments like FastAPI.

* **Raw LLM Client Usage (Stateless):**
  * [examples/run_llm.py](https://github.com/kaustubh-tr/literun/blob/main/examples/run_llm.py): How to use `ChatOpenAI` synchronously without the `Agent` supervisor.
  * [async_llm.py](https://github.com/kaustubh-tr/literun/blob/main/examples/async_llm.py): How to use the `ChatOpenAI` client asynchronously.

---

# Part 2: The Core Public API

This section details the primary interfaces you will use to construct and execute workflows in LiteRun. By understanding the parameters and methods of the `Agent` and the `ChatOpenAI` client, you gain fine-grained control over how the framework orchestrates multi-turn loops and interacts with the provider's network.

---

## 4. Public API: `Agent`

The `Agent` is the central orchestration unit in LiteRun. While the provider client (`ChatOpenAI`) handles the mechanics of HTTP requests and JSON formatting, the `Agent` is responsible for state management, tool routing, and loop safety. It binds your LLM configuration and your tool registry into a single, executable entity.

### 4.1 Constructor Parameters

When you instantiate an `Agent`, you define its persistent identity, its rules of engagement, and its safety boundaries. The constructor accepts the following parameters:

* **`llm`** (`BaseLLM`) – **Required**
  * **Description**: The provider client instance that the agent will use to generate responses. In the current scope of this documentation, this will always be an instance of `ChatOpenAI`.
  * **Design Note**: By requiring an interface (`BaseLLM`) rather than a hardcoded OpenAI client, the `Agent` adheres to the Dependency Inversion Principle, remaining completely isolated from the provider's specific network logic.

* **`name`** (`str | None`, default: `None`)
  * **Description**: An optional string used to label the agent.
  * **Usage**: Highly recommended in complex architectures where multiple agents interact. It acts as an identifier in logs, traces, and telemetry payloads, allowing you to filter errors down to a specific agent instance.

* **`description`** (`str | None`, default: `None`)
  * **Description**: Optional human-readable metadata describing the agent's purpose or capabilities (e.g., "Customer Support Triage" or "Postgres Query Generator").

* **`system_instruction`** (`str | None`, default: `None`)
  * **Description**: The top-level foundational prompt. This defines the agent's persona, its rules, and its constraints.
  * **Usage**: LiteRun passes this to the provider separately from the standard conversation history, ensuring the model prioritizes these instructions above user inputs.

* **`tools`** (`list[Tool] | None`, default: `None`)
  * **Description**: A list of registered `Tool` objects. These are the Python functions the agent is authorized to invoke. If left as `None`, the agent will operate purely as a text-to-text generator.

* **`tool_choice`** (`str`, default: `"auto"`)
  * **Description**: Dictates the policy the LLM must follow regarding tool usage.
  * **Options**:
    * `"auto"`: The model uses its own discretion to decide if a tool is needed to answer the user's prompt.
    * `"required"` / `"any"`: Forces the model to call at least one tool before responding.
    * `"none"`: Explicitly forbids the model from using tools for this specific run, even if they are provided in the `tools` list.

* **`parallel_tool_calls`** (`bool`, default: `True`)
  * **Description**: Determines whether the LLM is permitted to output multiple tool execution requests in a single turn.
  * **Usage**: When set to `True`, the model may emit multiple tool calls in one turn (for example, weather for New York, London, and Tokyo). LiteRun currently executes tool calls sequentially in the loop.


* **`max_iterations`** (`int`, default: `20`)
  * **Description**: A critical safety mechanism. This defines the maximum number of loop cycles `(LLM Call -> Tool Execution -> Append Result -> LLM Call)` the agent can perform for a single user prompt.
  * **Usage**: If the model gets confused and repeatedly calls a failing tool, this cap prevents the system from entering an infinite loop and burning through API credits. If the cap is reached, LiteRun raises an `AgentMaxIterationsError`.

### 4.2 Execution Methods

The `Agent` provides four primary entry points to accommodate any deployment architecture—from simple CLI scripts to high-throughput, async web servers.

* **`run(messages, runtime_context=None) -> RunResult`**
  * Synchronous, non-streaming execution. The thread blocks until the entire conversational loop is complete and a final text response is synthesized.
  * *(See it in action: [examples/run_agent.py](https://github.com/kaustubh-tr/literun/blob/main/examples/run_agent.py))*

* **`arun(messages, runtime_context=None) -> RunResult`**
  * Asynchronous, non-streaming execution. Must be `await`ed. This is the preferred method for frameworks like FastAPI, as it yields control of the event loop back to the server while waiting for the provider's network response.
  * *(See it in action: [examples/async_agent.py](https://github.com/kaustubh-tr/literun/blob/main/examples/async_agent.py))*

* **`stream(messages, runtime_context=None) -> Iterator[RunStreamEvent]`**
  * Synchronous streaming. Returns a standard Python generator. It blocks while waiting for the next network chunk but yields parsed `RunStreamEvent` objects the moment they arrive.

* **`astream(messages, runtime_context=None) -> AsyncIterator[RunStreamEvent]`**
  * Asynchronous streaming. Returns an async generator. Used with `async for` loops, providing the highest performance for real-time, high-concurrency websocket or SSE endpoints.

LiteRun provides a symmetric API for both synchronous and asynchronous (`asyncio`) environments. The execution methods are strictly categorized by whether they block for a complete response or yield a real-time stream:

* **`run` / `arun`**: Returns a finalized `RunResult` containing the complete text, a full trace of items, and total token usage.
* **`stream` / `astream`**: Yields granular `RunStreamEvent` objects for driving real-time UIs.

| Environment | Non-Streaming (Complete Result) | Streaming (Real-time Deltas) |
| :--- | :--- | :--- |
| **Synchronous** | `result = agent.run(...)` | `for event in agent.stream(...)` |
| **Asynchronous** | `result = await agent.arun(...)` | `async for event in agent.astream(...)` |


### 4.3 The `messages` Input Argument

All four execution methods accept a `messages` parameter, which represents the user's input or the current conversation history. LiteRun is designed to be flexible, accepting three distinct types:

1. **`str`**: For simple, one-off queries. LiteRun wraps it in a standard user message.
2. **`PromptTemplate`**: LiteRun's strongly typed, canonical message builder. This is the safest and most structured way to pass complex, multi-turn history.
3. **`list[dict]`**: Raw, provider-native dictionary lists. This acts as an escape hatch for advanced users who want to construct exact OpenAI JSON payloads manually.

*(Note: These input modes are explored exhaustively in Part 3 of this documentation).*

### 4.4 The `runtime_context` Dependency Injection

The optional `runtime_context: dict[str, Any]` argument is one of LiteRun's most powerful enterprise features.

When building production tools, your Python functions often need contextual data to execute securely—such as the current user's database ID, an active HTTP request session, or an API bearer token. However, you **never** want the LLM to know or generate this sensitive data.

By passing a dictionary to `runtime_context`, LiteRun makes that data available to your tools internally via the `ToolRuntime` class, completely bypassing the LLM's prompt and generation scope.

---

## 5. Public API: `ChatOpenAI`

The `ChatOpenAI` class is the provider client. It translates LiteRun's agnostic commands into the highly specific, proprietary JSON schemas demanded by the **OpenAI Responses API**.

Because provider APIs mutate rapidly, the `ChatOpenAI` client is strictly typed using Pydantic, ensuring that invalid parameters are caught immediately upon instantiation, rather than failing silently during a network request.

### 5.1 Standard Configuration Parameters

These parameters manage the basic HTTP connection, authentication, and core model settings.

* **`model`** (`str`, default: `"gpt-5-nano"`)
  * **Description**: The specific OpenAI model identifier you wish to invoke.

* **`api_key`** (`str | None`)
  * **Description**: Your OpenAI secret API key.
  * **Usage**: If you leave this as `None`, the client will securely attempt to read the `OPENAI_API_KEY` environment variable.

* **`organization`** / **`project`** (`str | None`)
  * **Description**: Optional string identifiers. If your OpenAI account utilizes multiple projects or organizations, setting these ensures that billing and rate limits are routed to the correct administrative bucket.

* **`base_url`** (`str | None`)
  * **Description**: Overrides the default OpenAI API endpoint.
  * **Usage**: Essential if you are routing traffic through an enterprise API gateway, a proxy for logging/caching, or an OpenAI-compatible local model server like vLLM.

* **`temperature`** (`float | None`)
  * **Description**: Controls the randomness of the model's output. A value of `0.0` yields highly deterministic, analytical responses, while values approaching `1.0` or higher encourage creative, varied phrasing.  
  *(Note: Reasoning models (like `o-series` or `gpt-5-series`) generally ignore or strictly limit temperature settings).*

* **`max_output_tokens`** (`int | None`)
  * **Description**: A hard cap on the number of tokens the model is permitted to generate in its response, including reasoning tokens. Useful for preventing runaway generation costs.

* **`timeout`** (`float | None`, default: `60.0`)
  * **Description**: The maximum number of seconds the client will wait for a network response. If the provider hangs, the client will terminate the request and raise a `LiteAPIConnectionError`.

* **`max_retries`** (`int`, default: `3`)
  * **Description**: The number of times the underlying SDK should automatically retry a request if it encounters transient network issues or a retryable `429 Rate Limit` status code.

* **`model_kwargs`** (`dict[str, Any]`)
  * **Description**: The ultimate escape hatch. Any key-value pairs placed in this dictionary are merged directly into the final API payload sent to OpenAI. If OpenAI releases a new parameter tomorrow, you can pass it here immediately without waiting for a LiteRun version update.

### 5.2 Advanced Responses API Parameters

The OpenAI Responses API introduces several new control mechanisms for reasoning models and structured data. `ChatOpenAI` exposes these natively:

* **`reasoning_effort`** (`Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None`)
  * **Description**: Directs compatible reasoning models (e.g., `o3-mini`) on how much computational time and budget to allocate to internal "thinking" before outputting text. Higher effort yields better answers but costs more reasoning tokens and increases latency.

* **`reasoning_summary`** (`Literal["auto", "concise", "detailed"] | None`)
  * **Description**: Instructs the API on how verbose the visible "thought process" block should be in the final response trace.

* **`verbosity`** (`Literal["low", "medium", "high"] | None`)
  * **Description**: A general directive to the model regarding how exhaustive its textual response should be, independent of the system prompt.

* **`text_format`** (`Literal["text", "json_object", "json_schema"]`, default: `"text"`)
  * **Description**: Dictates the structure of the model's text output.
    * `"text"`: Standard markdown/plaintext.
    * `"json_object"`: Guarantees the output will be a valid JSON string.
    * `"json_schema"`: Guarantees the output will strictly adhere to the schema provided in the `response_format` parameter.

* **`response_format`** (`object | None`)
  * **Description**: Used in conjunction with `text_format="json_schema"`. You pass the strict JSON schema (or a Pydantic model's schema dump) here to force the model to output a specific data shape.

* **`store`** (`bool`, default: `False`)
  * **Description**: When set to `True`, instructs OpenAI to store the inputs and outputs of this request on their servers. This is used for platform features like persistent memory or training data collection.

### 5.3 Internal Method Lifecycle

While you rarely need to call these methods directly (as the `Agent` handles them), understanding them clarifies how LiteRun translates data.

* **`normalize_messages()`**: Accepts `str`, `list`, or `PromptTemplate` and normalizes by input type:
  * `str` -> a simple user message payload,
  * `list` -> shallow copy pass-through for provider-native payloads,
  * `PromptTemplate` -> serialized OpenAI Responses block structure.

* **`generate()` / `agenerate()`**: Merges the normalized messages, tools, and configuration parameters into a final dictionary, fires the network request via the official `openai` Python SDK, and catches any native exceptions to remap them to LiteRun errors.

* **`get_response_adapter()` / `get_stream_adapter()`**: Returns the `OpenAIResponseAdapter` or `OpenAIStreamAdapter`. These isolated classes are responsible for taking the messy, raw OpenAI response objects or streaming SSE chunks and extracting the normalized text, tool calls, and token usage for the `Agent` to process.

> **Want to bypass the Agent and use the LLM client directly?**  
> If you only need LiteRun's strict Pydantic normalization and token accounting without the multi-turn tool loop, you can instantiate and call `ChatOpenAI` directly. 
> * For synchronous usage, *see: [examples/run_llm.py](https://github.com/kaustubh-tr/literun/blob/main/examples/run_llm.py)*
> * For asynchronous usage, *see: [examples/async_llm.py](https://github.com/kaustubh-tr/literun/blob/main/examples/async_llm.py)*

---

## 6. Public API: `Tool` and `@tool`

Tooling is where the majority of production failures occur in agentic systems. When you connect an LLM to your backend, you are translating unpredictable, probabilistic text generation into deterministic Python function calls. If the argument schemas are misaligned, if context is leaked, or if the function returns an un-serializable object, the entire loop crashes.

LiteRun’s `Tool` class is designed as a strict, defensive boundary between the LLM and your application logic.

### 6.1 The `Tool` Object Model

At its core, a `Tool` object wraps a Python executable alongside the metadata required to generate a provider-compliant JSON schema.

The underlying model supports the following attributes:

* **`func`**: A synchronous Python callable (`def`).
* **`coroutine`**: An asynchronous Python callable (`async def`).
* *Note: While a tool can technically hold both, you typically provide one or the other. LiteRun will automatically route the execution based on whether you used `agent.run()` or `agent.arun()`.*

* **`name`**: The exact string identifier the LLM must output to invoke this tool. If omitted, LiteRun derives it from the function's `__name__`.
* **`description`**: A plain-English explanation of what the tool does. **This is critical.** The LLM uses this description as a prompt to decide *when* and *how* to use the tool.
* **`input_schema`**: An optional Pydantic `BaseModel` used to validate arguments before the function is executed.
* **`output_schema`**: An optional Pydantic `BaseModel` used to validate the result before it is returned to the LLM.
* **`strict`**: A boolean flag that, when set to `True`, uses OpenAI strict structured output behavior where supported. This enforces tighter schema adherence but should not be treated as an absolute guarantee in every model/version scenario.

### 6.2 Preferred Creation Styles

LiteRun offers two primary ways to define and register tools.

**Style A: Explicit Instantiation (Preferred for Production)**
Using the `Tool` constructor (or a wrapper like `Tool.from_callable`) is the most robust method for production systems. It clearly separates the function definition from the LLM metadata. It natively handles both synchronous and asynchronous functions.

```python
from literun import Tool
import asyncio

# 1. Define your standard Python logic
async def fetch_city_weather(location: str) -> str:
    await asyncio.sleep(0.1) # Simulate network call
    return f"{location}: 25C"

# 2. Wrap it in a Tool object
weather_tool = Tool(
    name="get_weather",
    description="Fetches the current weather for a specific location.",
    coroutine=fetch_city_weather,
    strict=True
)
```

**Style B: The Decorator (`@tool`)**
For quick scripts or prototyping, you can use the decorator style. This binds the metadata directly to the function definition.

```python
from literun import tool

@tool(name="ping", description="Check system connectivity.")
def ping() -> str:
    return "pong"
```

### 6.3 Secure Context Injection via `ToolRuntime`

This is one of LiteRun's most powerful features for enterprise backends.

Often, a tool requires context to execute—such as the user's `tenant_id`, a secure database connection pool, or an HTTP request session. You **never** want to define `tenant_id` as a standard argument in your tool's schema, because that would expose it to the LLM, forcing the LLM to generate it (a massive security risk).

LiteRun solves this via dependency injection using the `ToolRuntime` class.

If you type-hint a parameter as `ToolRuntime` in your function, LiteRun will automatically strip it from the JSON schema sent to OpenAI. When the LLM calls the tool, LiteRun intercepts the call and injects the `runtime_context` dictionary you provided at the `Agent.run()` level.

```python
from literun import Tool, ToolRuntime

def query_orders(user_id: str, runtime: ToolRuntime) -> str:
    # 'user_id' is generated by the LLM based on the conversation.
    # 'runtime.tenant' is securely injected by your backend.
    
    tenant = getattr(runtime, "tenant", "default")
    db_conn = runtime.db_connection
    
    # Execute securely using the injected tenant scope
    return f"Found 3 orders for {user_id} in tenant {tenant}."

query_tool = Tool(func=query_orders)

# Execution at the API route level:
agent.run(
    "Where is my latest order?", 
    runtime_context={
        "tenant": "acme_corp",
        "db_connection": "postgres://..."
    }
)
```

### 6.4 Input and Output Validation

By default, LiteRun inspects your function's Python type hints (`location: str`) to generate the schema. For complex objects, you should provide explicit Pydantic models.

* **`input_schema`**: If provided, LiteRun will parse the LLM's raw JSON arguments through this Pydantic model *before* calling your function. If the LLM hallucinates an argument, Pydantic raises a validation error, which LiteRun catches and wraps in an `AgentToolCallError`.
* **`output_schema`**: If provided, the output of your Python function is passed through this model *after* execution. If your function returns a malformed dictionary, Pydantic catches it, raising an `AgentToolExecutionError`. This prevents malformed data from poisoning the LLM's context window.

### 6.5 Design Guidance: Correct vs. Incorrect Tool Usage

To maintain a stable runtime, adhere to the following rules:

**Correct Usage:**

* Keep tool signatures simple and explicit. Use primitives (`str`, `int`, `bool`) or simple dictionaries whenever possible.
* Use `ToolRuntime` exclusively for execution-time context that must remain hidden from the model.
* Use `input_schema` if argument validity is business-critical.

**Incorrect (Risky) Usage:**

* **Returning un-serializable objects**: Tools must return strings, numbers, or dictionaries. If your tool returns a complex object (like a raw `SQLAlchemy` model instance or an open file handler), the JSON serialization step will crash before the data reaches OpenAI.
* **Massive tool descriptions**: Do not write a 500-word essay for a tool description. Keep it concise. Overly long descriptions confuse the LLM's routing logic and consume valuable context window tokens.
* **Leaking sensitive context**: Never put secrets, API keys, or private internal state into standard tool arguments. Always use `ToolRuntime`.

---

## 7. Public API: `PromptTemplate` and Canonical Messages

When interacting with the OpenAI API, conversation history is represented as a list of dictionaries (e.g., `[{"role": "user", "content": "..."}]`).

While LiteRun allows you to pass these raw dictionaries directly, doing so in a dynamic application is dangerous. A typo in a key name, an invalid role string, or appending a `tool_call` result without the matching `call_id` will result in an immediate `400 Bad Request` from the provider.

To solve this, LiteRun provides the `PromptTemplate` and canonical message blocks.

### 7.1 Why Use Templates Instead of Raw Dicts?

The `PromptTemplate` acts as a strongly-typed, object-oriented builder for your conversation state. It utilizes Pydantic to ensure that your message history is structurally flawless before a network request is ever initiated. It provides:

1. **Autocomplete & Type Safety**: Your IDE will catch missing parameters immediately.
2. **Invariant Enforcement**: It mathematically prevents you from creating impossible states (e.g., assigning a tool execution result to the `system` role).
3. **Provider Agnosticism**: Canonical blocks isolate your business logic from OpenAI's specific formatting quirks. LiteRun's serializer handles the translation perfectly.

### 7.2 The Canonical Message Schema

At the core of this system is the `PromptMessage` model. Every turn in the conversation is an instance of `PromptMessage`, which contains exactly two fields:

* **`role`**: A strictly typed literal accepting only `"system"`, `"user"`, or `"assistant"`.
* **`content`**: A list of `MessageContentBlock` objects.

Because an LLM can return mixed content in a single turn (e.g., a paragraph of text, followed by a tool request), the `content` list uses a **Discriminated Union** to allow different types of data blocks to coexist safely.

### 7.3 Block Variants

There are four canonical block types you can append to a message:

* **`TextBlock`**: Contains standard plaintext strings.
* **`ToolCallBlock`**: Represents the LLM asking to execute a tool. Requires a `call_id`, `name`, and an `arguments` dictionary.
* **`ToolOutputBlock`**: Represents the result of a tool execution. Requires the matching `call_id` and the raw `output` data.
* **`ReasoningBlock`**: Represents the internal thought process of an advanced model (like `o-series` or `gpt-5-series`). Contains fields for `summary`, `signature` (for verifying the thought process), and `reasoning_id`.

### 7.4 Role Invariants

To guarantee that the resulting payload is accepted by the provider API, LiteRun enforces strict invariants on which blocks belong to which roles. If you violate these rules, the framework raises an immediate `ValueError`.

* **The `system` role**: May *only* contain `TextBlock` instances.
* **The `user` role**: May contain `TextBlock` instances (standard prompts) or `ToolOutputBlock` instances (returning the result of a tool back to the model).
* **The `assistant` role**: May contain `TextBlock` (the model speaking), `ToolCallBlock` (the model requesting a tool), or `ReasoningBlock` (the model thinking).

| Message Role | Allowed Content Blocks | Typical Purpose |
| :--- | :--- | :--- |
| `system` | `TextBlock` | Defining instructions and agent persona. |
| `user` | `TextBlock`, `ToolOutputBlock` | User prompts and returning local tool results. |
| `assistant` | `TextBlock`, `ToolCallBlock`, `ReasoningBlock` | Model responses, tool requests, and thoughts. |

### 7.5 Builder Methods Overview

The `PromptTemplate` exposes a clean, chainable API for constructing these blocks rapidly without instantiating the underlying Pydantic classes manually.

```python
from literun import PromptTemplate

prompt = PromptTemplate()

# Add standard text messages
prompt.add_system("You are a database analyzer.")
prompt.add_user("What is the average transaction value?")

# Advanced: Manually injecting a historical tool execution loop
# (Useful for few-shot prompting or restoring state from a database)
prompt.add_assistant("Let me query the database to find out.")
prompt.add_tool_call(
    name="run_sql", 
    arguments={"query": "SELECT AVG(value) FROM transactions"}, 
    call_id="call_abc123"
)

prompt.add_tool_output(
    call_id="call_abc123", 
    output="Average value is $45.50", 
    name="run_sql"
)

# Pass the strictly validated template to the agent
result = agent.run(prompt)
```

#### The Builder API Reference

The `PromptTemplate` exposes a clean, chainable API for constructing messages rapidly. Under the hood, these methods automatically instantiate the correct `PromptMessage` and `MessageContentBlock` classes, ensuring your history remains structurally sound.

* **`add_system(text: str) -> PromptTemplate`**
  * **Description**: Appends a `system` role message containing a `TextBlock`.
  * **Usage**: Used to define the agent's persona, global instructions, and constraints. Typically called only once at the beginning of the template.

* **`add_user(text: str) -> PromptTemplate`**
  * **Description**: Appends a `user` role message containing a `TextBlock`.
  * **Usage**: Represents the human's input or the primary prompt the model needs to respond to.

* **`add_assistant(text: str) -> PromptTemplate`**
  * **Description**: Appends an `assistant` role message containing a `TextBlock`.
  * **Usage**: Used to inject previous model responses into the history, establishing context for few-shot prompting or restoring a past conversation.

* **`add_tool_call(name: str, arguments: dict | str, call_id: str) -> PromptTemplate`**
  * **Description**: Appends an `assistant` role message specifically formatted with a `ToolCallBlock`.
  * **Parameters**:
    * `name`: The exact name of the registered tool.
    * `arguments`: A dictionary (or JSON string) of the arguments the model requested.
    * `call_id`: The unique execution ID assigned by the provider.
  * **Usage**: Crucial for rebuilding a conversation trace where the model previously asked to execute a tool.

* **`add_tool_output(call_id: str, output: str | dict, name: str = None, is_error: bool = None) -> PromptTemplate`**
  * **Description**: Appends a `user` role message specifically formatted with a `ToolOutputBlock`.
  * **Parameters**:
    * `call_id`: **Must identically match** the `call_id` from the preceding `add_tool_call`.
    * `output`: The raw result returned by your local Python function.
    * `is_error`: Optional flag indicating if the tool execution failed, helping the model realize it needs to fix its arguments.
  * **Usage**: Used to feed the results of a local function execution back into the model's context window.

* **`add_reasoning(summary: str = None, signature: str = None, reasoning_id: str = None, provider_meta: dict = None) -> PromptTemplate`**
  * **Description**: Appends an `assistant` role message containing a `ReasoningBlock`.
  * **Parameters**:
    * `summary`: The plaintext thought process of the model.
    * `signature`: The cryptographic hash required by the provider (e.g., OpenAI) to verify the reasoning block during replay.
  * **Usage**: Used exclusively when restoring conversation history for advanced reasoning models (like OpenAI `o-series` or `gpt-5-series`) to maintain their chain of thought.

* **`add_message(message: PromptMessage) -> PromptTemplate`**
  * **Description**: A lower-level method that appends an already-constructed `PromptMessage` instance directly to the template.

* **`add_messages(messages: Iterable[PromptMessage]) -> PromptTemplate`**
  * **Description**: Bulk-appends a list or generator of `PromptMessage` objects.

---

# Part 3: Inputs, Standardization, & Schemas

This section bridges the gap between how you pass data into the framework and how the framework guarantees the shape of the data it returns. Understanding these data contracts is essential for integrating LiteRun into larger enterprise architectures, such as front-end streaming UIs, database auditing layers, and telemetry dashboards.

---

## 8. Input Modes in Practice

The `messages` parameter in LiteRun's execution methods (`run`, `arun`, `stream`, `astream`) is polymorphic. It accepts three entirely different input types. This design choice provides a sliding scale between developer convenience, strict safety, and absolute low-level control.

### Mode A: The `str` Input (Simplicity)

When you pass a standard Python string, LiteRun automatically wraps it into a canonical user message behind the scenes.

**When to use it:** This mode is perfect for stateless endpoints, CLI tools, simple scripts, or the very first turn of a conversation where no prior history exists.

```python
# Simple, frictionless execution
result = agent.run("Please summarize the following text: ...")
```

### Mode B: The `list[dict]` Input (Absolute Control)

This mode acts as LiteRun's "escape hatch." If you pass a list of dictionaries, LiteRun assumes you are an expert user passing provider-native payloads. The framework applies minimal transformation, passing your dictionaries almost directly to the underlying OpenAI API.

**When to use it:** Use this when you have existing legacy code that already formats OpenAI payloads, or when OpenAI releases a brand-new message property that LiteRun's `PromptTemplate` does not yet natively support.

**Warning:** This mode is powerful but dangerous. You bypass LiteRun's validation invariants. If you format the dictionary incorrectly, the OpenAI API will reject the request with a `400 Bad Request` error.

*Example 1: Simple Provider-Native List*

```python
messages = [
    {"role": "user", "content": "Tell me one short joke."}
]
result = agent.run(messages)
```

*Example 2: Explicit OpenAI Responses API Content Blocks*
OpenAI's latest APIs support highly complex internal content blocks. You can pass them directly:

```python
messages = [
    {
        "role": "system",
        "content": [{"type": "input_text", "text": "You are a concise AI."}],
    },
    {
        "role": "user",
        "content": [{"type": "input_text", "text": "Say hello."}],
    },
]
result = agent.run(messages)
```

*Example 3: Manual Tool Continuation Payload (Advanced)*
If you are restoring a conversation from a database where a tool was previously called and resolved, you must pass the exact provider-native sequence of events.

```python
messages = [
    {"role": "user", "content": "What is the weather in Paris?"},
    {
        "type": "function_call",
        "call_id": "call_abc123",
        "name": "get_weather",
        "arguments": '{"location":"Paris"}',
    },
    {
        "type": "function_call_output",
        "call_id": "call_abc123",
        "output": "Paris: 22C",
    },
]
result = agent.run(messages)
```

### Mode C: `PromptTemplate` (Strict Safety)

As detailed in Section 7, passing a `PromptTemplate` instance is the recommended path for production systems handling multi-turn conversations. It provides compile-time type safety and guarantees that your message history is structurally valid before network transmission.

```python
prompt = PromptTemplate()
prompt.add_user("Find weather in Paris")
result = agent.run(prompt)
```

---

## 9. The Standardization Model

At the core of LiteRun is a strict standardization contract. Provider APIs can change their JSON structures, streaming chunks, and usage reporting formats over time. If your application code binds directly to raw provider response objects, your application becomes fragile under API evolution.

LiteRun isolates your application from this chaos by enforcing a normalization layer.

### 9.1 Run-Level Standardization

Instead of returning a raw `httpx.Response` or an `openai.types.Responses` object, LiteRun parses the entire multi-turn execution loop and collapses it into a single, unified `RunResult` (for synchronous runs) or a stream of `RunStreamEvent` wrappers. Your application logic only ever interacts with these LiteRun objects.

### 9.2 Event-Level Standardization

During a streaming request, an LLM might emit 500 individual Server-Sent Events (SSE). Some are useless heartbeat pings, some are text deltas, and others are fragmented JSON strings representing tool arguments. LiteRun's stream adapters catch this chaos and flatten it into exactly 11 deterministic, strongly-typed state variants (e.g., `stream.start`, `message.output.delta`, `tool.call.done`).

### 9.3 Usage-Level Standardization

Token accounting varies wildly across the industry. Some APIs include cached tokens in the main `input_tokens` count; others separate them. LiteRun enforces a strict `TokenUsage` schema that forces all token metrics into explicit, non-overlapping buckets (cached vs. reasoning vs. standard).

### 9.4 Why This Contract Matters for Enterprise Integration

By guaranteeing these data shapes, LiteRun enables you to:

1. **Build One UI:** You can write a single React/Vue frontend that handles streaming text and tool-execution loading spinners, confident that the backend will always emit the same `stream.type` strings.
2. **Audit Safely:** You can design a rigid SQL database schema to log agent traces (`RunItem`) without worrying about unstructured JSON columns breaking your analytics.
3. **Forecast Costs:** You can build reliable financial dashboards because the token math is fully isolated and standardized.

---

## 10. Run and RunItem Schemas in Detail

When an agent completes a loop, it yields highly structured dataclasses. This section deconstructs exactly what fields exist on those objects and how you should utilize them in your application logic.

### 10.1 The `RunResult` Schema

The `RunResult` is the final artifact returned by `run()` or `arun()`.

* **`output`** (`str`): The final, synthesized textual response from the assistant.
  * *Practical Usage*: This is the payload you return to the end-user or API client as the final answer.

* **`new_items`** (`list[RunItem]`): An ordered list of all normalized events that occurred during this specific run (e.g., the model spoke, a tool was called, the tool returned a result, the model spoke again).
  * *Practical Usage*: Serialize this list and save it to your database for observability, auditing, and debugging. It acts as the exact historical trace of the run.

* **`token_usage`** (`TokenUsage | None`): The mathematical accumulation of all tokens consumed across the entire run.
  * *Practical Usage*: Send this to your telemetry system (e.g., Datadog, Grafana) to monitor cost per request.

* **`timing`** (`Timing`): A snapshot containing `start_time`, `end_time`, and an automatically computed `duration` property.
  * *Practical Usage*: Use the `duration` for Service Level Objective (SLO) monitoring to track latency. *(Note: `start_time` is a monotonic clock value, not a wall-clock UTC timestamp. It is designed for measuring durations, not for absolute database timestamping).*

### 10.2 The `RunStreamEvent` Schema

When using `stream()` or `astream()`, LiteRun yields a `RunStreamEvent` for every chunk of data received over the network.

* **`event`** (`StreamEvent`): The normalized delta event itself (e.g., a `MessageOutputStreamDelta` containing three characters of text).
* **`output`** (`str`): A rolling, cumulative string buffer of all textual output received *up to this exact point in time*.
* *Practical Usage*: If your UI architecture prefers receiving the entire updated string rather than appending tiny deltas manually, you can just overwrite the UI state with this value.


* **`token_usage`** (`TokenUsage | None`): A point-in-time cumulative usage snapshot. (Note: OpenAI typically only emits usage metrics on the final `response.completed` event, so this will often be `None` until the stream concludes).
* **`timing`** (`Timing`): A timing snapshot at the moment this specific event was yielded.

### 10.3 The `RunItem` (Trace Schema) in Depth

The `new_items` list inside a `RunResult` is not just a loose list of text strings. It is a precise, normalized trace of the execution loop built from `RunItem` dataclasses. Each item represents a distinct semantic step in the chain of thought.

#### Shared Base Fields

Every `RunItem` subclass contains these foundational fields:

* **`type`** (`str`): The specific string discriminant (e.g., `"message.output.item"`). Useful for branching logic during serialization.
* **`id`** (`str | None`): The unique ID assigned to this item by the provider, if available.
* **`role`** (`str | None`): The semantic role (`assistant`, `tool_call`, `tool_output`).
* **`raw_item`** (`Any | None`): The original, unaltered object returned by the underlying SDK (e.g., the raw `openai.types.chat.ChatCompletionMessageToolCall`).
* *Practical Usage*: This is your ultimate debug tool. If LiteRun's normalization drops an obscure, brand-new metadata field that OpenAI just released, you can always recover it from `raw_item`.

* **`token_usage`** (`TokenUsage | None`): Token usage attributable to this specific step, if the provider isolates it.

#### Specific Item Variants

**1. `MessageOutputItem`**
Represents a block of textual output generated by the LLM.

* `type`: `"message.output.item"`
* `role`: `"assistant"`
* `content`: The generated string.

**2. `ToolCallItem`**
Represents the LLM explicitly pausing generation to request a function execution.

* `type`: `"tool.call.item"`
* `role`: `"tool_call"`
* `call_id`: The unique identifier for this specific tool execution request.
* `name`: The name of the Python function to execute.
* `arguments`: A parsed Python dictionary containing the exact parameters the model hallucinated for the tool.

**3. `ToolCallOutputItem`**
Represents the data returned by your local Python function, injected back into the conversation history.

* `type`: `"tool.output.item"`
* `role`: `"tool_output"`
* `call_id`: Must identically match the `call_id` of the `ToolCallItem` that triggered it.
* `result`: The string or dictionary output generated by your function.

**4. `ReasoningItem`**
Represents internal "thought processes" exposed by advanced reasoning models.

* `type`: `"reasoning.item"`
* `role`: `"assistant"`
* `summary`: A readable summary of what the model was thinking about.
* `signature`: A cryptographic signature provided by the API, required to safely replay this reasoning block in future conversational turns.

#### Example Trace Visualization

If you asked an agent, "What is the weather in Paris?", the `RunResult.new_items` trace would logically look like this:

```python
[
    MessageOutputItem(
        type="message.output.item", 
        content="Let me check the current weather data for you."
    ),
    ToolCallItem(
        type="tool.call.item", 
        call_id="call_abc123", 
        name="get_weather", 
        arguments={"location": "Paris"}
    ),
    ToolCallOutputItem(
        type="tool.output.item", 
        call_id="call_abc123", 
        name="get_weather", 
        result="The temperature in Paris is 22°C."
    ),
    MessageOutputItem(
        type="message.output.item", 
        content="The current weather in Paris is 22°C."
    )
]
```

This structured trace enables you to rebuild exact historical context, prove to auditors exactly what data was passed to local tools, and debug conversational loops with surgical precision.

---

# Part 4: Streaming, Accounting & Internals

This section delves into the mechanical core of LiteRun. It explains how the framework handles the chaotic nature of real-time network streams, how it normalizes the notoriously complex token economics of modern LLMs, and how it strictly maps your data to OpenAI’s proprietary schemas.

---

## 11. Streaming Event Schemas in Detail

When you invoke a model using `.stream()` or `.astream()`, the provider does not send a single, neatly packaged response. Instead, it holds a long-lived HTTP connection open and rapidly fires Server-Sent Events (SSEs). Some chunks contain a single letter, some contain fragments of JSON for a tool call, and others are simply blank heartbeats.

If you attempt to parse this raw stream manually, your application logic will quickly become an unmaintainable web of `if/else` statements. LiteRun abstracts this chaos into a finite state machine, yielding predictable, strictly typed `StreamEvent` objects.

### 11.1 Shared Stream Event Fields

Every event yielded by the stream inherits from a common base contract. Regardless of whether the event represents text, tool data, or an error, you can always rely on the following fields:

* **`type`** (`str`): The exact string literal defining the event (e.g., `"message.output.delta"`).
* **`id`** (`str | None`): The provider-assigned identifier for the current generation item or interaction, if available.
* **`raw_event`** (`Any | None`): The raw, unaltered chunk emitted by the underlying provider SDK. **This is your escape hatch.** If you need to access a brand-new, experimental field that OpenAI just added to their stream, you can extract it from `raw_event`.
* **`token_usage`** (`TokenUsage | None`): Cumulative usage up to this point in the stream. Note that OpenAI typically only populates this on the final termination event.

### 11.2 Text Message Events

These events are responsible for delivering the actual conversational output from the assistant.

* **`message.output.delta`**
  * **Payload**: `delta` (`str`)
  * **Purpose**: Emitted continuously as the model generates textual output. This is the event you will listen for to power a "typewriter" effect in a chat UI.

* **`message.output.done`**
  * **Payload**: `output` (`str`)
  * **Purpose**: Emitted when the model has finished generating a contiguous block of text. It contains the fully assembled string of all preceding deltas.

### 11.3 Tool Execution Events

When the model decides to invoke a tool, it stops streaming conversational text and begins streaming a JSON object representing the function's arguments.

* **`tool.call.delta`**
  * **Payload**: `call_id` (`str`), `name` (`str`), `delta` (`str | dict`)
  * **Purpose**: Carries fragments of the JSON arguments (e.g., `{"locati`, `on": "Pa`, `ris"}`). Useful if you want to show the user a live, updating code block of what the model is trying to execute.

* **`tool.call.done`**
  * **Payload**: `call_id` (`str`), `name` (`str`), `output` (`dict`)
  * **Purpose**: Emitted the moment the JSON stream is complete. LiteRun has successfully parsed the fragments into a valid Python dictionary. This signals that the LLM has finished its request.

* **`tool.output.done`**
  * **Payload**: `id` (`str`, set to the originating `call_id`), `name` (`str`), `output` (`str | dict`)
  * **Purpose**: **This is emitted by LiteRun, not the LLM.** After `tool.call.done`, LiteRun intercepts the loop, executes your local Python function, and emits this event to let your UI know that the local execution succeeded and the result is being sent back to the network.

### 11.4 Reasoning Events

Advanced models (like OpenAI's `o-series` and `gpt-5-series`) expose their internal thought processes before outputting an answer.

* **`reasoning.delta`**: Contains fragments of the model's internal thought summary.
* **`reasoning.done`**: Contains the complete, assembled thought process. You can use this to render a collapsible "Thought Process" UI component, similar to the native ChatGPT interface.

### 11.5 Lifecycle and Fallback Events

* **`stream.start`**: Emitted exactly once at the beginning of the generator, confirming the network connection is established.
* **`stream.end`**: The definitive termination signal. No further events will be yielded.
* **`stream.error`**: Wraps network drops or parsing failures into a standardized payload.
* **`other.event`**: Emitted when the provider sends an event that does not map to text, tools, or reasoning. It prevents data loss without forcing unsafe assumptions into the main event loop.

### 11.6 End Semantics and Multi-Turn Loops

A common pitfall in agent architecture is prematurely closing the UI stream when a tool is called.

If the model outputs a tool call, the provider's API naturally sends a "completed" signal for that specific network request. However, **LiteRun suppresses `stream.end` during tool calls.** Because LiteRun manages the multi-turn loop, it knows it must execute the tool and initiate *another* network request.

LiteRun only yields the `stream.end` event when it determines that the model has provided a final, text-complete turn with no further tool calls pending. This guarantees that your application logic can safely wait for `stream.end` before tearing down UI loading states.

When streaming, LiteRun categorizes chaotic network chunks into a clean, predictable lifecycle. Below is the quick-reference cheat sheet for the normalized `event.type` strings yielded by the stream, and where to find their data:

| Normalized Event Type | Description | Primary Payload Field |
| :--- | :--- | :--- |
| `stream.start` | Network connection opened successfully. | N/A |
| `message.output.delta` | A chunk of generated assistant text. | `event.delta` (str) |
| `message.output.done` | Finalized, contiguous block of text. | `event.output` (str) |
| `tool.call.delta` | A fragment/snapshot of tool arguments. | `event.delta` (str / dict) |
| `tool.call.done` | Model finished requesting a tool. | `event.output` (dict) |
| `tool.output.done` | Local Python function finished executing. | `event.id` (call_id), `event.output` (str / dict) |
| `reasoning.delta` | A fragment of internal model thought. | `event.delta` (str) |
| `stream.end` | Absolute termination of the multi-turn loop. | N/A |

---

## 12. Token Usage Model in Detail

Usage accounting is one of the most frustrating aspects of building production AI systems. Providers frequently change how they report tokens. For instance, OpenAI might report an `input_tokens` total of 10,000, but note in a nested dictionary that 9,000 of those were "cached." If you bill your users based on the raw `input_tokens` number, you will overcharge them massively, as cached tokens cost a fraction of standard tokens.

LiteRun’s `TokenUsage` dataclass acts as a strict accounting ledger, enforcing un-blended, mutually exclusive buckets.

### 12.1 Conceptual Meaning of Each Field

* **`input_tokens`**: The count of standard, un-cached tokens sent in the prompt. (Billable at the standard input rate).
* **`output_tokens`**: The count of standard text tokens generated by the model. (Billable at the standard output rate).
* **`cached_read_tokens`**: Tokens that were served directly from the provider's prompt cache. (Discounted rate).
* **`cached_write_tokens`**: Tokens that were newly written into the cache during this request.
* **`reasoning_tokens`**: Tokens burned by the model's internal thought process. These are not visible in the final output text but are billable as output tokens.
* **`tool_use_tokens`**: Overhead tokens consumed by the provider to format and understand tool schemas.
* **`total_tokens`**: Provider-reported total when available; otherwise a runtime-resolved aggregate used by LiteRun.

### 12.2 The OpenAI Mapping Rationale (Unblending)

OpenAI's native API reports token totals as blended figures.
For example, a raw OpenAI response might look like this:

* `usage.input_tokens`: 500
* `usage.input_tokens_details.cached_tokens`: 400

If you simply read `input_tokens`, you miss the nuance. LiteRun **deliberately separates these buckets** to prevent hidden costs.

When LiteRun processes the above example, it populates the `TokenUsage` object like this:

* `input_tokens`: 100 *(LiteRun subtracted the cached tokens to give you the true standard input)*
* `cached_read_tokens`: 400

This deliberate unblending means your downstream analytics and billing dashboards can simply multiply each LiteRun field by its respective pricing tier without fear of double-counting.

### 12.3 The Timing Schema

Included in both `RunResult` and `RunStreamEvent` is the `Timing` object.

* **`start_time`** / **`end_time`**: These are generated using Python's `time.perf_counter()`. They represent highly precise monotonic clock values, immune to system clock updates.
* **`duration`**: Computed as `end_time - start_time`.

**Guidance**: Do not treat `start_time` as a wall-clock UTC timestamp for your database (`created_at`). It is strictly a relative measurement tool for calculating `duration` to monitor latency SLOs. If you need a database timestamp, generate one at the application boundary (e.g., `datetime.now(UTC)`).

---

## 13. OpenAI-Specific Behavior and Mapping

While the `Agent` keeps your application abstracted from the provider, it is vital for senior engineers to understand exactly how the `ChatOpenAI` client translates your canonical data into the specific expectations of the OpenAI Responses API.

### 13.1 Message Normalization

When you call `agent.run(prompt_template)`, the `ChatOpenAI.normalize_messages()` method unpacks the `PromptTemplate` and transforms it.

OpenAI's latest schema requires highly nested content blocks. LiteRun maps them as follows:

* Canonical `TextBlock` in a `user` role:  `{"type": "input_text", "text": "..."}`
* Canonical `TextBlock` in an `assistant` role:  `{"type": "output_text", "text": "..."}`
* Canonical `ToolCallBlock`:  `{"type": "function_call", "call_id": "...", "name": "...", "arguments": "{...}"}`
* Canonical `ToolOutputBlock`:  `{"type": "function_call_output", "call_id": "...", "output": "..."}`

If LiteRun encounters a block that the OpenAI adapter does not support, it will raise an `AgentSerializationError` rather than silently dropping the data.

### 13.2 Reasoning Replay Expectations

If you are using an `o-series` or `gpt-5-series` model, the model may output a `ReasoningBlock`. If you want to continue the conversation, you must "replay" this thought process back to the model in the next turn to maintain context.

To successfully serialize a `ReasoningBlock` back to OpenAI, LiteRun requires:

1. **`reasoning_id`**: The unique ID of the thought block.
2. **`summary`**: The plaintext summary of the thought.
3. **`signature`** *(optional)*: If present, this is mapped to OpenAI's `encrypted_content`.

If `reasoning_id` or `summary` is missing, serialization fails with `AgentSerializationError`.

### 13.3 Stream Tool-Call Correlation

One of the most complex engineering feats inside the `OpenAIStreamAdapter` is tool-call correlation.

When OpenAI streams a tool call, it often separates the metadata (the tool name and ID) from the argument deltas (the JSON string). It emits a `response.output_item.added` event to define the ID, followed by dozens of `response.function_call_arguments.delta` events.

LiteRun maintains a highly optimized **internal registry dictionary** tracked by `item_id`.
As fragments arrive over the network, LiteRun buffers the JSON string in memory, tied strictly to that `item_id`. It only attempts to parse the JSON and emit a `tool.call.done` event when OpenAI explicitly transmits the `response.function_call_arguments.done` signal. This prevents the framework from attempting to parse malformed, half-streamed JSON, guaranteeing absolute stability in your tool execution loop.

---

# Part 5: Observability & Best Practices

When deploying agentic workflows to production, the framework's ability to gracefully handle failures and emit clear, structured telemetry is just as important as its ability to generate text. This final section covers LiteRun's error hierarchy, its structured logging subsystem, and architectural best practices to ensure your integrations remain stable at scale.

---

## 14. Error Model: What Fails, Why, and Where to Fix

Agentic loops involve constant network traversal, dynamic schema generation, and unpredictable model outputs. Failures are inevitable. If a framework throws generic `Exception` or `KeyError` messages, debugging becomes a nightmare.

LiteRun maps all internal and provider SDK exceptions to a unified, predictable hierarchy rooted in `LiteRunError`. This allows you to write defensive application code that can catch, triage, and potentially retry specific failure categories without having to import `openai.OpenAIError`.

### 14.1 Agent-Level Errors

These errors originate from within the LiteRun orchestration loop or prompt builder. They usually indicate a structural flaw in how your code is interacting with the framework.

* **`AgentInputError`**
  * **Cause**: The top-level `messages` value is an unsupported type (for example, `int` instead of `str | list[dict] | PromptTemplate`).
  * **Fix**: Ensure your input is strictly a `str`, provider-native `list[dict]`, or a `PromptTemplate`.
  * **Note**: Structural errors *inside* a `list[dict]` payload typically surface as provider-side request errors (`InvalidRequestError`), because raw list payloads are intentionally treated as expert pass-through mode.

* **`AgentSerializationError`**
  * **Cause**: LiteRun attempted to translate your canonical `PromptTemplate` blocks into the provider's API payload, but a strict invariant was missing.
  * **Example**: Attempting to append an OpenAI `ReasoningBlock` that is missing `reasoning_id` or `summary`.
  * **Fix**: Review your `PromptTemplate` construction to ensure it complies with the specific provider's structural requirements (as outlined in Section 13).

* **`AgentParsingError`**
  * **Cause**: The provider returned an API response or SSE chunk that violates its own documented schema, preventing LiteRun from normalizing it into a `RunItem` or `StreamEvent`.
  * **Fix**: Inspect the raw event/response shape in your logs. This is often a transient provider-side bug, but it may require updating the LiteRun adapter if the provider permanently changed their API schema.

* **`AgentExecutionError`**
  * **Cause**: An unexpected internal state exception occurred within the `while` loop of the runner.
  * **Fix**: Inspect the deep traceback and context payload.

* **`AgentMaxIterationsError`**
  * **Cause**: The model entered a death spiral—repeatedly calling tools unsuccessfully without ever outputting a final, user-facing textual response—triggering the `max_iterations` safety cap.
  * **Fix**: Improve your tool descriptions (they might be ambiguous), harden your tool return values (the model might not understand the error string the tool is returning), or force the model to answer using a stricter `tool_choice` policy.

### 14.2 Tool Errors

These errors represent a breakdown at the boundary between the LLM and your local Python environment.

* **`AgentToolCallError`**
  * **Cause**: The LLM requested a tool execution, but the arguments it provided were either malformed JSON or failed to pass the strict Pydantic validation defined in your `input_schema`.
  * **Fix**: Simplify the `input_schema`, or enable the `strict=True` Structured Outputs parameter to force the model to adhere to the exact JSON structure.

* **`AgentToolExecutionError`**
  * **Cause**: Your underlying Python function raised an unhandled exception (e.g., a database connection dropped during a query), or the function's output failed the `output_schema` validation.
  * **Fix**: Harden your tool's internal `try/except` logic. Tools should ideally return graceful error strings (e.g., `"Error: User not found"`) back to the model, rather than letting Python exceptions crash the entire agent loop.

### 14.3 Provider Errors

These errors are remapped directly from the underlying API SDK.

* **`AuthenticationError`**: Your API key is missing, expired, or lacks the correct project scope.
* **`RateLimitError`**: You have exceeded the provider's tokens-per-minute (TPM) or requests-per-minute (RPM) quotas.
* *Note*: This error class sets `retryable_error = True`, signaling to your application that it should apply exponential backoff.

* **`InvalidRequestError`**: The payload sent to the API was structurally rejected (a `400 Bad Request`).
* **`APIConnectionError`**: A network-level failure, DNS resolution error, or read timeout occurred. Also marked as `retryable_error = True`.
* **`APIStatusError` / `LLMError`**: A generic `500 Internal Server Error` or `503 Bad Gateway` originating from the provider's infrastructure.

### 14.4 Triage Order: Where to Look First

When an incident pages you at 2:00 AM, follow this triage path:

1. **Check the Exception Class**: Is it an `AgentToolExecutionError` (your code) or an `APIStatusError` (their servers)?
2. **Check the `error_code`**: Look at the normalized string (e.g., `"api.rate_limited"`) to route the alert to the correct team.
3. **Inspect the `context` Payload**: This dictionary contains the exact model name, provider, and execution state when the crash occurred.
4. **Check `cause` Details**: If LiteRun wrapped a lower-level exception, the original stack trace is preserved here.
5. **Examine the `raw_event`**: If streaming failed mid-chunk, the `raw_event` property on the last yielded `StreamEvent` contains the exact bytes that broke the parser.

LiteRun maps all internal and provider SDK exceptions to a unified hierarchy. Use this table to quickly route errors to the correct layer of your application.

| Exception Class | Origin | Typical Cause | `retryable` |
| :--- | :--- | :--- | :--- |
| `AgentInputError` | Application | Passed an invalid data type to `.run()`. | `False` |
| `AgentSerializationError` | Application | Invalid canonical block state (e.g., missing `reasoning_id`/`summary`). | `False` |
| `AgentToolExecutionError`| Application | Your local Python tool function crashed. | `False` |
| `AgentMaxIterationsError`| Agent Loop | Model stuck in an infinite tool-calling loop. | `False` |
| `AuthenticationError` | Provider | Invalid or missing API key. | `False` |
| `RateLimitError` | Provider | Exceeded tokens/requests per minute quota. | **`True`** |
| `APIConnectionError` | Network | DNS failure or network timeout. | **`True`** |
| `APIStatusError` | Provider | Provider's servers are down (500 / 503). | `False` |

---

## 15. Logging Model: How to Observe and Debug Runs

Printing `str(exception)` to standard output is insufficient for an agentic backend. LiteRun ships with `AgentLogger`, a utility designed specifically to emit highly structured, machine-readable logs.

### 15.1 Structured Payload Fields

Whenever LiteRun catches or wraps a critical exception, it generates a normalized dictionary payload. This payload is injected into the standard Python `logging` module under the `extra={"literun_error": ...}` parameter.

The payload guarantees the following keys:

* `error_type`: The literal class name of the exception (e.g., `"AgentMaxIterationsError"`).
* `message`: The human-readable error description.
* `error_code`: A stable enumeration string (e.g., `"agent.max_iterations"`).
* `retryable`: A boolean indicating if the network/application should safely re-attempt the run.
* `context`: A rich dictionary. For example, if a tool fails, the context will contain `{"tool_name": "fetch_data", "execution_mode": "async"}`.
* `cause_type` / `cause_message`: Information extracted from the underlying nested `__cause__` exception.

### 15.2 Why Structured Logging Matters

If you pipe your logs into a centralized observability platform like Datadog, Splunk, or ELK, structured logging changes how you operate:

* **Alerting**: You can set a PagerDuty alert specifically for `error_code: "api.auth.failed"` while ignoring spikes in `"tool.execution.failed"`.
* **Correlation**: By capturing the `tool_name` inside the `context` dictionary, you can instantly run a query to see that 95% of your agent failures are originating from one specific, flaky Python tool function.

### 15.3 Practical Usage at the Application Boundary

To take full advantage of this, ensure your top-level application framework (like a FastAPI route handler) captures these exceptions and injects its own trace IDs before logging them:

```python
import logging
from fastapi import APIRouter, HTTPException
from literun.errors import LiteRunError

router = APIRouter()
logger = logging.getLogger("my_app_logger")

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        result = await agent.arun(request.prompt)
        return {"response": result.output}
        
    except LiteRunError as exc:
        # Extract LiteRun's structured context
        log_payload = {
            "literun_code": exc.error_code.value,
            "literun_context": exc.context,
            "user_id": request.user_id, # Your application's context
            "trace_id": request.trace_id
        }
        
        logger.error(f"Agent execution failed: {exc}", extra=log_payload)
        
        if exc.retryable_error:
            raise HTTPException(status_code=503, detail="Service temporarily unavailable.")
        raise HTTPException(status_code=400, detail="Unable to process request.")
```

---

## 16. Design Guidance: Correct vs Incorrect Usage Patterns

Mastering LiteRun requires understanding the architectural boundaries of the framework. Here are concrete patterns to adopt and anti-patterns to avoid.

### 16.1 Message Input Choice

* **Correct**: Use a plain `str` for zero-shot prompts.
* **Correct**: Use `PromptTemplate` for managing all multi-turn conversation state to leverage Pydantic's strict type-checking and structural validation.
* **Correct**: Use `list[dict]` only if you have deep, expert knowledge of the provider's JSON schema and need to bypass LiteRun's canonical abstractions to use experimental API features.
* **Incorrect**: Mixing partially-canonical and partially-provider-native formats in a single payload. If you use `list[dict]`, the entire payload must conform to the provider's native schema.

### 16.2 Tool Design

* **Correct**: Design tools with extremely narrow scopes. Instead of one massive `manage_database` tool, create `read_user_record` and `update_user_email` tools.
* **Correct**: Write deterministic, concise docstrings for `description`.
* **Correct**: Inject ephemeral application state (tokens, session IDs) exclusively via the `ToolRuntime` context parameter.
* **Incorrect**: Defining broad, generic JSON argument schemas (`args: dict`). This encourages the LLM to hallucinate keys. Force the model to conform to explicit fields using `input_schema`.
* **Incorrect**: Leaking system secrets by defining an `api_key: str` argument that the LLM is expected to provide.

### 16.3 Streaming UI Design

* **Correct**: Accumulate the final UI text progressively by listening only to `message.output.delta` events.
* **Correct**: Provide user feedback (*"Agent is thinking..."* or *"Calling database..."*) by intercepting `tool.call.delta` and `tool.output.done` events.
* **Correct**: Use `other.event` as a signal to log unrecognized provider chunks to your telemetry system so you are aware when the provider introduces new API behaviors.
* **Incorrect**: Closing the user's websocket or tearing down the UI on intermediate provider terminal signals. You must wait for LiteRun's normalized terminal semantics (including `stream.end`) to confirm loop completion.

### 16.4 Usage Analytics

* **Correct**: Report billing metrics using the normalized `output_tokens` and `input_tokens` fields, while keeping `cached_read_tokens` separate.
* **Correct**: Monitor `reasoning_tokens` continuously if using `o-series` or `gpt-5-series` models, as reasoning loops can silently consume massive amounts of budget if left unchecked.
* **Incorrect**: Treating LiteRun's normalized `input_tokens` as the raw, total provider input count for billing algorithms without adding the cached buckets back in.

---

## 17. Future Work

LiteRun’s architecture is fundamentally provider-agnostic. While this branch of the documentation strictly governs the OpenAI Responses API implementation, the underlying `BaseLLM` and `BaseAdapter` contracts are already capable of housing other major providers.

Planned expansions for subsequent major releases include:

* **Gemini Provider Support**: Full documentation for the Google GenAI `Interactions API`, including native support for Gemini 2.0 thought blocks, search grounding tools, and multimodal input schemas.
* **Anthropic Provider Support**: Full documentation for the Anthropic `Messages API`, detailing cumulative cache-read stream logic, extended thinking budgets, and Claude's specific `tool_use` block invariants.
* **The Multi-Provider Migration Guide**: Cookbook patterns for seamlessly hot-swapping `ChatOpenAI` for `ChatAnthropic` in production environments using identical canonical `PromptTemplates` and Pydantic `Tool` registries.

---

## 18. Closing Notes

LiteRun was built to keep your runtime predictable without burying the hard edges of AI integration beneath unmanageable layers of "magic."

If you adopt a single principle from this developer guide, let it be this: **Treat the normalized schemas (`RunResult`, `RunStreamEvent`, `RunItem`, `TokenUsage`) as your unbreakable integration contract.** Write your databases, your front-end components, and your billing analytics strictly against these objects. Treat provider-native payloads and raw events as advanced detail available only when deep, surgical debugging is necessary.

By respecting that boundary, you gain absolute stability for your product code, while preserving the transparency required to debug and scale your agentic systems to production.

### Found a Documentation Issue?

Please report it here: [New Issue](https://github.com/kaustubh-tr/literun/issues/new)  
When reporting, include the section heading, what is incorrect, expected behavior, and a minimal example if possible.
