import os
import sys
import unittest
from types import SimpleNamespace

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun.events import OtherStreamEvent, StreamEndEvent, ToolCallStreamDone, ToolCallStreamDelta
from literun.providers.openai.responses import OpenAIResponseAdapter
from literun.providers.openai.streams import OpenAIStreamAdapter


class TestOpenAIResponseAdapter(unittest.TestCase):
    def test_extract_token_usage_independent_buckets(self):
        usage = SimpleNamespace(
            input_tokens=100,
            output_tokens=80,
            total_tokens=180,
            input_tokens_details=SimpleNamespace(cached_tokens=30),
            output_tokens_details=SimpleNamespace(reasoning_tokens=20),
        )
        response = SimpleNamespace(usage=usage)

        adapter = OpenAIResponseAdapter()
        token_usage = adapter.extract_token_usage(response)
        self.assertEqual(token_usage.input_tokens, 70)
        self.assertEqual(token_usage.cached_read_tokens, 30)
        self.assertEqual(token_usage.output_tokens, 60)
        self.assertEqual(token_usage.reasoning_tokens, 20)
        self.assertEqual(token_usage.total_tokens, 180)

    def test_build_tool_call_message_skips_invalid_reasoning(self):
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="reasoning",
                    id=None,
                    summary=[{"type": "summary_text", "text": "S"}],
                    encrypted_content="enc",
                ),
                SimpleNamespace(
                    type="reasoning",
                    id="r1",
                    summary=[{"type": "summary_text", "text": "S"}],
                    encrypted_content="enc",
                ),
            ]
        )
        adapter = OpenAIResponseAdapter()
        messages = adapter.build_tool_call_message(response)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["type"], "reasoning")
        self.assertEqual(messages[0]["id"], "r1")


class TestOpenAIStreamAdapter(unittest.TestCase):
    def test_tool_call_correlation_delta_and_done(self):
        adapter = OpenAIStreamAdapter()

        start = adapter._process_chunk(
            SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_1"))
        )
        self.assertEqual(start.type, "stream.start")

        added = SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(
                type="function_call",
                id="item_1",
                call_id="call_1",
                name="get_weather",
            ),
        )
        self.assertIsNone(adapter._process_chunk(added))

        delta_event = adapter._process_chunk(
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id="item_1",
                delta='{"location":"Par',
            )
        )
        self.assertIsInstance(delta_event, ToolCallStreamDelta)
        self.assertEqual(delta_event.call_id, "call_1")

        done_event = adapter._process_chunk(
            SimpleNamespace(
                type="response.function_call_arguments.done",
                item_id="item_1",
                arguments='{"location":"Paris"}',
            )
        )
        self.assertIsInstance(done_event, ToolCallStreamDone)
        self.assertEqual(done_event.call_id, "call_1")
        self.assertEqual(done_event.output, {"location": "Paris"})

    def test_terminal_event_with_tool_turn_is_other_event(self):
        adapter = OpenAIStreamAdapter()
        adapter._process_chunk(
            SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_1"))
        )
        adapter._process_chunk(
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(type="function_call", id="item_1", call_id="call_1", name="x"),
            )
        )
        completed = adapter._process_chunk(
            SimpleNamespace(type="response.completed", response=SimpleNamespace(id="resp_1", usage=None))
        )
        self.assertIsInstance(completed, OtherStreamEvent)

    def test_terminal_event_with_text_turn_is_stream_end(self):
        adapter = OpenAIStreamAdapter()
        adapter._process_chunk(
            SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_2"))
        )
        adapter._process_chunk(
            SimpleNamespace(type="response.output_text.done", item_id="msg_1", text="done text")
        )
        completed = adapter._process_chunk(
            SimpleNamespace(type="response.completed", response=SimpleNamespace(id="resp_2", usage=None))
        )
        self.assertIsInstance(completed, StreamEndEvent)

    def test_extract_token_usage_independent_buckets(self):
        adapter = OpenAIStreamAdapter()
        usage = SimpleNamespace(
            input_tokens=90,
            output_tokens=70,
            total_tokens=160,
            input_tokens_details=SimpleNamespace(cached_tokens=20),
            output_tokens_details=SimpleNamespace(reasoning_tokens=10),
        )
        chunk = SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="resp", usage=usage),
        )
        token_usage = adapter.extract_token_usage(chunk)
        self.assertEqual(token_usage.input_tokens, 70)
        self.assertEqual(token_usage.cached_read_tokens, 20)
        self.assertEqual(token_usage.output_tokens, 60)
        self.assertEqual(token_usage.reasoning_tokens, 10)
        self.assertEqual(token_usage.total_tokens, 160)


if __name__ == "__main__":
    unittest.main()
