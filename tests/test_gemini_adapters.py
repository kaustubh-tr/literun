import os
import sys
import unittest
from types import SimpleNamespace

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun.events import OtherStreamEvent, StreamEndEvent
from literun.providers.gemini.responses import GeminiResponseAdapter
from literun.providers.gemini.streams import GeminiStreamAdapter


class TestGeminiResponseAdapter(unittest.TestCase):
    def test_extract_token_usage_independent_buckets(self):
        usage = SimpleNamespace(
            total_input_tokens=120,
            total_output_tokens=40,
            total_tokens=200,
            total_cached_tokens=25,
            total_thought_tokens=30,
            total_tool_use_tokens=10,
        )
        response = SimpleNamespace(usage=usage)

        adapter = GeminiResponseAdapter()
        token_usage = adapter.extract_token_usage(response)
        self.assertEqual(token_usage.input_tokens, 95)
        self.assertEqual(token_usage.cached_read_tokens, 25)
        self.assertEqual(token_usage.output_tokens, 40)
        self.assertEqual(token_usage.reasoning_tokens, 30)
        self.assertEqual(token_usage.tool_use_tokens, 10)
        self.assertEqual(token_usage.total_tokens, 200)


class TestGeminiStreamAdapter(unittest.TestCase):
    def test_terminal_event_with_tool_turn_is_other_event(self):
        adapter = GeminiStreamAdapter()
        adapter._process_chunk(SimpleNamespace(event_type="interaction.start", interaction_id="i1"))
        adapter._process_chunk(
            SimpleNamespace(
                event_type="content.start",
                index=0,
                content=SimpleNamespace(type="function_call", id="c1", name="get_weather", arguments={}),
            )
        )
        completed = adapter._process_chunk(
            SimpleNamespace(
                event_type="interaction.complete",
                interaction_id="i1",
                interaction=SimpleNamespace(usage=None),
            )
        )
        self.assertIsInstance(completed, OtherStreamEvent)

    def test_terminal_event_with_text_turn_is_stream_end(self):
        adapter = GeminiStreamAdapter()
        adapter._process_chunk(SimpleNamespace(event_type="interaction.start", interaction_id="i2"))
        adapter._process_chunk(
            SimpleNamespace(event_type="content.start", index=0, content=SimpleNamespace(type="text"))
        )
        adapter._process_chunk(
            SimpleNamespace(
                event_type="content.delta",
                event_id="e1",
                index=0,
                delta=SimpleNamespace(type="text", text="done"),
            )
        )
        adapter._process_chunk(SimpleNamespace(event_type="content.stop", event_id="e2", index=0))
        completed = adapter._process_chunk(
            SimpleNamespace(
                event_type="interaction.complete",
                interaction_id="i2",
                interaction=SimpleNamespace(usage=None),
            )
        )
        self.assertIsInstance(completed, StreamEndEvent)

    def test_extract_token_usage_from_interaction_complete_only(self):
        adapter = GeminiStreamAdapter()
        usage = SimpleNamespace(
            total_input_tokens=90,
            total_output_tokens=70,
            total_cached_tokens=20,
            total_thought_tokens=10,
            total_tool_use_tokens=5,
            total_tokens=195,
        )
        chunk = SimpleNamespace(
            event_type="interaction.complete",
            interaction_id="i3",
            interaction=SimpleNamespace(usage=usage),
        )
        token_usage = adapter.extract_token_usage(chunk)
        self.assertEqual(token_usage.input_tokens, 70)
        self.assertEqual(token_usage.cached_read_tokens, 20)
        self.assertEqual(token_usage.output_tokens, 70)
        self.assertEqual(token_usage.reasoning_tokens, 10)
        self.assertEqual(token_usage.tool_use_tokens, 5)
        self.assertEqual(token_usage.total_tokens, 195)


if __name__ == "__main__":
    unittest.main()
