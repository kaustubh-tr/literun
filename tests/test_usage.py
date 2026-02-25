import os
import sys
import unittest

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun.usage import TokenUsage


class TestTokenUsage(unittest.TestCase):
    def test_resolved_total_uses_provider_total_when_present(self):
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cached_read_tokens=10,
            reasoning_tokens=5,
            total_tokens=1234,
        )
        self.assertEqual(usage.resolved_total_tokens, 1234)
        self.assertEqual(usage.dict()["total_tokens"], 1234)

    def test_resolved_total_fallback_sum_when_total_missing(self):
        usage = TokenUsage(
            input_tokens=80,
            output_tokens=20,
            cached_read_tokens=10,
            cached_write_tokens=2,
            reasoning_tokens=5,
            tool_use_tokens=3,
            total_tokens=None,
        )
        self.assertEqual(usage.resolved_total_tokens, 120)
        self.assertEqual(usage.dict()["total_tokens"], 120)

    def test_addition_merges_optional_fields(self):
        a = TokenUsage(
            input_tokens=10,
            output_tokens=4,
            cached_read_tokens=1,
            reasoning_tokens=None,
            total_tokens=14,
        )
        b = TokenUsage(
            input_tokens=7,
            output_tokens=3,
            cached_read_tokens=2,
            reasoning_tokens=5,
            total_tokens=10,
        )
        merged = a + b
        self.assertEqual(merged.input_tokens, 17)
        self.assertEqual(merged.output_tokens, 7)
        self.assertEqual(merged.cached_read_tokens, 3)
        self.assertEqual(merged.reasoning_tokens, 5)
        self.assertEqual(merged.total_tokens, 24)

    def test_addition_uses_fallback_when_provider_total_missing(self):
        a = TokenUsage(input_tokens=5, output_tokens=5, total_tokens=None)
        b = TokenUsage(input_tokens=1, output_tokens=2, total_tokens=None)
        merged = a + b
        self.assertEqual(merged.total_tokens, 13)


if __name__ == "__main__":
    unittest.main()
