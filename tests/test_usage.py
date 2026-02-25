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
        # Both sides have explicit provider totals; result should be their sum.
        self.assertEqual(merged.total_tokens, 24)
        self.assertEqual(merged.resolved_total_tokens, 24)

    def test_addition_uses_fallback_when_provider_total_missing(self):
        # Neither side has a provider-reported total; total is computed from buckets.
        a = TokenUsage(
            input_tokens=5,
            output_tokens=5,
            cached_read_tokens=2,
            reasoning_tokens=3,
            total_tokens=None,
        )
        b = TokenUsage(
            input_tokens=1,
            output_tokens=2,
            cached_read_tokens=1,
            total_tokens=None,
        )
        merged = a + b
        # resolved: (5+1) + (5+2) + (2+1) + (3+0) = 6 + 7 + 3 + 3 = 19
        self.assertEqual(merged.total_tokens, 19)
        self.assertEqual(merged.resolved_total_tokens, 19)

    def test_addition_mixed_provider_and_fallback(self):
        # One side has a provider total, the other does not; resolved is always used.
        a = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
        b = TokenUsage(input_tokens=3, output_tokens=2, total_tokens=None)
        merged = a + b
        # a.resolved = 15 (provider), b.resolved = 3+2 = 5 → sum = 20
        self.assertEqual(merged.total_tokens, 20)
        self.assertEqual(merged.resolved_total_tokens, 20)


if __name__ == "__main__":
    unittest.main()
