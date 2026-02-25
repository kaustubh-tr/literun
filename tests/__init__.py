"""Test package."""

from __future__ import annotations

import logging
import warnings

# Keep unittest output clean; tests assert behavior directly.
logging.disable(logging.CRITICAL)

# Gemini SDK emits an expected experimental warning during client init in tests.
warnings.filterwarnings(
    "ignore",
    message="Interactions usage is experimental and may change in future versions.",
    category=UserWarning,
    module=r"google\.genai\.client",
)
