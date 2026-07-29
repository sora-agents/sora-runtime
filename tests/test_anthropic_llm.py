"""AnthropicLLMClient's usage extraction (the provider-native token accounting the instrumented
client surfaces). Gated on the optional ``[llm]`` extra — the SDK import lives under adapters."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")

from sora.adapters.anthropic_llm import _usage_of  # noqa: E402
from sora.llm import LLMUsage  # noqa: E402


def test_usage_of_reads_the_token_block_and_carries_the_answer_length() -> None:
    message = SimpleNamespace(usage=SimpleNamespace(input_tokens=1200, output_tokens=800))
    usage = _usage_of(message, answer_chars=40)
    assert usage == LLMUsage(1200, 800, answer_chars=40)
    assert usage.thinking_share == 790 / 800  # output minus the ~10-token answer


def test_usage_of_tolerates_a_missing_or_partial_usage_block() -> None:
    # Instrumentation must never break a call: a message with no `usage` (or a partial one) degrades
    # to zeros rather than raising, so a metering gap is silent, not fatal.
    assert _usage_of(SimpleNamespace(), answer_chars=7) == LLMUsage(0, 0, answer_chars=7)
    partial = SimpleNamespace(usage=SimpleNamespace(input_tokens=42))
    assert _usage_of(partial, answer_chars=0).input_tokens == 42
