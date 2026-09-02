"""AnthropicLLMClient's usage extraction (the provider-native token accounting the instrumented
client surfaces). Gated on the optional ``[llm]`` extra — the SDK import lives under adapters."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("anthropic")

from sora.adapters.anthropic_llm import AnthropicLLMClient, _usage_of  # noqa: E402
from sora.llm import CompletionProfile, CompletionRequest, LLMUsage  # noqa: E402


def test_usage_of_reads_the_token_block_and_carries_the_answer_length() -> None:
    message = SimpleNamespace(usage=SimpleNamespace(input_tokens=1200, output_tokens=800))
    usage = _usage_of(message, answer_chars=40)
    assert usage == LLMUsage(1200, 800, answer_chars=40)
    assert usage.thinking_share == 790 / 800  # output minus the ~10-token answer


def test_usage_of_normalizes_anthropic_cache_counts_into_total_input() -> None:
    message = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=200,
            cache_creation_input_tokens=300,
            cache_read_input_tokens=500,
            output_tokens=80,
        )
    )

    usage = _usage_of(message, answer_chars=12)

    assert usage == LLMUsage(1000, 80, answer_chars=12, cached_input_tokens=500)


def test_usage_of_treats_explicit_zero_cached_input_as_zero() -> None:
    message = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=42,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            output_tokens=0,
        )
    )

    usage = _usage_of(message, answer_chars=0)
    assert usage is not None
    assert usage.cached_input_tokens == 0


def test_usage_of_tolerates_a_missing_or_partial_usage_block() -> None:
    # Missing accounting is unavailable, not an exact zero-token provider round trip.
    assert _usage_of(SimpleNamespace(), answer_chars=7) is None
    partial = SimpleNamespace(usage=SimpleNamespace(input_tokens=42))
    usage = _usage_of(partial, answer_chars=0)
    assert usage is not None
    assert usage.input_tokens == 42
    assert usage.cached_input_tokens is None


class _FakeMessageStream:
    def __init__(self, message: SimpleNamespace) -> None:
        self.message = message

    async def __aenter__(self) -> _FakeMessageStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def get_final_message(self) -> SimpleNamespace:
        return self.message


class _FakeMessages:
    def __init__(self, message: SimpleNamespace) -> None:
        self.message = message
        self.kwargs: dict[str, Any] = {}

    def stream(self, **kwargs: Any) -> _FakeMessageStream:
        self.kwargs = kwargs
        return _FakeMessageStream(self.message)


@pytest.fixture
def _llm_logging_enabled() -> Iterator[None]:
    logger = logging.getLogger("sora.llm")
    previous = logger.level
    logger.setLevel(logging.INFO)
    yield
    logger.setLevel(previous)


@pytest.mark.asyncio
async def test_streamed_message_emits_its_cached_input_usage(
    _llm_logging_enabled: None,
) -> None:
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")],
        stop_reason="max_tokens",
        usage=SimpleNamespace(
            input_tokens=20,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=70,
            output_tokens=2,
        ),
    )
    client = AnthropicLLMClient(model="m", api_key="test", instrument=True)
    client._client = SimpleNamespace(messages=_FakeMessages(message))  # type: ignore[assignment]
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign,assignment]
    logger = logging.getLogger("sora.llm")
    logger.addHandler(handler)
    try:
        assert await client.complete(CompletionRequest("s", "p", "plan", "1")) == "hi"
    finally:
        logger.removeHandler(handler)

    usage_records = [r for r in records if r.__dict__.get("llm_event") == "usage"]
    assert len(usage_records) == 1
    (record,) = usage_records
    assert record.__dict__["llm_input_tokens"] == 100
    assert record.__dict__["llm_cached_input_tokens"] == 70
    assert record.__dict__["llm_finish_reason"] == "max_tokens"
    assert record.__dict__["llm_semantic_label"] == "plan"
    assert record.__dict__["llm_prompt_version"] == "1"


@pytest.mark.asyncio
async def test_request_profile_overrides_output_cap_and_ignores_unsupported_hints() -> None:
    message = SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")])
    messages = _FakeMessages(message)
    client = AnthropicLLMClient(model="m", api_key="test", max_tokens=8192)
    client._client = SimpleNamespace(messages=messages)  # type: ignore[assignment]

    request = CompletionRequest(
        "system",
        "user",
        "condition",
        "1",
        profile=CompletionProfile(max_output_tokens=64, reasoning="low"),
    )
    assert await client.complete(request) == "hi"

    assert messages.kwargs["max_tokens"] == 64
    assert messages.kwargs["system"] == "system"
    assert messages.kwargs["messages"] == [{"role": "user", "content": "user"}]
    assert "reasoning" not in messages.kwargs
