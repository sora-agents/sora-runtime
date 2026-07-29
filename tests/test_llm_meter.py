"""MeteredLLMClient (transparent timing/logging decorator) + LLMMeter (log-driven tally)."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from sora.llm import LLMMeter, LLMUsage, MeteredLLMClient, log_llm_usage


@pytest.fixture
def _llm_logging_enabled() -> Iterator[None]:
    # The per-call cue is logged at INFO; without a configured level the record is dropped before
    # any handler sees it (root defaults to WARNING). Real run surfaces enable it — the CLI sets
    # `sora` to DEBUG, the example runners set the root to INFO — so mirror that here.
    logger = logging.getLogger("sora.llm")
    previous = logger.level
    logger.setLevel(logging.INFO)
    yield
    logger.setLevel(previous)


class _StubClient:
    def __init__(self, response: str = "ok") -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    async def complete(self, *, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        return self.response

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_metered_client_is_transparent_and_forwards_arguments() -> None:
    inner = _StubClient("the answer")
    metered = MeteredLLMClient(inner)

    result = await metered.complete(system="sys", prompt="usr")

    assert result == "the answer"  # passes the inner result straight through
    assert inner.calls == [("sys", "usr")]  # forwards keyword args unchanged


@pytest.mark.asyncio
async def test_metered_client_logs_one_timed_cue_per_call(_llm_logging_enabled: None) -> None:
    metered = MeteredLLMClient(_StubClient())
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign,assignment]
    logger = logging.getLogger("sora.llm")
    logger.addHandler(handler)
    try:
        await metered.complete(system="s", prompt="p")
    finally:
        logger.removeHandler(handler)

    assert len(records) == 1
    (record,) = records
    assert record.name == "sora.llm"
    # The structured fields ride in via `extra=`, so they live in __dict__, not typed attributes.
    assert record.__dict__["llm_event"] == "done"
    assert isinstance(record.__dict__["llm_seconds"], float)
    assert "llm" in record.getMessage()


@pytest.mark.asyncio
async def test_metered_client_logs_even_when_inner_raises(_llm_logging_enabled: None) -> None:
    class _Boom(_StubClient):
        async def complete(self, *, system: str, prompt: str) -> str:
            raise RuntimeError("boom")

    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            await MeteredLLMClient(_Boom()).complete(system="s", prompt="p")
    finally:
        logger.removeHandler(meter)

    assert meter.calls == 1  # the finally-clause cue still fired


@pytest.mark.asyncio
async def test_metered_client_forwards_aclose() -> None:
    inner = _StubClient()
    await MeteredLLMClient(inner).aclose()
    assert inner.closed is True


@pytest.mark.asyncio
async def test_llm_meter_tallies_calls_and_seconds(_llm_logging_enabled: None) -> None:
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    metered = MeteredLLMClient(_StubClient())
    try:
        await metered.complete(system="s", prompt="p")
        await metered.complete(system="s", prompt="p")
    finally:
        logger.removeHandler(meter)

    assert meter.calls == 2
    assert meter.total_seconds >= 0.0


def test_llm_meter_summary_singular_plural_and_wall() -> None:
    meter = LLMMeter()
    assert meter.summary() == "0 LLM calls, 0.0s in-model"

    meter.calls = 1
    meter.total_seconds = 1.24
    assert meter.summary() == "1 LLM call, 1.2s in-model"
    assert meter.summary(12.7) == "1 LLM call, 1.2s in-model, 12.7s wall"

    meter.calls = 3
    assert meter.summary().startswith("3 LLM calls,")


def test_llm_meter_ignores_unrelated_records() -> None:
    meter = LLMMeter()
    meter.handle(logging.LogRecord("sora.cycle", logging.INFO, __file__, 0, "observe: x", (), None))
    assert meter.calls == 0  # only records carrying llm_event="done" are counted


def test_llm_usage_estimates_thinking_from_output_minus_answer() -> None:
    # A tiny answer (40 chars ~= 10 tokens) behind a large output (800) reads as thinking-bound —
    # the exact case adaptive thinking hides from a thinking-block char count (which would see ~0).
    usage = LLMUsage(1200, 800, answer_chars=40)
    assert usage.answer_tokens == 10  # 40 / _CHARS_PER_TOKEN
    assert usage.thinking_tokens == 790  # output the answer doesn't explain
    assert usage.thinking_share == 790 / 800
    # An answer as long as the output is answer-bound: near-0 thinking, clamped, never negative.
    assert LLMUsage(500, 200, answer_chars=1000).thinking_tokens == 0
    assert LLMUsage(500, 200, answer_chars=1000).thinking_share == 0.0
    # No output at all -> 0.0, never a divide-by-zero.
    assert LLMUsage(10, 0, answer_chars=0).thinking_share == 0.0


def test_log_llm_usage_emits_one_usage_record(_llm_logging_enabled: None) -> None:
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign,assignment]
    logger = logging.getLogger("sora.llm")
    logger.addHandler(handler)
    try:
        log_llm_usage(LLMUsage(1200, 800, answer_chars=40))
    finally:
        logger.removeHandler(handler)

    (record,) = records
    assert record.__dict__["llm_event"] == "usage"  # distinct from the timing "done" event
    assert record.__dict__["llm_input_tokens"] == 1200
    assert record.__dict__["llm_output_tokens"] == 800
    assert record.__dict__["llm_answer_chars"] == 40
    # The cue shows the answer size beside the output for a quick eyeball: ~10 answer of 800 out.
    assert "~10 answer, ~99% thinking" in record.getMessage()  # 790/800, rounded


def test_llm_meter_tallies_usage_and_summary_reports_tokens(_llm_logging_enabled: None) -> None:
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    try:
        # Two instrumented calls: a thinking-bound one (tiny answer) and an answer-heavy one.
        log_llm_usage(LLMUsage(1000, 900, answer_chars=40))  # ~10 answer tok, ~890 thinking
        log_llm_usage(LLMUsage(500, 200, answer_chars=600))  # ~150 answer tok, ~50 thinking
    finally:
        logger.removeHandler(meter)

    assert meter.usage_calls == 2
    assert (meter.input_tokens, meter.output_tokens) == (1500, 1100)
    # Aggregate share is over pooled totals: (1100 out - 640/4 answer tok) / 1100.
    assert meter.thinking_share == (1100 - round(640 / 4)) / 1100
    summary = meter.summary()
    assert "1500 in / 1100 out tokens (~160 answer, ~85% thinking)" in summary


def test_llm_meter_summary_omits_tokens_when_uninstrumented() -> None:
    # With no usage records (client not instrumented), summary reports timing only — unchanged.
    meter = LLMMeter()
    meter.calls = 2
    meter.total_seconds = 3.0
    assert "tokens" not in meter.summary()
