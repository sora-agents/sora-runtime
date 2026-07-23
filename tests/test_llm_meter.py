"""MeteredLLMClient (transparent timing/logging decorator) + LLMMeter (log-driven tally)."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from sora.llm import LLMMeter, MeteredLLMClient


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
