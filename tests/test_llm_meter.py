"""MeteredLLMClient (transparent timing/logging decorator) + LLMMeter (log-driven tally)."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from sora.llm import (
    LLMMeter,
    LLMUsage,
    MeteredLLMClient,
    current_inference_id,
    log_llm_discarded,
    log_llm_usage,
)


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


def test_metered_client_falls_back_to_the_clients_own_model() -> None:
    """An `llm:` block may omit `model:` and let the client use its default — that run still has a
    model, so reporting "none" would be a lie rather than a missing detail."""

    class _NamedClient(_StubClient):
        model = "claude-sonnet-4-5"

    assert MeteredLLMClient(_NamedClient()).model == "claude-sonnet-4-5"


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


def test_llm_usage_prefers_an_exact_reasoning_count_over_the_estimate() -> None:
    # A provider that reports thinking tokens directly (OpenAI reasoning_tokens, Gemini
    # thoughts_token_count) wins: thinking is that exact number, not the output-minus-answer
    # estimate. The answer-token display is unaffected — it still reflects the visible answer size.
    usage = LLMUsage(1200, 800, answer_chars=40, reasoning_tokens=600)
    assert usage.thinking_tokens == 600  # exact, not 790 the estimate would give
    assert usage.thinking_share == 600 / 800
    assert usage.answer_tokens == 10
    # reasoning_tokens=None (the Anthropic path / default) leaves the estimate in force, unchanged.
    assert LLMUsage(1200, 800, answer_chars=40).thinking_tokens == 790


def test_llm_meter_pools_exact_reasoning_tokens_when_providers_report_them(
    _llm_logging_enabled: None,
) -> None:
    # When usage records carry reasoning_tokens, the pooled thinking share is the summed exact
    # counts over summed output — not the char estimate — so a multi-call run reports true thinking.
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    try:
        log_llm_usage(LLMUsage(1000, 900, answer_chars=40, reasoning_tokens=850))
        log_llm_usage(LLMUsage(500, 200, answer_chars=600, reasoning_tokens=50))
    finally:
        logger.removeHandler(meter)

    assert (meter.input_tokens, meter.output_tokens) == (1500, 1100)
    assert meter.thinking_share == (850 + 50) / 1100  # exact, not the answer_chars estimate


def test_llm_meter_mixes_exact_and_estimated_thinking_per_call(
    _llm_logging_enabled: None,
) -> None:
    # A run that mixes providers — one call reports an exact reasoning count (OpenAI/Gemini), the
    # other does not (Anthropic) — must sum EACH call's own figure: the exact count for the first
    # plus the answer-subtraction estimate for the second. The regression this guards: a single
    # exact-reporting call must not blank out the estimated thinking of the estimate-only calls.
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    try:
        log_llm_usage(LLMUsage(1000, 1000, answer_chars=40))  # Anthropic: estimate 1000 - 10 = 990
        log_llm_usage(LLMUsage(500, 200, answer_chars=40, reasoning_tokens=50))  # OpenAI: exact 50
    finally:
        logger.removeHandler(meter)

    # 990 (estimated) + 50 (exact) over 1200 pooled output — NOT 50/1200, which the old all-or-
    # nothing flag produced once any call reported an exact count.
    assert meter.thinking_share == (990 + 50) / 1200


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


def test_log_llm_discarded_emits_a_discard_record(_llm_logging_enabled: None) -> None:
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign,assignment]
    logger = logging.getLogger("sora.llm")
    logger.addHandler(handler)
    try:
        log_llm_discarded("inf-9")
    finally:
        logger.removeHandler(handler)

    (record,) = records
    assert record.__dict__["llm_event"] == "discarded"  # distinct from "done"/"usage"
    assert record.__dict__["llm_inference_id"] == "inf-9"


@pytest.mark.asyncio
async def test_per_call_cues_carry_a_short_inference_id_tag(_llm_logging_enabled: None) -> None:
    # The usage/timing/discarded cues share a `[id]` tag so a discarded cue is visibly the *same*
    # call as its metrics lines — they can't be marked discarded at print time (that's known a cycle
    # later), so the shared tag is how a reader correlates them.
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign,assignment]
    logger = logging.getLogger("sora.llm")
    logger.addHandler(handler)
    token = current_inference_id.set("32a3ad56cdb2421c")  # a full id; the tag shows the first 8
    try:
        await MeteredLLMClient(_StubClient()).complete(system="s", prompt="p")
        log_llm_usage(LLMUsage(1000, 800, answer_chars=40))
    finally:
        current_inference_id.reset(token)
    log_llm_discarded("32a3ad56cdb2421c")
    logger.removeHandler(handler)

    messages = [r.getMessage() for r in records]
    assert all("[32a3ad56]" in m for m in messages)  # every cue for this call carries the tag
    assert any(m.startswith("~ llm [32a3ad56] usage:") for m in messages)
    assert any(m.startswith("~ llm [32a3ad56] (") for m in messages)  # the timing cue
    assert any(m == "~ llm [32a3ad56] discarded (result superseded)" for m in messages)


@pytest.mark.asyncio
async def test_per_call_cues_have_no_tag_without_an_inference_id(
    _llm_logging_enabled: None,
) -> None:
    # A call not driven by an off-cycle inference (id unset) keeps the byte-for-byte pre-existing
    # cue text — no tag — so uninstrumented / non-inference LLM use reads exactly as before.
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign,assignment]
    logger = logging.getLogger("sora.llm")
    logger.addHandler(handler)
    try:
        await MeteredLLMClient(_StubClient()).complete(system="s", prompt="p")
        log_llm_usage(LLMUsage(1000, 800, answer_chars=40))
    finally:
        logger.removeHandler(handler)

    messages = [r.getMessage() for r in records]
    assert any(m.startswith("~ llm usage:") for m in messages)  # no "[...]" tag
    assert any(m.startswith("~ llm (") for m in messages)
    assert all("[" not in m for m in messages)


@pytest.mark.asyncio
async def test_llm_meter_moves_a_discarded_inference_to_the_wasted_bucket(
    _llm_logging_enabled: None,
) -> None:
    # A call attributed to an inference id (via the contextvar) that is later discarded: its metered
    # time and tokens are counted in the grand totals (real, billed cost) AND broken out as wasted.
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    token = current_inference_id.set("inf-1")
    try:
        await MeteredLLMClient(_StubClient()).complete(system="s", prompt="p")  # done, tagged inf-1
        log_llm_usage(LLMUsage(1000, 800, answer_chars=40))  # usage, tagged inf-1
        log_llm_discarded("inf-1")  # the result was invalidated/superseded -> wasted
    finally:
        current_inference_id.reset(token)
        logger.removeHandler(meter)

    assert meter.calls == 1  # still counted in the totals (the call really ran and was billed)
    assert meter.wasted_calls == 1  # ...and broken out as wasted
    assert meter.wasted_seconds == meter.total_seconds  # the one call's whole time was wasted
    assert (meter.wasted_input_tokens, meter.wasted_output_tokens) == (1000, 800)


@pytest.mark.asyncio
async def test_llm_meter_keeps_a_used_inference_out_of_the_wasted_bucket(
    _llm_logging_enabled: None,
) -> None:
    # The complement: an inference that resolves normally (no discard cue) stays fully used.
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    token = current_inference_id.set("inf-1")
    try:
        await MeteredLLMClient(_StubClient()).complete(system="s", prompt="p")
        log_llm_usage(LLMUsage(1000, 800, answer_chars=40))
    finally:
        current_inference_id.reset(token)
        logger.removeHandler(meter)

    assert meter.calls == 1
    assert meter.wasted_calls == 0
    assert meter.wasted_seconds == 0.0
    assert (meter.wasted_input_tokens, meter.wasted_output_tokens) == (0, 0)


def test_llm_meter_summary_shows_used_and_wasted_split() -> None:
    # When something was discarded, summary breaks out used vs. wasted side by side; the grand
    # totals stay the headline (real cost), with the wasted subset annotated.
    meter = LLMMeter()
    meter.calls = 5
    meter.total_seconds = 12.3
    meter.wasted_calls = 1
    meter.wasted_seconds = 2.1
    meter.usage_calls = 5
    meter.input_tokens = 1500
    meter.output_tokens = 1100
    meter.answer_chars = 640
    meter.wasted_output_tokens = 300

    summary = meter.summary(30.0)

    assert "5 LLM calls (4 used, 1 discarded)" in summary
    assert "12.3s in-model (10.2s used, 2.1s wasted)" in summary
    assert "30.0s wall" in summary
    assert "1500 in / 1100 out tokens" in summary
    assert "300 out discarded" in summary


def test_llm_meter_summary_unchanged_when_nothing_discarded() -> None:
    # No discards -> the terse pre-existing format, so a normal run reads exactly as before.
    meter = LLMMeter()
    meter.calls = 3
    meter.total_seconds = 4.0
    assert meter.summary() == "3 LLM calls, 4.0s in-model"
    assert "used" not in meter.summary()
    assert "wasted" not in meter.summary()
