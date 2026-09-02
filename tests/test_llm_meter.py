"""MeteredLLMClient (transparent timing/logging decorator) + LLMMeter (log-driven tally)."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from sora.llm import (
    CompletionRequest,
    LLMCallLimitExceeded,
    LLMMeter,
    LLMUsage,
    MeteredLLMClient,
    PromptSection,
    current_inference_id,
    llm_call_scope,
    log_llm_discarded,
    log_llm_malformed,
    log_llm_outcome,
    log_llm_terminal_parse_failure,
    log_llm_usage,
)
from sora.memory import (
    _parse_condition_verdict,
    _parse_keep,
    _parse_relevance,
    _parse_verdict,
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
        self.calls: list[CompletionRequest] = []
        self.closed = False

    async def complete(self, request: CompletionRequest) -> str:
        self.calls.append(request)
        return self.response

    async def aclose(self) -> None:
        self.closed = True


def _request() -> CompletionRequest:
    return CompletionRequest("s", "p", "test", "1")


@pytest.mark.asyncio
async def test_metered_client_is_transparent_and_forwards_arguments() -> None:
    inner = _StubClient("the answer")
    metered = MeteredLLMClient(inner)
    request = CompletionRequest("sys", "usr", "plan", "1")

    result = await metered.complete(request)

    assert result == "the answer"  # passes the inner result straight through
    assert inner.calls == [request]  # forwards the same metadata-bearing value unchanged


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
        await metered.complete(_request())
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
        async def complete(self, request: CompletionRequest) -> str:
            raise RuntimeError("boom")

    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            await MeteredLLMClient(_Boom()).complete(_request())
    finally:
        logger.removeHandler(meter)

    assert meter.calls == 1  # the finally-clause cue still fired


@pytest.mark.asyncio
async def test_metered_client_forwards_aclose() -> None:
    inner = _StubClient()
    await MeteredLLMClient(inner).aclose()
    assert inner.closed is True


@pytest.mark.asyncio
async def test_metered_client_rejects_before_admitting_a_call_past_the_logical_limit() -> None:
    inner = _StubClient()
    metered = MeteredLLMClient(inner, max_logical_calls=2)

    await metered.complete(_request())
    await metered.complete(_request())
    with pytest.raises(LLMCallLimitExceeded, match="2 logical agent LLM calls"):
        await metered.complete(_request())

    assert len(inner.calls) == 2
    assert metered.logical_calls_admitted == 2
    assert metered.logical_call_limit_exceeded is True


@pytest.mark.asyncio
async def test_failed_logical_call_consumes_admission_budget() -> None:
    class _Boom(_StubClient):
        async def complete(self, request: CompletionRequest) -> str:
            raise RuntimeError("boom")

    metered = MeteredLLMClient(_Boom(), max_logical_calls=1)
    with pytest.raises(RuntimeError, match="boom"):
        await metered.complete(_request())
    with pytest.raises(LLMCallLimitExceeded):
        await metered.complete(_request())

    assert metered.logical_calls_admitted == 1


@pytest.mark.asyncio
async def test_parse_repair_round_trip_under_one_call_scope_consumes_one_admission() -> None:
    inner = _StubClient()
    metered = MeteredLLMClient(inner, max_logical_calls=1)

    with llm_call_scope():
        await metered.complete(_request())
        await metered.complete(_request())

    assert len(inner.calls) == 2
    assert metered.logical_calls_admitted == 1
    with pytest.raises(LLMCallLimitExceeded):
        await metered.complete(_request())


@pytest.mark.asyncio
async def test_admission_deduplication_retains_no_completed_call_ids() -> None:
    metered = MeteredLLMClient(_StubClient())

    for _ in range(100):
        await metered.complete(_request())

    assert metered.logical_calls_admitted == 100
    assert not hasattr(metered, "_admitted_call_ids")


@pytest.mark.asyncio
async def test_llm_meter_tallies_calls_and_seconds(_llm_logging_enabled: None) -> None:
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    metered = MeteredLLMClient(_StubClient())
    try:
        await metered.complete(_request())
        await metered.complete(_request())
    finally:
        logger.removeHandler(meter)

    assert meter.calls == 2
    assert meter.total_seconds >= 0.0
    report = meter.report()
    assert report.section_characters is None
    assert report.dynamic_section_characters is None
    assert report.inferences[0].dynamic_section_share is None


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


def test_meter_reports_terminal_parse_failure_separately_from_field_repairs(
    _llm_logging_enabled: None,
) -> None:
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    try:
        with llm_call_scope():
            log_llm_terminal_parse_failure()
    finally:
        logger.removeHandler(meter)

    report = meter.report()
    assert report.terminal_parse_failures == 1
    assert report.malformed_fields_repaired == 0
    assert report.inferences[0].terminal_parse_failures == 1


@pytest.mark.asyncio
async def test_meter_reports_prompt_hashes_and_provider_observations(
    _llm_logging_enabled: None,
) -> None:
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    client = MeteredLLMClient(_StubClient())
    request = CompletionRequest("system", "user", "plan", "1")
    try:
        with llm_call_scope():
            log_llm_usage(
                LLMUsage(10, 5, 4, reasoning_tokens=3),
                request,
                observed_model="snapshot-model",
                sdk_name="provider-sdk",
                sdk_version="1.2.3",
                provider_observation={
                    "requested": "moonshotai/kimi-k2.5",
                    "summary": "selected=Moonshot AI",
                },
            )
            await client.complete(request)
    finally:
        logger.removeHandler(meter)

    (call,) = meter.report().inferences
    assert len(call.system_prompt_sha256 or "") == 64
    assert len(call.user_prompt_sha256 or "") == 64
    assert call.observed_models == ("snapshot-model",)
    assert call.sdk_observations == (("provider-sdk", "1.2.3"),)
    assert call.provider_observations == (
        {
            "requested": "moonshotai/kimi-k2.5",
            "summary": "selected=Moonshot AI",
        },
    )
    assert (call.reasoning_tokens, call.reasoning_tokens_exact) == (3, True)


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
        log_llm_usage(LLMUsage(1200, 800, answer_chars=40, cached_input_tokens=900))
    finally:
        logger.removeHandler(handler)

    (record,) = records
    assert record.__dict__["llm_event"] == "usage"  # distinct from the timing "done" event
    assert record.__dict__["llm_input_tokens"] == 1200
    assert record.__dict__["llm_output_tokens"] == 800
    assert record.__dict__["llm_answer_chars"] == 40
    assert record.__dict__["llm_cached_input_tokens"] == 900
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


def test_llm_meter_tallies_cached_input_and_reports_hit_rate(
    _llm_logging_enabled: None,
) -> None:
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    try:
        log_llm_usage(LLMUsage(1000, 100, answer_chars=40, cached_input_tokens=800))
        log_llm_usage(LLMUsage(500, 100, answer_chars=40))
    finally:
        logger.removeHandler(meter)

    assert meter.cached_input_tokens == 800
    assert meter.cache_observed_input_tokens == 1000
    assert meter.cache_unknown_input_tokens == 500
    assert meter.cache_hit_rate == 800 / 1000
    assert meter.cache_coverage == 1000 / 1500
    assert "800 cached, 80% hit (67% cache coverage)" in meter.summary()
    report = meter.report()
    assert report.cache_observed_input_tokens == 1000
    assert report.cache_unknown_input_tokens == 500
    assert len(report.inferences) == 2
    assert sum(row.cache_observed_input_tokens for row in report.inferences) == 1000
    assert sum(row.cache_unknown_input_tokens for row in report.inferences) == 500


def test_llm_meter_reports_an_explicit_zero_as_a_measured_cache_miss(
    _llm_logging_enabled: None,
) -> None:
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    try:
        log_llm_usage(LLMUsage(42, 1, answer_chars=1, cached_input_tokens=0))
    finally:
        logger.removeHandler(meter)

    assert meter.cache_hit_rate == 0.0
    assert meter.cache_coverage == 1.0
    assert "0 cached, 0% hit (100% cache coverage)" in meter.summary()


def test_llm_meter_reports_cache_usage_as_unavailable_when_unobserved(
    _llm_logging_enabled: None,
) -> None:
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    try:
        log_llm_usage(LLMUsage(42, 1, answer_chars=1))
    finally:
        logger.removeHandler(meter)

    assert meter.cache_hit_rate is None
    assert meter.cache_coverage == 0.0
    assert meter.cache_unknown_input_tokens == 42
    assert "cache usage unavailable" in meter.summary()


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
        await MeteredLLMClient(_StubClient()).complete(_request())
        log_llm_usage(LLMUsage(1000, 800, answer_chars=40))
    finally:
        current_inference_id.reset(token)
    log_llm_discarded("32a3ad56cdb2421c")
    logger.removeHandler(handler)

    messages = [r.getMessage() for r in records]
    assert all("[32a3ad56]" in m for m in messages)  # every cue for this call carries the tag
    assert any(m.startswith("~ llm [32a3ad56] usage:") for m in messages)
    assert any(m.startswith("~ llm [32a3ad56] test/v1 (") for m in messages)  # timing
    assert any(m == "~ llm [32a3ad56] discarded (result superseded)" for m in messages)


@pytest.mark.asyncio
async def test_per_call_cues_get_a_logical_call_tag_without_an_inference_id(
    _llm_logging_enabled: None,
) -> None:
    # Background judgements are not pending activity inferences, but they are still logical LLM
    # calls. Their timing/usage cues need a generated id so concurrent retirement/relevance calls
    # can be correlated rather than appearing as anonymous, interchangeable lines.
    class _UsageClient(_StubClient):
        async def complete(self, request: CompletionRequest) -> str:
            log_llm_usage(LLMUsage(1000, 800, answer_chars=40), request)
            return await super().complete(request)

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign,assignment]
    logger = logging.getLogger("sora.llm")
    logger.addHandler(handler)
    try:
        await MeteredLLMClient(_UsageClient()).complete(_request())
    finally:
        logger.removeHandler(handler)

    messages = [r.getMessage() for r in records]
    call_ids = {r.__dict__["llm_call_id"] for r in records}
    assert len(call_ids) == 1
    (call_id,) = call_ids
    assert isinstance(call_id, str)
    assert all(f"[{call_id[:8]}]" in message for message in messages)
    assert any(f"~ llm [{call_id[:8]}] test/v1 usage:" in m for m in messages)
    assert any(f"~ llm [{call_id[:8]}] test/v1 (" in m for m in messages)
    assert all(r.__dict__["llm_inference_id"] is None for r in records)


@pytest.mark.asyncio
async def test_background_calls_have_distinct_report_rows(_llm_logging_enabled: None) -> None:
    class _UsageClient(_StubClient):
        async def complete(self, request: CompletionRequest) -> str:
            log_llm_usage(LLMUsage(10, 2, answer_chars=4), request)
            return await super().complete(request)

    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    client = MeteredLLMClient(_UsageClient())
    try:
        await client.complete(CompletionRequest("s", "u", "retirement", "1"))
        await client.complete(CompletionRequest("s", "u", "retirement", "1"))
    finally:
        logger.removeHandler(meter)

    assert len(meter.report().inferences) == 2
    assert {row.inference_id for row in meter.report().inferences} == {None}
    call_ids = {row.call_id for row in meter.report().inferences}
    assert None not in call_ids
    assert len(call_ids) == 2


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
        await MeteredLLMClient(_StubClient()).complete(_request())  # done, tagged inf-1
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
        await MeteredLLMClient(_StubClient()).complete(_request())
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


@pytest.mark.asyncio
async def test_request_metadata_reaches_timing_and_usage_records(
    _llm_logging_enabled: None,
) -> None:
    request = CompletionRequest(
        "system",
        "user",
        semantic_label="plan",
        prompt_version="3",
        sections=(
            PromptSection("contract", characters=80, dynamic=False),
            PromptSection("observations", characters=20, dynamic=True),
        ),
    )
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign,assignment]
    logger = logging.getLogger("sora.llm")
    logger.addHandler(handler)
    try:
        await MeteredLLMClient(_StubClient()).complete(request)
        log_llm_usage(LLMUsage(100, 10, answer_chars=8), request, finish_reason="stop")
    finally:
        logger.removeHandler(handler)

    attributed = [
        record for record in records if record.__dict__.get("llm_event") in {"done", "usage"}
    ]
    assert len(attributed) == 2
    for record in attributed:
        assert record.__dict__["llm_semantic_label"] == "plan"
        assert record.__dict__["llm_prompt_version"] == "3"
        assert record.__dict__["llm_section_characters"] == 100
        assert record.__dict__["llm_dynamic_section_characters"] == 20
    assert attributed[-1].__dict__["llm_finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_structured_report_aggregates_round_trips_by_inference(
    _llm_logging_enabled: None,
) -> None:
    """A parse retry is two provider round trips but one off-cycle inference and outcome."""
    request = CompletionRequest(
        "system",
        "user",
        semantic_label="plan",
        prompt_version="3",
        sections=(
            PromptSection("static", characters=75, dynamic=False),
            PromptSection("goal", characters=25, dynamic=True),
        ),
    )
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    token = current_inference_id.set("inf-plan")
    try:
        for finish_reason in ("length", "stop"):
            log_llm_usage(
                LLMUsage(100, 20, answer_chars=20, cached_input_tokens=40),
                request,
                finish_reason=finish_reason,
            )
            await MeteredLLMClient(_StubClient()).complete(request)
        log_llm_malformed(dropped=2, repaired=1)
        log_llm_outcome("inf-plan", "success")
        log_llm_discarded("inf-plan")
    finally:
        current_inference_id.reset(token)
        logger.removeHandler(meter)

    report = meter.report()
    assert report.calls == 2
    assert report.latency_seconds == meter.total_seconds
    assert report.input_tokens == 200
    assert report.cached_input_tokens == 80
    assert report.output_tokens == 40
    assert report.section_characters == 200
    assert report.dynamic_section_characters == 50
    assert report.malformed_fields_dropped == 2
    assert report.malformed_fields_repaired == 1
    assert len(report.inferences) == 1
    inference = report.inferences[0]
    assert inference.inference_id == "inf-plan"
    assert inference.semantic_label == "plan"
    assert inference.prompt_version == "3"
    assert inference.round_trips == 2
    assert len(inference.usages) == 2
    assert [usage.input_tokens for usage in inference.usages] == [100, 100]
    assert len(inference.prompt_hashes) == 2
    assert inference.user_prompt_sha256 == inference.prompt_hashes[0].user_sha256
    assert inference.latency_seconds == meter.total_seconds
    assert inference.dynamic_section_share == 0.25
    assert inference.finish_reasons == ("length", "stop")
    assert inference.outcome == "success"
    assert inference.discarded is True
    assert inference.malformed_fields_dropped == 2
    assert inference.malformed_fields_repaired == 1


@pytest.mark.asyncio
async def test_structured_report_keeps_primary_and_repair_prompt_hashes(
    _llm_logging_enabled: None,
) -> None:
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    client = MeteredLLMClient(_StubClient())
    primary = CompletionRequest("system", "primary", "plan", "1")
    repair = CompletionRequest("system", "repair", "plan", "1")
    try:
        with llm_call_scope():
            await client.complete(primary)
            await client.complete(repair)
    finally:
        logger.removeHandler(meter)

    (inference,) = meter.report().inferences
    assert len(inference.prompt_hashes) == 2
    assert inference.prompt_hashes[0].user_sha256 != inference.prompt_hashes[1].user_sha256
    assert inference.user_prompt_sha256 == inference.prompt_hashes[0].user_sha256


@pytest.mark.asyncio
async def test_call_scope_attributes_malformed_cue_without_an_activity_inference(
    _llm_logging_enabled: None,
) -> None:
    request = CompletionRequest("system", "user", semantic_label="plan", prompt_version="3")
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    try:
        with llm_call_scope():
            await MeteredLLMClient(_StubClient()).complete(request)
            log_llm_usage(LLMUsage(100, 20, answer_chars=20), request)
            log_llm_malformed(dropped=2, repaired=1)
    finally:
        logger.removeHandler(meter)

    report = meter.report()
    assert report.malformed_fields_dropped == 2
    assert report.malformed_fields_repaired == 1
    assert len(report.inferences) == 1
    (inference,) = report.inferences
    assert inference.semantic_label == "plan"
    assert inference.prompt_version == "3"
    assert inference.inference_id is None
    assert inference.call_id is not None
    assert inference.malformed_fields_dropped == 2
    assert inference.malformed_fields_repaired == 1


def test_structured_report_keeps_the_first_terminal_outcome(
    _llm_logging_enabled: None,
) -> None:
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    try:
        log_llm_outcome("inf-1", "success")
        log_llm_outcome("inf-1", "error")  # a later stale result for the same terminal inference
    finally:
        logger.removeHandler(meter)

    (inference,) = meter.report().inferences
    assert inference.outcome == "success"


def test_structured_report_retains_an_off_cycle_inference_error(
    _llm_logging_enabled: None,
) -> None:
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    try:
        log_llm_outcome(
            "inf-1",
            "error",
            error="BadRequestError('maximum context length exceeded')",
        )
    finally:
        logger.removeHandler(meter)

    (inference,) = meter.report().inferences
    assert inference.error == "BadRequestError('maximum context length exceeded')"


def test_tolerant_response_parsers_report_every_dropped_or_repaired_field(
    _llm_logging_enabled: None,
) -> None:
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    logger.addHandler(meter)
    token = current_inference_id.set("inf-tolerant")
    try:
        assert _parse_verdict('{"valid": "yes"}') is True
        verdict = _parse_condition_verdict('{"fired": [0, "bad", 9, 0], "retired": "none"}', 1)
        assert verdict.fired == (0,)
        assert verdict.retired == ()
        assert _parse_keep('{"keep": [0, "bad", 9, 0]}', 1) == [0]
        assert (
            _parse_relevance(
                '{"relevant": true, "task": "zero", "goal": "", "question": 3}',
                [{"activity_id": "a1"}],
            )
            is None
        )
    finally:
        current_inference_id.reset(token)
        logger.removeHandler(meter)

    (inference,) = meter.report().inferences
    assert inference.malformed_fields_dropped == 8
    assert inference.malformed_fields_repaired == 3
