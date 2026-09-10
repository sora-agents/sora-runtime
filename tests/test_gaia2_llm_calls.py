"""Per-call model records: the join that produces them, and the ReAct engine that feeds ARE's clock.

Two things are worth locking down here, and neither is visible from either module alone.

The S-ORA side is a *join across two log records* — tokens from the concrete client, latency from
the timing decorator — so the tests below are written against real ``sora.llm`` emissions rather
than hand-built ``LogRecord``s: a rename of one of those ``extra`` keys is exactly the failure a
hand-built record would hide.

The ReAct side is a *bracket*. ARE's ``step()`` keeps only the last metadata the engine returned,
so on a malformed-output retry the earlier round-trips' tokens are lost unless the engine
accumulates them — and the charge that ends up on ARE's clock is that dict's
``completion_duration``. The retry case is the one that undercounts silently in production, so it
is the one asserted directly.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from examples.gaia2.llm_calls import LLMCallRecord, LLMCallWriter, SoraCallRecorder

from sora.llm import (
    CompletionRequest,
    LLMUsage,
    MeteredLLMClient,
    llm_call_scope,
    log_llm_usage,
)
from sora.memory import _complete_and_parse


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _done(call_id: str, seconds: float, label: str | None = None) -> None:
    """The record ``MeteredLLMClient`` emits from its ``finally``; the client itself is async and
    needs a provider, so only this one line of it is reproduced."""
    logging.getLogger("sora.llm").info(
        "~ llm done",
        extra={
            "llm_event": "done",
            "llm_seconds": seconds,
            "llm_call_id": call_id,
            "llm_semantic_label": label,
        },
    )


Rows = Callable[[], list[dict[str, Any]]]


@pytest.fixture
def recorded(tmp_path: Path) -> Iterator[Rows]:
    """Attach a recorder to the ``sora`` logger and yield a ``rows()`` reader.

    ``rows()`` closes the recorder before reading, because closing is when rows are written: a
    logical call is only known to be finished once the stream ends (see ``_Partial``), so nothing
    reaches disk before then. Calling it is a test's stand-in for the run ending.
    """
    writer = LLMCallWriter(tmp_path / "llm_calls.jsonl").open()
    recorder = SoraCallRecorder(writer, model="fallback-model", scenario_id="scen-1", run_number=2)
    log = logging.getLogger("sora")
    log.setLevel(logging.INFO)
    log.addHandler(recorder)

    def rows() -> list[dict[str, Any]]:
        recorder.close()
        return _rows(writer.path)

    try:
        yield rows
    finally:
        log.removeHandler(recorder)
        recorder.close()
        writer.close()


def _request() -> CompletionRequest:
    return CompletionRequest(system="sys", user="usr", semantic_label="plan", prompt_version="1")


def test_usage_and_timing_join_into_one_row(recorded: Rows) -> None:
    with llm_call_scope() as call_id:
        log_llm_usage(
            LLMUsage(
                input_tokens=1200,
                output_tokens=340,
                answer_chars=800,
                reasoning_tokens=90,
                cached_input_tokens=1024,
            ),
            _request(),
            finish_reason="stop",
            observed_model="gpt-5.5",
        )
        _done(call_id, 4.25, "plan")

    (row,) = recorded()
    assert row["call_id"] == call_id
    assert row["arm"] == "sora"
    assert row["input_tokens"] == 1200
    assert row["output_tokens"] == 340
    # The field the text log drops and the cache-aware fit needs; a provider that reports it must
    # survive the round trip through the record, not just through the summary tally.
    assert row["cached_input_tokens"] == 1024
    assert row["reasoning_tokens"] == 90
    assert row["seconds"] == pytest.approx(4.25)
    assert row["round_trips"] == 1
    assert row["finish_reason"] == "stop"
    assert row["model"] == "gpt-5.5"
    assert row["semantic_label"] == "plan"
    assert row["scenario_id"] == "scen-1" and row["run_number"] == 2
    assert row["usage_captured"] is True


async def test_repair_round_trip_sums_into_the_same_call(
    recorded: Rows,
) -> None:
    """A parser-repair pass reuses its logical call id: one decision, two billed round-trips.

    Driven through the real ``_complete_and_parse`` rather than a hand-built record sequence,
    because the *ordering* is the thing that has to be right. The retry is a second
    ``MeteredLLMClient.complete``, not a second round trip inside one, so the stream reads
    ``usage, done, usage, done`` — and a recorder that flushed on the first ``done`` would split
    this one decision into two rows that each claim a single round trip and half the tokens.
    """

    class _Repairing:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: CompletionRequest) -> str:
            self.calls += 1
            log_llm_usage(LLMUsage(input_tokens=1000, output_tokens=100, answer_chars=200), request)
            return "not json at all" if self.calls == 1 else '{"ok": true}'

    inner = _Repairing()
    client = MeteredLLMClient(inner, model="m")
    parsed = await _complete_and_parse(client, _request(), json.loads, what="test inference")

    assert parsed == {"ok": True}
    assert inner.calls == 2
    (row,) = recorded()
    assert row["round_trips"] == 2
    assert row["input_tokens"] == 2000
    assert row["output_tokens"] == 200
    # Both crossings are billed, so the latency is the sum of the two, not just the last.
    assert row["seconds"] is not None


async def test_a_repair_whose_retry_dies_is_not_a_captured_row(recorded: Rows) -> None:
    """The retry crosses the wire and never reports its tokens: ``usage, done, done``.

    ``done`` comes from a ``finally``, so the second crossing is counted whatever happened to it,
    while ``usage`` comes from inside the client and simply never arrives. The row therefore has
    two round-trips' latency and one round-trip's tokens, which is exactly the shape that would
    read as a complete, cheap call — so it has to be flagged, not left looking measured.
    """

    class _FailingRepair:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: CompletionRequest) -> str:
            self.calls += 1
            if self.calls == 1:
                log_llm_usage(
                    LLMUsage(input_tokens=900, output_tokens=80, answer_chars=100), request
                )
                return "not json at all"
            raise RuntimeError("provider fell over on the repair")

    client = MeteredLLMClient(_FailingRepair(), model="m")
    with pytest.raises(RuntimeError):
        await _complete_and_parse(client, _request(), json.loads, what="test inference")

    (row,) = recorded()
    assert row["round_trips"] == 2
    assert row["input_tokens"] == 900  # what the one reporting trip cost, not what the call cost
    assert row["usage_captured"] is False


def test_absent_cache_field_stays_unknown_not_zero(recorded: Rows) -> None:
    """A provider that reports no cache read is unknown coverage, not a measured miss — the
    distinction the fit needs in order to know which rows it may use."""
    with llm_call_scope() as call_id:
        log_llm_usage(LLMUsage(input_tokens=10, output_tokens=5, answer_chars=10), _request())
        _done(call_id, 1.0)

    (row,) = recorded()
    assert row["cached_input_tokens"] is None
    assert row["reasoning_tokens"] is None


def test_uninstrumented_client_records_timing_only(recorded: Rows) -> None:
    """No ``usage`` record ever arrives, so the row is timing with the tokens flagged uncaptured —
    a truthful account of what that configuration can know rather than a silent row of zeros."""
    _done("abc123", 2.5)

    (row,) = recorded()
    assert row["usage_captured"] is False
    assert row["input_tokens"] is None and row["output_tokens"] is None
    assert row["seconds"] == pytest.approx(2.5)


def test_close_flushes_a_call_whose_timing_never_arrived(tmp_path: Path) -> None:
    """The process dying between the two records must not lose the tokens it already paid for."""
    writer = LLMCallWriter(tmp_path / "calls.jsonl").open()
    recorder = SoraCallRecorder(writer)
    log = logging.getLogger("sora")
    log.setLevel(logging.INFO)
    log.addHandler(recorder)
    try:
        with llm_call_scope():
            log_llm_usage(LLMUsage(input_tokens=7, output_tokens=3, answer_chars=6), _request())
    finally:
        log.removeHandler(recorder)
        recorder.close()
        writer.close()

    (row,) = _rows(writer.path)
    assert row["input_tokens"] == 7
    assert row["seconds"] is None


def test_writer_appends_across_sessions(tmp_path: Path) -> None:
    """A capability's sweep points every scenario at one file, so a second open must not truncate
    what the first one wrote."""
    path = tmp_path / "nested" / "calls.jsonl"

    def record(call_id: str) -> LLMCallRecord:
        return LLMCallRecord(
            call_id=call_id,
            arm="sora",
            model="m",
            semantic_label=None,
            input_tokens=1,
            cached_input_tokens=None,
            output_tokens=1,
            reasoning_tokens=None,
            seconds=1.0,
            round_trips=1,
            finish_reason=None,
        )

    with LLMCallWriter(path) as w:
        w.write(record("a"))
    with LLMCallWriter(path) as w:
        w.write(record("b"))

    assert [r["call_id"] for r in _rows(path)] == ["a", "b"]


def test_reset_truncates_once_at_the_start_of_a_sweep(tmp_path: Path) -> None:
    """Re-running a capability replaces every other artifact in its output directory, so the call
    log has to go too: rows from the previous sweep carry the same scenario_id and run_number as
    the new ones, and the fit would count them twice. Truncation is once per writer, though — a
    close/reopen partway through the sweep must keep what the sweep has already written."""
    path = tmp_path / "calls.jsonl"

    def record(call_id: str) -> LLMCallRecord:
        return LLMCallRecord(
            call_id=call_id,
            arm="sora",
            model="m",
            semantic_label=None,
            input_tokens=1,
            cached_input_tokens=None,
            output_tokens=1,
            reasoning_tokens=None,
            seconds=1.0,
            round_trips=1,
            finish_reason=None,
        )

    with LLMCallWriter(path) as stale:
        stale.write(record("previous-sweep"))

    writer = LLMCallWriter(path, reset=True)
    with writer:
        writer.write(record("a"))
    with writer:  # same writer, reopened mid-sweep
        writer.write(record("b"))

    assert [r["call_id"] for r in _rows(path)] == ["a", "b"]


async def test_join_against_the_real_metered_client(
    recorded: Rows,
) -> None:
    """The one test that hand-rolls nothing: a real ``MeteredLLMClient`` around a client that
    reports usage the way a concrete one does. ``_done`` above reproduces a single line of this
    client, so this is what keeps that reproduction honest — a rename of ``llm_seconds`` or
    ``llm_call_id`` breaks the join in production and must break here rather than there."""

    class _Client:
        async def complete(self, request: CompletionRequest) -> str:
            log_llm_usage(
                LLMUsage(input_tokens=900, output_tokens=120, answer_chars=300),
                request,
                finish_reason="stop",
                observed_model="claude-opus-5",
            )
            return "{}"

    client = MeteredLLMClient(_Client(), model="claude-opus-5")
    await client.complete(_request())

    (row,) = recorded()
    assert row["input_tokens"] == 900 and row["output_tokens"] == 120
    assert row["seconds"] is not None and row["seconds"] >= 0
    assert row["model"] == "claude-opus-5"
    assert row["semantic_label"] == "plan"


# -- the ReAct arm -------------------------------------------------------------------------------

# Skip-gated per test rather than at module scope: the S-ORA half above needs no ARE, and a
# module-level importorskip would take it down with the ReAct half everywhere ARE is not installed
# — which includes CI, where the `are` dependency-group is deliberately not synced.
requires_are = pytest.mark.skipif(
    importlib.util.find_spec("are") is None, reason="ARE is an optional dependency-group"
)


def _fake_response(prompt: int, cached: int | None, completion: int, reasoning: int | None) -> Any:
    """``cached``/``reasoning`` of None omit the detail block entirely, which is what an ordinary
    non-caching, non-reasoning provider actually returns."""
    from litellm.types.utils import (
        Choices,
        CompletionTokensDetailsWrapper,
        Message,
        ModelResponse,
        PromptTokensDetailsWrapper,
        Usage,
    )

    response = ModelResponse(
        choices=[Choices(message=Message(content="Thought: ok\nAction: {}"), finish_reason="stop")]
    )
    response.usage = Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=(
            PromptTokensDetailsWrapper(cached_tokens=cached) if cached is not None else None
        ),
        completion_tokens_details=(
            CompletionTokensDetailsWrapper(reasoning_tokens=reasoning)
            if reasoning is not None
            else None
        ),
    )
    return response


@pytest.fixture
def engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Any, list[Any], LLMCallWriter]]:
    """A metered engine whose provider call is replaced by a queue of canned responses.

    The double is installed *before* the engine exists and the engine re-arms its capture per
    call, which is the property that keeps the interception working when anything else patches the
    same module global."""
    from are.simulation.agents.llm.litellm import litellm_engine as module
    from are.simulation.agents.llm.litellm.litellm_engine import LiteLLMModelConfig
    from examples.gaia2.react_engine import MeteredLiteLLMEngine

    queue: list[Any] = []

    def fake_completion(**kwargs: Any) -> Any:
        return queue.pop(0)

    monkeypatch.setattr(module, "completion", fake_completion)
    writer = LLMCallWriter(tmp_path / "react.jsonl").open()
    eng = MeteredLiteLLMEngine(
        LiteLLMModelConfig(model_name="test/model", provider="local"),
        writer=writer,
        scenario_id="scen-9",
    )
    try:
        yield eng, queue, writer
    finally:
        writer.close()


@requires_are
def test_engine_feeds_the_metadata_are_reads(engine: tuple[Any, list[Any], LLMCallWriter]) -> None:
    """ARE reads exactly these keys off the dict; stock ``LiteLLMEngine`` returns ``None`` for all
    of them, which is why every published leaderboard row thinks instantaneously."""
    eng, queue, writer = engine
    queue.append(_fake_response(prompt=5000, cached=4000, completion=250, reasoning=120))

    _text, metadata = eng.chat_completion([{"role": "user", "content": "hi"}])

    assert metadata is not None
    assert metadata["prompt_tokens"] == 5000
    assert metadata["completion_tokens"] == 250
    assert metadata["total_tokens"] == 5250
    assert metadata["reasoning_tokens"] == 120
    assert metadata["cached_prompt_tokens"] == 4000
    assert metadata["completion_duration"] > 0
    assert _rows(writer.path)[0]["usage_captured"] is True


@requires_are
def test_omitted_detail_blocks_stay_unknown_not_zero(
    engine: tuple[Any, list[Any], LLMCallWriter],
) -> None:
    """The common case: a provider returns ``usage`` but no ``prompt_tokens_details`` and no
    ``completion_tokens_details``. Reading those as 0 would write a *measured* cache miss into a
    row the cache-aware fit then trusts — a fabricated observation, and the one kind of error the
    fit cannot detect, since a real all-miss run looks identical."""
    eng, queue, writer = engine
    queue.append(_fake_response(prompt=5000, cached=None, completion=250, reasoning=None))

    _text, metadata = eng.chat_completion([{"role": "user", "content": "hi"}])

    (row,) = _rows(writer.path)
    assert row["cached_input_tokens"] is None
    assert row["reasoning_tokens"] is None
    # The usage block itself *was* read, so the row is not a capture failure: the counts it does
    # carry are real and usable.
    assert row["usage_captured"] is True
    assert row["input_tokens"] == 5000
    assert metadata is not None
    assert metadata["cached_prompt_tokens"] is None
    # ARE's own key is the exception: it lands in an int field of `LLMOutputThoughtActionLog` and
    # is collected into a list by its exporter, so the unknown flattens to 0 there. The row above
    # is where the distinction is kept.
    assert metadata["reasoning_tokens"] == 0


@requires_are
def test_zero_cache_read_is_kept_as_a_measurement(
    engine: tuple[Any, list[Any], LLMCallWriter],
) -> None:
    """The other side of the same distinction: a provider that *reports* ``cached_tokens: 0``
    measured a miss, and that has to survive as 0 rather than being folded into unknown."""
    eng, queue, writer = engine
    queue.append(_fake_response(prompt=100, cached=0, completion=10, reasoning=None))

    eng.chat_completion([{"role": "user", "content": "hi"}])

    assert _rows(writer.path)[0]["cached_input_tokens"] == 0


@requires_are
def test_retry_inside_one_bracket_accumulates(engine: tuple[Any, list[Any], LLMCallWriter]) -> None:
    """``step()`` keeps only the last metadata, so the bracket total has to be *in* it. Without
    this the tokens of every malformed-output retry are charged at zero."""
    eng, queue, writer = engine
    queue.extend(
        [
            _fake_response(prompt=4000, cached=0, completion=100, reasoning=0),
            _fake_response(prompt=4200, cached=0, completion=300, reasoning=0),
        ]
    )

    pause_calls: list[int] = []
    paused = eng.wrap_pause_env(lambda: pause_calls.append(1))
    paused()  # ARE pauses once at the top of step(), then may call the engine repeatedly
    eng.chat_completion([{"role": "user", "content": "a"}])
    _text, metadata = eng.chat_completion([{"role": "user", "content": "b"}])

    assert pause_calls == [1]
    assert metadata is not None
    assert metadata["round_trips"] == 2
    assert metadata["prompt_tokens"] == 8200
    assert metadata["completion_tokens"] == 400
    # Both round-trips reach the JSONL as their own rows regardless: it is the source of truth for
    # the fit, and stays complete even where ARE's single float cannot be.
    assert len(_rows(writer.path)) == 2


@requires_are
def test_retry_rows_carry_the_bracket_that_produced_them(
    engine: tuple[Any, list[Any], LLMCallWriter],
) -> None:
    """The retry's row and the first attempt's row have to be groupable *in the file*.

    Nothing else survives the run: ARE keeps only the last metadata a bracket returned, and
    ``LLMOutputThoughtActionLog`` stores a fixed set of fields that does not include the engine's
    ``round_trips``. Without the id on the row, one retried decision is indistinguishable from two
    decisions, and a per-decision cost read off the file is wrong by a whole round-trip.
    """
    eng, queue, writer = engine
    queue.extend(
        _fake_response(prompt=4000, cached=0, completion=100, reasoning=0) for _ in range(3)
    )
    paused = eng.wrap_pause_env(lambda: None)

    paused()
    eng.chat_completion([{"role": "user", "content": "a"}])
    eng.chat_completion([{"role": "user", "content": "b"}])  # ARE re-calling after a bad parse
    paused()
    eng.chat_completion([{"role": "user", "content": "c"}])

    first, retry, next_step = _rows(writer.path)
    assert first["bracket_id"] == retry["bracket_id"]
    assert next_step["bracket_id"] != first["bracket_id"]
    # Still one row each — the bracket groups them, it does not merge them.
    assert first["call_id"] != retry["call_id"]


@requires_are
def test_next_bracket_starts_clean(engine: tuple[Any, list[Any], LLMCallWriter]) -> None:
    """The accumulation is per generation bracket, not per run — otherwise every step would be
    charged the whole trajectory so far."""
    eng, queue, writer = engine
    queue.extend(
        [
            _fake_response(prompt=4000, cached=0, completion=100, reasoning=0),
            _fake_response(prompt=9000, cached=0, completion=200, reasoning=0),
        ]
    )
    paused = eng.wrap_pause_env(lambda: None)

    paused()
    eng.chat_completion([{"role": "user", "content": "a"}])
    paused()
    _text, metadata = eng.chat_completion([{"role": "user", "content": "b"}])

    assert metadata is not None
    assert metadata["round_trips"] == 1
    assert metadata["prompt_tokens"] == 9000


@requires_are
def test_unwired_brackets_undercount_rather_than_double_charge(
    engine: tuple[Any, list[Any], LLMCallWriter],
) -> None:
    """Nothing calls ``begin_bracket``, so the engine cannot know a bracket ended. It reports the
    single round-trip — losing a retry's tokens, the known ARE behaviour — instead of growing
    without bound, which would charge each step for the whole run."""
    eng, queue, _writer = engine
    queue.extend(
        [
            _fake_response(prompt=4000, cached=0, completion=100, reasoning=0),
            _fake_response(prompt=4000, cached=0, completion=100, reasoning=0),
        ]
    )
    eng.chat_completion([{"role": "user", "content": "a"}])
    _text, metadata = eng.chat_completion([{"role": "user", "content": "b"}])

    assert metadata is not None
    assert metadata["round_trips"] == 1
    assert metadata["prompt_tokens"] == 4000


@requires_are
def test_charge_model_replaces_measured_latency(
    engine: tuple[Any, list[Any], LLMCallWriter],
) -> None:
    """The point of the class beyond instrumentation: the frozen coefficients, not wall clock, are
    what ARE's clock advances by — computed in one place for both arms."""
    eng, queue, writer = engine
    eng.charge = lambda inp, cached, out: 1.0 + inp / 10_000 + out / 100
    queue.append(_fake_response(prompt=20_000, cached=0, completion=500, reasoning=0))

    _text, metadata = eng.chat_completion([{"role": "user", "content": "a"}])

    assert metadata is not None
    assert metadata["completion_duration"] == pytest.approx(1.0 + 2.0 + 5.0)
    assert _rows(writer.path)[0]["charged_seconds"] == pytest.approx(8.0)
    # The measurement survives beside the charge: a pricing rule has to stay auditable against it.
    assert metadata["measured_seconds"] > 0


@requires_are
def test_provider_error_is_still_recorded(engine: tuple[Any, list[Any], LLMCallWriter]) -> None:
    """A failed round-trip was still emitted. It reaches the JSONL flagged, so the charge can
    include it while the fit — which would otherwise be measuring the provider's bad day —
    excludes it. This is the *no response at all* case: nothing was captured, so the tokens are
    genuinely unknown rather than discarded."""
    eng, _queue, writer = engine  # empty queue: the double raises IndexError

    with pytest.raises(IndexError):
        eng.chat_completion([{"role": "user", "content": "a"}])

    (row,) = _rows(writer.path)
    assert row["finish_reason"].startswith("error:")
    assert row["usage_captured"] is False
    assert row["seconds"] is not None
    # Charged, not free: measured wall clock with no charge model wired.
    assert row["charged_seconds"] == pytest.approx(row["seconds"])


@requires_are
def test_a_rejected_response_still_reports_what_it_cost(
    engine: tuple[Any, list[Any], LLMCallWriter],
) -> None:
    """ARE raises *after* the provider answered, and that answer was billed.

    ``LiteLLMEngine.chat_completion`` asserts its way through the response once ``completion()``
    has returned — ``assert res is not None`` is the one a content-filtered reply trips. The
    capture already holds the usage block at that point, so discarding it would drop a real, paid
    crossing out of the fit's denominator and out of the bill.
    """
    from litellm.types.utils import Choices, Message

    eng, queue, writer = engine
    rejected = _fake_response(prompt=123, cached=None, completion=7, reasoning=None)
    rejected.choices = [Choices(message=Message(content=None), finish_reason="content_filter")]
    queue.append(rejected)

    with pytest.raises(AssertionError):
        eng.chat_completion([{"role": "user", "content": "a"}])

    (row,) = _rows(writer.path)
    assert row["input_tokens"] == 123 and row["output_tokens"] == 7
    assert row["usage_captured"] is True
    # Still marked an error, so the fit excludes it — the row is for the bill and the audit.
    assert row["finish_reason"].startswith("error:")


@requires_are
def test_a_failed_round_trip_is_charged_its_fixed_term(
    engine: tuple[Any, list[Any], LLMCallWriter],
) -> None:
    """With a charge model wired, a crossing that reported no tokens still bills its per-call
    term — the coefficient that exists precisely to price the part of a call that is not tokens."""
    eng, _queue, writer = engine  # empty queue: the double raises IndexError
    eng.charge = lambda inp, cached, out: 3.0 + inp / 10_000 + out / 100

    with pytest.raises(IndexError):
        eng.chat_completion([{"role": "user", "content": "a"}])

    assert _rows(writer.path)[0]["charged_seconds"] == pytest.approx(3.0)
