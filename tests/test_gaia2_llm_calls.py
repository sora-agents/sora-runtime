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

import asyncio
import importlib.util
import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from examples.gaia2.evaluation.core import ChargeCoefficients, TotalInputCharge
from examples.gaia2.llm_calls import (
    ClockedSoraLLMClient,
    LLMCallRecord,
    LLMCallWriter,
    RoundTripWindow,
    SoraCallRecorder,
    charged_time_union_seconds,
    round_trip_concurrency,
    wall_time_union_seconds,
)

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


def _charge() -> TotalInputCharge:
    return TotalInputCharge(
        ChargeCoefficients(
            model="model",
            provider="provider",
            provider_routing_json=None,
            a0_seconds=1.0,
            seconds_per_uncached_input_token=0.1,
            seconds_per_cached_input_token=0.01,
            seconds_per_output_token=0.2,
            decode_includes_reasoning=True,
            fit={},
            note="test",
        )
    )


class _GenerationClock:
    def __init__(self) -> None:
        self.next_token = 0
        self.active: set[int] = set()
        self.completed: list[tuple[int, float]] = []
        self.charge_time = 0.0
        self.pause_origin: float | None = None
        self.pending_charge = 0.0
        self.parallel_frontier = 0.0
        self.charge_starts: dict[int, float] = {}

    def pause_generation(self) -> int:
        if not self.active:
            self.pause_origin = self.charge_time
            self.pending_charge = 0.0
            self.parallel_frontier = self.charge_time
        self.next_token += 1
        self.active.add(self.next_token)
        self.charge_starts[self.next_token] = self.parallel_frontier
        return self.next_token

    def generation_charge_time(self, token: int) -> float:
        return self.charge_starts[token]

    def resume_generation(self, token: int, offset: float) -> None:
        self.active.remove(token)
        self.completed.append((token, offset))
        self.pending_charge += offset
        self.parallel_frontier = max(self.parallel_frontier, self.charge_starts.pop(token) + offset)
        if not self.active:
            assert self.pause_origin is not None
            self.charge_time = self.pause_origin + self.pending_charge
            self.pause_origin = None


def test_wall_time_union_counts_overlapping_crossings_once() -> None:
    windows = (
        RoundTripWindow(started_at=10.0, finished_at=20.0, charged_seconds=7.0),
        RoundTripWindow(started_at=12.0, finished_at=14.0, charged_seconds=2.0),
        RoundTripWindow(started_at=25.0, finished_at=28.0, charged_seconds=3.0),
    )

    assert sum(window.wall_seconds for window in windows) == 15.0
    assert wall_time_union_seconds(windows) == 13.0


def test_charged_union_uses_charge_duration_not_provider_latency() -> None:
    windows = (
        RoundTripWindow(
            started_at=10.0,
            finished_at=11.0,
            charged_seconds=4.0,
            charged_started_at=2.0,
        ),
        RoundTripWindow(
            started_at=10.7,
            finished_at=11.0,
            charged_seconds=4.0,
            charged_started_at=2.0,
        ),
    )

    assert sum(window.charged_seconds for window in windows) == 8.0
    assert wall_time_union_seconds(windows) == pytest.approx(1.0)
    assert charged_time_union_seconds(windows) == pytest.approx(4.0)


def test_round_trip_concurrency_counts_depth_and_each_overlapped_crossing() -> None:
    windows = (
        RoundTripWindow(0.0, 4.0, 1.0),
        RoundTripWindow(1.0, 2.0, 1.0),
        RoundTripWindow(2.0, 3.0, 1.0),  # touches the second; overlaps only the first
        RoundTripWindow(5.0, 6.0, 1.0),
    )

    stats = round_trip_concurrency(windows)

    assert stats.round_trips == 4
    assert stats.max_in_flight == 2
    assert stats.overlapped_round_trips == 3


async def test_sora_parser_repair_charges_each_physical_round_trip(tmp_path: Path) -> None:
    class _Client:
        async def complete(self, request: CompletionRequest) -> str:
            log_llm_usage(
                LLMUsage(input_tokens=10, cached_input_tokens=4, output_tokens=2, answer_chars=2),
                request,
            )
            return "{}"

    writer = LLMCallWriter(tmp_path / "calls.jsonl").open()
    charge = _charge()
    recorder = SoraCallRecorder(writer, charge=charge)
    clock = _GenerationClock()
    client = ClockedSoraLLMClient(MeteredLLMClient(_Client()), clock=clock, recorder=recorder)
    log = logging.getLogger("sora")
    log.setLevel(logging.INFO)
    log.addHandler(recorder)
    try:
        with llm_call_scope():
            await client.complete(_request())
            await client.complete(_request())
    finally:
        log.removeHandler(recorder)
        recorder.close()
        writer.close()

    (row,) = _rows(writer.path)
    per_trip = 1.0 + 6 * 0.1 + 4 * 0.01 + 2 * 0.2
    assert row["round_trips"] == 2
    assert len(row["round_trip_windows"]) == 2
    assert [window["charged_started_at"] for window in row["round_trip_windows"]] == pytest.approx(
        [0.0, per_trip]
    )
    assert charged_time_union_seconds(recorder.round_trip_windows) == pytest.approx(2 * per_trip)
    assert row["charged_seconds"] == pytest.approx(2 * per_trip)
    assert charge.charged_seconds == pytest.approx(2 * per_trip)
    assert [token for token, _seconds in clock.completed] == [1, 2]
    assert [seconds for _token, seconds in clock.completed] == pytest.approx([per_trip, per_trip])
    # Closing flushes rows but must not erase the offline sensitivity source.
    assert len(recorder.round_trip_windows) == 2
    assert recorder.wall_seconds > 0.0


async def test_concurrent_sora_calls_stay_paused_until_reverse_order_completion(
    tmp_path: Path,
) -> None:
    started = {name: asyncio.Event() for name in ("first", "second")}
    release = {name: asyncio.Event() for name in ("first", "second")}

    class _Client:
        async def complete(self, request: CompletionRequest) -> str:
            started[request.user].set()
            await release[request.user].wait()
            log_llm_usage(LLMUsage(input_tokens=1, output_tokens=1, answer_chars=1), request)
            return "{}"

    writer = LLMCallWriter(tmp_path / "calls.jsonl").open()
    recorder = SoraCallRecorder(writer, charge=_charge())
    clock = _GenerationClock()
    client = ClockedSoraLLMClient(MeteredLLMClient(_Client()), clock=clock, recorder=recorder)
    log = logging.getLogger("sora")
    log.setLevel(logging.INFO)
    log.addHandler(recorder)
    requests = [
        CompletionRequest(system="s", user=name, semantic_label="plan", prompt_version="1")
        for name in ("first", "second")
    ]
    wall_seconds = 0.0
    wall_union_seconds = 0.0
    charged_union_seconds: float | None = 0.0
    try:
        tasks = [asyncio.create_task(client.complete(request)) for request in requests]
        await asyncio.gather(*(event.wait() for event in started.values()))
        assert len(clock.active) == 2
        release["second"].set()
        await tasks[1]
        assert len(clock.active) == 1
        release["first"].set()
        await tasks[0]
        wall_seconds = recorder.wall_seconds
        wall_union_seconds = recorder.wall_union_seconds
        charged_union_seconds = recorder.charged_union_seconds
    finally:
        log.removeHandler(recorder)
        recorder.close()
        writer.close()

    assert [token for token, _charge_seconds in clock.completed] == [2, 1]
    rows = _rows(writer.path)
    assert len(rows) == 2
    assert all(len(row["round_trip_windows"]) == 1 for row in rows)
    assert {row["round_trip_windows"][0]["charged_started_at"] for row in rows} == {0.0}
    assert wall_union_seconds < wall_seconds
    assert charged_union_seconds is not None
    assert charged_union_seconds < recorder.charged_seconds
    assert recorder.concurrency.round_trips == 2
    assert recorder.concurrency.max_in_flight == 2
    assert recorder.concurrency.overlapped_round_trips == 2


async def test_sora_failure_without_usage_still_pays_the_fixed_term(tmp_path: Path) -> None:
    class _Client:
        async def complete(self, request: CompletionRequest) -> str:
            raise RuntimeError("provider failed")

    writer = LLMCallWriter(tmp_path / "calls.jsonl").open()
    charge = _charge()
    recorder = SoraCallRecorder(writer, charge=charge)
    clock = _GenerationClock()
    client = ClockedSoraLLMClient(MeteredLLMClient(_Client()), clock=clock, recorder=recorder)
    log = logging.getLogger("sora")
    log.setLevel(logging.INFO)
    log.addHandler(recorder)
    try:
        with pytest.raises(RuntimeError, match="provider failed"):
            await client.complete(_request())
    finally:
        log.removeHandler(recorder)
        recorder.close()
        writer.close()

    (row,) = _rows(writer.path)
    assert row["usage_captured"] is False
    assert row["charged_seconds"] == pytest.approx(1.0)
    assert clock.completed == [(1, 1.0)]


async def test_wall_clock_sora_records_a_window_without_pausing_or_charging(tmp_path: Path) -> None:
    class _Client:
        async def complete(self, request: CompletionRequest) -> str:
            log_llm_usage(LLMUsage(input_tokens=2, output_tokens=1, answer_chars=1), request)
            return "{}"

    writer = LLMCallWriter(tmp_path / "calls.jsonl").open()
    recorder = SoraCallRecorder(writer, charge=None)
    client = ClockedSoraLLMClient(MeteredLLMClient(_Client()), clock=None, recorder=recorder)
    log = logging.getLogger("sora")
    log.setLevel(logging.INFO)
    log.addHandler(recorder)
    try:
        await client.complete(_request())
    finally:
        log.removeHandler(recorder)
        recorder.close()
        writer.close()

    (row,) = _rows(writer.path)
    assert row["charged_seconds"] is None
    assert len(row["round_trip_windows"]) == 1
    assert row["round_trip_windows"][0]["charged_seconds"] == 0.0


async def test_sora_cache_clamp_has_an_independent_raw_anomaly_count(tmp_path: Path) -> None:
    class _Client:
        async def complete(self, request: CompletionRequest) -> str:
            log_llm_usage(
                LLMUsage(input_tokens=5, cached_input_tokens=8, output_tokens=1, answer_chars=1),
                request,
            )
            return "{}"

    charge = _charge()
    recorder = SoraCallRecorder(None, charge=charge)
    client = ClockedSoraLLMClient(
        MeteredLLMClient(_Client()), clock=_GenerationClock(), recorder=recorder
    )
    log = logging.getLogger("sora")
    log.setLevel(logging.INFO)
    log.addHandler(recorder)
    try:
        await client.complete(_request())
    finally:
        log.removeHandler(recorder)
        recorder.close()

    assert charge.cached_input_clamps == 1
    assert recorder.raw_cached_input_anomalies == 1


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
    row = _rows(writer.path)[0]
    assert row["charged_seconds"] == pytest.approx(8.0)
    assert len(row["round_trip_windows"]) == 1
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


# -- the ReAct arm, streamed ---------------------------------------------------------------------

# Streaming is not a preference here. S-ORA's client streams by default — its stall timeout means
# "the provider went quiet", which is only observable on a streamed call — so a non-streaming ReAct
# arm differs from it in transport, on exactly the per-arm comparison the charge model exists to
# make. ARE's own engine hardcodes a non-streaming `litellm.completion()` and ignores its
# `**kwargs`, so both the streaming and the settings have to be applied at the interception seam.


def _stream_chunks(text: str, *, usage: Any = None, finish_reason: str = "stop") -> list[Any]:
    """Real LiteLLM stream chunks, not stand-ins.

    ``stream_chunk_builder`` reaches into ``_hidden_params`` and indexes chunks as mappings, and
    the response it returns has to satisfy ARE's ``type(response) is ModelResponse`` — a duck-typed
    double would pass a test that the installed LiteLLM would fail."""
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    def chunk(**choice: Any) -> Any:
        return ModelResponseStream(
            id="chatcmpl-stream", model="test/model", choices=[StreamingChoices(index=0, **choice)]
        )

    # Sliced, not split: a split on whitespace would drop the separators and the test would then
    # be asserting on text the provider never sent.
    chunks = [
        chunk(delta=Delta(role="assistant", content=text[i : i + 4]))
        for i in range(0, len(text), 4)
    ]
    chunks.append(chunk(delta=Delta(content=None), finish_reason=finish_reason))
    if usage is not None:
        # The usage block rides a trailing chunk carrying no choices at all.
        chunks.append(
            ModelResponseStream(id="chatcmpl-stream", model="test/model", choices=[], usage=usage)
        )
    return chunks


def _stream_usage(prompt: int, cached: int | None, completion: int) -> Any:
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage

    details = PromptTokensDetailsWrapper(cached_tokens=cached) if cached is not None else None
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=details,
    )


@pytest.fixture
def streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Any, list[Any], list[dict[str, Any]], LLMCallWriter]]:
    """A streaming metered engine, plus the queue it draws from and every kwarg dict the provider
    call was made with."""
    from are.simulation.agents.llm.litellm import litellm_engine as module
    from are.simulation.agents.llm.litellm.litellm_engine import LiteLLMModelConfig
    from examples.gaia2.react_engine import MeteredLiteLLMEngine

    queue: list[Any] = []
    seen: list[dict[str, Any]] = []

    def fake_completion(**kwargs: Any) -> Any:
        seen.append(kwargs)
        nxt = queue.pop(0)
        return iter(nxt) if kwargs.get("stream") else nxt

    monkeypatch.setattr(module, "completion", fake_completion)
    writer = LLMCallWriter(tmp_path / "react.jsonl").open()
    eng = MeteredLiteLLMEngine(
        LiteLLMModelConfig(model_name="test/model", provider="local"),
        writer=writer,
        scenario_id="scen-s",
        request_kwargs={"reasoning_effort": "high", "max_completion_tokens": 16384},
        stream=True,
    )
    try:
        yield eng, queue, seen, writer
    finally:
        writer.close()


@requires_are
def test_a_streamed_call_reaches_are_as_an_ordinary_response(
    streaming: tuple[Any, list[Any], list[dict[str, Any]], LLMCallWriter],
) -> None:
    """ARE's body runs unchanged over the reassembled response — including the ``True``/``False``
    lowercasing its own checkers depend on, which is the reason the chunks are rebuilt rather than
    joined by hand here."""
    eng, queue, seen, writer = streaming
    queue.append(_stream_chunks("Thought: True", usage=_stream_usage(1200, None, 40)))

    text, metadata = eng.chat_completion([{"role": "user", "content": "a"}])

    assert text == "Thought: true"
    assert seen[0]["stream"] is True
    assert seen[0]["stream_options"] == {"include_usage": True}
    assert metadata["prompt_tokens"] == 1200 and metadata["completion_tokens"] == 40
    (row,) = _rows(writer.path)
    assert row["input_tokens"] == 1200 and row["output_tokens"] == 40
    assert row["finish_reason"] == "stop"
    assert row["usage_captured"] is True


@requires_are
def test_a_streamed_cache_read_survives_the_rebuild(
    streaming: tuple[Any, list[Any], list[dict[str, Any]], LLMCallWriter],
) -> None:
    """The cached count is what makes the ReAct arm's growing prefix affordable to charge, and it
    rides the same trailing chunk as the totals."""
    eng, queue, _seen, writer = streaming
    queue.append(_stream_chunks("Thought: ok", usage=_stream_usage(9000, 8192, 30)))

    eng.chat_completion([{"role": "user", "content": "a"}])

    (row,) = _rows(writer.path)
    assert row["input_tokens"] == 9000 and row["cached_input_tokens"] == 8192


@requires_are
def test_a_stream_without_usage_is_never_estimated(
    streaming: tuple[Any, list[Any], list[dict[str, Any]], LLMCallWriter],
) -> None:
    """The trap this guard exists for.

    ``stream_chunk_builder`` fills a *missing* usage block by re-tokenizing prompt and completion
    locally, so a provider that ignores ``include_usage`` yields counts that are plausible, wrong,
    and indistinguishable from reported ones once written to a row. The first assertion pins that
    the estimate really is produced — without it this test would pass for the wrong reason if
    LiteLLM ever started returning zeros instead."""
    from litellm import stream_chunk_builder

    eng, queue, _seen, writer = streaming
    chunks = _stream_chunks("Thought: ok", usage=None)
    rebuilt = stream_chunk_builder(list(chunks), messages=[{"role": "user", "content": "a " * 200}])
    assert rebuilt is not None and rebuilt.usage.prompt_tokens > 0  # the estimate, not a report

    queue.append(chunks)
    eng.chat_completion([{"role": "user", "content": "a " * 200}])

    (row,) = _rows(writer.path)
    assert row["usage_captured"] is False
    assert row["input_tokens"] is None and row["output_tokens"] is None
    # Still a paid crossing, and still charged — undercounted loudly, never dropped.
    assert row["seconds"] is not None


@requires_are
def test_a_stream_that_dies_midway_is_still_recorded(
    streaming: tuple[Any, list[Any], list[dict[str, Any]], LLMCallWriter],
) -> None:
    """Tokens streamed before the failure were billed by the provider, but nothing reported them:
    the usage chunk never arrived. The row says so rather than guessing."""

    def dying() -> Iterator[Any]:
        yield from _stream_chunks("Thought: ok", usage=None)[:1]
        raise TimeoutError("stream stalled")

    eng, queue, _seen, writer = streaming
    queue.append(dying())

    with pytest.raises(TimeoutError):
        eng.chat_completion([{"role": "user", "content": "a"}])

    (row,) = _rows(writer.path)
    assert row["finish_reason"] == "error:TimeoutError"
    assert row["usage_captured"] is False


@requires_are
def test_profile_settings_ride_the_call_are_would_send_bare(
    streaming: tuple[Any, list[Any], list[dict[str, Any]], LLMCallWriter],
) -> None:
    """ARE hands ``litellm.completion`` five fixed arguments and drops its own ``**kwargs``, so
    without this the baseline runs at the provider's defaults while the other arm runs at the
    profile's."""
    eng, queue, seen, _writer = streaming
    queue.append(_stream_chunks("ok", usage=_stream_usage(10, None, 2)))

    eng.chat_completion([{"role": "user", "content": "a"}])

    assert seen[0]["reasoning_effort"] == "high"
    assert seen[0]["max_completion_tokens"] == 16384
    # ARE's own five are untouched.
    assert seen[0]["model"] == "test/model" and "messages" in seen[0]


@requires_are
def test_a_non_streaming_engine_sends_no_stream_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A profile that does not stream must reach the provider as the request it always was — the
    flag is added by the wrapper, so its absence has to be asserted, not assumed."""
    from are.simulation.agents.llm.litellm import litellm_engine as module
    from are.simulation.agents.llm.litellm.litellm_engine import LiteLLMModelConfig
    from examples.gaia2.react_engine import MeteredLiteLLMEngine

    seen: list[dict[str, Any]] = []

    def fake_completion(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return _fake_response(prompt=10, cached=None, completion=2, reasoning=None)

    monkeypatch.setattr(module, "completion", fake_completion)
    writer = LLMCallWriter(tmp_path / "react.jsonl").open()
    eng = MeteredLiteLLMEngine(
        LiteLLMModelConfig(model_name="test/model", provider="local"),
        writer=writer,
        request_kwargs={"temperature": 0.5},
    )
    try:
        eng.chat_completion([{"role": "user", "content": "a"}])
    finally:
        writer.close()

    assert "stream" not in seen[0] and "stream_options" not in seen[0]
    assert seen[0]["temperature"] == 0.5


@requires_are
def test_settings_do_not_leak_to_an_unmetered_engine(
    streaming: tuple[Any, list[Any], list[dict[str, Any]], LLMCallWriter],
) -> None:
    """The wrapper is a module global and the settings live on a thread-local. Left set, this
    engine's operating point would silently reach whatever calls ``completion`` next on the same
    thread — a stock ``LiteLLMEngine``, another harness — which is exactly the kind of
    cross-contamination that makes two arms look like one."""
    from are.simulation.agents.llm.litellm.litellm_engine import (
        LiteLLMEngine,
        LiteLLMModelConfig,
    )

    eng, queue, seen, _writer = streaming
    queue.append(_stream_chunks("ok", usage=_stream_usage(10, None, 2)))
    eng.chat_completion([{"role": "user", "content": "a"}])

    queue.append(_fake_response(prompt=1, cached=None, completion=1, reasoning=None))
    plain = LiteLLMEngine(LiteLLMModelConfig(model_name="other/model", provider="local"))
    plain.chat_completion([{"role": "user", "content": "b"}])

    assert "stream" not in seen[1] and "reasoning_effort" not in seen[1]


@requires_are
def test_from_profile_runs_the_arm_at_the_profiles_operating_point() -> None:
    """The constructor a driver should reach for: the operating point comes from the same profile
    the S-ORA arm is configured from, so the two cannot drift apart by omission."""
    from examples.gaia2.evaluation.core import load_profiles
    from examples.gaia2.react_engine import MeteredLiteLLMEngine

    root = Path(__file__).parents[1] / "examples" / "gaia2" / "evaluation"
    kimi = load_profiles(root / "profiles.json")["kimi-k2.5-prompt"]

    eng = MeteredLiteLLMEngine.from_profile(kimi, scenario_id="scen-k")

    assert eng.model_config.model_name == "moonshotai/kimi-k2.5"
    assert eng.model_config.endpoint == "https://openrouter.ai/api/v1"
    # Transport follows the profile, which the latency grid reads too — coefficients fitted on one
    # transport do not price an arm running on another.
    assert eng.stream is False
    assert (
        MeteredLiteLLMEngine.from_profile(replace(kimi, stream=True), scenario_id="scen-k").stream
        is True
    )
    sent = dict(eng.settings.request_kwargs)
    assert sent["temperature"] == 0.5
    assert sent["max_completion_tokens"] == 16384
    assert sent["extra_body"]["reasoning"] == {"enabled": True}
    assert sent["extra_body"]["provider"]["allow_fallbacks"] is False
    assert sent["extra_headers"] == {"X-OpenRouter-Metadata": "enabled"}
    # Transport, added here rather than by the profile's operating point. Non-streamed, the
    # timeout is a total-duration cap rather than an inter-chunk silence bound.
    assert sent["timeout"] == 600 and sent["num_retries"] == 0
    # Omitted settings are absent, never present-and-null.
    assert "reasoning_effort" not in sent and "top_p" not in sent


def test_charged_union_is_unavailable_when_the_collection_mixes_time_axes() -> None:
    """Scenario-elapsed and host-monotonic starts cannot share an interval sweep.

    A run produces the mixture for real: a crossing admitted after the online judge stopped the
    environment gets the no-op token and so carries no charged coordinate, while every earlier
    crossing in the same run does. Sweeping both together yields a number in no coordinate system,
    so the union is declared unavailable instead.
    """
    charged = RoundTripWindow(
        started_at=1_000_000.0, finished_at=1_000_002.0, charged_seconds=2.0, charged_started_at=5.0
    )
    uncharted = RoundTripWindow(
        started_at=1_000_003.0, finished_at=1_000_004.0, charged_seconds=1.0
    )

    assert charged_time_union_seconds([charged, uncharted]) is None
    # Each half on its own is a single coherent axis and still reports.
    assert charged_time_union_seconds([charged]) == pytest.approx(2.0)
    assert charged_time_union_seconds([uncharted]) == pytest.approx(1.0)
    assert charged_time_union_seconds([]) == 0.0


async def test_a_failing_charge_probe_still_releases_the_frozen_world(tmp_path: Path) -> None:
    """A raise between the freeze and the `try` used to park the scenario, not fail the call.

    ARE's loop thread waits on `pause_event` with no timeout, so a crossing that acquires the
    freeze and then dies before reaching its resume does not surface as an inference error -- the
    environment simply never ticks again, and only the wall-clock watchdog ends the run.
    """

    class _Client:
        async def complete(self, request: CompletionRequest) -> str:
            raise AssertionError("the probe should fail before the provider is reached")

    class _UnreadableClock(_GenerationClock):
        def generation_charge_time(self, token: int) -> float:
            raise RuntimeError("charge axis unavailable")

    writer = LLMCallWriter(tmp_path / "calls.jsonl").open()
    recorder = SoraCallRecorder(writer, charge=_charge())
    clock = _UnreadableClock()
    client = ClockedSoraLLMClient(MeteredLLMClient(_Client()), clock=clock, recorder=recorder)
    try:
        with pytest.raises(RuntimeError, match="charge axis unavailable"):
            await client.complete(_request())
    finally:
        recorder.close()
        writer.close()

    assert clock.active == set(), "the freeze outlived the crossing that took it"
    assert clock.completed == [(1, 1.0)], "the failed crossing still owes its fixed term"


async def test_a_failing_recorder_still_releases_the_frozen_world(tmp_path: Path) -> None:
    """The same leak on the other side of the call: teardown must reach `resume_generation`."""

    class _Client:
        async def complete(self, request: CompletionRequest) -> str:
            return "answer"

    class _BrokenRecorder(SoraCallRecorder):
        def finish_round_trip(self, *args: Any, **kwargs: Any) -> float:
            raise RuntimeError("recorder failed")

    writer = LLMCallWriter(tmp_path / "calls.jsonl").open()
    recorder = _BrokenRecorder(writer, charge=_charge())
    clock = _GenerationClock()
    client = ClockedSoraLLMClient(MeteredLLMClient(_Client()), clock=clock, recorder=recorder)
    try:
        with pytest.raises(RuntimeError, match="recorder failed"):
            await client.complete(_request())
    finally:
        writer.close()

    assert clock.active == set()
    # Zero is the deliberate direction: an uncharged call understates simulated time, where a
    # leaked freeze would hang the scenario outright.
    assert clock.completed == [(1, 0.0)]


# -- the generation-free clock on the S-ORA arm --------------------------------------------------


async def test_generation_free_freezes_the_sora_clock_and_resumes_it_by_zero(
    tmp_path: Path,
) -> None:
    """The branch this pins is the one that made the mode worth building.

    A generation-free run reaches every S-ORA call site with ``charge=None`` — exactly as a
    wall-clock run does — and before the mode was named, that meant no clock was handed to the
    client at all, so the scenario ran *unfrozen* while its artifact said otherwise. The claim has
    two halves and both matter: the scenario is stopped for the duration of the crossing, and it is
    resumed by zero rather than by measured latency."""

    class _Client:
        async def complete(self, request: CompletionRequest) -> str:
            log_llm_usage(
                LLMUsage(
                    input_tokens=900, cached_input_tokens=0, output_tokens=400, answer_chars=2
                ),
                request,
            )
            return "{}"

    writer = LLMCallWriter(tmp_path / "calls.jsonl").open()
    recorder = SoraCallRecorder(writer, charge=None, generation_free=True)
    clock = _GenerationClock()
    client = ClockedSoraLLMClient(MeteredLLMClient(_Client()), clock=clock, recorder=recorder)
    log = logging.getLogger("sora")
    log.setLevel(logging.INFO)
    log.addHandler(recorder)
    try:
        with llm_call_scope():
            await client.complete(_request())
    finally:
        log.removeHandler(recorder)
        recorder.close()
        writer.close()

    # Frozen: the crossing took a pause token, and gave it back.
    assert clock.active == set()
    # Resumed by zero — not by the call's measured latency, which is what an unfrozen run would
    # have charged and what makes a wall-clock result irreproducible across serving endpoints.
    assert clock.completed == [(1, 0.0)]

    row = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[-1])
    # 0.0, not None: under this clock zero is an applied policy, where None means no clock billed
    # the crossing at all. A row that cannot tell those apart cannot state its own convention.
    assert row["charged_seconds"] == 0.0


async def test_a_wall_clock_sora_run_is_handed_no_clock_at_all(tmp_path: Path) -> None:
    """The contrast case: same ``charge=None``, no freeze, and nothing billed.

    Kept beside the test above because the two are one decision read two ways, and the bug was
    that only this behavior existed."""

    class _Client:
        async def complete(self, request: CompletionRequest) -> str:
            log_llm_usage(
                LLMUsage(
                    input_tokens=900, cached_input_tokens=0, output_tokens=400, answer_chars=2
                ),
                request,
            )
            return "{}"

    writer = LLMCallWriter(tmp_path / "calls.jsonl").open()
    recorder = SoraCallRecorder(writer, charge=None)
    client = ClockedSoraLLMClient(MeteredLLMClient(_Client()), clock=None, recorder=recorder)
    log = logging.getLogger("sora")
    log.setLevel(logging.INFO)
    log.addHandler(recorder)
    try:
        with llm_call_scope():
            await client.complete(_request())
    finally:
        log.removeHandler(recorder)
        recorder.close()
        writer.close()

    row = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[-1])
    assert row["charged_seconds"] is None


def test_both_frozen_modes_freeze_and_only_the_unfrozen_one_does_not() -> None:
    """The predicate the S-ORA arm reads to decide whether to hand the client a clock. Derived from
    the recorded mode rather than from ``charge is not None``, so a row cannot be labelled
    ``generation_free`` and have run unfrozen."""
    from examples.gaia2.evaluation.core import (
        CLOCK_MODE_GENERATION_FREE,
        CLOCK_MODE_TOKEN_CHARGED,
        CLOCK_MODE_WALL,
        freezes_clock,
    )

    assert freezes_clock(CLOCK_MODE_TOKEN_CHARGED) is True
    assert freezes_clock(CLOCK_MODE_GENERATION_FREE) is True
    assert freezes_clock(CLOCK_MODE_WALL) is False
