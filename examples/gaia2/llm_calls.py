"""Per-call model records for both benchmark arms, written as one JSONL file per run.

Why this exists
---------------
The charge model that freezes ARE's clock during generation is
``charge = a0 + uncached_input/R_in + cached_input/R_cache + output/R_out``, computed identically
for both the S-ORA arm and the ReAct baseline. Computing it — and, before that, validating the
coefficients against calls the two arms really made — needs one row per model round-trip with the
token counts, the measured latency and enough identity to say which arm and which call site
produced it.

Neither arm's native run artifact records that. S-ORA emits the numbers as ``sora.llm`` log
records, but their *text* form drops ``cached_input_tokens`` — which the cache-aware fit needs —
and ARE's ReAct agent discards its token counts entirely (see ``react_engine``). This module is the
one durable schema both harnesses write into, so a run's calls are comparable across arms without
a per-arm parser.

The JSONL is the source of truth for the fit and for billing. It is deliberately *not* the clock
control channel: the harness resumes each arm's pause bracket from its own per-crossing account,
while every physical round-trip lands here regardless.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Protocol

from sora.llm import CompletionRequest, LLMClient, current_llm_call_id, llm_call_scope

Arm = Literal["sora", "react", "grid"]


class Charge(Protocol):
    def __call__(self, total_input: int, cached_input: int, output: int) -> float: ...


class GenerationClock(Protocol):
    def pause_generation(self) -> int: ...
    def generation_charge_time(self, token: int) -> float: ...
    def resume_generation(self, token: int, offset: float) -> None: ...


def read_finish_reason(response: Any) -> str | None:
    """The first choice's ``finish_reason``, or None where the object carries no choices.

    Separate from :func:`read_usage` because a streamed call splits the two: the usage block rides
    a final chunk that carries no choices, and the finish reason rides a content chunk that carries
    no usage."""
    choices = getattr(response, "choices", None) or []
    return getattr(choices[0], "finish_reason", None) if choices else None


def read_usage(response: Any) -> tuple[int, int | None, int, int | None, str | None, bool]:
    """(input, cached_input, output, reasoning, finish_reason, captured) from a usage-carrying
    response.

    Duck-typed on the OpenAI response shape, which is why it lives here rather than beside its
    first caller: LiteLLM's ``ModelResponse``, the OpenAI SDK's completion object and a streamed
    usage chunk all carry the same fields, and the latency grid reads them through a client that
    is neither arm's.

    Every field is read defensively: providers differ in which detail blocks they populate, and a
    provider that omits ``cached_tokens`` must not make the run look like a measured cache miss —
    the caller writes ``usage_captured=False`` when the whole block was unreachable, which is the
    signal that separates the two."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, None, 0, None, None, False
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    finish_reason = read_finish_reason(response)
    # `is not None`, not truthiness: a provider reporting `cached_tokens: 0` measured a cache miss,
    # which is a different fact from a provider that shipped no `prompt_tokens_details` block at all
    # (the ordinary case for most models). Collapsing the second into 0 would put a fabricated cache
    # miss into the cache-aware fit.
    cached = getattr(prompt_details, "cached_tokens", None)
    reasoning = getattr(completion_details, "reasoning_tokens", None)
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(cached) if cached is not None else None,
        int(getattr(usage, "completion_tokens", 0) or 0),
        int(reasoning) if reasoning is not None else None,
        finish_reason,
        True,
    )


@dataclass(frozen=True)
class RoundTripWindow:
    """One physical model crossing on the wall and counterfactual charged-clock axes."""

    started_at: float
    finished_at: float
    charged_seconds: float
    # Scenario elapsed time on the parallel-charge sensitivity axis when this crossing was
    # admitted. Calls admitted before any sibling settles share a coordinate; a follow-up admitted
    # after a completion starts at the settled parallel frontier instead.
    charged_started_at: float | None = None

    @property
    def wall_seconds(self) -> float:
        return max(0.0, self.finished_at - self.started_at)


def _interval_union_seconds(intervals: Iterable[tuple[float, float]]) -> float:
    intervals = sorted((start, max(start, end)) for start, end in intervals)
    if not intervals:
        return 0.0
    total = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start


def wall_time_union_seconds(windows: Iterable[RoundTripWindow]) -> float:
    """Return the union of physical model-call wall intervals, counting overlap once."""
    return _interval_union_seconds((window.started_at, window.finished_at) for window in windows)


def charged_time_union_seconds(windows: Iterable[RoundTripWindow]) -> float | None:
    """Return the union counterfactual for parallel token-charged inference, or None.

    The axis is chosen once for the whole collection, never per window. ``charged_started_at`` is
    scenario-elapsed time on the charged clock; ``started_at`` is host ``time.monotonic``. Those
    are different origins *and* different rates, so sweeping an interval union across a mixture of
    the two produces a number in no coordinate system at all — it is not merely imprecise.

    A run can genuinely produce the mixture: a crossing admitted after the online judge stopped
    the environment gets the no-op token and so carries no charged coordinate, while every earlier
    crossing in the same run does. Rather than emit a plausible-looking figure into a paid
    artifact, the union is declared unavailable for that run. A collection where *no* window has
    the coordinate is a pre-freeze record and still has one coherent axis, so it keeps the host
    monotonic reading.
    """
    material = tuple(windows)
    if not material:
        return 0.0
    charged = [window.charged_started_at is not None for window in material]
    if any(charged) and not all(charged):
        return None
    return _interval_union_seconds(
        (
            start := (
                window.charged_started_at
                if window.charged_started_at is not None
                else window.started_at
            ),
            start + max(0.0, window.charged_seconds),
        )
        for window in material
    )


@dataclass(frozen=True)
class RoundTripConcurrency:
    """Host-wall concurrency observed across physical model crossings."""

    round_trips: int
    max_in_flight: int
    overlapped_round_trips: int


def round_trip_concurrency(windows: Iterable[RoundTripWindow]) -> RoundTripConcurrency:
    """Count physical crossings and positive-duration overlap on the host wall clock."""
    material = tuple(windows)
    overlapped: set[int] = set()
    events: list[tuple[float, int, int]] = []
    for index, window in enumerate(material):
        start = window.started_at
        end = max(start, window.finished_at)
        if end <= start:
            continue
        # Ends sort before starts at the same instant, so touching intervals do not overlap.
        events.append((start, 1, index))
        events.append((end, -1, index))

    active: set[int] = set()
    max_in_flight = 0
    for _at, delta, index in sorted(events, key=lambda event: (event[0], event[1])):
        if delta < 0:
            active.discard(index)
            continue
        if active:
            overlapped.add(index)
            overlapped.update(active)
        active.add(index)
        max_in_flight = max(max_in_flight, len(active))
    return RoundTripConcurrency(len(material), max_in_flight, len(overlapped))


@dataclass(frozen=True)
class LLMCallRecord:
    """One logical model call. ``round_trips`` is how many times the wire was actually crossed for
    it — a parser-repair pass on the S-ORA side, a malformed-output retry on the ReAct side — with
    the token fields summed across them, because both arms are billed for every round-trip but
    only one of them is a decision.

    ``seconds`` is measured wall clock at the client, and is what the fit uses; ``charged_seconds``
    is what the run's charge model billed, and is None until a charge model is wired. Keeping both
    is the point: the second is a *pricing rule* and has to stay auditable against the first.

    A None token field means *not reported*, never a measured zero — a provider that omits
    ``cached_tokens`` is unknown coverage, not a measured cache miss.
    """

    call_id: str
    arm: Arm
    model: str | None
    semantic_label: str | None
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    seconds: float | None
    round_trips: int
    finish_reason: str | None
    charged_seconds: float | None = None
    # One entry per physical crossing. Unlike the aggregate ``seconds``, these intervals preserve
    # overlap on both the host and charged-clock axes after a paid sweep.
    round_trip_windows: tuple[RoundTripWindow, ...] = ()
    scenario_id: str | None = None
    run_number: int | None = None
    # Groups the rows that one charged step produced. Set on the ReAct arm, where ARE calls the
    # engine again on a malformed output and each call becomes its own row — there is no
    # logical-call id there to reuse, so without this a retry is indistinguishable from two
    # separate decisions. None on the S-ORA arm, where `call_id` already does the grouping.
    bracket_id: str | None = None
    # False when the tokens are not a complete account of what the call cost: the usage block was
    # unreadable, or only some of the round-trips reported one. Either way the fit has to skip the
    # row rather than read the fields it does carry as the whole call.
    usage_captured: bool = True


class LLMCallWriter:
    """JSONL sink, safe to share across threads and across scenarios.

    Flushed per line so an aborted sweep still leaves the scenarios it already paid for on disk.
    The batch harness points every scenario in a capability at one file, so ``scenario_id`` on the
    record — not the filename — is what separates them.

    ``reset=True`` truncates the file once, on the first open of this writer. A sweep re-run into
    an existing output directory replaces ``output.jsonl`` and the traces, so the call log has to
    go with them: appending would leave two sweeps' rows carrying the same ``scenario_id`` and
    ``run_number`` with nothing to tell them apart, and the fit would silently count the stale ones
    twice. The default appends, which is what a driver aimed at an operator-chosen path wants.
    """

    def __init__(self, path: str | Path, *, reset: bool = False) -> None:
        self.path = Path(path)
        self._reset = reset
        self._truncated = False
        self._lock = threading.Lock()
        self._handle: Any = None
        self.written = 0

    def open(self) -> LLMCallWriter:
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Only the first open truncates: a close/reopen partway through a sweep must not
            # discard the rows the sweep has already written.
            mode = "w" if self._reset and not self._truncated else "a"
            self._handle = self.path.open(mode, encoding="utf-8")
            self._truncated = True
        return self

    def write(self, record: LLMCallRecord) -> None:
        with self._lock:
            if self._handle is None:
                self.open()
            self._handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            self._handle.flush()
            self.written += 1

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def __enter__(self) -> LLMCallWriter:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


@dataclass
class _Partial:
    """Every round-trip seen so far for one logical call id, summed.

    Held until the recorder closes rather than flushed on the call's ``done``. S-ORA's parser repair
    (``_complete_and_parse``) issues its retry as a *second* ``MeteredLLMClient.complete`` under the
    same ``llm_call_scope``, so a repaired call emits ``usage, done, usage, done`` — flushing on the
    first ``done`` would split one logical call across two rows that both claim
    ``round_trips == 1`` and neither of which carries the call's real cost. Nothing in the stream
    says a logical call is *over* (``llm_call_scope`` exits silently, and adding a record there
    would put a line per call into the CLI's verbose trace), so "over" is only known at close.

    Round-trips are counted from ``done``, not ``usage``: ``MeteredLLMClient`` emits ``done`` from a
    ``finally`` on every crossing, while ``usage`` comes from the concrete client and is absent
    entirely when that client is uninstrumented.
    """

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    seconds: float | None = None
    usage_records: int = 0
    done_records: int = 0
    finish_reason: str | None = None
    model: str | None = None
    semantic_label: str | None = None
    charged_seconds: float | None = None
    # Physical usage samples are retained because the frozen model has one intercept per crossing.
    # Charging a repaired call from the aggregate row would pay that intercept only once.
    usage_samples: list[tuple[int, int | None, int]] = field(default_factory=list)
    round_trip_windows: list[RoundTripWindow] = field(default_factory=list)

    @property
    def usage_complete(self) -> bool:
        """True only when every crossing that happened also reported its tokens.

        A repair whose retry fails before ``log_llm_usage`` runs emits ``usage, done, done``: two
        crossings, one token report. Calling that captured would hand the fit a row that looks
        measured, is short by a whole round trip, and is indistinguishable from a call that really
        did cost what its first trip cost.
        """
        return self.usage_records > 0 and self.usage_records >= self.round_trips

    @property
    def round_trips(self) -> int:
        # `done` is the unconditional one; `usage` only covers for a client whose timing decorator
        # is missing, which is also the only way a partial can exist with no `done` at all.
        return self.done_records or self.usage_records

    def add_usage(self, record: logging.LogRecord) -> None:
        self.usage_records += 1
        # Kept None-if-never-reported rather than defaulting to 0: see LLMCallRecord.
        for field_name, attr in (
            ("input_tokens", "llm_input_tokens"),
            ("output_tokens", "llm_output_tokens"),
            ("cached_input_tokens", "llm_cached_input_tokens"),
            ("reasoning_tokens", "llm_reasoning_tokens"),
        ):
            value = getattr(record, attr, None)
            if value is not None:
                current: int | None = getattr(self, field_name)
                setattr(self, field_name, (current or 0) + int(value))
        self.finish_reason = getattr(record, "llm_finish_reason", None) or self.finish_reason
        self.model = getattr(record, "llm_observed_model", None) or self.model
        self.semantic_label = getattr(record, "llm_semantic_label", None) or self.semantic_label
        input_tokens = getattr(record, "llm_input_tokens", None)
        output_tokens = getattr(record, "llm_output_tokens", None)
        cached_input_tokens = getattr(record, "llm_cached_input_tokens", None)
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            self.usage_samples.append(
                (
                    input_tokens,
                    cached_input_tokens if isinstance(cached_input_tokens, int) else None,
                    output_tokens,
                )
            )

    def add_done(self, record: logging.LogRecord) -> None:
        self.done_records += 1
        seconds = getattr(record, "llm_seconds", None)
        if isinstance(seconds, int | float):
            self.seconds = (self.seconds or 0.0) + float(seconds)
        self.semantic_label = self.semantic_label or getattr(record, "llm_semantic_label", None)


class SoraCallRecorder(logging.Handler):
    """Turns S-ORA's ``sora.llm`` stream into ``LLMCallRecord`` rows.

    The tokens and the latency arrive on *two different records* — ``log_llm_usage`` emits the
    first from inside the concrete client, ``MeteredLLMClient`` the second from its ``finally`` —
    joined here on ``llm_call_id``, together with any further round-trips the same logical call
    made (see ``_Partial`` for why that join can only be settled at close).

    Attach to the ``sora`` logger (or ``sora.llm``) for the duration of a run, and close it when
    the run ends — ``close`` is what writes the rows. With an uninstrumented client no ``usage``
    ever arrives and the rows carry timing only, marked ``usage_captured=False`` so an absent token
    count reads as unmeasured rather than as zero; the same flag catches the partial case, where
    one round-trip of a repaired call reported its tokens and another died before it could.
    """

    def __init__(
        self,
        writer: LLMCallWriter | None,
        *,
        model: str | None = None,
        scenario_id: str | None = None,
        run_number: int | None = None,
        charge: Charge | None = None,
    ) -> None:
        super().__init__(level=logging.INFO)
        self._writer = writer
        self._model = model
        self._scenario_id = scenario_id
        self._run_number = run_number
        self._charge = charge
        self._partials: dict[str, _Partial] = {}
        self._closed_round_trip_windows: tuple[RoundTripWindow, ...] = ()
        self._lock = threading.RLock()
        self.raw_cached_input_anomalies = 0
        self.charged_seconds = 0.0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            event = getattr(record, "llm_event", None)
            call_id = getattr(record, "llm_call_id", None)
            if not isinstance(call_id, str):
                return
            with self._lock:
                if event == "usage":
                    partial = self._partials.setdefault(call_id, _Partial())
                    partial.add_usage(record)
                    cached = getattr(record, "llm_cached_input_tokens", None)
                    total = getattr(record, "llm_input_tokens", None)
                    if isinstance(cached, int) and isinstance(total, int) and cached > total:
                        self.raw_cached_input_anomalies += 1
                elif event == "done":
                    self._partials.setdefault(call_id, _Partial()).add_done(record)
        except Exception:  # a diagnostic must never take down the run it is observing
            self.handleError(record)

    def _flush(self, call_id: str, partial: _Partial) -> None:
        if self._writer is None:
            return
        self._writer.write(
            LLMCallRecord(
                call_id=call_id,
                arm="sora",
                model=partial.model or self._model,
                semantic_label=partial.semantic_label,
                input_tokens=partial.input_tokens,
                cached_input_tokens=partial.cached_input_tokens,
                output_tokens=partial.output_tokens,
                reasoning_tokens=partial.reasoning_tokens,
                seconds=partial.seconds,
                round_trips=partial.round_trips,
                finish_reason=partial.finish_reason,
                scenario_id=self._scenario_id,
                run_number=self._run_number,
                usage_captured=partial.usage_complete,
                charged_seconds=partial.charged_seconds,
                round_trip_windows=tuple(partial.round_trip_windows),
            )
        )

    def begin_round_trip(self, call_id: str) -> int:
        """Return this logical call's usage index before one physical crossing begins."""
        with self._lock:
            return len(self._partials.setdefault(call_id, _Partial()).usage_samples)

    def finish_round_trip(
        self,
        call_id: str,
        usage_index: int,
        *,
        started_at: float,
        finished_at: float,
        charged_started_at: float | None = None,
    ) -> float:
        """Price exactly one crossing, including the fixed term when usage was unavailable."""
        with self._lock:
            partial = self._partials.setdefault(call_id, _Partial())
            sample = (
                partial.usage_samples[usage_index]
                if usage_index < len(partial.usage_samples)
                else None
            )
            if self._charge is None:
                charged = 0.0
            elif sample is None:
                charged = self._charge(0, 0, 0)
            else:
                total, cached, output = sample
                # Unknown cache usage is conservatively uncached input.
                charged = self._charge(total, cached or 0, output)
            if self._charge is not None:
                partial.charged_seconds = (partial.charged_seconds or 0.0) + charged
            partial.round_trip_windows.append(
                RoundTripWindow(
                    started_at=started_at,
                    finished_at=max(started_at, finished_at),
                    charged_seconds=charged,
                    charged_started_at=charged_started_at,
                )
            )
            if self._charge is not None:
                self.charged_seconds += charged
            return charged

    @property
    def round_trip_windows(self) -> tuple[RoundTripWindow, ...]:
        with self._lock:
            current = tuple(
                window
                for partial in self._partials.values()
                for window in partial.round_trip_windows
            )
            return self._closed_round_trip_windows + current

    @property
    def wall_seconds(self) -> float:
        return sum(window.wall_seconds for window in self.round_trip_windows)

    @property
    def wall_union_seconds(self) -> float:
        return wall_time_union_seconds(self.round_trip_windows)

    @property
    def charged_union_seconds(self) -> float | None:
        # None when the run mixed charged and host-monotonic coordinates; see the function.
        return charged_time_union_seconds(self.round_trip_windows)

    @property
    def concurrency(self) -> RoundTripConcurrency:
        return round_trip_concurrency(self.round_trip_windows)

    def close(self) -> None:
        # Every row is written here, in the order the calls first appeared. A call that never got
        # its `done` (the process died between the two records) still lands, with seconds=None
        # rather than being dropped: the file has to stay a complete account of what was billed.
        with self._lock:
            for call_id, partial in list(self._partials.items()):
                self._flush(call_id, partial)
            self._closed_round_trip_windows += tuple(
                window
                for partial in self._partials.values()
                for window in partial.round_trip_windows
            )
            self._partials.clear()
        super().close()


class ClockedSoraLLMClient:
    """Time every physical S-ORA round-trip and optionally freeze ARE around it."""

    def __init__(
        self,
        inner: LLMClient,
        *,
        clock: GenerationClock | None,
        recorder: SoraCallRecorder,
    ) -> None:
        self._inner = inner
        self._clock = clock
        self._recorder = recorder
        self.model = getattr(inner, "model", None)

    @property
    def logical_calls_admitted(self) -> int:
        value = getattr(self._inner, "logical_calls_admitted", 0)
        return int(value) if isinstance(value, int) else 0

    @property
    def logical_call_limit_exceeded(self) -> bool:
        return bool(getattr(self._inner, "logical_call_limit_exceeded", False))

    async def complete(self, request: CompletionRequest) -> str:
        # Establishing the scope outside the metering decorator gives this wrapper the same id the
        # usage/done records carry. A parser repair enters an existing scope and therefore charges
        # a second physical crossing without creating a second logical call.
        with llm_call_scope() as call_id:
            assert current_llm_call_id.get() == call_id
            started_at = time.monotonic()
            usage_index = self._recorder.begin_round_trip(call_id)
            token = self._clock.pause_generation() if self._clock is not None else None
            charged_started_at: float | None = None
            try:
                if self._clock is not None and token is not None and token != 0:
                    # Probed inside the guarded region on purpose. The line above has already
                    # frozen the world, so from here on every exit path owes a resume; reading
                    # the charge axis before the `try` put one statement outside that guarantee.
                    charged_started_at = self._clock.generation_charge_time(token)
                return await self._inner.complete(request)
            finally:
                # Releasing the freeze is the one step that must happen. ARE's loop thread waits
                # on `pause_event` with no timeout, so a raise between here and `resume_generation`
                # does not fail one call -- it parks the scenario forever, and the wall-clock
                # watchdog is then the only thing that ends the run.
                charged = 0.0
                try:
                    charged = self._recorder.finish_round_trip(
                        call_id,
                        usage_index,
                        started_at=started_at,
                        finished_at=time.monotonic(),
                        charged_started_at=charged_started_at,
                    )
                finally:
                    # Charging zero for a crossing whose usage could not be recorded understates
                    # simulated time for that call. That is the deliberate direction: the run
                    # stays live and the loss is visible as a missing round-trip in the trace.
                    if self._clock is not None and token is not None:
                        self._clock.resume_generation(token, charged)
