"""``MeteredLiteLLMEngine`` — the ReAct baseline arm's instrumentation.

Why this exists
---------------
ARE's own ReAct agent is charged for its thinking through exactly one channel: ``step()`` pauses
the environment clock, calls the engine, and resumes it by ``completion_duration`` taken from the
engine's metadata dict (``base_agent.py``; ``get_offset_from_time_config_mode`` returns that float
verbatim in the default ``"measured"`` mode). The channel is ARE's own and is fully wired — but
``LiteLLMEngine.chat_completion`` ends with ``return res, None``, throwing away a ``ModelResponse``
that carries the entire ``usage`` block. Every shipped engine does the same, which is why every
published leaderboard row describes an agent that thinks instantaneously.

Feeding that dict is the whole integration: no patch to ARE is needed, a subclass suffices. The
same object later carries the frozen charge (pass ``charge=``), so instrumenting the arm and
charging it through one code path shared with S-ORA are the same class rather than two.

Three things that would otherwise bite
--------------------------------------
1. **Retries live inside the pause bracket.** On a malformed output ``step()`` re-calls the engine
   and keeps only the *last* metadata, so a retried step silently loses the earlier round-trips'
   tokens. ``begin_bracket()`` (wire it with ``wrap_pause_env``) makes the returned metadata
   cumulative over the bracket, which is exact. Without that wiring the metadata covers one
   round-trip — undercounting, never double-charging.
   The JSONL keeps its own account: one row per round-trip, carrying ``bracket_id``. The extra
   keys the engine puts in the metadata dict do *not* survive — ``LLMOutputThoughtActionLog`` has
   a fixed field set and drops everything outside it — so the row is the only place a retry can
   still be told apart from two separate decisions after the run.
2. **The fit and the bill want different latency.** Every round-trip, failures included, reaches
   the JSONL and the charge, because the agent really did emit them; a row whose ``finish_reason``
   starts with ``error:`` is what the fit excludes, since it measures the provider's bad day rather
   than the model's work. "Failed" does not imply "unbilled": ARE validates the response after
   ``litellm.completion`` returned, so a content-filtered answer raises with a full ``usage`` block
   already captured, and the row reports what that crossing really cost. Only a crossing that
   produced no response at all bills the charge model's fixed per-call term alone. Either way it
   never reaches ARE's clock, because the exception leaves ``step()`` before metadata is returned.
3. **The response is intercepted, not re-implemented.** ``chat_completion`` still runs ARE's own
   body — including its ``True``/``False`` lowercasing, which downstream checkers depend on — and
   the ``ModelResponse`` is captured from ``litellm.completion`` on the way past. Copying the body
   here to reach the response would fork it from the installed ARE and drift silently.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from are.simulation.agents.llm.litellm.litellm_engine import (
    LiteLLMEngine,
    LiteLLMModelConfig,
)

from examples.gaia2.llm_calls import LLMCallRecord, LLMCallWriter

# Seconds to charge for one round-trip, given (input_tokens, cached_input_tokens, output_tokens).
# The frozen charge model plugs in here; None bills measured wall clock instead.
ChargeModel = Callable[[int, int, int], float]

_capture = threading.local()


def _install_response_capture() -> None:
    """Wrap the ``completion`` name ARE's engine module calls, stashing each ``ModelResponse`` on a
    thread-local.

    Thread-local rather than an instance attribute because a batch may run scenarios in threads
    while sharing one patched module global, and the wrapper always runs on its caller's thread.
    Idempotence is decided by a marker on the *currently installed* function rather than by a
    module flag, and the check is re-run per call: anything that replaces ``module.completion``
    after us — a test double, another harness — would otherwise silently unhook the capture and
    leave every row reading ``usage_captured=False``."""
    from are.simulation.agents.llm.litellm import litellm_engine as module

    inner = module.completion
    if getattr(inner, "_sora_capture", False):
        return

    def capturing(*args: Any, **kwargs: Any) -> Any:
        response = inner(*args, **kwargs)
        _capture.response = response
        return response

    capturing._sora_capture = True  # type: ignore[attr-defined]
    module.completion = capturing


@dataclass(frozen=True)
class _RoundTrip:
    input_tokens: int
    # None where the provider reported no such field — never a measured zero (see LLMCallRecord).
    cached_input_tokens: int | None
    output_tokens: int
    reasoning_tokens: int | None
    seconds: float
    charged: float


def _sum_known(values: Iterable[int | None]) -> int | None:
    """Sum of the reported values, or None when nothing in the bracket reported one."""
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _read_usage(response: Any) -> tuple[int, int | None, int, int | None, str | None, bool]:
    """(input, cached_input, output, reasoning, finish_reason, captured) from a ``ModelResponse``.

    Every field is read defensively: providers differ in which detail blocks they populate, and a
    provider that omits ``cached_tokens`` must not make the run look like a measured cache miss —
    the caller writes ``usage_captured=False`` when the whole block was unreachable, which is the
    signal that separates the two."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, None, 0, None, None, False
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    finish_reason: str | None = None
    choices = getattr(response, "choices", None) or []
    if choices:
        finish_reason = getattr(choices[0], "finish_reason", None)
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


class MeteredLiteLLMEngine(LiteLLMEngine):  # type: ignore[misc]  # ARE is untyped
    """``LiteLLMEngine`` that returns a populated metadata dict instead of ``None``.

    ``writer`` receives one ``LLMCallRecord`` per round-trip (``arm="react"``), tagged with the
    ``bracket_id`` of the step it belongs to; ``charge`` decides what ``completion_duration``
    bills. ``scenario_id``/``run_number`` are set per scenario by the driver so one file can hold
    a whole capability's sweep."""

    def __init__(
        self,
        model_config: LiteLLMModelConfig,
        *,
        writer: LLMCallWriter | None = None,
        charge: ChargeModel | None = None,
        scenario_id: str | None = None,
        run_number: int | None = None,
    ) -> None:
        super().__init__(model_config)
        _install_response_capture()
        self.writer = writer
        self.charge = charge
        self.scenario_id = scenario_id
        self.run_number = run_number
        self._bracket: list[_RoundTrip] = []
        self._bracket_wired = False
        self.round_trips = 0
        self.brackets = 0

    # -- bracket -----------------------------------------------------------------------------

    def begin_bracket(self) -> None:
        """Start a new generation bracket, discarding the previous one's tally.

        Called from the wrapped ``pause_env``, which ARE invokes once at the top of every
        ``step()`` — the only signal available to an engine for "the metadata you returned last
        time has been consumed"."""
        self._bracket_wired = True
        self.brackets += 1
        self._bracket.clear()

    def wrap_pause_env(self, pause_env: Callable[[], None]) -> Callable[[], None]:
        """Wrap ARE's ``env.pause`` so brackets open in step with it. Pass the result as the agent's
        ``pause_env=`` at construction — ``ARESimulationAgent.initialize`` forwards it to the ReAct
        agent, so nothing is monkey-patched."""

        def paused() -> None:
            self.begin_bracket()
            pause_env()

        return paused

    # -- the call ----------------------------------------------------------------------------

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        stop_sequences: Any = [],  # noqa: B006 — ARE's signature; overriding it would break callers
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any] | None]:
        _install_response_capture()  # cheap; re-arms if anything replaced the module global
        _capture.response = None
        if not self._bracket_wired:
            # Unwired, a bracket is one round-trip — all the engine can know without ARE's pause.
            self.brackets += 1
        bracket_id = f"react-{self.scenario_id or 'run'}-bracket-{self.brackets:05d}"
        started = time.perf_counter()
        try:
            text, _ = super().chat_completion(messages, stop_sequences, **kwargs)
        except Exception as exc:
            # A failure here does not mean the provider never answered. ARE validates the response
            # *after* `completion()` returned — four asserts, and `assert res is not None` fires on
            # a content-filtered `message.content=None`, a crossing that was billed in full. The
            # captured response is read on this path for exactly that reason; only a failure that
            # produced no response at all falls through to a fixed-term row.
            trip, _reason, captured = self._trip(
                self._take_response(), time.perf_counter() - started
            )
            self._record(
                trip,
                finish_reason=f"error:{type(exc).__name__}",
                captured=captured,
                bracket_id=bracket_id,
            )
            raise
        trip, finish_reason, captured = self._trip(
            self._take_response(), time.perf_counter() - started
        )
        self._record(trip, finish_reason=finish_reason, captured=captured, bracket_id=bracket_id)
        self._bracket.append(trip)
        return text, self._metadata(trip)

    def _take_response(self) -> Any:
        """The captured ``ModelResponse``, cleared as it is read so no later call can see it."""
        response = getattr(_capture, "response", None)
        _capture.response = None
        return response

    def _trip(self, response: Any, elapsed: float) -> tuple[_RoundTrip, str | None, bool]:
        """One round-trip's record, priced. ``response`` of None is the crossing that produced no
        answer at all: no tokens, and a charge model prices it at its fixed per-call term — zero
        would be the one reading the design rules out, since the agent did emit the call and did
        wait for it to fail."""
        input_tokens, cached, output_tokens, reasoning, finish_reason, captured = _read_usage(
            response
        )
        charged = self._charge_for(input_tokens, cached, output_tokens, elapsed)
        trip = _RoundTrip(input_tokens, cached, output_tokens, reasoning, elapsed, charged)
        return trip, finish_reason, captured

    def _charge_for(
        self, input_tokens: int, cached: int | None, output_tokens: int, elapsed: float
    ) -> float:
        # An unreported cache figure bills as uncached: the charge model has to return a number,
        # and pricing the unknown at the cheaper rate would understate the arm.
        if self.charge is None:
            return elapsed
        return self.charge(input_tokens, cached or 0, output_tokens)

    def _metadata(self, trip: _RoundTrip) -> dict[str, Any]:
        """Cumulative over the bracket when brackets are wired, this round-trip alone otherwise —
        because ARE keeps only the last metadata a bracket produced."""
        trips = self._bracket if self._bracket_wired else [trip]
        prompt_tokens = sum(t.input_tokens for t in trips)
        completion_tokens = sum(t.output_tokens for t in trips)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            # ARE stores this straight into `LLMOutputThoughtActionLog.reasoning_tokens`, an int
            # its exporter collects into a list, so the unknown flattens to 0 on the way into
            # ARE's own artifact. The honest null lives in the JSONL row, which is the artifact
            # whose schema documents the distinction.
            "reasoning_tokens": _sum_known(t.reasoning_tokens for t in trips) or 0,
            "completion_duration": sum(t.charged for t in trips),
            # None of the keys below are read by ARE, and `LLMOutputThoughtActionLog` drops every
            # field outside its own fixed set — so they reach a live trace and nothing else. What
            # has to survive the run is on the JSONL row.
            "round_trips": len(trips),
            "cached_prompt_tokens": _sum_known(t.cached_input_tokens for t in trips),
            "measured_seconds": sum(t.seconds for t in trips),
        }

    def _record(
        self, trip: _RoundTrip, *, finish_reason: str | None, captured: bool, bracket_id: str
    ) -> None:
        self.round_trips += 1
        if self.writer is None:
            return
        self.writer.write(
            LLMCallRecord(
                # ARE's ReAct loop has no logical-call id of its own: one round-trip is one call,
                # and a retry is a distinct row rather than a second trip on the same id — the
                # opposite of S-ORA's repair pass, which reuses its id. `bracket_id` is what puts
                # a step's rows back together, and it has to be *on the row*: the `round_trips`
                # the engine returns is dropped with the rest of the metadata ARE does not read.
                call_id=f"react-{self.scenario_id or 'run'}-{self.round_trips:05d}",
                bracket_id=bracket_id,
                arm="react",
                model=self.model_config.model_name,
                semantic_label="react_step",
                input_tokens=trip.input_tokens if captured else None,
                cached_input_tokens=trip.cached_input_tokens if captured else None,
                output_tokens=trip.output_tokens if captured else None,
                reasoning_tokens=trip.reasoning_tokens if captured else None,
                seconds=trip.seconds,
                round_trips=1,
                finish_reason=finish_reason,
                charged_seconds=trip.charged,
                scenario_id=self.scenario_id,
                run_number=self.run_number,
                usage_captured=captured,
            )
        )
