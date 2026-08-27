"""Shared per-scenario run core for the Gaia2 drivers.

Both the single-scenario correctness gate (``run_benchmark.py``) and the batch harness
(``batch.py``) need the same thing: take one already-loaded ARE scenario (judge attached if
scoring), run S-ORA against it to *timeline completion*, and score it. That logic — the turn-aware
``stop_when`` predicate especially — lives here once so the two entry points can't drift.

ARE (and the LLM client) are optional dependency groups, so every import of them is lazy, done
inside the functions rather than at module top; the pure ``dataclass`` below is importable without
them, which keeps ``batch.py``'s formatting/aggregation helpers unit-testable without the ``are``
extra installed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# How long ARE may hold the timeline paused before the run is treated as stalled. The pause bracket
# around a per-turn judge call is the only thing that pauses a benchmark run, and a judge that is
# answering at all answers in seconds; three minutes is slack for a slow endpoint, not for a call
# that will never return. Deliberately not a CLI flag: it separates "the judge died" from "this
# scenario is slow", which is a property of the judge, not of the run being scored.
MAX_PAUSE_SECONDS = 180.0


@dataclass
class RunResult:
    """One scenario run's outcome. ``environment`` is the live ARE ``Environment`` (handed to the
    trace exporter); ``exception`` is set when the run *or* ``validate()`` raised, so a batch can
    record an ``exception`` result for this one scenario and carry on instead of aborting the sweep.
    ``outcome`` is a ``sora.adapters.are_sim.ValidationOutcome`` (``success=None`` when unscored or
    when the run failed before scoring). ``awaiting_input`` holds the prompts of any activity the
    run ended on a question from — empty for every ordinary run. ``write_counts`` is a
    ``sora.adapters.are_sim.WriteCountCheck`` when an oracle log was available (None otherwise) —
    ARE's tool-call-count gate recomputed offline, which costs no model tokens and so is filled in
    for *unscored* runs too. ``timeline_expired`` is True when ARE's event loop ran out of
    ``scenario.duration`` — a real-time budget, since the loop is wall-clock paced — before the run
    finished; every result below it then describes a world that stopped early, not an agent that
    chose badly."""

    outcome: Any
    environment: Any
    duration: float
    exception: Exception | None = None
    awaiting_input: list[str] = field(default_factory=list)
    write_counts: Any = None
    timeline_expired: bool = False


def _awaiting_input(agent: Any) -> list[str]:
    """The await-input prompts of every activity currently asking a question — the replan breaker,
    the sub-goal recursion breaker, or a user stop, all of which park on ``InputWait``. Read as a
    list rather than a bool because the prompt is the whole value: it names the specific defects
    that led there, which is what tells a swept run apart from one that merely scored badly."""
    from sora.activity import ActivityState
    from sora.types import InputWait

    return [
        a.blocked_on.prompt or ""
        for a in agent.working.activities.values()
        if a.state is ActivityState.BLOCKED and isinstance(a.blocked_on, InputWait)
    ]


def _timeline_expired(simulation: Any) -> bool:
    """Whether ARE's own clock, not the agent, ended the run. Tolerates a simulation that predates
    the probe (a fake in a test) rather than requiring it on the ``Simulation`` Protocol."""
    probe = getattr(simulation, "timeline_expired", None)
    if probe is None:
        return False
    try:
        return bool(probe())
    except Exception:  # a diagnostic must never cost the run its real result
        log.warning("timeline-expiry probe failed", exc_info=True)
        return False


def _make_stop_when(
    simulation: Any,
    agent: Any,
    exit_when_idle: float | None,
    max_wall_seconds: float,
) -> Callable[[], bool] | None:
    """The turn-aware done predicate (see ``run_benchmark.py``'s module docstring). Returns None
    when the caller opts into ``TerminalSession``'s own quiet-window heuristic (``exit_when_idle``
    set), letting the session drive its old single-turn behavior unchanged."""
    from sora.activity import ActivityState
    from sora.types import ConditionWait, InputWait

    if exit_when_idle is not None:
        return None
    deadline = time.monotonic() + max_wall_seconds
    paused_since: float | None = None

    def _timeline_done() -> bool:
        nonlocal paused_since
        now = time.monotonic()
        if now >= deadline:
            return True
        # A paused environment is a judge call in flight, so the timeline is mid-turn, not over --
        # but bound that wait on its own. The whole-run clock already backstops a stalled judge;
        # what it cannot do is tell a stall from a slow scenario, so it can only be set to the
        # larger of the two and then pays that price per hung scenario across a sweep. This keeps
        # the two independent: a judge that answers in seconds is never touched, and one that never
        # answers costs MAX_PAUSE_SECONDS instead of the full wall clock.
        if simulation.is_paused():
            paused_since = now if paused_since is None else paused_since
            return now - paused_since >= MAX_PAUSE_SECONDS
        paused_since = None
        if simulation.is_running():
            return False  # timeline live — keep going, more turns may arrive
        acts = list(agent.working.activities.values())
        if not acts:
            return False
        # Done also when what is left is a wait nothing will satisfy. An activity parked on
        # InputWait is waiting for a user Message, and one parked on ConditionWait is waiting for a
        # tool signal (a declared pending condition, ADR-0022); past the end of the timeline
        # neither is coming, so it can never reach TERMINATED and the run would otherwise sit out
        # its whole wall clock in silence. Deliberately *below* the is_running() guard: while the
        # timeline is live a later turn genuinely can satisfy either wait — Observe resumes on a
        # user Message or on a matching signal — and cutting the run short would throw away a
        # recoverable state. This is a harness-level bound standing in for the timers that would
        # bound an absent trigger in a long-running agent; it does not generalize beyond a
        # simulation with an end.
        return all(
            a.state is ActivityState.TERMINATED
            or (
                a.state is ActivityState.BLOCKED
                and isinstance(a.blocked_on, InputWait | ConditionWait)
            )
            for a in acts
        )

    return _timeline_done


def run_scenario(
    scenario: Any,
    *,
    config: str,
    verbose: bool = False,
    log_file: str | None = None,
    max_wall_seconds: float = 1200.0,
    exit_when_idle: float | None = None,
) -> RunResult:
    """Run S-ORA against one loaded scenario to completion, then score it. Attach the judge (via
    ``are_sim.attach_judge``) *before* calling this if a real score is wanted; without it the run is
    unscored (``outcome.success is None``). A run-time crash or a ``validate()`` error is captured
    on ``RunResult.exception`` rather than raised, so a batch loop can record it and move on;
    ``KeyboardInterrupt`` still propagates so an operator can abort."""
    from sora.adapters.are_sim import AreSimulation, ValidationOutcome, write_count_check
    from sora.bootstrap import build_agent
    from sora.cli import TerminalSession

    simulation = AreSimulation(scenario)
    agent = build_agent(config, simulation=simulation)
    stop_when = _make_stop_when(simulation, agent, exit_when_idle, max_wall_seconds)

    session = TerminalSession(
        agent,
        verbose=verbose,
        initial_task=None,  # the Gaia2 scenario delivers its own task via the AUI timeline
        exit_when_idle=exit_when_idle,
        stop_when=stop_when,
        log_file=log_file,
    )

    exc: Exception | None = None
    started = time.monotonic()
    try:
        asyncio.run(session.run())
    except Exception as e:  # a run-time crash: record it, sweep continues (KI still propagates)
        exc = e
    duration = time.monotonic() - started
    # Sampled here, immediately after the session's teardown, rather than in the RunResult below:
    # everything between is scoring work (a judge pass over the oracle graph, which can take
    # minutes), and this reads a wall clock. `AreSimulation` latches the verdict at stop() so the
    # position no longer matters for the real adapter, but a fake or a future Simulation that does
    # not latch still gets a value measured against the run rather than against the judge.
    expired = _timeline_expired(simulation)

    outcome: Any = ValidationOutcome(success=None)
    if exc is None and getattr(scenario, "judge", None) is not None:
        # Only trust validate() when a scoring judge is actually attached (attach_judge → ARE's
        # preprocess_scenario sets scenario.judge). Without one, the scenario falls back to ARE's
        # base Scenario.validate, which returns ``env.state != FAILED`` — a spurious True for any
        # run that merely didn't crash, meaningless as a score. So a judge-less run stays unscored
        # (success=None) rather than reporting a false PASS.
        try:
            outcome = simulation.validate()
        except Exception as e:  # judge/oracle failure surfaces here, not as a silent unscored run
            exc = e

    # Deliberately outside the judge guard and after validate(): it needs no judge and no tokens, so
    # an unscored dev run — where it is the only pass/fail signal there is — gets it too. Never lets
    # a reporting failure cost the run's real result.
    counts: Any = None
    try:
        counts = write_count_check(scenario, simulation.environment())
    except Exception:  # a diagnostic must never cost the run its real result
        log.warning("write-count check failed", exc_info=True)

    return RunResult(
        outcome=outcome,
        environment=simulation.environment(),
        duration=duration,
        exception=exc,
        # Read after the session returns, so this reflects where the run actually stopped. Not an
        # error: the agent halting to ask rather than looping is the designed behavior, and the
        # scenario is still scored normally — this only records *why* it stopped short.
        awaiting_input=_awaiting_input(agent),
        write_counts=counts,
        # Sampled above, right after teardown; never allowed to raise, because it reinterprets
        # every field beside it and losing it to a probe failure would be worse than losing any
        # single one of them.
        timeline_expired=expired,
    )
