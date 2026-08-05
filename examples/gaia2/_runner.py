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
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class RunResult:
    """One scenario run's outcome. ``environment`` is the live ARE ``Environment`` (handed to the
    trace exporter); ``exception`` is set when the run *or* ``validate()`` raised, so a batch can
    record an ``exception`` result for this one scenario and carry on instead of aborting the sweep.
    ``outcome`` is a ``sora.adapters.are_sim.ValidationOutcome`` (``success=None`` when unscored or
    when the run failed before scoring)."""

    outcome: Any
    environment: Any
    duration: float
    exception: Exception | None = None


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

    if exit_when_idle is not None:
        return None
    deadline = time.monotonic() + max_wall_seconds

    def _timeline_done() -> bool:
        if time.monotonic() >= deadline:
            return True
        if simulation.is_running():
            return False  # timeline live — keep going, more turns may arrive
        acts = list(agent.working.activities.values())
        return bool(acts) and all(a.state is ActivityState.TERMINATED for a in acts)

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
    from sora.adapters.are_sim import AreSimulation, ValidationOutcome
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
    return RunResult(
        outcome=outcome,
        environment=simulation.environment(),
        duration=duration,
        exception=exc,
    )
