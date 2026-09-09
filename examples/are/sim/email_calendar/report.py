"""``sora run --report examples.are.sim.email_calendar.report.report`` hook.

Everything a run's trace needs live (trajectory dump, LLM-call summary) is already printed by
``TerminalSession`` itself, the same as for any other agent. This hook adds only the two lines
that are specific to running an ARE scenario: the agent's own outcome, and ARE's separate
environment-level ``scenario.validate()`` score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sora.activity import ActivityState

if TYPE_CHECKING:
    from sora.cycle import Agent


def report(agent: Agent, simulation: Any | None) -> None:
    activities = list(agent.working.activities.values())
    # "Failed" is a judgment call owned by whichever ReflectStrategy this agent.yaml configures
    # (see ReflectStrategy.failed) — going through agent.cycle.strategies.reflect rather than a
    # hardcoded rule means this line keeps working if that strategy is ever swapped for one with
    # different terminal-failure semantics. Ask it only about TERMINATED activities: live work is
    # reported by state below, rather than treating a transient failed operation as the outcome of
    # the whole activity.
    reflect = agent.cycle.strategies.reflect
    failed = any(reflect.failed(a) for a in activities if a.state is ActivityState.TERMINATED)
    if failed:
        outcome = "❌ FAILED"
    elif any(a.state is ActivityState.BLOCKED for a in activities):
        # Deliberately the state, not a diagnosis of why: an activity waits on a completion signal,
        # on a declared condition, or on the user (a bounded-recovery breaker asking for guidance),
        # and this line does not distinguish them. What it does say is that work is parked rather
        # than finished — which reporting it as completed would hide, and `incomplete` would blur.
        outcome = "⏸ BLOCKED"
    elif any(a.state is not ActivityState.TERMINATED for a in activities):
        outcome = "incomplete"
    else:
        outcome = "completed"
    print(f"\nagent outcome: {outcome}")

    if simulation is None:
        return
    # The base Scenario.validate only checks the environment didn't enter a FAILED state and runs
    # any oracle validators — declaring `.oracle()` events alone does NOT make this non-vacuous
    # (the base validate() never inspects them); a scenario needs its own validate() override
    # checking final app state, like the bundled EmailScheduleScenario does.
    try:
        outcome = simulation.validate()
        print(
            f"ARE validation: {'✅ PASS' if outcome.success else 'FAIL'} "
            "(environment-level; vacuous unless the scenario overrides validate())"
        )
        if outcome.rationale:
            print(f"    {outcome.rationale}")
    except Exception as exc:  # a scenario whose validators need oracle events can raise instead
        print(f"ARE validation: n/a ({exc})")
