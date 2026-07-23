"""On-demand runner for the in-process ARE showcase (real ARE Environment + real Claude).

    uv sync --all-extras --group are
    export ANTHROPIC_API_KEY=sk-ant-...                                  # or a .gitignored .env
    uv run python -m examples.are_scenario.run                           # default scenario
    uv run python -m examples.are_scenario.run --scenario path/to.json   # a Gaia2 JSON scenario
    uv run python -m examples.are_scenario.run --scenario pkg.mod.MyScenario

The scenario is a **command-line argument**, not config: the runner resolves it (``load_scenario``),
wraps it in an ``AreSimulation``, and injects that into the otherwise-generic ``agent.yaml`` via
``build_agent(config, simulation=...)``. ``Agent.run()`` then joins the ``are-sim`` workspace (which
starts the Environment event loop), and the task/timeline flow in through the ARE transport — no
manual ``transport.submit``. A docs showcase, not a test (needs a live model + the ARE package).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from pathlib import Path

from sora.activity import ActivityState
from sora.adapters.are_sim import AreSimulation, load_scenario
from sora.bootstrap import build_agent

log = logging.getLogger(__name__)

CONFIG = Path(__file__).with_name("agent.yaml")
DEFAULT_SCENARIO = "examples.are_scenario.scenario.EmailScheduleScenario"
_MAX_WAIT_S = 120.0
_POLL_S = 0.1
# Once every activity is terminated, wait this long for more work before exiting — long enough that
# a mid-run follow-up (which may land after the first activity completes and spawn a corrective one)
# still arrives and resets the timer. A newly created/live activity resets it; the deadline caps the
# total. Deliberately generous over precise: the runner can't tell a follow-up's state_changed
# signal from one the agent caused itself, so it waits out a quiet window rather than guessing.
_SETTLE_S = 8.0


async def main(scenario_ref: str) -> None:
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format="%(message)s")
    for noisy in ("httpx", "anthropic", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    print(f"loading scenario {scenario_ref!r} ...", flush=True)
    simulation = AreSimulation(load_scenario(scenario_ref))
    agent = build_agent(str(CONFIG), simulation=simulation)

    print("joining workspace + running the ARE event loop ...", flush=True)
    runner = asyncio.create_task(agent.run())
    try:
        deadline = time.monotonic() + _MAX_WAIT_S
        settled_since: float | None = None
        while time.monotonic() < deadline:
            await asyncio.sleep(_POLL_S)
            if runner.done():  # surface a startup/tick failure immediately, not after the timeout
                break
            # Don't stop on the first TERMINATED activity: the dynamic follow-up lands mid-run and
            # may spawn corrective work *after* the original completes. Exit only once every
            # activity has been terminated and stayed that way for _SETTLE_S — a corrective (once
            # created) is READY, which breaks "all terminated" and resets the timer.
            acts = list(agent.working.activities.values())
            quiescent = bool(acts) and all(a.state is ActivityState.TERMINATED for a in acts)
            if not quiescent:
                settled_since = None
            else:
                settled_since = settled_since or time.monotonic()
                if time.monotonic() - settled_since >= _SETTLE_S:
                    break
    finally:
        await agent.stop()
        if not runner.done():
            runner.cancel()
            try:
                await runner
            except (Exception, asyncio.CancelledError, BaseExceptionGroup) as exc:
                log.debug("run() raised during cancel-triggered shutdown: %r", exc)
        else:
            await runner

    print("\n--- trajectory ---")
    activities = list(agent.working.activities.values())
    for activity in activities:
        # A failed operation terminates the activity, so TERMINATED alone doesn't mean success — an
        # unresolved-ok last_operation is the failure signal (mirrors DefaultReflectStrategy).
        failed = activity.last_operation is not None and not activity.last_operation.ok
        note = "  (failed)" if failed else ""
        print(f"activity {activity.id!r}: {activity.goal!r} -> {activity.state.name}{note}")
        if activity.plan is not None:
            for i, step in enumerate(activity.plan.steps):
                marker = ">" if i == activity.step_index else " "
                print(f"  {marker} {step.next_action} {step.params}")

    # The agent's own outcome — the thing this showcase actually demonstrates.
    agent_failed = any(a.last_operation is not None and not a.last_operation.ok for a in activities)
    print(f"\nagent outcome: {'FAILED' if agent_failed else 'completed'}")

    # ARE's environment-level validation is a *separate* axis. The base Scenario.validate only runs
    # the scenario's oracle validators and checks the environment didn't enter a FAILED state — so a
    # scenario with no oracle events (like this default one) always reports PASS, even when the
    # agent failed the task. Author oracle events to make this a real task score.
    try:
        ok = simulation.validate()
        print(
            f"ARE validation: {'PASS' if ok else 'FAIL'} "
            "(environment-level; vacuous unless the scenario defines oracle events)"
        )
    except Exception as exc:  # a scenario whose validators need oracle events can raise instead
        print(f"ARE validation: n/a ({exc})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an ARE scenario in-process against S-ORA.")
    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO,
        help="dotted Scenario subclass path or a Gaia2 .json file (default: the bundled scenario)",
    )
    asyncio.run(main(parser.parse_args().scenario))
