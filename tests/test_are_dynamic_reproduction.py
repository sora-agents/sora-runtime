"""Skip-gated end-to-end reproduction: a **dynamic** ARE scenario, in-process, against real Claude.

The counterpart to ``test_are_scenario_reproduction.py`` (which drives the *static*, seeded MCP
world). Here the ARE ``Environment`` event loop runs for real: the task is delivered through the
``AgentUserInterface`` (no manual ``transport.submit``), and a mid-run follow-up email lands off the
agent's own action — the dynamic story the static world can't tell. Same ``build_agent`` path as the
showcase (``examples/are_scenario/run.py``), scenario injected as a runtime ``AreSimulation``.

**Opt-in and skip-gated** (marked ``integration``, excluded from the default ``pytest`` run): needs
the ``llm`` extra, the ARE package (``uv sync --all-extras --group are``), and a live
``ANTHROPIC_API_KEY``. CI stays deterministic — the bridge's behavior is pinned over fakes in
``test_are_sim.py`` and against a real Environment (no model) in ``test_are_sim_integration.py``.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from sora.activity import ActivityState

_CONFIG = Path(__file__).resolve().parent.parent / "examples" / "are_scenario" / "agent.yaml"
_SCENARIO = "examples.are_scenario.scenario.EmailScheduleScenario"


def _config_with_tmp_memory(tmp_path: Path) -> Path:
    """The real showcase config with its file:// memory dirs redirected under tmp_path (empty
    procedural store per run, and the repo's ``.sora/`` is left untouched)."""
    text = _CONFIG.read_text(encoding="utf-8").replace(
        "file://./.sora/are_scenario", f"file://{tmp_path}/memory"
    )
    out = tmp_path / "agent.yaml"
    out.write_text(text, encoding="utf-8")
    return out


@pytest.mark.integration
async def test_dynamic_are_scenario_reproduction(tmp_path: Path) -> None:
    pytest.importorskip("are.simulation")
    pytest.importorskip("anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    from sora.adapters.are_sim import AreSimulation, load_scenario
    from sora.bootstrap import build_agent

    simulation = AreSimulation(load_scenario(_SCENARIO))
    agent = build_agent(str(_config_with_tmp_memory(tmp_path)), simulation=simulation)

    tool_ids: set[str] = set()
    runner = asyncio.create_task(agent.run())
    try:
        deadline = time.monotonic() + 150.0
        while time.monotonic() < deadline:
            await asyncio.sleep(0.2)
            if runner.done():  # a startup/tick failure — surface it now
                break
            activities = list(agent.working.activities.values())
            if activities and any(a.state is ActivityState.TERMINATED for a in activities):
                break
        # Snapshot the joined tool ids *before* teardown — Agent.run()'s finally leaves every
        # workspace (deregistering its tools), so all_tools() is empty once the runner returns.
        tool_ids = {t.id for t in agent.registry.all_tools()}
    finally:
        await agent.stop()
        if not runner.done():
            runner.cancel()
        try:
            await runner
        except (Exception, asyncio.CancelledError, BaseExceptionGroup):
            pass

    # The task arrived through the ARE transport (AUI), not a manual submit -> an activity exists.
    activities = list(agent.working.activities.values())
    assert activities, "no activity was created from the AUI task message"
    activity = activities[0]
    assert activity.plan is not None and activity.plan.steps, "the model produced no plan"
    for step in activity.plan.steps:
        if step.next_action == "invoke":
            assert step.params["tool_id"] in tool_ids  # no made-up tool ids
    assert activity.history, "the agent planned but never invoked a real ARE operation"

    # The Environment event loop actually ran the timeline: the follow-up email was injected off the
    # agent's own action, so the live inbox now holds Alice's original *and* her follow-up.
    email_app = next(a for a in simulation.apps() if a.app_name() == "EmailClientApp")
    inbox = email_app.get_state().get("folders", {})
    assert "Tuesday" in str(email_app.get_state()), (
        f"timeline follow-up email never fired; inbox state did not change: {inbox}"
    )
