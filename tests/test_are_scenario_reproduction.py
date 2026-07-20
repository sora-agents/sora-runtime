"""Skip-gated end-to-end reproduction: the gaia2 showcase against real ARE apps + Claude.

Builds the agent from ``examples/gaia2/email_calendar/agent.yaml`` via ``build_agent`` — the same
path the on-demand showcase (``run.py``) uses — then drives the full decision cycle: a seeded task
message becomes an activity, the model-backed ``ScheduleFromEmailStrategy`` plans against the live
ARE apps, and the invokes hit our *seeded* ARE MCP launcher (``are_server.py``) over stdio — so
``list_emails`` returns a real email from Alice with a real id, which Reason grounds
``reply_to_email`` against.

**Opt-in and skip-gated** (marked ``integration``, excluded from the default ``pytest`` run — see
pyproject ``addopts``): it needs the ``mcp``/``llm`` extras, the ARE package (``uv sync --all-extras
--group are``), and a live ``ANTHROPIC_API_KEY``. CI stays deterministic — the scenario's *shape* is
pinned deterministically in ``test_gaia2_reproduction.py`` over fakes.

Must run from the repo root (like the showcase): the launcher is spawned as ``python -m
examples.gaia2.email_calendar.are_server``, which needs the repo root on the path. This asserts the
runnable-agent contract (the plan executes end-to-end against real ARE apps with real data), not a
scenario validator score — the world here is a static seeded snapshot, not a running simulation.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from sora.activity import ActivityState
from sora.bootstrap import build_agent
from sora.perception import Message
from sora.transport import InProcessTransport

_CONFIG = (
    Path(__file__).resolve().parent.parent / "examples" / "gaia2" / "email_calendar" / "agent.yaml"
)
_TASK = "Set up a 30-minute team sync with Bob and Carol next Monday, then reply to Alice."


def _config_with_tmp_memory(tmp_path: Path) -> Path:
    """The real showcase config, but with its file:// memory dirs redirected under tmp_path — so the
    test doesn't touch the repo's ``.sora/`` and each run starts with an empty procedural store. CWD
    stays the repo root (the launcher subprocess imports ``examples`` from there), so we can't just
    rely on the config's relative paths."""
    text = _CONFIG.read_text(encoding="utf-8").replace(
        "file://./.sora/memory", f"file://{tmp_path}/memory"
    )
    out = tmp_path / "agent.yaml"
    out.write_text(text, encoding="utf-8")
    return out


@pytest.mark.integration
async def test_gaia2_agent_runs_against_real_are(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    pytest.importorskip("are.simulation")
    pytest.importorskip("anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    agent = build_agent(str(_config_with_tmp_memory(tmp_path)))
    transport = agent.communication
    assert isinstance(transport, InProcessTransport)
    transport.submit(Message(sender="user", content={"text": _TASK}, received_at=time.time()))

    runner = asyncio.create_task(agent.run())
    try:
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            await asyncio.sleep(0.2)
            if runner.done():  # a startup/tick failure — surface it now, don't wait out the timeout
                break
            activities = list(agent.working.activities.values())
            if activities and any(a.state is ActivityState.TERMINATED for a in activities):
                break
    finally:
        await agent.stop()
        if not runner.done():
            runner.cancel()
        try:
            await runner
        except asyncio.CancelledError:
            pass

    activities = list(agent.working.activities.values())
    assert activities, "no activity was ever created from the seeded task message"
    activity = activities[0]
    assert activity.plan is not None, "the model produced no plan"
    assert activity.plan.steps, "the model returned an empty plan"
    # Every invoke step references a real ARE tool that discover() registered (no made-up ids).
    tool_ids = {t.id for t in agent.registry.all_tools()}
    for step in activity.plan.steps:
        if step.next_action == "invoke":
            assert step.params["tool_id"] in tool_ids
    # It actually executed against real ARE (not just planned): at least one operation resolved, and
    # reading the inbox was part of it (the seeded email is the whole point).
    assert activity.history, "the agent planned but never invoked a real ARE operation"
    ran = {c.invocation.operation_name for c in activity.history}
    assert "list_emails" in ran or "search_emails" in ran, f"agent never read the inbox; ran={ran}"
