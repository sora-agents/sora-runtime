"""On-demand runner for the gaia2 email/calendar showcase (real ARE MCP server + real Claude).

    uv sync --all-extras --group are
    export ANTHROPIC_API_KEY=sk-ant-...          # or a .gitignored .env
    uv run python -m examples.gaia2.email_calendar.run

Builds the agent from ``agent.yaml``, seeds the scenario's initial task as a user message (ARE's own
USER_MESSAGE routing into the transport is not wired yet — deferred), drives the decision cycle
until the activity terminates, and prints the trajectory. A docs showcase, not a test: it needs a
live model and the ARE package, so it is deliberately outside the pytest suite.

Note on output: the runtime does not log yet, so ``--verbose``-style per-phase tracing (README)
isn't available here. To watch progress, set the log level — this surfaces the runtime's own records
once logging lands, and today shows the ARE server + MCP logs:

    LOGLEVEL=INFO uv run python -m examples.gaia2.email_calendar.run
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

from sora.activity import ActivityState
from sora.bootstrap import build_agent
from sora.perception import Message
from sora.transport import InProcessTransport

CONFIG = Path(__file__).with_name("agent.yaml")
TASK = "Set up a 30-minute team sync with Bob and Carol next Monday, then reply to Alice."
_MAX_WAIT_S = 90.0
_POLL_S = 0.1


async def main() -> None:
    # Default to the runtime's own INFO trace (join / plan / invoke / resolve / terminate). Mute the
    # in-process HTTP/model libs; ARE's server logs come from the subprocess's stderr regardless.
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format="%(message)s")
    for noisy in ("httpx", "anthropic", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    print(f"building agent from {CONFIG.name} ...", flush=True)
    agent = build_agent(str(CONFIG))
    transport = agent.communication
    assert isinstance(transport, InProcessTransport)  # the single-agent default
    transport.submit(Message(sender="user", content={"text": TASK}, received_at=time.time()))

    print("joining workspaces + running (ARE startup takes a few seconds) ...", flush=True)
    runner = asyncio.create_task(agent.run())
    try:
        deadline = time.monotonic() + _MAX_WAIT_S
        while time.monotonic() < deadline:
            await asyncio.sleep(_POLL_S)
            # Surface a startup-join / tick failure immediately — without this the loop would wait
            # out the whole timeout before `await runner` re-raised, presenting as a silent hang.
            if runner.done():
                break
            if any(a.state is ActivityState.TERMINATED for a in agent.working.activities.values()):
                break
    finally:
        await agent.stop()
        if not runner.done():
            runner.cancel()  # unblock a stuck join/tick so teardown can't hang forever
        try:
            await runner  # re-raises any real failure from run()
        except asyncio.CancelledError:
            pass

    print("\n--- trajectory ---")
    for activity in agent.working.activities.values():
        print(f"activity {activity.id!r}: {activity.goal!r} -> {activity.state.name}")
        if activity.plan is not None:
            for i, step in enumerate(activity.plan.steps):
                marker = ">" if i == activity.step_index else " "
                print(f"  {marker} {step.next_action} {step.params}")


if __name__ == "__main__":
    asyncio.run(main())
