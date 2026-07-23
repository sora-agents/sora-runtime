"""On-demand runner for the gaia2 email/calendar showcase (real ARE MCP server + real Claude).

    uv sync --all-extras --group are
    export ANTHROPIC_API_KEY=sk-ant-...          # or a .gitignored .env
    uv run python -m examples.gaia2.email_calendar.run

Builds the agent from ``agent.yaml``, seeds the scenario's initial task (``task.txt``) as a user
message (ARE's own USER_MESSAGE routing into the transport is not wired yet — deferred), drives the
decision cycle until the activity terminates, and prints the trajectory. A docs showcase, not a
test: it needs a live model and the ARE package, so it is deliberately outside the pytest suite.
It's also a reference for driving an ``Agent`` programmatically — see README's "Driving an agent
programmatically" — as opposed to `TerminalSession`/`sora run`, the CLI path this same scenario is
also runnable through:

    uv run sora run examples/gaia2/email_calendar/agent.yaml \
        --task-file examples/gaia2/email_calendar/task.txt --verbose

Note on output: the runtime's own INFO trace (join / plan / invoke / resolve / terminate) prints by
default. Mute it, or raise/lower the level, via LOGLEVEL:

    LOGLEVEL=WARNING uv run python -m examples.gaia2.email_calendar.run
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

from sora.activity import ActivityState
from sora.bootstrap import build_agent
from sora.llm import LLMMeter
from sora.perception import Message
from sora.transport import InProcessTransport

log = logging.getLogger(__name__)

CONFIG = Path(__file__).with_name("agent.yaml")
TASK = Path(__file__).with_name("task.txt").read_text(encoding="utf-8").strip()
_MAX_WAIT_S = 90.0
_POLL_S = 0.1


async def main() -> None:
    # Default to the runtime's own INFO trace (join / plan / invoke / resolve / terminate). Mute the
    # in-process HTTP/model libs; ARE's server logs come from the subprocess's stderr regardless.
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format="%(message)s")
    for noisy in ("httpx", "anthropic", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Count/time the model round-trips (per-call `~ llm (…s)` cue prints via basicConfig above) and
    # start the wall clock, for the scenario-run footer below.
    meter = LLMMeter()
    logging.getLogger("sora").addHandler(meter)

    print(f"building agent from {CONFIG.name} ...", flush=True)
    agent = build_agent(str(CONFIG))
    transport = agent.communication
    assert isinstance(transport, InProcessTransport)  # the single-agent default
    transport.submit(Message(sender="user", content={"text": TASK}, received_at=time.time()))

    print("joining workspaces + running (ARE startup takes a few seconds) ...", flush=True)
    wall_start = time.monotonic()
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
                await runner
            except (Exception, asyncio.CancelledError, BaseExceptionGroup) as exc:
                # Cancelling mid-join/mid-tick can surface as a messy anyio BaseExceptionGroup
                # (from the MCP stdio client's own task group), not a plain CancelledError — since
                # *we* triggered this cancellation, treat it as shutdown noise, not a crash.
                log.debug("run() raised during cancel-triggered shutdown: %r", exc)
        else:
            await runner  # finished on its own — a real failure here should propagate

    print("\n--- trajectory ---")
    for activity in agent.working.activities.values():
        print(f"activity {activity.id!r}: {activity.goal!r} -> {activity.state.name}")
        if activity.plan is not None:
            for i, step in enumerate(activity.plan.steps):
                marker = ">" if i == activity.step_index else " "
                print(f"  {marker} {step.next_action} {step.params}")

    print(f"\n-- {meter.summary(time.monotonic() - wall_start)} --")


if __name__ == "__main__":
    asyncio.run(main())
