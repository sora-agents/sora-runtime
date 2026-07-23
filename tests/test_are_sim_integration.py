"""In-process ARE bridge against a **real, running** ``Environment`` (no model) — the dynamic path.

Drives a tiny in-code scenario whose event timeline fires on a background thread: the task arrives
via the ``AgentUserInterface`` at t0, a follow-up email is injected at delay T. Asserts the bridge
surfaces both — the follow-up as a ``state_changed`` signal (the thing MCP could not push
off-request), the task as a transport ``Message`` — and that an app op invokes. Opt-in
(``-m integration``, needs ``uv sync --all-extras --group are``); excluded from the default/CI run.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

pytest.importorskip("are.simulation.environment")

from are.simulation.apps.agent_user_interface import AgentUserInterface  # noqa: E402
from are.simulation.apps.email_client import (  # noqa: E402
    Email,
    EmailClientApp,
    EmailFolderName,
)
from are.simulation.scenarios.scenario import Scenario  # noqa: E402
from are.simulation.types import Event  # noqa: E402

from sora.adapters.are_sim import (  # noqa: E402
    AreInProcessWorkspaceAdapter,
    AreSimulation,
    AreTransport,
)
from sora.environment import WorkspaceOrigin  # noqa: E402
from sora.perception import NotificationQueueSink  # noqa: E402

pytestmark = pytest.mark.integration

_USER = "me@corp.com"


class _DynamicScenario(Scenario):  # type: ignore[misc]
    start_time = 0
    duration = 15

    def init_and_populate_apps(self, *args: object, **kwargs: object) -> None:
        self.email = EmailClientApp()
        self.aui = AgentUserInterface()
        self.email.add_email(
            Email(
                sender="alice@corp.com",
                recipients=[_USER],
                subject="Team sync?",
                content="Set up a 30-minute sync with Bob and Carol on Monday.",
            ),
            folder_name=EmailFolderName.INBOX,
        )
        self.apps = [self.email, self.aui]

    def build_events_flow(self) -> None:
        task = Event.from_function(
            self.aui.send_message_to_agent,
            content="Schedule the sync Alice asked for.",
        ).depends_on(None, delay_seconds=0)
        follow_up = Event.from_function(
            self.email.add_email,
            email=Email(
                sender="alice@corp.com",
                recipients=[_USER],
                subject="Re: Team sync?",
                content="Actually, make it Tuesday.",
            ),
            folder_name=EmailFolderName.INBOX,
        ).depends_on(None, delay_seconds=2)
        self.events = [task, follow_up]


async def _poll(produce: Callable[[], Awaitable[Any]], *, max_wait: float = 10.0) -> Any:
    """Poll ``produce`` (async, returns truthy or None) until it yields or ``max_wait``: the real
    Environment thread advances on wall-clock time, so the bridge is polled the same way."""
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        result = await produce()
        if result:
            return result
        await asyncio.sleep(0.1)
    return None


async def test_dynamic_scenario_bridges_task_and_timeline_email() -> None:
    sim = AreSimulation(_DynamicScenario())
    origin = WorkspaceOrigin(adapter="are-sim", address="insim:are")
    adapter = AreInProcessWorkspaceAdapter(workspace_id="are", origin=origin, simulation=sim)
    transport = AreTransport(sim)

    workspace = (await adapter.discover())[0]
    email_tool = next(t for t in workspace.tools() if "Email" in t.id)
    try:
        # ops arrive as bare names (ARE's <App>__ prefix stripped)
        assert "list_emails" in {op.name for op in email_tool.manual.operations}

        # the task is delivered by the scenario's AUI event -> a transport Message
        async def _task() -> list[object]:
            return [m async for m in transport.receive()]

        got = await _poll(_task)
        assert got and got[0].sender == "user"

        # the timeline injects a follow-up email off-cycle -> a state_changed signal
        sink: NotificationQueueSink[object] = NotificationQueueSink()
        await email_tool.focus(sink)

        async def _signal() -> list[object]:
            email_tool.observe()
            return [s async for _src, s in sink.drain()]

        signals = await _poll(_signal)
        assert signals and signals[0].name == "state_changed"

        # an app op invokes through the bridge
        ack = await email_tool.invoke("list_emails")
        assert ack.ok is True
    finally:
        await workspace.close()
