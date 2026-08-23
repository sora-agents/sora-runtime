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
from datetime import UTC, datetime
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
from are.simulation.types import Event, EventType  # noqa: E402

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


async def test_invoke_is_logged_as_are_agent_event() -> None:
    """A plain ``invoke`` already lands an ARE ``AGENT`` ``CompletedEvent`` in ``env.event_log`` —
    the thing Gaia2's event-graph judge matches against. ARE's app operation methods are decorated
    ``@event_registered(event_type=AGENT)``, so calling one inside a running ``Environment``
    self-registers the event; S-ORA does not (and must not) wrap invokes in ARE's ``register_event``
    on top, which would double-log. This test guards that invariant."""
    sim = AreSimulation(_DynamicScenario())
    origin = WorkspaceOrigin(adapter="are-sim", address="insim:are")
    adapter = AreInProcessWorkspaceAdapter(workspace_id="are", origin=origin, simulation=sim)

    workspace = (await adapter.discover())[0]
    email_tool = next(t for t in workspace.tools() if "Email" in t.id)
    try:

        def agent_events() -> list[Any]:
            env = sim._env
            assert env is not None  # started by discover()
            return [e for e in env.event_log.list_view() if e.event_type == EventType.AGENT]

        before = len(agent_events())
        ack = await email_tool.invoke("list_emails")
        assert ack.ok is True

        after = agent_events()
        # exactly one new AGENT event (no double-logging): the app + operation the judge keys on
        assert len(after) == before + 1
        assert after[-1].tool_name == "EmailClientApp__list_emails"
        assert after[-1].action.function_name == "list_emails"
    finally:
        await workspace.close()


# -- the simulated clock ---------------------------------------------------------------------------
#
# `Environment.run` copies `duration` and `time_increment_in_seconds` off the scenario but NOT
# `start_time`: that is read from the config alone, and `EnvironmentConfig` defaults it to None,
# which the Environment reads as 0. Left unset, every scenario ran with its clock at the Unix epoch
# — counting real seconds up from 1970-01-01 — while its data and its oracle sat in the scenario's
# own year. A goal like "cancel my appointments this upcoming Saturday" was then computed against
# 1970: a silent wrong answer, not an error, and one that looks like a model failure in a trace.

# `start_time` is a dataclass FIELD on ARE's `Scenario`, defaulted by a `time.time()` factory, so it
# has to be passed to the constructor — a class attribute on a subclass never reaches the instance.
_SCENARIO_EPOCH = 1728975600.0  # 2024-10-15 07:00:00 UTC, a Tuesday — a real Gaia2 start_time


def test_simulated_clock_starts_at_the_scenario_start_time() -> None:
    sim = AreSimulation(_DynamicScenario(start_time=_SCENARIO_EPOCH))
    sim.start()
    try:
        now = sim.environment().time_manager.time()
        # the scenario's own clock — not seconds counted up from the Unix epoch
        assert _SCENARIO_EPOCH <= now < _SCENARIO_EPOCH + 600
        moment = datetime.fromtimestamp(now, tz=UTC)
        assert moment.strftime("%Y-%m-%d") == "2024-10-15"
        assert moment.strftime("%A") == "Tuesday"
    finally:
        sim.stop()


def test_explicit_config_start_time_wins_over_the_scenario() -> None:
    """A caller who pinned the clock keeps it, and their config object is not mutated behind
    them."""
    from are.simulation.environment import EnvironmentConfig

    pinned = _SCENARIO_EPOCH + 86_400
    config = EnvironmentConfig(start_time=pinned)
    sim = AreSimulation(_DynamicScenario(start_time=_SCENARIO_EPOCH), config=config)
    sim.start()
    try:
        assert sim.environment().time_manager.time() >= pinned
        assert config.start_time == pinned  # copied, not mutated
    finally:
        sim.stop()


def test_undated_scenario_starts_now_rather_than_at_the_epoch() -> None:
    """An in-code scenario that names no date gets ARE's own default — the current wall-clock time,
    from the `Scenario.start_time` field factory — and that is what reaches the clock. This is the
    same rule ARE's ScenarioRunner applies (`if scenario.start_time and > 0`), and it is still a
    real date rather than 1970."""
    before = time.time()
    sim = AreSimulation(_DynamicScenario())
    sim.start()
    try:
        assert sim.environment().time_manager.time() >= before
    finally:
        sim.stop()
