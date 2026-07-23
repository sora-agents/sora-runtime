"""In-process ARE bridge — S-ORA-side contract, exercised against **fakes** (no ARE, runs in CI).

The adapter/transport depend only on a small duck-typed app/AUI interface, so plain fakes stand in
for live ARE apps. The real-Environment round-trip (a timeline actually firing) lives in the
integration-gated ``test_are_sim_integration.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from sora.adapters.are_sim import (
    AreInProcessWorkspaceAdapter,
    AreTransport,
    _params_schema,
)
from sora.environment import WorkspaceOrigin
from sora.manual import Manual
from sora.perception import NotificationQueueSink

# ------------------------------------------------------------------------------------------------
# Fakes: minimal stand-ins for ARE App / AppTool / AgentUserInterface / the simulation runtime.
# ------------------------------------------------------------------------------------------------


class FakeArg:
    def __init__(self, name: str, arg_type: str = "str", *, has_default: bool = False) -> None:
        self.name = name
        self.arg_type = arg_type
        self.description = f"{name} arg"
        self.has_default = has_default
        self.default = None


class FakeAppTool:
    def __init__(self, name: str, fn: Any, *, args: list[FakeArg] | None = None) -> None:
        self.name = name
        self.function = fn
        self.function_description = f"{name} description"
        self.args = args or []
        self.write_operation = False

    def __call__(self, **kwargs: Any) -> Any:
        return self.function(**kwargs)


class FakeEmailApp:
    def __init__(self) -> None:
        self._emails: list[dict[str, str]] = [{"id": "e1", "subject": "Team sync?"}]

    def app_name(self) -> str:
        return "EmailClientApp"

    def get_state(self) -> dict[str, Any]:
        return {"emails": [dict(e) for e in self._emails]}

    def list_emails(self) -> list[dict[str, str]]:
        return [dict(e) for e in self._emails]

    def add_email(self, subject: str) -> str:
        self._emails.append({"id": f"e{len(self._emails) + 1}", "subject": subject})
        return "added"

    def get_tools(self) -> list[FakeAppTool]:
        return [
            FakeAppTool("list_emails", self.list_emails),
            FakeAppTool("add_email", self.add_email, args=[FakeArg("subject")]),
        ]


class FlakyStateApp:
    """get_state raises the concurrent-modification RuntimeError on its first N calls, then
    succeeds — models ARE's event-loop thread mutating app state mid-read (no shared lock)."""

    def __init__(self, *, fail_times: int) -> None:
        self._remaining_failures = fail_times
        self.calls = 0

    def app_name(self) -> str:
        return "FlakyApp"

    def get_tools(self) -> list[FakeAppTool]:
        return []

    def get_state(self) -> dict[str, Any]:
        self.calls += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise RuntimeError("dictionary changed size during iteration")
        return {"calls": self.calls}


class FakeAui:
    def __init__(self) -> None:
        self.sent_to_user: list[str] = []
        self._unread: list[Any] = []
        self.wait_for_user_response = True

    def app_name(self) -> str:
        return "AgentUserInterface"

    def get_state(self) -> dict[str, Any]:
        return {"sent": len(self.sent_to_user)}

    def get_tools(self) -> list[FakeAppTool]:
        return []

    def deliver_user_message(  # what the timeline / user proxy does
        self, content: str, timestamp: float = 1.0
    ) -> None:
        self._unread.append(SimpleNamespace(sender="User", content=content, timestamp=timestamp))

    def get_last_unread_messages(self) -> list[Any]:
        msgs, self._unread = list(self._unread), []
        return msgs

    def send_message_to_user(self, content: str) -> None:
        self.sent_to_user.append(content)


class FakeSimulation:
    def __init__(self, apps: list[Any]) -> None:
        self._apps = apps
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def apps(self) -> list[Any]:
        return list(self._apps)

    @property
    def aui(self) -> Any:
        return next((a for a in self._apps if a.app_name() == "AgentUserInterface"), None)

    def run(self, fn: Any) -> Any:
        return fn()


def _origin() -> WorkspaceOrigin:
    return WorkspaceOrigin(adapter="are-sim", address="insim:are")


def _adapter(sim: FakeSimulation, **kw: Any) -> AreInProcessWorkspaceAdapter:
    return AreInProcessWorkspaceAdapter(workspace_id="are", origin=_origin(), simulation=sim, **kw)


# ------------------------------------------------------------------------------------------------
# Adapter: discovery, manual, invoke
# ------------------------------------------------------------------------------------------------


async def test_discover_builds_one_tool_per_app_excluding_aui() -> None:
    sim = FakeSimulation([FakeEmailApp(), FakeAui()])
    workspaces = await _adapter(sim).discover()
    assert sim.started is True
    tools = workspaces[0].tools()
    assert [t.id for t in tools] == ["insim:are/EmailClientApp"]  # AUI is not a tool
    manual = tools[0].manual
    assert {op.name for op in manual.operations} == {"list_emails", "add_email"}
    assert [p.name for p in manual.observable_properties] == ["state"]
    assert [s.name for s in manual.signals] == ["state_changed"]


def test_params_schema_marks_required_and_types() -> None:
    tool = FakeAppTool(
        "add_email", lambda **k: None, args=[FakeArg("subject"), FakeArg("cc", has_default=True)]
    )
    schema = _params_schema(tool)
    assert schema["properties"]["subject"]["type"] == "string"
    assert schema["required"] == ["subject"]  # cc has a default -> not required


def test_params_schema_maps_list_types_to_arrays() -> None:
    # A container arg must reach the grounding model as an array, not be collapsed to "string" —
    # otherwise the model fills e.g. attendees with "Alice, Bob" and ARE's type-check rejects it.
    tool = FakeAppTool(
        "add_event",
        lambda **k: None,
        args=[FakeArg("attendees", "list[str] | None", has_default=True)],
    )
    schema = _params_schema(tool)
    assert schema["properties"]["attendees"] == {
        "type": "array",
        "items": {"type": "string"},
        "description": "attendees arg",
    }


def test_params_schema_maps_numeric_union_to_number() -> None:
    # `int | float | None` is a multi-member union; it must reach the model as JSON "number"
    # (admits both), not collapse to "string" — else ARE's runtime check rejects a "1500".
    tool = FakeAppTool(
        "search",
        lambda **k: None,
        args=[FakeArg("min_price", "int | float | None", has_default=True)],
    )
    schema = _params_schema(tool)
    assert schema["properties"]["min_price"]["type"] == "number"


async def test_invoke_calls_the_app_op_and_returns_ack() -> None:
    sim = FakeSimulation([FakeEmailApp()])
    tool = (await _adapter(sim).discover())[0].tools()[0]
    ack = await tool.invoke("list_emails")
    assert ack.ok is True
    assert ack.result == [{"id": "e1", "subject": "Team sync?"}]


async def test_invoke_unknown_operation_is_a_failed_ack() -> None:
    sim = FakeSimulation([FakeEmailApp()])
    tool = (await _adapter(sim).discover())[0].tools()[0]
    ack = await tool.invoke("nope")
    assert ack.ok is False


async def test_discover_merges_authored_manual_when_source_resolves_one() -> None:
    authored = Manual(
        id="EmailClientApp",
        metadata={},
        description="",
        observable_properties=[],
        signals=[],
        operations=[],
        raw_text="# Email\nemail_id always comes from list_emails.",
    )

    class _Source:
        async def get(self, manual_id: str) -> Manual | None:
            return authored if manual_id == "EmailClientApp" else None

    sim = FakeSimulation([FakeEmailApp()])
    tool = (await _adapter(sim, manual_source=_Source()).discover())[0].tools()[0]
    assert tool.manual.raw_text is not None  # prose channel merged in
    assert {op.name for op in tool.manual.operations} == {
        "list_emails",
        "add_email",
    }  # adapter specs kept


# ------------------------------------------------------------------------------------------------
# Adapter: observe emits state_changed on a diff (the in-process signal path)
# ------------------------------------------------------------------------------------------------


async def test_observe_emits_signal_only_on_state_change() -> None:
    app = FakeEmailApp()
    sim = FakeSimulation([app])
    tool = (await _adapter(sim).discover())[0].tools()[0]
    sink: NotificationQueueSink[Any] = NotificationQueueSink()
    await tool.focus(sink)

    tool.observe()  # state unchanged since focus primed the cache -> no signal
    assert [s async for s in sink.drain()] == []

    app.add_email("Follow-up: actually Tuesday")  # an off-cycle (timeline-style) change
    props = tool.observe()
    drained = [sig async for _src, sig in sink.drain()]
    assert len(drained) == 1 and drained[0].name == "state_changed"
    assert len(props[0].value["emails"]) == 2  # property snapshot reflects the new email

    tool.observe()  # no further change -> no repeat signal
    assert [s async for s in sink.drain()] == []


async def test_read_state_retries_past_a_transient_concurrent_modification() -> None:
    # The ARE event-loop thread can mutate app state while observe() reads it; a get_state() that
    # raises "changed size during iteration" once must be retried, not propagated as a crash.
    app = FlakyStateApp(fail_times=1)
    tool = (await _adapter(FakeSimulation([app])).discover())[0].tools()[0]
    props = tool.observe()  # first read fails once, retry succeeds
    assert props[0].value == {"calls": app.calls}
    assert app.calls >= 2  # proves a retry happened


async def test_read_state_gives_up_after_exhausting_retries() -> None:
    app = FlakyStateApp(fail_times=99)  # never settles
    tool = (await _adapter(FakeSimulation([app])).discover())[0].tools()[0]
    with pytest.raises(RuntimeError, match="changed size"):
        tool.observe()


async def test_unfocus_stops_signal_emission() -> None:
    app = FakeEmailApp()
    sim = FakeSimulation([app])
    tool = (await _adapter(sim).discover())[0].tools()[0]
    sink: NotificationQueueSink[Any] = NotificationQueueSink()
    await tool.focus(sink)
    await tool.unfocus()
    app.add_email("change after unfocus")
    tool.observe()
    assert [s async for s in sink.drain()] == []


# ------------------------------------------------------------------------------------------------
# Transport over the AUI
# ------------------------------------------------------------------------------------------------


async def test_transport_receive_yields_unread_user_messages_once() -> None:
    aui = FakeAui()
    aui.deliver_user_message("schedule a sync with Bob and Carol")
    transport = AreTransport(FakeSimulation([FakeEmailApp(), aui]))

    got = [m async for m in transport.receive()]
    assert len(got) == 1
    assert got[0].sender == "user"
    assert got[0].content == {"text": "schedule a sync with Bob and Carol"}

    assert [m async for m in transport.receive()] == []  # already read


async def test_transport_preserves_a_zero_relative_timestamp() -> None:
    # The t0 task message has sim-relative timestamp 0.0; it must survive, not be overwritten with
    # wall-clock time by a `... or time.time()` falsy check.
    aui = FakeAui()
    aui.deliver_user_message("the t0 task", timestamp=0.0)
    transport = AreTransport(FakeSimulation([aui]))
    got = [m async for m in transport.receive()]
    assert got[0].received_at == 0.0


async def test_transport_send_posts_to_the_user() -> None:
    aui = FakeAui()
    transport = AreTransport(FakeSimulation([aui]))
    await transport.send("user", {"text": "Booked Monday 10:00 with Bob and Carol."})
    assert aui.sent_to_user == ["Booked Monday 10:00 with Bob and Carol."]
