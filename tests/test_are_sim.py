"""In-process ARE bridge — S-ORA-side contract, exercised against **fakes** (no ARE, runs in CI).

The adapter/transport depend only on a small duck-typed app/AUI interface, so plain fakes stand in
for live ARE apps. The real-Environment round-trip (a timeline actually firing) lives in the
integration-gated ``test_are_sim_integration.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from sora.adapters.are_sim import (
    AreInProcessWorkspaceAdapter,
    AreSimulation,
    AreTransport,
    ValidationOutcome,
    _params_schema,
    _returns_schema,
    _type_to_schema,
)
from sora.environment import WorkspaceOrigin
from sora.manual import Manual
from sora.perception import Message, NotificationQueueSink

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
    def __init__(
        self,
        name: str,
        fn: Any,
        *,
        args: list[FakeArg] | None = None,
        return_type: Any = None,
        return_description: str | None = None,
        write_operation: bool = False,
    ) -> None:
        self.name = name
        self.function = fn
        self.function_description = f"{name} description"
        self.args = args or []
        self.write_operation = write_operation
        self.return_type = return_type
        self.return_description = return_description

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
            FakeAppTool(
                "add_email", self.add_email, args=[FakeArg("subject")], write_operation=True
            ),
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

    def send_message_to_agent(self, content: str) -> None:  # the user side: user -> agent
        self._unread.append(SimpleNamespace(sender="User", content=content, timestamp=1.0))


class FakeSimulation:
    def __init__(self, apps: list[Any]) -> None:
        self._apps = apps
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def is_running(self) -> bool:
        return self.started and not self.stopped

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
    # ARE's AppTool.write_operation flows into OperationSpecification.side_effecting (ADR-0024):
    # add_email is a write, list_emails a read — the before_writes checkpoint keys on it.
    assert {op.name: op.side_effecting for op in manual.operations} == {
        "list_emails": False,
        "add_email": True,
    }
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


# A local mirror of ARE's Email/ReturnedEmails so the return-shape introspection is tested without
# importing ARE: search_emails returns list[Email] (a bare list), list_emails -> ReturnedEmails.
@dataclass
class _Email:
    sender: str
    email_id: str
    is_read: bool


@dataclass
class _ReturnedEmails:
    emails: list[_Email]
    total_emails: int


@dataclass
class _Node:
    # A minimal self-referential record: `get_type_hints` resolves `children` back to the live
    # class, so `_type_to_schema` would recurse forever without the depth cap. Module-level (not
    # local to a test) so that resolution against module globals actually finds `_Node`.
    label: str
    children: list[_Node]


def test_type_to_schema_expands_a_list_of_records_to_field_names() -> None:
    # The bug's core: search_emails returns a *bare* list[Email], so a resolvable $from path is
    # `0.email_id` — the schema must surface the record's field names, not a fictional wrapper.
    schema = _type_to_schema(list[_Email])
    assert schema["type"] == "array"
    assert schema["items"]["type"] == "object"
    assert set(schema["items"]["properties"]) == {"sender", "email_id", "is_read"}
    assert schema["items"]["properties"]["email_id"] == {"type": "string"}


def test_type_to_schema_expands_a_wrapped_record_of_records() -> None:
    schema = _type_to_schema(_ReturnedEmails)
    assert schema["type"] == "object"
    assert schema["properties"]["total_emails"] == {"type": "integer"}
    emails = schema["properties"]["emails"]
    assert emails["type"] == "array"
    assert set(emails["items"]["properties"]) == {"sender", "email_id", "is_read"}


def test_type_to_schema_handles_leaf_and_optional_types() -> None:
    assert _type_to_schema(str) == {"type": "string"}  # a bare id string (add_calendar_event)
    assert _type_to_schema(str | None) == {"type": "string"}  # unwrap X | None
    # An all-numeric union collapses to `number` (admits both), like the arg-type mapper; any other
    # heterogeneous union has no single faithful type, so it degrades to `string`.
    assert _type_to_schema(int | float) == {"type": "number"}
    assert _type_to_schema(int | str) == {"type": "string"}
    # A string annotation (an unresolved `from __future__` hint) routes through the union-aware
    # mapper, so a `X | None` *string* unwraps too — not the `string` a single-atom mapper gives.
    assert _type_to_schema("int | None") == {"type": "integer"}


def test_type_to_schema_bounds_a_self_referential_record_and_logs_at_the_cap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # No ARE app record is self-referential, but a foreign/future one could be. The depth cap must
    # bound it (no RecursionError) and emit one DEBUG line noting the elision, so a dropped $from
    # path has a trail. `_Node.children: list[_Node]` is the minimal cycle.
    with caplog.at_level(logging.DEBUG, logger="sora.adapters.are_sim"):
        schema = _type_to_schema(_Node)

    # Bounded: the deepest reached `children` is a bare array with no `items` (shape below elided).
    deepest = schema["properties"]["children"]["items"]["properties"]["children"]
    assert deepest == {"type": "array"}
    cap_logs = [r for r in caplog.records if "depth cap" in r.getMessage()]
    assert len(cap_logs) == 1
    assert cap_logs[0].levelno == logging.DEBUG


def test_returns_schema_seeds_the_description_and_is_none_without_a_type() -> None:
    tool = FakeAppTool(
        "search_emails",
        lambda **k: None,
        return_type=list[_Email],
        return_description="A list of emails that match the query.",
    )
    schema = _returns_schema(tool)
    assert schema is not None
    assert schema["type"] == "array"
    assert schema["description"] == "A list of emails that match the query."
    # No return_type declared -> no returns schema (an unannotated op, a hand-authored manual).
    assert _returns_schema(FakeAppTool("noop", lambda **k: None)) is None


def test_returns_schema_resolves_via_the_functions_own_annotation() -> None:
    # The production path: ARE's AppTool.return_type is a *string* (`from __future__ annotations`),
    # so `_returns_schema` resolves the real type off the underlying function's return annotation
    # via get_type_hints against its module globals — not the raw `return_type`. Here `return_type`
    # is deliberately left None so only the function-resolution path can produce a shape.
    def search() -> list[_Email]:  # annotation resolves against this module (where _Email lives)
        return []

    schema = _returns_schema(FakeAppTool("search_emails", search))
    assert schema is not None
    assert schema["type"] == "array"
    assert set(schema["items"]["properties"]) == {"sender", "email_id", "is_read"}


def test_returns_schema_is_none_for_a_void_operation() -> None:
    # A `-> None` op has no result to reference; it must declare no shape, not a fictitious `string`
    # leaf a planner could bind an empty-path $from against.
    def send() -> None:
        return None

    assert _returns_schema(FakeAppTool("send_email", send)) is None


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
    # Thin: the event names which app moved and WHERE, never a copy of the state (ADR-0004/0019).
    # The snapshot travels on the `state` observable property alone; `changes` carries identities
    # only, so it is the one thing the replace-by-key snapshot cannot express, duplicating nothing.
    assert drained[0].payload["app"] == app.app_name()
    changes = drained[0].payload["changes"]
    assert [c.path for c in changes] == ["emails"]
    assert len(changes[0].added) == 1  # the identity of the new email, not the email
    assert not any(isinstance(v, dict) for c in changes for v in c.added)
    assert len(props[0].value["emails"]) == 2  # property snapshot reflects the new email

    tool.observe()  # no further change -> no repeat signal
    assert [s async for s in sink.drain()] == []


async def test_observe_is_reentrant_from_the_push_screen() -> None:
    # Since the signal is thin, a push-time consumer (an InterruptPolicy) reads the state back off
    # the tool — i.e. it calls observe() from inside push(). That must terminate: observe() records
    # the new state BEFORE pushing, so the re-entrant call sees no diff and pushes nothing further.
    app = FakeEmailApp()
    sim = FakeSimulation([app])
    tool = (await _adapter(sim).discover())[0].tools()[0]
    sink: NotificationQueueSink[Any] = NotificationQueueSink()
    await tool.focus(sink)

    seen: list[Any] = []
    sink.on_push = lambda _source, _signal: seen.append(tool.observe())

    app.add_email("Follow-up: actually Tuesday")
    tool.observe()

    assert len(seen) == 1  # the screen ran once; its own observe() did not push again
    assert len(seen[0][0].value["emails"]) == 2  # and it saw the POST-change state
    assert len([sig async for _src, sig in sink.drain()]) == 1


async def test_observe_returns_the_state_the_reentrant_screen_advanced_to() -> None:
    # The push screen re-enters observe(). If ARE's thread mutates state in that window, the nested
    # call advances the tool's recorded state past what the outer call read — and the outer call is
    # what feeds the once-per-cycle property snapshot. It must return the newer state, not the
    # value it happened to read first, or working memory carries the pre-change world for a tick.
    app = FakeEmailApp()
    sim = FakeSimulation([app])
    tool = (await _adapter(sim).discover())[0].tools()[0]
    sink: NotificationQueueSink[Any] = NotificationQueueSink()
    await tool.focus(sink)

    mutated = False

    def screen(_source: Any, _signal: Any) -> None:
        nonlocal mutated
        if not mutated:  # mutate once, so the recursion terminates on the nested screen
            mutated = True
            app.add_email("Third, arriving mid-observe")  # the concurrent-mutation window
        tool.observe()  # the ADR-0020 pattern: read the live tool at push time

    sink.on_push = screen

    app.add_email("Follow-up: actually Tuesday")
    props = tool.observe()

    assert len(props[0].value["emails"]) == 3  # not the 2 the outer _read_state() saw


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


async def test_transport_submit_injects_an_ad_hoc_user_message() -> None:
    # A typed CLI line (or a /stop resume) reaches the agent via send_message_to_agent and surfaces
    # on the next receive() drain, indistinguishable from a scripted timeline message.
    aui = FakeAui()
    transport = AreTransport(FakeSimulation([aui]))
    msg = Message(sender="user", content={"text": "Never mind, continue"}, received_at=0.0)
    transport.submit(msg)
    got = [m async for m in transport.receive()]
    assert len(got) == 1
    assert got[0].sender == "user"
    assert got[0].content == {"text": "Never mind, continue"}


# ------------------------------------------------------------------------------------------------
# AreSimulation.validate() — surfaces ARE's in-band validation exception, preserves unscored None.
# ------------------------------------------------------------------------------------------------


def _sim_with_validate_result(result: Any) -> AreSimulation:
    # AreSimulation.validate() only needs a scenario with a validate(env) method and a non-None
    # _env (past the start() assert) — no ARE Environment required.
    sim = AreSimulation(SimpleNamespace(validate=lambda env: result))
    sim._env = object()
    return sim


def test_validate_raises_when_judge_errors_in_band() -> None:
    # ARE reports a judge/validator error as success=None *with* an in-band exception on the result
    # (it does not raise). validate() must re-raise it so the caller records an 'exception', not a
    # silent unscored run — otherwise a judge crash is indistinguishable from 'no judge attached'.
    boom = RuntimeError("judge boom")
    sim = _sim_with_validate_result(
        SimpleNamespace(success=None, exception=boom, rationale="graph error")
    )
    with pytest.raises(RuntimeError, match="judge boom"):
        sim.validate()


def test_validate_preserves_unscored_none_without_exception() -> None:
    # success=None and no exception is a genuine unscored verdict, not an error — pass it through.
    sim = _sim_with_validate_result(SimpleNamespace(success=None, exception=None, rationale=None))
    assert sim.validate() == ValidationOutcome(success=None, rationale=None)


def test_validate_passes_through_scored_verdicts() -> None:
    sim = _sim_with_validate_result(
        SimpleNamespace(success=False, exception=None, rationale="wrong tool")
    )
    assert sim.validate() == ValidationOutcome(success=False, rationale="wrong tool")
