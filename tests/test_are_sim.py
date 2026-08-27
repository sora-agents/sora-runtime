"""In-process ARE bridge — S-ORA-side contract, exercised against **fakes** (no ARE, runs in CI).

The adapter/transport depend only on a small duck-typed app/AUI interface, so plain fakes stand in
for live ARE apps. The real-Environment round-trip (a timeline actually firing) lives in the
integration-gated ``test_are_sim_integration.py``.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, TypedDict

import pytest

from sora.adapters.are_sim import (
    AreInProcessWorkspaceAdapter,
    AreSimulation,
    AreTransport,
    ValidationOutcome,
    _params_schema,
    _record_fields,
    _returns_schema,
    _type_to_schema,
    relax_judge_verdict_case,
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


# Mirrors of ARE's *TypedDict* envelopes. ARE spells its plain records as dataclasses but every
# paginated envelope as a TypedDict, so this is the shape that used to fall through to a bare
# `string` — see `_record_fields`. `_CalendarEventsResult` is the four-level shape that sets the
# depth cap: envelope -> list of records -> a record's own list of scalars.
@dataclass
class _CalendarEvent:
    event_id: str
    title: str
    attendees: list[str]


class _CalendarEventsResult(TypedDict):
    events: list[_CalendarEvent]
    range: tuple[int, int]
    total: int


class _ProductMetadata(TypedDict):
    range: tuple[int, int]
    total: int


class _ProductListResult(TypedDict):
    # A TypedDict whose own field is another TypedDict — recognizing the outer one is not enough.
    products: dict[str, str]
    metadata: _ProductMetadata


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


def test_type_to_schema_expands_a_typed_dict_envelope() -> None:
    # The gap this closes: ARE spells `get_calendar_events_from_to -> CalendarEventsResult` as a
    # TypedDict, and a dataclass-only record branch let it fall through to a bare `string`. A
    # planner reading that manual is told the op returns a scalar, so there is no `$from` path to
    # author into `events` at all — the very ops whose payload most needs indexing were the ones
    # declaring the least. The whole point is the *named fields*, so assert them, not just the type.
    schema = _type_to_schema(_CalendarEventsResult)
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"events", "range", "total"}
    assert schema["properties"]["total"] == {"type": "integer"}
    events = schema["properties"]["events"]
    assert events["type"] == "array"
    assert set(events["items"]["properties"]) == {"event_id", "title", "attendees"}


def test_type_to_schema_expands_a_typed_dict_nested_in_a_typed_dict() -> None:
    # ARE's ProductListResult carries a ProductMetadata: recognizing only the outer envelope would
    # leave the inner one a `string` leaf. A `dict[str, str]` field stays a bare `object` — its keys
    # are data (product names), so there are no field names to declare.
    schema = _type_to_schema(_ProductListResult)
    assert schema["properties"]["products"] == {"type": "object"}
    metadata = schema["properties"]["metadata"]
    assert metadata["type"] == "object"
    assert set(metadata["properties"]) == {"range", "total"}


def test_the_depth_cap_admits_the_deepest_real_are_shape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # envelope -> list of records -> a record's own list of scalars is four levels, and it is the
    # deepest shape ARE actually returns. The cap must not clip it: `attendees` losing its element
    # type is exactly the elision that matters, since attendees is what a plan paths into. No cap
    # log either — a DEBUG line here would be a false lead on a shape that resolved fine.
    with caplog.at_level(logging.DEBUG, logger="sora.adapters.are_sim"):
        schema = _type_to_schema(_CalendarEventsResult)

    attendees = schema["properties"]["events"]["items"]["properties"]["attendees"]
    assert attendees == {"type": "array", "items": {"type": "string"}}
    assert [r for r in caplog.records if "depth cap" in r.getMessage()] == []


def test_record_fields_reads_both_record_spellings_and_nothing_else() -> None:
    assert _record_fields(_ProductMetadata) is not None
    assert set(_record_fields(_ProductMetadata) or {}) == {"range", "total"}
    assert set(_record_fields(_Email) or {}) == {"sender", "email_id", "is_read"}
    # Not records: a plain dict (arbitrary keys, no field names), a primitive, a parameterized
    # generic. Each must fall through to its own branch rather than being read as an empty record.
    assert _record_fields(dict) is None
    assert _record_fields(str) is None
    assert _record_fields(list[_Email]) is None


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

    # Bounded: descend `children` -> `items` as far as the schema goes and assert it *ends* — in a
    # leaf or an `items`-less array — rather than pinning the exact number of levels, which is the
    # tunable `_MAX_RETURN_DEPTH` and not what this test is about.
    node: dict[str, Any] = schema
    levels = 0
    while "properties" in node:
        children = node["properties"]["children"]
        assert children["type"] == "array"
        if "items" not in children:
            break  # clipped: shape below the cap elided
        node = children["items"]
        levels += 1
    assert 0 < levels < 10  # terminated, and not by exhausting the recursion limit
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
# AreSimulation.timeline_expired() — "ARE's clock ended the run", not "the agent did nothing".
#
# ARE's event loop sleeps one real second per tick, so a scenario's `duration` is a wall-clock
# budget for the agent (1000s by default for a JSON benchmark scenario). Spend it on inference and
# the environment stops mid-run: no later turn is delivered, because a Gaia2 turn is released by a
# ConditionCheckEvent only the live loop ticks. The judge then reports the turn index never
# advanced and the write-count gate reports the whole turn's oracle calls as missing — the same
# output a capable agent that chose to do nothing would produce.
# ------------------------------------------------------------------------------------------------


def _sim_with_clock(duration: Any, passed: float) -> AreSimulation:
    # timeline_expired() reads only env.duration and env.time_manager.time_passed(), so it needs no
    # ARE Environment — same white-box shortcut as _sim_with_validate_result above.
    sim = AreSimulation(SimpleNamespace())
    sim._env = SimpleNamespace(
        duration=duration, time_manager=SimpleNamespace(time_passed=lambda: passed)
    )
    sim._started = True
    return sim


def test_timeline_expired_mirrors_ares_own_loop_exit_test() -> None:
    # ARE's loop runs `while time_passed() <= duration`, so equality is still *running* — the run
    # has not expired on the tick that exactly reaches the budget. Pinned to the boundary because
    # an off-by-one here would label an ordinary completed run as truncated, which is worse than
    # not reporting at all: it would excuse a real agent failure.
    assert _sim_with_clock(1000, 999.0).timeline_expired() is False
    assert _sim_with_clock(1000, 1000.0).timeline_expired() is False
    assert _sim_with_clock(1000, 1000.5).timeline_expired() is True


def test_timeline_expired_is_false_when_the_scenario_runs_indefinitely() -> None:
    # ARE reads duration=None as "no limit", so there is no budget to overrun however long it ran.
    assert _sim_with_clock(None, 10_000.0).timeline_expired() is False


def test_timeline_expired_is_false_before_the_simulation_starts() -> None:
    sim = AreSimulation(SimpleNamespace())
    assert sim.timeline_expired() is False


def test_timeline_expired_never_raises_when_the_clock_is_unreadable() -> None:
    # A diagnostic must never cost the run its real result: a probe that raised here would turn a
    # scored run into an 'exception' record.
    sim = AreSimulation(SimpleNamespace())
    sim._env = SimpleNamespace(duration=1000, time_manager=SimpleNamespace(time_passed=_boom))
    sim._started = True
    assert sim.timeline_expired() is False


def _boom() -> float:
    raise RuntimeError("clock gone")


def _stoppable_sim(duration: Any, clock: list[float]) -> AreSimulation:
    """A sim whose clock keeps ticking after stop() — which is what ARE's really does.
    `Environment.stop()` sets the stop event and the state but never pauses the TimeManager, so
    `time_passed()` = `time.time() - real_start + offset` goes right on advancing afterwards.
    A frozen clock is what hid both defects below."""
    sim = AreSimulation(SimpleNamespace())
    sim._env = SimpleNamespace(
        duration=duration,
        time_manager=SimpleNamespace(time_passed=lambda: clock[0]),
        stop=lambda: None,
    )
    sim._started = True
    return sim


def test_timeline_expired_still_answers_after_the_simulation_is_stopped() -> None:
    """The shipped path only ever asks *after* the run: the session's teardown leaves every joined
    workspace, which closes the ARE workspace, which calls stop(). A probe that goes quiet once the
    simulation is stopped is dead exactly where it is needed, while still reading True in a test
    that never stopped it."""
    clock = [5000.0]
    sim = _stoppable_sim(1000, clock)
    assert sim.timeline_expired() is True  # while running
    sim.stop()
    assert sim.timeline_expired() is True  # and after, which is when anyone actually asks


def test_timeline_expiry_probe_does_not_guard_on_started() -> None:
    """The probe is a post-mortem, not a liveness check — unlike is_running/is_paused, which are
    rightly False once the run is over. stop()'s latch normally answers first, so this pins the
    fallback directly: any path that clears `_started` without latching (a future teardown, a
    caller stopping ARE's Environment itself) must still get the verdict rather than a flat False.
    """
    sim = _stoppable_sim(1000, [5000.0])
    sim._started = False  # cleared without a latch
    assert sim._expired is None

    assert sim.timeline_expired() is True


def test_timeline_expiry_is_latched_at_stop_not_at_read() -> None:
    """A run that finished well inside its budget must not drift into "expired" just because its
    result was read slowly — and the caller reads this after validate(), whose judge pass over the
    oracle graph can take minutes against a wall clock nothing paused."""
    clock = [10.0]
    sim = _stoppable_sim(1000, clock)
    sim.stop()
    clock[0] = 99_999.0  # the judge pass, as ARE's clock sees it

    assert sim.timeline_expired() is False


def test_timeline_expiry_latched_at_stop_survives_a_later_read() -> None:
    # The converse: a genuinely expired run stays expired no matter when the verdict is read.
    clock = [1500.0]
    sim = _stoppable_sim(1000, clock)
    sim.stop()
    clock[0] = 1500.0

    assert sim.timeline_expired() is True


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


# --- ARE's case-sensitive judge-verdict parse (upstream defect) ---------------------------------
#
# `relax_judge_verdict_case` patches a class it imports from ARE, which is an optional extra. These
# stub that module in `sys.modules` so the contract is exercised in CI either way — and so the test
# pins OUR replacement's behavior rather than whatever ARE happens to ship.


class _StubChecker:
    """Mirrors the surface `relax_judge_verdict_case` replaces: ARE's `LLMChecker`, whose `__call__`
    tallies votes with a case-SENSITIVE membership test and returns None when nothing parsed."""

    def __init__(
        self, response: str, success: str = "[[True]]", failure: str = "[[False]]"
    ) -> None:
        self.success_str = success
        self.failure_str = failure
        self.num_votes = 1

        def judge(_args: dict[str, str]) -> str:
            return response

        self.judge = judge

    def __call__(self, user_prompt_args: dict[str, str]) -> bool | None:
        votes: list[bool] = []
        for _ in range(self.num_votes):
            response = self.judge(user_prompt_args)
            if response is None:
                continue
            if self.success_str in response:
                votes.append(True)
            elif self.failure_str in response:
                votes.append(False)
        if len(votes) == 0:
            return None
        return sum(votes) >= len(votes) / 2


def _stub_are(monkeypatch: pytest.MonkeyPatch) -> type[_StubChecker]:
    """Install a fresh stub `LLMChecker` under ARE's import path and return the class."""

    class LLMChecker(_StubChecker):
        pass

    module = SimpleNamespace(LLMChecker=LLMChecker)
    monkeypatch.setitem(sys.modules, "are.simulation.validation.utils.llm_utils", module)
    return LLMChecker


def test_are_discards_a_lowercase_verdict_before_the_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    # The defect itself, stated as a test: the judge signed off, and ARE records no vote at all.
    checker = _stub_are(monkeypatch)("The tone is polite and suitable. Evaluation: [[true]]")
    assert checker({}) is None  # not False — the verdict was discarded, not disagreed with


def test_relax_judge_verdict_case_reads_a_lowercase_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    cls = _stub_are(monkeypatch)
    assert relax_judge_verdict_case() is True
    assert cls("The tone is polite and suitable. Evaluation: [[true]]")({}) is True


def test_relax_judge_verdict_case_reads_a_lowercase_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Relaxing the case must not turn every verdict into a pass — a real rejection still rejects.
    cls = _stub_are(monkeypatch)
    relax_judge_verdict_case()
    assert cls("The signature names someone else. Evaluation: [[false]]")({}) is False


def test_relax_judge_verdict_case_keeps_none_for_an_unparseable_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only the marker's CASE is relaxed; a response carrying no marker still records no vote, so a
    # genuinely broken judge is still distinguishable from one that answered.
    cls = _stub_are(monkeypatch)
    relax_judge_verdict_case()
    assert cls("I am not sure what to make of this email.")({}) is None


def test_relax_judge_verdict_case_leaves_the_other_marker_family_working(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Four checkers use [[True]]; the rest use [[Success]]. Both must keep parsing.
    cls = _stub_are(monkeypatch)
    relax_judge_verdict_case()
    checker = cls("Evaluation: [[success]]", success="[[Success]]", failure="[[Failure]]")
    assert checker({}) is True


def test_relax_judge_verdict_case_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_are(monkeypatch)
    assert relax_judge_verdict_case() is True
    assert relax_judge_verdict_case() is False  # already patched — not re-wrapped


def test_relax_judge_verdict_case_is_a_no_op_without_are(monkeypatch: pytest.MonkeyPatch) -> None:
    # ARE is an optional extra, so the patch must degrade quietly rather than break an import.
    monkeypatch.setitem(sys.modules, "are.simulation.validation.utils.llm_utils", None)
    assert relax_judge_verdict_case() is False


def test_ares_own_engine_mangles_a_true_verdict_before_the_checker_sees_it() -> None:
    """The root cause, pinned against the real package: the casing is ARE's, not the model's.

    Both engines ARE ships end ``chat_completion`` with ``.replace("True", "true")`` — a
    JSON-shaped normalization of *agent* output that also rewrites judge verdicts in transit.
    ``create_judge_engine`` returns a ``LiteLLMEngine``, so ``success_str="[[True]]"`` is
    unsatisfiable on the shipped path regardless of which model answers, and the
    ``[[True]]``-family checkers can only ever return ``None``. Uses LiteLLM's ``mock_response``,
    so no network and no model — the "model output" is fixed here and we observe the transit.
    """
    litellm_engine = pytest.importorskip("are.simulation.agents.llm.litellm.litellm_engine")
    llm_utils = pytest.importorskip("are.simulation.validation.utils.llm_utils")
    prompts = pytest.importorskip("are.simulation.validation.prompts")

    def engine(verdict: str) -> Any:
        eng = litellm_engine.LiteLLMEngine(
            model_config=litellm_engine.LiteLLMModelConfig(model_name="gpt-4o", provider="openai")
        )
        eng.mock_response = verdict
        return eng

    assert engine("[[True]]")([{"role": "user", "content": "x"}])[0] == "[[true]]"
    assert engine("[[False]]")([{"role": "user", "content": "x"}])[0] == "[[false]]"
    # The other marker family survives the replace — which is why only some tools are affected.
    assert engine("[[Success]]")([{"role": "user", "content": "x"}])[0] == "[[Success]]"

    def verdict(response: str) -> bool | None:
        checker = llm_utils.LLMChecker(
            engine=engine(response),
            prompt_templates=prompts.SIGNATURE_CHECKER_TEMPLATES,
            num_votes=1,
            success_str="[[True]]",
            failure_str="[[False]]",
        )
        result: bool | None = checker({"agent_action_call": "hi", "user_name": "n"})
        return result

    # Not False — no vote was recorded at all, for a pass *and* for a fail.
    assert verdict("Evaluation: [[True]]") is None
    assert verdict("Evaluation: [[False]]") is None

    relax_judge_verdict_case()
    assert verdict("Evaluation: [[True]]") is True
    assert verdict("Evaluation: [[False]]") is False
