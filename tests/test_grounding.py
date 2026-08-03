"""Parameter grounding — references resolved mechanically, escalated to the model when they can't.

A plan is a reusable *skeleton*; a param whose value depends on a prior step's result is emitted as
a *reference* the Reason phase grounds each run against the activity's execution history. Two layers
are pinned here:

* ``resolve_references`` — the pure, deterministic resolver (hard ``$from``/``path`` refs; anything
  it can't resolve is reported for escalation, never raised);
* ``DefaultReasonStrategy`` grounding — resolve mechanically when possible (no model call), else
  escalate to ``procedural.ground`` (one model call), while the *stored* plan keeps its references.

Grounding lives in Reason (deciding a value is reasoning); Act stays mechanistic. See ADR-0017.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from fakes import FakeAdapter, FakeTool, FakeWorkspace
from sora.action import default_action_registry, invoke_step
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, Tool, WorkspaceOrigin
from sora.manual import Manual, OperationSpecification
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    PerceptSnapshot,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from sora.perception import Message, Percept
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReasonStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    Strategies,
    TickResult,
    resolve_references,
)
from sora.transport import MessageTransport
from sora.types import (
    CompletedOperation,
    ObservableProperty,
    OperationAck,
    OperationInvocation,
    Plan,
    Signal,
    Step,
)

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")


def _history(operation_name: str, result: object, *, ok: bool = True) -> CompletedOperation:
    return CompletedOperation(
        OperationInvocation("email", operation_name, {}), OperationAck(ok=ok, result=result)
    )


# --------------------------------------------------------------------------------------------------
# resolve_references — the pure deterministic layer
# --------------------------------------------------------------------------------------------------


def test_hard_reference_resolves_from_history() -> None:
    history = [_history("search_emails", {"emails": [{"id": 42}]})]
    params = {"email_id": {"$from": "search_emails", "path": "emails.0.id"}, "body": "hi"}
    resolved, unresolved = resolve_references(params, history)
    assert resolved == {"email_id": 42, "body": "hi"}
    assert unresolved == []


def test_concrete_params_pass_through_untouched() -> None:
    resolved, unresolved = resolve_references({"folder": "inbox", "limit": 5}, [])
    assert resolved == {"folder": "inbox", "limit": 5}
    assert unresolved == []  # nothing to resolve -> no escalation


def test_bad_path_missing_step_and_soft_ref_are_unresolved() -> None:
    history = [_history("search_emails", {"emails": [{"id": 42}]})]
    cases = [
        {"x": {"$from": "search_emails", "path": "emails.9.id"}},  # index out of range
        {"x": {"$from": "search_emails", "path": "no_such_key"}},  # bad key
        {"x": {"$from": "never_ran", "path": "a"}},  # no matching history entry
        {"x": {"$decide": "the right id"}},  # soft ref always escalates
    ]
    for params in cases:
        resolved, unresolved = resolve_references(params, history)
        assert unresolved == ["x"], params
        assert resolved["x"] == params["x"]  # left in place for the escalation to replace


def test_bare_list_result_resolves_by_index_then_field() -> None:
    # ARE's search_emails returns a *bare* list[Email] keyed by `email_id` — so the resolvable path
    # is `0.email_id` / `1.email_id`. Surfacing that return shape to the planner (so it emits these
    # instead of the fictional `emails.0.id`) is what keeps both emails read mechanically, with no
    # model escalation — the fix for the second email never being read.
    history = [
        _history(
            "search_emails",
            [{"email_id": "followup", "content": "Tuesday"}, {"email_id": "original"}],
        )
    ]
    ref0 = {"id": {"$from": "search_emails", "path": "0.email_id"}}
    ref1 = {"id": {"$from": "search_emails", "path": "1.email_id"}}
    first, u1 = resolve_references(ref0, history)
    second, u2 = resolve_references(ref1, history)
    assert (first["id"], u1) == ("followup", [])
    assert (second["id"], u2) == ("original", [])  # the second record is reachable, not truncated
    # The old guessed shape (a fictional `emails` wrapper) does NOT resolve against a bare list —
    # it escalates, which is exactly the failure the returns rendering removes.
    _, u3 = resolve_references({"id": {"$from": "search_emails", "path": "emails.0.id"}}, history)
    assert u3 == ["id"]


def test_latest_matching_history_entry_wins() -> None:
    history = [_history("list", {"v": 1}), _history("list", {"v": 2})]
    resolved, _ = resolve_references({"x": {"$from": "list", "path": "v"}}, history)
    assert resolved["x"] == 2  # most recent, not the first


# --------------------------------------------------------------------------------------------------
# DefaultReasonStrategy grounding — mechanistic when it can be, model escalation when it must be
# --------------------------------------------------------------------------------------------------


class ScriptedProcedural(ProceduralMemory):
    """Spies ``ground`` (and would raise if it were called without being configured) so a test can
    prove the *mechanistic* path took no model call, and assert the escalation payload."""

    def __init__(self, *, ground_result: dict[str, object] | None = None) -> None:
        super().__init__(FileMemoryBackend("unused"))
        self._ground_result = ground_result
        self.ground_calls: list[tuple[str, dict[str, object]]] = []
        self.ground_percepts: list[PerceptSnapshot] = []

    async def ground(
        self,
        activity: Activity,
        operation_name: str,
        manual: Manual | None,
        partial_params: dict[str, object],
        observed: PerceptSnapshot | None = None,
    ) -> dict[str, object]:
        self.ground_calls.append((operation_name, dict(partial_params)))
        self.ground_percepts.append(observed or PerceptSnapshot())
        if self._ground_result is None:
            raise AssertionError("ground() called but no ground_result configured")
        return self._ground_result


class _NullTransport:
    async def send(self, to: str, content: dict[str, object]) -> None: ...

    def receive(self) -> AsyncIterator[Message]:
        async def _drain() -> AsyncIterator[Message]:
            return
            yield  # pragma: no cover — never-yielding async generator

        return _drain()


def _cycle(
    tmp_path: Path, procedural: ProceduralMemory, tool: Tool
) -> tuple[DecisionCycle, WorkingMemory, EnvironmentRegistry]:
    registry = EnvironmentRegistry(
        adapters={_ORIGIN: FakeAdapter("fake", FakeWorkspace("ws", _ORIGIN, [tool]))}
    )
    working = WorkingMemory(registry=registry)
    transport: MessageTransport = _NullTransport()
    strategies = Strategies(
        observe=DefaultObserveStrategy(),
        reflect=DefaultReflectStrategy(),
        situate=DefaultSituateStrategy(),
        reason=DefaultReasonStrategy(),
        act=DefaultActStrategy(),
    )
    cycle = DecisionCycle(
        strategies=strategies,
        communication=transport,
        actions=default_action_registry(),
        registry=registry,
        working=working,
        semantic=SemanticMemory(FileMemoryBackend(tmp_path / "semantic")),
        procedural=procedural,
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "episodic")),
    )
    return cycle, working, registry


async def test_reason_grounds_reference_mechanically_without_model(tmp_path: Path) -> None:
    tool = FakeTool("email", invoke_results={"reply_to_email": {"sent": True}})
    spy = ScriptedProcedural()  # ground() would raise if reached
    cycle, working, registry = _cycle(tmp_path, spy, tool)
    await registry.join(_ORIGIN)
    ref = {"$from": "list_emails", "path": "emails.0.id"}
    plan_step = invoke_step("email", "reply_to_email", email_id=ref, body="hi")
    activity = Activity(
        id="a",
        goal="reply",
        context={},
        plan=Plan(id="p", goal="reply", steps=[plan_step]),
        step_index=0,
        history=[_history("list_emails", {"emails": [{"id": 7}]})],
    )

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step is not None
    assert result.step.params["email_id"] == 7  # resolved from history
    assert result.step.params["body"] == "hi"
    assert spy.ground_calls == []  # mechanistic -> no model call
    # The *stored* plan keeps the reference (a reusable skeleton); only the per-cycle step grounds.
    assert activity.plan is not None
    assert activity.plan.steps[0].params["email_id"] == ref


async def test_reason_escalates_unresolvable_reference_to_model(tmp_path: Path) -> None:
    tool = FakeTool("email", invoke_results={"reply_to_email": {"sent": True}})
    spy = ScriptedProcedural(ground_result={"email_id": 99, "body": "hi"})
    cycle, working, registry = _cycle(tmp_path, spy, tool)
    await registry.join(_ORIGIN)
    # A soft ref always escalates; history present but no mechanical resolution.
    plan_step = invoke_step(
        "email", "reply_to_email", email_id={"$decide": "Alice's email"}, body="hi"
    )
    activity = Activity(
        id="a",
        goal="reply",
        context={},
        plan=Plan(id="p", goal="reply", steps=[plan_step]),
        step_index=0,
        history=[_history("search_emails", {"emails": [{"id": 99}]})],
    )
    working.activities["a"] = activity  # the off-cycle _ground_ action looks it up here

    # A soft ref can't resolve mechanically: Reason fires _ground_ off-cycle -> RUNNING, no step.
    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert result.step is None
    # state read into a fresh local before each assert so mypy doesn't carry a narrowing across the
    # observe() that mutates it (same idiom as test_interrupt.py).
    state = activity.state
    assert state is ActivityState.RUNNING
    assert activity.pending_inference is not None
    assert activity.pending_inference.kind == "ground"
    assert activity.step_index == 0  # not advanced while the escalation is in flight

    await asyncio.sleep(0)  # let the background _ground_ task run and push its result
    assert len(spy.ground_calls) == 1
    assert spy.ground_calls[0][0] == "reply_to_email"

    # The grounded params land in a later Observe; the next Reason pass consumes them into the step.
    await DefaultObserveStrategy().observe(cycle)
    assert activity.grounded_params == {"email_id": 99, "body": "hi"}
    state = activity.state
    assert state is ActivityState.READY

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert result.step is not None
    assert result.step.params["email_id"] == 99  # from the model escalation
    assert activity.grounded_params is None  # consumed
    assert activity.step_index == 1
    assert len(spy.ground_calls) == 1  # not re-escalated


async def test_reason_grounds_send_content_mechanically_without_model(tmp_path: Path) -> None:
    tool = FakeTool("clock", invoke_results={"get_time": "12:00"})
    spy = ScriptedProcedural()  # ground() would raise if reached
    cycle, working, registry = _cycle(tmp_path, spy, tool)
    await registry.join(_ORIGIN)
    ref = {"$from": "get_time", "path": ""}
    plan_step = Step(next_action="send", params={"to": "user", "content": {"time": ref}})
    activity = Activity(
        id="a",
        goal="what time is it?",
        context={},
        plan=Plan(id="p", goal="what time is it?", steps=[plan_step]),
        step_index=0,
        history=[_history("get_time", "12:00")],
    )

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step is not None
    assert result.step.params["content"] == {"time": "12:00"}  # resolved from history
    assert result.step.params["to"] == "user"
    assert spy.ground_calls == []  # mechanistic -> no model call
    # The *stored* plan keeps the reference (a reusable skeleton); only the per-cycle step grounds.
    assert activity.plan is not None
    assert activity.plan.steps[0].params["content"] == {"time": ref}


async def test_reason_escalates_unresolvable_send_content_to_model(tmp_path: Path) -> None:
    tool = FakeTool("clock", invoke_results={"get_time": "12:00"})
    spy = ScriptedProcedural(ground_result={"time": "12:00"})
    cycle, working, registry = _cycle(tmp_path, spy, tool)
    await registry.join(_ORIGIN)
    plan_step = Step(
        next_action="send",
        params={"to": "user", "content": {"time": {"$decide": "the observed time"}}},
    )
    activity = Activity(
        id="a",
        goal="what time is it?",
        context={},
        plan=Plan(id="p", goal="what time is it?", steps=[plan_step]),
        step_index=0,
        history=[_history("get_time", "12:00")],
    )
    working.activities["a"] = activity  # the off-cycle _ground_ action looks it up here

    # A soft ref in send content escalates the same off-cycle way: RUNNING, no step this cycle.
    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert result.step is None
    assert activity.pending_inference is not None
    assert activity.pending_inference.kind == "ground"

    await asyncio.sleep(0)  # let the background _ground_ task run and push its result
    assert len(spy.ground_calls) == 1
    assert spy.ground_calls[0][0] == "send"

    # The grounded content lands in a later Observe; the next Reason pass folds it into the step.
    await DefaultObserveStrategy().observe(cycle)
    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert result.step is not None
    assert result.step.params["content"] == {"time": "12:00"}  # from the model escalation
    assert activity.step_index == 1


async def test_reason_send_without_dict_content_is_untouched(tmp_path: Path) -> None:
    tool = FakeTool("clock", invoke_results={"get_time": "12:00"})
    spy = ScriptedProcedural()  # would raise if ground() were called
    cycle, working, registry = _cycle(tmp_path, spy, tool)
    await registry.join(_ORIGIN)
    plan_step = Step(next_action="send", params={"to": "user", "content": "plain text"})
    activity = Activity(
        id="a",
        goal="hi",
        context={},
        plan=Plan(id="p", goal="hi", steps=[plan_step]),
        step_index=0,
    )

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step is plan_step  # non-dict content -> nothing to ground, untouched
    assert spy.ground_calls == []


async def test_reason_ground_escalation_receives_current_properties_and_signals(
    tmp_path: Path,
) -> None:
    # The escalation shouldn't decide blind either — currently observed world state reaches
    # ground() alongside the operation schema/partial params/history.
    tool = FakeTool("email", invoke_results={"reply_to_email": {"sent": True}})
    spy = ScriptedProcedural(ground_result={"email_id": 99, "body": "hi"})
    cycle, working, registry = _cycle(tmp_path, spy, tool)
    await registry.join(_ORIGIN)
    prop_percept = Percept("email", ObservableProperty("unread_count", 3), 0.0)
    signal_percept = Percept("email", Signal("new_email", {"id": 99}), 0.0)
    working.properties[("email", "unread_count")] = prop_percept
    working.signals.append(signal_percept)
    plan_step = invoke_step(
        "email", "reply_to_email", email_id={"$decide": "Alice's email"}, body="hi"
    )
    activity = Activity(
        id="a",
        goal="reply",
        context={},
        plan=Plan(id="p", goal="reply", steps=[plan_step]),
        step_index=0,
        history=[_history("search_emails", {"emails": [{"id": 99}]})],
    )
    working.activities["a"] = activity  # the off-cycle _ground_ action looks it up here

    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    await asyncio.sleep(0)  # let the background _ground_ task run so it records what it was asked

    assert len(spy.ground_percepts) == 1
    assert spy.ground_percepts[0] == PerceptSnapshot([prop_percept], [signal_percept])


async def test_reason_reference_free_step_is_cheap_no_ground(tmp_path: Path) -> None:
    tool = FakeTool("email", invoke_results={"list_emails": {"emails": []}})
    spy = ScriptedProcedural()  # would raise if ground() were called
    cycle, working, registry = _cycle(tmp_path, spy, tool)
    await registry.join(_ORIGIN)
    plan_step = invoke_step("email", "list_emails", folder="inbox")
    activity = Activity(
        id="a",
        goal="list",
        context={},
        plan=Plan(id="p", goal="list", steps=[plan_step]),
        step_index=0,
    )

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step is plan_step  # no references -> the exact same Step object, untouched
    assert isinstance(result.step, Step)
    assert spy.ground_calls == []


# --------------------------------------------------------------------------------------------------
# DefaultActStrategy.bind — guarded skip when a *required* param resolves to null
#
# Grounding (above) runs first in Reason, so by bind time a param is either a concrete value or a
# genuine null the model declined/could-not fill. Dispatching an invoke with a null *required* param
# is a mis-action (the historic blind-delete crash, now degraded to a probabilistic mis-action held
# off only by a prompt fragment). bind makes it a mechanical guarantee: consult the operation's
# schema (adapter-synthesized OperationSpecification.parameters), and skip the dispatch when a
# required param is null. It stays *mechanistic* (schema-driven, no judgment) — ADR-0017. Without a
# schema (no manual / spec / declared `required`) required-ness is unknown, so it cannot guard and
# falls back to dispatching — the same structured-spec dependency A6 has.
# --------------------------------------------------------------------------------------------------


def _manual_with_required(tool_id: str, op: str, required: list[str]) -> Manual:
    return Manual(
        id=tool_id,
        metadata={},
        description="",
        observable_properties=[],
        signals=[],
        operations=[
            OperationSpecification(
                name=op,
                description="",
                parameters={"properties": {k: {} for k in required}, "required": list(required)},
            )
        ],
    )


async def test_bind_skips_invoke_when_required_param_is_null(tmp_path: Path) -> None:
    cycle, _, _ = _cycle(tmp_path, ScriptedProcedural(), FakeTool("email"))
    manual = _manual_with_required("email", "reply_to_email", ["email_id", "body"])
    step = invoke_step("email", "reply_to_email", email_id=None, body="hi")

    result = await DefaultActStrategy().bind(step, manual, cycle, TickResult())

    assert result.invocation is None  # skipped -> the cycle dispatches nothing this cycle


async def test_bind_skips_invoke_when_required_param_absent(tmp_path: Path) -> None:
    # A required key missing from the step params entirely resolves to null the same as an explicit
    # None — a schema-invalid invoke, so the guard skips it rather than dispatch it malformed.
    cycle, _, _ = _cycle(tmp_path, ScriptedProcedural(), FakeTool("email"))
    manual = _manual_with_required("email", "reply_to_email", ["email_id", "body"])
    step = invoke_step("email", "reply_to_email", body="hi")  # email_id absent

    result = await DefaultActStrategy().bind(step, manual, cycle, TickResult())

    assert result.invocation is None


async def test_bind_dispatches_when_all_required_params_present(tmp_path: Path) -> None:
    cycle, _, _ = _cycle(tmp_path, ScriptedProcedural(), FakeTool("email"))
    manual = _manual_with_required("email", "reply_to_email", ["email_id", "body"])
    step = invoke_step("email", "reply_to_email", email_id="e1", body="hi")

    result = await DefaultActStrategy().bind(step, manual, cycle, TickResult())

    assert result.invocation == OperationInvocation(
        tool_id="email", operation_name="reply_to_email", params={"email_id": "e1", "body": "hi"}
    )


async def test_bind_dispatches_when_required_param_is_falsy_not_null(tmp_path: Path) -> None:
    # 0 / "" / False are legitimate values, not null — the guard keys on null (None/absent) only.
    cycle, _, _ = _cycle(tmp_path, ScriptedProcedural(), FakeTool("clock"))
    manual = _manual_with_required("clock", "set_volume", ["level"])
    step = invoke_step("clock", "set_volume", level=0)

    result = await DefaultActStrategy().bind(step, manual, cycle, TickResult())

    assert result.invocation is not None
    assert result.invocation.params == {"level": 0}


async def test_bind_dispatches_when_null_param_is_optional(tmp_path: Path) -> None:
    # A null value on a param the schema does *not* mark required is legitimate (many operations
    # take optional params) — it passes straight through, only required-null is guarded.
    cycle, _, _ = _cycle(tmp_path, ScriptedProcedural(), FakeTool("email"))
    manual = _manual_with_required("email", "search_emails", ["query"])  # cc not required
    step = invoke_step("email", "search_emails", query="alice", cc=None)

    result = await DefaultActStrategy().bind(step, manual, cycle, TickResult())

    assert result.invocation is not None
    assert result.invocation.params == {"query": "alice", "cc": None}


async def test_bind_dispatches_null_when_schema_unavailable(tmp_path: Path) -> None:
    # No schema (manual=None, or a manual that doesn't describe this op, or one with no declared
    # `required`) => required-ness is unknowable, so the guard cannot fire and bind falls back to
    # dispatching. This is the structured-spec dependency the ARE example satisfies (specs are
    # adapter-synthesized); a hand-authored-only manual would not, and stays on the model judgment.
    cycle, _, _ = _cycle(tmp_path, ScriptedProcedural(), FakeTool("email"))
    step = invoke_step("email", "reply_to_email", email_id=None, body="hi")

    no_manual = await DefaultActStrategy().bind(step, None, cycle, TickResult())
    assert no_manual.invocation is not None  # can't know required-ness -> dispatch (unchanged)

    # A manual that describes the op but declares no `required` key is equally non-committal.
    bare = Manual(
        id="email",
        metadata={},
        description="",
        observable_properties=[],
        signals=[],
        operations=[OperationSpecification(name="reply_to_email", description="", parameters={})],
    )
    bare_result = await DefaultActStrategy().bind(step, bare, cycle, TickResult())
    assert bare_result.invocation is not None


async def test_cycle_skips_invoke_dispatch_when_required_param_null(tmp_path: Path) -> None:
    # End-to-end through the cycle: a plan step whose required param grounds to null must never
    # reach the tool. The step is *not* re-run — step_index advances (skip-and-continue), so the
    # activity progresses to its next step rather than deadlocking on the guarded one.
    manual = _manual_with_required("email", "reply_to_email", ["email_id", "body"])
    tool = FakeTool("email", manual=manual, invoke_results={"reply_to_email": {"sent": True}})
    cycle, working, registry = _cycle(tmp_path, ScriptedProcedural(), tool)
    await registry.join(_ORIGIN)
    step = invoke_step("email", "reply_to_email", email_id=None, body="hi")
    activity = Activity(
        id="a",
        goal="reply",
        context={},
        plan=Plan(id="p", goal="reply", steps=[step]),
        step_index=0,
    )
    working.activities["a"] = activity

    await cycle.tick()
    await asyncio.sleep(0)  # let any dispatched background invoke run before asserting

    assert tool.invocations == []  # guarded — never dispatched with a null email_id
    assert activity.pending_operation is None  # never went RUNNING on an op
    state = activity.state
    assert state is ActivityState.READY  # not stuck RUNNING; free to advance
    assert activity.step_index == 1  # advanced -> skip-and-continue, not stuck on the step


async def test_cycle_dispatches_invoke_when_required_params_present(tmp_path: Path) -> None:
    # Control for the skip test: identical setup, required param present -> dispatches normally.
    manual = _manual_with_required("email", "reply_to_email", ["email_id", "body"])
    tool = FakeTool("email", manual=manual, invoke_results={"reply_to_email": {"sent": True}})
    cycle, working, registry = _cycle(tmp_path, ScriptedProcedural(), tool)
    await registry.join(_ORIGIN)
    step = invoke_step("email", "reply_to_email", email_id="e1", body="hi")
    activity = Activity(
        id="a",
        goal="reply",
        context={},
        plan=Plan(id="p", goal="reply", steps=[step]),
        step_index=0,
    )
    working.activities["a"] = activity

    await cycle.tick()
    await asyncio.sleep(0)  # let the dispatched background invoke run

    assert tool.invocations == [("reply_to_email", {"email_id": "e1", "body": "hi"})]
