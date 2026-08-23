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

from fakes import FakeAdapter, FakeLLMClient, FakeTool, FakeWorkspace
from sora.action import default_action_registry, invoke_step
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, Tool, WorkspaceOrigin
from sora.manual import Manual, MarkdownManualParser, OperationSpecification
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    PerceptSnapshot,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
    default_plan_prompt,
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
    _null_required_params,
    _undeclared_params,
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
    UnresolvableGrounding,
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


# -- references nested in a list or dict ----------------------------------------------------------
#
# The resolver used to iterate top-level param values only, so a reference inside a list was neither
# resolved nor reported unresolved and the raw token dict was serialized to the tool. It surfaced
# only when the *sole* reference in a step was nested — any other unresolved param escalated the
# step anyway and the model grounder papered over it. The plan schema forces the nesting: a param
# typed `list[str]` whose element is known only at run time has no other spelling.


def test_hard_reference_nested_in_a_list_resolves_without_escalating() -> None:
    history = [_history("search_contacts", [{"email": "ake@example.com"}])]
    params = {"recipients": [{"$from": "search_contacts", "path": "0.email"}]}
    resolved, unresolved = resolve_references(params, history)
    assert resolved == {"recipients": ["ake@example.com"]}
    assert unresolved == []  # mechanically resolvable — no model call


def test_hard_reference_nested_in_a_dict_resolves() -> None:
    history = [_history("search_contacts", [{"email": "ake@example.com"}])]
    params = {"event": {"who": {"$from": "search_contacts", "path": "0.email"}, "when": "09:00"}}
    resolved, unresolved = resolve_references(params, history)
    assert resolved == {"event": {"who": "ake@example.com", "when": "09:00"}}
    assert unresolved == []


def test_soft_reference_nested_in_a_list_escalates_under_its_top_level_key() -> None:
    # The exact shape that reached ARE as a literal: attendees is list[str], the name is unknown at
    # plan time. `unresolved` names the *param*, not a path, because grounding escalates per-param.
    params = {"attendees": [{"$decide": "full name of the first contact"}], "title": "Standup"}
    resolved, unresolved = resolve_references(params, [])
    assert unresolved == ["attendees"]
    assert resolved["attendees"] == [{"$decide": "full name of the first contact"}]
    assert resolved["title"] == "Standup"  # untouched


def test_unresolvable_nested_reference_escalates_rather_than_reaching_the_tool() -> None:
    history = [_history("search_contacts", [{"email": "ake@example.com"}])]
    cases: list[dict[str, object]] = [
        {"x": [{"$from": "never_ran", "path": "0.email"}]},  # no such history entry
        {"x": [{"$from": "search_contacts", "path": "9.email"}]},  # index out of range
        {"x": {"deep": {"deeper": {"$bind": "no_such_binding"}}}},  # missing binding, 3 levels down
    ]
    for params in cases:
        _, unresolved = resolve_references(params, history)
        assert unresolved == ["x"], params


def test_a_partly_resolvable_list_keeps_what_resolved_and_still_escalates() -> None:
    # Grounding should be handed as much settled context as possible, so the resolvable element is
    # filled in even though the param as a whole escalates on its stubborn sibling.
    history = [_history("search_contacts", [{"email": "ake@example.com"}])]
    params = {"to": [{"$from": "search_contacts", "path": "0.email"}, {"$decide": "the manager"}]}
    resolved, unresolved = resolve_references(params, history)
    assert unresolved == ["to"]
    assert resolved["to"] == ["ake@example.com", {"$decide": "the manager"}]


def test_nested_non_reference_structures_pass_through_unchanged() -> None:
    params = {"filters": [{"op": "eq", "value": 3}], "meta": {"tags": ["a", "b"], "n": None}}
    resolved, unresolved = resolve_references(params, [])
    assert resolved == params
    assert unresolved == []


# --------------------------------------------------------------------------------------------------
# DefaultReasonStrategy grounding — mechanistic when it can be, model escalation when it must be
# --------------------------------------------------------------------------------------------------


class ScriptedProcedural(ProceduralMemory):
    """Spies ``ground`` (and would raise if it were called without being configured) so a test can
    prove the *mechanistic* path took no model call, and assert the escalation payload."""

    def __init__(
        self,
        *,
        ground_result: dict[str, object] | None = None,
        ground_unresolvable: str | None = None,
        ground_error: str | None = None,
    ) -> None:
        super().__init__(FileMemoryBackend("unused"))
        self._ground_result = ground_result
        self._ground_unresolvable = ground_unresolvable
        self._ground_error = ground_error
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
        if self._ground_unresolvable is not None:
            raise UnresolvableGrounding(self._ground_unresolvable)
        if self._ground_error is not None:
            raise RuntimeError(self._ground_error)
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


async def test_bind_skips_an_invoke_whose_params_still_carry_a_reference(tmp_path: Path) -> None:
    """Backstop for a resolver bug: a reference reaching parameter binding must not be serialized to
    the tool. Left unguarded, the tool rejects it with a type error on the *enclosing* list, which
    names the wrong culprit and sends debugging after a phantom schema problem. No manual is needed
    — the check is structural, so it fires where the required-param guard (which needs a schema)
    cannot."""
    cycle, _, _ = _cycle(tmp_path, ScriptedProcedural(), FakeTool("email"))
    step = invoke_step("email", "add_calendar_event", attendees=[{"$decide": "the manager"}])

    result = await DefaultActStrategy().bind(step, None, cycle, TickResult())

    assert result.invocation is None  # skipped -> nothing dispatched


async def test_bind_dispatches_normally_when_no_reference_survives(tmp_path: Path) -> None:
    # The guard must not fire on ordinary structured params that merely *contain* dicts and lists.
    cycle, _, _ = _cycle(tmp_path, ScriptedProcedural(), FakeTool("email"))
    step = invoke_step(
        "email", "add_calendar_event", attendees=["ake@example.com"], title="Standup"
    )

    result = await DefaultActStrategy().bind(step, None, cycle, TickResult())

    assert result.invocation == OperationInvocation(
        tool_id="email",
        operation_name="add_calendar_event",
        params={"attendees": ["ake@example.com"], "title": "Standup"},
    )


# -- the grounding failure channel -----------------------------------------------------------------
#
# The grounder used to be given no way to fail: GROUND_SYSTEM_PROMPT demanded {"params": {...}}
# unconditionally, so a reference naming data the run never produced (`search_contacts` -> []) left
# fabrication as the only response that satisfied the contract — and a real run duly invented the
# USER as the attendee of a meeting with a friend, then emailed the appointment to the user's own
# address. The model was obeying its instructions. These pin the second legal answer.


def _procedural(tmp_path: Path, response: str) -> ProceduralMemory:
    return ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=FakeLLMClient(response))


def _bare_activity() -> Activity:
    return Activity(id="a", goal="g", context={}, history=[_history("search_contacts", [])])


async def test_ground_reports_a_gap_instead_of_inventing_params(tmp_path: Path) -> None:
    proc = _procedural(tmp_path, '{"unresolvable": "recipients: search_contacts returned nothing"}')
    try:
        await proc.ground(_bare_activity(), "send_email", None, {"recipients": []})
    except UnresolvableGrounding as exc:
        assert "search_contacts returned nothing" in str(exc)
    else:  # pragma: no cover — the assertion below is the failure report
        raise AssertionError("expected UnresolvableGrounding")


async def test_a_response_carrying_both_is_read_as_the_gap(tmp_path: Path) -> None:
    """A hedging model that reports the gap AND offers params is taken at its first word. The params
    half of such an answer is exactly the fabrication this channel exists to stop, so the safe
    reading is the reported gap — never the invented value."""
    proc = _procedural(
        tmp_path, '{"unresolvable": "no such contact", "params": {"recipients": ["someone@x.com"]}}'
    )
    try:
        await proc.ground(_bare_activity(), "send_email", None, {})
    except UnresolvableGrounding:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected the unresolvable report to win")


async def test_an_empty_report_is_not_a_report(tmp_path: Path) -> None:
    # An empty string carries no information about what was missing, so it is not treated as the
    # escape hatch — the params alongside it are parsed as usual.
    proc = _procedural(tmp_path, '{"unresolvable": "", "params": {"recipients": ["a@b.com"]}}')
    got = await proc.ground(_bare_activity(), "send_email", None, {})
    assert got == {"recipients": ["a@b.com"]}


async def test_a_normal_params_response_is_unaffected(tmp_path: Path) -> None:
    proc = _procedural(tmp_path, '{"params": {"recipients": ["friend@example.com"]}}')
    got = await proc.ground(_bare_activity(), "send_email", None, {})
    assert got == {"recipients": ["friend@example.com"]}


# --------------------------------------------------------------------------------------------------
# the third way out of a list parameter: keep the shape, drop the element
# --------------------------------------------------------------------------------------------------


async def test_a_shortened_list_is_the_gap_it_declined_to_report(tmp_path: Path) -> None:
    """A real run created a calendar event with ``attendees: []`` from a step that asked for one
    attendee, because the reference behind that element resolved to nothing and the model returned
    the shorter list rather than reporting it. That parses, reads as success, and gets invoked. The
    element count is a pre-image the runtime already holds, so it is checked rather than trusted."""
    proc = _procedural(tmp_path, '{"params": {"attendees": []}}')
    partial = {"attendees": [{"$decide": "the friend's full name"}]}
    try:
        await proc.ground(_bare_activity(), "add_calendar_event", None, partial)
    except UnresolvableGrounding as exc:
        assert "attendees" in str(exc)
        assert "supplied 1" in str(exc) and "returned 0" in str(exc)
    else:  # pragma: no cover — the assertion below is the failure report
        raise AssertionError("expected the dropped element to be reported as a gap")


async def test_dropping_the_parameter_altogether_counts_as_dropping_its_elements(
    tmp_path: Path,
) -> None:
    proc = _procedural(tmp_path, '{"params": {"title": "Film Production Day"}}')
    partial = {"attendees": [{"$decide": "the friend"}], "title": "Film Production Day"}
    try:
        await proc.ground(_bare_activity(), "add_calendar_event", None, partial)
    except UnresolvableGrounding as exc:
        assert "attendees" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an omitted list parameter is still a dropped element")


async def test_a_list_that_grows_is_exactly_what_a_reference_element_is_for(tmp_path: Path) -> None:
    """Only shrinkage is a claim that something asked for is absent. One reference element standing
    for "everyone the search found" resolving to three recipients is the feature working."""
    proc = _procedural(tmp_path, '{"params": {"recipients": ["a@x.com", "b@x.com", "c@x.com"]}}')
    partial = {"recipients": [{"$from": "search_contacts", "path": "*.email"}]}
    got = await proc.ground(_bare_activity(), "send_email", None, partial)
    assert got == {"recipients": ["a@x.com", "b@x.com", "c@x.com"]}


async def test_an_empty_list_going_in_is_not_a_floor_to_trip_over(tmp_path: Path) -> None:
    """A step that supplied no elements is not asking for any, so an empty answer is not a drop —
    the check must not manufacture a gap out of a parameter nobody filled."""
    proc = _procedural(tmp_path, '{"params": {"attendees": [], "title": "Solo day"}}')
    got = await proc.ground(
        _bare_activity(), "add_calendar_event", None, {"attendees": [], "title": "Solo day"}
    )
    assert got == {"attendees": [], "title": "Solo day"}


def _superseded_activity(defect: str | None) -> Activity:
    activity = Activity(
        id="a",
        goal="book a day with my friend the film producer",
        context={},
        plan=Plan(
            id="p",
            goal="book a day",
            steps=[invoke_step("insim:are/Calendar", "add_calendar_event", title="Film Day")],
        ),
        history=[_history("search_contacts", [])],
    )
    activity.reset_for_replan(defect=defect)
    return activity


def test_a_defective_plan_is_not_described_as_merely_stale() -> None:
    activity = _superseded_activity("attendees: search_contacts returned an empty list")
    _system, user = default_plan_prompt(activity, {})

    # The specific gap reaches the planner — without it the brief is an unactionable "that failed".
    assert "search_contacts returned an empty list" in user
    # And the reconsideration advice, which is what made the planner re-emit the doomed step, is
    # explicitly NOT given: this plan is wrong, not overtaken by events.
    assert "the world moved" not in user
    assert "stale, not wrong throughout" not in user
    assert "will fail in the same place" in user  # told to route around, not to re-derive
    # The steps that had nothing to do with the gap are still blessed — a brief that reads as "that
    # was all wrong" throws away the decomposition several calls already paid for.
    assert "should be reused" in user
    assert "add_calendar_event" in user  # the un-run tail is still rendered


def test_a_stale_plan_keeps_the_reconsideration_brief() -> None:
    """The other half: nothing changes for a plan dropped because the world moved."""
    _system, user = default_plan_prompt(_superseded_activity(None), {})

    assert "the world moved" in user
    assert "stale, not wrong throughout" in user
    assert "will fail in the same place" not in user


def _contacts_manual() -> Manual:
    return Manual(
        id="contacts",
        metadata={},
        description="",
        observable_properties=[],
        signals=[],
        operations=[
            OperationSpecification(
                name="get_contacts",
                description="Gets contacts from an offset. There is a view limit.",
                parameters={"properties": {"offset": {"type": "integer"}}},
            )
        ],
    )


async def _reason_once(tmp_path: Path, tool: FakeTool, step: Step) -> tuple[Activity, TickResult]:
    cycle, working, registry = _cycle(tmp_path, ScriptedProcedural(), tool)
    await registry.join(_ORIGIN)
    activity = Activity(
        id="a",
        goal="find the film producer",
        context={},
        plan=Plan(id="p", goal="find the film producer", steps=[step]),
        step_index=0,
        history=[],
    )
    working.activities["a"] = activity
    emitted = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    return activity, emitted


async def test_an_undeclared_param_replans_instead_of_invoking(tmp_path: Path) -> None:
    tool = FakeTool("contacts", manual=_contacts_manual(), invoke_results={"get_contacts": {}})
    step = invoke_step("contacts", "get_contacts", offset=0, limit=100)

    activity, emitted = await _reason_once(tmp_path, tool, step)

    assert emitted.step is None  # no step commits this cycle
    assert tool.invocations == []  # never reached the wire, so never a TypeError
    state = activity.state
    assert state is ActivityState.READY  # and not TERMINATED, which a failed op would have caused
    assert activity.plan is None  # dropped -> Reason re-infers next cycle
    superseded = activity.superseded
    assert superseded is not None
    # The reason is what makes the retry differ from the attempt: it must name the offending param
    # AND what the operation does accept, or the planner is left guessing at the same fork.
    assert superseded.defect is not None
    assert "'limit'" in superseded.defect
    assert "offset" in superseded.defect


async def test_a_fully_declared_call_is_untouched(tmp_path: Path) -> None:
    tool = FakeTool("contacts", manual=_contacts_manual(), invoke_results={"get_contacts": {}})
    step = invoke_step("contacts", "get_contacts", offset=0)

    activity, emitted = await _reason_once(tmp_path, tool, step)

    assert emitted.step == step  # the guard let it through and the step commits
    assert activity.step_index == 1
    assert activity.plan is not None  # nothing dropped
    assert activity.superseded is None


def test_an_operation_with_no_declared_properties_is_left_alone() -> None:
    """Required-ness is unknowable without a structured schema, so the guard cannot fire — the same
    dependency the null-required guard has. A prose-only manual must not become un-invokable."""
    manual = Manual(
        id="t",
        metadata={},
        description="",
        observable_properties=[],
        signals=[],
        operations=[OperationSpecification(name="op", description="", parameters={})],
    )
    assert _undeclared_params(manual, "op", {"anything": 1}) == []
    assert _undeclared_params(None, "op", {"anything": 1}) == []
    assert _undeclared_params(manual, "no_such_op", {"anything": 1}) == []


def test_a_schema_that_opts_into_extras_is_left_alone() -> None:
    """An operation that genuinely takes a free-form bag declares additionalProperties, and is then
    exempt — the closed-by-default reading is for schemas synthesized from real callables."""
    open_manual = Manual(
        id="t",
        metadata={},
        description="",
        observable_properties=[],
        signals=[],
        operations=[
            OperationSpecification(
                name="op",
                description="",
                parameters={"properties": {"a": {}}, "additionalProperties": True},
            )
        ],
    )
    assert _undeclared_params(open_manual, "op", {"a": 1, "b": 2}) == []


# A hand-authored manual with an interface block: the parser lifts the operation's *required*
# keys and nothing else, so `limit` — a real, optional param — appears nowhere in the schema.
_AUTHORED_MANUAL = """# Tool Metadata
id: calendar

# Functional Description
Reads and writes calendar events.

# Observable Properties
(none)

# Signals
(none)

# Operations
```yaml
- name: search_events
  required: [query]
```
- search_events(query, limit): finds events matching a query; limit defaults to 20.

# Usage Protocols & Safety
No special precautions.
"""


def test_an_authored_manuals_optional_params_are_not_undeclared() -> None:
    """MarkdownManualParser lifts only the *required* keys into `properties` (the adapter is meant
    to supply the real shapes), so reading `properties` as the closed universe of accepted names
    would reject every legitimately-optional param of a manual that never merges with an adapter
    schema — dropping a correct plan, and dropping the replacement for the same reason. Both
    built-in adapters do merge, so this is a trap for the authoring extension point rather than a
    live failure; `_null_required_params` reads `required` and is safe either way, and the
    asymmetry is what makes it a trap."""
    manual = MarkdownManualParser().parse(_AUTHORED_MANUAL)

    assert _undeclared_params(manual, "search_events", {"query": "x", "limit": 10}) == []
    assert _null_required_params(manual, "search_events", {"query": None}) == ["query"]


def test_every_undeclared_param_is_reported_not_just_the_first() -> None:
    # A planner that misread the schema usually misreads it more than once; reporting one name at a
    # time would cost a full replan per name.
    assert _undeclared_params(_contacts_manual(), "get_contacts", {"limit": 1, "page": 2}) == [
        "limit",
        "page",
    ]
