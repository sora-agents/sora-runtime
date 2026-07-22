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

from collections.abc import AsyncIterator
from pathlib import Path

from fakes import FakeAdapter, FakeTool, FakeWorkspace
from sora.action import default_action_registry, invoke_step
from sora.activity import Activity
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, Tool, WorkspaceOrigin
from sora.manual import Manual
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

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step is not None
    assert result.step.params["email_id"] == 99  # from the model escalation
    assert len(spy.ground_calls) == 1
    assert spy.ground_calls[0][0] == "reply_to_email"


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

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step is not None
    assert result.step.params["content"] == {"time": "12:00"}  # from the model escalation
    assert len(spy.ground_calls) == 1
    assert spy.ground_calls[0][0] == "send"


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

    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

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
