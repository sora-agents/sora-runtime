"""Permanent TDD tests for the deterministic Observe and Reflect phases and the cycle paths that
exercise them.

Observe is the one deterministic, no-LLM phase that always runs first and never selects an activity
(the decision chain starts at Situate). ``DefaultObserveStrategy.observe(cycle)`` runs four channels
and returns an empty ``TickResult()``:

* focused-tool ``observe()`` properties -> ``PROPERTY`` percepts;
* ``signal_sink`` -> ``SIGNAL`` percepts;
* ``result_sink`` -> the automatic 1:1 running-resolution (an ack matching an activity's
  ``pending_operation.id`` drives ``RUNNING -> READY``, clears pending, sets ``last_operation`` — no
  ``Percept``, no strategy judgment; a non-matching ack is silently dropped);
* ``communication.receive()`` -> inbound ``messages``.

Reflect (``DefaultReflectStrategy``) runs per activity right after Observe: it judges each ``READY``
activity **completed** (plan fully consumed) or **failed** (last operation not ``ok``), transitions
it to ``TERMINATED`` *synchronously* (so Situate, which selects only ``READY`` activities, never
re-selects it this cycle), and *dispatches* the episodic/procedural stores asynchronously so they
never block the cycle. Only success stores the plan to procedural memory. The episode learned for
either outcome carries the enriched record — outcome, the attempted plan (the only surviving copy
on failure, since procedural memory does not store failed plans), step progress, and the last
operation result.

The harness reuses ``tests/fakes.py`` and a real ``FileMemoryBackend`` — one directory per memory
module (as ``agent.yaml`` wires them), since ``ProceduralMemory.retrieve`` and
``EpisodicMemory.consult`` both filter by ``goal`` and a shared directory would cross-match plan and
episode records. A local scripted transport's ``receive()`` drains a preset message list (modeled on
``test_action.py``'s ``RecordingTransport``). Promotes the spike's
``observe_resolves_running_activity`` from ``tests/test_cycle_wiring.py`` (triage row in
``docs/phase-3-test-triage.md``). The Situate/Reason/Act defaults are inert plumbing here — asserted
only for their non-dispatch, not their own behavior (their own tests come with those strategies).
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from fakes import FakeAdapter, FakeTool, FakeWorkspace, ScriptedTransport
from sora.action import SendAction, default_action_registry, invoke_step
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, WorkspaceOrigin
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from sora.perception import Message
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    Strategies,
    TickResult,
)
from sora.types import (
    OPERATION_NAME,
    TOOL_ID,
    WAIT,
    ActionAck,
    ObservableProperty,
    OperationAck,
    OperationInvocation,
    PendingOperation,
    Plan,
    Signal,
    Step,
)

# --------------------------------------------------------------------------------------------------
# Harness — the shared fakes plus a scripted transport and a real FileMemoryBackend-backed cycle.
# --------------------------------------------------------------------------------------------------


class _InertReason:
    """A ReasonStrategy stand-in; only reached once an activity is selected, which the Observe-only
    tests deliberately avoid."""

    async def reason(
        self, activity: Any, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        return result


class _RecordingReason:
    """A ReasonStrategy that flags whether the cycle ever reached Reason — i.e. whether Situate
    selected an activity. Used to prove a Reflect-terminated activity is not re-selected."""

    def __init__(self) -> None:
        self.called = False

    async def reason(
        self, activity: Any, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        self.called = True
        return result


def _cycle(
    tmp_path: Path,
    transport: ScriptedTransport | None = None,
    *,
    reflect: Any = None,
    reason: Any = None,
    act: Any = None,
    registry: EnvironmentRegistry | None = None,
) -> tuple[DecisionCycle, WorkingMemory]:
    # One directory per memory module (mirroring agent.yaml): procedural/episodic both query by
    # `goal`, so a shared directory would let a plan and an episode with the same goal cross-match.
    registry = registry if registry is not None else EnvironmentRegistry()
    working = WorkingMemory(registry=registry)
    strategies = Strategies(
        observe=DefaultObserveStrategy(),
        reflect=reflect or DefaultReflectStrategy(),
        situate=DefaultSituateStrategy(),
        reason=reason or _InertReason(),
        act=act or DefaultActStrategy(),
    )
    cycle = DecisionCycle(
        strategies=strategies,
        communication=transport or ScriptedTransport(),
        actions=default_action_registry(),
        registry=registry,
        working=working,
        semantic=SemanticMemory(FileMemoryBackend(tmp_path / "semantic")),
        procedural=ProceduralMemory(FileMemoryBackend(tmp_path / "procedural")),
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "episodic")),
    )
    return cycle, working


def _running_activity(activity_id: str, op_id: str) -> Activity:
    """A RUNNING activity with a bound ``pending_operation`` keyed by ``op_id`` — the state that
    Observe's 1:1 running-resolution consumes (as InvokeAction leaves it, minus the round-trip)."""
    return Activity(
        id=activity_id,
        goal="list emails",
        context={},
        state=ActivityState.RUNNING,
        pending_operation=PendingOperation(
            id=op_id,
            invocation=OperationInvocation(
                tool_id="EmailClientApp", operation_name="list_emails", params={}
            ),
            invoked_at=0.0,
        ),
    )


# --------------------------------------------------------------------------------------------------
# Observe — the four deterministic channels
# --------------------------------------------------------------------------------------------------


async def test_observe_returns_empty_tick_result(tmp_path: Path) -> None:
    # Observe never selects an activity — the decision chain starts at Situate.
    cycle, _ = _cycle(tmp_path)
    result = await DefaultObserveStrategy().observe(cycle)
    assert result == TickResult()


async def test_observe_emits_property_percepts_from_focused_tools(tmp_path: Path) -> None:
    prop = ObservableProperty(name="unread", value=3)
    tool = FakeTool("EmailClientApp", properties=[prop])
    cycle, working = _cycle(tmp_path)
    working.focused_tools["EmailClientApp"] = tool

    await DefaultObserveStrategy().observe(cycle)

    assert len(working.properties) == 1
    percept = working.properties[(tool.id, "unread")]
    assert percept.source == tool.id
    assert percept.payload == prop


async def test_observe_emits_signal_percepts_from_signal_sink(tmp_path: Path) -> None:
    signal = Signal(name="new_email", payload={"n": 1})
    cycle, working = _cycle(tmp_path)
    cycle.signal_sink.push("EmailClientApp", signal)

    await DefaultObserveStrategy().observe(cycle)

    assert len(working.signals) == 1
    percept = working.signals[0]
    assert percept.source == "EmailClientApp"
    assert percept.payload == signal


async def test_observe_appends_inbound_messages(tmp_path: Path) -> None:
    message = Message(sender="agent-b", content={"greeting": "hi"}, received_at=0.0)
    transport = ScriptedTransport(inbound=[message])
    cycle, working = _cycle(tmp_path, transport=transport)

    await DefaultObserveStrategy().observe(cycle)

    assert working.messages == [message]


# --------------------------------------------------------------------------------------------------
# Observe — properties are a replaced (source, name) snapshot; signals stay an append log
# --------------------------------------------------------------------------------------------------


async def test_observe_replaces_property_snapshot_not_append(tmp_path: Path) -> None:
    # A property is persistent, re-observed state: re-observing the same (source, name) replaces the
    # prior percept (last value wins) rather than accumulating a growing history of stale values.
    tool = FakeTool("EmailClientApp", properties=[ObservableProperty(name="unread", value=3)])
    cycle, working = _cycle(tmp_path)
    working.focused_tools["EmailClientApp"] = tool
    strategy = DefaultObserveStrategy()

    await strategy.observe(cycle)
    tool._properties = [ObservableProperty(name="unread", value=5)]  # the live value moved on
    await strategy.observe(cycle)

    props = list(working.properties.values())
    assert len(props) == 1
    assert props[0].source == "EmailClientApp"
    assert props[0].payload == ObservableProperty(name="unread", value=5)


async def test_observe_snapshots_each_property_by_name(tmp_path: Path) -> None:
    # The snapshot key is (source, name): distinct property names coexist, one entry each, and
    # re-observing does not duplicate them.
    tool = FakeTool(
        "EmailClientApp",
        properties=[
            ObservableProperty(name="unread", value=3),
            ObservableProperty(name="drafts", value=1),
        ],
    )
    cycle, working = _cycle(tmp_path)
    working.focused_tools["EmailClientApp"] = tool
    strategy = DefaultObserveStrategy()

    await strategy.observe(cycle)
    await strategy.observe(cycle)

    props = list(working.properties.values())
    assert len(props) == 2
    assert {p.payload.name for p in props} == {"unread", "drafts"}
    assert set(working.properties) == {("EmailClientApp", "unread"), ("EmailClientApp", "drafts")}


async def test_observe_keeps_signal_append_semantics(tmp_path: Path) -> None:
    # Signals are the opposite of properties: transient and fire-and-forget, so they accumulate
    # across cycles (never replaced), even from the same source with the same name.
    cycle, working = _cycle(tmp_path)
    strategy = DefaultObserveStrategy()

    cycle.signal_sink.push("EmailClientApp", Signal(name="new_email", payload={"n": 1}))
    await strategy.observe(cycle)
    cycle.signal_sink.push("EmailClientApp", Signal(name="new_email", payload={"n": 2}))
    await strategy.observe(cycle)

    signals = working.signals
    assert len(signals) == 2
    assert [p.payload.payload["n"] for p in signals] == [1, 2]


# --------------------------------------------------------------------------------------------------
# Observe — the automatic 1:1 running-resolution off result_sink
# --------------------------------------------------------------------------------------------------


async def test_observe_resolves_running_activity(tmp_path: Path) -> None:
    cycle, working = _cycle(tmp_path)
    activity = _running_activity("a1", op_id="op-1")
    working.activities["a1"] = activity
    ack = OperationAck(ok=True, result={"emails": []})
    cycle.result_sink.push("op-1", ack)

    await DefaultObserveStrategy().observe(cycle)

    assert activity.state is ActivityState.READY
    assert activity.pending_operation is None
    assert activity.last_operation == ack
    # A resolved operation result is never surfaced as a Percept.
    assert working.properties == {}
    assert working.signals == []


async def test_observe_resolves_only_the_matching_activity(tmp_path: Path) -> None:
    cycle, working = _cycle(tmp_path)
    matched = _running_activity("a1", op_id="op-1")
    other = _running_activity("a2", op_id="op-2")
    working.activities["a1"] = matched
    working.activities["a2"] = other
    cycle.result_sink.push("op-1", OperationAck(ok=True, result={"emails": []}))

    await DefaultObserveStrategy().observe(cycle)

    assert matched.state is ActivityState.READY
    assert matched.pending_operation is None
    # The unmatched activity keeps waiting on its own in-flight operation.
    assert other.state is ActivityState.RUNNING
    assert other.pending_operation is not None
    assert other.last_operation is None


async def test_observe_ignores_unmatched_result_ack(tmp_path: Path) -> None:
    # An ack whose id matches no pending operation is silently dropped (the 1:1 guard) — no raise,
    # no state change, and never re-surfaced as a Percept.
    cycle, working = _cycle(tmp_path)
    activity = _running_activity("a1", op_id="op-1")
    working.activities["a1"] = activity
    cycle.result_sink.push("stranger", OperationAck(ok=True, result="unrelated"))

    await DefaultObserveStrategy().observe(cycle)

    assert activity.state is ActivityState.RUNNING
    assert activity.pending_operation is not None
    assert activity.last_operation is None
    assert working.properties == {}
    assert working.signals == []
    assert cycle.result_sink._queue.empty()


# --------------------------------------------------------------------------------------------------
# The Observe-only tick() path
# --------------------------------------------------------------------------------------------------


async def test_tick_observes_then_returns_without_external_action(tmp_path: Path) -> None:
    # A real tick() runs Observe (percepts + messages populated); the inbound message becomes an
    # activity in Situate, but the inert Reason yields no step, so no external action is dispatched.
    prop = ObservableProperty(name="unread", value=3)
    tool = FakeTool("EmailClientApp", properties=[prop])
    signal = Signal(name="new_email", payload={"n": 1})
    message = Message(sender="agent-b", content={"greeting": "hi"}, received_at=0.0)
    transport = ScriptedTransport(inbound=[message])
    # A tool can only be focused once its workspace is joined (per A&A) — so its property percept
    # survives Situate's _filter_, which keeps only percepts from the joined workspaces' tools.
    origin = WorkspaceOrigin(adapter="fake", address="fake://ws")
    registry = EnvironmentRegistry(
        adapters={origin: FakeAdapter("fake", FakeWorkspace("ws", origin, [tool]))}
    )
    cycle, working = _cycle(tmp_path, transport=transport, registry=registry)
    await registry.join(origin)
    working.focused_tools["EmailClientApp"] = tool
    cycle.signal_sink.push("EmailClientApp", signal)

    await cycle.tick()

    assert [p.payload for p in working.properties.values()] == [prop]
    assert [p.payload for p in working.signals] == [signal]
    assert working.messages == [message]
    # The inert Reason produced no step, so no external action was dispatched.
    assert tool.invoked_with is None


# --------------------------------------------------------------------------------------------------
# Reflect — deterministic completion/failure judgment + async store-on-success
# --------------------------------------------------------------------------------------------------


def _planned_activity(
    activity_id: str,
    *,
    steps: int = 1,
    step_index: int | None = None,
    state: ActivityState = ActivityState.READY,
    last_ok: bool | None = True,
) -> Activity:
    """An activity carrying a plan whose ``goal`` matches its own (so ``episodic.consult``, keyed on
    goal, can find the episode Reflect records — plans are no longer auto-cached). ``step_index``
    defaults to *fully consumed* (plan complete); ``last_ok`` sets ``last_operation`` (``None`` = no
    op)."""
    goal = f"goal-{activity_id}"
    plan = Plan(
        id=f"plan-{activity_id}",
        goal=goal,
        steps=[Step(next_action="wait", params={}) for _ in range(steps)],
    )
    return Activity(
        id=activity_id,
        goal=goal,
        context={},
        state=state,
        plan=plan,
        step_index=steps if step_index is None else step_index,
        last_operation=None if last_ok is None else OperationAck(ok=last_ok),
    )


async def _drain(strategy: DefaultReflectStrategy) -> None:
    """Await the fire-and-forget episodic/procedural stores Reflect dispatched (whitebox: the
    strategy tracks them in ``_tasks``, mirroring ``InvokeAction``)."""
    await asyncio.gather(*list(strategy._tasks))


async def test_reflect_terminates_completed_activity_and_records_episode(tmp_path: Path) -> None:
    cycle, working = _cycle(tmp_path)
    activity = _planned_activity("a1")  # plan consumed, last op ok -> completed
    working.activities["a1"] = activity
    strategy = DefaultReflectStrategy()

    result = await strategy.reflect(activity, working, cycle, TickResult())

    # The completion judgment is synchronous: state flips before the cycle continues to Situate.
    assert activity.state is ActivityState.TERMINATED
    # The episode write is *dispatched*, not awaited — a task exists but hasn't necessarily run yet.
    assert strategy._tasks
    # Reflect never fills in the decision fields.
    assert result == TickResult()

    await _drain(strategy)
    episodes = await cycle.episodic.consult(activity)
    assert len(episodes) == 1
    episode = episodes[0]
    # The episode is a self-contained experience: outcome, the attempted plan, step progress,
    # and the last operation result.
    assert episode["succeeded"] is True
    assert activity.plan is not None
    assert episode["plan"] == asdict(activity.plan)
    assert episode["step_index"] == 1
    assert episode["step_count"] == 1
    assert episode["last_result"] == asdict(OperationAck(ok=True))
    # The completed plan is deliberately NOT auto-cached to procedural memory (unsound to replay
    # verbatim); the episode above is the only durable record.
    assert await cycle.procedural.retrieve(activity) is None


async def test_reflect_terminates_failed_activity_without_storing_plan(tmp_path: Path) -> None:
    cycle, working = _cycle(tmp_path)
    # Mid-plan, but the last operation failed -> failed judgment regardless of remaining steps.
    activity = _planned_activity("a1", steps=3, step_index=1, last_ok=False)
    working.activities["a1"] = activity
    strategy = DefaultReflectStrategy()

    await strategy.reflect(activity, working, cycle, TickResult())

    assert activity.state is ActivityState.TERMINATED
    await _drain(strategy)
    # The failure is recorded to episodic memory — and the failure episode is the *only* surviving
    # copy of the attempted plan, since procedural deliberately stores no failed plan.
    episodes = await cycle.episodic.consult(activity)
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode["succeeded"] is False
    assert activity.plan is not None
    assert episode["plan"] == asdict(activity.plan)
    assert episode["step_index"] == 1
    assert episode["step_count"] == 3
    assert episode["last_result"] == asdict(OperationAck(ok=False))
    # A failed plan is never stored for future reuse.
    assert await cycle.procedural.retrieve(activity) is None


async def test_reflect_ignores_incomplete_activity(tmp_path: Path) -> None:
    cycle, working = _cycle(tmp_path)
    # Mid-plan (step 1 of 3 done, succeeded) — not yet complete, not failed.
    activity = _planned_activity("a1", steps=3, step_index=1, last_ok=True)
    working.activities["a1"] = activity
    strategy = DefaultReflectStrategy()

    await strategy.reflect(activity, working, cycle, TickResult())

    assert activity.state is ActivityState.READY
    assert not strategy._tasks
    assert await cycle.episodic.consult(activity) == []
    assert await cycle.procedural.retrieve(activity) is None


async def test_reflect_ignores_planless_activity(tmp_path: Path) -> None:
    # No plan -> never auto-completed by the mechanical default, even with a successful last op.
    # This is the boundary that keeps plan-less flows (e.g. the spike's Reason) from terminating.
    cycle, working = _cycle(tmp_path)
    activity = Activity(
        id="a1",
        goal="g",
        context={},
        state=ActivityState.READY,
        last_operation=OperationAck(ok=True),
    )
    working.activities["a1"] = activity
    strategy = DefaultReflectStrategy()

    await strategy.reflect(activity, working, cycle, TickResult())

    assert activity.state is ActivityState.READY
    assert not strategy._tasks
    assert await cycle.episodic.consult(activity) == []


async def test_reflect_terminates_planless_failed_activity(tmp_path: Path) -> None:
    # Failure is independent of the plan: a plan-less activity whose last op failed still
    # terminates, and the episode logs cleanly with no plan (plan=None, step_count=None).
    cycle, working = _cycle(tmp_path)
    activity = Activity(
        id="a1",
        goal="g",
        context={},
        state=ActivityState.READY,
        last_operation=OperationAck(ok=False),
    )
    working.activities["a1"] = activity
    strategy = DefaultReflectStrategy()

    await strategy.reflect(activity, working, cycle, TickResult())

    assert activity.state is ActivityState.TERMINATED
    await _drain(strategy)
    episodes = await cycle.episodic.consult(activity)
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode["succeeded"] is False
    assert episode["plan"] is None
    assert episode["step_index"] == 0
    assert episode["step_count"] is None
    assert episode["last_result"] == asdict(OperationAck(ok=False))
    assert await cycle.procedural.retrieve(activity) is None


async def test_reflect_skips_running_activity(tmp_path: Path) -> None:
    # A RUNNING activity's operation is still in flight — Reflect can't judge it yet.
    cycle, working = _cycle(tmp_path)
    activity = _planned_activity("a1", state=ActivityState.RUNNING)
    working.activities["a1"] = activity
    strategy = DefaultReflectStrategy()

    await strategy.reflect(activity, working, cycle, TickResult())

    assert activity.state is ActivityState.RUNNING
    assert not strategy._tasks


async def test_reflect_is_idempotent_for_terminated_activity(tmp_path: Path) -> None:
    # Reflect runs every cycle for every activity; an already-TERMINATED one must not re-record.
    cycle, working = _cycle(tmp_path)
    activity = _planned_activity("a1", state=ActivityState.TERMINATED)
    working.activities["a1"] = activity
    strategy = DefaultReflectStrategy()

    await strategy.reflect(activity, working, cycle, TickResult())

    assert not strategy._tasks
    assert await cycle.episodic.consult(activity) == []
    assert await cycle.procedural.retrieve(activity) is None


async def test_tick_reflect_terminates_completed_activity_and_is_not_reselected(
    tmp_path: Path,
) -> None:
    # Integrated: a completed activity is terminated during Reflect and, because Situate selects
    # only READY activities, is never re-selected the same cycle (Reason is never reached).
    reflect = DefaultReflectStrategy()
    reason = _RecordingReason()
    cycle, working = _cycle(tmp_path, reflect=reflect, reason=reason)
    activity = _planned_activity("a1")
    working.activities["a1"] = activity

    await cycle.tick()

    assert activity.state is ActivityState.TERMINATED
    assert reason.called is False  # Situate did not re-select the just-terminated activity
    await _drain(reflect)
    assert len(await cycle.episodic.consult(activity)) == 1
    assert await cycle.procedural.retrieve(activity) is None  # plan auto-caching is disabled


# --------------------------------------------------------------------------------------------------
# Reason + Act — the decision chain's tail (§7b: the _act() bind-then-dispatch boundary)
# --------------------------------------------------------------------------------------------------
#
# The cycle no longer special-cases `next_action == "invoke"`. Act binds a step iff its *action*
# declares `requires_binding`, then dispatches exactly one external action (the bound invocation's
# routing keys + params when present, else the raw step params). WAIT stays the cycle's no-op
# sentinel — it is not a registered ExternalAction, so it is guarded before any registry lookup.
#
# Reason is driven here by a deterministic, no-LLM *test fixture* (a seeded plan the cycle
# advances), not a shipped default: planning needs inference, so the real default is model-backed
# and lands with ProceduralMemory.infer() (kept out of this deterministic suite). This fixture
# replaces the walking-skeleton spike's string-typed ListEmailsReasonStrategy; the end-to-end test
# below re-drives that spike's last group (its assertions survive; the harness is rebuilt),
# retiring tests/test_cycle_wiring.py.


class _PlanFollowingReason:
    """Deterministic, no-LLM Reason fixture: reads the selected activity's current plan step and
    advances step_index; emits WAIT once the plan is exhausted (or absent). No procedural retrieval
    — the test seeds the plan directly. Stands in for the app-specific/LLM Reason the runtime does
    not ship a mechanical default for."""

    async def reason(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        plan = activity.plan
        if plan is None or activity.step_index >= len(plan.steps):
            return replace(result, step=Step(next_action=WAIT, params={}))
        step = plan.steps[activity.step_index]
        activity.step_index += 1
        return replace(result, step=step)


class _SpyAct:
    """Wraps DefaultActStrategy, counting bind() calls — so a test can assert the cycle binds based
    on the action's own requires_binding, not the removed `next_action == "invoke"` hardcode."""

    def __init__(self) -> None:
        self.bind_calls = 0
        self._inner = DefaultActStrategy()

    async def bind(
        self, step: Step, manual: Any, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        self.bind_calls += 1
        return await self._inner.bind(step, manual, cycle, result)


class _BindingProbeAction:
    """A custom external action (named "probe", *not* "invoke") that declares requires_binding=True,
    to prove the cycle's decision to bind is the action's own — the old hardcode only ever bound
    "invoke", so it would never bind this one."""

    name = "probe"
    requires_binding = True

    def __init__(self) -> None:
        self.received: OperationInvocation | None = None

    async def execute(
        self,
        registry: EnvironmentRegistry,
        cycle: DecisionCycle,
        *,
        activity_id: str,
        **kwargs: Any,
    ) -> ActionAck:
        self.received = OperationInvocation(
            tool_id=kwargs["tool_id"],
            operation_name=kwargs["operation_name"],
            params={k: v for k, v in kwargs.items() if k not in (TOOL_ID, OPERATION_NAME)},
        )
        return ActionAck(ok=True)


def _registry_with(tool: FakeTool) -> tuple[EnvironmentRegistry, WorkspaceOrigin]:
    origin = WorkspaceOrigin(adapter="fake", address="fake://ws")
    registry = EnvironmentRegistry(
        adapters={origin: FakeAdapter("fake", FakeWorkspace("ws", origin, [tool]))}
    )
    return registry, origin


async def test_default_act_binds_invoke_step_splitting_routing_keys(tmp_path: Path) -> None:
    # DefaultActStrategy.bind lifts the routing keys out of the step params into the invocation's
    # tool_id/operation_name, leaving only the operation's own params behind.
    cycle, _ = _cycle(tmp_path)
    step = invoke_step("clock", "get_time", tz="UTC")

    result = await DefaultActStrategy().bind(step, None, cycle, TickResult())

    assert result.invocation == OperationInvocation(
        tool_id="clock", operation_name="get_time", params={"tz": "UTC"}
    )


async def test_tick_binds_when_action_requires_binding(tmp_path: Path) -> None:
    # The action declares it needs binding, so Act binds it — even though it is not "invoke". This
    # fails on the pre-§7b cycle, whose hardcoded `next_action == "invoke"` branch never bound it.
    tool = FakeTool("clock", invoke_results={"get_time": "10:00"})
    registry, origin = _registry_with(tool)
    spy = _SpyAct()
    probe = _BindingProbeAction()
    cycle, working = _cycle(tmp_path, reason=_PlanFollowingReason(), act=spy, registry=registry)
    cycle.actions.register_external(probe)
    await registry.join(origin)  # so bind can resolve the tool's manual through the registry
    plan = Plan(
        id="p1",
        goal="g",
        steps=[
            Step(
                next_action="probe",
                params={TOOL_ID: "clock", OPERATION_NAME: "get_time", "tz": "UTC"},
            )
        ],
    )
    working.activities["a1"] = Activity(id="a1", goal="g", context={}, plan=plan)

    await cycle.tick()

    assert spy.bind_calls == 1  # bound because the action declared requires_binding, not by name
    assert probe.received == OperationInvocation(
        tool_id="clock", operation_name="get_time", params={"tz": "UTC"}
    )


async def test_tick_skips_bind_for_non_binding_action(tmp_path: Path) -> None:
    # A non-binding external action (send) is dispatched straight from the step params — Act.bind is
    # never called, and the raw params reach the action.
    transport = ScriptedTransport()
    spy = _SpyAct()
    cycle, working = _cycle(tmp_path, transport=transport, reason=_PlanFollowingReason(), act=spy)
    plan = Plan(
        id="p1",
        goal="notify",
        steps=[
            Step(next_action=SendAction.name, params={"to": "agent-b", "content": {"text": "hi"}})
        ],
    )
    working.activities["a1"] = Activity(id="a1", goal="notify", context={}, plan=plan)

    await cycle.tick()

    assert spy.bind_calls == 0  # send declares no binding
    assert transport.sent == [("agent-b", {"text": "hi"})]


async def test_tick_end_to_end_invoke_then_resolve(tmp_path: Path) -> None:
    # Re-drives the walking-skeleton spike's last group: a one-step invoke plan runs end-to-end
    # through the real tick() — Situate selects, Reason yields the invoke step, Act binds and
    # dispatches, the off-cycle result lands on result_sink, and the next Observe resolves it.
    tool = FakeTool(
        "EmailClientApp", invoke_results={"list_emails": {"emails": [], "total_emails": 0}}
    )
    registry, origin = _registry_with(tool)
    reflect = DefaultReflectStrategy()
    cycle, working = _cycle(
        tmp_path, reason=_PlanFollowingReason(), reflect=reflect, registry=registry
    )
    await registry.join(origin)
    plan = Plan(
        id="p1",
        goal="list emails",
        steps=[invoke_step("EmailClientApp", "list_emails")],
    )
    working.activities["a1"] = Activity(id="a1", goal="list emails", context={}, plan=plan)

    # Drive the cycle until the single step resolves (Observe picks up the off-cycle result).
    for _ in range(5):
        await cycle.tick()
        await asyncio.sleep(0)  # let the off-cycle invoke task land before the next observe
        if working.activities["a1"].last_operation is not None:
            break

    activity = working.activities["a1"]
    assert activity.last_operation is not None
    assert activity.last_operation.ok is True
    assert activity.last_operation.result == {"emails": [], "total_emails": 0}
    assert tool.invoked_with == ("list_emails", {})
    await _drain(reflect)  # settle the async success store so no task dangles past the test


async def test_tick_wait_step_dispatches_no_action(tmp_path: Path) -> None:
    # WAIT is the cycle's no-op sentinel, not a registered ExternalAction: reaching it must dispatch
    # nothing and must not raise (it is guarded before the registry lookup). A plan-less activity
    # makes Reason emit WAIT (and Reflect never auto-terminates a plan-less activity, so it is
    # selected and reaches Reason).
    tool = FakeTool("clock", invoke_results={"get_time": "10:00"})
    registry, origin = _registry_with(tool)
    cycle, working = _cycle(tmp_path, reason=_PlanFollowingReason(), registry=registry)
    await registry.join(origin)
    working.activities["a1"] = Activity(id="a1", goal="g", context={})  # no plan -> WAIT

    await cycle.tick()

    assert tool.invoked_with is None
    assert working.activities["a1"].state is ActivityState.READY
