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
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fakes import FakeAdapter, FakeTool, FakeWorkspace
from sora.action import default_action_registry
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
from sora.perception import Message, PerceptKind
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    Strategies,
    TickResult,
)
from sora.types import (
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


class ScriptedTransport:
    """Satisfies MessageTransport: ``receive()`` drains a preset inbound list (so a second tick
    doesn't re-yield the same messages); ``send()`` logs its args."""

    def __init__(self, inbound: list[Message] | None = None) -> None:
        self._inbound = list(inbound or [])
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def send(self, to: str, content: dict[str, Any]) -> None:
        self.sent.append((to, content))

    def receive(self) -> AsyncIterator[Message]:
        async def _drain() -> AsyncIterator[Message]:
            while self._inbound:
                yield self._inbound.pop(0)

        return _drain()


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
        act=DefaultActStrategy(),
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

    assert len(working.perceptions) == 1
    percept = working.perceptions[0]
    assert percept.source == tool.id
    assert percept.kind is PerceptKind.PROPERTY
    assert percept.payload == prop


async def test_observe_emits_signal_percepts_from_signal_sink(tmp_path: Path) -> None:
    signal = Signal(name="new_email", payload={"n": 1})
    cycle, working = _cycle(tmp_path)
    cycle.signal_sink.push("EmailClientApp", signal)

    await DefaultObserveStrategy().observe(cycle)

    assert len(working.perceptions) == 1
    percept = working.perceptions[0]
    assert percept.source == "EmailClientApp"
    assert percept.kind is PerceptKind.SIGNAL
    assert percept.payload == signal


async def test_observe_appends_inbound_messages(tmp_path: Path) -> None:
    message = Message(sender="agent-b", content={"greeting": "hi"}, received_at=0.0)
    transport = ScriptedTransport(inbound=[message])
    cycle, working = _cycle(tmp_path, transport=transport)

    await DefaultObserveStrategy().observe(cycle)

    assert working.messages == [message]


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
    assert working.perceptions == []


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
    assert working.perceptions == []
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

    kinds = {p.kind for p in working.perceptions}
    assert kinds == {PerceptKind.PROPERTY, PerceptKind.SIGNAL}
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
    """An activity carrying a plan whose ``goal`` matches its own (so ``procedural.retrieve`` and
    ``episodic.consult``, both keyed on goal, can find what Reflect stores). ``step_index`` defaults
    to *fully consumed* (plan complete); ``last_ok`` sets ``last_operation`` (``None`` = no op)."""
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


async def test_reflect_terminates_completed_activity_and_stores(tmp_path: Path) -> None:
    cycle, working = _cycle(tmp_path)
    activity = _planned_activity("a1")  # plan consumed, last op ok -> completed
    working.activities["a1"] = activity
    strategy = DefaultReflectStrategy()

    result = await strategy.reflect(activity, working, cycle, TickResult())

    # The completion judgment is synchronous: state flips before the cycle continues to Situate.
    assert activity.state is ActivityState.TERMINATED
    # The stores are *dispatched*, not awaited — a task exists but hasn't necessarily run yet.
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
    stored_plan = await cycle.procedural.retrieve(activity)
    assert stored_plan is not None
    assert stored_plan.id == "plan-a1"


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
    assert await cycle.procedural.retrieve(activity) is not None
