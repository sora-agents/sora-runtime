"""Hard-interrupt path: DecisionCycle.interrupt(), phase-boundary checkpoints, mid-flight model-call
abandonment, and the pluggable InterruptHandler / InterruptPolicy seams (Phase 4 A3).

A hard interrupt is an *authoritative* preemption of current work (the wired source is a user stop),
modelled on process scheduling: it saves context (durable on Activity; the per-tick TickResult is
discarded, immune to interrupt staleness per ADR-0011), runs a handler, then the scheduler picks
next. No new activity state — the default handler pauses a schedulable activity to
BLOCKED via an InputWait (await the user's next instruction). An in-flight *external* op is never
abandoned; only a Reason model call / the disposable TickResult is. Reuses tests/fakes.py + a real
FileMemoryBackend cycle, mirroring tests/test_blocked.py.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from fakes import ScriptedTransport
from sora.action import default_action_registry
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry
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
    DefaultInterruptHandler,
    DefaultObserveStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    InterruptPolicy,
    NeverInterruptPolicy,
    Strategies,
    TickResult,
)
from sora.types import (
    Abandoned,
    InputWait,
    InterruptRequest,
    OperationAck,
    OperationInvocation,
    PendingOperation,
    Plan,
    Signal,
    SignalWait,
    Step,
)


class _RecordingReason:
    """Flags whether Reason was reached — proves a checkpoint aborted the tick before it, or that a
    resumed activity became selectable again."""

    def __init__(self) -> None:
        self.called = False

    async def reason(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        self.called = True
        return result


class _SelectFirst:
    """A minimal SituateStrategy: select the first ready activity, no activity-creation or wm
    adjustment, so tests over Reason/Act see exactly the activity they set up."""

    async def situate(
        self,
        activities: list[Activity],
        wm: WorkingMemory,
        cycle: DecisionCycle,
        result: TickResult,
    ) -> TickResult:
        if result.activity is None and activities:
            return TickResult(activity=activities[0])
        return result


class _NoopSituate:
    async def situate(
        self,
        activities: list[Activity],
        wm: WorkingMemory,
        cycle: DecisionCycle,
        result: TickResult,
    ) -> TickResult:
        return result


class _HangingReason:
    """A Reason whose model call is a hangable coroutine routed through `cycle.abandon_on_interrupt`
    — the shape the default Reason uses. `entered` fires once the call is in flight; `release` lets
    the abandoned call finish cleanly at teardown (an abandoned call is *not* cancelled in
    production; the test releases it only to avoid a leaked pending task)."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def reason(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        async def _model_call() -> str:
            self.entered.set()
            await self.release.wait()
            return "a plan"  # a stand-in model result, discarded on abandonment

        outcome = await cycle.abandon_on_interrupt(_model_call())
        if isinstance(outcome, Abandoned):
            return result  # bail without mutating; the checkpoint after Reason aborts the tick
        return result


class _AbandonableInferReason:
    """Mirrors DefaultReasonStrategy's structure: a hangable model call via `abandon_on_interrupt`
    whose result is written to `activity.plan` ONLY when not abandoned. Lets a test prove an
    abandoned call never lands its mutation late (the stale-plan race the guard closes)."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.plan = Plan(id="p", goal="g", steps=[Step(next_action="wait", params={})])

    async def reason(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        async def _infer() -> Plan:
            self.entered.set()
            await self.release.wait()
            return self.plan

        outcome = await cycle.abandon_on_interrupt(_infer())
        if isinstance(outcome, Abandoned):
            return result  # guarded: no mutation when the call was abandoned
        activity.plan = outcome
        return result


class _FiresOnPolicy:
    """An InterruptPolicy that preempts when a signal of the configured name is pushed — the shape a
    real, application-specific policy (e.g. an inbox diff) takes."""

    def __init__(self, trigger: str) -> None:
        self.trigger = trigger

    def decide(self, source: str, signal: Signal, wm: WorkingMemory) -> InterruptRequest | None:
        if signal.name == self.trigger:
            return InterruptRequest(Signal("preempt", {"from": source}), target=None)
        return None


def _cycle(
    tmp_path: Path,
    *,
    reason: object | None = None,
    situate: object | None = None,
    interrupt_policy: InterruptPolicy | None = None,
) -> tuple[DecisionCycle, WorkingMemory, ScriptedTransport]:
    registry = EnvironmentRegistry(adapters={})
    working = WorkingMemory(registry=registry)
    transport = ScriptedTransport()
    cycle = DecisionCycle(
        strategies=Strategies(
            observe=DefaultObserveStrategy(),
            reflect=DefaultReflectStrategy(),
            situate=situate or DefaultSituateStrategy(),  # type: ignore[arg-type]
            reason=reason or _RecordingReason(),  # type: ignore[arg-type]
            act=DefaultActStrategy(),
        ),
        communication=transport,
        actions=default_action_registry(),
        registry=registry,
        working=working,
        semantic=SemanticMemory(FileMemoryBackend(tmp_path / "semantic")),
        procedural=ProceduralMemory(FileMemoryBackend(tmp_path / "procedural")),
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "episodic")),
        interrupt_policy=interrupt_policy,
    )
    return cycle, working, transport


def _ready(activity_id: str, goal: str = "g") -> Activity:
    return Activity(id=activity_id, goal=goal, context={}, state=ActivityState.READY)


def _running(activity_id: str, op_id: str) -> Activity:
    return Activity(
        id=activity_id,
        goal="g",
        context={},
        state=ActivityState.RUNNING,
        pending_operation=PendingOperation(
            id=op_id,
            invocation=OperationInvocation(tool_id="t", operation_name="move", params={}),
            invoked_at=0.0,
        ),
    )


# --------------------------------------------------------------------------------------------------
# Wake latency — an interrupt cuts the inter-tick idle wait (the reactive target), deterministically
# --------------------------------------------------------------------------------------------------


async def test_interrupt_wakes_the_idle_wait(tmp_path: Path) -> None:
    cycle, _, _ = _cycle(tmp_path)
    await cycle.interrupt(Signal("user_stop", {}))  # sets the wake edge
    # A huge interval would hang for ~an hour if the wake edge were ignored; it must return at once.
    await asyncio.wait_for(cycle.wait_between_ticks(3600.0), timeout=1.0)
    assert not cycle._wake.is_set()  # the edge is consumed (cleared) so it only shortens one wait


async def test_idle_wait_sleeps_when_no_interrupt(tmp_path: Path) -> None:
    cycle, _, _ = _cycle(tmp_path)
    # No interrupt pending -> the wait runs to its (tiny) timeout rather than returning on a set.
    await asyncio.wait_for(cycle.wait_between_ticks(0.0), timeout=1.0)
    assert cycle._interrupt is None


# --------------------------------------------------------------------------------------------------
# Phase-boundary checkpoint — a pending interrupt aborts the rest of the tick before Reason/Act
# --------------------------------------------------------------------------------------------------


async def test_pending_interrupt_aborts_tick_before_reason(tmp_path: Path) -> None:
    reason = _RecordingReason()
    cycle, working, _ = _cycle(tmp_path, reason=reason, situate=_SelectFirst())
    working.activities["a1"] = _ready("a1")
    await cycle.interrupt(Signal("user_stop", {}))

    await cycle.tick()

    assert reason.called is False  # the checkpoint after Observe aborted the tick
    assert working.activities["a1"].state is ActivityState.BLOCKED  # routed by the handler
    assert isinstance(working.activities["a1"].blocked_on, InputWait)
    assert cycle._interrupt is None  # discharged (no RUNNING activity left pending)


# --------------------------------------------------------------------------------------------------
# Default handler — a user stop pauses schedulable activities to await input, and a message resumes
# --------------------------------------------------------------------------------------------------


async def test_user_stop_pauses_then_message_resumes(tmp_path: Path) -> None:
    cycle, working, transport = _cycle(tmp_path, situate=_NoopSituate())
    working.activities["a1"] = _ready("a1")

    await cycle.interrupt(Signal("user_stop", {}))
    await cycle.tick()
    paused = working.activities["a1"]
    # state read into a fresh local before each assert so mypy doesn't carry a narrowing across the
    # tick() that mutates it (same idiom as test_blocked.py).
    state = paused.state
    assert state is ActivityState.BLOCKED
    assert isinstance(paused.blocked_on, InputWait)

    # The user's next instruction resumes the paused activity through the normal cycle.
    transport._inbound.append(Message(sender="user", content={"text": "carry on"}, received_at=0.0))
    await cycle.tick()
    state = paused.state
    assert state is ActivityState.READY
    assert paused.blocked_on is None


async def test_default_handler_falls_back_and_warns_on_a_non_user_stop_interrupt(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # DefaultInterruptHandler is a *user-stop* handler, not a general router: an interrupt with any
    # other signal (a custom policy raised it but no paired handler claimed it) still halts to await
    # input — the fail-safe fallback — but is logged at warning level so the strand isn't silent.
    cycle, working, _ = _cycle(tmp_path)
    working.activities["a1"] = _ready("a1")

    request = InterruptRequest(Signal("some_custom_signal", {}), target=None)
    with caplog.at_level("WARNING", logger="sora.strategies"):
        discharged = await DefaultInterruptHandler().handle(request, working, cycle)

    assert discharged is True
    activity = working.activities["a1"]
    assert activity.state is ActivityState.BLOCKED  # still routed (not dropped)
    assert isinstance(activity.blocked_on, InputWait)
    assert any("no interrupt handler routed" in r.message for r in caplog.records)


# --------------------------------------------------------------------------------------------------
# Mid-flight model-call abandonment — interrupt during Reason bails without waiting on the call
# --------------------------------------------------------------------------------------------------


async def test_interrupt_abandons_in_flight_reason(tmp_path: Path) -> None:
    reason = _HangingReason()
    cycle, working, _ = _cycle(tmp_path, reason=reason, situate=_SelectFirst())
    working.activities["a1"] = _ready("a1")

    tick = asyncio.ensure_future(cycle.tick())
    await asyncio.wait_for(reason.entered.wait(), timeout=1.0)  # Reason is mid-flight
    await cycle.interrupt(Signal("user_stop", {}))
    await asyncio.wait_for(tick, timeout=1.0)  # the tick completes instead of hanging on the model

    assert working.activities["a1"].state is ActivityState.BLOCKED  # handler ran, paused it
    assert (
        cycle._abandoned
    )  # the hung call was abandoned (kept referenced), not awaited or cancelled

    reason.release.set()  # let the abandoned call finish so no pending task leaks past the test
    await asyncio.sleep(0)


async def test_abandoned_model_call_never_lands_its_mutation(tmp_path: Path) -> None:
    # The stale-plan race the guard closes: an abandoned infer must not write its (now-stale) plan
    # onto an activity *after* the interrupt handler has re-routed it. With the mutation guarded on
    # the abandon outcome, the plan the handler left in place is never clobbered by the late result.
    strat = _AbandonableInferReason()
    cycle, working, _ = _cycle(tmp_path, reason=strat, situate=_SelectFirst())
    working.activities["a1"] = _ready("a1")  # plan is None

    tick = asyncio.ensure_future(cycle.tick())
    await asyncio.wait_for(strat.entered.wait(), timeout=1.0)  # infer is mid-flight
    await cycle.interrupt(Signal("user_stop", {}))
    await asyncio.wait_for(tick, timeout=1.0)

    activity = working.activities["a1"]
    assert activity.state is ActivityState.BLOCKED  # handler ran (user stop -> paused)
    assert activity.plan is None  # guarded: the abandoned infer's mutation was skipped

    strat.release.set()  # let the abandoned infer resolve in the background
    await asyncio.sleep(0)
    assert working.activities["a1"].plan is None  # still not clobbered after it completes


async def test_abandon_on_interrupt_returns_the_value_absent_an_interrupt(tmp_path: Path) -> None:
    # The no-interrupt path: the raced call completes normally, its value is returned, and the
    # caller applies the mutation as usual (nothing is abandoned).
    strat = _AbandonableInferReason()
    cycle, working, _ = _cycle(tmp_path, reason=strat, situate=_SelectFirst())
    working.activities["a1"] = _ready("a1")
    strat.release.set()  # the call completes immediately; no interrupt is raised

    await cycle.tick()

    assert working.activities["a1"].plan is strat.plan  # value applied
    assert not cycle._abandoned  # nothing was abandoned


# --------------------------------------------------------------------------------------------------
# The never-abandon-external-op invariant — a RUNNING op finishes; the interrupt is honored after
# --------------------------------------------------------------------------------------------------


async def test_running_external_op_is_not_abandoned(tmp_path: Path) -> None:
    cycle, working, _ = _cycle(tmp_path, situate=_NoopSituate())
    activity = _running("a1", op_id="op-1")
    working.activities["a1"] = activity

    await cycle.interrupt(Signal("user_stop", {}))
    await cycle.tick()
    # The op is still in flight: the handler leaves it RUNNING and the request stays pending. state
    # read into a fresh local before each assert so mypy doesn't carry a narrowing across tick().
    state = activity.state
    assert state is ActivityState.RUNNING
    assert activity.pending_operation is not None
    assert cycle._interrupt is not None

    # Once its ack lands the op resolves normally (RUNNING -> READY), then the interrupt is honored.
    cycle.result_sink.push("op-1", OperationAck(ok=True, result={}))
    await cycle.tick()
    state = activity.state
    assert state is ActivityState.BLOCKED
    assert isinstance(activity.blocked_on, InputWait)
    assert cycle._interrupt is None  # discharged now


# --------------------------------------------------------------------------------------------------
# Late-ack safety — a late ack for an activity a hard interrupt already routed away never resurrects
# --------------------------------------------------------------------------------------------------


async def test_late_ack_does_not_resurrect_a_routed_activity(tmp_path: Path) -> None:
    cycle, working, _ = _cycle(tmp_path)
    # An activity no longer RUNNING (here TERMINATED) that still carries a pending_operation id.
    activity = Activity(
        id="a1",
        goal="g",
        context={},
        state=ActivityState.TERMINATED,
        pending_operation=PendingOperation(
            id="op-1",
            invocation=OperationInvocation(tool_id="t", operation_name="move", params={}),
            invoked_at=0.0,
        ),
    )
    working.activities["a1"] = activity
    cycle.result_sink.push("op-1", OperationAck(ok=True, result={}))

    await DefaultObserveStrategy().observe(cycle)

    assert activity.state is ActivityState.TERMINATED  # not flipped to READY by the late ack
    assert activity.last_operation is None  # the resolve was skipped entirely


# --------------------------------------------------------------------------------------------------
# InterruptPolicy seam — a policy promotes a pushed signal to a hard interrupt; default never does
# --------------------------------------------------------------------------------------------------


async def test_policy_promotes_a_signal_to_an_interrupt(tmp_path: Path) -> None:
    cycle, _, _ = _cycle(tmp_path, interrupt_policy=_FiresOnPolicy("new_email"))

    cycle.signal_sink.push("inbox", Signal("state_changed", {}))  # not the trigger
    assert cycle._interrupt is None
    assert not cycle._wake.is_set()

    cycle.signal_sink.push("inbox", Signal("new_email", {"id": "e-9"}))  # the trigger
    assert cycle._interrupt is not None
    assert cycle._interrupt.signal.name == "preempt"
    assert cycle._wake.is_set()


async def test_default_policy_never_preempts(tmp_path: Path) -> None:
    cycle, working, _ = _cycle(tmp_path)
    assert isinstance(cycle.interrupt_policy, NeverInterruptPolicy)

    # The cooperative signal path is unchanged: a pushed signal is drained into wm.signals, never a
    # hard interrupt, and can still resume a BLOCKED (SignalWait) activity.
    activity = Activity(
        id="a1",
        goal="g",
        context={},
        state=ActivityState.BLOCKED,
        blocked_on=SignalWait("done", source="t"),
    )
    working.activities["a1"] = activity
    cycle.signal_sink.push("t", Signal("done", {}))
    assert cycle._interrupt is None  # no preemption

    await DefaultObserveStrategy().observe(cycle)
    assert activity.state is ActivityState.READY  # cooperative resume still works
    assert len(working.signals) == 1
