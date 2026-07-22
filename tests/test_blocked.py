"""Blocked-state machinery: mechanical, Observe-hosted suspend/resume driven by a manual-declared
operation completion signal.

The *second* kind of waiting (the first being the automatic RUNNING->READY resolve): after a
long-running op's own ack comes back, if the op's manual declares a completion signal
(``OperationSpecification.completion_signal``), the activity blocks until that signal is observed.
Both halves live in ``DefaultObserveStrategy`` and are fully mechanical (name-equality match, no
LLM, no planner) — the wait is structurally declared, so there is no judgment to make:

* **suspend** — layered *on top of* the automatic resolve (last_operation/history still set), never
  fused into it: a resolved, successful, signal-declaring op goes READY then immediately BLOCKED;
* **resume** — a BLOCKED activity whose ``blocked_on`` matches an observed signal returns to READY;
  the matched signal is *not* evicted — it stays in the shared, append-only ``wm.signals`` log so it
  can also satisfy another activity blocked on the same wait, or a strategy reading it directly;
* **early signal** — a completion signal that beats its op's ack is recognized at suspend time, so
  the two waits compose rather than deadlock — again without consuming the signal;
* **orphan** — a signal with no (current) waiter is retained (fire-and-forget; it may match a later
  suspend or resume), bounded only by ``_SIGNAL_RETENTION`` so the store can't grow without limit.

Robotic-arm is the canonical driver (``move_to`` -> wait ``target_reached`` before actuating the
gripper — a safety interlock). The manual is built here with a structured ``completion_signal`` (the
shipped ``robotic-arm.md`` fixture states it in prose; the parser lift of the ``completes_on`` block
is covered in ``test_manual.py``). Reuses ``tests/fakes.py`` + a real ``FileMemoryBackend`` cycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fakes import FakeAdapter, FakeTool, FakeWorkspace, ScriptedTransport
from sora.action import default_action_registry
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, WorkspaceOrigin
from sora.manual import Manual, OperationSpecification
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from sora.strategies import (
    _SIGNAL_RETENTION,
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    Strategies,
    TickResult,
)
from sora.types import (
    OperationAck,
    OperationInvocation,
    PendingOperation,
    Signal,
    SignalWait,
)

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://arm")


class _RecordingReason:
    """Flags whether the cycle reached Reason — i.e. whether Situate selected the activity. Proves a
    BLOCKED activity is skipped and a resumed one becomes selectable again."""

    def __init__(self) -> None:
        self.called = False

    async def reason(
        self, activity: Any, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        self.called = True
        return result  # no step -> Act is never reached


def _arm_manual() -> Manual:
    # move_to declares a completion signal; close_gripper is synchronous (None). Structured specs
    # are what the mechanical suspend reads (the fixture states the same in prose — see module doc).
    return Manual(
        id="robotic-arm",
        metadata={},
        description="arm",
        observable_properties=[],
        signals=[],
        operations=[
            OperationSpecification(
                name="move_to", description="", parameters={}, completion_signal="target_reached"
            ),
            OperationSpecification(name="close_gripper", description="", parameters={}),
        ],
    )


async def _joined_arm(
    tmp_path: Path, *, reason: _RecordingReason | None = None
) -> tuple[DecisionCycle, WorkingMemory, FakeTool]:
    tool = FakeTool(
        "robotic-arm",
        manual=_arm_manual(),
        invoke_results={"move_to": {"accepted": True}, "close_gripper": {"aperture": 0}},
    )
    registry = EnvironmentRegistry(
        adapters={_ORIGIN: FakeAdapter("fake", FakeWorkspace("arm-ws", _ORIGIN, [tool]))}
    )
    await registry.join(_ORIGIN)
    working = WorkingMemory(registry=registry)
    cycle = DecisionCycle(
        strategies=Strategies(
            observe=DefaultObserveStrategy(),
            reflect=DefaultReflectStrategy(),
            situate=DefaultSituateStrategy(),
            reason=reason or _RecordingReason(),
            act=DefaultActStrategy(),
        ),
        communication=ScriptedTransport(),
        actions=default_action_registry(),
        registry=registry,
        working=working,
        semantic=SemanticMemory(FileMemoryBackend(tmp_path / "semantic")),
        procedural=ProceduralMemory(FileMemoryBackend(tmp_path / "procedural")),
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "episodic")),
    )
    return cycle, working, tool


def _running_move_to(activity_id: str, op_id: str) -> Activity:
    """A RUNNING activity mid-``move_to`` — the state InvokeAction leaves behind (minus the
    off-cycle round-trip), ready for Observe's result-sink resolve to consume."""
    return Activity(
        id=activity_id,
        goal="pick up the block",
        context={},
        state=ActivityState.RUNNING,
        pending_operation=PendingOperation(
            id=op_id,
            invocation=OperationInvocation(
                tool_id="robotic-arm", operation_name="move_to", params={}
            ),
            invoked_at=0.0,
        ),
    )


# --------------------------------------------------------------------------------------------------
# Suspend — a resolved op that declares a completion signal blocks the activity
# --------------------------------------------------------------------------------------------------


async def test_completed_op_with_completion_signal_suspends(tmp_path: Path) -> None:
    cycle, working, _ = await _joined_arm(tmp_path)
    activity = _running_move_to("a1", op_id="op-1")
    working.activities["a1"] = activity
    cycle.result_sink.push("op-1", OperationAck(ok=True, result={"accepted": True}))

    await DefaultObserveStrategy().observe(cycle)

    assert activity.state is ActivityState.BLOCKED
    assert activity.blocked_on == SignalWait("target_reached", source="robotic-arm")
    # Layered, not fused: the automatic resolve still happened (belief to ground on), the blocked
    # wait is a *second*, separate step on top of it.
    assert activity.last_operation == OperationAck(ok=True, result={"accepted": True})
    assert activity.pending_operation is None
    assert len(activity.history) == 1


async def test_synchronous_op_does_not_suspend(tmp_path: Path) -> None:
    # close_gripper declares no completion signal -> the resolve leaves it READY, no block.
    cycle, working, _ = await _joined_arm(tmp_path)
    activity = Activity(
        id="a1",
        goal="g",
        context={},
        state=ActivityState.RUNNING,
        pending_operation=PendingOperation(
            id="op-1",
            invocation=OperationInvocation(
                tool_id="robotic-arm", operation_name="close_gripper", params={}
            ),
            invoked_at=0.0,
        ),
    )
    working.activities["a1"] = activity
    cycle.result_sink.push("op-1", OperationAck(ok=True, result={"aperture": 0}))

    await DefaultObserveStrategy().observe(cycle)

    assert activity.state is ActivityState.READY
    assert activity.blocked_on is None


async def test_failed_op_terminates_not_blocks(tmp_path: Path) -> None:
    # A failure isn't a completion to wait past: the op resolved not-ok, so no suspend (Reflect will
    # terminate it). Guards against blocking forever on a signal a failed op will never emit.
    cycle, working, _ = await _joined_arm(tmp_path)
    activity = _running_move_to("a1", op_id="op-1")
    working.activities["a1"] = activity
    cycle.result_sink.push("op-1", OperationAck(ok=False, result="motor fault"))

    await DefaultObserveStrategy().observe(cycle)

    assert activity.state is ActivityState.READY  # not BLOCKED
    assert activity.blocked_on is None


# --------------------------------------------------------------------------------------------------
# Resume — an observed signal that satisfies the wait returns the activity to READY and is evicted
# --------------------------------------------------------------------------------------------------


async def test_blocked_activity_resumes_when_signal_arrives(tmp_path: Path) -> None:
    cycle, working, _ = await _joined_arm(tmp_path)
    activity = Activity(
        id="a1",
        goal="g",
        context={},
        state=ActivityState.BLOCKED,
        blocked_on=SignalWait("target_reached", source="robotic-arm"),
    )
    working.activities["a1"] = activity
    cycle.signal_sink.push("robotic-arm", Signal("target_reached", {}))

    await DefaultObserveStrategy().observe(cycle)

    assert activity.state is ActivityState.READY
    assert activity.blocked_on is None
    assert len(working.signals) == 1  # retained, not evicted — still visible to other readers


async def test_wrong_signal_does_not_resume(tmp_path: Path) -> None:
    cycle, working, _ = await _joined_arm(tmp_path)
    activity = Activity(
        id="a1",
        goal="g",
        context={},
        state=ActivityState.BLOCKED,
        blocked_on=SignalWait("target_reached", source="robotic-arm"),
    )
    working.activities["a1"] = activity
    cycle.signal_sink.push("robotic-arm", Signal("gripper_stalled", {}))

    await DefaultObserveStrategy().observe(cycle)

    assert activity.state is ActivityState.BLOCKED  # still waiting
    assert len(working.signals) == 1  # the non-matching signal is retained, not consumed


async def test_signal_from_another_source_does_not_resume(tmp_path: Path) -> None:
    # The wait is source-scoped to the completing tool: a same-named signal from another tool is not
    # the one we're waiting on.
    cycle, working, _ = await _joined_arm(tmp_path)
    activity = Activity(
        id="a1",
        goal="g",
        context={},
        state=ActivityState.BLOCKED,
        blocked_on=SignalWait("target_reached", source="robotic-arm"),
    )
    working.activities["a1"] = activity
    cycle.signal_sink.push("other-arm", Signal("target_reached", {}))

    await DefaultObserveStrategy().observe(cycle)

    assert activity.state is ActivityState.BLOCKED
    assert len(working.signals) == 1


# --------------------------------------------------------------------------------------------------
# The early-signal race and orphan lifecycle
# --------------------------------------------------------------------------------------------------


async def test_early_completion_signal_never_blocks(tmp_path: Path) -> None:
    # The completion signal beats (or arrives with) the op's ack: the suspend pass finds it already
    # present and stays READY — the two waits compose, they don't deadlock.
    cycle, working, _ = await _joined_arm(tmp_path)
    activity = _running_move_to("a1", op_id="op-1")
    working.activities["a1"] = activity
    cycle.signal_sink.push("robotic-arm", Signal("target_reached", {}))  # arrives this cycle
    cycle.result_sink.push("op-1", OperationAck(ok=True, result={"accepted": True}))

    await DefaultObserveStrategy().observe(cycle)

    assert activity.state is ActivityState.READY  # never blocked
    assert activity.blocked_on is None
    assert len(working.signals) == 1  # not consumed — left for any other waiter/reader


async def test_early_signal_also_resumes_a_separately_blocked_activity(tmp_path: Path) -> None:
    # Regression: the early-consume branch used to *delete* the signal from wm.signals the moment
    # it satisfied one invocation's early check — stealing it from a *different* activity already
    # BLOCKED on the identical wait, since suspend runs before resume in the same tick. Now that
    # matching never mutates wm.signals, both see it in the same tick.
    cycle, working, _ = await _joined_arm(tmp_path)
    already_blocked = Activity(
        id="a1",
        goal="already waiting",
        context={},
        state=ActivityState.BLOCKED,
        blocked_on=SignalWait("target_reached", source="robotic-arm"),
    )
    just_finishing = _running_move_to("a2", op_id="op-2")
    working.activities["a1"] = already_blocked
    working.activities["a2"] = just_finishing
    cycle.signal_sink.push("robotic-arm", Signal("target_reached", {}))
    cycle.result_sink.push("op-2", OperationAck(ok=True, result={"accepted": True}))

    await DefaultObserveStrategy().observe(cycle)

    assert already_blocked.state is ActivityState.READY  # resumed, not starved
    assert already_blocked.blocked_on is None
    assert just_finishing.state is ActivityState.READY  # never blocked (early signal)
    assert just_finishing.blocked_on is None


async def test_two_activities_blocked_on_the_same_signal_both_resume(tmp_path: Path) -> None:
    # Regression: resume used to evict the matched signal on the first hit, starving a second
    # activity blocked on the identical (name, source) wait until a second signal occurrence — which
    # may never come. Now the signal is left in place for later activities in the same pass to see.
    cycle, working, _ = await _joined_arm(tmp_path)
    first = Activity(
        id="a1",
        goal="g1",
        context={},
        state=ActivityState.BLOCKED,
        blocked_on=SignalWait("target_reached", source="robotic-arm"),
    )
    second = Activity(
        id="a2",
        goal="g2",
        context={},
        state=ActivityState.BLOCKED,
        blocked_on=SignalWait("target_reached", source="robotic-arm"),
    )
    working.activities["a1"] = first
    working.activities["a2"] = second
    cycle.signal_sink.push("robotic-arm", Signal("target_reached", {}))

    await DefaultObserveStrategy().observe(cycle)

    assert first.state is ActivityState.READY
    assert second.state is ActivityState.READY


async def test_orphan_signal_is_retained_for_a_future_waiter(tmp_path: Path) -> None:
    # A signal with no current waiter (arrives before its op resolves, e.g.) is fire-and-forget and
    # kept, so a later suspend in a subsequent cycle can still find it (the early-signal window).
    cycle, working, _ = await _joined_arm(tmp_path)
    cycle.signal_sink.push("robotic-arm", Signal("target_reached", {}))

    await DefaultObserveStrategy().observe(cycle)

    assert len(working.signals) == 1  # retained, not dropped


async def test_signal_store_is_bounded(tmp_path: Path) -> None:
    # Orphans can't grow the store without bound; the newest _SIGNAL_RETENTION win.
    cycle, working, _ = await _joined_arm(tmp_path)
    for i in range(_SIGNAL_RETENTION + 10):
        cycle.signal_sink.push("x", Signal("s", {"i": i}))

    await DefaultObserveStrategy().observe(cycle)

    assert len(working.signals) == _SIGNAL_RETENTION
    assert working.signals[-1].payload.payload["i"] == _SIGNAL_RETENTION + 9  # newest kept


# --------------------------------------------------------------------------------------------------
# Full tick() sequence — suspend hides the activity from selection; resume makes it selectable again
# --------------------------------------------------------------------------------------------------


async def test_tick_suspends_then_resumes_across_cycles(tmp_path: Path) -> None:
    reason = _RecordingReason()
    cycle, working, _ = await _joined_arm(tmp_path, reason=reason)
    activity = _running_move_to("a1", op_id="op-1")
    working.activities["a1"] = activity

    # Cycle 1: the op resolves and the activity blocks — Situate must not select a BLOCKED activity.
    # (state re-read into a fresh local each cycle so mypy doesn't carry a stale narrowing across
    # the tick() that mutates it.)
    cycle.result_sink.push("op-1", OperationAck(ok=True, result={"accepted": True}))
    await cycle.tick()
    state = activity.state
    assert state is ActivityState.BLOCKED
    assert reason.called is False  # skipped by selection while blocked

    # A cycle with neither the signal nor anything else keeps it blocked and unselected.
    await cycle.tick()
    state = activity.state
    assert state is ActivityState.BLOCKED
    assert reason.called is False

    # Cycle 3: the awaited signal arrives -> resume -> selectable, so Reason is now reached.
    cycle.signal_sink.push("robotic-arm", Signal("target_reached", {}))
    await cycle.tick()
    state = activity.state
    assert state is ActivityState.READY
    assert activity.blocked_on is None
    assert len(working.signals) == 1
    assert reason.called is True
