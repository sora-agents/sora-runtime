"""Hard-interrupt path: DecisionCycle.interrupt(), phase-boundary checkpoints, the stale-inference
reconciliation, and the pluggable InterruptHandler / InterruptPolicy seams.

A hard interrupt is an *authoritative* preemption of current work (the wired source is a user stop),
modelled on process scheduling: it saves context (durable on Activity; the per-tick TickResult is
discarded, immune to interrupt staleness per ADR-0011), runs a handler, then the scheduler picks
next. No new activity state — the default handler pauses a schedulable activity to
BLOCKED via an InputWait (await the user's next instruction). Nothing in-flight is cut mid-flight:
an external op finishes and its ack is honored at the next checkpoint; an off-cycle infer/ground
(ADR-0021) likewise runs to completion, but a result whose id no longer matches the activity's live
``pending_inference`` (a handler re-routed it) is discarded on resolve rather than applied — the
same ``state is RUNNING`` reconciliation the late-ack guard uses. Reuses tests/fakes.py + a real
FileMemoryBackend cycle, mirroring tests/test_blocked.py.
"""

from __future__ import annotations

import asyncio
import logging
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
    InferenceResult,
    InputWait,
    InterruptRequest,
    OperationAck,
    OperationInvocation,
    PendingInference,
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


class _FiresOnPolicy:
    """An InterruptPolicy that preempts when a signal of the configured name is pushed — the shape a
    real, application-specific policy (e.g. an inbox diff) takes."""

    def __init__(self, trigger: str) -> None:
        self.trigger = trigger

    def decide(self, source: str, signal: Signal, wm: WorkingMemory) -> InterruptRequest | None:
        if signal.name == self.trigger:
            return InterruptRequest(Signal("preempt", {"from": source}), target=None)
        return None


class _RaisingPolicy:
    """An InterruptPolicy whose decide() fails — the realistic shape once a policy reads the live
    tool at push time (ADR-0020): that is real I/O against a concurrently-mutating artifact."""

    def decide(self, source: str, signal: Signal, wm: WorkingMemory) -> InterruptRequest | None:
        raise RuntimeError("dictionary changed size during iteration")


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


def _inferring(activity_id: str, *, inf_id: str, kind: str) -> Activity:
    """An activity RUNNING on an off-cycle infer/ground — the deferred-result waiting state."""
    return Activity(
        id=activity_id,
        goal="g",
        context={},
        state=ActivityState.RUNNING,
        pending_inference=PendingInference(id=inf_id, kind=kind, requested_at=0.0),
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
    # Real Situate (not a noop) so the resume path and activity-creation both run this tick: the
    # follow-up must resume the paused activity AND not also spawn a ghost activity from its text.
    cycle, working, transport = _cycle(tmp_path)
    a1 = _ready("a1")
    a1.plan = Plan(id="p", goal="g", steps=[Step(next_action="wait", params={})])
    a1.step_index = 1  # mid-plan, to prove the resume clears it rather than resuming in place
    working.activities["a1"] = a1

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
    paused = working.activities["a1"]
    state = paused.state
    assert state is ActivityState.READY
    assert paused.blocked_on is None
    # Plan cleared so Reason re-infers with the follow-up + executed history visible, rather than
    # silently advancing the stale plan and never seeing the instruction (the resume bug).
    assert paused.plan is None
    assert paused.step_index == 0
    # The follow-up was claimed as reconsideration input (messages_cursor), so Situate did NOT
    # mint a ghost activity from its text — the double-duty bug this fix closes.
    assert list(working.activities) == ["a1"]
    assert working.messages_cursor == len(working.messages)


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
# Off-cycle inference resolve — an infer/ground result lands 1:1 on its RUNNING activity, next cycle
# --------------------------------------------------------------------------------------------------


async def test_inference_result_resolves_plan_and_readies_activity(tmp_path: Path) -> None:
    # The happy path of the deferred-result mechanism: an activity RUNNING on a pending_inference
    # (kind="plan") picks up the Plan from inference_sink in Observe, resets step_index, goes READY.
    cycle, working, _ = _cycle(tmp_path)
    activity = _inferring("a1", inf_id="inf-1", kind="plan")
    activity.step_index = 7  # a stale index from before; the resolve must reset it to 0
    working.activities["a1"] = activity
    plan = Plan(id="p", goal="g", steps=[Step(next_action="wait", params={})])
    cycle.inference_sink.push("inf-1", InferenceResult(id="inf-1", value=plan))

    await DefaultObserveStrategy().observe(cycle)

    assert activity.plan is plan
    assert activity.step_index == 0
    assert activity.pending_inference is None
    assert activity.state is ActivityState.READY


async def test_ground_inference_result_parks_grounded_params(tmp_path: Path) -> None:
    # kind="ground": the resolved params land on grounded_params (for Reason's next pass), not plan.
    cycle, working, _ = _cycle(tmp_path)
    activity = _inferring("a1", inf_id="inf-1", kind="ground")
    working.activities["a1"] = activity
    cycle.inference_sink.push("inf-1", InferenceResult(id="inf-1", value={"to": "boss@x"}))

    await DefaultObserveStrategy().observe(cycle)

    assert activity.grounded_params == {"to": "boss@x"}
    assert activity.plan is None  # a ground result never touches the plan
    assert activity.pending_inference is None
    assert activity.state is ActivityState.READY


# --------------------------------------------------------------------------------------------------
# Stale-inference discard — a result for an activity a handler re-routed never lands its mutation
# --------------------------------------------------------------------------------------------------


async def test_stale_inference_is_discarded_when_pending_cleared(tmp_path: Path) -> None:
    # The reconsideration shape (ReconsiderInterruptHandler): a handler cleared pending_inference
    # and returned the activity to READY. The in-flight call still finishes, but its result no
    # longer matches any live pending_inference, so Observe discards it, not writing a stale plan.
    cycle, working, _ = _cycle(tmp_path)
    activity = _ready("a1")  # pending_inference is None — the handler invalidated it
    working.activities["a1"] = activity
    stale = Plan(id="stale", goal="g", steps=[Step(next_action="wait", params={})])
    cycle.inference_sink.push("inf-1", InferenceResult(id="inf-1", value=stale))

    await DefaultObserveStrategy().observe(cycle)

    assert activity.plan is None  # the stale plan was not applied
    assert activity.state is ActivityState.READY  # not flipped by the discarded result


async def test_stale_inference_is_discarded_after_reinference(tmp_path: Path) -> None:
    # A handler re-fired inference: pending_inference now carries a *new* id. The old call's result
    # (inf-1) no longer matches the live one (inf-2), so it's discarded and the activity stays
    # RUNNING awaiting the fresh result — the id guard, mirroring the external-op late-ack guard.
    cycle, working, _ = _cycle(tmp_path)
    activity = _inferring("a1", inf_id="inf-2", kind="plan")  # the *new* in-flight inference
    working.activities["a1"] = activity
    stale = Plan(id="stale", goal="g", steps=[Step(next_action="wait", params={})])
    cycle.inference_sink.push("inf-1", InferenceResult(id="inf-1", value=stale))  # the old one

    await DefaultObserveStrategy().observe(cycle)

    assert activity.plan is None  # the superseded result was discarded
    assert activity.pending_inference is not None
    assert activity.pending_inference.id == "inf-2"  # still awaiting the fresh inference
    assert activity.state is ActivityState.RUNNING


async def test_failed_inference_replans_instead_of_stranding(tmp_path: Path) -> None:
    # A model call that raised (malformed output, no LLM, a network error) resolves with an error
    # InferenceResult rather than dying silently and leaving the activity RUNNING forever. The
    # failure surfaces cycle-synchronized (like a failed op), never a permanent hang — and it
    # degrades to a replan rather than terminating: nothing was attempted, so the activity has
    # nothing wrong with it beyond one unusable model response.
    cycle, working, _ = _cycle(tmp_path)
    activity = _inferring("a1", inf_id="inf-1", kind="plan")
    working.activities["a1"] = activity
    # The shape InferAction actually reports: repr(exc), whose message quotes the offending output.
    cycle.inference_sink.push(
        "inf-1", InferenceResult(id="inf-1", error="ValueError('bad plan JSON: {\"steps\":')")
    )

    await DefaultObserveStrategy().observe(cycle)

    state = activity.state
    assert state is ActivityState.READY  # surfaced and retryable, not stranded RUNNING
    assert activity.pending_inference is None
    assert activity.plan is None  # Reason will infer again on the next pass
    # The attempt is on the record so the breaker can see a second one repeat — and the trail entry
    # is normalized to the cause, since the quoted output differs every attempt and a raw entry
    # would never compare equal to the next one.
    assert activity.replan_trail == [
        "the plan inference did not return a usable result (ValueError)"
    ]


async def test_discarded_inference_emits_a_meter_cue(tmp_path: Path) -> None:
    # The Observe->meter wiring: a result no live activity claims (invalidated/superseded) emits a
    # `discarded` sora.llm cue, so LLMMeter can fold that call's already-metered cost into its
    # wasted bucket instead of counting it as useful work.
    cycle, working, _ = _cycle(tmp_path)
    working.activities["a1"] = _ready("a1")  # pending_inference None -> the result is orphaned
    cycle.inference_sink.push(
        "inf-1", InferenceResult(id="inf-1", value=Plan(id="p", goal="g", steps=[]))
    )

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign,assignment]
    logger = logging.getLogger("sora.llm")
    previous = logger.level
    logger.setLevel(logging.INFO)  # the cue is INFO; without this it's dropped before any handler
    logger.addHandler(handler)
    try:
        await DefaultObserveStrategy().observe(cycle)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)

    discarded = [r for r in records if r.__dict__.get("llm_event") == "discarded"]
    assert len(discarded) == 1
    assert discarded[0].__dict__["llm_inference_id"] == "inf-1"


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


async def test_user_stop_pauses_an_inference_running_activity_at_once(tmp_path: Path) -> None:
    # The counterpart to the external-op invariant: an activity RUNNING only on an off-cycle
    # infer/ground has no side effect to protect, so a user stop must NOT wait out the (unbounded)
    # model call. The handler drops the inference and pauses to BLOCKED immediately, discharging the
    # interrupt this cycle — unlike an in-flight external op, which keeps it pending (ADR-0021).
    cycle, working, _ = _cycle(tmp_path, situate=_NoopSituate())
    activity = _inferring("a1", inf_id="inf-1", kind="plan")
    working.activities["a1"] = activity

    await cycle.interrupt(Signal("user_stop", {}))
    await cycle.tick()

    state = activity.state
    assert state is ActivityState.BLOCKED  # paused now, not left RUNNING for the model to finish
    assert isinstance(activity.blocked_on, InputWait)
    assert activity.pending_inference is None  # inference invalidated (discarded on resolve)
    assert cycle._interrupt is None  # discharged this cycle — no external op held it pending

    # The now-stale result, arriving after the handler dropped it, is discarded — never resurrects.
    stale = Plan(id="p", goal="g", steps=[])
    cycle.inference_sink.push("inf-1", InferenceResult(id="inf-1", value=stale))
    await DefaultObserveStrategy().observe(cycle)
    state = activity.state
    assert state is ActivityState.BLOCKED
    assert activity.plan is None


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


async def test_a_failing_policy_degrades_to_no_interrupt(tmp_path: Path) -> None:
    # decide() runs on the *pusher's* stack — an adapter callback, not a tick — so an exception
    # would unwind through the adapter and out of Agent.run's loop, killing the agent. A failed
    # screen must degrade to "no interrupt"; the signal still reaches the cooperative drain.
    cycle, working, _ = _cycle(tmp_path, interrupt_policy=_RaisingPolicy())

    cycle.signal_sink.push("inbox", Signal("state_changed", {}))  # must not raise

    assert cycle._interrupt is None
    assert not cycle._wake.is_set()
    assert [sig.name async for _src, sig in cycle.signal_sink.drain()] == ["state_changed"]


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
