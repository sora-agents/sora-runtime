"""Context-adaptation plan reconsideration (ADR-0024).

The pluggable ``ReconsiderationPolicy`` levels, the cheap perception change-gate, the off-cycle
plan-validity re-check (``ProceduralMemory.revalidate`` / ``RevalidateAction``), and the Reason
checkpoint that ties them together: before committing a side-effecting step, if the policy asks and
perception has moved since the plan was baselined, fire the revalidation; on an ``invalid`` verdict
re-infer, on ``valid`` proceed. All deterministic — a ``FakeLLMClient`` stands in for the
revalidation, no network, no model.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from fakes import (
    FakeAdapter,
    FakeLLMClient,
    FakeTool,
    FakeWorkspace,
    ScriptedTransport,
    plan_json,
)
from sora.action import (
    FocusAction,
    InferAction,
    RevalidateAction,
    SendAction,
    default_action_registry,
    invoke_step,
)
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, WorkspaceOrigin
from sora.manual import Manual, OperationSpecification
from sora.memory import (
    REVALIDATE_SYSTEM_PROMPT,
    EpisodicMemory,
    FileMemoryBackend,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from sora.perception import Percept
from sora.strategies import (
    _DEFAULT_MAX_REPLAN_ATTEMPTS,
    BeforeEachOp,
    BeforeWrites,
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReasonStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    NoneReconsideration,
    PerceptionSignatureGate,
    Strategies,
    TickResult,
    _perception_signature,
    _step_side_effecting,
)
from sora.types import (
    CompletedOperation,
    InferenceKind,
    InferenceResult,
    InputWait,
    ObservableProperty,
    OperationAck,
    OperationInvocation,
    PendingInference,
    Plan,
    Signal,
    Step,
)

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")

# A tool whose manual marks write_op side-effecting and read_op a pure read — so the before_writes
# checkpoint can tell them apart via OperationSpecification.side_effecting.
_MANUAL = Manual(
    id="t",
    metadata={},
    description="",
    observable_properties=[],
    signals=[],
    operations=[
        OperationSpecification(name="write_op", description="", parameters={}, side_effecting=True),
        OperationSpecification(name="read_op", description="", parameters={}, side_effecting=False),
    ],
)


async def _joined_registry() -> EnvironmentRegistry:
    tool = FakeTool("t", manual=_MANUAL, invoke_results={"write_op": "ok", "read_op": "ok"})
    workspace = FakeWorkspace("ws", _ORIGIN, [tool])
    registry = EnvironmentRegistry(adapters={_ORIGIN: FakeAdapter("fake", workspace)})
    await registry.join(_ORIGIN)  # populate live tools so registry.get("t") resolves the manual
    return registry


async def _cycle(
    tmp_path: Path,
    *,
    reconsideration: object,
    verdict_response: str | list[str] = '{"valid": true}',
    change_gate: object | None = None,
    llm: FakeLLMClient | None = None,
) -> tuple[DecisionCycle, WorkingMemory]:
    registry = await _joined_registry()
    working = WorkingMemory(registry=registry)
    strategies = Strategies(
        observe=DefaultObserveStrategy(),
        reflect=DefaultReflectStrategy(),
        situate=DefaultSituateStrategy(),
        reason=DefaultReasonStrategy(),
        act=DefaultActStrategy(),
    )
    cycle = DecisionCycle(
        strategies=strategies,
        communication=ScriptedTransport(),
        actions=default_action_registry(),
        registry=registry,
        working=working,
        semantic=SemanticMemory(FileMemoryBackend(tmp_path / "sem")),
        procedural=ProceduralMemory(
            FileMemoryBackend(tmp_path / "proc"), llm=llm or FakeLLMClient(verdict_response)
        ),
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "epi")),
        reconsideration=reconsideration,  # type: ignore[arg-type]
        change_gate=change_gate,  # type: ignore[arg-type]
    )
    return cycle, working


async def _resolve_revalidate(cycle: DecisionCycle) -> None:
    """Let the off-cycle revalidation task finish, then run Observe so the verdict is parked."""
    await asyncio.gather(*list(cycle.actions.internal(RevalidateAction.name)._tasks))  # type: ignore[attr-defined]
    await DefaultObserveStrategy().observe(cycle)


# ── Policy truth table ──────────────────────────────────────────────────────────────────────────


def test_none_reconsideration_never_checks() -> None:
    policy = NoneReconsideration()
    assert not any(policy.should_check(s) for s in (True, False, None))


def test_before_writes_checks_writes_and_unknowns_not_known_reads() -> None:
    policy = BeforeWrites()
    assert policy.should_check(True) is True  # a write
    assert policy.should_check(None) is True  # unknown -> treated as a write (conservative)
    assert policy.should_check(False) is False  # a known read is skipped


def test_before_each_op_always_checks() -> None:
    policy = BeforeEachOp()
    assert all(policy.should_check(s) for s in (True, False, None))


# ── side-effecting classification + change-gate ──────────────────────────────────────────────────


async def test_step_side_effecting_reads_the_manual_for_invoke() -> None:
    wm = WorkingMemory(registry=await _joined_registry())
    assert _step_side_effecting(invoke_step("t", "write_op"), wm) is True
    assert _step_side_effecting(invoke_step("t", "read_op"), wm) is False
    assert _step_side_effecting(invoke_step("t", "mystery"), wm) is None  # op not in the manual


async def test_step_side_effecting_classifies_non_invoke_actions() -> None:
    wm = WorkingMemory(registry=await _joined_registry())
    assert _step_side_effecting(Step(FocusAction.name, {"tool_id": "t"}), wm) is False
    assert _step_side_effecting(Step(SendAction.name, {"to": "x", "content": {}}), wm) is None


async def test_perception_signature_moves_on_any_observable_change() -> None:
    wm = WorkingMemory(registry=await _joined_registry())
    base = _perception_signature(wm)
    wm.signals.append(Percept("t", Signal("new_email", {}), 0.0))
    assert _perception_signature(wm) != base  # a new signal
    wm2 = WorkingMemory(registry=await _joined_registry())
    wm2.properties[("t", "p")] = Percept("t", ObservableProperty("p", 1), 0.0)
    before = _perception_signature(wm2)
    wm2.properties[("t", "p")] = Percept("t", ObservableProperty("p", 2), 0.0)  # value changed
    assert _perception_signature(wm2) != before


async def test_perception_signature_is_stable_when_a_value_is_re_observed() -> None:
    # Regression: the signature keys on the property *payload* (value), NOT the whole Percept —
    # _snapshot_properties refreshes `observed_at` with time.time() on every re-observation, so
    # hashing the envelope would make an unchanged property look moved every cycle and fire a
    # revalidation on every write even in a static world (defeating "free when static").
    wm = WorkingMemory(registry=await _joined_registry())
    wm.properties[("t", "p")] = Percept("t", ObservableProperty("p", 1), 100.0)
    base = _perception_signature(wm)
    wm.properties[("t", "p")] = Percept("t", ObservableProperty("p", 1), 999.0)  # same value, later
    assert _perception_signature(wm) == base  # unchanged value -> unchanged signature


async def test_default_change_gate_wraps_the_perception_signature() -> None:
    # The runtime default ChangeGate is just the module signature helper behind the Protocol, so the
    # checkpoint's compare is unchanged when no domain gate is named.
    wm = WorkingMemory(registry=await _joined_registry())
    wm.properties[("t", "p")] = Percept("t", ObservableProperty("p", 1), 0.0)
    wm.signals.append(Percept("t", Signal("s", {}), 0.0))
    assert PerceptionSignatureGate().signature(wm) == _perception_signature(wm)


class _ConstantGate:
    """A ChangeGate that reports the world never moves — projects everything away."""

    def signature(self, wm: WorkingMemory) -> object:
        return "constant"


async def test_change_gate_governs_whether_the_revalidation_fires(tmp_path: Path) -> None:
    # A domain gate that reports "nothing moved" keeps the checkpoint cold even when raw perception
    # moved, proving the reconsideration compare routes through cycle.change_gate — not the built-in
    # signature. This is the efference filter's mechanism (a self-write projected away never fires).
    cycle, working = await _cycle(
        tmp_path, reconsideration=BeforeWrites(), change_gate=_ConstantGate()
    )
    activity = _write_activity(working)
    activity.reconsider_baseline = cycle.change_gate.signature(working)  # "constant"
    working.signals.append(Percept("t", Signal("follow_up", {}), 0.0))  # raw perception moved

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step == invoke_step("t", "write_op")  # gate cold -> committed, no revalidation
    assert activity.pending_inference is None


# ── ProceduralMemory.revalidate ─────────────────────────────────────────────────────────────────


async def test_revalidate_parses_verdict_and_sees_goal_and_remaining_steps(tmp_path: Path) -> None:
    llm = FakeLLMClient('{"valid": false}')
    proc = ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm)
    plan = Plan(id="p", goal="schedule Monday sync", steps=[invoke_step("t", "write_op")])
    activity = Activity(id="a", goal="schedule Monday sync", context={}, plan=plan)

    assert await proc.revalidate(activity) is False
    assert (llm.requests[-1].semantic_label, llm.requests[-1].prompt_version) == (
        "revalidate",
        "1",
    )
    system, user = llm.calls[-1]
    assert system == REVALIDATE_SYSTEM_PROMPT
    assert "schedule Monday sync" in user  # goal is in the prompt
    assert "write_op" in user  # the remaining step is in the prompt


async def test_revalidate_sees_the_operations_already_executed(tmp_path: Path) -> None:
    # The checkpoint fires late in a plan too, where `remaining` is a single step and everything
    # the goal asked for lives in history. Without it the model judges a nearly-finished plan
    # against a goal whose work is nowhere in evidence and reasonably answers "invalid" — the
    # observed-state renderers can't stand in for it (they're truncated snapshots of the world,
    # not a record of what this activity did).
    llm = FakeLLMClient('{"valid": true}')
    proc = ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm)
    plan = Plan(id="p", goal="schedule the sync and reply", steps=[invoke_step("t", "report_op")])
    activity = Activity(id="a", goal="schedule the sync and reply", context={}, plan=plan)
    activity.history.append(
        CompletedOperation(
            invocation=OperationInvocation(
                tool_id="t", operation_name="add_calendar_event", params={"title": "Team sync"}
            ),
            ack=OperationAck(ok=True, result="event_created_42"),
        )
    )

    assert await proc.revalidate(activity) is True
    _system, user = llm.calls[-1]
    assert "add_calendar_event" in user  # the executed operation is in the prompt
    assert "event_created_42" in user  # ...with its result


async def test_revalidate_sees_the_bindings_the_data_ops_produced(tmp_path: Path) -> None:
    # A data-op leaves nothing in history — it writes a named binding — so a plan that spends its
    # early steps narrowing a collection reaches the checkpoint with almost none of its work in
    # evidence. An observed run replanned at step 9 because the only executed result the judge
    # could see was a clock reading, and the replacement plan was the same plan with its bindings
    # renamed: a 19k-token call to rediscover what had already been computed. Unlike a replan
    # (which discards them, so the plan prompt is right not to show them), these are live for the
    # plan being judged, which makes them part of the work already done.
    llm = FakeLLMClient('{"valid": true}')
    proc = ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm)
    goal = "cancel Saturday's appointments"
    plan = Plan(id="p", goal=goal, steps=[invoke_step("t", "delete_op")])
    activity = Activity(id="a", goal=goal, context={}, plan=plan)
    activity.bindings["saturday_appointments"] = [{"event_id": "evt-7", "title": "Brand meeting"}]

    assert await proc.revalidate(activity) is True
    _system, user = llm.calls[-1]
    assert "saturday_appointments" in user  # the binding an earlier data-op step produced
    assert "evt-7" in user  # ...and what it actually holds, not just its name


async def test_revalidate_fail_soft_treats_malformed_answer_as_valid(tmp_path: Path) -> None:
    proc = ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=FakeLLMClient("not json at all"))
    activity = Activity(id="a", goal="g", context={}, plan=Plan(id="p", goal="g", steps=[]))
    assert (
        await proc.revalidate(activity) is True
    )  # a flaky revalidation must not force a replan storm


async def test_revalidate_sees_remaining_steps_across_the_subgoal_stack(tmp_path: Path) -> None:
    # Inside a sub-plan (ADR-0022), the remaining tail spans the active sub-plan *and* each
    # suspended parent's steps after its sub-goal. Judging only the sub-plan would call it "still
    # valid" while a now-stale parent step still waits, so the revalidation must see both.
    llm = FakeLLMClient('{"valid": true}')
    proc = ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm)
    parent_plan = Plan(
        id="parent", goal="g", steps=[invoke_step("t", "read_op"), invoke_step("t", "write_op")]
    )
    subplan = Plan(id="sub", goal="sub", steps=[invoke_step("t", "read_op")])
    activity = Activity(id="a", goal="g", context={}, plan=subplan, step_index=0)
    activity.parent_frames.append((parent_plan, 0, 0))  # sub-goal was parent step 0

    assert await proc.revalidate(activity) is True
    _system, user = llm.calls[-1]
    # the sub-plan's own step and the parent's step *after* the sub-goal (write_op) both appear
    assert "write_op" in user  # parent tail is not dropped
    assert "read_op" in user


# ── Reason checkpoint integration ────────────────────────────────────────────────────────────────


def _write_activity(working: WorkingMemory) -> Activity:
    plan = Plan(id="p", goal="g", steps=[invoke_step("t", "write_op")])
    activity = Activity(id="a", goal="g", context={}, plan=plan, step_index=0)
    working.activities["a"] = activity
    return activity


async def test_none_policy_emits_the_write_without_a_revalidation(tmp_path: Path) -> None:
    cycle, working = await _cycle(tmp_path, reconsideration=NoneReconsideration())
    activity = _write_activity(working)
    activity.reconsider_baseline = _perception_signature(working)
    working.signals.append(Percept("t", Signal("changed", {}), 0.0))  # world moved

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step == invoke_step("t", "write_op")  # committed, no reconsideration
    assert activity.pending_inference is None


async def test_cold_gate_emits_and_baselines(tmp_path: Path) -> None:
    cycle, working = await _cycle(tmp_path, reconsideration=BeforeWrites())
    activity = _write_activity(working)  # no baseline yet, world static

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step == invoke_step("t", "write_op")  # first checkpoint is cold -> proceed
    assert activity.reconsider_baseline is not None  # baselined for next time
    assert activity.pending_inference is None


async def test_hot_gate_fires_revalidation_then_invalid_reinfers(tmp_path: Path) -> None:
    cycle, working = await _cycle(
        tmp_path, reconsideration=BeforeWrites(), verdict_response='{"valid": false}'
    )
    activity = _write_activity(working)
    activity.reconsider_baseline = _perception_signature(working)  # baselined against empty world
    working.signals.append(Percept("t", Signal("follow_up", {}), 0.0))  # world moved -> gate hot

    fired = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert fired.step is None  # no step committed
    assert activity.state.value == "running"
    assert activity.pending_inference is not None
    assert activity.pending_inference.kind == "revalidate"

    await _resolve_revalidate(cycle)
    assert activity.reconsider_verdict is False
    assert activity.state.value == "ready"

    replanned = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert replanned.step is None
    assert activity.plan is None  # invalidated -> Reason will re-infer next
    assert activity.reconsider_verdict is None  # consumed


async def test_invalidated_then_re_inferred_plans_are_both_traced(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The trace of a replan is only useful if it shows *both* plans: the invalidated one (with the
    step it had reached) and the replacement inferred against the moved world. Body text is DEBUG —
    it belongs in the --log-file mirror, not the terminal's one-line-per-event view."""
    cycle, working = await _cycle(
        tmp_path, reconsideration=BeforeWrites(), verdict_response='{"valid": false}'
    )
    activity = _write_activity(working)
    activity.reconsider_baseline = _perception_signature(working)
    working.signals.append(Percept("t", Signal("follow_up", {}), 0.0))  # gate hot
    # Taken inside a sub-plan: reset_for_replan drops the suspended parent too, so a dump of the
    # active plan alone would understate the discard.
    parent_steps = [invoke_step("t", "sub_op"), invoke_step("t", "tail_op")]
    parent = Plan(id="p0", goal="g", steps=parent_steps)
    activity.parent_frames.append((parent, 0, 0))

    with caplog.at_level(logging.DEBUG, logger="sora.strategies"):
        await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
        await _resolve_revalidate(cycle)
        await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())  # invalid
        assert activity.plan is None

        # The replacement inference lands the way Reason's re-infer would.
        replacement = Plan(id="p2", goal="g", steps=[invoke_step("t", "read_op")])
        activity.state = ActivityState.RUNNING
        activity.pending_inference = PendingInference(
            id="inf-2",
            kind=InferenceKind.PLAN,
            requested_at=0.0,
            baseline=_perception_signature(working),
        )
        cycle.inference_sink.push("inf-2", InferenceResult(id="inf-2", value=replacement))
        await DefaultObserveStrategy().observe(cycle)

    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    discarded = [m for m in debug if m.startswith("reason: discarded plan")]
    installed = [m for m in debug if m.startswith("observe: plan for activity")]
    assert len(discarded) == 1
    assert "was at step 0" in discarded[0]
    assert "0: invoke" in discarded[0] and "write_op" in discarded[0]
    # The whole stack, not just the active frame: the parent is dropped with it, so its body is
    # traced too — and in full, unlike the un-run tail the replanning prompt is given.
    assert "-- suspended parent, at sub-goal step 0 --" in discarded[0]
    assert "tail_op" in discarded[0]
    assert len(installed) == 1
    assert "0: invoke" in installed[0] and "read_op" in installed[0]


async def test_invalid_verdict_parks_the_whole_discarded_stack_for_the_replan(
    tmp_path: Path,
) -> None:
    """Invalidation stays a whole-activity redirect — the intention stack is cleared, never popped
    to the stale frame (ADR-0024). What that costs is recovered by parking the discard for the next
    inference to read, rather than by making the reset frame-local."""
    cycle, working = await _cycle(
        tmp_path, reconsideration=BeforeWrites(), verdict_response='{"valid": false}'
    )
    activity = _write_activity(working)
    activity.reconsider_baseline = _perception_signature(working)
    working.signals.append(Percept("t", Signal("follow_up", {}), 0.0))  # gate hot
    parent_steps = [invoke_step("t", "sub_op"), invoke_step("t", "tail_op")]
    parent = Plan(id="p0", goal="g", steps=parent_steps)
    activity.parent_frames.append((parent, 0, 0))

    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    await _resolve_revalidate(cycle)
    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())  # invalid

    assert activity.plan is None and not activity.parent_frames  # blank slate, as before
    assert activity.superseded is not None
    assert activity.superseded.step_index == 0
    assert activity.superseded.parent_frames == [(parent, 0)]  # the suspended parent, not just
    assert activity.superseded.plan.steps == [invoke_step("t", "write_op")]  # the active frame


async def test_superseded_plan_reaches_the_replanning_prompt(tmp_path: Path) -> None:
    """End to end: the plan discarded inside a sub-goal is what the *replacement* inference is
    written against. Without this the re-infer starts blank and re-derives a decomposition that
    several model calls already paid for."""
    llm = FakeLLMClient(['{"valid": false}', plan_json({"tool_id": "t", "op": "read_op"})])
    cycle, working = await _cycle(tmp_path, reconsideration=BeforeWrites(), llm=llm)
    activity = _write_activity(working)
    activity.reconsider_baseline = _perception_signature(working)
    working.signals.append(Percept("t", Signal("follow_up", {}), 0.0))  # gate hot
    parent_steps = [invoke_step("t", "sub_op"), invoke_step("t", "tail_op")]
    parent = Plan(id="p0", goal="g", steps=parent_steps)
    activity.parent_frames.append((parent, 0, 0))

    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    await _resolve_revalidate(cycle)
    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())  # invalid
    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())  # fires _infer_
    await asyncio.gather(*list(cycle.actions.internal(InferAction.name)._tasks))  # type: ignore[attr-defined]

    _system, user = llm.calls[-1]
    assert "previous plan for this goal was abandoned" in user
    assert "write_op" in user  # the discarded active frame's un-run step
    assert "tail_op" in user  # ...and the suspended parent's, which the reset dropped too


async def test_superseded_plan_is_cleared_once_the_replacement_installs(tmp_path: Path) -> None:
    # Parked for exactly one inference: a bundle that outlived its replan would leak into a later,
    # unrelated one as a phantom "previous plan".
    cycle, working = await _cycle(tmp_path, reconsideration=BeforeWrites())
    activity = _write_activity(working)
    activity.reset_for_replan()
    assert activity.superseded is not None

    activity.state = ActivityState.RUNNING
    activity.pending_inference = PendingInference(
        id="inf-1", kind=InferenceKind.PLAN, requested_at=0.0
    )
    replacement = Plan(id="p2", goal="g", steps=[invoke_step("t", "read_op")])
    cycle.inference_sink.push("inf-1", InferenceResult(id="inf-1", value=replacement))
    await DefaultObserveStrategy().observe(cycle)

    assert activity.plan == replacement
    assert activity.superseded is None


async def test_a_reset_with_no_plan_leaves_an_earlier_bundle_intact(tmp_path: Path) -> None:
    # reset_for_replan runs on paths where there may be nothing in flight (a stop arriving while
    # already replanning). Overwriting with an empty capture there would erase the real discard the
    # pending inference is still waiting to read.
    _unused, working = await _cycle(tmp_path, reconsideration=BeforeWrites())
    activity = _write_activity(working)
    activity.reset_for_replan()
    parked = activity.superseded

    activity.reset_for_replan()  # nothing to discard this time

    assert activity.superseded is parked


async def test_hot_gate_valid_verdict_proceeds_and_rebaselines(tmp_path: Path) -> None:
    cycle, working = await _cycle(
        tmp_path, reconsideration=BeforeWrites(), verdict_response='{"valid": true}'
    )
    activity = _write_activity(working)
    activity.reconsider_baseline = _perception_signature(working)
    working.signals.append(Percept("t", Signal("follow_up", {}), 0.0))

    await DefaultReasonStrategy().reason(
        activity, working, cycle, TickResult()
    )  # fires the revalidation
    await _resolve_revalidate(cycle)
    assert activity.reconsider_verdict is True

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert result.step == invoke_step("t", "write_op")  # still valid -> committed
    assert activity.plan is not None  # not invalidated
    # re-baselined to the revalidation's fire-time world (== current here: nothing drifted during
    # flight), so the same change won't re-fire
    assert activity.reconsider_baseline == _perception_signature(working)


async def test_valid_verdict_does_not_absorb_a_change_that_landed_during_revalidation_flight(
    tmp_path: Path,
) -> None:
    # Regression: on "valid" the baseline advances to the world the revalidation was FIRED against,
    # not the verdict-time world. A follow-up that arrives while it is in flight must stay outside
    # the new baseline and earn its own reconsideration at the next write — re-baselining to "now"
    # would silently swallow it. Two writes: the revalidation OK'd the first (so it commits); the
    # mid-flight change must then trip the second write's checkpoint.
    cycle, working = await _cycle(
        tmp_path, reconsideration=BeforeWrites(), verdict_response='{"valid": true}'
    )
    plan = Plan(
        id="p", goal="g", steps=[invoke_step("t", "write_op"), invoke_step("t", "write_op")]
    )
    activity = Activity(id="a", goal="g", context={}, plan=plan, step_index=0)
    working.activities["a"] = activity
    activity.reconsider_baseline = _perception_signature(working)
    working.signals.append(Percept("t", Signal("first", {}), 0.0))  # gate hot

    await DefaultReasonStrategy().reason(
        activity, working, cycle, TickResult()
    )  # fires the revalidation
    fire_time_world = _perception_signature(working)
    working.signals.append(Percept("t", Signal("during_flight", {}), 0.0))  # lands mid-flight
    await _resolve_revalidate(cycle)  # verdict: valid (about the first change only)

    assert (
        activity.reconsider_baseline == fire_time_world
    )  # advanced to fire time, not verdict time
    assert activity.reconsider_baseline != _perception_signature(working)  # the mid-flight drift

    # The revalidation OK'd the first write, so it commits — the mid-flight change is NOT swallowed.
    committed = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert committed.step == invoke_step("t", "write_op")
    assert activity.step_index == 1

    # The second write's checkpoint now trips on the mid-flight change rather than committing blind.
    refired = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert refired.step is None
    assert activity.pending_inference is not None
    assert activity.pending_inference.kind == "revalidate"


async def test_before_writes_skips_a_read_even_with_a_hot_gate(tmp_path: Path) -> None:
    cycle, working = await _cycle(tmp_path, reconsideration=BeforeWrites())
    plan = Plan(id="p", goal="g", steps=[invoke_step("t", "read_op")])
    activity = Activity(id="a", goal="g", context={}, plan=plan)
    working.activities["a"] = activity
    activity.reconsider_baseline = _perception_signature(working)
    working.signals.append(Percept("t", Signal("changed", {}), 0.0))  # hot, but it's a read

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step == invoke_step("t", "read_op")  # a known read is never checked
    assert activity.pending_inference is None


async def test_before_each_op_checks_even_a_read(tmp_path: Path) -> None:
    cycle, working = await _cycle(
        tmp_path, reconsideration=BeforeEachOp(), verdict_response='{"valid": true}'
    )
    plan = Plan(id="p", goal="g", steps=[invoke_step("t", "read_op")])
    activity = Activity(id="a", goal="g", context={}, plan=plan)
    working.activities["a"] = activity
    activity.reconsider_baseline = _perception_signature(working)
    working.signals.append(Percept("t", Signal("changed", {}), 0.0))  # hot

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step is None  # before_each_op judges reads too
    assert activity.pending_inference is not None
    assert activity.pending_inference.kind == "revalidate"


# ── infer-time baseline (ADR-0024): a plan is baselined against the world it was inferred in ──────


def _inferring_plan(working: WorkingMemory, *, baseline: object) -> Activity:
    """An activity RUNNING on an in-flight plan inference carrying its infer-time baseline."""
    activity = Activity(
        id="a",
        goal="g",
        context={},
        state=ActivityState.RUNNING,
        pending_inference=PendingInference(
            id="inf-1", kind=InferenceKind.PLAN, requested_at=0.0, baseline=baseline
        ),
    )
    working.activities["a"] = activity
    return activity


async def test_plan_install_baselines_to_infer_time_world_not_the_drifted_one(
    tmp_path: Path,
) -> None:
    cycle, working = await _cycle(tmp_path, reconsideration=BeforeWrites())
    infer_world = _perception_signature(working)  # W0 — what the planner actually saw
    activity = _inferring_plan(working, baseline=infer_world)
    working.signals.append(Percept("t", Signal("arrived_mid_inference", {}), 0.0))  # world moved
    assert _perception_signature(working) != infer_world  # the drift is real

    plan = Plan(id="p", goal="g", steps=[invoke_step("t", "write_op")])
    cycle.inference_sink.push("inf-1", InferenceResult(id="inf-1", value=plan))
    await DefaultObserveStrategy().observe(cycle)

    assert activity.plan == plan
    assert activity.state.value == "ready"
    # baselined to the infer-time world, not the world after the mid-inference drift
    assert activity.reconsider_baseline == infer_world
    assert activity.reconsider_baseline != _perception_signature(working)


async def test_inference_window_change_is_caught_at_the_first_checkpoint(tmp_path: Path) -> None:
    # The gap infer-time baselining closes: a change lands *during* inference and the world then
    # goes static. An entry-time baseline would fold the change in and gate cold — missing it.
    cycle, working = await _cycle(
        tmp_path, reconsideration=BeforeWrites(), verdict_response='{"valid": false}'
    )
    infer_world = _perception_signature(working)
    activity = _inferring_plan(working, baseline=infer_world)
    working.signals.append(Percept("t", Signal("cancel_during_inference", {}), 0.0))

    plan = Plan(id="p", goal="g", steps=[invoke_step("t", "write_op")])
    cycle.inference_sink.push("inf-1", InferenceResult(id="inf-1", value=plan))
    await DefaultObserveStrategy().observe(cycle)  # installs plan, baseline = infer_world

    # No further change — world static from here. The first write checkpoint still trips.
    fired = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert fired.step is None  # gate hot vs the infer-time baseline -> revalidation fired
    assert activity.pending_inference is not None
    assert activity.pending_inference.kind == "revalidate"

    await _resolve_revalidate(cycle)
    replanned = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert replanned.step is None
    assert activity.plan is None  # invalidated -> re-infer against the current world


# ── invoke-pass placement (ADR-0024): the checkpoint guards the invoke, not the grounding ─────────


async def test_revalidate_between_grounding_and_invoke_does_not_force_a_reground(
    tmp_path: Path,
) -> None:
    # The checkpoint runs on the invoke pass, *after* grounding resolved. When it fires a
    # revalidation, the already-grounded params must stay parked so the next cycle re-emits the same
    # step rather than re-escalating _ground_ (a second, costlier model call). Regression for the
    # pre-invoke check plus the peek-don't-consume grounding lifecycle.
    cycle, working = await _cycle(
        tmp_path, reconsideration=BeforeWrites(), verdict_response='{"valid": true}'
    )
    step = invoke_step("t", "write_op", target={"$decide": "x"})  # a ref: escalates if not parked
    plan = Plan(id="p", goal="g", steps=[step])
    activity = Activity(id="a", goal="g", context={}, plan=plan, step_index=0)
    activity.grounded_params = {"target": "resolved"}  # as if a prior _ground_ escalation resolved
    working.activities["a"] = activity
    activity.reconsider_baseline = _perception_signature(working)
    working.signals.append(Percept("t", Signal("moved_during_grounding", {}), 0.0))  # gate hot

    # Invoke pass: grounding is already done (params parked), so the checkpoint re-checks the plan.
    fired = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert fired.step is None
    assert activity.pending_inference is not None
    assert activity.pending_inference.kind == "revalidate"  # a revalidation, not a re-fired ground
    assert activity.grounded_params == {"target": "resolved"}  # parked params survive the deferral

    await _resolve_revalidate(cycle)
    assert activity.reconsider_verdict is True

    # Next pass: re-emits the *same* grounded step; no second escalation.
    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert result.step is not None
    assert result.step.params["target"] == "resolved"  # from the parked params, not a re-ground
    assert activity.grounded_params is None  # consumed at commit
    assert activity.step_index == 1
    assert activity.pending_inference is None  # did not re-escalate _ground_


# ── Runaway-replan breaker ──────────────────────────────────────────────────────────────────────
#
# A plan is dropped, its replacement is dropped too, and nothing ever runs — each turn costing a
# full planning inference (minutes apiece on a local model in the run that motivated this). The
# bound is deliberately progress-relative: replanning without limit is what an agent in a dynamic
# environment is *for*, so what gets counted is only replans that executed no operation at all.


def _ran_an_op(page: int = 0) -> CompletedOperation:
    """One completed call. ``page`` varies the arguments: only a call the activity has not already
    made counts as progress, so a fixture that re-ran one identical call would be modelling the
    stuck agent rather than the working one."""
    return CompletedOperation(
        OperationInvocation("t", "read_op", {"page": page}), OperationAck(ok=True, result="ok")
    )


def _replanned(activity: Activity, defect: str | None) -> None:
    """One full drop-the-plan turn: a plan has to be present for the reset to record anything."""
    activity.plan = Plan(id="p", goal="g", steps=[invoke_step("t", "write_op")])
    activity.reset_for_replan(defect=defect)


def _halt_prompt(activity: Activity) -> str:
    """The await-input text a tripped breaker parked. Asserts the shape on the way through, since
    every caller below is checking what the text *says*, not that there is text at all."""
    wait = activity.blocked_on
    assert isinstance(wait, InputWait)
    prompt = wait.prompt
    assert prompt is not None
    return prompt


def _stuck(working: WorkingMemory, trail: list[str | None]) -> Activity:
    """An activity whose last ``len(trail)`` plans were abandoned with no operation in between."""
    activity = Activity(id="a", goal="book a day with the film producer", context={})
    working.activities["a"] = activity
    for defect in trail:
        _replanned(activity, defect)
    return activity


def test_an_operation_running_between_replans_forgives_the_trail() -> None:
    """The whole point of counting progress rather than attempts: an agent that keeps adjusting
    while actually getting somewhere must never approach the cap, however long it runs."""
    activity = Activity(id="a", goal="g", context={})
    for page in range(_DEFAULT_MAX_REPLAN_ATTEMPTS * 3):
        _replanned(activity, None)  # the world moved again
        activity.history.append(_ran_an_op(page))  # ...and the new plan still got somewhere new
    _replanned(activity, None)

    assert activity.replan_trail == [None]  # only the latest; every earlier one was forgiven


def test_re_running_an_already_made_call_does_not_forgive_the_trail() -> None:
    """The observed runaway: every replan re-issued ``get_contacts(offset=0)``, which cleared the
    trail each time and kept the breaker asleep while nothing new was ever learned."""
    activity = Activity(id="a", goal="g", context={})
    activity.history.append(_ran_an_op(0))
    for defect in ("no such parameter 'limit'", "friend_contact is empty", "selected is empty"):
        _replanned(activity, defect)
        activity.history.append(_ran_an_op(0))  # the same call, all over again

    assert activity.replan_trail == [
        "no such parameter 'limit'",
        "friend_contact is empty",
        "selected is empty",
    ]


def test_replans_with_nothing_run_between_them_accumulate() -> None:
    activity = Activity(id="a", goal="g", context={})
    _replanned(activity, "no such parameter 'limit'")
    _replanned(activity, None)
    _replanned(activity, "search_contacts returned []")

    assert activity.replan_trail == [
        "no such parameter 'limit'",
        None,
        "search_contacts returned []",
    ]


def test_progress_is_counted_from_the_last_replan_not_from_zero() -> None:
    """A single op run early must not keep forgiving later replans — the mark advances with it."""
    activity = Activity(id="a", goal="g", context={})
    activity.history.append(_ran_an_op())
    _replanned(activity, "a")
    _replanned(activity, "b")

    assert activity.replan_trail == ["a", "b"]


async def test_the_same_defect_twice_halts_before_a_third_plan(tmp_path: Path) -> None:
    """The precise check: the planner was handed that defect in its brief and wrote past it, so a
    third attempt would fail identically. Trips at two, well under the count backstop."""
    cycle, working = await _cycle(tmp_path, reconsideration=NoneReconsideration())
    defect = "get_contacts: no such parameter(s) 'limit' — that operation accepts only offset"
    activity = _stuck(working, [defect, defect])

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step is None
    state = activity.state
    assert state is ActivityState.BLOCKED
    assert activity.pending_inference is None  # no planning inference was spent
    assert defect in _halt_prompt(activity)  # the specific evidence, not just "it failed"


async def test_two_different_defects_are_still_worth_another_attempt(tmp_path: Path) -> None:
    """The inverse: failing differently each time is floundering, not a loop — the planner is at
    least trying something new, so it gets the full count before anyone is interrupted."""
    cycle, working = await _cycle(tmp_path, reconsideration=NoneReconsideration())
    activity = _stuck(working, ["no such parameter 'limit'", "search_contacts returned []"])

    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    state = activity.state
    assert state is not ActivityState.BLOCKED
    assert activity.pending_inference is not None  # planned again


async def test_a_world_that_keeps_moving_does_not_trip_the_same_defect_check(
    tmp_path: Path,
) -> None:
    """Consecutive reconsiderations carry no defect. Two in a row is evidence about the world, not
    about a stuck planner, so they must not halt at two the way a repeated defect does."""
    cycle, working = await _cycle(tmp_path, reconsideration=NoneReconsideration())
    activity = _stuck(working, [None, None])

    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    state = activity.state
    assert state is not ActivityState.BLOCKED
    assert activity.pending_inference is not None


async def test_the_count_backstops_attempts_that_all_fail_differently(tmp_path: Path) -> None:
    cycle, working = await _cycle(tmp_path, reconsideration=NoneReconsideration())
    trail: list[str | None] = [f"distinct failure {i}" for i in range(_DEFAULT_MAX_REPLAN_ATTEMPTS)]
    activity = _stuck(working, trail)

    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    state = activity.state
    assert state is ActivityState.BLOCKED
    assert activity.pending_inference is None
    prompt = _halt_prompt(activity)
    # Mechanically rendered (no model call): every attempt is listed, in order, verbatim.
    for i, reason in enumerate(trail, start=1):
        assert f"  {i}. {reason}" in prompt
    assert "How should I proceed?" in prompt


async def test_a_world_that_keeps_moving_never_trips_the_count_either(tmp_path: Path) -> None:
    """The moving world this runtime is built for must not be able to halt the agent on its own.

    A defect-free entry is a reconsideration: the plan was fine and the world moved under it, so the
    next plan is inferred against a different world — an attempt, not a repeat. A plan whose first
    checkpointed step is a write has run no operation, so nothing forgives its trail, and counting
    those entries halted an agent that had adapted honestly every time. Far past the cap here, and
    still planning."""
    cycle, working = await _cycle(tmp_path, reconsideration=NoneReconsideration())
    trail: list[str | None] = [None] * (_DEFAULT_MAX_REPLAN_ATTEMPTS * 3)
    activity = _stuck(working, trail)

    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    state = activity.state
    assert state is not ActivityState.BLOCKED
    assert activity.pending_inference is not None


async def test_the_halt_prompt_names_a_reconsideration_as_such(tmp_path: Path) -> None:
    """A None entry has no defect string to quote, so the mechanical renderer supplies the reason
    rather than printing "None" at whoever has to answer the question. Reached via a trail that
    halts on its *defects* — reconsiderations no longer trip the breaker themselves, but they are
    still part of the story the halt prompt has to tell."""
    cycle, working = await _cycle(tmp_path, reconsideration=NoneReconsideration())
    trail: list[str | None] = [None]
    trail += [f"distinct failure {i}" for i in range(_DEFAULT_MAX_REPLAN_ATTEMPTS)]
    activity = _stuck(working, trail)

    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    prompt = _halt_prompt(activity)
    assert "the world changed under the plan" in prompt
    assert "None" not in prompt


async def test_the_cap_is_configurable(tmp_path: Path) -> None:
    """A task with legitimately many false starts can raise it, exactly like max_subgoal_depth."""
    cycle, working = await _cycle(tmp_path, reconsideration=NoneReconsideration())
    trail: list[str | None] = [f"distinct failure {i}" for i in range(_DEFAULT_MAX_REPLAN_ATTEMPTS)]
    activity = _stuck(working, trail)

    strategy = DefaultReasonStrategy(max_replan_attempts=_DEFAULT_MAX_REPLAN_ATTEMPTS + 1)
    await strategy.reason(activity, working, cycle, TickResult())

    state = activity.state
    assert state is not ActivityState.BLOCKED  # the default would have halted here
    assert activity.pending_inference is not None


async def test_answering_the_halt_lets_the_activity_plan_again(tmp_path: Path) -> None:
    """Without clearing the trail the breaker re-trips on the resumed activity's first Reason pass
    and the halt is permanent — the guidance just given could never be acted on."""
    cycle, working = await _cycle(tmp_path, reconsideration=NoneReconsideration())
    defect = "get_contacts: no such parameter(s) 'limit'"
    activity = _stuck(working, [defect, defect])
    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    blocked = activity.state
    assert blocked is ActivityState.BLOCKED  # precondition

    resumed = DefaultObserveStrategy._resume_on_input(working)

    assert resumed is True
    assert activity.replan_trail == []  # including the resume's own reset_for_replan entry
    ready = activity.state
    assert ready is ActivityState.READY
    # And it really can plan now, rather than halting again on the trail it was carrying.
    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    after = activity.state
    assert after is not ActivityState.BLOCKED
    assert activity.pending_inference is not None
