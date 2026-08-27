"""What a fired pending condition does to the intention stack.

Four coupled rules, all motivated by one ARE run (the Gaia2 "Time" scenario, 2026-08-26) in which
an agent deleted one of four conflicting calendar events and then stopped for good:

* a fired `then` is *committed work*, pursued once the body is idle — never a preemption that
  abandons the steps the plan has not reached yet;
* a **maintenance** sub-goal (ADR-0027) is not finished when its steps run out — those were its
  first iteration — so its frame is held by the conditions *it* declared, and pops only once every
  one of them has retired. An **achievement** sub-goal is unchanged: steps exhausted, frame popped,
  and a contingency condition it declared keeps watching from the activity (ADR-0022);
* the `then` runs at the depth its condition was declared at, and is exempt from the ancestor
  overlap check — a watch's `then` restates the goal that declared it by construction.

The complementary "queue it, drain it in order" rules live in `test_pending_conditions.py`.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from fakes import FakeAdapter, FakeLLMClient, FakeTool, FakeWorkspace
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
from sora.perception import Percept
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReasonStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    Strategies,
    _eligible_conditions,
    _goal_token_overlap,
    _lift_pending_conditions,
)
from sora.types import (
    Change,
    ConditionVerdict,
    ConditionWait,
    PendingCondition,
    PendingConditionState,
    Plan,
    Signal,
    SignalWait,
    Step,
)

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")

_CALENDAR = SignalWait(
    signal_name="state_changed", source="insim:are/Calendar", path="events", kind="added"
)


class _NullTransport:
    async def send(self, to: str, content: dict[str, Any]) -> None:  # pragma: no cover - unused
        raise AssertionError("no message expected")

    def receive(self) -> Any:
        async def _empty() -> Any:
            return
            yield  # pragma: no cover - never reached

        return _empty()


def _cycle(tmp_path: Path, llm: FakeLLMClient | None = None) -> tuple[DecisionCycle, WorkingMemory]:
    tool = FakeTool("insim:are/Calendar")
    registry = EnvironmentRegistry(
        adapters={_ORIGIN: FakeAdapter("fake", FakeWorkspace("ws", _ORIGIN, [tool]))}
    )
    working = WorkingMemory(registry=registry)
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm)
    cycle = DecisionCycle(
        strategies=Strategies(
            observe=DefaultObserveStrategy(),
            reflect=DefaultReflectStrategy(),
            situate=DefaultSituateStrategy(),
            reason=DefaultReasonStrategy(),
            act=DefaultActStrategy(),
        ),
        communication=_NullTransport(),
        actions=default_action_registry(),
        registry=registry,
        working=working,
        semantic=SemanticMemory(FileMemoryBackend(tmp_path / "semantic")),
        procedural=procedural,
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "episodic")),
    )
    return cycle, working


# The wording is the point: `then` is contained in the ancestor sub-goal's goal, because the planner
# is instructed to phrase a `then` "like the original goal". Both were authored by the same model in
# the same response on the motivating run, and scored 0.94 overlap against a 0.7 threshold.
_THEN = "delete every overlapping preexisting calendar event"
_ANCESTOR_GOAL = (
    "Whenever one or more calendar events are added, delete every overlapping preexisting "
    "calendar event and do not notify the user during this monitoring window"
)


def _condition() -> PendingCondition:
    return PendingCondition(
        watch=_CALENDAR,
        when="one or more calendar events are added",
        then=_THEN,
        until="four minutes after the get_current_time result",
    )


_OTHER_THEN = "tell the user the conservator has replied"


def _other() -> PendingCondition:
    """A second, distinct condition — one the TOP-LEVEL plan declares, so lifting has two frames to
    tell apart."""
    return PendingCondition(
        watch=SignalWait(signal_name="state_changed", source="insim:are/Email"),
        when="the conservator replies",
        then=_OTHER_THEN,
        until="the restoration slot has taken place",
    )


def _monitoring_frame(
    goal_kind: str = "maintenance", pending: tuple[PendingCondition, ...] = ()
) -> tuple[Plan, int, int]:
    """A suspended parent frame whose sub-goal step carries the ancestor goal — what the overlap
    check reads — followed by the step that reports completion to the user. `goal_kind` is the
    declaration ADR-0027 reads to decide whether the frame it pushed survives its own steps."""
    parent = Plan(
        id="parent",
        goal="watch the calendar and clear conflicts",
        steps=[
            Step(
                next_action="subgoal",
                params={"goal": _ANCESTOR_GOAL, "mode": "deliberative", "goal_kind": goal_kind},
            ),
            Step("wait", {}),
        ],
        pending=pending,
    )
    return (parent, 0, 0)


def _fanned_out(
    step_index: int,
    goal_kind: str = "maintenance",
    pending: tuple[PendingCondition, ...] = (),
    parent_pending: tuple[PendingCondition, ...] = (),
) -> Activity:
    """The sub-plan mid-fan-out: two delete steps, nested under the monitoring sub-goal."""
    plan = Plan(
        id="sub",
        goal="clear the conflicts",
        steps=[Step("wait", {}), Step("wait", {})],
        pending=pending,
    )
    activity = Activity(id="a1", goal="clear the conflicts", context={}, plan=plan)
    activity.step_index = step_index
    activity.parent_frames.append(_monitoring_frame(goal_kind, parent_pending))
    return activity


def _fired(activity: Activity) -> PendingConditionState:
    """Park a resolved verdict that fired the activity's one condition, as Observe would."""
    state = PendingConditionState(condition=_condition(), evaluated_through=99)
    activity.pending_conditions = [state]
    activity.condition_batch = [state]
    activity.condition_verdict = ConditionVerdict(fired=(0,))
    return state


def _tick() -> Any:
    from sora.strategies import TickResult

    return TickResult()


async def _settle() -> None:
    """Let the spawned off-cycle call run — these drive phases directly, not through the loop."""
    import asyncio

    for _ in range(6):
        await asyncio.sleep(0)


async def test_a_fire_does_not_preempt_an_unfinished_body(tmp_path: Path) -> None:
    """A `then` waits for the body to finish, exactly as an already-queued fire does.

    Pursuing it mid-body abandons every step the plan has not reached: on the motivating run a
    four-step fan-out of deletes had executed one when that very delete opened the gate it was
    watching, and the remaining three never ran.
    """
    llm = FakeLLMClient(json.dumps({"steps": []}))
    cycle, working = _cycle(tmp_path, llm)
    activity = _fanned_out(step_index=0)  # a step still to run
    state = _fired(activity)
    working.activities[activity.id] = activity

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())
    await _settle()

    assert llm.calls == []  # nothing planned while the body still has steps
    assert activity.pending_inference is None
    assert [s.condition for s in activity.condition_fired] == [state.condition]  # queued, not lost


async def test_an_exhausted_maintenance_sub_plan_does_not_pop_past_its_own_condition(
    tmp_path: Path,
) -> None:
    """A maintenance sub-goal's steps were its FIRST ITERATION, not its completion (ADR-0027).

    Popping resumes the parent at the step after the sub-goal — on the motivating run, the message
    telling the user everything was done — while the monitoring window was still open.
    """
    cycle, working = _cycle(tmp_path)
    activity = _fanned_out(step_index=2, goal_kind="maintenance", pending=(_condition(),))
    _lift_pending_conditions(activity, working)
    working.activities[activity.id] = activity

    result = await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert result.step is None
    assert len(activity.parent_frames) == 1  # not popped
    assert activity.state is ActivityState.BLOCKED
    assert isinstance(activity.blocked_on, ConditionWait)


async def test_an_exhausted_achievement_sub_plan_pops_past_its_own_live_condition(
    tmp_path: Path,
) -> None:
    """ADR-0022's contingency case, which the pre-ADR-0027 stopgap had broken: send the mail, pop,
    keep watching. The condition outlives the frame *on the activity* — that is the design, not a
    leak — so the parent resumes at the step after the sub-goal with the watch still live."""
    cycle, working = _cycle(tmp_path)
    activity = _fanned_out(step_index=2, goal_kind="achievement", pending=(_condition(),))
    _lift_pending_conditions(activity, working)
    working.activities[activity.id] = activity

    result = await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert activity.parent_frames == []  # popped
    assert activity.plan is not None and activity.plan.id == "parent"
    assert result.step is not None  # the parent ran the step after the sub-goal
    assert len(activity.pending_conditions) == 1  # still watching, from the activity


async def test_a_sub_goal_with_no_declared_kind_is_an_achievement_goal(tmp_path: Path) -> None:
    """The default is what every sub-goal written before ADR-0027 meant, so nothing migrates and no
    old plan is reinterpreted."""
    cycle, working = _cycle(tmp_path)
    activity = _fanned_out(step_index=2, pending=(_condition(),))
    parent, index, mark = activity.parent_frames[0]
    undeclared = Step(
        next_action="subgoal", params={"goal": _ANCESTOR_GOAL, "mode": "deliberative"}
    )
    activity.parent_frames[0] = (replace(parent, steps=[undeclared, parent.steps[1]]), index, mark)
    _lift_pending_conditions(activity, working)
    working.activities[activity.id] = activity

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert activity.parent_frames == []  # popped, exactly as before the goal kind existed


async def test_a_maintenance_sub_goal_that_declared_nothing_still_ends(tmp_path: Path) -> None:
    """`until` is the only thing that ends a maintenance goal, so one whose plan declared no
    condition ends when its steps do. That is the safe direction for a label nothing verifies — the
    alternative is a frame, and a parent, held open by a goal with no way to finish."""
    cycle, working = _cycle(tmp_path)
    activity = _fanned_out(step_index=2, goal_kind="maintenance")
    _lift_pending_conditions(activity, working)
    working.activities[activity.id] = activity

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert activity.parent_frames == []


async def test_another_frames_condition_does_not_hold_a_maintenance_sub_plan(
    tmp_path: Path,
) -> None:
    """Only the conditions a frame declared ITSELF hold it.

    Lifting deliberately erases which frame declared what, so without attribution a condition
    declared by the TOP-LEVEL plan holds the pop out of an entirely unrelated sub-plan: the
    activity goes BLOCKED with its frames intact and the parent's own remaining steps — the report
    to the user — never run.
    """
    cycle, working = _cycle(tmp_path)
    activity = _fanned_out(step_index=2, goal_kind="maintenance", parent_pending=(_condition(),))
    _lift_pending_conditions(activity, working)
    working.activities[activity.id] = activity
    assert len(activity.pending_conditions) == 1  # lifted off the parent frame, not this one

    result = await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert activity.parent_frames == []  # popped: this frame declared nothing
    assert result.step is not None  # the report to the user, which the stopgap never reached


async def test_a_maintenance_frame_pops_once_its_condition_retires(tmp_path: Path) -> None:
    """The maintenance sub-goal completes when every condition it declared has retired — that is
    what its `until` bounds, and the only thing that ever releases the parent."""
    cycle, working = _cycle(tmp_path)
    activity = _fanned_out(step_index=2, goal_kind="maintenance", pending=(_condition(),))
    _lift_pending_conditions(activity, working)
    working.activities[activity.id] = activity
    activity.condition_batch = list(activity.pending_conditions)
    activity.condition_verdict = ConditionVerdict(retired=(0,))

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())  # applies the verdict
    assert activity.pending_conditions == []
    assert len(activity.parent_frames) == 1  # the pop is Reason's next pass, not the verdict's

    result = await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert activity.parent_frames == []  # window closed -> the parent resumes
    assert result.step is not None


async def test_committed_work_holds_an_achievement_frame_too(tmp_path: Path) -> None:
    """Not a goal-kind question: a queued fire is work the frame already accepted, and the parent
    must not run ahead of it whichever kind of goal declared it."""
    llm = FakeLLMClient(json.dumps({"steps": []}))
    cycle, working = _cycle(tmp_path, llm)
    activity = _fanned_out(step_index=2, goal_kind="achievement", pending=(_condition(),))
    _lift_pending_conditions(activity, working)
    activity.condition_fired = list(activity.pending_conditions)
    working.activities[activity.id] = activity

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())
    await _settle()

    assert len(activity.parent_frames) == 1  # not popped past the queued `then`
    assert activity.pending_inference is not None  # pursuing it instead


async def test_a_then_keeps_the_maintenance_frame_held_after_replacing_the_body(
    tmp_path: Path,
) -> None:
    """A firing replaces the maintenance body with the `then`'s own plan at the same depth — the
    frame is the same frame, so it stays held by the condition that declared it. Attribution that
    named the plan rather than the frame would release it after the first firing, which is the
    motivating run's failure exactly."""
    llm = FakeLLMClient(json.dumps({"steps": []}))
    cycle, working = _cycle(tmp_path, llm)
    activity = _fanned_out(step_index=2, goal_kind="maintenance", pending=(_condition(),))
    _lift_pending_conditions(activity, working)
    working.activities[activity.id] = activity
    activity.condition_batch = list(activity.pending_conditions)
    activity.condition_verdict = ConditionVerdict(fired=(0,))

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())  # pursues the `then`
    await _settle()
    await cycle.strategies.observe.observe(cycle)  # installs the empty `then` plan at this depth
    assert activity.plan is not None and activity.plan.id != "sub"
    await cycle.strategies.reflect.reflect(activity, working, cycle, _tick())  # re-lifts

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert len(activity.parent_frames) == 1  # still monitoring
    assert activity.state is ActivityState.BLOCKED


def test_a_lifted_condition_records_the_frame_that_declared_it(tmp_path: Path) -> None:
    """Attribution is by FRAME, not by plan: a `then` that replaces a frame's body keeps the
    frame's identity, while a sibling sub-goal pushed at another step of the same parent is a
    different frame."""
    working = WorkingMemory(registry=EnvironmentRegistry(adapters={}))
    activity = _fanned_out(step_index=2, pending=(_condition(),), parent_pending=(_other(),))
    _lift_pending_conditions(activity, working)

    parent, index, _mark = activity.parent_frames[0]
    by_condition = {
        state.condition.then: state.declared_by for state in activity.pending_conditions
    }
    assert by_condition[_THEN] == ((parent.id, index),)  # the sub-plan's own frame
    assert by_condition[_OTHER_THEN] == ()  # declared one level up, by the top-level plan


async def test_a_condition_then_is_exempt_from_the_ancestor_overlap_check(tmp_path: Path) -> None:
    """Containment in an ancestor is the *shape* of a `then`, not a failure to reduce.

    The reduction is in the data, not the wording: the `then` is planned against a change that did
    not exist when the ancestor was planned. The depth cap still applies — only this check is
    skipped.
    """
    llm = FakeLLMClient(json.dumps({"steps": []}))
    cycle, working = _cycle(tmp_path, llm)
    activity = _fanned_out(step_index=2)
    state = _fired(activity)
    working.activities[activity.id] = activity
    assert _goal_token_overlap(state.condition.then, _ANCESTOR_GOAL) == 1.0  # would trip

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())
    await _settle()

    assert activity.state is not ActivityState.BLOCKED  # not halted to await input
    assert activity.pending_inference is not None
    assert any(_THEN in prompt for _system, prompt in llm.calls)


async def test_a_condition_then_runs_at_the_same_depth(tmp_path: Path) -> None:
    """No frame is pushed for a `then`. A watch fires as many times as the world moves, and a stack
    that grows once per firing walks a healthy monitor into the depth cap. The body is idle before
    the `then` starts, so there is nothing underneath it to return to."""
    llm = FakeLLMClient(json.dumps({"steps": []}))
    cycle, working = _cycle(tmp_path, llm)
    activity = _fanned_out(step_index=2)
    _fired(activity)
    working.activities[activity.id] = activity

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())
    await _settle()
    await cycle.strategies.observe.observe(cycle)

    assert len(activity.parent_frames) == 1  # the monitoring frame, still the only one
    assert activity.plan is not None
    assert activity.plan.id != "sub"  # the `then`'s own plan replaced the exhausted body
    assert activity.step_index == 0


# --------------------------------------------------------------------------------------------------
# The gate the agent's own write must not open
# --------------------------------------------------------------------------------------------------


def _observed(working: WorkingMemory, change: Change) -> None:
    working.signals.append(
        Percept("insim:are/Calendar", Signal("state_changed", {"changes": [change]}), 0.0)
    )
    working.signals_appended += 1


async def test_the_agents_own_delete_does_not_open_the_gate_it_watches(tmp_path: Path) -> None:
    """The cost control, end to end. Watching `events` for additions while deleting from `events`
    puts the agent's own write on the exact path it watches, so `path` alone cannot exclude it: on
    the motivating run each delete bought a judge call, ~138 cycles of latency, and a verdict that a
    removal is not an addition — all while the rest of the fan-out sat blocked behind it."""
    cycle, working = _cycle(tmp_path)
    activity = _fanned_out(step_index=2)
    activity.pending_conditions = [
        PendingConditionState(condition=_condition(), evaluated_through=0)
    ]
    working.activities[activity.id] = activity
    _observed(working, Change(path="events", removed=("075c5ad9",)))  # the agent's own delete

    assert _eligible_conditions(activity, working) == []

    _observed(working, Change(path="events", added=("3fa4a347",)))  # the world's addition

    assert len(_eligible_conditions(activity, working)) == 1
