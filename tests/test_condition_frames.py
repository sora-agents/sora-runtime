"""What a fired pending condition does to the intention stack.

Three coupled rules, all motivated by one ARE run (the Gaia2 "Time" scenario, 2026-08-26) in which
an agent deleted one of four conflicting calendar events and then stopped for good:

* a fired `then` is *committed work*, pursued once the body is idle — never a preemption that
  abandons the steps the plan has not reached yet;
* an exhausted sub-plan does not pop past a frame whose conditions are still live, because the
  `until` outlives the steps;
* the `then` runs at the depth its condition was declared at, and is exempt from the ancestor
  overlap check — a watch's `then` restates the goal that declared it by construction.

The complementary "queue it, drain it in order" rules live in `test_pending_conditions.py`.
"""

from __future__ import annotations

import json
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
)
from sora.types import (
    Change,
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


def _monitoring_frame() -> tuple[Plan, int, int]:
    """A suspended parent frame whose sub-goal step carries the ancestor goal — what the overlap
    check reads — followed by the step that reports completion to the user."""
    parent = Plan(
        id="parent",
        goal="watch the calendar and clear conflicts",
        steps=[
            Step(next_action="subgoal", params={"goal": _ANCESTOR_GOAL, "mode": "deliberative"}),
            Step("wait", {}),
        ],
    )
    return (parent, 0, 0)


def _fanned_out(step_index: int) -> Activity:
    """The sub-plan mid-fan-out: two delete steps, nested under the monitoring sub-goal."""
    plan = Plan(id="sub", goal="clear the conflicts", steps=[Step("wait", {}), Step("wait", {})])
    activity = Activity(id="a1", goal="clear the conflicts", context={}, plan=plan)
    activity.step_index = step_index
    activity.parent_frames.append(_monitoring_frame())
    return activity


def _fired(activity: Activity) -> PendingConditionState:
    """Park a resolved verdict that fired the activity's one condition, as Observe would."""
    from sora.types import ConditionVerdict

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


async def test_an_exhausted_sub_plan_does_not_pop_past_a_live_condition(tmp_path: Path) -> None:
    """The condition is declared by the sub-plan, but its `until` outlives the sub-plan's steps.

    Popping resumes the parent at the step after the sub-goal — on the motivating run, the message
    telling the user everything was done — while the monitoring window was still open.
    """
    cycle, working = _cycle(tmp_path)
    activity = _fanned_out(step_index=2)  # every step has run
    activity.pending_conditions = [
        PendingConditionState(condition=_condition(), evaluated_through=99)
    ]
    working.activities[activity.id] = activity

    result = await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert result.step is None
    assert len(activity.parent_frames) == 1  # not popped
    assert activity.state is ActivityState.BLOCKED
    assert isinstance(activity.blocked_on, ConditionWait)


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
