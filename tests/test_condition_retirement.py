"""Garbage collection of pending conditions whose watch has gone quiet (ADR-0027 §4).

`until` used to be judged in exactly one place: the batched condition evaluation, which runs only
when an observed change makes a condition *eligible*. So a condition nothing ever moves against was
never re-judged, never retired, and its `ConditionWait` held the activity BLOCKED for good —
reachable for a contingency condition, load-bearing for a maintenance sub-goal, whose frame lives
exactly as long as its `until`.

The sweep closes that. It is **retire-only** (a firing pass would duplicate the eligibility gate's
job and reopen the cost question that gate settled), **idle-scheduled** (ADR-0026's cadence — it
never runs on a tick where some activity could advance), and split the way ADR-0027 names it:
**Observe retires, Reason pops.**
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fakes import FakeAdapter, FakeLLMClient, FakeTool, FakeWorkspace
from sora.action import default_action_registry
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, WorkspaceOrigin
from sora.memory import (
    RETIREMENT_SYSTEM_PROMPT,
    EpisodicMemory,
    FileMemoryBackend,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReasonStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    Strategies,
    TickResult,
    _lift_pending_conditions,
)
from sora.types import (
    ConditionWait,
    PendingCondition,
    PendingConditionState,
    Plan,
    SignalWait,
    Step,
    Until,
)

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")

_CALENDAR = SignalWait(
    signal_name="state_changed", source="insim:are/Calendar", path="events", kind="added"
)
_EMAIL = SignalWait(signal_name="state_changed", source="insim:are/Email")


class _NullTransport:
    async def send(self, to: str, content: dict[str, Any]) -> None:  # pragma: no cover - unused
        raise AssertionError("no message expected")

    def receive(self) -> Any:
        async def _empty() -> Any:
            return
            yield  # pragma: no cover - never reached

        return _empty()


def _cycle(
    tmp_path: Path,
    llm: FakeLLMClient | None = None,
    *,
    observe: DefaultObserveStrategy | None = None,
) -> tuple[DecisionCycle, WorkingMemory]:
    tool = FakeTool("insim:are/Calendar")
    registry = EnvironmentRegistry(
        adapters={_ORIGIN: FakeAdapter("fake", FakeWorkspace("ws", _ORIGIN, [tool]))}
    )
    working = WorkingMemory(registry=registry)
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm)
    cycle = DecisionCycle(
        strategies=Strategies(
            observe=observe or DefaultObserveStrategy(retirement_interval=0.0),
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


def _monitoring() -> PendingCondition:
    return PendingCondition(
        watch=_CALENDAR,
        when="one or more calendar events are added",
        then="delete every overlapping preexisting calendar event",
        until=Until(text="four minutes after the get_current_time result"),
    )


def _contingency() -> PendingCondition:
    return PendingCondition(
        watch=_EMAIL,
        when="the conservator replies",
        then="tell the user the conservator has replied",
        until=Until(text="the restoration slot has taken place"),
    )


def _blocked_on(activity: Activity, *conditions: PendingCondition) -> Activity:
    """An activity parked exactly where Reflect/Reason park one whose body is spent: BLOCKED on the
    watches of its live conditions, with nothing arriving to reopen a gate."""
    activity.pending_conditions = [
        PendingConditionState(condition=c, evaluated_through=0) for c in conditions
    ]
    activity.state = ActivityState.BLOCKED
    activity.blocked_on = ConditionWait(watches=tuple(c.watch for c in conditions))
    return activity


def _maintenance(pending: tuple[PendingCondition, ...]) -> Activity:
    """A spent maintenance sub-plan: its steps ran (they were the first iteration), its frame is
    held open by the condition it declared, and the parent still owes the user a report."""
    parent = Plan(
        id="parent",
        goal="watch the calendar and clear conflicts",
        steps=[
            Step(
                next_action="subgoal",
                params={"goal": "clear conflicts as they appear", "goal_kind": "maintenance"},
            ),
            Step("wait", {}),
        ],
    )
    plan = Plan(id="sub", goal="clear the conflicts", steps=[Step("wait", {})], pending=pending)
    activity = Activity(id="a1", goal="clear the conflicts", context={}, plan=plan)
    activity.step_index = 1
    activity.parent_frames.append((parent, 0, 0))
    return activity


async def _settle() -> None:
    """Let the spawned off-cycle judgement run — these drive Observe directly, not via a loop."""
    for _ in range(6):
        await asyncio.sleep(0)


def _verdict(**kwargs: Any) -> str:
    return json.dumps(kwargs)


async def _sweep(cycle: DecisionCycle) -> None:
    """Two Observes: the first fires the off-cycle judgement, the second applies what it parked.
    The same discipline every off-cycle call in the runtime follows — the background task only ever
    sets a field, and working memory is mutated on-cycle."""
    await cycle.strategies.observe.observe(cycle)
    await _settle()
    await cycle.strategies.observe.observe(cycle)


# ------------------------------------------------------------------------------------------------
# The sweep itself
# ------------------------------------------------------------------------------------------------


async def test_a_quiet_condition_is_retired(tmp_path: Path) -> None:
    """The gap this closes: nothing ever moves on the watched collection, so the eligibility gate
    never opens and the batched evaluation — the only thing that ever read `until` — never runs."""
    llm = FakeLLMClient(_verdict(retired=[0]))
    cycle, working = _cycle(tmp_path, llm)
    activity = _blocked_on(Activity(id="a1", goal="clear the conflicts", context={}), _monitoring())
    working.activities[activity.id] = activity

    await _sweep(cycle)

    assert activity.pending_conditions == []
    assert llm.calls and llm.calls[0][0] == RETIREMENT_SYSTEM_PROMPT


async def test_a_retired_condition_is_remembered_so_the_next_lift_leaves_it_retired(
    tmp_path: Path,
) -> None:
    """`Plan.pending` is a frozen skeleton that still declares what retirement removed, so without
    the record the very next Reflect would put the condition straight back on watch, for good."""
    llm = FakeLLMClient(_verdict(retired=[0]))
    condition = _monitoring()
    cycle, working = _cycle(tmp_path, llm)
    activity = _maintenance((condition,))
    _lift_pending_conditions(activity, working)
    activity.state = ActivityState.BLOCKED
    activity.blocked_on = ConditionWait(watches=(condition.watch,))
    working.activities[activity.id] = activity

    await _sweep(cycle)

    assert condition in activity.retired_conditions
    _lift_pending_conditions(activity, working)
    assert activity.pending_conditions == []


async def test_the_sweep_never_fires_a_condition(tmp_path: Path) -> None:
    """Retire only. A firing pass would duplicate the eligibility gate's job and reopen the cost
    question that gate settled — so a `fired` coming back from this call is ignored, not queued."""
    llm = FakeLLMClient(_verdict(fired=[0], retired=[]))
    cycle, working = _cycle(tmp_path, llm)
    activity = _blocked_on(Activity(id="a1", goal="clear the conflicts", context={}), _monitoring())
    working.activities[activity.id] = activity

    await _sweep(cycle)

    assert activity.condition_fired == []
    assert activity.condition_verdict is None
    assert len(activity.pending_conditions) == 1  # still waiting
    assert activity.state is ActivityState.BLOCKED


async def test_a_partly_retired_activity_keeps_waiting_on_what_is_left(tmp_path: Path) -> None:
    """One of two retires: the activity stays BLOCKED, and its `blocked_on` is re-derived so the
    wait describes what it is actually still waiting for — that is what a diagnostic renders."""
    llm = FakeLLMClient(_verdict(retired=[0]))
    cycle, working = _cycle(tmp_path, llm)
    activity = _blocked_on(
        Activity(id="a1", goal="clear the conflicts", context={}), _monitoring(), _contingency()
    )
    working.activities[activity.id] = activity

    await _sweep(cycle)

    assert [s.condition for s in activity.pending_conditions] == [_contingency()]
    assert activity.state is ActivityState.BLOCKED
    assert activity.blocked_on == ConditionWait(watches=(_EMAIL,))


async def test_a_failed_judgement_leaves_the_condition_waiting(tmp_path: Path) -> None:
    """Degrade to the state the activity was already in. The opposite default would let a flaky
    call retire a live commitment, which is unrecoverable — the declaration is gone from watch."""
    llm = FakeLLMClient("not json at all")
    cycle, working = _cycle(tmp_path, llm)
    activity = _blocked_on(Activity(id="a1", goal="clear the conflicts", context={}), _monitoring())
    working.activities[activity.id] = activity

    await _sweep(cycle)

    assert len(activity.pending_conditions) == 1
    assert activity.state is ActivityState.BLOCKED


# ------------------------------------------------------------------------------------------------
# Observe retires, Reason pops
# ------------------------------------------------------------------------------------------------


async def test_retirement_releases_the_maintenance_frame_it_was_holding(tmp_path: Path) -> None:
    """The load-bearing case. A maintenance frame lives exactly as long as its `until`, so nothing
    but retirement ever releases it — and the activity has to come back to READY for Reason to be
    given the chance to pop, since Situate only ever selects a READY activity."""
    llm = FakeLLMClient(_verdict(retired=[0]))
    cycle, working = _cycle(tmp_path, llm)
    activity = _maintenance((_monitoring(),))
    _lift_pending_conditions(activity, working)
    activity.state = ActivityState.BLOCKED
    activity.blocked_on = ConditionWait(watches=(_CALENDAR,))
    working.activities[activity.id] = activity

    await _sweep(cycle)

    assert activity.pending_conditions == []
    assert activity.state is ActivityState.READY  # Observe retired and resumed
    assert activity.blocked_on is None
    assert len(activity.parent_frames) == 1  # the pop is Reason's, not Observe's

    result = await cycle.strategies.reason.reason(activity, working, cycle, TickResult())

    assert activity.parent_frames == []  # window closed -> the parent resumes
    assert result.step is not None


async def test_a_spent_top_level_activity_finishes_once_its_last_contingency_retires(
    tmp_path: Path,
) -> None:
    """No frame to pop: the body is done and the only thing keeping the activity alive was the
    contingency. Retiring it is what lets Reflect finally record the episode."""
    llm = FakeLLMClient(_verdict(retired=[0]))
    cycle, working = _cycle(tmp_path, llm)
    plan = Plan(id="p", goal="mail the conservator", steps=[Step("wait", {})])
    activity = Activity(id="a1", goal="mail the conservator", context={}, plan=plan)
    activity.step_index = 1
    _blocked_on(activity, _contingency())
    working.activities[activity.id] = activity

    await _sweep(cycle)

    resumed = activity.state  # a local, so the assert does not narrow the attribute for mypy
    assert resumed is ActivityState.READY
    await cycle.strategies.reflect.reflect(activity, working, cycle, TickResult())

    assert activity.state is ActivityState.TERMINATED


# ------------------------------------------------------------------------------------------------
# Scheduling: out of slack, never off the critical path
# ------------------------------------------------------------------------------------------------


async def test_the_sweep_stands_down_while_an_activity_could_advance(tmp_path: Path) -> None:
    """ADR-0026's cadence: eligible the moment a condition goes quiet, but it never runs in
    preference to an activity that could actually advance. A READY activity is exactly the tick
    where Situate will select something, so the sweep waits."""
    llm = FakeLLMClient(_verdict(retired=[0]))
    cycle, working = _cycle(tmp_path, llm)
    blocked = _blocked_on(Activity(id="a1", goal="clear the conflicts", context={}), _monitoring())
    working.activities[blocked.id] = blocked
    busy = Activity(id="a2", goal="book the studio", context={})
    working.activities[busy.id] = busy  # READY by default

    await _sweep(cycle)

    assert llm.calls == []
    assert len(blocked.pending_conditions) == 1


async def test_the_sweep_is_paced(tmp_path: Path) -> None:
    """A blocked maintenance activity makes *every* tick idle, so an unpaced sweep would spend a
    model call per tick for as long as the window is open. The interval is what bounds that."""
    llm = FakeLLMClient(_verdict(retired=[]))
    cycle, working = _cycle(
        tmp_path, llm, observe=DefaultObserveStrategy(retirement_interval=3600.0)
    )
    activity = _blocked_on(Activity(id="a1", goal="clear the conflicts", context={}), _monitoring())
    working.activities[activity.id] = activity

    for _ in range(4):
        await cycle.strategies.observe.observe(cycle)
        await _settle()

    assert len(llm.calls) == 1


async def test_the_sweep_can_be_switched_off(tmp_path: Path) -> None:
    """`None` disables it, the way `inference_deadline=None` disables the watchdog — for a run that
    wants no unattended model spend at all."""
    llm = FakeLLMClient(_verdict(retired=[0]))
    cycle, working = _cycle(tmp_path, llm, observe=DefaultObserveStrategy(retirement_interval=None))
    activity = _blocked_on(Activity(id="a1", goal="clear the conflicts", context={}), _monitoring())
    working.activities[activity.id] = activity

    await _sweep(cycle)

    assert llm.calls == []
    assert len(activity.pending_conditions) == 1


async def test_one_judgement_is_in_flight_at_a_time(tmp_path: Path) -> None:
    """Two blocked activities, one call: the sweep takes one activity per pass. Its whole cost
    argument is that it comes out of slack, and N calls fired on one idle tick is not slack."""
    llm = FakeLLMClient(_verdict(retired=[0]))
    cycle, working = _cycle(tmp_path, llm)
    first = _blocked_on(Activity(id="a1", goal="clear the conflicts", context={}), _monitoring())
    second = _blocked_on(Activity(id="a2", goal="mail the conservator", context={}), _contingency())
    working.activities[first.id] = first
    working.activities[second.id] = second

    await cycle.strategies.observe.observe(cycle)
    await _settle()

    assert len(llm.calls) == 1


async def test_a_terminated_activitys_conditions_are_not_swept(tmp_path: Path) -> None:
    """A terminated activity's watch set is already gone as far as the runtime is concerned; paying
    a judgement to tidy it would spend on a question nothing can act on."""
    llm = FakeLLMClient(_verdict(retired=[0]))
    cycle, working = _cycle(tmp_path, llm)
    activity = _blocked_on(Activity(id="a1", goal="clear the conflicts", context={}), _monitoring())
    activity.state = ActivityState.TERMINATED
    working.activities[activity.id] = activity

    await _sweep(cycle)

    assert llm.calls == []
