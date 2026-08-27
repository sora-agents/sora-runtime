"""Change events derived from the property snapshot — the recovery path when a signal never lands.

The failure these exist for: on a benchmark run six calendar events were added while the agent
watched for them, and only four `state_changed` signals ever reached working memory. The agent
acted correctly on all four and scored 4/5, because the two it never heard about were, by then,
plainly visible in its own property snapshot. Nothing read them: a pending condition's gate opened
only on a signal, and a signal is transient — dropped once, it is gone, and no later snapshot can
reconstruct it.

So the runtime derives the change itself, by diffing the re-observed property against the value it
last saw. This is the same move AgentSpeak's belief revision makes (percepts are diffed against the
belief base to produce belief-change events); the adapter's own signal remains the fast path, and
this is what makes losing it survivable rather than scoring.

Two things are pinned hardest here, because both are easy to break silently. First the cost
control: a change the adapter *did* signal must not also be derived, or every condition is judged
twice and the judge bill doubles for no new information. Second the direction of the safety net —
it must never *miss*, so a derived change matches wider than a signal does (it has no signal name to
match on), and that widening is deliberate.
"""

from __future__ import annotations

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
)
from sora.types import (
    Change,
    ConditionWait,
    ObservableProperty,
    PendingCondition,
    PendingConditionState,
    Plan,
    Signal,
    SignalWait,
    Step,
)

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")
_TOOL = "insim:are/Calendar"


class _NullTransport:
    async def send(self, to: str, content: dict[str, Any]) -> None: ...

    async def receive(self) -> Any:
        return
        yield  # pragma: no cover — never reached; makes this an async generator


async def _cycle(tmp_path: Path, llm: Any = None) -> tuple[DecisionCycle, WorkingMemory, FakeTool]:
    tool = FakeTool(_TOOL)
    registry = EnvironmentRegistry(
        adapters={_ORIGIN: FakeAdapter("fake", FakeWorkspace("ws", _ORIGIN, [tool]))}
    )
    working = WorkingMemory(registry=registry)
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
        procedural=ProceduralMemory(FileMemoryBackend(tmp_path / "procedural"), llm=llm),
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "episodic")),
    )
    await registry.join(_ORIGIN)  # the tool has to be joined before Observe can attend it
    return cycle, working, tool


def _events(*ids: str) -> ObservableProperty:
    """The shape an app-state property takes: a nested, identity-keyed collection."""
    return ObservableProperty(name="state", value={"events": {i: {"title": i} for i in ids}})


def _tick() -> Any:
    from sora.strategies import TickResult

    return TickResult()


def _derived_paths(working: WorkingMemory) -> list[str]:
    return [c.path for percept in working.property_changes for c in percept.payload.changes]


_WATCH = SignalWait(signal_name="state_changed", source=_TOOL, path="events", kind="added")


def _blocked_on(working: WorkingMemory, watch: SignalWait = _WATCH) -> Activity:
    condition = PendingCondition(
        watch=watch,
        when="one or more events are added",
        then="Resolve the conflict",
        until="the four minutes are up",
    )
    activity = Activity(
        id="a1",
        goal="watch the calendar",
        context={},
        plan=Plan(id="p1", goal="watch the calendar", steps=[Step("wait", {})]),
        step_index=1,
    )
    activity.state = ActivityState.BLOCKED
    activity.pending_conditions = [PendingConditionState(condition=condition)]
    activity.blocked_on = ConditionWait(watches=(watch,))
    working.activities[activity.id] = activity
    return activity


# --------------------------------------------------------------------------------------------------
# Deriving the change
# --------------------------------------------------------------------------------------------------


async def test_first_sighting_of_a_property_reports_no_change(tmp_path: Path) -> None:
    # A newly attended tool has not *changed*; it has only just become visible. Reporting its whole
    # value as an addition would open every gate watching it the moment attention reached it.
    cycle, working, tool = await _cycle(tmp_path)
    tool.set_properties([_events("e1", "e2")])

    await cycle.strategies.observe.observe(cycle)

    assert working.property_changes == []


async def test_a_property_that_moved_without_a_signal_produces_a_change(tmp_path: Path) -> None:
    cycle, working, tool = await _cycle(tmp_path)
    tool.set_properties([_events("e1")])
    await cycle.strategies.observe.observe(cycle)  # baseline

    tool.set_properties([_events("e1", "e2")])
    await cycle.strategies.observe.observe(cycle)

    assert len(working.property_changes) == 1
    change = working.property_changes[0]
    assert change.source == _TOOL
    assert change.payload.name == "state"
    assert [(c.path, c.added) for c in change.payload.changes] == [("events", ("e2",))]


async def test_a_property_that_did_not_move_produces_nothing(tmp_path: Path) -> None:
    cycle, working, tool = await _cycle(tmp_path)
    tool.set_properties([_events("e1")])
    await cycle.strategies.observe.observe(cycle)
    await cycle.strategies.observe.observe(cycle)

    assert working.property_changes == []


async def test_a_change_the_adapter_already_signalled_is_not_derived_again(tmp_path: Path) -> None:
    # THE cost control. The adapter refreshes the property and pushes the signal in one breath, so
    # both land in the same Observe. Deriving a twin would double every condition's judge calls
    # while telling it nothing it did not already know.
    cycle, working, tool = await _cycle(tmp_path)
    tool.set_properties([_events("e1")])
    await cycle.strategies.observe.observe(cycle)

    tool.set_properties([_events("e1", "e2")])
    cycle.signal_sink.push(
        _TOOL, Signal("state_changed", {"changes": [Change(path="events", added=("e2",))]})
    )
    await cycle.strategies.observe.observe(cycle)

    assert len(working.signals) == 1
    assert working.property_changes == []


async def test_a_signal_on_another_path_does_not_suppress_the_derivation(tmp_path: Path) -> None:
    # Dedup is per-path, not per-source: an adapter that reported one change and dropped another in
    # the same tick must still have the dropped one recovered.
    cycle, working, tool = await _cycle(tmp_path)
    tool.set_properties([ObservableProperty(name="state", value={"events": {}, "tasks": {}})])
    await cycle.strategies.observe.observe(cycle)

    tool.set_properties(
        [ObservableProperty(name="state", value={"events": {"e2": {}}, "tasks": {"t1": {}}})]
    )
    cycle.signal_sink.push(
        _TOOL, Signal("state_changed", {"changes": [Change(path="tasks", added=("t1",))]})
    )
    await cycle.strategies.observe.observe(cycle)

    assert _derived_paths(working) == ["events"]


async def test_the_baseline_survives_an_unfocus_so_changes_while_away_are_reported(
    tmp_path: Path,
) -> None:
    # The blind window this also closes: dropping the property with the tool means a later
    # re-observation compares against nothing, folds everything that happened in between into a
    # fresh baseline, and reports silence. Inert while the default policy attends everything;
    # live the moment attention narrows.
    cycle, working, tool = await _cycle(tmp_path)
    tool.set_properties([_events("e1")])
    await cycle.strategies.observe.observe(cycle)

    working.drop_properties(lambda source: False)  # _unfocus_/_filter_ prune the snapshot
    tool.set_properties([_events("e1", "e2")])
    await cycle.strategies.observe.observe(cycle)

    assert [(c.path, c.added) for c in working.property_changes[0].payload.changes] == [
        ("events", ("e2",))
    ]


async def test_derived_changes_are_bounded_by_a_retention_cap(tmp_path: Path) -> None:
    cycle, working, tool = await _cycle(tmp_path)
    from sora.strategies import _DERIVED_RETENTION, _SIGNAL_RETENTION

    # Sized against a different window than signals are (a derived change fills at tick rate, not
    # event rate), so a shared constant would be the wrong bound -- pin that they stay distinct.
    assert _DERIVED_RETENTION > _SIGNAL_RETENTION

    # One id swapped per cycle rather than a growing list: same one-append-per-cycle fill, without
    # making the diff itself grow with the cap.
    for n in range(_DERIVED_RETENTION + 20):
        tool.set_properties([_events(f"e{n}")])
        await cycle.strategies.observe.observe(cycle)

    assert len(working.property_changes) == _DERIVED_RETENTION


# --------------------------------------------------------------------------------------------------
# What a derived change is allowed to open
# --------------------------------------------------------------------------------------------------


async def test_a_derived_change_opens_a_conditions_gate(tmp_path: Path) -> None:
    # The whole point: the signal never came, and the condition still gets its chance to fire.
    cycle, working, tool = await _cycle(tmp_path)
    tool.set_properties([_events("e1")])
    await cycle.strategies.observe.observe(cycle)
    activity = _blocked_on(working)

    tool.set_properties([_events("e1", "e2")])
    await cycle.strategies.observe.observe(cycle)

    assert activity.state is ActivityState.READY
    assert activity.blocked_on is None


async def test_a_derived_change_on_another_path_does_not_open_the_gate(tmp_path: Path) -> None:
    cycle, working, tool = await _cycle(tmp_path)
    tool.set_properties([ObservableProperty(name="state", value={"events": {}, "tasks": {}})])
    await cycle.strategies.observe.observe(cycle)
    activity = _blocked_on(working)

    tool.set_properties(
        [ObservableProperty(name="state", value={"events": {}, "tasks": {"t1": {}}})]
    )
    await cycle.strategies.observe.observe(cycle)

    assert activity.state is ActivityState.BLOCKED


async def test_a_derived_change_in_the_wrong_direction_does_not_open_the_gate(
    tmp_path: Path,
) -> None:
    # The agent's own delete lands on the exact watched path; `kind` is what separates it from the
    # world's addition, and it has to keep separating them on the derived path too.
    cycle, working, tool = await _cycle(tmp_path)
    tool.set_properties([_events("e1", "e2")])
    await cycle.strategies.observe.observe(cycle)
    activity = _blocked_on(working)

    tool.set_properties([_events("e1")])
    await cycle.strategies.observe.observe(cycle)

    assert activity.state is ActivityState.BLOCKED


async def test_a_derived_change_is_not_judged_twice(tmp_path: Path) -> None:
    # Same livelock the signal path has its own mark for: resume, find nothing new, re-block, spin —
    # once per cycle for as long as the change stays in retention.
    cycle, working, tool = await _cycle(tmp_path)
    tool.set_properties([_events("e1")])
    await cycle.strategies.observe.observe(cycle)
    activity = _blocked_on(working)

    tool.set_properties([_events("e1", "e2")])
    await cycle.strategies.observe.observe(cycle)
    activity.state = ActivityState.BLOCKED  # as if Reason judged it and nothing held
    activity.blocked_on = ConditionWait(watches=(_WATCH,))
    activity.pending_conditions[0].derived_through = working.property_changes_appended

    await cycle.strategies.observe.observe(cycle)

    assert activity.state is ActivityState.BLOCKED


async def test_a_new_condition_ignores_the_derived_backlog(tmp_path: Path) -> None:
    # A condition declared now cannot be about a change from before it existed — the same reasoning
    # that starts `evaluated_through` at the signal high-water mark, applied to the second log.
    cycle, working, tool = await _cycle(tmp_path)
    tool.set_properties([_events("e1")])
    await cycle.strategies.observe.observe(cycle)
    tool.set_properties([_events("e1", "e2")])
    await cycle.strategies.observe.observe(cycle)  # change arrives BEFORE the plan exists

    activity = Activity(
        id="a2",
        goal="g",
        context={},
        plan=Plan(
            id="p2",
            goal="g",
            steps=[Step("wait", {})],
            pending=(
                PendingCondition(watch=_WATCH, when="events added", then="handle", until="done"),
            ),
        ),
        step_index=1,
    )
    working.activities[activity.id] = activity
    await cycle.strategies.reflect.reflect(activity, working, cycle, _tick())

    assert activity.pending_conditions[0].derived_through == working.property_changes_appended


async def test_a_derived_change_does_not_resume_a_completion_wait(tmp_path: Path) -> None:
    # Scoped deliberately: a condition that fires spuriously costs one judge call and is told no,
    # while a completion wait that resumes spuriously lets an activity proceed as though a
    # long-running operation had finished. Extending to blocked_on is a separate, later step.
    cycle, working, tool = await _cycle(tmp_path)
    tool.set_properties([_events("e1")])
    await cycle.strategies.observe.observe(cycle)

    activity = Activity(
        id="a3",
        goal="g",
        context={},
        plan=Plan(id="p3", goal="g", steps=[Step("wait", {})]),
        step_index=0,
    )
    activity.state = ActivityState.BLOCKED
    activity.blocked_on = SignalWait(signal_name="state_changed", source=_TOOL, path="events")
    working.activities[activity.id] = activity

    tool.set_properties([_events("e1", "e2")])
    await cycle.strategies.observe.observe(cycle)

    assert activity.state is ActivityState.BLOCKED
    assert isinstance(activity.blocked_on, SignalWait)


async def test_a_derived_change_is_not_stored_as_a_signal(tmp_path: Path) -> None:
    # The two logs stay distinct (ADR-0004): `signals` means the environment announced something.
    # Anything rendering working memory must not start showing derived events as observed ones.
    cycle, working, tool = await _cycle(tmp_path)
    tool.set_properties([_events("e1")])
    await cycle.strategies.observe.observe(cycle)

    tool.set_properties([_events("e1", "e2")])
    await cycle.strategies.observe.observe(cycle)

    assert working.signals == []
    assert working.signals_appended == 0
    assert len(working.property_changes) == 1


def test_a_derived_change_is_not_a_percept_of_the_signal_type() -> None:
    from sora.types import PropertyChange

    change = PropertyChange(name="state", changes=(Change(path="events", added=("e2",)),))
    percept = Percept(_TOOL, change, 0.0)

    assert not isinstance(percept.payload, Signal)
    assert percept.payload.changes[0].added == ("e2",)


async def test_a_derived_change_reaches_the_condition_judge(tmp_path: Path) -> None:
    # Opening the gate is only half of it: the judgement itself has to be able to read what moved.
    # The two logs carry their changes on differently-shaped payloads, and a matcher that resumes an
    # activity onto a judge that cannot parse the percept has moved the failure, not fixed it.
    import json

    llm = FakeLLMClient(json.dumps({"fired": [], "retired": []}))
    cycle, working, tool = await _cycle(tmp_path, llm=llm)
    tool.set_properties([_events("e1")])
    await cycle.strategies.observe.observe(cycle)
    activity = _blocked_on(working)
    activity.state = ActivityState.READY
    activity.blocked_on = None

    tool.set_properties([_events("e1", "e2")])
    await cycle.strategies.observe.observe(cycle)
    await cycle.strategies.reason.reason(activity, working, cycle, _tick())
    await _settle()

    assert len(llm.calls) == 1
    assert "e2" in llm.calls[0][1]


async def _settle() -> None:
    """Let the off-cycle judge call run: these drive phases directly, with no tick to resolve
    it."""
    import asyncio

    for _ in range(6):
        await asyncio.sleep(0)


# --------------------------------------------------------------------------------------------------
# The diagnostic
# --------------------------------------------------------------------------------------------------


async def test_observation_heartbeat_reports_per_tool_counts_and_rate(
    tmp_path: Path, caplog: Any
) -> None:
    # What the run log could not answer: 229 cycles covered the first 91 seconds against 11.6/s
    # afterwards, and nothing recorded whether those cycles were spread across the window or bunched
    # at the end of it — the difference between "the agent was never looking" and "it looked and the
    # change was dropped". Counts per tool plus elapsed wall time say which, without a guess.
    import logging

    from sora.strategies import _OBSERVATION_HEARTBEAT

    cycle, working, tool = await _cycle(tmp_path)
    tool.set_properties([_events("e1")])
    with caplog.at_level(logging.DEBUG, logger="sora.strategies"):
        for _ in range(_OBSERVATION_HEARTBEAT):
            await cycle.strategies.observe.observe(cycle)

    beats = [r.getMessage() for r in caplog.records if "cycles in" in r.getMessage()]
    assert len(beats) == 1
    assert f"{_TOOL}={_OBSERVATION_HEARTBEAT}" in beats[0]


async def test_the_heartbeat_stays_silent_below_debug(tmp_path: Path, caplog: Any) -> None:
    # It runs every cycle at 11/s; the counting itself is skipped unless someone is listening.
    import logging

    from sora.strategies import _OBSERVATION_HEARTBEAT

    cycle, working, tool = await _cycle(tmp_path)
    tool.set_properties([_events("e1")])
    with caplog.at_level(logging.INFO, logger="sora.strategies"):
        for _ in range(_OBSERVATION_HEARTBEAT + 1):
            await cycle.strategies.observe.observe(cycle)

    assert [r for r in caplog.records if "cycles in" in r.getMessage()] == []
