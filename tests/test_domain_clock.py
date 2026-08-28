"""The domain clock an `until` is answered against (ADR-0027 §5 and §6).

Every `time.time()` in the runtime is *host* wall-clock and correct as infrastructure timing — a
percept's `observed_at`, the inference watchdog, how often the retirement sweep asks. Domain time
is a different clock: under a simulation it starts elsewhere and can run at a different rate, and
merging the two is a silent wrong answer rather than an error (ARE once told an agent it was
1 Jan 1970 through a scenario set in October 2024).

So domain time reaches the runtime through a `DomainClock` on the **workspace** — per workspace,
because a simulated clock's rate differs and two workspaces may legitimately disagree — and never
through `time.time()`. Two consumers here:

* **retirement** — a time-bounded `until` is resolved mechanically against that clock, ahead of
  (and instead of) the retirement judge, at no model cost;
* **plan validation** — a maintenance sub-goal whose `until` asks about time, watching a workspace
  that cannot tell it, could never terminate, so it is refused with a named defect and replanned.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fakes import FakeAdapter, FakeLLMClient, FakeTool, FakeWorkspace
from sora.action import default_action_registry
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import DomainClock, EnvironmentRegistry, HostClock, WorkspaceOrigin
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
    pending_from_raw,
)
from sora.perception import Message
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReasonStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    Strategies,
    TickResult,
    _clock_for_source,
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
    signal_name="state_changed", source="realestate", path="events", kind="added"
)
# The condition as the planner writes it, minus the `until` each test varies.
_RAW = {
    "watch": {"signal": "state_changed", "source": "realestate", "path": "events", "kind": "added"},
    "when": "one or more calendar events are added",
    "then": "delete every overlapping preexisting calendar event",
}
_FOUR_MINUTES = Until(text="four minutes after the get_current_time result", seconds=240.0)


def _monitoring(until: Until | None = _FOUR_MINUTES) -> PendingCondition:
    return PendingCondition(
        watch=_CALENDAR,
        when="one or more calendar events are added",
        then="delete every overlapping preexisting calendar event",
        until=until,
    )


class FakeClock:
    """A domain clock a test can move independently of host wall-clock — which is the whole point:
    a run whose domain time sits in 2024 must not be answerable from `time.time()`."""

    def __init__(self, start: datetime) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


class _NullTransport:
    async def send(self, to: str, content: dict[str, Any]) -> None: ...

    def receive(self) -> AsyncIterator[Message]:
        async def _drain() -> AsyncIterator[Message]:
            return
            yield  # pragma: no cover — never-yielding async generator

        return _drain()


def _cycle(
    tmp_path: Path,
    *,
    clock: DomainClock | None = None,
    llm: FakeLLMClient | None = None,
) -> tuple[DecisionCycle, WorkingMemory, EnvironmentRegistry]:
    tool = FakeTool("realestate")
    workspace = FakeWorkspace("ws", _ORIGIN, [tool], clock=clock)
    registry = EnvironmentRegistry(adapters={_ORIGIN: FakeAdapter("fake", workspace)})
    working = WorkingMemory(registry=registry)
    cycle = DecisionCycle(
        strategies=Strategies(
            observe=DefaultObserveStrategy(retirement_interval=0.0),
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
        procedural=ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm),
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "episodic")),
    )
    return cycle, working, registry


# ------------------------------------------------------------------------------------------------
# The `until` contract: a bound is declared, never inferred
# ------------------------------------------------------------------------------------------------


def test_a_plain_string_until_is_event_shaped_and_asks_nothing_of_the_clock() -> None:
    """The default, and every plan written before bounds existed. Nothing here is a deadline, so
    nothing is resolved mechanically — the retirement judge answers it, exactly as before."""
    for text in (
        "the restoration slot has taken place",
        "the Film Production Day has taken place",
        "the submission deadline has passed",
        "the 2024-10-15T09:00Z meeting has been rescheduled",
    ):
        cond = pending_from_raw({**_RAW, "until": text})
        assert cond is not None and cond.until == Until(text=text), text
        assert not cond.until.is_time_bounded, text


def test_the_object_form_declares_the_window_the_runtime_may_close() -> None:
    cond = pending_from_raw({**_RAW, "until": {"text": "30 minutes have passed", "seconds": 1800}})
    assert cond is not None
    assert cond.until == Until(text="30 minutes have passed", seconds=1800.0)
    assert cond.until.is_time_bounded


def test_an_unreadable_bound_keeps_the_clause_and_falls_back_to_the_judge() -> None:
    """The asymmetry that shapes this whole parser. Dropping the clause would leave a condition
    nothing ends; trusting a number nobody can read could close a window that is still open. So the
    text survives and the bound does not — the condition still ends, just by judgement."""
    for seconds in ("1800", None, -60, 0, True, {"n": 1}):
        cond = pending_from_raw({**_RAW, "until": {"text": "in a while", "seconds": seconds}})
        assert cond is not None, seconds
        assert cond.until == Until(text="in a while"), seconds


def test_a_bound_no_timeline_could_hold_is_unreadable_too() -> None:
    """A number can be positive and still name no instant. `timedelta` tops out around 8.6e13
    seconds and the datetime it is added to tops out at year 9999, so a window the planner writes
    as 1e15 seconds — or as the bare `Infinity` token, which `json.loads` accepts — raises
    OverflowError rather than producing a deadline.

    That matters far past tidiness: the arithmetic runs in `_retire_expired_conditions`, inside
    Observe, and `Agent.run` puts no `except` around `tick()`. One out-of-range number from one
    plan would end the whole run, and re-raise on every tick after. It is unreadable in exactly the
    sense the rest of this parser means, so it degrades the same way — the text survives, the bound
    does not, and the judge is asked."""
    for seconds in (float("inf"), float("-inf"), float("nan"), 1e15, 10**18):
        cond = pending_from_raw({**_RAW, "until": {"text": "in a while", "seconds": seconds}})
        assert cond is not None, seconds
        assert cond.until == Until(text="in a while"), seconds


def test_an_out_of_range_bound_is_unanswerable_rather_than_fatal() -> None:
    """Belt and braces on the seam itself: `Until` is also rebuilt from stored plans, so the parser
    is not the only way one is constructed. An instant no calendar can represent is the same answer
    as no anchor at all — unanswerable, which means "keep waiting", never "expired now"."""
    declared = datetime(2024, 10, 15, 12, 0, tzinfo=UTC)
    assert Until(text="in a while", seconds=1e15).deadline(declared) is None
    assert Until(text="in a while", seconds=float("inf")).deadline(declared) is None


def test_an_until_with_no_text_is_no_until_at_all() -> None:
    for raw in ({"seconds": 1800}, {"text": "   "}, "", "   ", 17, None):
        cond = pending_from_raw({**_RAW, "until": raw})
        assert cond is not None and cond.until is None, raw


def test_a_condition_survives_a_round_trip_through_procedural_memory(tmp_path: Path) -> None:
    """`asdict` flattens the nested Until to a dict, so the rehydration path has to rebuild it or a
    stored plan comes back with a bound no clock could read — silently unbounded, one run later."""
    plan = Plan(
        id="p1",
        goal="watch the calendar",
        steps=[Step(next_action="report", params={})],
        pending=(_monitoring(),),
    )
    restored = ProceduralMemory._from_dict(json.loads(json.dumps(asdict(plan))))
    assert restored.pending[0].until == _FOUR_MINUTES


def test_a_bound_resolves_to_a_deadline_against_the_moment_the_wait_began() -> None:
    declared = datetime(2024, 10, 15, 12, 0, tzinfo=UTC)
    assert _FOUR_MINUTES.deadline(declared) == datetime(2024, 10, 15, 12, 4, tzinfo=UTC)
    # No anchor is not "expired now": the workspace could not tell domain time when this was
    # lifted, and retiring on that would close a window that is still open.
    assert _FOUR_MINUTES.deadline(None) is None
    assert Until(text="the slot has taken place").deadline(declared) is None


# ------------------------------------------------------------------------------------------------
# The clock seam itself
# ------------------------------------------------------------------------------------------------


def test_the_host_clock_is_an_aware_instant() -> None:
    now = HostClock().now()
    assert now.tzinfo is not None
    assert abs((now - datetime.now(UTC)).total_seconds()) < 5


async def test_a_watchs_clock_is_the_one_of_the_workspace_owning_its_source(
    tmp_path: Path,
) -> None:
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    _cycle_, working, registry = _cycle(tmp_path, clock=clock)
    await registry.join(_ORIGIN)

    assert registry.workspace_of("realestate") is not None
    assert registry.workspace_of("nope") is None
    assert _clock_for_source(working.registry, "realestate") is clock
    assert _clock_for_source(working.registry, "nope") is None
    # A watch that names no source names no workspace, so no clock is determinate for it.
    assert _clock_for_source(working.registry, None) is None


async def test_a_workspace_that_cannot_tell_domain_time_has_no_clock(tmp_path: Path) -> None:
    _cycle_, working, registry = _cycle(tmp_path)
    await registry.join(_ORIGIN)
    assert _clock_for_source(working.registry, "realestate") is None


async def test_leaving_a_workspace_takes_its_clock_with_it(tmp_path: Path) -> None:
    _cycle_, working, registry = _cycle(tmp_path, clock=HostClock())
    await registry.join(_ORIGIN)
    await registry.leave("ws")
    assert registry.workspace_of("realestate") is None
    assert _clock_for_source(working.registry, "realestate") is None


# ------------------------------------------------------------------------------------------------
# Retirement, resolved mechanically
# ------------------------------------------------------------------------------------------------


def _blocked_on(activity: Activity, *conditions: PendingCondition) -> Activity:
    activity.pending_conditions = [
        PendingConditionState(condition=c, evaluated_through=0) for c in conditions
    ]
    activity.state = ActivityState.BLOCKED
    activity.blocked_on = ConditionWait(watches=tuple(c.watch for c in conditions))
    return activity


async def test_a_lifted_condition_records_the_domain_time_it_was_declared_at(
    tmp_path: Path,
) -> None:
    """The anchor a relative bound is measured from. Taken from the *workspace's* clock at the
    moment the condition is lifted onto the activity, never from `time.time()`."""
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    _cycle_, working, registry = _cycle(tmp_path, clock=clock)
    await registry.join(_ORIGIN)
    plan = Plan(id="p", goal="watch", steps=[Step("wait", {})], pending=(_monitoring(),))
    activity = Activity(id="a", goal="watch", context={}, plan=plan)

    _lift_pending_conditions(activity, working)

    assert activity.pending_conditions[0].declared_at == datetime(2024, 10, 15, 9, 0, tzinfo=UTC)


async def test_a_window_that_has_closed_retires_without_a_model_call(tmp_path: Path) -> None:
    """The common case ADR-0027 §4 says must not be taxed: a time-bounded `until` is answered from
    the clock, so the retirement judge is never asked."""
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    llm = FakeLLMClient()  # no canned response: any call raises
    cycle, working, registry = _cycle(tmp_path, clock=clock, llm=llm)
    await registry.join(_ORIGIN)
    activity = _blocked_on(Activity(id="a", goal="watch", context={}), _monitoring())
    activity.pending_conditions[0].declared_at = clock.now()
    working.activities["a"] = activity

    clock.advance(241)
    await cycle.strategies.observe.observe(cycle)

    assert activity.pending_conditions == []
    assert llm.calls == []
    assert activity.state is ActivityState.READY  # released, not left blocked on nothing


async def test_a_window_still_open_is_not_retired(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    cycle, working, registry = _cycle(tmp_path, clock=clock)
    await registry.join(_ORIGIN)
    activity = _blocked_on(Activity(id="a", goal="watch", context={}), _monitoring())
    activity.pending_conditions[0].declared_at = clock.now()
    working.activities["a"] = activity

    clock.advance(239)
    await cycle.strategies.observe.observe(cycle)

    assert len(activity.pending_conditions) == 1
    assert activity.state is ActivityState.BLOCKED


async def test_host_time_passing_does_not_close_a_domain_window(tmp_path: Path) -> None:
    """The 1970-vs-2024 bug, in the one place it would now bite: the domain clock is stopped while
    host wall-clock runs on, and the window must stay open."""
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    cycle, working, registry = _cycle(tmp_path, clock=clock)
    await registry.join(_ORIGIN)
    activity = _blocked_on(Activity(id="a", goal="watch", context={}), _monitoring())
    # Declared "now" in domain time — which is years away from whatever time.time() says.
    activity.pending_conditions[0].declared_at = clock.now()
    working.activities["a"] = activity

    for _ in range(3):
        await cycle.strategies.observe.observe(cycle)

    assert len(activity.pending_conditions) == 1


async def test_without_a_clock_a_time_bounded_until_is_left_to_the_judge(tmp_path: Path) -> None:
    """The workspace cannot tell domain time, so there is nothing to resolve against. Answering
    from `time.time()` instead is precisely the silent wrong answer this seam exists to prevent."""
    llm = FakeLLMClient(json.dumps({"retired": []}))
    cycle, working, registry = _cycle(tmp_path, llm=llm)
    await registry.join(_ORIGIN)
    activity = _blocked_on(Activity(id="a", goal="watch", context={}), _monitoring())
    activity.pending_conditions[0].declared_at = None
    working.activities["a"] = activity

    await cycle.strategies.observe.observe(cycle)
    for _ in range(6):
        await asyncio.sleep(0)

    assert len(activity.pending_conditions) == 1
    assert llm.calls  # fell through to the judge rather than guessing


async def test_an_event_shaped_until_is_never_resolved_mechanically(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    llm = FakeLLMClient(json.dumps({"retired": []}))
    cycle, working, registry = _cycle(tmp_path, clock=clock, llm=llm)
    await registry.join(_ORIGIN)
    activity = _blocked_on(
        Activity(id="a", goal="watch", context={}),
        _monitoring(Until(text="the Film Production Day has taken place")),
    )
    activity.pending_conditions[0].declared_at = clock.now()
    working.activities["a"] = activity

    clock.advance(86_400 * 30)
    await cycle.strategies.observe.observe(cycle)
    for _ in range(6):
        await asyncio.sleep(0)

    assert len(activity.pending_conditions) == 1


async def test_the_mechanical_pass_is_not_paced_by_the_judged_sweeps_interval(
    tmp_path: Path,
) -> None:
    """The sweep's backoff exists to stop paying a model per tick for the same "still waiting"
    answer, and it can hold an activity off for many minutes. A closed window costs nothing to
    notice, so it must not wait behind that — a four-minute window noticed sixteen minutes late is
    the very over-run the maintenance goal was declared to avoid."""
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    llm = FakeLLMClient()
    cycle, working, registry = _cycle(tmp_path, clock=clock, llm=llm)
    await registry.join(_ORIGIN)
    observe = DefaultObserveStrategy(retirement_interval=3600.0)
    activity = _blocked_on(Activity(id="a", goal="watch", context={}), _monitoring())
    activity.pending_conditions[0].declared_at = clock.now()
    working.activities["a"] = activity

    clock.advance(241)
    await observe.observe(cycle)

    assert activity.pending_conditions == []
    assert llm.calls == []


async def test_mechanical_retirement_releases_the_maintenance_frame_it_was_holding(
    tmp_path: Path,
) -> None:
    """Observe retires, Reason pops — the same split the judged sweep follows. Once the window's
    last condition is gone the frame is no longer held, so the parent resumes at its report."""
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    cycle, working, registry = _cycle(tmp_path, clock=clock)
    await registry.join(_ORIGIN)
    report = Step(next_action="send", params={"to": "user", "content": {"text": "done"}})
    parent = Plan(
        id="parent",
        goal="watch the calendar and clear conflicts",
        steps=[
            Step(
                next_action="subgoal",
                params={"goal": "clear conflicts as they appear", "goal_kind": "maintenance"},
            ),
            report,
        ],
    )
    condition = _monitoring()
    sub = Plan(id="sub", goal="clear conflicts", steps=[Step("wait", {})], pending=(condition,))
    activity = _blocked_on(Activity(id="a", goal="watch", context={}, plan=sub), condition)
    activity.step_index = 1
    activity.parent_frames.append((parent, 0, 0))
    activity.pending_conditions[0].declared_at = clock.now()
    working.activities["a"] = activity

    clock.advance(241)
    await cycle.strategies.observe.observe(cycle)
    assert activity.pending_conditions == []
    assert activity.state is ActivityState.READY

    resumed = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert resumed.step is report
    assert activity.parent_frames == []


# ------------------------------------------------------------------------------------------------
# Plan validation: a maintenance window nothing could ever close (ADR-0027 §6)
# ------------------------------------------------------------------------------------------------


_BOUNDED = {"text": "four minutes after the get_current_time result", "seconds": 240}


def _subplan_json(until: Any, *, source: str | None = "realestate") -> str:
    watch: dict[str, Any] = {"signal": "state_changed", "path": "events", "kind": "added"}
    if source is not None:
        watch["source"] = source
    return json.dumps(
        {
            "steps": [{"action": "send", "to": "user", "content": {"text": "done"}}],
            "pending": [
                {
                    "watch": watch,
                    "when": "one or more calendar events are added",
                    "then": "delete every overlapping preexisting calendar event",
                    "until": until,
                }
            ],
        }
    )


async def _land_subplan(
    tmp_path: Path,
    subplan: str,
    *,
    clock: DomainClock | None = None,
    goal_kind: str = "maintenance",
) -> tuple[Activity, Plan]:
    """Drive a deliberative sub-goal all the way through: Reason fires `_infer_`, the sub-plan
    lands in Observe. Validation happens where the plan arrives, before anything is entered."""
    llm = FakeLLMClient(subplan)
    cycle, working, registry = _cycle(tmp_path, clock=clock, llm=llm)
    await registry.join(_ORIGIN)
    subgoal = Step(
        next_action="subgoal",
        params={
            "goal": "for the next four minutes clear conflicts as they appear",
            "mode": "deliberative",
            "goal_kind": goal_kind,
        },
    )
    parent = Plan(id="p", goal="watch the calendar", steps=[subgoal, Step("wait", {})])
    activity = Activity(id="a", goal="watch the calendar", context={}, plan=parent)
    working.activities["a"] = activity

    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    await asyncio.sleep(0)
    await cycle.strategies.observe.observe(cycle)
    return activity, parent


async def test_a_maintenance_window_no_clock_could_close_is_refused(tmp_path: Path) -> None:
    """It would hold its frame — and its parent's remaining steps — silently forever. Refused with
    a named defect, which is what makes the replan differ from the attempt (ADR-0025)."""
    activity, _parent = await _land_subplan(tmp_path, _subplan_json(_BOUNDED))

    assert activity.plan is None  # nothing entered
    assert activity.parent_frames == []
    assert activity.state is ActivityState.READY  # re-plans next cycle
    assert activity.superseded is not None
    defect = activity.superseded.defect
    assert defect is not None and "clock" in defect
    assert activity.replan_trail and activity.replan_trail[-1] == defect


async def test_the_same_window_is_accepted_where_the_clock_can_close_it(tmp_path: Path) -> None:
    activity, parent = await _land_subplan(
        tmp_path,
        _subplan_json(_BOUNDED),
        clock=FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC)),
    )

    assert activity.plan is not None and activity.plan is not parent
    assert activity.parent_frames == [(parent, 0, 0)]
    assert activity.state is ActivityState.READY


async def test_an_event_shaped_window_needs_no_clock(tmp_path: Path) -> None:
    """The retirement judge answers it from the observed world, so a clock-less workspace is no
    obstacle — refusing it would ground a perfectly terminable plan."""
    activity, parent = await _land_subplan(
        tmp_path, _subplan_json("the Film Production Day has taken place")
    )

    assert activity.plan is not None and activity.plan is not parent
    assert activity.parent_frames == [(parent, 0, 0)]


async def test_an_achievement_sub_goal_is_not_refused(tmp_path: Path) -> None:
    """Only a maintenance frame is held by its own conditions. An achievement frame pops when its
    steps run out and the condition keeps watching from the activity — ADR-0022's contingency case,
    which a clock has no bearing on."""
    activity, parent = await _land_subplan(
        tmp_path,
        _subplan_json(_BOUNDED),
        goal_kind="achievement",
    )

    assert activity.plan is not None and activity.plan is not parent
    assert activity.parent_frames == [(parent, 0, 0)]


async def test_a_watch_with_no_source_names_no_clock_and_is_refused(tmp_path: Path) -> None:
    """`until` resolves against the workspace owning the watch's source. With no source there is no
    workspace to ask, so a time-bounded window is as unclosable as one with no clock — and the
    defect says which of the two it was, since the planner's fix differs."""
    activity, _parent = await _land_subplan(
        tmp_path,
        _subplan_json(_BOUNDED, source=None),
        clock=FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC)),
    )

    assert activity.plan is None
    assert activity.superseded is not None
    defect = activity.superseded.defect
    assert defect is not None and "source" in defect
