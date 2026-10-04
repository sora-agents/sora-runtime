"""The expiry branch of a pending condition: `otherwise` (ADR-0028).

The failure these exist for: "send the colleagues a message asking who orders the cab; **if after
three minutes there is no response**, order a default cab yourself." An agent sent the messages
correctly and then had nowhere to put the second half. `then` is reachable only by a FIRING, and
`until` ends the watching without attaching anything to the ending — so the branch that the goal is
actually about, the one taken when the awaited reply never comes, had no representable form. The
planner restated the goal five times across a replan searching for the construct, and the cab was
never ordered.

What is pinned hardest here is the *negative* side of the trigger, because it is what distinguishes
this from a firing and what a plausible-looking implementation gets wrong: the branch is owed only
to a window that closed having never fired, it must be owed from **both** retirement paths, and a
verdict that fires and retires the same condition in one answer — the ordinary one-shot shape — owes
nothing. The last of those is the case that would otherwise run both branches of one condition.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeAdapter, FakeLLMClient, FakeTool, FakeWorkspace
from sora.action import default_action_registry
from sora.activity import SEEDED_BINDINGS, Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import DomainClock, EnvironmentRegistry, WorkspaceOrigin
from sora.llm import CompletionRequest, LLMClient
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
    pending_from_raw,
)
from sora.perception import Message, Percept
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReasonStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    Strategies,
    TickResult,
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
    Until,
)

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")
_MESSAGES = SignalWait(
    signal_name="state_changed", source="messaging", path="conversations", kind="added"
)
# The condition as the planner writes it once it can state the branch at all.
_RAW: dict[str, Any] = {
    "watch": {
        "signal": "state_changed",
        "source": "messaging",
        "path": "conversations",
        "kind": "added",
    },
    "when": "one of the colleagues replies saying who will order the cab",
    "then": "Confirm to the user who is ordering the cab",
    "until": {"text": "three minutes have passed", "seconds": 180},
    "otherwise": "Order a default cab from the Department of Physics to The Eagle Pub",
}
_THREE_MINUTES = Until(text="three minutes have passed", seconds=180.0)


def _waiting(
    *, otherwise: str | None = "Order a default cab", until: Until | None = _THREE_MINUTES
) -> PendingCondition:
    return PendingCondition(
        watch=_MESSAGES,
        when="one of the colleagues replies saying who will order the cab",
        then="Confirm to the user who is ordering the cab",
        until=until,
        otherwise=otherwise,
    )


class FakeClock:
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
    # The Protocol, not `FakeLLMClient`: one test needs a client that can hold an answer back.
    llm: LLMClient | None = None,
) -> tuple[DecisionCycle, WorkingMemory, EnvironmentRegistry]:
    tool = FakeTool("messaging")
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


def _blocked_on(activity: Activity, *conditions: PendingCondition) -> Activity:
    activity.pending_conditions = [
        PendingConditionState(condition=c, evaluated_through=0) for c in conditions
    ]
    activity.state = ActivityState.BLOCKED
    activity.blocked_on = ConditionWait(watches=tuple(c.watch for c in conditions))
    return activity


def _exhausted(pending: tuple[PendingCondition, ...] = ()) -> Activity:
    """An activity whose body has run out — messages sent, now waiting for a reply."""
    plan = Plan(id="p1", goal="ask who orders the cab", steps=[Step("wait", {})], pending=pending)
    return Activity(id="a", goal="ask who orders the cab", context={}, plan=plan, step_index=1)


def _tick() -> TickResult:
    return TickResult()


async def _planned_goal(llm: FakeLLMClient) -> str:
    """Which goal the pursued sub-plan was actually inferred for.

    `Activity.pursued_goals` is not it: that is written when the inference RESOLVES, and the call is
    off-cycle, so it is still empty the moment Reason returns. The prompt is what is in hand.
    """
    for _ in range(3):  # let the spawned off-cycle call reach the client
        await asyncio.sleep(0)
    assert llm.calls, "no planning call was made"
    return llm.requests[-1].user.splitlines()[0].removeprefix("Goal: ").strip()


async def _settle() -> None:
    """Let a spawned off-cycle judgement run — these drive Observe directly, not via a loop."""
    for _ in range(6):
        await asyncio.sleep(0)


def _mid_body() -> Activity:
    """An activity with body still to run, so a queued branch stays observable instead of being
    pursued and popped in the same call."""
    plan = Plan(id="p1", goal="ask who orders the cab", steps=[Step("wait", {}), Step("wait", {})])
    return Activity(id="a", goal="ask who orders the cab", context={}, plan=plan, step_index=0)


# --------------------------------------------------------------------------------------------------
# Parsing: what the planner emits
# --------------------------------------------------------------------------------------------------


def test_the_expiry_branch_parses_from_planner_json() -> None:
    cond = pending_from_raw(_RAW)
    assert cond is not None
    assert cond.otherwise == "Order a default cab from the Department of Physics to The Eagle Pub"


def test_a_condition_without_an_expiry_branch_is_unchanged() -> None:
    """Every plan written before this construct existed, and every plan that simply has no
    alternative to take. `None` is the whole of the old behaviour: retirement stops the watching and
    attaches nothing to the ending."""
    cond = pending_from_raw({k: v for k, v in _RAW.items() if k != "otherwise"})
    assert cond is not None
    assert cond.otherwise is None


def test_an_unusable_expiry_branch_drops_the_branch_not_the_condition() -> None:
    """The asymmetry against `when`/`then`, which are fatal. A condition with no `then` has nothing
    to do at all; one with a mis-shaped `otherwise` still has a working watch and positive branch,
    and dropping it whole takes the branch that IS well-formed down with the one that is not."""
    unusable: tuple[Any, ...] = ("", "   ", 180, [], {"goal": "order a cab"}, True)
    for bad in unusable:
        cond = pending_from_raw({**_RAW, "otherwise": bad})
        assert cond is not None, bad
        assert cond.otherwise is None, bad
        assert cond.then == "Confirm to the user who is ordering the cab", bad


async def test_a_stored_plan_keeps_its_expiry_branch(tmp_path: Path) -> None:
    """A JSON round-trip through the procedural store. The sibling bug this guards against is live
    in this file's history: `pending` and its nested `SignalWait` both had to be rebuilt by hand,
    and a field left out of that rebuild comes back as a plan whose branch silently never runs."""
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"))
    condition = _waiting()
    plan = Plan(
        id="p1", goal="ask who orders the cab", steps=[Step("wait", {})], pending=(condition,)
    )
    await procedural.store(plan)

    restored = await procedural.retrieve(
        Activity(id="a", goal="ask who orders the cab", context={})
    )

    assert restored is not None
    assert restored.pending[0].otherwise == "Order a default cab"
    assert restored.pending == (condition,)


# --------------------------------------------------------------------------------------------------
# The clock's retirement path — the one the motivating goal needs
# --------------------------------------------------------------------------------------------------


async def test_a_window_that_closes_unfired_pursues_its_expiry_branch(tmp_path: Path) -> None:
    """The motivating case end to end: three minutes pass, nobody replied, the cab gets ordered.

    Note which phase does what. Observe notices the window is spent and queues the branch; Reason
    pursues it as an ordinary deliberative sub-goal, exactly as it would a `then`. Nothing here
    costs a judgement — the bound was declared, so closing the window is a comparison.
    """
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    llm = FakeLLMClient(json.dumps({"steps": []}))
    cycle, working, registry = _cycle(tmp_path, clock=clock, llm=llm)
    await registry.join(_ORIGIN)
    activity = _blocked_on(_exhausted(), _waiting())
    activity.pending_conditions[0].declared_at = clock.now()
    working.activities[activity.id] = activity

    clock.advance(181)
    await cycle.strategies.observe.observe(cycle)

    # Released, with the branch owed rather than forgotten.
    assert activity.pending_conditions == []
    assert activity.state is ActivityState.READY
    assert [f.goal for f in activity.condition_fired] == ["Order a default cab"]
    assert activity.condition_fired[0].on_expiry is True

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    # Pursued down the same path a firing takes: a goal planned fresh when the moment comes.
    assert activity.pending_inference is not None
    assert activity.pending_inference.kind == "then"
    assert activity.condition_fired == []


async def test_a_window_still_open_owes_nothing(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    cycle, working, registry = _cycle(tmp_path, clock=clock)
    await registry.join(_ORIGIN)
    activity = _blocked_on(_exhausted(), _waiting())
    activity.pending_conditions[0].declared_at = clock.now()
    working.activities[activity.id] = activity

    clock.advance(60)
    await cycle.strategies.observe.observe(cycle)

    assert activity.condition_fired == []
    assert activity.state is ActivityState.BLOCKED


async def test_a_retiring_condition_that_declared_no_branch_owes_nothing(tmp_path: Path) -> None:
    """The pre-existing behaviour, pinned so this construct cannot change it by accident: a window
    with no `otherwise` still just stops waiting."""
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    llm = FakeLLMClient()  # no canned response: any model call raises
    cycle, working, registry = _cycle(tmp_path, clock=clock, llm=llm)
    await registry.join(_ORIGIN)
    activity = _blocked_on(_exhausted(), _waiting(otherwise=None))
    activity.pending_conditions[0].declared_at = clock.now()
    working.activities[activity.id] = activity

    clock.advance(181)
    await cycle.strategies.observe.observe(cycle)

    assert activity.pending_conditions == []
    assert activity.condition_fired == []
    assert llm.calls == []


async def test_a_window_that_already_fired_owes_nothing_when_it_closes(tmp_path: Path) -> None:
    """ "If there is no response" means exactly that. A firing does not consume a condition, `until`
    is what ends it, so a window can fire and then expire — and that expiry has already had its
    awaited event. Without `ever_fired` the runtime cannot tell the two endings apart and would
    order the cab after being told who is ordering it."""
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    cycle, working, registry = _cycle(tmp_path, clock=clock)
    await registry.join(_ORIGIN)
    activity = _blocked_on(_exhausted(), _waiting())
    activity.pending_conditions[0].declared_at = clock.now()
    activity.pending_conditions[0].ever_fired = True
    working.activities[activity.id] = activity

    clock.advance(181)
    await cycle.strategies.observe.observe(cycle)

    assert activity.pending_conditions == []
    assert activity.condition_fired == []


async def test_an_expiry_resumes_the_activity_even_with_a_sibling_still_watched(
    tmp_path: Path,
) -> None:
    """The guard that is invisible in every test retiring an activity's only condition.

    An expiry branch is reached by Reason and Situate only ever selects a READY activity, so an
    activity left BLOCKED holds its queued branch for good — the signal that could have woken it has
    already been and gone, which is why the branch is owed at all. So retiring one window while a
    sibling is still watched must still release the activity.
    """
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    llm = FakeLLMClient(json.dumps({"steps": []}))
    cycle, working, registry = _cycle(tmp_path, clock=clock, llm=llm)
    await registry.join(_ORIGIN)
    sibling = PendingCondition(
        watch=_MESSAGES,
        when="the lunch event is moved",
        then="Re-check the pickup time",
        until=Until(text="the lunch has taken place"),  # event-shaped: the clock cannot close it
    )
    activity = _blocked_on(_exhausted(), _waiting(), sibling)
    activity.pending_conditions[0].declared_at = clock.now()
    working.activities[activity.id] = activity

    clock.advance(181)
    await cycle.strategies.observe.observe(cycle)

    assert [s.condition for s in activity.pending_conditions] == [sibling]
    assert [f.goal for f in activity.condition_fired] == ["Order a default cab"]
    assert activity.state is ActivityState.READY


async def test_the_expiry_branch_seeds_no_changed_ids(tmp_path: Path) -> None:
    """A `then` is seeded with the ids its firing reported, so its sub-plan can name them
    mechanically. An expiry reports none — nothing moved, which is the entire point of the branch —
    so the seeds are CLEARED rather than filled. Handing the `otherwise` plan an authoritative empty
    set instead would make an `in` select nothing and a `not_in` exclude nothing, silently."""
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    llm = FakeLLMClient(json.dumps({"steps": []}))
    cycle, working, registry = _cycle(tmp_path, clock=clock, llm=llm)
    await registry.join(_ORIGIN)
    activity = _blocked_on(_exhausted(), _waiting())
    activity.pending_conditions[0].declared_at = clock.now()
    for name in SEEDED_BINDINGS:
        activity.bindings[name] = ["stale-from-an-earlier-firing"]
    working.activities[activity.id] = activity

    clock.advance(181)
    await cycle.strategies.observe.observe(cycle)
    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    for name in SEEDED_BINDINGS:
        assert name not in activity.bindings, name


# --------------------------------------------------------------------------------------------------
# The judge's retirement path — the same debt, owed from the other side
# --------------------------------------------------------------------------------------------------


async def test_an_event_shaped_window_the_judge_retires_also_owes_its_branch(
    tmp_path: Path,
) -> None:
    """Both retirement paths owe the branch, and this is the half a fix hooked in one place misses.

    Restricting the branch to the clock reads safer — a false retire would then only stop watching
    instead of acting — but it silently drops the branch for every plan whose `until` is
    event-shaped, which reproduces the original defect invisibly. The retirement judge is instructed
    to default to KEEPING, so its error direction is a branch that does not fire, which is the
    pre-existing behaviour rather than a new wrong action.
    """
    llm = FakeLLMClient(json.dumps({"steps": []}))
    cycle, working, _registry = _cycle(tmp_path, llm=llm)
    condition = _waiting(until=Until(text="the lunch has taken place"))
    activity = _blocked_on(_exhausted(), condition)
    activity.condition_batch = list(activity.pending_conditions)
    activity.condition_verdict = ConditionVerdict(retired=(0,))
    working.activities[activity.id] = activity

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert activity.pending_conditions == []
    assert activity.pending_inference is not None  # pursued straight away: the body is idle
    assert activity.pending_inference.kind == "then"


async def test_one_verdict_that_fires_and_retires_owes_only_the_then(tmp_path: Path) -> None:
    """`{"fired": [0], "retired": [0]}` is the ordinary ONE-SHOT answer, not a contradiction — the
    awaited thing happened and nothing further is being waited for. Reading it as a window that
    closed unfired would run both branches of one condition, in the most common shape rather than an
    exotic one. This is why firings are marked before retirement is applied."""
    llm = FakeLLMClient(json.dumps({"steps": []}))
    cycle, working, _registry = _cycle(tmp_path, llm=llm)
    activity = _blocked_on(_exhausted(), _waiting())
    activity.condition_batch = list(activity.pending_conditions)
    activity.condition_verdict = ConditionVerdict(fired=(0,), retired=(0,))
    working.activities[activity.id] = activity

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    # Exactly one goal owed, and it is the positive branch. An empty queue after the pop is the
    # load-bearing half: a second item would still be sitting in it.
    assert activity.pending_inference is not None
    assert activity.condition_fired == []
    assert await _planned_goal(llm) == "Confirm to the user who is ordering the cab"


async def test_a_firing_and_a_separate_expiry_pursue_the_awaited_event_first(
    tmp_path: Path,
) -> None:
    """Two conditions, one verdict: one fired, the other closed unfired. Both goals are owed, and
    the firing goes first — ordering between distinct conditions is otherwise arbitrary, and the
    event that actually happened is the better claim on the next sub-plan."""
    llm = FakeLLMClient(json.dumps({"steps": []}))
    cycle, working, _registry = _cycle(tmp_path, llm=llm)
    closing = PendingCondition(
        watch=_MESSAGES,
        when="a colleague declines",
        then="Ask the next colleague",
        until=Until(text="the lunch has taken place"),
        otherwise="Order a default cab",
    )
    activity = _blocked_on(_exhausted(), _waiting(), closing)
    activity.condition_batch = list(activity.pending_conditions)
    activity.condition_verdict = ConditionVerdict(fired=(0,), retired=(1,))
    working.activities[activity.id] = activity

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert await _planned_goal(llm) == "Confirm to the user who is ordering the cab"
    assert [f.goal for f in activity.condition_fired] == ["Order a default cab"]


# --------------------------------------------------------------------------------------------------
# Races between the sites that can close one window
# --------------------------------------------------------------------------------------------------


async def test_a_window_closing_under_an_outstanding_judgement_defers_its_branch(
    tmp_path: Path,
) -> None:
    """The near-deadline case, which is the one the construct is actually for: the reply arrives
    just before the three minutes are up, its judgement is dispatched, and the clock closes the
    window while that verdict is still in flight.

    `ever_fired` is only true once a verdict has been APPLIED, and Observe's clock sweep runs after
    the inference drain but before Reason — so at the moment the window closes the condition reads
    as "never fired" even though the answer saying otherwise is already in hand. Committing the
    negative branch there orders a default cab *and* confirms who is ordering one, off a single
    verdict. So the window still closes on time, and only the branch decision waits for the
    judgement that is already paid for.
    """
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    llm = FakeLLMClient(json.dumps({"steps": []}))
    cycle, working, registry = _cycle(tmp_path, clock=clock, llm=llm)
    await registry.join(_ORIGIN)
    activity = _blocked_on(_exhausted(), _waiting())
    state = activity.pending_conditions[0]
    state.declared_at = clock.now()
    activity.condition_batch = [state]  # a reply landed; the batched judge is mid-answer
    working.activities[activity.id] = activity

    clock.advance(181)
    await cycle.strategies.observe.observe(cycle)

    assert activity.pending_conditions == []  # the window closes on time regardless
    assert activity.condition_fired == []  # but nothing is committed on an unread answer
    assert state.expiry_owed is True
    # Resumed, so Reason can reach the verdict. With a sibling condition still watched this would
    # stay BLOCKED instead, and the wake is the verdict's own resolution setting the activity READY.
    assert activity.state is ActivityState.READY

    # The verdict lands, and the colleague did reply: only the positive branch was ever owed.
    activity.condition_verdict = ConditionVerdict(fired=(0,))
    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert await _planned_goal(llm) == "Confirm to the user who is ordering the cab"
    assert activity.condition_fired == []  # no second goal trailing behind the `then`


async def test_a_deferred_branch_is_owed_once_the_verdict_reports_nothing_fired(
    tmp_path: Path,
) -> None:
    """The other half of the deferral, and what keeps it from being a silent drop. The signal that
    closed the gate was judged NOT to be the awaited reply — someone wrote about something else —
    so the window really did close unfired and the cab is owed after all. The verdict is where that
    becomes knowable, which is why it is also where the branch is committed."""
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    llm = FakeLLMClient(json.dumps({"steps": []}))
    cycle, working, registry = _cycle(tmp_path, clock=clock, llm=llm)
    await registry.join(_ORIGIN)
    activity = _blocked_on(_exhausted(), _waiting())
    state = activity.pending_conditions[0]
    state.declared_at = clock.now()
    activity.condition_batch = [state]
    working.activities[activity.id] = activity

    clock.advance(181)
    await cycle.strategies.observe.observe(cycle)
    activity.condition_verdict = ConditionVerdict()  # judged: not the awaited thing
    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert await _planned_goal(llm) == "Order a default cab"


async def test_two_retirement_paths_closing_one_window_owe_the_branch_once(
    tmp_path: Path,
) -> None:
    """Two sites can retire the same event-shaped condition — the quiet retirement sweep and the
    batched eligibility verdict — and each carries a list of states captured before the call it was
    waiting on resolved.

    Dropping the condition deduplicates itself: the removal is by identity, so a state another site
    already removed is simply no longer found. Queuing has no such natural no-op, so the second
    arrival would queue the same `otherwise` again and the timeout action would run twice — two cabs
    ordered off one closed window. `expiry_settled` is what makes the decision once-per-waiter
    regardless of how many sites reach it, and regardless of whether a site decides before dropping
    (the clock) or after (the verdict).
    """
    llm = FakeLLMClient(json.dumps({"retired": [0]}))
    cycle, working, _registry = _cycle(tmp_path, llm=llm)
    activity = _blocked_on(_mid_body(), _waiting(until=Until(text="the lunch has taken place")))
    working.activities[activity.id] = activity

    # The sweep fires its judgement off-cycle and parks a verdict retiring the window.
    await cycle.strategies.observe.observe(cycle)
    await _settle()
    assert llm.calls, "the retirement sweep never fired"

    # Before that lands, the batched judge answers the same question and Reason applies it.
    activity.condition_batch = list(activity.pending_conditions)
    activity.condition_verdict = ConditionVerdict(retired=(0,))
    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert [f.goal for f in activity.condition_fired] == ["Order a default cab"]

    # Now the sweep's parked verdict lands, naming a state nothing holds any more.
    await cycle.strategies.observe.observe(cycle)

    assert [f.goal for f in activity.condition_fired] == ["Order a default cab"]


def _reply(working: WorkingMemory) -> None:
    """A colleague's reply landing on the watched path — a change the condition has not judged."""
    working.signals.append(
        Percept(
            "messaging",
            Signal("state_changed", {"changes": [Change(path="conversations", added=("c9",))]}),
            0.0,
        )
    )
    working.signals_appended += 1


async def test_a_window_closing_over_an_unjudged_change_declines_its_branch(
    tmp_path: Path,
) -> None:
    """A reply lands two seconds before the deadline and the clock closes the window before any
    Reason pass could judge it.

    The gate opened, so the evidence that would settle the branch is sitting in working memory
    unread — and "no response arrived" is the one thing that is now demonstrably not known. Acting
    anyway orders a cab over a reply nobody looked at. Declining is a permanent decision here rather
    than a deferral: no judgement will ever be dispatched for a condition that has stopped existing,
    so there is nothing to wait for. That the reply itself is lost is the behaviour that preceded
    this construct — a change landing in the last tick of a window has never been judged — and it is
    deliberately not widened into here.
    """
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    llm = FakeLLMClient()  # no canned response: any model call raises
    cycle, working, registry = _cycle(tmp_path, clock=clock, llm=llm)
    await registry.join(_ORIGIN)
    activity = _blocked_on(_exhausted(), _waiting())
    activity.pending_conditions[0].declared_at = clock.now()
    working.activities[activity.id] = activity

    clock.advance(179)
    _reply(working)
    clock.advance(2)
    await cycle.strategies.observe.observe(cycle)

    assert activity.pending_conditions == []  # the window still closes on time
    assert activity.condition_fired == []  # but nothing is owed off an unread change
    assert llm.calls == []


async def test_a_failed_judgement_does_not_authorize_the_expiry_branch(tmp_path: Path) -> None:
    """A condition judgement that ERRORS parks an empty verdict — the runtime's fail-soft for a
    flaky seam, chosen because "the activity was already waiting, so keeping it waiting changes
    nothing". That reasoning does not survive a window that has since closed: an empty verdict then
    reads as "nothing fired", which is exactly the authorization the negative branch needs. A
    network failure while judging the reply would order the cab. The verdict therefore says whether
    it is an answer at all, and a non-answer decides nothing."""
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    llm = FakeLLMClient()  # no canned response: any model call raises
    cycle, working, registry = _cycle(tmp_path, clock=clock, llm=llm)
    await registry.join(_ORIGIN)
    # Body still to run, so a wrongly-authorized branch sits in the queue to be seen rather than
    # being pursued and popped in the same call.
    activity = _blocked_on(_mid_body(), _waiting())
    state = activity.pending_conditions[0]
    state.declared_at = clock.now()
    activity.condition_batch = [state]
    working.activities[activity.id] = activity

    clock.advance(181)
    await cycle.strategies.observe.observe(cycle)

    assert state.expiry_owed is True  # the branch waits on the judgement, as it should

    activity.condition_verdict = ConditionVerdict(failed=True)  # ... which then errored
    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert activity.condition_fired == []
    assert llm.calls == []


# --------------------------------------------------------------------------------------------------
# A non-answer arriving over a working seam
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "owed"),
    [
        ("the colleagues have not replied yet", False),  # no JSON at all
        ('{"fired": "unknown", "retired": []}', False),  # the field is not a list
        ('{"retired": []}', False),  # the field is absent
        ('{"fired": [7], "retired": []}', False),  # emptied by dropping an out-of-range index
        ('{"fired": [], "retired": []}', True),  # answered, and the answer is "nothing"
    ],
)
async def test_only_an_answered_empty_fired_authorizes_the_deferred_branch(
    tmp_path: Path, answer: str, owed: bool
) -> None:
    """The same defect as a seam failure, reached through a seam that worked.

    A raised error is the obvious non-answer; malformed output is the likely one, and it arrives as
    an ordinary successful call. The parser's fail-soft turns all of it into an empty verdict —
    deliberately, because for every other consumer an absent fire does nothing. For the expiry
    branch "nothing fired" is the authorization, so `not valid JSON` ordered a cab, and so did a
    `fired` the model filled with prose. Only the last row here establishes that no condition fired,
    and it is in the table so that the fix cannot be "never commit a deferred branch".

    Driven through the real judge and the real parser rather than a hand-built verdict: the
    distinction lives in `_parse_condition_verdict`, so a test that parks a verdict directly cannot
    see it get lost.
    """
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    llm = FakeLLMClient(answer)
    cycle, working, registry = _cycle(tmp_path, clock=clock, llm=llm)
    await registry.join(_ORIGIN)
    # Body still to run, so a wrongly-authorized branch sits in the queue to be seen rather than
    # being pursued and popped in the same call.
    activity = _blocked_on(_mid_body(), _waiting())
    activity.pending_conditions[0].declared_at = clock.now()
    working.activities[activity.id] = activity

    # Two seconds before the deadline a reply lands, and Reason dispatches the batched judgement.
    clock.advance(179)
    _reply(working)
    await cycle.strategies.observe.observe(cycle)
    await cycle.strategies.reason.reason(activity, working, cycle, _tick())
    assert activity.condition_batch, "the judgement was not dispatched"

    # The window then closes with that judgement outstanding, so the branch defers to it.
    clock.advance(2)
    await cycle.strategies.observe.observe(cycle)
    assert activity.pending_conditions == []
    assert activity.condition_fired == []

    # The judgement comes back and the next Observe parks it, read by the real parser.
    await _settle()
    await cycle.strategies.observe.observe(cycle)
    assert activity.condition_verdict is not None, "the judgement never came back"
    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert [f.on_expiry for f in activity.condition_fired] == ([True] if owed else [])


@pytest.mark.parametrize(
    ("answer", "owed"),
    [
        ('{"fired": "unknown", "retired": [0]}', False),  # retires readably, says nothing of firing
        ('{"retired": [0]}', False),  # the firing claim is simply absent
        ('{"fired": [], "retired": [0]}', True),  # both halves answered: closed, and unfired
    ],
)
async def test_a_readable_retirement_in_an_unreadable_answer_owes_no_branch(
    tmp_path: Path, answer: str, owed: bool
) -> None:
    """Half an answer can be perfectly readable, and the readable half is the one that retires.

    `{"fired": "unknown", "retired": [0]}` is a *valid* retirement: the `until` clause was judged
    satisfied, so the watching legitimately ends and the condition is dropped. What the same answer
    failed to say is whether the awaited event happened — and the branch needs both halves, because
    "the window closed" and "the awaited thing never happened" are different claims. Reading the
    retirement as authorization lets an unreadable `fired` order the cab through the one group that
    does not come from a deferral, which is why the failure gate governs both.

    Event-shaped `until` on purpose: that is the only `until` the batched judge retires, since a
    declared duration is the clock's to close.
    """
    llm = FakeLLMClient(answer)
    cycle, working, registry = _cycle(tmp_path, llm=llm)
    await registry.join(_ORIGIN)
    # Body still to run, so a wrongly-authorized branch stays in the queue to be seen.
    activity = _blocked_on(_mid_body(), _waiting(until=Until(text="the lunch has taken place")))
    working.activities[activity.id] = activity

    # A change opens the gate and Reason dispatches the real batched judgement over it.
    _reply(working)
    await cycle.strategies.observe.observe(cycle)
    await cycle.strategies.reason.reason(activity, working, cycle, _tick())
    assert activity.condition_batch, "the judgement was not dispatched"

    await _settle()
    await cycle.strategies.observe.observe(cycle)
    assert activity.condition_verdict is not None, "the judgement never came back"
    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert activity.pending_conditions == []  # the retirement half is honoured either way
    assert [f.on_expiry for f in activity.condition_fired] == ([True] if owed else [])


class _GatedLLM:
    """Two answers, and the retirement one held until the test releases it.

    `FakeLLMClient` answers immediately, which makes the order the two judgement paths resolve in a
    property of where the event loop happens to yield. This race is *about* that order, so it is
    pinned here instead: the quiet sweep's call is dispatched first and parks its answer last.
    """

    def __init__(self, retirement: str, batched: str) -> None:
        self.release = asyncio.Event()
        self._retirement = retirement
        self._batched = batched
        self.calls: list[tuple[str, str]] = []
        self.requests: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> str:
        self.requests.append(request)
        self.calls.append((request.system, request.user))
        if request.semantic_label == "retirement":
            await self.release.wait()
            return self._retirement
        return self._batched


async def test_a_late_sweep_cannot_resurrect_a_rejected_branch(tmp_path: Path) -> None:
    """The last way into the branch, and the one a commit-only mark cannot close.

    A quiet retirement judgement goes out. While it is in flight a reply arrives, the batched judge
    answers `{"fired": "unknown", "retired": [0]}`, and Reason does the right thing: the readable
    `retired` half retires the waiter, the unreadable `fired` half authorizes nothing. Then the
    older sweep lands, still holding the waiter it captured before any of that, and asks for the
    branch that was just refused.

    Nothing on the waiter says no. `ever_fired` is false — nothing was ever judged to have fired —
    the evaluation mark has advanced past the reply so the unjudged-change guard sees a quiet watch,
    and a mark that records only commitments is false because nothing was committed. A rejection
    that is merely *performed* is indistinguishable from a decision never taken, which is why
    `expiry_settled` records the decision rather than the queueing.
    """
    llm = _GatedLLM(
        retirement=json.dumps({"retired": [0]}),
        batched=json.dumps({"fired": "unknown", "retired": [0]}),
    )
    cycle, working, registry = _cycle(tmp_path, llm=llm)
    await registry.join(_ORIGIN)
    activity = _blocked_on(_mid_body(), _waiting(until=Until(text="the lunch has taken place")))
    working.activities[activity.id] = activity

    # 1. The sweep fires its judgement, which then hangs on the gate.
    await cycle.strategies.observe.observe(cycle)
    await _settle()
    assert [r.semantic_label for r in llm.requests] == ["retirement"]

    # 2. A reply arrives and Reason dispatches the batched judgement over it.
    _reply(working)
    await cycle.strategies.observe.observe(cycle)
    await cycle.strategies.reason.reason(activity, working, cycle, _tick())
    assert activity.condition_batch, "the batched judgement was not dispatched"

    # 3-4. It comes back unreadable on `fired`: the waiter retires, and no branch is owed.
    await _settle()
    await cycle.strategies.observe.observe(cycle)
    await cycle.strategies.reason.reason(activity, working, cycle, _tick())
    assert activity.pending_conditions == []
    assert activity.condition_fired == []

    # 5. Only now does the older sweep answer, naming the waiter it captured in step 1.
    llm.release.set()
    await _settle()
    await cycle.strategies.observe.observe(cycle)

    assert activity.condition_fired == []  # the refusal stands


@pytest.mark.parametrize(
    ("answer", "owed"),
    [
        ('{"fired": "unknown", "retired": []}', False),  # unreadable, applied before the deadline
        ("not valid JSON", False),  # same non-answer, nothing parsed at all
        ('{"fired": [], "retired": []}', True),  # answered: the reply was read and did not fire
    ],
)
async def test_an_unreadable_answer_before_the_deadline_drops_the_branch(
    tmp_path: Path, answer: str, owed: bool
) -> None:
    """The rejection has to outlive the call that made it, even for a waiter still on watch.

    Every earlier round of this defect had the window already closed when the unreadable answer
    landed, so the waiter was either retired by that same verdict or carrying a deferral — both
    groups the drop was recorded for. A reply judged *mid-window* is in neither: the verdict applies
    while the condition is still live, and two ticks later the clock closes the window on a waiter
    that looks pristine. It is not. Judging it advanced its evaluation cursor past the reply, and an
    advanced cursor is exactly what a watch nothing ever matched looks like — so the sweep finds no
    open gate, no judgement in flight, and reads this method's own side effect as evidence that the
    colleagues never answered.

    A clock-owned `until` on purpose: the batched judge cannot retire it (that window is answered by
    comparison in `retire_expired`), which is what keeps the waiter live past the verdict and makes
    the clock the site that queues.
    """
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    llm = FakeLLMClient(answer)
    cycle, working, registry = _cycle(tmp_path, clock=clock, llm=llm)
    await registry.join(_ORIGIN)
    activity = _blocked_on(_mid_body(), _waiting())
    activity.pending_conditions[0].declared_at = clock.now()
    working.activities[activity.id] = activity

    # A reply lands two seconds before the deadline and Reason dispatches its judgement.
    clock.advance(179)
    _reply(working)
    await cycle.strategies.observe.observe(cycle)
    await cycle.strategies.reason.reason(activity, working, cycle, _tick())
    assert activity.condition_batch, "the judgement was not dispatched"

    # It comes back and is applied while the window is still open, read by the real parser.
    await _settle()
    await cycle.strategies.observe.observe(cycle)
    assert activity.condition_verdict is not None, "the judgement never came back"
    await cycle.strategies.reason.reason(activity, working, cycle, _tick())
    assert activity.pending_conditions, "the clock's window is not the batched judge's to retire"
    assert activity.condition_fired == []

    # Only now does the window close, and the clock sweeps a waiter that was already decided.
    clock.advance(2)
    await cycle.strategies.observe.observe(cycle)

    assert activity.pending_conditions == []
    assert [f.on_expiry for f in activity.condition_fired] == ([True] if owed else [])
