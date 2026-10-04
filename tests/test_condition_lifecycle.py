"""The lifecycle of an expiry decision, asserted by what the agent actually *did* (ADR-0028).

`tests/test_expiry_branch.py` pins the construct and one reproduction per defect. This file exists
because that was not enough, and the reason it was not is worth stating: six review rounds each
found a different route by which unresolved evidence became a confirmed negative, each fix was
locally correct, and every one of those tests asserted a *flag* — `condition_fired`, `on_expiry`,
`expiry_settled`. A flag is one layer away from the failure anyone cares about, which is an agent
ordering a cab nobody asked for. So every test here drives the real cycle to the point where an
external operation is dispatched, and asserts the dispatch:

* `confirm` on the tool means the positive branch ran — a reply was read and it fired;
* `order_cab` means the negative branch ran — the window closed and nothing had fired;
* no invocation at all means the branch was refused, which is the correct outcome whenever the
  evidence could not be resolved.

The matrix is the ADR's transition table rather than the list of reported bugs: each row is a way
an evaluation can end, and both controls — a valid reply producing its `then`, a genuine timeout
producing its `otherwise` — are re-asserted alongside the failures, because the cheap way to pass a
failure test is to stop committing the branch at all.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeAdapter, FakeTool, FakeWorkspace
from sora._strategies.observe import _SIGNAL_RETENTION
from sora.action import default_action_registry
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, WorkspaceOrigin
from sora.llm import CompletionRequest
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from sora.perception import Message, Percept
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
    PendingCondition,
    PendingConditionState,
    Plan,
    Signal,
    SignalWait,
    Step,
    Until,
    changes_of,
)

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")
_MESSAGES = SignalWait(
    signal_name="state_changed", source="messaging", path="conversations", kind="added"
)
_THEN = "Confirm to the user who is ordering the cab"
_OTHERWISE = "Order a default cab"
# Which operation each branch's sub-plan reaches for, so a dispatched call names the branch.
_OPERATION = {_THEN: "confirm", _OTHERWISE: "order_cab"}

# A clock-owned window: `seconds` is declared, so `retire_expired` answers it by comparison and no
# retirement judge is ever called. That matters for the scripted seam below — every non-planning
# request it sees is the batched eligibility judgement, and nothing else can consume a scripted
# answer out from under a test.
_WINDOW = Until(text="three minutes have passed", seconds=180.0)

_FIRED = '{"fired": [0], "retired": []}'
_NOTHING_FIRED = '{"fired": [], "retired": []}'
_UNREADABLE = '{"fired": "unknown", "retired": []}'
# Readable about index 0, unreadable about an index that does not exist. Half an answer, and the
# half that parsed is a FIRING — the one shape where a failed verdict still carries something the
# runtime must act on exactly once.
_PARTIAL = '{"fired": [0, 7], "retired": []}'


class _ScriptedSeam:
    """An LLM seam that answers condition judgements from a script and plans every sub-goal into one
    identifiable operation call.

    Gated rather than canned: a single response string cannot express "the judge was unreadable the
    first time and answered the second", which is the whole of the retry lifecycle. Judgement
    answers are consumed in order (the last one sticks); a `BaseException` in the script is raised
    instead of returned, which is how the seam-error producer is reached without patching anything.
    """

    def __init__(self, judgements: Sequence[str | BaseException]) -> None:
        self.model = None
        self._judgements = list(judgements)
        self.requests: list[CompletionRequest] = []
        self.judged = 0
        self.planned: list[str] = []

    async def complete(self, request: CompletionRequest) -> str:
        self.requests.append(request)
        first = request.user.splitlines()[0]
        if first.startswith("Goal: "):
            goal = first.removeprefix("Goal: ").strip()
            self.planned.append(goal)
            assert goal in _OPERATION, f"unexpected sub-goal {goal!r}"
            return json.dumps(
                {
                    "steps": [
                        {
                            "action": "invoke",
                            "tool_id": "messaging",
                            "operation_name": _OPERATION[goal],
                            "params": {},
                        }
                    ]
                }
            )
        self.judged += 1
        assert self._judgements, "the seam was asked to judge more often than the script allows"
        answer = self._judgements[0] if len(self._judgements) == 1 else self._judgements.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


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


def _condition() -> PendingCondition:
    return PendingCondition(
        watch=_MESSAGES,
        when="one of the colleagues replies saying who will order the cab",
        then=_THEN,
        until=_WINDOW,
        otherwise=_OTHERWISE,
    )


def _waiting_agent(
    tmp_path: Path, judgements: Sequence[str | BaseException], *, ready: bool = False
) -> tuple[DecisionCycle, Activity, FakeTool, _ScriptedSeam, FakeClock]:
    """An agent that has sent the messages and is now blocked on the reply, with its window open.

    The body is exhausted on purpose — that is the real shape of the motivating goal, and it is what
    makes the branch reach Act in the same run rather than sitting behind unrun steps.

    `ready=True` starts the same activity READY with nothing to wake it instead, which is not a
    contrived state: an agent is READY whenever it has work in hand, and its windows are open the
    whole time. The difference that matters is that none of Observe's per-activity passes look at a
    READY activity's conditions, so nothing checks a gate before the retention trim runs.
    """
    clock = FakeClock(datetime(2024, 10, 15, 9, 0, tzinfo=UTC))
    tool = FakeTool(
        "messaging", invoke_results={"confirm": {"ok": True}, "order_cab": {"ok": True}}
    )
    workspace = FakeWorkspace("ws", _ORIGIN, [tool], clock=clock)
    registry = EnvironmentRegistry(adapters={_ORIGIN: FakeAdapter("fake", workspace)})
    working = WorkingMemory(registry=registry)
    seam = _ScriptedSeam(judgements)
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
        procedural=ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=seam),
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "episodic")),
    )
    condition = _condition()
    plan = Plan(id="p1", goal="ask who orders the cab", steps=[Step("wait", {})])
    activity = Activity(id="a", goal="ask who orders the cab", context={}, plan=plan, step_index=1)
    activity.pending_conditions = [
        PendingConditionState(condition=condition, declared_at=clock.now())
    ]
    if ready:
        activity.state = ActivityState.READY
    else:
        activity.state = ActivityState.BLOCKED
        activity.blocked_on = ConditionWait(watches=(condition.watch,))
    working.activities[activity.id] = activity
    return cycle, activity, tool, seam, clock


def _reply(cycle: DecisionCycle) -> None:
    """A colleague's reply landing on the watched path — evidence accepted into the window."""
    cycle.working.signals.append(
        Percept(
            "messaging",
            Signal("state_changed", {"changes": [Change(path="conversations", added=("c9",))]}),
            0.0,
        )
    )
    cycle.working.signals_appended += 1


def _unrelated_traffic(cycle: DecisionCycle, count: int) -> None:
    """Enough non-matching signals on the watched tool to push the retention cap over.

    Same signal name and same source as the reply, different path: what has to decline these is the
    watch, not the source, or the test would pass for the wrong reason.
    """
    for _ in range(count):
        cycle.working.signals.append(
            Percept(
                "messaging",
                Signal("state_changed", {"changes": [Change(path="drafts", added=("d1",))]}),
                0.0,
            )
        )
        cycle.working.signals_appended += 1


def _reply_retained(cycle: DecisionCycle) -> bool:
    """Is the reply still in the log? A test about eviction has to prove the eviction."""
    return any(
        percept.payload.name == "state_changed"
        and any(change.path == "conversations" for change in changes_of(percept.payload))
        for percept in cycle.working.signals
    )


async def _turns(cycle: DecisionCycle, count: int, clock: FakeClock, *, each: float = 0.0) -> None:
    """Run the real decision cycle `count` times, letting off-cycle calls land in between.

    `each` advances the domain clock per turn. The settle is what makes the off-cycle calls
    (ADR-0021) observable: a judgement fired this tick resolves through `inference_sink` and is
    picked up by a later Observe, so a lifecycle assertion needs several turns, not one.
    """
    for _ in range(count):
        for _ in range(6):
            await asyncio.sleep(0)
        if each:
            clock.advance(each)
        await cycle.tick()


# --------------------------------------------------------------------------------------------------
# The two controls. Everything below is only meaningful if these keep passing.
# --------------------------------------------------------------------------------------------------


async def test_a_read_reply_that_fires_runs_the_positive_branch(tmp_path: Path) -> None:
    """A reply arrives, the judge reads it, it fires: the `then` is planned and dispatched."""
    cycle, _activity, tool, seam, clock = _waiting_agent(tmp_path, [_FIRED])
    await cycle.registry.join(_ORIGIN)
    clock.advance(60)
    _reply(cycle)

    await _turns(cycle, 6, clock)

    assert seam.planned == [_THEN]
    assert [op for op, _ in tool.invocations] == ["confirm"]


async def test_a_genuine_timeout_runs_the_negative_branch(tmp_path: Path) -> None:
    """Nothing ever arrives and the clock closes the window: the `otherwise` is dispatched.

    The control that keeps every failure test honest — the cheapest way to pass all of them is to
    stop committing the branch at all, and this is the row that notices.
    """
    cycle, _activity, tool, seam, clock = _waiting_agent(tmp_path, [_NOTHING_FIRED])
    await cycle.registry.join(_ORIGIN)

    await _turns(cycle, 9, clock, each=40.0)

    assert seam.judged == 0, "a clock-owned window must not buy a judgement"
    assert seam.planned == [_OTHERWISE]
    assert [op for op, _ in tool.invocations] == ["order_cab"]


# --------------------------------------------------------------------------------------------------
# A judgement that was not an answer: retry, then decide
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "first",
    [
        pytest.param(_UNREADABLE, id="unreadable-fired-claim"),
        pytest.param("not JSON at all", id="unparseable"),
        pytest.param(RuntimeError("the provider hung up"), id="seam-error"),
    ],
)
async def test_an_unreadable_judgement_retries_and_the_retry_decides(
    tmp_path: Path, first: str | BaseException
) -> None:
    """The row the whole close-out turns on, and the one a rollback alone would have broken.

    A reply lands inside the window and the judgement over it comes back unusable — unreadable
    text, or no answer at all because the seam raised. That is *unresolved*, not negative: the
    change has to go back so the question can be asked again, and until it is asked again nothing
    may read the advanced cursor as an absence.

    Then the retry answers readably, and says the reply was not the awaited event. Now — and only
    now — the window may close on a confirmed negative and order the cab. Rolling the cursor back
    without also changing the terminal decision produced exactly this sequence and then refused the
    branch, because the first failure had already settled it: a genuine timeout silently producing
    nothing, which is the same invariant broken from the other side.

    The three parameters are the three producers of a non-answer, reaching one transition.
    """
    cycle, activity, tool, seam, clock = _waiting_agent(tmp_path, [first, _NOTHING_FIRED])
    await cycle.registry.join(_ORIGIN)
    clock.advance(60)
    _reply(cycle)

    # The first judgement fails, and the change is given back rather than consumed.
    await _turns(cycle, 3, clock)
    assert seam.judged == 1, "the first judgement was never dispatched"
    state = activity.pending_conditions[0]
    assert state.retried_after_failure is True, "the allowance was not spent"
    assert state.expiry_settled is False, "an unreadable answer is not a decision"
    assert tool.invocations == []

    # The retry is asked, answers "nothing fired", and the window then closes on that answer.
    await _turns(cycle, 8, clock, each=30.0)

    assert seam.judged == 2, "the change was not re-judged"
    assert seam.planned == [_OTHERWISE]
    assert [op for op, _ in tool.invocations] == ["order_cab"]


async def test_a_retry_that_fires_runs_the_positive_branch(tmp_path: Path) -> None:
    """The same retry, answering the other way: the reply *was* the awaited event.

    Pairs with the row above so that "give the change back" is pinned as a real re-judgement rather
    than a way of quietly dropping the evidence — a rollback that lost the change would show up
    here as a cab, not as a confirmation.
    """
    cycle, _activity, tool, seam, clock = _waiting_agent(tmp_path, [_UNREADABLE, _FIRED])
    await cycle.registry.join(_ORIGIN)
    clock.advance(60)
    _reply(cycle)

    await _turns(cycle, 8, clock)

    assert seam.judged == 2
    assert seam.planned == [_THEN]
    assert [op for op, _ in tool.invocations] == ["confirm"]


async def test_an_exhausted_retry_refuses_the_branch(tmp_path: Path) -> None:
    """A judge that cannot be read twice buys one retry and then nothing.

    Two things are pinned at once. The bound: the gate reopening on a rolled-back cursor must not
    re-dispatch for as long as the signal stays in retention, which is a model call per cycle. And
    the decision: once the retry is spent the evidence is unresolvable for good, so the window
    closes without a branch — a no-op, which is the degradation this construct is allowed, rather
    than a cab ordered over a reply nobody could read.
    """
    cycle, activity, tool, seam, clock = _waiting_agent(tmp_path, [_UNREADABLE])
    await cycle.registry.join(_ORIGIN)
    clock.advance(60)
    _reply(cycle)

    await _turns(cycle, 10, clock, each=30.0)

    assert seam.judged == 2, "the retry is one per failure, not one per cycle"
    assert seam.planned == []
    assert tool.invocations == []
    assert activity.pending_conditions == [], "the window still closed on time"


async def test_the_retry_allowance_is_per_failure_not_per_activity(tmp_path: Path) -> None:
    """A failure, a readable answer, then a second failure — which gets its own retry.

    The allowance is a budget against one unresolved question, so an answered question has to give
    it back. It used to be restored by the *seam* answering, which is not the same thing: an
    unreadable reply arrives over a working seam, so that reading refunded a retry it had never
    spent while permanently consuming the change it could not read.
    """
    cycle, activity, tool, seam, clock = _waiting_agent(
        tmp_path, [_UNREADABLE, _NOTHING_FIRED, _UNREADABLE, _NOTHING_FIRED]
    )
    await cycle.registry.join(_ORIGIN)
    clock.advance(30)
    _reply(cycle)

    await _turns(cycle, 4, clock)
    assert seam.judged == 2, "the first failure's retry was not asked"
    state = activity.pending_conditions[0]
    assert state.retried_after_failure is False, "a readable answer must return the allowance"

    # A second reply, a second failure — and a retry is available again.
    _reply(cycle)
    await _turns(cycle, 4, clock)
    assert seam.judged == 4, "the second failure had no retry of its own"
    assert tool.invocations == []


# --------------------------------------------------------------------------------------------------
# The two ways an evaluation disappears
# --------------------------------------------------------------------------------------------------


async def test_a_stalled_judgement_is_failed_and_its_late_answer_discarded(
    tmp_path: Path,
) -> None:
    """The stall watchdog gives up, and the real answer arrives afterwards.

    Observe drains delivered results *before* running the watchdog (and again straight after, which
    is how the synthetic error resolves in the same tick), so the watchdog only ever fires on a call
    that has delivered nothing yet. It is still reachable by a call that was about to answer — the
    real answer lands in the gap and meets the next tick's drain, where ADR-0021's identity guard
    drops it. Nothing cancels the underlying call, so "discarded" has to be a no-op on a batch some
    other path already resolved, not a second resolution of it.

    `{"fired": [0]}` is the late answer on purpose: if the discard were *not* a no-op it would fire
    the positive branch here, which is the loudest possible failure.
    """
    cycle, activity, tool, seam, clock = _waiting_agent(tmp_path, [_FIRED, _NOTHING_FIRED])
    await cycle.registry.join(_ORIGIN)
    clock.advance(60)
    _reply(cycle)

    # Dispatch the judgement, then expire it before its result is drained.
    await cycle.tick()
    assert activity.condition_batch, "the judgement was not dispatched"
    observe = cycle.strategies.observe
    assert isinstance(observe, DefaultObserveStrategy)
    observe._inference_deadline = 0.0  # the stall this watchdog exists for, without the wait
    await cycle.tick()
    observe._inference_deadline = None  # the sub-goal plan is allowed to finish
    assert activity.condition_batch == [], "the stall was not resolved as a failure"

    state = activity.pending_conditions[0]
    assert state.retried_after_failure is True
    assert state.expiry_settled is False, "a stall decides nothing on its own"

    # The retry is asked and answers; the branch follows that answer, never the discarded one.
    await _turns(cycle, 8, clock, each=30.0)
    assert seam.planned == [_OTHERWISE]
    assert [op for op, _ in tool.invocations] == ["order_cab"]


async def test_an_abandoned_judgement_does_not_strand_its_waiter(tmp_path: Path) -> None:
    """A replan discards an in-flight condition judgement, so no verdict will ever be applied.

    Distinct from the late-result discard above, and the distinction is the defect: a late result
    lands on a batch something else already resolved, while this removes the only evaluation that
    was ever going to resolve one. `condition_batch` is cleared by nothing but an applied verdict,
    so left alone the waiter sits unresolved forever — an expiry over it defers rather than decides,
    and the next dispatch leaves it outstanding nowhere with its cursor still advanced past the
    change nobody read. A sweep then reads that advance as the absence that authorizes the cab.

    What it must do instead is take the ordinary failure transition, which is why the assertion is
    both halves: the change comes back (so no absence can be claimed off it), and the window still
    closes on time rather than hanging.
    """
    cycle, activity, tool, seam, clock = _waiting_agent(tmp_path, [_NOTHING_FIRED])
    await cycle.registry.join(_ORIGIN)
    clock.advance(60)
    _reply(cycle)

    await cycle.tick()
    assert activity.condition_batch, "the judgement was not dispatched"
    state = activity.pending_conditions[0]
    judged_through = state.evaluated_through

    activity.reset_for_replan()  # the world moved under the plan; the judgement goes with it

    assert activity.condition_batch == [], "the abandoned batch was left outstanding"
    assert state.evaluated_through < judged_through, "the change was not given back"
    assert state.retried_after_failure is True
    assert state.expiry_settled is False, "an abandoned call decides nothing"

    # The reply is now unjudged evidence again, so the closing window may not claim an absence over
    # it — and must not hang on the verdict that can never arrive.
    await _turns(cycle, 8, clock, each=30.0)
    assert activity.pending_conditions == [], "the window did not close"
    assert "order_cab" not in [op for op, _ in tool.invocations]


# --------------------------------------------------------------------------------------------------
# Evidence accepted, not yet resolved
# --------------------------------------------------------------------------------------------------


async def test_a_second_accepted_change_keeps_the_waiter_unresolved(tmp_path: Path) -> None:
    """Two replies land; the judgement covers the first. An empty answer resolves only that one.

    The cursor advances past exactly the change being judged and no further, so the second reply is
    still accepted-but-unjudged when the first verdict comes back empty. "Nothing fired" is an
    answer about the first reply alone, and reading it as the window's absence would commit the
    branch over evidence nobody has looked at.
    """
    cycle, activity, tool, seam, clock = _waiting_agent(tmp_path, [_NOTHING_FIRED])
    await cycle.registry.join(_ORIGIN)
    clock.advance(60)
    _reply(cycle)
    _reply(cycle)

    await _turns(cycle, 3, clock)
    assert seam.judged >= 1

    # Close the window. The second change is either still unjudged (so the branch declines) or has
    # since been judged readably — never committed over an unread one.
    await _turns(cycle, 6, clock, each=30.0)
    assert activity.pending_conditions == []
    if "order_cab" in [op for op, _ in tool.invocations]:
        assert seam.judged == 2, "the branch committed with a change left unjudged"


async def test_a_partial_answer_fires_once_and_is_not_retried(tmp_path: Path) -> None:
    """A verdict that fires index 0 and claims a nonexistent index 7: one `confirm`, no retry.

    The failed flag and the firing are both real here. Resolving the failure over the whole batch
    rewound the very waiter the same verdict had just fired, which left its change eligible again;
    the retry was then asked the identical question about the identical change, answered the same
    way, and the agent confirmed twice. Two external operations out of one reply — the duplicate
    `otherwise` of round one, reached through the positive branch instead.

    The script holds a single answer, so it is also what any re-judgement would receive: a second
    `confirm` is the regression, and `seam.judged` says whether the gate was re-opened at all.
    """
    cycle, activity, tool, seam, clock = _waiting_agent(tmp_path, [_PARTIAL])
    await cycle.registry.join(_ORIGIN)
    clock.advance(60)
    _reply(cycle)

    await _turns(cycle, 12, clock)

    assert [op for op, _ in tool.invocations] == ["confirm"]
    assert seam.judged == 1, "the answered waiter was re-judged over a change already consumed"
    state = activity.pending_conditions[0]
    assert state.ever_fired is True
    assert state.retried_after_failure is False, "a usable answer spends no retry"


async def test_evidence_evicted_before_judgement_refuses_the_branch(tmp_path: Path) -> None:
    """The reply is recognized, retention drops it before any judgement, and the window closes.

    `wm.signals` is a capped broadcast log, and the cap is the only eviction — so a reply that lands
    inside the window and is followed by enough unrelated traffic is gone before the activity is
    free to judge it. Observe recognizes the match (that pass runs ahead of the trim) and the trim
    then removes it, after which nothing re-opens the gate: the eligibility sweep scans the same
    evicted log.

    What made that a wrong action rather than a lost judgement is that the expiry sweep asked the
    same evicted log whether any evidence was outstanding. It said no, and "no evidence" over a
    closed window is the authorization for `otherwise` — so the agent ordered a cab having bought
    zero judgements, with the awaited reply having arrived. Acceptance is durable now, so the window
    closes on a refusal: no branch, and no operation.
    """
    cycle, activity, tool, seam, clock = _waiting_agent(tmp_path, [_NOTHING_FIRED])
    await cycle.registry.join(_ORIGIN)
    _reply(cycle)
    _unrelated_traffic(cycle, _SIGNAL_RETENTION + 32)

    await _turns(cycle, 9, clock, each=40.0)

    assert not activity.pending_conditions, "the clock-owned window should have retired"
    assert not _reply_retained(cycle), "the reply was not evicted — the test proves nothing"
    assert seam.judged == 0, "nothing was judgeable, so nothing should have been judged"
    assert seam.planned == [], "no branch is available over evidence nobody could read"
    assert [op for op, _ in tool.invocations] == []


async def test_evidence_is_accepted_even_while_the_agent_is_busy(tmp_path: Path) -> None:
    """The window is open on a READY activity, so nothing checks its gate before the trim.

    The first version of the retention fix recorded acceptance inside the gate checks themselves,
    which made it conditional on scheduling: Observe's eligibility sweep only looks at activities
    BLOCKED on a `ConditionWait`, and the expiry sweep only at windows it is already closing. An
    agent that is READY — which is to say, an agent with work in hand — is neither, so the reply
    was recognized by nothing at all and the trim took the only record of it. The window then
    closed on an empty log and ordered the cab, having bought no judgement.

    Acceptance is reconciled for every live waiter immediately before the trim now, so a reply that
    lands while the agent is busy still costs the branch rather than authorizing it.
    """
    cycle, activity, tool, seam, clock = _waiting_agent(tmp_path, [_NOTHING_FIRED], ready=True)
    await cycle.registry.join(_ORIGIN)
    _reply(cycle)
    _unrelated_traffic(cycle, _SIGNAL_RETENTION + 32)

    await _turns(cycle, 9, clock, each=40.0)

    assert not _reply_retained(cycle), "the reply was not evicted — the test proves nothing"
    assert seam.judged == 0, "nothing was judgeable, so nothing should have been judged"
    assert seam.planned == [], "no branch is available over evidence nobody could read"
    assert [op for op, _ in tool.invocations] == []


async def test_a_later_judgement_cannot_erase_evidence_lost_to_retention(tmp_path: Path) -> None:
    """A reply is lost to the trim, a later change is judged negative, and the window still refuses.

    The evaluation cursor is a single scalar, so this is the sequence that defeats a high-water
    mark: the lost reply sits at some sequence the cursor has not reached, then a later matching
    change is judged, and `_Match.marks_for` moves the cursor past *both*. The acceptance mark it
    meets then reads as "everything this window accepted has been answered" — and a negative answer
    over a closed window is the authorization. One reply, one eviction, one later change was enough
    to order the cab.

    So the loss is decided when it happens, while the entry is still in hand, and nothing clears it:
    a later judgement about later evidence says nothing about the change that went missing. The
    second reply is judged honestly here — `seam.judged == 1` — and the branch is still refused.
    """
    cycle, activity, tool, seam, clock = _waiting_agent(tmp_path, [_NOTHING_FIRED], ready=True)
    await cycle.registry.join(_ORIGIN)
    _reply(cycle)
    _unrelated_traffic(cycle, _SIGNAL_RETENTION + 32)

    # One tick: the reply is accepted and evicted, and the branch is refused on the spot.
    await _turns(cycle, 1, clock)
    assert not _reply_retained(cycle), "the reply was not evicted — the test proves nothing"
    state = activity.pending_conditions[0]
    assert state.expiry_settled is True, "evidence lost to the trim did not decide the branch"

    # Now a second matching change, which the judge reads and answers negatively. That is a true
    # statement about the second change and says nothing about the first.
    _reply(cycle)
    await _turns(cycle, 9, clock, each=40.0)

    assert seam.judged == 1, "the second change should have been judged exactly once"
    assert seam.planned == [], "a later negative answer cannot resurrect a refused branch"
    assert [op for op, _ in tool.invocations] == []


async def _dispatch_then_evict(
    cycle: DecisionCycle, activity: Activity, clock: FakeClock
) -> PendingConditionState:
    """Get a judgement out over the reply, then lose the reply to the trim while it is still out.

    The window this exercises is one tick wide and cannot be reached with `_turns`, which settles
    the event loop before every tick: the first tick dispatches the judgement and is deliberately
    *not* settled, so the traffic arrives while the call is outstanding and the next Observe does
    the reconciliation and the trim with the batch still in flight.
    """
    _reply(cycle)
    await cycle.tick()
    assert activity.condition_batch, "no judgement was dispatched — the test proves nothing"
    state = activity.pending_conditions[0]
    assert state.evaluated_through > state.fired_from_signals, (
        "the fire-time advance did not happen"
    )
    _unrelated_traffic(cycle, _SIGNAL_RETENTION + 32)
    await _turns(cycle, 1, clock)
    assert not _reply_retained(cycle), "the reply was not evicted — the test proves nothing"
    return state


async def test_a_seam_error_over_evidence_already_evicted_refuses_the_branch(
    tmp_path: Path,
) -> None:
    """The judgement is dispatched, its change is evicted, and the call then fails.

    This is the gap the fire-time cursor advance opens, and it is narrow enough to be worth
    spelling out: the advance happens when the call goes OUT, so from that moment the change being
    judged sits below the evaluation cursor — which is where the pre-trim reconciliation stops
    looking. Evicting it there recorded nothing at all. The failure transition then rewound the
    cursor, which gives back a sequence number and not a log entry, so the waiter was re-judgeable
    in form only; and when a later matching change was judged, the single scalar cursor advanced
    past both and the acceptance mark read as fully answered. The window expired into `order_cab`
    having never once been told what the reply said.
    """
    cycle, activity, tool, seam, clock = _waiting_agent(
        tmp_path, [RuntimeError("seam down"), _NOTHING_FIRED]
    )
    await cycle.registry.join(_ORIGIN)
    state = await _dispatch_then_evict(cycle, activity, clock)
    assert state.expiry_settled is True, "a failure over lost evidence did not decide the branch"

    # A second matching change, read and answered negatively. True about that change, and no
    # statement at all about the one nobody could read.
    _reply(cycle)
    await _turns(cycle, 9, clock, each=40.0)

    assert seam.planned == [], "a later negative answer cannot resurrect a refused branch"
    assert [op for op, _ in tool.invocations] == []


async def test_a_malformed_answer_over_evicted_evidence_refuses_the_branch(
    tmp_path: Path,
) -> None:
    """The other producer of a non-answer reaches the same place: `{"fired": "unknown"}`.

    Worth its own row rather than folding into the seam-error one, because the two arrive by
    different routes — a raised exception parks an empty verdict, an unreadable `fired` parses to
    one — and only the second proves the parser's failure flag is what drives the refusal.
    """
    cycle, activity, tool, seam, clock = _waiting_agent(tmp_path, [_UNREADABLE, _NOTHING_FIRED])
    await cycle.registry.join(_ORIGIN)
    state = await _dispatch_then_evict(cycle, activity, clock)
    assert state.expiry_settled is True, "a failure over lost evidence did not decide the branch"

    _reply(cycle)
    await _turns(cycle, 9, clock, each=40.0)

    assert seam.planned == [], "a later negative answer cannot resurrect a refused branch"
    assert [op for op, _ in tool.invocations] == []


async def test_a_usable_answer_still_resolves_its_own_evicted_evidence(tmp_path: Path) -> None:
    """The control for the two rows above: eviction under judgement is not itself a refusal.

    A judgement that comes back readable answered the question it was asked, about the change it
    was given, so whether retention still holds that change decides nothing — the window is free to
    expire into its negative branch on the ordinary basis that nothing fired. Without this row the
    cheap way to pass the two above is to refuse on any eviction at all, which would turn the
    construct's normal near-deadline case into silence.
    """
    cycle, activity, tool, seam, clock = _waiting_agent(tmp_path, [_NOTHING_FIRED])
    await cycle.registry.join(_ORIGIN)
    state = await _dispatch_then_evict(cycle, activity, clock)
    assert state.dispatched_evidence_lost is False, "a usable answer should have forgiven the loss"

    await _turns(cycle, 9, clock, each=40.0)

    assert seam.planned == [_OTHERWISE]
    assert [op for op, _ in tool.invocations] == ["order_cab"]
