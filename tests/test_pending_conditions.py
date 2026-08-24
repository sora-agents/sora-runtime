"""Declared pending conditions (ADR-0022) and undeclared-relevance recovery (ADR-0026).

The failure these exist for: an agent scheduled an event, emailed the attendee, confirmed to the
user and TERMINATED — then the attendee replied "can't make it, Thursday instead" and nothing was
left alive to read it. The plan's own prose had stated the conditional; its body encoded none of it,
because nothing in the representation distinguished *this goal is finished* from *this goal's body
is finished*.

Layer 1 gives a plan somewhere to declare that, and blocks instead of terminating. Layer 3 covers
the case the planner declares nothing at all — which is what actually happened — by judging changes
that opened no declared gate against recent episodes, and asking the user before acting.

The invariants pinned hardest are the cost controls, because they are what make either layer
affordable and are the easiest thing to quietly break: a condition is only ever judged against a
change that mechanically matched its declared watch, and it is never judged twice for the same
signal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fakes import FakeAdapter, FakeLLMClient, FakeTool, FakeWorkspace, ScriptedTransport
from sora.action import default_action_registry
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, WorkspaceOrigin
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    PerceptSnapshot,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
    pending_from_raw,
)
from sora.perception import Percept
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReasonStrategy,
    DefaultReflectStrategy,
    DefaultRelevanceJudge,
    DefaultSituateStrategy,
    Strategies,
)
from sora.transport import MessageTransport
from sora.types import (
    Change,
    ConditionWait,
    InputWait,
    ObservableProperty,
    PendingCondition,
    PendingConditionState,
    Plan,
    Signal,
    SignalWait,
    Step,
)

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")


class _NullTransport:
    async def send(self, to: str, content: dict[str, Any]) -> None: ...

    async def receive(self) -> Any:
        return
        yield  # pragma: no cover — never reached; makes this an async generator


def _cycle(
    tmp_path: Path, procedural: ProceduralMemory, transport: MessageTransport | None = None
) -> tuple[DecisionCycle, WorkingMemory]:
    tool = FakeTool("insim:are/Emails")
    registry = EnvironmentRegistry(
        adapters={_ORIGIN: FakeAdapter("fake", FakeWorkspace("ws", _ORIGIN, [tool]))}
    )
    working = WorkingMemory(registry=registry)
    transport = transport if transport is not None else _NullTransport()
    cycle = DecisionCycle(
        strategies=Strategies(
            observe=DefaultObserveStrategy(),
            reflect=DefaultReflectStrategy(),
            situate=DefaultSituateStrategy(),
            reason=DefaultReasonStrategy(),
            act=DefaultActStrategy(),
        ),
        communication=transport,
        actions=default_action_registry(),
        registry=registry,
        working=working,
        semantic=SemanticMemory(FileMemoryBackend(tmp_path / "semantic")),
        procedural=procedural,
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "episodic")),
    )
    return cycle, working


_INBOX = SignalWait(
    signal_name="state_changed", source="insim:are/Emails", path="folders.INBOX.emails"
)


def _condition(watch: SignalWait = _INBOX) -> PendingCondition:
    return PendingCondition(
        watch=watch,
        when="the attendee replies that the date does not work",
        then="Rebook for the day the attendee proposes",
        until="the booking has taken place",
    )


def _exhausted(plan_pending: tuple[PendingCondition, ...] = ()) -> Activity:
    """An activity whose body has just run out — the moment the old code terminated it."""
    plan = Plan(
        id="p1", goal="book it and tell them", steps=[Step("wait", {})], pending=plan_pending
    )
    return Activity(id="a1", goal="book it and tell them", context={}, plan=plan, step_index=1)


def _signal(working: WorkingMemory, path: str, source: str = "insim:are/Emails") -> None:
    working.signals.append(
        Percept(
            source, Signal("state_changed", {"changes": [Change(path=path, added=("e9",))]}), 0.0
        )
    )
    working.signals_appended += 1


# --------------------------------------------------------------------------------------------------
# Parsing: what the planner emits
# --------------------------------------------------------------------------------------------------


def test_pending_condition_parses_from_planner_json() -> None:
    cond = pending_from_raw(
        {
            "watch": {"signal": "state_changed", "source": "t1", "path": "folders.INBOX.emails"},
            "when": "they reply",
            "then": "rebook",
            "until": "it happened",
        }
    )
    assert cond is not None
    assert cond.watch == SignalWait("state_changed", "t1", "folders.INBOX.emails")
    assert (cond.when, cond.then, cond.until) == ("they reply", "rebook", "it happened")


def test_condition_without_a_watch_is_dropped() -> None:
    # A gate is REQUIRED. Without one the condition would have to be evaluated against every signal
    # the agent ever sees — the unbounded keep-alive this design exists to reject, wearing a field
    # name. Dropping lands the run back on "terminate when the body ends", which is honest; keeping
    # it would create a wait that can never fire.
    assert pending_from_raw({"when": "they reply", "then": "rebook"}) is None
    assert pending_from_raw({"watch": {}, "when": "they reply", "then": "rebook"}) is None


def test_malformed_condition_is_dropped_not_raised() -> None:
    # The body is the part that does the work; failing a whole plan because an optional
    # forward-looking clause was mis-shaped would trade a partial success for a total failure.
    assert pending_from_raw({"watch": {"signal": "s"}, "when": "", "then": "x"}) is None
    assert pending_from_raw("not a dict") is None  # type: ignore[arg-type]


async def test_infer_parses_pending_alongside_steps(tmp_path: Path) -> None:
    response = json.dumps(
        {
            "steps": [{"action": "focus", "tool_id": "t1"}],
            "pending": [
                {
                    "watch": {"signal": "state_changed", "path": "folders.INBOX.emails"},
                    "when": "they reply",
                    "then": "rebook",
                }
            ],
        }
    )
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path), llm=FakeLLMClient(response))
    plan = await procedural.infer(Activity(id="a", goal="g", context={}), {})
    assert len(plan.steps) == 1
    assert len(plan.pending) == 1 and plan.pending[0].then == "rebook"


async def test_a_plan_that_declares_nothing_is_unchanged(tmp_path: Path) -> None:
    # Most goals are unconditional; the feature existing must cost them nothing.
    procedural = ProceduralMemory(
        FileMemoryBackend(tmp_path), llm=FakeLLMClient(json.dumps({"steps": []}))
    )
    plan = await procedural.infer(Activity(id="a", goal="g", context={}), {})
    assert plan.pending == ()


# --------------------------------------------------------------------------------------------------
# Layer 1: block instead of terminate
# --------------------------------------------------------------------------------------------------


async def test_exhausted_body_with_a_condition_blocks_instead_of_terminating(
    tmp_path: Path,
) -> None:
    # The core fix. Before this, an exhausted body meant TERMINATED and the reply had nothing to
    # reach.
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    activity = _exhausted((_condition(),))
    working.activities[activity.id] = activity

    await cycle.strategies.reflect.reflect(activity, working, cycle, _tick())

    assert activity.state is ActivityState.BLOCKED
    assert isinstance(activity.blocked_on, ConditionWait)
    assert activity.blocked_on.watches == (_INBOX,)


async def test_exhausted_body_without_conditions_still_terminates(tmp_path: Path) -> None:
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    activity = _exhausted()
    working.activities[activity.id] = activity

    await cycle.strategies.reflect.reflect(activity, working, cycle, _tick())

    assert activity.state is ActivityState.TERMINATED


async def test_blocking_records_no_episode_yet(tmp_path: Path) -> None:
    # An episode is a claim about how work ENDED. Writing one for an activity that is still waiting
    # would be a claim about an outcome that has not happened.
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    activity = _exhausted((_condition(),))
    working.activities[activity.id] = activity

    await cycle.strategies.reflect.reflect(activity, working, cycle, _tick())

    assert await cycle.episodic.consult(activity) == []


async def test_conditions_are_live_before_the_body_finishes(tmp_path: Path) -> None:
    # Lifted at plan install, not when the body runs out — so a reply that BEATS the last step is
    # watched for, rather than being discovered only after the plan happens to end.
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    plan = Plan(
        id="p", goal="g", steps=[Step("wait", {}), Step("wait", {})], pending=(_condition(),)
    )
    activity = Activity(id="a1", goal="g", context={}, plan=plan, step_index=0)  # mid-body
    working.activities[activity.id] = activity

    await cycle.strategies.reflect.reflect(activity, working, cycle, _tick())

    assert len(activity.pending_conditions) == 1
    assert activity.state is ActivityState.READY  # still running its body, not blocked


async def test_lifting_is_idempotent_across_cycles(tmp_path: Path) -> None:
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    activity = _exhausted((_condition(),))
    working.activities[activity.id] = activity

    for _ in range(3):
        activity.state = ActivityState.READY
        await cycle.strategies.reflect.reflect(activity, working, cycle, _tick())

    assert len(activity.pending_conditions) == 1


async def test_a_new_condition_ignores_the_signal_backlog(tmp_path: Path) -> None:
    # A condition declared now cannot be about a change from before it existed, and the retention
    # log holds hundreds. Starting the mark at zero would re-judge the whole backlog immediately.
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    _signal(working, "folders.INBOX.emails")  # arrived BEFORE the plan existed
    activity = _exhausted((_condition(),))
    working.activities[activity.id] = activity

    await cycle.strategies.reflect.reflect(activity, working, cycle, _tick())

    assert activity.pending_conditions[0].evaluated_through == working.signals_appended


# --------------------------------------------------------------------------------------------------
# Layer 1: the mechanical gate — what reaches a model and what never does
# --------------------------------------------------------------------------------------------------


def _blocked_with_condition(working: WorkingMemory, watch: SignalWait = _INBOX) -> Activity:
    activity = _exhausted()
    activity.state = ActivityState.BLOCKED
    activity.pending_conditions = [
        PendingConditionState(condition=_condition(watch), evaluated_through=0)
    ]
    activity.blocked_on = ConditionWait(watches=(watch,))
    working.activities[activity.id] = activity
    return activity


async def test_matching_change_resumes_the_blocked_activity(tmp_path: Path) -> None:
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    activity = _blocked_with_condition(working)
    _signal(working, "folders.INBOX.emails")

    await cycle.strategies.observe.observe(cycle)

    assert activity.state is ActivityState.READY
    assert activity.blocked_on is None


async def test_the_agents_own_outbound_write_does_not_wake_it(tmp_path: Path) -> None:
    # THE cost control, and the general replacement for a domain efference filter: the agent's own
    # send lands in SENT, an inbound reply in INBOX. Same signal name, same source, told apart by
    # where they landed — with no reasoning about which changes the agent caused.
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    activity = _blocked_with_condition(working)
    _signal(working, "folders.SENT.emails")

    await cycle.strategies.observe.observe(cycle)

    assert activity.state is ActivityState.BLOCKED


async def test_a_different_tool_does_not_wake_it(tmp_path: Path) -> None:
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    activity = _blocked_with_condition(working)
    _signal(working, "folders.INBOX.emails", source="insim:are/Calendar")

    await cycle.strategies.observe.observe(cycle)

    assert activity.state is ActivityState.BLOCKED


async def test_an_already_judged_signal_does_not_wake_it_again(tmp_path: Path) -> None:
    # Without the per-condition mark the activity would resume, find nothing new, re-block, and
    # spin — burning a model call every cycle for as long as the signal stays in retention.
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    activity = _blocked_with_condition(working)
    _signal(working, "folders.INBOX.emails")
    activity.pending_conditions[0].evaluated_through = working.signals_appended  # already judged

    await cycle.strategies.observe.observe(cycle)

    assert activity.state is ActivityState.BLOCKED


async def test_reflect_does_not_undo_the_resume_observe_just_did(tmp_path: Path) -> None:
    # The livelock. Observe resumes a gate-opened activity so Reason can judge it, but Reflect runs
    # next in the SAME cycle, saw an exhausted body with pending conditions, and blocked it straight
    # back — before Situate could select it. Both phases passed their own tests; the defect lived
    # only in their composition, which is why this test drives them together.
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    activity = _blocked_with_condition(working)
    _signal(working, "folders.INBOX.emails")

    await cycle.strategies.observe.observe(cycle)
    assert activity.state is ActivityState.READY  # Observe woke it
    await cycle.strategies.reflect.reflect(activity, working, cycle, _tick())

    assert activity.state is ActivityState.READY  # ...and Reflect left it awake to be judged
    assert activity.blocked_on is None


async def test_the_gate_opening_reaches_the_judgement_instead_of_spinning(tmp_path: Path) -> None:
    # End to end over two cycles, which is what the log showed going wrong ~1400 times: cycle one
    # must actually spend the judgement (advancing the marks), and cycle two must then find nothing
    # eligible and settle back to BLOCKED rather than waking on the same signal again.
    llm = FakeLLMClient(json.dumps({"fired": [], "retired": []}))
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm))
    activity = _blocked_with_condition(working)
    _signal(working, "folders.INBOX.emails")

    await cycle.strategies.observe.observe(cycle)
    await cycle.strategies.reflect.reflect(activity, working, cycle, _tick())
    # Situate is the phase the livelock actually starved: it only ever selects a READY activity, so
    # a Reflect that re-blocked meant Reason was never handed the activity at all. Go through it
    # rather than calling reason() directly, or the test proves nothing about the spin.
    situated = await cycle.strategies.situate.situate([activity], working, cycle, _tick())
    assert situated.activity is activity
    await cycle.strategies.reason.reason(activity, working, cycle, situated)
    await _settle()

    assert len(llm.calls) == 1  # the judgement the spin never reached
    assert activity.pending_conditions[0].evaluated_through == working.signals_appended

    await cycle.strategies.observe.observe(cycle)  # second cycle: same signal, now judged
    await cycle.strategies.reflect.reflect(activity, working, cycle, _tick())
    # Reflect leaves it READY here, because that same Observe parked the (empty) verdict and Reason
    # is the phase that applies it. Settling back to BLOCKED is Reason's call, one phase later.
    situated = await cycle.strategies.situate.situate([activity], working, cycle, _tick())
    await cycle.strategies.reason.reason(activity, working, cycle, situated)

    assert activity.state is ActivityState.BLOCKED
    assert len(llm.calls) == 1  # and no second call for a signal already judged


async def test_an_exhausted_body_with_no_open_gate_still_blocks(tmp_path: Path) -> None:
    # The guard is narrow: only an UNJUDGED open gate keeps the activity awake. With nothing
    # eligible, an exhausted body must still park on its conditions rather than sit ready forever.
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    activity = _exhausted()
    activity.pending_conditions = [
        PendingConditionState(condition=_condition(), evaluated_through=0)
    ]
    working.activities[activity.id] = activity
    _signal(working, "folders.SENT.emails")  # never passes the gate

    await cycle.strategies.reflect.reflect(activity, working, cycle, _tick())

    assert activity.state is ActivityState.BLOCKED
    assert isinstance(activity.blocked_on, ConditionWait)


async def test_evaluation_fires_one_call_for_several_eligible_conditions(tmp_path: Path) -> None:
    # Batched: the gate is what makes this affordable, and batching is what keeps the saving as
    # conditions accumulate. One call, not one per condition.
    llm = FakeLLMClient(json.dumps({"fired": [], "retired": []}))
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm))
    activity = _exhausted()
    activity.pending_conditions = [
        PendingConditionState(condition=_condition(), evaluated_through=0),
        PendingConditionState(
            condition=PendingCondition(watch=_INBOX, when="something else", then="do that"),
            evaluated_through=0,
        ),
    ]
    working.activities[activity.id] = activity
    _signal(working, "folders.INBOX.emails")

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())
    await _settle()

    assert len(llm.calls) == 1
    assert activity.condition_batch and len(activity.condition_batch) == 2


async def test_marks_advance_at_fire_time_so_the_same_signal_is_not_rejudged(
    tmp_path: Path,
) -> None:
    llm = FakeLLMClient(json.dumps({"fired": [], "retired": []}))
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm))
    activity = _exhausted()
    activity.pending_conditions = [
        PendingConditionState(condition=_condition(), evaluated_through=0)
    ]
    working.activities[activity.id] = activity
    _signal(working, "folders.INBOX.emails")

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert activity.pending_conditions[0].evaluated_through == working.signals_appended


async def test_no_eligible_condition_costs_no_call(tmp_path: Path) -> None:
    # The common case by construction. A procedural with no LLM raises if anything escalates, so
    # this proves nothing did.
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    activity = _exhausted()
    activity.pending_conditions = [
        PendingConditionState(condition=_condition(), evaluated_through=0)
    ]
    working.activities[activity.id] = activity
    _signal(working, "folders.SENT.emails")  # never passes the gate

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())  # must not raise


# --------------------------------------------------------------------------------------------------
# Layer 1: applying the verdict
# --------------------------------------------------------------------------------------------------


async def test_a_fired_condition_pursues_its_then_as_a_subgoal(tmp_path: Path) -> None:
    llm = FakeLLMClient(json.dumps({"steps": []}))
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm))
    activity = _exhausted()
    state = PendingConditionState(condition=_condition(), evaluated_through=0)
    activity.pending_conditions = [state]
    activity.condition_batch = [state]
    activity.condition_verdict = _verdict(fired=(0,))
    working.activities[activity.id] = activity

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    # Pursued through the ordinary deliberative sub-goal path — `then` is a goal, planned fresh when
    # the moment comes, which is why it is prose rather than steps.
    assert activity.pending_inference is not None
    assert activity.pending_inference.kind == "subgoal"


async def test_nothing_fired_goes_back_to_waiting(tmp_path: Path) -> None:
    # Must NOT fall through to Reflect, which would see an exhausted plan and terminate outright.
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    activity = _exhausted()
    state = PendingConditionState(condition=_condition(), evaluated_through=99)
    activity.pending_conditions = [state]
    activity.condition_batch = [state]
    activity.condition_verdict = _verdict()
    working.activities[activity.id] = activity

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert activity.state is ActivityState.BLOCKED
    assert isinstance(activity.blocked_on, ConditionWait)


def _other_condition(then: str) -> PendingCondition:
    return PendingCondition(watch=_INBOX, when="a team member replies", then=then, until="never")


async def test_every_fired_condition_is_pursued_not_just_the_first(tmp_path: Path) -> None:
    """A batched verdict fires plural on purpose — one call judges every eligible condition, and a
    single reply can satisfy two gates. Dropping the extras is silent and unrecoverable: every
    judged condition's mark was advanced at fire time, so the signal that opened those gates is no
    longer eligible and nothing re-fires them."""
    llm = FakeLLMClient(json.dumps({"steps": []}))
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm))
    activity = _exhausted()
    first = PendingConditionState(condition=_condition(), evaluated_through=99)
    second = PendingConditionState(
        condition=_other_condition("Tell the user the schedule slipped"), evaluated_through=99
    )
    activity.pending_conditions = [first, second]
    activity.condition_batch = [first, second]
    activity.condition_verdict = _verdict(fired=(0, 1))
    working.activities[activity.id] = activity

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())
    await _settle()

    assert activity.pending_inference is not None  # planning the first `then`
    assert first.condition.then in llm.calls[-1][1]

    # The first `then` has been planned and run: its sub-plan popped, so the body is exhausted
    # again and nothing new has arrived. The second fired condition is what is left to do.
    activity.pending_inference = None
    activity.state = ActivityState.READY
    await cycle.strategies.reason.reason(activity, working, cycle, _tick())
    await _settle()

    assert second.condition.then in llm.calls[-1][1]


async def test_a_queued_fired_condition_does_not_nest_inside_the_first(tmp_path: Path) -> None:
    """In order and at the same depth. While the first `then`'s sub-plan is still running the body
    is not exhausted, so pursuing the second there would push a frame inside a frame — inverting the
    order the verdict listed them in and walking the activity toward the recursion breaker."""
    llm = FakeLLMClient(json.dumps({"steps": []}))
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm))
    activity = _exhausted()
    first = PendingConditionState(condition=_condition(), evaluated_through=99)
    second = PendingConditionState(
        condition=_other_condition("Tell the user the schedule slipped"), evaluated_through=99
    )
    activity.pending_conditions = [first, second]
    activity.condition_batch = [first, second]
    activity.condition_verdict = _verdict(fired=(0, 1))
    working.activities[activity.id] = activity
    await cycle.strategies.reason.reason(activity, working, cycle, _tick())
    await _settle()
    calls_after_first = len(llm.calls)
    assert calls_after_first == 1  # the first `then` was planned

    # The first `then`'s sub-plan landed and is mid-body: one step left to run, nothing exhausted.
    activity.pending_inference = None
    activity.state = ActivityState.READY
    activity.parent_frames.append((activity.plan, 1))  # type: ignore[arg-type]  # plan is set
    activity.plan = Plan(id="sub", goal="rebook", steps=[Step("wait", {})])
    activity.step_index = 0

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())
    await _settle()

    assert len(llm.calls) == calls_after_first  # no second `then` planned on top of the first


async def test_reflect_does_not_terminate_over_a_fired_condition(tmp_path: Path) -> None:
    """The queue outlives the condition that produced it — a fired condition is usually retired in
    the same verdict, so `pending_conditions` can be empty while committed work is still queued.
    Terminating there would write a success episode for a goal with a `then` still owed."""
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    activity = _exhausted()
    activity.condition_fired = [PendingConditionState(condition=_condition(), evaluated_through=99)]
    working.activities[activity.id] = activity

    await cycle.strategies.reflect.reflect(activity, working, cycle, _tick())

    assert activity.state is ActivityState.READY


async def test_reflect_does_not_block_over_an_unconsumed_verdict(tmp_path: Path) -> None:
    """Observe parks a resolved verdict and advances the marks in the same breath, so by the time
    Reflect runs the gate that produced it is no longer eligible and `condition_fired` is still
    empty — only Reason fills that. Blocking here strands the verdict: Situate skips a BLOCKED
    activity, so Reason never consumes it, and a judgement already paid for is thrown away. Seen on
    2026-08-24 as `fired=(0, 2)` immediately followed by `blocking on 3 pending condition(s)`.
    Reason's own no-fire path re-blocks, so there is nothing for Reflect to protect here."""
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    activity = _exhausted()
    state = PendingConditionState(condition=_condition(), evaluated_through=99)  # marks advanced
    activity.pending_conditions = [state]
    activity.condition_batch = [state]
    activity.condition_verdict = _verdict(fired=(0,))
    working.activities[activity.id] = activity

    await cycle.strategies.reflect.reflect(activity, working, cycle, _tick())

    assert activity.state is ActivityState.READY


async def test_a_parked_verdict_survives_reflect_and_reaches_its_then(tmp_path: Path) -> None:
    """The cross-phase version of the above: the two phases in the order the cycle runs them."""
    llm = FakeLLMClient(json.dumps({"steps": []}))
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm))
    activity = _exhausted()
    state = PendingConditionState(condition=_condition(), evaluated_through=99)
    activity.pending_conditions = [state]
    activity.condition_batch = [state]
    activity.condition_verdict = _verdict(fired=(0,))
    working.activities[activity.id] = activity

    await cycle.strategies.reflect.reflect(activity, working, cycle, _tick())
    # Through Situate, not straight to reason(): Situate only ever selects a READY activity, so a
    # Reflect that blocked means Reason is never handed the activity and the verdict rots in place.
    # Calling reason() directly would pass even with the bug.
    situated = await cycle.strategies.situate.situate([activity], working, cycle, _tick())
    assert situated.activity is activity
    await cycle.strategies.reason.reason(activity, working, cycle, situated)

    assert activity.pending_inference is not None
    assert activity.pending_inference.kind == "subgoal"


async def test_an_empty_parked_verdict_still_ends_up_blocked(tmp_path: Path) -> None:
    """Staying READY for the verdict must not lose the re-block when nothing fired — Reason owns
    that decision (and a failed evaluation degrades to an empty verdict, so it takes this path)."""
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    activity = _exhausted()
    state = PendingConditionState(condition=_condition(), evaluated_through=99)
    activity.pending_conditions = [state]
    activity.condition_batch = [state]
    activity.condition_verdict = _verdict()
    working.activities[activity.id] = activity

    await cycle.strategies.reflect.reflect(activity, working, cycle, _tick())
    after_reflect = activity.state  # a local: asserting in place would narrow it for mypy
    assert after_reflect is ActivityState.READY  # deferred to Reason, not blocked here
    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert activity.state is ActivityState.BLOCKED
    assert isinstance(activity.blocked_on, ConditionWait)


async def test_a_retired_condition_stops_waiting(tmp_path: Path) -> None:
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    activity = _exhausted()
    state = PendingConditionState(condition=_condition(), evaluated_through=99)
    activity.pending_conditions = [state]
    activity.condition_batch = [state]
    activity.condition_verdict = _verdict(retired=(0,))
    working.activities[activity.id] = activity

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert activity.pending_conditions == []
    # With nothing left to wait for, Reflect completes it on the next pass as it always did.
    activity.state = ActivityState.READY
    await cycle.strategies.reflect.reflect(activity, working, cycle, _tick())
    assert activity.state is ActivityState.TERMINATED


async def test_a_retired_condition_is_not_re_lifted_from_the_plan(tmp_path: Path) -> None:
    """Retiring removes the per-run state, but `Plan.pending` is the frozen skeleton and never
    changes — so the declaration is still there for Reflect to lift again on the very next pass,
    putting the condition back on watch for an `until` the verdict already judged satisfied."""
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    condition = _condition()
    activity = _exhausted(plan_pending=(condition,))
    state = PendingConditionState(condition=condition, evaluated_through=99)
    activity.pending_conditions = [state]
    activity.condition_batch = [state]
    activity.condition_verdict = _verdict(retired=(0,))
    working.activities[activity.id] = activity

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())
    assert activity.pending_conditions == []

    activity.state = ActivityState.READY
    await cycle.strategies.reflect.reflect(activity, working, cycle, _tick())

    assert activity.pending_conditions == []  # stays retired; the plan does not resurrect it
    assert activity.state is ActivityState.TERMINATED


async def test_a_condition_the_plan_declares_is_still_lifted_after_an_unrelated_retirement(
    tmp_path: Path,
) -> None:
    """Retirement is per-condition, not a switch on the plan: a second declared condition must go on
    watching after its sibling retires."""
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    done, live = _condition(), _other_condition("Tell the user the schedule slipped")
    activity = _exhausted(plan_pending=(done, live))
    state = PendingConditionState(condition=done, evaluated_through=99)
    activity.pending_conditions = [state]
    activity.condition_batch = [state]
    activity.condition_verdict = _verdict(retired=(0,))
    working.activities[activity.id] = activity

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())
    activity.state = ActivityState.READY
    await cycle.strategies.reflect.reflect(activity, working, cycle, _tick())

    assert [s.condition for s in activity.pending_conditions] == [live]
    assert activity.state is ActivityState.BLOCKED  # still waiting on the one that did not retire


async def test_a_hallucinated_index_cannot_retire_a_live_condition(tmp_path: Path) -> None:
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    activity = _exhausted()
    state = PendingConditionState(condition=_condition(), evaluated_through=99)
    activity.pending_conditions = [state]
    activity.condition_batch = [state]
    activity.condition_verdict = _verdict(retired=(7,))  # out of range
    working.activities[activity.id] = activity

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())

    assert activity.pending_conditions == [state]


# --------------------------------------------------------------------------------------------------
# Layer 3: undeclared-relevance recovery
# --------------------------------------------------------------------------------------------------


async def test_judge_is_off_unless_opted_in(tmp_path: Path) -> None:
    # It spends a call on an unverifiable judgement and, when it fires, interrupts a person. An
    # unattended run has nobody to ask, which is exactly where acting on a guess is worst.
    cycle, _ = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    assert cycle.relevance is None
    await cycle.tick()  # idle tick with no judge configured — must not raise


async def test_judge_proposes_an_amending_activity_blocked_on_the_user(tmp_path: Path) -> None:
    llm = FakeLLMClient(
        json.dumps(
            {
                "relevant": True,
                "task": 0,
                "goal": "Rebook for Thursday",
                "question": "Åke can't make the 19th — rebook for Thursday?",
            }
        )
    )
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm))
    judge = DefaultRelevanceJudge()
    done = Activity(id="old", goal="book it", context={}, state=ActivityState.TERMINATED)
    await cycle.episodic.learn(done, "booked and emailed", succeeded=True)
    _signal(working, "folders.INBOX.emails")  # claimed by NO declared condition

    await judge.consider(cycle)  # fires
    await _settle()
    await judge.consider(cycle)  # applies, inside the tick

    (amendment,) = [a for a in working.activities.values() if a.id != "old"]
    assert amendment.goal == "Rebook for Thursday"
    # Born BLOCKED on the user: never act on a goal nobody stated.
    assert amendment.state is ActivityState.BLOCKED
    assert isinstance(amendment.blocked_on, InputWait)
    assert "Thursday" in (amendment.blocked_on.prompt or "")
    # The amendment points back at what it amends; the closed episode is untouched.
    assert amendment.context["amends"] == "old"
    assert done.state is ActivityState.TERMINATED


async def test_the_amendment_question_is_actually_delivered(tmp_path: Path) -> None:
    """Setting `blocked_on` without sending `prompt` parks the agent on a question nobody can hear.
    Worse than silent: `_resume_on_input` clears an InputWait on the user's next Message whatever it
    says, so an unrelated instruction would be read as consent to an amendment never shown."""
    llm = FakeLLMClient(
        json.dumps(
            {
                "relevant": True,
                "task": 0,
                "goal": "Rebook for Thursday",
                "question": "Åke can't make the 19th — rebook for Thursday?",
            }
        )
    )
    transport = ScriptedTransport()
    cycle, working = _cycle(
        tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm), transport
    )
    judge = DefaultRelevanceJudge()
    done = Activity(id="old", goal="book it", context={}, state=ActivityState.TERMINATED)
    await cycle.episodic.learn(done, "booked and emailed", succeeded=True)
    _signal(working, "folders.INBOX.emails")

    await judge.consider(cycle)
    await _settle()
    await judge.consider(cycle)

    (amendment,) = [a for a in working.activities.values() if a.id != "old"]
    assert isinstance(amendment.blocked_on, InputWait)
    # The same text, on the same channel `send_message_to_user` uses — the wait and the asking are
    # two halves of one act.
    assert transport.sent == [("user", {"text": amendment.blocked_on.prompt})]
    assert transport.sent[0][1]["text"] == "Åke can't make the 19th — rebook for Thursday?"


async def test_a_declined_repeat_asks_nothing_twice(tmp_path: Path) -> None:
    """The dedup guard returns before the ask, so a repeat proposal must cost no second message —
    otherwise the throttle bounds activities while the user still gets pinged every tick."""
    responses = json.dumps({"relevant": True, "task": 0, "goal": "same goal", "question": "q?"})
    transport = ScriptedTransport()
    cycle, _ = _cycle(
        tmp_path,
        ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=FakeLLMClient(responses)),
        transport,
    )
    judge = DefaultRelevanceJudge(max_asks=9)
    done = Activity(id="old", goal="book it", context={}, state=ActivityState.TERMINATED)
    await cycle.episodic.learn(done, "booked", succeeded=True)

    for i in range(3):
        _signal(cycle.working, f"folders.INBOX.{i}")
        await judge.consider(cycle)
        await _settle()
        await judge.consider(cycle)

    assert transport.sent == [("user", {"text": "q?"})]


async def test_a_change_claimed_by_a_declared_gate_never_reaches_the_judge(
    tmp_path: Path,
) -> None:
    # The subtraction that defines this layer's input. A procedural with no LLM raises if the judge
    # escalates, so reaching the end proves it did not.
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p")))
    judge = DefaultRelevanceJudge()
    _blocked_with_condition(working)  # declares a watch on folders.INBOX.emails
    done = Activity(id="old", goal="book it", context={}, state=ActivityState.TERMINATED)
    await cycle.episodic.learn(done, "booked", succeeded=True)
    _signal(working, "folders.INBOX.emails")  # matches the declared gate

    await judge.consider(cycle)
    await _settle()

    assert not [a for a in working.activities.values() if a.id not in {"a1", "old"}]


async def test_judge_stops_asking_after_its_cap(tmp_path: Path) -> None:
    # No principled value exists for this number, which is why it is a setting — but SOME bound is
    # required: a mechanism that interrupts the user on every stray change is worse than one that
    # misses.
    responses = json.dumps({"relevant": True, "task": 0, "goal": "g", "question": "q?"})
    cycle, working = _cycle(
        tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=FakeLLMClient(responses))
    )
    judge = DefaultRelevanceJudge(max_asks=1)
    done = Activity(id="old", goal="book it", context={}, state=ActivityState.TERMINATED)
    await cycle.episodic.learn(done, "booked", succeeded=True)

    for i in range(4):
        _signal(working, f"folders.INBOX.{i}")
        await judge.consider(cycle)
        await _settle()
        await judge.consider(cycle)

    assert len([a for a in working.activities.values() if a.id != "old"]) == 1


async def test_judge_does_not_repropose_the_same_amendment(tmp_path: Path) -> None:
    responses = json.dumps({"relevant": True, "task": 0, "goal": "same goal", "question": "q?"})
    cycle, working = _cycle(
        tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=FakeLLMClient(responses))
    )
    judge = DefaultRelevanceJudge(max_asks=9)
    done = Activity(id="old", goal="book it", context={}, state=ActivityState.TERMINATED)
    await cycle.episodic.learn(done, "booked", succeeded=True)

    for i in range(3):
        _signal(working, f"folders.INBOX.{i}")
        await judge.consider(cycle)
        await _settle()
        await judge.consider(cycle)

    assert len([a for a in working.activities.values() if a.id != "old"]) == 1


async def test_judge_says_nothing_when_the_answer_is_no(tmp_path: Path) -> None:
    cycle, working = _cycle(
        tmp_path,
        ProceduralMemory(
            FileMemoryBackend(tmp_path / "p"), llm=FakeLLMClient(json.dumps({"relevant": False}))
        ),
    )
    judge = DefaultRelevanceJudge()
    done = Activity(id="old", goal="book it", context={}, state=ActivityState.TERMINATED)
    await cycle.episodic.learn(done, "booked", succeeded=True)
    _signal(working, "folders.INBOX.emails")

    await judge.consider(cycle)
    await _settle()
    await judge.consider(cycle)

    assert list(working.activities) == []


async def test_malformed_judgement_degrades_to_silence(tmp_path: Path) -> None:
    # A fabricated follow-up interrupts a person about work that was already done properly; a missed
    # one leaves a gap they can still notice. The asymmetry decides the fail-soft direction.
    cycle, working = _cycle(
        tmp_path,
        ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=FakeLLMClient("not json at all")),
    )
    judge = DefaultRelevanceJudge()
    done = Activity(id="old", goal="book it", context={}, state=ActivityState.TERMINATED)
    await cycle.episodic.learn(done, "booked", succeeded=True)
    _signal(working, "folders.INBOX.emails")

    await judge.consider(cycle)
    await _settle()
    await judge.consider(cycle)

    assert list(working.activities) == []


async def test_consult_recent_orders_by_recency_not_key(tmp_path: Path) -> None:
    # consult() retrieves by goal-equality, which is useless to a caller holding a change rather
    # than a goal — and the backend's stable order is the activity id, saying nothing about time.
    memory = EpisodicMemory(FileMemoryBackend(tmp_path))
    for name in ("zzz", "aaa", "mmm"):
        await memory.learn(Activity(id=name, goal=f"goal-{name}", context={}), "s", succeeded=True)

    recent = await memory.consult_recent(2)

    assert [e["activity_id"] for e in recent] == ["mmm", "aaa"]
    assert all(e["ended_at"] for e in recent)


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------


def _tick() -> Any:
    from sora.strategies import TickResult

    return TickResult()


def _verdict(fired: tuple[int, ...] = (), retired: tuple[int, ...] = ()) -> Any:
    from sora.types import ConditionVerdict

    return ConditionVerdict(fired=fired, retired=retired)


async def _settle() -> None:
    """Let a spawned off-cycle call run to completion — the tests drive phases directly rather than
    through the loop, so there is no tick to carry the resolve."""
    import asyncio

    for _ in range(6):
        await asyncio.sleep(0)


# --------------------------------------------------------------------------------------------------
# Rendering: a declared condition has to be visible in the trace
# --------------------------------------------------------------------------------------------------
# Without this, a plan that declared a good gate and a plan that silently declared none produce
# byte-identical logs — and telling those two apart is the entire question when a run ends early.


def test_render_plan_shows_declared_conditions() -> None:
    from sora.memory import render_plan

    plan = Plan(
        id="p1",
        goal="g",
        steps=[Step("invoke", {"tool_id": "t1", "operation_name": "send_email"})],
        pending=(
            PendingCondition(
                watch=SignalWait("state_changed", "t1", "folders.INBOX.emails"),
                when="he replies that he cannot make it",
                then="rebook for the date he proposes",
                until="the day has passed",
            ),
        ),
    )
    rendered = render_plan(plan)
    assert "0: invoke" in rendered
    assert "pending:" in rendered
    assert "folders.INBOX.emails" in rendered
    assert "he replies that he cannot make it" in rendered
    assert "rebook for the date he proposes" in rendered
    assert "the day has passed" in rendered


def test_render_plan_without_conditions_is_exactly_the_body() -> None:
    """A body-only plan must render as it did before conditions existed — no trailing header."""
    from sora.memory import render_plan, render_steps

    plan = Plan(
        id="p1", goal="g", steps=[Step("invoke", {"tool_id": "t1", "operation_name": "send_email"})]
    )
    assert render_plan(plan) == render_steps(plan.steps)
    assert "pending" not in render_plan(plan)


def test_render_plan_omits_an_absent_until() -> None:
    from sora.memory import render_plan

    plan = Plan(
        id="p1",
        goal="g",
        steps=[],
        pending=(
            PendingCondition(
                watch=SignalWait("state_changed", "t1", None), when="w", then="t", until=None
            ),
        ),
    )
    rendered = render_plan(plan)
    assert "when" in rendered and "until" not in rendered


# --------------------------------------------------------------------------------------------------
# The judgement must be able to READ what changed, not just be told where it is
# --------------------------------------------------------------------------------------------------
# ADR-0022 divides the labour: `Change` carries identities only, and "the values behind them come
# from the observed property snapshot" — so the judge "reads one named path instead of re-scanning a
# whole property". The dereference half was never implemented. `evaluate_conditions` rendered the
# raw ids plus `render_properties`, which collapses any property over its length cap to a shape
# sketch, so a 129-email ARE inbox arrived as `emails: [{... x 10}] x 129` and the id that had just
# landed was never resolved to its body. In the gaia2 adaptability run the gate opened correctly,
# the activity resumed correctly, and the judge then answered fired=() — having been shown an id and
# a shape, and asked whether someone had declined an invitation.


def _inbox(*emails: dict[str, Any]) -> Percept:
    """An ARE-shaped Emails.state: bulk enough that render_properties must shape-sketch it."""
    return Percept(
        "insim:are/Emails",
        ObservableProperty(
            "state",
            {
                "user_email": "astrid@example.com",
                "folders": {
                    "INBOX": {"folder_name": "INBOX", "emails": list(emails)},
                    "SENT": {"folder_name": "SENT", "emails": []},
                },
            },
        ),
        0.0,
    )


_REPLY = {
    "email_id": "e9",
    "sender": "ake@filmproduktion.se",
    "subject": "Re: Film Production Day",
    "content": "Sorry, I cannot make Saturday. How about Tuesday the 22nd at 10:00?",
}


async def test_the_judgement_is_shown_the_record_the_change_points_at(tmp_path: Path) -> None:
    llm = FakeLLMClient(json.dumps({"fired": [], "retired": []}))
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm)
    activity = _exhausted()
    observed = PerceptSnapshot([_inbox({"email_id": "old"}, _REPLY)], [])

    await procedural.evaluate_conditions(
        activity,
        [_condition()],
        [("insim:are/Emails", Change(path="folders.INBOX.emails", added=("e9",)))],
        observed,
    )

    _system, prompt = llm.calls[0]
    assert "cannot make Saturday" in prompt  # the body, not merely the id
    assert "Tuesday the 22nd" in prompt


async def test_only_the_changed_record_is_dereferenced_not_the_whole_property(
    tmp_path: Path,
) -> None:
    # The point of a located change: one named path, not a re-scan. An inbox holding hundreds of
    # unrelated emails must not drag them all into the prompt just because one arrived.
    llm = FakeLLMClient(json.dumps({"fired": [], "retired": []}))
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm)
    noise = [{"email_id": f"n{i}", "content": f"unrelated chatter {i}"} for i in range(200)]
    observed = PerceptSnapshot([_inbox(*noise, _REPLY)], [])

    await procedural.evaluate_conditions(
        activity := _exhausted(),
        [_condition()],
        [("insim:are/Emails", Change(path="folders.INBOX.emails", added=("e9",)))],
        observed,
    )
    assert activity is not None

    _system, prompt = llm.calls[0]
    assert "cannot make Saturday" in prompt
    assert "unrelated chatter 7" not in prompt


async def test_a_coarse_change_carries_no_ids_and_still_judges(tmp_path: Path) -> None:
    # Adapters DEGRADE rather than fail (types.py): a coarse Change names a path with all three
    # tuples empty. There is nothing to dereference, and that must not raise.
    llm = FakeLLMClient(json.dumps({"fired": [], "retired": []}))
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm)
    observed = PerceptSnapshot([_inbox(_REPLY)], [])

    await procedural.evaluate_conditions(
        _exhausted(),
        [_condition()],
        [("insim:are/Emails", Change(path="folders.INBOX.emails"))],
        observed,
    )

    assert len(llm.calls) == 1


async def test_a_change_path_that_no_longer_resolves_is_skipped_not_raised(tmp_path: Path) -> None:
    # The snapshot is read at judgement time, not change time, so a path can have gone away.
    llm = FakeLLMClient(json.dumps({"fired": [], "retired": []}))
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm)
    observed = PerceptSnapshot([_inbox(_REPLY)], [])

    await procedural.evaluate_conditions(
        _exhausted(),
        [_condition()],
        [("insim:are/Emails", Change(path="folders.ARCHIVE.emails", added=("e9",)))],
        observed,
    )

    assert len(llm.calls) == 1


async def test_the_gate_hands_the_judgement_the_source_of_each_change(tmp_path: Path) -> None:
    # End to end through Reason: the strategy used to flatten (percept, Change) into a bare
    # list[Change], discarding the source — which is what says WHICH property to dereference.
    llm = FakeLLMClient(json.dumps({"fired": [], "retired": []}))
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm))
    activity = _exhausted()
    activity.pending_conditions = [
        PendingConditionState(condition=_condition(), evaluated_through=0)
    ]
    working.activities[activity.id] = activity
    working.properties[("insim:are/Emails", "state")] = _inbox(_REPLY)
    _signal(working, "folders.INBOX.emails")

    await cycle.strategies.reason.reason(activity, working, cycle, _tick())
    await _settle()

    _system, prompt = llm.calls[0]
    assert "cannot make Saturday" in prompt


# The relevance judge is the same judgement one layer out, and it reads the same kind of change —
# so it needs the same dereference. It is the harder case, not the easier one: the condition judge
# is at least handed a `when` clause naming what to look for, while this one is asked the open
# question "does anything here bear on work that finished?" against episode summaries. Answering
# that from ids and a shape sketch is not a judgement at all. Undeclared-relevance recovery being
# off by default bounds the blast radius but does not make the prompt correct — and leaving two
# sibling judges disagreeing about what a `Change` renders as is how the first one came to be wrong.


async def test_the_relevance_judgement_is_shown_the_record_the_change_points_at(
    tmp_path: Path,
) -> None:
    llm = FakeLLMClient(json.dumps({"relevant": False}))
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm)
    episodes = [{"activity_id": "old", "goal": "book it", "succeeded": True, "summary": "booked"}]
    observed = PerceptSnapshot([_inbox({"email_id": "old"}, _REPLY)], [])

    await procedural.judge_relevance(
        episodes,
        [("insim:are/Emails", Change(path="folders.INBOX.emails", added=("e9",)))],
        observed,
    )

    _system, prompt = llm.calls[0]
    assert "cannot make Saturday" in prompt  # the body, not merely the id
    assert "Tuesday the 22nd" in prompt


async def test_the_relevance_judge_hands_the_judgement_the_source_of_each_change(
    tmp_path: Path,
) -> None:
    # End to end through the judge: `consider` flattened (percept, Change) into a bare list[Change],
    # discarding the source — which is what says WHICH property to dereference. The inbox is bulk on
    # purpose: under its length cap `render_properties` prints the whole property and the body
    # reaches the prompt regardless, so a one-email fixture passes without the dereference existing.
    llm = FakeLLMClient(json.dumps({"relevant": False}))
    cycle, working = _cycle(tmp_path, ProceduralMemory(FileMemoryBackend(tmp_path / "p"), llm=llm))
    judge = DefaultRelevanceJudge()
    done = Activity(id="old", goal="book it", context={}, state=ActivityState.TERMINATED)
    await cycle.episodic.learn(done, "booked and emailed", succeeded=True)
    noise = [{"email_id": f"n{i}", "content": f"unrelated chatter {i}"} for i in range(200)]
    working.properties[("insim:are/Emails", "state")] = _inbox(*noise, _REPLY)
    _signal(working, "folders.INBOX.emails")

    await judge.consider(cycle)
    await _settle()

    _system, prompt = llm.calls[0]
    assert "cannot make Saturday" in prompt
    assert "unrelated chatter 7" not in prompt
