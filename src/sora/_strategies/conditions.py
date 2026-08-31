"""Condition lifting, matching, frame ownership, and retirement mechanics."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

from sora.action import (
    ResumeAction,
    _spawn_tracked,
)
from sora.activity import Activity, ActivityState
from sora.memory import (
    PerceptSnapshot,
    pending_from_raw,
)
from sora.perception import Percept
from sora.types import (
    GOAL_KIND_ACHIEVEMENT,
    GOAL_KIND_MAINTENANCE,
    SUBGOAL,
    ConditionWait,
    PendingCondition,
    PendingConditionState,
    SignalWait,
    Step,
    SubgoalMode,
    changes_of,
    goal_kind_of,
    watch_matches,
)

if TYPE_CHECKING:
    from sora.cycle import DecisionCycle
    from sora.environment import DomainClock, EnvironmentView
    from sora.memory import WorkingMemory
    from sora.types import Plan

log = logging.getLogger("sora.strategies")


def _lift_pending_conditions(activity: Activity, wm: WorkingMemory) -> None:
    """Copy any conditions the activity's live frames declare onto the activity itself.

    Conditions are declared per-plan (the reusable skeleton) but must outlive the frame that
    declared them — a deliberative sub-goal is usually where the agent first learns a branch exists
    (it sent the mail; now a reply may come), and that sub-plan's frame pops long before the reply
    arrives. Lifting is idempotent: a condition already present is not re-added, so this can run
    every cycle without accumulating duplicates.

    A newly-lifted condition starts its mark at the CURRENT signal count, not at zero. Signals that
    arrived before the condition was declared cannot be the event it is waiting for, and the
    retention log still holds a few hundred of them — starting at zero would make every new
    condition immediately re-judge the whole backlog.

    Retired conditions are excluded explicitly rather than by absence. Dedup alone cannot tell "not
    lifted yet" from "lifted and then retired", and `Plan.pending` is a frozen skeleton that still
    declares what retirement removed — so without `retired_conditions` a satisfied `until` would be
    undone by the very next lift, putting the condition back on watch for good.

    Each lifted condition records the frame that declared it (`declared_by`), because lifting is
    precisely what erases that and a maintenance frame's completion rule needs it back — see
    `_conditions_hold_frame`. A condition two frames declare identically is attributed to the
    innermost one, which is the frame reading it as its own.
    """
    # Innermost-first, because that is how `_frame_key` counts `depth`. `parent_frames` is a stack
    # stored outermost-first, so walking it in storage order would pair every plan past the first
    # with another frame's key.
    frames = [activity.plan, *(plan for plan, _, _ in reversed(activity.parent_frames))]
    known = {state.condition for state in activity.pending_conditions}
    known |= activity.retired_conditions
    for depth, plan in enumerate(frames):
        if plan is None:
            continue
        for condition in plan.pending:
            if condition in known:
                continue
            known.add(condition)
            activity.pending_conditions.append(
                _lifted(condition, activity, wm, _frame_key(activity, depth))
            )


def _lifted(
    condition: PendingCondition,
    activity: Activity,
    wm: WorkingMemory,
    declared_by: tuple[tuple[str, int], ...],
) -> PendingConditionState:
    """Build the run state for one newly-lifted condition, remembering or inheriting its deadline.

    The anchor a relative `until` is measured from is taken NOW, because now is when the waiting
    starts — and from the watched workspace's own clock, since host wall-clock is a different clock
    entirely (ADR-0027 §5).

    On top of that, a resolved deadline is remembered for this condition and handed to a later
    declaration of the same condition that cannot resolve one of its own. That asymmetry is the
    point: a declared bound always wins, and inheritance only ever fills a hole. It closes the case
    where a window outlives the plan that declared it — a replan mid-window is told (rightly) never
    to guess a `seconds` it was not given, so it writes an event-shaped string, and the clock exit
    the first plan had is gone. A shared `watch` is not enough: it is only the mechanical gate, and
    independent conditions may legitimately inspect the same changes.
    """
    declared_at = _domain_now(wm, condition.watch.source)
    deadline = condition.until.deadline(declared_at) if condition.until else None
    window_key = _condition_window_key(condition)
    if deadline is not None:
        activity.window_deadlines[window_key] = deadline
    return PendingConditionState(
        condition=condition,
        declared_by=declared_by,
        declared_at=declared_at,
        inherited_deadline=(
            activity.window_deadlines.get(window_key) if deadline is None else None
        ),
        evaluated_through=wm.signals_appended,
        derived_through=wm.property_changes_appended,
    )


def _condition_window_key(condition: PendingCondition) -> tuple[SignalWait, str, str]:
    """The stable part of a condition across a replan that cannot restate its relative bound."""
    return (condition.watch, condition.when, condition.then)


def _step_pending_conditions(step: Step) -> tuple[PendingCondition, ...]:
    """Parse the valid pending conditions declared by a sub-goal step."""
    raw = step.params.get("pending")
    if not isinstance(raw, list):
        return ()
    parsed = (pending_from_raw(entry) if isinstance(entry, dict) else None for entry in raw)
    return tuple(condition for condition in parsed if condition is not None)


def _lift_step_conditions(
    step: Step, activity: Activity, wm: WorkingMemory, mode: SubgoalMode
) -> None:
    """Lift any `pending` a SUB-GOAL STEP declares, at the moment that step is reached.

    A plan declares conditions; a sub-goal step is not a plan, so this looks like the wrong home for
    one. It is the only home a maintenance sub-goal has. Its window has to be bounded by an `until`,
    and the two places the planner could otherwise put it are both unreachable: a MECHANICAL
    sub-goal has no plan of its own at all (it splices its fan-out into the caller's plan), and a
    DELIBERATIVE one's sub-plan is written a later cycle, by which point the window has already been
    open for a while and its `seconds` can no longer be stated honestly. So the planner writes it
    here — the last point where "for the next four minutes" is still literally true — and before
    this it was read by nobody and dropped without a word.

    A mechanical sub-goal pushes no frame, so its condition belongs to the current frame. A
    deliberative sub-goal does push one once inference resolves, and the step's condition governs
    that future child frame: assigning it to the caller would let the child exhaust and pop while
    the window was still open. The future key is already determinate from the current plan id and
    step index; it is the same tuple ``InferAction`` appends to the target copy it prompts against.

    Lifting is idempotent against the same dedup set the plan-level lift uses, so re-reaching the
    step (a mechanical fan-out re-reads `step_index`) cannot double-declare a window.
    """
    known = {state.condition for state in activity.pending_conditions} | activity.retired_conditions
    owner = _frame_key(activity, 0)
    if mode is SubgoalMode.DELIBERATIVE:
        plan = activity.plan
        assert plan is not None  # a sub-goal step is dispatched only from a set plan
        owner += ((plan.id, activity.step_index),)
    for condition in _step_pending_conditions(step):
        if condition in known:
            continue
        known.add(condition)
        log.info(
            "reason: sub-goal step declares a pending condition -> %r (until %r)",
            condition.when,
            condition.until.text if condition.until else None,
        )
        activity.pending_conditions.append(_lifted(condition, activity, wm, owner))


def _clock_for_source(view: EnvironmentView, source: str | None) -> DomainClock | None:
    """The domain clock an `until` watching ``source`` is answered against: the clock of the
    workspace owning that tool, or None when there isn't one to ask (ADR-0027 §5).

    Three ways to get None, and they are one answer on purpose — none of them may fall back to
    `time.time()`: the watch names no source (so no workspace is determinate), the source is not a
    joined tool, or its workspace cannot tell domain time. Note what the per-workspace scoping
    buys: because each watch names one source, each `until` resolves against exactly one clock, so
    the "which of two disagreeing clocks" case ADR-0027 calls a plan defect cannot arise in the
    data at all."""
    if source is None:
        return None
    workspace = view.workspace_of(source)
    return workspace.clock if workspace is not None else None


def _domain_now(wm: WorkingMemory, source: str | None) -> datetime | None:
    clock = _clock_for_source(wm.registry, source)
    return clock.now() if clock is not None else None


def _unclosable_window(activity: Activity, sub_plan: Plan, wm: WorkingMemory) -> str | None:
    """Plan validation for a maintenance sub-goal whose window nothing could ever close — the
    defect string for a replan, or None when the plan is fine (ADR-0027 §6).

    A maintenance sub-goal is finished when every condition it declared has retired, and nothing
    else ends it. So an `until` that asks about time, watching a workspace that cannot tell domain
    time, is not a slow plan: it is a frame — and every remaining step of the parent, including the
    report the user is waiting for — held open silently and forever. Refusing it costs a replan;
    accepting it costs the run.

    Refused **at the plan**, not at the moment the wait would go quiet, for two reasons. The
    planner is still holding the goal here, so the correction reaches the only party that can act
    on it; and by the time the window would have mattered the agent has already run the body and
    told nobody anything is wrong. It is the ordinary ADR-0025 replan path from there: the defect
    rides the `superseded` bundle into the next planning prompt, and a planner that writes past it
    twice trips the runaway-replan breaker into await-input — which is the honest end state, since
    what the agent is actually missing is the ability to tell the time here.

    Only **maintenance**, and only a bound the runtime can see. An achievement frame pops when its
    steps run out and its condition keeps watching from the activity afterwards (ADR-0022's
    contingency case), so no clock is load-bearing there. An event-shaped `until` is the retirement
    judge's question and needs no clock either. What is left is exactly the intersection this
    checks."""
    plan = activity.plan
    if plan is None or activity.step_index >= len(plan.steps):
        return None  # no sub-goal step to read a kind off — nothing to validate against
    step = plan.steps[activity.step_index]
    if step.next_action != SUBGOAL or goal_kind_of(step) != GOAL_KIND_MAINTENANCE:
        return None
    goal = step.params.get("goal")
    # A deliberative maintenance window is normally declared on the invoking step: that is the
    # last point where a relative duration can be stated honestly, and the condition is assigned
    # to the future child frame. The child prompt consequently tells the planner not to repeat it
    # in Plan.pending. Both declaration sites can hold the frame, so both must pass this guard.
    conditions = (*sub_plan.pending, *_step_pending_conditions(step))
    for condition in conditions:
        until = condition.until
        if until is None or not until.is_time_bounded:
            continue  # event-shaped: the judge can answer it without a clock
        source = condition.watch.source
        if source is None:
            return (
                f"the maintenance sub-goal {goal!r} is bounded by a window in time "
                f"({until.text!r}), but its condition's watch names no `source`, so there is "
                "no workspace whose clock could tell when that window closes — name the tool the "
                "watch is on, or bound the window by an observable event instead"
            )
        if _clock_for_source(wm.registry, source) is None:
            return (
                f"the maintenance sub-goal {goal!r} is bounded by a window in time "
                f"({until.text!r}), but {source!r}'s workspace has no domain clock, so "
                "nothing could ever tell that the window has closed and the sub-goal would never "
                "finish — bound it by an observable event instead, or say that this environment "
                "provides no way to tell the time"
            )
    return None


def _frame_key(activity: Activity, depth: int = 0) -> tuple[tuple[str, int], ...]:
    """Identity of the frame ``depth`` levels above the current one: the chain of (plan id, sub-goal
    step index) that reaches it, `()` for the top-level plan. See `PendingConditionState`."""
    frames = activity.parent_frames[: len(activity.parent_frames) - depth]
    return tuple((plan.id, index) for plan, index, _ in frames)


def _frame_goal_kind(activity: Activity) -> str:
    """The completion criterion declared for the frame the activity is currently executing.

    Read off the `subgoal` step that pushed it — the same place `_ancestor_subgoal_goals` reads a
    frame's goal from — rather than stored on the frame, so nothing has to migrate and a step's
    params stay the one declaration. The top-level plan is nobody's sub-goal and is always an
    achievement goal: it has no frame to hold, and an exhausted body with live conditions blocks
    there anyway.

    Only a frame has a kind, so a `goal_kind` on a MECHANICAL sub-goal reads as nothing: that
    fan-out splices into the plan in place and pushes no frame of its own, and declares no
    `pending` either (the conditions belong to the plan it was spliced into). Maintenance is served
    by
    such a fan-out from inside its own sub-plan — the frame that carries the kind — not by
    labelling the fan-out step itself.
    """
    if not activity.parent_frames:
        return GOAL_KIND_ACHIEVEMENT
    parent_plan, index, _mark = activity.parent_frames[-1]
    if index >= len(parent_plan.steps):  # defensive: a frame whose parent was replanned under it
        return GOAL_KIND_ACHIEVEMENT
    return goal_kind_of(parent_plan.steps[index])


def _conditions_hold_frame(activity: Activity) -> bool:
    """Work this frame still owes, which popping to the parent would skip past (ADR-0027).

    Two independent reasons, and only the second is a goal-kind question:

    * **Committed work** — a queued `condition_fired` or a verdict nothing has applied yet. Both
      kinds of goal owe it: the judgement is already paid for and the `then` runs at this depth, so
      the parent must not run ahead of them.
    * **A maintenance goal's own live conditions** — its steps were the first iteration, so it is
      finished only when every condition it declared has retired. On the motivating run, popping
      resumed the parent at the message telling the user everything was done, sent while the
      monitoring window was still open.

    An **achievement** frame is never held by a condition: ADR-0022's contingency case (send the
    mail, pop, keep watching) is exactly a condition meant to outlive the frame that declared it,
    and it keeps watching from the activity after the pop. The stopgap this replaced held every
    frame by every live condition, which broke that case in one direction and, because lifting
    erases provenance, let a condition declared by the top-level plan pin an unrelated sub-plan
    open in the other.
    """
    if activity.condition_fired or activity.condition_verdict is not None:
        return True
    if _frame_goal_kind(activity) != GOAL_KIND_MAINTENANCE:
        return False
    key = _frame_key(activity)
    return any(state.declared_by == key for state in activity.pending_conditions)


def _body_exhausted(activity: Activity) -> bool:
    """Has the activity run out of body to execute *now*?

    Deliberately not "is the activity finished": a just-exhausted sub-plan usually has parent
    frames waiting to pop, and until they do it still has body left. The exception is a frame its
    own conditions hold (`_conditions_hold_frame`) — Reason will not pop past that, so there is no
    next step to run and a fired `then` may start. Both callers ask this to decide exactly that, so
    what they need is "is the intention stack idle". Reflect's completion test is separately
    stricter and still demands an empty stack.
    """
    return (
        activity.plan is not None
        and activity.step_index >= len(activity.plan.steps)
        and (not activity.parent_frames or _conditions_hold_frame(activity))
    )


def _condition_watches(activity: Activity) -> tuple[SignalWait, ...]:
    """The distinct watches of an activity's unsatisfied conditions, order-stable.

    Denormalized onto the ConditionWait so `blocked_on` describes the wait on its own — what a
    diagnostic renders, and what a harness inspects to tell a live wait from a stuck activity.
    Matching itself reads `pending_conditions`, because that is where the per-condition marks live.
    """
    seen: dict[SignalWait, None] = {}
    for state in activity.pending_conditions:
        seen.setdefault(state.condition.watch, None)
    return tuple(seen)


def _match_signal(wm: WorkingMemory, wait: SignalWait, *, since: int = 0) -> Percept | None:
    """Return the first retained signal satisfying a declared wait."""
    first_seq = wm.signals_appended - len(wm.signals)
    for offset, percept in enumerate(wm.signals):
        if first_seq + offset < since:
            continue
        if (
            percept.payload.name == wait.signal_name
            and (wait.source is None or percept.source == wait.source)
            and watch_matches(wait.path, wait.kind, changes_of(percept.payload))
        ):
            return percept
    return None


def _match_derived(wm: WorkingMemory, wait: SignalWait, *, since: int = 0) -> Percept | None:
    """Return the first retained derived change satisfying a declared wait."""
    first_seq = wm.property_changes_appended - len(wm.property_changes)
    for offset, percept in enumerate(wm.property_changes):
        if first_seq + offset < since:
            continue
        if wait.source is not None and percept.source != wait.source:
            continue
        if watch_matches(wait.path, wait.kind, list(percept.payload.changes)):
            return percept
    return None


def _eligible_conditions(
    activity: Activity, wm: WorkingMemory
) -> list[tuple[PendingConditionState, Percept]]:
    """The conditions whose gate has opened on a signal they have not already judged.

    This is the whole cost control: a condition is only ever evaluated against a change that
    mechanically matched its declared watch — name, source, path, and direction — so the unrelated
    events an agent observes never reach a model.

    `path` alone is not enough, and assuming it was cost a run: the note here used to claim an
    agent's own outbound write "lands somewhere different from what the condition watches", which
    holds for an agent that watches an inbox and writes to a sent folder, and fails completely for
    one that watches a collection it also deletes from. There the write lands on the *exact*
    watched path, opens the gate, and buys a model call to be told that a deletion is not an
    addition. `SignalWait.kind` is what separates the two, and it degrades open (see
    `_kind_matches_one`), so a watch stays correct on an adapter that cannot report direction.
    """
    eligible: list[tuple[PendingConditionState, Percept]] = []
    for state in activity.pending_conditions:
        match = _match_signal(wm, state.condition.watch, since=state.evaluated_through)
        if match is None:
            match = _match_derived(wm, state.condition.watch, since=state.derived_through)
        if match is not None:
            eligible.append((state, match))
    return eligible


_RETIREMENT_BACKOFF = 5


class ConditionRetirement:
    """Stateful retirement of quiet or expired pending conditions."""

    def __init__(self, interval: float | None) -> None:
        self._interval = interval
        self._marks: dict[str, tuple[float, int, int]] = {}
        self._in_flight = False
        self._result: tuple[str, list[PendingConditionState], tuple[int, ...]] | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def retire_quiet(self, cycle: DecisionCycle) -> None:
        """Garbage-collect pending conditions whose watch has gone quiet (ADR-0027 §4).

        The hole this fills: `until` was judged in exactly one place, the batched condition
        evaluation, which runs only when an observed change makes a condition *eligible*. So the
        one condition guaranteed never to be re-judged was the one nothing ever moves against —
        and it held its activity BLOCKED for good. Reachable for a contingency condition,
        load-bearing for a maintenance sub-goal, whose frame lives exactly as long as its `until`.

        Three properties, each of which is the decision rather than an implementation detail:

        * **Retire only, never fire.** A pass that could also fire would duplicate the eligibility
          gate's job and reopen the cost question that gate settled — and it would be firing on no
          evidence, since its premise is that nothing arrived. Enforced in `judge_retirement`, not
          merely asked for in the prompt.
        * **Idle-scheduled** (ADR-0026's cadence). A READY activity is one Situate will select on
          this very tick, so the sweep stands down; Reflect only ever demotes between here and
          there, so "nothing is READY at the end of Observe" is exactly "Situate will select
          nothing". Being conservative in that direction costs at most a deferral to the next tick.
        * **Observe retires, Reason pops.** This drops the retired conditions and releases the
          wait; the pop that follows is Reason's, which owns plan advancement everywhere else.

        What it answers is an **event-shaped** `until` ("the Film Production Day has taken place"),
        judged from the observed world. A **time-bounded** one is a comparison, not a judgement,
        and `retire_expired` has already resolved it against the workspace's clock
        before this runs — which is what keeps the common case free. What still reaches here is the
        residue that comparison could not place: a bound with no clock behind its watch, a
        wall-clock reading with no zone, a duration anchored on an event. The judge is shown no
        clock even so, deliberately: the only time this module could pass it is `time.time()`, and
        answering a domain question with host wall-clock is the silent wrong answer ADR-0027 §5
        exists to prevent. For that residue the backoff keeps the unanswerable case cheap rather
        than making it correct.

        The call itself is spawned off-cycle and its result parked, like every model call in the
        runtime — but deliberately *not* through `_infer_`/`pending_inference`, which would move the
        activity to RUNNING. The activity is BLOCKED and must stay blocked: a garbage collection
        pass is not a reason to make an activity look like it is doing something, and RUNNING is
        mutually exclusive with `blocked_on`. So this follows `DefaultRelevanceJudge`'s shape — the
        background task only ever sets a field, and every mutation of working memory happens here,
        on-cycle.
        """
        if self._interval is None:
            return
        # Apply first, and only apply: a tick that lands a verdict does not also fire the next one.
        if self._result is not None:
            await self._apply(cycle)
            return
        if self._in_flight:
            return
        activity = self._candidate(cycle.working)
        if activity is None:
            return
        judged = list(activity.pending_conditions)
        _checked, misses, _seen = self._marks.get(activity.id, (0.0, 0, 0))
        # Marked at fire time, not on resolve, so a slow call cannot be re-fired underneath itself.
        self._marks[activity.id] = (time.time(), misses, len(judged))
        # Agent-level, like both other judges and unlike a `_ground_`/`_select_` call: retirement
        # asks whether waiting is over, and the evidence for that ("the slot has taken place") is
        # routinely on a tool the waiting activity never touches.
        observed = PerceptSnapshot(
            list(cycle.working.properties.values()), list(cycle.working.signals)
        )
        self._in_flight = True
        log.info(
            "observe: judging retirement of %d quiet condition(s) on %s",
            len(judged),
            activity.id,
        )
        _spawn_tracked(self._tasks, self._judge(cycle, activity, judged, observed))

    def _candidate(self, wm: WorkingMemory) -> Activity | None:
        """The one activity to sweep this tick, or None.

        One per tick, longest-unchecked first. The sweep's whole cost argument is that it comes out
        of slack, and N calls fired together on a single idle tick is not slack — nor would it stay
        one call per activity, since an agent that accumulates watches is exactly the one this runs
        for most often.
        """
        if len(self._marks) > len(wm.activities):
            # Pacing state for an activity working memory no longer holds. Bounded rather than
            # correctness-critical, but this dict is keyed by a per-run id and the runtime is meant
            # to stay up indefinitely.
            self._marks = {key: mark for key, mark in self._marks.items() if key in wm.activities}
        if any(a.state is ActivityState.READY for a in wm.activities.values()):
            return None  # something can advance; the sweep never competes with it
        assert self._interval is not None  # guarded by the caller
        now = time.time()
        due: list[tuple[float, Activity]] = []
        for activity in wm.activities.values():
            if activity.state is ActivityState.TERMINATED or not activity.pending_conditions:
                continue
            checked, misses, seen = self._marks.get(activity.id, (0.0, 0, 0))
            if misses and len(activity.pending_conditions) > seen:
                # A condition declared since the last check deserves a prompt look — the backoff's
                # premise ("the last look bought nothing") says nothing about one never looked at.
                # Persisted, not just local: the stored count is what the next mark is built from,
                # so a local reset makes the look prompt exactly once and then restores the whole
                # accumulated backoff.
                misses = 0
                self._marks[activity.id] = (checked, misses, seen)
            if checked and now - checked < self._interval * 2 ** min(misses, _RETIREMENT_BACKOFF):
                continue
            due.append((checked, activity))
        if not due:
            return None
        return min(due, key=lambda entry: entry[0])[1]

    async def _judge(
        self,
        cycle: DecisionCycle,
        activity: Activity,
        judged: list[PendingConditionState],
        observed: PerceptSnapshot,
    ) -> None:
        try:
            retired = await cycle.procedural.judge_retirement(
                activity, [state.condition for state in judged], observed
            )
        except Exception:  # noqa: BLE001 — a failed sweep means "keep waiting", not a crash
            log.warning("observe: retirement judgement failed on %s", activity.id, exc_info=True)
            retired = ()
        finally:
            self._in_flight = False
        self._result = (activity.id, judged, retired)

    async def _apply(self, cycle: DecisionCycle) -> None:
        """Consume a parked retirement verdict — the judged sweep's half of `_drop_retired`, plus
        the backoff bookkeeping a verdict that retired nothing earns."""
        parked, self._result = self._result, None
        assert parked is not None  # guarded by the caller
        activity_id, judged, indices = parked
        activity = cycle.working.activities.get(activity_id)
        if activity is None or activity.state is ActivityState.TERMINATED:
            return
        if not indices:
            # Back off: the same answer is what the next call would most likely buy too.
            checked, misses, seen = self._marks.get(activity_id, (time.time(), 0, 0))
            self._marks[activity_id] = (checked, misses + 1, seen)
            return
        await self._drop_retired(cycle, activity, [judged[i] for i in indices if i < len(judged)])

    async def retire_expired(self, cycle: DecisionCycle) -> None:
        """Retire every condition whose `until` the workspace's own clock says is spent (ADR-0027).

        The cheap half of retirement, and the common case the ADR says must not be taxed: a
        time-bounded `until` is a comparison, not a judgement, so it never reaches a model. That is
        also why this pass is unlike its judged sibling in every dimension — it sweeps **all**
        activities, **every** tick, with no idle gate, no one-in-flight rule and no backoff. Those
        exist to ration model calls; there is nothing here to ration, and a maintenance window
        noticed late is a window the agent kept working past.

        What makes it correct rather than merely cheap is which clock it reads. The bound is
        measured against the clock of the workspace owning the watch — never `time.time()`, whose
        answer under a simulation is off by decades (ADR-0027 §5). Anything the runtime cannot
        place on that timeline — an event-shaped `until`, a wall-clock reading with no zone, a
        duration anchored on something that has not happened yet, or a watch whose workspace cannot
        tell domain time at all — is left untouched here and falls through to the judged sweep.
        Erring that way is deliberate: retiring early ends a window that is still open, which is
        precisely the failure this machinery was built to fix.
        """
        for activity in list(cycle.working.activities.values()):
            if activity.state is ActivityState.TERMINATED or not activity.pending_conditions:
                continue
            expired = [
                state
                for state in activity.pending_conditions
                if self._has_expired(cycle.working, state)
            ]
            if expired:
                await self._drop_retired(cycle, activity, expired)

    @staticmethod
    def _has_expired(wm: WorkingMemory, state: PendingConditionState) -> bool:
        until = state.condition.until
        # An inherited deadline is consulted even when this condition declares no `until` at all: a
        # replan that drops the clause entirely is the same lost bound as one that restates it
        # event-shaped, and neither makes the window it was given eternal.
        declared = until.deadline(state.declared_at) if until else None
        deadline = declared or state.inherited_deadline
        if deadline is None:
            # Either event-shaped (the judge's question, not the clock's), or a declared bound with
            # no anchor because the workspace could not tell domain time when it was lifted.
            return False
        now = _domain_now(wm, state.condition.watch.source)
        if now is None or now.tzinfo is None or deadline.tzinfo is None:
            # No clock to ask, or one answering with a naive instant — which names no point two
            # clocks could be compared on. Either way this is not the pass that guesses.
            return False
        return now >= deadline

    async def _drop_retired(
        self, cycle: DecisionCycle, activity: Activity, retiring: list[PendingConditionState]
    ) -> None:
        """Drop retired conditions off an activity, then release the wait they were holding.

        Releasing is the half that is easy to leave out and impossible to notice missing. An
        activity BLOCKED on a `ConditionWait` is resumed by exactly one thing — a signal making one
        of its conditions eligible — so an activity whose last condition just retired has nothing
        left that could ever wake it, and Situate only ever selects a READY activity. Retiring
        without resuming would swap one permanent block for another.

        There are therefore **two** ways a retirement releases the wait, and only the first is
        "nothing left to watch". A frame is held solely by the conditions it declared
        (`_conditions_hold_frame`), so retiring a maintenance frame's own condition frees the frame
        — Reason can pop it and run the parent's remaining steps — even while an unrelated
        contingency declared one level up is still being watched, and keeps `pending_conditions`
        non-empty. Gating the resume on emptiness alone leaves exactly that activity blocked for
        good, which is the failure the frame rule was written to end, one layer down.

        A queued `condition_fired` is deliberately not scrubbed when its condition retires: that is
        work the frame already accepted and paid a judgement for, and ADR-0027 holds both kinds of
        frame open for it. Retirement ends the *watching*, not a firing that already happened.
        """
        # Identity, not equality: PendingConditionState is mutable (its marks advance), so it is
        # unhashable, and two conditions can compare equal while being distinct waiters. A state the
        # eligibility gate retired while a judgement was in flight is simply no longer found.
        retired = {id(state) for state in retiring}
        activity.pending_conditions = [
            state for state in activity.pending_conditions if id(state) not in retired
        ]
        # Remember WHAT retired: the declaration survives on the frozen `Plan.pending`, so the next
        # lift would otherwise put the condition straight back on watch, undoing this for good.
        activity.retired_conditions.update(state.condition for state in retiring)
        # Reset the miss counter — a set that actually moved earns a prompt look at what is left —
        # but leave "last judged at" where it was. 0.0 (never) for an activity the judged sweep has
        # not seen: a free mechanical retirement is not a judgement, and recording it as one would
        # push that activity's first judged sweep an interval into the future.
        checked, _misses, _seen = self._marks.get(activity.id, (0.0, 0, 0))
        self._marks[activity.id] = (checked, 0, len(activity.pending_conditions))
        log.info(
            "observe: retired %d condition(s) on %s; %d left",
            len(retiring),
            activity.id,
            len(activity.pending_conditions),
        )
        if not isinstance(activity.blocked_on, ConditionWait):
            return
        freed_frame = bool(activity.parent_frames) and not _conditions_hold_frame(activity)
        if activity.pending_conditions and not freed_frame:
            # Still waiting, on less. Re-derive the wait so `blocked_on` keeps describing what the
            # activity is actually waiting for — what a diagnostic renders and a harness inspects.
            activity.blocked_on = ConditionWait(watches=_condition_watches(activity))
            return
        resume = cycle.actions.internal(ResumeAction.name)
        await resume.execute(cycle, activity_id=activity.id)
