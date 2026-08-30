"""One pluggable strategy per phase, threaded through a shared TickResult."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard

from sora.action import (
    CollectAction,
    CreateActivityAction,
    EvaluateConditionsAction,
    FilterAction,
    FilterPerceptionsAction,
    FocusAction,
    GroundAction,
    InferAction,
    InvokeAction,
    JoinAction,
    LeaveAction,
    LoadManualAction,
    ResumeAction,
    RevalidateAction,
    SendAction,
    SuspendAction,
    UnfocusAction,
    UnloadManualAction,
    _spawn_tracked,
    attend,
    pluck,
    release,
)
from sora.activity import SEEDED_BINDINGS, Activity, ActivityState
from sora.llm import LLMOutcome, log_llm_discarded, log_llm_outcome
from sora.memory import (
    PerceptSnapshot,
    pending_from_raw,
    render_plan,
    render_steps,
    step_from_raw,
)
from sora.perception import Percept
from sora.types import (
    GOAL_KIND_ACHIEVEMENT,
    GOAL_KIND_MAINTENANCE,
    OPERATION_NAME,
    SUBGOAL,
    TOOL_ID,
    USER_STOP,
    WAIT,
    Change,
    CompletedOperation,
    ConditionFiring,
    ConditionVerdict,
    ConditionWait,
    InferenceResult,
    InputWait,
    OperationInvocation,
    PendingCondition,
    PendingConditionState,
    PropertyChange,
    RelevanceCandidate,
    SignalWait,
    Step,
    changes_of,
    diff_values,
    goal_kind_of,
    path_matches,
    walk_path,
    watch_matches,
)

# The dotted-path walker lives in sora.types now (shared with the data-ops in sora.action, which
# can't import this module); keep the private alias so the many call sites below stay untouched.
_walk_path = walk_path


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


def _lift_step_conditions(step: Step, activity: Activity, wm: WorkingMemory) -> None:
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
    if step.params.get("mode", "deliberative") == "deliberative":
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
        match = DefaultObserveStrategy._match_signal(
            wm, state.condition.watch, since=state.evaluated_through
        )
        if match is None:
            match = DefaultObserveStrategy._match_derived(
                wm, state.condition.watch, since=state.derived_through
            )
        if match is not None:
            eligible.append((state, match))
    return eligible


# Bound on retained signals. Nothing is ever evicted for being *matched* — `wm.signals` is a shared
# broadcast log, so one waiter's satisfaction must not remove an event another waiter (or a strategy
# reading the log directly) still needs, and a signal that arrives before its waiter has to survive
# to the cycle that suspends. The retention cap is therefore the only eviction, and it runs after
# both the suspend and resume passes so a signal that arrived this tick is matched before it can be
# a candidate. The newest win. Deliberately simple; note the cap now bounds something with more
# consequence than it used to — losing a signal loses not just the fact that an event happened but
# the only record of *where*, which no later snapshot can reconstruct.
_SIGNAL_RETENTION = 256

# Bound on retained derived changes. Deliberately larger than `_SIGNAL_RETENTION` and deliberately
# a separate number: the two logs are sized against different windows. A signal accrues per
# environment event, a derived change per observation cycle in which anything moved -- so on a tool
# whose property churns, this log fills at tick rate (~11/s on a live scenario) rather than at event
# rate. What an entry must survive is the gap between its append and the cycle the watching activity
# is free to judge it, which is bounded by how long an inference can occupy that activity
# (`DEFAULT_INFERENCE_DEADLINE`). 1024 covers ~90s of worst-case single-property churn, which clears
# a slow thinking-model call; sizing it to the full deadline would not help, since under that much
# churn the newest-win policy retains the churn and evicts the real change regardless -- that regime
# needs the property excluded from derivation, not a bigger buffer.
_DERIVED_RETENTION = 1024

# How long an in-flight infer/ground may say nothing before Observe gives up on it. Generous by
# design: it is a watchdog for a seam that has stopped answering, not a latency budget — a thinking
# model legitimately spends tens of seconds, and expiring a call that was about to succeed buys a
# replan nobody needed. The client-side stall timeout is the tighter, better-informed bound (it can
# tell a quiet socket from a slow one); this one exists because not every failure reaches it.
DEFAULT_INFERENCE_DEADLINE = 300.0

# Cycles between per-tool observation summaries (DEBUG only). The question this answers is the one a
# `[cycle N]` trace structurally cannot: cycle numbers alone say how many ticks ran, never WHEN, so
# a run whose ticks were bunched at the end of a window is indistinguishable from one that ticked
# evenly through it — and those two readings of a missed change (never looked / looked and lost it)
# call for completely different fixes. Counts plus elapsed wall time separate them. Periodic rather
# than per-cycle because Observe runs at ~11/s on a live scenario, where a line per tick buries the
# trace it is meant to explain.
_OBSERVATION_HEARTBEAT = 100

# How long the retire-only condition sweep waits before re-judging one activity's quiet conditions,
# and how far it backs off after a judgement that retired nothing (the interval doubles per miss,
# topping out at 2**_RETIREMENT_BACKOFF times the base). Backoff is not a refinement here but the
# thing that makes the sweep affordable: an activity BLOCKED on a condition makes *every* tick idle,
# so the eligibility that schedules this never lapses on its own, and a fixed interval would keep
# paying for the same "still waiting" answer for as long as the agent lives. A miss counter resets
# whenever a condition is declared or retired, so a set that actually moves is checked promptly
# again. What the backoff does not solve — an activity waiting on a healthy-looking condition that
# has been quiet for a very long time — is a question for the user, and deliberately not answered
# here (ADR-0027's last open consequence).
#
# Wall-clock on purpose, and this is emphatically NOT the clock an `until` is answered against.
# This paces how often the runtime *asks*, which is infrastructure timing like the inference
# watchdog above; the `until` itself is a question about **domain** time, which reaches the runtime
# through the workspace and never through `time.time()` (ADR-0027 §5). A tick count would be the
# wrong unit for the pacing anyway: tick rate varies by two orders of magnitude between an
# interactive session and a simulation loop.
DEFAULT_RETIREMENT_INTERVAL = 30.0
_RETIREMENT_BACKOFF = 5

if TYPE_CHECKING:
    from sora.activity import Activity
    from sora.cycle import DecisionCycle
    from sora.environment import DomainClock, EnvironmentView
    from sora.manual import Manual
    from sora.memory import WorkingMemory
    from sora.perception import Message
    from sora.types import InterruptRequest, Plan, Signal

log = logging.getLogger("sora.strategies")


# ── Context-adaptation reconsideration (ADR-0024) ───────────────────────────────────────────────
# How eagerly the cycle re-validates an in-progress plan against new perception, as a pluggable
# policy gating an off-cycle revalidation. Per-agent (via agent.yaml strategies.context_adaptation);
# the *act* of reconsidering stays cycle-owned (ADR-0022/0019) — the policy only decides WHEN.


class ReconsiderationPolicy(Protocol):
    def should_check(self, side_effecting: bool | None) -> bool:
        """Given the side-effecting-ness of the step Reason is about to commit (True write, False
        read, None unknown), decide whether to run the (gated) validity check before it."""
        ...


class NoneReconsideration:
    """``context_adaptation: none`` — never reconsider on ambient percepts (blind commitment).
    Failure-driven re-planning stays orthogonal and always on."""

    def should_check(self, side_effecting: bool | None) -> bool:
        return False


class BeforeWrites:
    """``context_adaptation: before_writes`` (the default) — check before a side-effecting step,
    where acting on a stale plan does damage. Skips reads (side_effecting is False); an unknown
    (None) is treated as a write, so it is checked (conservative)."""

    def should_check(self, side_effecting: bool | None) -> bool:
        return side_effecting is not False


class BeforeEachOp:
    """``context_adaptation: before_each_op`` — check before EVERY external step, read or write.
    Maximum caution; still op-gated, so it skips planning/grounding/waiting cycles."""

    def should_check(self, side_effecting: bool | None) -> bool:
        return True


# WM/attention actions and WAIT never mutate the world, so they are never "writes"; every other
# non-invoke external action (e.g. send) is unknown -> treated as a write by before_writes.
_NON_SIDE_EFFECTING_ACTIONS = frozenset(
    {FocusAction.name, UnfocusAction.name, JoinAction.name, LeaveAction.name, WAIT}
)


def _step_side_effecting(step: Step, wm: WorkingMemory) -> bool | None:
    """Whether committing ``step`` mutates the world: an invoke defers to the operation's
    ``OperationSpecification.side_effecting`` (None = unknown); a WM/attention action or WAIT is a
    definite read (False); any other external action is unknown (None)."""
    if step.next_action == InvokeAction.name:
        manual = _manual_for(wm, step.params.get(TOOL_ID))
        op = manual.operation(step.params.get(OPERATION_NAME, "")) if manual is not None else None
        return op.side_effecting if op is not None else None
    if step.next_action in _NON_SIDE_EFFECTING_ACTIONS:
        return False
    return None


def _perception_signature(wm: WorkingMemory) -> tuple[Any, ...]:
    """A compact, comparable signature of current perception — the cheap mechanical change-gate
    behind the reconsideration check (ADR-0024). No domain knowledge: the replace-by-key property
    snapshot (each property by its *payload* repr) plus the append-log lengths. Equal signatures
    mean nothing observable moved since the plan was baselined, so the re-check is skipped (free
    when the world is static). Keyed on `percept.payload` (the ObservableProperty value), NOT whole
    Percept — the envelope's `observed_at` is refreshed with `time.time()` on every re-observation
    (`_snapshot_properties`), so hashing the whole Percept would make an unchanged property look
    like it moved every cycle and revalidate on every write even in a static world."""
    properties = tuple(
        sorted(
            (f"{source}\x1f{name}", repr(percept.payload))
            for (source, name), percept in wm.properties.items()
        )
    )
    return (properties, len(wm.signals), len(wm.messages))


class ChangeGate(Protocol):
    """The cheap mechanical test the reconsideration checkpoint runs *before* a revalidation:
    produce a comparable signature of perception, so equal signatures across cycles mean nothing
    observable moved since the plan was baselined (ADR-0024). Orthogonal to ReconsiderationPolicy,
    which decides *which* steps are checkpoints (WHEN); the gate decides *whether* the world moved.
    A domain gate that projects perception onto only its externally-meaningful part filters the
    agent's *own* writes here — the same efference trick a stateful InterruptPolicy uses, applied to
    the cooperative path. The signature is stored as ``object`` (PendingInference.baseline /
    Activity.reconsider_baseline), so a gate may return any comparable value."""

    def signature(self, wm: WorkingMemory) -> object: ...


class PerceptionSignatureGate:
    """The runtime default ChangeGate: domain-free. The replace-by-key property snapshot (by repr)
    plus the signal/message append-log lengths. A self-caused write still moves it (a new
    ``state_changed`` signal, a changed property), so under this default the checkpoint spends one
    revalidation on the agent's own writes; a domain ChangeGate that projects to only the external
    surface is how an application removes that (e.g. an INBOX-id gate that self-writes to SENT /
    read-flags / calendar don't move)."""

    def signature(self, wm: WorkingMemory) -> object:
        return _perception_signature(wm)


@dataclass(frozen=True)
class TickResult:
    """The decision surface for one cycle. Every phase strategy receives and returns one of these.
    Whatever's still None, DecisionCycle fills in by calling the next phase's own strategy — so a
    fully-decomposed configuration produces one field at a time, and a fused Situate can fill in
    step/invocation too, deciding the rest of the cycle in one call. Lives only for the duration of
    one tick() call — nothing persists across cycles, so there's no cache to key or invalidate.

    A freeform per-tick scratchpad for multi-call strategy configurations (e.g. a fused Situate
    passing notes to a separate, focused Act) is a foreseen addition, deferred until the first such
    configuration actually exists."""

    activity: Activity | None = None
    step: Step | None = None  # this cycle's decision — not the whole (possibly multi-step) Plan
    invocation: OperationInvocation | None = None


class ObserveStrategy(Protocol):
    async def observe(self, cycle: DecisionCycle) -> TickResult:
        """Mutates cycle.working (perceptions, messages) as a side effect — same as the default
        below. Default: mechanical, no model call, returns an empty TickResult(). An LLM-backed
        Observe is for interpreting raw perception itself (e.g. describing a camera snapshot), not
        for deciding the cycle — decision-chain fusion starts at Situate, not here."""
        ...


class ReflectStrategy(Protocol):
    async def reflect(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        """Decides whether this activity just completed or failed — deterministic or model-backed,
        depending on the application — and if so, summarizes and stores to episodic memory. (The
        default does NOT auto-cache the completed plan to procedural memory — replaying a stored
        plan verbatim is unsound; distilling reusable procedures from episodes is future work.) The
        completion judgment is
        synchronous — it must land before Situate selects, so a just-completed activity is never
        re-selected the same cycle — while the summarize/store side effects are dispatched
        asynchronously and never block the cycle; several activities may terminate in the same
        cycle. Passes `result` through, optionally adding to it. Default: performs the completion
        check and the store-on-success, leaves TickResult's other fields untouched. `cycle` is what
        makes these memory calls possible at all — previously missing from this Protocol despite
        the calls it was already documented as making."""
        ...

    def failed(self, activity: Activity) -> bool:
        """This strategy's own judgment of whether `activity` has failed — a *judgment call*, not
        a fact recorded on `Activity` itself, since a different ReflectStrategy may define failure
        differently (e.g. from a signal, or a partial-success rule) than the default's "resolved
        operation, not ok" rule. Exposed on the Protocol (not just the default) so callers outside
        the decision cycle — a reporting hook, a test assertion — go through whichever strategy is
        actually configured rather than re-deriving the rule themselves."""
        ...


class SituateStrategy(Protocol):
    async def situate(
        self,
        activities: list[Activity],
        wm: WorkingMemory,
        cycle: DecisionCycle,
        result: TickResult,
    ) -> TickResult:
        """Selects the next activity and adjusts wm for it. Always runs — unlike Reason/Act it is
        not gated on its own output field, because adjusting wm (selecting tools, loading/unloading
        manuals, filtering percepts) must reflect this cycle's fresh percepts even for an
        already-selected activity. Selects only if result.activity is still None; a pre-set
        selection (uncommon — e.g. an Observe that pins the activity handling a critical signal) is
        respected and situated, not overridden. Also responsible for activity creation: if
        wm.messages includes a new goal delegation, invokes the internal _create_activity_ action
        (via cycle) before selecting. Head of the decision chain (Situate -> Reason -> Act) and the
        intended entry point for fusing the remaining phases into one model call — it runs after
        this cycle's percepts and messages are already in working memory. May additionally fill in
        step/invocation, short-circuiting Reason/Act (those forward-fusion gates remain; only
        Situate's own activity gate is removed)."""
        ...


class ActivitySelectionStrategy(Protocol):
    async def select(
        self, ready: list[Activity], wm: WorkingMemory, cycle: DecisionCycle
    ) -> Activity | None:
        """Picks the activity to progress this cycle from the ready set (empty -> None). A
        scheduling policy, not a phase: it decides *which* ready activity runs, nothing else — the
        caller (Situate) folds the pick into TickResult; fusing step/invocation stays a full
        SituateStrategy concern. `async` + the `cycle` handle are for a richer policy (priority,
        aging, deadlines, or an LLM-based scheduler) that consults memory or a model; the mechanical
        default consults neither."""
        ...


class FocusPolicy(Protocol):
    def attend(self, wm: WorkingMemory) -> set[str]:
        """Tool ids the agent should be attending to this cycle.

        Pure and set-valued: the caller owns the diff against `wm.focused_tools` and performs the
        focus/unfocus, so a policy never touches the environment and can be unit-tested as a
        function. A scheduling-style sub-strategy, not a phase — the same shape as
        `ActivitySelectionStrategy` (ADR-0016), and injected the same way (a constructor argument
        to the default Observe strategy, not an `agent.yaml` key)."""
        ...


class ReasonStrategy(Protocol):  # pluggable; default targets 1 LLM call/cycle
    async def reason(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        """Only called if result.step is still None. Typical implementation: if activity.plan is
        already set and still valid, just read activity.plan.steps[activity.step_index] and
        advance the index — no model call. Otherwise, retrieve a cached Plan via
        cycle.procedural.retrieve() or infer a new one (the expensive path), reset step_index to
        0, and use its first Step. Deciding when a plan counts as invalidated is entirely up to
        the implementation. May additionally fill in invocation, short-circuiting Act — this is
        where the historical 'tool hallucination' risk lives if it does."""
        ...


class ActStrategy(Protocol):
    async def bind(
        self, step: Step, manual: Manual | None, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        """Only called if result.invocation is still None. This is *parameter binding*: grounding
        an abstract Step into a concrete, schema-conformant OperationInvocation (the tool-
        hallucination-prone step — where "email the boss" becomes validated `{to, subject, ...}`).
        Distinct from a *protocol binding* (WoT forms/security, an MCP session), which is how the
        adapter's Tool actually reaches the instance and never surfaces here — see ADR-0015. `cycle`
        is available for implementations that cache bindings (e.g. belief-state -> params) rather
        than re-deriving one every time."""
        ...


@dataclass(frozen=True)
class Strategies:  # bundles the five, so DecisionCycle.__init__ doesn't take five loose params
    observe: ObserveStrategy
    reflect: ReflectStrategy
    situate: SituateStrategy
    reason: ReasonStrategy
    act: ActStrategy


class RelevanceJudge(Protocol):
    def consider(self, cycle: DecisionCycle) -> Awaitable[None]:
        """Called on an IDLE tick — one where Situate selected nothing, so either there is no
        schedulable activity or everything schedulable is already awaiting a model (ADR-0026).

        Scheduling, not triggering: an unclaimed change makes this eligible the moment it lands, but
        it never runs in preference to an activity that could actually advance. Implementations must
        return promptly — fire any model call off-cycle and apply the result on a later call — so an
        idle tick stays responsive to arriving signals.
        """
        ...


class DefaultRelevanceJudge:
    """Undeclared-relevance recovery (ADR-0026): notice that a change bears on work that already
    finished, ask the user, and amend rather than reopen.

    Deliberately **opt-in**. It spends a model call on an unverifiable judgement and, when it fires,
    interrupts a person — and its own ADR records that with nobody available to ask, the safe
    degradation is to not act. An unattended run should therefore get the declared-condition layer
    and nothing else unless someone chose otherwise.

    Its input is only what the declared gates left unclaimed, so every condition the planner learns
    to declare removes work from here.
    """

    def __init__(self, *, window: int = 10, max_asks: int = 3) -> None:
        # Neither number is principled, which is why both are settings rather than constants.
        # `window` too small silently drops old-but-live commitments and too large grows the prompt
        # and the error rate together; `max_asks` too high pesters the user until they stop reading
        # and too low reproduces the miss this exists to prevent.
        self._window = window
        self._max_asks = max_asks
        self._mark = 0  # high-water over wm.signals_appended — this judge's own, like any waiter's
        self._asks = 0
        self._in_flight = False
        self._result: RelevanceCandidate | None = None
        self._declined: set[tuple[str, str]] = set()
        self._tasks: set[asyncio.Task[None]] = set()

    async def consider(self, cycle: DecisionCycle) -> None:
        # Apply first: a parked result is applied INSIDE the tick, so every mutation of working
        # memory happens on-cycle. The background call only ever sets a field — the same discipline
        # the sinks enforce for infer/ground/invoke (ADR-0021), with a one-slot mailbox instead of
        # a queue because at most one judgement is ever in flight.
        if self._result is not None:
            candidate, self._result = self._result, None
            await self._amend(cycle, candidate)
            return
        if self._in_flight or self._asks >= self._max_asks:
            return
        unclaimed = self._unclaimed(cycle.working)
        # Advance the mark whether or not anything is judged: a change that opened no gate and did
        # not reach a call has still been considered, and re-considering it every idle tick would
        # turn an idle agent into a spend loop.
        self._mark = cycle.working.signals_appended
        if not unclaimed:
            return
        episodes = await cycle.episodic.consult_recent(self._window)
        episodes = [e for e in episodes if isinstance(e, dict) and e.get("activity_id")]
        if not episodes:
            return
        # Paired with the source that reported them, exactly as the declared-condition gate does:
        # a `Change` names the path that moved but not the tool it moved on, and the judgement
        # needs both to dereference the ids back into records (ProceduralMemory.render_changes).
        changes: list[tuple[str, Change]] = []
        for percept in unclaimed:
            changes.extend((percept.source, change) for change in changes_of(percept.payload))
        observed = PerceptSnapshot(
            list(cycle.working.properties.values()), list(cycle.working.signals)
        )
        self._in_flight = True
        _spawn_tracked(self._tasks, self._call(cycle, episodes, changes, observed))

    async def _call(
        self,
        cycle: DecisionCycle,
        episodes: list[Any],
        changes: list[tuple[str, Change]],
        observed: PerceptSnapshot,
    ) -> None:
        try:
            candidate = await cycle.procedural.judge_relevance(episodes, changes, observed)
        except Exception:  # noqa: BLE001 — a failed judgement means "nothing follows up", not a crash
            log.exception("relevance: judgement failed")
            candidate = None
        finally:
            self._in_flight = False
        self._result = candidate

    def _unclaimed(self, wm: WorkingMemory) -> list[Percept]:
        """Signals past this judge's mark that opened NO declared gate.

        The subtraction that defines this layer's input. A change claimed by some activity's
        pending condition is layer 1's business and is never offered here — an accepted false
        negative, since that same change might also have borne on an unrelated finished activity,
        but the alternative is judging every signal against every terminated activity.
        """
        watches = [
            state.condition.watch
            for activity in wm.activities.values()
            for state in activity.pending_conditions
        ]
        first_seq = wm.signals_appended - len(wm.signals)
        out: list[Percept] = []
        for offset, percept in enumerate(wm.signals):
            if first_seq + offset < self._mark:
                continue
            signal = percept.payload
            claimed = any(
                w.signal_name == signal.name
                and (w.source is None or percept.source == w.source)
                # `path`, deliberately NOT `kind`, unlike the eligibility gate: `kind` says what may
                # *open* a gate, not what a gate is answerable for. Narrowing here inverts the
                # purpose it was added for — a watch declared `added` on a collection the agent also
                # deletes from is exactly the shape `kind` exists to spare a judge call, and reading
                # it here hands that same delete to *this* judge instead, which additionally
                # interrupts a person. The wider test keeps the saving where it was won.
                and path_matches(w.path, changes_of(signal))
                for w in watches
            )
            if not claimed:
                out.append(percept)
        return out

    async def _amend(self, cycle: DecisionCycle, candidate: RelevanceCandidate) -> None:
        """Create the amending activity — born BLOCKED on an InputWait, so the user is asked before
        the agent acts on a goal nobody stated.

        A NEW activity, never the terminated one revived: an episode is a historical claim about
        what was attempted and how it ended, and editing one to make it current would turn
        `succeeded` into a retrospective lie in the very record the agent learns from.

        Through `_await_input` like every other breaker, because the wait and the *asking* are two
        halves of one act: parking on a question that was never delivered would leave this layer
        silently inert, and would let the user's next unrelated message be read by
        `_resume_on_input` as consent to an amendment they were never shown.

        Consent needs no mechanism of its own. `_resume_on_input` already clears an InputWait on a
        user Message, drops the (empty) plan, and re-infers with the reply and history visible — so
        a decline is answered by the same path a go-ahead is.
        """
        if (candidate.episode_id, candidate.goal) in self._declined:
            return
        self._declined.add((candidate.episode_id, candidate.goal))
        self._asks += 1
        activity = Activity(
            id=uuid.uuid4().hex,
            goal=candidate.goal,
            # The amendment points back at what it amends; the original stays terminated and its
            # episode untouched.
            context={"amends": candidate.episode_id},
        )
        # Ask first, register after: nothing else runs between the two, and an activity is never
        # visible in working memory in any state but the blocked one it is born in.
        await _await_input(cycle, activity, candidate.question)
        cycle.working.activities[activity.id] = activity
        log.info(
            "relevance: proposing amendment to episode %s -> %r",
            candidate.episode_id,
            candidate.goal,
        )


class InterruptPolicy(Protocol):
    def decide(self, source: str, signal: Signal, wm: WorkingMemory) -> InterruptRequest | None:
        """Consulted synchronously the instant a signal is pushed to signal_sink — before the once-
        per-cycle Observe drain. Return an InterruptRequest to preempt the current phase (a hard
        interrupt), or None to let the signal flow cooperatively (reacted to at the next cycle
        boundary). Sync because push is sync. A stateful policy may diff the signal vs remembered
        state (e.g. a set of inbox ids) to fire only on a new external event and filter the
        agent's own writes — the only distinguishable-external test available until read-write/
        efference tagging lands. Default: NeverInterruptPolicy — no signal ever preempts."""
        ...


class NeverInterruptPolicy:
    """The runtime default: a pushed signal never becomes an interrupt, so the cooperative signal
    path (drained in Observe, resumes a BLOCKED activity) is unchanged. With no runtime way yet to
    tell the agent's own writes from external events, preempting on a signal would risk a self-write
    loop; opting in is a deliberate, application-supplied policy."""

    def decide(self, source: str, signal: Signal, wm: WorkingMemory) -> InterruptRequest | None:
        return None


class InterruptHandler(Protocol):
    async def handle(
        self, request: InterruptRequest, wm: WorkingMemory, cycle: DecisionCycle
    ) -> bool:
        """Runs after tick() aborts on a pending interrupt — the process-scheduling 'interrupt
        handler'. The interrupted activity's context is already saved (durable on Activity; the per-
        tick TickResult was discarded, immune to interrupt staleness per ADR-0011), so this only
        decides the follow-up: map each targeted activity onto an existing state — READY (resume, or
        replan by clearing plan/step_index), BLOCKED via InputWait (await the user's next
        instruction), or TERMINATED (drop) — then the ActivitySelectionStrategy picks
        next. Never abandons an in-flight *external* op (side effects): a RUNNING activity is left
        RUNNING and revisited at the next checkpoint once its ack resolves. Returns True once every
        targeted activity is routed (the request is discharged and cleared), False when some are
        still RUNNING and the request must be revisited next tick."""
        ...


class DefaultInterruptHandler:
    """The runtime default: a user stop. Pauses each targeted, schedulable (READY) activity to a
    resumable point via an InputWait, so the agent halts current work but stays alive; a later user
    Message resumes it (DefaultObserveStrategy._resume_on_input). An activity mid-external-op
    (RUNNING on a pending_operation) is left to finish and routed on a later checkpoint, so a
    physical side effect always runs to completion. An activity RUNNING only on an off-cycle
    infer/ground (pending_inference) has no side effect to protect, so it is paused *now* — its
    inference invalidated (discarded on resolve) — rather than waiting out an unbounded model call
    (ADR-0021). target=None is agent-wide; a named target pauses just that activity.

    It is a *user-stop* handler, not a general router: it recognizes only the USER_STOP signal and
    treats any other interrupt as unrouted — pausing to await human instruction is the fail-safe
    fallback (halt-and-ask, never barrel ahead), logged at warning level since no handler claimed
    it. There is no general runtime answer for an arbitrary interrupt signal; that is why a custom
    InterruptPolicy should ship a paired InterruptHandler for its own signals (see ADR-0020). With
    the default components only a CLI /stop ever reaches here."""

    async def handle(
        self, request: InterruptRequest, wm: WorkingMemory, cycle: DecisionCycle
    ) -> bool:
        if request.signal.name != USER_STOP:
            # Unrouted interrupt: a custom policy raised it but no paired handler claimed it. Fall
            # back to the same halt-to-await-input as a user stop (fail-safe), but log it — a silent
            # strand (an activity blocked on input no user message ever arrives to satisfy, e.g.
            # headless) is otherwise invisible. Still routed, not dropped: swallowing an interrupt a
            # policy deliberately raised would be as surprising as stranding on it.
            log.warning(
                "no interrupt handler routed signal %r; pausing targeted activities to await input "
                "as a fallback (a custom InterruptPolicy should ship a paired InterruptHandler)",
                request.signal.name,
            )
        if request.target is None:
            targets = list(wm.activities.values())
        else:
            target = wm.activities.get(request.target)
            targets = [target] if target is not None else []
        pending = False
        for activity in targets:
            if activity.state is ActivityState.RUNNING and activity.pending_operation is not None:
                pending = True  # external op in flight: let it finish, route on a later checkpoint
            elif activity.state in (ActivityState.READY, ActivityState.RUNNING):
                # READY, or RUNNING only on a side-effect-free inference: drop the inference (no
                # reason to wait out an unbounded model call) and pause to await the user's next
                # instruction.
                activity.discard_inference()
                activity.state = ActivityState.BLOCKED
                activity.blocked_on = InputWait(prompt=request.signal.name)
        return not pending


# Inference kinds whose outright failure is recoverable by planning again: nothing was attempted, so
# the world is untouched and a fresh attempt is free to differ. `select`/`condition`/`revalidate`
# are absent because they already degrade in place (an empty shortlist, nothing fired, assume
# valid) — cheaper still, since they keep the plan.
#
# `then` belongs here for the same reason `subgoal` does — it *is* a sub-goal, planned
# deliberatively and landing a plan, differing only in that it pushes no frame. Leaving it out was
# worse than leaving `subgoal` out: a `then` is pursued only once the body is idle, which for the
# monitoring goals that declare conditions means a suspended parent frame is intact underneath it
# every time, so the residual branch's terminate destroyed an activity mid-window.
_REPLANNABLE_INFERENCE = frozenset({"plan", "subgoal", "ground", "then"})


def _inference_defect(kind: str, error: str) -> str:
    """The replan defect for an inference that failed outright, normalized to its *cause*.

    Normalization is what makes the runaway-replan breaker able to see a repeat. Trail entries are
    compared for equality (`_replanning_would_loop`), and `InferAction` reports `repr(exc)`, whose
    message quotes the model output that defeated the parse — different every attempt. Carrying it
    verbatim would mean two hopeless calls never compared equal, so the precise "abandoned for the
    same reason twice" check could never fire and even a permanent failure would be paid for
    `max_replan_attempts` times. The full message is not lost: it is logged at the failure site.
    """
    cause = error.split("(", 1)[0].strip() or error
    return f"the {kind} inference did not return a usable result ({cause})"


async def _report_to_user(cycle: DecisionCycle, text: str) -> None:
    """Say something to the user on the agent's own channel, from the runtime rather than a plan.

    The same transport call `runtime-io`'s `send_message_to_user` makes — used directly because
    there is no plan left to route through at the points that need it (an activity being abandoned,
    or parked on a question). Failures are logged, never raised: this runs on paths that are already
    reporting bad news, and a dead transport must not replace one failure with another.
    """
    try:
        await cycle.communication.send("user", {"text": text})
    except Exception:  # noqa: BLE001 — a transport failure must not mask what we were reporting
        log.exception("could not deliver a runtime message to the user: %s", text)


async def _await_input(cycle: DecisionCycle, activity: Activity, prompt: str) -> None:
    """Park an activity on the user's next instruction *and actually ask the question*.

    A breaker that sets `blocked_on` without delivering `prompt` stops the agent on a question no
    one can hear: `_resume_on_input` waits for a Message that the user has no reason to send. The
    two halves belong together, so every breaker goes through here rather than setting the fields
    itself. Deliberately not used for the hard-interrupt pause — the user caused that one and does
    not need to be told they did it.
    """
    activity.state = ActivityState.BLOCKED
    activity.blocked_on = InputWait(prompt=prompt)
    await _report_to_user(cycle, prompt)


def _prop_tool_ids(params: Any, joined: set[str]) -> set[str]:
    """Tool ids named by any ``$prop`` reference nested anywhere inside a step's params.

    A ``$prop`` is the one way a step reads a tool it never invokes, and it lives *inside* the
    param bag — under a data-op's ``in``, a sub-goal's collection — not under ``TOOL_ID``. Missing
    it releases the tool mid-plan and the reference then resolves to nothing, which is the silent
    blindness intention-scoped focus exists to remove. Recursive for that reason: depth is
    not bounded by the schema.

    The tool id is found by matching against the *joined* id set rather than by splitting on the
    first dot, for the reason `_property_ref` documents at length: neither half of a property key
    is dot-free. Longest match wins. A bare property name (``{"$prop": "state"}``) names no tool
    and contributes nothing — it stays resolvable only while something else attends its tool.
    """
    found: set[str] = set()
    if isinstance(params, dict):
        ref = params.get(_REF_PROP)
        if isinstance(ref, str):
            candidates = [t for t in joined if ref == t or ref.startswith(f"{t}.")]
            if candidates:
                found.add(max(candidates, key=len))
        for value in params.values():
            found |= _prop_tool_ids(value, joined)
    elif isinstance(params, (list, tuple)):
        for item in params:
            found |= _prop_tool_ids(item, joined)
    return found


def _wait_sources(wait: SignalWait | InputWait | ConditionWait | None) -> set[str]:
    if isinstance(wait, SignalWait):
        return {wait.source} if wait.source else set()
    if isinstance(wait, ConditionWait):
        return {w.source for w in wait.watches if w.source}
    return set()  # InputWait waits on the user, not on a tool


def referenced_tools(activity: Activity, joined: set[str]) -> set[str] | None:
    """The tools one activity's live intentions reference — or ``None`` for "the whole world".

    ``None`` is not "nothing": an activity with no plan is about to be planned, and planning is
    deliberately broad (the planner picks ``$prop`` over paginated scanning by reading property
    *shapes* in the plan prompt, so narrowing it there would go dark on ADR-0023 discovery). Since
    `reset_for_replan` clears `plan`, that same clause covers the whole replan window — which is
    why attention needs no grace period and no history retention.

    Shared by the attention policy (agent-level union) and by the Stage-2 per-activity prompt view,
    so the two layers cannot disagree about what an activity's tools are.
    """
    if activity.state is ActivityState.TERMINATED:
        return set()
    if activity.plan is None:
        return None
    ids: set[str] = set()
    for plan in [activity.plan, *(frame[0] for frame in activity.parent_frames)]:
        for step in plan.steps:
            # An `unfocus` step names a tool in order to STOP attending to it; counting it here
            # would attend the tool right back and make the step a permanent no-op. Note this only
            # gets the tool no OTHER step names: the scan covers every step of a live plan, not the
            # un-run tail (that is what removes the need for history retention), so an `unfocus`
            # following an `invoke` of the same tool still reads as referenced here. That is not the
            # last word — once the `unfocus` step actually runs, `WorkingMemory.suppressed_tools`
            # holds the release against this derived floor, so the explicit act wins.
            if step.next_action != UnfocusAction.name:
                tool_id = step.params.get(TOOL_ID)
                if isinstance(tool_id, str):
                    ids.add(tool_id)
            ids |= _prop_tool_ids(step.params, joined)
        # A declared condition watches a tool the body may never touch (waiting on a reply in a
        # messaging app while every step touches email) — not redundant with the step scan.
        for condition in plan.pending:
            if condition.watch.source:
                ids.add(condition.watch.source)
    for state in activity.pending_conditions:
        if state.condition.watch.source:
            ids.add(state.condition.watch.source)
    if activity.pending_operation is not None:
        ids.add(activity.pending_operation.invocation.tool_id)
    ids |= _wait_sources(activity.blocked_on)
    return ids


class IntentionScopedFocus:
    """Attend exactly the tools the agent's live *intentions* reference — the narrowing policy,
    opt-in rather than default.

    BDI vocabulary on purpose — this set is derived from committed plans, not from candidate goals,
    so it is intention-scoped and not desire-scoped. The union over live activities is an eagerly
    evaluated refcount: a tool leaves focus precisely when no live plan names it any more. Kept as
    a recomputation rather than incremental leases because a lease has to be maintained at every
    plan mutation site, and a missed decrement leaks cost silently while a double decrement blinds
    a tool that is still in use — going silently blind, the failure this policy exists to
    remove. Analysis and measurement: `docs/architecture/notes/
    attention-scoped-to-live-intentions.md`.

    Broad while nothing is planned (no activities at all, or any activity awaiting a plan), narrow
    while executing.
    """

    def attend(self, wm: WorkingMemory) -> set[str]:
        joined = {tool.id for tool in wm.registry.all_tools()}
        live = [a for a in wm.activities.values() if a.state is not ActivityState.TERMINATED]
        if not live:
            return joined  # an idle agent stays observant (ADR-0026's judge reads the world)
        attended: set[str] = set()
        for activity in live:
            referenced = referenced_tools(activity, joined)
            if referenced is None:
                return joined
            attended |= referenced
        # Intersect last, so an id left over from a departed workspace can never be re-attended.
        return attended & joined


class FocusAllJoined:
    """Attend every joined tool, ignoring activities — **the default**.

    Not a relic of the auto-focus-on-join fallback: attention is still reconciled every
    Observe by the same mechanism, so perception never hinges on the model emitting a
    `focus` step. What this policy declines to do is *narrow*. Measured, narrowing is worth
    ~825 prompt tokens per model call and **zero** judge calls on the shipped configs — the
    condition judge already gates on `watch.source`, and the relevance judge is opt-in and
    off — while it introduces churn across replans, where re-attaching re-baselines an
    adapter and a change occurring in the gap is absorbed rather than reported. That failure
    is silent, so it is not worth cents against a benchmark result.

    `IntentionScopedFocus` is the opt-in for a dynamic, many-workspace run, where the
    broad set grows with the environment while the narrow one stays constant."""

    def attend(self, wm: WorkingMemory) -> set[str]:
        return {tool.id for tool in wm.registry.all_tools()}


def scoped_snapshot(wm: WorkingMemory, activity: Activity) -> PerceptSnapshot:
    """The agent's percepts narrowed to one activity's own tools — the second attention layer.

    Attention is agent-level (a union over live activities, because focusing is a subscription and
    one subscription serves everyone). A *prompt* is not: a model call fired for one activity has
    no use for a sibling activity's tools, and with several activities in flight the union is again
    the thing that grows. This closes that gap at the `PerceptSnapshot` boundary every call already
    takes, which makes it **non-destructive**: `wm.properties` is untouched, so the ADR-0024 change
    gate keeps hashing a stable agent-level world and the idle-tick relevance judge keeps reading
    it. A destructive per-activity `_filter_` would instead make the shared store depend on which
    activity the scheduler happened to pick, and move the change signature when nothing moved.

    Deliberately *not* applied everywhere. Planning stays broad (`_infer_` reads property shapes to
    choose `$prop` over paginated scanning), both judges stay broad (they dereference change ids
    against these same properties, and a watch source is routinely a tool no step names), and plan
    revalidation stays broad — its gate fires on *any* perception change, so narrowing it to the
    plan's own tools would blind the judge to the very change that woke it. It applies where the
    question genuinely concerns one step of one activity: `_ground_` and `_select_`.
    """
    # Follows the FocusPolicy rather than deciding for itself. An agent on the broad default has
    # declined to narrow *because* a wrongly narrowed view fails silently — the model simply reasons
    # without the property and nothing reports a miss — so narrowing its prompts anyway would
    # reintroduce exactly the risk the policy choice was made to avoid, one layer down and
    # invisibly. This layer only sharpens a narrowing the agent already opted into: from the
    # agent-level union to the one activity the call is actually about.
    if not wm.attention_narrowed:
        return PerceptSnapshot(list(wm.properties.values()), list(wm.signals))
    joined = {tool.id for tool in wm.registry.all_tools()}
    referenced = referenced_tools(activity, joined)
    if referenced is None:  # unplanned -> the same breadth attention gives it
        return PerceptSnapshot(list(wm.properties.values()), list(wm.signals))
    return PerceptSnapshot(
        [percept for key, percept in wm.properties.items() if key[0] in referenced],
        [percept for percept in wm.signals if percept.source in referenced],
    )


class DefaultObserveStrategy:
    """The runtime's built-in default — purely mechanical, no LLM.

    Owns two reconciliations of activity state against the world, both before any perception is
    read: what the agent *attends to* and what it is *waiting for* (`_suspend_`/
    `_resume_`, ADR-0019). Attention belongs here rather than in Situate because focusing is a
    subscription to the environment, not a view over memory — it decides what can be perceived at
    all, so deciding it after Observe would leave it structurally one tick behind."""

    def __init__(
        self,
        focus: FocusPolicy | None = None,
        *,
        inference_deadline: float | None = DEFAULT_INFERENCE_DEADLINE,
        retirement_interval: float | None = DEFAULT_RETIREMENT_INTERVAL,
    ) -> None:
        # Defaults to attending everything joined. Narrowing to live intentions is a
        # deliberate opt-in (`focus=IntentionScopedFocus()`), not the default: measured, it
        # is worth ~825 prompt tokens per model call and zero judge calls, against a
        # re-baselining risk that fails silently. A benchmark number is worth more than
        # that saving. See the attention design note.
        self._focus = focus or FocusAllJoined()
        # None disables the watchdog entirely — for a deliberately unbounded interactive session,
        # or a debugger sitting on a breakpoint inside the client.
        self._inference_deadline = inference_deadline
        # Retire-only sweep over quiet pending conditions (ADR-0027 §4). ON by default — retirement
        # is required machinery, not an optimization: without it a condition whose watched
        # collection goes quiet is never re-judged and holds its activity BLOCKED for good. `None`
        # switches it off the way `inference_deadline=None` disables the watchdog. Per-activity
        # pacing state is (last judged at, consecutive misses, conditions seen at that check); one
        # judgement is in flight at a time, parked in a one-slot mailbox and applied on-cycle.
        self._retirement_interval = retirement_interval
        self._retirement_marks: dict[str, tuple[float, int, int]] = {}
        self._retirement_in_flight = False
        self._retirement_result: tuple[str, list[PendingConditionState], tuple[int, ...]] | None = (
            None
        )
        self._retirement_tasks: set[asyncio.Task[None]] = set()
        # DEBUG-only diagnostic state, reset at each heartbeat; see _OBSERVATION_HEARTBEAT.
        self._observed: dict[str, int] = {}
        self._heartbeat_cycles = 0
        self._heartbeat_wall = time.time()

    async def observe(self, cycle: DecisionCycle) -> TickResult:
        wm = cycle.working
        # Before the snapshot, so a tool a plan just started referencing is perceived on THIS tick
        # rather than the next one.
        attention_moved = await self._reconcile_attention(cycle)
        self._snapshot_properties(wm)
        if attention_moved:
            self._rebaseline(cycle)
        signals_before = wm.signals_appended
        async for source, signal in cycle.signal_sink.drain():
            wm.signals.append(Percept(source, signal, time.time()))
            # Bumped with the append, never with the eviction: this is what a per-waiter high-water
            # mark is measured against, so it has to survive the retention trim below.
            wm.signals_appended += 1
            log.info("observe: signal %s from %s", signal.name, source)
        self._count_observations(wm, cycle)
        # After the drain, because this reads the signals that just landed to avoid deriving a
        # duplicate of a change the adapter did announce.
        self._derive_property_changes(wm, wm.signals_appended - signals_before)
        just_resolved: list[tuple[Activity, OperationInvocation]] = []
        async for invocation_id, ack in cycle.result_sink.drain():
            # Unambiguous 1:1 match: the invoke's own result resolves its activity automatically to
            # READY — manual-agnostic, no strategy involved. The *second* kind of waiting (block on
            # a declared completion signal) is layered on top below, never fused into this resolve.
            for activity in wm.activities.values():
                # Guarded on RUNNING: a late ack for an activity a hard interrupt already routed
                # away (paused to BLOCKED/InputWait, or dropped to TERMINATED) must not resurrect it
                # to READY. The in-flight external op was allowed to finish; its result is just no
                # longer awaited. A RUNNING activity still resolves normally (the interrupt is
                # honored on the checkpoint *after* this resolve — see DefaultInterruptHandler).
                if (
                    activity.pending_operation
                    and activity.pending_operation.id == invocation_id
                    and activity.state is ActivityState.RUNNING
                ):
                    invocation = activity.pending_operation.invocation
                    op = invocation.operation_name
                    activity.last_operation = ack
                    activity.history.append(
                        CompletedOperation(invocation, ack)
                    )  # belief to ground on
                    activity.pending_operation = None
                    activity.state = ActivityState.READY
                    just_resolved.append((activity, invocation))
                    if ack.ok:
                        log.info("observe: resolved %s -> ok", op)
                        log.debug("observe: %s result\n%r", op, ack.result)
                    else:
                        # Surface *why*: a failed op terminates the activity in Reflect, and without
                        # the error the trace just says failed with no cause (e.g. a schema error).
                        log.warning("observe: resolved %s -> FAILED: %s", op, _truncate(ack.result))
                    break
        # Drain first: a provider result may already be queued even though Observe starts after its
        # deadline. It is a real resolution, so do not append a synthetic timeout behind it. Then
        # expire requests that remain pending and drain again so a true absence still resolves in
        # this same Observe pass.
        await self._resolve_inferences(cycle)
        self._expire_stalled_inferences(cycle)
        await self._resolve_inferences(cycle)
        await self._suspend_on_completion_signal(cycle, just_resolved)
        await self._resume_on_signal(cycle)
        # After the resume pass, so an activity a signal just woke counts as one that can advance
        # and stands the sweep down — and so a condition the gate is about to re-judge for free is
        # never also paid for here.
        # Ahead of the judged sweep, and unthrottled by it: a window the clock has already closed
        # costs nothing to notice, so it must neither buy a model call nor wait behind that sweep's
        # backoff (which can hold one activity off for many minutes — long enough to over-run a
        # four-minute maintenance window by four times its own length).
        await self._retire_expired_conditions(cycle)
        await self._retire_quiet_conditions(cycle)
        # Trim last: a signal that just arrived this tick must survive to be matched by the two
        # passes above before it's ever subject to eviction (bound orphan growth; newest win).
        if len(wm.signals) > _SIGNAL_RETENTION:
            del wm.signals[:-_SIGNAL_RETENTION]
        if len(wm.property_changes) > _DERIVED_RETENTION:
            del wm.property_changes[:-_DERIVED_RETENTION]
        received_message = False
        async for message in cycle.communication.receive():
            wm.messages.append(message)
            received_message = True
            log.info("observe: message from %s: %r", message.sender, _goal_from_message(message))
        if received_message and self._resume_on_input(wm):
            # A resume consumed this batch as reconsideration input for the resumed activity —
            # claim it so Situate won't also mint a ghost activity from the same follow-up (the
            # double-duty bug). A normal message (no InputWait) is turned into a goal as before.
            wm.messages_cursor = len(wm.messages)
        return TickResult()

    def _count_observations(self, wm: WorkingMemory, cycle: DecisionCycle) -> None:
        """Per-tool observation counts and the wall-clock rate, summarized every
        `_OBSERVATION_HEARTBEAT` cycles.

        Purely diagnostic — it decides nothing and is skipped entirely unless someone is listening
        at DEBUG, because the counting runs on the hot path. A tool appears here once per cycle it
        was focused for, which is exactly once per cycle its properties were read, so a gap in the
        count is a gap in perception and not merely a quiet log.
        """
        if not log.isEnabledFor(logging.DEBUG):
            return
        for tool_id in wm.focused_tools:
            self._observed[tool_id] = self._observed.get(tool_id, 0) + 1
        self._heartbeat_cycles += 1
        if self._heartbeat_cycles < _OBSERVATION_HEARTBEAT:
            return
        elapsed = time.time() - self._heartbeat_wall
        log.debug(
            "observe: %d cycles in %.1fs (%.1f/s) through cycle %d; observed %s",
            self._heartbeat_cycles,
            elapsed,
            self._heartbeat_cycles / elapsed if elapsed > 0 else 0.0,
            cycle.cycle_count,
            ", ".join(f"{tool}={n}" for tool, n in sorted(self._observed.items())),
        )
        self._observed.clear()
        self._heartbeat_cycles = 0
        self._heartbeat_wall = time.time()

    @staticmethod
    def _derive_property_changes(wm: WorkingMemory, arrived: int) -> None:
        """Recover the changes no adapter announced, by diffing each re-observed property.

        A signal is transient: pushed once, and if it never lands there is nothing left to find,
        because the property snapshot answers "what is true now" and never "what just moved". On the
        run that motivated this, six calendar events were added, four `state_changed` signals
        arrived, and the two the agent never heard about were sitting in its own snapshot the whole
        time — visible, and read by nothing. Diffing against the value last seen reconstructs
        exactly the missing half, the way AgentSpeak's belief revision derives belief-change
        events by comparing new percepts against the belief base.

        Deliberately a *safety net over* the adapter's signal, not a replacement for it. The signal
        stays the fast path (it can say things a diff cannot infer, and it arrives without waiting
        for a snapshot to move), and an adapter with no signal facility at all now gets change
        detection for free.

        Two rules do the real work:

        A property seen for the FIRST time reports nothing. It has not changed; it has only just
        become visible, and treating its whole value as an addition would open every gate watching
        it the instant attention arrived — exactly what `tool.focus()` establishing an adapter-side
        baseline already avoids on the signal path.

        A change the adapter DID signal this tick is not derived again. That is the cost control:
        an adapter refreshes the property and pushes the signal in one breath (ADR-0004 requires
        that order), so both land in the same Observe, and a derived twin would double every
        watching condition's judge calls while carrying nothing new. Dedup is per (source, path)
        rather than per source, so an adapter that reported one change and dropped another in the
        same tick still has the dropped one recovered. `arrived` is how many signals this Observe
        just drained, which is what scopes that check to the current tick.
        """
        signalled: set[tuple[str, str]] = set()
        for percept in wm.signals[len(wm.signals) - arrived :] if arrived else []:
            for change in changes_of(percept.payload):
                signalled.add((percept.source, change.path))
        for (source, name), percept in wm.properties.items():
            current = percept.payload.value
            key = (source, name)
            if key not in wm.property_baseline:
                wm.property_baseline[key] = current
                continue
            previous = wm.property_baseline[key]
            if previous == current:
                continue
            wm.property_baseline[key] = current
            # `path_matches` both ways, as everywhere else: a signal reported above or below the
            # derived path still covers it, so a coarse announcement suppresses the fine twin.
            changes = tuple(
                change
                for change in diff_values(previous, current)
                if not any(
                    src == source and path_matches(path, [change]) for src, path in signalled
                )
            )
            if not changes:
                continue
            wm.property_changes.append(
                Percept(source, PropertyChange(name=name, changes=changes), time.time())
            )
            wm.property_changes_appended += 1
            log.info(
                "observe: derived change on %s.%s from %s (no signal)",
                source,
                name,
                ", ".join(change.path or "<root>" for change in changes),
            )

    def _expire_stalled_inferences(self, cycle: DecisionCycle) -> None:
        """Give up on an inference that has been in flight too long, as though the call had failed.

        The provider seam can go quiet in ways no client setting reaches — a proxy that accepts a
        request and never answers, a retry loop inside an SDK, a process paused mid-call — and the
        activity waiting on it is `RUNNING` with no other way out: ADR-0021's stale-inference guard
        is identity-based, so it discards a *late* result but never notices an absent one. On the
        run that motivated this, one plan inference held the agent for ~14 minutes and cost it the
        whole scenario.

        Deliberately expressed as a synthetic errored `InferenceResult` rather than a second
        recovery path: every `kind` already has a considered degradation (replan for plan/subgoal/
        ground, fail-soft for select/condition/revalidate), and a deadline has no business
        inventing a different one. Nothing cancels the underlying call — an LLM call cannot be cut
        mid-generation — so if it does eventually answer, the id no longer matches the live
        `pending_inference` and the existing guard drops it, exactly as it drops any other stale
        result.

        This is *infrastructure* time, so the host wall-clock is the right one and
        `requested_at`'s `time.time()` needs no domain clock: it measures how long a request has
        been outstanding on this machine, never how far a simulated world has moved.
        """
        if self._inference_deadline is None:
            return
        now = time.time()
        for activity in cycle.working.activities.values():
            pending = activity.pending_inference
            if pending is None or activity.state is not ActivityState.RUNNING:
                continue
            waited = now - pending.requested_at
            if waited < self._inference_deadline:
                continue
            log.warning(
                "observe: %s for activity %s exceeded %.0fs (waited %.0fs) -> giving up",
                pending.kind,
                activity.id,
                self._inference_deadline,
                waited,
            )
            cycle.inference_sink.push(
                pending.id,
                InferenceResult(
                    id=pending.id,
                    error=f"inference stalled: no result after {waited:.0f}s",
                ),
            )

    @staticmethod
    async def _resolve_inferences(cycle: DecisionCycle) -> None:
        """Drain the off-cycle infer()/ground() results and apply each to the activity still RUNNING
        on it. An unambiguous 1:1 match on pending_inference.id, resolved to READY — never a Percept
        (deliberation output, not observed state — ADR-0019/0021), so it never touches the
        perception path. `kind == "plan"` lands the Plan and resets step_index; `"ground"` parks
        the resolved params on grounded_params for Reason's next pass to consume. A result carrying
        an `error` (the model call raised) never strands the activity RUNNING: the failure surfaces
        cycle-synchronized, the way a failed op does, and every kind degrades rather than dying —
        in place for select/condition/revalidate, into a replan carrying the defect for
        plan/subgoal/ground, with the runaway-replan breaker bounding the retries. Stale results
        are discarded by the same guard the external-op late-ack uses: a result whose id no longer
        matches the live pending_inference (an interrupt handler re-routed or re-inferred the
        activity), or whose activity is no longer RUNNING, is dropped — the background call ran to
        completion (an LLM call can't be cut mid-generation) but its result is no longer wanted."""
        wm = cycle.working
        async for inf_id, res in cycle.inference_sink.drain():
            outcome: LLMOutcome = (
                "unresolvable"
                if res.unresolvable is not None
                else "error"
                if res.error is not None
                else "success"
            )
            log_llm_outcome(inf_id, outcome)
            for activity in wm.activities.values():
                if (
                    activity.pending_inference is not None
                    and activity.pending_inference.id == inf_id
                    and activity.state is ActivityState.RUNNING
                ):
                    kind = activity.pending_inference.kind
                    out = activity.pending_inference.out  # set only for kind=="select"
                    baseline = (
                        activity.pending_inference.baseline
                    )  # set for plan/subgoal (ADR-0024)
                    activity.pending_inference = None
                    if res.unresolvable is not None:
                        # An escalation asked to resolve a reference reported that it names data
                        # this run never produced, instead of fabricating a value for it — grounding
                        # a param, or a $decide filter predicate. The defect is in the PLAN — it
                        # assumed an earlier step would yield something it didn't — so the repair is
                        # a replan, not termination: the replanning prompt carries the executed
                        # history, so the next plan SEES the empty result that defeated this one and
                        # can narrow differently (or just report the gap to the user). Terminating
                        # here would be safe but useless — it leaves the user with no answer at all,
                        # and the activity had nothing wrong with it beyond one bad assumption.
                        # Routed through the single funnel, so the discarded plan is parked as
                        # `superseded` for the re-inference exactly like any other invalidation —
                        # but tagged with the defect, because unlike a reconsideration this plan was
                        # not merely overtaken by events: the value it reaches for is not there, and
                        # a replacement that reaches for it the same way fails the same way.
                        # Only for `ground`: a $decide filter's gap is about its predicate, and an
                        # empty `in` there is already an answer rather than a defect, so there is
                        # nothing to re-attribute.
                        defect = (
                            _with_empty_binding_origin(activity, res.unresolvable)
                            if kind == "ground"
                            else res.unresolvable
                        )
                        activity.reset_for_replan(defect=defect)
                        activity.state = ActivityState.READY
                        log.warning(
                            "observe: %s for activity %s resolved nothing (%s) -> replan",
                            kind,
                            activity.id,
                            defect,
                        )
                    elif res.error is not None and kind == "select":
                        # A $decide filter is a transform, not control flow: a transient model or
                        # parse failure degrades to an empty shortlist (the pipeline does nothing
                        # this run) rather than terminating the activity — keeps the data-op alive.
                        assert out is not None
                        activity.bindings[out] = []
                        activity.state = ActivityState.READY
                        log.warning(
                            "observe: $decide filter for %s failed (%s) -> empty binding %r",
                            activity.id,
                            res.error,
                            out,
                        )
                    elif res.error is not None and kind == "condition":
                        # A failed condition evaluation degrades to "nothing fired" — the same
                        # fail-soft as select/revalidate, and the same reasoning: the activity was
                        # already waiting, so keeping it waiting changes nothing, while the opposite
                        # default would invent follow-up work nobody asked for off a flaky call.
                        #
                        # But "keeping it waiting" is only true if the change that opened the gate
                        # is still judgeable, and the fire-time mark advance means it is not: the
                        # gate cannot re-open for a change already past the mark, the real verdict
                        # arriving late is dropped by the stale-id guard, and a collection that goes
                        # quiet afterwards never makes the condition eligible again — so a failure
                        # here silently loses the wake rather than deferring it. That is reachable
                        # on a HEALTHY call, since the client retries a stalled request twice at its
                        # own stall timeout each, which can outlast this deadline. Give the change
                        # back, once per condition (see `retried_after_failure`).
                        for state in activity.condition_batch:
                            if state.retried_after_failure:
                                continue
                            state.retried_after_failure = True
                            state.evaluated_through = state.fired_from_signals
                            state.derived_through = state.fired_from_derived
                        activity.condition_verdict = ConditionVerdict()
                        activity.state = ActivityState.READY
                        log.warning(
                            "observe: condition evaluation for %s failed (%s) -> nothing fired",
                            activity.id,
                            res.error,
                        )
                    elif res.error is not None and kind == "revalidate":
                        # A failed revalidation must not force a replan (would thrash): degrade to
                        # "still valid" so Reason proceeds — mirrors select's fail-soft (ADR-0024).
                        # Advance the baseline to the re-check's fire-time world (like a valid
                        # verdict) so a static world doesn't re-fire on the next write.
                        activity.reconsider_verdict = True
                        activity.reconsider_baseline = baseline
                        activity.state = ActivityState.READY
                        log.warning(
                            "observe: plan revalidation for %s failed (%s) -> assume valid",
                            activity.id,
                            res.error,
                        )
                    elif res.error is not None and kind in _REPLANNABLE_INFERENCE:
                        # A plan/sub-plan/grounding call that raised is a *deliberation* failure,
                        # not evidence that the goal cannot be reached: nothing was attempted, the
                        # world is untouched, and a fresh attempt is free to differ. So it degrades
                        # the same way an unresolvable grounding does — replan carrying the defect
                        # — rather than terminating. Terminating here was the destructive default:
                        # for `subgoal` it destroyed an activity whose parent frames were intact,
                        # and for `plan` it threw away a whole activity over one malformed model
                        # response. The retry budget is NOT open-ended and needs no counter of its
                        # own: `_replanning_would_loop` already refuses a further attempt once two
                        # plans in a row were abandoned for the *same* defect (or after
                        # max_replan_attempts distinct ones) and blocks on an InputWait instead —
                        # which is exactly the right disposition for a failure that keeps
                        # repeating, including a permanent one like "no LLM is configured". That is
                        # why `_inference_defect` normalizes the error to its cause rather than
                        # carrying the raw message: two parse failures quote different model output
                        # and would never compare equal, so the precise check would never fire and
                        # a hopeless call would be paid for five times instead of twice.
                        activity.grounded_params = None
                        activity.reset_for_replan(defect=_inference_defect(kind, res.error))
                        activity.state = ActivityState.READY
                        log.warning(
                            "observe: %s for activity %s failed (%s) -> replan",
                            kind,
                            activity.id,
                            res.error,
                        )
                    elif res.error is not None:
                        # Residual net for an inference kind with no degradation of its own (every
                        # kind the runtime ships routes above). Terminating is right when there is
                        # no defined way to continue — but it must not be *silent*, which is what
                        # this branch used to be: no episode, so the failure never reached memory
                        # and Reflect's "TERMINATED was already recorded" was untrue for this path,
                        # and no word to the user, so an activity born from an instruction ended
                        # without an answer. Both are repaired here, and awaited rather than
                        # dispatched: this is the activity's last cycle, so there is no later pass
                        # to finish the work on.
                        activity.grounded_params = None
                        activity.superseded = None  # no replacement is coming; don't keep it parked
                        activity.state = ActivityState.TERMINATED
                        log.error(
                            "observe: %s for activity %s failed (%s) -> terminated",
                            kind,
                            activity.id,
                            res.error,
                        )
                        await cycle.episodic.learn(
                            activity,
                            f"failed: {activity.goal} ({kind} inference failed: {res.error})",
                            succeeded=False,
                        )
                        await _report_to_user(
                            cycle,
                            f"I could not carry on with {activity.goal!r}: the {kind} step of my "
                            f"own reasoning failed ({res.error}). Nothing was changed.",
                        )
                    elif kind == "plan":
                        inferred: Plan = res.value  # type: ignore[assignment]  # kind=="plan" => Plan
                        activity.plan = inferred
                        activity.step_index = 0
                        activity.history_mark = len(activity.history)
                        # The superseded bundle has now been consumed by the inference that
                        # produced this plan (ADR-0024): drop it so it can never reach a later,
                        # unrelated inference.
                        activity.superseded = None
                        # Anchor the context-adaptation gate to the world this plan was inferred
                        # against (ADR-0024), so a change that landed *during* inference is caught
                        # at the first checkpoint rather than folded into a later baseline.
                        activity.reconsider_baseline = baseline
                        activity.state = ActivityState.READY
                        log.info("observe: resolved inferred plan for activity %s", activity.id)
                        # The plan body itself only at DEBUG: it belongs in the full --log-file
                        # trace (next to the prompt that produced it), not in the terminal's
                        # one-line-per-event view. Same rendering as the revalidation prompt, so a
                        # re-inferred plan can be diffed against the one it replaced by eye.
                        log.debug(
                            "observe: plan for activity %s\n%s",
                            activity.id,
                            render_plan(inferred),
                        )
                    elif kind in ("subgoal", "then"):
                        # A mid-plan sub-goal's synthesized sub-plan: push the parent frame (its
                        # plan + the sub-goal's step_index) and enter the sub-plan, so Reason
                        # advances it and pops back to the parent when it exhausts (ADR-0022). It
                        # lands like a top-level plan, only onto a stacked frame not the activity.
                        #
                        # A fired condition's `then` ("then") lands the same way but pushes NO
                        # frame: it only ever starts once the body is idle, so the plan it replaces
                        # has nothing left to return to, and a watch fires as many times as the
                        # world moves — a stack that grew once per firing would walk a healthy
                        # monitor into the depth cap for doing its job.
                        sub_plan: Plan = res.value  # type: ignore[assignment]  # both kinds => Plan
                        if kind == "subgoal":
                            # Validate before entering, so a sub-goal that could never end is never
                            # half-installed and the superseded bundle the replan reads is the
                            # PARENT plan (the one still holding the goal) rather than a frame that
                            # was pushed only to be torn down again.
                            window = _unclosable_window(activity, sub_plan, cycle.working)
                            if window is not None:
                                log.warning(
                                    "observe: plan defect for activity %s — %s", activity.id, window
                                )
                                activity.reset_for_replan(defect=window)
                                activity.state = ActivityState.READY
                                break  # this activity claimed the result; it just refused it
                            frame = (activity.plan, activity.step_index, activity.history_mark)
                            activity.parent_frames.append(frame)  # type: ignore[arg-type]  # plan set mid-plan
                        activity.plan = sub_plan
                        activity.step_index = 0
                        # The sub-plan collects only what it runs itself, not what the parent left
                        # in history before it was entered.
                        activity.history_mark = len(activity.history)
                        # Re-anchor the gate to the sub-plan's own infer-time world (ADR-0024).
                        activity.reconsider_baseline = baseline
                        activity.state = ActivityState.READY
                        log.info(
                            "observe: entered %s for activity %s",
                            "sub-plan" if kind == "subgoal" else "a fired condition's `then`",
                            activity.id,
                        )
                        log.debug(
                            "observe: sub-plan for activity %s (nested under %d frame(s))\n%s",
                            activity.id,
                            len(activity.parent_frames),
                            render_plan(sub_plan),
                        )
                    elif kind == "select":
                        # A $decide data-op filter (ADR-0023): the surviving subset lands into the
                        # named binding, exactly like a mechanical filter would have written it. Not
                        # a Percept (deliberation output, not observed state) — same as plan/ground.
                        assert out is not None  # a "select" pending always carries its target name
                        activity.bindings[out] = (
                            res.value
                        )  # value is Any-typed dict; no cast needed
                        activity.state = ActivityState.READY
                        log.info(
                            "observe: resolved $decide filter -> binding %r for activity %s",
                            out,
                            activity.id,
                        )
                    elif kind == "condition":
                        # The batched pending-condition verdict (ADR-0022): park it for Reason's
                        # next pass, which applies it against the eligible list it re-derives.
                        # Deliberation output, like plan/ground/select/revalidate — never a Percept.
                        verdict = res.value
                        activity.condition_verdict = (
                            verdict if isinstance(verdict, ConditionVerdict) else ConditionVerdict()
                        )
                        # The seam answered, so the retry this condition is owed on a failure is
                        # restored — the bound is one retry per failure, not one per activity life.
                        for state in activity.condition_batch:
                            state.retried_after_failure = False
                        activity.state = ActivityState.READY
                        log.info(
                            "observe: condition verdict fired=%s retired=%s for activity %s",
                            activity.condition_verdict.fired,
                            activity.condition_verdict.retired,
                            activity.id,
                        )
                    elif kind == "revalidate":
                        # The context-adaptation validity verdict (ADR-0024): park the bool for
                        # Reason's next pass (proceed / reset_for_replan) — deliberation output,
                        # like plan/ground/select. Advance the baseline to the world this re-check
                        # was fired against (carried on pending_inference.baseline), so a "valid"
                        # verdict re-baselines to re-check-fire, not verdict, time: a change that
                        # arrived mid-flight stays outside the baseline and earns its own
                        # reconsideration. A "False" verdict discards it in reset_for_replan.
                        activity.reconsider_verdict = bool(res.value)
                        activity.reconsider_baseline = baseline
                        activity.state = ActivityState.READY
                        log.info(
                            "observe: plan-validity verdict %s for activity %s",
                            activity.reconsider_verdict,
                            activity.id,
                        )
                    else:
                        activity.grounded_params = res.value  # type: ignore[assignment]  # => dict
                        activity.state = ActivityState.READY
                        log.info("observe: resolved grounded params for activity %s", activity.id)
                    break
            else:
                # No live activity claimed this result: it was invalidated (an interrupt re-routed
                # the activity) or superseded (a re-inference gave a new id). The background model
                # call ran to completion and was already metered, so its cost is real but wasted —
                # tell the meter to move it to the wasted bucket (a no-op when uninstrumented).
                log_llm_discarded(inf_id)

    @staticmethod
    def _resume_on_input(wm: WorkingMemory) -> bool:
        """A user Message satisfies an InputWait: any activity a hard interrupt paused (a user stop)
        returns to READY, so the decision cycle can reconsider it with the new instruction now in
        working memory. The mirror of _resume_on_signal, but the awaited stimulus is inbound user
        input rather than a tool signal — mechanical, no judgment. The plan is *cleared* on resume
        (not kept) so Reason re-infers with the follow-up message and the executed history visible —
        a bare resume would keep advancing the stale plan and never see the instruction. Returns
        whether any activity was resumed, so the caller can claim the message batch as
        reconsideration input rather than letting Situate mint a ghost activity from it."""
        resumed = False
        for activity in wm.activities.values():
            if isinstance(activity.blocked_on, InputWait):
                activity.blocked_on = None
                # Route the plan-drop through the single funnel (clears plan/step_index, the whole
                # intention stack, and any parked/in-flight deliberation) so Reason re-infers with
                # the follow-up instruction and executed history visible — a bare resume would keep
                # advancing the stale plan and never see it. blocked_on and the unconditional READY
                # stay here: resume-specific, not plan invalidation (cf. the _resume_ action).
                activity.reset_for_replan()
                # The instruction just received is the new direction, so the attempts that led here
                # no longer bear on the next plan. Without this the breaker re-trips on the resumed
                # activity's first Reason pass and the halt becomes permanent instead of a question
                # — including for the reset_for_replan immediately above, which appends to the very
                # trail being read. Cleared after it, so the resume's own entry goes too.
                activity.clear_replan_trail()
                activity.state = ActivityState.READY
                resumed = True
        return resumed

    async def _suspend_on_completion_signal(
        self, cycle: DecisionCycle, just_resolved: list[tuple[Activity, OperationInvocation]]
    ) -> None:
        """For each activity whose op just resolved: if the op's manual declares a completion signal
        that hasn't already arrived, suspend the activity until it does. Layered on the automatic
        RUNNING->READY resolve above (a failed op still terminates in Reflect; only a successful,
        signal-declaring op suspends). If the signal already arrived (it beat the ack), stay READY
        without blocking — the two waits compose, they don't deadlock. The signal itself is never
        consumed here: it's left in `wm.signals` for `_resume_on_signal` (or any other activity
        blocked on the same wait, or a strategy reading `wm.signals` directly) to still see it."""
        wm = cycle.working
        suspend = cycle.actions.internal(SuspendAction.name)
        for activity, invocation in just_resolved:
            last = activity.last_operation
            if last is None or not last.ok:  # a failure isn't a completion to wait past
                continue
            completion = self._completion_signal(wm, invocation)
            if completion is None:
                continue
            wait = SignalWait(signal_name=completion, source=invocation.tool_id)
            if self._match_signal(wm, wait) is not None:  # early signal: already satisfied
                log.info("observe: completion signal %s already present", completion)
                continue
            await suspend.execute(cycle, activity_id=activity.id, wait=wait)

    async def _resume_on_signal(self, cycle: DecisionCycle) -> None:
        """For each BLOCKED activity, if an observed signal satisfies its wait, resume it. The
        matched signal is left in `wm.signals` rather than evicted — it's a shared, append-only log
        any other blocked activity (waiting on the identical name+source) or strategy reading it
        directly may still need; only the fixed retention cap (see observe()) ever evicts it."""
        wm = cycle.working
        resume = cycle.actions.internal(ResumeAction.name)
        for activity in wm.activities.values():
            # An InputWait waits on a user Message and is resumed in _resume_on_input, not here.
            if isinstance(activity.blocked_on, ConditionWait):
                # A gate opening only makes a condition ELIGIBLE — whether it actually holds is a
                # Reason judgment, because matching prose against an email body is irreducibly
                # semantic. Resuming is how the activity gets selected so Reason can make it.
                # Gated on an UNJUDGED match (each condition's own mark): resuming on a signal
                # every condition has already dismissed would re-block immediately and spin.
                if _eligible_conditions(activity, wm):
                    log.info("observe: activity %s has an eligible pending condition", activity.id)
                    await resume.execute(cycle, activity_id=activity.id)
                continue
            if not isinstance(activity.blocked_on, SignalWait):
                continue
            if self._match_signal(wm, activity.blocked_on) is not None:
                await resume.execute(cycle, activity_id=activity.id)

    async def _retire_quiet_conditions(self, cycle: DecisionCycle) -> None:
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
        and `_retire_expired_conditions` has already resolved it against the workspace's clock
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
        if self._retirement_interval is None:
            return
        # Apply first, and only apply: a tick that lands a verdict does not also fire the next one.
        if self._retirement_result is not None:
            await self._apply_retirement(cycle)
            return
        if self._retirement_in_flight:
            return
        activity = self._retirement_candidate(cycle.working)
        if activity is None:
            return
        judged = list(activity.pending_conditions)
        _checked, misses, _seen = self._retirement_marks.get(activity.id, (0.0, 0, 0))
        # Marked at fire time, not on resolve, so a slow call cannot be re-fired underneath itself.
        self._retirement_marks[activity.id] = (time.time(), misses, len(judged))
        # Agent-level, like both other judges and unlike a `_ground_`/`_select_` call: retirement
        # asks whether waiting is over, and the evidence for that ("the slot has taken place") is
        # routinely on a tool the waiting activity never touches.
        observed = PerceptSnapshot(
            list(cycle.working.properties.values()), list(cycle.working.signals)
        )
        self._retirement_in_flight = True
        log.info(
            "observe: judging retirement of %d quiet condition(s) on %s",
            len(judged),
            activity.id,
        )
        _spawn_tracked(
            self._retirement_tasks, self._judge_retirement(cycle, activity, judged, observed)
        )

    def _retirement_candidate(self, wm: WorkingMemory) -> Activity | None:
        """The one activity to sweep this tick, or None.

        One per tick, longest-unchecked first. The sweep's whole cost argument is that it comes out
        of slack, and N calls fired together on a single idle tick is not slack — nor would it stay
        one call per activity, since an agent that accumulates watches is exactly the one this runs
        for most often.
        """
        if len(self._retirement_marks) > len(wm.activities):
            # Pacing state for an activity working memory no longer holds. Bounded rather than
            # correctness-critical, but this dict is keyed by a per-run id and the runtime is meant
            # to stay up indefinitely.
            self._retirement_marks = {
                key: mark for key, mark in self._retirement_marks.items() if key in wm.activities
            }
        if any(a.state is ActivityState.READY for a in wm.activities.values()):
            return None  # something can advance; the sweep never competes with it
        assert self._retirement_interval is not None  # guarded by the caller
        now = time.time()
        due: list[tuple[float, Activity]] = []
        for activity in wm.activities.values():
            if activity.state is ActivityState.TERMINATED or not activity.pending_conditions:
                continue
            checked, misses, seen = self._retirement_marks.get(activity.id, (0.0, 0, 0))
            if misses and len(activity.pending_conditions) > seen:
                # A condition declared since the last check deserves a prompt look — the backoff's
                # premise ("the last look bought nothing") says nothing about one never looked at.
                # Persisted, not just local: the stored count is what the next mark is built from,
                # so a local reset makes the look prompt exactly once and then restores the whole
                # accumulated backoff.
                misses = 0
                self._retirement_marks[activity.id] = (checked, misses, seen)
            if checked and now - checked < self._retirement_interval * 2 ** min(
                misses, _RETIREMENT_BACKOFF
            ):
                continue
            due.append((checked, activity))
        if not due:
            return None
        return min(due, key=lambda entry: entry[0])[1]

    async def _judge_retirement(
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
            self._retirement_in_flight = False
        self._retirement_result = (activity.id, judged, retired)

    async def _apply_retirement(self, cycle: DecisionCycle) -> None:
        """Consume a parked retirement verdict — the judged sweep's half of `_drop_retired`, plus
        the backoff bookkeeping a verdict that retired nothing earns."""
        parked, self._retirement_result = self._retirement_result, None
        assert parked is not None  # guarded by the caller
        activity_id, judged, indices = parked
        activity = cycle.working.activities.get(activity_id)
        if activity is None or activity.state is ActivityState.TERMINATED:
            return
        if not indices:
            # Back off: the same answer is what the next call would most likely buy too.
            checked, misses, seen = self._retirement_marks.get(activity_id, (time.time(), 0, 0))
            self._retirement_marks[activity_id] = (checked, misses + 1, seen)
            return
        await self._drop_retired(cycle, activity, [judged[i] for i in indices if i < len(judged)])

    async def _retire_expired_conditions(self, cycle: DecisionCycle) -> None:
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
        checked, _misses, _seen = self._retirement_marks.get(activity.id, (0.0, 0, 0))
        self._retirement_marks[activity.id] = (checked, 0, len(activity.pending_conditions))
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

    @staticmethod
    def _completion_signal(wm: WorkingMemory, invocation: OperationInvocation) -> str | None:
        """The completion signal the invoked op declares in its manual, or None (unknown tool,
        unknown op, or a synchronous op)."""
        try:
            tool = wm.registry.get(invocation.tool_id)
        except KeyError:
            return None  # tool left since the invoke — nothing to wait on
        op = tool.manual.operation(invocation.operation_name)
        return op.completion_signal if op is not None else None

    @staticmethod
    def _match_signal(wm: WorkingMemory, wait: SignalWait, *, since: int = 0) -> Percept | None:
        """The first stored signal satisfying `wait` (name equality, source when scoped, and path
        when scoped), or None. Mechanical — no LLM judgment — since every field is declared.

        `since` is a high-water mark over `wm.signals_appended`: only signals appended *after*
        it are considered. The default of 0 considers everything retained, which a completion-signal
        wait wants (it must see a signal that beat its own ack). A pending-condition waiter passes
        its own mark so it never re-judges a signal it already judged — per-waiter, never shared.
        """
        # signals[i]'s sequence number; the cap front-evicts, so this is not the list index.
        first_seq = wm.signals_appended - len(wm.signals)
        for offset, percept in enumerate(wm.signals):
            if first_seq + offset < since:
                continue
            if percept.payload.name == wait.signal_name and (
                wait.source is None or percept.source == wait.source
            ):
                if watch_matches(wait.path, wait.kind, changes_of(percept.payload)):
                    return percept
        return None

    @staticmethod
    def _match_derived(wm: WorkingMemory, wait: SignalWait, *, since: int = 0) -> Percept | None:
        """The first DERIVED change satisfying `wait`, or None — `_match_signal` over the other log,
        minus one field: a derived change has no signal name, because no tool named it, so source,
        path and direction gate it and `signal_name` cannot.

        That makes this match strictly WIDER than the declared watch, and the widening is
        deliberate. This path exists because an announcement went missing, and a safety net that can
        itself miss is not one — insisting on a name nothing produced would satisfy no watch at all.
        It is the same trade the rest of this machinery already makes: `path_matches` matches
        bidirectionally and `_kind_matches_one` degrades open, both because a redundant evaluation
        costs one judge call answered "no" while a missed wake costs the run. The cost control the
        gate exists for survives intact — source, path and `kind` still gate it, so the agent's own
        write on the watched path is told apart from the world's exactly as it is on the signal
        path.
        """
        # As in _match_signal: the cap front-evicts, so this is a sequence number, not a list index.
        first_seq = wm.property_changes_appended - len(wm.property_changes)
        for offset, percept in enumerate(wm.property_changes):
            if first_seq + offset < since:
                continue
            if wait.source is not None and percept.source != wait.source:
                continue
            if watch_matches(wait.path, wait.kind, list(percept.payload.changes)):
                return percept
        return None

    async def _reconcile_attention(self, cycle: DecisionCycle) -> bool:
        """Diff the policy's target set against what is actually focused, and close the gap.

        Returns whether the attended set actually moved, so the caller can re-anchor the
        reconsideration gate against the new observation window.

        A mechanical internal effect, the same class as the auto-focus this replaces — not an
        external-action dispatch — so one-external-action-per-cycle (ADR-0009) is untouched.
        Ordering is safe in the other direction too: `tool.focus()` establishes the adapter's
        change baseline, so the `observe()` immediately below compares equal and a newly attended
        tool emits no spurious `state_changed` on its first tick.
        """
        wm = cycle.working
        joined = {tool.id for tool in wm.registry.all_tools()}
        target = self._focus.attend(wm)
        # Recorded from the policy's own target, *before* suppression is subtracted: "is this agent
        # narrowing?" is a question about the policy. Deriving it from `focused_tools` instead would
        # conflate a narrowing policy with an explicit `_unfocus_` or a tool that failed to resolve,
        # and would switch on the per-activity prompt view for an agent that chose the broad policy.
        wm.attention_narrowed = not joined <= target
        # An explicit `_unfocus_` outranks the policy. The policy is a derived floor recomputed from
        # scratch each tick; the suppression is a decision the agent took and nothing here undoes.
        attended = target - wm.suppressed_tools
        before = set(wm.focused_tools)
        for tool_id in wm.focused_tools.keys() - attended:
            await release(cycle, tool_id)
        for tool_id in attended - wm.focused_tools.keys():
            try:
                tool = wm.registry.get(tool_id)
            except KeyError:  # named by a plan but not (or no longer) joined
                continue
            await attend(cycle, tool)
        if before != wm.focused_tools.keys():  # only on a real transition — this is THE diagnostic
            log.info("observe: attending %s", ", ".join(sorted(wm.focused_tools)) or "(nothing)")
            return True
        return False

    @staticmethod
    def _rebaseline(cycle: DecisionCycle) -> None:
        """Re-anchor the reconsideration gate after the *observation window* moved, rather than the
        world (ADR-0024).

        The gate hashes the property store, so attending or releasing a tool moves the signature
        with nothing in the environment having changed — and the first checkpoint after a plan lands
        (narrowing) or a workspace is joined (broadening) would spend a revalidation call on the
        agent's own attention. Re-anchoring reads that as "the window moved", which is what it is.

        The cost is a one-tick blind spot: a genuine change landing on the same tick as an attention
        transition is absorbed into the new baseline instead of firing. That is bounded — a
        transition happens when a plan lands, on join/leave, and on an explicit focus/unfocus, not
        continuously — and the alternative is a false positive on every one of those.
        """
        signature = cycle.change_gate.signature(cycle.working)
        for activity in cycle.working.activities.values():
            if activity.reconsider_baseline is not None:
                activity.reconsider_baseline = signature

    @staticmethod
    def _snapshot_properties(wm: WorkingMemory) -> None:
        """Represent observable properties as a replace-by-(source, name) snapshot: one percept per
        property, last value wins. A property is persistent, re-observed state, so re-observing the
        same (source, name) overwrites its entry in the keyed store rather than accumulating — the
        store *is* the snapshot. Signals are the opposite (transient, fire-and-forget) and keep
        append semantics in their own list, handled in observe()."""
        for tool in wm.focused_tools.values():
            for prop in tool.observe():
                wm.properties[(tool.id, prop.name)] = Percept(tool.id, prop, time.time())


class DefaultReflectStrategy:
    """The runtime's built-in default — purely mechanical, no LLM.

    Judges each activity completed or failed by two deterministic rules, and on a terminal outcome
    records the experience: the state transition is synchronous (so Situate, which runs later this
    cycle and selects only READY activities, never re-selects a just-terminated one), while the
    episodic/procedural writes are dispatched as background tasks and never block the cycle (several
    activities may terminate in the same cycle). Strong refs to the in-flight tasks are held so they
    aren't GC'd mid-write — the same pattern as InvokeAction.

    The two rules are deliberately asymmetric. **Failure** fires on any resolved-but-not-ok
    ``last_operation``, independent of the plan: a failed operation is definite negative evidence,
    so the activity terminates even mid-plan. **Completion** requires positive evidence that all
    planned work is done — a plan present and fully consumed (``step_index >= len(plan.steps)``) —
    so a plan-less activity is never auto-completed here (what a plan-following Reason, and any
    application driving activities without a plan, relies on). Both outcomes record an episode;
    neither auto-caches the plan to procedural memory — replaying a stored plan verbatim is unsound,
    so plan storage is disabled until reusable procedures are distilled from episodes (Reason still
    consults ``procedural.retrieve``, which simply finds nothing until then)."""

    def __init__(self) -> None:
        # Hold strong refs to in-flight background stores so they aren't GC'd mid-write.
        self._tasks: set[asyncio.Task[None]] = set()

    async def reflect(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        # Lift any conditions the live frames declare onto the activity, before the state check so
        # it happens on every cycle regardless of state. Idempotent (dedup by condition value), so
        # this is also what makes a condition live from plan ENTRY rather than only once the body is
        # exhausted — the early-reply case (a reply that beats the confirmation step) needs the
        # condition already watching while the body is still running.
        _lift_pending_conditions(activity, wm)
        # Only READY activities are judged: RUNNING has an operation still in flight (nothing to
        # judge yet), BLOCKED is waiting on a signal or on the user, and TERMINATED already recorded
        # its own episode — every path that sets TERMINATED writes one before handing back (this
        # strategy below, and Observe's residual inference-failure branch), which is what lets
        # reflect() skip them and stay idempotent across the cycles it runs on every activity.
        if activity.state is not ActivityState.READY:
            return result
        if self.failed(activity):
            activity.state = ActivityState.TERMINATED  # synchronous — Situate sees it this cycle
            log.info("reflect: activity %s failed; storing episode", activity.id)
            self._dispatch(self._record_failure(cycle, activity))
        elif (
            activity.plan is not None
            and activity.step_index >= len(activity.plan.steps)
            and not activity.parent_frames
        ):
            # Complete only when the *top-level* plan is exhausted: a just-exhausted sub-plan still
            # has parent frames to pop (Reason does that next cycle), so it isn't done (ADR-0022).
            eligible = _eligible_conditions(activity, wm) if activity.pending_conditions else []
            if activity.condition_fired or activity.condition_verdict is not None:
                # Work Reason owes: a fired condition whose `then` is still unrun, or a resolved
                # verdict Observe parked that nothing has applied yet. The queue outlives the
                # condition that produced it — a fired condition is usually retired by the same
                # verdict, so `pending_conditions` can be empty while committed work is still
                # queued. Leave it READY for Reason to drain; terminating here would write a success
                # episode for a goal that has an unrun `then`, and BLOCKING here would strand the
                # verdict, because Situate only ever selects a READY activity — so Reason would
                # never apply it and a judgement already paid for would be silently discarded, with
                # the marks already advanced past the signal that could re-open the gate. Reason's
                # own no-fire path re-blocks (and a failed evaluation parks an empty verdict that
                # takes exactly that path), so deferring costs nothing.
                log.info(
                    "reflect: activity %s body exhausted; leaving ready to pursue %d fired "
                    "condition(s) and %d unapplied verdict(s)",
                    activity.id,
                    len(activity.condition_fired),
                    0 if activity.condition_verdict is None else 1,
                )
            elif eligible:
                # A gate has opened on a signal no condition has judged yet, and Observe resumed
                # this activity precisely so Reason can judge it. Re-blocking here would undo that
                # resume in the same cycle, before Situate could ever select it — and since the
                # per-condition mark advances only when Reason *fires* the batched judgement, the
                # same unjudged signal would reopen the gate next cycle, forever. That is the
                # Observe-resume/Reflect-reblock livelock seen on 2026-08-21: ~1400 cycles of
                # resume->reblock after a single Emails `state_changed`, spending no model calls and
                # making no progress, with the pending condition never once evaluated. Leave it
                # READY; Reason advances the marks and re-blocks (or retires) from its own verdict.
                log.info(
                    "reflect: activity %s body exhausted; leaving ready to judge %d condition(s)",
                    activity.id,
                    len(eligible),
                )
            elif activity.pending_conditions:
                # The body is finished but the GOAL is not: unsatisfied declared conditions mean
                # this plan said what would make it relevant again. Block rather than terminate, and
                # record no episode yet — the activity has not ended, so an episode written here
                # would be a claim about an outcome that hasn't happened (ADR-0022/ADR-0026).
                activity.state = ActivityState.BLOCKED
                activity.blocked_on = ConditionWait(watches=_condition_watches(activity))
                log.info(
                    "reflect: activity %s body exhausted; blocking on %d pending condition(s)",
                    activity.id,
                    len(activity.pending_conditions),
                )
            else:
                activity.state = ActivityState.TERMINATED
                log.info("reflect: activity %s completed; storing episode", activity.id)
                self._dispatch(self._record_success(cycle, activity))
        # Reflect never fills in the decision fields (activity/step/invocation) — it threads
        # `result` through untouched.
        return result

    def failed(self, activity: Activity) -> bool:
        """The default rule: a resolved-but-not-ok last_operation is definite negative evidence,
        independent of the plan (see the class docstring's "asymmetric" rules)."""
        return activity.last_operation is not None and not activity.last_operation.ok

    def _dispatch(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _record_success(self, cycle: DecisionCycle, activity: Activity) -> None:
        # Records the episode only. The completed plan is deliberately NOT stored to procedural
        # memory: auto-caching a plan and replaying it verbatim is unsound (a corrected or
        # observation-coupled plan is not reusable). Distilling reusable procedures from episodes is
        # future work; cycle.procedural.store stays available for that deliberate step.
        await cycle.episodic.learn(activity, _summarize(activity, succeeded=True), succeeded=True)

    async def _record_failure(self, cycle: DecisionCycle, activity: Activity) -> None:
        await cycle.episodic.learn(activity, _summarize(activity, succeeded=False), succeeded=False)


def _summarize(activity: Activity, *, succeeded: bool) -> str:
    """A deterministic, no-LLM episode summary. A model-backed ReflectStrategy would substitute a
    richer natural-language summary here; the mechanical default just states outcome and goal."""
    outcome = "completed" if succeeded else "failed"
    return f"{outcome}: {activity.goal}"


def _truncate(value: Any, limit: int = 300) -> str:
    """One-line, length-capped rendering of an operation result for a log line (a tool error can be
    a long multi-line traceback)."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit] + "…"


def _goal_from_message(message: Message) -> str:
    """The default's deterministic goal derivation from a message — no interpretation, no model
    call: the message's own text if it carries a conventional ``text`` field, else the whole content
    rendered. A model-backed Situate would derive a richer goal instead."""
    text = message.content.get("text")
    return text if isinstance(text, str) else str(message.content)


class RoundRobinActivitySelection:
    """Deterministic anti-starvation default: rotate through the ready set by carrying a cursor
    (last-selected activity id) across cycles. Cold start (or when the last pick is no longer ready)
    falls back to ready[0] — the oldest — so behavior matches a static priority-by-age default until
    an activity lingers READY, at which point selection rotates instead of pinning it. Genuine
    cross-cycle state (unlike a stateless default), feasible because the strategy instance persists
    for the agent's lifetime — cf. DefaultReflectStrategy's task set."""

    def __init__(self) -> None:
        self._last_id: str | None = None

    async def select(
        self, ready: list[Activity], wm: WorkingMemory, cycle: DecisionCycle
    ) -> Activity | None:
        if not ready:
            return None
        ids = [a.id for a in ready]
        # Rotate off the last pick; wrap via modulo. Single-ready -> (0+1)%1 == 0 re-picks it (no
        # starvation possible). Last pick gone from the ready set -> restart at the oldest.
        nxt = (ids.index(self._last_id) + 1) % len(ids) if self._last_id in ids else 0
        chosen = ready[nxt]
        self._last_id = chosen.id
        return chosen


class DefaultSituateStrategy:
    """The runtime's built-in default — mechanical, no LLM. Always runs: it adjusts working memory
    for the joined workspaces every cycle (even for an already-selected activity), then selects only
    if result.activity is still None. Creates an activity from any unhandled message (deduped by
    derived goal) via the internal _create_activity_ action, and adjusts wm via the internal
    working-memory actions — loads joined tools' manuals (_load_), unloads manuals no longer backed
    by a joined tool (_unload_), and filters observable-property percepts to the *attended* tools
    (_filter_). _filter_ only prunes properties (a re-observed snapshot, safe to drop); signals are
    retained regardless of source — they're fire-and-forget, and their retention and eviction is
    consumption-driven, owned by the blocked-state machinery, not this prune. Deciding what to
    attend to is *not* done here: it is a subscription to the environment, reconciled in Observe
    against the live intentions, and a plan can still override either way with an
    explicit `focus`/`unfocus` step dispatched as the cycle's one external action (at Act). Which
    ready activity runs is delegated to a pluggable ActivitySelectionStrategy (default
    RoundRobinActivitySelection — fair rotation over the ready set), so a richer scheduler can be
    swapped in without re-authoring the mechanical activity-creation and wm-adjustment above."""

    def __init__(self, selection: ActivitySelectionStrategy | None = None) -> None:
        self._activity_selection = selection or RoundRobinActivitySelection()

    async def situate(
        self,
        activities: list[Activity],
        wm: WorkingMemory,
        cycle: DecisionCycle,
        result: TickResult,
    ) -> TickResult:
        await self._create_activities_from_messages(wm, cycle)
        await self._adjust_working_memory(wm, cycle)
        if result.activity is not None:
            return result  # a pre-set selection is respected, not overridden
        # Recompute from wm (not the passed snapshot) so a just-created activity is selectable now.
        # wm.activities preserves insertion (creation) order and is never reordered, so the ready
        # list is oldest-first; the pick itself is delegated to the selection sub-strategy.
        ready = [a for a in wm.activities.values() if a.state is ActivityState.READY]
        selected = await self._activity_selection.select(ready, wm, cycle)
        return result if selected is None else replace(result, activity=selected)

    @staticmethod
    async def _create_activities_from_messages(wm: WorkingMemory, cycle: DecisionCycle) -> None:
        if wm.messages_cursor >= len(wm.messages):
            return  # nothing new since last processed -> the internal action isn't required
        create = cycle.actions.internal(CreateActivityAction.name)
        goals = {a.goal for a in wm.activities.values()}
        for message in wm.messages[wm.messages_cursor :]:  # only messages not yet routed/claimed
            goal = _goal_from_message(message)
            if goal not in goals:  # an unhandled message maps to no existing activity (by goal)
                await create.execute(cycle, goal=goal)
                goals.add(goal)
        wm.messages_cursor = len(wm.messages)  # claim the batch -> each message handled once

    @staticmethod
    async def _adjust_working_memory(wm: WorkingMemory, cycle: DecisionCycle) -> None:
        tools = wm.registry.all_tools()
        manual_ids = {tool.manual.id for tool in tools}
        # Manuals track the joined workspaces; percepts track the narrower ATTENDED set. Only a
        # focused tool is re-observed, so the moment attention narrows a released tool's
        # snapshot is frozen and misleading — this is the housekeeping backstop that drops it even
        # if `release` did not. Signals ignore this set: _filter_ never drops them.
        relevant_ids = set(wm.focused_tools)
        load = cycle.actions.internal(LoadManualAction.name)
        unload = cycle.actions.internal(UnloadManualAction.name)
        filter_ = cycle.actions.internal(FilterPerceptionsAction.name)
        for manual_id in manual_ids - wm.loaded_manuals.keys():
            await load.execute(cycle, manual_id=manual_id)
        for manual_id in wm.loaded_manuals.keys() - manual_ids:
            await unload.execute(cycle, manual_id=manual_id)
        await filter_.execute(cycle, tool_ids=relevant_ids)


# --- parameter grounding: references + a deterministic resolver ----------------------------------
# A plan is a reusable *skeleton*; a param whose value depends on a prior step's result can't be a
# literal at plan time, so the planner emits a *reference* the Reason phase grounds each run against
# the activity's execution history. Two forms (see ADR-0017):
#   hard: {"$from": "<operation_name>", "path": "<dotted path>"}  -> resolved deterministically
#   soft: {"$decide": "<natural-language description>"}           -> always escalates to the model
_REF_FROM = "$from"
_REF_PATH = "path"
_REF_DECIDE = "$decide"
# The named-binding read token. Distinct from $from (which reads Activity.history): $bind reads a
# named binding — either a data-op's output binding in Activity.bindings (ADR-0023), resolved here
# at ground/fan-out time against that dict, or the current loop element of a mechanical sub-goal
# (ADR-0022), which is substituted eagerly at fan-out by _substitute_bindings and so never reaches
# this resolver. The two coexist: the loop element is gone before grounding runs, so any $bind left
# here is a binding read.
_REF_BIND = "$bind"
_REF_NAME = "$bind"  # the binding-name key inside a $bind reference (same token, read as a key)
# The observed-world-state read token. The third and last binding source (ADR-0022): $from reads
# Activity.history, $bind reads the named-binding namespace, $prop reads WorkingMemory.properties —
# the snapshot Observe refreshes each cycle for every focused tool. It resolves per step, at the
# same point $from does, because a property is re-observed state whose whole value is being current;
# binding it once at plan entry would freeze a moving value for the plan's life.
_REF_PROP = "$prop"
_MISSING = object()  # sentinel: no matching history entry (distinct from a genuine None result)
_AMBIGUOUS = object()  # sentinel: a bare property name several focused tools expose
_BAD_PATH = object()  # sentinel: the source IS present, its `path` names nothing inside it


def _is_reference(value: Any) -> TypeGuard[dict[str, Any]]:
    return isinstance(value, dict) and (
        _REF_FROM in value or _REF_DECIDE in value or _REF_BIND in value or _REF_PROP in value
    )


def _latest_result(history: list[CompletedOperation], reference: str) -> Any:
    """The result of the most recent completed operation this reference names, or _MISSING.

    Three accepted spellings, tried most-precise first, because a plan may name an operation any of
    the ways its own brief shows one. The bare ``operation_name`` is the canonical form. The
    fully-qualified ``tool_id.operation_name`` is what a planner reaches for after reading a catalog
    that addresses every operation that way, and it is the *more* specific match when two joined
    workspaces expose the same operation (ARE's Contacts and InternalContacts both have
    ``get_contacts``), so it is honored rather than merely tolerated. Last, the segment after the
    final dot, for a qualification whose prefix matches no tool actually invoked (an abbreviated or
    misremembered tool id) — the operation name never contains a dot, so this is unambiguous.

    Accepting all three is not laxity: a reference the runtime refuses resolves to nothing and, at
    a fan-out, used to vanish silently. Refusing a reference whose *intent* is unambiguous buys no
    safety and costs a whole plan."""

    def _latest(matches: Callable[[OperationInvocation], bool]) -> Any:
        for completed in reversed(history):
            if matches(completed.invocation):
                return completed.ack.result
        return _MISSING

    result = _latest(lambda inv: inv.operation_name == reference)
    if result is not _MISSING:
        return result
    result = _latest(lambda inv: f"{inv.tool_id}.{inv.operation_name}" == reference)
    if result is not _MISSING or "." not in reference:
        return result
    tail = reference.rsplit(".", 1)[-1]
    return _latest(lambda inv: inv.operation_name == tail)


def _property_ref(properties: dict[tuple[str, str], Percept], reference: str) -> tuple[Any, str]:
    """Resolve a ``$prop`` reference to ``(value, residual_path)``, ``_MISSING``, or ``_AMBIGUOUS``.

    Two spellings reach here and both name one value. The canonical one keeps the sub-path in its
    own ``path`` key; a planner that has just read a catalog addressing everything by dotted name
    folds the whole route into the token instead — ``insim:are/Contacts.state.contacts`` for
    ``{"$prop": "insim:are/Contacts.state", "path": "contacts"}``. Honoring only the first cost a
    whole plan on the 2026-08-21 adaptability run, for a spelling difference; this is the tolerance
    ``_latest_result`` already grants ``$from``, applied to the token that lacked it.

    The split is found by **matching against the live key set**, never by parsing the string. That
    matters because the store's key is ``(source, name)`` and *neither half is dot-free*: a WoT tool
    id contains them (``wot:lamp.local/Lamp``), and while today's adapters happen to mint dot-free
    property names, nothing in ``ObservablePropertySpecification`` or the adapter boundary forbids a
    property called ``sensor.temp`` — the runtime does not author names (ADR-0003), so it cannot
    assume their shape. Joining each candidate key back to ``f"{source}.{name}"`` and comparing asks
    the store what it actually holds, so a dotted property name resolves and a dotted tool id keeps
    resolving, with no rule about where the boundary "should" be.

    Longest reference first, so an exact key always beats a folded reading of the same string, and
    within one length a **qualified** match beats a bare one (naming the tool is more specific).
    Where a length is genuinely ambiguous — several tools exposing one bare name, or two distinct
    keys that join to the same string — it comes back ``_AMBIGUOUS`` rather than whichever the dict
    happened to yield first. ARE gives thirteen tools a ``state`` property, so guessing here would
    be a silent wrong answer: worse than a missing one, because it is harder to see."""
    cuts = [i for i, ch in enumerate(reference) if ch == "."]
    for cut in [len(reference), *reversed(cuts)]:
        head, residual = reference[:cut], reference[cut + 1 :]
        keys = [key for key in properties if f"{key[0]}.{key[1]}" == head]
        if not keys:  # unqualified: the planner named the property without its tool
            keys = [key for key in properties if key[1] == head]
        if len(keys) == 1:
            return properties[keys[0]].payload.value, residual
        if keys:
            return _AMBIGUOUS, ""
    return _MISSING, ""


def _resolve_ref(
    ref: dict[str, Any],
    history: list[CompletedOperation],
    bindings: dict[str, Any],
    properties: dict[tuple[str, str], Percept] | None = None,
) -> Any:
    """Resolve one *hard* reference — ``$from`` (history) or ``$bind`` (a named binding) — to its
    value, walking the ``path`` into it. ``_MISSING`` when the source is absent (no such op ran / no
    such binding). Raises on a bad path (a present source, wrong path) so the caller can distinguish
    "escalate" from "left in place". ``$decide`` is soft and never resolved here."""
    if _REF_FROM in ref:
        result = _latest_result(history, str(ref[_REF_FROM]))
        return _MISSING if result is _MISSING else _walk_path(result, ref.get(_REF_PATH, ""))
    if _REF_NAME in ref:
        name = ref[_REF_NAME]
        if name not in bindings:
            return _MISSING
        return _walk_path(bindings[name], ref.get(_REF_PATH, ""))
    if _REF_PROP in ref:
        value, residual = _property_ref(properties or {}, str(ref[_REF_PROP]))
        if value is _MISSING or value is _AMBIGUOUS:
            return _MISSING  # both escalate; _collection_defect re-reads which, to say why
        # A sub-path folded into the token is walked *before* an explicit `path`, since it names the
        # outer route; the two compose so a half-folded reference resolves the same as either form.
        folded = ".".join(p for p in (residual, str(ref.get(_REF_PATH, ""))) if p)
        return _walk_path(value, folded)
    return _MISSING


def _resolve_nested(
    value: Any,
    history: list[CompletedOperation],
    bindings: dict[str, Any],
    properties: dict[tuple[str, str], Percept] | None = None,
) -> tuple[Any, bool]:
    """Resolve every reference *anywhere* in ``value``, returning ``(resolved, fully_resolved)``.

    References nest because the plan schema makes them nest: a param typed ``list[str]`` whose one
    element is only known at run time can *only* be written ``[{"$decide": ...}]`` — a reference as
    the whole value would yield a string, not a list. The resolver used to look at top-level param
    values only, so such a reference was neither resolved nor reported unresolved, and the raw
    ``{"$decide": ...}`` dict was serialized to the tool as a literal (ARE reported it as
    ``Argument 'attendees' must be of type list[str] | None, got <class 'list'>``, naming the wrong
    culprit). It surfaced only when the *sole* reference in a step was nested — any independently
    unresolved top-level param escalated the whole step anyway and the model grounder filled it in,
    which is why it hid for so long.

    A dict that *is* a reference is resolved, not descended into; every other dict/list is rebuilt
    element-wise. Partial resolution is deliberate: a list holding one resolvable ``$from`` and one
    ``$decide`` comes back with the ``$from`` filled and the ``$decide`` left in place, so the
    escalation the caller raises hands the grounder as much settled context as possible."""
    if _is_reference(value):
        if _REF_DECIDE in value:
            return value, False  # soft — always escalates, left in place for the model
        try:
            got = _resolve_ref(value, history, bindings, properties)
        except (KeyError, IndexError, TypeError, ValueError):
            return value, False  # bad path against a present source
        return (value, False) if got is _MISSING else (got, True)
    if isinstance(value, dict):
        pairs = [
            (key, _resolve_nested(item, history, bindings, properties))
            for key, item in value.items()
        ]
        return {key: got for key, (got, _ok) in pairs}, all(ok for _key, (_got, ok) in pairs)
    if isinstance(value, list):
        items = [_resolve_nested(item, history, bindings, properties) for item in value]
        return [got for got, _ok in items], all(ok for _got, ok in items)
    return value, True


def _reference_paths(value: Any, prefix: str = "") -> list[str]:
    """Dotted paths of every reference token surviving in ``value`` — at any depth. ``[]`` is the
    healthy case. Used as Act's leak guard; see ``DefaultActStrategy``."""
    if _is_reference(value):
        return [prefix or "<root>"]
    if isinstance(value, dict):
        return [
            path
            for key, item in value.items()
            for path in _reference_paths(item, f"{prefix}.{key}" if prefix else str(key))
        ]
    if isinstance(value, list):
        return [
            path
            for index, item in enumerate(value)
            for path in _reference_paths(item, f"{prefix}.{index}" if prefix else str(index))
        ]
    return []


def _manual_for(wm: WorkingMemory, tool_id: str | None) -> Manual | None:
    """The joined tool's manual (the operation schema the model grounds against), or None."""
    if tool_id is None:
        return None
    try:
        return wm.registry.get(tool_id).manual
    except KeyError:
        return None


def resolve_references(
    op_params: dict[str, Any],
    history: list[CompletedOperation],
    bindings: dict[str, Any] | None = None,
    properties: dict[tuple[str, str], Percept] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Resolve a step's operation params against execution history and named bindings. Non-reference
    values pass through; a hard reference (``$from``/``$bind``) is resolved deterministically;
    anything that can't be resolved mechanically (soft ``$decide``, missing source, bad path) is
    left in place and its key returned in ``unresolved`` for the caller to escalate. Never raises on
    a bad path — that's an escalation signal, not an error.

    A reference is found wherever it sits — as a param's whole value, or nested inside a list or
    dict the param holds (see ``_resolve_nested``). ``unresolved`` names the *top-level key*
    whatever the depth, because that is the unit grounding escalates on (``partial_params`` is
    per-param), so one stubborn leaf re-grounds its whole param."""
    binds = bindings or {}
    resolved: dict[str, Any] = {}
    unresolved: list[str] = []
    for key, value in op_params.items():
        got, ok = _resolve_nested(value, history, binds, properties)
        resolved[key] = got
        if not ok:
            unresolved.append(key)
    return resolved, unresolved


# --- irreversibility guard: never commit a write on a plan already known to be dead ---------------
# A run showed why this is needed. A filter chain wrote `friend_contact = []` at step 3 (the planner
# had read only the first page of contacts, so nobody matched); the plan then kept going — reading
# the calendar at step 4 and DELETING the user's real appointment at step 5 — and only tripped over
# the empty binding at step 7, where it finally needed the friend's address. The plan was already
# unfinishable two steps before the irreversible act, and the evidence was sitting in `bindings`.
#
# So this is not an ordering rule: that plan *was* ordered correctly, gathering before destroying.
# It is a viability rule. Nothing rolls a delete back, and the asymmetry is stark — abandoning a
# plan that might still have worked costs one more inference, while acting on a plan that cannot
# work costs the user something real and unrecoverable. So: check before a write, and only a write.
#
# "Provably" is meant strictly. Only a binding an earlier step *already produced* and produced empty
# counts, and only where a later step reads a VALUE out of it — a collection position (`in`, a
# membership `where`) is exempt, because an empty collection there is a legitimate answer ("nothing
# to iterate", "exclude nothing"), which is the same line _data_op already draws. A name a later
# step rewrites is exempt from that point on, since it is no longer provably anything.
#
# The same proof holds for a `$from` read of an operation that already ran and came back empty, and
# a later run showed the guard missing it for want of scanning that token: a plan invoked
# `search_contacts`, got `[]`, and the runtime still committed `add_calendar_event` two steps on,
# creating the event with no attendee. Identical evidence, identical asymmetry, different spelling —
# so both references are scanned, with `refreshed` playing for operations the part `out` plays for
# bindings (a step ahead of the read that re-invokes the operation makes it no longer provably
# anything, which is what keeps a replan's second attempt at a search from being condemned by its
# first attempt's empty result).
#
# What does NOT count for a `$from`, deliberately: an operation that has not run yet (a plan
# normally reads at step 3 what it invokes at step 1 — absence here is not evidence), and a
# *present but mis-pathed* source. The latter is a real defect but a recoverable one: grounding
# reads the actual history and routinely resolves a value the path spelled wrong, so condemning
# would pre-empt a repair that works. An empty source admits no such repair — there is no value at
# any path — which is the line between the two.

# Keys whose value is a collection rather than a value read out of one: an empty binding in these
# positions is an answer, not a defect (see above). `from` is the collect data-op's operation name.
_COLLECTION_KEYS = frozenset({"in", "where", "from"})


def _is_empty(value: Any) -> bool:
    """Empty in the sense that a step reading a value out of it cannot get one. Deliberately not
    falsiness: 0 and False are perfectly good values a step can act on."""
    if value is None:
        return True
    return len(value) == 0 if isinstance(value, str | bytes | list | tuple | set | dict) else False


def _dereferenced_bindings(step: Step) -> set[str]:
    """Binding names this step reads a *value* out of, at any nesting depth.

    A mechanical sub-goal's template always contains ``{"$bind": "<as>"}`` — its own loop element,
    bound per iteration at fan-out, not a name the plan produced. Nothing stops a plan from
    spelling ``as`` the same as a real binding (``filter(out: "contacts")`` then
    ``subgoal(in: {"$bind": "contacts"}, as: "contacts")`` is a natural thing to write), and when
    that binding is empty the template's read looked like a dereference of it — so a fan-out that
    legitimately reduces to zero steps ("nothing matched, nothing to do") condemned the plan at the
    next write. The loop name is excluded explicitly rather than assumed distinct."""
    loop_var = step.params.get("as")
    names: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            name = value.get(_REF_BIND)
            if isinstance(name, str):
                if name != loop_var:
                    names.add(name)
                return
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    for key, value in step.params.items():
        if key not in _COLLECTION_KEYS:
            walk(value)
    return names


# --- attributing an empty binding to the step that produced it ------------------------------------
# An empty binding is evidence about the step that WROTE it, not the step that tripped over it, and
# the two defect channels both used to describe it from the reader's side. Grounding does so for a
# reason it cannot help: it is handed one step's params and the history, never the plan, so the most
# it can honestly report is which parameter came up short. The replanning prompt then says the
# replacement "has to differ THERE" — and THERE resolved to the reader. A run showed the cost: the
# planner rewrote the invoke that read the binding, re-emitted the identical `eq` filter that had
# written it empty, and bound empty again, three times until the replan breaker parked the activity.
#
# The runtime holds the plan that the grounder does not, so it names the producer mechanically —
# no prose parsing, no second model call. It only INFORMS: the empty collection may be the true
# answer ("that record really is absent"), and the existing framing's "tell the user instead" escape
# has to stay reachable, so nothing here instructs the planner to change that step.
#
# The producing step is deliberately identified by its action and its own params rather than by
# index. It has already RUN, so it is absent from the discarded plan's rendered tail — and
# render_superseded_plan renumbers that tail from 0 precisely so a listed step is not misread as an
# executed one. An index quoted here would point at a different step than the one meant.


def _binding_source(step: Step) -> str | None:
    """The binding a data-op step reads its input collection from, when that is where its input came
    from. Only a ``$bind`` input continues a chain of emptiness: a ``$prop``/``$from``/literal input
    is where the chain ends, and ``collect`` reads an operation rather than a collection.

    Attribution therefore stops at a ``filter`` whose own ``$prop``/``$from`` input was already
    empty, describing that step's predicate when the source it read was the cause. The outcome
    wording stays true and the replan prompt still carries the history showing the upstream
    miss, but the clause points one step downstream. Closing that needs the operation history
    (for ``$from``) and ``WorkingMemory`` (for ``$prop``) threaded into
    :func:`_empty_binding_origin`, which today is given only the activity."""
    source = step.params.get("in")
    if isinstance(source, dict):
        name = source.get(_REF_BIND)
        if isinstance(name, str):
            return name
    return None


def _executed_steps(activity: Activity) -> list[Step]:
    """The activity's already-run steps in execution order, flattened across the frame stack: each
    suspended parent's executed prefix (it ran before the sub-plan was spliced in), then the live
    frame's. Whatever wrote a binding is in here — a binding exists only because a step ran."""
    steps: list[Step] = []
    for parent, index, _ in activity.parent_frames:
        steps.extend(parent.steps[:index])
    if activity.plan is not None:
        steps.extend(activity.plan.steps[: activity.step_index])
    return steps


def _root_empty_producer(
    executed: list[Step], name: str, empty: frozenset[str]
) -> tuple[Step, bool] | None:
    """``(the step an empty binding originates in, whether the chain was walked)``, else ``None``.

    A data-op fed an input binding that was ALREADY empty is only passing the emptiness along;
    naming it would misattribute by one link, which is the same defect this exists to fix. So the
    walk continues through such producers to the first one whose own input was not empty — the
    ``$decide`` filter over an empty ``eq`` result names the ``eq``. ``seen`` guards a plan that
    writes two bindings from each other rather than trusting it cannot."""
    seen: set[str] = set()
    producer: Step | None = None
    walked = False
    while name not in seen:
        seen.add(name)
        wrote = next((s for s in reversed(executed) if s.params.get("out") == name), None)
        if wrote is None:
            break
        walked = producer is not None
        producer = wrote
        source = _binding_source(wrote)
        if source is None or source not in empty:
            break
        name = source
    return None if producer is None else (producer, walked)


def _and_list(names: list[str]) -> str:
    quoted = [repr(name) for name in names]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def _empty_binding_origin(activity: Activity, names: list[str], empty: frozenset[str]) -> str:
    """The clause naming where the empty bindings came from, or ``""`` when none can be attributed
    (a binding no step of this plan wrote — a sub-goal element, or one carried in). Silence is the
    right degradation: the defect without it is what shipped before.

    Bindings sharing one root are named in ONE clause. A `filter` then a `take` off its result are
    two dead bindings with a single cause, and describing that step twice reads as two independent
    defects — noise in a message whose whole point is to say what to write differently."""
    executed = _executed_steps(activity)
    grouped: dict[int, tuple[Step, bool, list[str]]] = {}
    for name in names:
        found = _root_empty_producer(executed, name, empty)
        if found is None:
            continue
        step, walked = found
        origin, seen, bound = grouped.get(id(step), (step, False, []))
        grouped[id(step)] = (origin, seen or walked, [*bound, name])
    clauses: list[str] = []
    for step, walked, bound in grouped.values():
        # `in`/`out` are the plumbing; what discriminates the step is the rest (a `where`, a `by`).
        params = {key: value for key, value in step.params.items() if key not in ("in", "out")}
        # A step with nothing left to show (a `distinct` over whole items) gets no empty `{}` — the
        # dash-clause is there to quote the part that missed, not to prove one was looked for.
        shown = f" — {json.dumps(params, default=str)} —" if params else ","
        outcome = (
            "matched no items in its input collection"
            if step.next_action == "filter"
            else "produced an empty result"
        )
        derived = "; every binding derived from it was empty in consequence" if walked else ""
        clauses.append(
            f" The empty {_and_list(bound)} {'was' if len(bound) == 1 else 'were'} produced by an "
            f"earlier `{step.next_action}` step of this plan{shown} which {outcome}{derived}."
        )
    return "".join(clauses)


def _with_empty_binding_origin(activity: Activity, defect: str) -> str:
    """Append the origin clause to a grounder-authored defect, for the empty bindings the step it
    failed on actually reads. Computed from the plan rather than read out of the grounder's prose:
    the names are already in ``bindings``, so parsing a model's sentence for them would be a
    fragility with nothing to buy it."""
    plan = activity.plan
    if plan is None or not 0 <= activity.step_index < len(plan.steps):
        return defect
    empty = frozenset(name for name, value in activity.bindings.items() if _is_empty(value))
    dead = sorted(_dereferenced_bindings(plan.steps[activity.step_index]) & empty)
    origin = _empty_binding_origin(activity, dead, empty) if dead else ""
    if not origin:
        return defect
    # The grounder is asked for "<which parameter, and what was missing>" and answers with a
    # fragment ("product_id: matches is empty"), not a sentence — close it, or the two run together.
    head = defect.rstrip()
    return head + ("" if head.endswith((".", "!", "?", ":", ";")) else ".") + origin


# Shared tail of both defect strings: what the planner should do about it. The corrections are the
# same whichever token carried the dead reference, and naming them is what makes the retry differ.
_REPLAN_HINT = (
    "Re-plan a way to obtain that data (read the whole collection rather than one page, filter on "
    "a different field, search by another term) before any step that changes the world; the "
    "runtime stopped short of the next one."
)


def _dereferenced_operations(step: Step) -> list[dict[str, Any]]:
    """The ``$from`` reference objects this step reads a *value* out of, at any nesting depth — the
    reference itself, not just its name, since whether it can yield a value depends on its ``path``.

    Nesting matters more here than for ``$bind``: a ``$decide`` element carries its own source under
    a plain ``from`` key (``{"$decide": "...", "from": {"$from": "search", "path": "0"}}``), which
    is the shape that actually slipped a write past this guard. Note that ``_COLLECTION_KEYS``
    filters only the step's *top-level* params, so such a nested ``from`` is still walked — the two
    senses of the word do not collide."""
    refs: list[dict[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get(_REF_FROM), str):
                refs.append(value)
                return
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    for key, value in step.params.items():
        if key not in _COLLECTION_KEYS:
            walk(value)
    return refs


def _spent_operation_read(
    ref: dict[str, Any], history: list[CompletedOperation], refreshed: set[str]
) -> tuple[str, str] | None:
    """``(operation, what it yielded)`` when this reference provably cannot produce a value, else
    ``None``. See the guard's header for why "never ran" and "wrong path" are both excluded."""
    name = str(ref[_REF_FROM])
    # Tail comparison covers all three spellings _latest_result accepts; erring toward "refreshed"
    # errs toward NOT condemning, which is the safe direction for a guard that abandons plans.
    if name in refreshed or name.rsplit(".", 1)[-1] in refreshed:
        return None
    result = _latest_result(history, name)
    if result is _MISSING:
        return None  # not yet run — the step that runs it may be ahead of this read
    if _is_empty(result):
        return name, "returned an empty result"  # no path finds a value in it
    try:
        value = _walk_path(result, str(ref.get(_REF_PATH, "")))
    except (KeyError, IndexError, TypeError, ValueError):
        return None  # present but mis-pathed: grounding can still recover the value
    path = str(ref.get(_REF_PATH, ""))
    return (name, f"returned nothing at {path!r}") if _is_empty(value) else None


def _invoked_operation(step: Step) -> str | None:
    if step.next_action != "invoke":
        return None
    name = step.params.get("operation_name")
    return name if isinstance(name, str) else None


def _unsatisfiable_reference(activity: Activity) -> str | None:
    """Where the rest of the plan dereferences data that provably is not there — a binding an
    earlier step produced empty, or a ``$from`` naming an operation that already ran and came back
    empty — described as a defect for the replanning prompt; ``None`` when nothing is provably dead.
    Scans the active frame and then every suspended parent in resume order, since a sub-plan's
    caller runs later and reads the same flat `bindings` and the same history."""
    plan = activity.plan
    if plan is None:
        return None
    empty = {name for name, value in activity.bindings.items() if _is_empty(value)}
    # Captured before the forward scan starts discarding rewritten names: attribution asks what is
    # empty NOW, which is what the producer walk has to chase through.
    empty_now = frozenset(empty)
    refreshed: set[str] = set()
    frames = [(plan, activity.step_index)]
    frames += [(parent, index + 1) for parent, index, _ in reversed(activity.parent_frames)]
    for frame, start in frames:
        for index in range(start, len(frame.steps)):
            step = frame.steps[index]
            dead = sorted(_dereferenced_bindings(step) & empty)
            if dead:
                return (
                    f"step {index} ({step.next_action}) reads {', '.join(repr(n) for n in dead)}, "
                    "which an earlier step of this plan produced EMPTY — nothing matched it, so "
                    "that step cannot work and the plan cannot finish as written."
                    f"{_empty_binding_origin(activity, dead, empty_now)} {_REPLAN_HINT}"
                )
            for ref in _dereferenced_operations(step):
                spent = _spent_operation_read(ref, activity.history, refreshed)
                if spent is not None:
                    name, yielded = spent
                    return (
                        f"step {index} ({step.next_action}) reads a value out of {name!r}, which "
                        f"already ran in this run and {yielded} — so that step cannot work and the "
                        f"plan cannot finish as written. {_REPLAN_HINT}"
                    )
            out = step.params.get("out")
            if isinstance(out, str):
                empty.discard(out)  # rewritten here -> no longer provably empty further down
            invoked = _invoked_operation(step)
            if invoked is not None:
                refreshed.add(invoked)  # re-run here -> its old empty result proves nothing below
    return None


# --- sub-goals: mechanical fan-out over a collection (ADR-0022) -----------------------------------
# A `subgoal` Step with mode="mechanical" is expanded in Reason into one concrete step per element
# of a run-time collection — the count is len(data), not a model guess (the RentAFlat "for each"
# fix). _SUBGOAL_RUNNING / _SUBGOAL_SPLICED are the two outcomes _subgoal reports to reason():
# a deliberative sub-goal fired _infer_ and is RUNNING (return, no step); a mechanical one spliced
# its expansion into the plan in place (re-loop and read the first expanded step); a deliberative
# one the loop-guard refused pauses the activity to await input (no step) -> _SUBGOAL_HALTED;
# a mechanical one whose collection could not be read dropped the plan (no step) -> _SUBGOAL_DEFECT.
_SUBGOAL_RUNNING = object()
_SUBGOAL_SPLICED = object()
_SUBGOAL_HALTED = object()
_SUBGOAL_DEFECT = object()

# Circuit breaker for runaway deliberative sub-goal recursion (ADR-0022's deferred overflow valve,
# pulled forward). Synthesis-as-selection has no termination guarantee an *authored* plan library
# has: the model can satisfy "plan for goal G" by emitting a plan whose body is another deliberative
# sub-goal for ~G, deferring instead of reducing, and recurse until a budget (or credit) runs out.
# Two mechanical detectors, tripped before the _infer_ spend: a depth cap on the intention stack
# (configurable per DefaultReasonStrategy, wired from agent.yaml's `max_subgoal_depth`), and
# token-overlap against the ancestor sub-goals — a new sub-goal whose tokens are largely contained
# in one still on the stack is re-stating it, not reducing. Overlap (|A&B| / min), not Jaccard: the
# observed regress *elaborates* the same goal (piling on qualifiers), which grows the union and
# sinks Jaccard while the core token set stays contained, so containment is what catches the reword.
# Tripping pauses to await-input (ADR-0020) rather than terminating, so a deep-but-legitimate task
# can be redirected, not killed. Both are coarse backstops; the real fix is making the common
# map/filter/distinct shapes expressible without deliberation at all.
_DEFAULT_MAX_SUBGOAL_DEPTH = 4
_SUBGOAL_GOAL_OVERLAP = 0.7

# Circuit breaker for runaway *replanning* — the same failure mode one level out. A plan is dropped
# (a defect found in it, or reconsideration invalidating it), the replacement is dropped too, and
# nothing ever executes; each turn of that loop costs a full planning inference, which on a local
# model was minutes apiece in an observed run. Counted against `Activity.replan_trail`, which holds
# only replans with no operation between them, so this never limits an agent adapting to a world
# that keeps moving — the design center — and limits only one that is getting nowhere. Two
# mechanical detectors, tripped before the _infer_ spend, mirroring the sub-goal breaker above:
# a repeated *defect* (the planner was told what was wrong and wrote it again — no third attempt
# will differ, so this trips at two), and a plain count as the coarse backstop for the case where
# every attempt fails differently. Tripping pauses to await-input (ADR-0020) rather than
# terminating, so the run can be redirected rather than killed.
_DEFAULT_MAX_REPLAN_ATTEMPTS = 5


def _replan_halt_prompt(activity: Activity, halt: str) -> str:
    """The await-input text for a tripped replan breaker: the goal, why it stopped, and what each
    abandoned plan ran into, oldest first. Rendered mechanically — no model call, matching the
    sub-goal breaker. Summarizing why the model keeps failing is the last place to spend another
    inference, and the trail is already the specific, quotable evidence a person needs to answer."""
    attempts = "\n".join(
        f"  {i}. {reason if reason is not None else 'the world changed under the plan'}"
        for i, reason in enumerate(activity.replan_trail, start=1)
    )
    return (
        f"Stuck on {activity.goal!r}: {halt}.\n"
        f"What each abandoned plan ran into:\n{attempts}\n"
        "How should I proceed?"
    )


def _goal_token_overlap(a: str, b: str) -> float:
    """Token overlap coefficient over two goal strings — ``|A&B| / min(|A|, |B|)``, 1.0 when the
    smaller token set is contained in the larger, 0.0 disjoint. Cheap and deterministic (no model
    call). Overlap, not Jaccard: the non-reducing recursion re-states an ancestor's goal with extra
    qualifiers, growing the union (which sinks Jaccard) while the core stays contained — containment
    is the signal that survives the reword."""
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _ancestor_subgoal_goals(activity: Activity) -> list[str]:
    """The goals of the deliberative sub-goals still suspended on the intention stack — each parent
    frame's ``(plan, idx)`` points back at the ``subgoal`` step that pushed it. The root
    ``activity.goal`` is deliberately excluded: the first decomposition legitimately shares its
    vocabulary, so comparing against it would false-trip a single, valid refinement."""
    goals: list[str] = []
    for plan, idx, _ in activity.parent_frames:
        if 0 <= idx < len(plan.steps):
            goal = plan.steps[idx].params.get("goal")
            if isinstance(goal, str):
                goals.append(goal)
    return goals


# The names a windowed list operation uses for the metadata it returns *beside* its payload. Closed
# and deliberately short: it is the only thing separating a paginated envelope from a record that
# happens to carry one list field, so every addition widens what gets read as a collection. A name
# belongs here only if no tool would plausibly use it for a record's own field.
_PAGE_META = frozenset(
    {
        "count",
        "cursor",
        "has_more",
        "limit",
        "next_cursor",
        "next_offset",
        "offset",
        "page",
        "page_size",
        "per_page",
        "range",
        "total",
        "total_count",
        "view_limit",
    }
)


def _paginated_payload(value: dict[str, Any]) -> list[Any] | None:
    """The payload list out of ``{"events": [...], "range": ..., "total": ...}``, or ``None`` when
    this is not that shape: exactly one list-valued key, every other key a ``_PAGE_META`` scalar.
    The vocabulary check is the whole load-bearing part — without it a one-list-field record
    qualifies (see ``_as_collection`` tier 3)."""
    payload: list[Any] | None = None
    for key, item in value.items():
        if isinstance(item, list):
            if payload is not None:
                return None  # two candidate payloads: which one was meant is not mechanical
            payload = item
        elif isinstance(item, dict) or key not in _PAGE_META:
            return None
    return payload


def _as_collection(value: Any) -> list[Any] | None:
    """Coerce a resolved value to the list a fan-out/pipeline iterates, in deterministic tiers so a
    plan author never has to hand-shape a tool's return:

    1. **list** -> itself.
    2. **single-key envelope** (a lone key wrapping the payload, e.g. ``{"apartments": {id -> r}}``
       or ``{"results": [...]}``): unwrap and recurse into the one value, *iff* that value is itself
       a collection. A single-element ``{id -> record}`` map whose record has *any* scalar field
       falls through here (the recursion refuses a mixed/scalar record) and tier 3 catches it. The
       residual ambiguity is any single-element map ``{"a1": {record}}`` whose lone record's fields
       are *all* mapping-valued: that record is itself indistinguishable from an ``{id -> record}``
       map (a one-field record ``{"a1": {"photos": [...]}}`` and a many-field one
       ``{"a1": {"loc": {...}, "meta": {...}}}`` both recurse to a collection), so it is unwrapped
       into the record's field-*values* rather than kept as one record — a genuine
       misclassification for such a shape. This is **undecidable** at this layer
       (``{K: {k1: {...}, k2: {...}}}`` is structurally identical whether ``K`` is an id or a
       wrapper name); the unwrap is chosen because ARE's records always carry scalar fields (so they
       never reach it) and its real envelopes are plural ``{id -> record}`` maps that must unwrap.
       The principled fix for a tool that returns all-mapping-field records is the deferred
       model-escalated extraction below, not another shape heuristic — every mechanical tie-break
       here only shifts *which* shape misfires.
    3. **paginated envelope** (a lone list-valued key beside pagination metadata, e.g. ARE's
       ``get_calendar_events_from_to`` -> ``{"events": [...], "range": "(0, 1)", "total": 1}``):
       take the list. Unlike tier 2 this cannot be decided structurally — a record with one list
       field and scalar siblings (``{"event_id": ..., "title": ..., "attendees": [...]}``, an ARE
       calendar event) is *shape-identical* to the envelope, and reading that as a collection of
       attendees would be worse than refusing. So the siblings must additionally all be scalars
       drawn from a closed pagination vocabulary (``_PAGE_META``), which a record's own field names
       are not in. Narrow on purpose: it buys the one shape ARE's windowed list operations actually
       return, and nothing else.
    4. **``{id -> record}`` mapping** (ARE's ``list_all_apartments`` / ``search_apartments`` /
       ``list_saved_apartments``): every value a mapping -> iterate the *values*; the id is carried
       inside each record, so values-iteration is lossless.
    5. anything else -> ``None``: a single record's fields, an ``{id -> scalar}`` map, a scalar. NOT
       mechanically a collection, so the caller logs the shape rather than fanning out over garbage.

    An empty dict is an empty collection (``[]``), not "unresolvable". (Model-escalated extraction
    for shapes these tiers still refuse is deferred.)"""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if not value:
            return []  # an empty {id -> record} map is an empty collection, not "unresolvable"
        if len(value) == 1:
            inner = _as_collection(next(iter(value.values())))
            if inner is not None:
                return inner  # single-key envelope: the lone value is the real collection
        paginated = _paginated_payload(value)
        if paginated is not None:
            return paginated
        if all(isinstance(v, dict) for v in value.values()):
            return list(value.values())
    return None  # single record / envelope-of-scalars / id->scalar / scalar: refuse to guess


def _collection_defect(
    ref: Any,
    value: Any,
    history: list[CompletedOperation],
    properties: dict[tuple[str, str], Percept] | None = None,
) -> str:
    """Why a collection reference could not be read, phrased for the *planner* rather than the log:
    it goes into the replan brief, so it has to say what to write instead. Both cases have a
    concrete correction available, and naming it is the difference between a retry that differs and
    one that repeats — the same reason the undeclared-parameter defect lists the accepted names."""
    if isinstance(ref, dict) and _REF_PROP in ref and value is _MISSING:
        name = str(ref[_REF_PROP])
        props = properties or {}
        candidates = sorted({source for (source, prop) in props if prop == name})
        if candidates:
            return (
                f"{name!r} is exposed by several focused tools ({', '.join(candidates)}) — "
                "qualify it as '<tool_id>.<property_name>' so it names exactly one."
            )
        observed = ", ".join(sorted(f"{s}.{p}" for (s, p) in props))
        # Deliberately does NOT prescribe a 'focus' step. The runtime already attends to every tool
        # a live plan names, so a focus step cannot conjure a property that is not in this list —
        # the name is wrong, or its tool's workspace was never joined. Sending the planner to add a
        # focus step buys a replan that repeats the same reference, which is the one outcome a
        # defect message exists to prevent.
        return (
            f"{name!r} names no observed property; currently observed: "
            f"{observed or 'none — no tool is being observed'}. Reference one of those, qualified "
            "as '<tool_id>.<property_name>'. If the property you want belongs to a tool that is "
            "not listed, its workspace has not been joined — plan a 'join' step first, or reach "
            "the value through an operation instead."
        )
    if value is _MISSING:
        ran = sorted({c.invocation.operation_name for c in history})
        available = ", ".join(ran) if ran else "none yet — no operation has run"
        return (
            f"{ref!r} names no result the plan has produced; operations run so far: {available}. "
            "Reference a collection only after a step has produced it."
        )
    if isinstance(value, dict):
        keys = ", ".join(repr(k) for k in list(value)[:8])
        return (
            f"{ref!r} resolved to a dict with keys {keys}, not a list the runtime can iterate — "
            "add a 'path' naming the field that holds the list."
        )
    return (
        f"{ref!r} resolved to a {type(value).__name__}, not a list the runtime can iterate — "
        "reference something that is a collection, or narrow to one with a data-op first."
    )


def _path_defect(
    ref: dict[str, Any],
    history: list[CompletedOperation],
    bindings: dict[str, Any],
    properties: dict[tuple[str, str], Percept] | None = None,
) -> str:
    """Why a ``path`` failed against a source that *is* present — kept apart from the missing-source
    defect on purpose. Collapsing the two told the planner ``{'$from': 'search_events', 'path':
    'events'} names no result the plan has produced; operations run so far: search_events`` — a
    brief that contradicts itself and aims the repair at a step that had just run. The planner
    rewrote the same reference, the trail saw the same defect twice, and the activity halted on a
    question the user could not usefully answer. So: say the source ran, name the segment that did
    not fit, and show what is actually there to name instead.

    A ``$prop`` is split by `_property_ref` rather than by dropping ``path``, because its route can
    be *folded into the token* — for `{"$prop": "Contacts.state.nope"}` there is no ``path`` key to
    drop, so re-resolving the stripped ref just re-ran the same failing walk and re-raised out of
    this function, out of `_resolve_collection`, and out of `tick()`: the plan-defect layer aborting
    the run it exists to recover. Splitting head from route composes the folded and explicit halves
    the same way `_resolve_ref` does, so either spelling reports the same segment."""
    if _REF_PROP in ref:
        source, residual = _property_ref(properties or {}, str(ref[_REF_PROP]))
        if source is _MISSING or source is _AMBIGUOUS:
            # The HEAD did not resolve: an unobserved or ambiguous property is a different question
            # from a bad route, and _collection_defect is the one that names focusing/qualifying.
            return _collection_defect(ref, _MISSING, history, properties)
        path = ".".join(p for p in (residual, str(ref.get(_REF_PATH, ""))) if p)
        return _walk_defect(ref, source, path)
    source = _resolve_ref(
        {k: v for k, v in ref.items() if k != _REF_PATH}, history, bindings, properties
    )
    path = str(ref.get(_REF_PATH, ""))
    return _walk_defect(ref, source, path)


def _walk_defect(ref: dict[str, Any], source: Any, path: str) -> str:
    """Walk `path` into an already-resolved `source` and describe where it stopped fitting."""
    value: Any = source
    walked: list[str] = []
    failed = path
    for segment in filter(None, path.split(".")):
        try:
            value = value[int(segment)] if segment.isdigit() else value[segment]
        except (KeyError, IndexError, TypeError, ValueError):
            failed = segment
            break
        walked.append(segment)
    at = ".".join(walked) or "the result itself"
    if isinstance(value, dict):
        keys = ", ".join(repr(k) for k in list(value)[:8]) or "no keys"
        holds = f"{at} is a mapping with keys {keys}"
    elif isinstance(value, list):
        holds = f"{at} is a list of {len(value)} item(s) — only a numeric segment indexes it"
    else:
        holds = f"{at} is a {type(value).__name__}"
    return (
        f"{ref!r} names a source that IS present and does not need to run again, but its 'path' "
        f"does not fit that result: {failed!r} is not readable there, because {holds}. Correct the "
        "'path' to a field that is, or drop 'path' if the source is already the collection."
    )


def _resolve_collection(
    ref: Any,
    history: list[CompletedOperation],
    bindings: dict[str, Any] | None = None,
    properties: dict[tuple[str, str], Percept] | None = None,
) -> tuple[list[Any] | None, str | None]:
    """The collection a mechanical sub-goal iterates or a data-op transforms: a ``$from`` reference
    resolved against history, a ``$bind`` reference resolved against named bindings (a prior data-op
    output), or a literal list. A resolved mapping iterates its values (see ``_as_collection``).
    Returns ``(collection, defect)``. A ``defect`` is set exactly when the reference could not be
    *read* — a missing source, a bad path, or a resolved value of a shape these tiers refuse — and
    the caller replans on it rather than proceeding. That distinction is the whole point of the
    pair: a collection that is legitimately empty and a collection the runtime could not read both
    used to come back as "nothing to do", so a sub-goal over an unreadable reference fanned out to
    zero steps and the plan sailed past it as though the work were done. An observed run dropped
    three calendar cancellations that way without a single error surfacing. Empty is an answer;
    unreadable is a question, and only the second is a plan defect worth another inference.

    ``(None, None)`` means soft, not failed: a ``$decide`` collection is resolved off-cycle, so
    there is nothing to read here yet and nothing to blame the plan for."""
    if _is_reference(ref):
        if _REF_DECIDE in ref:
            return None, None  # a $decide collection is soft — resolved off-cycle, not a defect
        try:
            value: Any = _resolve_ref(ref, history, bindings or {}, properties)
        except (KeyError, IndexError, TypeError, ValueError):
            return None, _path_defect(ref, history, bindings or {}, properties)
        if value is _MISSING:
            return None, _collection_defect(ref, value, history, properties)
    else:
        value = ref  # a literal (already a list, mapping, or a plan author's mistake)
    collection = _as_collection(value)
    if collection is None:
        return None, _collection_defect(ref, value, history, properties)
    return collection, None


def _enrich_with_params(result: Any, params: dict[str, Any]) -> Any:
    """``collect`` carries each fanned-out call's input params alongside its result, so a downstream
    ``filter``/membership can correlate a result back to the input that produced it — a crime rate
    back to its ``zip_code`` when ``get_crime_rate`` doesn't echo the zip. A dict result is enriched
    in place, the params filling in only the keys the result doesn't already carry (the
    authoritative return wins on collision); a non-dict result is wrapped as
    ``{**params, "result": <value>}`` so the key stays reachable. Empty params (or none) -> the
    result untouched."""
    if not params:
        return result
    if isinstance(result, dict):
        return {**params, **result}
    return {**params, "result": result}


_ORDERED_OPS = ("lt", "le", "gt", "ge")  # the ops whose operand must be a single comparable value


def _shape_of(value: Any) -> str:
    """A resolved value's shape, phrased for a plan defect — what the planner needs to see is what
    it got instead of a comparable value, not the value itself (which can be a whole record)."""
    if value is None:
        return "None"
    if isinstance(value, dict):
        keys = ", ".join(repr(k) for k in list(value)[:8]) or "no keys"
        return f"a mapping with keys {keys}"
    if isinstance(value, (list, tuple)):
        return f"a list of {len(value)} item(s)"
    return f"a {type(value).__name__}"


def _resolve_operand_items(
    items: list[Any] | tuple[Any, ...],
    history: list[CompletedOperation],
    bindings: dict[str, Any],
    properties: dict[tuple[str, str], Percept] | None = None,
) -> tuple[list[Any], str | None]:
    """Resolve references written *inside* a list operand, element-wise.

    A list is not itself a reference, so ``_is_reference`` says no to it and the whole thing used
    to pass through as a literal. But the natural way to write a ``between`` whose ends are each
    computed is exactly a list of references — ``[{"$bind": "lo"}, {"$bind": "hi"}]`` — since a
    reference in the *whole-value* position would have to resolve to the pair already assembled,
    which no single earlier step produces. Written the natural way, the reference dicts reached
    ``_matches`` intact, every ``lo <= actual`` raised ``TypeError``, that was caught as a
    non-match, and the filter kept nothing at all while reporting an ordinary empty result. The
    same applies to a literal ``in`` set with a reference among its members.

    Resolving per element makes both spellings mean what they read as. A defect here is a defect
    for the whole predicate: a pair with one unreadable end is no more comparable than one with
    two."""
    resolved: list[Any] = []
    for item in items:
        if not _is_reference(item):
            resolved.append(item)
            continue
        if _REF_DECIDE in item:
            return [], (
                f"a filter predicate's 'value' cannot contain a $decide reference "
                f"({item[_REF_DECIDE]!r}) — the comparison runs mechanically, so every part of "
                "the operand has to be known before it. Compute it in an earlier step and "
                "reference that binding, or make the whole 'where' a $decide predicate."
            )
        try:
            value = _resolve_ref(item, history, bindings, properties)
        except (KeyError, IndexError, TypeError, ValueError):
            return [], _path_defect(item, history, bindings, properties)
        if value is _MISSING:
            return [], _collection_defect(item, value, history, properties)
        resolved.append(value)
    return resolved, None


def _operand_defect(where: dict[str, Any], written: Any, operand: Any) -> str | None:
    """Why a predicate operand that *read* cleanly still cannot be compared against.

    Resolving is not the same as being usable, and the gap between the two is silent: an operand
    that lands on ``None``, a whole record, or a list makes every ordered comparison raise
    ``TypeError`` inside ``_matches``, where it is caught as a non-match by design — so no element
    survives and the step writes an empty binding that reads downstream as a fact about the world.
    ``between`` is the same trap one level up: it compares against exactly a two-element pair and
    treats anything else as a blanket non-match. In both cases the filter is not selecting badly,
    it is not selecting at all, so this is reported as a plan defect rather than an answer.

    ``eq``/``ne`` are deliberately excluded. An operand of any shape can genuinely match there — a
    field that really is null, an object compared whole — so refusing one would refuse a
    legitimate predicate to guard against a mistake that isn't provable from the shape alone.
    ``in``/``not_in`` are excluded too: their operand is a collection, already checked as one."""
    op = where.get("op", "eq")
    if op == "between":
        if not (isinstance(operand, (list, tuple)) and len(operand) == 2):
            return (
                f"the 'between' predicate's value {written!r} is {_shape_of(operand)}, not the "
                "[lo, hi] pair 'between' compares against — every element then fails the "
                "comparison and the filter keeps nothing at all. Give it two ends: a reference "
                "that resolves to a pair, or the two bounds written out as [<lo>, <hi>]."
            )
        if any(end is None for end in operand):
            return (
                f"the 'between' predicate's value {written!r} resolved to {list(operand)!r}, a "
                "pair with a missing end — a null bound excludes every element. Produce both "
                "bounds before comparing against them, or use 'lt'/'gt' against the one that "
                "exists."
            )
        return None
    if op in _ORDERED_OPS and (operand is None or isinstance(operand, (dict, list, tuple))):
        return (
            f"the {op!r} predicate's value {written!r} is {_shape_of(operand)}, not a single "
            f"value to compare against — {op!r} then fails for every element and the filter "
            "keeps nothing at all. Add a 'path' naming the field that holds the threshold, or "
            "compute the threshold in an earlier step and reference that binding."
        )
    return None


# Mirrors sora.action's predicate grammar — the evaluator there walks these, the resolver here
# fills them in. Kept as constants on both sides rather than imported, for the same reason
# `_REF_DECIDE` is: the two modules share a wire format, not an implementation.
_COMPOSE_ALL = "all"
_COMPOSE_ANY = "any"
_OP_OVERLAPS = "overlaps"


def _composition_defect(key: str, clauses: Any) -> str | None:
    """Why an ``all``/``any`` composition cannot be walked at all.

    Both failures are silent and both are dangerous in the same direction. A non-list has no
    clauses to evaluate; an EMPTY list is worse, because an empty conjunction is vacuously *true*
    in logic — written naively it would keep the whole collection, and a filter's output is
    routinely fanned out into one external action per item. The evaluator therefore matches nothing
    for either, and this reports it rather than let the step write an empty binding that reads
    downstream as a fact about the world."""
    if not isinstance(clauses, list):
        return (
            f"the {key!r} predicate's clauses are {_shape_of(clauses)}, not a list of predicates "
            f"to combine — nothing can be evaluated and the filter keeps nothing at all. Write "
            f"{key!r} as a list, each entry its own clause."
        )
    if not clauses:
        return (
            f"the {key!r} predicate has an empty clause list, so it selects nothing. If there is "
            "only one condition, write that comparison directly instead of composing it; if a "
            "clause was meant to come from an earlier step, produce it first."
        )
    return None


def _resolve_overlaps_against(
    where: dict[str, Any],
    history: list[CompletedOperation],
    bindings: dict[str, Any],
    properties: dict[tuple[str, str], Percept] | None = None,
) -> tuple[Any, str | None]:
    """Resolve an ``overlaps`` clause's ``against`` into plain ``[start, end]`` pairs.

    The third projection shape, alongside the membership set and the bare operand — and the reason
    ``overlaps`` needs no per-member alias grammar. ``against`` names a *collection*, so it resolves
    once here exactly as an ``in`` set does, and the two paths that read each member's interval are
    applied at resolution time. What reaches ``_matches`` is a literal list of pairs, which keeps
    the "resolve once in Reason, compare literals in the evaluator" invariant that every other op
    already relies on; deferring a reference to evaluation time instead would need a second
    resolution regime for one op's benefit.

    An unreadable ``against`` fails *closed* (nothing overlaps nothing), so it is reported rather
    than swallowed, like every other operand here."""
    against = where.get("against")
    if _is_reference(against):
        if _REF_DECIDE in against:
            return where, (
                f"an 'overlaps' predicate's 'against' cannot be a $decide reference "
                f"({against[_REF_DECIDE]!r}) — the comparison runs mechanically, so the intervals "
                "have to be known before it. Produce that collection in an earlier step and "
                "reference the binding it wrote."
            )
        resolved_members, defect = _resolve_collection(against, history, bindings, properties)
        if defect is not None:
            return where, defect
        members = resolved_members or []
    elif isinstance(against, list):
        members = against
    else:
        return where, (
            f"the 'overlaps' predicate's 'against' is {_shape_of(against)}, not a collection of "
            "intervals to compare with — the comparison then fails for every element and the "
            "filter keeps nothing at all. Give it a reference to the collection whose items carry "
            "the other interval, plus 'against_start_path'/'against_end_path' naming its two ends."
        )
    start_path = where.get("against_start_path", "")
    end_path = where.get("against_end_path", "")
    projected = [[pluck(m, start_path), pluck(m, end_path)] for m in members]
    # The sibling of the membership-set warning below: a member whose ends do not project to
    # comparable values can never overlap anything, so it drops out of the comparison silently and
    # the filter quietly narrows more than the plan asked. The usual cause is a path naming a field
    # the records don't carry (-> None) or pointing at a nested object. An empty collection stays
    # silent — `_resolve_collection` already logged why, and genuinely nothing-was-added is a real
    # and common answer here.
    if members and any(
        end is None or isinstance(end, (dict, list)) for pair in projected for end in pair
    ):
        bad = sum(
            1 for pair in projected if any(e is None or isinstance(e, (dict, list)) for e in pair)
        )
        log.warning(
            "filter: 'overlaps' against-collection has %d/%d member(s) with an end that projected "
            "to a non-scalar or None (against_start_path=%r, against_end_path=%r); those can never "
            "overlap anything — check the two paths",
            bad,
            len(projected),
            start_path,
            end_path,
        )
    resolved = {
        k: v for k, v in where.items() if k not in ("against_start_path", "against_end_path")
    }
    resolved["against"] = projected
    return resolved, None


def _resolve_predicate_value(
    params: dict[str, Any],
    history: list[CompletedOperation],
    bindings: dict[str, Any],
    properties: dict[tuple[str, str], Percept] | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Resolve a ``filter`` predicate's ``value`` when it is a reference rather than a literal, so
    ``_matches`` stays a pure literal comparison and the resolution lives here in Reason, next to
    the ``in``-collection resolution (ADR-0023 extension). Two shapes, because the ops read
    ``value`` differently:

    * ``in``/``not_in`` compare against a *set*: the reference resolves as a collection and is
      projected by ``value_path`` (default: the elements themselves, for a reference that already
      resolves to a list of scalars) into a list of comparable keys.
    * every other op compares against the value *itself*: the reference resolves to whatever it
      names — a scalar for ``eq``/``ne``/``lt``/``le``/``gt``/``ge``, the pair for ``between`` —
      and is **not** projected, since here the value is the operand rather than a collection to
      key on. This is the threshold shape ADR-0023's own reduce-then-compare pipeline is written
      in, and it was unreachable while resolution was gated on the membership ops: the raw
      reference dict reached ``_matches``, every comparison against it raised ``TypeError``, that
      was caught as a non-match, and the filter silently kept *nothing*.

    A reference may also sit *inside* a list operand rather than being the whole of it — the only
    way to write a ``between`` whose two ends come from two different steps — so those are resolved
    element-wise first (``_resolve_operand_items``); a list is not a reference, so without that the
    pair reached ``_matches`` with its reference dicts intact and kept nothing.

    A whole-predicate ``$decide`` (which ``FilterAction`` escalates intact) passes through
    untouched. A literal ``value`` has nothing to resolve, but is still shape-checked like a
    resolved one: whether an uncomparable operand was written out or arrived through a reference
    makes no difference to the filter it kills. Returns the resolved params and a ``defect``, set
    when the reference could not be read, or when it read cleanly but landed on a shape the op
    cannot compare against (``_operand_defect``) — reported rather than swallowed for the same
    reason the fan-out reports it:
    an unreadable predicate value fails *open* in some direction for every op (``in`` matches
    nothing, ``not_in`` keeps everything, an unreadable threshold excludes everything), so the
    filter confidently does the wrong thing to the whole collection. A resolved-but-unusable
    projection (a ``value_path`` that plucks to ``None`` or a non-scalar for every member) is the
    neighbouring trap the warning below guards, and fails open the same way."""
    where = params.get("where")
    if not isinstance(where, dict):
        return params, None  # no predicate
    resolved, defect = _resolve_predicate_clause(where, history, bindings, properties)
    if defect is not None:
        return params, defect
    return ({**params, "where": resolved} if resolved is not where else params), None


def _resolve_predicate_clause(
    where: Any,
    history: list[CompletedOperation],
    bindings: dict[str, Any],
    properties: dict[tuple[str, str], Percept] | None = None,
) -> tuple[Any, str | None]:
    """One clause of a predicate, resolved — the recursive worker behind
    ``_resolve_predicate_value``, and where that function's documented shapes are actually applied.

    Recursive because a predicate composes. ``all``/``any`` hold child clauses, each resolving its
    own operand exactly as a lone clause would, and a defect anywhere is a defect for the *whole*
    predicate: a conjunction with one dead clause selects nothing, and a disjunction with one
    silently drops whatever that clause was meant to catch. Composition resolves nothing itself —
    it is structure the evaluator walks — so all that is checked of it here is that it can be
    walked."""
    if not isinstance(where, dict):
        return where, (
            f"a composed predicate contains {_shape_of(where)}, not an object describing a "
            "predicate clause — the malformed clause selects nothing. Write each 'all'/'any' "
            "entry as its own predicate object."
        )
    if _REF_DECIDE in where:
        return where, None  # a soft clause is escalated whole rather than resolved mechanically
    for key in (_COMPOSE_ALL, _COMPOSE_ANY):
        if key not in where:
            continue
        clauses = where[key]
        if (composition_defect := _composition_defect(key, clauses)) is not None:
            return where, composition_defect
        resolved_clauses: list[Any] = []
        for clause in clauses:
            got, clause_defect = _resolve_predicate_clause(clause, history, bindings, properties)
            if clause_defect is not None:
                return where, clause_defect
            resolved_clauses.append(got)
        return {**where, key: resolved_clauses}, None
    if where.get("op") == _OP_OVERLAPS:
        return _resolve_overlaps_against(where, history, bindings, properties)
    written = where.get("value")  # what the plan wrote, kept for the defect messages
    value: Any = written
    if isinstance(value, (list, tuple)) and any(_is_reference(v) for v in value):
        items, list_defect = _resolve_operand_items(value, history, bindings, properties)
        if list_defect is not None:
            return where, list_defect
        where = {**where, "value": items}
        value = items
    if not _is_reference(value):
        # A literal, or a list whose members just resolved: nothing left to resolve, but the shape
        # still has to be one the op can compare against — an uncomparable operand is as dead
        # written out as it is referenced.
        return where, _operand_defect(where, written, value)
    if _REF_DECIDE in value:
        # A soft reference in the *operand* position, which no prompt documents and no op can
        # evaluate: the comparison itself is mechanical. It used to resolve to an empty membership
        # set (silently fails open) or a raw dict (silently excludes everything), so say what to
        # write instead rather than let either happen.
        return where, (
            f"a filter predicate's 'value' cannot be a $decide reference "
            f"({value[_REF_DECIDE]!r}) — the comparison runs mechanically, so the operand has to "
            "be known before it. Compute it in an earlier step and reference that binding, or "
            "make the whole 'where' a $decide predicate."
        )
    if where.get("op") not in ("in", "not_in"):
        try:
            operand: Any = _resolve_ref(value, history, bindings, properties)
        except (KeyError, IndexError, TypeError, ValueError):
            return where, _path_defect(value, history, bindings, properties)
        if operand is _MISSING:
            return where, _collection_defect(value, operand, history, properties)
        if (shape_defect := _operand_defect(where, written, operand)) is not None:
            return where, shape_defect
        return {**where, "value": operand}, None
    resolved_members, defect = _resolve_collection(value, history, bindings, properties)
    if defect is not None:
        return where, defect
    members = resolved_members or []
    projected = [pluck(m, where.get("value_path", "")) for m in members]
    # A membership set is compared element-by-element against a scalar key, so only scalar members
    # can ever match. A non-scalar (dict/list) or ``None`` projection is dead weight: `in` silently
    # drops it, `not_in` silently keeps it. The usual cause is a `value_path` that's missing (the
    # referenced collection is records, not bare keys), wrong (names a field the records don't
    # carry -> None), or points at a nested object — surface any such member rather than let the
    # filter fail open invisibly (an all-None projection is exactly the duplicate-action trap:
    # "not already saved" keeps everything). An empty set (nothing resolved) stays silent here —
    # _resolve_collection already logged *why*, and a genuinely empty exclusion list is benign.
    if members and any(p is None or isinstance(p, (dict, list)) for p in projected):
        bad = sum(1 for p in projected if p is None or isinstance(p, (dict, list)))
        log.warning(
            "filter: membership set for %r has %d/%d member(s) that projected to a non-scalar or "
            "None key (value_path=%r); those can never match — `in` drops them, `not_in` keeps "
            "them — check `value_path`",
            where.get("path"),
            bad,
            len(projected),
            where.get("value_path"),
        )
    resolved_where = {k: v for k, v in where.items() if k != "value_path"}
    resolved_where["value"] = projected
    return resolved_where, None


def _substitute_bindings(obj: Any, name: str, element: Any) -> Any:
    """Replace every ``{"$bind": name, "path": ...}`` in a template with the value at that path of
    the current loop ``element``, recursively. Only the named binding is substituted;
    ``$from``/``$decide`` references (and a ``$bind`` for a different name) pass through untouched,
    to be grounded later by the ordinary Reason path. A path that doesn't resolve substitutes a
    ``None`` — which the Act required-param guard skips, not a literal ``$bind`` dict reaching the
    tool."""
    if isinstance(obj, dict):
        if obj.get(_REF_BIND) == name:
            try:
                return _walk_path(element, obj.get(_REF_PATH, ""))
            except (KeyError, IndexError, TypeError, ValueError):
                log.warning("subgoal: $bind path %r did not resolve against %r", obj, element)
                return None
        return {k: _substitute_bindings(v, name, element) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_bindings(v, name, element) for v in obj]
    return obj


def _expand_mechanical(
    step: Step,
    history: list[CompletedOperation],
    bindings: dict[str, Any] | None = None,
    properties: dict[tuple[str, str], Percept] | None = None,
) -> tuple[list[Step], str | None]:
    """Fan a mechanical sub-goal out to one concrete ``Step`` per element of its ``in`` collection,
    the element substituted for ``{"$bind": "<as>"}`` in its ``template``. The ``in`` collection may
    be a ``$from`` (history), a ``$bind`` (a data-op output binding, e.g. a filtered shortlist), or
    a ``$prop`` (bulk state an adapter publishes as an observable property).

    Returns the expansion and a ``defect``. An empty collection expands to no steps and *is* the
    answer — the sub-goal had nothing to do and the plan should continue. A collection that could
    not be read expands to no steps too, but means the opposite, so it comes back as a defect for
    the caller to replan on. Collapsing the two is how "cancel each event on Saturday" quietly
    became a no-op in a real run while the event sat in history, correctly fetched, all along."""
    elements, defect = _resolve_collection(step.params.get("in"), history, bindings, properties)
    if defect is not None:
        return [], defect
    if not elements:
        return [], None
    loop_var = step.params.get("as", "")
    template = step.params.get("template", {})
    return [
        step_from_raw(_substitute_bindings(template, loop_var, element)) for element in elements
    ], None


class DefaultReasonStrategy:
    """The runtime's Reason default. Reason is the one phase with no *mechanical* default —
    planning inherently needs a model — so this is deterministic orchestration around the single
    model call, which is isolated in ``ProceduralMemory.infer``:

    * an activity that already has a plan with steps left is the cheap path — read the current step
      and advance ``step_index``: no model call, no procedural lookup;
    * an activity with no plan gets one by *reuse* first (``procedural.retrieve``) and only *infers*
      a fresh one on a miss — firing the ``_infer_`` internal action, which moves the activity to
      RUNNING and yields no step this cycle; the plan lands a later cycle via ``inference_sink`` and
      Reason then advances it. Reuse is currently always a miss — the default Reflect no longer
      stores completed plans (auto-caching a plan for verbatim replay is unsound), so every activity
      infers until reusable procedures are distilled from episodes. Infer passes the currently-
      joined tools (id -> Manual) as the planning catalog and a ``PerceptSnapshot`` of
      ``wm.properties``/``wm.signals`` as the agent's known world state (the strategy holds `wm`; a
      memory module never reaches into the environment or working memory itself — it only sees
      what's extracted and handed to it);
    * an exhausted plan yields no step — the cycle returns, and Reflect terminates the activity the
      next cycle on the same "plan present and fully consumed" rule (so this branch is normally only
      reached by a just-inferred empty plan).

    Mutates the activity (plan/step_index) in place, like the other phase defaults. Both model calls
    run **off-cycle** as the ``_infer_``/``_ground_`` internal actions (isolated in
    ``ProceduralMemory.infer``/``ground``): the cheap path — advancing an existing plan's step_index
    and mechanically resolving its params — makes zero model calls and stays same-cycle; only the
    escalations park the activity in RUNNING and resolve a later cycle. No phase blocks on a model
    call, so there is nothing to race or abandon (ADR-0021)."""

    def __init__(
        self,
        max_subgoal_depth: int = _DEFAULT_MAX_SUBGOAL_DEPTH,
        max_replan_attempts: int = _DEFAULT_MAX_REPLAN_ATTEMPTS,
    ) -> None:
        # Depth cap for the deliberative sub-goal breaker; wired from agent.yaml's
        # `max_subgoal_depth` so a legitimately deep task can raise it past the default.
        self._max_subgoal_depth = max_subgoal_depth
        # Backstop count for the replanning breaker, from agent.yaml's `max_replan_attempts`.
        self._max_replan_attempts = max_replan_attempts

    async def reason(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        # Pending conditions come first: a condition that has fired redirects the activity, so
        # deciding that before advancing the body is what lets a reply that beats the last step be
        # acted on rather than discovered after the fact (ADR-0022).
        if activity.condition_verdict is not None:
            return await self._apply_condition_verdict(activity, wm, cycle, result)
        if activity.condition_fired and _body_exhausted(activity):
            # A verdict that fired several conditions pursued the first and queued the rest. Drain
            # the queue before judging anything new: this work has already been judged and paid for.
            # Gated on an idle intention stack so each `then` runs after its predecessor rather than
            # nested inside it.
            return await self._pursue_fired_condition(activity, wm, cycle, result)
        if activity.pending_conditions:
            fired = await self._evaluate_pending_conditions(activity, wm, cycle, result)
            if fired is not None:
                return fired
        if activity.plan is None:
            halt = self._replanning_would_loop(activity)
            if halt is not None:
                # Refuse to plan again: pause for guidance rather than spend another planning
                # inference (and another...) on an activity that is not getting anywhere. Gated
                # ahead of the procedural retrieve as well as the inference — a cached plan is
                # exactly as stuck as a synthesized one when it is the plan that just failed. Set
                # BLOCKED directly, as the sub-goal breaker and the interrupt handler do, and with
                # a mechanically rendered prompt for the same reason they use one.
                log.warning("reason: halting replanning for activity %s: %s", activity.id, halt)
                await _await_input(cycle, activity, _replan_halt_prompt(activity, halt))
                return result
            plan = await cycle.procedural.retrieve(activity)  # reuse across runs (cheap)
            if plan is None:
                # Miss -> fire _infer_ off-cycle: it moves the activity to RUNNING and returns at
                # once, so the cycle never blocks on the model. The plan lands a later cycle via
                # inference_sink (Observe attaches it and resets step_index); this Reason yields no
                # step now. Situate won't reselect a RUNNING activity, so no re-fire meanwhile.
                catalog = {tool.id: tool.manual for tool in wm.registry.all_tools()}
                observed = PerceptSnapshot(list(wm.properties.values()), list(wm.signals))
                infer = cycle.actions.internal(InferAction.name)
                await infer.execute(
                    cycle,
                    activity_id=activity.id,
                    tools=catalog,
                    observed=observed,
                    messages=list(wm.messages),  # recent user instructions, snapshot at fire time
                    # The world the plan is being inferred against, so the context-adaptation gate
                    # baselines against it once the plan installs (ADR-0024) — not a later cycle's
                    # already-drifted perception. Via the pluggable ChangeGate so the baseline and
                    # every later comparison share one signature space.
                    baseline=cycle.change_gate.signature(wm),
                )
                return result  # RUNNING on the inference; plan lands a later cycle
            log.info(
                "reason: reusing cached plan (%d steps) for %r", len(plan.steps), activity.goal
            )
            log.debug("reason: cached plan for activity %s\n%s", activity.id, render_plan(plan))
            activity.plan = plan
            activity.step_index = 0
            activity.history_mark = len(activity.history)
            # A cached plan installs without an inference, so nothing consumed a parked superseded
            # bundle (ADR-0024) — drop it here too, or it outlives the re-plan it was parked for.
            activity.superseded = None
        while True:
            plan = activity.plan
            assert plan is not None  # set above, and by every branch that continues this loop
            if activity.step_index >= len(plan.steps):
                if activity.parent_frames and not _conditions_hold_frame(activity):
                    # Sub-plan exhausted: pop the frame and resume the parent at the step *after*
                    # its sub-goal, then loop to read it (or pop again if that frame is exhausted).
                    parent_plan, parent_index, parent_mark = activity.parent_frames.pop()
                    activity.plan = parent_plan
                    activity.step_index = parent_index + 1
                    activity.history_mark = parent_mark  # the parent collects over its own span
                    continue
                if activity.pending_conditions and not (
                    activity.condition_fired or activity.condition_verdict is not None
                ):
                    # Nothing left to run and nothing owed: wait on the watches. Reaching here with
                    # an open gate is impossible — the head of reason() fires the judgement first
                    # and returns — so this cannot re-block over a signal that is still unjudged.
                    activity.state = ActivityState.BLOCKED
                    activity.blocked_on = ConditionWait(watches=_condition_watches(activity))
                    log.info(
                        "reason: no body left on %s; waiting on %d pending condition(s)",
                        activity.id,
                        len(activity.pending_conditions),
                    )
                return result  # nothing more to run this cycle
            step = plan.steps[activity.step_index]
            if step.next_action == SUBGOAL:
                outcome = await self._subgoal(step, activity, wm, cycle)
                if outcome is _SUBGOAL_SPLICED:
                    continue  # mechanical: spliced the expansion in place -> read the first step
                # Deliberative fired _infer_ (RUNNING), the guard halted (BLOCKED), or the
                # collection was unreadable and the plan was dropped (READY, re-infers next cycle).
                return result
            if cycle.actions.is_data_op(step.next_action):
                # A data-op transforms a run-time value into a named binding (ADR-0023). Advance
                # past it either way: a mechanical op wrote its binding now (read the next step this
                # cycle); a $decide filter fired an off-cycle model call and the activity is now
                # RUNNING, its result landing in bindings[out] a later cycle — so on resume we
                # continue after the op, not re-run it (unlike _ground_, whose step still runs).
                parked = await self._data_op(step, activity, cycle)
                if activity.plan is None:
                    # The op's input was unreadable, so the plan was dropped and step_index
                    # rewound to 0 — advancing would carry a stale index into its replacement.
                    return result
                activity.step_index += 1
                if parked:
                    return result  # RUNNING on the select escalation; binding lands a later cycle
                continue
            # A write is the last moment the runtime can still decline to act, so before one it
            # asks whether the plan can still finish at all (_unsatisfiable_reference). Ahead of
            # grounding, unlike the reconsideration checkpoint below: this reads only settled state,
            # costs nothing, and there is no sense buying a grounding call for a dead plan. Not
            # routed through `cycle.reconsideration` deliberately — that policy is configurable and
            # may legitimately be switched off, whereas refusing to act on a plan that provably
            # cannot work is not a tuning knob.
            if _step_side_effecting(step, wm) is not False:
                dead = _unsatisfiable_reference(activity)
                if dead is not None:
                    log.warning("reason: plan defect for activity %s — %s", activity.id, dead)
                    activity.reset_for_replan(defect=dead)
                    return result  # no step this cycle; Reason re-infers against the current world
            # Ground first, *then* reconsider — the checkpoint guards the side-effecting commitment
            # (the invoke), not the grounding before it (ADR-0024). Grounding is itself an
            # off-cycle, side-effect-free model call, so checking before it would (a) spend a
            # revalidation to maybe save a grounding — a model call to save a model call,
            # net-negative when the plan still holds (the common case) — and (b) miss a change that
            # lands *during* the grounding window (grounding read its world at dispatch), where a
            # slow grounding call is most exposed. So each step is checked once, just before commit.
            grounded = await self._ground(step, activity, wm, cycle)
            if grounded is None:
                # Either escalated via _ground_ (RUNNING; checked on the invoke pass later), or the
                # plan was dropped as defective (READY, plan cleared -> re-infers next cycle).
                # Neither commits a step this cycle, which is all this site needs to know.
                return result
            checkpoint = await self._reconsider(step, activity, wm, cycle, result)
            if checkpoint is not None:
                # Fired a revalidation (RUNNING), or the plan was invalidated (reset_for_replan) —
                # either way no step commits this cycle. The grounded params stay parked (see
                # _resolve) so the next cycle re-emits this step without a second _ground_ call.
                return checkpoint
            # Commit: consume the parked grounding now that the step dispatches. Advance only once a
            # concrete step is emitted, so a step awaiting its grounding escalation isn't skipped —
            # step_index stays put across RUNNING cycles until it lands.
            activity.grounded_params = None
            activity.step_index += 1
            return replace(result, activity=activity, step=grounded)

    async def _reconsider(
        self,
        step: Step,
        activity: Activity,
        wm: WorkingMemory,
        cycle: DecisionCycle,
        result: TickResult,
    ) -> TickResult | None:
        """Context-adaptation checkpoint (ADR-0024), run just before committing an external step.
        Returns a ``TickResult`` to yield (a revalidation was fired -> RUNNING, or the plan was
        invalidated -> re-infer) or ``None`` to proceed and emit the step. Two-tier: a resolved
        verdict from a prior cycle's revalidation is consumed first; otherwise a cheap mechanical
        gate (did perception move since the plan was baselined?) decides whether to spend one —
        so a static world costs zero model calls."""
        # 1) A prior cycle's revalidation resolved: act on its verdict.
        if activity.reconsider_verdict is not None:
            valid = activity.reconsider_verdict
            activity.reconsider_verdict = None
            if not valid:
                log.info("reason: plan invalidated by context-adaptation for %r", activity.goal)
                activity.reset_for_replan()  # -> re-infer next cycle against the current world
                # A moving world no longer counts toward the replan breaker (see
                # _replanning_would_loop), so this is the only place the pile-up shows: the agent is
                # re-planning honestly each time and still never reaching its first write.
                churn = sum(1 for d in activity.replan_trail if d is None)
                if churn >= self._max_replan_attempts:
                    log.warning(
                        "reason: %d consecutive plans for activity %s invalidated by a moving "
                        "world with no operation run — the world may be changing faster than the "
                        "agent can commit to it",
                        churn,
                        activity.id,
                    )
                # Trace what was dropped, from the bundle the reset parked (ADR-0024). Every frame's
                # *whole* body, not the un-run tail the replanning prompt gets: a prompt pays per
                # token and separately receives what already ran as history, whereas this is read by
                # a human diffing it against the replacement, which installs logged whole. Suspended
                # parents included — the reset drops the entire intention stack, so a discard taken
                # inside a sub-plan throws away more than the frame in hand.
                if activity.superseded is not None:
                    sup = activity.superseded
                    bodies = [render_steps(sup.plan.steps)]
                    bodies += [
                        f"-- suspended parent, at sub-goal step {i} --\n{render_steps(p.steps)}"
                        for p, i in reversed(sup.parent_frames)
                    ]
                    log.debug(
                        "reason: discarded plan for activity %s (was at step %d)\n%s",
                        activity.id,
                        sup.step_index,
                        "\n".join(bodies),
                    )
                return result
            # Valid: proceed. The baseline was already advanced by Observe to the world the re-check
            # was fired against (not now) — so a change that landed during its flight stays outside
            # the baseline and is reconsidered on the next write rather than folded in here.
            return None
        # 2) Anchor the baseline if unset. An inferred plan already carries its infer-time baseline
        # (installed by Observe); this is the fallback for a *reused* plan, which has no fresh
        # inference. Done before the policy gate so even a pre-write read prefix anchors the
        # reference the first write compares against (entry-time, not first-write-time).
        if activity.reconsider_baseline is None:
            activity.reconsider_baseline = cycle.change_gate.signature(wm)
        # 3) Does the policy want a check before this step?
        if not cycle.reconsideration.should_check(_step_side_effecting(step, wm)):
            return None
        # 4) Cheap mechanical gate: has anything observable moved since the plan was baselined?
        current = cycle.change_gate.signature(wm)
        if current == activity.reconsider_baseline:
            return None  # nothing moved -> proceed (free when the world is static)
        # 5) Gate hot: fire the revalidation off-cycle (RUNNING); the verdict lands a later cycle.
        # Carry the fire-time signature so Observe advances the baseline to *this* world on resolve:
        # a change landing mid-flight then earns its own reconsideration rather than being absorbed.
        # Agent-level on purpose, unlike ground/select: the gate above fires on ANY
        # perception change, so scoping this to the plan's own tools would hand the judge a world
        # in which the change that woke it is invisible — it would revalidate against nothing.
        observed = PerceptSnapshot(list(wm.properties.values()), list(wm.signals))
        revalidate = cycle.actions.internal(RevalidateAction.name)
        await revalidate.execute(
            cycle,
            activity_id=activity.id,
            observed=observed,
            messages=list(wm.messages),
            baseline=current,
        )
        return result

    @staticmethod
    async def _data_op(step: Step, activity: Activity, cycle: DecisionCycle) -> bool:
        """Execute one data-op step (ADR-0023) and report whether it *parked* the activity on an
        off-cycle model call (only a ``$decide`` filter does). Resolves the op's input collection
        here — a ``collect`` gathers the per-element results of a fanned-out operation straight from
        ``history`` (its ``from`` is an operation name, not an ``in`` reference), each result
        carrying its invoking params so a downstream op can correlate it to its input (a ``collect``
        whose operation ran only *before* this plan's span is a plan defect, not an empty result);
        every other op resolves ``in`` from history/bindings, an unresolvable input becoming an
        empty collection (the fan-out's never-raise contract). Dispatches from the data-op bucket;
        the op writes its result into ``activity.bindings[out]`` (or, for the escalation, Observe
        does so later)."""
        if step.next_action == CollectAction.name:
            op_name = step.params.get("from")
            # Scoped to the active frame's span (Activity.history_mark): collect takes EVERY match,
            # so an unscoped read accumulates results from plans that already finished — a sub-plan
            # collecting its parent's stale calendar query alongside its own, then deleting over it.
            collection: list[Any] = [
                _enrich_with_params(completed.ack.result, completed.invocation.params)
                for completed in activity.history[activity.history_mark :]
                if completed.invocation.operation_name == op_name
            ]
            if not collection and any(
                completed.invocation.operation_name == op_name
                for completed in activity.history[: activity.history_mark]
            ):
                # The operation ran, but only OUTSIDE this plan's span — under a plan that has since
                # been replaced. The mark stays where it is (results a discarded plan gathered may
                # no longer hold, which is usually *why* it was discarded), but the planner is shown
                # the full history, so writing this collect is a reasonable mistake to make. Say so
                # precisely: left alone, the empty binding surfaces a step later as the generic
                # "an earlier step produced EMPTY — nothing matched", which reads as a fact about
                # the world, and the planner re-plans into the same collect on that false premise.
                defect: str | None = (
                    f"the {step.next_action!r} step reads {op_name!r}, which ran only under a "
                    "PREVIOUS plan that has since been replaced — a collect reaches only the runs "
                    "its own plan performs, so this yields nothing. Invoke it again in this plan "
                    f"if its results are needed. {_REPLAN_HINT}"
                )
                log.warning("reason: plan defect for activity %s — %s", activity.id, defect)
                activity.reset_for_replan(defect=defect)
                return True  # no step this cycle; Reason re-infers against the current world
            passthrough = {k: v for k, v in step.params.items() if k != "from"}
        else:
            resolved, defect = _resolve_collection(
                step.params.get("in"),
                activity.history,
                activity.bindings,
                cycle.working.properties,
            )
            passthrough = {k: v for k, v in step.params.items() if k != "in"}
            if defect is None and step.next_action == FilterAction.name:
                passthrough, defect = _resolve_predicate_value(
                    passthrough, activity.history, activity.bindings, cycle.working.properties
                )
            if defect is not None:
                # Same reason as the fan-out: a data-op over an unreadable input would write an
                # empty binding, and an empty binding reads downstream as a real answer ("no such
                # contact") rather than as a question. Replan on it instead.
                defect = f"data-op {step.next_action!r} could not read its input: {defect}"
                log.warning("reason: plan defect for activity %s — %s", activity.id, defect)
                activity.reset_for_replan(defect=defect)
                return True  # no step this cycle; Reason re-infers against the current world
            collection = resolved if resolved is not None else []
        op = cycle.actions.data_op(step.next_action)
        # `observed` is only read by the $decide filter's escalation (the others are mechanical and
        # ignore it), but it is passed uniformly rather than branched on: a soft predicate resolves
        # its references against the same world grounding does, and deciding per-op which context a
        # model call may see is how the two drifted apart in the first place.
        await op.execute(
            cycle,
            activity_id=activity.id,
            collection=collection,
            observed=scoped_snapshot(cycle.working, activity),
            **passthrough,
        )
        return activity.pending_inference is not None

    async def _evaluate_pending_conditions(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult | None:
        """Fire the batched condition judgement if any gate has opened on an unjudged signal.

        Returns the threaded result when a call was fired (no step this cycle — the activity is
        RUNNING on the inference), or None to let Reason carry on with the body. The common case is
        None: gates are mechanical and narrow, so most cycles nothing is eligible and this costs a
        list comprehension.
        """
        eligible = _eligible_conditions(activity, wm)
        if not eligible:
            return None
        # Advance every judged condition's mark NOW, at fire time rather than on resolve. A signal
        # that lands while the call is in flight gets a higher sequence number and so earns its own
        # evaluation later; advancing on resolve instead would either re-judge the same signal (a
        # spin) or swallow one that arrived mid-flight.
        # Keeping where they stood is what lets a judgement that ERRORS give the change back
        # (see `PendingConditionState.fired_from_signals`); a call that answers never reads them.
        for state, percept in eligible:
            state.fired_from_signals = state.evaluated_through
            state.fired_from_derived = state.derived_through
            state.evaluated_through = wm.signals_appended
            state.derived_through = wm.property_changes_appended
            # Keep the change that opened this gate, for the `then` plan to reference mechanically
            # (see `_pursue_fired_condition`). Recorded here because this is the last moment it is
            # in hand: the judgement is off-cycle, and by the time its verdict is applied the tick
            # that carried the change has passed.
            state.fired_changes = tuple(
                (percept.source, change) for change in changes_of(percept.payload)
            )
        # Paired with the source that reported them, not flattened into a bare list: a `Change`
        # names the path that moved but not the tool it moved on, and the judgement needs both to
        # dereference the ids back into records (see ProceduralMemory.render_changes).
        changes: list[tuple[str, Change]] = []
        for _, percept in eligible:
            changes.extend((percept.source, change) for change in changes_of(percept.payload))
        observed = PerceptSnapshot(list(wm.properties.values()), list(wm.signals))
        evaluate = cycle.actions.internal(EvaluateConditionsAction.name)
        await evaluate.execute(
            cycle,
            activity_id=activity.id,
            conditions=[state.condition for state, _ in eligible],
            changes=changes,
            observed=observed,
        )
        activity.condition_batch = [state for state, _ in eligible]
        return result

    async def _apply_condition_verdict(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        """Consume a resolved condition verdict: retire what is done, pursue what fired.

        A fired condition's `then` is pursued through the ordinary deliberative sub-goal path — it
        is a goal, planned fresh when the moment comes, which is the whole reason `then` is prose
        rather than steps. That reuse is also why a fired condition costs a second call and a
        non-firing one costs none: the judgement and the planning stay separate concerns.

        Firing is recorded the moment the verdict lands, but pursued only once the body is idle
        (the same rule the queued-fire path in `reason` applies). A `then` that preempts an
        unfinished plan abandons every step it never reached: on the run that motivated this, a
        four-step fan-out of deletes had executed one when that delete opened the gate it was
        watching, and the remaining three never ran. Nothing is lost by waiting — the fire sits in
        `condition_fired` and Reflect keeps the activity READY for it (see its own comment).
        """
        verdict = activity.condition_verdict or ConditionVerdict()
        activity.condition_verdict = None
        judged = activity.condition_batch
        activity.condition_batch = []
        # Identity, not equality: PendingConditionState is mutable (its mark advances), so it is
        # unhashable, and two conditions can compare equal while being distinct waiters.
        retired = {id(judged[i]) for i in verdict.retired if i < len(judged)}
        if retired:
            activity.pending_conditions = [
                state for state in activity.pending_conditions if id(state) not in retired
            ]
            # Remember WHAT retired, not just that something did: the declaration survives on the
            # frozen Plan.pending, so the next lift would otherwise put it straight back on watch.
            activity.retired_conditions.update(
                judged[i].condition for i in verdict.retired if i < len(judged)
            )
            log.info("reason: retired %d pending condition(s) on %s", len(retired), activity.id)
        fired = [
            ConditionFiring(
                condition=judged[i].condition,
                fired_changes=judged[i].fired_changes,
            )
            for i in verdict.fired
            if i < len(judged) and id(judged[i]) not in retired
        ]
        # Queue every fire, pursue one. The verdict is plural on purpose — one call judges the whole
        # eligible batch, and a single reply can satisfy two gates — so keeping only the first would
        # discard judgements already paid for and unrecoverable: the marks advanced at fire time, so
        # the signal that opened the other gates can no longer make them eligible.
        activity.condition_fired.extend(fired)
        if activity.condition_fired and _body_exhausted(activity):
            return await self._pursue_fired_condition(activity, wm, cycle, result)
        # Nothing fired. If the body is finished, go back to waiting rather than falling through to
        # Reflect, which would otherwise see an exhausted plan and terminate the activity outright.
        if activity.pending_conditions and _body_exhausted(activity):
            activity.state = ActivityState.BLOCKED
            activity.blocked_on = ConditionWait(watches=_condition_watches(activity))
            log.info("reason: no pending condition fired on %s; waiting again", activity.id)
        return result

    async def _pursue_fired_condition(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        """Take the oldest queued fire and pursue its `then` as a deliberative sub-goal.

        One at a time, in the order the verdict listed them: each `then` is a goal in its own right,
        and pursuing a second while the first one's sub-plan is still running would nest it inside
        its predecessor — inverting the order and walking the activity toward the recursion breaker
        for reasons that have nothing to do with the goals themselves. The queue is therefore only
        drained when the intention stack is otherwise idle (both callers check `_body_exhausted`).

        `from_condition` marks the step as a `then` rather than an authored sub-goal, which changes
        two things downstream: it is exempt from the ancestor-overlap breaker, and it installs
        without pushing a frame (see `_subgoal`).

        The firing's own change is seeded into `SEEDED_BINDINGS` first, so the `then` plan can name
        the ids that just moved *mechanically*. Without it the runtime holds the answer — the watch
        gate matched on those very ids — and offers the planner no way to reach it: the reference
        grammar addresses history, bindings and properties, none of which is a change. What a plan
        wrote instead was a `$decide` filter re-deriving the added set from the whole property, one
        model call over every record in the collection to recover a set the signal had already
        reported verbatim. On the run that motivated this that was four of nine such calls and
        about a third of the entire run's input tokens, for set membership.
        """
        state = activity.condition_fired.pop(0)
        log.info("reason: pending condition fired on %s -> %r", activity.id, state.condition.then)
        # Replace the previous firing's ids even when this firing cannot supply new ones. A coarse
        # Change means "something moved, ids and direction unknown", so turning its three empty
        # tuples into authoritative empty sets would silently make `in` select nothing and
        # `not_in` exclude nothing. Since these flattened bindings cannot mark a precise subset as
        # incomplete, one coarse member makes the whole firing unavailable. Otherwise seed all
        # three kinds, including genuinely known-empty directions, ordered and de-duplicated.
        for name in SEEDED_BINDINGS:
            activity.bindings.pop(name, None)
        changes_are_precise = bool(state.fired_changes) and all(
            change.added or change.removed or change.updated
            for _source, change in state.fired_changes
        )
        if changes_are_precise:
            seeded: dict[str, list[Any]] = {name: [] for name in SEEDED_BINDINGS}
            for _source, change in state.fired_changes:
                for name, moved in zip(
                    SEEDED_BINDINGS, (change.added, change.removed, change.updated), strict=True
                ):
                    seeded[name].extend(m for m in moved if m not in seeded[name])
            activity.bindings.update(seeded)
        step = Step(
            next_action=SUBGOAL,
            params={
                "goal": state.condition.then,
                "mode": "deliberative",
                "from_condition": True,
            },
        )
        await self._subgoal(step, activity, wm, cycle)
        return result

    async def _subgoal(
        self, step: Step, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle
    ) -> object:
        """Handle a ``subgoal`` step (ADR-0022). **Deliberative** -> fire ``_infer_`` for the
        sub-goal's own goal (activity goes RUNNING; the sub-plan lands a later cycle and Observe
        enters it on a pushed frame), reporting ``_SUBGOAL_RUNNING``. **Mechanical** -> fan the
        sub-goal out over its collection and splice the expansion into the plan *in place* — a
        per-run copy via ``replace``, so the stored skeleton keeps its ``subgoal`` step — leaving
        ``step_index`` on the first expanded step and reporting ``_SUBGOAL_SPLICED`` so ``reason``
        re-reads it. An empty collection expands to nothing and the sub-goal simply vanishes; one
        that could not be *read* is a plan defect instead -> replan (``_SUBGOAL_DEFECT``)."""
        mode = step.params.get("mode", "deliberative")
        _lift_step_conditions(step, activity, wm)
        if mode == "deliberative":
            goal = step.params["goal"]
            # A `then` (see `_pursue_fired_condition`) restates the goal that declared it — the
            # planner is told to phrase it "like the original goal" — so containment in an ancestor
            # is its shape, not a failure to reduce, and the overlap check reads it as recursion
            # every time. Its reduction is in the data: it is planned against a change that did not
            # exist when the ancestor was. The depth cap still applies, as the coarse backstop.
            from_condition = bool(step.params.get("from_condition"))
            halt = self._deliberation_would_loop(activity, goal, check_overlap=not from_condition)
            if halt is not None:
                # Refuse to recurse: pause to await the user's guidance instead of spending another
                # (and another...) _infer_ on a sub-goal that isn't reducing. Set BLOCKED directly,
                # as the interrupt handler does for its InputWait — no _suspend_ (that's for a
                # manual-declared SignalWait), and no model call for the prompt (kept mechanical).
                log.warning(
                    "reason: halting sub-goal recursion for activity %s: %s", activity.id, halt
                )
                await _await_input(
                    cycle, activity, f"Stuck on {goal!r}: {halt}. How should I proceed?"
                )
                return _SUBGOAL_HALTED
            catalog = {tool.id: tool.manual for tool in wm.registry.all_tools()}
            observed = PerceptSnapshot(list(wm.properties.values()), list(wm.signals))
            infer = cycle.actions.internal(InferAction.name)
            await infer.execute(
                cycle,
                activity_id=activity.id,
                tools=catalog,
                observed=observed,
                messages=list(wm.messages),
                # "then" lands like a sub-plan but at the same depth (Observe pushes no frame).
                kind="then" if from_condition else "subgoal",
                goal=goal,
                # A sub-plan has its own assumptions; baseline the gate against the world it is
                # synthesized in, so entering a sub-goal re-anchors reconsideration (ADR-0024).
                baseline=cycle.change_gate.signature(wm),
            )
            return _SUBGOAL_RUNNING
        plan = activity.plan
        assert plan is not None  # reason() only dispatches a step off a set plan
        i = activity.step_index
        expanded, defect = _expand_mechanical(
            step, activity.history, activity.bindings, cycle.working.properties
        )
        if defect is not None:
            # Splicing in zero steps here would mean "this sub-goal had nothing to do", which is a
            # claim the runtime cannot make when it could not read the collection at all. Replan
            # carrying the reason, exactly as the undeclared-parameter check does: the planner gets
            # a correction it can act on, and a planner that writes the same bad reference twice
            # trips the replan breaker rather than looping.
            defect = f"sub-goal {step.params.get('goal')!r} could not be expanded: {defect}"
            log.warning("reason: plan defect for activity %s — %s", activity.id, defect)
            activity.reset_for_replan(defect=defect)
            return _SUBGOAL_DEFECT
        activity.plan = replace(plan, steps=plan.steps[:i] + expanded + plan.steps[i + 1 :])
        log.info(
            "reason: sub-goal %r fanned out to %d step(s)", step.params.get("goal"), len(expanded)
        )
        # The spliced body, not just its size: every later "act: invoke" and "was at step N"
        # indexes into *this* plan, so without it the trace's last plan is the pre-splice one.
        log.debug(
            "reason: plan for activity %s after fan-out (at step %d)\n%s",
            activity.id,
            i,
            render_steps(activity.plan.steps),
        )
        return _SUBGOAL_SPLICED

    def _replanning_would_loop(self, activity: Activity) -> str | None:
        """Whether inferring another plan for this activity now would be a loop rather than an
        attempt — the reason string if so (for the log and the await-input prompt), else ``None``.
        Read off ``Activity.replan_trail``, which accumulates only across replans that executed no
        operation, so adapting to a world that keeps moving is never what trips this.

        Both checks read only the *defect-bearing* entries. A defect-free entry is a
        reconsideration (ADR-0024): the plan was fine and the world moved under it, so the next plan
        is inferred against a genuinely different world and is an attempt, not a repeat. Counting
        those meant a plan whose first checkpointed step is a write — nothing has run yet, so
        nothing forgives the trail — halted the agent on an unanswerable question after five honest
        adaptations, in exactly the moving world this runtime exists for. A world that will not hold
        still long enough to commit a write is a livelock bounded by the operator's wall clock, not
        evidence of a stuck planner, so it is logged (see ``_reconsider``) rather than blocked on.

        Two mechanical checks. The precise one first: the replacement plan was abandoned for the
        *same* defect as the plan it replaced, meaning the planner was handed that defect in its
        brief and wrote past it anyway — a third attempt is not going to differ, so this trips at
        two. A reconsideration landing between the two does not excuse it: a structural complaint
        (a reference to nothing, a param the operation does not take) is wrong about the plan, not
        about the world, so a moved world makes repeating it no more forgivable. Then the coarse
        count, for floundering where each attempt at least fails differently."""
        defects = [d for d in activity.replan_trail if d is not None]
        if len(defects) >= 2 and defects[-1] == defects[-2]:
            return (
                "the replacement plan was abandoned for the same reason as the plan it "
                f"replaced ({defects[-1]})"
            )
        if len(defects) >= self._max_replan_attempts:
            return (
                f"{len(defects)} plans in a row were abandoned with a defect and without a single "
                f"operation running (>= {self._max_replan_attempts})"
            )
        return None

    def _deliberation_would_loop(
        self, activity: Activity, goal: str, *, check_overlap: bool = True
    ) -> str | None:
        """Whether firing a deliberative sub-goal for ``goal`` now would be runaway recursion rather
        than progress — the reason string if so (for the log and the await-input prompt), else
        ``None``. Two mechanical checks: the intention stack is already ``max_subgoal_depth`` deep,
        or ``goal``'s tokens are largely contained in a sub-goal still suspended above it (an
        elaborated re-statement, not a reduction).

        ``check_overlap=False`` keeps only the depth cap, for a goal whose containment in an
        ancestor carries no information — a fired condition's ``then`` (see ``_subgoal``)."""
        depth = len(activity.parent_frames)
        if depth >= self._max_subgoal_depth:
            return (
                f"sub-goal recursion reached the depth cap ({depth} >= {self._max_subgoal_depth})"
            )
        if check_overlap:
            for ancestor in _ancestor_subgoal_goals(activity):
                if _goal_token_overlap(goal, ancestor) >= _SUBGOAL_GOAL_OVERLAP:
                    return (
                        "sub-goal goal repeats an ancestor's without reducing to concrete actions"
                    )
        return None

    async def _ground(
        self,
        step: Step,
        activity: Activity,
        wm: WorkingMemory,
        cycle: DecisionCycle,
    ) -> Step | None:
        """Ground a step's reference-bearing params against this run's execution history for *this
        cycle* plus the agent's currently observed properties/signals, leaving the stored plan's
        references intact so procedural reuse keeps a reusable skeleton. Deciding a param value is
        a *reasoning* act, so it lives here, not in Act (which stays mechanistic). Hybrid: resolve
        references deterministically; escalate to one **off-cycle** model call (the ``_ground_``
        internal action) only for what can't be resolved mechanically. Returns the concrete Step
        when grounding is complete (no references, references resolved mechanically, or a prior
        ``_ground_`` escalation's params peeked from ``activity.grounded_params``), or ``None``
        when it just fired ``_ground_`` — the activity is now RUNNING and the step lands a later
        cycle. A step with no references is a pure no-op — the cheap path makes no model call. Only
        ``invoke`` (an operation's params) and ``send`` (its ``content``) carry a groundable bag
        today; ``focus``/``unfocus`` carry only a bare ``tool_id``, nothing to ground."""
        if step.next_action == InvokeAction.name:
            routing = {k: v for k, v in step.params.items() if k in (TOOL_ID, OPERATION_NAME)}
            op_params = {k: v for k, v in step.params.items() if k not in (TOOL_ID, OPERATION_NAME)}
            manual = _manual_for(wm, routing.get(TOOL_ID))
            resolved = await self._resolve(
                activity, wm, cycle, op_params, routing[OPERATION_NAME], manual
            )
            if resolved is None:
                return None  # escalated to _ground_; RUNNING now
            undeclared = _undeclared_params(manual, routing[OPERATION_NAME], resolved)
            if undeclared:
                # A param the operation does not take. Invoking would raise an unexpected-keyword
                # TypeError at the wire, and a failed op terminates the activity (DefaultReflect) —
                # so a plan that is otherwise right dies on a name. It IS a plan defect: the model
                # had the schema and wrote past it (a real run took `limit` from get_contacts' prose
                # description, which mentions a view limit it does not accept). Dropping the key
                # instead was rejected — silently changing what an operation is asked to do is worse
                # on a `send`-shaped op than failing, and a misspelled *required* param would only
                # trade this error for a null-required skip. So: replan, carrying the reason, which
                # is what makes the retry differ from the attempt. Checked after grounding, so a key
                # the grounder invents is caught too, not just one the planner wrote.
                defect = (
                    f"{routing[OPERATION_NAME]}: no such parameter(s) "
                    f"{', '.join(repr(k) for k in undeclared)} — that operation accepts only "
                    f"{', '.join(sorted(_declared_param_names(manual, routing[OPERATION_NAME])))}"
                )
                log.warning("reason: plan defect for activity %s — %s", activity.id, defect)
                activity.reset_for_replan(defect=defect)
                return None
            mistyped = _mistyped_params(manual, routing[OPERATION_NAME], resolved, op_params)
            if mistyped:
                # Same defect class as `undeclared` above, one line down the same schema: the name
                # was right and the value is not something the operation can take. Checked after
                # grounding for the same reason, so a value the grounder produced is covered too.
                defect = f"{routing[OPERATION_NAME]}: {'; '.join(mistyped)}"
                log.warning("reason: plan defect for activity %s — %s", activity.id, defect)
                activity.reset_for_replan(defect=defect)
                return None
            if resolved == op_params:
                return step  # no references -> unchanged, reuse the original Step
            return replace(step, params={**routing, **resolved})
        if step.next_action == SendAction.name:
            content = step.params.get("content")
            if not isinstance(content, dict):
                return step  # nothing groundable
            resolved = await self._resolve(activity, wm, cycle, content, SendAction.name, None)
            if resolved is None:
                return None
            if resolved == content:
                return step
            return replace(step, params={**step.params, "content": resolved})
        return step

    @staticmethod
    async def _resolve(
        activity: Activity,
        wm: WorkingMemory,
        cycle: DecisionCycle,
        params: dict[str, Any],
        operation_name: str,
        manual: Manual | None,
    ) -> dict[str, Any] | None:
        """Mechanically resolve ``params`` against history; if anything can't be, escalate to the
        off-cycle ``_ground_`` action and return ``None``. When a prior escalation already resolved
        (``activity.grounded_params`` set by Observe), *peek* and return those instead — the 1:1
        counterpart to the plan landing on ``activity.plan``. step_index hasn't advanced across the
        RUNNING cycles, so the parked params belong to exactly this step. Peek, not consume: a
        checkpoint (ADR-0024) can run between grounding and the invoke and defer the step a cycle,
        so reason() clears the params at the commit site once the step dispatches — leaving them
        parked here means a deferred re-entry re-emits the same step instead of re-escalating."""
        if activity.grounded_params is not None:
            return activity.grounded_params  # the escalation resolved; peek (cleared at commit)
        resolved, unresolved = resolve_references(
            params, activity.history, activity.bindings, wm.properties
        )
        if not unresolved:
            return resolved  # cheap path — resolved mechanically, no model call
        # Prompt context only (the mechanical resolve above already read the whole store), so
        # narrowing it to this activity's own tools costs nothing and is where the tokens are.
        observed = scoped_snapshot(wm, activity)
        ground = cycle.actions.internal(GroundAction.name)
        await ground.execute(
            cycle,
            activity_id=activity.id,
            operation_name=operation_name,
            manual=manual,
            partial_params=resolved,
            observed=observed,
        )
        return None  # RUNNING on the grounding escalation; params land a later cycle


class DefaultActStrategy:
    """The mechanical, no-LLM default for *parameter binding* (not protocol binding — see
    ActStrategy): bind an ``invoke`` Step straight to an OperationInvocation, splitting the
    tool_id/operation_name routing keys out of the operation's own params. A model-backed
    ActStrategy would instead ground under-specified params against the manual's schema here; the
    default assumes the Step already carries concrete params, so binding is just the key-split.

    Two mechanical guards sit here (both structural checks, no judgment, so Act stays mechanistic
    per ADR-0017). The first is a **leak guard**: grounding has already run by the time a step
    reaches binding, so a ``$from``/``$decide``/``$bind`` dict still present in the params is not a
    reference waiting to be filled — it is one the resolver failed to *see*, and binding it would
    serialize the reference itself to the wire as a literal object. The tool then rejects it with a
    message that names the wrong culprit (a type error on the enclosing list), so the guard skips
    the invoke and logs the offending paths instead. It is a backstop for a resolver bug, not part
    of normal flow; a healthy run never trips it.

    The second: a **required** param that resolves to null is a schema violation, so the invoke is
    *skipped* — no
    invocation is emitted and the cycle dispatches nothing this step (`_act`). Grounding (Reason)
    has already run by now, so a null at bind time is a value the model declined or could not fill,
    not an un-grounded reference; dispatching the operation anyway is the historic blind-`delete`
    mis-action, degraded to a probabilistic one and previously held off only by a prompt fragment.
    The guard needs the operation's schema (`OperationSpecification.parameters`, adapter-
    synthesized) to know which params are required; with no manual/spec/declared ``required`` it
    cannot tell, so it does not fire and binds as before — the same structured-spec dependency the
    thread-reading Manual relocation has. It narrows, not eliminates, a null-invoke: an *optional*
    null still passes through by design (many operations take legitimately-optional params)."""

    async def bind(
        self, step: Step, manual: Manual | None, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        params = {k: v for k, v in step.params.items() if k not in (TOOL_ID, OPERATION_NAME)}
        operation_name = step.params[OPERATION_NAME]
        leaked = _reference_paths(params)
        if leaked:
            log.error(
                "act: skipping invoke %s.%s — unresolved reference(s) at %s reached parameter "
                "binding; grounding never saw them (resolver bug, not a plan bug)",
                step.params[TOOL_ID],
                operation_name,
                leaked,
            )
            return result  # no invocation -> _act dispatches nothing this step (skip-and-continue)
        null_required = _null_required_params(manual, operation_name, params)
        if null_required:
            log.warning(
                "act: skipping invoke %s.%s — required param(s) %s resolved to null",
                step.params[TOOL_ID],
                operation_name,
                null_required,
            )
            return result  # no invocation -> _act dispatches nothing this step (skip-and-continue)
        invocation = OperationInvocation(
            tool_id=step.params[TOOL_ID],
            operation_name=operation_name,
            params=params,
        )
        return replace(result, invocation=invocation)


def _declared_param_names(manual: Manual | None, operation_name: str) -> list[str]:
    """The param names the operation's schema declares — for naming the alternatives in a defect
    message, so the replanning prompt says what IS accepted, not only what was not."""
    spec = manual.operation(operation_name) if manual is not None else None
    declared = spec.parameters.get("properties") if spec is not None else None
    return list(declared) if isinstance(declared, dict) else []


def _undeclared_params(
    manual: Manual | None, operation_name: str, params: dict[str, Any]
) -> list[str]:
    """Params the operation's schema does not declare — the ones an invoke would pass as unexpected
    keyword arguments. Empty when required-ness is unknowable (no manual, no spec, no declared
    ``properties``), the same structured-spec dependency ``_null_required_params`` has.

    Treats a schema without an explicit ``additionalProperties`` as *closed*, which inverts the
    JSON Schema default. Deliberate: these schemas are synthesized from real callables (an ARE
    ``AppTool`` signature, an MCP ``inputSchema``), so an undeclared key is a TypeError at the
    wire, not a tolerated extra. An adapter whose operation genuinely takes a free-form bag
    declares ``additionalProperties`` itself, and is then left alone."""
    spec = manual.operation(operation_name) if manual is not None else None
    if spec is None:
        return []
    if spec.parameters.get("additionalProperties", False) is not False:
        return []
    declared = spec.parameters.get("properties")
    if not isinstance(declared, dict):
        return []
    return sorted(key for key in params if key not in declared)


# What each JSON-Schema `type` accepts, as Python. `bool` is excluded from the numeric rows on
# purpose: it IS an `int` to Python, and a tool asking for a count that is handed True has been
# mis-planned, not satisfied. Types outside this table (unions, `null`, anything an adapter invents)
# are absent rather than empty — absent means "no opinion", which is how the check fails open.
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def _mistyped_params(
    manual: Manual | None,
    operation_name: str,
    params: dict[str, Any],
    raw: dict[str, Any],
) -> list[str]:
    """Params whose resolved value contradicts the type its schema declares, for a planner.

    The sibling of `_undeclared_params`, and it exists for the identical reason: the wire raises
    (ARE answers `Argument 'start_datetime' must be of type <class 'str'>, got <class 'float'>`), a
    failed op terminates the activity, and so a plan that is otherwise right dies on a conversion.
    It is a plan defect in the same sense too — the schema was in the catalog and the step wrote
    past it. The motivating run piped `get_calendar_event`'s epoch-float `start_datetime` straight
    into `get_calendar_events_from_to`, which declares a `YYYY-MM-DD HH:MM:SS` STRING; the run died
    there, mid-maintenance-window, on the first firing.

    Nothing is coerced. Turning 1729375200.0 into "1729375200.0" would satisfy the type and still be
    the wrong argument — the operation wants a formatted date, and only the planner can know that.
    Reporting it is the whole fix: the replan gets a defect naming the format and picks the field
    that already carries it.

    Fails open at every step where the schema is less than explicit — no manual, no spec, no
    declared `properties`, a `type` this table has no row for, a value of None (that is
    `_null_required_params`' question). A false positive here would refuse a call that WOULD have
    worked, which is strictly worse than the failure being guarded against.

    Names the reference a bad value came from, not just the parameter it landed in: the value is one
    hop from its producer and the producer is what has to change.
    """
    spec = manual.operation(operation_name) if manual is not None else None
    if spec is None:
        return []
    declared = spec.parameters.get("properties")
    if not isinstance(declared, dict):
        return []
    problems = []
    for key, value in params.items():
        schema = declared.get(key)
        if not isinstance(schema, dict) or value is None:
            continue
        declared_type = schema.get("type")
        # A union (`["string", "null"]`) is a list, and an unhashable one — so this reads the type
        # before looking it up, rather than after. A union means the schema has more than one
        # opinion, which is no single opinion to check against.
        accepted = _JSON_TYPES.get(declared_type) if isinstance(declared_type, str) else None
        if accepted is None:
            continue
        # `bool` is an `int` to Python, so the isinstance below would wave True through for a
        # declared number. Decide it first, in both directions.
        if isinstance(value, bool):
            if accepted == (bool,):
                continue
        elif declared_type == "integer" and isinstance(value, float) and value.is_integer():
            continue  # 3.0 after a JSON round trip still represents an integer; 3.14 does not
        elif isinstance(value, accepted):
            continue
        origin = raw.get(key)
        came_from = f" (from {origin!r})" if isinstance(origin, dict) else ""
        described = schema.get("description")
        wants = f", which wants {described}" if isinstance(described, str) and described else ""
        problems.append(
            f"{key!r} must be {declared_type} but got {type(value).__name__} "
            f"{value!r}{came_from}{wants}"
        )
    return problems


def _null_required_params(
    manual: Manual | None, operation_name: str, params: dict[str, Any]
) -> list[str]:
    """The operation's *required* params (per its schema) that resolve to null in ``params`` —
    either an explicit None or absent entirely (a required key the step never supplied). Empty when
    the schema is unavailable (no manual, the manual doesn't describe this op, or it declares no
    ``required``): required-ness is then unknowable, so the guard can't fire and binding goes on."""
    spec = manual.operation(operation_name) if manual is not None else None
    if spec is None:
        return []
    required = spec.parameters.get("required", [])
    return [key for key in required if params.get(key) is None]
