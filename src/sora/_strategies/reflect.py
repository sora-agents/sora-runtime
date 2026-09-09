"""Default Reflect strategy and episode summarization."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any

from sora._strategies.conditions import (
    _condition_watches,
    _eligible_conditions,
    _lift_pending_conditions,
)
from sora._strategies.contracts import (
    TickResult,
)
from sora._strategies.interaction import _truncate
from sora.activity import Activity, ActivityState
from sora.references import _manual_for
from sora.types import (
    ConditionWait,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from sora.cycle import DecisionCycle
    from sora.memory import WorkingMemory

log = logging.getLogger("sora.strategies")


def _summarize(activity: Activity) -> str:
    """A deterministic, no-LLM episode summary. A model-backed ReflectStrategy would substitute a
    richer natural-language summary here; the mechanical default just states outcome and goal."""
    return f"completed: {activity.goal}"


def _operation_failure_defect(activity: Activity, wm: WorkingMemory) -> str:
    """Name a rejected operation for the replacement plan without duplicating its parameters.

    Observe has already appended the full invocation and result to ``history``; the defect is the
    short, stable reason the replan breaker compares and the planning prompt places beside that
    execution evidence. The fallback admits a hand-built or custom plan-less activity carrying
    only ``last_operation`` without pretending an invocation identity is known.

    A not-ok ack is not proof the effect did not land: an adapter turns a timeout, a transport
    error, or a post-write validation failure into the same rejection as a bad argument, and the
    runtime cannot tell those apart. So unless the manual declares the operation a read
    (``side_effecting is False``), the defect says so — an operation that never declared it counts
    as a write here, the same conservative reading `before_writes` takes. Deliberately a warning to
    the planner rather than a refusal to replan: only a read can establish what the world now holds,
    and only the planner can put that read in front of the retry.
    """
    failed = activity.last_operation
    assert failed is not None and not failed.ok
    operation = "external operation"
    side_effecting: bool | None = None
    if activity.history and activity.history[-1].ack is failed:
        invocation = activity.history[-1].invocation
        operation = f"{invocation.tool_id}.{invocation.operation_name}"
        manual = _manual_for(wm, invocation.tool_id)
        spec = manual.operation(invocation.operation_name) if manual is not None else None
        side_effecting = spec.side_effecting if spec is not None else None
    defect = f"{operation} failed: {_truncate(failed.result)}"
    if side_effecting is not False:
        # No trailing period: the planning prompt continues the sentence this is spliced into.
        defect += (
            " (the rejection does not prove the call had no effect — check the current state "
            "before repeating it)"
        )
    return defect


class DefaultReflectStrategy:
    """The runtime's built-in default — purely mechanical, no LLM.

    Judges each activity completed or failed by two deterministic rules, and on a terminal outcome
    records the experience: the state transition is synchronous (so Situate, which runs later this
    cycle and selects only READY activities, never re-selects a just-terminated one), while the
    episodic/procedural writes are dispatched as background tasks and never block the cycle (several
    activities may terminate in the same cycle). Strong refs to the in-flight tasks are held so they
    aren't GC'd mid-write — the same pattern as InvokeAction.

    The two rules are deliberately asymmetric. A resolved-but-not-ok ``last_operation`` is definite
    negative evidence about the current plan, so the default drops that plan with a named defect and
    lets Reason's existing replan breaker bound recovery. It does not terminate an otherwise-live
    goal for one rejected call. **Completion** requires positive evidence that all planned work is
    done — a plan present and fully consumed (``step_index >= len(plan.steps)``) — so a plan-less
    activity is never auto-completed here (what a plan-following Reason, and any application driving
    activities without a plan, relies on). Only terminal completion records an episode here, and it
    never auto-caches the plan to procedural memory — replaying a stored plan verbatim is unsound,
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
        # its own episode — the completion branch below is the only path left that terminates an
        # activity, and it writes one before handing back, which is what lets reflect() skip them
        # and stay idempotent across the cycles it runs on every activity.
        if activity.state is not ActivityState.READY:
            return result
        if self.failed(activity):
            defect = _operation_failure_defect(activity, wm)
            log.warning("reflect: activity %s replanning after %s", activity.id, defect)
            activity.reset_for_replan(defect=defect)
            # The failure remains in history as execution evidence. Clear only the one-shot trigger
            # Reflect just handled, or the next tick would abandon the replacement before it could
            # run. The broader stale-last-operation path across InputWait remains separate.
            activity.last_operation = None
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
        """Whether a resolved operation rejects the current plan and requires recovery."""
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
        await cycle.episodic.learn(activity, _summarize(activity), succeeded=True)
