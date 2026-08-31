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
from sora.activity import Activity, ActivityState
from sora.types import (
    ConditionWait,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from sora.cycle import DecisionCycle
    from sora.memory import WorkingMemory

log = logging.getLogger("sora.strategies")


def _summarize(activity: Activity, *, succeeded: bool) -> str:
    """A deterministic, no-LLM episode summary. A model-backed ReflectStrategy would substitute a
    richer natural-language summary here; the mechanical default just states outcome and goal."""
    outcome = "completed" if succeeded else "failed"
    return f"{outcome}: {activity.goal}"


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
