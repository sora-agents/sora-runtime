"""Deferred-inference deadlines, reconciliation support, and failure recovery."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from sora._strategies.conditions import (
    _unclosable_window,
)
from sora.activity import ActivityState
from sora.llm import LLMOutcome, log_llm_discarded, log_llm_late_completion, log_llm_outcome
from sora.memory import (
    render_plan,
)
from sora.references import (
    _with_empty_binding_origin,
)
from sora.types import (
    ConditionVerdict,
    InferenceKind,
    InferenceResult,
)

if TYPE_CHECKING:
    from sora.cycle import DecisionCycle
    from sora.types import Plan

log = logging.getLogger("sora.strategies")

# How long an in-flight infer/ground may say nothing before Observe gives up on it. Generous by
# design: it is a watchdog for a seam that has stopped answering, not a latency budget — a thinking
# model legitimately spends tens of seconds, and expiring a call that was about to succeed buys a
# replan nobody needed. The client-side stall timeout is the tighter, better-informed bound (it can
# tell a quiet socket from a slow one); this one exists because not every failure reaches it.
DEFAULT_INFERENCE_DEADLINE = 300.0


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
_REPLANNABLE_INFERENCE = frozenset(
    {
        InferenceKind.PLAN,
        InferenceKind.SUBGOAL,
        InferenceKind.GROUND,
        InferenceKind.CONDITION_FOLLOWUP,
    }
)


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


def _expire_stalled_inferences(cycle: DecisionCycle, deadline: float | None) -> None:
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
    if deadline is None:
        return
    now = time.time()
    for activity in cycle.working.activities.values():
        pending = activity.pending_inference
        if pending is None or activity.state is not ActivityState.RUNNING:
            continue
        waited = now - pending.requested_at
        if waited < deadline:
            continue
        log.warning(
            "observe: %s for activity %s exceeded %.0fs (waited %.0fs) -> giving up",
            pending.kind,
            activity.id,
            deadline,
            waited,
        )
        cycle.inference_sink.push(
            pending.id,
            InferenceResult(
                id=pending.id,
                error=f"inference stalled: no result after {waited:.0f}s",
            ),
        )


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
        claimed = any(
            activity.pending_inference is not None
            and activity.pending_inference.id == inf_id
            and activity.state is ActivityState.RUNNING
            for activity in wm.activities.values()
        )
        if claimed:
            log_llm_outcome(inf_id, outcome, error=res.error)
        else:
            # A watchdog/interrupt/superseding inference already resolved the runtime side.
            # The provider finishing later is useful observability, but it is not a second
            # terminal outcome and must not read like an earlier error turned into success.
            log_llm_late_completion(inf_id, outcome)
        for activity in wm.activities.values():
            if (
                activity.pending_inference is not None
                and activity.pending_inference.id == inf_id
                and activity.state is ActivityState.RUNNING
            ):
                kind = activity.pending_inference.kind
                out = activity.pending_inference.out  # set only for kind=="select"
                baseline = activity.pending_inference.baseline  # set for plan/subgoal (ADR-0024)
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
                    if kind is InferenceKind.GROUND:
                        defect = _with_empty_binding_origin(activity, res.unresolvable)
                    elif kind is InferenceKind.SELECT:
                        defect = res.unresolvable
                    else:
                        raise AssertionError(
                            f"{kind.value} inference cannot resolve as unresolvable"
                        )
                    activity.reset_for_replan(defect=defect)
                    activity.state = ActivityState.READY
                    log.warning(
                        "observe: %s for activity %s resolved nothing (%s) -> replan",
                        kind,
                        activity.id,
                        defect,
                    )
                elif res.error is not None and kind is InferenceKind.SELECT:
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
                elif res.error is not None and kind is InferenceKind.CONDITION:
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
                elif res.error is not None and kind is InferenceKind.REVALIDATE:
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
                    raise AssertionError(f"unhandled failure for {kind.value} inference")
                elif kind is InferenceKind.PLAN:
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
                elif kind in (InferenceKind.SUBGOAL, InferenceKind.CONDITION_FOLLOWUP):
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
                    if kind is InferenceKind.SUBGOAL:
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
                        "sub-plan"
                        if kind is InferenceKind.SUBGOAL
                        else "a fired condition's `then`",
                        activity.id,
                    )
                    log.debug(
                        "observe: sub-plan for activity %s (nested under %d frame(s))\n%s",
                        activity.id,
                        len(activity.parent_frames),
                        render_plan(sub_plan),
                    )
                elif kind is InferenceKind.SELECT:
                    # A $decide data-op filter (ADR-0023): the surviving subset lands into the
                    # named binding, exactly like a mechanical filter would have written it. Not
                    # a Percept (deliberation output, not observed state) — same as plan/ground.
                    assert out is not None  # a "select" pending always carries its target name
                    activity.bindings[out] = res.value  # value is Any-typed dict; no cast needed
                    activity.state = ActivityState.READY
                    log.info(
                        "observe: resolved $decide filter -> binding %r for activity %s",
                        out,
                        activity.id,
                    )
                elif kind is InferenceKind.CONDITION:
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
                elif kind is InferenceKind.REVALIDATE:
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
                elif kind is InferenceKind.GROUND:
                    activity.grounded_params = res.value  # type: ignore[assignment]  # => dict
                    activity.state = ActivityState.READY
                    log.info("observe: resolved grounded params for activity %s", activity.id)
                else:
                    raise AssertionError(f"unhandled successful {kind.value} inference")
                break
        else:
            # No live activity claimed this result: it was invalidated (an interrupt re-routed
            # the activity) or superseded (a re-inference gave a new id). The background model
            # call ran to completion and was already metered, so its cost is real but wasted —
            # tell the meter to move it to the wasted bucket (a no-op when uninstrumented).
            log_llm_discarded(inf_id)
