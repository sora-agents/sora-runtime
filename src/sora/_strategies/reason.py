"""Default Reason strategy and replanning safeguards."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from sora._strategies.conditions import (
    _body_exhausted,
    _condition_watches,
    _conditions_hold_frame,
    _eligible_conditions,
    _lift_step_conditions,
)
from sora._strategies.contracts import (
    TickResult,
)
from sora._strategies.interaction import (
    _await_input,
)
from sora._strategies.observe import (
    scoped_snapshot,
)
from sora._strategies.parameters import (
    _declared_param_names,
    _mistyped_params,
    _undeclared_params,
)
from sora._strategies.reconsideration import (
    _step_side_effecting,
)
from sora._strategies.subgoals import (
    _DEFAULT_MAX_SUBGOAL_DEPTH,
    _SUBGOAL_DEFECT,
    _SUBGOAL_GOAL_OVERLAP,
    _SUBGOAL_HALTED,
    _SUBGOAL_RUNNING,
    _SUBGOAL_SPLICED,
    _ancestor_subgoal_goals,
    _expand_mechanical,
    _goal_token_overlap,
)
from sora.action import (
    CollectAction,
    EvaluateConditionsAction,
    FilterAction,
    GroundAction,
    InferAction,
    InvokeAction,
    RevalidateAction,
    SendAction,
)
from sora.activity import SEEDED_BINDINGS, Activity, ActivityState
from sora.data_ops import (
    _enrich_with_params,
    _resolve_collection,
    _resolve_predicate_value,
)
from sora.memory import (
    PerceptSnapshot,
    render_plan,
    render_steps,
)
from sora.references import (
    _REPLAN_HINT,
    _manual_for,
    _unsatisfiable_reference,
    resolve_references,
)
from sora.types import (
    OPERATION_NAME,
    SUBGOAL,
    TOOL_ID,
    Change,
    ConditionFiring,
    ConditionVerdict,
    ConditionWait,
    InferenceKind,
    Step,
    SubgoalMode,
    changes_of,
    subgoal_mode_of,
)

if TYPE_CHECKING:
    from sora.cycle import DecisionCycle
    from sora.manual import Manual
    from sora.memory import WorkingMemory

log = logging.getLogger("sora.strategies")

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
                "mode": SubgoalMode.DELIBERATIVE,
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
        try:
            mode = subgoal_mode_of(step)
        except ValueError as exc:
            mode_defect = str(exc)
            log.warning("reason: plan defect for activity %s — %s", activity.id, mode_defect)
            activity.reset_for_replan(defect=mode_defect)
            return _SUBGOAL_DEFECT
        if mode is SubgoalMode.DELIBERATIVE:
            # A deliberative sub-goal is accepted once it has passed the recursion guard below:
            # the child inference needs its step-owned conditions in the prompt, but a rejected
            # step must not leave a condition behind for a replacement plan to inherit.
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
            _lift_step_conditions(step, activity, wm, mode)
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
                kind=(
                    InferenceKind.CONDITION_FOLLOWUP if from_condition else InferenceKind.SUBGOAL
                ),
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
        # The fan-out is now known to be a committed step (including a known-empty one), so its
        # maintenance window can safely outlive the splice.  Do this after validation: conditions
        # on a rejected mechanical step belong to no plan and must not survive its replan.
        _lift_step_conditions(step, activity, wm, mode)
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
        return None
