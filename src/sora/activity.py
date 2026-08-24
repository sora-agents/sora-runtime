"""Activity: the sole first-class unit of work (see ADR-0002)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from sora.types import SupersededPlan  # constructed at run time by reset_for_replan

if TYPE_CHECKING:
    from sora.types import (
        CompletedOperation,
        ConditionVerdict,
        ConditionWait,
        InputWait,
        OperationAck,
        PendingCondition,
        PendingConditionState,
        PendingInference,
        PendingOperation,
        Plan,
        SignalWait,
    )


class ActivityState(Enum):
    RUNNING = "running"
    BLOCKED = "blocked"
    READY = "ready"
    TERMINATED = "terminated"


@dataclass
class Activity:
    id: str
    goal: str
    context: dict[str, Any]
    state: ActivityState = ActivityState.READY
    plan: Plan | None = None  # once set, Reason can just advance it instead of (re)planning
    step_index: int = 0
    # The intention stack for sub-goals (ADR-0022): each entry is a suspended parent frame (its
    # Plan, the step_index of the sub-goal that pushed it, and that frame's `history_mark` below).
    # `plan`/`step_index` above are the *active* frame; entering a deliberative sub-goal pushes the
    # parent here (Observe) and exhausting a sub-plan pops it, resuming the parent at the step after
    # its sub-goal (Reason). Empty for a flat plan — generalizes step_index rather than adding a
    # separate intention type (ADR-0002).
    parent_frames: list[tuple[Plan, int, int]] = field(default_factory=list)
    # Where in `history` the *active* frame's plan began. `history` is flat and frame-agnostic (see
    # reset_for_replan below, which relies on that), which is right for `$from` — it reads the
    # LATEST match, so it stays current, and a sub-plan reading the event its parent created is the
    # normal case. It is wrong for `collect`, which takes EVERY match and so silently accumulates
    # results from plans that already finished: an observed run collected its parent's calendar
    # query alongside its own and fanned out a delete over the stale set. So collect reads
    # `history[history_mark:]`. Set wherever a plan is installed (inferred, cached, or a sub-plan),
    # saved into the frame on push and restored on pop; a replan needs no reset of its own, since
    # installing the replacement plan re-marks it past everything the discarded plan ran.
    history_mark: int = 0
    pending_operation: PendingOperation | None = None  # set while RUNNING; cleared on resolve
    # set while RUNNING on an off-cycle infer()/ground() (the _infer_/_ground_ internal actions),
    # mutually exclusive with pending_operation. RUNNING thus has two resolve sources — an invoke
    # ack (result_sink) and an inference result (inference_sink) — but stays one state. Cleared in
    # Observe when the matching InferenceResult resolves; a result whose id no longer matches is
    # discarded (the stale-inference guard).
    pending_inference: PendingInference | None = None
    # A resolved _ground_ escalation's concrete params, parked here by Observe for Reason's next
    # pass to consume (build the concrete Step) — the ground counterpart to a plan on `plan`.
    grounded_params: dict[str, Any] | None = None
    last_operation: OperationAck | None = None  # most recently resolved result, for Reason to read
    # set while BLOCKED; what the activity waits for before returning to READY. Orthogonal to
    # pending_operation: RUNNING waits on an operation result (automatic 1:1 resolve), BLOCKED waits
    # on one of two declared things. A SignalWait — a manual-declared completion signal, set by
    # _suspend_ and cleared by _resume_, matched in Observe. Or an InputWait — the user's next
    # instruction, set by the interrupt handler when a hard interrupt pauses it, cleared in
    # Observe when a user Message arrives. Or a ConditionWait — the plan's own declared pending
    # conditions outliving an exhausted body, set when the body runs out with conditions still
    # unsatisfied. Named generally (not blocked_on_signal) to admit these later variants.
    blocked_on: SignalWait | InputWait | ConditionWait | None = None
    # Per-run state for Plan.pending: which declared conditions are still unsatisfied and how far
    # each has evaluated. Conditions declared by ANY frame are lifted here when that frame pops —
    # the point of a condition is to outlive the plan that noticed it, and a deliberative sub-goal
    # is usually where the agent first learns a branch exists (it sent the mail; now a reply may
    # come). Transient run state, like history/bindings: the durable copy is Plan.pending.
    pending_conditions: list[PendingConditionState] = field(default_factory=list)
    # A resolved condition evaluation parked for Reason's next pass — the pending-condition
    # counterpart to reconsider_verdict. Holds indices into the eligible list the call was made
    # about, so Reason re-derives that list to apply it. Transient run state.
    condition_verdict: ConditionVerdict | None = None
    # The conditions the in-flight (or just-resolved) evaluation was made about, in the order the
    # call presented them — the verdict's indices are into THIS list. Kept next to the verdict so
    # the correspondence travels with the activity rather than in strategy-held state, which would
    # not survive a strategy being rebuilt and could leak across activities. Transient run state.
    condition_batch: list[PendingConditionState] = field(default_factory=list)
    # Conditions a verdict judged fired whose `then` has not been pursued yet, oldest first. One
    # call judges the whole eligible batch and may fire several at once (a single reply can satisfy
    # two gates), but each `then` is a goal in its own right and runs one at a time — so the rest
    # queue here rather than being dropped. A queue is required, not a convenience: every judged
    # condition's mark is advanced at fire time, so the signal that opened those gates is already
    # behind them and nothing would ever re-fire the ones not pursued. Transient run state.
    condition_fired: list[PendingConditionState] = field(default_factory=list)
    # Conditions whose `until` a verdict judged satisfied. Retiring drops the per-run state, but
    # Plan.pending is the frozen skeleton and never changes — so without a record here the next
    # lift would read the same declaration off the same plan and put the condition straight back on
    # watch. By condition VALUE (the same key the lift dedups on), not by state identity, since the
    # state being retired is exactly the object being thrown away. Transient run state.
    retired_conditions: set[PendingCondition] = field(default_factory=set)
    # Append-only trace of resolved operations this activity ran — a later step grounds its params
    # against it (last_operation keeps only the newest, overwritten each step). Transient:
    # not persisted, and episodic learn() captures selectively, not a blind asdict(activity).
    history: list[CompletedOperation] = field(default_factory=list)
    # Named bindings a data-op step writes and a later step reads via {"$bind": "<name>"} (ADR-0023)
    # — the imperative pipeline's intermediate values (a filtered/deduped/sorted collection, a
    # reduced scalar). Transient run state like `history`/`grounded_params` (not persisted); cleared
    # on replan since the values are coupled to the plan that produced them. Distinct from the
    # mechanical sub-goal's eager loop-element $bind, which is substituted at fan-out, never stored.
    bindings: dict[str, Any] = field(default_factory=dict)
    # Context-adaptation reconsideration (ADR-0024). reconsider_baseline is a compact signature of
    # the perception the current plan was inferred against (its assumptions) — captured at infer
    # time and installed with the plan by Observe, so a change that landed *during* inference is
    # caught at the first checkpoint; a reused plan (no fresh inference) falls back to an entry-time
    # baseline. Reason's checkpoint compares it to the live signature and, when they differ, fires
    # an off-cycle revalidation. reconsider_verdict parks that re-check's bool result for Reason's
    # next pass (True -> proceed and re-baseline; False -> reset_for_replan). Both are transient run
    # state, cleared on reset_for_replan.
    reconsider_baseline: object | None = None
    reconsider_verdict: bool | None = None
    # The plan the last reset_for_replan() discarded, kept for exactly one inference: the planning
    # prompt renders its un-run tail so the replacement is written against the intent it replaces
    # rather than from nothing (ADR-0024). Cleared by Observe once that replacement installs, so it
    # can never leak into a later, unrelated inference. Transient run state; never persisted.
    superseded: SupersededPlan | None = None
    # Why each of the *consecutive* replans that has run no operation was taken, oldest first (None
    # = no defect, the plan was sound and the world moved). Reason reads it to decide whether
    # another plan is worth inferring at all. Deliberately progress-relative rather than a lifetime
    # count: an agent in a dynamic environment is *supposed* to replan without limit, so an absolute
    # budget on adapting would cap the thing the runtime exists to do. Executing a single operation
    # clears the trail, which is what separates "kept adjusting while getting somewhere" from
    # "produced N plans and never moved". Transient run state; never persisted.
    replan_trail: list[str | None] = field(default_factory=list)
    replan_history_mark: int = 0  # len(history) as of the last replan — the progress marker above
    # context is exclusively for strategy-author data — the runtime itself never writes into it,
    # which is what keeps pending_operation/last_operation as dedicated fields instead of context
    # keys with a naming convention: no shared namespace means no collision to avoid in the first
    # place

    def discard_inference(self) -> None:
        """Invalidate an off-cycle infer/ground in flight (and any params it already parked): the
        late result is discarded on resolve because its id no longer matches the (now-cleared)
        pending_inference. Side-effect-free, unlike an external op, so always safe to drop. Leaves
        `state` untouched — the caller decides where the activity goes next (BLOCKED to await input,
        READY to re-plan, ...)."""
        self.pending_inference = None
        self.grounded_params = None

    def _progressed_since_replan(self) -> bool:
        """Whether anything genuinely new has run since the last replan.

        The obvious test — did ``history`` grow — is too generous, and an observed run showed how
        it fails. Five plans in a row each re-issued ``get_contacts(offset=0)``, a call already in
        history, so every replan looked like progress, the trail cleared each time, and the breaker
        never came near its cap while the agent went nowhere and the plans stayed equally stuck.
        Re-running a call whose arguments already appear in history yields no fact the next plan
        did not already have, so it cannot be what forgives a replan.

        The test is mechanical — same tool, operation and params — and deliberately errs toward
        *not* forgiving: re-reading state that has since changed scores as no progress even though
        the result may differ. That direction is the safe one, because the trail only ever counts,
        and what it counts toward is asking the user rather than terminating anything.
        """
        for i in range(self.replan_history_mark, len(self.history)):
            call = self.history[i].invocation
            if not any(
                done.invocation.tool_id == call.tool_id
                and done.invocation.operation_name == call.operation_name
                and done.invocation.params == call.params
                for done in self.history[:i]
            ):
                return True
        return False

    def clear_replan_trail(self) -> None:
        """Forget the consecutive-replan trail: new direction has arrived, so the attempts that led
        to a halt no longer bear on whether the *next* plan is worth inferring. Without this a
        resumed activity re-trips the breaker on its first pass and could never act on the guidance
        it just received — the halt would be permanent rather than a question."""
        self.replan_trail.clear()
        self.replan_history_mark = len(self.history)

    def reset_for_replan(self, defect: str | None = None) -> None:
        """Drop the current plan and any in-flight/parked deliberation so Reason re-plans from
        scratch. A droppable inference in flight is invalidated and the activity returns to READY;
        an in-flight *external* op is left RUNNING (its physical side effect must complete) with
        only its stale plan cleared — it resolves to READY later and Reason re-plans then. The one
        place every plan-invalidation site (interrupt handlers, signal-driven re-planners) routes
        through, so new deliberation state can't be forgotten at one call site.

        ``defect`` distinguishes the two reasons a plan gets dropped: pass the specific defect when
        the plan itself cannot work (its assumption about the world is false and will stay false),
        and leave it None when the plan was fine but the world moved under it. The replanning prompt
        reads it to decide whether to tell the planner to reuse this plan or to route around it."""
        was_inferring = self.pending_inference is not None
        # Park what is being dropped for the *next* inference to read (ADR-0024): a blank-slate
        # replan is correct but wasteful, and the planner reuses what still applies far better than
        # the runtime could decide for it. Captured before the fields below are cleared, and only
        # when there is a plan — a reset with nothing in flight leaves any earlier bundle alone.
        if self.plan is not None:
            self.superseded = SupersededPlan(
                plan=self.plan,
                step_index=self.step_index,
                # The mark is live runtime scoping, meaningless in a record that only ever gets
                # rendered into a prompt — so the bundle keeps the (plan, step_index) shape.
                parent_frames=[(plan, index) for plan, index, _ in self.parent_frames],
                defect=defect,
            )
        self.plan = None
        self.step_index = 0
        # Clears the whole intention stack, deliberately: reconsideration is a whole-activity
        # redirect, never frame-local. Popping only to the stale frame was considered and rejected
        # (ADR-0024) — `bindings`/`history` are flat on the activity with no frame ownership, so a
        # surviving parent step could read a binding produced by the sub-plan just discarded, and
        # the frame's own goal string is authored by the parent's now-stale reasoning. The
        # superseded bundle above is what recovers the lost work instead.
        self.parent_frames.clear()
        # Drop the pipeline's intermediate bindings too: they were produced by (and are only
        # meaningful within) the plan being discarded.
        self.bindings.clear()
        # Reconsideration baseline/verdict are coupled to the discarded plan (ADR-0024): drop them
        # so the next plan re-baselines against its own starting world rather than a stale one.
        self.reconsider_baseline = None
        self.reconsider_verdict = None
        # Runaway-replan bookkeeping. Counting only — the cap, and what to do when it trips, belong
        # to the Reason strategy (as with the sub-goal depth breaker), not to a value type. Progress
        # since the last replan forgives everything before it: only replans that got *nowhere*
        # accumulate.
        if self._progressed_since_replan():
            self.replan_trail.clear()
        self.replan_trail.append(defect)
        self.replan_history_mark = len(self.history)
        self.discard_inference()
        if was_inferring:
            self.state = ActivityState.READY
