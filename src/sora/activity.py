"""Activity: the sole first-class unit of work (see ADR-0002)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from sora.types import SupersededPlan  # constructed at run time by reset_for_replan

if TYPE_CHECKING:
    from sora.types import (
        CompletedOperation,
        InputWait,
        OperationAck,
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
    # The intention stack for sub-goals (ADR-0022): each entry is a suspended parent frame (its Plan
    # and the step_index of the sub-goal that pushed it). `plan`/`step_index` above are the *active*
    # frame; entering a deliberative sub-goal pushes the parent here (Observe) and exhausting a
    # sub-plan pops it, resuming the parent at the step after its sub-goal (Reason). Empty for a
    # flat plan — generalizes step_index rather than adding a separate intention type (ADR-0002).
    parent_frames: list[tuple[Plan, int]] = field(default_factory=list)
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
    # Observe when a user Message arrives. Named generally (not blocked_on_signal) to admit
    # this second variant.
    blocked_on: SignalWait | InputWait | None = None
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

    def reset_for_replan(self) -> None:
        """Drop the current plan and any in-flight/parked deliberation so Reason re-plans from
        scratch. A droppable inference in flight is invalidated and the activity returns to READY;
        an in-flight *external* op is left RUNNING (its physical side effect must complete) with
        only its stale plan cleared — it resolves to READY later and Reason re-plans then. The one
        place every plan-invalidation site (interrupt handlers, signal-driven re-planners) routes
        through, so new deliberation state can't be forgotten at one call site."""
        was_inferring = self.pending_inference is not None
        # Park what is being dropped for the *next* inference to read (ADR-0024): a blank-slate
        # replan is correct but wasteful, and the planner reuses what still applies far better than
        # the runtime could decide for it. Captured before the fields below are cleared, and only
        # when there is a plan — a reset with nothing in flight leaves any earlier bundle alone.
        if self.plan is not None:
            self.superseded = SupersededPlan(
                plan=self.plan, step_index=self.step_index, parent_frames=list(self.parent_frames)
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
        self.discard_inference()
        if was_inferring:
            self.state = ActivityState.READY
