"""Activity: the sole first-class unit of work (see ADR-0002)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

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
        self.plan = None
        self.step_index = 0
        # Clears the whole intention stack: right for a whole-activity redirect (the only callers,
        # interrupt handlers and signal-driven re-planners, invalidate the suspended parents too).
        # A frame-local sub-goal replan (keep the parents, re-infer only the active sub-plan for
        # its own goal — recoverable from parent_frames[-1]'s (parent_plan, subgoal_index)) is a
        # separate future method; there's no trigger for it yet (a failed sub-plan inference
        # currently terminates the activity rather than replanning the frame).
        self.parent_frames.clear()
        # Drop the pipeline's intermediate bindings too: they were produced by (and are only
        # meaningful within) the plan being discarded.
        self.bindings.clear()
        self.discard_inference()
        if was_inferring:
            self.state = ActivityState.READY
