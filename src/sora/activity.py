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
        self.discard_inference()
        if was_inferring:
            self.state = ActivityState.READY
