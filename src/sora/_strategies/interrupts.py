"""Hard-interrupt policies and handlers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from sora.activity import ActivityState
from sora.types import (
    USER_STOP,
    InputWait,
)

if TYPE_CHECKING:
    from sora.cycle import DecisionCycle
    from sora.memory import WorkingMemory
    from sora.types import InterruptRequest, Signal

log = logging.getLogger("sora.strategies")


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
