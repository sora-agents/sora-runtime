"""Plan-reconsideration policies and perception change gates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from sora.action import (
    FocusAction,
    InvokeAction,
    JoinAction,
    LeaveAction,
    UnfocusAction,
)
from sora.references import (
    _manual_for,
)
from sora.types import (
    OPERATION_NAME,
    TOOL_ID,
    WAIT,
    Step,
)

if TYPE_CHECKING:
    from sora.memory import WorkingMemory

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
