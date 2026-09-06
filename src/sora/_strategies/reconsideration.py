"""Plan-reconsideration policies and perception change gates."""

from __future__ import annotations

from enum import Enum
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


class ReconsiderationOutcome(Enum):
    """What a policy wants done at a commitment point whose change-gate has gone hot.

    Three-valued rather than a bool because the *resolution* is a second axis from the *timing*: a
    model revalidation is only worth its price when the percepts it will be shown carry enough to
    answer with. In an environment whose perception is announcements rather than evidence (a signal
    that says the world moved but not to what), the judge is asked to rule on the thing that woke
    it and can only say "maybe" — so the call buys nothing the gate did not already know, and
    discarding the plan outright is both cheaper and more honest.
    """

    SKIP = "skip"  # proceed: this step is not a commitment point worth guarding
    CHECK = "check"  # spend a model revalidation and act on its verdict
    REPLAN = "replan"  # discard the plan on a relevant change, with no model call


class ReconsiderationPolicy(Protocol):
    def decide(self, side_effecting: bool | None) -> ReconsiderationOutcome:
        """Given the side-effecting-ness of the step Reason is about to commit (True write, False
        read, None unknown), decide whether to guard it and, if so, how."""
        ...


class NoneReconsideration:
    """``context_adaptation: none`` — never reconsider on ambient percepts (blind commitment).
    Failure-driven re-planning stays orthogonal and always on."""

    def decide(self, side_effecting: bool | None) -> ReconsiderationOutcome:
        return ReconsiderationOutcome.SKIP


class BeforeWrites:
    """``context_adaptation: before_writes`` (the default) — check before a side-effecting step,
    where acting on a stale plan does damage. Skips reads (side_effecting is False); an unknown
    (None) is treated as a write, so it is checked (conservative)."""

    def decide(self, side_effecting: bool | None) -> ReconsiderationOutcome:
        if side_effecting is False:
            return ReconsiderationOutcome.SKIP
        return ReconsiderationOutcome.CHECK


class BeforeEachOp:
    """``context_adaptation: before_each_op`` — check before EVERY external step, read or write.
    Maximum caution; still op-gated, so it skips planning/grounding/waiting cycles."""

    def decide(self, side_effecting: bool | None) -> ReconsiderationOutcome:
        return ReconsiderationOutcome.CHECK


class ReplanOnChange:
    """``context_adaptation: replan_on_change`` — guard the same points ``before_writes`` guards,
    but resolve a hot gate by discarding the plan outright instead of asking a model whether it
    still holds.

    For environments whose perception is *announcement* rather than *evidence*. Where a tool
    publishes observable state, a revalidation is worth its price: shown the changed state, the
    judge can genuinely answer "that email is a newsletter, proceed" and save the replan. Where the
    only percept is a signal saying something moved — no identifiers, no values — the judge is
    handed the announcement that woke it and nothing else, so it can rarely do better than "maybe",
    and the call is spent to learn what the gate already established. Replanning directly costs one
    fewer model call and, unlike the judge, cannot rubber-stamp a plan it had no evidence about.

    The trade is real and runs the other way on a rich environment: this policy never proceeds on a
    change a judge would have dismissed, so on a noisy world it replans where ``before_writes``
    would not. That is why the discard is scoped to changes on tools the plan actually references
    (see `relevant_change_since`) — without that scope an unrelated app's signal would discard a
    plan that never touched it.

    **Precondition, and the reason this is not the default:** it is unsound on an adapter whose
    signals include the agent's *own* writes. Such a signal lands on a tool the plan references by
    construction, so every write would discard the plan that issued it. A judge filters that out by
    reading the change; a mechanical trigger cannot. Use this only where perception is
    environment-initiated, or where self-writes are otherwise excluded.
    """

    def decide(self, side_effecting: bool | None) -> ReconsiderationOutcome:
        if side_effecting is False:
            return ReconsiderationOutcome.SKIP
        return ReconsiderationOutcome.REPLAN


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


def relevant_change_since(
    wm: WorkingMemory, cursor: tuple[int, int] | None, sources: set[str] | None
) -> bool:
    """Did a percept from one of ``sources`` land at or after ``cursor``?

    The scope a mechanical discard needs and a judged one deliberately does not have: a model
    revalidation is shown the whole world *because* it has to reason about the change that woke it
    (see the checkpoint's own note), whereas a discard with no reader would otherwise throw away a
    calendar plan because a shopping app spoke.

    Fails OPEN three ways, all for the same reason — a spurious replan costs one plan call, a
    missed one costs the scenario. No cursor (the plan predates scope tracking) or no scope (the
    activity has no plan, so every tool is potentially in play) means relevant. So does a cursor the
    retention cap has outrun: entries between it and the oldest retained percept are gone, and
    absence of evidence there is not evidence of absence."""
    if cursor is None or sources is None:
        return True
    signals_from, changes_from = cursor
    for log, appended, since in (
        (wm.signals, wm.signals_appended, signals_from),
        (wm.property_changes, wm.property_changes_appended, changes_from),
    ):
        first = appended - len(log)
        if first > since:
            return True  # the cap evicted part of the window; cannot rule the change out
        if any(
            first + offset >= since and percept.source in sources
            for offset, percept in enumerate(log)
        ):
            return True
    return False
