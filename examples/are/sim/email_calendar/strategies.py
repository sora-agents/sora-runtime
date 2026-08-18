"""Dynamic-scenario strategies for the in-process ARE showcase.

The static MCP demo (``examples/are/mcp/email_calendar``) runs one plan to completion. This dynamic
scenario's timeline fires a mid-run follow-up email that *changes the answer* (Monday -> Tuesday).
The agent reconsiders through its **own decision cycle** rather than blindly finishing the original
plan.

The **primary** reconsideration path is now the runtime's general context-adaptation mechanism
(ADR-0024), configured in ``agent.yaml`` as ``context_adaptation: before_writes`` — before each
side-effecting step Reason re-validates the plan against new perception, and a mid-run follow-up
trips it before the calendar write. The mechanism itself is pure runtime; this file contributes only
one optional, domain-shaped piece — ``InboxChangeGate``, wired as ``strategies.change_gate``. It is
the *pre-revalidation* change-gate: the runtime default trips on any observable movement (including
the agent's own reply/read-flags/calendar writes), so without it the checkpoint would spend a
revalidation on the agent's own actions. ``InboxChangeGate`` projects perception to just the INBOX
email ids, so a self-write leaves the signature unchanged (no revalidation) and only a genuine
follow-up fires — the same
efference filter ``MailDiffInterruptPolicy`` applies on the hard-interrupt path, sharing the one
``_inbox_ids_from_state`` projection.

What lives here is the **opt-in deterministic override** — a *preemptive*, no-model alternative that
also covers the post-completion case (a follow-up after the goal already finished), which the
cooperative before-writes checkpoint structurally can't. It is wired only when ``agent.yaml``
uncomments the ``interrupt_policy``/``interrupt`` lines. It is a **hard interrupt**, not a phase
strategy:

* ``MailDiffInterruptPolicy`` — an ``InterruptPolicy`` screening the ``state_changed`` signals the
  ARE tools push. It diffs the INBOX email ids carried in the signal payload against what it has
  already seen; a genuinely new inbound email raises a hard interrupt (``DecisionCycle.interrupt``),
  preempting the current phase. This is the whole reason the reconsideration is *preemptive* rather
  than only at a cycle boundary.
* ``ReconsiderInterruptHandler`` — the paired ``InterruptHandler``. On that interrupt it clears the
  in-flight activity's plan so the (default, model-backed) Reason re-infers a fresh plan against the
  now-updated observations; if the change landed *after* the goal already completed (no live
  activity), it spawns one corrective activity. Any other interrupt (a user stop) is delegated to
  the runtime default (pause to await input). Reconsideration thus lives in *one* seam — the
  interrupt handler — instead of being split across bespoke Reason/Situate strategies.

Why *inbound-email ids* and not "a signal arrived": the agent writes to the very tool it watches
(its reply), and every write emits a ``state_changed`` — so a bare "a signal arrived -> reconsider"
trigger fires on the agent's *own* actions, forever (reply -> signal -> re-plan -> reply -> ...).
Diffing the **INBOX** ids sidesteps that: a follow-up grows the inbox, while the agent's
reply lands in SENT, so a self-write never changes the set and never fires. This is example-level,
ARE-email-shaped logic (``_inbox_ids_from_signal`` knows the ``folders/INBOX/emails`` state shape);
the general fix — efference / read-write tags so *any* self-caused change is filtered — is deferred.

Timing caveat: today the ARE bridge emits ``state_changed`` from ``tool.observe()``, i.e. *during*
the Observe phase (Observe-cadence, for determinism), not off a background thread. So the interrupt
fires inside the current tick's Observe and aborts that tick before Reason/Act. Reason's model calls
already run off-cycle (``_infer_``/``_ground_`` — ADR-0021), so an inference may be in flight from
an earlier tick when the interrupt lands; the handler invalidates it (``pending_inference`` cleared)
and its result is discarded when it resolves, rather than writing a stale plan. Making the ARE push
itself off-cycle (a genuinely asynchronous signal source) is separately deferred.

Observation precondition — met by the runtime now, not the prompt. Observable properties/signals
are only produced for a *focused* tool (``DefaultObserveStrategy``), and the follow-up path is dead
without them (no ``state_changed`` from the inbox -> the policy never fires; no view of what the
agent already created). ``JoinAction`` now auto-focuses every tool of a joined workspace, so this
holds mechanically — the plan no longer has to emit a ``focus`` step. That auto-focus is a
*temporary* fallback (the goal is reliable intentional focus/unfocus); until then
``reconciling_plan_prompt`` no longer scaffolds it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sora.action import CreateActivityAction
from sora.activity import ActivityState
from sora.memory import default_plan_prompt
from sora.strategies import DefaultInterruptHandler
from sora.types import InterruptRequest, Signal

if TYPE_CHECKING:
    from sora.activity import Activity
    from sora.cycle import DecisionCycle
    from sora.manual import Manual
    from sora.memory import PerceptSnapshot, WorkingMemory
    from sora.perception import Message

log = logging.getLogger("examples.are.sim.email_calendar")

# A distinct goal for the corrective activity so DefaultSituateStrategy (dedups by goal) treats
# it as new work with its own fresh plan, not the already-completed scheduling activity.
_CORRECTIVE_GOAL = (
    "Re-check the inbox for changes to the team sync you scheduled and reconcile the calendar "
    "event and your reply if a follow-up changed the plan."
)

_NEW_INBOUND = "new_inbound_email"  # the interrupt signal the policy raises and the handler routes


def _inbox_ids_from_state(state: object) -> frozenset[str] | None:
    """The INBOX email ids in an ARE ``EmailClientApp`` state, or None if this state has no inbox
    (a non-email tool's state, or a malformed value). State shape:
    ``{"folders": {"INBOX": {"emails": [{"email_id": ...}]}}}``. Scoping to INBOX is the whole point
    — the agent's own outbound reply lands in SENT, so it never grows this set. This is the shared
    ``state -> id-set`` projection behind both the hard-interrupt policy (fed the signal payload's
    ``value``) and the cooperative change-gate (fed an observable ``state`` property's value)."""
    if not isinstance(state, dict):
        return None
    folders = state.get("folders")
    inbox = folders.get("INBOX") if isinstance(folders, dict) else None
    emails = inbox.get("emails") if isinstance(inbox, dict) else None
    if not isinstance(emails, list):
        return None
    return frozenset(str(e["email_id"]) for e in emails if isinstance(e, dict) and "email_id" in e)


def _inbox_ids_from_signal(signal: Signal) -> frozenset[str] | None:
    """The INBOX email ids carried in a ``state_changed`` signal payload (``{"value": state}``), or
    None if it carries no inbox. A thin wrapper over the shared ``_inbox_ids_from_state`` projection
    — the hard-interrupt path unwraps the fat signal's ``value``."""
    return _inbox_ids_from_state(signal.payload.get("value"))


class MailDiffInterruptPolicy:
    """An ``InterruptPolicy`` that preempts on a genuinely new inbound email. Reads INBOX ids
    from the ``state_changed`` signal payload and diffs them against what it has already seen, so a
    follow-up that grows the inbox raises a hard interrupt while the agent's own reply (landing in
    SENT, leaving the INBOX unchanged) never does — the structural self-write filter. The first
    non-empty observation is the *baseline*, not a follow-up: only mail on top of
    it fires. Stateful, so it also dedups — each new id fires exactly once."""

    def __init__(self) -> None:
        self._seen: frozenset[str] = frozenset()

    def decide(self, source: str, signal: Signal, wm: WorkingMemory) -> InterruptRequest | None:
        current = _inbox_ids_from_signal(signal)
        if current is None:
            return None  # not an inbox-bearing state_changed signal
        new_inbound = current - self._seen
        had_baseline = bool(self._seen)  # first non-empty inbox is the baseline, not a follow-up
        self._seen = self._seen | current
        if had_baseline and new_inbound:
            log.info("interrupt-policy: new inbound email %s -> preempt", sorted(new_inbound))
            return InterruptRequest(Signal(_NEW_INBOUND, {"email_ids": sorted(new_inbound)}))
        return None


class InboxChangeGate:
    """A domain ``ChangeGate`` for the cooperative context-adaptation path (ADR-0024), the
    counterpart to ``MailDiffInterruptPolicy`` on the hard-interrupt path. The runtime default
    ``PerceptionSignatureGate`` trips on *any* observable movement — including the agent's own
    reply, read-flags, and calendar writes, each of which mutates a watched ``state`` property — so
    the before-writes checkpoint would spend a revalidation on the agent's own actions. This gate
    projects perception onto only its externally-meaningful part: the union of INBOX email ids
    across every observable ``state`` property, via the same ``_inbox_ids_from_state`` projection
    the interrupt policy uses. A self-write (reply -> SENT, a read-flag flip, a calendar add) leaves
    the INBOX id-set unchanged, so the signature is equal and the revalidation is skipped; a genuine
    follow-up grows the set and the checkpoint fires. Stateless — the baseline lives on the
    activity, not here."""

    def signature(self, wm: WorkingMemory) -> object:
        ids: frozenset[str] = frozenset()
        for (_source, name), percept in wm.properties.items():
            if name != "state":
                continue
            inbox = _inbox_ids_from_state(getattr(percept.payload, "value", None))
            if inbox is not None:
                ids |= inbox
        return ids


class ReconsiderInterruptHandler:
    """The ``InterruptHandler`` paired with ``MailDiffInterruptPolicy``. Routes a new-inbound-email
    interrupt into the agent's own decision cycle: every live (non-terminated) activity has its plan
    cleared so the default Reason re-infers against the updated observations; if the change landed
    after the goal completed (no live activity), one corrective activity is spawned. Any other
    interrupt — a user stop — is delegated to the runtime default (pause to await input).

    Because infer/ground now run off-cycle (ADR-0021), an activity may be RUNNING on an in-flight
    inference when the interrupt lands. Clearing that ``pending_inference`` invalidates it — the
    background call still finishes (an LLM call can't be cut mid-generation) but its result is
    discarded on resolve (its id no longer matches the live one) — and the activity returns to READY
    so it re-infers against the now-updated observations. An inference has no external side effect,
    so dropping it is safe; an in-flight *external* op is left running (the base handler's
    invariant), only its stale plan cleared."""

    def __init__(self) -> None:
        self._default = DefaultInterruptHandler()

    async def handle(
        self, request: InterruptRequest, wm: WorkingMemory, cycle: DecisionCycle
    ) -> bool:
        if request.signal.name != _NEW_INBOUND:
            return await self._default.handle(request, wm, cycle)
        live = [a for a in wm.activities.values() if a.state is not ActivityState.TERMINATED]
        if live:
            for activity in live:
                # Drop the stale plan (and any parked/in-flight deliberation) -> the default Reason
                # re-infers a fresh one. reset_for_replan invalidates a RUNNING infer/ground back to
                # READY (side-effect-free, discarded on resolve); an external op is left running.
                activity.reset_for_replan()
            return True
        # The change landed after the goal completed: no live activity to replan -> spawn corrective
        # work (a distinct goal so DefaultSituateStrategy treats it as new, its own fresh plan).
        create = cycle.actions.internal(CreateActivityAction.name)
        await create.execute(cycle, goal=_CORRECTIVE_GOAL, context={})
        return True


# The reconciling plan prompt appends dynamic-environment guidance to the default planning content.
# It is split into fragments with *different fates*, so the "scaffolding vs. domain knowledge" seam
# is explicit rather than buried in one blob:
#
#   _RECONCILE_AGAINST_OBSERVED — SCAFFOLDING for a runtime gap (limitation 1): no guarded/skip-if-
#       empty step, so the prompt must tell the model to delete/update only a stale item it can
#       currently see. Domain-neutral RULE; email/calendar only in the examples. Retired by guarded
#       steps / skip-invoke-on-null.
#   _THREAD_READING — NOT scaffolding: genuine email-thread domain knowledge (a follow-up is usually
#       a partial correction; read every relevant message, not just the top search hit). This does
#       not belong in a runtime prompt — its home is the email-client Manual (or semantic memory),
#       so it travels with the tool instead of being re-tuned per example. It lives here only
#       because the are-sim adapter *synthesizes* manuals and has no hand-authored one to pair
#       (ADR-0015);
#       relocating it is the tracked follow-up.

_RECONCILE_AGAINST_OBSERVED = (
    "You may be re-planning AFTER earlier steps already took effect: an item may already have been "
    "created, or a message already sent, for the OLD answer — leaving it in place would be a "
    "duplicate or an obsolete artifact (two meetings on the calendar, an outdated reply). "
    "Reconcile against what you can CURRENTLY SEE in the observed state. Every step you plan WILL "
    "run — there is no 'skip if empty' and no conditionals — so only plan to DELETE or UPDATE a "
    "stale item that is actually visible in the current state right now, referencing its id from a "
    "fresh search/list step. If no stale item is visible, do NOT plan a removal; just create or "
    "correct what the new answer needs. If the current state already satisfies the goal, plan just "
    "a short send confirming.\n"
)

# Domain knowledge, quarantined — belongs in the email-client Manual (see the note above).
_THREAD_READING = (
    "A follow-up in a thread like this is often a short correction, not a full restatement: it may "
    "mention only what changed (e.g. the day) and say nothing about details that did not change "
    "(duration, attendees, location, ...). The most recently arrived email is not necessarily the "
    "most complete one — do not treat whichever one a search happens to rank first as the whole "
    "story. When your CURRENT observations already show more than one relevant email in the thread "
    "(e.g. a follow-up has arrived and the inbox you focused now holds both the original and the "
    "correction), plan a `get_email_by_id`-style step for EACH of them, not just the top one, so "
    "both are in front of you when you decide. But do NOT speculatively read results that may not "
    "exist: if only one relevant email is visible to you right now, plan exactly ONE read (the top "
    "result). You never need to pre-plan for a follow-up that has not arrived — when new mail "
    "lands the runtime re-plans automatically, and that fresh plan will see and read every email "
    "then present."
)

_RECONCILE_INSTRUCTION = _RECONCILE_AGAINST_OBSERVED + _THREAD_READING


def reconciling_plan_prompt(
    activity: Activity,
    tools: dict[str, Manual],
    observed: PerceptSnapshot | None = None,
    messages: list[Message] | None = None,
) -> tuple[str, str]:
    """A commitment-aware ``PlanPrompt``: the default planning content plus an instruction to
    reconcile against the *observed* current world — deleting/updating only a stale item that is
    actually visible, never blindly — instead of assuming a fresh start, and to read every relevant
    email in a thread rather than assuming the most recent search hit is the complete request (a
    follow-up correction typically omits whatever didn't change). That last clause is the
    ``_THREAD_READING`` fragment — email domain knowledge quarantined here until it can move to the
    email-client Manual (ADR-0015); the first is gap-scaffolding for limitation 1. The focus-first
    scaffolding (``_OBSERVE_TO_NOTICE_CHANGE``) is gone — ``JoinAction`` auto-focuses joined tools
    now, so perception no longer hinges on a model focus step. Wired via ``agent.yaml``'s
    ``procedural.plan_prompt``; the ``{"steps": [...]}`` response contract is unchanged."""
    system, user = default_plan_prompt(activity, tools, observed, messages)
    return system + _RECONCILE_INSTRUCTION, user
