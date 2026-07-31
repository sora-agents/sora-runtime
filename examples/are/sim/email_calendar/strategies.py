"""Dynamic-scenario strategies for the in-process ARE showcase.

The static MCP demo (``examples/are/mcp/email_calendar``) runs one plan to completion. This dynamic
scenario's timeline fires a mid-run follow-up email that *changes the answer* (Monday -> Tuesday).
The agent reconsiders through its **own decision cycle** rather than blindly finishing the original
plan — and the trigger for that reconsideration is a **hard interrupt**, not a phase strategy:

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

Precondition — the plan MUST focus the tools it reconciles against: the inbox (or ``state_changed``
signals never carry INBOX state, so the policy never fires) and every tool whose state it changes
(here the calendar). Observable properties/signals are only produced for a *focused* tool
(``DefaultObserveStrategy``), so without a ``focus`` step the agent runs
blind to a follow-up and can't see what it already created (and would blindly delete a non-existent
item, since a step has no "skip if empty"). Focus is optional to the base planner,
so ``reconciling_plan_prompt`` asks for it explicitly.
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

log = logging.getLogger("examples.are.sim.email_calendar")

# A distinct goal for the corrective activity so DefaultSituateStrategy (dedups by goal) treats
# it as new work with its own fresh plan, not the already-completed scheduling activity.
_CORRECTIVE_GOAL = (
    "Re-check the inbox for changes to the team sync you scheduled and reconcile the calendar "
    "event and your reply if a follow-up changed the plan."
)

_NEW_INBOUND = "new_inbound_email"  # the interrupt signal the policy raises and the handler routes


def _inbox_ids_from_signal(signal: Signal) -> frozenset[str] | None:
    """The INBOX email ids in a ``state_changed`` signal's payload, or None if it carries no inbox
    (a non-email tool's state, or a malformed payload). ARE ``EmailClientApp`` state shape:
    ``{"value": {"folders": {"INBOX": {"emails": [{"email_id": ...}]}}}}``. Scoping to INBOX is the
    whole point — the agent's own outbound reply lands in SENT, so it never grows this set."""
    value = signal.payload.get("value")
    if not isinstance(value, dict):
        return None
    folders = value.get("folders")
    inbox = folders.get("INBOX") if isinstance(folders, dict) else None
    emails = inbox.get("emails") if isinstance(inbox, dict) else None
    if not isinstance(emails, list):
        return None
    return frozenset(str(e["email_id"]) for e in emails if isinstance(e, dict) and "email_id" in e)


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
# It is split into three fragments with *different fates*, so the "scaffolding vs. domain knowledge"
# seam is explicit rather than buried in one blob:
#
#   _OBSERVE_TO_NOTICE_CHANGE  — SCAFFOLDING for a runtime gap (limitation 4): perception is gated
#       on a model-driven `focus` step, and the base prompt motivates focus only to "read what you
#       need now, unfocus when done" — it has no notion of holding focus to catch a *future* change
#       or to re-see your own writes across a replan. Domain-neutral RULE; email/calendar only in
#       the examples. Candidate to promote (example-free) into PLAN_SYSTEM_PROMPT, or to retire once
#       the runtime auto-focuses tools a plan touches.
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

_OBSERVE_TO_NOTICE_CHANGE = (
    "\nThis is a DYNAMIC environment: the task can change WHILE you work — a new input may arrive "
    "mid-task and change the answer. To notice such a change you must keep *observing* the tools "
    "involved, because an unfocused tool's state is not observed. So make your FIRST steps a "
    "`focus` on every channel where an update could arrive (here, the email inbox) AND on every "
    "tool whose state this task changes (here, the calendar), and keep them focused until the task "
    "is done — otherwise you will neither see a later change nor see what you have already "
    "created.\n"
)

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

_RECONCILE_INSTRUCTION = _OBSERVE_TO_NOTICE_CHANGE + _RECONCILE_AGAINST_OBSERVED + _THREAD_READING


def reconciling_plan_prompt(
    activity: Activity,
    tools: dict[str, Manual],
    observed: PerceptSnapshot | None = None,
) -> tuple[str, str]:
    """A commitment-aware ``PlanPrompt``: the default planning content plus an instruction to focus
    the tools it reconciles against (the inbox, so a mid-task email is observed, and any tool it
    changes, so it can see what it already created), to reconcile against the *observed* current
    world — deleting/updating only a stale item that is actually visible, never blindly — instead of
    assuming a fresh start, and to read every relevant email in a thread rather than assuming the
    most recent search hit is the complete request (a follow-up correction typically omits whatever
    didn't change). That last clause is the ``_THREAD_READING`` fragment — email domain knowledge
    quarantined here until it can move to the email-client Manual (ADR-0015); the other two
    fragments are gap-scaffolding for limitations 1 and 4. Wired via ``agent.yaml``'s
    ``procedural.plan_prompt``; the ``{"steps": [...]}`` response contract is unchanged."""
    system, user = default_plan_prompt(activity, tools, observed)
    return system + _RECONCILE_INSTRUCTION, user
