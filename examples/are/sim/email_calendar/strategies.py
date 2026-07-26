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
fires inside the current tick's Observe and aborts that tick's Reason/Act — there is no in-flight
model call to abandon yet. The seam is the same one a genuinely asynchronous signal source (or an
off-cycle ARE push) would reuse to abandon an in-flight inference; making the ARE push off-cycle is
separately deferred.

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
    interrupt — a user stop — is delegated to the runtime default (pause to await input)."""

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
                activity.plan = (
                    None  # drop the stale plan -> the default Reason re-infers a fresh one
                )
                activity.step_index = 0
            return True
        # The change landed after the goal completed: no live activity to replan -> spawn corrective
        # work (a distinct goal so DefaultSituateStrategy treats it as new, its own fresh plan).
        create = cycle.actions.internal(CreateActivityAction.name)
        await create.execute(cycle, goal=_CORRECTIVE_GOAL, context={})
        return True


_RECONCILE_INSTRUCTION = (
    "\nThis is a DYNAMIC environment: the task can change WHILE you work — a follow-up email may "
    "arrive mid-task and change the answer (a different day, time, or attendee). To notice such a "
    "change you must keep *observing* the tools involved, so make your FIRST steps a `focus` on "
    "the email inbox (where updates arrive) AND on any other tool whose state this task changes "
    "(e.g. the calendar), and keep them focused until the task is done — an unfocused tool's state "
    "is not observed, so you will neither see a follow-up email that arrives later nor see what "
    "you have already created. "
    "You may be re-planning AFTER earlier steps already took effect: an item may already have been "
    "created, a message already sent, for the OLD answer — leaving it in place would be a "
    "duplicate or an obsolete artifact (two meetings on the calendar, an outdated reply). "
    "Reconcile against what you can CURRENTLY SEE in the observed state. Every step you plan WILL "
    "run — there is no "
    "'skip if empty' and no conditionals — so only plan to DELETE or UPDATE a stale item that is "
    "actually visible in the current state right now, referencing its id from a fresh search/list "
    "step. If no stale item is visible, do NOT plan a removal; just create or correct what the new "
    "answer needs. If the current state already satisfies the goal, plan just a short send "
    "confirming.\n"
    "A follow-up in a thread like this is often a short correction, not a full restatement: it may "
    "mention only what changed (e.g. the day) and say nothing about details that did not change "
    "(duration, attendees, location, ...). The most recently arrived email is not necessarily the "
    "most complete one — do not treat whichever one a search happens to rank first as the whole "
    "story. Before you finalize a parameter, make sure you have actually read every email in the "
    "thread that could bear on it: if a search for the topic returns more than one relevant "
    "result, plan a `get_email_by_id`-style step for each of them, not just the top one, so both "
    "the original request and the correction are in front of you when you decide."
)


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
    didn't change). Wired via ``agent.yaml``'s ``procedural.plan_prompt``; the ``{"steps": [...]}``
    response contract is unchanged."""
    system, user = default_plan_prompt(activity, tools, observed)
    return system + _RECONCILE_INSTRUCTION, user
