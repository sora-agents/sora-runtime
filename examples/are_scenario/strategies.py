"""Dynamic-scenario strategies for the in-process ARE showcase.

The static gaia2 demo (``examples/gaia2/email_calendar``) runs one plan to completion. This dynamic
scenario's timeline fires a mid-run follow-up email that *changes the answer* (Monday -> Tuesday).
Two custom strategies make the agent reconsider through its **own decision cycle** rather than
blindly finishing the original plan, handling the change whenever it lands:

* ``ReconcilingReasonStrategy`` — while an activity is in flight, a **new inbound email** re-plans
  it. It re-*infers* directly rather than letting the default advance the existing in-flight plan
  (its cheap path), replacing it with a fresh plan from the now-updated observations.
* ``CorrectiveSituateStrategy`` — if the new email lands *after* the goal already completed (an
  email arriving is only observed state, and never spawns an activity the way a USER message does),
  it spawns one fresh corrective activity so the agent still reconciles. Situate is already the
  activity-creation phase; this just triggers on observed inbox state rather than a message.

Why *inbound-email content* and not "a signal arrived": the agent writes to the very tool it watches
(its reply), and every write emits a ``state_changed`` — so a signal-count trigger re-infers on the
agent's *own* actions, forever (reply -> signal -> re-plan -> reply -> ...). Keying on the set of
**INBOX** email ids sidesteps that structurally: a follow-up grows the inbox, while the agent's
reply lands in SENT, so a self-write is invisible to the trigger. This is example-level,
ARE-email-shaped logic (`_inbound_email_ids` knows the ``folders/INBOX/emails`` state shape); the
general fix — efference/read-write tags so *any* self-caused change is filtered — is deferred.

Precondition — the plan MUST focus the tools it reconciles against: the inbox (or the re-inference
above never triggers) and every tool whose state it changes (here the calendar). Observable
properties are only snapshotted for a *focused* tool (``DefaultObserveStrategy``), so without a
``focus`` step the tool's state never reaches working memory — the agent both runs blind to a
follow-up and can't see what it already created, so it can't tell a stale item to reconcile from
none (and would blindly delete a non-existent one, since a step has no "skip if empty"). Focus is a
plan step the base planner treats as optional, so ``reconciling_plan_prompt`` asks for it
explicitly.

Coordination: exactly one path handles a given new email. Reason owns it while an activity is alive
(``CorrectiveSituateStrategy`` refuses to spawn while any activity is non-terminated); once the
activity ends, Situate spawns corrective work only for inbox ids no activity has yet accounted for.
The shared ``_SEEN_INBOUND`` set on ``activity.context`` (the runtime never writes there) hands off
between them without double-fixing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sora.action import CreateActivityAction
from sora.activity import ActivityState
from sora.memory import PerceptSnapshot, default_plan_prompt
from sora.strategies import DefaultReasonStrategy, DefaultSituateStrategy

if TYPE_CHECKING:
    from sora.activity import Activity
    from sora.cycle import DecisionCycle
    from sora.manual import Manual
    from sora.memory import WorkingMemory
    from sora.strategies import TickResult

log = logging.getLogger("examples.are_scenario")

# Inbound email ids an activity has already accounted for, stored on activity.context. Both
# strategies honor it so a given email is handled once: Reason stamps the activity it re-infers for;
# Situate refuses to spawn corrective work for an inbox id any activity (live or since-terminated)
# already accounts for.
_SEEN_INBOUND = "_seen_inbound_email_ids"

# A distinct goal for the corrective activity so DefaultSituateStrategy (which dedups by goal)
# treats it as new work with its own fresh plan, not the already-completed scheduling activity.
_CORRECTIVE_GOAL = (
    "Re-check the inbox for changes to the team sync you scheduled and reconcile the calendar "
    "event and your reply if a follow-up changed the plan."
)


def _inbound_email_ids(wm: WorkingMemory) -> frozenset[str]:
    """Email ids currently in any focused tool's INBOX (ARE ``EmailClientApp`` state shape:
    ``{"folders": {"INBOX": {"emails": [{"email_id": ...}]}}}``). Scoping to INBOX is the whole
    point — the agent's own outbound reply lands in SENT, so it never grows this set and so never
    looks like new external input. Tolerant of any non-email focused tool (contributes nothing)."""
    ids: set[str] = set()
    for percept in wm.properties.values():
        state = getattr(percept.payload, "value", None)
        if not isinstance(state, dict):
            continue
        folders = state.get("folders")
        inbox = folders.get("INBOX") if isinstance(folders, dict) else None
        emails = inbox.get("emails") if isinstance(inbox, dict) else None
        if isinstance(emails, list):
            ids.update(
                str(e["email_id"]) for e in emails if isinstance(e, dict) and "email_id" in e
            )
    return frozenset(ids)


class ReconcilingReasonStrategy:
    """Model-backed Reason (the runtime default) plus new-inbound-email-driven re-inference."""

    def __init__(self) -> None:
        self._default = DefaultReasonStrategy()

    async def reason(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        current = _inbound_email_ids(wm)
        seen: frozenset[str] = activity.context.get(_SEEN_INBOUND, frozenset())
        new_inbound = current - seen
        activity.context[_SEEN_INBOUND] = seen | current  # account for all inbox mail seen so far
        # `seen` truthiness gates the first non-empty observation: the inbox the agent starts with
        # is its *baseline* (the original task), not a follow-up — only mail on top of it is.
        if activity.plan is not None and seen and new_inbound:
            log.info("reason: new inbound email mid-plan -> re-inferring %r", activity.goal)
            catalog: dict[str, Manual] = {t.id: t.manual for t in wm.registry.all_tools()}
            observed = PerceptSnapshot(list(wm.properties.values()), list(wm.signals))
            activity.plan = await cycle.procedural.infer(activity, catalog, observed)
            activity.step_index = 0
        return await self._default.reason(activity, wm, cycle, result)


class CorrectiveSituateStrategy:
    """DefaultSituateStrategy plus: spawn a corrective activity when a new inbound email is observed
    with no live activity to replan — the change arrived after the original goal was done."""

    def __init__(self) -> None:
        self._default = DefaultSituateStrategy()

    async def situate(
        self,
        activities: list[Activity],
        wm: WorkingMemory,
        cycle: DecisionCycle,
        result: TickResult,
    ) -> TickResult:
        # Create corrective work first so the default's own selection can pick it up this cycle.
        await self._maybe_spawn_corrective(wm, cycle)
        return await self._default.situate(activities, wm, cycle, result)

    @staticmethod
    async def _maybe_spawn_corrective(wm: WorkingMemory, cycle: DecisionCycle) -> None:
        current = _inbound_email_ids(wm)
        if not current:
            return
        activities = list(wm.activities.values())
        # No prior work to correct: the initial task is handled the normal way (message -> activity)
        # and this runs *before* that creation each cycle, so bail rather than correct the baseline.
        if not activities:
            return
        if any(a.state is not ActivityState.TERMINATED for a in activities):
            return  # something in flight -> Reason reconsiders it; don't also spawn corrective work
        accounted: set[str] = set()
        for a in activities:
            accounted |= a.context.get(_SEEN_INBOUND, frozenset())
        if not (current - accounted):
            return  # every inbox email is already accounted for by some activity
        log.info("situate: new inbound email with no live activity -> spawning corrective activity")
        create = cycle.actions.internal(CreateActivityAction.name)
        await create.execute(cycle, goal=_CORRECTIVE_GOAL, context={_SEEN_INBOUND: current})


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
    "confirming."
)


def reconciling_plan_prompt(
    activity: Activity,
    tools: dict[str, Manual],
    observed: PerceptSnapshot | None = None,
) -> tuple[str, str]:
    """A commitment-aware ``PlanPrompt``: the default planning content plus an instruction to focus
    the tools it reconciles against (the inbox, so a mid-task email is observed, and any tool it
    changes, so it can see what it already created) and to reconcile against the *observed* current
    world — deleting/updating only a stale item that is actually visible, never blindly — instead of
    assuming a fresh start. Wired via ``agent.yaml``'s ``procedural.plan_prompt``; the
    ``{"steps": [...]}`` response contract is unchanged."""
    system, user = default_plan_prompt(activity, tools, observed)
    return system + _RECONCILE_INSTRUCTION, user
