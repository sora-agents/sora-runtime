"""Deterministic tests for the ARE *dynamic*-scenario interrupt seam — over fakes, no model.

The dynamic scenario's follow-up (a mid-run Monday -> Tuesday change) reaches the agent as a **new
inbound email**, and reconsideration is triggered by a hard interrupt rather than a phase strategy:

* ``MailDiffInterruptPolicy`` raises an interrupt on a genuinely new INBOX email id (read from the
  ``state_changed`` signal payload) — and crucially NOT on the baseline inbox, NOT the agent's own
  reply landing in SENT (the loop a bare signal-count trigger caused), and NOT a non-inbox signal.
* ``ReconsiderInterruptHandler`` routes that interrupt: clear a live activity's plan (the default
  Reason then re-infers) or spawn one corrective activity when the goal completed; a user stop
  is delegated to the runtime default.

The real ARE Environment + real Claude version is the skip-gated ``test_are_sim_reproduction.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from examples.are.sim.email_calendar.strategies import (
    _CORRECTIVE_GOAL,
    MailDiffInterruptPolicy,
    ReconsiderInterruptHandler,
    reconciling_plan_prompt,
)

from sora.action import default_action_registry
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReasonStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    Strategies,
)
from sora.transport import InProcessTransport
from sora.types import InputWait, InterruptRequest, PendingInference, Plan, Signal, Step

_GOAL = "schedule the team sync Alice emailed about, then reply to her"


def _email_state(*, inbox: list[str], sent: list[str] | None = None) -> dict[str, Any]:
    """The ARE ``EmailClientApp.get_state()`` shape the policy reads — INBOX + SENT folders."""
    return {
        "folders": {
            "INBOX": {"folder_name": "INBOX", "emails": [{"email_id": i} for i in inbox]},
            "SENT": {"folder_name": "SENT", "emails": [{"email_id": i} for i in (sent or [])]},
        }
    }


def _state_changed(*, inbox: list[str], sent: list[str] | None = None) -> Signal:
    """The ``state_changed`` signal the ARE email tool pushes on a state diff."""
    return Signal(
        "state_changed",
        {"app": "EmailClientApp", "value": _email_state(inbox=inbox, sent=sent)},
    )


def _wm() -> WorkingMemory:
    return WorkingMemory(registry=EnvironmentRegistry(adapters={}))


def _cycle(tmp_path: Path) -> tuple[DecisionCycle, WorkingMemory]:
    registry = EnvironmentRegistry(adapters={})
    working = WorkingMemory(registry=registry)
    cycle = DecisionCycle(
        strategies=Strategies(
            observe=DefaultObserveStrategy(),
            reflect=DefaultReflectStrategy(),
            situate=DefaultSituateStrategy(),
            reason=DefaultReasonStrategy(),
            act=DefaultActStrategy(),
        ),
        communication=InProcessTransport(),
        actions=default_action_registry(),
        registry=registry,
        working=working,
        semantic=SemanticMemory(FileMemoryBackend(tmp_path / "semantic")),
        procedural=ProceduralMemory(FileMemoryBackend(tmp_path / "procedural")),
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "episodic")),
    )
    return cycle, working


# --------------------------------------------------------------------------------------------------
# MailDiffInterruptPolicy: a new INBOX id fires once; the baseline, a SENT self-write, a non-inbox
# signal do not
# --------------------------------------------------------------------------------------------------


def test_policy_baseline_does_not_fire_but_a_new_inbound_does() -> None:
    policy = MailDiffInterruptPolicy()
    wm = _wm()

    # First observation is the baseline (the original task) — not a follow-up.
    assert policy.decide("EmailClientApp", _state_changed(inbox=["orig"]), wm) is None

    # A follow-up grows the inbox -> a hard interrupt carrying the new id.
    request = policy.decide("EmailClientApp", _state_changed(inbox=["orig", "followup"]), wm)
    assert isinstance(request, InterruptRequest)
    assert request.signal.name == "new_inbound_email"
    assert request.signal.payload["email_ids"] == ["followup"]

    # Dedup: the same inbox state does not fire again.
    assert policy.decide("EmailClientApp", _state_changed(inbox=["orig", "followup"]), wm) is None


def test_policy_ignores_the_agents_own_reply_landing_in_sent() -> None:
    # The self-write bug: the agent's reply lands in SENT, not INBOX. A bare signal trigger looped
    # here; the INBOX-id diff must ignore it (INBOX unchanged -> no new id).
    policy = MailDiffInterruptPolicy()
    wm = _wm()
    assert policy.decide("EmailClientApp", _state_changed(inbox=["orig"]), wm) is None
    fired = policy.decide("EmailClientApp", _state_changed(inbox=["orig"], sent=["myreply"]), wm)
    assert fired is None


def test_policy_ignores_a_non_inbox_signal() -> None:
    # A calendar (or any non-email) state_changed carries no INBOX and must not fire or disturb the
    # baseline of the inbox diff.
    policy = MailDiffInterruptPolicy()
    wm = _wm()
    assert (
        policy.decide("CalendarApp", Signal("state_changed", {"value": {"events": []}}), wm) is None
    )
    assert policy.decide("EmailClientApp", _state_changed(inbox=["orig"]), wm) is None  # baseline
    request = policy.decide("EmailClientApp", _state_changed(inbox=["orig", "new"]), wm)
    assert request is not None  # still fires normally afterwards


# --------------------------------------------------------------------------------------------------
# ReconsiderInterruptHandler: replan a live activity, spawn corrective when none live, delegate stop
# --------------------------------------------------------------------------------------------------


async def test_handler_clears_a_live_plan_for_reinference(tmp_path: Path) -> None:
    cycle, working = _cycle(tmp_path)
    activity = Activity(
        id="a1",
        goal=_GOAL,
        context={},
        state=ActivityState.READY,
        plan=Plan(id="p1", goal=_GOAL, steps=[Step(next_action="wait", params={})]),
        step_index=1,
    )
    working.activities["a1"] = activity

    request = InterruptRequest(Signal("new_inbound_email", {"email_ids": ["followup"]}))
    discharged = await ReconsiderInterruptHandler().handle(request, working, cycle)

    assert discharged is True
    assert activity.plan is None  # stale plan dropped -> the default Reason re-infers a fresh one
    assert activity.step_index == 0
    assert activity.state is ActivityState.READY  # still selectable
    assert len(working.activities) == 1  # no corrective spawned while an activity is live


async def test_handler_invalidates_an_in_flight_inference(tmp_path: Path) -> None:
    # A follow-up can land while the activity is RUNNING on an off-cycle infer/ground (ADR-0021).
    # The handler clears pending_inference (so the background result is discarded on resolve) and
    # grounded_params, and returns the activity to READY so the default Reason re-infers.
    cycle, working = _cycle(tmp_path)
    activity = Activity(
        id="a1",
        goal=_GOAL,
        context={},
        state=ActivityState.RUNNING,
        plan=Plan(id="p1", goal=_GOAL, steps=[Step(next_action="wait", params={})]),
        step_index=1,
        pending_inference=PendingInference(id="inf-old", kind="plan", requested_at=0.0),
        grounded_params={"stale": True},
    )
    working.activities["a1"] = activity

    request = InterruptRequest(Signal("new_inbound_email", {"email_ids": ["followup"]}))
    discharged = await ReconsiderInterruptHandler().handle(request, working, cycle)

    assert discharged is True
    assert activity.plan is None  # stale plan dropped
    assert activity.pending_inference is None  # in-flight inference invalidated (discarded later)
    assert activity.grounded_params is None  # a resolved escalation's params were stale too
    assert activity.state is ActivityState.READY  # returned from RUNNING so Reason re-infers


async def test_handler_spawns_one_corrective_when_no_live_activity(tmp_path: Path) -> None:
    cycle, working = _cycle(tmp_path)
    # The original goal already completed (terminated) — the change landed after the fact.
    working.activities["a1"] = Activity(
        id="a1", goal=_GOAL, context={}, state=ActivityState.TERMINATED
    )

    request = InterruptRequest(Signal("new_inbound_email", {"email_ids": ["followup"]}))
    discharged = await ReconsiderInterruptHandler().handle(request, working, cycle)

    assert discharged is True
    corrective = [a for a in working.activities.values() if a.goal == _CORRECTIVE_GOAL]
    assert len(corrective) == 1  # exactly one corrective activity spawned
    assert corrective[0].state is ActivityState.READY


async def test_handler_delegates_a_user_stop_to_the_default(tmp_path: Path) -> None:
    cycle, working = _cycle(tmp_path)
    activity = Activity(id="a1", goal=_GOAL, context={}, state=ActivityState.READY)
    working.activities["a1"] = activity

    request = InterruptRequest(Signal("user_stop", {}))
    discharged = await ReconsiderInterruptHandler().handle(request, working, cycle)

    assert discharged is True
    assert activity.state is ActivityState.BLOCKED  # default: paused to await input, not replanned
    assert isinstance(activity.blocked_on, InputWait)


# --------------------------------------------------------------------------------------------------
# reconciling_plan_prompt: no longer instructs focusing (the runtime auto-focuses joined tools now)
# --------------------------------------------------------------------------------------------------


def test_reconciling_prompt_no_longer_scaffolds_focus() -> None:
    # The _OBSERVE_TO_NOTICE_CHANGE fragment is retired: joining a workspace auto-focuses all its
    # tools, so the prompt no longer *directs* the model to make focus its first steps and hold it.
    # (The base prompt still lists focus as an available action, and the surviving _THREAD_READING
    # fragment still mentions the focused inbox — neither is the retired directive.) The reconcile
    # guidance (don't duplicate a stale item) stays until A5/A6 land.
    activity = Activity(id="probe", goal=_GOAL, context={})
    system, _user = reconciling_plan_prompt(activity, {})
    lowered = system.lower()
    assert "make your first steps a" not in lowered  # the retired focus-first directive
    assert "keep them focused until the task is done" not in lowered
    assert "duplicate" in lowered  # keeps the reconcile-don't-duplicate guidance
