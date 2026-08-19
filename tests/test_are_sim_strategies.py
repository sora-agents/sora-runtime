"""Deterministic tests for the ARE *dynamic*-scenario reconsideration paths — over fakes, no model.

The dynamic scenario's follow-up (a mid-run Monday -> Tuesday change) reaches the agent as a **new
inbound email**. Two reconsideration paths are covered:

* **Primary — context-adaptation (ADR-0024).** `agent.yaml` sets `context_adaptation:
  before_writes`; the follow-up moves perception, so before the calendar write Reason's change-gate
  goes hot and a revalidation (a fake, here) re-checks the plan — "invalid" re-plans, "valid" runs.
  General runtime machinery, no example code (the last section below).
* **Opt-in override — the MailDiff interrupt seam.** ``MailDiffInterruptPolicy`` raises a hard
  interrupt on a genuinely new INBOX email id (read off the emitting tool, since the
  ``state_changed`` signal is a bare event and ``wm.properties`` is still a snapshot behind at push
  time) — NOT the baseline inbox, NOT the agent's own reply landing in SENT, NOT a non-inbox tool —
  and
  ``ReconsiderInterruptHandler`` routes it: clear a live activity's plan (Reason re-infers) or spawn
  one corrective activity when the goal completed; a user stop is delegated to the runtime default.

The real ARE Environment + real Claude version is the skip-gated ``test_are_sim_reproduction.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from examples.are.sim.email_calendar.strategies import (
    _CORRECTIVE_GOAL,
    InboxChangeGate,
    MailDiffInterruptPolicy,
    ReconsiderInterruptHandler,
    reconciling_plan_prompt,
)

from fakes import FakeAdapter, FakeLLMClient, FakeTool, FakeWorkspace
from sora.action import RevalidateAction, default_action_registry, invoke_step
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, WorkspaceOrigin
from sora.manual import Manual, OperationSpecification
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from sora.perception import Percept
from sora.strategies import (
    BeforeWrites,
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReasonStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    Strategies,
    TickResult,
    _perception_signature,
)
from sora.transport import InProcessTransport
from sora.types import (
    InputWait,
    InterruptRequest,
    ObservableProperty,
    PendingInference,
    Plan,
    Signal,
    Step,
    SupersededPlan,
)

_GOAL = "schedule the team sync Alice emailed about, then reply to her"


def _email_state(*, inbox: list[str], sent: list[str] | None = None) -> dict[str, Any]:
    """The ARE ``EmailClientApp.get_state()`` shape the policy reads — INBOX + SENT folders."""
    return {
        "folders": {
            "INBOX": {"folder_name": "INBOX", "emails": [{"email_id": i} for i in inbox]},
            "SENT": {"folder_name": "SENT", "emails": [{"email_id": i} for i in (sent or [])]},
        }
    }


def _state_changed(app: str = "EmailClientApp") -> Signal:
    """The ``state_changed`` event the ARE tools push on a state diff — identity only, no state
    (ADR-0004: a signal never duplicates the observable property it accompanies)."""
    return Signal("state_changed", {"app": app})


def _focus(wm: WorkingMemory, tool_id: str, state: Any) -> None:
    """Put a focused tool holding `state` in its `state` observable into working memory. Calling it
    again replaces the tool with one holding the new state — which is what the real adapter does
    (``_AreTool.observe`` records the new state *before* pushing). The policy reads the tool, so
    this, not ``wm.properties``, is where a follow-up has to land for it to be seen."""
    wm.focused_tools[tool_id] = FakeTool(tool_id, properties=[ObservableProperty("state", state)])


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
# InboxChangeGate: the cooperative-path efference filter — the change-gate signature tracks only the
# INBOX id-set, so a SENT self-write / a non-email state never moves it, but a new inbound does
# --------------------------------------------------------------------------------------------------


def _state_property(wm: WorkingMemory, tool_id: str, state: Any) -> None:
    wm.properties[(tool_id, "state")] = Percept(tool_id, ObservableProperty("state", state), 0.0)


def test_inbox_change_gate_projects_to_inbox_ids() -> None:
    wm = _wm()
    _state_property(wm, "email", _email_state(inbox=["e1", "e2"]))
    assert InboxChangeGate().signature(wm) == frozenset({"e1", "e2"})


def test_inbox_change_gate_ignores_a_sent_self_write() -> None:
    # The whole point: the agent's own reply lands in SENT and never grows the INBOX id-set, so the
    # signature is unchanged and the before-writes checkpoint stays cold on a self-write.
    gate = InboxChangeGate()
    wm = _wm()
    _state_property(wm, "email", _email_state(inbox=["e1"]))
    before = gate.signature(wm)
    _state_property(wm, "email", _email_state(inbox=["e1"], sent=["reply"]))
    assert gate.signature(wm) == before


def test_inbox_change_gate_moves_on_a_new_inbound() -> None:
    gate = InboxChangeGate()
    wm = _wm()
    _state_property(wm, "email", _email_state(inbox=["e1"]))
    before = gate.signature(wm)
    _state_property(wm, "email", _email_state(inbox=["e1", "e2"]))  # a follow-up arrived
    assert gate.signature(wm) != before


def test_inbox_change_gate_ignores_non_email_state() -> None:
    # A calendar tool's state has no INBOX -> projected away, so a self calendar write never fires.
    wm = _wm()
    _state_property(wm, "cal", {"events": [{"event_id": "ev1"}]})
    assert InboxChangeGate().signature(wm) == frozenset()


# --------------------------------------------------------------------------------------------------
# MailDiffInterruptPolicy: a new INBOX id fires once; the baseline, a SENT self-write, a non-inbox
# signal do not
# --------------------------------------------------------------------------------------------------


def test_policy_baseline_does_not_fire_but_a_new_inbound_does() -> None:
    policy = MailDiffInterruptPolicy()
    wm = _wm()

    # First observation is the baseline (the original task) — not a follow-up.
    _focus(wm, "EmailClientApp", _email_state(inbox=["orig"]))
    assert policy.decide("EmailClientApp", _state_changed(), wm) is None

    # A follow-up grows the inbox -> a hard interrupt carrying the new id.
    _focus(wm, "EmailClientApp", _email_state(inbox=["orig", "followup"]))
    request = policy.decide("EmailClientApp", _state_changed(), wm)
    assert isinstance(request, InterruptRequest)
    assert request.signal.name == "new_inbound_email"
    assert request.signal.payload["email_ids"] == ["followup"]

    # Dedup: the same inbox state does not fire again.
    assert policy.decide("EmailClientApp", _state_changed(), wm) is None


def test_policy_reads_the_tool_not_the_property_snapshot() -> None:
    # The regression this whole seam turns on. An InterruptPolicy is screened at push time, upstream
    # of DefaultObserveStrategy._snapshot_properties, so wm.properties still holds the PRE-change
    # world — here, the inbox before the follow-up. A policy that diffed the snapshot would find no
    # new id, and since the tool won't re-emit for an unchanged state it would never fire again:
    # silent death, not an error. Reading the tool is what makes the follow-up visible.
    policy = MailDiffInterruptPolicy()
    wm = _wm()
    _focus(wm, "EmailClientApp", _email_state(inbox=["orig"]))
    assert policy.decide("EmailClientApp", _state_changed(), wm) is None  # baseline

    _state_property(wm, "EmailClientApp", _email_state(inbox=["orig"]))  # stale snapshot
    _focus(wm, "EmailClientApp", _email_state(inbox=["orig", "followup"]))  # tool is current
    request = policy.decide("EmailClientApp", _state_changed(), wm)
    assert request is not None
    assert request.signal.payload["email_ids"] == ["followup"]


def test_policy_ignores_the_agents_own_reply_landing_in_sent() -> None:
    # The self-write bug: the agent's reply lands in SENT, not INBOX. A bare signal trigger looped
    # here; the INBOX-id diff must ignore it (INBOX unchanged -> no new id).
    policy = MailDiffInterruptPolicy()
    wm = _wm()
    _focus(wm, "EmailClientApp", _email_state(inbox=["orig"]))
    assert policy.decide("EmailClientApp", _state_changed(), wm) is None
    _focus(wm, "EmailClientApp", _email_state(inbox=["orig"], sent=["myreply"]))
    assert policy.decide("EmailClientApp", _state_changed(), wm) is None


def test_policy_ignores_a_non_inbox_tool() -> None:
    # A calendar (or any non-email) state_changed comes from a tool whose state has no INBOX, and
    # must not fire or disturb the baseline of the inbox diff.
    policy = MailDiffInterruptPolicy()
    wm = _wm()
    _focus(wm, "CalendarApp", {"events": []})
    assert policy.decide("CalendarApp", _state_changed("CalendarApp"), wm) is None

    _focus(wm, "EmailClientApp", _email_state(inbox=["orig"]))
    assert policy.decide("EmailClientApp", _state_changed(), wm) is None  # baseline
    _focus(wm, "EmailClientApp", _email_state(inbox=["orig", "new"]))
    assert policy.decide("EmailClientApp", _state_changed(), wm) is not None  # still fires after


def test_policy_ignores_an_unknown_signal_and_an_unfocused_source() -> None:
    # The payload no longer identifies the signal, so the name is the discriminator; and a signal
    # whose source isn't focused has no readable state (an unfocus racing a push) -> no fire, and
    # neither case may disturb the baseline.
    policy = MailDiffInterruptPolicy()
    wm = _wm()
    _focus(wm, "EmailClientApp", _email_state(inbox=["orig", "followup"]))
    assert policy.decide("EmailClientApp", Signal("user_stop", {}), wm) is None
    assert policy.decide("UnfocusedApp", _state_changed("UnfocusedApp"), wm) is None

    # Baseline intact: the first inbox read is still the baseline, so it doesn't fire either.
    assert policy.decide("EmailClientApp", _state_changed(), wm) is None


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


def test_reconciling_prompt_inherits_the_superseded_plan_section() -> None:
    # A custom PlanPrompt composes the default's *user* half rather than reimplementing it, so the
    # replanning context arrives without the example being touched. This is what the bundle riding
    # on the Activity buys over widening the PlanPrompt signature, which every implementor would
    # have had to adopt by hand (ADR-0024).
    activity = Activity(id="probe", goal=_GOAL, context={})
    plan = Plan(id="p", goal=_GOAL, steps=[invoke_step("Cal", "book_monday")])
    activity.superseded = SupersededPlan(plan=plan, step_index=0, parent_frames=[])

    _system, user = reconciling_plan_prompt(activity, {})

    assert "previous plan for this goal was abandoned" in user
    assert "book_monday" in user


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


# --------------------------------------------------------------------------------------------------
# The primary reconsideration path: context-adaptation (ADR-0024). agent.yaml sets
# `context_adaptation: before_writes`, so the mid-run Monday -> Tuesday follow-up trips the
# checkpoint and the plan is re-inferred — general mechanism, no MailDiff interrupt, fake re-check.
# --------------------------------------------------------------------------------------------------

_CAL_MANUAL = Manual(
    id="calendar",
    metadata={},
    description="",
    observable_properties=[],
    signals=[],
    operations=[
        OperationSpecification(
            name="add_calendar_event", description="", parameters={}, side_effecting=True
        )
    ],
)


async def _reconsidering_cycle(
    tmp_path: Path, *, verdict: str
) -> tuple[DecisionCycle, WorkingMemory]:
    """A cycle wired the way the example's agent.yaml wires it — before_writes context-adaptation —
    with a fake revalidation for the model so the Monday -> Tuesday regression is deterministic."""
    origin = WorkspaceOrigin(adapter="fake", address="fake://cal")
    tool = FakeTool("calendar", manual=_CAL_MANUAL, invoke_results={"add_calendar_event": "ok"})
    registry = EnvironmentRegistry(
        adapters={origin: FakeAdapter("fake", FakeWorkspace("cal", origin, [tool]))}
    )
    await registry.join(origin)  # populate live tools so the manual (op.side_effecting) resolves
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
        procedural=ProceduralMemory(
            FileMemoryBackend(tmp_path / "procedural"), llm=FakeLLMClient(verdict)
        ),
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "episodic")),
        reconsideration=BeforeWrites(),
    )
    return cycle, working


async def _resolve_revalidate(cycle: DecisionCycle) -> None:
    await asyncio.gather(*list(cycle.actions.internal(RevalidateAction.name)._tasks))  # type: ignore[attr-defined]
    await DefaultObserveStrategy().observe(cycle)


async def test_context_adaptation_replans_on_the_mid_run_followup(tmp_path: Path) -> None:
    # The Monday -> Tuesday regression, driven by the general mechanism rather than the interrupt.
    cycle, working = await _reconsidering_cycle(tmp_path, verdict='{"valid": false}')
    plan = Plan(id="p", goal=_GOAL, steps=[invoke_step("calendar", "add_calendar_event")])
    activity = Activity(id="a", goal=_GOAL, context={}, plan=plan, step_index=0)
    working.activities["a"] = activity
    activity.reconsider_baseline = _perception_signature(working)  # baselined in the Monday world

    # The follow-up lands mid-run: a new inbound email surfaces as a state change. The default gate
    # tracks signal arrival, not signal content, so the thin event is all it takes to go hot.
    working.signals.append(Percept("EmailClientApp", _state_changed(), 0.0))

    fired = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert (
        fired.step is None
    )  # the calendar write is NOT committed; the gate went hot -> revalidation
    assert activity.pending_inference is not None
    assert activity.pending_inference.kind == "revalidate"

    await _resolve_revalidate(cycle)  # verdict: invalid
    replanned = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert replanned.step is None
    assert activity.plan is None  # stale Monday plan dropped -> Reason re-infers against Tuesday


async def test_context_adaptation_keeps_a_still_valid_plan(tmp_path: Path) -> None:
    # Control: a "still valid" verdict commits the write — no spurious replan storm on self-writes.
    cycle, working = await _reconsidering_cycle(tmp_path, verdict='{"valid": true}')
    plan = Plan(id="p", goal=_GOAL, steps=[invoke_step("calendar", "add_calendar_event")])
    activity = Activity(id="a", goal=_GOAL, context={}, plan=plan, step_index=0)
    working.activities["a"] = activity
    activity.reconsider_baseline = _perception_signature(working)
    working.signals.append(Percept("EmailClientApp", _state_changed(), 0.0))

    await DefaultReasonStrategy().reason(
        activity, working, cycle, TickResult()
    )  # fires the revalidation
    await _resolve_revalidate(cycle)  # verdict: valid
    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert result.step == invoke_step("calendar", "add_calendar_event")  # still valid -> committed
    assert activity.plan is not None
