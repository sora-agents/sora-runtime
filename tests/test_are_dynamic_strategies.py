"""Deterministic tests for the in-process ARE *dynamic*-scenario strategies — over fakes, no model.

The dynamic scenario's follow-up (a mid-run Monday -> Tuesday change) reaches the agent as a **new
inbound email**. Three behaviors distinguish these strategies from the static gaia2 one and are
pinned here:

* ``ReconcilingReasonStrategy`` re-*infers* mid-plan when a new INBOX email appears — and crucially
  does NOT loop when the agent's own reply lands in SENT (the bug a signal-count trigger caused).
* ``CorrectiveSituateStrategy`` spawns exactly one corrective activity when the email lands after
  the original goal completed, and stays quiet otherwise.

The real ARE Environment + real Claude version is the skip-gated
``test_are_dynamic_reproduction.py``.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from examples.are_scenario.strategies import (
    _CORRECTIVE_GOAL,
    CorrectiveSituateStrategy,
    ReconcilingReasonStrategy,
    reconciling_plan_prompt,
)

from fakes import FakeAdapter, FakeLLMClient, FakeTool, FakeWorkspace, plan_json
from sora.action import default_action_registry
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, WorkspaceOrigin
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from sora.perception import Message
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReflectStrategy,
    Strategies,
)
from sora.transport import InProcessTransport
from sora.types import ObservableProperty

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")
_GOAL = "schedule the team sync Alice emailed about, then reply to her"

_PLAN_A = plan_json(
    {"action": "invoke", "tool_id": "EmailClientApp", "operation_name": "list_emails"},
    {"action": "invoke", "tool_id": "CalendarApp", "operation_name": "get_calendar_events_from_to"},
    {"action": "invoke", "tool_id": "CalendarApp", "operation_name": "add_calendar_event"},
    {
        "action": "invoke",
        "tool_id": "EmailClientApp",
        "operation_name": "reply_to_email",
        "params": {"email_id": 1},
    },
)

# The corrected continuation the model returns after the follow-up: new day booked + a fresh reply.
_PLAN_B = plan_json(
    {"action": "invoke", "tool_id": "CalendarApp", "operation_name": "get_calendar_events_from_to"},
    {"action": "invoke", "tool_id": "CalendarApp", "operation_name": "add_calendar_event"},
    {
        "action": "invoke",
        "tool_id": "EmailClientApp",
        "operation_name": "reply_to_email",
        "params": {"email_id": 2},
    },
)

# What the corrective activity plans from scratch: re-read the inbox, then send the corrected reply.
_PLAN_CORRECTIVE = plan_json(
    {"action": "invoke", "tool_id": "EmailClientApp", "operation_name": "list_emails"},
    {
        "action": "invoke",
        "tool_id": "EmailClientApp",
        "operation_name": "reply_to_email",
        "params": {"email_id": 2},
    },
)


def _email_state(*, inbox: list[str], sent: list[str] | None = None) -> dict[str, Any]:
    """The ARE ``EmailClientApp.get_state()`` shape the strategy reads — INBOX + SENT folders."""
    return {
        "folders": {
            "INBOX": {"folder_name": "INBOX", "emails": [{"email_id": i} for i in inbox]},
            "SENT": {"folder_name": "SENT", "emails": [{"email_id": i} for i in (sent or [])]},
        }
    }


def _tools(*, inbox: list[str]) -> tuple[FakeTool, FakeTool]:
    email = FakeTool(
        "EmailClientApp",
        invoke_results={"list_emails": {"emails": [{"id": 1}]}, "reply_to_email": {"sent": True}},
        properties=[ObservableProperty("state", _email_state(inbox=inbox))],
    )
    calendar = FakeTool(
        "CalendarApp",
        invoke_results={
            "get_calendar_events_from_to": {"events": []},
            "add_calendar_event": {"event_id": "evt-1"},
        },
    )
    return email, calendar


def _cycle(
    tmp_path: Path,
    *,
    procedural_dir: Path,
    llm: FakeLLMClient,
    tools: tuple[FakeTool, FakeTool],
    reflect: DefaultReflectStrategy,
) -> tuple[DecisionCycle, WorkingMemory, InProcessTransport, EnvironmentRegistry]:
    email, calendar = tools
    workspace = FakeWorkspace("are", _ORIGIN, [email, calendar])
    registry = EnvironmentRegistry(adapters={_ORIGIN: FakeAdapter("fake", workspace)})
    working = WorkingMemory(registry=registry)
    working.focused_tools[email.id] = email  # simulate the plan's focus step on the inbox
    transport = InProcessTransport()
    strategies = Strategies(
        observe=DefaultObserveStrategy(),
        reflect=reflect,
        situate=CorrectiveSituateStrategy(),
        reason=ReconcilingReasonStrategy(),
        act=DefaultActStrategy(),
    )
    cycle = DecisionCycle(
        strategies=strategies,
        communication=transport,
        actions=default_action_registry(),
        registry=registry,
        working=working,
        semantic=SemanticMemory(FileMemoryBackend(tmp_path / "semantic")),
        procedural=ProceduralMemory(FileMemoryBackend(procedural_dir), llm=llm),
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "episodic")),
    )
    return cycle, working, transport, registry


def _seed_goal(transport: InProcessTransport) -> None:
    transport.submit(Message(sender="user", content={"text": _GOAL}, received_at=time.time()))


def _set_inbox(email: FakeTool, *, inbox: list[str], sent: list[str] | None = None) -> None:
    email._properties = [ObservableProperty("state", _email_state(inbox=inbox, sent=sent))]


async def _drive(cycle: DecisionCycle, working: WorkingMemory, *, budget: int = 30) -> None:
    for _ in range(budget):
        await cycle.tick()
        await asyncio.sleep(0)  # let each off-cycle invoke result land before the next observe
        acts = list(working.activities.values())
        if acts and all(a.state is ActivityState.TERMINATED for a in acts):
            return


async def _settle(reflect: DefaultReflectStrategy) -> None:
    await asyncio.gather(*list(reflect._tasks))  # Reflect's async success store


# --------------------------------------------------------------------------------------------------
# ReconcilingReasonStrategy: a new INBOX email re-infers once; the agent's own SENT reply does NOT
# --------------------------------------------------------------------------------------------------


async def test_new_inbound_midplan_reinfers_once_and_the_reply_does_not_loop(
    tmp_path: Path,
) -> None:
    email, calendar = _tools(inbox=["orig"])
    llm = FakeLLMClient([_PLAN_A, _PLAN_B])
    reflect = DefaultReflectStrategy()
    cycle, working, transport, registry = _cycle(
        tmp_path,
        procedural_dir=tmp_path / "procedural",
        llm=llm,
        tools=(email, calendar),
        reflect=reflect,
    )
    await registry.join(_ORIGIN)
    _seed_goal(transport)

    followed_up = False
    modelled_reply = False
    for _ in range(40):
        await cycle.tick()
        await asyncio.sleep(0)
        if not followed_up and email.invocations:  # plan running -> the follow-up email lands
            _set_inbox(email, inbox=["orig", "followup"])
            followed_up = True
        if not modelled_reply and any(op == "reply_to_email" for op, _ in email.invocations):
            # The agent's reply lands in SENT, not INBOX — a signal-count trigger looped here; the
            # inbound-id trigger must ignore it. Grow SENT to prove it does not re-fire.
            _set_inbox(email, inbox=["orig", "followup"], sent=["myreply"])
            modelled_reply = True
        acts = list(working.activities.values())
        if acts and all(a.state is ActivityState.TERMINATED for a in acts):
            break

    assert len(working.activities) == 1  # Reason handled the live activity; no corrective spawned
    assert next(iter(working.activities.values())).state is ActivityState.TERMINATED
    assert (
        len(llm.calls) == 2
    )  # inferred plan A + re-inferred plan B once; the SENT reply did NOT loop
    assert email.invocations[-1] == (
        "reply_to_email",
        {"email_id": 2},
    )  # plan B, the corrected reply
    await _settle(reflect)


# --------------------------------------------------------------------------------------------------
# CorrectiveSituateStrategy: a new email after completion spawns exactly one corrective activity
# --------------------------------------------------------------------------------------------------


async def test_new_inbound_after_completion_spawns_one_corrective(tmp_path: Path) -> None:
    email, calendar = _tools(inbox=["orig"])
    llm = FakeLLMClient([_PLAN_A, _PLAN_CORRECTIVE])  # schedule infer, then corrective infer
    reflect = DefaultReflectStrategy()
    cycle, working, transport, registry = _cycle(
        tmp_path,
        procedural_dir=tmp_path / "procedural",
        llm=llm,
        tools=(email, calendar),
        reflect=reflect,
    )
    await registry.join(_ORIGIN)
    _seed_goal(transport)

    # Drive the schedule activity to completion with NO new email (inbox stays {orig}).
    await _drive(cycle, working)
    assert len(working.activities) == 1
    assert next(iter(working.activities.values())).state is ActivityState.TERMINATED

    # The follow-up email lands AFTER completion -> a new inbox id with no live activity.
    _set_inbox(email, inbox=["orig", "followup"])
    for _ in range(40):
        await cycle.tick()
        await asyncio.sleep(0)
        acts = list(working.activities.values())
        if len(acts) >= 2 and all(a.state is ActivityState.TERMINATED for a in acts):
            break

    assert len(working.activities) == 2  # exactly one corrective activity (no duplicate spawns)
    corrective = [a for a in working.activities.values() if a.goal != _GOAL]
    assert len(corrective) == 1
    assert corrective[0].goal == _CORRECTIVE_GOAL
    assert corrective[0].state is ActivityState.TERMINATED
    assert len(llm.calls) == 2  # schedule plan + corrective plan (the corrective goal isn't cached)
    await _settle(reflect)


async def test_no_corrective_and_no_reinfer_without_a_new_email(tmp_path: Path) -> None:
    # Guard: both paths are new-inbound-driven — a plain run (inbox never grows) does nothing extra.
    email, calendar = _tools(inbox=["orig"])
    llm = FakeLLMClient(_PLAN_A)
    reflect = DefaultReflectStrategy()
    cycle, working, transport, registry = _cycle(
        tmp_path,
        procedural_dir=tmp_path / "procedural",
        llm=llm,
        tools=(email, calendar),
        reflect=reflect,
    )
    await registry.join(_ORIGIN)
    _seed_goal(transport)

    await _drive(cycle, working)
    for _ in range(3):  # a few idle ticks — inbox unchanged, so nothing new happens
        await cycle.tick()
        await asyncio.sleep(0)

    assert [a.goal for a in working.activities.values()] == [_GOAL]  # no corrective
    assert next(iter(working.activities.values())).state is ActivityState.TERMINATED
    assert len(llm.calls) == 1  # inferred once; the baseline inbox never counted as "new"
    await _settle(reflect)


# --------------------------------------------------------------------------------------------------
# reconciling_plan_prompt: asks the planner to focus the inbox (the observation precondition)
# --------------------------------------------------------------------------------------------------


def test_reconciling_prompt_instructs_focusing_the_inbox() -> None:
    # The whole dynamic path is dead without a focus step (an unfocused tool's state isn't
    # observed), and the base planner treats focus as optional — so the prompt must ask for it.
    activity = Activity(id="probe", goal=_GOAL, context={})
    system, _user = reconciling_plan_prompt(activity, {})
    assert "focus" in system.lower()
    assert "inbox" in system.lower()
    assert "duplicate" in system.lower()  # keeps the reconcile-don't-duplicate guidance
