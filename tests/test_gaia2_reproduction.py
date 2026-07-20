"""Deterministic reproduction of EXAMPLES.md's ARE ``email_calendar`` scenario — over fakes.

It reproduces the scenario's *shape* without a network or a model, so it runs in the default suite.
It exercises the whole assembled path — the initial task arrives as a ``Message`` through the
transport, the default Situate turns it into an activity, the shipped ``ScheduleFromEmailStrategy``
(model-backed planning via a ``FakeLLMClient`` + signal-driven replanning) drives it, and the four
invokes land on the fake ARE tools in order. Three behaviors EXAMPLES.md promises are pinned here:

* the four-step plan (read email -> check calendar -> create event -> reply), inferred once;
* procedural-memory *reuse across runs* — a second run on the same store re-plans with zero model
  calls;
* *signal-driven replanning* — a mid-plan ``resource_updated`` signal invalidates the plan and the
  agent re-plans from the updated working memory.

The real ARE server + real Claude version is the skip-gated ``test_are_scenario_reproduction.py``;
the on-demand showcase is ``examples/gaia2/email_calendar/run.py``.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from examples.gaia2.email_calendar import ScheduleFromEmailStrategy

from fakes import FakeAdapter, FakeLLMClient, FakeTool, FakeWorkspace, plan_json
from sora.action import default_action_registry
from sora.activity import Activity, ActivityState
from sora.cycle import Agent, DecisionCycle
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
    DefaultSituateStrategy,
    Strategies,
)
from sora.transport import InProcessTransport
from sora.types import Signal

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")
_GOAL = "schedule a 30-min sync with Bob and Carol next Monday and reply to Alice"

_FOUR_STEP_PLAN = plan_json(
    {
        "action": "invoke",
        "tool_id": "EmailClientApp",
        "operation_name": "list_emails",
        "params": {"folder": "inbox", "limit": 5},
    },
    {"action": "invoke", "tool_id": "CalendarApp", "operation_name": "get_calendar_events_from_to"},
    {"action": "invoke", "tool_id": "CalendarApp", "operation_name": "add_calendar_event"},
    {
        "action": "invoke",
        "tool_id": "EmailClientApp",
        "operation_name": "reply_to_email",
        "params": {"email_id": 1},
    },
)


def _tools() -> tuple[FakeTool, FakeTool]:
    email = FakeTool(
        "EmailClientApp",
        invoke_results={"list_emails": {"emails": [{"id": 1}]}, "reply_to_email": {"sent": True}},
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
    workspace = FakeWorkspace("gaia2", _ORIGIN, [email, calendar])
    registry = EnvironmentRegistry(adapters={_ORIGIN: FakeAdapter("fake", workspace)})
    working = WorkingMemory(registry=registry)
    transport = InProcessTransport()
    strategies = Strategies(
        observe=DefaultObserveStrategy(),
        reflect=reflect,
        situate=DefaultSituateStrategy(),
        reason=ScheduleFromEmailStrategy(),
        act=DefaultActStrategy(),
    )
    cycle = DecisionCycle(
        strategies=strategies,
        communication=transport,
        actions=default_action_registry(),
        registry=registry,
        working=working,
        # Separate dirs: procedural + episodic both query by `goal`, so a shared dir would let a
        # plan and an episode cross-match (same rationale as test_reason.py).
        semantic=SemanticMemory(FileMemoryBackend(tmp_path / "semantic")),
        procedural=ProceduralMemory(FileMemoryBackend(procedural_dir), llm=llm),
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "episodic")),
    )
    return cycle, working, transport, registry


def _seed_goal(transport: InProcessTransport) -> None:
    transport.submit(Message(sender="user", content={"text": _GOAL}, received_at=time.time()))


async def _drive(cycle: DecisionCycle, working: WorkingMemory, *, budget: int = 20) -> Activity:
    for _ in range(budget):
        await cycle.tick()
        await asyncio.sleep(0)  # let each off-cycle invoke result land before the next observe
        activities = list(working.activities.values())
        if activities and all(a.state is ActivityState.TERMINATED for a in activities):
            break
    return next(iter(working.activities.values()))


async def _settle(reflect: DefaultReflectStrategy) -> None:
    await asyncio.gather(*list(reflect._tasks))  # Reflect's async success store


def _agent(cycle: DecisionCycle) -> Agent:
    """Wrap a cycle in an Agent sharing its instances (tick_interval=0 -> run() spins fast)."""
    return Agent(
        cycle=cycle,
        registry=cycle.registry,
        working=cycle.working,
        semantic=cycle.semantic,
        procedural=cycle.procedural,
        episodic=cycle.episodic,
        communication=cycle.communication,
        tick_interval=0.0,
    )


# --------------------------------------------------------------------------------------------------
# Four-step plan, inferred once, followed to completion, stored for reuse
# --------------------------------------------------------------------------------------------------


async def test_reproduces_four_step_plan_and_stores_it(tmp_path: Path) -> None:
    email, calendar = _tools()
    llm = FakeLLMClient(_FOUR_STEP_PLAN)
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

    activity = await _drive(cycle, working)

    assert activity.goal == _GOAL  # the message became an activity (transport -> Situate)
    assert activity.state is ActivityState.TERMINATED
    assert len(llm.calls) == 1  # inferred once; every later cycle advanced the cached plan
    # Each tool ran its own operations in plan order (the two interleave: email, cal, cal, email).
    assert [op for op, _ in email.invocations] == ["list_emails", "reply_to_email"]
    assert [op for op, _ in calendar.invocations] == [
        "get_calendar_events_from_to",
        "add_calendar_event",
    ]
    assert email.invocations[1] == ("reply_to_email", {"email_id": 1})  # params bound through

    await _settle(reflect)
    stored = await cycle.procedural.retrieve(Activity(id="probe", goal=_GOAL, context={}))
    assert stored is not None  # the followed plan was stored for reuse across runs
    assert [s.params["operation_name"] for s in stored.steps] == [
        "list_emails",
        "get_calendar_events_from_to",
        "add_calendar_event",
        "reply_to_email",
    ]


# --------------------------------------------------------------------------------------------------
# Procedural reuse across runs: a second run re-plans with zero model calls
# --------------------------------------------------------------------------------------------------


async def test_second_run_reuses_stored_plan_without_model_call(tmp_path: Path) -> None:
    procedural_dir = tmp_path / "procedural"  # shared across both runs -> the plan persists

    # Run 1: infer + store.
    reflect1 = DefaultReflectStrategy()
    cycle1, working1, transport1, registry1 = _cycle(
        tmp_path / "run1",
        procedural_dir=procedural_dir,
        llm=FakeLLMClient(_FOUR_STEP_PLAN),
        tools=_tools(),
        reflect=reflect1,
    )
    await registry1.join(_ORIGIN)
    _seed_goal(transport1)
    a1 = await _drive(cycle1, working1)
    assert a1.state is ActivityState.TERMINATED
    await _settle(reflect1)

    # Run 2: same goal, same procedural store, a model that would RAISE if inference were attempted.
    reflect2 = DefaultReflectStrategy()
    empty_llm = FakeLLMClient([])  # no configured response -> .complete() raises if ever called
    cycle2, working2, transport2, registry2 = _cycle(
        tmp_path / "run2",
        procedural_dir=procedural_dir,
        llm=empty_llm,
        tools=_tools(),
        reflect=reflect2,
    )
    await registry2.join(_ORIGIN)
    _seed_goal(transport2)
    a2 = await _drive(cycle2, working2)

    assert a2.state is ActivityState.TERMINATED  # completed purely from the reused plan
    assert empty_llm.calls == []  # the cache hit skipped the model entirely
    await _settle(reflect2)


# --------------------------------------------------------------------------------------------------
# Signal-driven replanning: a mid-plan resource_updated invalidates the plan
# --------------------------------------------------------------------------------------------------


async def test_signal_midplan_triggers_replan(tmp_path: Path) -> None:
    email, calendar = _tools()
    # Plan A is the original 4 steps; after a signal lands post-list_emails, the strategy
    # invalidates it and the model returns plan B (a corrected 3-step continuation).
    plan_b = plan_json(
        {
            "action": "invoke",
            "tool_id": "CalendarApp",
            "operation_name": "get_calendar_events_from_to",
        },
        {"action": "invoke", "tool_id": "CalendarApp", "operation_name": "add_calendar_event"},
        {
            "action": "invoke",
            "tool_id": "EmailClientApp",
            "operation_name": "reply_to_email",
            "params": {"email_id": 2},
        },
    )
    llm = FakeLLMClient([_FOUR_STEP_PLAN, plan_b])
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

    pushed = False
    for _ in range(24):
        await cycle.tick()
        await asyncio.sleep(0)
        if not pushed and any(op == "list_emails" for op, _ in email.invocations):
            # ARE emits resource_updated after the inbox changes (a follow-up email); simulate it by
            # pushing the signal — the next observe surfaces it as a SIGNAL percept mid-plan.
            cycle.signal_sink.push(
                "EmailClientApp", Signal("state_changed", {"uri": "app://EmailClientApp/state"})
            )
            pushed = True
        activities = list(working.activities.values())
        if activities and all(a.state is ActivityState.TERMINATED for a in activities):
            break

    activity = next(iter(working.activities.values()))
    assert activity.state is ActivityState.TERMINATED
    assert len(llm.calls) == 2  # inferred plan A, then re-inferred plan B after the signal
    # list_emails ran (plan A step 1); the calendar work + reply came from the re-planned plan B.
    assert [op for op, _ in email.invocations] == ["list_emails", "reply_to_email"]
    assert [op for op, _ in calendar.invocations] == [
        "get_calendar_events_from_to",
        "add_calendar_event",
    ]
    assert email.invocations[-1] == ("reply_to_email", {"email_id": 2})  # from plan B, not plan A
    await _settle(reflect)


# --------------------------------------------------------------------------------------------------
# The *assembled* agent: Agent.run() drives the whole scenario (startup join -> plan -> teardown)
# --------------------------------------------------------------------------------------------------


async def test_agent_run_reproduces_scenario_end_to_end(tmp_path: Path) -> None:
    email, calendar = _tools()
    llm = FakeLLMClient(_FOUR_STEP_PLAN)
    reflect = DefaultReflectStrategy()
    # NOT pre-joined: Agent.run() performs the startup join itself (as it would in production).
    cycle, working, transport, _ = _cycle(
        tmp_path,
        procedural_dir=tmp_path / "procedural",
        llm=llm,
        tools=(email, calendar),
        reflect=reflect,
    )
    agent = _agent(cycle)
    _seed_goal(transport)

    runner = asyncio.create_task(agent.run())
    try:
        for _ in range(2000):
            await asyncio.sleep(0)
            acts = list(working.activities.values())
            if acts and all(a.state is ActivityState.TERMINATED for a in acts):
                break
        else:
            runner.cancel()
            raise AssertionError("agent did not terminate the activity within the loop budget")
    finally:
        await agent.stop()
        await runner

    activity = next(iter(working.activities.values()))
    assert activity.state is ActivityState.TERMINATED
    assert len(llm.calls) == 1  # inferred once through the running agent
    assert [op for op, _ in email.invocations] == ["list_emails", "reply_to_email"]
    assert [op for op, _ in calendar.invocations] == [
        "get_calendar_events_from_to",
        "add_calendar_event",
    ]
    assert not agent.registry.joined_workspaces()  # left on teardown
    await _settle(reflect)


# --------------------------------------------------------------------------------------------------
# Dynamic parameter grounding: a later step's param references an earlier step's result
# --------------------------------------------------------------------------------------------------


def _reply_tools() -> tuple[FakeTool, FakeTool]:
    email = FakeTool(
        "EmailClientApp",
        invoke_results={
            "search_emails": {"emails": [{"id": 42, "sender": "alice@corp.com"}]},
            "reply_to_email": {"sent": True},
        },
    )
    return email, FakeTool("CalendarApp", invoke_results={})  # calendar unused here


async def test_reference_param_grounds_from_prior_result_without_model(tmp_path: Path) -> None:
    email, calendar = _reply_tools()
    # reply_to_email's email_id is a hard reference to search_emails' result — resolvable
    # mechanically once search has run, so no second model call.
    plan = plan_json(
        {
            "action": "invoke",
            "tool_id": "EmailClientApp",
            "operation_name": "search_emails",
            "params": {"query": "Alice"},
        },
        {
            "action": "invoke",
            "tool_id": "EmailClientApp",
            "operation_name": "reply_to_email",
            "params": {
                "email_id": {"$from": "search_emails", "path": "emails.0.id"},
                "body": "hi Alice",
            },
        },
    )
    llm = FakeLLMClient(plan)
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

    activity = await _drive(cycle, working)

    assert activity.state is ActivityState.TERMINATED
    assert email.invocations == [
        ("search_emails", {"query": "Alice"}),
        ("reply_to_email", {"email_id": 42, "body": "hi Alice"}),  # id bound from the prior result
    ]
    assert len(llm.calls) == 1  # inferred once; the reference resolved mechanically (no model call)
    await _settle(reflect)


async def test_soft_reference_escalates_to_model_grounding(tmp_path: Path) -> None:
    email, calendar = _reply_tools()
    # A $decide reference always escalates: the model grounds email_id from the search result.
    plan = plan_json(
        {
            "action": "invoke",
            "tool_id": "EmailClientApp",
            "operation_name": "search_emails",
            "params": {"query": "Alice"},
        },
        {
            "action": "invoke",
            "tool_id": "EmailClientApp",
            "operation_name": "reply_to_email",
            "params": {"email_id": {"$decide": "Alice's email id"}, "body": "hi"},
        },
    )
    ground = json.dumps({"params": {"email_id": 42, "body": "hi"}})
    llm = FakeLLMClient([plan, ground])  # call 1: infer, call 2: ground escalation
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

    activity = await _drive(cycle, working)

    assert activity.state is ActivityState.TERMINATED
    assert ("reply_to_email", {"email_id": 42, "body": "hi"}) in email.invocations
    assert len(llm.calls) == 2  # infer + one ground escalation
    await _settle(reflect)
