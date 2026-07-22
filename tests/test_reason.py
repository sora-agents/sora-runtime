"""Permanent tests for the Reason phase — ``DefaultReasonStrategy``.

Reason is the one phase with no *mechanical* default: planning inherently needs inference. The
runtime's default is therefore **deterministic orchestration around a model call**, not a model
call itself. ``DefaultReasonStrategy``:

* if the activity already has a plan with steps remaining, reads the current step and advances
  ``step_index`` — the cheap path, no ``retrieve``/``infer`` at all;
* otherwise retrieves a cached ``Plan`` for the goal (reuse across runs), or, on a miss, calls
  ``ProceduralMemory.infer(...)`` — the single model call, passing the currently-joined tools as
  the planning catalog;
* on an exhausted plan, yields no step (the cycle returns; Reflect terminates it next cycle).

The model itself lives behind ``ProceduralMemory.infer`` (see ``test_procedural_memory.py``); these
tests pin the orchestration deterministically with a scripted ``ProceduralMemory`` (unit rules) and
a ``FakeLLMClient``-backed real one (the end-to-end tick). No network, no model.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fakes import FakeAdapter, FakeLLMClient, FakeTool, FakeWorkspace, plan_json
from sora.action import default_action_registry, invoke_step
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, Tool, WorkspaceOrigin
from sora.manual import Manual
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    PerceptSnapshot,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from sora.perception import Message, Percept
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReasonStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    Strategies,
    TickResult,
)
from sora.types import ObservableProperty, Plan, Signal

# --------------------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------------------

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")


class ScriptedTransport:
    """Satisfies MessageTransport: ``receive()`` yields nothing; ``send()`` logs."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def send(self, to: str, content: dict[str, Any]) -> None:
        self.sent.append((to, content))

    def receive(self) -> AsyncIterator[Message]:
        async def _drain() -> AsyncIterator[Message]:
            return
            yield  # pragma: no cover — never-yielding async generator

        return _drain()


class SpyProcedural(ProceduralMemory):
    """A ``ProceduralMemory`` whose ``retrieve``/``infer`` are scripted and logged, so a reason test
    can assert exactly which planning path ``DefaultReasonStrategy`` took (and prove it did *not*
    plan on the cheap path). The backend is never touched — both methods are overridden."""

    def __init__(self, *, retrieve: Plan | None = None, infer: Plan | None = None) -> None:
        super().__init__(FileMemoryBackend("unused-spy-backend"))
        self._retrieve_plan = retrieve
        self._infer_plan = infer
        self.retrieve_calls: list[Activity] = []
        self.infer_calls: list[tuple[Activity, dict[str, Manual], PerceptSnapshot]] = []

    async def retrieve(self, activity: Activity) -> Plan | None:
        self.retrieve_calls.append(activity)
        return self._retrieve_plan

    async def infer(
        self,
        activity: Activity,
        tools: dict[str, Manual],
        observed: PerceptSnapshot | None = None,
    ) -> Plan:
        self.infer_calls.append((activity, tools, observed or PerceptSnapshot()))
        if self._infer_plan is None:
            raise AssertionError("infer() was called but no infer plan was configured")
        return self._infer_plan


def _registry_with(*tools: Tool) -> tuple[EnvironmentRegistry, WorkspaceOrigin]:
    workspace = FakeWorkspace("ws", _ORIGIN, list(tools))
    adapter = FakeAdapter("fake", workspace)
    return EnvironmentRegistry(adapters={_ORIGIN: adapter}), _ORIGIN


def _cycle(
    registry: EnvironmentRegistry,
    tmp_path: Path,
    *,
    procedural: ProceduralMemory,
    reflect: DefaultReflectStrategy | None = None,
) -> tuple[DecisionCycle, WorkingMemory]:
    # Separate dirs per module: procedural/episodic both query by `goal`, so a shared directory
    # would let a plan and an episode with the same goal cross-match (mirrors test_cycle.py).
    working = WorkingMemory(registry=registry)
    strategies = Strategies(
        observe=DefaultObserveStrategy(),
        reflect=reflect or DefaultReflectStrategy(),
        situate=DefaultSituateStrategy(),
        reason=DefaultReasonStrategy(),
        act=DefaultActStrategy(),
    )
    cycle = DecisionCycle(
        strategies=strategies,
        communication=ScriptedTransport(),
        actions=default_action_registry(),
        registry=registry,
        working=working,
        semantic=SemanticMemory(FileMemoryBackend(tmp_path / "semantic")),
        procedural=procedural,
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "episodic")),
    )
    return cycle, working


async def _settle(reflect: DefaultReflectStrategy) -> None:
    """Await Reflect's in-flight background stores (same dangling-task pattern as test_cycle)."""
    await asyncio.gather(*list(reflect._tasks))


# --------------------------------------------------------------------------------------------------
# Cheap path — advance an existing plan without planning
# --------------------------------------------------------------------------------------------------


async def test_reason_advances_existing_plan_without_planning(tmp_path: Path) -> None:
    registry, _ = _registry_with()
    spy = SpyProcedural()  # infer() would raise, retrieve() returns None — neither must be called
    cycle, working = _cycle(registry, tmp_path, procedural=spy)
    plan = Plan(id="p", goal="g", steps=[invoke_step("t", "op1"), invoke_step("t", "op2")])
    activity = Activity(id="a", goal="g", context={}, plan=plan, step_index=0)

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step == invoke_step("t", "op1")  # current step
    assert activity.step_index == 1  # advanced
    assert result.activity is activity
    assert spy.retrieve_calls == []  # the cheap path never consults procedural memory
    assert spy.infer_calls == []


async def test_reason_advances_across_successive_calls(tmp_path: Path) -> None:
    registry, _ = _registry_with()
    spy = SpyProcedural()
    cycle, working = _cycle(registry, tmp_path, procedural=spy)
    plan = Plan(id="p", goal="g", steps=[invoke_step("t", "op1"), invoke_step("t", "op2")])
    activity = Activity(id="a", goal="g", context={}, plan=plan, step_index=0)

    first = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    second = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert first.step == invoke_step("t", "op1")
    assert second.step == invoke_step("t", "op2")
    assert activity.step_index == 2


# --------------------------------------------------------------------------------------------------
# Planning path — retrieve (reuse) then infer (the model call)
# --------------------------------------------------------------------------------------------------


async def test_reason_reuses_retrieved_plan_without_inferring(tmp_path: Path) -> None:
    registry, _ = _registry_with()
    cached = Plan(id="p", goal="g", steps=[invoke_step("t", "op1")])
    spy = SpyProcedural(retrieve=cached)  # infer() would raise if reached
    cycle, working = _cycle(registry, tmp_path, procedural=spy)
    activity = Activity(id="a", goal="g", context={})

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert activity.plan is cached  # reused, not re-planned
    assert activity.step_index == 1
    assert result.step == cached.steps[0]
    assert len(spy.retrieve_calls) == 1
    assert spy.infer_calls == []  # a cache hit skips the model entirely


async def test_reason_infers_plan_on_cache_miss_with_joined_tool_catalog(tmp_path: Path) -> None:
    tool = FakeTool("EmailClientApp", invoke_results={"list_emails": {"emails": []}})
    registry, origin = _registry_with(tool)
    await registry.join(origin)
    inferred = Plan(id="p", goal="g", steps=[invoke_step("EmailClientApp", "list_emails")])
    spy = SpyProcedural(retrieve=None, infer=inferred)
    cycle, working = _cycle(registry, tmp_path, procedural=spy)
    activity = Activity(id="a", goal="g", context={})

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert activity.plan is inferred
    assert activity.step_index == 1
    assert result.step == inferred.steps[0]
    assert len(spy.retrieve_calls) == 1
    assert len(spy.infer_calls) == 1
    called_activity, called_tools, called_observed = spy.infer_calls[0]
    assert called_activity is activity
    # The planning catalog is the currently-joined tools, keyed by tool id -> its manual.
    assert called_tools == {tool.id: tool.manual}
    # No percepts observed yet in this test -> infer() still gets the (empty) current snapshot.
    assert called_observed == PerceptSnapshot()


async def test_reason_infer_receives_current_properties_and_signals(tmp_path: Path) -> None:
    # Reason shouldn't hand the planner only past action results — the agent's currently-known
    # world state (WorkingMemory.properties/.signals) must reach infer() too.
    tool = FakeTool("EmailClientApp", invoke_results={"list_emails": {"emails": []}})
    registry, origin = _registry_with(tool)
    await registry.join(origin)
    inferred = Plan(id="p", goal="g", steps=[invoke_step("EmailClientApp", "list_emails")])
    spy = SpyProcedural(retrieve=None, infer=inferred)
    cycle, working = _cycle(registry, tmp_path, procedural=spy)
    activity = Activity(id="a", goal="g", context={})
    prop_percept = Percept("EmailClientApp", ObservableProperty("unread_count", 3), 0.0)
    signal_percept = Percept("EmailClientApp", Signal("new_email", {"id": 1}), 0.0)
    working.properties[("EmailClientApp", "unread_count")] = prop_percept
    working.signals.append(signal_percept)

    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    _activity, _tools, called_observed = spy.infer_calls[0]
    assert called_observed == PerceptSnapshot([prop_percept], [signal_percept])


# --------------------------------------------------------------------------------------------------
# Exhausted plan
# --------------------------------------------------------------------------------------------------


async def test_reason_exhausted_plan_yields_no_step(tmp_path: Path) -> None:
    registry, _ = _registry_with()
    spy = SpyProcedural()
    cycle, working = _cycle(registry, tmp_path, procedural=spy)
    plan = Plan(id="p", goal="g", steps=[invoke_step("t", "op1")])
    activity = Activity(id="a", goal="g", context={}, plan=plan, step_index=1)  # past the end

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step is None  # nothing to advance; the cycle returns, Reflect terminates it
    assert spy.infer_calls == []


# --------------------------------------------------------------------------------------------------
# End-to-end: infer once, advance a 2-step plan through the real cycle, terminate + store
# --------------------------------------------------------------------------------------------------


async def test_tick_infers_advances_and_terminates_two_step_plan(tmp_path: Path) -> None:
    tool = FakeTool(
        "EmailClientApp",
        invoke_results={"list_emails": {"emails": []}, "read_email": {"body": "hi"}},
    )
    registry, origin = _registry_with(tool)
    llm = FakeLLMClient(
        plan_json(
            {"action": "invoke", "tool_id": "EmailClientApp", "operation_name": "list_emails"},
            {
                "action": "invoke",
                "tool_id": "EmailClientApp",
                "operation_name": "read_email",
                "params": {"id": 1},
            },
        )
    )
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "procedural"), llm=llm)
    reflect = DefaultReflectStrategy()
    cycle, working = _cycle(registry, tmp_path, procedural=procedural, reflect=reflect)
    await registry.join(origin)
    working.activities["a1"] = Activity(id="a1", goal="triage inbox", context={})

    for _ in range(12):
        await cycle.tick()
        await asyncio.sleep(0)  # let each off-cycle invoke result land before the next observe
        if working.activities["a1"].state is ActivityState.TERMINATED:
            break

    activity = working.activities["a1"]
    assert activity.state is ActivityState.TERMINATED  # completed the full plan
    assert [op for op, _ in tool.invocations] == ["list_emails", "read_email"]  # both, in order
    assert tool.invocations[1] == ("read_email", {"id": 1})  # params bound through
    assert len(llm.calls) == 1  # inferred once; every later cycle advanced the cached plan

    await _settle(reflect)  # settle the async success store so no task dangles past the test
    stored = await procedural.retrieve(Activity(id="probe", goal="triage inbox", context={}))
    assert stored is not None  # the followed plan was stored for reuse across runs
    assert [s.params["operation_name"] for s in stored.steps] == ["list_emails", "read_email"]


async def test_tick_plan_focuses_then_invokes(tmp_path: Path) -> None:
    # focus/unfocus are external actions the LLM can emit as plan steps (D3/D4 deferred this to
    # a richer strategy — the model-backed Reason is it): a focus step dispatches through
    # FocusAction, then the plan advances to the invoke on the next cycle.
    tool = FakeTool("clock", invoke_results={"get_time": "10:00"})
    registry, origin = _registry_with(tool)
    llm = FakeLLMClient(
        plan_json(
            {"action": "focus", "tool_id": "clock"},
            {"action": "invoke", "tool_id": "clock", "operation_name": "get_time"},
        )
    )
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "procedural"), llm=llm)
    reflect = DefaultReflectStrategy()
    cycle, working = _cycle(registry, tmp_path, procedural=procedural, reflect=reflect)
    await registry.join(origin)
    working.activities["a1"] = Activity(id="a1", goal="what time is it", context={})

    for _ in range(12):
        await cycle.tick()
        await asyncio.sleep(0)
        if working.activities["a1"].state is ActivityState.TERMINATED:
            break

    assert tool.focused is True  # the focus step subscribed the agent to the tool
    assert "clock" in working.focused_tools
    assert [op for op, _ in tool.invocations] == ["get_time"]  # then the invoke step ran
    assert working.activities["a1"].state is ActivityState.TERMINATED
    await _settle(reflect)
