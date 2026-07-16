"""Fast, ARE-free wiring tests for the walking skeleton.

These cover exactly what one ``DecisionCycle.tick()`` touches, exercised against an in-process
fake ``WorkspaceAdapter`` (no MCP, no subprocess). The single *real* end-to-end assertion against
ARE lives in ``test_are_walking_skeleton.py``; this file keeps the wiring itself deterministic and
instant.

This started as spike code, but most of these assertions are the natural first tests for the
permanent build-out. ``docs/phase-3-test-triage.md`` records, per group, which assertions are
promoted to permanent TDD tests (and under which task), which get re-driven, and which scaffolding
is replaced. Each layer is lifted out of this file into its permanent module as it is re-driven;
this file is deleted once the last group has moved.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from sora.action import default_action_registry
from sora.activity import Activity
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, Tool, Workspace, WorkspaceOrigin
from sora.manual import Manual
from sora.memory import EpisodicMemory, ProceduralMemory, SemanticMemory, WorkingMemory
from sora.perception import Message
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    Strategies,
    TickResult,
)
from sora.transport import MessageTransport
from sora.types import ObservableProperty, OperationAck, Step

# --------------------------------------------------------------------------------------------------
# In-process fakes (stand in for the MCP adapter / ARE server / transport / memory backend)
# --------------------------------------------------------------------------------------------------


def _manual(tool_id: str, op: str) -> Manual:
    from sora.manual import OperationSpecification

    return Manual(
        id=tool_id,
        metadata={},
        description=f"fake {tool_id}",
        observable_properties=[],
        signals=[],
        operations=[OperationSpecification(name=op, description="", parameters={})],
    )


class FakeTool:
    def __init__(self, tool_id: str, op: str, result: Any) -> None:
        self.id = tool_id
        self.manual = _manual(tool_id, op)
        self.address: str | None = None
        self._result = result
        self.invoked_with: tuple[str, dict[str, Any]] | None = None

    async def invoke(self, operation_name: str, **params: Any) -> OperationAck:
        self.invoked_with = (operation_name, params)
        return OperationAck(ok=True, result=self._result)

    async def focus(self, sink: Any) -> None: ...

    async def unfocus(self) -> None: ...

    def observe(self) -> list[ObservableProperty]:
        return []


class FakeWorkspace:
    def __init__(self, ws_id: str, origin: WorkspaceOrigin, tools: list[Tool]) -> None:
        self.id = ws_id
        self.origin = origin
        self._tools = tools
        self.closed = False

    def tools(self) -> list[Tool]:
        return self._tools

    async def close(self) -> None:
        self.closed = True


class FakeAdapter:
    name = "fake"

    def __init__(self, ws_id: str, origin: WorkspaceOrigin, tools: list[Tool]) -> None:
        self._ws = FakeWorkspace(ws_id, origin, tools)

    async def discover(self) -> list[Workspace]:
        return [self._ws]

    async def connect(self, *a: Any, **k: Any) -> Workspace:  # unused by the spike
        return self._ws


class NullTransport:
    async def send(self, to: str, content: dict[str, Any]) -> None: ...

    def receive(self) -> AsyncIterator[Message]:
        async def _empty() -> AsyncIterator[Message]:
            return
            yield  # pragma: no cover — makes this a (never-yielding) async generator

        return _empty()


class DictBackend:
    def __init__(self) -> None:
        self._d: dict[str, Any] = {}

    async def get(self, key: str) -> Any:
        return self._d.get(key)

    async def put(self, key: str, value: Any) -> None:
        self._d[key] = value

    async def query(self, **filters: Any) -> list[Any]:
        return list(self._d.values())


class ListEmailsReasonStrategy:
    """The hardcoded spike strategy: always advance to invoking ``<tool>.<op>`` once."""

    def __init__(self, tool_id: str, operation_name: str) -> None:
        self._tool_id = tool_id
        self._operation_name = operation_name

    async def reason(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        if activity.last_operation is not None:  # already done its single step
            return TickResult(activity=activity, step=Step(next_action="wait", params={}))
        return TickResult(
            activity=activity,
            step=Step(
                next_action="invoke",
                params={"tool_id": self._tool_id, "operation_name": self._operation_name},
            ),
        )


def _cycle(
    registry: EnvironmentRegistry, reason: Any, transport: MessageTransport | None = None
) -> tuple[DecisionCycle, WorkingMemory]:
    working = WorkingMemory(registry=registry)
    backend = DictBackend()
    actions = default_action_registry()
    strategies = Strategies(
        observe=DefaultObserveStrategy(),
        reflect=DefaultReflectStrategy(),
        situate=DefaultSituateStrategy(),
        reason=reason,
        act=DefaultActStrategy(),
    )
    cycle = DecisionCycle(
        strategies=strategies,
        communication=transport or NullTransport(),
        actions=actions,
        registry=registry,
        working=working,
        semantic=SemanticMemory(backend),
        procedural=ProceduralMemory(backend),
        episodic=EpisodicMemory(backend),
    )
    return cycle, working


# --------------------------------------------------------------------------------------------------
# Group: EnvironmentRegistry join/get/leave
# --------------------------------------------------------------------------------------------------


def _registry_with(tool: FakeTool) -> tuple[EnvironmentRegistry, WorkspaceOrigin, FakeWorkspace]:
    origin = WorkspaceOrigin(adapter="fake", address="fake://ws")
    adapter = FakeAdapter("ws", origin, [tool])
    registry = EnvironmentRegistry(adapters={origin: adapter})
    return registry, origin, adapter._ws


async def test_registry_join_registers_workspace_and_tools() -> None:
    tool = FakeTool("EmailClientApp", "list_emails", {"emails": []})
    registry, origin, ws = _registry_with(tool)
    joined = await registry.join(origin)
    assert joined is ws
    assert registry.get("EmailClientApp") is tool
    assert registry.get_workspace("ws") is ws
    assert [t.id for t in registry.all_tools()] == ["EmailClientApp"]


async def test_registry_leave_closes_and_deregisters() -> None:
    tool = FakeTool("EmailClientApp", "list_emails", {"emails": []})
    registry, origin, ws = _registry_with(tool)
    await registry.join(origin)
    await registry.leave("ws")
    assert ws.closed is True
    with pytest.raises(KeyError):
        registry.get("EmailClientApp")


async def test_joined_workspaces_reflects_join_and_leave() -> None:
    tool = FakeTool("EmailClientApp", "list_emails", {"emails": []})
    registry, origin, ws = _registry_with(tool)
    assert registry.joined_workspaces() == []
    await registry.join(origin)
    assert registry.joined_workspaces() == [ws]
    await registry.leave("ws")
    assert registry.joined_workspaces() == []


async def test_working_memory_registry_is_the_shared_instance() -> None:
    # WorkingMemory holds the same EnvironmentRegistry object (typed read-only as EnvironmentView),
    # so a strategy reasons over the live joined set with no separate copy to keep in sync.
    tool = FakeTool("EmailClientApp", "list_emails", {"emails": []})
    registry, origin, ws = _registry_with(tool)
    working = WorkingMemory(registry=registry)
    await registry.join(origin)
    assert working.registry is registry
    assert working.registry.joined_workspaces() == [ws]


# --------------------------------------------------------------------------------------------------
# Group: DecisionCycle.tick end-to-end on the fake adapter
# --------------------------------------------------------------------------------------------------


async def test_tick_end_to_end_invoke_then_resolve() -> None:
    tool = FakeTool("EmailClientApp", "list_emails", {"emails": [], "total_emails": 0})
    registry, origin, _ = _registry_with(tool)
    await registry.join(origin)
    cycle, working = _cycle(registry, ListEmailsReasonStrategy("EmailClientApp", "list_emails"))
    working.activities["a1"] = Activity(id="a1", goal="list emails", context={})

    # Drive the cycle until the single step resolves (observe picks up the off-cycle result).
    for _ in range(5):
        await cycle.tick()
        await asyncio.sleep(0)  # let the off-cycle invoke task land before the next observe
        if working.activities["a1"].last_operation is not None:
            break

    activity = working.activities["a1"]
    assert activity.last_operation is not None
    assert activity.last_operation.ok is True
    assert activity.last_operation.result == {"emails": [], "total_emails": 0}
    assert tool.invoked_with == ("list_emails", {})
