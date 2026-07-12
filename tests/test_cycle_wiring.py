"""Fast, ARE-free wiring tests for the walking skeleton.

These cover exactly what one ``DecisionCycle.tick()`` touches, exercised against an in-process
fake ``WorkspaceAdapter`` (no MCP, no subprocess). The single *real* end-to-end assertion against
ARE lives in ``test_are_walking_skeleton.py``; this file keeps the wiring itself deterministic and
instant. All of this is throwaway spike code — to be replaced with a proper TDD build-out.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from sora.action import ActionRegistry, InvokeAction
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, Tool, Workspace, WorkspaceOrigin
from sora.manual import Manual
from sora.memory import EpisodicMemory, ProceduralMemory, SemanticMemory, WorkingMemory
from sora.perception import Message, NotificationQueueSink
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
        usage_protocols="",
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
    actions = ActionRegistry()
    actions.register_external(InvokeAction())
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
        working=working,
        semantic=SemanticMemory(backend),
        procedural=ProceduralMemory(backend),
        episodic=EpisodicMemory(backend),
    )
    return cycle, working


# --------------------------------------------------------------------------------------------------
# Group: NotificationQueueSink
# --------------------------------------------------------------------------------------------------


async def test_sink_push_then_drain_yields_in_order() -> None:
    sink: NotificationQueueSink[int] = NotificationQueueSink()
    sink.push("a", 1)
    sink.push("b", 2)
    drained = [item async for item in sink.drain()]
    assert drained == [("a", 1), ("b", 2)]


async def test_sink_drain_is_empty_after_draining() -> None:
    sink: NotificationQueueSink[int] = NotificationQueueSink()
    sink.push("a", 1)
    assert [item async for item in sink.drain()] == [("a", 1)]
    assert [item async for item in sink.drain()] == []


async def test_sink_drain_snapshots_current_depth() -> None:
    # An item pushed *during* a drain waits for the next drain (no starvation within a cycle).
    sink: NotificationQueueSink[int] = NotificationQueueSink()
    sink.push("a", 1)
    seen = []
    async for item in sink.drain():
        seen.append(item)
        sink.push("late", 99)  # must not be yielded by this same drain
    assert seen == [("a", 1)]
    assert [i async for i in sink.drain()] == [("late", 99)]


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


# --------------------------------------------------------------------------------------------------
# Group: InvokeAction + ActionRegistry
# --------------------------------------------------------------------------------------------------


async def test_invoke_action_sets_running_then_pushes_result() -> None:
    tool = FakeTool("EmailClientApp", "list_emails", {"emails": [], "total_emails": 0})
    registry, origin, _ = _registry_with(tool)
    await registry.join(origin)
    cycle, working = _cycle(registry, ListEmailsReasonStrategy("EmailClientApp", "list_emails"))
    activity = Activity(id="a1", goal="list emails", context={})
    working.activities["a1"] = activity

    ack = await InvokeAction().execute(
        registry, cycle, activity_id="a1", tool_id="EmailClientApp", operation_name="list_emails"
    )
    assert ack.ok is True
    assert activity.state is ActivityState.RUNNING
    assert activity.pending_operation is not None

    # The tool round-trip runs off-cycle; yield to the loop so the background task lands its result.
    await asyncio.sleep(0)
    drained = [item async for item in cycle.result_sink.drain()]
    assert len(drained) == 1
    op_id, op_ack = drained[0]
    assert op_id == activity.pending_operation.id
    assert op_ack.ok is True
    assert tool.invoked_with == ("list_emails", {})


async def test_action_registry_lookup() -> None:
    reg = ActionRegistry()
    invoke = InvokeAction()
    reg.register_external(invoke)
    assert reg.external("invoke") is invoke


# --------------------------------------------------------------------------------------------------
# Group: DefaultObserveStrategy resolves a RUNNING activity
# --------------------------------------------------------------------------------------------------


async def test_observe_resolves_running_activity() -> None:
    tool = FakeTool("EmailClientApp", "list_emails", {"emails": []})
    registry, origin, _ = _registry_with(tool)
    await registry.join(origin)
    cycle, working = _cycle(registry, ListEmailsReasonStrategy("EmailClientApp", "list_emails"))
    activity = Activity(id="a1", goal="list emails", context={})
    working.activities["a1"] = activity
    await InvokeAction().execute(
        registry, cycle, activity_id="a1", tool_id="EmailClientApp", operation_name="list_emails"
    )
    op_id = activity.pending_operation.id  # type: ignore[union-attr]
    # Simulate the off-cycle result having landed, then observe.
    cycle.result_sink.push(op_id, OperationAck(ok=True, result={"emails": []}))
    await DefaultObserveStrategy().observe(cycle)
    assert activity.state is ActivityState.READY
    assert activity.pending_operation is None
    assert activity.last_operation == OperationAck(ok=True, result={"emails": []})


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
