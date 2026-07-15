"""Permanent TDD tests for the predefined external actions.

Covers the five still-stubbed actions — Focus/Unfocus (working.focused_tools + signal_sink wiring),
Join/Leave (registry mutation + record persistence via SemanticMemory), Send (MessageTransport
delegation). Each action's ``execute(...)`` is driven directly against the in-process fakes
(``tests/fakes.py``) and a real ``FileMemoryBackend``, so the Join persistence path exercises the
actual store/retrieve serialization round-trip rather than a mock. Also promotes the spike's
``action_registry_lookup`` (triage rows 27, 87).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeAdapter, FakeTool, FakeWorkspace
from sora.action import (
    ActionRegistry,
    ExternalAction,
    FocusAction,
    InvokeAction,
    JoinAction,
    LeaveAction,
    SendAction,
    UnfocusAction,
)
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, Tool, WorkspaceOrigin
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
    TickResult,
)
from sora.types import Signal

# --------------------------------------------------------------------------------------------------
# Harness — the fakes plus a recording transport and a real FileMemoryBackend-backed DecisionCycle.
# --------------------------------------------------------------------------------------------------

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")


class RecordingTransport:
    """Satisfies MessageTransport: send() logs its args; receive() yields nothing."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def send(self, to: str, content: dict[str, Any]) -> None:
        self.sent.append((to, content))

    def receive(self) -> AsyncIterator[Message]:
        async def _empty() -> AsyncIterator[Message]:
            return
            yield  # pragma: no cover — makes this a (never-yielding) async generator

        return _empty()


class _UnusedReason:
    """A ReasonStrategy stand-in; never invoked (tests call execute() directly, not tick())."""

    async def reason(
        self, activity: Any, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        return result


def _registry_with(*tools: Tool) -> tuple[EnvironmentRegistry, FakeWorkspace]:
    workspace = FakeWorkspace("ws", _ORIGIN, list(tools))
    adapter = FakeAdapter("fake", workspace)
    registry = EnvironmentRegistry(adapters={_ORIGIN: adapter})
    return registry, workspace


def _cycle(
    registry: EnvironmentRegistry,
    tmp_path: Path,
    transport: RecordingTransport | None = None,
) -> tuple[DecisionCycle, WorkingMemory, SemanticMemory]:
    backend = FileMemoryBackend(tmp_path)
    semantic = SemanticMemory(backend)
    working = WorkingMemory(registry=registry)
    strategies = Strategies(
        observe=DefaultObserveStrategy(),
        reflect=DefaultReflectStrategy(),
        situate=DefaultSituateStrategy(),
        reason=_UnusedReason(),
        act=DefaultActStrategy(),
    )
    cycle = DecisionCycle(
        strategies=strategies,
        communication=transport or RecordingTransport(),
        actions=ActionRegistry(),
        registry=registry,
        working=working,
        semantic=semantic,
        procedural=ProceduralMemory(backend),
        episodic=EpisodicMemory(backend),
    )
    return cycle, working, semantic


# --------------------------------------------------------------------------------------------------
# Focus / Unfocus
# --------------------------------------------------------------------------------------------------


async def test_focus_subscribes_records_and_wires_signal_sink(tmp_path: Path) -> None:
    signal = Signal(name="new_email", payload={"n": 1})
    tool = FakeTool("EmailClientApp", signals_on_focus=[signal])
    registry, _ = _registry_with(tool)
    await registry.join(_ORIGIN)
    cycle, working, _ = _cycle(registry, tmp_path)

    ack = await FocusAction().execute(registry, cycle, activity_id="a1", tool_id="EmailClientApp")

    assert ack.ok is True
    assert working.focused_tools["EmailClientApp"] is tool
    assert tool.focused is True
    # The tool replayed its signal into whatever sink it got — proving it was the cycle's own.
    drained = [item async for item in cycle.signal_sink.drain()]
    assert drained == [("EmailClientApp", signal)]


async def test_unfocus_removes_and_calls_unfocus(tmp_path: Path) -> None:
    tool = FakeTool("EmailClientApp")
    registry, _ = _registry_with(tool)
    await registry.join(_ORIGIN)
    cycle, working, _ = _cycle(registry, tmp_path)
    await FocusAction().execute(registry, cycle, activity_id="a1", tool_id="EmailClientApp")

    ack = await UnfocusAction().execute(registry, cycle, activity_id="a1", tool_id="EmailClientApp")

    assert ack.ok is True
    assert "EmailClientApp" not in working.focused_tools
    assert tool.focused is False


async def test_unfocus_unknown_tool_is_noop(tmp_path: Path) -> None:
    registry, _ = _registry_with(FakeTool("EmailClientApp"))
    cycle, working, _ = _cycle(registry, tmp_path)

    ack = await UnfocusAction().execute(registry, cycle, activity_id="a1", tool_id="never-focused")

    assert ack.ok is True
    assert working.focused_tools == {}


# --------------------------------------------------------------------------------------------------
# Join / Leave
# --------------------------------------------------------------------------------------------------


async def test_join_registers_tools_and_persists_records(tmp_path: Path) -> None:
    tool = FakeTool("EmailClientApp", invoke_results={"list_emails": {"emails": []}})
    registry, workspace = _registry_with(tool)
    cycle, _, semantic = _cycle(registry, tmp_path)

    ack = await JoinAction().execute(registry, cycle, activity_id="a1", origin=_ORIGIN)

    # Registered live in the shared registry.
    assert ack.ok is True
    assert ack.result == {"workspace_id": "ws", "tool_ids": ["EmailClientApp"]}
    assert registry.get("EmailClientApp") is tool
    assert registry.get_workspace("ws") is workspace

    # Persisted durably via SemanticMemory (real file round-trip).
    ws_record = await semantic.retrieve_workspace_record("ws")
    assert ws_record is not None
    assert ws_record.origin == _ORIGIN

    tool_record = await semantic.retrieve_tool_record("EmailClientApp")
    assert tool_record is not None
    assert tool_record.manual_id == tool.manual.id
    assert tool_record.workspace_id == "ws"
    assert tool_record.address == tool.address

    stored_manual = await semantic.retrieve_manual(tool.manual.id)
    assert stored_manual is not None
    assert stored_manual.id == tool.manual.id


async def test_leave_closes_and_deregisters(tmp_path: Path) -> None:
    tool = FakeTool("EmailClientApp")
    registry, workspace = _registry_with(tool)
    await registry.join(_ORIGIN)
    cycle, _, _ = _cycle(registry, tmp_path)

    ack = await LeaveAction().execute(registry, cycle, activity_id="a1", workspace_id="ws")

    assert ack.ok is True
    assert workspace.closed is True
    with pytest.raises(KeyError):
        registry.get("EmailClientApp")
    with pytest.raises(KeyError):
        registry.get_workspace("ws")


# --------------------------------------------------------------------------------------------------
# Send
# --------------------------------------------------------------------------------------------------


async def test_send_delegates_to_transport(tmp_path: Path) -> None:
    transport = RecordingTransport()
    registry, _ = _registry_with(FakeTool("EmailClientApp"))
    cycle, _, _ = _cycle(registry, tmp_path, transport=transport)

    ack = await SendAction().execute(
        registry, cycle, activity_id="a1", to="agent-b", content={"greeting": "hi"}
    )

    assert ack.ok is True
    assert transport.sent == [("agent-b", {"greeting": "hi"})]


# --------------------------------------------------------------------------------------------------
# ActionRegistry lookup (promotes the spike's action_registry_lookup, generalized)
# --------------------------------------------------------------------------------------------------


async def test_action_registry_lookup_external() -> None:
    reg = ActionRegistry()
    actions: list[ExternalAction] = [
        InvokeAction(),
        FocusAction(),
        UnfocusAction(),
        JoinAction(),
        LeaveAction(),
        SendAction(),
    ]
    for action in actions:
        reg.register_external(action)
    for action in actions:
        assert reg.external(action.name) is action
