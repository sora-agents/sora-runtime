"""``EnvironmentRegistry`` — join/get/leave, ADR-0014 id-uniqueness, and ``restore()``.

Promotes the walking-skeleton spike's registry assertions (``test_cycle_wiring.py``) into their
permanent home and extends them with what the spike never covered
([docs/phase-3-test-triage.md](../docs/phase-3-test-triage.md), task C2):

* **[ADR-0014](../docs/adrs/0014-tool-identity-globally-unique.md) enforcement** — a registry can't
  verify *global* tool-id uniqueness, but it enforces the slice it sees: a duplicate ``Tool.id`` at
  join fails loud and *atomically* (no half-registered workspace), and ``leave`` never deregisters a
  tool that belongs to another workspace.
* **``restore()``** — reconnect known workspaces via ``adapter.connect()`` (never ``discover()``),
  resolving each tool's manual from ``SemanticMemory``, one ``connect`` per workspace record.

Builds on the shared in-process double in ``tests/fakes.py`` (task C1) rather than bespoke stubs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fakes import FakeAdapter, FakeTool, FakeWorkspace, fake_manual
from sora.environment import (
    EnvironmentRegistry,
    Workspace,
    WorkspaceAdapter,
    WorkspaceOrigin,
)
from sora.manual import Manual, ToolRecord, WorkspaceRecord
from sora.memory import FileMemoryBackend, SemanticMemory

# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _origin(address: str = "fake://ws") -> WorkspaceOrigin:
    return WorkspaceOrigin(adapter="fake", address=address)


def _registry_with_workspace(
    ws_id: str, origin: WorkspaceOrigin, tools: list[FakeTool]
) -> tuple[EnvironmentRegistry, FakeWorkspace]:
    ws = FakeWorkspace(ws_id, origin, list(tools))
    registry = EnvironmentRegistry(adapters={origin: FakeAdapter("fake", ws)})
    return registry, ws


def _registry_with_workspaces(
    *specs: tuple[str, WorkspaceOrigin, list[FakeTool]],
) -> tuple[EnvironmentRegistry, dict[str, FakeWorkspace]]:
    adapters: dict[WorkspaceOrigin, WorkspaceAdapter] = {}
    workspaces: dict[str, FakeWorkspace] = {}
    for ws_id, origin, tools in specs:
        ws = FakeWorkspace(ws_id, origin, list(tools))
        adapters[origin] = FakeAdapter("fake", ws)
        workspaces[ws_id] = ws
    return EnvironmentRegistry(adapters=adapters), workspaces


class RecordingFakeAdapter(FakeAdapter):
    """Wraps the shared fake adapter to count ``discover``/``connect`` calls and capture the records
    handed to ``connect`` — so a test can prove ``restore()`` takes the connect (not discover) path
    and groups tool records per workspace."""

    def __init__(self, name: str, workspace: FakeWorkspace) -> None:
        super().__init__(name, workspace)
        self.discover_calls = 0
        self.connect_calls = 0
        self.connected_tool_ids: list[str] | None = None
        self.connected_manual_ids: list[str] | None = None

    async def discover(self) -> list[Workspace]:
        self.discover_calls += 1
        return await super().discover()

    async def connect(
        self,
        workspace_record: WorkspaceRecord,
        tool_records: list[ToolRecord],
        manuals: dict[str, Manual],
    ) -> Workspace:
        self.connect_calls += 1
        self.connected_tool_ids = [r.id for r in tool_records]
        self.connected_manual_ids = sorted(manuals)
        return await super().connect(workspace_record, tool_records, manuals)


def _tool_record(tool_id: str, manual_id: str, workspace_id: str) -> ToolRecord:
    return ToolRecord(
        id=tool_id,
        manual_id=manual_id,
        workspace_id=workspace_id,
        address=None,
        discovered_at=0.0,
        last_seen_at=0.0,
    )


def _workspace_record(ws_id: str, origin: WorkspaceOrigin) -> WorkspaceRecord:
    return WorkspaceRecord(id=ws_id, origin=origin, discovered_at=0.0, last_seen_at=0.0)


# --------------------------------------------------------------------------------------------------
# Promoted: join / get / leave (from the walking-skeleton spike)
# --------------------------------------------------------------------------------------------------


async def test_join_registers_workspace_and_tools() -> None:
    origin = _origin()
    tool = FakeTool("EmailClientApp", invoke_results={"list_emails": {"emails": []}})
    registry, ws = _registry_with_workspace("ws", origin, [tool])
    joined = await registry.join(origin)
    assert joined is ws
    assert registry.get("EmailClientApp") is tool
    assert registry.get_workspace("ws") is ws
    assert [t.id for t in registry.all_tools()] == ["EmailClientApp"]


async def test_leave_closes_and_deregisters() -> None:
    origin = _origin()
    registry, ws = _registry_with_workspace("ws", origin, [FakeTool("EmailClientApp")])
    await registry.join(origin)
    await registry.leave("ws")
    assert ws.closed is True
    with pytest.raises(KeyError):
        registry.get("EmailClientApp")
    with pytest.raises(KeyError):
        registry.get_workspace("ws")


async def test_joined_workspaces_reflects_join_and_leave() -> None:
    origin = _origin()
    registry, ws = _registry_with_workspace("ws", origin, [FakeTool("EmailClientApp")])
    assert registry.joined_workspaces() == []
    await registry.join(origin)
    assert registry.joined_workspaces() == [ws]
    await registry.leave("ws")
    assert registry.joined_workspaces() == []


# --------------------------------------------------------------------------------------------------
# ADR-0014: id-uniqueness enforced fail-loud and atomically at join
# --------------------------------------------------------------------------------------------------


async def test_join_duplicate_tool_id_across_workspaces_raises() -> None:
    origin_a, origin_b = _origin("fake://a"), _origin("fake://b")
    registry, _ = _registry_with_workspaces(
        ("ws-a", origin_a, [FakeTool("shared-tool")]),
        ("ws-b", origin_b, [FakeTool("shared-tool")]),
    )
    await registry.join(origin_a)
    with pytest.raises(ValueError, match="shared-tool"):
        await registry.join(origin_b)


async def test_failed_join_is_atomic() -> None:
    # The rejected workspace leaves *no* trace: not its id, not its non-colliding tools, and the
    # already-registered tool still resolves to the original instance.
    origin_a, origin_b = _origin("fake://a"), _origin("fake://b")
    original = FakeTool("shared-tool")
    registry, workspaces = _registry_with_workspaces(
        ("ws-a", origin_a, [original]),
        ("ws-b", origin_b, [FakeTool("shared-tool"), FakeTool("unique-tool")]),
    )
    await registry.join(origin_a)
    with pytest.raises(ValueError):
        await registry.join(origin_b)
    assert registry.joined_workspaces() == [workspaces["ws-a"]]
    with pytest.raises(KeyError):
        registry.get_workspace("ws-b")
    assert registry.get("shared-tool") is original
    with pytest.raises(KeyError):
        registry.get("unique-tool")  # rejected workspace's other tool did not leak in


async def test_join_duplicate_tool_id_within_workspace_raises() -> None:
    origin = _origin()
    registry, _ = _registry_with_workspace("ws", origin, [FakeTool("dup"), FakeTool("dup")])
    with pytest.raises(ValueError, match="dup"):
        await registry.join(origin)
    assert registry.joined_workspaces() == []  # nothing partially registered


async def test_join_duplicate_workspace_id_raises() -> None:
    origin_a, origin_b = _origin("fake://a"), _origin("fake://b")
    registry, _ = _registry_with_workspaces(
        ("dup-ws", origin_a, [FakeTool("tool-a")]),
        ("dup-ws", origin_b, [FakeTool("tool-b")]),
    )
    await registry.join(origin_a)
    with pytest.raises(ValueError, match="dup-ws"):
        await registry.join(origin_b)
    assert registry.get("tool-a").id == "tool-a"
    with pytest.raises(KeyError):
        registry.get("tool-b")  # the rejected workspace's tool never registered


# --------------------------------------------------------------------------------------------------
# ADR-0014: leave never pops a shared id (no cross-workspace deregistration)
# --------------------------------------------------------------------------------------------------


async def test_leave_does_not_deregister_other_workspaces_tools() -> None:
    origin_a, origin_b = _origin("fake://a"), _origin("fake://b")
    tool_a, tool_b = FakeTool("tool-a"), FakeTool("tool-b")
    registry, workspaces = _registry_with_workspaces(
        ("ws-a", origin_a, [tool_a]),
        ("ws-b", origin_b, [tool_b]),
    )
    await registry.join(origin_a)
    await registry.join(origin_b)
    await registry.leave("ws-a")
    assert workspaces["ws-a"].closed is True
    # ws-b and its tool are wholly untouched — leave only removed ws-a's own ids.
    assert registry.get("tool-b") is tool_b
    assert registry.get_workspace("ws-b") is workspaces["ws-b"]
    assert workspaces["ws-b"].closed is False


# --------------------------------------------------------------------------------------------------
# restore(): reconnect known workspaces via connect(), resolving manuals from SemanticMemory
# --------------------------------------------------------------------------------------------------


async def test_restore_reconnects_via_connect_not_discover(tmp_path: Path) -> None:
    origin = _origin()
    semantic = SemanticMemory(FileMemoryBackend(tmp_path))
    manual = fake_manual("email-client")
    await semantic.store_manual(manual)
    adapter = RecordingFakeAdapter("fake", FakeWorkspace("ws", origin, []))
    registry = EnvironmentRegistry(adapters={origin: adapter})

    restored = await registry.restore(
        [_workspace_record("ws", origin)],
        [_tool_record("EmailClientApp", "email-client", "ws")],
        semantic,
    )

    assert adapter.discover_calls == 0  # restore skips discovery entirely
    assert adapter.connect_calls == 1
    assert [w.id for w in restored] == ["ws"]
    assert registry.get_workspace("ws").id == "ws"
    # The manual resolved from SemanticMemory reached connect and was stamped on the rebuilt tool.
    assert registry.get("EmailClientApp").manual == manual
    assert adapter.connected_manual_ids == ["email-client"]


async def test_restore_groups_tool_records_by_workspace(tmp_path: Path) -> None:
    origin_a, origin_b = _origin("fake://a"), _origin("fake://b")
    semantic = SemanticMemory(FileMemoryBackend(tmp_path))
    await semantic.store_manual(fake_manual("man-a"))
    await semantic.store_manual(fake_manual("man-b"))
    adapter_a = RecordingFakeAdapter("fake", FakeWorkspace("ws-a", origin_a, []))
    adapter_b = RecordingFakeAdapter("fake", FakeWorkspace("ws-b", origin_b, []))
    registry = EnvironmentRegistry(adapters={origin_a: adapter_a, origin_b: adapter_b})

    restored = await registry.restore(
        [_workspace_record("ws-a", origin_a), _workspace_record("ws-b", origin_b)],
        [_tool_record("tool-a", "man-a", "ws-a"), _tool_record("tool-b", "man-b", "ws-b")],
        semantic,
    )

    assert {w.id for w in restored} == {"ws-a", "ws-b"}
    # Each connect saw only its own workspace's records — the grouping, made explicit.
    assert adapter_a.connect_calls == 1
    assert adapter_a.connected_tool_ids == ["tool-a"]
    assert adapter_b.connect_calls == 1
    assert adapter_b.connected_tool_ids == ["tool-b"]
    assert registry.get("tool-a").manual == fake_manual("man-a")
    assert registry.get("tool-b").manual == fake_manual("man-b")


async def test_restore_duplicate_tool_id_raises(tmp_path: Path) -> None:
    # restore goes through the same id-uniqueness enforcement as join.
    origin_a, origin_b = _origin("fake://a"), _origin("fake://b")
    semantic = SemanticMemory(FileMemoryBackend(tmp_path))
    await semantic.store_manual(fake_manual("man"))
    adapter_a = FakeAdapter("fake", FakeWorkspace("ws-a", origin_a, []))
    adapter_b = FakeAdapter("fake", FakeWorkspace("ws-b", origin_b, []))
    registry = EnvironmentRegistry(adapters={origin_a: adapter_a, origin_b: adapter_b})

    with pytest.raises(ValueError, match="dup"):
        await registry.restore(
            [_workspace_record("ws-a", origin_a), _workspace_record("ws-b", origin_b)],
            [_tool_record("dup", "man", "ws-a"), _tool_record("dup", "man", "ws-b")],
            semantic,
        )
