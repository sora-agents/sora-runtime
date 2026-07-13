"""Usage interface + adapters (S-ORA does not author tools, only consumes them)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from sora.manual import Manual, ToolRecord, WorkspaceRecord
    from sora.memory import SemanticMemory
    from sora.perception import SignalSink
    from sora.types import ObservableProperty, OperationAck


class Tool(Protocol):
    id: str
    manual: Manual
    address: str | None  # overrides the workspace's address when this tool has its own endpoint

    async def invoke(self, operation_name: str, **params: Any) -> OperationAck: ...

    async def focus(self, sink: SignalSink) -> None: ...

    async def unfocus(self) -> None: ...

    def observe(self) -> list[ObservableProperty]: ...


@dataclass(frozen=True)
class WorkspaceOrigin:
    """The part of a WorkspaceRecord only the adapter can know: how to (re)connect."""

    adapter: str  # e.g. "mcp", "wot" — matches WorkspaceAdapter.name
    address: str  # e.g. an MCP server URI, or a WoT directory's base href


class Workspace(Protocol):
    """A shared connection/lifecycle and tool-use-protocol boundary: e.g. one MCP session, or one
    WoT-described environment, however many tools it exposes. Tools within a workspace stay
    individually focusable, and may have their own address; the workspace's own connection —
    however many of its tools actually use it — is (re)established once."""

    id: str  # matches WorkspaceRecord.id / ToolRecord.workspace_id
    origin: WorkspaceOrigin

    def tools(self) -> list[Tool]: ...

    async def close(self) -> None: ...  # contained tools go stale together


class WorkspaceAdapter(Protocol):  # was ToolAdapter — it always operated at workspace granularity
    """Imports externally-defined tools (MCP, OpenAPI, WoT, ...) into the S-ORA usage interface.
    The tool-use protocol is fixed once per workspace (e.g. all-MCP, all-WoT); per-tool addressing
    within that protocol (see Tool.address) is a separate, orthogonal concern."""

    name: str  # e.g. "mcp" — matches WorkspaceOrigin.adapter

    async def discover(self) -> list[Workspace]:
        """Enumerates workspaces this adapter can reach. Today, each configured adapter instance is
        scoped to exactly one workspace (config-driven join — see Tool Model and Use); the same
        method is what open, dynamic discovery would call too, once that's in scope."""
        ...

    async def connect(
        self,
        workspace_record: WorkspaceRecord,
        tool_records: list[ToolRecord],
        manuals: dict[str, Manual],
    ) -> Workspace:
        """Re-establishes a workspace from its known records — one connection, all its tools
        rebuilt, no re-fetching manuals. Per tool_record: uses tool_record.address if set, else
        falls back to workspace_record.origin.address."""
        ...


class EnvironmentView(Protocol):
    """Read-only projection of the live environment that WorkingMemory exposes to strategies: they
    reason over the currently-joined workspaces and tools — a legitimate part of the agent's current
    context — but cannot mutate connections through it (join/leave/restore live only in the action
    space; mypy --strict enforces that). EnvironmentRegistry satisfies this structurally and adds
    the mutators. See ADR-0013."""

    def get(self, tool_id: str) -> Tool: ...

    def get_workspace(self, workspace_id: str) -> Workspace: ...

    def all_tools(self) -> list[Tool]: ...

    def joined_workspaces(self) -> list[Workspace]: ...


class EnvironmentRegistry:
    """Live, in-process handles for workspaces (and their tools) the agent currently has a
    connection to. Populated by join()/restore() — never persisted directly (see
    WorkspaceRecord/ToolRecord). The single shared instance: DecisionCycle holds it mutation-capable
    for action dispatch; WorkingMemory mirrors it read-only as an EnvironmentView."""

    def __init__(self, adapters: dict[WorkspaceOrigin, WorkspaceAdapter] | None = None) -> None:
        """Keyed by the full origin (adapter + address), not just adapter name — an agent can
        join multiple workspaces that share a protocol (e.g. two separate MCP servers) without
        ambiguity."""
        self._adapters: dict[WorkspaceOrigin, WorkspaceAdapter] = adapters or {}
        self._workspaces: dict[str, Workspace] = {}
        self._tools: dict[str, Tool] = {}
        self._workspace_tools: dict[str, list[str]] = {}  # ws id -> its tool ids, for leave()

    def get(self, tool_id: str) -> Tool:
        return self._tools[tool_id]

    def get_workspace(self, workspace_id: str) -> Workspace:
        return self._workspaces[workspace_id]

    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def joined_workspaces(self) -> list[Workspace]:  # the live joined set (EnvironmentView)
        return list(self._workspaces.values())

    async def join(self, origin: WorkspaceOrigin) -> Workspace:
        """Predefined external action _join_: looks up the adapter registered for this exact origin,
        calls its discover() (config-scoped to just this target today), registers the workspace."""
        adapter = self._adapters[origin]
        workspace = (await adapter.discover())[0]  # config-scoped: exactly one workspace today
        self._register(workspace)
        return workspace

    def _register(self, workspace: Workspace) -> None:
        self._workspaces[workspace.id] = workspace
        tool_ids = []
        for tool in workspace.tools():
            self._tools[tool.id] = tool
            tool_ids.append(tool.id)
        self._workspace_tools[workspace.id] = tool_ids

    async def leave(self, workspace_id: str) -> None:
        """Predefined external action _leave_: closes the workspace's connection, deregisters it
        and all its tools."""
        workspace = self._workspaces.pop(workspace_id)
        for tool_id in self._workspace_tools.pop(workspace_id, []):
            self._tools.pop(tool_id, None)
        await workspace.close()

    async def restore(
        self,
        workspace_records: list[WorkspaceRecord],
        tool_records: list[ToolRecord],
        semantic: SemanticMemory,
    ) -> list[Workspace]:
        """Reconnects to already-known workspaces via adapter.connect() — one call per workspace,
        looking up each one's adapter by workspace_record.origin, resolving each tool's manual from
        SemanticMemory first. Skips discovery entirely."""
        raise NotImplementedError  # not built yet — untouched by the walking-skeleton spike

    def __repr__(self) -> str:
        return f"EnvironmentRegistry(workspaces={sorted(self._workspaces)})"
