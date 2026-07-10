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


class EnvironmentRegistry:  # was ToolRegistry — now tracks workspaces, not just flattened tools
    """Live, in-process handles for workspaces (and their tools) the agent currently has a
    connection to. Populated by join()/restore() — never persisted directly (see
    WorkspaceRecord/ToolRecord)."""

    def __init__(self, adapters: dict[WorkspaceOrigin, WorkspaceAdapter] | None = None) -> None:
        """Keyed by the full origin (adapter + address), not just adapter name — an agent can
        join multiple workspaces that share a protocol (e.g. two separate MCP servers) without
        ambiguity."""

    def get(self, tool_id: str) -> Tool:
        raise NotImplementedError

    def get_workspace(self, workspace_id: str) -> Workspace:
        raise NotImplementedError

    def all_tools(self) -> list[Tool]:
        raise NotImplementedError

    async def join(self, origin: WorkspaceOrigin) -> Workspace:
        """Predefined external action _join_: looks up the adapter registered for this exact origin,
        calls its discover() (config-scoped to just this target today), registers the workspace."""
        raise NotImplementedError

    async def leave(self, workspace_id: str) -> None:
        """Predefined external action _leave_: closes the workspace's connection, deregisters it
        and all its tools."""
        raise NotImplementedError

    async def restore(
        self,
        workspace_records: list[WorkspaceRecord],
        tool_records: list[ToolRecord],
        semantic: SemanticMemory,
    ) -> list[Workspace]:
        """Reconnects to already-known workspaces via adapter.connect() — one call per workspace,
        looking up each one's adapter by workspace_record.origin, resolving each tool's manual from
        SemanticMemory first. Skips discovery entirely."""
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError  # return joined workspace ids
