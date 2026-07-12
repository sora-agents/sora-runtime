"""ARE-over-MCP ``WorkspaceAdapter`` — the walking-skeleton spike.

This adapter is **ARE-specific**: it speaks MCP as the wire protocol but bakes in ARE's own data
model on top of it — the ``<App>__<operation>`` tool-naming convention and (later) ARE's
``app://{app}/state`` resources. It is not a general-purpose MCP adapter.

Planned split (not built yet — see docs/phase-2-findings.md): the sections marked *generic MCP
mechanics* below (stdio spawn, session, ``call_tool``, result parsing, workspace lifecycle) extract
into a protocol-only ``McpWorkspaceAdapter`` base in ``sora.adapters.mcp``, and this class becomes
``AreMcpWorkspaceAdapter(McpWorkspaceAdapter)`` overriding just two hooks — *grouping*
(``list[mcp.Tool] -> S-ORA Tools``) and its inverse *name assembly* (``(tool_id, op) -> mcp_name``).
The base's default grouping for a plain MCP server is deliberately left undesigned until there's a
second, non-ARE MCP consumer to draw the seam from, rather than guessing it from one example.

Deliberately minimal and throwaway: only the ``invoke`` path is fleshed out; ``focus``/``observe``
(ARE observable properties + the ``resource_updated`` signal) are stubbed and noted as gaps. stdio
(not SSE) is used here — no port to bind, no long-lived HTTP server — far more robust for a gated
integration test; EXAMPLES.md's SSE form remains valid, this is a transport choice.
"""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from sora.environment import WorkspaceOrigin
from sora.manual import Manual, OperationSpecification
from sora.types import ObservableProperty, OperationAck

if TYPE_CHECKING:
    from sora.environment import Tool, Workspace
    from sora.manual import ToolRecord, WorkspaceRecord
    from sora.perception import SignalSink

_ARE_SEP = "__"  # ARE-specific: the <App>__<operation> tool-namespacing separator


class AreMcpWorkspaceAdapter:
    """Spawns ARE's MCP server as a stdio subprocess and imports its app tools. One adapter instance
    is config-scoped to exactly one workspace (see WorkspaceAdapter.discover)."""

    name = "are-mcp"  # matches WorkspaceOrigin.adapter; distinct from the future generic "mcp"

    def __init__(
        self,
        *,
        command: str,
        args: list[str],
        workspace_id: str,
        origin: WorkspaceOrigin,
        env: dict[str, str] | None = None,
    ) -> None:
        self._params = StdioServerParameters(command=command, args=args, env=env)
        self._workspace_id = workspace_id
        self._origin = origin

    async def discover(self) -> list[Workspace]:
        # --- generic MCP mechanics (candidate for a shared McpWorkspaceAdapter base) ---
        stack = AsyncExitStack()
        read, write = await stack.enter_async_context(stdio_client(self._params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = await session.list_tools()

        # --- ARE-specific: group the flat <App>__<op> tool list into one Tool per app ---
        grouped: dict[str, list[OperationSpecification]] = {}
        for mcp_tool in listed.tools:
            app, _, op = mcp_tool.name.partition(_ARE_SEP)
            if not op:  # un-namespaced tool — skip in the spike, note as a gap
                continue
            grouped.setdefault(app, []).append(
                OperationSpecification(
                    name=op,
                    description=mcp_tool.description or "",
                    parameters=dict(mcp_tool.inputSchema or {}),
                )
            )

        tools: list[Tool] = [
            _AreMcpTool(tool_id=app, manual=_synth_manual(app, ops), session=session)
            for app, ops in grouped.items()
        ]
        workspace = _AreMcpWorkspace(
            ws_id=self._workspace_id, origin=self._origin, tools=tools, stack=stack
        )
        return [workspace]

    async def connect(
        self,
        workspace_record: WorkspaceRecord,
        tool_records: list[ToolRecord],
        manuals: dict[str, Manual],
    ) -> Workspace:
        raise NotImplementedError  # restore-from-records is out of scope for the spike


def _synth_manual(app: str, operations: list[OperationSpecification]) -> Manual:
    # ARE-specific: an ARE "app" becomes one S-ORA tool type.
    return Manual(
        id=app,
        metadata={"source": "are-mcp", "app": app},
        description=f"ARE app {app}, imported over MCP",
        observable_properties=[],  # not built yet: derive from ARE's app://{app}/state resource
        signals=[],  # not built yet: derive from ARE's resource_updated notifications
        operations=operations,
        usage_protocols="",
    )


class _AreMcpTool:
    def __init__(self, *, tool_id: str, manual: Manual, session: ClientSession) -> None:
        self.id = tool_id
        self.manual = manual
        self.address: str | None = None  # rides the workspace's single stdio connection
        self._session = session

    async def invoke(self, operation_name: str, **params: Any) -> OperationAck:
        # ARE-specific name assembly (inverse of discover's grouping); the call_tool round-trip
        # itself and _parse_result are generic MCP mechanics.
        result = await self._session.call_tool(
            f"{self.id}{_ARE_SEP}{operation_name}", arguments=params or None
        )
        return OperationAck(ok=not result.isError, result=_parse_result(result))

    async def focus(self, sink: SignalSink) -> None:
        raise NotImplementedError  # ARE resource_updated -> Signal wiring not built yet

    async def unfocus(self) -> None:
        raise NotImplementedError  # not built yet

    def observe(self) -> list[ObservableProperty]:
        return []  # not built yet — poll ARE's app://{id}/state


class _AreMcpWorkspace:
    # Generic MCP mechanics (candidate for the shared base) — nothing ARE-specific here.
    def __init__(
        self, *, ws_id: str, origin: WorkspaceOrigin, tools: list[Tool], stack: AsyncExitStack
    ) -> None:
        self.id = ws_id
        self.origin = origin
        self._tools = tools
        self._stack = stack

    def tools(self) -> list[Tool]:
        return self._tools

    async def close(self) -> None:
        await self._stack.aclose()  # tears down the ClientSession + the stdio subprocess


def _parse_result(result: Any) -> Any:
    """Generic MCP: prefer structured output; otherwise decode text content (JSON if it parses)."""
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    texts = [c.text for c in result.content if isinstance(c, TextContent)]
    if not texts:
        return None
    blob = texts[0] if len(texts) == 1 else "\n".join(texts)
    try:
        return json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return blob
