"""ARE-over-MCP ``WorkspaceAdapter`` — a curating subclass of the generic ``McpWorkspaceAdapter``.

ARE lifts its own abstraction on top of plain MCP: the flat MCP tool list is namespaced
``<App>__<operation>``, and each app also exposes a state resource (``app://{app}/state``) whose
``resource_updated`` notifications are signals. This adapter overrides just the curation hooks —
grouping (flat names -> one tool per app), name assembly (its inverse), and the observable bindings
(the app's state resource) — reusing the base for all the generic MCP mechanics (stdio spawn,
session, ``call_tool``, result parsing, resource read/subscribe, signal routing, lifecycle).

The ``<App>__`` convention and the ``app://{app}/state`` mapping are **ARE curation**, not canonical
MCP: a vanilla MCP server has neither, and the base surfaces no observables for one (see ADR-0003,
ADR-0004, and the base module docstring).
"""

from __future__ import annotations

from typing import Any

from sora.adapters.mcp import (
    McpWorkspaceAdapter,
    _ResourceBinding,
    _ToolBlueprint,
)
from sora.manual import OperationSpecification

_ARE_SEP = "__"  # ARE-specific: the <App>__<operation> tool-namespacing separator


class AreMcpWorkspaceAdapter(McpWorkspaceAdapter):
    name = "are-mcp"  # matches WorkspaceOrigin.adapter; distinct from the base's generic "mcp"

    def _group(self, mcp_tools: list[Any]) -> list[_ToolBlueprint]:
        # Group the flat <App>__<op> list into one blueprint per app; un-namespaced tools are
        # skipped (not part of ARE's app model).
        grouped: dict[str, list[OperationSpecification]] = {}
        for mcp_tool in mcp_tools:
            app, _, op = mcp_tool.name.partition(_ARE_SEP)
            if not op:
                continue
            grouped.setdefault(app, []).append(
                OperationSpecification(
                    name=op,
                    description=mcp_tool.description or "",
                    parameters=dict(mcp_tool.inputSchema or {}),
                )
            )
        return [
            _ToolBlueprint(
                seed=app,
                manual_id=app,
                description=f"ARE app {app}, imported over MCP",
                operations=operations,
                metadata={"app": app},
            )
            for app, operations in grouped.items()
        ]

    def _mcp_name(self, seed: str, operation_name: str) -> str:
        return f"{seed}{_ARE_SEP}{operation_name}"

    def _observable_bindings(self, seed: str) -> list[_ResourceBinding]:
        # Each ARE app exposes its state as one resource; its updates are one signal.
        return [
            _ResourceBinding(
                uri=f"app://{seed}/state",
                property_name="state",
                signal_name="state_changed",
            )
        ]
