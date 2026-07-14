"""Reusable in-process fake ``WorkspaceAdapter`` (and its ``Tool``/``Workspace``) for tests.

A deterministic, subprocess-free stand-in for a real adapter (MCP, WoT, ...). It satisfies the
``Tool``/``Workspace``/``WorkspaceAdapter`` Protocols in ``sora.environment`` structurally, so it
plugs into the real ``EnvironmentRegistry.join()``/``restore()`` paths — no bespoke registry needed.
Promoted from the walking-skeleton spike's in-file fakes into one shared double, so the environment,
action, and decision-cycle tests import it instead of copy-pasting their own tool/workspace/adapter
stubs.

This is a helper module, not a test module (pytest only collects ``test_*.py``); its own contract is
pinned by ``tests/test_fakes.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from sora.environment import Tool, Workspace, WorkspaceOrigin
from sora.manual import Manual, OperationSpecification, ToolRecord, WorkspaceRecord
from sora.perception import SignalSink
from sora.types import ObservableProperty, OperationAck, Signal


def fake_manual(manual_id: str, operations: Iterable[str] = ()) -> Manual:
    """An envelope ``Manual`` mirroring the hand-authored Markdown channel: id + description, the
    structured spec lists empty by default (see ADR-0015). Pass ``operations`` to add minimally
    described ``OperationSpecification`` entries when a test wants named ops."""
    return Manual(
        id=manual_id,
        metadata={},
        description=f"fake {manual_id}",
        observable_properties=[],
        signals=[],
        operations=[
            OperationSpecification(name=op, description="", parameters={}) for op in operations
        ],
    )


class FakeTool:
    """Canned-response tool. ``invoke`` returns the ``OperationAck`` configured for the operation
    (``ok=False`` for an unconfigured one) and records the call; ``focus`` replays configured
    signals into the sink; ``observe`` returns configured properties."""

    def __init__(
        self,
        tool_id: str,
        *,
        manual: Manual | None = None,
        address: str | None = None,
        invoke_results: dict[str, Any] | None = None,
        properties: Sequence[ObservableProperty] = (),
        signals_on_focus: Sequence[Signal] = (),
    ) -> None:
        self.id = tool_id
        self.manual = (
            manual if manual is not None else fake_manual(tool_id, (invoke_results or {}).keys())
        )
        self.address = address
        self._invoke_results = invoke_results or {}
        self._properties = list(properties)
        self._signals_on_focus = list(signals_on_focus)
        self.invoked_with: tuple[str, dict[str, Any]] | None = None  # last call, for assertions
        self.invocations: list[tuple[str, dict[str, Any]]] = []  # full call log
        self.focused = False

    async def invoke(self, operation_name: str, **params: Any) -> OperationAck:
        self.invoked_with = (operation_name, params)
        self.invocations.append((operation_name, params))
        if operation_name not in self._invoke_results:
            return OperationAck(ok=False, result=f"unknown operation {operation_name!r}")
        return OperationAck(ok=True, result=self._invoke_results[operation_name])

    async def focus(self, sink: SignalSink) -> None:
        self.focused = True
        for signal in self._signals_on_focus:
            sink.push(self.id, signal)

    async def unfocus(self) -> None:
        self.focused = False

    def observe(self) -> list[ObservableProperty]:
        return list(self._properties)


class FakeWorkspace:
    # Method returns are typed as the Protocol types (list[Tool], not list[FakeTool]) so the fake
    # structurally satisfies Workspace — list is invariant, so a concrete element type would not.
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
    """``discover()`` yields the workspace it was built with (config-scoped: exactly one, matching
    the real single-workspace-per-adapter join today). ``connect()`` rebuilds a workspace from
    durable records, resolving each tool's address (own address, else the workspace origin's) and
    its manual."""

    def __init__(self, name: str, workspace: FakeWorkspace) -> None:
        self.name = name
        self._workspace = workspace

    async def discover(self) -> list[Workspace]:
        return [self._workspace]

    async def connect(
        self,
        workspace_record: WorkspaceRecord,
        tool_records: list[ToolRecord],
        manuals: dict[str, Manual],
    ) -> Workspace:
        tools: list[Tool] = [
            FakeTool(
                record.id,
                manual=manuals[record.manual_id],
                address=record.address or workspace_record.origin.address,
            )
            for record in tool_records
        ]
        return FakeWorkspace(workspace_record.id, workspace_record.origin, tools)
