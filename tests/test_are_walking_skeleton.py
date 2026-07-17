"""Walking skeleton: one Observe→Act decision cycle against the *real* ARE MCP server.

This is the single integration-level test the walking skeleton asks for (not per-layer units). It
spawns ARE's in-package MCP server over stdio, exposing the EmailClientApp, and drives S-ORA's real
decision cycle + real MCP adapter through exactly one external action — ``invoke
EmailClientApp.list_emails`` — asserting the tool's ``OperationAck`` comes back through the cycle.

It is **opt-in and skip-gated**: marked ``integration`` (excluded from the default ``pytest`` run —
see pyproject ``addopts``), so run it explicitly with ``uv run pytest -m integration``. It also
needs both the ``mcp`` extra and the ARE package (a PEP 735 ``are`` dependency-group, deliberately
not pulled by ``uv sync --all-extras``), so CI skips it and it runs only when ARE is installed
(``uv sync --all-extras --group are``). It exercises the (now hardened) adapter end-to-end against
the real server; the deterministic per-layer contract lives in ``test_mcp_adapter.py``.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from typing import Any

import pytest

from sora.action import default_action_registry
from sora.activity import Activity
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, WorkspaceOrigin
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
from sora.types import Step

SERVER = "are.simulation.apps.mcp.server.are_simulation_mcp_server"
EMAIL_APP = "are.simulation.apps.email_client.EmailClientApp"


class NullTransport:
    async def send(self, to: str, content: dict[str, Any]) -> None: ...

    async def receive(self) -> AsyncIterator[Message]:
        return
        yield  # pragma: no cover — makes this a (never-yielding) async generator


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
    """The hardcoded spike ReasonStrategy: the scenario's single first step, nothing more. The tool
    id is now origin-qualified (ADR-0014), so it's injected rather than a bare ``"EmailClientApp"``
    literal."""

    def __init__(self, tool_id: str) -> None:
        self._tool_id = tool_id

    async def reason(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        if activity.last_operation is not None:
            return TickResult(activity=activity, step=Step(next_action="wait", params={}))
        return TickResult(
            activity=activity,
            step=Step(
                next_action="invoke",
                params={"tool_id": self._tool_id, "operation_name": "list_emails"},
            ),
        )


@pytest.mark.integration
async def test_walking_skeleton_list_emails_against_are() -> None:
    pytest.importorskip("mcp")
    pytest.importorskip("are.simulation")
    from sora.adapters.are_mcp import AreMcpWorkspaceAdapter

    origin = WorkspaceOrigin(adapter="are-mcp", address="stdio:are-email")
    # Tool ids are origin-qualified (ADR-0014) and derived deterministically, so the expected id is
    # known before the connection exists.
    email_tool_id = f"{origin.address}/EmailClientApp"
    adapter = AreMcpWorkspaceAdapter(
        command=sys.executable,
        args=["-m", SERVER, "--apps", EMAIL_APP, "--transport", "stdio"],
        workspace_id="are",
        origin=origin,
    )
    registry = EnvironmentRegistry(adapters={origin: adapter})
    working = WorkingMemory(registry=registry)
    backend = DictBackend()
    actions = default_action_registry()
    strategies = Strategies(
        observe=DefaultObserveStrategy(),
        reflect=DefaultReflectStrategy(),
        situate=DefaultSituateStrategy(),
        reason=ListEmailsReasonStrategy(email_tool_id),
        act=DefaultActStrategy(),
    )
    cycle = DecisionCycle(
        strategies=strategies,
        communication=NullTransport(),
        actions=actions,
        registry=registry,
        working=working,
        semantic=SemanticMemory(backend),
        procedural=ProceduralMemory(backend),
        episodic=EpisodicMemory(backend),
    )

    workspace = await registry.join(origin)  # spawns + connects to the real ARE MCP server
    try:
        assert email_tool_id in [t.id for t in workspace.tools()]

        working.activities["schedule"] = Activity(
            id="schedule", goal="list emails from the inbox", context={}
        )
        activity = working.activities["schedule"]

        for _ in range(10):
            await cycle.tick()
            await asyncio.sleep(0.05)  # real subprocess I/O — let the off-cycle invoke resolve
            if activity.last_operation is not None:
                break

        assert activity.last_operation is not None, "list_emails never resolved through the cycle"
        assert activity.last_operation.ok is True
        assert isinstance(activity.last_operation.result, dict)
        assert "emails" in activity.last_operation.result
    finally:
        await registry.leave("are")  # closes the MCP session + terminates the subprocess
