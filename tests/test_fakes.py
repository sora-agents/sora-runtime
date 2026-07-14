"""Characterization tests for the reusable in-process fake adapter double (``tests/fakes.py``).

Deliberately lean: the fake is a *test double*, so its structural conformance to the
``Tool``/``Workspace``/``WorkspaceAdapter`` Protocols is already enforced by ``mypy --strict``
(which scans ``tests/``), and its behaviour is exercised transitively by the environment, focus, and
observe tests that build on it. This file pins only the non-trivial logic that has *no current
consumer* yet — the parts a downstream author will assume already work. Trivial getters
(``tools()``/``close()``/``observe()``) and conformance are intentionally not re-asserted here.
"""

from __future__ import annotations

from fakes import FakeAdapter, FakeTool, FakeWorkspace, fake_manual
from sora.environment import EnvironmentRegistry, WorkspaceOrigin
from sora.manual import ToolRecord, WorkspaceRecord
from sora.perception import NotificationQueueSink
from sora.types import Signal


async def test_invoke_canned_result_and_unconfigured_op_returns_not_ok() -> None:
    tool = FakeTool("EmailClientApp", invoke_results={"list_emails": {"emails": [], "total": 0}})

    ack = await tool.invoke("list_emails", limit=5)
    assert ack.ok is True
    assert ack.result == {"emails": [], "total": 0}
    assert tool.invoked_with == ("list_emails", {"limit": 5})

    # An operation the fake wasn't configured for is rejected via the ack, not an exception — the
    # tool-level failure travels the same async-ack channel as a success (see OperationAck).
    rejected = await tool.invoke("nonexistent_op")
    assert rejected.ok is False


async def test_focus_emits_configured_signals_to_sink() -> None:
    signals = [
        Signal(name="email_received", payload={"from": "a@x"}),
        Signal(name="email_received", payload={"from": "b@x"}),
    ]
    tool = FakeTool("EmailClientApp", signals_on_focus=signals)
    sink: NotificationQueueSink[Signal] = NotificationQueueSink()

    await tool.focus(sink)

    drained = [item async for item in sink.drain()]
    assert drained == [("EmailClientApp", signals[0]), ("EmailClientApp", signals[1])]


async def test_connect_rebuilds_from_records_with_address_fallback() -> None:
    origin = WorkspaceOrigin(adapter="fake", address="fake://ws")
    manual = fake_manual("email-client")
    ws_record = WorkspaceRecord(id="ws", origin=origin, discovered_at=0.0, last_seen_at=0.0)
    tool_records = [
        # No own address -> resolves to the workspace origin's address.
        ToolRecord(
            id="EmailClientApp",
            manual_id="email-client",
            workspace_id="ws",
            address=None,
            discovered_at=0.0,
            last_seen_at=0.0,
        ),
        # Own address -> overrides the workspace origin's address.
        ToolRecord(
            id="Device",
            manual_id="email-client",
            workspace_id="ws",
            address="dev://own",
            discovered_at=0.0,
            last_seen_at=0.0,
        ),
    ]
    adapter = FakeAdapter("fake", FakeWorkspace("ws", origin, []))

    workspace = await adapter.connect(ws_record, tool_records, {"email-client": manual})

    rebuilt = {t.id: t for t in workspace.tools()}
    assert set(rebuilt) == {"EmailClientApp", "Device"}
    assert rebuilt["EmailClientApp"].manual is manual
    assert rebuilt["EmailClientApp"].address == "fake://ws"  # origin fallback
    assert rebuilt["Device"].address == "dev://own"  # own address wins


async def test_discover_join_roundtrip() -> None:
    # The fake plugs into the *real* EnvironmentRegistry.join() path — no bespoke registry.
    tool = FakeTool("EmailClientApp", invoke_results={"list_emails": {"emails": []}})
    origin = WorkspaceOrigin(adapter="fake", address="fake://ws")
    workspace = FakeWorkspace("ws", origin, [tool])
    registry = EnvironmentRegistry(adapters={origin: FakeAdapter("fake", workspace)})

    joined = await registry.join(origin)

    assert joined is workspace
    assert registry.get("EmailClientApp") is tool
    assert [t.id for t in registry.all_tools()] == ["EmailClientApp"]
