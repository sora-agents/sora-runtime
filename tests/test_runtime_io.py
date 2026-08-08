"""The runtime-IO adapter: the agent's user-reply channel surfaced as a uniform tool
(``send_message_to_user``) rather than a special ``send`` action. The tool delegates to
whatever MessageTransport bootstrap wired, so one always-joined tool is correct in every
environment."""

from __future__ import annotations

import pytest

from sora.adapters.runtime_io import (
    RUNTIME_IO_ADAPTER,
    RUNTIME_IO_ADDRESS,
    SEND_MESSAGE_TO_USER,
    RuntimeIOAdapter,
    user_channel_manual,
)
from sora.environment import WorkspaceOrigin
from sora.manual import WorkspaceRecord
from sora.transport import InProcessTransport

_ORIGIN = WorkspaceOrigin(adapter=RUNTIME_IO_ADAPTER, address=RUNTIME_IO_ADDRESS)


def _adapter() -> tuple[RuntimeIOAdapter, InProcessTransport]:
    transport = InProcessTransport()
    return RuntimeIOAdapter(origin=_ORIGIN, transport=transport), transport


async def test_discover_yields_a_workspace_with_the_user_channel_tool() -> None:
    adapter, _ = _adapter()
    (workspace,) = await adapter.discover()
    (tool,) = workspace.tools()
    assert tool.id.startswith(f"{RUNTIME_IO_ADDRESS}/")
    assert tool.manual.operation(SEND_MESSAGE_TO_USER) is not None
    assert tool.observe() == []  # no observable state on the reply channel


async def test_send_message_to_user_delegates_to_the_transport() -> None:
    adapter, transport = _adapter()
    (workspace,) = await adapter.discover()
    (tool,) = workspace.tools()
    ack = await tool.invoke(SEND_MESSAGE_TO_USER, text="Booked Monday 10:00.")
    assert ack.ok
    assert transport.sent == [("user", {"text": "Booked Monday 10:00."})]


async def test_unknown_operation_returns_a_failed_ack_not_a_raise() -> None:
    adapter, transport = _adapter()
    (workspace,) = await adapter.discover()
    (tool,) = workspace.tools()
    ack = await tool.invoke("send_message_to_agent", to="peer", text="hi")
    assert not ack.ok
    assert transport.sent == []  # nothing delivered on an unknown op


async def test_connect_rebuilds_the_same_channel_from_a_record() -> None:
    adapter, transport = _adapter()
    ws = await adapter.connect(
        WorkspaceRecord(id="runtime-io", origin=_ORIGIN, discovered_at=0.0, last_seen_at=0.0),
        [],
        {},
    )
    (tool,) = ws.tools()
    ack = await tool.invoke(SEND_MESSAGE_TO_USER, text="hello")
    assert ack.ok
    assert transport.sent == [("user", {"text": "hello"})]


def test_manual_bakes_the_recipient_into_the_operation_name() -> None:
    # The name asserts *who* it reaches, so a plan can't mistake it for messaging a third party.
    op = user_channel_manual().operation(SEND_MESSAGE_TO_USER)
    assert op is not None
    assert op.name.endswith("_to_user")
    assert "text" in op.parameters["properties"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
