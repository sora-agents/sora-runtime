"""Hardened MCP adapter contract — the generic ``McpWorkspaceAdapter`` base and its ARE subclass.

Runs in the default suite: every MCP round-trip goes through an injected **fake session**, so
there's no subprocess and no real ARE server. The fake stands in for ``mcp.ClientSession`` — canned
``list_tools``/``read_resource``/``call_tool`` plus a ``trigger_resource_updated`` driver that
replays a ``resource_updated`` notification synchronously, exercising the adapter's real URI-routing
and signal-delivery path. The live subscribe/notify round-trip against the real ARE server stays in
``test_are_walking_skeleton.py``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("mcp")

from sora.adapters.are_mcp import AreMcpWorkspaceAdapter  # noqa: E402
from sora.adapters.mcp import (  # noqa: E402
    McpSession,
    McpWorkspaceAdapter,
    ResourceUpdateCallback,
)
from sora.environment import (  # noqa: E402
    EnvironmentRegistry,
    WorkspaceAdapter,
    WorkspaceOrigin,
)
from sora.manual import (  # noqa: E402
    Manual,
    OperationSpecification,
    ToolRecord,
    WorkspaceRecord,
)
from sora.perception import NotificationQueueSink  # noqa: E402
from sora.types import Signal  # noqa: E402

# ------------------------------------------------------------------------------------------------
# Fake MCP session + adapter factory seam
# ------------------------------------------------------------------------------------------------
_UNSET = object()


class FakeMcpSession:
    """Structural stand-in for ``mcp.ClientSession`` (satisfies the ``McpSession`` Protocol). Canned
    tool list + resources; records ``call_tool`` calls and live subscriptions; replays a
    ``resource_updated`` notification via ``trigger_resource_updated`` straight into the adapter's
    routing callback."""

    def __init__(
        self,
        *,
        tools: list[tuple[str, str, dict[str, Any]]] | None = None,
        resources: dict[str, Any] | None = None,
    ) -> None:
        self._tools = tools or []
        self.resources = dict(resources or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.subscribed: set[str] = set()
        self.on_update: ResourceUpdateCallback | None = None

    async def initialize(self) -> Any:
        return None

    async def list_tools(self) -> Any:
        return SimpleNamespace(
            tools=[SimpleNamespace(name=n, description=d, inputSchema=s) for n, d, s in self._tools]
        )

    async def read_resource(self, uri: Any) -> Any:
        text = json.dumps(self.resources[str(uri)])
        return SimpleNamespace(contents=[SimpleNamespace(text=text)])

    async def subscribe_resource(self, uri: Any) -> Any:
        self.subscribed.add(str(uri))
        return None

    async def unsubscribe_resource(self, uri: Any) -> Any:
        self.subscribed.discard(str(uri))
        return None

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        self.calls.append((name, arguments or {}))
        return SimpleNamespace(isError=False, structuredContent={"called": name}, content=[])

    async def trigger_resource_updated(self, uri: str, new_value: Any = _UNSET) -> None:
        if new_value is not _UNSET:
            self.resources[str(uri)] = new_value
        assert self.on_update is not None, "no tool has focused a resource yet"
        await self.on_update(str(uri))


def _factory_for(session: FakeMcpSession) -> Any:
    """Build the ``session_factory`` seam the adapter calls with its routing callback."""

    @asynccontextmanager
    async def _cm(on_update: ResourceUpdateCallback) -> AsyncIterator[McpSession]:
        session.on_update = on_update
        yield session

    return _cm


def _mcp_tool(name: str, schema: dict[str, Any] | None = None) -> tuple[str, str, dict[str, Any]]:
    return (name, f"{name} description", schema or {})


def _origin(address: str = "stdio:are-email", adapter: str = "are-mcp") -> WorkspaceOrigin:
    return WorkspaceOrigin(adapter=adapter, address=address)


def _are_adapter(session: FakeMcpSession, origin: WorkspaceOrigin) -> AreMcpWorkspaceAdapter:
    return AreMcpWorkspaceAdapter(
        command="python",
        args=["-m", "server"],
        workspace_id="are",
        origin=origin,
        session_factory=_factory_for(session),
    )


def _vanilla_adapter(
    session: FakeMcpSession, origin: WorkspaceOrigin, workspace_id: str = "srv"
) -> McpWorkspaceAdapter:
    return McpWorkspaceAdapter(
        command="python",
        args=["-m", "server"],
        workspace_id=workspace_id,
        origin=origin,
        session_factory=_factory_for(session),
    )


# ------------------------------------------------------------------------------------------------
# 1–2. Vanilla grouping: one S-ORA tool per MCP tool, single op, no observables
# ------------------------------------------------------------------------------------------------
async def test_vanilla_one_tool_per_mcp_tool_no_observables() -> None:
    origin = _origin(address="mcp://localhost/weather", adapter="mcp")
    session = FakeMcpSession(tools=[_mcp_tool("get_forecast"), _mcp_tool("get_alerts")])
    workspace = (await _vanilla_adapter(session, origin).discover())[0]

    tools = {t.id: t for t in workspace.tools()}
    assert set(tools) == {
        "mcp://localhost/weather/get_forecast",
        "mcp://localhost/weather/get_alerts",
    }
    forecast = tools["mcp://localhost/weather/get_forecast"]
    # A single operation named after the MCP tool, and no curated observables for vanilla MCP.
    assert [op.name for op in forecast.manual.operations] == ["get_forecast"]
    assert forecast.manual.observable_properties == []
    assert forecast.manual.signals == []
    assert forecast.observe() == []


async def test_vanilla_invoke_uses_identity_name_assembly() -> None:
    origin = _origin(address="mcp://localhost/weather", adapter="mcp")
    session = FakeMcpSession(tools=[_mcp_tool("get_forecast")])
    workspace = (await _vanilla_adapter(session, origin).discover())[0]
    tool = workspace.tools()[0]

    ack = await tool.invoke("get_forecast", city="NYC")
    assert ack.ok is True
    assert session.calls == [("get_forecast", {"city": "NYC"})]


# ------------------------------------------------------------------------------------------------
# 3–4. ARE grouping: one tool per app, <App>__<op> name assembly, origin-qualified ids
# ------------------------------------------------------------------------------------------------
def _are_session() -> FakeMcpSession:
    return FakeMcpSession(
        tools=[
            _mcp_tool("EmailClientApp__list_emails"),
            _mcp_tool("EmailClientApp__send_email"),
            _mcp_tool("CalendarApp__list_events"),
        ],
        resources={
            "app://EmailClientApp/state": {"unread": 3},
            "app://CalendarApp/state": {"events": 0},
        },
    )


async def test_are_groups_flat_names_into_one_tool_per_app() -> None:
    origin = _origin()
    workspace = (await _are_adapter(_are_session(), origin).discover())[0]

    tools = {t.id: t for t in workspace.tools()}
    assert set(tools) == {"stdio:are-email/EmailClientApp", "stdio:are-email/CalendarApp"}
    email = tools["stdio:are-email/EmailClientApp"]
    assert {op.name for op in email.manual.operations} == {"list_emails", "send_email"}


async def test_are_invoke_assembles_app_op_name() -> None:
    origin = _origin()
    session = _are_session()
    workspace = (await _are_adapter(session, origin).discover())[0]
    email = next(t for t in workspace.tools() if t.id.endswith("EmailClientApp"))

    await email.invoke("list_emails")
    assert session.calls == [("EmailClientApp__list_emails", {})]


async def test_tool_id_is_origin_qualified_and_deterministic() -> None:
    # Same origin + same server -> same ids on a re-discover (so restore reproduces them);
    # a different origin address -> different ids (globally unique — ADR-0014).
    origin_a = _origin(address="stdio:are-email")
    ids_1 = {t.id for t in (await _are_adapter(_are_session(), origin_a).discover())[0].tools()}
    ids_2 = {t.id for t in (await _are_adapter(_are_session(), origin_a).discover())[0].tools()}
    assert ids_1 == ids_2

    origin_b = _origin(address="stdio:are-work")
    ids_b = {t.id for t in (await _are_adapter(_are_session(), origin_b).discover())[0].tools()}
    assert ids_1.isdisjoint(ids_b)


# ------------------------------------------------------------------------------------------------
# 5–8. Resources -> ObservableProperty / Signal (gap 2)
# ------------------------------------------------------------------------------------------------
async def test_are_manual_populates_observable_specs() -> None:
    origin = _origin()
    workspace = (await _are_adapter(_are_session(), origin).discover())[0]
    email = next(t for t in workspace.tools() if t.id.endswith("EmailClientApp"))

    assert [p.name for p in email.manual.observable_properties] == ["state"]
    assert [s.name for s in email.manual.signals] == ["state_changed"]


async def test_focus_reads_resource_into_observable_property() -> None:
    origin = _origin()
    session = _are_session()
    workspace = (await _are_adapter(session, origin).discover())[0]
    email = next(t for t in workspace.tools() if t.id.endswith("EmailClientApp"))

    assert email.observe() == []  # nothing until focused
    await email.focus(NotificationQueueSink[Signal]())
    assert "app://EmailClientApp/state" in session.subscribed
    props = {p.name: p.value for p in email.observe()}
    assert props == {"state": {"unread": 3}}


async def test_resource_updated_pushes_signal_and_refreshes_property() -> None:
    origin = _origin()
    session = _are_session()
    workspace = (await _are_adapter(session, origin).discover())[0]
    email = next(t for t in workspace.tools() if t.id.endswith("EmailClientApp"))

    sink: NotificationQueueSink[Signal] = NotificationQueueSink()
    await email.focus(sink)
    await session.trigger_resource_updated("app://EmailClientApp/state", {"unread": 4})

    drained = [item async for item in sink.drain()]
    assert len(drained) == 1
    source, signal = drained[0]
    assert source == email.id
    assert signal.name == "state_changed"
    assert signal.payload["value"] == {"unread": 4}
    # observe() reflects the refreshed snapshot.
    assert {p.name: p.value for p in email.observe()} == {"state": {"unread": 4}}


async def test_unfocus_unsubscribes_and_clears_properties() -> None:
    origin = _origin()
    session = _are_session()
    workspace = (await _are_adapter(session, origin).discover())[0]
    email = next(t for t in workspace.tools() if t.id.endswith("EmailClientApp"))

    await email.focus(NotificationQueueSink[Signal]())
    await email.unfocus()
    assert session.subscribed == set()
    assert email.observe() == []


# ------------------------------------------------------------------------------------------------
# manual_source pairing — merge a hand-authored manual with the adapter-synthesized one (ADR-0018)
# ------------------------------------------------------------------------------------------------
class _FakeManualSource:
    def __init__(self, manuals: dict[str, Manual]) -> None:
        self._manuals = manuals

    async def get(self, manual_id: str) -> Manual | None:
        return self._manuals.get(manual_id)


async def test_discover_merges_authored_manual_when_manual_source_resolves_one() -> None:
    origin = _origin(address="mcp://localhost/weather", adapter="mcp")
    session = FakeMcpSession(tools=[_mcp_tool("get_forecast")])
    authored = Manual(
        id="get_forecast",
        metadata={"category": "Weather"},
        description="Forecasts the weather.",
        observable_properties=[],
        signals=[],
        operations=[],
        raw_text="# Tool Metadata\nid: get_forecast\n\n# Usage Protocols & Safety\nBe kind.\n",
    )
    adapter = McpWorkspaceAdapter(
        command="python",
        args=["-m", "server"],
        workspace_id="srv",
        origin=origin,
        session_factory=_factory_for(session),
        manual_source=_FakeManualSource({"get_forecast": authored}),
    )
    workspace = (await adapter.discover())[0]
    tool = workspace.tools()[0]

    assert tool.manual.raw_text == authored.raw_text  # authored channel supplies raw_text
    assert tool.manual.description == "Forecasts the weather."
    assert [op.name for op in tool.manual.operations] == [
        "get_forecast"
    ]  # adapter still owns specs
    assert tool.manual.metadata == {"source": "mcp", "category": "Weather"}  # union, authored wins


async def test_discover_leaves_manual_adapter_only_when_no_authored_manual_for_id() -> None:
    origin = _origin(address="mcp://localhost/weather", adapter="mcp")
    session = FakeMcpSession(tools=[_mcp_tool("get_forecast")])
    adapter = McpWorkspaceAdapter(
        command="python",
        args=["-m", "server"],
        workspace_id="srv",
        origin=origin,
        session_factory=_factory_for(session),
        manual_source=_FakeManualSource({}),  # no manual for "get_forecast"
    )
    workspace = (await adapter.discover())[0]
    assert workspace.tools()[0].manual.raw_text is None  # unmerged, adapter-synthesized manual


async def test_discover_without_manual_source_is_unchanged() -> None:
    origin = _origin(address="mcp://localhost/weather", adapter="mcp")
    session = FakeMcpSession(tools=[_mcp_tool("get_forecast")])
    workspace = (await _vanilla_adapter(session, origin).discover())[0]  # no manual_source passed
    assert workspace.tools()[0].manual.raw_text is None


# ------------------------------------------------------------------------------------------------
# 9. connect() — restore from records + manuals (lazy rebuild)
# ------------------------------------------------------------------------------------------------
async def test_connect_rebuilds_tool_from_records_and_manual() -> None:
    origin = _origin()
    session = _are_session()
    adapter = _are_adapter(session, origin)

    manual = Manual(
        id="EmailClientApp",
        metadata={},
        description="",
        observable_properties=[],
        signals=[],
        operations=[OperationSpecification(name="list_emails", description="", parameters={})],
        raw_text=None,
    )
    ws_record = WorkspaceRecord(id="are", origin=origin, discovered_at=0.0, last_seen_at=0.0)
    tool_record = ToolRecord(
        id="stdio:are-email/EmailClientApp",
        manual_id="EmailClientApp",
        workspace_id="are",
        address=None,
        discovered_at=0.0,
        last_seen_at=0.0,
    )
    workspace = await adapter.connect(ws_record, [tool_record], {"EmailClientApp": manual})

    tool = workspace.tools()[0]
    assert tool.id == "stdio:are-email/EmailClientApp"
    await tool.invoke("list_emails")
    assert session.calls == [("EmailClientApp__list_emails", {})]


# ------------------------------------------------------------------------------------------------
# 10. Registry integration — join two workspaces, global id uniqueness holds
# ------------------------------------------------------------------------------------------------
async def test_two_joined_workspaces_have_globally_unique_ids() -> None:
    origin_a = _origin(address="stdio:are-email")
    origin_b = _origin(address="stdio:are-work")
    # Distinct workspace ids per adapter so the registry accepts both.
    adapters: dict[WorkspaceOrigin, WorkspaceAdapter] = {
        origin_a: _are_adapter(_are_session(), origin_a),
        origin_b: AreMcpWorkspaceAdapter(
            command="python",
            args=["-m", "server"],
            workspace_id="are-work",
            origin=origin_b,
            session_factory=_factory_for(_are_session()),
        ),
    }
    registry = EnvironmentRegistry(adapters=adapters)
    await registry.join(origin_a)
    await registry.join(origin_b)

    ids = [t.id for t in registry.all_tools()]
    assert len(ids) == len(set(ids)) == 4  # 2 apps x 2 workspaces, all distinct


# ------------------------------------------------------------------------------------------------
# Transport selection: stdio (spawns/owns a subprocess) vs a remote SSE / streamable-HTTP server
# ------------------------------------------------------------------------------------------------


def test_transport_is_stdio_when_command_given() -> None:
    adapter = McpWorkspaceAdapter(
        workspace_id="s", origin=_origin("stdio:x", "mcp"), command="python", args=["-m", "srv"]
    )
    assert adapter._transport == "stdio"


def test_transport_defaults_to_sse_when_url_given() -> None:
    adapter = McpWorkspaceAdapter(
        workspace_id="s", origin=_origin("http://h/sse", "mcp"), url="http://h/sse"
    )
    assert adapter._transport == "sse"


def test_transport_streamable_http_when_requested() -> None:
    adapter = McpWorkspaceAdapter(
        workspace_id="s",
        origin=_origin("http://h/mcp", "mcp"),
        url="http://h/mcp",
        transport="streamable-http",
    )
    assert adapter._transport == "streamable-http"


def test_transport_requires_a_source() -> None:
    with pytest.raises(ValueError, match="needs a transport"):
        McpWorkspaceAdapter(workspace_id="s", origin=_origin("x", "mcp"))


async def test_sse_transport_connects_to_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    @asynccontextmanager
    async def fake_sse(url: str) -> AsyncIterator[tuple[str, str]]:
        seen["url"] = url
        yield ("read", "write")

    monkeypatch.setattr("sora.adapters.mcp.sse_client", fake_sse)
    adapter = McpWorkspaceAdapter(
        workspace_id="s", origin=_origin("http://remote/sse", "mcp"), url="http://remote/sse"
    )
    async with adapter._open_transport() as streams:
        assert streams == ("read", "write")
    assert seen["url"] == "http://remote/sse"  # connected to the existing server, nothing spawned


async def test_streamable_http_transport_drops_the_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @asynccontextmanager
    async def fake_http(url: str) -> AsyncIterator[tuple[str, str, str]]:
        yield ("read", "write", "session-id-callback")  # streamable-http yields a 3rd element

    monkeypatch.setattr("sora.adapters.mcp.streamablehttp_client", fake_http)
    adapter = McpWorkspaceAdapter(
        workspace_id="s",
        origin=_origin("http://remote/mcp", "mcp"),
        url="http://remote/mcp",
        transport="streamable-http",
    )
    async with adapter._open_transport() as streams:
        assert streams == ("read", "write")  # the session-id element is dropped for ClientSession


# ------------------------------------------------------------------------------------------------
# adapter_for wiring: agent.yaml chooses stdio vs remote by what the workspace entry carries
# ------------------------------------------------------------------------------------------------


def test_adapter_for_remote_url_needs_no_command() -> None:
    from sora.bootstrap import adapter_for

    origin, adapter = adapter_for(
        {"origin": {"adapter": "mcp", "address": "http://host:8080/sse"}, "workspace_id": "remote"}
    )
    assert isinstance(adapter, McpWorkspaceAdapter)
    assert adapter._transport == "sse"  # remote, no subprocess deployed


def test_adapter_for_remote_transport_passthrough() -> None:
    from sora.bootstrap import adapter_for

    _origin_, adapter = adapter_for(
        {
            "origin": {"adapter": "are-mcp", "address": "http://host/mcp"},
            "workspace_id": "remote",
            "transport": "streamable-http",
        }
    )
    assert isinstance(adapter, AreMcpWorkspaceAdapter)
    assert adapter._transport == "streamable-http"


def test_adapter_for_stdio_still_spawns_a_command() -> None:
    from sora.bootstrap import adapter_for

    _origin_, adapter = adapter_for(
        {
            "origin": {"adapter": "mcp", "address": "stdio:local"},
            "workspace_id": "local",
            "command": "python",
            "args": ["-m", "srv"],
        }
    )
    assert isinstance(adapter, McpWorkspaceAdapter)
    assert adapter._transport == "stdio"
