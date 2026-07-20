"""Protocol-generic MCP ``WorkspaceAdapter`` — the shared base for every MCP-over-anything adapter.

This is a concrete, directly-usable adapter for a **vanilla** MCP server: an MCP server maps to one
workspace, each MCP tool maps to one S-ORA ``Tool`` with a single operation and no observable
properties or signals. This modeling choice is deliberate: while MCP resources could represent
observable state, they are application-controlled and carry no protocol-level guarantee of belonging
to any tool. A *curating* adapter (see ``are_mcp``) subclasses this and overrides the
grouping/name-assembly/observable hooks to lift a richer abstraction (an ARE "app" spanning several
MCP tools plus a state resource) on top — see ADR-0003 (adapters approximate the source faithfully,
no more) and ADR-0004.

The runtime-facing extension points stay Protocols (``WorkspaceAdapter``, ``Tool``, ``Workspace`` in
``sora.environment``); this base is implementation-sharing among concrete MCP adapters, not a
runtime seam, so a template-method base with overridable hooks is the right tool here (cf.
ADR-0008).
"""

from __future__ import annotations

import functools
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import ResourceUpdatedNotification, ServerNotification, TextContent

from sora.manual import (
    Manual,
    ObservablePropertySpecification,
    OperationSpecification,
    SignalSpecification,
)
from sora.types import ObservableProperty, OperationAck, Signal

if TYPE_CHECKING:
    from sora.environment import Tool, Workspace, WorkspaceOrigin
    from sora.manual import ToolRecord, WorkspaceRecord
    from sora.perception import SignalSink

# A resource_updated notification, reduced to just the URI that changed. The base's session
# message-handler unwraps MCP's notification envelope down to this; the routing that follows
# (URI -> the tool that focused it) is protocol-agnostic and unit-testable without MCP objects.
ResourceUpdateCallback = Callable[[str], Awaitable[None]]


class McpSession(Protocol):
    """The subset of ``mcp.ClientSession`` this adapter uses. A Protocol so tests can inject a
    subprocess-free fake; the real ``ClientSession`` satisfies it structurally."""

    async def initialize(self) -> Any: ...
    async def list_tools(self) -> Any: ...
    async def read_resource(self, uri: Any) -> Any: ...
    async def subscribe_resource(self, uri: Any) -> Any: ...
    async def unsubscribe_resource(self, uri: Any) -> Any: ...
    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any: ...


# The session factory is the test seam: given the adapter's resource-update routing callback, it
# yields a live (or fake) session for the duration of a workspace's life.
McpSessionFactory = Callable[[ResourceUpdateCallback], AbstractAsyncContextManager[McpSession]]


@dataclass(frozen=True)
class _ResourceBinding:
    """Curation, not a raw resource pass-through: names the specific ObservableProperty a resource
    surfaces as and the specific Signal its ``resource_updated`` fires (see ADR-0004). Vanilla MCP
    declares none; a curating adapter (ARE) declares one per app state resource."""

    uri: str
    property_name: str
    signal_name: str


@dataclass(frozen=True)
class _ToolBlueprint:
    """One S-ORA tool the grouping hook decided to build, before it becomes a live ``_McpTool``.
    ``seed`` drives both name assembly (``_mcp_name``) and id derivation; ``manual_id`` is the tool
    *type* identity (shared across instances/agents), distinct from the origin-qualified instance
    id."""

    seed: str
    manual_id: str
    description: str
    operations: list[OperationSpecification]
    metadata: dict[str, Any] = field(default_factory=dict)


class McpWorkspaceAdapter:
    """Directly usable for vanilla MCP; a base for curating adapters. One instance is config-scoped
    to exactly one workspace (see ``WorkspaceAdapter.discover``).

    Connects over any MCP transport: ``command``/``args`` spawns and owns a local **stdio**
    subprocess, while ``url`` connects to an **already-running remote** server (**SSE** by default,
    or **streamable-HTTP** via ``transport="streamable-http"``) — nothing is deployed in that case.
    ``discover``/``connect`` are transport-agnostic; only ``_open_transport`` differs."""

    name = "mcp"  # matches WorkspaceOrigin.adapter

    def __init__(
        self,
        *,
        workspace_id: str,
        origin: WorkspaceOrigin,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        transport: str | None = None,
        session_factory: McpSessionFactory | None = None,
    ) -> None:
        self._workspace_id = workspace_id
        self._origin = origin
        self._session_factory = session_factory
        # Transport selection — exactly one source, resolved once:
        #   * an injected session_factory wins (tests / a custom transport);
        #   * else `command` spawns and owns a local stdio subprocess;
        #   * else `url` connects to an already-running remote server — SSE by default, or
        #     streamable-HTTP (`transport="streamable-http"`).
        self._url = url
        self._params: StdioServerParameters | None = None
        if session_factory is not None:
            self._transport = "custom"
        elif command is not None:
            self._transport = "stdio"
            self._params = StdioServerParameters(command=command, args=args or [], env=env)
        elif url is not None:
            self._transport = transport or "sse"
        else:
            raise ValueError(
                "McpWorkspaceAdapter needs a transport: `command` (stdio), `url` (sse / "
                "streamable-http), or an injected `session_factory`"
            )

    # -- WorkspaceAdapter Protocol -------------------------------------------------------------

    async def discover(self) -> list[Workspace]:
        router = _SignalRouter()
        stack = AsyncExitStack()
        session = await stack.enter_async_context(self._open(router))
        listed = await session.list_tools()
        tools = [self._build_tool(bp, session, router) for bp in self._group(list(listed.tools))]
        return [_McpWorkspace(self._workspace_id, self._origin, tools, stack)]

    async def connect(
        self,
        workspace_record: WorkspaceRecord,
        tool_records: list[ToolRecord],
        manuals: dict[str, Manual],
    ) -> Workspace:
        # Lazy rebuild from records — no re-list_tools(): the manual is a complete blueprint
        # (operations, observable bindings, and the name-assembly seed all reconstruct from it).
        # A tool the live server no longer has still rebuilds; its failure is deferred to invoke
        # time (dynamic workspaces are deferred as out of scope here).
        router = _SignalRouter()
        stack = AsyncExitStack()
        session = await stack.enter_async_context(self._open(router))
        tools: list[Tool] = []
        for record in tool_records:
            manual = manuals[record.manual_id]
            tools.append(
                self._make_tool(
                    tool_id=record.id,
                    seed=manual.id,
                    manual=manual,
                    session=session,
                    router=router,
                )
            )
        return _McpWorkspace(workspace_record.id, workspace_record.origin, tools, stack)

    # -- overridable hooks (defaults = vanilla MCP) --------------------------------------------

    def _group(self, mcp_tools: list[Any]) -> list[_ToolBlueprint]:
        """Vanilla policy: one S-ORA tool per MCP tool, a single operation named after it. A
        curating adapter overrides this to group several MCP tools under one abstraction."""
        return [
            _ToolBlueprint(
                seed=t.name,
                manual_id=t.name,
                description=t.description or "",
                operations=[
                    OperationSpecification(
                        name=t.name,
                        description=t.description or "",
                        parameters=dict(t.inputSchema or {}),
                    )
                ],
            )
            for t in mcp_tools
        ]

    def _mcp_name(self, seed: str, operation_name: str) -> str:
        """Assemble the wire-level MCP tool name from (seed, operation). Vanilla is identity — the
        operation *is* the MCP tool. Kept as its own hook (not folded into a discover-time map)
        because ``connect`` has no live tool list to derive a map from and must reassemble names
        from the seed alone."""
        return operation_name

    def _observable_bindings(self, seed: str) -> list[_ResourceBinding]:
        """Vanilla MCP surfaces no observables (see module docstring). A curating adapter declares
        the resource(s) it curates for this tool here."""
        return []

    def _synth_manual(self, blueprint: _ToolBlueprint, bindings: list[_ResourceBinding]) -> Manual:
        """Adapter-provenance manual: names + JSON-schema data shapes, no ``raw_text`` (that's the
        hand-authored Markdown channel's job — see ADR-0015). Observable property/signal specs
        mirror the curated bindings, so a plain MCP tool's manual carries operations only."""
        return Manual(
            id=blueprint.manual_id,
            metadata={"source": self.name, **blueprint.metadata},
            description=blueprint.description,
            observable_properties=[
                ObservablePropertySpecification(name=b.property_name, description="", schema={})
                for b in bindings
            ],
            signals=[
                SignalSpecification(name=b.signal_name, description="", schema={}) for b in bindings
            ],
            operations=blueprint.operations,
            raw_text=None,
        )

    # -- internals -----------------------------------------------------------------------------

    def _open(self, router: _SignalRouter) -> AbstractAsyncContextManager[McpSession]:
        factory = self._session_factory or self._default_session
        return factory(router.dispatch)

    @asynccontextmanager
    async def _default_session(
        self, on_update: ResourceUpdateCallback
    ) -> AsyncIterator[McpSession]:
        async def message_handler(message: Any) -> None:
            if isinstance(message, ServerNotification) and isinstance(
                message.root, ResourceUpdatedNotification
            ):
                await on_update(str(message.root.params.uri))

        async with self._open_transport() as (read, write):
            async with ClientSession(read, write, message_handler=message_handler) as session:
                await session.initialize()
                yield session

    def _open_transport(self) -> AbstractAsyncContextManager[Any]:
        """The chosen transport's read/write stream pair. stdio spawns/owns a subprocess; sse and
        streamable-http connect to an already-running server at ``self._url`` (streamable-http
        yields a third session-id element the client protocol doesn't need, so it's dropped)."""
        if self._transport == "stdio":
            assert self._params is not None
            return stdio_client(self._params)
        if self._transport == "sse":
            assert self._url is not None
            return sse_client(self._url)
        if self._transport == "streamable-http":
            assert self._url is not None
            return self._http_streams(self._url)
        raise ValueError(f"unknown MCP transport {self._transport!r}")

    @asynccontextmanager
    async def _http_streams(self, url: str) -> AsyncIterator[tuple[Any, Any]]:
        async with streamablehttp_client(url) as (read, write, _get_session_id):
            yield read, write

    def _derive_tool_id(self, seed: str) -> str:
        # ADR-0014: globally unique, adapter-derived, deterministic. MCP gives no per-tool global
        # URI, so synthesize one from the workspace origin's address + the local seed; the same
        # records rebuild the same id, keeping restore() valid.
        return f"{self._origin.address}/{seed}"

    def _build_tool(
        self, blueprint: _ToolBlueprint, session: McpSession, router: _SignalRouter
    ) -> Tool:
        return self._make_tool(
            tool_id=self._derive_tool_id(blueprint.seed),
            seed=blueprint.seed,
            manual=self._synth_manual(blueprint, self._observable_bindings(blueprint.seed)),
            session=session,
            router=router,
        )

    def _make_tool(
        self,
        *,
        tool_id: str,
        seed: str,
        manual: Manual,
        session: McpSession,
        router: _SignalRouter,
    ) -> Tool:
        return _McpTool(
            tool_id=tool_id,
            manual=manual,
            session=session,
            router=router,
            name_of=functools.partial(self._mcp_name, seed),
            bindings=self._observable_bindings(seed),
        )


class _SignalRouter:
    """Per-connection fan-out from a ``resource_updated`` URI to the tool that focused it. One
    router per workspace (the connection is shared); the session's message-handler calls
    ``dispatch``, tools register/unregister themselves as they focus/unfocus."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[], Awaitable[None]]] = {}

    def register(self, uri: str, handler: Callable[[], Awaitable[None]]) -> None:
        self._handlers[uri] = handler

    def unregister(self, uri: str) -> None:
        self._handlers.pop(uri, None)

    async def dispatch(self, uri: str) -> None:
        handler = self._handlers.get(uri)
        if handler is not None:
            await handler()


class _McpTool:
    """One live tool over a shared MCP session. ``invoke`` assembles the wire name via ``name_of``;
    ``focus``/``observe`` are driven entirely by ``bindings`` — empty for vanilla MCP, so both are
    no-ops there."""

    def __init__(
        self,
        *,
        tool_id: str,
        manual: Manual,
        session: McpSession,
        router: _SignalRouter,
        name_of: Callable[[str], str],
        bindings: list[_ResourceBinding],
    ) -> None:
        self.id = tool_id
        self.manual = manual
        self.address: str | None = None  # rides the workspace's single connection
        self._session = session
        self._router = router
        self._name_of = name_of
        self._bindings = bindings
        self._sink: SignalSink | None = None
        self._cache: dict[str, Any] = {}  # property_name -> last-known value

    async def invoke(self, operation_name: str, **params: Any) -> OperationAck:
        result = await self._session.call_tool(
            self._name_of(operation_name), arguments=params or None
        )
        return OperationAck(ok=not result.isError, result=_parse_result(result))

    async def focus(self, sink: SignalSink) -> None:
        self._sink = sink
        for binding in self._bindings:
            self._cache[binding.property_name] = await self._read(binding.uri)
            await self._session.subscribe_resource(binding.uri)
            self._router.register(binding.uri, self._make_handler(binding))

    async def unfocus(self) -> None:
        for binding in self._bindings:
            self._router.unregister(binding.uri)
            await self._session.unsubscribe_resource(binding.uri)
        self._cache.clear()
        self._sink = None

    def observe(self) -> list[ObservableProperty]:
        return [
            ObservableProperty(name=b.property_name, value=self._cache[b.property_name])
            for b in self._bindings
            if b.property_name in self._cache
        ]

    def _make_handler(self, binding: _ResourceBinding) -> Callable[[], Awaitable[None]]:
        async def handler() -> None:
            value = await self._read(binding.uri)
            self._cache[binding.property_name] = value
            if self._sink is not None:
                self._sink.push(
                    self.id, Signal(binding.signal_name, {"uri": binding.uri, "value": value})
                )

        return handler

    async def _read(self, uri: str) -> Any:
        return _parse_resource(await self._session.read_resource(uri))


class _McpWorkspace:
    def __init__(
        self, ws_id: str, origin: WorkspaceOrigin, tools: list[Tool], stack: AsyncExitStack
    ) -> None:
        self.id = ws_id
        self.origin = origin
        self._tools = tools
        self._stack = stack

    def tools(self) -> list[Tool]:
        return self._tools

    async def close(self) -> None:
        await self._stack.aclose()  # tears down the session + the stdio subprocess


def _parse_result(result: Any) -> Any:
    """Prefer structured output; otherwise decode text content (JSON if it parses)."""
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    texts = [c.text for c in result.content if isinstance(c, TextContent)]
    if not texts:
        return None
    blob = texts[0] if len(texts) == 1 else "\n".join(texts)
    return _loads_or_raw(blob)


def _parse_resource(result: Any) -> Any:
    """Decode a ``read_resource`` result's first text content (JSON if it parses)."""
    for content in getattr(result, "contents", []):
        text = getattr(content, "text", None)
        if text is not None:
            return _loads_or_raw(text)
    return None


def _loads_or_raw(blob: str) -> Any:
    try:
        return json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return blob
