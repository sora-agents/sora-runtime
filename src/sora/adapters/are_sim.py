"""In-process ARE ``WorkspaceAdapter`` + ``MessageTransport`` — talk to live ARE apps directly.

Unlike ``are_mcp`` (which reaches ARE over an MCP subprocess serving a *static* app snapshot), this
runs the ARE ``Environment`` **event loop** in the same process (a bg thread) so a scenario's
timeline actually fires — mid-run email injections, follow-up user messages, task delivery — and
bridges the two off-cycle event channels into S-ORA directly, as method calls on shared objects:

  * app state changes  -> ``state_changed`` Signal into the focused tool's ``signal_sink``
    (poll-on-observe: the tool re-reads ``app.get_state()`` each Observe and diffs — see
    ``_AreTool.observe``).  This is what MCP could not push off-request (ARE's MCP server only emits
    ``resource_updated`` from inside a write-tool request), so we go in-process instead.
  * ``AgentUserInterface`` USER messages  -> ``MessageTransport`` (``AreTransport`` over the AUI).

``AreSimulation`` owns the ``Environment``/scenario lifecycle and is the single object both seams
share (see the new ADR). The adapter's *workspace* owns start/stop (start on ``discover``, stop on
``close``), exactly as ``_McpWorkspace`` owns its subprocess. ``ARE`` (``are.simulation.*``) is an
optional dependency-group, so every import of it is lazy; the adapter/transport depend only
on a small duck-typed app/AUI interface (``app_name``/``get_tools``/``get_state`` and AUI
``get_last_unread_messages``/``send_message_to_user``), which fakes satisfy, so S-ORA-side logic
stays testable without ARE (see ADR-0003).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from sora.manual import (
    Manual,
    ObservablePropertySpecification,
    OperationSpecification,
    SignalSpecification,
    merge_manuals,
)
from sora.perception import Message
from sora.types import ObservableProperty, OperationAck, Signal

if TYPE_CHECKING:
    from sora.environment import Tool, Workspace, WorkspaceOrigin
    from sora.manual import ManualSource, ToolRecord, WorkspaceRecord

_T = TypeVar("_T")

_AUI_APP = "AgentUserInterface"  # ARE's user-message app; routed via the transport, not as a tool

# ARE mutates app state on its own event-loop thread with no lock we can share (see AreSimulation),
# so a ``get_state()`` that iterates a dict the event loop is concurrently growing can raise
# "changed size during iteration". Mutation happens in sub-second bursts, so an immediate re-read
# sees a settled snapshot — retry a few times before giving up.
_STATE_READ_ATTEMPTS = 3


class Simulation(Protocol):
    """The runtime surface the adapter/transport use, decoupled from ARE so fakes satisfy it. The
    concrete ``AreSimulation`` implements it over a live ARE ``Environment``; a test fake implements
    it over plain app/AUI objects."""

    @property
    def aui(self) -> Any:  # the live AgentUserInterface app, or None
        ...

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def apps(self) -> list[Any]: ...
    def run(self, fn: Callable[[], _T]) -> _T: ...  # serialize S-ORA's own concurrent app calls


class AreSimulation:
    """Owns the ARE ``Environment`` + scenario lifecycle — the shared object the in-process adapter
    and transport both reference. ``start`` runs the scenario's event loop on a background thread
    (``env.run(..., wait_for_end=False)``). The ``Lock`` serializes S-ORA's *own* concurrent app
    calls (e.g. an ``invoke`` on a worker thread vs an ``observe`` on the cycle thread); it does
    **not** — and cannot — serialize against ARE's event-loop thread, which mutates app state with
    no lock we can share, so reads tolerate a transient concurrent-modification error by retry (see
    ``_AreTool._read_state`` / ``_STATE_READ_ATTEMPTS``). The agent replies without blocking
    (``aui.wait_for_user_response = False``) — a follow-up user message arrives via the timeline and
    is picked up by ``AreTransport.receive``."""

    def __init__(self, scenario: Any, *, config: Any | None = None) -> None:
        self._scenario = scenario
        self._config = config
        self._env: Any | None = None
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        from are.simulation.environment import Environment, EnvironmentConfig

        if not getattr(self._scenario, "_initialized", False):
            self._scenario.initialize()
        self._env = Environment(config=self._config or EnvironmentConfig())
        # wait_for_end=False: registers apps, schedules the timeline, starts the event-loop thread,
        # and returns — the agent then drives its cycle against the live, ticking world.
        self._env.run(self._scenario, wait_for_end=False)
        aui = self.aui
        if aui is not None:
            aui.wait_for_user_response = False
        self._started = True

    def stop(self) -> None:
        if self._env is not None and self._started:
            self._env.stop()
        self._started = False

    def apps(self) -> list[Any]:
        return list(getattr(self._scenario, "apps", None) or [])

    @property
    def aui(self) -> Any:
        return next((a for a in self.apps() if a.app_name() == _AUI_APP), None)

    def run(self, fn: Callable[[], _T]) -> _T:
        # Serializes S-ORA's own concurrent calls only; ARE's event-loop thread does not take this
        # lock (it's ARE-internal), so it does not guard app reads against that thread — see the
        # class docstring and _AreTool._read_state's retry.
        with self._lock:
            return fn()

    def validate(self) -> bool:
        """Oracle scoring: run the scenario's validators against the final environment state."""
        assert self._env is not None, "start() the simulation before validating"
        return bool(self._scenario.validate(self._env).success)


def load_scenario(ref: str) -> Any:
    """Resolve a scenario reference to an ARE ``Scenario`` — the "any ARE scenario" seam. ``ref`` is
    either a ``.json`` benchmark scenario path or a dotted path to a ``Scenario`` subclass (or a
    ready instance). Instances are returned as-is (``AreSimulation.start`` initializes them)."""
    if ref.endswith(".json"):
        from are.simulation.benchmark.scenario_loader import load_scenario as _are_load

        scenario, _ = _are_load(
            Path(ref).read_text(encoding="utf-8"), ref, load_completed_events=False
        )
        if scenario is None:
            raise ValueError(f"failed to load ARE scenario from {ref!r}")
        return scenario
    from sora.bootstrap import import_object  # lazy: avoid a bootstrap<->are_sim import cycle

    obj = import_object(ref)
    return obj() if isinstance(obj, type) else obj


# -- app -> S-ORA usage-interface extraction ------------------------------------------------------

_JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}

_ARE_SEP = "__"  # ARE namespaces an app's tools as <App>__<operation> (as the flat MCP names do)


def _op_name(app: Any, app_tool: Any) -> str:
    """The bare operation name, stripping ARE's ``<App>__`` prefix so ops read as ``list_emails``
    (matching the ``are_mcp`` adapter and the hand-authored manuals), not ``EmailClientApp__…``."""
    prefix = f"{app.app_name()}{_ARE_SEP}"
    name: str = app_tool.name
    return name.removeprefix(prefix)


def _json_atom(t: str) -> dict[str, Any]:
    """One non-union ARE type string -> a JSON-Schema fragment. Recursive on ``list[...]`` so the
    item type is faithful too (``list[str]`` -> array of string, not array of anything)."""
    if t.startswith("list[") and t.endswith("]"):
        return {"type": "array", "items": _json_atom(t[len("list[") : -1].strip())}
    if t == "list":
        return {"type": "array"}
    if t == "dict" or t.startswith("dict["):
        return {"type": "object"}
    return {"type": _JSON_TYPES.get(t, "string")}


def _json_type(arg_type: Any) -> dict[str, Any]:
    """Map an ARE ``AppTool`` arg-type *string* (``str``, ``int``, ``list[str]``,
    ``list[str] | None``, ``int | float | None``, ``dict[str, Any]``, ...) to a JSON-Schema type
    fragment. Every arg the grounding model fills has to be represented faithfully: collapsing
    ``list[str]`` to ``string`` is what led the model to fill ``attendees`` with ``"Alice, Bob"`` —
    which ARE's own runtime type-check then rejects (``must be of type list[str] | None, got str``).
    Unions are split on ``|`` (``None`` dropped): a lone member maps directly; an all-numeric union
    (``int | float``) becomes JSON ``number`` (which admits both); any other heterogeneous union
    has no single faithful JSON type, so it falls back to ``string``. (Assumes ARE's flat vocabulary
    — no ``|`` nested inside brackets, which its apps never emit.)"""
    if not isinstance(arg_type, str):
        return {"type": "string"}
    members = [m.strip() for m in arg_type.split("|")]
    members = [m for m in members if m and m != "None"]
    if len(members) == 1:
        return _json_atom(members[0])
    if {_json_atom(m).get("type") for m in members} <= {"integer", "number"}:
        return {"type": "number"}
    return {"type": "string"}


def _params_schema(app_tool: Any) -> dict[str, Any]:
    """A JSON-Schema object for an ARE ``AppTool``'s args (same shape the ARE MCP server uses)."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for arg in app_tool.args:
        properties[arg.name] = {**_json_type(arg.arg_type), "description": arg.description or ""}
        if not getattr(arg, "has_default", False):
            required.append(arg.name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _operation_specs(app: Any) -> list[OperationSpecification]:
    return [
        OperationSpecification(
            name=_op_name(app, at),
            description=getattr(at, "function_description", None) or "",
            parameters=_params_schema(at),
        )
        for at in app.get_tools()
    ]


def _to_serializable(value: Any) -> Any:
    """Make an app op's result JSON-friendly so a later step can ground on its fields (e.g. an
    ``email_id`` from ``list_emails``). Falls back to the raw value when ARE isn't importable (fakes
    already return plain data)."""
    try:
        from are.simulation.utils import make_serializable

        return make_serializable(value)
    except Exception:
        return value


# -- Tool / Workspace / Adapter -------------------------------------------------------------------


class _AreTool:
    """One live tool over one ARE app. ``invoke`` calls the app op (off-thread, lock-guarded
    against the Environment thread); ``observe`` polls ``get_state`` and emits ``state_changed`` on
    diff into the sink handed at ``focus`` — the in-process analogue of MCP's resource-update push,
    tied to the cycle's own Observe cadence so it's deterministic."""

    def __init__(
        self, *, tool_id: str, manual: Manual, app: Any, ops: dict[str, Any], simulation: Simulation
    ) -> None:
        self.id = tool_id
        self.manual = manual
        self.address: str | None = None
        self._app = app
        self._ops = ops
        self._sim = simulation
        self._sink: Any | None = None
        self._state: Any = None  # last observed state, for the diff

    async def invoke(self, operation_name: str, **params: Any) -> OperationAck:
        app_tool = self._ops.get(operation_name)
        if app_tool is None:
            return OperationAck(ok=False, result=f"unknown operation {operation_name!r}")
        try:
            result = await asyncio.to_thread(self._sim.run, lambda: app_tool(**params))
            return OperationAck(ok=True, result=_to_serializable(result))
        except Exception as exc:  # an app op raising is a failed ack, not a runtime crash
            return OperationAck(ok=False, result=str(exc))

    async def focus(self, sink: Any) -> None:
        self._sink = sink
        self._state = self._read_state()

    async def unfocus(self) -> None:
        self._sink = None
        self._state = None

    def observe(self) -> list[ObservableProperty]:
        state = self._read_state()
        if self._sink is not None and state != self._state:
            self._sink.push(
                self.id, Signal("state_changed", {"app": self._app.app_name(), "value": state})
            )
        self._state = state
        return [ObservableProperty(name="state", value=state)]

    def _read_state(self) -> Any:
        # ARE's event-loop thread can mutate app state mid-read (no shared lock), so a get_state()
        # iterating a dict it's concurrently growing may raise RuntimeError. Retry the snapshot —
        # mutation is bursty, so an immediate re-read almost always settles (_STATE_READ_ATTEMPTS).
        last: RuntimeError | None = None
        for _ in range(_STATE_READ_ATTEMPTS):
            try:
                return self._sim.run(self._app.get_state)
            except RuntimeError as exc:  # concurrent modification by the ARE event-loop thread
                last = exc
        assert last is not None
        raise last


class _AreWorkspace:
    def __init__(
        self, ws_id: str, origin: WorkspaceOrigin, tools: list[Tool], simulation: Simulation
    ) -> None:
        self.id = ws_id
        self.origin = origin
        self._tools = tools
        self._sim = simulation

    def tools(self) -> list[Tool]:
        return self._tools

    async def close(self) -> None:
        await asyncio.to_thread(self._sim.stop)  # stops the Environment event-loop thread


class AreInProcessWorkspaceAdapter:
    """Imports the live ARE apps of a running ``AreSimulation`` as S-ORA tools (one per app, its
    ops from ``app.get_tools()``, plus a ``state`` observable + ``state_changed`` signal). The
    ``AgentUserInterface`` app is deliberately excluded — user messages are a transport concern
    (``AreTransport``), not a tool. The workspace owns the Environment lifecycle: ``discover``
    it, ``close`` stops it."""

    name = "are-sim"  # matches WorkspaceOrigin.adapter

    def __init__(
        self,
        *,
        workspace_id: str,
        origin: WorkspaceOrigin,
        simulation: Simulation,
        manual_source: ManualSource | None = None,
    ) -> None:
        self._workspace_id = workspace_id
        self._origin = origin
        self._sim = simulation
        self._manual_source = manual_source

    async def discover(self) -> list[Workspace]:
        await asyncio.to_thread(self._sim.start)
        tools = [await self._build_tool(app) for app in self._tool_apps()]
        return [_AreWorkspace(self._workspace_id, self._origin, tools, self._sim)]

    async def connect(
        self,
        workspace_record: WorkspaceRecord,
        tool_records: list[ToolRecord],
        manuals: dict[str, Manual],
    ) -> Workspace:
        # In-process apps live in the current simulation, so rebuild directly from them (no snapshot
        # reconstruction needed — the process holds the live objects).
        await asyncio.to_thread(self._sim.start)
        by_name = {app.app_name(): app for app in self._tool_apps()}
        tools: list[Tool] = []
        for record in tool_records:
            app = by_name.get(record.manual_id)
            if app is not None:
                tools.append(self._make_tool(record.id, app, manuals[record.manual_id]))
        return _AreWorkspace(workspace_record.id, workspace_record.origin, tools, self._sim)

    def _tool_apps(self) -> list[Any]:
        return [a for a in self._sim.apps() if a.app_name() != _AUI_APP]

    async def _build_tool(self, app: Any) -> Tool:
        manual = await self._paired_manual(app.app_name(), self._synth_manual(app))
        return self._make_tool(self._derive_tool_id(app.app_name()), app, manual)

    def _make_tool(self, tool_id: str, app: Any, manual: Manual) -> Tool:
        ops = {_op_name(app, at): at for at in app.get_tools()}
        return _AreTool(tool_id=tool_id, manual=manual, app=app, ops=ops, simulation=self._sim)

    def _synth_manual(self, app: Any) -> Manual:
        name = app.app_name()
        return Manual(
            id=name,
            metadata={"source": self.name, "app": name},
            description=f"ARE app {name}, in-process",
            observable_properties=[
                ObservablePropertySpecification(name="state", description="", schema={})
            ],
            signals=[SignalSpecification(name="state_changed", description="", schema={})],
            operations=_operation_specs(app),
            raw_text=None,
        )

    async def _paired_manual(self, manual_id: str, adapter_manual: Manual) -> Manual:
        if self._manual_source is None:
            return adapter_manual
        authored = await self._manual_source.get(manual_id)
        return adapter_manual if authored is None else merge_manuals(adapter_manual, authored)

    def _derive_tool_id(self, seed: str) -> str:
        # ADR-0014: globally unique, adapter-derived, deterministic (origin address + app name).
        return f"{self._origin.address}/{seed}"


class AreTransport:
    """``MessageTransport`` over the scenario's ``AgentUserInterface``. ``receive`` drains unread
    USER messages (the task + timeline follow-ups) as ``Message``s; ``send`` posts the agent's reply
    via ``send_message_to_user``. Shares the running ``AreSimulation`` with the adapter."""

    def __init__(self, simulation: Simulation) -> None:
        self._sim = simulation

    async def send(self, to: str, content: dict[str, Any]) -> None:
        aui = self._sim.aui
        if aui is None:
            return
        text = content.get("text", "") if isinstance(content, dict) else str(content)
        await asyncio.to_thread(self._sim.run, lambda: aui.send_message_to_user(text))

    def receive(self) -> AsyncIterator[Message]:
        async def _drain() -> AsyncIterator[Message]:
            aui = self._sim.aui
            if aui is None:
                return
            for m in self._sim.run(aui.get_last_unread_messages):
                # ARE message timestamps are sim-relative time; the t0 task message legitimately
                # has timestamp 0.0, so distinguish an absent timestamp (None) from a falsy 0.0
                # rather than `... or time.time()`, which would stamp wall-clock over a real 0.0.
                ts = getattr(m, "timestamp", None)
                yield Message(
                    sender="user",
                    content={"text": m.content},
                    received_at=time.time() if ts is None else ts,
                )

        return _drain()
