"""Permanent TDD tests for the predefined action space and the ``ActionRegistry``.

Covers all six predefined external actions plus the registry that dispatches them:

* **Invoke** — returns an immediate ``ActionAck`` (dispatch, not outcome), moves the activity to
  ``RUNNING`` with a bound ``pending_operation`` (the implicit, unconditional wait), runs the tool
  round-trip *off-cycle*, and lands the ``OperationAck`` on ``result_sink`` keyed by the per-invoke
  ``invocation_id`` (never a ``Percept``), so Observe can resolve it 1:1.
* **Focus/Unfocus** — ``working.focused_tools`` + ``signal_sink`` wiring.
* **Join/Leave** — registry mutation + record persistence via ``SemanticMemory``.
* **Send** — ``MessageTransport`` delegation.

Each action's ``execute(...)`` is driven directly against the in-process fakes (``tests/fakes.py``)
and a real ``FileMemoryBackend``, so the Join persistence path exercises the actual store/retrieve
serialization round-trip rather than a mock. The ``DecisionCycle`` here is real (its constructor is
the typed seam ``execute`` reaches through), but its strategies/transport are inert plumbing — these
tests call ``execute`` directly, never ``tick()``. Promotes the spike's
``invoke_action_sets_running_then_pushes_result`` and ``action_registry_lookup`` from
``tests/test_cycle_wiring.py`` (triage rows in ``docs/phase-3-test-triage.md``).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeAdapter, FakeTool, FakeWorkspace, fake_manual
from sora.action import (
    ActionRegistry,
    CreateActivityAction,
    ExternalAction,
    FilterPerceptionsAction,
    FocusAction,
    InvokeAction,
    JoinAction,
    LeaveAction,
    LoadManualAction,
    SendAction,
    UnfocusAction,
    UnloadManualAction,
    default_action_registry,
    invoke_step,
)
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, Tool, WorkspaceOrigin
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from sora.perception import Message, Percept, PerceptKind
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    Strategies,
    TickResult,
)
from sora.types import OPERATION_NAME, TOOL_ID, ObservableProperty, Signal

# --------------------------------------------------------------------------------------------------
# Harness — the fakes plus a recording transport and a real FileMemoryBackend-backed DecisionCycle.
# --------------------------------------------------------------------------------------------------

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")


class RecordingTransport:
    """Satisfies MessageTransport: send() logs its args; receive() yields nothing."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def send(self, to: str, content: dict[str, Any]) -> None:
        self.sent.append((to, content))

    def receive(self) -> AsyncIterator[Message]:
        async def _empty() -> AsyncIterator[Message]:
            return
            yield  # pragma: no cover — makes this a (never-yielding) async generator

        return _empty()


class _UnusedReason:
    """A ReasonStrategy stand-in; never invoked (tests call execute() directly, not tick())."""

    async def reason(
        self, activity: Any, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        return result


def _registry_with(*tools: Tool) -> tuple[EnvironmentRegistry, FakeWorkspace]:
    workspace = FakeWorkspace("ws", _ORIGIN, list(tools))
    adapter = FakeAdapter("fake", workspace)
    registry = EnvironmentRegistry(adapters={_ORIGIN: adapter})
    return registry, workspace


def _cycle(
    registry: EnvironmentRegistry,
    tmp_path: Path,
    transport: RecordingTransport | None = None,
) -> tuple[DecisionCycle, WorkingMemory, SemanticMemory]:
    backend = FileMemoryBackend(tmp_path)
    semantic = SemanticMemory(backend)
    working = WorkingMemory(registry=registry)
    strategies = Strategies(
        observe=DefaultObserveStrategy(),
        reflect=DefaultReflectStrategy(),
        situate=DefaultSituateStrategy(),
        reason=_UnusedReason(),
        act=DefaultActStrategy(),
    )
    cycle = DecisionCycle(
        strategies=strategies,
        communication=transport or RecordingTransport(),
        actions=ActionRegistry(),
        registry=registry,
        working=working,
        semantic=semantic,
        procedural=ProceduralMemory(backend),
        episodic=EpisodicMemory(backend),
    )
    return cycle, working, semantic


def _add_activity(cycle: DecisionCycle, activity_id: str) -> Activity:
    """Invoke reads ``working.activities[activity_id]`` (unlike the other actions), so its tests
    seed the activity first."""
    activity = Activity(id=activity_id, goal="list emails", context={})
    cycle.working.activities[activity_id] = activity
    return activity


# --------------------------------------------------------------------------------------------------
# Invoke
# --------------------------------------------------------------------------------------------------


async def test_invoke_returns_immediate_ack_and_sets_running(tmp_path: Path) -> None:
    tool = FakeTool("EmailClientApp", invoke_results={"list_emails": {"emails": []}})
    registry, _ = _registry_with(tool)
    await registry.join(_ORIGIN)
    cycle, _, _ = _cycle(registry, tmp_path)
    activity = _add_activity(cycle, "a1")

    ack = await InvokeAction().execute(
        registry, cycle, activity_id="a1", tool_id="EmailClientApp", operation_name="list_emails"
    )

    # The ack is dispatch, not outcome — immediate, with the tool's own result still in flight.
    assert ack.ok is True
    assert activity.state is ActivityState.RUNNING
    assert activity.pending_operation is not None


async def test_invoke_records_bound_invocation_on_pending_operation(tmp_path: Path) -> None:
    tool = FakeTool("EmailClientApp", invoke_results={"list_emails": {"emails": []}})
    registry, _ = _registry_with(tool)
    await registry.join(_ORIGIN)
    cycle, _, _ = _cycle(registry, tmp_path)
    _add_activity(cycle, "a1")

    before = asyncio.get_running_loop().time()
    await InvokeAction().execute(
        registry, cycle, activity_id="a1", tool_id="EmailClientApp", operation_name="list_emails"
    )

    pending = cycle.working.activities["a1"].pending_operation
    assert pending is not None
    assert pending.id  # non-empty per-invoke id
    assert pending.invocation.tool_id == "EmailClientApp"
    assert pending.invocation.operation_name == "list_emails"
    assert pending.invocation.params == {}
    assert pending.invoked_at >= before  # a real timestamp, stamped at dispatch


async def test_invoke_result_lands_off_cycle_keyed_by_invocation_id(tmp_path: Path) -> None:
    tool = FakeTool("EmailClientApp", invoke_results={"list_emails": {"emails": [], "total": 0}})
    registry, _ = _registry_with(tool)
    await registry.join(_ORIGIN)
    cycle, _, _ = _cycle(registry, tmp_path)
    activity = _add_activity(cycle, "a1")

    await InvokeAction().execute(
        registry, cycle, activity_id="a1", tool_id="EmailClientApp", operation_name="list_emails"
    )
    assert activity.pending_operation is not None
    invocation_id = activity.pending_operation.id

    # The round-trip runs off-cycle; yield so the background task lands its result on result_sink.
    await asyncio.sleep(0)
    drained = [item async for item in cycle.result_sink.drain()]

    assert len(drained) == 1  # exactly one ack, and never delivered as a Percept
    op_id, op_ack = drained[0]
    assert op_id == invocation_id  # keyed by the invocation id, not the tool id
    assert op_ack.ok is True
    assert op_ack.result == {"emails": [], "total": 0}
    assert tool.invoked_with == ("list_emails", {})


async def test_invoke_passes_params_through_to_tool_and_invocation(tmp_path: Path) -> None:
    tool = FakeTool("EmailClientApp", invoke_results={"search": ["hit"]})
    registry, _ = _registry_with(tool)
    await registry.join(_ORIGIN)
    cycle, _, _ = _cycle(registry, tmp_path)
    activity = _add_activity(cycle, "a1")

    await InvokeAction().execute(
        registry,
        cycle,
        activity_id="a1",
        tool_id="EmailClientApp",
        operation_name="search",
        folder="inbox",
        query="invoice",
    )

    # Params (everything beyond tool_id/operation_name) flow to both the bound invocation...
    assert activity.pending_operation is not None
    assert activity.pending_operation.invocation.params == {"folder": "inbox", "query": "invoice"}
    # ...and through to the tool call itself.
    await asyncio.sleep(0)
    assert tool.invoked_with == ("search", {"folder": "inbox", "query": "invoice"})


async def test_concurrent_invokes_get_distinct_ids_and_both_land(tmp_path: Path) -> None:
    tool = FakeTool("EmailClientApp", invoke_results={"list_emails": {"emails": []}})
    registry, _ = _registry_with(tool)
    await registry.join(_ORIGIN)
    cycle, _, _ = _cycle(registry, tmp_path)
    a1 = _add_activity(cycle, "a1")
    a2 = _add_activity(cycle, "a2")
    invoke = InvokeAction()

    await invoke.execute(
        registry, cycle, activity_id="a1", tool_id="EmailClientApp", operation_name="list_emails"
    )
    await invoke.execute(
        registry, cycle, activity_id="a2", tool_id="EmailClientApp", operation_name="list_emails"
    )

    assert a1.pending_operation is not None and a2.pending_operation is not None
    assert a1.pending_operation.id != a2.pending_operation.id  # per-invoke, not per-tool

    await asyncio.sleep(0)
    drained = dict([item async for item in cycle.result_sink.drain()])
    assert set(drained) == {a1.pending_operation.id, a2.pending_operation.id}
    assert all(ack.ok for ack in drained.values())


# --------------------------------------------------------------------------------------------------
# Focus / Unfocus
# --------------------------------------------------------------------------------------------------


async def test_focus_subscribes_records_and_wires_signal_sink(tmp_path: Path) -> None:
    signal = Signal(name="new_email", payload={"n": 1})
    tool = FakeTool("EmailClientApp", signals_on_focus=[signal])
    registry, _ = _registry_with(tool)
    await registry.join(_ORIGIN)
    cycle, working, _ = _cycle(registry, tmp_path)

    ack = await FocusAction().execute(registry, cycle, activity_id="a1", tool_id="EmailClientApp")

    assert ack.ok is True
    assert working.focused_tools["EmailClientApp"] is tool
    assert tool.focused is True
    # The tool replayed its signal into whatever sink it got — proving it was the cycle's own.
    drained = [item async for item in cycle.signal_sink.drain()]
    assert drained == [("EmailClientApp", signal)]


async def test_unfocus_removes_and_calls_unfocus(tmp_path: Path) -> None:
    tool = FakeTool("EmailClientApp")
    registry, _ = _registry_with(tool)
    await registry.join(_ORIGIN)
    cycle, working, _ = _cycle(registry, tmp_path)
    await FocusAction().execute(registry, cycle, activity_id="a1", tool_id="EmailClientApp")

    ack = await UnfocusAction().execute(registry, cycle, activity_id="a1", tool_id="EmailClientApp")

    assert ack.ok is True
    assert "EmailClientApp" not in working.focused_tools
    assert tool.focused is False


async def test_unfocus_unknown_tool_is_noop(tmp_path: Path) -> None:
    registry, _ = _registry_with(FakeTool("EmailClientApp"))
    cycle, working, _ = _cycle(registry, tmp_path)

    ack = await UnfocusAction().execute(registry, cycle, activity_id="a1", tool_id="never-focused")

    assert ack.ok is True
    assert working.focused_tools == {}


async def test_unfocus_removes_the_tools_property_percepts(tmp_path: Path) -> None:
    # Unfocusing stops re-observing the tool, so its observable-property snapshot is permanently
    # stale and must be dropped. Only *this* tool's PROPERTY percepts go: another source's
    # properties and the tool's own (fire-and-forget) signals are retained.
    tool = FakeTool("EmailClientApp")
    registry, _ = _registry_with(tool)
    await registry.join(_ORIGIN)
    cycle, working, _ = _cycle(registry, tmp_path)
    await FocusAction().execute(registry, cycle, activity_id="a1", tool_id="EmailClientApp")
    own_prop = Percept("EmailClientApp", PerceptKind.PROPERTY, ObservableProperty("unread", 3), 0.0)
    other_prop = Percept("CalendarApp", PerceptKind.PROPERTY, ObservableProperty("busy", True), 0.0)
    own_signal = Percept("EmailClientApp", PerceptKind.SIGNAL, Signal("new_email", {"n": 1}), 0.0)
    working.perceptions.extend([own_prop, other_prop, own_signal])

    await UnfocusAction().execute(registry, cycle, activity_id="a1", tool_id="EmailClientApp")

    assert working.perceptions == [other_prop, own_signal]


# --------------------------------------------------------------------------------------------------
# Join / Leave
# --------------------------------------------------------------------------------------------------


async def test_join_registers_tools_and_persists_records(tmp_path: Path) -> None:
    tool = FakeTool("EmailClientApp", invoke_results={"list_emails": {"emails": []}})
    registry, workspace = _registry_with(tool)
    cycle, _, semantic = _cycle(registry, tmp_path)

    ack = await JoinAction().execute(registry, cycle, activity_id="a1", origin=_ORIGIN)

    # Registered live in the shared registry.
    assert ack.ok is True
    assert ack.result == {"workspace_id": "ws", "tool_ids": ["EmailClientApp"]}
    assert registry.get("EmailClientApp") is tool
    assert registry.get_workspace("ws") is workspace

    # Persisted durably via SemanticMemory (real file round-trip).
    ws_record = await semantic.retrieve_workspace_record("ws")
    assert ws_record is not None
    assert ws_record.origin == _ORIGIN

    tool_record = await semantic.retrieve_tool_record("EmailClientApp")
    assert tool_record is not None
    assert tool_record.manual_id == tool.manual.id
    assert tool_record.workspace_id == "ws"
    assert tool_record.address == tool.address

    stored_manual = await semantic.retrieve_manual(tool.manual.id)
    assert stored_manual is not None
    assert stored_manual.id == tool.manual.id


async def test_leave_closes_and_deregisters(tmp_path: Path) -> None:
    tool = FakeTool("EmailClientApp")
    registry, workspace = _registry_with(tool)
    await registry.join(_ORIGIN)
    cycle, _, _ = _cycle(registry, tmp_path)

    ack = await LeaveAction().execute(registry, cycle, activity_id="a1", workspace_id="ws")

    assert ack.ok is True
    assert workspace.closed is True
    with pytest.raises(KeyError):
        registry.get("EmailClientApp")
    with pytest.raises(KeyError):
        registry.get_workspace("ws")


async def test_leave_unfocuses_the_workspaces_focused_tools(tmp_path: Path) -> None:
    tool = FakeTool("EmailClientApp", signals_on_focus=[Signal("new_email", {"n": 1})])
    registry, workspace = _registry_with(tool)
    await registry.join(_ORIGIN)
    cycle, working, _ = _cycle(registry, tmp_path)
    await FocusAction().execute(registry, cycle, activity_id="a1", tool_id="EmailClientApp")
    assert "EmailClientApp" in working.focused_tools

    await LeaveAction().execute(registry, cycle, activity_id="a1", workspace_id="ws")

    # No stale focus survives leave: the handle is dropped and the subscription torn down.
    assert "EmailClientApp" not in working.focused_tools
    assert tool.focused is False
    assert workspace.closed is True


# --------------------------------------------------------------------------------------------------
# Send
# --------------------------------------------------------------------------------------------------


async def test_send_delegates_to_transport(tmp_path: Path) -> None:
    transport = RecordingTransport()
    registry, _ = _registry_with(FakeTool("EmailClientApp"))
    cycle, _, _ = _cycle(registry, tmp_path, transport=transport)

    ack = await SendAction().execute(
        registry, cycle, activity_id="a1", to="agent-b", content={"greeting": "hi"}
    )

    assert ack.ok is True
    assert transport.sent == [("agent-b", {"greeting": "hi"})]


# --------------------------------------------------------------------------------------------------
# Internal working-memory actions: _create_activity_ / _load_ / _unload_ / _filter_
# These touch only memory (no EnvironmentRegistry) and take the (cycle, **kwargs) InternalAction
# signature. Driven directly against the real FileMemoryBackend-backed cycle.
# --------------------------------------------------------------------------------------------------


async def test_create_activity_adds_ready_activity_to_working_memory(tmp_path: Path) -> None:
    registry, _ = _registry_with(FakeTool("EmailClientApp"))
    cycle, working, _ = _cycle(registry, tmp_path)

    activity = await CreateActivityAction().execute(cycle, goal="list emails")

    assert activity.goal == "list emails"
    assert activity.state is ActivityState.READY
    assert activity.id  # a generated, non-empty id
    assert working.activities[activity.id] is activity


async def test_create_activity_honors_explicit_id_and_context(tmp_path: Path) -> None:
    registry, _ = _registry_with(FakeTool("EmailClientApp"))
    cycle, working, _ = _cycle(registry, tmp_path)

    activity = await CreateActivityAction().execute(
        cycle, goal="g", activity_id="act-1", context={"k": "v"}
    )

    assert activity.id == "act-1"
    assert activity.context == {"k": "v"}
    assert working.activities["act-1"] is activity


async def test_load_manual_pulls_from_semantic_into_working_memory(tmp_path: Path) -> None:
    registry, _ = _registry_with(FakeTool("EmailClientApp"))
    cycle, working, semantic = _cycle(registry, tmp_path)
    await semantic.store_manual(fake_manual("clock", ["get_time"]))

    await LoadManualAction().execute(cycle, manual_id="clock")

    # Loaded from the durable store (a fresh copy, so compare by id, not identity).
    assert "clock" in working.loaded_manuals
    assert working.loaded_manuals["clock"].id == "clock"


async def test_load_manual_unknown_id_is_noop(tmp_path: Path) -> None:
    registry, _ = _registry_with(FakeTool("EmailClientApp"))
    cycle, working, _ = _cycle(registry, tmp_path)

    await LoadManualAction().execute(cycle, manual_id="missing")  # no raise

    assert working.loaded_manuals == {}


async def test_unload_manual_removes_from_working_memory(tmp_path: Path) -> None:
    registry, _ = _registry_with(FakeTool("EmailClientApp"))
    cycle, working, _ = _cycle(registry, tmp_path)
    working.loaded_manuals["clock"] = fake_manual("clock")

    await UnloadManualAction().execute(cycle, manual_id="clock")

    assert "clock" not in working.loaded_manuals


async def test_unload_manual_absent_id_is_noop(tmp_path: Path) -> None:
    registry, _ = _registry_with(FakeTool("EmailClientApp"))
    cycle, working, _ = _cycle(registry, tmp_path)

    await UnloadManualAction().execute(cycle, manual_id="never-loaded")  # no raise

    assert working.loaded_manuals == {}


async def test_filter_keeps_only_relevant_source_percepts(tmp_path: Path) -> None:
    registry, _ = _registry_with(FakeTool("EmailClientApp"))
    cycle, working, _ = _cycle(registry, tmp_path)
    keep = Percept("EmailClientApp", PerceptKind.PROPERTY, ObservableProperty("unread", 1), 0.0)
    drop = Percept("stranger", PerceptKind.PROPERTY, ObservableProperty("x", 2), 0.0)
    working.perceptions.extend([keep, drop])

    await FilterPerceptionsAction().execute(cycle, tool_ids={"EmailClientApp"})

    assert working.perceptions == [keep]  # mutated in place, stranger dropped


async def test_filter_retains_signals_regardless_of_source(tmp_path: Path) -> None:
    registry, _ = _registry_with(FakeTool("EmailClientApp"))
    cycle, working, _ = _cycle(registry, tmp_path)
    prop = Percept("stranger", PerceptKind.PROPERTY, ObservableProperty("x", 2), 0.0)
    signal = Percept("stranger", PerceptKind.SIGNAL, Signal("new_email", {"n": 1}), 0.0)
    working.perceptions.extend([prop, signal])

    await FilterPerceptionsAction().execute(cycle, tool_ids={"EmailClientApp"})

    # The property from an irrelevant source is pruned; the fire-and-forget signal is always kept
    # (its lifecycle is consumption-driven, owned by the blocked-state machinery, not _filter_).
    assert working.perceptions == [signal]


# --------------------------------------------------------------------------------------------------
# ActionRegistry lookup
# --------------------------------------------------------------------------------------------------


class _NoopInternal:
    name = "noop"

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> Any:
        return None


async def test_action_registry_lookup_external() -> None:
    reg = ActionRegistry()
    actions: list[ExternalAction] = [
        InvokeAction(),
        FocusAction(),
        UnfocusAction(),
        JoinAction(),
        LeaveAction(),
        SendAction(),
    ]
    for action in actions:
        reg.register_external(action)
    for action in actions:
        assert reg.external(action.name) is action  # keyed by each action's own name constant


async def test_action_registry_lookup_internal() -> None:
    reg = ActionRegistry()
    action = _NoopInternal()
    reg.register_internal(action)
    assert reg.internal("noop") is action


async def test_action_registry_unknown_raises_keyerror() -> None:
    reg = ActionRegistry()
    with pytest.raises(KeyError):
        reg.external("nope")
    with pytest.raises(KeyError):
        reg.internal("nope")


async def test_default_action_registry_has_all_predefined_actions() -> None:
    reg = default_action_registry()
    for name in ("invoke", "focus", "unfocus", "join", "leave", "send"):
        assert reg.external(name).name == name
    for name in ("create_activity", "load", "unload", "filter"):
        assert reg.internal(name).name == name


# --------------------------------------------------------------------------------------------------
# requires_binding — the action declares whether the cycle's Act phase binds its step
# --------------------------------------------------------------------------------------------------


def test_requires_binding_only_for_invoke() -> None:
    # The cycle binds a step into a concrete OperationInvocation iff its action declares
    # requires_binding — only invoke does. This replaces the removed `next_action == "invoke"`
    # hardcode in tick(): binding is now a property of the action, not a branch in the cycle.
    assert InvokeAction().requires_binding is True
    for action in (FocusAction(), UnfocusAction(), JoinAction(), LeaveAction(), SendAction()):
        assert action.requires_binding is False


def test_invoke_step_packs_routing_and_args() -> None:
    # invoke_step is the one-source-of-truth constructor for an `invoke` Step: it packs the routing
    # keys (tool_id, operation_name) alongside the operation arguments in params, under the
    # TOOL_ID/OPERATION_NAME constants — so no call site hand-writes that magic-keyed dict.
    step = invoke_step("EmailClientApp", "search_emails", query="urgent", limit=5)

    assert step.next_action == InvokeAction.name
    assert step.params == {
        TOOL_ID: "EmailClientApp",
        OPERATION_NAME: "search_emails",
        "query": "urgent",
        "limit": 5,
    }
