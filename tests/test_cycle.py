"""Permanent TDD tests for the Observe phase and the Observe-only ``tick()`` path.

Observe is the one deterministic, no-LLM phase that always runs first and never selects an activity
(the decision chain starts at Situate). ``DefaultObserveStrategy.observe(cycle)`` runs four channels
and returns an empty ``TickResult()``:

* focused-tool ``observe()`` properties -> ``PROPERTY`` percepts;
* ``signal_sink`` -> ``SIGNAL`` percepts;
* ``result_sink`` -> the automatic 1:1 running-resolution (an ack matching an activity's
  ``pending_operation.id`` drives ``RUNNING -> READY``, clears pending, sets ``last_operation`` — no
  ``Percept``, no strategy judgment; a non-matching ack is silently dropped);
* ``communication.receive()`` -> inbound ``messages``.

The harness reuses ``tests/fakes.py`` and a real ``FileMemoryBackend``, with a local scripted
transport whose ``receive()`` drains a preset message list (modeled on ``test_action.py``'s
``RecordingTransport``). Promotes the spike's ``observe_resolves_running_activity`` from
``tests/test_cycle_wiring.py`` (triage row in ``docs/phase-3-test-triage.md``). The Reflect/Situate/
Reason/Act defaults are inert plumbing here — the Observe-only tick relies on them but asserts only
Observe's effects and non-dispatch.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fakes import FakeTool
from sora.action import ActionRegistry
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from sora.perception import Message, PerceptKind
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    Strategies,
    TickResult,
)
from sora.types import (
    ObservableProperty,
    OperationAck,
    OperationInvocation,
    PendingOperation,
    Signal,
)

# --------------------------------------------------------------------------------------------------
# Harness — the shared fakes plus a scripted transport and a real FileMemoryBackend-backed cycle.
# --------------------------------------------------------------------------------------------------


class ScriptedTransport:
    """Satisfies MessageTransport: ``receive()`` drains a preset inbound list (so a second tick
    doesn't re-yield the same messages); ``send()`` logs its args."""

    def __init__(self, inbound: list[Message] | None = None) -> None:
        self._inbound = list(inbound or [])
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def send(self, to: str, content: dict[str, Any]) -> None:
        self.sent.append((to, content))

    def receive(self) -> AsyncIterator[Message]:
        async def _drain() -> AsyncIterator[Message]:
            while self._inbound:
                yield self._inbound.pop(0)

        return _drain()


class _InertReason:
    """A ReasonStrategy stand-in; only reached once an activity is selected, which the Observe-only
    tests deliberately avoid."""

    async def reason(
        self, activity: Any, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        return result


def _cycle(
    tmp_path: Path, transport: ScriptedTransport | None = None
) -> tuple[DecisionCycle, WorkingMemory]:
    backend = FileMemoryBackend(tmp_path)
    registry = EnvironmentRegistry()
    working = WorkingMemory(registry=registry)
    strategies = Strategies(
        observe=DefaultObserveStrategy(),
        reflect=DefaultReflectStrategy(),
        situate=DefaultSituateStrategy(),
        reason=_InertReason(),
        act=DefaultActStrategy(),
    )
    cycle = DecisionCycle(
        strategies=strategies,
        communication=transport or ScriptedTransport(),
        actions=ActionRegistry(),
        registry=registry,
        working=working,
        semantic=SemanticMemory(backend),
        procedural=ProceduralMemory(backend),
        episodic=EpisodicMemory(backend),
    )
    return cycle, working


def _running_activity(activity_id: str, op_id: str) -> Activity:
    """A RUNNING activity with a bound ``pending_operation`` keyed by ``op_id`` — the state that
    Observe's 1:1 running-resolution consumes (as InvokeAction leaves it, minus the round-trip)."""
    return Activity(
        id=activity_id,
        goal="list emails",
        context={},
        state=ActivityState.RUNNING,
        pending_operation=PendingOperation(
            id=op_id,
            invocation=OperationInvocation(
                tool_id="EmailClientApp", operation_name="list_emails", params={}
            ),
            invoked_at=0.0,
        ),
    )


# --------------------------------------------------------------------------------------------------
# Observe — the four deterministic channels
# --------------------------------------------------------------------------------------------------


async def test_observe_returns_empty_tick_result(tmp_path: Path) -> None:
    # Observe never selects an activity — the decision chain starts at Situate.
    cycle, _ = _cycle(tmp_path)
    result = await DefaultObserveStrategy().observe(cycle)
    assert result == TickResult()


async def test_observe_emits_property_percepts_from_focused_tools(tmp_path: Path) -> None:
    prop = ObservableProperty(name="unread", value=3)
    tool = FakeTool("EmailClientApp", properties=[prop])
    cycle, working = _cycle(tmp_path)
    working.focused_tools["EmailClientApp"] = tool

    await DefaultObserveStrategy().observe(cycle)

    assert len(working.perceptions) == 1
    percept = working.perceptions[0]
    assert percept.source == tool.id
    assert percept.kind is PerceptKind.PROPERTY
    assert percept.payload == prop


async def test_observe_emits_signal_percepts_from_signal_sink(tmp_path: Path) -> None:
    signal = Signal(name="new_email", payload={"n": 1})
    cycle, working = _cycle(tmp_path)
    cycle.signal_sink.push("EmailClientApp", signal)

    await DefaultObserveStrategy().observe(cycle)

    assert len(working.perceptions) == 1
    percept = working.perceptions[0]
    assert percept.source == "EmailClientApp"
    assert percept.kind is PerceptKind.SIGNAL
    assert percept.payload == signal


async def test_observe_appends_inbound_messages(tmp_path: Path) -> None:
    message = Message(sender="agent-b", content={"greeting": "hi"}, received_at=0.0)
    transport = ScriptedTransport(inbound=[message])
    cycle, working = _cycle(tmp_path, transport=transport)

    await DefaultObserveStrategy().observe(cycle)

    assert working.messages == [message]


# --------------------------------------------------------------------------------------------------
# Observe — the automatic 1:1 running-resolution off result_sink
# --------------------------------------------------------------------------------------------------


async def test_observe_resolves_running_activity(tmp_path: Path) -> None:
    cycle, working = _cycle(tmp_path)
    activity = _running_activity("a1", op_id="op-1")
    working.activities["a1"] = activity
    ack = OperationAck(ok=True, result={"emails": []})
    cycle.result_sink.push("op-1", ack)

    await DefaultObserveStrategy().observe(cycle)

    assert activity.state is ActivityState.READY
    assert activity.pending_operation is None
    assert activity.last_operation == ack
    # A resolved operation result is never surfaced as a Percept.
    assert working.perceptions == []


async def test_observe_resolves_only_the_matching_activity(tmp_path: Path) -> None:
    cycle, working = _cycle(tmp_path)
    matched = _running_activity("a1", op_id="op-1")
    other = _running_activity("a2", op_id="op-2")
    working.activities["a1"] = matched
    working.activities["a2"] = other
    cycle.result_sink.push("op-1", OperationAck(ok=True, result={"emails": []}))

    await DefaultObserveStrategy().observe(cycle)

    assert matched.state is ActivityState.READY
    assert matched.pending_operation is None
    # The unmatched activity keeps waiting on its own in-flight operation.
    assert other.state is ActivityState.RUNNING
    assert other.pending_operation is not None
    assert other.last_operation is None


async def test_observe_ignores_unmatched_result_ack(tmp_path: Path) -> None:
    # An ack whose id matches no pending operation is silently dropped (the 1:1 guard) — no raise,
    # no state change, and never re-surfaced as a Percept.
    cycle, working = _cycle(tmp_path)
    activity = _running_activity("a1", op_id="op-1")
    working.activities["a1"] = activity
    cycle.result_sink.push("stranger", OperationAck(ok=True, result="unrelated"))

    await DefaultObserveStrategy().observe(cycle)

    assert activity.state is ActivityState.RUNNING
    assert activity.pending_operation is not None
    assert activity.last_operation is None
    assert working.perceptions == []
    assert cycle.result_sink._queue.empty()


# --------------------------------------------------------------------------------------------------
# The Observe-only tick() path
# --------------------------------------------------------------------------------------------------


async def test_tick_observes_then_returns_when_nothing_selectable(tmp_path: Path) -> None:
    # With no ready activity, a real tick() still runs Observe (percepts + messages populated) but
    # returns without dispatching any external action.
    prop = ObservableProperty(name="unread", value=3)
    tool = FakeTool("EmailClientApp", properties=[prop])
    signal = Signal(name="new_email", payload={"n": 1})
    message = Message(sender="agent-b", content={"greeting": "hi"}, received_at=0.0)
    transport = ScriptedTransport(inbound=[message])
    cycle, working = _cycle(tmp_path, transport=transport)
    working.focused_tools["EmailClientApp"] = tool
    cycle.signal_sink.push("EmailClientApp", signal)

    await cycle.tick()

    kinds = {p.kind for p in working.perceptions}
    assert kinds == {PerceptKind.PROPERTY, PerceptKind.SIGNAL}
    assert working.messages == [message]
    # No activity was selectable, so no external action was dispatched.
    assert tool.invoked_with is None
