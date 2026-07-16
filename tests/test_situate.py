"""Permanent TDD tests for the Situate phase — ``DefaultSituateStrategy`` and the §7a fix.

Situate is the head of the decision chain (Situate -> Reason -> Act). Unlike Reason/Act it is *not*
gated on its own output field: it **always runs** so it can re-adjust working memory for the
(possibly already-selected) activity every cycle, selecting an activity only if ``result.activity``
is still ``None``. The deterministic default:

* **creates** an activity from any unhandled message (one that maps to no existing activity, keyed
  by derived goal) via the internal ``_create_activity_`` action;
* **adjusts** working memory for the currently-joined workspaces — loads their tools' manuals into
  ``wm.loaded_manuals`` (``_load_``), unloads manuals no longer backed by a joined tool
  (``_unload_``), and filters ``wm.perceptions`` down to joined-tool sources (``_filter_``);
* **selects** the first ready activity if none is pre-set.

Focusing tools is *not* done here — ``_focus_`` is an external action (one external action per
cycle, dispatched at Act), so Situate performs only the internal adjustments. The harness reuses
``tests/fakes.py`` + a real ``FileMemoryBackend`` and a scripted transport (modeled on
``test_cycle.py``). ``DefaultSituateStrategy`` is exercised directly and through an Observe-only
``tick()`` with an inert Reason.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fakes import FakeAdapter, FakeTool, FakeWorkspace, fake_manual
from sora.action import default_action_registry
from sora.activity import Activity
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
    RoundRobinActivitySelection,
    Strategies,
    TickResult,
)
from sora.types import ObservableProperty, Signal

# --------------------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------------------

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")


class ScriptedTransport:
    """Satisfies MessageTransport: ``receive()`` drains a preset inbound list; ``send()`` logs."""

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
    """A ReasonStrategy stand-in that yields no step — so a tick() reaching Reason dispatches no
    external action, isolating what Situate did."""

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
    registry: EnvironmentRegistry, tmp_path: Path, transport: ScriptedTransport | None = None
) -> tuple[DecisionCycle, WorkingMemory, SemanticMemory]:
    backend = FileMemoryBackend(tmp_path)
    semantic = SemanticMemory(backend)
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
        actions=default_action_registry(),
        registry=registry,
        working=working,
        semantic=semantic,
        procedural=ProceduralMemory(backend),
        episodic=EpisodicMemory(backend),
    )
    return cycle, working, semantic


def _user_message(text: str) -> Message:
    return Message(sender="user", content={"text": text}, received_at=0.0)


# --------------------------------------------------------------------------------------------------
# Activity creation from unhandled messages
# --------------------------------------------------------------------------------------------------


async def test_situate_creates_and_selects_activity_from_unhandled_message(tmp_path: Path) -> None:
    registry, _ = _registry_with()
    cycle, working, _ = _cycle(registry, tmp_path)
    working.messages.append(_user_message("what time is it?"))

    result = await DefaultSituateStrategy().situate([], working, cycle, TickResult())

    assert len(working.activities) == 1
    created = next(iter(working.activities.values()))
    assert created.goal == "what time is it?"  # derived from message content["text"]
    assert result.activity is created  # created then selected the same cycle
    # Non-destructive: the message stays in working memory for Reason strategies to read.
    assert len(working.messages) == 1


async def test_situate_dedups_message_matching_existing_activity(tmp_path: Path) -> None:
    registry, _ = _registry_with()
    cycle, working, _ = _cycle(registry, tmp_path)
    existing = Activity(id="a1", goal="what time is it?", context={})
    working.activities["a1"] = existing
    working.messages.append(_user_message("what time is it?"))

    await DefaultSituateStrategy().situate([existing], working, cycle, TickResult())

    # Same derived goal as an existing activity -> no duplicate created.
    assert list(working.activities) == ["a1"]


async def test_situate_derives_goal_from_content_without_text_key(tmp_path: Path) -> None:
    registry, _ = _registry_with()
    cycle, working, _ = _cycle(registry, tmp_path)
    working.messages.append(Message(sender="agent-b", content={"query": "ping"}, received_at=0.0))

    await DefaultSituateStrategy().situate([], working, cycle, TickResult())

    created = next(iter(working.activities.values()))
    assert created.goal == str({"query": "ping"})  # deterministic fallback, no interpretation


# --------------------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------------------


async def test_situate_selects_first_ready_activity(tmp_path: Path) -> None:
    registry, _ = _registry_with()
    cycle, working, _ = _cycle(registry, tmp_path)
    a1 = Activity(id="a1", goal="g1", context={})
    a2 = Activity(id="a2", goal="g2", context={})
    working.activities["a1"] = a1
    working.activities["a2"] = a2

    result = await DefaultSituateStrategy().situate([a1, a2], working, cycle, TickResult())

    assert result.activity is a1


async def test_situate_returns_no_activity_when_none_ready(tmp_path: Path) -> None:
    registry, _ = _registry_with()
    cycle, working, _ = _cycle(registry, tmp_path)

    result = await DefaultSituateStrategy().situate([], working, cycle, TickResult())

    assert result.activity is None


# --------------------------------------------------------------------------------------------------
# RoundRobinActivitySelection — the default selection sub-strategy, driven directly
# --------------------------------------------------------------------------------------------------


def _ready(*ids: str) -> list[Activity]:
    return [Activity(id=i, goal=i, context={}) for i in ids]


async def test_round_robin_rotates_across_persistently_ready_activities(tmp_path: Path) -> None:
    registry, _ = _registry_with()
    cycle, working, _ = _cycle(registry, tmp_path)
    selection = RoundRobinActivitySelection()
    ready = _ready("a", "b", "c")

    picks = [await selection.select(ready, working, cycle) for _ in range(4)]

    # Cold start picks the oldest, then the cursor advances and wraps: a -> b -> c -> a.
    assert [p.id for p in picks if p is not None] == ["a", "b", "c", "a"]


async def test_round_robin_single_ready_is_repicked_every_call(tmp_path: Path) -> None:
    registry, _ = _registry_with()
    cycle, working, _ = _cycle(registry, tmp_path)
    selection = RoundRobinActivitySelection()
    ready = _ready("solo")

    picks = [await selection.select(ready, working, cycle) for _ in range(3)]

    # (0 + 1) % 1 == 0 -> the sole activity is re-picked, never starved, never None.
    assert [p.id for p in picks if p is not None] == ["solo", "solo", "solo"]


async def test_round_robin_empty_ready_returns_none(tmp_path: Path) -> None:
    registry, _ = _registry_with()
    cycle, working, _ = _cycle(registry, tmp_path)

    assert await RoundRobinActivitySelection().select([], working, cycle) is None


async def test_round_robin_falls_back_to_oldest_when_last_pick_gone(tmp_path: Path) -> None:
    registry, _ = _registry_with()
    cycle, working, _ = _cycle(registry, tmp_path)
    selection = RoundRobinActivitySelection()

    first = await selection.select(_ready("a", "b"), working, cycle)  # cold start -> a
    # 'a' is no longer ready next cycle; the cursor's target is gone -> restart at the oldest.
    second = await selection.select(_ready("b", "c"), working, cycle)

    assert first is not None and first.id == "a"
    assert second is not None and second.id == "b"


async def test_round_robin_cold_start_first_pick_is_oldest(tmp_path: Path) -> None:
    registry, _ = _registry_with()
    cycle, working, _ = _cycle(registry, tmp_path)

    pick = await RoundRobinActivitySelection().select(_ready("a", "b", "c"), working, cycle)

    # A fresh cursor reproduces the old priority-by-age behavior on its first pick.
    assert pick is not None and pick.id == "a"


# --------------------------------------------------------------------------------------------------
# DefaultSituateStrategy delegates selection to its (pluggable) sub-strategy
# --------------------------------------------------------------------------------------------------


async def test_situate_reused_instance_rotates_selection(tmp_path: Path) -> None:
    registry, _ = _registry_with()
    cycle, working, _ = _cycle(registry, tmp_path)
    a1 = Activity(id="a1", goal="g1", context={})
    a2 = Activity(id="a2", goal="g2", context={})
    working.activities["a1"] = a1
    working.activities["a2"] = a2
    situate = DefaultSituateStrategy()  # one instance reused across cycles -> cursor persists

    first = await situate.situate([a1, a2], working, cycle, TickResult())
    second = await situate.situate([a1, a2], working, cycle, TickResult())

    # Both stay READY, so a static priority-by-age default would pin a1 twice; the round-robin
    # cursor on the Situate instance rotates to a2 instead — selection is delegated, not inlined.
    assert first.activity is a1
    assert second.activity is a2


async def test_situate_uses_injected_selection_strategy(tmp_path: Path) -> None:
    registry, _ = _registry_with()
    cycle, working, _ = _cycle(registry, tmp_path)
    a1 = Activity(id="a1", goal="g1", context={})
    a2 = Activity(id="a2", goal="g2", context={})
    working.activities["a1"] = a1
    working.activities["a2"] = a2

    class _PinSecond:
        """A custom selection sub-strategy: always the second ready activity, never ready[0]."""

        async def select(
            self, ready: list[Activity], wm: WorkingMemory, cycle: DecisionCycle
        ) -> Activity | None:
            return ready[1] if len(ready) > 1 else None

    result = await DefaultSituateStrategy(selection=_PinSecond()).situate(
        [a1, a2], working, cycle, TickResult()
    )

    # The injected policy overrides the round-robin default end-to-end through situate().
    assert result.activity is a2


# --------------------------------------------------------------------------------------------------
# §7a — Situate always runs: a pre-set selection is respected, but wm is still adjusted
# --------------------------------------------------------------------------------------------------


async def test_situate_respects_preset_activity_but_still_adjusts(tmp_path: Path) -> None:
    tool = FakeTool("clock")
    registry, _ = _registry_with(tool)
    await registry.join(_ORIGIN)
    cycle, working, semantic = _cycle(registry, tmp_path)
    await semantic.store_manual(tool.manual)  # so _load_ can retrieve it from the durable store
    preset = Activity(id="pinned", goal="urgent", context={})
    working.activities["pinned"] = preset

    result = await DefaultSituateStrategy().situate(
        [preset], working, cycle, TickResult(activity=preset)
    )

    # The pre-set selection is respected (not overridden)...
    assert result.activity is preset
    # ...yet wm adjustment still ran — the joined tool's manual was loaded.
    assert tool.manual.id in working.loaded_manuals


# --------------------------------------------------------------------------------------------------
# wm adjustment: load / unload / filter for the joined workspaces
# --------------------------------------------------------------------------------------------------


async def test_situate_loads_manuals_for_joined_tools(tmp_path: Path) -> None:
    tool = FakeTool("clock")
    registry, _ = _registry_with(tool)
    await registry.join(_ORIGIN)
    cycle, working, semantic = _cycle(registry, tmp_path)
    await semantic.store_manual(tool.manual)
    a1 = Activity(id="a1", goal="g", context={})
    working.activities["a1"] = a1

    await DefaultSituateStrategy().situate([a1], working, cycle, TickResult())

    assert tool.manual.id in working.loaded_manuals
    assert working.loaded_manuals[tool.manual.id].id == tool.manual.id


async def test_situate_unloads_manual_no_longer_backed_by_a_joined_tool(tmp_path: Path) -> None:
    registry, _ = _registry_with()  # nothing joined
    cycle, working, _ = _cycle(registry, tmp_path)
    working.loaded_manuals["gone"] = fake_manual("gone")
    a1 = Activity(id="a1", goal="g", context={})
    working.activities["a1"] = a1

    await DefaultSituateStrategy().situate([a1], working, cycle, TickResult())

    assert "gone" not in working.loaded_manuals


async def test_situate_filters_properties_but_retains_signals(tmp_path: Path) -> None:
    tool = FakeTool("clock")
    registry, _ = _registry_with(tool)
    await registry.join(_ORIGIN)
    cycle, working, semantic = _cycle(registry, tmp_path)
    await semantic.store_manual(tool.manual)
    keep_prop = Percept("clock", PerceptKind.PROPERTY, ObservableProperty("time", "10:00"), 0.0)
    drop_prop = Percept("stranger", PerceptKind.PROPERTY, ObservableProperty("x", 1), 0.0)
    keep_signal = Percept("stranger", PerceptKind.SIGNAL, Signal("blip", {}), 0.0)
    working.perceptions.extend([keep_prop, drop_prop, keep_signal])
    a1 = Activity(id="a1", goal="g", context={})
    working.activities["a1"] = a1

    await DefaultSituateStrategy().situate([a1], working, cycle, TickResult())

    # The property from an unengaged source is pruned; the joined tool's property and the
    # fire-and-forget signal (even from an unengaged source) are retained.
    assert working.perceptions == [keep_prop, keep_signal]


# --------------------------------------------------------------------------------------------------
# tick() integration — Situate always runs inside the real cycle
# --------------------------------------------------------------------------------------------------


async def test_tick_creates_selects_and_adjusts_from_message(tmp_path: Path) -> None:
    tool = FakeTool("clock")
    registry, _ = _registry_with(tool)
    transport = ScriptedTransport(inbound=[_user_message("what time is it?")])
    cycle, working, semantic = _cycle(registry, tmp_path, transport=transport)
    await registry.join(_ORIGIN)
    await semantic.store_manual(tool.manual)

    await cycle.tick()

    # Observe drained the message; Situate turned it into a selected activity and adjusted wm.
    assert len(working.activities) == 1
    created = next(iter(working.activities.values()))
    assert created.goal == "what time is it?"
    assert tool.manual.id in working.loaded_manuals
    # The inert Reason yields no step, so no external action was dispatched.
    assert tool.invoked_with is None
