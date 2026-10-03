from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from fakes import FakeTool
from sora._strategies.observe import DefaultObserveStrategy
from sora.action import ActionRegistry, CreateActivityAction
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.diagnostics import (
    RuntimeEventCollector,
    collect_runtime_events,
    emit_runtime_event,
    runtime_event_context,
    runtime_phase,
)
from sora.environment import EnvironmentRegistry
from sora.memory import WorkingMemory
from sora.types import (
    ActionAck,
    InferenceKind,
    ObservableProperty,
    OperationInvocation,
    PendingInference,
    PendingOperation,
    Plan,
    Step,
)


class _FailingSink:
    def emit(self, event: dict[str, Any]) -> None:
        raise RuntimeError("collector failed")


class _Internal:
    name = "diagnose"

    async def execute(self, cycle: Any, **kwargs: Any) -> dict[str, Any]:
        return {"seen": kwargs["value"]}


class _External:
    name = "send-diagnostic"
    requires_binding = False

    async def execute(
        self,
        registry: Any,
        cycle: Any,
        *,
        activity_id: str,
        **kwargs: Any,
    ) -> ActionAck:
        return ActionAck(True, {"activity_id": activity_id, "seen": kwargs["value"]})


class _DataOp:
    name = "diagnostic-filter"

    async def execute(self, cycle: Any, **kwargs: Any) -> None:
        return None


def test_events_are_json_safe_ordered_and_cover_activity_states() -> None:
    collector = RuntimeEventCollector()
    with collect_runtime_events(collector), runtime_event_context(cycle=3, phase="reason"):
        activity = Activity("a1", "goal", {})
        emit_runtime_event(
            "activity.created", activity_id=activity.id, payload={"state": activity.state}
        )
        activity.state = ActivityState.RUNNING
        activity.pending_operation = PendingOperation(
            "o1", OperationInvocation("tool", "operation", {}), 9.0
        )
        activity.pending_operation = None
        activity.pending_inference = PendingInference("i1", InferenceKind.PLAN, 10.0)
        activity.pending_inference = None
        activity.state = ActivityState.BLOCKED
        activity.state = ActivityState.READY
        activity.plan = Plan(id="p1", goal="goal", steps=[])
        activity.reset_for_replan("bad reference")
        activity.state = ActivityState.TERMINATED

    rows = collector.snapshot()
    assert [row["sequence"] for row in rows] == list(range(len(rows)))
    assert {row["payload"]["to"] for row in rows if row["event"] == "activity.transition"} >= {
        "running",
        "blocked",
        "ready",
        "terminated",
    }
    assert {row["event"] for row in rows} >= {
        "pending_inference.created",
        "pending_inference.cleared",
        "pending_operation.created",
        "pending_operation.cleared",
        "plan.installed",
        "plan.replan",
        "plan.cleared",
    }
    json.dumps(rows)


def test_activity_assignment_skips_old_value_lookup_without_diagnostics() -> None:
    class _Unreadable:
        def __getattribute__(self, name: str) -> Any:
            raise AssertionError(f"unexpected read of {name}")

    probe = object.__new__(_Unreadable)

    Activity.__setattr__(cast(Any, probe), "value", 3)

    assert object.__getattribute__(probe, "value") == 3


def test_plan_install_and_replan_events_bound_large_plans() -> None:
    collector = RuntimeEventCollector()
    plan = Plan(
        id="large",
        goal="diagnose",
        steps=[
            Step(next_action="inspect", params={"items": list(range(1_000))}) for _ in range(100)
        ],
    )
    with collect_runtime_events(collector):
        activity = Activity("a1", "goal", {})
        activity.plan = plan
        activity.reset_for_replan("bad reference")

    rows = collector.snapshot()
    installed = next(row for row in rows if row["event"] == "plan.installed")
    replanned = next(row for row in rows if row["event"] == "plan.replan")
    assert installed["payload"]["plan"]["steps"][-1] == {
        "__diagnostic_truncated__": True,
        "length": 100,
    }
    assert replanned["payload"]["plan"]["steps"][-1] == {
        "__diagnostic_truncated__": True,
        "length": 100,
    }


@pytest.mark.asyncio
async def test_activity_creation_bounds_large_context() -> None:
    collector = RuntimeEventCollector()
    cycle = SimpleNamespace(working=SimpleNamespace(activities={}))
    with collect_runtime_events(collector):
        await CreateActivityAction().execute(
            cast(Any, cycle), goal="goal", context={"items": list(range(10_000))}
        )

    created = next(row for row in collector.snapshot() if row["event"] == "activity.created")
    assert created["payload"]["context"]["items"][-1] == {
        "__diagnostic_truncated__": True,
        "length": 10_000,
    }


@pytest.mark.asyncio
async def test_action_dispatch_and_result_share_activity_correlation() -> None:
    registry = ActionRegistry()
    registry.register_internal(_Internal())
    collector = RuntimeEventCollector()
    with collect_runtime_events(collector), runtime_phase(9, "situate"):
        result = await registry.internal("diagnose").execute(
            cast(Any, SimpleNamespace()), activity_id="a1", value=4
        )

    assert result == {"seen": 4}
    action_rows = [row for row in collector.snapshot() if row["event"].startswith("action.")]
    assert [row["event"] for row in action_rows] == ["action.dispatch", "action.result"]
    assert all(row["activity_id"] == "a1" for row in action_rows)
    assert all(row["cycle"] == 9 and row["phase"] == "situate" for row in action_rows)


@pytest.mark.asyncio
async def test_internal_action_diagnostics_do_not_walk_large_arguments() -> None:
    registry = ActionRegistry()
    registry.register_internal(_Internal())
    collector = RuntimeEventCollector()
    collection = list(range(10_000))
    with collect_runtime_events(collector):
        await registry.internal("diagnose").execute(
            cast(Any, SimpleNamespace()), activity_id="a1", value=collection
        )

    dispatch = next(row for row in collector.snapshot() if row["event"] == "action.dispatch")
    assert dispatch["payload"]["arguments"]["value"] == {"type": "list", "length": 10_000}


@pytest.mark.asyncio
async def test_data_op_diagnostics_keep_predicate_but_summarize_collection() -> None:
    registry = ActionRegistry()
    registry.register_data_op(_DataOp())
    collector = RuntimeEventCollector()
    where = {
        "all": [
            {"path": "status", "op": "eq", "value": "open"},
            {"path": "priority", "op": "ge", "value": 3},
        ]
    }
    with collect_runtime_events(collector):
        await registry.data_op("diagnostic-filter").execute(
            cast(Any, SimpleNamespace()),
            activity_id="a1",
            collection=list(range(10_000)),
            out="matches",
            where=where,
        )

    dispatch = next(row for row in collector.snapshot() if row["event"] == "action.dispatch")
    arguments = dispatch["payload"]["arguments"]
    assert arguments["collection"] == {"type": "list", "length": 10_000}
    assert arguments["out"] == "matches"
    assert arguments["where"] == where


@pytest.mark.asyncio
async def test_external_action_dispatch_and_result_are_correlated() -> None:
    registry = ActionRegistry()
    registry.register_external(_External())
    collector = RuntimeEventCollector()
    with collect_runtime_events(collector), runtime_phase(10, "act", activity_id="a2"):
        result = await registry.external("send-diagnostic").execute(
            cast(Any, SimpleNamespace()),
            cast(Any, SimpleNamespace()),
            activity_id="a2",
            value=5,
        )

    assert result == ActionAck(True, {"activity_id": "a2", "seen": 5})
    action_rows = [row for row in collector.snapshot() if row["event"].startswith("action.")]
    assert [row["event"] for row in action_rows] == ["action.dispatch", "action.result"]
    assert all(row["activity_id"] == "a2" for row in action_rows)
    assert all(row["payload"]["category"] == "external" for row in action_rows)


@pytest.mark.asyncio
async def test_external_action_diagnostics_bound_arguments_and_results() -> None:
    registry = ActionRegistry()
    registry.register_external(_External())
    collector = RuntimeEventCollector()
    collection = list(range(10_000))
    with collect_runtime_events(collector):
        await registry.external("send-diagnostic").execute(
            cast(Any, SimpleNamespace()),
            cast(Any, SimpleNamespace()),
            activity_id="a2",
            value=collection,
        )

    dispatch, result = [row for row in collector.snapshot() if row["event"].startswith("action.")]
    projected = dispatch["payload"]["arguments"]["value"]
    assert len(projected) == 33
    assert projected[-1] == {"__diagnostic_truncated__": True, "length": 10_000}
    result_value = result["payload"]["result"]["result"]["seen"]
    assert len(result_value) == 33
    assert result_value[-1]["__diagnostic_truncated__"] is True


@pytest.mark.asyncio
async def test_concurrent_emission_has_one_total_order() -> None:
    collector = RuntimeEventCollector()

    async def emit(index: int) -> None:
        await asyncio.sleep(0)
        emit_runtime_event("inference.completed", payload={"index": index})

    with collect_runtime_events(collector):
        await asyncio.gather(*(emit(index) for index in range(40)))

    rows = collector.snapshot()
    assert [row["sequence"] for row in rows] == list(range(40))
    assert {row["payload"]["index"] for row in rows} == set(range(40))


@pytest.mark.asyncio
async def test_cycle_context_covers_events_between_phases() -> None:
    cycle = cast(Any, object.__new__(DecisionCycle))
    cycle._cycle_count = 0

    async def body(cycle_number: int) -> None:
        assert cycle_number == 1
        emit_runtime_event("between.phases")

    cycle._tick_body = body
    collector = RuntimeEventCollector()
    with collect_runtime_events(collector):
        await cycle.tick()

    assert [row["cycle"] for row in collector.snapshot()] == [1, 1, 1]


def test_failing_sink_cannot_change_runtime_behavior() -> None:
    with collect_runtime_events(_FailingSink()):
        emit_runtime_event("ignored", payload={"value": object()})
        activity = Activity("a1", "goal", {})
        activity.state = ActivityState.TERMINATED
    assert activity.state is ActivityState.TERMINATED


# --- property snapshot: an unchanged re-observation must not repeat the value -----------------
#
# The trajectory these events land in reached 4.31 GiB (and on another run 7.85 GiB) for a single
# scenario, 99.9% of it byte-identical repeats of a value already recorded, held in memory for the
# length of the run. The elision has to stay lossless in both directions, which is what the first
# two tests pin: an unchanged property still produces an event (so the per-cycle attended set is
# still readable), and a value that actually moved is still recorded in full.


def _wm_with(tool: FakeTool) -> WorkingMemory:
    wm = WorkingMemory(registry=EnvironmentRegistry())
    wm.focused_tools[tool.id] = cast(Any, tool)
    return wm


def _property_rows(collector: RuntimeEventCollector) -> list[dict[str, Any]]:
    return [row for row in collector.snapshot() if row["event"].startswith("boundary.property.")]


def test_an_unchanged_property_is_recorded_without_repeating_its_value() -> None:
    tool = FakeTool("app", properties=[ObservableProperty("state", {"items": [1, 2, 3]})])
    wm = _wm_with(tool)
    collector = RuntimeEventCollector()

    with collect_runtime_events(collector):
        DefaultObserveStrategy._snapshot_properties(wm)  # first sight
        DefaultObserveStrategy._snapshot_properties(wm)  # unchanged
        DefaultObserveStrategy._snapshot_properties(wm)  # unchanged

    rows = _property_rows(collector)
    # One event per observation either way — a reader shown only changes could not tell an
    # unchanged property from a tool that stopped being attended.
    assert [row["event"] for row in rows] == [
        "boundary.property.received",
        "boundary.property.unchanged",
        "boundary.property.unchanged",
    ]
    assert rows[0]["payload"]["property"]["value"] == {"items": [1, 2, 3]}
    # The whole point: the repeats carry the key and nothing else.
    assert rows[1]["payload"] == {"source": "app", "name": "state"}
    assert "value" not in json.dumps(rows[2]["payload"])


def test_a_changed_property_value_is_still_recorded_in_full() -> None:
    # The regression that would make the bundle useless in the other direction.
    tool = FakeTool("app", properties=[ObservableProperty("state", {"items": [1]})])
    wm = _wm_with(tool)
    collector = RuntimeEventCollector()

    with collect_runtime_events(collector):
        DefaultObserveStrategy._snapshot_properties(wm)
        tool.set_properties([ObservableProperty("state", {"items": [1, 2]})])
        DefaultObserveStrategy._snapshot_properties(wm)
        DefaultObserveStrategy._snapshot_properties(wm)  # the new value is now the baseline

    rows = _property_rows(collector)
    assert [row["event"] for row in rows] == [
        "boundary.property.received",
        "boundary.property.received",
        "boundary.property.unchanged",
    ]
    assert rows[1]["payload"]["property"]["value"] == {"items": [1, 2]}


def test_the_snapshot_store_is_written_on_every_tick_even_when_the_event_is_elided() -> None:
    # The elision is a diagnostics decision only. `observed_at` is refreshed on every
    # re-observation and the reconsideration gate relies on that (it hashes the payload rather than
    # the envelope precisely because of it), so the store write must stay unconditional.
    tool = FakeTool("app", properties=[ObservableProperty("state", 1)])
    wm = _wm_with(tool)
    collector = RuntimeEventCollector()

    with collect_runtime_events(collector):
        DefaultObserveStrategy._snapshot_properties(wm)
        first = wm.properties[("app", "state")]
        DefaultObserveStrategy._snapshot_properties(wm)
        second = wm.properties[("app", "state")]

    assert second is not first  # re-stored, not skipped
    assert second.observed_at >= first.observed_at
    assert second.payload == ObservableProperty("state", 1)


def test_an_uncomparable_property_value_is_recorded_rather_than_elided() -> None:
    # A property value is adapter data of arbitrary shape, and the comparison runs *outside*
    # emit_runtime_event, which is the thing that swallows failures so a diagnostic can never alter
    # control flow. A numpy array inside a value is the real-world shape of this: the dataclass's
    # tuple compare raises on its truthiness. The fallback has to be "changed" — a redundant record
    # is recoverable, a missing one is not — and it must not propagate.
    class _Uncomparable:
        def __eq__(self, other: object) -> bool:
            raise ValueError("ambiguous truth value")

        __hash__ = None  # type: ignore[assignment]

    tool = FakeTool("app", properties=[ObservableProperty("state", _Uncomparable())])
    wm = _wm_with(tool)
    collector = RuntimeEventCollector()

    with collect_runtime_events(collector):
        DefaultObserveStrategy._snapshot_properties(wm)
        # A *distinct* instance, which is what a real adapter yields: ARE rebuilds the whole
        # value tree through `_to_serializable` on every `observe()` (its own
        # `changed = state != previous` depends on that), so the comparison is a genuine deep one
        # and actually reaches `__eq__`. Re-observing the identical object instead would be
        # answered by the tuple compare's identity shortcut before `__eq__` ran, testing nothing.
        tool.set_properties([ObservableProperty("state", _Uncomparable())])
        DefaultObserveStrategy._snapshot_properties(wm)

    assert [row["event"] for row in _property_rows(collector)] == [
        "boundary.property.received",
        "boundary.property.received",
    ]


def test_the_snapshot_still_updates_with_no_diagnostic_sink_installed() -> None:
    # No sink means the comparison is skipped entirely, so an undiagnosed run pays nothing for the
    # dedup. Perception must be identical regardless.
    tool = FakeTool("app", properties=[ObservableProperty("state", 1)])
    wm = _wm_with(tool)

    DefaultObserveStrategy._snapshot_properties(wm)
    tool.set_properties([ObservableProperty("state", 2)])
    DefaultObserveStrategy._snapshot_properties(wm)

    assert wm.properties[("app", "state")].payload == ObservableProperty("state", 2)
