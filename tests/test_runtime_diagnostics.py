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
    _stream_contexts,
    _stream_ledgers,
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


class _FlakySink:
    """Drops its first ``failures`` rows on the floor, then records normally."""

    def __init__(self, failures: int) -> None:
        self._remaining = failures
        self.events: list[dict[str, Any]] = []

    def emit(self, event: dict[str, Any]) -> None:
        if self._remaining > 0:
            self._remaining -= 1
            raise RuntimeError("sink unavailable")
        self.events.append(event)


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


def _wm_with(*tools: FakeTool) -> WorkingMemory:
    wm = WorkingMemory(registry=EnvironmentRegistry())
    for tool in tools:
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


def test_a_collector_armed_mid_run_receives_the_value_before_any_marker() -> None:
    """The paid harness arms one collector per attempt against an agent that has already been
    observing, so this stream's first sight of a property is a re-observation of an unchanged one.
    Asking only the keyed store would elide it and leave the marker deferring to a `received` row
    that exists in no stream at all — the unreadable bundle the elision exists to avoid."""
    tool = FakeTool("app", properties=[ObservableProperty("state", {"items": [1]})])
    wm = _wm_with(tool)

    DefaultObserveStrategy._snapshot_properties(wm)  # observed before anything was listening

    collector = RuntimeEventCollector()
    with collect_runtime_events(collector):
        DefaultObserveStrategy._snapshot_properties(wm)  # unchanged, but new to this stream
        DefaultObserveStrategy._snapshot_properties(wm)  # now it may be elided

    rows = _property_rows(collector)
    assert [row["event"] for row in rows] == [
        "boundary.property.received",
        "boundary.property.unchanged",
    ]
    assert rows[0]["payload"]["property"]["value"] == {"items": [1]}


def test_each_streams_first_observation_carries_its_own_value() -> None:
    """Two attempts over one long-lived agent. The second collector is a second reader and is owed
    the same seed, which a ledger kept on working memory or on the runtime would not give it."""
    tool = FakeTool("app", properties=[ObservableProperty("state", 1)])
    wm = _wm_with(tool)

    first = RuntimeEventCollector()
    with collect_runtime_events(first):
        DefaultObserveStrategy._snapshot_properties(wm)
        DefaultObserveStrategy._snapshot_properties(wm)
    second = RuntimeEventCollector()
    with collect_runtime_events(second):
        DefaultObserveStrategy._snapshot_properties(wm)

    assert [row["event"] for row in _property_rows(first)] == [
        "boundary.property.received",
        "boundary.property.unchanged",
    ]
    assert [row["event"] for row in _property_rows(second)] == ["boundary.property.received"]


def test_no_marker_in_a_stream_lacks_an_earlier_value_for_its_key() -> None:
    """The invariant the whole elision rests on, stated directly: every name-only marker has an
    earlier full value for the same key *in the same stream*. Over several tools and cycles, with a
    value moving partway through and history the stream never saw, since the defect only shows where
    the store's history and the stream's beginning disagree."""
    app = FakeTool("app", properties=[ObservableProperty("state", 1)])
    other = FakeTool("other", properties=[ObservableProperty("mode", "idle")])
    wm = _wm_with(app, other)
    DefaultObserveStrategy._snapshot_properties(wm)  # history no collector witnessed

    collector = RuntimeEventCollector()
    with collect_runtime_events(collector):
        for cycle in range(3):
            if cycle == 2:
                app.set_properties([ObservableProperty("state", 2)])
            DefaultObserveStrategy._snapshot_properties(wm)

    carried: set[tuple[str, str]] = set()
    for row in _property_rows(collector):
        payload = row["payload"]
        if row["event"] == "boundary.property.unchanged":
            assert (payload["source"], payload["name"]) in carried
        else:
            carried.add((payload["source"], payload["property"]["name"]))
    assert carried == {("app", "state"), ("other", "mode")}


def test_a_nested_collectors_observation_does_not_make_the_outer_stream_stale() -> None:
    """Two collectors over one agent, the inner armed inside the outer. The keyed store is shared
    between them, so a value observed only while the inner stream was installed moves the store
    ahead of what the outer reader was ever told. Comparing against the store would then call the
    outer stream's next observation unchanged and send its reader back to the superseded value."""
    tool = FakeTool("app", properties=[ObservableProperty("state", 1)])
    wm = _wm_with(tool)

    outer = RuntimeEventCollector()
    inner = RuntimeEventCollector()
    with collect_runtime_events(outer):
        DefaultObserveStrategy._snapshot_properties(wm)  # the outer reader is told 1
        tool.set_properties([ObservableProperty("state", 2)])
        with collect_runtime_events(inner):
            DefaultObserveStrategy._snapshot_properties(wm)  # only the inner reader is told 2
        DefaultObserveStrategy._snapshot_properties(wm)  # unchanged in the store, news to outer

    outer_rows = _property_rows(outer)
    assert [row["event"] for row in outer_rows] == [
        "boundary.property.received",
        "boundary.property.received",
    ]
    assert [row["payload"]["property"]["value"] for row in outer_rows] == [1, 2]
    # And the inner stream is owed its own seed rather than inheriting the outer one's.
    assert [row["event"] for row in _property_rows(inner)] == ["boundary.property.received"]


def test_a_value_the_sink_never_took_is_re_emitted_rather_than_elided() -> None:
    """`emit_runtime_event` swallows a sink failure by design, so a dropped row is invisible to the
    runtime — but it must not be invisible to the ledger. If the one row carrying the value never
    landed, every later re-observation is unchanged, and a ledger seeded regardless would elide all
    of them against a row no reader can see. A sink that recovers would not repair that."""
    sink = _FlakySink(failures=1)
    tool = FakeTool("app", properties=[ObservableProperty("state", {"items": [1]})])
    wm = _wm_with(tool)

    with collect_runtime_events(sink):
        DefaultObserveStrategy._snapshot_properties(wm)  # the value is dropped on the floor
        DefaultObserveStrategy._snapshot_properties(wm)  # unchanged, but this stream is still owed
        DefaultObserveStrategy._snapshot_properties(wm)  # now it may be elided

    rows = [row for row in sink.events if row["event"].startswith("boundary.property.")]
    assert [row["event"] for row in rows] == [
        "boundary.property.received",
        "boundary.property.unchanged",
    ]
    assert rows[0]["payload"]["property"]["value"] == {"items": [1]}


def test_a_reader_reconstructs_every_observed_value_from_its_own_stream() -> None:
    """The guarantee the elision owes a bundle reader, stated as the reader's own procedure: walk
    the rows carrying the last full value forward across the markers, and recover exactly what the
    agent observed on each of this stream's cycles. Run against the case that breaks a key-only
    ledger — a nested collector moving the shared store between two of the outer stream's own
    observations — since that is where the store and the stream disagree about "unchanged"."""
    tool = FakeTool("app", properties=[ObservableProperty("state", 1)])
    wm = _wm_with(tool)

    outer = RuntimeEventCollector()
    with collect_runtime_events(outer):
        DefaultObserveStrategy._snapshot_properties(wm)
        tool.set_properties([ObservableProperty("state", 2)])
        with collect_runtime_events(RuntimeEventCollector()):
            DefaultObserveStrategy._snapshot_properties(wm)
        DefaultObserveStrategy._snapshot_properties(wm)
        DefaultObserveStrategy._snapshot_properties(wm)
        tool.set_properties([ObservableProperty("state", 3)])
        DefaultObserveStrategy._snapshot_properties(wm)

    carried: dict[tuple[str, str], Any] = {}
    reconstructed: list[Any] = []
    for row in _property_rows(outer):
        payload = row["payload"]
        if row["event"] == "boundary.property.unchanged":
            key = (payload["source"], payload["name"])
            assert key in carried, "a marker with no value earlier in this stream"
        else:
            key = (payload["source"], payload["property"]["name"])
            carried[key] = payload["property"]["value"]
        reconstructed.append(carried[key])

    # The four observations this stream witnessed, in order. The nested collector's is not among
    # them, and the value it moved must not be attributed to the outer reader for free.
    assert reconstructed == [1, 2, 2, 3]


def test_contexts_sharing_a_sink_share_one_view_of_what_it_was_sent() -> None:
    """Nesting `collect_runtime_events` on one sink — or entering it from two tasks at once — makes
    several contexts append to a single log. A ledger owned by the context rather than the sink
    would then describe a stream it does not own: the inner context's value *is* in the log the
    outer context's reader walks, so an outer ledger that had not seen it would call the next
    observation unchanged and hand that reader a value the stream has already superseded."""
    tool = FakeTool("app", properties=[ObservableProperty("state", 1)])
    wm = _wm_with(tool)
    collector = RuntimeEventCollector()

    with collect_runtime_events(collector):
        DefaultObserveStrategy._snapshot_properties(wm)
        tool.set_properties([ObservableProperty("state", 2)])
        with collect_runtime_events(collector):  # same sink, so the same stream
            DefaultObserveStrategy._snapshot_properties(wm)
        tool.set_properties([ObservableProperty("state", 1)])
        DefaultObserveStrategy._snapshot_properties(wm)  # back to 1, which this log last saw as 2

    rows = _property_rows(collector)
    assert [row["event"] for row in rows] == ["boundary.property.received"] * 3
    # What a reader carrying values forward over this one log recovers. The third observation is
    # equal to what the *outer context* last emitted, and unequal to what the *stream* holds; the
    # stream is what a marker would defer to, so the value has to be restated.
    assert [row["payload"]["property"]["value"] for row in rows] == [1, 2, 1]


def test_a_sinks_ledger_does_not_outlive_the_contexts_open_on_it() -> None:
    """Owning the ledger by sink identity is only safe if the entry goes away with the last context
    on it: a surviving entry keyed on a recycled id would seed an unrelated stream with values it
    never received, and entries that accumulated would be a leak on a run that arms one collector
    per attempt. Nesting therefore refcounts rather than replacing or discarding."""
    collector = RuntimeEventCollector()
    key = id(collector)
    assert key not in _stream_ledgers

    with collect_runtime_events(collector):
        assert _stream_contexts[key] == 1
        ledger = _stream_ledgers[key]
        with collect_runtime_events(collector):
            assert _stream_contexts[key] == 2
            assert _stream_ledgers[key] is ledger  # shared, not a second one
        assert _stream_contexts[key] == 1
        assert _stream_ledgers[key] is ledger  # kept, not discarded by the inner exit

    assert key not in _stream_ledgers
    assert key not in _stream_contexts


def test_stream_yields_exactly_what_snapshot_returns() -> None:
    collector = RuntimeEventCollector()
    with collect_runtime_events(collector):
        for index in range(4):
            emit_runtime_event("phase.entry", payload={"n": index})
    assert list(collector.stream()) == list(collector.snapshot())


def test_stream_reads_each_row_only_when_it_is_reached() -> None:
    """`stream` exists so the export holds one row rather than the whole log, and that rests on the
    per-row read being deferred — not merely on consumption being lazy. `iter(snapshot())` would
    satisfy every other assertion here while copying everything up front, so this probes the
    deferral itself: a row reached after the walk started is read at that point, not captured when
    `stream` was called. The mutation is the probe, not a supported semantic — nothing in the
    runtime rewrites a committed row."""
    collector = RuntimeEventCollector()
    with collect_runtime_events(collector):
        for index in range(3):
            emit_runtime_event("phase.entry", payload={"n": index})

    rows = collector.stream()
    assert next(rows)["payload"]["n"] == 0
    collector._events[2]["event"] = "read-after-the-walk-began"
    assert [row["event"] for row in rows] == ["phase.entry", "read-after-the-walk-began"]


def test_stream_walks_a_stable_prefix_while_emission_continues() -> None:
    """A concurrent inference task can emit during the export walk. The event list is append-only,
    so the walk sees the prefix that existed when it started — the same view a snapshot taken at
    that moment would have given — rather than a partially-extended or shifting list."""
    collector = RuntimeEventCollector()
    with collect_runtime_events(collector):
        emit_runtime_event("phase.entry", payload={"n": 0})
        emit_runtime_event("phase.entry", payload={"n": 1})
        rows = collector.stream()
        first = next(rows)
        emit_runtime_event("phase.entry", payload={"n": 2})  # arrives mid-walk
        remaining = list(rows)

    assert first["payload"]["n"] == 0
    assert [row["payload"]["n"] for row in remaining] == [1]
    assert [row["payload"]["n"] for row in collector.snapshot()] == [0, 1, 2]
