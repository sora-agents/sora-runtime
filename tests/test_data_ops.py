"""Structured-value data-ops — the plan's composable data-processing layer (ADR-0023).

A data-op is a ``Step`` whose ``next_action`` names a registered data-op (``filter``, ``distinct``,
``sort``, ``take``, ``collect``, ``reduce``). Reason dispatches it from ``ActionRegistry``'s
dedicated data-op bucket, transforming a run-time collection (read from history via ``$from`` or a
prior binding via ``$bind``) and writing a **named binding** (``Activity.bindings[out]``) that a
later step reads via ``{"$bind": "<name>"}``. Mechanical ops run inline and advance the plan; a
``filter`` with a ``$decide`` predicate escalates to one off-cycle model call (``select``) whose
result lands in the binding a later cycle — mirroring ``_ground_`` (ADR-0021).

It closes the RentAFlat "save each *qualifying* apartment" gap — narrowing a collection before a
mechanical sub-goal fans out over it — that the documented-but-unwired ``where`` clause never
actually implemented.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from fakes import FakeAdapter, FakeLLMClient, FakeTool, FakeWorkspace
from sora.action import default_action_registry
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
from sora.perception import Message
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReasonStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    Strategies,
    TickResult,
    _as_collection,
)
from sora.transport import MessageTransport
from sora.types import (
    CompletedOperation,
    OperationAck,
    OperationInvocation,
    Plan,
    Step,
)

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")


def _history(operation_name: str, result: object, *, ok: bool = True) -> CompletedOperation:
    return CompletedOperation(
        OperationInvocation("realestate", operation_name, {}), OperationAck(ok=ok, result=result)
    )


class _NullTransport:
    async def send(self, to: str, content: dict[str, object]) -> None: ...

    def receive(self) -> AsyncIterator[Message]:
        async def _drain() -> AsyncIterator[Message]:
            return
            yield  # pragma: no cover

        return _drain()


def _cycle(
    tmp_path: Path, procedural: ProceduralMemory, tool: Tool
) -> tuple[DecisionCycle, WorkingMemory, EnvironmentRegistry]:
    registry = EnvironmentRegistry(
        adapters={_ORIGIN: FakeAdapter("fake", FakeWorkspace("ws", _ORIGIN, [tool]))}
    )
    working = WorkingMemory(registry=registry)
    transport: MessageTransport = _NullTransport()
    strategies = Strategies(
        observe=DefaultObserveStrategy(),
        reflect=DefaultReflectStrategy(),
        situate=DefaultSituateStrategy(),
        reason=DefaultReasonStrategy(),
        act=DefaultActStrategy(),
    )
    cycle = DecisionCycle(
        strategies=strategies,
        communication=transport,
        actions=default_action_registry(),
        registry=registry,
        working=working,
        semantic=SemanticMemory(FileMemoryBackend(tmp_path / "semantic")),
        procedural=procedural,
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "episodic")),
    )
    return cycle, working, registry


def _no_llm_procedural(tmp_path: Path) -> ProceduralMemory:
    # A mechanical data-op makes no model call; a procedural with no LLM proves it (it would raise
    # if the op ever escalated).
    return ProceduralMemory(FileMemoryBackend(tmp_path / "proc"))


def _activity_with_plan(steps: list[Step], history: list[CompletedOperation]) -> Activity:
    return Activity(
        id="a",
        goal="pipeline",
        context={},
        plan=Plan(id="p", goal="pipeline", steps=steps),
        step_index=0,
        history=history,
    )


async def _run_one_dataop(
    tmp_path: Path, step: Step, history: list[CompletedOperation]
) -> Activity:
    """Drive a single mechanical data-op step through reason(): it executes inline, writes its
    binding, advances past itself, and (plan exhausted) yields no step."""
    tool = FakeTool("realestate")
    cycle, working, _ = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    activity = _activity_with_plan([step], history)
    working.activities["a"] = activity
    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert result.step is None  # a lone data-op yields no external step
    return activity


# --------------------------------------------------------------------------------------------------
# filter — mechanical field-path predicate
# --------------------------------------------------------------------------------------------------

APARTMENTS = [
    {"id": "a1", "crime": 3},
    {"id": "a2", "crime": 6},
    {"id": "a3", "crime": 9},
    {"id": "a4", "crime": 12},
]


async def test_filter_between_keeps_matching_elements(tmp_path: Path) -> None:
    step = Step(
        next_action="filter",
        params={
            "in": {"$from": "search_apartments", "path": ""},
            "out": "qualifying",
            "where": {"path": "crime", "op": "between", "value": [5, 10]},
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [_history("search_apartments", APARTMENTS)])
    assert [e["id"] for e in activity.bindings["qualifying"]] == ["a2", "a3"]


async def test_filter_over_id_keyed_mapping_iterates_values(tmp_path: Path) -> None:
    # ARE's collection ops (list_all_apartments, search_apartments, list_saved_apartments) return an
    # {id -> record} mapping, not a bare list. A collection reference that resolves to a dict must
    # iterate its values (lossless here — each record carries its own id), not resolve to empty.
    mapping = {
        "a1": {"apartment_id": "a1", "crime": 3},
        "a2": {"apartment_id": "a2", "crime": 6},
        "a3": {"apartment_id": "a3", "crime": 9},
    }
    step = Step(
        next_action="filter",
        params={
            "in": {"$from": "list_all_apartments", "path": ""},
            "out": "qualifying",
            "where": {"path": "crime", "op": "between", "value": [5, 10]},
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [_history("list_all_apartments", mapping)])
    assert [e["apartment_id"] for e in activity.bindings["qualifying"]] == ["a2", "a3"]


async def test_filter_comparison_ops(tmp_path: Path) -> None:
    data = [{"n": 1}, {"n": 2}, {"n": 3}]
    for op, value, expected in [
        ("eq", 2, [2]),
        ("ne", 2, [1, 3]),
        ("lt", 2, [1]),
        ("le", 2, [1, 2]),
        ("gt", 2, [3]),
        ("ge", 2, [2, 3]),
        ("in", [1, 3], [1, 3]),
    ]:
        step = Step(
            next_action="filter",
            params={"in": data, "out": "o", "where": {"path": "n", "op": op, "value": value}},
        )
        activity = await _run_one_dataop(tmp_path, step, [])
        assert [e["n"] for e in activity.bindings["o"]] == expected, op


# --------------------------------------------------------------------------------------------------
# filter — $decide predicate escalates to one off-cycle model call
# --------------------------------------------------------------------------------------------------


async def test_filter_decide_escalates_and_lands_in_binding(tmp_path: Path) -> None:
    llm = FakeLLMClient('{"keep": [0, 2]}')  # model keeps elements 0 and 2
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm)
    tool = FakeTool("realestate")
    cycle, working, _ = _cycle(tmp_path, procedural, tool)
    step = Step(
        next_action="filter",
        params={
            "in": {"$from": "search_apartments", "path": ""},
            "out": "qualifying",
            "where": {"$decide": "keep only the affordable ones near good schools"},
        },
    )
    activity = _activity_with_plan([step], [_history("search_apartments", APARTMENTS)])
    working.activities["a"] = activity

    # Reaching the $decide filter fires the off-cycle select: RUNNING, no step, binding not yet set.
    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert result.step is None
    state = activity.state
    assert state is ActivityState.RUNNING
    assert activity.pending_inference is not None
    assert activity.pending_inference.kind == "select"
    assert "qualifying" not in activity.bindings

    await asyncio.sleep(0)  # let the background select task push its result
    assert any("keep only the affordable" in prompt for _system, prompt in llm.calls)

    # Observe resolves the select into the named binding; the activity returns to READY.
    await DefaultObserveStrategy().observe(cycle)
    assert [e["id"] for e in activity.bindings["qualifying"]] == ["a1", "a3"]
    state = activity.state
    assert state is ActivityState.READY


# --------------------------------------------------------------------------------------------------
# distinct / sort / take / reduce
# --------------------------------------------------------------------------------------------------


async def test_distinct_dedupes_scalars(tmp_path: Path) -> None:
    step = Step(next_action="distinct", params={"in": ["x", "y", "x", "z", "y"], "out": "u"})
    activity = await _run_one_dataop(tmp_path, step, [])
    assert activity.bindings["u"] == ["x", "y", "z"]  # first-occurrence order preserved


async def test_distinct_by_key_path(tmp_path: Path) -> None:
    data = [{"zip": "1"}, {"zip": "2"}, {"zip": "1"}]
    step = Step(next_action="distinct", params={"in": data, "out": "u", "by": "zip"})
    activity = await _run_one_dataop(tmp_path, step, [])
    assert [e["zip"] for e in activity.bindings["u"]] == ["1", "2"]


async def test_sort_by_path_ascending_and_descending(tmp_path: Path) -> None:
    data = [{"n": 3}, {"n": 1}, {"n": 2}]
    asc = await _run_one_dataop(
        tmp_path, Step(next_action="sort", params={"in": data, "out": "s", "by": "n"}), []
    )
    assert [e["n"] for e in asc.bindings["s"]] == [1, 2, 3]
    desc = await _run_one_dataop(
        tmp_path,
        Step(next_action="sort", params={"in": data, "out": "s", "by": "n", "desc": True}),
        [],
    )
    assert [e["n"] for e in desc.bindings["s"]] == [3, 2, 1]


async def test_take_first_n(tmp_path: Path) -> None:
    step = Step(next_action="take", params={"in": [1, 2, 3, 4, 5], "out": "t", "n": 3})
    activity = await _run_one_dataop(tmp_path, step, [])
    assert activity.bindings["t"] == [1, 2, 3]


async def test_reduce_aggregations(tmp_path: Path) -> None:
    data = [{"v": 2}, {"v": 4}, {"v": 6}]
    for op, expected in [("sum", 12), ("min", 2), ("max", 6), ("mean", 4.0), ("count", 3)]:
        step = Step(next_action="reduce", params={"in": data, "out": "r", "op": op, "by": "v"})
        activity = await _run_one_dataop(tmp_path, step, [])
        assert activity.bindings["r"] == expected, op


# --------------------------------------------------------------------------------------------------
# collect — gather a fan-out's per-operation history results into one binding
# --------------------------------------------------------------------------------------------------


async def test_collect_gathers_history_results_by_operation_name(tmp_path: Path) -> None:
    history = [
        _history("get_crime_rate", {"zip": "1", "rate": 7}),
        _history("get_crime_rate", {"zip": "2", "rate": 3}),
        _history("get_crime_rate", {"zip": "3", "rate": 9}),
        _history("search", ["1", "2", "3"]),  # a different op — not collected
    ]
    step = Step(next_action="collect", params={"from": "get_crime_rate", "out": "rates"})
    activity = await _run_one_dataop(tmp_path, step, history)
    assert [r["rate"] for r in activity.bindings["rates"]] == [7, 3, 9]


# --------------------------------------------------------------------------------------------------
# downstream $bind — bindings feed later steps
# --------------------------------------------------------------------------------------------------


async def test_binding_feeds_a_mechanical_subgoal_in(tmp_path: Path) -> None:
    # filter narrows the collection to a binding; a mechanical sub-goal iterates the binding and
    # fans out one invoke per survivor — the RentAFlat "save each qualifying apartment" shape.
    tool = FakeTool("realestate", invoke_results={"save_apartment": {"saved": True}})
    cycle, working, registry = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    await registry.join(_ORIGIN)
    filter_step = Step(
        next_action="filter",
        params={
            "in": {"$from": "search_apartments", "path": ""},
            "out": "qualifying",
            "where": {"path": "crime", "op": "between", "value": [5, 10]},
        },
    )
    subgoal = Step(
        next_action="subgoal",
        params={
            "goal": "save each qualifying apartment",
            "mode": "mechanical",
            "in": {"$bind": "qualifying"},  # the binding the filter wrote, not a $from
            "as": "apt",
            "template": {
                "action": "invoke",
                "tool_id": "realestate",
                "operation_name": "save_apartment",
                "params": {"apartment_id": {"$bind": "apt", "path": "id"}},
            },
        },
    )
    activity = _activity_with_plan(
        [filter_step, subgoal], [_history("search_apartments", APARTMENTS)]
    )
    working.activities["a"] = activity

    saved: list[str] = []
    for _ in range(2):  # exactly two apartments qualify (crime 6 and 9)
        result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
        assert result.step is not None
        assert result.step.next_action == "invoke"
        saved.append(result.step.params["apartment_id"])
    assert saved == ["a2", "a3"]

    exhausted = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert exhausted.step is None  # count == len(survivors), no third invoke


async def test_binding_grounds_an_invoke_param(tmp_path: Path) -> None:
    tool = FakeTool("realestate", invoke_results={"email": {"ok": True}})
    cycle, working, registry = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    await registry.join(_ORIGIN)
    take_step = Step(
        next_action="take", params={"in": ["r@x.com", "s@x.com"], "out": "recips", "n": 1}
    )
    invoke = Step(
        next_action="invoke",
        params={
            "tool_id": "realestate",
            "operation_name": "email",
            "to": {"$bind": "recips", "path": "0"},
        },
    )
    activity = _activity_with_plan([take_step, invoke], [])
    working.activities["a"] = activity

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert result.step is not None
    assert result.step.params["to"] == "r@x.com"  # $bind read the binding, walked path "0"


# --------------------------------------------------------------------------------------------------
# edge cases
# --------------------------------------------------------------------------------------------------


async def test_unresolvable_in_writes_an_empty_binding(tmp_path: Path) -> None:
    # `in` points at an op that never ran: mirror the fan-out's never-raise contract — empty
    # binding, not a crash.
    step = Step(
        next_action="filter",
        params={
            "in": {"$from": "never_ran", "path": ""},
            "out": "o",
            "where": {"path": "n", "op": "gt", "value": 0},
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [])
    assert activity.bindings["o"] == []


async def test_reset_for_replan_clears_bindings(tmp_path: Path) -> None:
    activity = Activity(id="a", goal="g", context={})
    activity.bindings["stale"] = [1, 2, 3]
    activity.reset_for_replan()
    assert activity.bindings == {}


# --------------------------------------------------------------------------------------------------
# collection-shape coercion — a dict is a collection only when it's an {id -> record} map
# --------------------------------------------------------------------------------------------------


def test_as_collection_only_treats_an_id_record_map_as_a_collection() -> None:
    # A list is itself; an {id -> record} map iterates its values (lossless); an empty dict is an
    # empty collection. But a single record, a {"results": [...]} envelope, an {id -> scalar} map,
    # and a scalar are NOT collections — coercing them would silently fan out over field values.
    assert _as_collection([{"id": "a1"}, {"id": "a2"}]) == [{"id": "a1"}, {"id": "a2"}]
    assert _as_collection({"a1": {"crime": 3}, "a2": {"crime": 6}}) == [{"crime": 3}, {"crime": 6}]
    assert _as_collection({}) == []
    assert _as_collection({"apartment_id": "a1", "crime": 6}) is None  # a single record's fields
    assert _as_collection({"results": [1, 2], "count": 2}) is None  # an envelope, not id->record
    assert _as_collection({"90210": 7, "10001": 3}) is None  # id -> scalar (lossy) -> refuse
    assert _as_collection("not-a-collection") is None
    assert _as_collection(5) is None


async def test_filter_between_excludes_incomparable_value(tmp_path: Path) -> None:
    # A `between` predicate over a collection with a non-numeric field must exclude the bad element,
    # not raise (like the lt/le/gt/ge TypeError guard) — dirty tool output must not abort reason.
    data = [{"crime": 6}, {"crime": "unknown"}, {"crime": 8}]
    step = Step(
        next_action="filter",
        params={
            "in": data,
            "out": "o",
            "where": {"path": "crime", "op": "between", "value": [5, 10]},
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [])
    assert [e["crime"] for e in activity.bindings["o"]] == [6, 8]


async def test_filter_decide_dedupes_repeated_indices(tmp_path: Path) -> None:
    # A model that repeats an index must not make the filter emit a duplicate item (which would
    # double-act a downstream fan-out): {"keep": [0, 0, 2]} -> two distinct survivors, not three.
    llm = FakeLLMClient('{"keep": [0, 0, 2]}')
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm)
    tool = FakeTool("realestate")
    cycle, working, _ = _cycle(tmp_path, procedural, tool)
    step = Step(
        next_action="filter",
        params={
            "in": {"$from": "search_apartments", "path": ""},
            "out": "qualifying",
            "where": {"$decide": "any"},
        },
    )
    activity = _activity_with_plan([step], [_history("search_apartments", APARTMENTS)])
    working.activities["a"] = activity
    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    await asyncio.sleep(0)
    await DefaultObserveStrategy().observe(cycle)
    assert [e["id"] for e in activity.bindings["qualifying"]] == ["a1", "a3"]  # no duplicate a1


async def test_filter_decide_error_degrades_to_empty_binding(tmp_path: Path) -> None:
    # A transient model/parse failure on a $decide filter degrades to an empty shortlist and leaves
    # the activity READY — a data-op is a transform, it must not terminate the whole activity.
    llm = FakeLLMClient("not json at all")  # select's _parse_keep raises -> InferenceResult.error
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm)
    tool = FakeTool("realestate")
    cycle, working, _ = _cycle(tmp_path, procedural, tool)
    step = Step(
        next_action="filter",
        params={
            "in": {"$from": "search_apartments", "path": ""},
            "out": "qualifying",
            "where": {"$decide": "any"},
        },
    )
    activity = _activity_with_plan([step], [_history("search_apartments", APARTMENTS)])
    working.activities["a"] = activity
    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    await asyncio.sleep(0)
    await DefaultObserveStrategy().observe(cycle)
    assert activity.bindings["qualifying"] == []
    state = activity.state
    assert state is ActivityState.READY


async def test_reduce_sum_of_empty_is_none(tmp_path: Path) -> None:
    # sum over an empty collection is None (absence of a total), consistent with min/max/mean —
    # not 0, which a downstream branch could misread as a real zero total.
    step = Step(next_action="reduce", params={"in": [], "out": "r", "op": "sum", "by": "v"})
    activity = await _run_one_dataop(tmp_path, step, [])
    assert activity.bindings["r"] is None
