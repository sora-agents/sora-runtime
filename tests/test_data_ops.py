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
import logging
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

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
    _MISSING,
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReasonStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    Strategies,
    TickResult,
    _as_collection,
    _latest_result,
    _resolve_collection,
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


async def test_filter_unwraps_a_single_key_envelope_return(tmp_path: Path) -> None:
    # A tool that wraps its {id -> record} payload under a lone key ({"apartments": {...}}) is
    # unwrapped deterministically before the predicate runs — the plan author writes no reshaping
    # step. Proves the envelope tier flows through _resolve_collection -> the live filter op.
    enveloped = {
        "apartments": {
            "a1": {"apartment_id": "a1", "crime": 3},
            "a2": {"apartment_id": "a2", "crime": 6},
            "a3": {"apartment_id": "a3", "crime": 9},
        }
    }
    step = Step(
        next_action="filter",
        params={
            "in": {"$from": "search_apartments", "path": ""},
            "out": "qualifying",
            "where": {"path": "crime", "op": "between", "value": [5, 10]},
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [_history("search_apartments", enveloped)])
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
        ("not_in", [1, 3], [2]),
    ]:
        step = Step(
            next_action="filter",
            params={"in": data, "out": "o", "where": {"path": "n", "op": op, "value": value}},
        )
        activity = await _run_one_dataop(tmp_path, step, [])
        assert [e["n"] for e in activity.bindings["o"]] == expected, op


# --------------------------------------------------------------------------------------------------
# filter — cross-collection membership: a reference-valued in/not_in predicate (ADR-0023 extension)
# --------------------------------------------------------------------------------------------------


async def test_filter_not_in_another_collection_by_reference(tmp_path: Path) -> None:
    # "Keep apartments NOT already saved": the membership set is another collection named by a
    # reference ($from the saved list, an {id -> record} map), projected by value_path to its ids.
    # Resolved once in Reason, so _matches stays a literal comparison.
    candidates = [{"id": "a1"}, {"id": "a2"}, {"id": "a3"}]
    saved = {"a2": {"apartment_id": "a2"}, "a3": {"apartment_id": "a3"}}
    step = Step(
        next_action="filter",
        params={
            "in": candidates,
            "out": "fresh",
            "where": {
                "path": "id",
                "op": "not_in",
                "value": {"$from": "list_saved_apartments"},
                "value_path": "apartment_id",
            },
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [_history("list_saved_apartments", saved)])
    assert [e["id"] for e in activity.bindings["fresh"]] == ["a1"]


async def test_filter_in_another_collection_by_reference(tmp_path: Path) -> None:
    # The intersection direction: keep only candidates whose id IS in the referenced set.
    candidates = [{"id": "a1"}, {"id": "a2"}, {"id": "a3"}]
    saved = {"a2": {"apartment_id": "a2"}, "a3": {"apartment_id": "a3"}}
    step = Step(
        next_action="filter",
        params={
            "in": candidates,
            "out": "known",
            "where": {
                "path": "id",
                "op": "in",
                "value": {"$from": "list_saved_apartments"},
                "value_path": "apartment_id",
            },
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [_history("list_saved_apartments", saved)])
    assert [e["id"] for e in activity.bindings["known"]] == ["a2", "a3"]


async def test_filter_membership_reference_of_scalars_needs_no_value_path(tmp_path: Path) -> None:
    # When the referenced collection already resolves to a list of scalars (e.g. a prior op's
    # output binding), value_path is omitted — the elements are the keys themselves.
    candidates = [{"zip": "90210"}, {"zip": "10001"}, {"zip": "60601"}]
    step = Step(
        next_action="filter",
        params={
            "in": candidates,
            "out": "covered",
            "where": {"path": "zip", "op": "in", "value": {"$from": "known_zips"}},
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [_history("known_zips", ["90210", "60601"])])
    assert [e["zip"] for e in activity.bindings["covered"]] == ["90210", "60601"]


async def test_filter_membership_unresolvable_reference_replans(tmp_path: Path) -> None:
    """An unreadable membership set used to become an empty set, which fails *open* in both
    directions — `not_in` keeps every element, `in` keeps none — so the filter confidently does the
    wrong thing to the whole collection either way. Neither answer is knowable, so neither is
    given: the plan is dropped for the same reason the fan-out drops it."""
    candidates = [{"id": "a1"}, {"id": "a2"}]
    for op in ("not_in", "in"):
        step = Step(
            next_action="filter",
            params={
                "in": candidates,
                "out": "kept",
                "where": {"path": "id", "op": op, "value": {"$from": "never_ran"}},
            },
        )
        activity = await _run_one_dataop(tmp_path, step, [])

        assert "kept" not in activity.bindings, op
        assert activity.plan is None, op
        assert activity.replan_trail[-1] is not None, op


async def test_filter_membership_wrong_value_path_warns_before_failing_open(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The silent trap: the reference RESOLVES (records exist) but value_path names a field they
    # don't carry, so every member projects to None -> the membership set is effectively empty and
    # not_in fails open, silently re-keeping already-saved items. The projection guard must surface
    # this (an all-None projection, not just a dict/list one) rather than let it mis-filter unseen.
    candidates = [{"id": "a1"}, {"id": "a2"}]
    saved = {"a2": {"apartment_id": "a2"}}  # the id lives under apartment_id, not "id"
    step = Step(
        next_action="filter",
        params={
            "in": candidates,
            "out": "kept",
            "where": {
                "path": "id",
                "op": "not_in",
                "value": {"$from": "list_saved_apartments"},
                "value_path": "id",  # WRONG: should be apartment_id -> plucks None for every record
            },
        },
    )
    with caplog.at_level(logging.WARNING, logger="sora.strategies"):
        activity = await _run_one_dataop(tmp_path, step, [_history("list_saved_apartments", saved)])
    # Fails open (the documented never-raise contract) — but now it is no longer silent.
    assert [e["id"] for e in activity.bindings["kept"]] == ["a1", "a2"]
    assert any(
        "value_path" in r.getMessage() and "never match" in r.getMessage() for r in caplog.records
    )


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


def _history_with_params(
    operation_name: str, params: dict[str, object], result: object
) -> CompletedOperation:
    return CompletedOperation(
        OperationInvocation("realestate", operation_name, params),
        OperationAck(ok=True, result=result),
    )


async def test_collect_carries_the_fanout_input_key(tmp_path: Path) -> None:
    # A fanned-out get_crime_rate whose result does NOT echo the zip it was called for: collect
    # merges each call's input params into its result, so the zip_code the rate belongs to is
    # recoverable downstream (the correlation the crime->zip join needs). Empty-param history stays
    # untouched — see test_collect_gathers_history_results_by_operation_name.
    history = [
        _history_with_params("get_crime_rate", {"zip_code": "1000"}, {"violent_crime": 7}),
        _history_with_params("get_crime_rate", {"zip_code": "2000"}, {"violent_crime": 3}),
    ]
    step = Step(next_action="collect", params={"from": "get_crime_rate", "out": "rates"})
    activity = await _run_one_dataop(tmp_path, step, history)
    assert activity.bindings["rates"] == [
        {"zip_code": "1000", "violent_crime": 7},
        {"zip_code": "2000", "violent_crime": 3},
    ]


async def test_collect_result_wins_on_key_collision(tmp_path: Path) -> None:
    # If the tool echoes a key that is also an input param, the authoritative RETURN value wins —
    # the param only fills in keys the result doesn't already carry.
    history = [
        _history_with_params(
            "get_crime_rate", {"zip_code": "1000"}, {"zip_code": "1000-canonical", "rate": 7}
        )
    ]
    step = Step(next_action="collect", params={"from": "get_crime_rate", "out": "rates"})
    activity = await _run_one_dataop(tmp_path, step, history)
    assert activity.bindings["rates"] == [{"zip_code": "1000-canonical", "rate": 7}]


async def test_collect_wraps_a_non_dict_result_with_params(tmp_path: Path) -> None:
    # A scalar result can't be merged, so it is wrapped as {**params, "result": <value>} — the input
    # key stays reachable (value_path) and the value lives under "result" (path).
    history = [_history_with_params("get_crime_rate", {"zip_code": "1000"}, 7)]
    step = Step(next_action="collect", params={"from": "get_crime_rate", "out": "rates"})
    activity = await _run_one_dataop(tmp_path, step, history)
    assert activity.bindings["rates"] == [{"zip_code": "1000", "result": 7}]


async def test_collect_then_mechanical_crime_pipeline(tmp_path: Path) -> None:
    # The end-to-end shape the Gaia2 run needed and couldn't express: a fanned-out get_crime_rate
    # whose result doesn't echo its zip. collect carries the zip via input params; a mechanical
    # `between` keeps the qualifying rates; a reference-valued membership `in` (value_path zip_code)
    # then joins the apartments back onto the qualifying zips — all mechanical, no $decide.
    apartments = [
        {"apartment_id": "a1", "zip_code": "1000"},
        {"apartment_id": "a2", "zip_code": "2000"},
        {"apartment_id": "a3", "zip_code": "3000"},
    ]
    history = [
        _history("list_all_apartments", apartments),
        _history_with_params("get_crime_rate", {"zip_code": "1000"}, {"violent_crime": 7}),
        _history_with_params("get_crime_rate", {"zip_code": "2000"}, {"violent_crime": 3}),
        _history_with_params("get_crime_rate", {"zip_code": "3000"}, {"violent_crime": 9}),
    ]
    steps = [
        Step(next_action="collect", params={"from": "get_crime_rate", "out": "crime"}),
        Step(
            next_action="filter",
            params={
                "in": {"$bind": "crime"},
                "out": "qualifying_crime",
                "where": {"path": "violent_crime", "op": "between", "value": [5, 10]},
            },
        ),
        Step(
            next_action="filter",
            params={
                "in": {"$from": "list_all_apartments", "path": ""},
                "out": "target_props",
                "where": {
                    "path": "zip_code",
                    "op": "in",
                    "value": {"$bind": "qualifying_crime"},
                    "value_path": "zip_code",
                },
            },
        ),
    ]
    tool = FakeTool("realestate")
    cycle, working, _ = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    activity = _activity_with_plan(steps, history)
    working.activities["a"] = activity
    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert result.step is None  # three mechanical data-ops, no external step
    # zips 1000 (7) and 3000 (9) qualify; 2000 (3) does not -> apartments a1 and a3.
    assert [p["apartment_id"] for p in activity.bindings["target_props"]] == ["a1", "a3"]


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


async def test_unresolvable_in_replans_rather_than_binding_nothing(tmp_path: Path) -> None:
    """`in` points at an op that never ran. Writing an empty binding would be worse than crashing:
    downstream, an empty binding reads as a *finding* — "no such contact" — and a real run reported
    exactly that to the user off a reference that had simply never resolved."""
    step = Step(
        next_action="filter",
        params={
            "in": {"$from": "never_ran", "path": ""},
            "out": "o",
            "where": {"path": "n", "op": "gt", "value": 0},
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [])

    assert "o" not in activity.bindings  # no fabricated empty answer
    assert activity.plan is None
    defect = activity.replan_trail[-1]
    assert defect is not None and "filter" in defect


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
    # empty collection. But a single record, an {id -> scalar} map, and a scalar are NOT
    # collections — coercing them fans out over garbage.
    assert _as_collection([{"id": "a1"}, {"id": "a2"}]) == [{"id": "a1"}, {"id": "a2"}]
    assert _as_collection({"a1": {"crime": 3}, "a2": {"crime": 6}}) == [{"crime": 3}, {"crime": 6}]
    assert _as_collection({}) == []
    assert _as_collection({"apartment_id": "a1", "crime": 6}) is None  # a single record's fields
    assert _as_collection({"90210": 7, "10001": 3}) is None  # id -> scalar (lossy) -> refuse
    assert _as_collection("not-a-collection") is None
    assert _as_collection(5) is None


def test_as_collection_takes_the_payload_out_of_a_paginated_envelope() -> None:
    # The shape ARE's windowed list operations return. One list-valued key, every sibling a
    # pagination-metadata scalar -> take the list.
    events = {"events": [{"id": "e1"}], "range": "(0, 1)", "total": 1}
    assert _as_collection(events) == [{"id": "e1"}]
    assert _as_collection({"results": [1, 2], "count": 2}) == [1, 2]
    assert _as_collection({"contacts": [], "range": "(0, 0)", "total": 0}) == []
    # The vocabulary is the whole safeguard: an identically-shaped RECORD is refused, because
    # fanning out over one event's attendees instead of over events is worse than not reading it.
    record = {"event_id": "e1", "title": "Standup", "attendees": [{"id": "a1"}]}
    assert _as_collection(record) is None
    # Mixed evidence stays refused: a non-metadata sibling, a dict sibling, or a second list all
    # mean the payload is not the one obvious thing in the dict.
    assert _as_collection({"events": [1], "notes": "hi"}) is None
    assert _as_collection({"events": [1], "total": 1, "page_info": {"n": 1}}) is None
    assert _as_collection({"events": [1], "errors": [], "total": 1}) is None


def test_as_collection_unwraps_a_single_key_envelope() -> None:
    # A lone key wrapping the payload is unwrapped and recursed into: {"apartments": {id -> rec}}
    # yields the records, {"results": [...]} yields the list, even nested one level deeper.
    assert _as_collection({"apartments": {"a1": {"crime": 3}, "a2": {"crime": 6}}}) == [
        {"crime": 3},
        {"crime": 6},
    ]
    assert _as_collection({"results": [{"id": "a1"}, {"id": "a2"}]}) == [{"id": "a1"}, {"id": "a2"}]
    assert _as_collection({"data": {"results": [1, 2, 3]}}) == [1, 2, 3]  # nested envelope
    # A single-element {id -> record} map is NOT an envelope: the lone record's fields are scalars,
    # so it falls through to the id->record tier and iterates (yields the one record), not unwraps.
    assert _as_collection({"a1": {"crime": 6, "rent": 1200}}) == [{"crime": 6, "rent": 1200}]
    # A lone key whose value is itself a scalar/non-collection is refused, not unwrapped.
    assert _as_collection({"count": 5}) is None


def test_as_collection_all_mapping_field_record_is_the_accepted_undecidable_residual() -> None:
    # DOCUMENTED, undecidable residual (ADR-0023): a single-entry {id -> record} map whose lone
    # record's fields are ALL mapping-valued is structurally identical to a single-key envelope
    # wrapping an {id -> record} map, so it is (wrongly) unwrapped into the record's field-VALUES.
    # This is NOT limited to a one-field record — a many-field all-dict record misfires the same
    # way. Locked here so the behavior can't drift silently; the principled fix is the deferred
    # model-escalated extraction, not a shape heuristic (any heuristic only moves the misfire).
    assert _as_collection({"a1": {"loc": {"lat": 1}, "meta": {"x": 2}}}) == [{"lat": 1}, {"x": 2}]
    # The mirror it is genuinely indistinguishable from: a real envelope wrapping a 2-record map.
    assert _as_collection({"apts": {"a1": {"p": 1}, "a2": {"p": 2}}}) == [{"p": 1}, {"p": 2}]
    # The safe boundary: one scalar field in the record makes the recursion refuse, so it is kept
    # as a single record (tier 3) — the shape ARE actually returns, hence why the misfire is inert.
    one_record = {"a1": {"loc": {"lat": 1}, "price": 2}}
    assert _as_collection(one_record) == [{"loc": {"lat": 1}, "price": 2}]


def test_resolve_collection_separates_empty_from_unreadable() -> None:
    """The distinction the caller replans on. "Nothing to do" and "I could not read this" both used
    to come back as an empty collection, which is how a sub-goal over a bad reference fanned out to
    zero steps and the plan continued as though the work were done."""
    # (1) an unresolved reference — the $from op never ran. The defect names what *has* run, so the
    #     planner gets a correction rather than a complaint.
    collection, defect = _resolve_collection({"$from": "search"}, [], {})
    assert collection is None
    assert defect is not None and "no operation has run" in defect

    # (2) a reference resolving to a value of a shape these tiers refuse — the type is named.
    collection, defect = _resolve_collection({"$bind": "x"}, [], {"x": "just-a-string"})
    assert collection is None
    assert defect is not None and "str" in defect

    # (3) a multi-key dict that is NOT a paginated envelope (the sibling is a record field, not
    #     pagination metadata). The defect must name the fix, since "add a path" is the only thing
    #     that makes a retry differ from the plan just abandoned.
    events = {"events": [{"id": "e1"}], "notes": "unfiled"}
    collection, defect = _resolve_collection({"$bind": "x"}, [], {"x": events})
    assert collection is None
    assert defect is not None and "'events'" in defect and "path" in defect

    # A genuinely empty collection is an answer, not a defect — this is the case that must NOT
    # replan, or every legitimately-empty fan-out would burn a planning inference.
    assert _resolve_collection({}, [], {}) == ([], None)
    assert _resolve_collection([], [], {}) == ([], None)


def test_a_bad_path_on_a_present_source_is_not_reported_as_a_missing_source() -> None:
    """The two failures need different repairs, so they must not share a defect string.

    Collapsing them told the planner that ``search_events`` had produced no result while naming
    ``search_events`` among the operations that had run — a self-contradictory brief, aimed at a
    step that did not need to run again. The planner wrote the same reference, the trail saw the
    same defect twice, and the activity halted on a question the user could not answer."""
    history = [_ran("Calendar", "search_events", [{"id": "e1"}])]
    ref = {"$from": "search_events", "path": "events"}

    collection, defect = _resolve_collection(ref, history, {})

    assert collection is None
    assert defect is not None
    assert "names no result the plan has produced" not in defect
    assert "IS present" in defect and "does not need to run again" in defect
    assert "'events'" in defect  # the segment that did not fit
    assert "list of 1 item(s)" in defect  # ...and what is actually there instead


def test_a_bad_path_names_the_keys_the_result_does_have() -> None:
    """Naming the alternatives is what makes the retry differ, same as the undeclared-param defect.
    The failing segment is reported where it breaks, not at the head of the path."""
    history = [_ran("Calendar", "search_events", {"events": [{"id": "e1"}]})]
    ref = {"$from": "search_events", "path": "events.0.title"}

    _collection, defect = _resolve_collection(ref, history, {})

    assert defect is not None
    assert "'title'" in defect
    assert "events.0" in defect and "'id'" in defect


def test_a_bad_path_on_a_present_binding_talks_about_the_binding_not_operations() -> None:
    """A $bind defect used to be phrased entirely in terms of operations run so far — advice about
    the wrong half of the plan, since no operation produces a binding."""
    _collection, defect = _resolve_collection({"$bind": "x", "path": "nope"}, [], {"x": {"a": 1}})

    assert defect is not None
    assert "operations run so far" not in defect
    assert "'nope'" in defect and "'a'" in defect


def _ran(tool_id: str, operation_name: str, result: object) -> CompletedOperation:
    return CompletedOperation(
        OperationInvocation(tool_id, operation_name, {}), OperationAck(ok=True, result=result)
    )


def test_a_from_reference_resolves_bare_or_qualified() -> None:
    """A planner reading a catalog that addresses every operation as ``tool_id.operation_name``
    writes references that way too. Refusing that spelling resolved to nothing, and at a fan-out
    nothing meant zero steps — a plan lost to a naming convention the runtime itself taught."""
    history = [_ran("insim:are/Contacts", "get_contacts", {"contacts": [{"id": "c1"}]})]

    expected = {"contacts": [{"id": "c1"}]}
    assert _latest_result(history, "get_contacts") == expected
    assert _latest_result(history, "insim:are/Contacts.get_contacts") == expected
    # A qualification whose prefix matches no tool invoked still names the operation unambiguously
    # (an operation name never contains a dot), so the tail is honored rather than dropped.
    assert _latest_result(history, "Contacts.get_contacts") == expected

    assert _latest_result(history, "never_ran") is _MISSING


def test_a_qualified_reference_picks_the_tool_it_names() -> None:
    """Where the qualified form earns its precedence: two joined workspaces exposing the same
    operation (ARE's Contacts and InternalContacts both have ``get_contacts``). The bare name can
    only mean "most recent"; the qualified one is a genuine disambiguation and must be honored as
    such, not collapsed to the tail."""
    history = [
        _ran("insim:are/Contacts", "get_contacts", "external"),
        _ran("insim:are/InternalContacts", "get_contacts", "internal"),
    ]

    assert _latest_result(history, "insim:are/Contacts.get_contacts") == "external"
    assert _latest_result(history, "insim:are/InternalContacts.get_contacts") == "internal"
    assert _latest_result(history, "get_contacts") == "internal"  # bare -> most recent


def test_a_decide_collection_is_soft_not_a_defect() -> None:
    # Resolved off-cycle by the model, so there is nothing to read yet and nothing to blame.
    assert _resolve_collection({"$decide": "the interesting ones"}, [], {}) == (None, None)


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
