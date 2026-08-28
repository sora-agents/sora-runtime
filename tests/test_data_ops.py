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
    PerceptSnapshot,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from sora.perception import Message, Percept
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
    ObservableProperty,
    OperationAck,
    OperationInvocation,
    Plan,
    Step,
    UnresolvableGrounding,
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
# filter — a reference-valued predicate on a NON-membership op (a threshold, a computed range)
# --------------------------------------------------------------------------------------------------


async def test_filter_threshold_reads_a_reference_value(tmp_path: Path) -> None:
    # ADR-0023's own canonical pipeline: reduce to an aggregate, then keep what beats it. The
    # threshold is only known at run time, so it can only be written as a reference. Resolution
    # used to be gated on in/not_in, so this reference reached _matches as a raw dict, every
    # comparison raised TypeError, and the documented pipeline silently kept nothing.
    data = [{"v": 2}, {"v": 4}, {"v": 9}]
    steps = [
        Step(next_action="reduce", params={"in": data, "out": "mean_v", "op": "mean", "by": "v"}),
        Step(
            next_action="filter",
            params={
                "in": data,
                "out": "above",
                "where": {"path": "v", "op": "gt", "value": {"$bind": "mean_v"}},
            },
        ),
    ]
    tool = FakeTool("realestate")
    cycle, working, _ = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    activity = _activity_with_plan(steps, [])
    working.activities["a"] = activity
    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert activity.bindings["mean_v"] == 5.0
    assert [e["v"] for e in activity.bindings["above"]] == [9]  # not [] — the threshold resolved


async def test_filter_between_reads_a_reference_range(tmp_path: Path) -> None:
    # `between` takes a two-element value, so its reference resolves to a list without being
    # projected the way a membership set is — the pair IS the value, not a collection to key on.
    data = [{"v": 2}, {"v": 6}, {"v": 12}]
    step = Step(
        next_action="filter",
        params={
            "in": data,
            "out": "inrange",
            "where": {"path": "v", "op": "between", "value": {"$from": "get_band"}},
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [_history("get_band", [5, 10])])
    assert [e["v"] for e in activity.bindings["inrange"]] == [6]


async def test_filter_threshold_unresolvable_reference_replans(tmp_path: Path) -> None:
    """An unreadable threshold is the same hazard as an unreadable membership set: every ordered
    comparison against a raw reference dict raises TypeError, which _matches catches as a non-match,
    so the filter confidently keeps nothing. That answer isn't knowable, so it isn't given."""
    step = Step(
        next_action="filter",
        params={
            "in": [{"v": 2}, {"v": 9}],
            "out": "kept",
            "where": {"path": "v", "op": "gt", "value": {"$bind": "never_computed"}},
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [])

    assert "kept" not in activity.bindings
    assert activity.plan is None
    assert activity.replan_trail[-1] is not None


async def test_filter_decide_predicate_is_not_resolved_as_a_value(tmp_path: Path) -> None:
    # $decide is soft: FilterAction escalates the whole predicate to one model call. Widening
    # value resolution past in/not_in must not start treating it as an unresolvable hard reference
    # and replanning on it.
    llm = FakeLLMClient('{"keep": [1]}')
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm)
    tool = FakeTool("realestate")
    cycle, working, _ = _cycle(tmp_path, procedural, tool)
    step = Step(
        next_action="filter",
        params={"in": [{"v": 2}, {"v": 9}], "out": "kept", "where": {"$decide": "the big one"}},
    )
    activity = _activity_with_plan([step], [])
    working.activities["a"] = activity
    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert activity.plan is not None  # escalated, not replanned
    assert activity.state is ActivityState.RUNNING


async def test_filter_between_reads_a_reference_at_each_end_of_the_pair(tmp_path: Path) -> None:
    """Two bounds produced by two different steps can only be written as a pair OF references —
    no single earlier result holds the assembled [lo, hi]. A list isn't itself a reference, so
    both ends used to reach _matches as raw dicts, every comparison raised TypeError, and the
    filter kept nothing while reporting an ordinary empty result."""
    data = [{"v": 2}, {"v": 6}, {"v": 12}]
    step = Step(
        next_action="filter",
        params={
            "in": data,
            "out": "inrange",
            "where": {
                "path": "v",
                "op": "between",
                "value": [{"$from": "get_floor"}, {"$from": "get_ceiling"}],
            },
        },
    )
    activity = await _run_one_dataop(
        tmp_path, step, [_history("get_floor", 5), _history("get_ceiling", 10)]
    )
    assert [e["v"] for e in activity.bindings["inrange"]] == [6]


async def test_filter_between_with_one_unreadable_end_replans(tmp_path: Path) -> None:
    # A pair is only as comparable as its worse end: one unresolvable bound kills the whole
    # predicate exactly as two would, so it is a defect, not a half-usable range.
    step = Step(
        next_action="filter",
        params={
            "in": [{"v": 2}, {"v": 9}],
            "out": "inrange",
            "where": {
                "path": "v",
                "op": "between",
                "value": [{"$from": "get_floor"}, {"$bind": "never_computed"}],
            },
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [_history("get_floor", 5)])

    assert "inrange" not in activity.bindings
    assert activity.plan is None
    assert activity.superseded is not None
    assert "never_computed" in (activity.superseded.defect or "")


async def test_filter_threshold_resolving_to_a_record_is_a_defect_not_an_empty_result(
    tmp_path: Path,
) -> None:
    """Reading cleanly is not the same as being comparable. A reference that lands on the whole
    record instead of one of its fields — the commonest way to write this wrong — makes every
    ordered comparison raise TypeError inside _matches, where it is caught as a non-match by
    design. The filter then keeps nothing and the empty binding reads downstream as a fact about
    the world. Say which reference cannot compare instead."""
    step = Step(
        next_action="filter",
        params={
            "in": [{"v": 2}, {"v": 9}],
            "out": "above",
            "where": {"path": "v", "op": "gt", "value": {"$from": "get_budget"}},
        },
    )
    activity = await _run_one_dataop(
        tmp_path, step, [_history("get_budget", {"amount": 5, "currency": "EUR"})]
    )

    assert "above" not in activity.bindings
    assert activity.plan is None
    assert activity.superseded is not None
    defect = activity.superseded.defect or ""
    assert "get_budget" in defect
    assert "'amount'" in defect  # names what IS there, so the retry can reach for a path


async def test_filter_between_against_a_scalar_is_a_defect(tmp_path: Path) -> None:
    # `between` compares against a pair and treats anything else as a blanket non-match, so a
    # reference resolving to one bound is dead the same way an unreadable one is.
    step = Step(
        next_action="filter",
        params={
            "in": [{"v": 2}, {"v": 9}],
            "out": "inrange",
            "where": {"path": "v", "op": "between", "value": {"$from": "get_cap"}},
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [_history("get_cap", 5)])

    assert "inrange" not in activity.bindings
    assert activity.plan is None
    assert activity.superseded is not None
    assert "[lo, hi]" in (activity.superseded.defect or "")


async def test_filter_eq_against_a_resolved_none_still_runs(tmp_path: Path) -> None:
    """The shape check must not overreach: `eq` against a null is a legitimate predicate (keep the
    items whose field is unset), so an operand shape that CAN match is never a defect — only one
    that provably matches nothing is."""
    step = Step(
        next_action="filter",
        params={
            "in": [{"v": None}, {"v": 9}],
            "out": "unset",
            "where": {"path": "v", "op": "eq", "value": {"$from": "get_nothing"}},
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [_history("get_nothing", None)])

    assert activity.bindings["unset"] == [{"v": None}]
    assert activity.plan is not None


# --------------------------------------------------------------------------------------------------
# filter — boolean composition (all/any) and the `overlaps` range join
#
# Both exist to keep predicates OFF the $decide path. A real rule is rarely one clause, and before
# composition a single awkward clause dragged the whole predicate to a model call over every item.
# `overlaps` is the two-sided sibling of `between`: one item's range against a whole collection of
# them, resolved once in Reason like an `in` set, so no per-member alias reaches the evaluator.
# --------------------------------------------------------------------------------------------------

BOOKINGS = [
    {"id": "b1", "starts_at": "2026-05-01T09:00", "ends_at": "2026-05-01T10:00", "room": "blue"},
    {"id": "b2", "starts_at": "2026-05-01T10:00", "ends_at": "2026-05-01T11:00", "room": "blue"},
    {"id": "b3", "starts_at": "2026-05-01T10:30", "ends_at": "2026-05-01T11:30", "room": "red"},
    {"id": "b4", "starts_at": "2026-05-01T12:00", "ends_at": "2026-05-01T13:00", "room": "blue"},
]


async def test_filter_all_requires_every_clause(tmp_path: Path) -> None:
    step = Step(
        next_action="filter",
        params={
            "in": BOOKINGS,
            "out": "kept",
            "where": {
                "all": [
                    {"path": "room", "op": "eq", "value": "blue"},
                    {"path": "starts_at", "op": "ge", "value": "2026-05-01T10:00"},
                ]
            },
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [])
    assert [e["id"] for e in activity.bindings["kept"]] == ["b2", "b4"]


async def test_filter_any_requires_one_clause(tmp_path: Path) -> None:
    step = Step(
        next_action="filter",
        params={
            "in": BOOKINGS,
            "out": "kept",
            "where": {
                "any": [
                    {"path": "room", "op": "eq", "value": "red"},
                    {"path": "id", "op": "eq", "value": "b1"},
                ]
            },
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [])
    assert [e["id"] for e in activity.bindings["kept"]] == ["b1", "b3"]


async def test_filter_composition_nests(tmp_path: Path) -> None:
    step = Step(
        next_action="filter",
        params={
            "in": BOOKINGS,
            "out": "kept",
            "where": {
                "all": [
                    {"path": "room", "op": "eq", "value": "blue"},
                    {
                        "any": [
                            {"path": "id", "op": "eq", "value": "b1"},
                            {"path": "id", "op": "eq", "value": "b4"},
                        ]
                    },
                ]
            },
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [])
    assert [e["id"] for e in activity.bindings["kept"]] == ["b1", "b4"]


async def test_filter_composition_resolves_references_inside_clauses(tmp_path: Path) -> None:
    # A composed clause is not a second-class one: its operand resolves exactly as a lone clause's
    # would. Without the recursion the reference dict would reach `_matches` intact, every
    # comparison against it would raise TypeError, and the conjunction would keep nothing.
    step = Step(
        next_action="filter",
        params={
            "in": BOOKINGS,
            "out": "kept",
            "where": {
                "all": [
                    {"path": "room", "op": "eq", "value": "blue"},
                    {"path": "id", "op": "not_in", "value": {"$from": "already_done"}},
                ]
            },
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [_history("already_done", ["b1", "b2"])])
    assert [e["id"] for e in activity.bindings["kept"]] == ["b4"]


async def test_filter_empty_or_malformed_composition_is_a_defect_not_a_blanket_keep(
    tmp_path: Path,
) -> None:
    """An empty conjunction is vacuously TRUE in logic, which here would hand the whole collection
    to whatever fans out over it — one external action per item. The evaluator refuses both shapes
    and Reason reports them, so neither silently acts on everything nor silently acts on nothing."""
    malformed: list[dict[str, object]] = [{"all": []}, {"any": []}, {"all": "not a list"}]
    for where in malformed:
        step = Step(
            next_action="filter",
            params={"in": BOOKINGS, "out": "kept", "where": where},
        )
        activity = await _run_one_dataop(tmp_path, step, [])

        assert "kept" not in activity.bindings, where
        assert activity.plan is None, where
        assert activity.replan_trail[-1] is not None, where


async def test_filter_overlaps_is_half_open_by_default(tmp_path: Path) -> None:
    # b2 ends exactly when the probe starts and b4 starts after it ends: neither is a clash. Only
    # b3, which genuinely straddles the probe's range, is kept.
    probe = [{"starts_at": "2026-05-01T11:00", "ends_at": "2026-05-01T12:00"}]
    step = Step(
        next_action="filter",
        params={
            "in": BOOKINGS,
            "out": "clashing",
            "where": {
                "op": "overlaps",
                "start_path": "starts_at",
                "end_path": "ends_at",
                "against": {"$from": "the_new_one"},
                "against_start_path": "starts_at",
                "against_end_path": "ends_at",
            },
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [_history("the_new_one", probe)])
    assert [e["id"] for e in activity.bindings["clashing"]] == ["b3"]


async def test_filter_overlaps_inclusive_counts_a_shared_endpoint(tmp_path: Path) -> None:
    probe = [{"starts_at": "2026-05-01T11:00", "ends_at": "2026-05-01T12:00"}]
    step = Step(
        next_action="filter",
        params={
            "in": BOOKINGS,
            "out": "touching",
            "where": {
                "op": "overlaps",
                "start_path": "starts_at",
                "end_path": "ends_at",
                "against": {"$from": "the_new_one"},
                "against_start_path": "starts_at",
                "against_end_path": "ends_at",
                "boundaries": "inclusive",
            },
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [_history("the_new_one", probe)])
    assert [e["id"] for e in activity.bindings["touching"]] == ["b2", "b3", "b4"]


async def test_filter_overlaps_matches_any_member_of_the_against_collection(tmp_path: Path) -> None:
    # The existential: one item is kept if it clashes with AT LEAST ONE of several ranges. This is
    # what the general-quantifier design would have needed an alias grammar for; here the whole
    # collection resolves once and `_matches` loops over literals.
    probes = [
        {"starts_at": "2026-05-01T09:30", "ends_at": "2026-05-01T09:45"},
        {"starts_at": "2026-05-01T12:30", "ends_at": "2026-05-01T12:45"},
    ]
    step = Step(
        next_action="filter",
        params={
            "in": BOOKINGS,
            "out": "clashing",
            "where": {
                "op": "overlaps",
                "start_path": "starts_at",
                "end_path": "ends_at",
                "against": {"$from": "the_new_ones"},
                "against_start_path": "starts_at",
                "against_end_path": "ends_at",
            },
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [_history("the_new_ones", probes)])
    assert [e["id"] for e in activity.bindings["clashing"]] == ["b1", "b4"]


async def test_filter_overlaps_unresolvable_against_replans(tmp_path: Path) -> None:
    """`overlaps` fails CLOSED — nothing overlaps an unreadable collection — so an empty binding
    would read downstream as "no clashes", a real and actionable answer about the world."""
    step = Step(
        next_action="filter",
        params={
            "in": BOOKINGS,
            "out": "clashing",
            "where": {
                "op": "overlaps",
                "start_path": "starts_at",
                "end_path": "ends_at",
                "against": {"$from": "never_ran"},
                "against_start_path": "starts_at",
                "against_end_path": "ends_at",
            },
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [])

    assert "clashing" not in activity.bindings
    assert activity.plan is None
    assert activity.replan_trail[-1] is not None


async def test_filter_overlaps_without_a_collection_is_a_defect(tmp_path: Path) -> None:
    step = Step(
        next_action="filter",
        params={
            "in": BOOKINGS,
            "out": "clashing",
            "where": {
                "op": "overlaps",
                "start_path": "starts_at",
                "end_path": "ends_at",
                "against": "2026-05-01T11:00",
                "against_start_path": "starts_at",
                "against_end_path": "ends_at",
            },
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [])

    assert "clashing" not in activity.bindings
    assert activity.plan is None


async def test_filter_overlaps_bad_against_paths_warn_before_narrowing_silently(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The sibling of the membership value_path trap: the collection RESOLVES, but its ends project
    # to None, so nothing can overlap and the filter quietly keeps nothing.
    probe = [{"from": "2026-05-01T09:30", "to": "2026-05-01T12:45"}]
    step = Step(
        next_action="filter",
        params={
            "in": BOOKINGS,
            "out": "clashing",
            "where": {
                "op": "overlaps",
                "start_path": "starts_at",
                "end_path": "ends_at",
                "against": {"$from": "the_new_one"},
                "against_start_path": "starts_at",  # the probe calls them from/to
                "against_end_path": "ends_at",
            },
        },
    )
    with caplog.at_level(logging.WARNING, logger="sora.strategies"):
        activity = await _run_one_dataop(tmp_path, step, [_history("the_new_one", probe)])

    assert activity.bindings["clashing"] == []
    assert any("overlaps" in record.message for record in caplog.records)


async def test_composed_overlaps_semi_join_needs_no_model_call(tmp_path: Path) -> None:
    """The whole point, end to end: "the existing items that clash with the ones just added, not
    counting those additions themselves" is an anti-join AND an existential range join — three
    clauses that previously escalated together to one $decide over the entire collection. Composed,
    it runs inline with no LLM configured at all (`_run_one_dataop` builds a model-free cycle)."""
    added = [{"id": "b3", "starts_at": "2026-05-01T10:30", "ends_at": "2026-05-01T11:30"}]
    step = Step(
        next_action="filter",
        params={
            "in": BOOKINGS,
            "out": "to_cancel",
            "where": {
                "all": [
                    {
                        "path": "id",
                        "op": "not_in",
                        "value": {"$from": "just_added"},
                        "value_path": "id",
                    },
                    {
                        "op": "overlaps",
                        "start_path": "starts_at",
                        "end_path": "ends_at",
                        "against": {"$from": "just_added"},
                        "against_start_path": "starts_at",
                        "against_end_path": "ends_at",
                    },
                ]
            },
        },
    )
    activity = await _run_one_dataop(tmp_path, step, [_history("just_added", added)])
    # b2 (10:00-11:00) straddles the new 10:30 start; b1 ends at 10:00 and b4 starts at 12:00, so
    # neither clashes; b3 is excluded as one of the additions rather than kept as its own clash.
    assert [e["id"] for e in activity.bindings["to_cancel"]] == ["b2"]


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


async def test_collect_gathers_only_the_current_frames_results(tmp_path: Path) -> None:
    """``collect`` takes EVERY history match, which across a frame boundary means it accumulates
    results the *parent* plan produced, however stale. An observed run failed exactly here: a
    sub-plan collected `get_calendar_events_from_to` and got back both its own proposed-day query
    and the parent's Saturday query from 1,500 cycles earlier, then fanned out a delete over the
    stale set — whose event the agent had already deleted itself. It failed loudly only by luck;
    with a still-live event it would have silently deleted a real appointment on the wrong day.

    ``$from`` is deliberately *not* scoped this way and stays cross-frame: it reads the LATEST
    match, so it is naturally current, and a sub-plan referencing the event its parent created is
    the normal case. Only ``collect`` accumulates, so only ``collect`` needs the boundary."""
    history = [
        _history("get_events", {"day": "saturday"}),  # the parent frame's query
        _history("get_events", {"day": "thursday"}),  # this sub-plan's own query
    ]
    step = Step(next_action="collect", params={"from": "get_events", "out": "day_results"})
    tool = FakeTool("realestate")
    cycle, working, _ = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    activity = _activity_with_plan([step], history)
    activity.history_mark = 1  # this frame's plan was installed after history[0] had landed
    working.activities["a"] = activity
    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert activity.bindings["day_results"] == [{"day": "thursday"}]


async def test_collect_of_a_replaced_plans_results_is_a_defect_not_an_empty_binding(
    tmp_path: Path,
) -> None:
    """A replan resets the frame's span, so results the *replaced* plan fanned out are no longer
    collectible — deliberately, since a plan is usually discarded because the world moved under it.
    But the replanning planner is shown the whole history, so it can reasonably write a collect over
    them. Left alone that binds empty, and the emptiness resurfaces a step later as the generic
    "an earlier step produced EMPTY — nothing matched", which reads as a fact about the world: the
    planner re-plans into the same collect on that false premise until the replan breaker parks the
    activity. Naming the real cause is what breaks the loop."""
    history = [
        _history("get_crime_rate", {"zip": "1", "rate": 7}),
        _history("get_crime_rate", {"zip": "2", "rate": 3}),
    ]
    step = Step(next_action="collect", params={"from": "get_crime_rate", "out": "rates"})
    tool = FakeTool("realestate")
    cycle, working, _ = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    activity = _activity_with_plan([step], history)
    activity.history_mark = len(history)  # the replacement plan has run nothing of its own yet
    working.activities["a"] = activity

    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert "rates" not in activity.bindings  # no empty binding to mislead a later step
    assert activity.plan is None  # dropped for a replan
    assert activity.superseded is not None
    defect = activity.superseded.defect or ""
    assert "get_crime_rate" in defect
    assert "PREVIOUS plan" in defect  # the actual cause, not "nothing matched"


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


async def test_reset_for_replan_keeps_the_runtime_seeded_bindings(tmp_path: Path) -> None:
    """The exception is by the clearing rule, not against it: no plan produced these, so no plan's
    discarding makes them untrue. Clearing them would also make the replan unrecoverable — the next
    plan's `$bind` resolves to nothing, is reported as a defect, and replans again, on a reference
    the runtime itself told it to use."""
    activity = Activity(id="a", goal="g", context={})
    activity.bindings["stale"] = [1, 2, 3]
    activity.bindings["fired_added_ids"] = ["e9"]
    activity.reset_for_replan()
    assert activity.bindings == {"fired_added_ids": ["e9"]}


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


# --------------------------------------------------------------------------------------------------
# filter — the $decide predicate is judged against the SAME context grounding gets
# --------------------------------------------------------------------------------------------------
# The gaia2 adaptability run (examples/gaia2/logs/aug24-run1-gpt-5.5.log) failed here. The planner
# is explicitly taught to write predicates referencing an earlier result — "the upcoming Saturday
# computed from the get_current_time result" — and did exactly that. `select` then rendered only the
# goal, the predicate and the items, so the clock reading the predicate named was nowhere in the
# prompt: the model could not compute the date it was being asked to compare against and correctly
# answered {"keep": []}. An empty binding reads downstream as a real answer ("no appointments that
# day"), so the sub-goal fanned out to zero steps and the cancellation silently never happened.


async def test_decide_filter_sees_the_result_its_predicate_names(tmp_path: Path) -> None:
    llm = FakeLLMClient('{"keep": []}')
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm)
    activity = _activity_with_plan(
        [], [_history("get_current_time", {"current_datetime": "2024-10-15 07:00:54"})]
    )

    await procedural.select(activity, [{"id": "e1"}], "events on the Saturday after the clock read")

    _system, prompt = llm.calls[0]
    assert "2024-10-15 07:00:54" in prompt  # the referent, not merely the reference


async def test_decide_filter_sees_named_bindings_and_observed_state(tmp_path: Path) -> None:
    llm = FakeLLMClient('{"keep": []}')
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm)
    activity = _activity_with_plan([], [])
    activity.bindings["shortlist"] = [{"name": "Ake"}]
    observed = PerceptSnapshot(
        [Percept("clock", ObservableProperty("state", {"today": "2024-10-15"}), 0.0)], []
    )

    await procedural.select(activity, [{"id": "e1"}], "anything in the shortlist", observed)

    _system, prompt = llm.calls[0]
    assert "shortlist" in prompt
    assert "2024-10-15" in prompt


async def test_the_decide_filter_data_op_plumbs_observed_state_through(tmp_path: Path) -> None:
    # End to end: the escalation path must hand `select` the world, not just the collection —
    # otherwise the two tests above are unreachable from a real plan.
    llm = FakeLLMClient('{"keep": []}')
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm)
    tool = FakeTool("realestate")
    cycle, working, _ = _cycle(tmp_path, procedural, tool)
    working.properties[("clock", "state")] = Percept(
        "clock", ObservableProperty("state", {"today": "2024-10-15"}), 0.0
    )
    step = Step(
        next_action="filter",
        params={"in": [{"id": "e1"}], "out": "qualifying", "where": {"$decide": "todays events"}},
    )
    # The plan names the clock. Under intention-scoped focus that is not decoration: a tool
    # whose PROPERTY the
    # plan needs but whose operations it never calls is exactly the case an explicit `focus` step
    # exists for, and it is also what puts the clock in this activity's prompt view. A $decide
    # whose referent belongs to a tool no live plan mentions is not observed in the first place.
    activity = _activity_with_plan(
        [Step(next_action="focus", params={"tool_id": "clock"}), step],
        [_history("get_current_time", {"hour": 7})],
    )
    activity.step_index = 1  # the focus step already ran; the data-op is next
    working.activities["a"] = activity

    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    await asyncio.sleep(0)  # let the background select task run

    _system, prompt = llm.calls[0]
    assert "get_current_time" in prompt
    assert "2024-10-15" in prompt


# --------------------------------------------------------------------------------------------------
# filter — a $decide predicate whose referent is absent reports the gap instead of keeping nothing
# --------------------------------------------------------------------------------------------------
# Giving `select` the context makes a resolvable predicate resolve, but leaves the other half of the
# same silent failure open: when the referent genuinely is not there, {"keep": []} is the model's
# only legal answer and is indistinguishable from "nothing matched". It lands in a binding, and an
# empty binding reads downstream as a real answer ("no appointments that day") rather than as the
# question it actually is. Grounding already has the channel for this; so does select now.


async def test_a_predicate_naming_absent_data_reports_the_gap(tmp_path: Path) -> None:
    llm = FakeLLMClient(
        '{"unresolvable": "predicate names the get_current_time result; no clock '
        'reading is in the history"}'
    )
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm)

    try:
        await procedural.select(_activity_with_plan([], []), [{"id": "e1"}], "events on Saturday")
    except UnresolvableGrounding as exc:
        assert "no clock reading" in str(exc)
    else:  # pragma: no cover — the assertion below is the failure report
        raise AssertionError("expected the gap to be reported, not an empty keep-list")


async def test_a_select_response_carrying_both_is_read_as_the_gap(tmp_path: Path) -> None:
    # Same hedging rule grounding applies: the keep half of such an answer is the guess this channel
    # exists to stop, so the reported gap wins.
    llm = FakeLLMClient('{"unresolvable": "no clock reading", "keep": [0]}')
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm)

    try:
        await procedural.select(_activity_with_plan([], []), [{"id": "e1"}], "events on Saturday")
    except UnresolvableGrounding:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected the unresolvable report to win over the keep list")


async def test_an_unresolvable_predicate_replans_rather_than_writing_an_empty_binding(
    tmp_path: Path,
) -> None:
    # The whole point: an unresolvable predicate must NOT leave a binding a later step reads as an
    # answer. It is a defect in the PLAN — a step assumed an earlier one would yield something it
    # did not — so the repair is a replan carrying the defect, exactly as for grounding.
    llm = FakeLLMClient('{"unresolvable": "predicate names a clock reading that was never taken"}')
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm)
    tool = FakeTool("realestate")
    cycle, working, _ = _cycle(tmp_path, procedural, tool)
    step = Step(
        next_action="filter",
        params={"in": [{"id": "e1"}], "out": "qualifying", "where": {"$decide": "todays events"}},
    )
    activity = _activity_with_plan([step], [])
    working.activities["a"] = activity

    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    await asyncio.sleep(0)
    await DefaultObserveStrategy().observe(cycle)

    assert "qualifying" not in activity.bindings  # no empty binding masquerading as an answer
    assert activity.plan is None  # discarded for re-inference
    assert activity.state is ActivityState.READY
    assert activity.superseded is not None and activity.superseded.defect is not None
    assert "clock reading" in activity.superseded.defect


async def test_a_transient_select_failure_still_degrades_to_an_empty_binding(
    tmp_path: Path,
) -> None:
    # The distinction the new channel rests on: a model/parse FAILURE is not a reported gap. It
    # stays fail-soft (the pipeline does nothing this run) rather than triggering a replan, so a
    # flaky call cannot churn the plan.
    llm = FakeLLMClient("not json at all")
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm)
    tool = FakeTool("realestate")
    cycle, working, _ = _cycle(tmp_path, procedural, tool)
    step = Step(
        next_action="filter",
        params={"in": [{"id": "e1"}], "out": "qualifying", "where": {"$decide": "todays events"}},
    )
    activity = _activity_with_plan([step], [])
    working.activities["a"] = activity

    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    await asyncio.sleep(0)
    await DefaultObserveStrategy().observe(cycle)

    assert activity.bindings["qualifying"] == []
    assert activity.plan is not None  # not a replan
