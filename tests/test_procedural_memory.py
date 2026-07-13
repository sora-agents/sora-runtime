"""Tests for the deterministic, file-backed ``ProceduralMemory`` (store/retrieve).

``ProceduralMemory`` caches ``Plan``s a completed activity actually followed, so a later activity
with the same goal reuses one instead of re-planning. The default is deterministic: a ``Plan`` is
stored under its own stable ``id`` and retrieved by an exact match on its ``goal`` (the retrieval
key). ``infer()`` — the expensive, potentially LLM-backed path that synthesizes a fresh plan — is
deliberately still a stub here.

Backed by a real ``FileMemoryBackend`` (the deterministic default, already fast and covered by
``test_memory_backend.py``) rather than a fake, so these tests pin the actual serialization
round-trip the module owns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sora.activity import Activity
from sora.memory import FileMemoryBackend, ProceduralMemory
from sora.types import Plan, Step


def _activity(goal: str) -> Activity:
    return Activity(id=f"act-{goal}", goal=goal, context={})


def _plan(plan_id: str, goal: str, steps: list[Step] | None = None) -> Plan:
    if steps is None:
        steps = [Step(next_action="invoke", params={"tool_id": "EmailClientApp", "n": 1})]
    return Plan(id=plan_id, goal=goal, steps=steps)


def _memory(tmp_path: Path) -> ProceduralMemory:
    return ProceduralMemory(FileMemoryBackend(tmp_path))


# --------------------------------------------------------------------------------------------------
# store -> retrieve round-trip
# --------------------------------------------------------------------------------------------------


async def test_store_then_retrieve_by_goal_returns_equal_plan(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    plan = _plan("p1", "schedule from email")
    await mem.store(plan)
    got = await mem.retrieve(_activity("schedule from email"))
    assert got == plan


async def test_retrieve_unknown_goal_returns_none(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    await mem.store(_plan("p1", "schedule from email"))
    assert await mem.retrieve(_activity("book a flight")) is None


async def test_retrieve_on_empty_store_returns_none(tmp_path: Path) -> None:
    mem = _memory(tmp_path / "never_written")
    assert await mem.retrieve(_activity("anything")) is None


async def test_store_same_id_overwrites(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    await mem.store(_plan("p1", "schedule from email", [Step(next_action="wait", params={})]))
    updated = _plan("p1", "schedule from email", [Step(next_action="invoke", params={"n": 2})])
    await mem.store(updated)
    got = await mem.retrieve(_activity("schedule from email"))
    assert got == updated


async def test_retrieve_matches_on_goal_not_id(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    await mem.store(_plan("p-a", "goal A"))
    await mem.store(_plan("p-b", "goal B"))
    got = await mem.retrieve(_activity("goal B"))
    assert got is not None
    assert got.id == "p-b"


async def test_retrieve_is_exact_match_not_fuzzy(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    await mem.store(_plan("p1", "schedule from email"))
    # A similar-but-different goal must not resolve to the stored plan.
    assert await mem.retrieve(_activity("schedule email")) is None


# --------------------------------------------------------------------------------------------------
# Serialization fidelity of the Plan/Step shape
# --------------------------------------------------------------------------------------------------


async def test_multi_step_plan_round_trips_with_order_and_types(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    steps = [
        Step(next_action="invoke", params={"tool_id": "EmailClientApp", "n": 1}),
        Step(next_action="wait", params={}),
        Step(next_action="send", params={"to": "peer", "content": {"nested": [1, 2, 3]}}),
    ]
    plan = _plan("p-multi", "compose reply", steps)
    await mem.store(plan)
    got = await mem.retrieve(_activity("compose reply"))
    assert got == plan
    assert got is not None
    assert [s.next_action for s in got.steps] == ["invoke", "wait", "send"]


async def test_empty_steps_plan_round_trips(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    plan = _plan("p-empty", "noop goal", [])
    await mem.store(plan)
    got = await mem.retrieve(_activity("noop goal"))
    assert got == plan
    assert got is not None
    assert got.steps == []


# --------------------------------------------------------------------------------------------------
# Durability + isolation (the file backend re-reads from disk)
# --------------------------------------------------------------------------------------------------


async def test_plan_persists_across_memory_instances(tmp_path: Path) -> None:
    await _memory(tmp_path).store(_plan("p1", "durable goal"))
    # A fresh module over the same root — i.e. a process restart — sees the prior store.
    got = await _memory(tmp_path).retrieve(_activity("durable goal"))
    assert got is not None
    assert got.id == "p1"


async def test_retrieved_plan_is_isolated_from_store(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    await mem.store(_plan("p1", "schedule from email"))
    first = await mem.retrieve(_activity("schedule from email"))
    assert first is not None
    first.steps[0].params["mutated"] = True  # mutate the returned plan's step params
    second = await mem.retrieve(_activity("schedule from email"))
    assert second is not None
    assert "mutated" not in second.steps[0].params  # store untouched


# --------------------------------------------------------------------------------------------------
# infer(): deliberately deferred (the LLM path)
# --------------------------------------------------------------------------------------------------


async def test_infer_is_not_implemented(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    with pytest.raises(NotImplementedError):
        await mem.infer(_activity("anything"))
