"""Behavior tests for ``EpisodicMemory`` (learn / consult).

Episodic memory saves an experience of a completed activity (``learn``) and retrieves the ones
relevant to a new activity (``consult``). The deterministic default treats "relevant" as
goal-equality — the same cheap proxy ``ProceduralMemory.retrieve`` uses — so everything here is
exercised against the real ``FileMemoryBackend``, which also pins that the JSON round-trip works.

An episode is a self-contained experience: beyond the prose ``summary`` it carries the ``succeeded``
outcome, the plan snapshot, step progress (``step_index``/``step_count``), and the last operation
result — captured from the activity ``learn`` receives. ``consult`` returns those dicts (its README
signature is the deliberately loose ``list[Any]``).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from sora.activity import Activity, ActivityState
from sora.memory import EpisodicMemory, FileMemoryBackend
from sora.types import OperationAck, Plan, Step


def _activity(id: str, goal: str) -> Activity:
    return Activity(id=id, goal=goal, context={})


def _episode(
    id: str,
    goal: str,
    summary: str,
    *,
    succeeded: bool = True,
    step_index: int = 0,
    step_count: int | None = None,
    last_result: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The full stored record for a plain ``_activity`` (no plan, no last op) unless overridden."""
    return {
        "activity_id": id,
        "goal": goal,
        "succeeded": succeeded,
        "summary": summary,
        "step_index": step_index,
        "step_count": step_count,
        "last_result": last_result,
        "plan": plan,
    }


# --------------------------------------------------------------------------------------------------
# learn -> consult round-trip
# --------------------------------------------------------------------------------------------------


async def test_learn_then_consult_returns_episode(tmp_path: Path) -> None:
    memory = EpisodicMemory(FileMemoryBackend(tmp_path))
    activity = _activity("a1", "schedule from email")
    await memory.learn(activity, "booked the 3pm slot and replied", succeeded=True)

    episodes = await memory.consult(_activity("a2", "schedule from email"))

    assert episodes == [_episode("a1", "schedule from email", "booked the 3pm slot and replied")]


async def test_learn_records_the_full_experience(tmp_path: Path) -> None:
    # The enriched record: outcome, the plan snapshot, step progress, and the last operation result,
    # all captured from the activity learn receives.
    memory = EpisodicMemory(FileMemoryBackend(tmp_path))
    plan = Plan(
        id="plan-1",
        goal="schedule from email",
        steps=[Step(next_action="invoke", params={}), Step(next_action="send", params={})],
    )
    activity = Activity(
        id="a1",
        goal="schedule from email",
        context={},
        state=ActivityState.TERMINATED,
        plan=plan,
        step_index=1,
        last_operation=OperationAck(ok=False, result={"error": "conflict"}),
    )
    await memory.learn(activity, "hit a calendar conflict", succeeded=False)

    (episode,) = await memory.consult(_activity("a2", "schedule from email"))
    assert episode == _episode(
        "a1",
        "schedule from email",
        "hit a calendar conflict",
        succeeded=False,
        step_index=1,
        step_count=2,
        last_result=asdict(OperationAck(ok=False, result={"error": "conflict"})),
        plan=asdict(plan),
    )


# --------------------------------------------------------------------------------------------------
# consult relevance = goal-equality
# --------------------------------------------------------------------------------------------------


async def test_consult_with_no_episodes_returns_empty(tmp_path: Path) -> None:
    memory = EpisodicMemory(FileMemoryBackend(tmp_path))
    assert await memory.consult(_activity("a1", "anything")) == []


async def test_consult_only_returns_matching_goal(tmp_path: Path) -> None:
    memory = EpisodicMemory(FileMemoryBackend(tmp_path))
    await memory.learn(_activity("a1", "schedule from email"), "did the scheduling", succeeded=True)
    await memory.learn(_activity("a2", "triage inbox"), "did the triage", succeeded=True)

    episodes = await memory.consult(_activity("a3", "schedule from email"))
    assert episodes == [_episode("a1", "schedule from email", "did the scheduling")]


async def test_multiple_episodes_same_goal_all_returned(tmp_path: Path) -> None:
    memory = EpisodicMemory(FileMemoryBackend(tmp_path))
    await memory.learn(_activity("a1", "schedule from email"), "first time", succeeded=True)
    await memory.learn(_activity("a2", "schedule from email"), "second time", succeeded=True)

    episodes = await memory.consult(_activity("a3", "schedule from email"))
    assert len(episodes) == 2
    assert {e["activity_id"] for e in episodes} == {"a1", "a2"}
    assert {e["summary"] for e in episodes} == {"first time", "second time"}


# --------------------------------------------------------------------------------------------------
# keying: one episode per activity id, re-learning overwrites
# --------------------------------------------------------------------------------------------------


async def test_relearning_same_activity_overwrites(tmp_path: Path) -> None:
    memory = EpisodicMemory(FileMemoryBackend(tmp_path))
    activity = _activity("a1", "schedule from email")
    await memory.learn(activity, "first attempt notes", succeeded=True)
    await memory.learn(activity, "corrected final notes", succeeded=True)

    episodes = await memory.consult(activity)
    assert episodes == [_episode("a1", "schedule from email", "corrected final notes")]


# --------------------------------------------------------------------------------------------------
# persistence across instances (the point of "file-backed")
# --------------------------------------------------------------------------------------------------


async def test_episodes_persist_across_instances(tmp_path: Path) -> None:
    await EpisodicMemory(FileMemoryBackend(tmp_path)).learn(
        _activity("a1", "schedule from email"), "durable experience", succeeded=True
    )
    # A fresh module over the same root (i.e. a process restart) sees the prior write.
    episodes = await EpisodicMemory(FileMemoryBackend(tmp_path)).consult(
        _activity("a2", "schedule from email")
    )
    assert episodes == [_episode("a1", "schedule from email", "durable experience")]


# --------------------------------------------------------------------------------------------------
# consult results are fresh copies — mutating them can't corrupt the store
# --------------------------------------------------------------------------------------------------


async def test_consult_result_isolated_from_store(tmp_path: Path) -> None:
    memory = EpisodicMemory(FileMemoryBackend(tmp_path))
    await memory.learn(_activity("a1", "schedule from email"), "original", succeeded=True)

    episodes = await memory.consult(_activity("a1", "schedule from email"))
    episodes.append({"activity_id": "spoof", "goal": "schedule from email", "summary": "injected"})
    episodes[0]["summary"] = "tampered"

    fresh = await memory.consult(_activity("a1", "schedule from email"))
    assert fresh == [_episode("a1", "schedule from email", "original")]
