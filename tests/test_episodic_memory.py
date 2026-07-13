"""Behavior tests for ``EpisodicMemory`` (learn / consult).

Episodic memory saves a summary of a completed activity (``learn``) and retrieves the experiences
relevant to a new activity (``consult``). The deterministic default treats "relevant" as
goal-equality — the same cheap proxy ``ProceduralMemory.retrieve`` uses — so everything here is
exercised against the real ``FileMemoryBackend``, which also pins that the JSON round-trip works.

Episodes are the plain ``{activity_id, goal, summary}`` dict the module hands the backend;
``consult`` returns those dicts (its README signature is the deliberately loose ``list[Any]``).
"""

from __future__ import annotations

from pathlib import Path

from sora.activity import Activity
from sora.memory import EpisodicMemory, FileMemoryBackend


def _activity(id: str, goal: str) -> Activity:
    return Activity(id=id, goal=goal, context={})


# --------------------------------------------------------------------------------------------------
# learn -> consult round-trip
# --------------------------------------------------------------------------------------------------


async def test_learn_then_consult_returns_episode(tmp_path: Path) -> None:
    memory = EpisodicMemory(FileMemoryBackend(tmp_path))
    activity = _activity("a1", "schedule from email")
    await memory.learn(activity, "booked the 3pm slot and replied")

    episodes = await memory.consult(_activity("a2", "schedule from email"))

    assert episodes == [
        {
            "activity_id": "a1",
            "goal": "schedule from email",
            "summary": "booked the 3pm slot and replied",
        }
    ]


async def test_learn_records_activity_id_goal_and_summary(tmp_path: Path) -> None:
    memory = EpisodicMemory(FileMemoryBackend(tmp_path))
    await memory.learn(_activity("a1", "triage inbox"), "archived 4, flagged 1")

    (episode,) = await memory.consult(_activity("a1", "triage inbox"))
    assert episode == {
        "activity_id": "a1",
        "goal": "triage inbox",
        "summary": "archived 4, flagged 1",
    }


# --------------------------------------------------------------------------------------------------
# consult relevance = goal-equality
# --------------------------------------------------------------------------------------------------


async def test_consult_with_no_episodes_returns_empty(tmp_path: Path) -> None:
    memory = EpisodicMemory(FileMemoryBackend(tmp_path))
    assert await memory.consult(_activity("a1", "anything")) == []


async def test_consult_only_returns_matching_goal(tmp_path: Path) -> None:
    memory = EpisodicMemory(FileMemoryBackend(tmp_path))
    await memory.learn(_activity("a1", "schedule from email"), "did the scheduling")
    await memory.learn(_activity("a2", "triage inbox"), "did the triage")

    episodes = await memory.consult(_activity("a3", "schedule from email"))
    assert episodes == [
        {"activity_id": "a1", "goal": "schedule from email", "summary": "did the scheduling"}
    ]


async def test_multiple_episodes_same_goal_all_returned(tmp_path: Path) -> None:
    memory = EpisodicMemory(FileMemoryBackend(tmp_path))
    await memory.learn(_activity("a1", "schedule from email"), "first time")
    await memory.learn(_activity("a2", "schedule from email"), "second time")

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
    await memory.learn(activity, "first attempt notes")
    await memory.learn(activity, "corrected final notes")

    episodes = await memory.consult(activity)
    assert episodes == [
        {"activity_id": "a1", "goal": "schedule from email", "summary": "corrected final notes"}
    ]


# --------------------------------------------------------------------------------------------------
# persistence across instances (the point of "file-backed")
# --------------------------------------------------------------------------------------------------


async def test_episodes_persist_across_instances(tmp_path: Path) -> None:
    await EpisodicMemory(FileMemoryBackend(tmp_path)).learn(
        _activity("a1", "schedule from email"), "durable experience"
    )
    # A fresh module over the same root (i.e. a process restart) sees the prior write.
    episodes = await EpisodicMemory(FileMemoryBackend(tmp_path)).consult(
        _activity("a2", "schedule from email")
    )
    assert episodes == [
        {"activity_id": "a1", "goal": "schedule from email", "summary": "durable experience"}
    ]


# --------------------------------------------------------------------------------------------------
# consult results are fresh copies — mutating them can't corrupt the store
# --------------------------------------------------------------------------------------------------


async def test_consult_result_isolated_from_store(tmp_path: Path) -> None:
    memory = EpisodicMemory(FileMemoryBackend(tmp_path))
    await memory.learn(_activity("a1", "schedule from email"), "original")

    episodes = await memory.consult(_activity("a1", "schedule from email"))
    episodes.append({"activity_id": "spoof", "goal": "schedule from email", "summary": "injected"})
    episodes[0]["summary"] = "tampered"

    fresh = await memory.consult(_activity("a1", "schedule from email"))
    assert fresh == [{"activity_id": "a1", "goal": "schedule from email", "summary": "original"}]
