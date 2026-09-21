from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from examples.gaia2.evaluation.campaigns.prompt.capture import BufferedLLMExchangeCapture
from examples.gaia2.evaluation.core import (
    EvaluationRecord,
    MatrixEntry,
    load_judge_profile,
    load_profiles,
)
from examples.gaia2.evaluation.diagnostics import (
    BufferedSessionLog,
    allocate_attempt_directory,
    export_attempt,
)
from examples.gaia2.rescore import summarize

from sora.adapters.are_judge import JudgedEvent, JudgeRecording
from sora.diagnostics import RuntimeEventCollector
from sora.llm import CompletionRequest


def _inputs(tmp_path: Path) -> tuple[Any, ...]:
    del tmp_path
    profile = load_profiles(Path("examples/gaia2/evaluation/profiles.json"))[
        "gpt-5.4-medium-prompt"
    ]
    judge = load_judge_profile(Path("examples/gaia2/evaluation/campaigns/prompt/judge.json"))
    entry = MatrixEntry(
        profile=profile.name,
        suite="development",
        capability="search",
        case_id="case",
        arm="baseline",
        repeat=0,
        reserved_agent_cost=3.0,
        reserved_judge_cost=0.5,
    )
    record = EvaluationRecord(
        arm="baseline",
        profile=profile.name,
        suite="development",
        capability="search",
        case_id="case",
        repeat=0,
        score=1.0,
        passed=True,
        call_records=({"call_id": "c1", "inference_id": "i1"},),
        terminal_cause="verification_completion",
    )
    recording = JudgeRecording(
        events=(
            JudgedEvent(
                tool_name="Tool__write",
                agent_args={"value": 1},
                oracle_args={"value": 1},
                equality=True,
                verdict=True,
            ),
        ),
        scenario_id="scenario-1",
        run_number=0,
        verdict_parse="stock",
    )
    result = SimpleNamespace(
        outcome=SimpleNamespace(success=True, rationale="matched"),
        judge_recording=recording,
        write_counts=None,
        environment=None,
        exception=None,
        duration=1.5,
    )
    events = RuntimeEventCollector()
    events.emit({"event": "cycle.entry", "cycle": 1, "payload": {}})
    capture = BufferedLLMExchangeCapture()
    capture.record(
        CompletionRequest("system", "user", "plan", "1"),
        response='{"steps": []}',
        error=None,
        call_id="c1",
        inference_id="i1",
        elapsed_seconds=0.25,
    )
    session = BufferedSessionLog()
    session.emit(logging.LogRecord("sora.test", logging.INFO, __file__, 1, "hello", (), None))
    return profile, judge, entry, record, result, events, capture, session


def test_attempt_bundle_is_integrity_indexed_and_rescorable(tmp_path: Path) -> None:
    profile, judge, entry, record, result, events, capture, session = _inputs(tmp_path)
    attempt = allocate_attempt_directory(tmp_path, entry.key, entry.suite)
    reference = export_attempt(
        attempt,
        output_dir=tmp_path,
        entry=entry,
        record=record,
        result=result,
        scenario=SimpleNamespace(scenario_id="scenario-1"),
        config_path=tmp_path / "agent.yaml",
        profile=profile,
        judge_profile=judge,
        runtime_events=events,
        llm_capture=capture,
        session_log=session,
        source_revision="abc",
        source_dirty_diff_sha256=None,
    )

    expected = {
        "run.json",
        "trajectory.jsonl",
        "judge_recording.json",
        "verdict.json",
        "write_counts.json",
        "llm_calls.json",
        "llm/exchanges.jsonl",
        "session.log",
        "index.json",
    }
    paths = {path.relative_to(attempt).as_posix() for path in attempt.rglob("*") if path.is_file()}
    assert expected <= paths
    index = json.loads((attempt / "index.json").read_text())
    assert index["complete"] is True
    assert [row["path"] for row in index["files"]] == sorted(row["path"] for row in index["files"])
    for row in index["files"]:
        data = (attempt / row["path"]).read_bytes()
        assert row["bytes"] == len(data)
        assert row["sha256"] == hashlib.sha256(data).hexdigest()
    assert reference["error"] is None
    rescored = summarize([attempt])
    assert rescored["recordings"][0]["scores"]["stock"] == 1.0
    assert rescored["recordings"][0]["scores"]["case-insensitive"] == 1.0
    assert rescored["unreproduced_recordings"] == 0


def test_attempt_directories_never_overwrite_and_acceptance_is_segregated(
    tmp_path: Path,
) -> None:
    first = allocate_attempt_directory(tmp_path, "baseline:p:development:c:0", "development")
    second = allocate_attempt_directory(tmp_path, "baseline:p:development:c:0", "development")
    secret = allocate_attempt_directory(tmp_path, "baseline:p:acceptance:c:0", "acceptance")
    assert first.name == "attempt-0"
    assert second.name == "attempt-1"
    assert secret.is_relative_to(tmp_path / "artifacts" / "acceptance")


def test_component_export_failure_is_recorded_without_changing_score(tmp_path: Path) -> None:
    profile, judge, entry, record, result, events, capture, session = _inputs(tmp_path)
    attempt = allocate_attempt_directory(tmp_path, entry.key, entry.suite)

    def fail(_root: Any) -> tuple[Any, ...]:
        raise OSError("disk full")

    capture.export = fail
    reference = export_attempt(
        attempt,
        output_dir=tmp_path,
        entry=entry,
        record=record,
        result=result,
        scenario=SimpleNamespace(scenario_id="scenario-1"),
        config_path=tmp_path / "agent.yaml",
        profile=profile,
        judge_profile=judge,
        runtime_events=events,
        llm_capture=capture,
        session_log=session,
        source_revision="abc",
        source_dirty_diff_sha256=None,
    )
    assert "llm: OSError: disk full" in reference["error"]
    assert json.loads((attempt / "index.json").read_text())["complete"] is False
    assert record.score == 1.0
