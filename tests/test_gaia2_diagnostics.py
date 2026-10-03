from __future__ import annotations

import hashlib
import json
import logging
import tracemalloc
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from examples.gaia2.evaluation.campaigns.prompt.capture import BufferedLLMExchangeCapture
from examples.gaia2.evaluation.core import (
    EvaluationRecord,
    MatrixEntry,
    load_judge_profile,
    load_profiles,
)
from examples.gaia2.evaluation.diagnostics import (
    BufferedSessionLog,
    _index,
    _write_jsonl,
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


class _Unreprable:
    """A value no projection can render: ``json_safe``'s catch-all branch calls ``repr`` on an
    unrecognized object, so this is what an unserializable row actually looks like."""

    def __repr__(self) -> str:
        raise RuntimeError("this object refuses to be rendered")


def test_jsonl_rows_are_written_as_they_are_consumed_not_materialized_first(
    tmp_path: Path,
) -> None:
    """The defect was `write_text("".join(...))`: the finished file was built as one string before
    a byte reached disk, on top of the caller's own copy. Proven by a source that fails partway —
    the rows already consumed are on disk, which cannot be true of a write that happens only after
    the source is exhausted."""
    path = tmp_path / "trajectory.jsonl"

    def rows() -> Any:
        for index in range(3):
            yield {"sequence": index, "event": "phase.entry"}
        raise RuntimeError("the source died partway through")

    with pytest.raises(RuntimeError, match="died partway"):
        _write_jsonl(path, rows())

    written = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["sequence"] for row in written] == [0, 1, 2]


def test_jsonl_output_is_byte_identical_to_the_materialized_form(tmp_path: Path) -> None:
    """Streaming is an allocation change, not a format change: the bundle's integrity index and
    every tool that reads a trajectory must see exactly the bytes the joined write produced."""
    rows = [
        {"sequence": i, "event": "boundary.property.received", "payload": {"v": [i] * 4}}
        for i in range(5)
    ]
    streamed = tmp_path / "streamed.jsonl"
    _write_jsonl(streamed, iter(rows))
    joined = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    assert streamed.read_text() == joined


def test_one_unserializable_row_does_not_cost_the_whole_trajectory(tmp_path: Path) -> None:
    """A trajectory is what a failed run is read back from, so losing every row to one bad value is
    the worst possible failure mode. The bad row becomes a located placeholder; the rest
    survives."""
    path = tmp_path / "trajectory.jsonl"
    _write_jsonl(
        path,
        iter(
            [
                {"sequence": 0, "event": "phase.entry"},
                {"sequence": 1, "event": "action.dispatch", "payload": _Unreprable()},
                {"sequence": 2, "event": "phase.exit"},
            ]
        ),
    )

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "phase.entry",
        "diagnostic.row_unserializable",
        "phase.exit",
    ]
    assert rows[1]["sequence"] == 1  # the gap is locatable in the rest of the bundle
    assert "RuntimeError" in rows[1]["error"]


def test_session_log_lines_match_the_joined_text(tmp_path: Path) -> None:
    """`lines()` exists so session.log can be written without first joining it; it has to agree
    with `text()` exactly, since that is what the comparison against older bundles rests on."""
    del tmp_path
    session = BufferedSessionLog()
    for message in ("first", "second", "third"):
        session.emit(logging.LogRecord("sora", logging.INFO, __file__, 1, message, None, None))
    assert list(session.lines()) == session.text().splitlines()


def test_export_peak_allocation_does_not_scale_with_the_trajectory(tmp_path: Path) -> None:
    """The defect was an allocation shape, so this is the test that actually pins the fix: peak
    memory across the whole collector-to-file path must stay a small fraction of the payload rather
    than a multiple of it. The materializing form measured ~2x the file (the caller's copy plus the
    joined string); the streamed form holds one row. A quarter of the payload is a deliberately
    loose ceiling — it fails the old shape by roughly 8x while leaving room for interpreter
    noise."""
    collector = RuntimeEventCollector()
    blob = {"items": [{"id": f"item-{index}", "text": "x" * 64} for index in range(80)]}
    for _ in range(400):
        collector.emit({"event": "boundary.property.received", "payload": {"value": blob}})

    path = tmp_path / "trajectory.jsonl"
    tracemalloc.start()
    try:
        _write_jsonl(path, collector.stream())
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    payload_bytes = path.stat().st_size
    assert payload_bytes > 2_000_000  # the fixture is big enough for the ratio to mean something
    assert peak < payload_bytes / 4


def test_indexing_a_bundle_does_not_read_whole_files_into_memory(tmp_path: Path) -> None:
    """The index's own materialization, and the larger of the two: it ran `read_bytes()` over every
    file in the bundle purely to size and hash it, so the trajectory just written was pulled back
    in as one allocation — on observed runs that was a single 8.6 GiB read. Size now comes from
    stat and the digest is fed in fixed chunks.

    Asserted as a SCALING property rather than a fraction of the file, because the fixed chunk is
    the floor: peak is O(chunk), not O(file), so quadrupling the file must not move it. A ratio
    threshold would instead just be measuring the chunk size on a small fixture. Digest correctness
    is already pinned by the bundle-integrity test above; this pins the allocation."""
    line = b'{"event":"boundary.property.received"}\n'

    def peak_for(name: str, repeats: int) -> tuple[int, int]:
        root = tmp_path / name
        root.mkdir()
        (root / "trajectory.jsonl").write_bytes(line * repeats)
        tracemalloc.start()
        try:
            _index(root, {})
            return tracemalloc.get_traced_memory()[1], (root / "trajectory.jsonl").stat().st_size
        finally:
            tracemalloc.stop()

    small_peak, small_bytes = peak_for("small", 80_000)
    large_peak, large_bytes = peak_for("large", 320_000)

    assert large_bytes == small_bytes * 4  # the input really did grow fourfold
    assert large_peak < small_peak * 1.5  # ... and the peak did not follow it

    index = _index(tmp_path / "large", {})
    row = next(entry for entry in index["files"] if entry["path"] == "trajectory.jsonl")
    assert row["bytes"] == large_bytes
    assert (
        row["sha256"]
        == hashlib.sha256((tmp_path / "large" / "trajectory.jsonl").read_bytes()).hexdigest()
    )
