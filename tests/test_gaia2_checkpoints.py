"""Scenario-boundary pause/recovery without ARE, network calls, or model spend."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from examples.gaia2 import batch, checkpoints
from examples.gaia2.llm_calls import LLMCallRecord


def _inputs(tmp_path: Path) -> tuple[Namespace, Path, list[Any]]:
    plan = tmp_path / "launch-plan.json"
    plan.write_text('{"frozen": "offline-test"}')
    manifest = batch.SweepManifest(
        "test", "dataset", "revision", "validation", (("search", "a"), ("search", "b")), "digest"
    )
    args = Namespace(
        checkpoint_plan=plan,
        pause_file=tmp_path / "PAUSE",
        arm="sora",
        capability="search",
        num_runs=1,
        sweep_manifest=manifest,
    )
    return (
        args,
        tmp_path / "capability",
        [(SimpleNamespace(scenario_id=case), []) for case in ("a", "b")],
    )


def _fake_run(
    scenario: Any,
    repeat: int,
    args: Namespace,
    directory: str,
    *,
    record_judge: bool,
    llm_calls: Any,
) -> dict[str, Any]:
    root = Path(directory)
    trace = root / "trace.json"
    judge = root / "judge.json"
    trace.write_text("{}")
    judge.write_text('{"events": [{}]}')
    llm_calls.write(
        LLMCallRecord("call", "sora", "offline", "plan", 10, 0, 5, None, 0.01, 1, "stop")
    )
    return batch._jsonl_record(
        scenario_id=scenario.scenario_id,
        run_number=repeat,
        success=False,
        rationale="ordinary agent failure",
        exception=None,
        trace_id=str(trace),
        judge_recording_path=str(judge),
    )


def _run(args: Namespace, root: Path, scenarios: list[Any]) -> list[dict[str, Any]]:
    return checkpoints.run_checkpointed(args, root, scenarios, lambda: scenarios, record_judge=True)


@pytest.mark.parametrize("success", [False, True])
@pytest.mark.parametrize("errors", [None, []])
def test_scored_batch_record_without_inference_errors_is_retained(
    success: bool, errors: list[str] | None
) -> None:
    row = batch._jsonl_record(
        scenario_id="a",
        run_number=0,
        success=success,
        rationale="scored outcome",
        exception=None,
        trace_id="trace",
        judge_recording_path="judge.json",
    )
    assert "terminal_inference_errors" not in row["metadata"]
    assert not checkpoints.infrastructure_failure(row)
    row["metadata"]["terminal_inference_errors"] = errors
    assert not checkpoints.infrastructure_failure(row)


def test_pause_then_resume_keeps_completed_zero_score_and_preserves_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, root, scenarios = _inputs(tmp_path)
    calls = []

    def pause_after_first(*values: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(values[0].scenario_id)
        row = _fake_run(*values, **kwargs)
        args.pause_file.touch()
        return row

    monkeypatch.setattr(batch, "_run_one_scenario", pause_after_first)
    with pytest.raises(SystemExit) as paused:
        _run(args, root, scenarios)
    assert paused.value.code == checkpoints.PAUSED_EXIT
    store = checkpoints.ScenarioCheckpoints(
        root, args.checkpoint_plan, arm="sora", capability="search"
    )
    accepted = store.completed()
    assert set(accepted) == {("a", 0)}
    attempt = accepted[("a", 0)]
    before = batch._artifact_checksums(attempt)
    assert not (root / batch.ARTIFACT_ATTESTATION).exists()
    args.pause_file.unlink()
    monkeypatch.setattr(batch, "_run_one_scenario", _fake_run)
    rows = _run(args, root, scenarios)
    assert len(rows) == 2 and all(row["score"] == 0 for row in rows)
    assert calls == ["a"]
    assert before == batch._artifact_checksums(attempt)
    assert len(list((root / "attempts").glob("*/attempt-*"))) == 2
    assert batch.verify_artifact_attestation(str(root))[1] == ()
    with pytest.raises(ValueError, match="immutable"):
        _run(args, root, scenarios)


def test_runtime_crash_preserves_call_logs_and_restarts_only_interrupted_scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, root, scenarios = _inputs(tmp_path)
    called = []

    def crash_at_second(*values: Any, **kwargs: Any) -> dict[str, Any]:
        called.append(values[0].scenario_id)
        row = _fake_run(*values, **kwargs)
        if values[0].scenario_id == "b":
            raise RuntimeError("simulated worker crash")
        return row

    monkeypatch.setattr(batch, "_run_one_scenario", crash_at_second)
    with pytest.raises(RuntimeError, match="worker crash"):
        _run(args, root, scenarios)
    logs = list((root / "attempts").glob("*/attempt-0/llm_calls.jsonl"))
    assert len(logs) == 2
    before = {path: path.read_bytes() for path in logs}
    monkeypatch.setattr(batch, "_run_one_scenario", _fake_run)
    assert len(_run(args, root, scenarios)) == 2
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert len(list((root / "attempts").glob("*/attempt-*"))) == 3
    recovery = json.loads((root / "recovery.json").read_text())
    assert len(recovery["preserved_incomplete_or_infrastructure_attempts"]) == 1
    assert len(batch._read_jsonl(str(root / "llm_calls.jsonl"))) == 2
    assert called == ["a", "b"]


def test_credit_failure_drains_then_explicit_resume_retains_failed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, root, scenarios = _inputs(tmp_path)

    def no_credits(*values: Any, **kwargs: Any) -> dict[str, Any]:
        row = _fake_run(*values, **kwargs)
        row["score"] = None
        row["metadata"]["has_exception"] = True
        return row

    monkeypatch.setattr(batch, "_run_one_scenario", no_credits)
    with pytest.raises(SystemExit) as failed:
        _run(args, root, scenarios)
    assert failed.value.code == 1
    assert args.pause_file.exists()
    assert len(list((root / "attempts").glob("*/attempt-*"))) == 1
    args.pause_file.unlink()
    monkeypatch.setattr(batch, "_run_one_scenario", _fake_run)
    assert len(_run(args, root, scenarios)) == 2
    assert len(list((root / "attempts").glob("*/attempt-*"))) == 3
    assert batch.verify_artifact_attestation(str(root))[1] == ()


def test_tampered_checkpoint_and_changed_plan_refused_before_paid_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, root, scenarios = _inputs(tmp_path)

    def first(*values: Any, **kwargs: Any) -> dict[str, Any]:
        row = _fake_run(*values, **kwargs)
        args.pause_file.touch()
        return row

    monkeypatch.setattr(batch, "_run_one_scenario", first)
    with pytest.raises(SystemExit):
        _run(args, root, scenarios)
    store = checkpoints.ScenarioCheckpoints(
        root, args.checkpoint_plan, arm="sora", capability="search"
    )
    attempt = store.completed()[("a", 0)]
    args.pause_file.unlink()
    log = attempt / "llm_calls.jsonl"
    original = log.read_bytes()
    log.write_text("modified")
    with pytest.raises(ValueError, match="tampered checkpoint"):
        _run(args, root, scenarios)
    log.write_bytes(original)
    args.checkpoint_plan.write_text('{"frozen": "changed"}')
    with pytest.raises(ValueError, match="mismatched"):
        _run(args, root, scenarios)


def test_atomic_attestation_crash_before_commit_is_restartable_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, root, scenarios = _inputs(tmp_path)
    monkeypatch.setattr(batch, "_run_one_scenario", _fake_run)
    original = batch._atomic_artifact_text

    def fail_at_commit(path: Path, text: str) -> None:
        if path.name == batch.ARTIFACT_ATTESTATION:
            raise OSError("power lost before checkpoint commit")
        original(path, text)

    monkeypatch.setattr(batch, "_atomic_artifact_text", fail_at_commit)
    with pytest.raises(OSError, match="before checkpoint"):
        _run(args, root, scenarios)
    raw = next((root / "attempts").glob("*/attempt-0/llm_calls.jsonl"))
    original_bytes = raw.read_bytes()
    monkeypatch.setattr(batch, "_atomic_artifact_text", original)
    assert len(_run(args, root, scenarios)) == 2
    assert raw.read_bytes() == original_bytes
    assert len(list((root / "attempts").glob("*/attempt-*"))) == 3


def test_offcycle_api_failure_is_retryable_but_model_parser_failure_is_retained() -> None:
    metadata = {
        "terminal_cause": "infrastructure_error",
        "terminal_inference_errors": ["RateLimitError('insufficient_quota')"],
        "trace_id": "unused",
        "judge_recording": "recording",
    }
    row = {"score": 0, "trace_id": "trace", "metadata": metadata}
    assert checkpoints.infrastructure_failure(row)
    metadata["terminal_inference_errors"] = ["ValueError('model returned invalid plan JSON')"]
    assert not checkpoints.infrastructure_failure(row)
    metadata["terminal_inference_errors"] = [
        "ValueError('invalid model plan mentioned RateLimitError')"
    ]
    assert not checkpoints.infrastructure_failure(row)


@pytest.mark.parametrize("seconds", [300, 301])
def test_inference_stall_drains_and_resume_retries_preserving_the_scored_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seconds: int
) -> None:
    args, root, scenarios = _inputs(tmp_path)

    def stall(*values: Any, **kwargs: Any) -> dict[str, Any]:
        row = _fake_run(*values, **kwargs)
        row["metadata"]["terminal_cause"] = "infrastructure_error"
        row["metadata"]["terminal_inference_errors"] = [
            f"inference stalled: no result after {seconds}s"
        ]
        return row

    monkeypatch.setattr(batch, "_run_one_scenario", stall)
    with pytest.raises(SystemExit) as stopped:
        _run(args, root, scenarios)
    assert stopped.value.code == 1
    assert args.pause_file.exists()
    store = checkpoints.ScenarioCheckpoints(
        root, args.checkpoint_plan, arm="sora", capability="search"
    )
    assert store.completed() == {}
    attempt = next((root / "attempts").glob("*/attempt-0"))
    before = batch._artifact_checksums(attempt)
    assert batch._read_jsonl(str(attempt / "output.jsonl"))[0]["score"] == 0
    args.pause_file.unlink()
    monkeypatch.setattr(batch, "_run_one_scenario", _fake_run)
    assert len(_run(args, root, scenarios)) == 2
    assert batch._artifact_checksums(attempt) == before
    assert len(list((root / "attempts").glob("*/attempt-*"))) == 3
    assert attempt not in store.completed().values()


def test_model_parser_quoting_stall_message_is_retained() -> None:
    row = {
        "score": 0,
        "trace_id": "trace",
        "metadata": {
            "judge_recording": "recording",
            "terminal_inference_errors": [
                "ValueError('invalid model plan: inference stalled: no result after 300s')"
            ],
        },
    }
    assert not checkpoints.infrastructure_failure(row)


def test_missing_supervisor_stops_at_boundary_before_any_paid_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, root, scenarios = _inputs(tmp_path)
    args.supervisor_pid = 12345

    def missing(_: int, __: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr("examples.gaia2.checkpoints.os.kill", missing)
    monkeypatch.setattr(
        batch, "_run_one_scenario", lambda *_args, **_kwargs: pytest.fail("should not run")
    )
    with pytest.raises(SystemExit) as stopped:
        _run(args, root, scenarios)
    assert stopped.value.code == checkpoints.PAUSED_EXIT
    assert args.pause_file.exists()
    assert not list((root / "attempts").glob("*/attempt-*"))
