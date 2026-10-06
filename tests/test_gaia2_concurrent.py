"""Offline launcher admission, isolation, provenance, and process lifecycle checks."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from examples.gaia2 import batch, concurrent
from examples.gaia2.evaluation.core import load_profiles, sha256_file, sha256_text


def _args(tmp_path: Path, **overrides: Any) -> Namespace:
    values = {
        "scenario_manifest": concurrent._CANARY,
        "profiles_path": concurrent._EVAL / "profiles.json",
        "judge_path": concurrent._EVAL / "campaigns/prompt/judge.json",
        "profiles": ["gpt-5.4-high-paper", "kimi-k2.5-prompt"],
        "arms": ["react", "sora"],
        "workers": 4,
        "num_runs": 1,
        "max_wall_seconds": 3600.0,
        "prompt_snapshot": None,
        "output_dir": tmp_path / "runs",
        "execute": False,
        "resume": False,
    }
    return Namespace(**{**values, **overrides})


def _artifact_job(job: concurrent.Job, plan: dict[str, Any], *, exception: bool = False) -> None:
    job.directory.mkdir(parents=True)
    manifest = batch._load_sweep_manifest(concurrent._CANARY)
    rows = []
    for scenario_id in manifest.scenario_ids(job.capability):
        trace = job.directory / f"{scenario_id}.json"
        trace.write_text("{}")
        judge = job.directory / f"{scenario_id}.judge.json"
        judge.write_text('{"events": [{}]}')
        rows.append(
            batch._jsonl_record(
                scenario_id=scenario_id,
                run_number=0,
                success=False,
                rationale="agent failure",
                exception=RuntimeError("provider failed") if exception else None,
                trace_id=str(trace),
                judge_recording_path=str(judge),
                model_profile=job.profile,
                clock_mode="generation_free",
                max_wall_seconds=3600,
                verdict_parse=plan["verdict_parse"],
                scenario_manifest_digest=manifest.digest,
                prompt_snapshot_digest=plan["prompt_snapshot_digest"]
                if job.arm == "sora"
                else None,
            )
        )
    batch._write_jsonl(str(job.directory / "output.jsonl"), rows)
    batch._write_jsonl(
        str(job.directory / "llm_calls.jsonl"),
        [
            {
                "input_tokens": 25,
                "cached_input_tokens": 0,
                "output_tokens": 50,
                "round_trips": 1,
                "usage_captured": True,
            }
        ],
    )
    batch.write_artifact_attestation(
        str(job.directory), arm=job.arm, capability=job.capability, records=rows
    )


def test_canary_uses_familiar_ids_and_pins_eight_runs(tmp_path: Path) -> None:
    plan, jobs = concurrent._build_plan(_args(tmp_path))
    assert plan["expected_scenario_runs"] == 8
    assert len(jobs) == 8
    assert {job.profile for job in jobs} == {"gpt-5.4-high-paper", "kimi-k2.5-prompt"}
    assert len({job.directory for job in jobs}) == 8
    for job in jobs:
        assert "--generation-free" in job.command
        if job.arm == "sora":
            assert "--isolate-memory" in job.command
            assert job.dependency == f"{job.profile}.react.{job.capability}"
        else:
            assert job.dependency is None
    familiar = json.loads(
        (concurrent._EVAL / "campaigns/prompt/manifests/familiar.json").read_text()
    )
    assert {case for _, case in batch._load_sweep_manifest(concurrent._CANARY).cases} <= {
        f"scenario_universe_{row['id']}" for row in familiar["cases"]
    }


@pytest.mark.parametrize("workers", [0, 5])
def test_refuses_worker_limit_outside_one_to_four(tmp_path: Path, workers: int) -> None:
    with pytest.raises(ValueError, match="global"):
        concurrent._build_plan(_args(tmp_path, workers=workers))


def test_paid_sora_requires_snapshot_and_mismatch_fails_offline(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prompt-snapshot"):
        concurrent._build_plan(_args(tmp_path, execute=True))
    from examples.gaia2.evaluation.campaigns.prompt.snapshot import build_prompt_snapshot

    snapshot = build_prompt_snapshot(identity="test", source_revision="test", reason="offline test")
    # A valid snapshot of different prompts must fail before credentials or ARE are loaded.
    snapshot["prompts"][0]["system"] += "changed"
    snapshot["prompts"][0]["system_sha256"] = sha256_text(snapshot["prompts"][0]["system"])
    from examples.gaia2.evaluation.campaigns.prompt.snapshot import prompt_rows_digest

    snapshot["prompts_digest"] = prompt_rows_digest(snapshot["prompts"])
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="runtime prompts no longer match"):
        concurrent._build_plan(_args(tmp_path, prompt_snapshot=path))


def test_memory_fresh_for_scenario_repeat_and_keeps_operating_point(tmp_path: Path) -> None:
    original = concurrent._CONFIGS["gpt-5.4-high-paper"]
    configs = [
        batch._isolated_scenario_config(str(original), str(tmp_path), case, repeat)
        for case, repeat in [("a", 0), ("b", 0), ("a", 1)]
    ]
    memory = [yaml.safe_load(Path(config).read_text())["agent"]["memory"] for config in configs]
    assert len({row["procedural"] for row in memory}) == 3
    for config in configs:
        batch._check_operating_point(
            load_profiles(concurrent._EVAL / "profiles.json")["gpt-5.4-high-paper"],
            config,
        )
    with pytest.raises(FileExistsError):
        batch._isolated_scenario_config(str(original), str(tmp_path), "a", 0)
    assert yaml.safe_load(original.read_text())["agent"]["memory"] != memory[0]


async def test_scheduler_fills_four_slots_and_preserves_counterpart_dependency(
    tmp_path: Path,
) -> None:
    _, jobs = concurrent._build_plan(_args(tmp_path))
    active = peak = 0
    completed: set[str] = set()
    results: dict[str, dict[str, Any]] = {}

    async def run(job: concurrent.Job) -> dict[str, Any]:
        nonlocal active, peak
        if job.dependency:
            assert job.dependency in completed
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        completed.add(job.key)
        return {"status": "completed"}

    await concurrent._schedule(jobs, 4, run, results, lambda: None)
    assert peak == 4
    assert len(results) == 8


async def test_failure_stops_new_admissions_but_finishes_active_jobs(tmp_path: Path) -> None:
    _, jobs = concurrent._build_plan(_args(tmp_path))
    admitted = []
    results: dict[str, dict[str, Any]] = {}

    async def run(job: concurrent.Job) -> dict[str, Any]:
        admitted.append(job.key)
        await asyncio.sleep(0.01 if job == jobs[0] else 0.02)
        return {"status": "failed" if job == jobs[0] else "completed"}

    await concurrent._schedule(jobs, 2, run, results, lambda: None)
    assert admitted == [jobs[0].key, jobs[1].key]
    assert len(results) == 2


async def test_real_subprocess_resume_preserves_completed_failed_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, arms=["react"], profiles=["gpt-5.4-high-paper"])
    plan, jobs = concurrent._build_plan(args)
    script = tmp_path / "fake_batch.py"
    # Real subprocess exercises logging/waiting, with token-free fixtures attested in advance.
    script.write_text("print('offline fake worker')\n")
    jobs = [replace(job, command=(sys.executable, str(script))) for job in jobs]
    monkeypatch.setattr(concurrent, "_verify_inputs", lambda _: None)
    for job in jobs:
        _artifact_job(job, plan)
    # Fresh output root still rejects pre-existing artifacts, rather than adopting foreign runs.
    with pytest.raises(ValueError, match="fresh"):
        await concurrent._execute(args, plan, jobs)
    (args.output_dir / "launch-plan.json").write_text(json.dumps(plan))
    args.resume = True
    assert await concurrent._execute(args, plan, jobs)
    summary = json.loads((args.output_dir / "launcher-summary.json").read_text())
    assert all(
        row["passed"] == 0 and row["status"] == "completed" for row in summary["jobs"].values()
    )
    before = {job.key: sha256_file(job.directory / "output.jsonl") for job in jobs}
    assert await concurrent._execute(args, plan, jobs)
    assert before == {job.key: sha256_file(job.directory / "output.jsonl") for job in jobs}
    # One changed paid byte disqualifies the whole resume before admitting work.
    (jobs[0].directory / "llm_calls.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="invalid artifacts"):
        await concurrent._execute(args, plan, jobs)


async def test_actual_workers_have_distinct_processes_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, arms=["react"])
    plan, jobs = concurrent._build_plan(args)
    script = tmp_path / "worker.py"
    script.write_text("import os, time\nprint(os.getpid(), flush=True)\ntime.sleep(0.05)\n")
    jobs = [replace(job, command=(sys.executable, str(script))) for job in jobs]
    monkeypatch.setattr(concurrent, "_verify_inputs", lambda _: None)
    monkeypatch.setattr(concurrent, "_inspect_job", lambda *_: {"infrastructure_errors": []})
    assert await concurrent._execute(args, plan, jobs)
    pids = [int((args.output_dir / "logs" / f"{job.key}.log").read_text()) for job in jobs]
    assert len(set(pids)) == 4
    assert os.getpid() not in pids
    with pytest.raises(ValueError, match="requires --resume"):
        await concurrent._execute(args, plan, jobs)


async def test_cancellation_stops_worker_and_preserves_interrupted_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, arms=["react"], profiles=["gpt-5.4-high-paper"])
    plan, jobs = concurrent._build_plan(args)
    marker = tmp_path / "pid"
    script = tmp_path / "wait.py"
    script.write_text(
        "import os, pathlib, time\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    jobs = [replace(jobs[0], command=(sys.executable, str(script)))]
    monkeypatch.setattr(concurrent, "_verify_inputs", lambda _: None)
    task = asyncio.create_task(concurrent._execute(args, plan, jobs))
    for _ in range(100):
        if marker.exists():
            break
        await asyncio.sleep(0.01)
    assert marker.exists()
    pid = int(marker.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    summary = json.loads((args.output_dir / "launcher-summary.json").read_text())
    assert summary["jobs"][jobs[0].key]["status"] == "interrupted"
    # The stable lock file remains, but its OS lock is automatically released.
    descriptor = concurrent._reserve_lock(args.output_dir)
    os.close(descriptor)
    args.resume = True
    jobs = [replace(jobs[0], command=(sys.executable, "-c", "print('explicit restart')"))]
    monkeypatch.setattr(concurrent, "_inspect_job", lambda *_: {"infrastructure_errors": []})
    assert await concurrent._execute(args, plan, jobs)
    assert len(list((args.output_dir / "logs").glob("*.log"))) == 2
    assert len((args.output_dir / "launcher-invocations.jsonl").read_text().splitlines()) == 2


def test_source_drift_and_input_drift_rejected_before_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _ = concurrent._build_plan(_args(tmp_path))
    monkeypatch.setattr(concurrent, "_source_identity", lambda _: {})
    with pytest.raises(ValueError, match="source changed"):
        concurrent._verify_inputs(plan)
    monkeypatch.setattr(concurrent, "_source_identity", lambda _: plan["source"])
    external = tmp_path / "selection.json"
    external.write_text("{}")
    plan["input_files"][str(external)] = "incorrect"
    with pytest.raises(ValueError, match="pinned input changed"):
        concurrent._verify_inputs(plan)


@pytest.fixture
def source_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    (root / "module.py").write_text("# tracked source\n")
    (root / "code").mkdir()
    (root / "code/module.py").write_text("# nested tracked source\n")
    subprocess.run(["git", "add", "module.py", "code/module.py"], cwd=root, check=True)
    check_output = subprocess.check_output

    def output(command: list[str], **kwargs: Any) -> Any:
        # This temporary index has no commits; source enumeration still uses real git.
        if command == ["git", "rev-parse", "HEAD"] and kwargs.get("cwd") == root:
            return b"offline-test-revision\n"
        return check_output(command, **kwargs)

    monkeypatch.setattr(concurrent, "_ROOT", root)
    monkeypatch.setattr(subprocess, "check_output", output)
    return root


async def test_nonignored_source_output_admits_workers_and_resumes_without_source_drift(
    source_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(source_repo, arms=["react"], profiles=["gpt-5.4-high-paper"])
    script = source_repo / "worker.py"
    script.write_text("print('offline worker')\n")
    plan, jobs = concurrent._build_plan(args)
    jobs = [replace(job, command=(sys.executable, str(script))) for job in jobs]
    monkeypatch.setattr(concurrent, "_inspect_job", lambda *_: {"infrastructure_errors": []})
    assert await concurrent._execute(args, plan, jobs)
    assert len(list((args.output_dir / "logs").glob("*.log"))) == 2
    rebuilt, _ = concurrent._build_plan(args)
    assert rebuilt == plan
    args.resume = True
    assert await concurrent._execute(args, rebuilt, jobs)
    assert len(list((args.output_dir / "logs").glob("*.log"))) == 4
    # Excluding runs must not exclude a sibling with the same prefix or tracked source.
    sibling = source_repo / "runs-source.py"
    sibling.write_text("# new source\n")
    with pytest.raises(ValueError, match="source changed"):
        concurrent._verify_inputs(plan)
    sibling.unlink()
    concurrent._verify_inputs(plan)
    (source_repo / "module.py").write_text("# changed tracked source\n")
    with pytest.raises(ValueError, match="source changed"):
        concurrent._verify_inputs(plan)


@pytest.mark.parametrize("location", ["root", "parent", "tracked"])
def test_output_cannot_exclude_tracked_source_or_entire_worktree(
    source_repo: Path, location: str
) -> None:
    output = {
        "root": source_repo,
        "parent": source_repo.parent,
        "tracked": source_repo / "code",
    }[location]
    with pytest.raises(ValueError, match="output directory"):
        concurrent._build_plan(_args(source_repo, output_dir=output))
    assert not (output / "launch-plan.json").exists()


def test_inspect_rejects_wrong_matrix_and_reports_infrastructure_failure(tmp_path: Path) -> None:
    plan, jobs = concurrent._build_plan(_args(tmp_path))
    job = jobs[0]
    _artifact_job(job, plan, exception=True)
    summary = concurrent._inspect_job(job, plan, batch._load_sweep_manifest(concurrent._CANARY))
    assert summary["infrastructure_errors"]
    assert summary["input_tokens"] == 25
    plan["num_runs"] = 2
    with pytest.raises(ValueError, match="matrix"):
        concurrent._inspect_job(job, plan, batch._load_sweep_manifest(concurrent._CANARY))


async def test_pause_command_drains_real_worker_and_stops_new_admissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, arms=["react"], profiles=["gpt-5.4-high-paper"], workers=1)
    plan, jobs = concurrent._build_plan(args)
    ready = tmp_path / "ready"
    pause = args.output_dir / "PAUSE"
    script = tmp_path / "drain.py"
    script.write_text(
        "import pathlib, time, sys\n"
        f"pathlib.Path({str(ready)!r}).touch()\n"
        f"pause = pathlib.Path({str(pause)!r})\n"
        "for _ in range(200):\n"
        "    if pause.exists():\n"
        "        print('scenario drained', flush=True)\n"
        "        sys.exit(75)\n"
        "    time.sleep(0.01)\n"
        "sys.exit(1)\n"
    )
    jobs = [replace(job, command=(sys.executable, str(script))) for job in jobs]
    monkeypatch.setattr(concurrent, "_verify_inputs", lambda _: None)
    task = asyncio.create_task(concurrent._execute(args, plan, jobs))
    for _ in range(100):
        if ready.exists():
            break
        await asyncio.sleep(0.01)
    assert ready.exists()
    concurrent.main(["--output-dir", str(args.output_dir), "--pause"])
    assert not await task
    status = json.loads((args.output_dir / "launcher-status.json").read_text())
    assert status["state"] == "paused"
    assert status["active_jobs"] == []
    assert list(status["jobs"]) == [jobs[0].key]
    assert status["jobs"][jobs[0].key]["status"] == "paused"


def test_short_resume_restores_original_selection_without_launching_paid_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, arms=["react"], profiles=["gpt-5.4-high-paper"], workers=2)
    plan, _ = concurrent._build_plan(args)
    args.output_dir.mkdir()
    (args.output_dir / "launch-plan.json").write_text(json.dumps(plan))
    # Dry resume must recover the original profile/arm/worker selection rather than defaults.
    import io
    from contextlib import redirect_stdout

    output = io.StringIO()
    with redirect_stdout(output):
        concurrent.main(["--output-dir", str(args.output_dir), "--resume"])
    resumed = json.loads(output.getvalue())
    assert resumed["selection"] == plan["selection"]
    assert resumed["jobs"] == plan["jobs"]
    # An explicit different worker count must be represented, not silently discarded.
    output = io.StringIO()
    with redirect_stdout(output):
        concurrent.main(["--output-dir", str(args.output_dir), "--resume", "--workers=1"])
    assert json.loads(output.getvalue())["workers"] == 1


def test_os_lock_automatically_releases_and_refuses_a_second_supervisor(tmp_path: Path) -> None:
    descriptor = concurrent._reserve_lock(tmp_path)
    try:
        with pytest.raises(ValueError, match="still holds"):
            concurrent._reserve_lock(tmp_path)
    finally:
        os.close(descriptor)
    descriptor = concurrent._reserve_lock(tmp_path)
    os.close(descriptor)


def test_surviving_worker_keeps_os_lock_after_supervisor_closes_its_descriptor(
    tmp_path: Path,
) -> None:
    import subprocess

    descriptor = concurrent._reserve_lock(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.1)"], pass_fds=(descriptor,)
    )
    os.close(descriptor)
    try:
        with pytest.raises(ValueError, match="surviving worker"):
            concurrent._reserve_lock(tmp_path)
    finally:
        process.wait(timeout=2)
    descriptor = concurrent._reserve_lock(tmp_path)
    os.close(descriptor)
