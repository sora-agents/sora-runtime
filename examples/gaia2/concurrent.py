"""Bounded subprocess supervisor for manifest-pinned Gaia2 capability jobs.

Dry run by default. Batch owns scenario execution and immutable paid artifacts; this module owns
job admission, process isolation, full job timing, and completed-job resume.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.gaia2 import batch
from examples.gaia2.checkpoints import PAUSED_EXIT, ScenarioCheckpoints, infrastructure_failure
from examples.gaia2.evaluation.core import (
    canonical_json,
    load_judge_profile,
    load_profiles,
    sha256_file,
)

_ROOT = Path(__file__).resolve().parents[2]
_EVAL = _ROOT / "examples/gaia2/evaluation"
_CANARY = _EVAL / "campaigns/paper2027/canary-not-evidence.json"
_CONFIGS = {
    "gpt-5.4-high-paper": _ROOT / "examples/gaia2/agent.gpt-high.yaml",
    "kimi-k2.5-prompt": _ROOT / "examples/gaia2/agent.kimi.yaml",
    "gpt-5.4-medium-prompt": _ROOT / "examples/gaia2/agent.yaml",
}


@dataclass(frozen=True)
class Job:
    key: str
    profile: str
    arm: str
    capability: str
    directory: Path
    command: tuple[str, ...]
    dependency: str | None = None


def _source_identity(output_dir: Path) -> dict[str, str]:
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=_ROOT)

    output_dir = output_dir.resolve()
    if _ROOT.resolve().is_relative_to(output_dir):
        raise ValueError("output directory must not contain the source worktree")
    tracked = set(git("ls-files", "-z", "--cached").split(b"\0")) - {b""}
    untracked = set(git("ls-files", "-z", "--others", "--exclude-standard").split(b"\0"))
    digest = hashlib.sha256()
    for raw in sorted(tracked | untracked):
        if raw:
            path = _ROOT / os.fsdecode(raw)
            if path.resolve().is_relative_to(output_dir):
                if raw in tracked:
                    raise ValueError(f"output directory overlaps tracked source: {path}")
                # Generated run artifacts change at every admission and checkpoint. Pin their
                # root in the plan, but verify their contents through artifact attestations.
                continue
            digest.update(raw + b"\0")
            digest.update(path.read_bytes() if path.is_file() else b"<deleted>")
            digest.update(b"\0")
    return {
        "revision": git("rev-parse", "HEAD").decode().strip(),
        "tree_sha256": digest.hexdigest(),
    }


def _dependencies() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version}
    for name in (
        "meta-agents-research-environments",
        "openai",
        "litellm",
        "datasets",
        "huggingface-hub",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _build_plan(args: argparse.Namespace) -> tuple[dict[str, Any], list[Job]]:
    manifest = batch._load_sweep_manifest(args.scenario_manifest)
    profiles = load_profiles(args.profiles_path)
    judge = load_judge_profile(args.judge_path)
    if args.workers < 1 or args.workers > 4:
        raise ValueError("--workers must be between 1 and 4 (global across profiles and arms)")
    if args.num_runs < 1 or not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0:
        raise ValueError("--num-runs and --max-wall-seconds must be positive")
    if len(args.profiles) != len(set(args.profiles)) or len(args.arms) != len(set(args.arms)):
        raise ValueError("duplicate profiles or arms")
    if args.execute and "sora" in args.arms and args.prompt_snapshot is None:
        raise ValueError("S-ORA execution requires --prompt-snapshot; dry run can omit it")
    inputs = [
        args.scenario_manifest,
        args.profiles_path,
        args.judge_path,
        _EVAL / "charge_model.json",
    ]
    prompt_digest = None
    if args.prompt_snapshot is not None:
        from examples.gaia2.evaluation.campaigns.prompt.snapshot import (
            load_prompt_snapshot,
            verify_live_prompts,
        )

        prompt_digest = verify_live_prompts(load_prompt_snapshot(args.prompt_snapshot))
        inputs.append(args.prompt_snapshot)
    capabilities = tuple(dict.fromkeys(capability for capability, _ in manifest.cases))
    jobs: list[Job] = []
    for profile_name in args.profiles:
        if profile_name not in profiles or profile_name not in _CONFIGS:
            raise ValueError(f"unsupported profile {profile_name!r}; supported: {list(_CONFIGS)}")
        config = _CONFIGS[profile_name]
        batch._check_operating_point(profiles[profile_name], str(config))
        inputs.append(config)
        output = args.output_dir / profile_name
        # ReAct comes first so the second arm never sees an incomplete counterpart matrix.
        for arm in ("react", "sora"):
            if arm not in args.arms:
                continue
            for capability in capabilities:
                if capability not in batch._CORE_CAPABILITIES:
                    raise ValueError(f"unsupported capability {capability!r}")
                key = f"{profile_name}.{arm}.{capability}"
                dependency = (
                    f"{profile_name}.react.{capability}"
                    if arm == "sora" and "react" in args.arms
                    else None
                )
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "examples.gaia2.batch",
                    "--scenario-manifest",
                    str(args.scenario_manifest),
                    "--hf-dataset",
                    manifest.dataset,
                    "--split",
                    manifest.split,
                    "--capability",
                    capability,
                    "--arm",
                    arm,
                    "--config",
                    str(config),
                    "--profiles-path",
                    str(args.profiles_path),
                    "--output-dir",
                    str(output),
                    "--num-runs",
                    str(args.num_runs),
                    "--generation-free",
                    "--max-wall-seconds",
                    str(args.max_wall_seconds),
                    "--checkpoint-plan",
                    str(args.output_dir / "launch-plan.json"),
                    "--pause-file",
                    str(args.output_dir / "PAUSE"),
                    "--judge-model",
                    judge.model,
                    "--judge-provider",
                    judge.provider,
                ]
                if judge.endpoint:
                    command.extend(["--judge-endpoint", judge.endpoint])
                if not judge.relax_verdict_case:
                    command.append("--strict-verdict-case")
                if arm == "react":
                    command.extend(["--profile", profile_name])
                else:
                    command.append("--isolate-memory")
                    if args.prompt_snapshot is not None:
                        command.extend(["--prompt-snapshot", str(args.prompt_snapshot)])
                jobs.append(
                    Job(
                        key,
                        profile_name,
                        arm,
                        capability,
                        Path(batch._arm_root(str(output), arm)) / "standard" / capability,
                        tuple(command),
                        dependency,
                    )
                )
    plan = {
        "schema_version": 2,
        "output_dir": str(args.output_dir.resolve()),
        "selection": {
            name: str(getattr(args, name))
            if isinstance(getattr(args, name), Path)
            else getattr(args, name)
            for name in (
                "scenario_manifest",
                "profiles_path",
                "judge_path",
                "prompt_snapshot",
                "profiles",
                "arms",
                "workers",
                "num_runs",
                "max_wall_seconds",
            )
        },
        "source": _source_identity(args.output_dir),
        "dependencies": _dependencies(),
        "scenario_manifest_digest": manifest.digest,
        "prompt_snapshot_digest": prompt_digest,
        "clock_mode": "generation_free",
        "verdict_parse": "case-insensitive" if judge.relax_verdict_case else "stock",
        "workers": args.workers,
        "num_runs": args.num_runs,
        "max_wall_seconds": args.max_wall_seconds,
        "expected_scenario_runs": len(manifest.cases)
        * args.num_runs
        * len(args.profiles)
        * len(args.arms),
        "input_files": {str(path.resolve()): sha256_file(path) for path in inputs},
        "jobs": [
            {"key": job.key, "command": list(job.command), "dependency": job.dependency}
            for job in jobs
        ],
    }
    return plan, jobs


def _verify_inputs(plan: dict[str, Any]) -> None:
    if _dependencies() != plan["dependencies"]:
        raise ValueError("installed dependency versions changed since the launch plan was pinned")
    if _source_identity(Path(plan["output_dir"])) != plan["source"]:
        raise ValueError(
            "source changed since the launch plan was pinned; stop and use a new run root"
        )
    for filename, digest in plan["input_files"].items():
        if sha256_file(Path(filename)) != digest:
            raise ValueError(f"pinned input changed: {filename}")


def _inspect_job(job: Job, plan: dict[str, Any], manifest: batch.SweepManifest) -> dict[str, Any]:
    attestation, reasons = batch.verify_artifact_attestation(str(job.directory))
    if reasons or attestation is None:
        raise ValueError(f"{job.key}: invalid artifacts: {'; '.join(reasons)}")
    if attestation.get("arm") != job.arm or attestation.get("capability") != job.capability:
        raise ValueError(f"{job.key}: artifact arm/capability disagrees with launch plan")
    rows = batch._read_jsonl(str(job.directory / "output.jsonl"))
    expected = {
        (case, repeat)
        for case in manifest.scenario_ids(job.capability)
        for repeat in range(plan["num_runs"])
    }
    actual = {
        (row["metadata"].get("scenario_id"), row["metadata"].get("run_number")) for row in rows
    }
    if actual != expected or len(rows) != len(expected):
        raise ValueError(f"{job.key}: incomplete scenario/repeat matrix")
    for row in rows:
        metadata = row["metadata"]
        fields = {
            "scenario_manifest_digest": manifest.digest,
            "model_profile": job.profile,
            "clock_mode": plan["clock_mode"],
            "max_wall_seconds": plan["max_wall_seconds"],
            "verdict_parse": plan["verdict_parse"],
        }
        if job.arm == "sora":
            fields["prompt_snapshot_digest"] = plan["prompt_snapshot_digest"]
        for field, value in fields.items():
            if metadata.get(field) != value:
                raise ValueError(f"{job.key}: recorded {field} disagrees with launch plan")
    calls = batch._read_jsonl(str(job.directory / "llm_calls.jsonl"))
    infra = [row["metadata"]["scenario_id"] for row in rows if infrastructure_failure(row)]
    recovery_path = job.directory / "recovery.json"
    recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else {}

    return {
        "artifact_set_sha256": attestation["artifact_set_sha256"],
        "preserved_incomplete_or_infrastructure_attempts": recovery.get(
            "preserved_incomplete_or_infrastructure_attempts", []
        ),
        "scenario_runs": len(rows),
        "passed": sum(row.get("score") == 1 for row in rows),
        "infrastructure_errors": infra,
        "harness_truncations": sum(bool(row["metadata"].get("harness_truncation")) for row in rows),
        "model_calls": len(calls),
        "round_trips": sum(row.get("round_trips", 0) for row in calls),
        "input_tokens": sum(row.get("input_tokens") or 0 for row in calls),
        "cached_input_tokens": sum(row.get("cached_input_tokens") or 0 for row in calls),
        "output_tokens": sum(row.get("output_tokens") or 0 for row in calls),
        "usage_missing_calls": sum(not row.get("usage_captured", False) for row in calls),
        "output_limit_calls": sum(row.get("finish_reason") == "length" for row in calls),
    }


async def _schedule(
    jobs: list[Job],
    workers: int,
    run_job: Callable[[Job], Coroutine[Any, Any, dict[str, Any]]],
    results: dict[str, dict[str, Any]],
    update: Callable[[], None],
    should_pause: Callable[[], bool] = lambda: False,
) -> None:
    pending = [job for job in jobs if job.key not in results]
    active: dict[asyncio.Task[dict[str, Any]], Job] = {}
    stopped = False
    try:
        while pending or active:
            stopped |= should_pause()
            for job in tuple(pending):
                if stopped or len(active) >= workers:
                    break
                if job.dependency and results.get(job.dependency, {}).get("status") != "completed":
                    continue
                pending.remove(job)
                active[asyncio.create_task(run_job(job))] = job
            if not active:
                break
            done, _ = await asyncio.wait(active, timeout=5, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                job = active.pop(task)
                results[job.key] = task.result()
                stopped |= results[job.key]["status"] != "completed"
                print(f"{job.key}: {results[job.key]['status']}", flush=True)
            update()
    finally:
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(canonical_json(value), encoding="utf-8")
    temporary.replace(path)


def _resource_sample(processes: dict[str, asyncio.subprocess.Process]) -> dict[str, Any]:
    # Process groups include ARE child workers; CPU percent is ps's lifetime average, not a
    # sampling-interval measurement. RSS is summed and can count shared pages more than once.
    groups = {process.pid: key for key, process in processes.items()}
    totals = {key: {"cpu_percent": 0.0, "rss_kib": 0} for key in processes}
    if groups:
        output = subprocess.check_output(["ps", "-axo", "pgid=,%cpu=,rss="], text=True)
        for line in output.splitlines():
            group, cpu, rss = line.split()
            key = groups.get(int(group))
            if key:
                totals[key]["cpu_percent"] += float(cpu)
                totals[key]["rss_kib"] += int(rss)
    return {"jobs": totals, "load_average": os.getloadavg()}


def _reserve_lock(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / ".launcher-lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(descriptor)
        raise ValueError(
            "another supervisor or surviving worker still holds this run root"
        ) from None
    os.ftruncate(descriptor, 0)
    os.write(descriptor, str(os.getpid()).encode())
    return descriptor


def _root_is_fresh(root: Path, lock: Path) -> bool:
    return set(root.iterdir()) == {lock}


async def _terminate(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        os.killpg(process.pid, signal.SIGKILL)
        await process.wait()
    except ProcessLookupError:
        pass


async def _execute(args: argparse.Namespace, plan: dict[str, Any], jobs: list[Job]) -> bool:
    root: Path = args.output_dir
    lock_descriptor = _reserve_lock(root)
    lock = root / ".launcher-lock"
    results: dict[str, dict[str, Any]] = {}
    processes: dict[str, asyncio.subprocess.Process] = {}
    started = time.monotonic()
    invocation_id = uuid.uuid4().hex
    resource_time = 0.0
    manifest = batch._load_sweep_manifest(args.scenario_manifest)
    pause_file = root / "PAUSE"
    try:
        plan_path = root / "launch-plan.json"
        if plan_path.exists():
            if not args.resume or json.loads(plan_path.read_text()) != plan:
                raise ValueError("run root already has a different plan or requires --resume")
            status = root / "launcher-status.json"
            previous = json.loads(status.read_text())["jobs"] if status.exists() else {}
            for job in jobs:
                if (job.directory / batch.ARTIFACT_ATTESTATION).exists():
                    summary = _inspect_job(job, plan, manifest)
                    if summary["infrastructure_errors"]:
                        raise ValueError(f"{job.key}: completed artifact has infrastructure errors")
                    results[job.key] = {
                        **previous.get(job.key, {}),
                        **summary,
                        "status": "completed",
                    }
                elif job.directory.exists() and any(job.directory.iterdir()):
                    store = ScenarioCheckpoints(
                        job.directory, plan_path, arm=job.arm, capability=job.capability
                    )
                    completed = store.completed()
                    expected = {
                        (case, repeat)
                        for case in manifest.scenario_ids(job.capability)
                        for repeat in range(plan["num_runs"])
                    }
                    if set(completed) - expected:
                        raise ValueError(f"{job.key}: checkpoint matrix disagrees with launch plan")
                    print(
                        f"resuming {job.key}: {len(completed)} completed scenarios retained",
                        flush=True,
                    )
            # An explicit resume authorizes restarting only interrupted/infra-failed scenarios.
            pause_file.unlink(missing_ok=True)
        else:
            # Only the lock created above may be present on an initial launch.
            if not _root_is_fresh(root, lock):
                raise ValueError("first launch requires a fresh output root")
            _write_json(plan_path, plan)
        (root / "logs").mkdir(exist_ok=True)

        def update() -> None:
            nonlocal resource_time
            elapsed = time.monotonic() - started
            _write_json(
                root / "launcher-status.json",
                {
                    "elapsed_seconds_this_invocation": elapsed,
                    "invocation_id": invocation_id,
                    "jobs": results,
                    "active_jobs": list(processes),
                    "state": "draining"
                    if pause_file.exists() and processes
                    else "paused"
                    if pause_file.exists()
                    else "running",
                },
            )
            if elapsed - resource_time >= 5:
                try:
                    sample = _resource_sample(processes)
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    sample = {"measurement_error": str(exc)}
                with (root / "resources.jsonl").open("a", encoding="utf-8") as file:
                    file.write(
                        json.dumps(
                            {"invocation_id": invocation_id, "elapsed_seconds": elapsed, **sample}
                        )
                        + "\n"
                    )
                resource_time = elapsed

        async def run_job(job: Job) -> dict[str, Any]:
            job_started = time.monotonic()
            process: asyncio.subprocess.Process | None = None
            try:
                _verify_inputs(plan)
                log_path = root / "logs" / f"{job.key}.log"
                attempt = 1
                while log_path.exists():
                    log_path = root / "logs" / f"{job.key}.attempt-{attempt}.log"
                    attempt += 1
                command = list(job.command)
                if "--checkpoint-plan" in command:
                    command.extend(["--supervisor-pid", str(os.getpid())])
                with log_path.open("x", encoding="utf-8") as log:
                    print(f"starting {job.key}", flush=True)
                    process = await asyncio.create_subprocess_exec(
                        *command,
                        cwd=_ROOT,
                        stdout=log,
                        stderr=asyncio.subprocess.STDOUT,
                        start_new_session=True,
                        pass_fds=(lock_descriptor,),
                        env={
                            **os.environ,
                            "PYTHONPATH": str(_ROOT / "src") + os.pathsep + str(_ROOT),
                        },
                    )
                    processes[job.key] = process
                    count = len(manifest.scenario_ids(job.capability)) * plan["num_runs"]
                    job_limit = 600 + count * (plan["max_wall_seconds"] + 600)
                    try:
                        code = await asyncio.wait_for(process.wait(), timeout=job_limit)
                    except TimeoutError:
                        await _terminate(process)
                        raise ValueError(f"{job.key}: full-job watchdog expired") from None
                if code == PAUSED_EXIT:
                    return {
                        "status": "paused",
                        "elapsed_seconds": time.monotonic() - job_started,
                        "log": str(log_path),
                    }
                if code != 0:
                    raise ValueError(f"batch exited {code}; see logs/{job.key}.log")
                summary = _inspect_job(job, plan, manifest)
                return {
                    **summary,
                    "status": "failed" if summary["infrastructure_errors"] else "completed",
                    "elapsed_seconds": time.monotonic() - job_started,
                    "returncode": code,
                    "log": str(log_path),
                }
            except asyncio.CancelledError:
                if process is not None:
                    await _terminate(process)
                results[job.key] = {
                    "status": "interrupted",
                    "elapsed_seconds": time.monotonic() - job_started,
                }
                raise
            except (OSError, ValueError, RuntimeError) as exc:
                pause_file.touch(exist_ok=True)
                return {
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_seconds": time.monotonic() - job_started,
                }
            finally:
                processes.pop(job.key, None)

        try:
            await _schedule(jobs, args.workers, run_job, results, update, pause_file.exists)
        finally:
            update()
            summary = {
                "invocation_id": invocation_id,
                "expected_jobs": len(jobs),
                "jobs": results,
                "elapsed_seconds_this_invocation": time.monotonic() - started,
                "state": "paused" if pause_file.exists() else "stopped",
                "cost_note": (
                    "Canonical agent tokens cover accepted attempts only. Preserved incomplete/"
                    "infra attempts and judge billing are additional spend."
                ),
            }
            _write_json(root / "launcher-summary.json", summary)
            with (root / "launcher-invocations.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(summary) + "\n")
                file.flush()
                os.fsync(file.fileno())
        return len(results) == len(jobs) and all(
            row["status"] == "completed" for row in results.values()
        )
    finally:
        os.close(lock_descriptor)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-manifest", type=Path, default=_CANARY)
    parser.add_argument("--profiles-path", type=Path, default=_EVAL / "profiles.json")
    parser.add_argument("--judge-path", type=Path, default=_EVAL / "campaigns/prompt/judge.json")
    parser.add_argument("--profiles", nargs="+", default=["gpt-5.4-high-paper", "kimi-k2.5-prompt"])
    parser.add_argument("--arms", nargs="+", choices=["react", "sora"], default=["react", "sora"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--max-wall-seconds", type=float, default=3600)
    parser.add_argument("--prompt-snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--execute", action="store_true", help="Launch paid jobs; otherwise print the plan only."
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Ask an existing sweep to drain active scenarios and pause.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed scenarios; explicitly retry interrupted/infra-failed attempts.",
    )
    raw_args = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(raw_args)
    if args.pause:
        if args.execute or args.resume or not (args.output_dir / "launch-plan.json").is_file():
            parser.error("--pause requires an existing output root and excludes --execute/--resume")
        (args.output_dir / "PAUSE").touch(exist_ok=True)
        print("Pause requested. Wait for launcher exit/paused status before closing the laptop.")
        return
    if args.resume:
        plan_path = args.output_dir / "launch-plan.json"
        if not plan_path.is_file():
            parser.error("--resume requires an existing launch-plan.json")
        stored = json.loads(plan_path.read_text())["selection"]
        for name, value in stored.items():
            if not any(
                arg == "--" + name.replace("_", "-")
                or arg.startswith("--" + name.replace("_", "-") + "=")
                for arg in raw_args
            ):
                if (
                    name in {"scenario_manifest", "profiles_path", "judge_path", "prompt_snapshot"}
                    and value is not None
                ):
                    value = Path(value)
                setattr(args, name, value)
    for name in (
        "scenario_manifest",
        "profiles_path",
        "judge_path",
        "prompt_snapshot",
        "output_dir",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    try:
        plan, jobs = _build_plan(args)
        if not args.execute:
            print(canonical_json(plan), end="")
            return
        profiles = load_profiles(args.profiles_path)
        judge = load_judge_profile(args.judge_path)
        required = {profiles[name].credential_env for name in args.profiles} | {
            judge.credential_env
        }
        missing = sorted(name for name in required if not os.environ.get(name))
        if missing:
            raise ValueError(f"missing credential environment variables: {', '.join(missing)}")
        from examples.gaia2._local_fs import ensure_local_fallback_fs

        path = ensure_local_fallback_fs()
        if path is None or not Path(path).is_dir():
            raise ValueError("local ARE filesystem staging failed; refusing a remote fallback")
        from examples.gaia2._local_fs import _REVISION

        plan["local_filesystem"] = {
            "path": path,
            "revision": os.environ.get("GAIA2_FS_REVISION", _REVISION),
        }
        _verify_inputs(plan)
        if not asyncio.run(_execute(args, plan, jobs)):
            summary_path = args.output_dir / "launcher-summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            if (args.output_dir / "PAUSE").exists() and all(
                row["status"] in {"completed", "paused"} for row in summary.get("jobs", {}).values()
            ):
                print("Paused safely. Resume with --output-dir DIR --execute --resume.")
                return
            raise SystemExit(1)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"concurrent: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "concurrent: interrupted; paid partial artifacts preserved\n")


if __name__ == "__main__":
    main()
