"""Immutable per-scenario attempts beneath a resumable batch capability directory."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from examples.gaia2 import batch
from examples.gaia2.evaluation.core import canonical_json, sha256_file, sha256_text
from examples.gaia2.llm_calls import LLMCallWriter

MARKER = "checkpoint-plan.json"
PAUSED_EXIT = 75


def _provider_failure(errors: list[str]) -> bool:
    # A syntax/contract failure by the model is an agent outcome, even when the runner labels
    # its inference suffix infrastructure_error. Retry transport/provider failures and the runtime's
    # synthetic inference watchdog result; never match error names quoted inside a parser failure.
    provider_classes = {
        "apiconnectionerror",
        "apitimeouterror",
        "ratelimiterror",
        "authenticationerror",
        "internalservererror",
        "serviceunavailableerror",
    }
    credit_markers = (
        "insufficient_quota",
        "insufficient credits",
        "insufficient balance",
        "credit balance",
        "error code: 402",
        "error code: 429",
    )
    for error in errors:
        if re.fullmatch(r"inference stalled: no result after \d+s", error):
            return True
        cause = error.split("(", 1)[0].strip().rsplit(".", 1)[-1].lower()
        if cause in provider_classes:
            return True
        if cause in {"apistatuserror", "badrequesterror", "apierror"} and any(
            marker in error.lower() for marker in credit_markers
        ):
            return True
    return False


def infrastructure_failure(row: dict[str, Any]) -> bool:
    metadata = row["metadata"]
    return bool(
        metadata.get("has_exception")
        or _provider_failure(metadata.get("terminal_inference_errors") or [])
        or metadata.get("harness_truncation")
        or metadata.get("timeline_expired")
        or row.get("score") is None
        or not row.get("trace_id")
        or not metadata.get("judge_recording")
    )


def _durable_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        file.write(canonical_json(value))
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)
    # Make the rename durable as well as the contents on a local POSIX filesystem.
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ScenarioCheckpoints:
    def __init__(self, directory: Path, plan_path: Path, *, arm: str, capability: str) -> None:
        self.directory = directory
        self.identity = {
            "plan_sha256": sha256_file(plan_path),
            "arm": arm,
            "capability": capability,
        }

    def verify(self) -> None:
        if batch._contamination_marker(str(self.directory)):
            raise ValueError(f"contaminated checkpoint root: {self.directory}")
        if not self.directory.exists() or not any(self.directory.iterdir()):
            return
        marker = self.directory / MARKER
        if not marker.is_file() or json.loads(marker.read_text()) != self.identity:
            raise ValueError(f"unknown or mismatched scenario checkpoints in {self.directory}")
        allowed = {
            MARKER,
            "attempts",
            "output.jsonl",
            "llm_calls.jsonl",
            "recovery.json",
            "output.tmp",
            "llm_calls.tmp",
            "recovery.tmp",
            batch.ARTIFACT_ATTESTATION,
            batch.ARTIFACT_CHECKSUMS,
            batch.ARTIFACT_CHECKSUMS + ".tmp",
            batch.ARTIFACT_ATTESTATION + ".tmp",
        }
        if any(path.name not in allowed for path in self.directory.iterdir()):
            raise ValueError(f"unknown files in checkpoint root: {self.directory}")

    def prepare(self) -> None:
        self.verify()
        if (self.directory / batch.ARTIFACT_ATTESTATION).exists():
            raise ValueError(f"completed capability is immutable: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)
        marker = self.directory / MARKER
        if not marker.exists():
            _durable_json(marker, self.identity)

    def completed(self) -> dict[tuple[str, int], Path]:
        self.verify()
        accepted: dict[tuple[str, int], Path] = {}
        for attempt in sorted((self.directory / "attempts").glob("*/attempt-*")):
            if not (attempt / batch.ARTIFACT_ATTESTATION).exists():
                # A killed worker may have created any prefix of its files. Keep it as evidence,
                # but never reconstruct a result from an unattested response or truncated row.
                continue
            _, reasons = batch.verify_artifact_attestation(str(attempt))
            if reasons:
                raise ValueError(f"tampered checkpoint {attempt}: {'; '.join(reasons)}")
            record = json.loads((attempt / "checkpoint.json").read_text())
            if record.get("identity") != self.identity:
                raise ValueError(f"checkpoint identity mismatch: {attempt}")
            rows = batch._read_jsonl(str(attempt / "output.jsonl"))
            if len(rows) != 1:
                raise ValueError(f"checkpoint must contain one scenario result: {attempt}")
            row = rows[0]
            key = (str(row["metadata"]["scenario_id"]), int(row["metadata"]["run_number"]))
            if record.get("scenario_id") != key[0] or record.get("run_number") != key[1]:
                raise ValueError(f"checkpoint scenario mismatch: {attempt}")
            if record.get("usable") != (not infrastructure_failure(row)):
                raise ValueError(f"checkpoint completion status mismatch: {attempt}")
            if record["usable"]:
                if key in accepted:
                    raise ValueError(f"duplicate completed checkpoint for {key}")
                accepted[key] = attempt
        return accepted

    def next_attempt(self, scenario_id: str, run_number: int) -> Path:
        key = f"{sha256_text(scenario_id)[:16]}.run{run_number}"
        parent = self.directory / "attempts" / key
        parent.mkdir(parents=True, exist_ok=True)
        index = 0
        while (parent / f"attempt-{index}").exists():
            index += 1
        attempt = parent / f"attempt-{index}"
        attempt.mkdir()
        _durable_json(
            attempt / "started.json",
            {
                "identity": self.identity,
                "scenario_id": scenario_id,
                "run_number": run_number,
            },
        )
        return attempt

    def finish_attempt(self, attempt: Path, row: dict[str, Any]) -> bool:
        usable = not infrastructure_failure(row)
        _durable_json(
            attempt / "checkpoint.json",
            {
                "identity": self.identity,
                "scenario_id": row["metadata"]["scenario_id"],
                "run_number": row["metadata"]["run_number"],
                "usable": usable,
            },
        )
        # The record becomes reusable only when the attestation pair is fully written. If the
        # process dies between those two writes, resume refuses rather than trusting partial proof.
        for path in batch._artifact_paths(attempt):
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        batch.write_artifact_attestation(
            str(attempt),
            arm=self.identity["arm"],
            capability=self.identity["capability"],
            records=[row],
        )
        return usable

    def publish(
        self, accepted: dict[tuple[str, int], Path], order: list[tuple[str, int]]
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        for key in order:
            if key in accepted:
                attempt = accepted[key]
                rows.extend(batch._read_jsonl(str(attempt / "output.jsonl")))
                calls.extend(batch._read_jsonl(str(attempt / "llm_calls.jsonl")))
        for name, values in (("output", rows), ("llm_calls", calls)):
            temporary = self.directory / f"{name}.tmp"
            batch._write_jsonl(str(temporary), values)
            temporary.replace(self.directory / f"{name}.jsonl")
        selected = set(accepted.values())
        discarded = [
            str(path)
            for path in sorted((self.directory / "attempts").glob("*/attempt-*"))
            if path not in selected
        ]
        _durable_json(
            self.directory / "recovery.json",
            {
                "completed_scenario_runs": len(accepted),
                "preserved_incomplete_or_infrastructure_attempts": discarded,
                "canonical_call_log": (
                    "Accepted attempts only; other paid calls remain in the preserved "
                    "attempt directories."
                ),
            },
        )
        return rows


def run_checkpointed(
    args: Any,
    directory: Path,
    first_scenarios: Any,
    scenarios_for_run: Callable[[], Any],
    *,
    record_judge: bool,
) -> list[dict[str, Any]]:
    store = ScenarioCheckpoints(
        directory, args.checkpoint_plan, arm=args.arm, capability=args.capability
    )
    store.prepare()
    accepted = store.completed()
    order = [
        (case, repeat)
        for repeat in range(args.num_runs)
        for case in args.sweep_manifest.scenario_ids(args.capability)
    ]
    if set(accepted) - set(order):
        raise ValueError("checkpoint scenario/repeat matrix does not match this manifest")
    pause_file: Path = args.pause_file
    rows = store.publish(accepted, order)
    for repeat in range(args.num_runs):
        scenarios = first_scenarios if repeat == 0 else scenarios_for_run()
        for scenario, _events in scenarios:
            key = (str(scenario.scenario_id), repeat)
            if key in accepted:
                print(f"checkpoint: keeping {key[0]} run {repeat}", flush=True)
                continue
            supervisor = getattr(args, "supervisor_pid", None)
            if supervisor is not None:
                try:
                    os.kill(supervisor, 0)
                except ProcessLookupError:
                    pause_file.touch(exist_ok=True)
            if pause_file.exists():
                raise SystemExit(PAUSED_EXIT)
            scenario.run_number = repeat
            attempt = store.next_attempt(*key)
            with LLMCallWriter(attempt / "llm_calls.jsonl", reset=True, durable=True) as writer:
                row = batch._run_one_scenario(
                    scenario,
                    repeat,
                    args,
                    str(attempt),
                    record_judge=record_judge,
                    llm_calls=writer,
                )
            batch._write_jsonl(str(attempt / "output.jsonl"), [row])
            usable = store.finish_attempt(attempt, row)
            if not usable:
                # Shared drain marker stops other workers at their next scenario boundary too.
                pause_file.touch(exist_ok=True)
                raise SystemExit(1)
            accepted[key] = attempt
            rows = store.publish(accepted, order)
            print(
                f"[{args.capability}] checkpointed {key[0]} run {repeat}: "
                f"{row['metadata']['status']}",
                flush=True,
            )
    batch.write_artifact_attestation(
        str(directory), arm=args.arm, capability=args.capability, records=rows
    )
    return rows
