"""Post-run export of replayable Gaia evaluation evidence bundles."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import threading
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

from examples.gaia2.evaluation.campaigns.prompt.capture import BufferedLLMExchangeCapture
from sora import __version__ as sora_version
from sora.diagnostics import RuntimeEventCollector, json_safe

DIAGNOSTIC_SCHEMA_VERSION = 1


class BufferedSessionLog(logging.Handler):
    """A human-readable log retained in memory until the environment has stopped."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self._lock = threading.Lock()
        self._lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = f"{record.created:.6f} {record.levelname} {record.name}: {record.getMessage()}"
            if record.exc_info is not None:
                formatter = self.formatter or logging.Formatter()
                line += "\n" + formatter.formatException(record.exc_info)
            with self._lock:
                self._lines.append(line)
        except BaseException:
            return

    def text(self) -> str:
        with self._lock:
            return "".join(line + "\n" for line in self._lines)


def allocate_attempt_directory(output_dir: Path, entry_key: str, suite: str) -> Path:
    safe_key = entry_key.replace(":", "--").replace("/", "-")
    root = output_dir / "artifacts"
    if suite == "acceptance":
        root /= "acceptance"
    entry_root = root / safe_key
    entry_root.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        candidate = entry_root / f"attempt-{attempt}"
        try:
            candidate.mkdir()
        except FileExistsError:
            attempt += 1
            continue
        return candidate


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(json_safe(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_counts_payload(counts: Any) -> dict[str, Any] | None:
    if counts is None:
        return None
    turns = []
    for turn in counts.turns:
        row = asdict(cast(Any, turn)) if is_dataclass(turn) else json_safe(turn)
        row.update(
            {
                "surplus": turn.surplus,
                "missing": turn.missing,
                "replies_within_band": turn.replies_within_band,
                "passed": turn.passed,
            }
        )
        turns.append(row)
    return {"passed": counts.passed, "turns": turns}


def _verdict_payload(record: Any, result: Any, judge_profile: dict[str, Any]) -> dict[str, Any]:
    recording = result.judge_recording
    event_summary: dict[str, Any] | None = None
    if recording is not None:
        from sora.adapters.are_judge import rescore

        live_parse = recording.verdict_parse
        rescored = rescore(recording, parse=live_parse) if live_parse is not None else None
        event_summary = {
            "events": len(recording.events),
            "accepted": sum(event.verdict is True for event in recording.events),
            "rejected": sum(event.verdict is False for event in recording.events),
            "unparsed": sum(event.verdict is None for event in recording.events),
            "stored_parse_score": rescored.score if rescored is not None else None,
            "disagreements_with_recorded": (
                list(rescored.disagreements_with_recorded) if rescored is not None else None
            ),
        }
    return {
        "score": record.score,
        "passed": record.passed,
        "rationale": getattr(result.outcome, "rationale", None),
        "verdict_parse": (
            result.judge_recording.verdict_parse if result.judge_recording is not None else None
        ),
        "judge_profile": judge_profile,
        "event_match_summary": event_summary,
        "judge_recording": "judge_recording.json",
    }


def _export_trace(
    attempt_dir: Path,
    *,
    result: Any,
    scenario: Any,
    model: str,
    arm: str,
) -> None:
    if result.environment is None:
        return
    from are.simulation.data_handler.exporter import JsonScenarioExporter
    from are.simulation.scenarios.scenario import ScenarioStatus

    decision = (
        ScenarioStatus.Valid.value if result.outcome.success else ScenarioStatus.Invalid.value
    )
    JsonScenarioExporter().export_to_json_file(
        result.environment,
        scenario,
        model_id=model,
        agent_id=arm,
        validation_decision=decision,
        validation_rationale=getattr(result.outcome, "rationale", None),
        run_duration=result.duration,
        output_dir=attempt_dir / "are_trace",
        trace_dump_format="hf",
        scenario_exception=result.exception,
    )


def _index(attempt_dir: Path, errors: dict[str, str]) -> dict[str, Any]:
    files = []
    for path in sorted(p for p in attempt_dir.rglob("*") if p.is_file() and p.name != "index.json"):
        data = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(attempt_dir).as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "files": files,
        "export_errors": {key: errors[key] for key in sorted(errors)},
        "complete": not errors,
    }


def export_attempt(
    attempt_dir: Path,
    *,
    output_dir: Path,
    entry: Any,
    record: Any,
    result: Any,
    scenario: Any,
    config_path: Path,
    profile: Any,
    judge_profile: Any,
    runtime_events: RuntimeEventCollector,
    llm_capture: BufferedLLMExchangeCapture,
    session_log: BufferedSessionLog,
    source_revision: str,
    source_dirty_diff_sha256: str | None,
) -> dict[str, Any]:
    """Export every component independently and return a compact checkpoint reference."""
    errors: dict[str, str] = {}

    def export(name: str, operation: Callable[[], object]) -> None:
        try:
            operation()
        except BaseException as error:
            errors[name] = f"{type(error).__name__}: {error}"

    export(
        "run",
        lambda: _write_json(
            attempt_dir / "run.json",
            {
                "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
                "entry_key": entry.key,
                "arm": entry.arm,
                "profile": entry.profile,
                "suite": entry.suite,
                "capability": entry.capability,
                "case_id": entry.case_id,
                "repeat": entry.repeat,
                "status": record.status,
                "score": record.score,
                "passed": record.passed,
                "terminal_cause": record.terminal_cause,
                "timing": {
                    "duration_seconds": record.duration_seconds,
                    "llm_latency_seconds": record.latency_seconds,
                },
                "scenario": {
                    "id": getattr(scenario, "scenario_id", None),
                    "source": (
                        str(source)
                        if (source := getattr(scenario, "source", None)) is not None
                        else None
                    ),
                },
                "config": str(config_path),
                "provenance": {
                    "source_revision": source_revision,
                    "source_dirty_diff_sha256": source_dirty_diff_sha256,
                    "python": platform.python_version(),
                    "sora": sora_version,
                    "profile": profile.to_dict(),
                    "judge_profile": judge_profile.to_dict(),
                },
                "seeds": {
                    "repeat": entry.repeat,
                    "profile_seed": profile.settings.get("seed").value
                    if "seed" in profile.settings
                    else None,
                },
            },
        ),
    )
    export(
        "trajectory",
        lambda: _write_jsonl(attempt_dir / "trajectory.jsonl", runtime_events.snapshot()),
    )
    export(
        "judge_recording",
        lambda: _write_json(
            attempt_dir / "judge_recording.json",
            result.judge_recording.to_dict(),
        ),
    )
    export(
        "verdict",
        lambda: _write_json(
            attempt_dir / "verdict.json",
            _verdict_payload(record, result, judge_profile.to_dict()),
        ),
    )
    export(
        "write_counts",
        lambda: _write_json(
            attempt_dir / "write_counts.json", _write_counts_payload(result.write_counts)
        ),
    )
    export(
        "llm_calls", lambda: _write_json(attempt_dir / "llm_calls.json", list(record.call_records))
    )
    export("llm", lambda: llm_capture.export(attempt_dir / "llm"))
    export(
        "session_log",
        lambda: (attempt_dir / "session.log").write_text(session_log.text(), encoding="utf-8"),
    )
    export(
        "are_trace",
        lambda: _export_trace(
            attempt_dir,
            result=result,
            scenario=scenario,
            model=profile.model,
            arm=entry.arm,
        ),
    )
    index_payload: dict[str, Any] | None = None
    try:
        index_payload = _index(attempt_dir, errors)
        _write_json(attempt_dir / "index.json", index_payload)
    except BaseException as error:
        errors["index"] = f"{type(error).__name__}: {error}"

    try:
        relative = attempt_dir.relative_to(output_dir).as_posix()
    except ValueError:
        relative = str(attempt_dir)
    index_hash = None
    index_path = attempt_dir / "index.json"
    try:
        if index_path.exists():
            index_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()
    except BaseException as error:
        errors["index_reference"] = f"{type(error).__name__}: {error}"
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "path": relative,
        "index_sha256": index_hash,
        "error": " | ".join(f"{key}: {value}" for key, value in sorted(errors.items())) or None,
    }
