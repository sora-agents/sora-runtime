from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import yaml

from examples.gaia2.evaluation.campaigns.prompt.contracts import run_contract_suite
from examples.gaia2.evaluation.campaigns.prompt.neutral import (
    NEUTRAL_CASES,
    run_neutral_suite,
)
from examples.gaia2.evaluation.campaigns.prompt.reporting import build_report
from examples.gaia2.evaluation.campaigns.prompt.snapshot import (
    PROMPT_SOURCES,
    build_campaign_configuration,
    build_prompt_snapshot,
    load_campaign_configuration,
    load_prompt_snapshot,
    verify_live_prompts,
)
from examples.gaia2.evaluation.core import (
    GAIA_SUITES,
    BudgetPolicy,
    CallUsage,
    ChargeModelSheet,
    EvaluationRecord,
    JudgeProfile,
    ManifestLockedError,
    ModelProfile,
    PriceSheet,
    RunMatrix,
    RunSelection,
    build_run_matrix,
    calculate_call_cost,
    canonical_json,
    load_judge_profile,
    load_manifests,
    load_profiles,
    record_from_dict,
    resolve_scenario,
    sha256_file,
    sha256_text,
)

EVAL_ROOT = Path(__file__).parent
PROMPT_ROOT = EVAL_ROOT / "campaigns" / "prompt"
BASELINE_PROMPT_SNAPSHOT = PROMPT_ROOT / "snapshots" / "pre-optimization-control.json"
CAMPAIGN_CONFIGURATION = PROMPT_ROOT / "campaign.json"
DEFAULT_SCENARIO_ROOT = Path("examples/gaia2/scenarios")
DEFAULT_PRICE_SHEET = EVAL_ROOT / "price_sheets" / "2026-09-12.json"


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _source_revision() -> str:
    return _git_output("rev-parse", "HEAD") or "unknown"


def _exclude_pathspec(root_raw: str | None, excluded: set[Path]) -> list[str]:
    """``--`` plus a ``:(exclude)`` pathspec per path, or nothing when there is none to exclude.

    A pathspec list made only of exclusions means "everything else" to git, so no positive entry
    is needed alongside them. ``top`` anchors each path at the repo root: without it git resolves
    against the process CWD, and a pathspec that matches nothing still means "everything else", so
    the file comes back into the diff rather than the command failing. The only caller already
    invokes git at the root, which would make it redundant — it is kept so the helper's output is
    correct on its own terms, not conditionally on where the caller happens to run git."""
    if not root_raw or not excluded:
        return []
    root = Path(root_raw).resolve()
    relatives: list[str] = []
    for path in sorted(excluded):
        try:
            relatives.append(path.relative_to(root).as_posix())
        except ValueError:  # outside the repo: nothing in the diff could name it anyway
            continue
    return ["--", *(f":(exclude,top){relative}" for relative in relatives)] if relatives else []


def _dirty_diff_hash(*, excluded: set[Path] | None = None) -> str | None:
    root_raw = _git_output("rev-parse", "--show-toplevel")
    # Both content commands are anchored at the toplevel rather than run in the process CWD: `git
    # ls-files --others` reports only the subtree it is run from, and names it relative to that, so
    # invoked from anywhere but the root this hash would silently cover a slice of the tree.
    at_root = ("-C", root_raw) if root_raw else ()
    untracked_raw = _git_output(*at_root, "ls-files", "--others", "--exclude-standard")
    excluded_resolved = {path.resolve() for path in excluded or set()}
    # The exclusion has to reach `git diff`, not just the untracked scan below: the snapshot this
    # hash is written into is itself a tracked file, so covering its own diff would mean the value
    # is stale the moment it is stored, and two regenerations with no edit between them would
    # disagree. It never converges while the snapshot is uncommitted — which, here, is its normal
    # reviewable state.
    diff = _git_output(
        *at_root, "diff", "--binary", "HEAD", *_exclude_pathspec(root_raw, excluded_resolved)
    )
    additions: list[dict[str, str]] = []
    if root_raw and untracked_raw:
        root = Path(root_raw)
        for relative in untracked_raw.splitlines():
            path = root / relative
            resolved = path.resolve()
            if any(resolved == item or resolved.is_relative_to(item) for item in excluded_resolved):
                continue
            if not path.is_file():
                continue
            additions.append({"path": relative, "sha256": sha256_file(path)})
    if not diff and not additions:
        return None
    return sha256_text(canonical_json({"tracked_diff": diff or "", "untracked": additions}))


def _prompt_source_dirty_diff_hash() -> str | None:
    source_paths = sorted(
        {source[0] for source in PROMPT_SOURCES.values()} | {"src/sora/memory.py"}
    )
    root_raw = _git_output("rev-parse", "--show-toplevel")
    at_root = ("-C", root_raw) if root_raw else ()
    diff = _git_output(*at_root, "diff", "--binary", "HEAD", "--", *source_paths)
    untracked_raw = _git_output(
        *at_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        *source_paths,
    )
    additions: list[dict[str, str]] = []
    if root_raw and untracked_raw:
        root = Path(root_raw)
        additions = [
            {"path": relative, "sha256": sha256_file(root / relative)}
            for relative in untracked_raw.splitlines()
        ]
    if not diff and not additions:
        return None
    return sha256_text(canonical_json({"tracked_diff": diff or "", "untracked": additions}))


def _snapshot_command(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    snapshot = build_prompt_snapshot(
        identity=args.identity,
        source_revision=_source_revision(),
        reason=args.reason,
        prompt_source_dirty_diff_sha256=_prompt_source_dirty_diff_hash(),
    )
    if args.output:
        assert output is not None
        _write_prompt_snapshot(output, snapshot)
        print(f"wrote prompt snapshot: {args.output}")
    else:
        print(canonical_json(snapshot), end="")
    return 0


def _write_prompt_snapshot(output: Path, snapshot: dict[str, Any]) -> None:
    identity = snapshot.get("identity")
    if not isinstance(identity, str) or output.stem != identity:
        raise ValueError("prompt snapshot filename must equal its immutable identity")
    rendered = canonical_json(snapshot)
    if output.exists() and output.read_text() != rendered:
        raise ValueError(f"prompt snapshot identity {identity!r} already exists and is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)


def _configuration_command(args: argparse.Namespace) -> int:
    configuration = canonical_json(build_campaign_configuration(root=EVAL_ROOT))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(configuration)
    print(f"wrote campaign configuration: {output}")
    return 0


def _check_scenario_availability(root: Path, manifests: dict[str, Any]) -> tuple[list[str], int]:
    missing: list[str] = []
    available = 0
    for manifest in manifests.values():
        for case in manifest.cases:
            matches = list((root / case.capability).glob(f"*{case.case_id}*.json"))
            if len(matches) != 1:
                missing.append(f"{manifest.suite}/{case.capability}/{case.case_id}")
            else:
                available += 1
    return missing, available


def _check_command(args: argparse.Namespace) -> int:
    profiles = load_profiles(EVAL_ROOT / "profiles.json")
    judge_profile = load_judge_profile(PROMPT_ROOT / "judge.json")
    manifests = load_manifests(PROMPT_ROOT / "manifests")
    sheets = [PriceSheet.load(path) for path in sorted((EVAL_ROOT / "price_sheets").glob("*.json"))]
    if not sheets:
        raise ValueError("no dated price sheet is available")
    for profile in profiles.values():
        if not any(profile.endpoint_identity() in sheet.endpoints for sheet in sheets):
            raise ValueError(f"no dated price sheet covers profile endpoint {profile.name}")
    frozen = load_prompt_snapshot(BASELINE_PROMPT_SNAPSHOT)
    verify_live_prompts(frozen)
    campaign_configuration = load_campaign_configuration(CAMPAIGN_CONFIGURATION)
    expected_configuration = build_campaign_configuration(root=EVAL_ROOT)
    if campaign_configuration != expected_configuration:
        raise ValueError("campaign.json no longer mirrors the live campaign configuration")
    expected_profiles = [profiles[name].to_dict() for name in sorted(profiles)]
    if campaign_configuration.get("evaluation_profiles") != expected_profiles:
        raise ValueError("the campaign evaluation profiles no longer match profiles.json")
    if campaign_configuration.get("judge_profile") != judge_profile.to_dict():
        raise ValueError("the campaign judge profile no longer matches judge.json")
    charge_model = ChargeModelSheet.load(EVAL_ROOT / "charge_model.json")
    if campaign_configuration.get("charge_model") != charge_model.to_dict():
        raise ValueError("the campaign charge model no longer matches charge_model.json")
    for profile in profiles.values():
        # Every profile a campaign can run has to be chargeable, or the arm silently falls back to
        # measured wall clock and stops being comparable with the arms that were charged.
        charge_model.for_profile(profile)
    contract = run_contract_suite()
    neutral = run_neutral_suite()
    if contract.failed or neutral.failed:
        raise ValueError(
            f"offline suites failed: contract={contract.failures}, neutral={neutral.failures}"
        )
    try:
        first = manifests["acceptance"].cases[0]
        resolve_scenario(
            Path(args.scenario_root),
            "acceptance",
            first.capability,
            first.case_id,
        )
    except ManifestLockedError:
        pass
    else:
        raise ValueError("acceptance payload resolver did not enforce its lock")
    missing, available = _check_scenario_availability(Path(args.scenario_root), manifests)
    if missing and args.require_scenarios:
        raise FileNotFoundError("missing ignored scenarios: " + ", ".join(missing))
    # Exercise report serialization and its schema without any scenario/model content. The
    # frozen snapshot is passed in the shape a real report declares, so this check covers the
    # same field a paired report is read through rather than a parallel one kept alive for it.
    report = build_report(
        [],
        expected_prompt_snapshots={
            "baseline": {
                "identity": str(frozen["identity"]),
                "digest": str(frozen["prompts_digest"]),
            }
        },
        judge_profile=judge_profile.to_dict(),
    )
    json.loads(canonical_json(report))
    print(
        f"check passed: {len(profiles)} profiles, {len(manifests)} Gaia manifests, "
        f"{contract.passed} contract cases, {neutral.passed} neutral cases, "
        f"{available}/15 ignored scenario files available"
    )
    if missing:
        print("scenario files unavailable (live runs only): " + ", ".join(missing))
    print("acceptance payloads remained locked and unopened")
    return 0


def _matrix_dict(
    matrix: RunMatrix,
    completed: set[str] | None = None,
    *,
    max_agent_llm_calls: int,
    judge_profile: JudgeProfile,
) -> dict[str, Any]:
    completed = completed or set()
    return {
        "gaia_runs": matrix.gaia_runs,
        "prior_gaia_runs": matrix.prior_gaia_runs,
        "cumulative_gaia_runs": matrix.prior_gaia_runs + matrix.gaia_runs,
        "agent_reserve": matrix.agent_reserve,
        "judge_reserve": matrix.judge_reserve,
        "total_reserve": matrix.total_reserve,
        "prior_spend": matrix.prior_spend,
        "cumulative_projected_spend": matrix.prior_spend + matrix.total_reserve,
        "run_limits": {
            "max_agent_llm_calls": max_agent_llm_calls,
            "unit": "logical_agent_llm_call",
        },
        "judge_profile": judge_profile.to_dict(),
        "entries": [
            asdict(entry)
            | {
                "matrix_key": entry.key,
                "checkpoint_status": "complete" if entry.key in completed else "pending",
            }
            for entry in matrix.entries
        ],
    }


def _write_profile_config(
    output_dir: Path,
    profile: ModelProfile,
    entry: Any,
    *,
    synthetic: bool = False,
    max_agent_llm_calls: int,
) -> Path:
    source = yaml.safe_load(Path("examples/gaia2/agent.yaml").read_text())
    source["agent"]["name"] = f"gaia2-eval-{profile.name}"
    source["agent"]["llm"] = profile.client_settings()
    source["agent"]["llm"]["max_logical_calls"] = max_agent_llm_calls
    state_root = (
        output_dir
        / "memory"
        / entry.arm
        / entry.profile
        / entry.suite
        / entry.case_id
        / str(entry.repeat)
    ).resolve()
    source["agent"]["memory"] = {
        kind: f"file://{state_root / kind}" for kind in ("semantic", "procedural", "episodic")
    }
    if synthetic:
        source["agent"].pop("transport", None)
        source["agent"]["workspaces"] = [
            {
                "origin": {
                    "adapter": "prompt-eval-synthetic",
                    "address": f"memory://prompt-eval-synthetic/{entry.case_id}",
                },
                "workspace_id": "prompt-eval-synthetic",
                "factory": "examples.gaia2.evaluation.campaigns.prompt.synthetic.make_adapter",
            }
        ]
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    safe_key = entry.key.replace(":", "-")
    path = config_dir / f"{safe_key}.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=False))
    return path


def _checkpoint_records(path: Path) -> tuple[set[str], list[EvaluationRecord]]:
    keys: set[str] = set()
    records: list[EvaluationRecord] = []
    if not path.exists():
        return keys, records
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        record = record_from_dict(row["record"])
        if record.completes_matrix_entry:
            keys.add(str(row["matrix_key"]))
        records.append(record)
    return keys, records


def _append_checkpoint(path: Path, matrix_key: str, record: EvaluationRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = record.to_dict(detailed_acceptance=True)
    if record.suite in GAIA_SUITES and record.diagnostics is not None:
        relative = record.diagnostics.get("path")
        if isinstance(relative, str):
            artifact = Path(relative)
            calls_path = (
                artifact if artifact.is_absolute() else path.parent / artifact
            ) / "llm_calls.json"
            try:
                exported_calls = json.loads(calls_path.read_text(encoding="utf-8"))
                stored_calls = json.loads(json.dumps(stored["call_records"]))
            except (OSError, TypeError, ValueError):
                pass
            else:
                if isinstance(exported_calls, list) and exported_calls == stored_calls:
                    stored["call_records"] = []
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"matrix_key": matrix_key, "record": stored}))
        handle.write("\n")
        handle.flush()


def _offline_record(entry: Any) -> EvaluationRecord:
    if entry.suite == "contract":
        result = run_contract_suite()
        return EvaluationRecord(
            arm=entry.arm,
            profile=entry.profile,
            suite=entry.suite,
            capability=entry.capability,
            case_id=entry.case_id,
            repeat=entry.repeat,
            score=1.0 if result.failed == 0 else 0.0,
            passed=result.failed == 0,
            contract_failures=result.failed,
        )
    case = next(case for case in NEUTRAL_CASES if case.case_id == entry.case_id)
    result = run_neutral_suite()
    passed = case.case_id not in result.failures
    return EvaluationRecord(
        arm=entry.arm,
        profile=entry.profile,
        suite=entry.suite,
        capability=case.topic,
        case_id=case.case_id,
        repeat=entry.repeat,
        score=1.0 if passed else 0.0,
        passed=passed,
        contract_failures=0 if passed else 1,
    )


def _call_records_and_cost(
    llm_report: Any, profile: ModelProfile, sheet: PriceSheet
) -> tuple[tuple[dict[str, Any], ...], float | None, bool]:
    if llm_report is None:
        return (), 0.0, True
    rows: list[dict[str, Any]] = []
    total = 0.0
    upper_bound = False
    usage_complete = True
    for call in llm_report.inferences:
        usages = tuple(getattr(call, "usages", ()))
        call_usage_complete = len(usages) == call.round_trips
        usage_complete = usage_complete and call_usage_complete
        round_trip_usage: list[dict[str, Any]] = []
        call_costs = []
        for usage in usages:
            cost = calculate_call_cost(
                sheet,
                profile,
                CallUsage(
                    input_tokens=usage.input_tokens,
                    cached_input_tokens=usage.cached_input_tokens,
                    output_tokens=usage.output_tokens,
                    reasoning_tokens=usage.reasoning_tokens,
                    cache_write_input_tokens=getattr(usage, "cache_write_input_tokens", None),
                ),
            )
            call_costs.append(cost)
            total += cost.agent_cost
            upper_bound = upper_bound or cost.upper_bound
            round_trip_usage.append(
                {
                    "input_tokens": usage.input_tokens,
                    "cached_input_tokens": usage.cached_input_tokens,
                    "cache_write_input_tokens": getattr(usage, "cache_write_input_tokens", None),
                    "output_tokens": usage.output_tokens,
                    "reasoning_tokens": usage.reasoning_tokens,
                    "reasoning_tokens_exact": usage.reasoning_tokens_exact,
                    "cost": asdict(cost),
                }
            )
        prompt_hashes = [
            {
                "system_sha256": prompt.system_sha256,
                "user_sha256": prompt.user_sha256,
            }
            for prompt in getattr(call, "prompt_hashes", ())
        ]
        aggregate_cost = (
            {
                "agent_cost": sum(cost.agent_cost for cost in call_costs),
                "uncached_input_cost": sum(cost.uncached_input_cost for cost in call_costs),
                "cached_input_cost": sum(cost.cached_input_cost for cost in call_costs),
                "output_cost": sum(cost.output_cost for cost in call_costs),
                "upper_bound": any(cost.upper_bound for cost in call_costs),
                "tier_max_input_tokens": None,
            }
            if call_usage_complete
            else None
        )
        rows.append(
            {
                "call_id": getattr(call, "call_id", None),
                "inference_id": getattr(call, "inference_id", None),
                "semantic_label": call.semantic_label,
                "prompt_version": call.prompt_version,
                "prompt_hashes": {
                    "system_sha256": call.system_prompt_sha256,
                    "user_sha256": call.user_prompt_sha256,
                    "round_trips": prompt_hashes,
                },
                "settings": {name: asdict(setting) for name, setting in profile.settings.items()},
                "profile_fields": profile.reported_fields(),
                "finish_reasons": list(call.finish_reasons),
                "retries_visible_to_runtime": max(0, call.round_trips - 1),
                "round_trips": call.round_trips,
                "round_trip_usage": round_trip_usage,
                "usage_available": call_usage_complete,
                "cached_input_tokens": call.cached_input_tokens,
                "cache_write_input_tokens": getattr(call, "cache_write_input_tokens", None),
                "uncached_input_tokens": (
                    call.input_tokens
                    if call.cache_unknown_input_tokens
                    else call.input_tokens - call.cached_input_tokens
                ),
                "cache_reporting_unavailable": bool(call.cache_unknown_input_tokens),
                "output_tokens": call.output_tokens,
                "reasoning_tokens": call.reasoning_tokens,
                "reasoning_tokens_exact": call.reasoning_tokens_exact,
                "observed_models": list(call.observed_models),
                "sdk_observations": [
                    {"name": name, "version": version} for name, version in call.sdk_observations
                ],
                "provider_observations": list(call.provider_observations),
                "latency_seconds": call.latency_seconds,
                "outcome": getattr(call, "outcome", None),
                "error": getattr(call, "error", None),
                "discarded": call.discarded,
                "cost": aggregate_cost,
                "terminal_parse_failures": getattr(call, "terminal_parse_failures", 0),
            }
        )
    return tuple(rows), total if usage_complete else None, upper_bound


def _arm_live_judge_recording() -> bool:
    from sora.adapters.are_judge import arm_judge_recording, judge_recording_armed

    arm_judge_recording()
    return judge_recording_armed()


def _live_gaia_record(
    entry: Any,
    *,
    scenario_root: Path,
    ack_locked_acceptance: bool,
    config_path: Path,
    profile: ModelProfile,
    sheet: PriceSheet,
    judge_profile: JudgeProfile,
    max_wall_seconds: float,
    max_agent_llm_calls: int,
    output_dir: Path,
    charge_sheet: ChargeModelSheet | None = None,
    wall_clock: bool = False,
    diagnostics: bool = False,
    source_revision: str = "unknown",
    source_dirty_diff_sha256: str | None = None,
) -> EvaluationRecord:
    # Every ARE/provider import is below all dry-run and budget gates.
    from examples.gaia2._runner import run_scenario
    from sora.adapters.are_sim import attach_judge, load_scenario

    path = resolve_scenario(
        scenario_root,
        entry.suite,
        entry.capability,
        entry.case_id,
        ack_locked_acceptance=ack_locked_acceptance,
    )
    scenario = load_scenario(str(path))
    attempt_dir: Path | None = None
    runtime_events: Any = None
    llm_capture: Any = None
    session_log: Any = None
    diagnostic_setup_error: str | None = None
    if diagnostics:
        from examples.gaia2.evaluation.campaigns.prompt.capture import (
            BufferedLLMExchangeCapture,
        )
        from examples.gaia2.evaluation.diagnostics import (
            BufferedSessionLog,
            allocate_attempt_directory,
        )
        from sora.diagnostics import RuntimeEventCollector

        try:
            attempt_dir = allocate_attempt_directory(output_dir, entry.key, entry.suite)
            runtime_events = RuntimeEventCollector()
            llm_capture = BufferedLLMExchangeCapture()
            session_log = BufferedSessionLog()
            if not _arm_live_judge_recording():
                raise RuntimeError("ARE judge recording could not be armed")
        except BaseException as error:
            diagnostic_setup_error = f"{type(error).__name__}: {error}"
            # An allocated directory may remain as an intentionally discoverable orphan, but a
            # partially armed collector must not alter how the paid attempt itself is run.
            attempt_dir = None
    attach_judge(
        scenario,
        model=judge_profile.model,
        provider=judge_profile.provider,
        endpoint=judge_profile.endpoint,
        offline_validation=judge_profile.offline_validation,
        relax_verdict_case=judge_profile.relax_verdict_case,
    )
    charge_sheet = charge_sheet or ChargeModelSheet.load(EVAL_ROOT / "charge_model.json")
    charge = None if wall_clock else charge_sheet.charge_for(profile)
    with ExitStack() as diagnostic_context:
        sora_logger = None
        previous_level = None
        if attempt_dir is not None:
            from sora.diagnostics import collect_runtime_events
            from sora.llm import capture_llm_exchanges

            diagnostic_context.enter_context(collect_runtime_events(runtime_events))
            diagnostic_context.enter_context(capture_llm_exchanges(llm_capture))
            import logging

            sora_logger = logging.getLogger("sora")
            previous_level = sora_logger.level
            sora_logger.setLevel(logging.DEBUG)
            sora_logger.addHandler(session_log)
        try:
            result = run_scenario(
                scenario,
                config=str(config_path),
                max_wall_seconds=max_wall_seconds,
                read_stdin=False,
                charge=charge,
                charge_model_identity=charge.identity if charge is not None else None,
                charge_model_digest=charge_sheet.digest if charge is not None else None,
                record_judge=attempt_dir is not None,
                verdict_parse=("case-insensitive" if judge_profile.relax_verdict_case else "stock"),
                scenario_id=getattr(scenario, "scenario_id", entry.case_id),
                run_number=entry.repeat,
            )
        finally:
            if sora_logger is not None and session_log is not None:
                sora_logger.removeHandler(session_log)
                assert previous_level is not None
                sora_logger.setLevel(previous_level)
    raw_anomalies = int(getattr(result, "raw_cached_input_anomalies", 0))
    calls, agent_cost, upper_bound = _call_records_and_cost(result.llm_report, profile, sheet)
    missing = surplus = 0
    if result.write_counts is not None:
        missing = sum(sum(turn.missing.values()) for turn in result.write_counts.turns)
        surplus = sum(sum(turn.surplus.values()) for turn in result.write_counts.turns)
    llm_report = result.llm_report
    valid_completion = (
        result.exception is None and result.terminal_cause == "verification_completion"
    )
    passed = result.outcome.success if valid_completion else None
    score = float(passed) if isinstance(passed, bool) else None
    if result.exception is not None:
        status = f"error: {result.exception}"
    elif not valid_completion:
        status = f"invalid: {result.terminal_cause or 'unknown terminal cause'}"
    else:
        status = "complete"
    record = EvaluationRecord(
        arm=entry.arm,
        profile=entry.profile,
        suite=entry.suite,
        capability=entry.capability,
        case_id=entry.case_id,
        repeat=entry.repeat,
        score=score,
        passed=passed,
        missing_writes=missing,
        surplus_writes=surplus,
        repair_count=(llm_report.malformed_fields_repaired if llm_report else 0),
        replan_count=result.replan_count,
        terminal_parse_failures=(llm_report.terminal_parse_failures if llm_report else 0),
        duration_seconds=result.duration,
        agent_llm_calls=result.agent_llm_calls,
        agent_llm_call_limit=max_agent_llm_calls,
        provider_round_trips=sum(call.get("round_trips", 0) for call in calls),
        external_actions=result.external_actions,
        latency_seconds=(llm_report.latency_seconds if llm_report else 0.0),
        input_tokens=(llm_report.input_tokens if llm_report else 0),
        cached_input_tokens=(llm_report.cached_input_tokens if llm_report else 0),
        cache_write_input_tokens=(
            getattr(llm_report, "cache_write_input_tokens", None) if llm_report else None
        ),
        cache_unknown_input_tokens=(llm_report.cache_unknown_input_tokens if llm_report else 0),
        output_tokens=(llm_report.output_tokens if llm_report else 0),
        reasoning_tokens=(llm_report.thinking_tokens if llm_report else 0),
        agent_cost=agent_cost,
        agent_cost_reserve=entry.reserved_agent_cost,
        agent_cost_upper_bound=upper_bound,
        judge_reserve=entry.reserved_judge_cost,
        judge_profile=judge_profile.to_dict(),
        call_records=calls,
        status=status,
        terminal_cause=cast(Any, result.terminal_cause),
        decision_cycles=result.decision_cycles,
        prop_reads=result.prop_reads,
        distinct_prop_reads=len(result.prop_reads_by_property),
        charged_seconds=float(
            getattr(
                result, "charged_seconds", charge.charged_seconds if charge is not None else 0.0
            )
        ),
        charge_model_identity=charge.identity if charge is not None else None,
        charge_model_digest=charge_sheet.digest if charge is not None else None,
        cached_input_clamps=charge.cached_input_clamps if charge is not None else 0,
        raw_cached_input_anomalies=raw_anomalies,
        charge_accounting_consistent=(
            None if charge is None else charge.cached_input_clamps == raw_anomalies
        ),
        clock_mode="wall" if charge is None else "token_charged",
        inference_charge_policy=getattr(
            result, "inference_charge_policy", "serialized_sum" if charge is not None else None
        ),
        llm_wall_seconds=float(getattr(result, "llm_wall_seconds", 0.0)),
        llm_wall_union_seconds=float(getattr(result, "llm_wall_union_seconds", 0.0)),
        llm_charged_union_seconds=float(getattr(result, "llm_charged_union_seconds", 0.0)),
        llm_round_trips=int(getattr(result, "llm_round_trips", 0)),
        llm_max_in_flight=int(getattr(result, "llm_max_in_flight", 0)),
        llm_overlapped_round_trips=int(getattr(result, "llm_overlapped_round_trips", 0)),
    )
    if attempt_dir is not None:
        try:
            from examples.gaia2.evaluation.diagnostics import export_attempt

            diagnostic_ref = export_attempt(
                attempt_dir,
                output_dir=output_dir,
                entry=entry,
                record=record,
                result=result,
                scenario=scenario,
                config_path=config_path,
                profile=profile,
                judge_profile=judge_profile,
                runtime_events=runtime_events,
                llm_capture=llm_capture,
                session_log=session_log,
                source_revision=source_revision,
                source_dirty_diff_sha256=source_dirty_diff_sha256,
            )
        except BaseException as error:
            try:
                artifact_path = attempt_dir.relative_to(output_dir).as_posix()
            except ValueError:
                artifact_path = str(attempt_dir)
            diagnostic_ref = {
                "schema_version": 1,
                "path": artifact_path,
                "index_sha256": None,
                "error": f"export: {type(error).__name__}: {error}",
            }
        record = replace(record, diagnostics=diagnostic_ref)
    elif diagnostics:
        record = replace(
            record,
            diagnostics={
                "schema_version": 1,
                "path": None,
                "index_sha256": None,
                "error": diagnostic_setup_error or "diagnostic setup failed",
            },
        )
    return record


def _headless_neutral_done(agent: Any) -> bool:
    """Stop once no activity can advance without another user message."""
    from sora.activity import ActivityState
    from sora.types import InputWait

    activities = list(agent.working.activities.values())
    if not activities:
        return False
    unfinished = [
        activity for activity in activities if activity.state is not ActivityState.TERMINATED
    ]
    return not unfinished or all(
        activity.state is ActivityState.BLOCKED and isinstance(activity.blocked_on, InputWait)
        for activity in unfinished
    )


def _input_wait_prompts(agent: Any) -> tuple[str, ...]:
    from sora.activity import ActivityState
    from sora.types import InputWait

    return tuple(
        activity.blocked_on.prompt or ""
        for activity in agent.working.activities.values()
        if activity.state is ActivityState.BLOCKED and isinstance(activity.blocked_on, InputWait)
    )


def _live_neutral_record(
    entry: Any,
    *,
    config_path: Path,
    profile: ModelProfile,
    sheet: PriceSheet,
) -> EvaluationRecord:
    from examples.gaia2._runner import _terminal_cause, _terminal_inference_errors
    from examples.gaia2.evaluation.campaigns.prompt.synthetic import (
        LIVE_TASKS,
        TOOL_ID,
        SyntheticTool,
        score_live_case,
    )
    from sora.bootstrap import build_agent
    from sora.cli import TerminalSession

    agent = build_agent(str(config_path))
    synthetic_tool: SyntheticTool | None = None

    def stop_when_done() -> bool:
        nonlocal synthetic_tool
        if synthetic_tool is None:
            try:
                candidate = agent.registry.get(TOOL_ID)
            except KeyError:
                pass  # The session may poll before Agent.run() finishes joining workspaces.
            else:
                if not isinstance(candidate, SyntheticTool):
                    raise TypeError(
                        "live neutral workspace returned an unexpected tool implementation"
                    )
                synthetic_tool = candidate
        return _headless_neutral_done(agent)

    session = TerminalSession(
        agent,
        color=False,
        initial_task=LIVE_TASKS[entry.case_id],
        stop_when=stop_when_done,
        read_stdin=False,
    )
    started = time.monotonic()
    asyncio.run(session.run())
    duration = time.monotonic() - started
    if synthetic_tool is None:
        raise RuntimeError("live neutral session ended before its synthetic tool became available")
    scored_passed, authorization_violations, safety_violations = score_live_case(
        entry.case_id,
        synthetic_tool.invocations,
        cast(Any, agent.communication).sent,
    )
    case = next(case for case in NEUTRAL_CASES if case.case_id == entry.case_id)
    calls, agent_cost, upper_bound = _call_records_and_cost(session.llm_report, profile, sheet)
    llm_report = session.llm_report
    inference_errors = _terminal_inference_errors(llm_report)
    input_wait_prompts = _input_wait_prompts(agent)
    passed = None if inference_errors else scored_passed
    if inference_errors:
        status = f"error: {' | '.join(inference_errors)}"
    elif input_wait_prompts:
        status = f"awaiting input: {' | '.join(input_wait_prompts)}"
    else:
        status = "complete"
    terminal_cause = _terminal_cause(
        None,
        False,
        None,
        passed,
        inference_errors=inference_errors,
    )
    return EvaluationRecord(
        arm=entry.arm,
        profile=entry.profile,
        suite=entry.suite,
        capability=case.topic,
        case_id=entry.case_id,
        repeat=entry.repeat,
        score=float(passed) if isinstance(passed, bool) else None,
        passed=passed,
        replan_count=sum(activity.replan_count for activity in agent.working.activities.values()),
        terminal_parse_failures=(llm_report.terminal_parse_failures if llm_report else 0),
        authorization_violations=authorization_violations,
        safety_violations=safety_violations,
        duration_seconds=duration,
        agent_llm_calls=agent.procedural.logical_calls_admitted,
        provider_round_trips=sum(call.get("round_trips", 0) for call in calls),
        external_actions=agent.cycle.external_action_count,
        prop_reads=sum(agent.working.prop_reads.values()),
        distinct_prop_reads=len(agent.working.prop_reads),
        latency_seconds=(llm_report.latency_seconds if llm_report else 0.0),
        input_tokens=(llm_report.input_tokens if llm_report else 0),
        cached_input_tokens=(llm_report.cached_input_tokens if llm_report else 0),
        cache_unknown_input_tokens=(llm_report.cache_unknown_input_tokens if llm_report else 0),
        output_tokens=(llm_report.output_tokens if llm_report else 0),
        reasoning_tokens=(llm_report.thinking_tokens if llm_report else 0),
        agent_cost=agent_cost,
        agent_cost_reserve=entry.reserved_agent_cost,
        agent_cost_upper_bound=upper_bound,
        call_records=calls,
        status=status,
        terminal_cause=cast(Any, terminal_cause),
    )


def _validate_judge_overrides(args: argparse.Namespace, judge_profile: JudgeProfile) -> None:
    pinned = {
        "judge_model": judge_profile.model,
        "judge_provider": judge_profile.provider,
        "judge_endpoint": judge_profile.endpoint,
    }
    for argument, expected in pinned.items():
        supplied = getattr(args, argument)
        if supplied is not None and supplied != expected:
            flag = "--" + argument.replace("_", "-")
            raise ValueError(f"{flag} must match the pinned judge value {expected!r}")


def _run_command(args: argparse.Namespace) -> int:
    if args.max_agent_llm_calls <= 0:
        raise ValueError("--max-agent-llm-calls must be positive")
    profiles = load_profiles(EVAL_ROOT / "profiles.json")
    judge_profile = load_judge_profile(PROMPT_ROOT / "judge.json")
    _validate_judge_overrides(args, judge_profile)
    unknown = sorted(set(args.profile) - set(profiles))
    if unknown:
        raise ValueError(f"unknown profiles: {', '.join(unknown)}")
    wrong_campaign = sorted(
        name for name in args.profile if "prompt" not in profiles[name].campaigns
    )
    if wrong_campaign:
        raise ValueError(f"profiles are not declared for prompt: {', '.join(wrong_campaign)}")
    manifests = load_manifests(PROMPT_ROOT / "manifests")
    # Before any credential, provider, or scenario is touched: the prompts this process would send
    # have to be the ones the declared snapshot recorded. A run that discovers a prompt edit only
    # afterwards has already paid for rows nothing can retroactively attribute, and the edit is
    # exactly what a prompt campaign spends money to measure — so this fails closed rather than
    # recording what it found.
    prompt_snapshot = load_prompt_snapshot(Path(args.prompt_snapshot))
    prompt_snapshot_digest = verify_live_prompts(prompt_snapshot)
    prompt_snapshot_identity = str(prompt_snapshot["identity"])
    sheet = PriceSheet.load(Path(args.price_sheet))
    charge_sheet = ChargeModelSheet.load(EVAL_ROOT / "charge_model.json")
    for name in args.profile:
        sheet.for_profile(profiles[name])
    selection = RunSelection(
        profiles=tuple(args.profile),
        suites=tuple(args.suite),
        arm=args.arm,
        repeats=args.repeats,
        live_neutral=args.live_neutral,
    )
    source_revision = _source_revision()
    source_dirty_diff_sha256 = _dirty_diff_hash(excluded={Path(args.output_dir).resolve()})
    policy = BudgetPolicy(
        max_gaia_runs=args.max_gaia_runs,
        max_total_spend=args.max_total_spend,
        unknown_gaia_agent_reserve=args.gaia_agent_reserve,
        unknown_gaia_judge_reserve=args.judge_reserve,
        unknown_neutral_live_reserve=args.neutral_reserve,
    )
    output_dir = Path(args.output_dir)
    checkpoint = output_dir / "checkpoint.jsonl"
    completed, records = _checkpoint_records(checkpoint)
    expected_judge = judge_profile.to_dict()
    incompatible_judge_records = [
        record
        for record in records
        if record.suite in GAIA_SUITES and record.judge_profile != expected_judge
    ]
    if incompatible_judge_records:
        raise ValueError(
            "checkpoint contains Gaia records without the pinned judge profile; use a separate "
            "output directory or remove those Gaia rows"
        )
    # Resuming into rows produced under other prompts would append this run's rows beside them and
    # leave one checkpoint describing two prompt versions, which the report can then only withhold.
    # Refused here instead, while it is still one directory choice rather than spent money.
    foreign_prompt_digests = sorted(
        {
            record.prompt_snapshot_digest or "(none recorded)"
            for record in records
            if record.prompt_snapshot_digest != prompt_snapshot_digest
        }
    )
    if foreign_prompt_digests:
        raise ValueError(
            f"checkpoint contains records run under other prompts "
            f"({', '.join(digest[:12] for digest in foreign_prompt_digests)}); this run declares "
            f"{prompt_snapshot_identity} ({prompt_snapshot_digest[:12]}) — use a separate output "
            f"directory"
        )
    matrix = build_run_matrix(selection, policy, manifests, prior_records=records)
    projected_campaign_spend = matrix.prior_spend + matrix.total_reserve
    if args.confirm_budget + 1e-9 < projected_campaign_spend:
        raise ValueError(
            f"--confirm-budget ${args.confirm_budget:.2f} is below the reserved "
            f"campaign total ${projected_campaign_spend:.2f}"
        )
    print(
        canonical_json(
            _matrix_dict(
                matrix,
                completed,
                max_agent_llm_calls=args.max_agent_llm_calls,
                judge_profile=judge_profile,
            )
        ),
        end="",
    )
    if args.dry_run:
        print("dry-run: no acceptance payload, credential, provider, or scenario loader was opened")
        return 0
    if (
        any(entry.suite == "acceptance" for entry in matrix.entries)
        and not args.ack_locked_acceptance
    ):
        raise ManifestLockedError(
            "acceptance run requires --ack-locked-acceptance before payloads may be opened"
        )

    def _assert_remaining_budget() -> None:
        spent = sum(record.accounted_agent_cost + record.judge_reserve for record in records)
        remaining = sum(
            entry.reserved_agent_cost + entry.reserved_judge_cost
            for entry in matrix.entries
            if entry.key not in completed
        )
        if spent + remaining > policy.max_total_spend + 1e-9:
            raise ValueError(
                f"checkpoint actuals plus remaining reserves project ${spent + remaining:.2f}, "
                f"above spend ceiling ${policy.max_total_spend:.2f}"
            )

    # This precedes credential resolution and every provider/scenario import. On resume, actual
    # costs replace (rather than hide behind) the reservations of completed entries.
    _assert_remaining_budget()
    pending_live_profiles = {
        entry.profile
        for entry in matrix.entries
        if entry.key not in completed
        and (entry.suite in GAIA_SUITES or (entry.suite == "neutral" and args.live_neutral))
    }
    for profile_name in sorted(pending_live_profiles):
        credential = profiles[profile_name].credential_env
        if not os.environ.get(credential):
            raise ValueError(f"profile {profile_name} requires environment variable {credential}")
    pending_gaia = any(
        entry.key not in completed and entry.suite in GAIA_SUITES for entry in matrix.entries
    )
    if pending_gaia and not os.environ.get(judge_profile.credential_env):
        raise ValueError(
            f"judge profile {judge_profile.name} requires environment variable "
            f"{judge_profile.credential_env}"
        )
    gaia_filesystem_staged = False
    for entry in matrix.entries:
        if entry.key in completed:
            continue
        _assert_remaining_budget()
        if entry.suite == "contract" or (entry.suite == "neutral" and not args.live_neutral):
            record = _offline_record(entry)
        elif entry.suite == "neutral":
            config = _write_profile_config(
                output_dir,
                profiles[entry.profile],
                entry,
                synthetic=True,
                max_agent_llm_calls=args.max_agent_llm_calls,
            )
            record = _live_neutral_record(
                entry,
                config_path=config,
                profile=profiles[entry.profile],
                sheet=sheet,
            )
        else:
            if not gaia_filesystem_staged:
                # ARE binds DEMO_FS_PATH while importing its config module. Stage once, as late as
                # possible after every no-spend gate but before _live_gaia_record imports ARE.
                from examples.gaia2._local_fs import ensure_local_fallback_fs

                ensure_local_fallback_fs()
                gaia_filesystem_staged = True
            config = _write_profile_config(
                output_dir,
                profiles[entry.profile],
                entry,
                max_agent_llm_calls=args.max_agent_llm_calls,
            )
            record = _live_gaia_record(
                entry,
                scenario_root=Path(args.scenario_root),
                ack_locked_acceptance=args.ack_locked_acceptance,
                config_path=config,
                profile=profiles[entry.profile],
                sheet=sheet,
                judge_profile=judge_profile,
                max_wall_seconds=args.max_wall_seconds,
                max_agent_llm_calls=args.max_agent_llm_calls,
                output_dir=output_dir,
                charge_sheet=charge_sheet,
                wall_clock=args.wall_clock,
                diagnostics=not args.no_diagnostics,
                source_revision=source_revision,
                source_dirty_diff_sha256=source_dirty_diff_sha256,
            )
        # Stamped here rather than threaded through the three record builders: the verified digest
        # is a property of the whole run, identical on every row it produces, and a builder that
        # takes it can be called without it.
        record = replace(
            record,
            prompt_snapshot_identity=prompt_snapshot_identity,
            prompt_snapshot_digest=prompt_snapshot_digest,
        )
        _append_checkpoint(checkpoint, entry.key, record)
        records.append(record)
        completed.add(entry.key)
        diagnostic_error = record.diagnostics and record.diagnostics.get("error")
        if diagnostic_error:
            print(
                f"diagnostics {entry.suite}/{entry.capability}/{entry.case_id}: "
                f"INCOMPLETE ({diagnostic_error})",
                file=sys.stderr,
            )
        if entry.suite in GAIA_SUITES:
            verdict = (
                "PASS" if record.passed is True else "FAIL" if record.passed is False else "INVALID"
            )
            print(
                f"judge {entry.suite}/{entry.capability}/{entry.case_id} "
                f"[{entry.profile}, {entry.arm}, repeat {entry.repeat}]: {verdict} "
                f"(terminal={record.terminal_cause})"
            )
    print(f"completed {len(records)} records; checkpoint: {checkpoint}")
    return 0


def _read_records(paths: list[str]) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    for raw_path in paths:
        path = Path(raw_path)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            record = raw.get("record", raw)
            parsed = record_from_dict(record)
            diagnostics = parsed.diagnostics
            if not parsed.call_records and diagnostics is not None:
                relative = diagnostics.get("path")
                if isinstance(relative, str):
                    calls_path = path.parent / relative / "llm_calls.json"
                    try:
                        calls = json.loads(calls_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        pass
                    else:
                        if isinstance(calls, list):
                            parsed = EvaluationRecord(
                                **(
                                    parsed.to_dict(detailed_acceptance=True)
                                    | {"call_records": tuple(calls)}
                                )
                            )
            records.append(parsed)
    return records


def _report_diff_exclusions(inputs: list[str], output: str) -> set[Path]:
    return {*(Path(raw_path).resolve().parent for raw_path in inputs), Path(output).resolve()}


def _expected_prompt_snapshots(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    """Resolve each arm's declared snapshot file to the identity and digest its rows must carry.

    Loading the file rather than trusting a digest passed on the command line: the loader already
    refuses a snapshot whose rows no longer hash to its recorded digest, so resolving through it
    means a mutated snapshot cannot certify the rows that were run against its earlier content.

    An arm with no declared snapshot is simply absent from the mapping. The report withholds its
    delta and names the arm — raising here instead would make a report unreadable in exactly the
    situation where its per-pair rows are the thing needed for diagnosis."""
    declared = {
        "baseline": getattr(args, "baseline_snapshot", None),
        "candidate": getattr(args, "candidate_snapshot", None),
    }
    expected: dict[str, dict[str, str]] = {}
    for arm, path in declared.items():
        if not path:
            continue
        snapshot = load_prompt_snapshot(Path(path))
        expected[arm] = {
            "identity": str(snapshot["identity"]),
            "digest": str(snapshot["prompts_digest"]),
        }
    return expected


def _report_command(args: argparse.Namespace) -> int:
    records = _read_records(args.input)
    profiles = load_profiles(EVAL_ROOT / "profiles.json")
    judge_profile = load_judge_profile(PROMPT_ROOT / "judge.json")
    manifests = load_manifests(PROMPT_ROOT / "manifests")
    sheet = PriceSheet.load(Path(args.price_sheet)) if args.price_sheet else None
    expected_prompt_snapshots = _expected_prompt_snapshots(args)
    selected_names = sorted({record.profile for record in records if record.profile in profiles})
    dirty_diff_sha256 = _dirty_diff_hash(excluded=_report_diff_exclusions(args.input, args.output))
    report = build_report(
        records,
        detailed_acceptance=args.include_acceptance_details,
        expected_prompt_snapshots=expected_prompt_snapshots,
        source_revision=_source_revision(),
        source_dirty_diff_sha256=dirty_diff_sha256,
        price_sheet_date=sheet.effective_date if sheet else None,
        price_sheet_digest=sheet.digest if sheet else None,
        harness_revision=_source_revision(),
        harness_dirty_diff_sha256=dirty_diff_sha256,
        manifest_digests={name: manifest.digest for name, manifest in manifests.items()},
        selected_profiles=[profiles[name].to_dict() for name in selected_names],
        judge_profile=judge_profile.to_dict(),
        safety_sensitive=args.safety_sensitive,
        reduces_tool_catalog=args.reduces_tool_catalog,
        fresh_expansion_payloads_available=args.expansion_payloads_available,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(report))
    print(f"wrote canonical report: {output}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m examples.gaia2.evaluation")
    campaigns = parser.add_subparsers(dest="campaign", required=True)
    prompt = campaigns.add_parser("prompt", help="frozen prompt-tuning evaluation campaign")
    commands = prompt.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot", help="render the seven canonical prompt inputs")
    snapshot.add_argument("--output", help="write JSON here; omit to print to stdout")
    snapshot.add_argument("--identity", required=True, help="immutable snapshot identity")
    snapshot.add_argument("--reason", required=True, help="why this prompt identity was captured")
    snapshot.set_defaults(handler=_snapshot_command)
    configuration = commands.add_parser(
        "configuration", help="render the mutable campaign-configuration mirror"
    )
    configuration.add_argument("--output", default=str(CAMPAIGN_CONFIGURATION))
    configuration.set_defaults(handler=_configuration_command)
    check = commands.add_parser("check", help="validate all tracked artifacts without network I/O")
    check.add_argument("--scenario-root", default=str(DEFAULT_SCENARIO_ROOT))
    check.add_argument("--require-scenarios", action="store_true")
    check.set_defaults(handler=_check_command)
    run = commands.add_parser("run", help="execute an explicit, budget-gated evaluation matrix")
    run.add_argument("--profile", action="append", required=True)
    run.add_argument(
        "--suite",
        action="append",
        required=True,
        choices=["contract", "neutral", *GAIA_SUITES],
    )
    run.add_argument("--arm", required=True, choices=["baseline", "candidate"])
    run.add_argument("--output-dir", required=True)
    run.add_argument("--price-sheet", required=True)
    run.add_argument(
        "--prompt-snapshot",
        default=str(BASELINE_PROMPT_SNAPSHOT),
        help=(
            "the prompt snapshot this run declares; the live renderer must match it exactly "
            "before anything is spent. Defaults to the pre-optimization control, which is what "
            "a baseline arm runs; a candidate arm names its own snapshot."
        ),
    )
    run.add_argument("--confirm-budget", required=True, type=float)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--gaia-repeats", "--repeats", dest="repeats", type=int, default=1)
    run.add_argument("--scenario-root", default=str(DEFAULT_SCENARIO_ROOT))
    run.add_argument("--ack-locked-acceptance", action="store_true")
    run.add_argument("--max-gaia-runs", type=int, default=60)
    run.add_argument("--max-total-spend", type=float, default=180.0)
    run.add_argument("--gaia-agent-reserve", type=float, default=3.0)
    run.add_argument("--judge-reserve", type=float, default=0.5)
    run.add_argument("--neutral-reserve", type=float, default=0.5)
    run.add_argument("--live-neutral", action="store_true")
    run.add_argument(
        "--judge-model", help="optional assertion; must match the campaign's pinned judge model"
    )
    run.add_argument(
        "--judge-provider",
        help="optional assertion; must match the campaign's pinned judge provider",
    )
    run.add_argument(
        "--judge-endpoint",
        help="optional assertion; must match the campaign's pinned judge endpoint",
    )
    run.add_argument("--max-wall-seconds", type=float, default=1200.0)
    run.add_argument(
        "--wall-clock",
        action="store_true",
        help=(
            "Run Gaia cases on elapsed wall time rather than the frozen token charge. Produces a "
            "separate robustness result that is not timing-comparable to charged runs."
        ),
    )
    run.add_argument("--max-agent-llm-calls", type=int, default=200)
    run.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="disable replayable evidence bundles for live Gaia attempts",
    )
    run.set_defaults(handler=_run_command)
    report = commands.add_parser("report", help="combine baseline/candidate checkpoints")
    report.add_argument("--input", action="append", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--price-sheet")
    # Which snapshot each arm is claimed to have run. Checked against what the rows actually
    # recorded; an arm whose expectation is absent or unmet has its paired delta withheld, because
    # two arms can each be internally consistent and still have run the same prompts.
    report.add_argument(
        "--baseline-snapshot",
        default=str(BASELINE_PROMPT_SNAPSHOT),
        help="prompt snapshot the baseline arm was run against",
    )
    report.add_argument(
        "--candidate-snapshot",
        help="prompt snapshot the candidate arm was run against (no default: it is frozen per "
        "campaign, and guessing it is the failure this check exists to catch)",
    )
    report.add_argument("--include-acceptance-details", action="store_true")
    report.add_argument("--safety-sensitive", action="store_true")
    report.add_argument("--reduces-tool-catalog", action="store_true")
    report.add_argument("--expansion-payloads-available", action="store_true")
    report.set_defaults(handler=_report_command)
    paper2027 = campaigns.add_parser(
        "paper2027",
        help="show the AAMAS 2027 protocol-freeze candidate",
    )
    paper2027.set_defaults(handler=_paper2027_status)
    return parser


def _paper2027_status(args: argparse.Namespace) -> int:
    del args
    print(
        "paper2027 has an uncommitted protocol-freeze candidate and exact Gaia2 mini manifest; "
        "run it through examples.gaia2.batch after the checkpoint is reviewed and committed"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, FileNotFoundError, ManifestLockedError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
