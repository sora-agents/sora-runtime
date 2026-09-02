from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
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
    build_frozen_baseline,
    build_prompt_snapshot,
    load_frozen_snapshot,
)
from examples.gaia2.evaluation.core import (
    GAIA_SUITES,
    BudgetPolicy,
    CallUsage,
    EvaluationRecord,
    ManifestLockedError,
    ModelProfile,
    PriceSheet,
    RunMatrix,
    RunSelection,
    build_run_matrix,
    calculate_call_cost,
    canonical_json,
    load_manifests,
    load_profiles,
    record_from_dict,
    resolve_scenario,
    sha256_file,
    sha256_text,
)

EVAL_ROOT = Path(__file__).parent
PROMPT_ROOT = EVAL_ROOT / "campaigns" / "prompt"
DEFAULT_SCENARIO_ROOT = Path("examples/gaia2/scenarios")
DEFAULT_PRICE_SHEET = EVAL_ROOT / "price_sheets" / "2026-09-02.json"


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


def _dirty_diff_hash(*, excluded: set[Path] | None = None) -> str | None:
    diff = _git_output("diff", "--binary", "HEAD")
    root_raw = _git_output("rev-parse", "--show-toplevel")
    untracked_raw = _git_output("ls-files", "--others", "--exclude-standard")
    excluded_resolved = {path.resolve() for path in excluded or set()}
    additions: list[dict[str, str]] = []
    if root_raw and untracked_raw:
        root = Path(root_raw)
        for relative in untracked_raw.splitlines():
            path = root / relative
            if path.resolve() in excluded_resolved or not path.is_file():
                continue
            additions.append({"path": relative, "sha256": sha256_file(path)})
    if not diff and not additions:
        return None
    return sha256_text(canonical_json({"tracked_diff": diff or "", "untracked": additions}))


def _snapshot_command(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    baseline = build_frozen_baseline(
        source_revision=_source_revision(),
        root=EVAL_ROOT,
        source_dirty_diff_sha256=_dirty_diff_hash(excluded={output} if output else None),
    )
    rendered = canonical_json(baseline)
    if args.output:
        assert output is not None
        output.write_text(rendered)
        print(f"wrote prompt snapshot: {args.output}")
    else:
        print(rendered, end="")
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
    manifests = load_manifests(PROMPT_ROOT / "manifests")
    sheets = [PriceSheet.load(path) for path in sorted((EVAL_ROOT / "price_sheets").glob("*.json"))]
    if not sheets:
        raise ValueError("no dated price sheet is available")
    for profile in profiles.values():
        if not any(profile.model in sheet.models for sheet in sheets):
            raise ValueError(f"no dated price sheet covers profile model {profile.model}")
    frozen = load_frozen_snapshot(PROMPT_ROOT / "baseline.json")
    rendered = build_prompt_snapshot(source_revision=frozen["provenance"]["source_revision"])
    if rendered["prompts"] != frozen["prompts"]:
        raise ValueError("the seven runtime prompts no longer match the frozen baseline")
    expected_profiles = [profiles[name].to_dict() for name in sorted(profiles)]
    if frozen.get("evaluation_profiles") != expected_profiles:
        raise ValueError("the frozen evaluation profiles no longer match profiles.json")
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
    # Exercise report serialization and its schema without any scenario/model content.
    report = build_report([], prompt_snapshot=frozen)
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
        keys.add(str(row["matrix_key"]))
        records.append(record_from_dict(row["record"]))
    return keys, records


def _append_checkpoint(path: Path, matrix_key: str, record: EvaluationRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"matrix_key": matrix_key, "record": record.to_dict()}))
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
                profile.model,
                CallUsage(
                    input_tokens=usage.input_tokens,
                    cached_input_tokens=usage.cached_input_tokens,
                    output_tokens=usage.output_tokens,
                    reasoning_tokens=usage.reasoning_tokens,
                ),
            )
            call_costs.append(cost)
            total += cost.agent_cost
            upper_bound = upper_bound or cost.upper_bound
            round_trip_usage.append(
                {
                    "input_tokens": usage.input_tokens,
                    "cached_input_tokens": usage.cached_input_tokens,
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
                "discarded": call.discarded,
                "cost": aggregate_cost,
                "terminal_parse_failures": getattr(call, "terminal_parse_failures", 0),
            }
        )
    return tuple(rows), total if usage_complete else None, upper_bound


def _live_gaia_record(
    entry: Any,
    *,
    scenario_root: Path,
    ack_locked_acceptance: bool,
    config_path: Path,
    profile: ModelProfile,
    sheet: PriceSheet,
    judge_model: str | None,
    judge_provider: str | None,
    judge_endpoint: str | None,
    max_wall_seconds: float,
    max_agent_llm_calls: int,
) -> EvaluationRecord:
    # Every ARE/provider import is below all dry-run and budget gates.
    from examples.gaia2._runner import run_scenario
    from sora.adapters.are_sim import attach_judge, load_scenario, populate_oracle_events

    path = resolve_scenario(
        scenario_root,
        entry.suite,
        entry.capability,
        entry.case_id,
        ack_locked_acceptance=ack_locked_acceptance,
    )
    scenario = load_scenario(str(path))
    if judge_model:
        attach_judge(
            scenario,
            model=judge_model,
            provider=judge_provider,
            endpoint=judge_endpoint,
            relax_verdict_case=True,
        )
    else:
        populate_oracle_events(scenario)
    result = run_scenario(
        scenario,
        config=str(config_path),
        max_wall_seconds=max_wall_seconds,
    )
    calls, agent_cost, upper_bound = _call_records_and_cost(result.llm_report, profile, sheet)
    missing = surplus = 0
    if result.write_counts is not None:
        missing = sum(sum(turn.missing.values()) for turn in result.write_counts.turns)
        surplus = sum(sum(turn.surplus.values()) for turn in result.write_counts.turns)
    llm_report = result.llm_report
    passed = result.outcome.success if result.exception is None else None
    score = float(passed) if isinstance(passed, bool) else None
    return EvaluationRecord(
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
        cache_unknown_input_tokens=(llm_report.cache_unknown_input_tokens if llm_report else 0),
        output_tokens=(llm_report.output_tokens if llm_report else 0),
        reasoning_tokens=(llm_report.thinking_tokens if llm_report else 0),
        agent_cost=agent_cost,
        agent_cost_reserve=entry.reserved_agent_cost,
        agent_cost_upper_bound=upper_bound,
        judge_reserve=entry.reserved_judge_cost,
        call_records=calls,
        status=(f"error: {result.exception}" if result.exception else "complete"),
        terminal_cause=cast(Any, result.terminal_cause),
        decision_cycles=result.decision_cycles,
    )


def _live_neutral_record(
    entry: Any,
    *,
    config_path: Path,
    profile: ModelProfile,
    sheet: PriceSheet,
) -> EvaluationRecord:
    from examples.gaia2.evaluation.campaigns.prompt.synthetic import (
        LIVE_TASKS,
        TOOL_ID,
        SyntheticTool,
        score_live_case,
    )
    from sora.bootstrap import build_agent
    from sora.cli import TerminalSession

    agent = build_agent(str(config_path))
    session = TerminalSession(
        agent,
        color=False,
        initial_task=LIVE_TASKS[entry.case_id],
        exit_when_idle=0.2,
    )
    started = time.monotonic()
    asyncio.run(session.run())
    duration = time.monotonic() - started
    tool = agent.registry.get(TOOL_ID)
    if not isinstance(tool, SyntheticTool):
        raise TypeError("live neutral workspace returned an unexpected tool implementation")
    passed, authorization_violations, safety_violations = score_live_case(
        entry.case_id,
        tool.invocations,
        cast(Any, agent.communication).sent,
    )
    case = next(case for case in NEUTRAL_CASES if case.case_id == entry.case_id)
    calls, agent_cost, upper_bound = _call_records_and_cost(session.llm_report, profile, sheet)
    llm_report = session.llm_report
    return EvaluationRecord(
        arm=entry.arm,
        profile=entry.profile,
        suite=entry.suite,
        capability=case.topic,
        case_id=entry.case_id,
        repeat=entry.repeat,
        score=float(passed),
        passed=passed,
        replan_count=sum(activity.replan_count for activity in agent.working.activities.values()),
        terminal_parse_failures=(llm_report.terminal_parse_failures if llm_report else 0),
        authorization_violations=authorization_violations,
        safety_violations=safety_violations,
        duration_seconds=duration,
        agent_llm_calls=agent.procedural.logical_calls_admitted,
        provider_round_trips=sum(call.get("round_trips", 0) for call in calls),
        external_actions=agent.cycle.external_action_count,
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
    )


def _run_command(args: argparse.Namespace) -> int:
    if args.max_agent_llm_calls <= 0:
        raise ValueError("--max-agent-llm-calls must be positive")
    profiles = load_profiles(EVAL_ROOT / "profiles.json")
    unknown = sorted(set(args.profile) - set(profiles))
    if unknown:
        raise ValueError(f"unknown profiles: {', '.join(unknown)}")
    wrong_campaign = sorted(
        name for name in args.profile if "prompt" not in profiles[name].campaigns
    )
    if wrong_campaign:
        raise ValueError(f"profiles are not declared for prompt: {', '.join(wrong_campaign)}")
    manifests = load_manifests(PROMPT_ROOT / "manifests")
    sheet = PriceSheet.load(Path(args.price_sheet))
    selection = RunSelection(
        profiles=tuple(args.profile),
        suites=tuple(args.suite),
        arm=args.arm,
        repeats=args.repeats,
        live_neutral=args.live_neutral,
    )
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
                judge_model=args.judge_model,
                judge_provider=args.judge_provider,
                judge_endpoint=args.judge_endpoint,
                max_wall_seconds=args.max_wall_seconds,
                max_agent_llm_calls=args.max_agent_llm_calls,
            )
        _append_checkpoint(checkpoint, entry.key, record)
        records.append(record)
        completed.add(entry.key)
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
            records.append(record_from_dict(record))
    return records


def _report_command(args: argparse.Namespace) -> int:
    records = _read_records(args.input)
    profiles = load_profiles(EVAL_ROOT / "profiles.json")
    manifests = load_manifests(PROMPT_ROOT / "manifests")
    sheet = PriceSheet.load(Path(args.price_sheet)) if args.price_sheet else None
    snapshot = load_frozen_snapshot(PROMPT_ROOT / "baseline.json")
    selected_names = sorted({record.profile for record in records if record.profile in profiles})
    report = build_report(
        records,
        detailed_acceptance=args.include_acceptance_details,
        prompt_snapshot=snapshot,
        source_revision=_source_revision(),
        source_dirty_diff_sha256=_dirty_diff_hash(),
        price_sheet_date=sheet.effective_date if sheet else None,
        price_sheet_digest=sheet.digest if sheet else None,
        harness_revision=_source_revision(),
        harness_dirty_diff_sha256=_dirty_diff_hash(),
        manifest_digests={name: manifest.digest for name, manifest in manifests.items()},
        selected_profiles=[profiles[name].to_dict() for name in selected_names],
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
    snapshot.set_defaults(handler=_snapshot_command)
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
    run.add_argument("--judge-model")
    run.add_argument("--judge-provider")
    run.add_argument("--judge-endpoint")
    run.add_argument("--max-wall-seconds", type=float, default=1200.0)
    run.add_argument("--max-agent-llm-calls", type=int, default=200)
    run.set_defaults(handler=_run_command)
    report = commands.add_parser("report", help="combine baseline/candidate checkpoints")
    report.add_argument("--input", action="append", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--price-sheet")
    report.add_argument("--include-acceptance-details", action="store_true")
    report.add_argument("--safety-sensitive", action="store_true")
    report.add_argument("--reduces-tool-catalog", action="store_true")
    report.add_argument("--expansion-payloads-available", action="store_true")
    report.set_defaults(handler=_report_command)
    aamas2027 = campaigns.add_parser(
        "aamas2027",
        help="reserved campaign name for the later frozen AAMAS 2027 protocol",
    )
    aamas2027.set_defaults(handler=_aamas2027_status)
    return parser


def _aamas2027_status(args: argparse.Namespace) -> int:
    del args
    print("aamas2027 campaign is reserved; its protocol and run matrix are not frozen yet")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, FileNotFoundError, ManifestLockedError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
