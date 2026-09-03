from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from examples.gaia2.evaluation.core import (
    GAIA_SUITES,
    NON_EVALUABLE_GAIA_TERMINAL_CAUSES,
    EvaluationRecord,
    decide_acceptance_expansion,
    load_manifests,
    load_profiles,
)

STATISTICAL_BOOTSTRAP_SEED = 20260831


def _quality_score(record: EvaluationRecord) -> float | None:
    """Return only scores produced by a complete, quality-evaluable run.

    ``None`` remains accepted for old records that predate terminal-cause reporting. A known bad
    Gaia terminal cause is different: ARE may still return a final-validation verdict for the
    truncated environment, but that verdict does not measure the requested trajectory.
    """
    if record.suite in GAIA_SUITES and record.terminal_cause in NON_EVALUABLE_GAIA_TERMINAL_CAUSES:
        return None
    return record.score


def _paired_runs(records: list[EvaluationRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, int], dict[str, EvaluationRecord]] = {}
    for record in records:
        key = (
            record.profile,
            record.suite,
            record.capability,
            record.case_id,
            record.repeat,
        )
        grouped.setdefault(key, {})[record.arm] = record
    pairs: list[dict[str, Any]] = []
    for key, arms in sorted(grouped.items()):
        baseline, candidate = arms.get("baseline"), arms.get("candidate")
        if baseline is None or candidate is None:
            continue
        baseline_score = _quality_score(baseline)
        candidate_score = _quality_score(candidate)
        delta = (
            candidate_score - baseline_score
            if candidate_score is not None and baseline_score is not None
            else None
        )
        pairs.append(
            {
                "profile": key[0],
                "suite": key[1],
                "capability": key[2],
                "case_id": key[3],
                "repeat": key[4],
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "score_delta": delta,
                "new_safety_violations": max(
                    0, candidate.safety_violations - baseline.safety_violations
                ),
                "new_authorization_violations": max(
                    0,
                    candidate.authorization_violations - baseline.authorization_violations,
                ),
            }
        )
    return pairs


def _cluster_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for pair in pairs:
        key = (
            str(pair["profile"]),
            str(pair["suite"]),
            str(pair["capability"]),
            str(pair["case_id"]),
        )
        grouped.setdefault(key, []).append(pair)
    clusters: list[dict[str, Any]] = []
    for key, runs in sorted(grouped.items()):
        complete = [
            run
            for run in runs
            if isinstance(run["baseline_score"], int | float)
            and isinstance(run["candidate_score"], int | float)
        ]
        baseline = (
            sum(float(run["baseline_score"]) for run in complete) / len(complete)
            if complete
            else None
        )
        candidate = (
            sum(float(run["candidate_score"]) for run in complete) / len(complete)
            if complete
            else None
        )
        clusters.append(
            {
                "profile": key[0],
                "suite": key[1],
                "capability": key[2],
                "case_id": key[3],
                "repeats": sorted(int(run["repeat"]) for run in runs),
                "baseline_score": baseline,
                "candidate_score": candidate,
                "score_delta": (
                    candidate - baseline if candidate is not None and baseline is not None else None
                ),
                "new_safety_violations": max(
                    (int(run["new_safety_violations"]) for run in runs), default=0
                ),
                "new_authorization_violations": max(
                    (int(run["new_authorization_violations"]) for run in runs), default=0
                ),
            }
        )
    return clusters


def _pass_at_1(records: list[EvaluationRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[EvaluationRecord]] = {}
    for record in records:
        grouped.setdefault((record.arm, record.profile, record.suite), []).append(record)
    rows: list[dict[str, Any]] = []
    for key, runs in sorted(grouped.items()):
        scores = [float(score) for run in runs if (score := _quality_score(run)) is not None]
        rows.append(
            {
                "arm": key[0],
                "profile": key[1],
                "suite": key[2],
                "runs": len(runs),
                "scenarios": len({run.case_id for run in runs}),
                "pass_at_1": sum(scores) / len(scores) if scores else None,
            }
        )
    return rows


def _bootstrap_interval(values: list[float], *, samples: int = 10_000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(STATISTICAL_BOOTSTRAP_SEED)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return [means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]]


def _familiarity_gap(pairs: list[dict[str, Any]]) -> float | None:
    by_suite: dict[str, list[float]] = {}
    for pair in pairs:
        delta = pair["score_delta"]
        if isinstance(delta, int | float):
            by_suite.setdefault(str(pair["suite"]), []).append(float(delta))
    familiar = by_suite.get("familiar")
    acceptance = by_suite.get("acceptance")
    if not familiar or not acceptance:
        return None
    familiar_loss = -sum(familiar) / len(familiar)
    acceptance_loss = -sum(acceptance) / len(acceptance)
    return familiar_loss - acceptance_loss


def build_report(
    records: list[EvaluationRecord],
    *,
    detailed_acceptance: bool = False,
    prompt_snapshot: dict[str, Any] | None = None,
    source_revision: str | None = None,
    source_dirty_diff_sha256: str | None = None,
    price_sheet_date: str | None = None,
    price_sheet_digest: str | None = None,
    harness_revision: str | None = None,
    harness_dirty_diff_sha256: str | None = None,
    manifest_digests: dict[str, str] | None = None,
    selected_profiles: list[dict[str, Any]] | None = None,
    judge_profile: dict[str, Any] | None = None,
    safety_sensitive: bool = False,
    reduces_tool_catalog: bool = False,
    fresh_expansion_payloads_available: bool = False,
) -> dict[str, Any]:
    gaia_records = [
        record for record in records if record.suite in {"familiar", "development", "acceptance"}
    ]
    if judge_profile is not None:
        mismatched = [
            record
            for record in gaia_records
            if record.judge_profile is not None and record.judge_profile != judge_profile
        ]
        if mismatched:
            raise ValueError("Gaia records contain a judge profile that differs from the report")
    matching_judge_records = sum(
        record.judge_profile == judge_profile
        for record in gaia_records
        if judge_profile is not None
    )
    run_pairs = _paired_runs(records)
    pairs = _cluster_pairs(run_pairs)
    deltas = [float(pair["score_delta"]) for pair in pairs if pair["score_delta"] is not None]
    hard_failure_count = sum(
        int(pair["new_safety_violations"] > 0 or pair["new_authorization_violations"] > 0)
        for pair in pairs
    )
    if not pairs:
        hard_failure_count = sum(
            int(record.safety_violations > 0 or record.authorization_violations > 0)
            for record in records
            if record.arm == "candidate"
        )
    acceptance_profiles = sorted(
        {record.profile for record in records if record.suite == "acceptance"}
    )
    by_profile: dict[str, dict[str, Any]] = {}
    for profile in acceptance_profiles:
        profile_pairs = [
            pair for pair in pairs if pair["suite"] == "acceptance" and pair["profile"] == profile
        ]
        complete = len(profile_pairs) == 5 and all(
            isinstance(pair["baseline_score"], int | float)
            and isinstance(pair["candidate_score"], int | float)
            for pair in profile_pairs
        )
        if not complete:
            by_profile[profile] = {
                "required": False,
                "reasons": [],
                "status": "not_evaluable",
            }
            continue
        decision = decide_acceptance_expansion(
            baseline_scores=[float(pair["baseline_score"]) for pair in profile_pairs],
            candidate_scores=[float(pair["candidate_score"]) for pair in profile_pairs],
            safety_sensitive=safety_sensitive,
            reduces_tool_catalog=reduces_tool_catalog,
            fresh_payloads_available=fresh_expansion_payloads_available,
        )
        by_profile[profile] = asdict(decision)

    evaluable = [row for row in by_profile.values() if row["status"] != "not_evaluable"]
    required = any(row["required"] for row in evaluable)
    reasons = list(dict.fromkeys(reason for row in evaluable for reason in row.get("reasons", ())))
    if required:
        status = "deferred" if any(row["status"] == "deferred" for row in evaluable) else "required"
    elif len(evaluable) != len(by_profile) or not evaluable:
        status = "not_evaluable"
    else:
        status = "not_required"
    expansion: dict[str, Any] = {
        "required": required,
        "reasons": reasons,
        "status": status,
        "by_profile": by_profile,
    }
    known_agent_cost = sum(record.agent_cost or 0.0 for record in records)
    unknown_agent_cost_reserve = sum(
        record.agent_cost_reserve for record in records if record.agent_cost is None
    )
    total_judge_reserve = sum(record.judge_reserve for record in records)
    observations: list[dict[str, Any]] = []
    observed: set[str] = set()
    for record in records:
        for call in record.call_records:
            for model in call.get("observed_models", []):
                item = {"kind": "model", "value": str(model)}
                key = json.dumps(item, sort_keys=True)
                if key not in observed:
                    observed.add(key)
                    observations.append(item)
            for sdk in call.get("sdk_observations", []):
                name = str(sdk.get("name", "unknown"))
                version = str(sdk.get("version", "unknown"))
                item = {"kind": f"sdk:{name}", "value": version}
                key = json.dumps(item, sort_keys=True)
                if key not in observed:
                    observed.add(key)
                    observations.append(item)
            for provider in call.get("provider_observations", []):
                item = {"kind": "provider", "value": provider}
                key = json.dumps(item, sort_keys=True)
                if key not in observed:
                    observed.add(key)
                    observations.append(item)
    return {
        "schema_version": 1,
        "campaign": "prompt",
        "provenance": {
            "harness_revision": harness_revision,
            "harness_dirty_diff_sha256": harness_dirty_diff_sha256,
            "source_revision": source_revision,
            "source_dirty_diff_sha256": source_dirty_diff_sha256,
            "prompt_snapshot": prompt_snapshot,
            "manifest_digests": manifest_digests or {},
            "selected_profiles": selected_profiles or [],
            "judge_profile": judge_profile,
            "judge_profile_coverage": {
                "gaia_records": len(gaia_records),
                "matching_records": matching_judge_records,
                "complete": matching_judge_records == len(gaia_records),
            },
            "actual_sdk_model_observations": observations,
            "price_sheet_date": price_sheet_date,
            "price_sheet_digest": price_sheet_digest,
            "statistical_bootstrap_seed": STATISTICAL_BOOTSTRAP_SEED,
            "agent_llm_call_limits": sorted(
                {
                    record.agent_llm_call_limit
                    for record in records
                    if record.agent_llm_call_limit is not None
                }
            ),
        },
        "cases": [record.to_dict(detailed_acceptance=detailed_acceptance) for record in records],
        "aggregates": {
            "pass_at_1": _pass_at_1(records),
            "paired_run_deltas": run_pairs,
            "paired_deltas": pairs,
            "mean_paired_score_delta": sum(deltas) / len(deltas) if deltas else None,
            "paired_delta_bootstrap_95_interval": _bootstrap_interval(deltas),
            "familiarity_gap": _familiarity_gap(pairs),
            "hard_failure_count": hard_failure_count,
            "budget_usage": {
                "exact_agent_cost": known_agent_cost,
                "unknown_agent_cost_reserve": unknown_agent_cost_reserve,
                "agent_cost_unavailable_count": sum(
                    record.agent_cost is None for record in records
                ),
                "estimated_judge_reserve": total_judge_reserve,
                "total": known_agent_cost + unknown_agent_cost_reserve + total_judge_reserve,
                "agent_cost_is_upper_bound": any(
                    record.agent_cost_upper_bound for record in records
                ),
            },
            "acceptance_expansion": expansion,
        },
    }


def default_provenance(eval_root: Path) -> dict[str, Any]:
    manifests = load_manifests(eval_root / "campaigns" / "prompt" / "manifests")
    profiles = load_profiles(eval_root / "profiles.json")
    return {
        "manifest_digests": {name: manifest.digest for name, manifest in manifests.items()},
        "selected_profiles": [profile.to_dict() for profile in profiles.values()],
    }
