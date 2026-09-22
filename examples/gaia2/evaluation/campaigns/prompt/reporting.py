from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from examples.gaia2.evaluation.core import (
    GAIA_SUITES,
    LEGACY_CLOCK_MODE,
    NON_EVALUABLE_GAIA_TERMINAL_CAUSES,
    EvaluationRecord,
    decide_acceptance_expansion,
    load_manifests,
    load_profiles,
)

STATISTICAL_BOOTSTRAP_SEED = 20260831


def _optional_sum(values: Iterable[float | None]) -> float | None:
    """Sum that propagates unavailability instead of dropping it.

    Used for the parallel-charge counterfactual, which a run reports as None when its crossings
    did not share one time axis. Skipping those would produce a total whose denominator no longer
    matches the row's run count, which is worse than having no number at all.
    """
    total = 0.0
    for value in values:
        if value is None:
            return None
        total += value
    return total


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


def _prompt_snapshots_by_arm(
    records: list[EvaluationRecord],
) -> dict[str, list[dict[str, str | None]]]:
    """The distinct prompt snapshots each arm's rows record, as observed rather than as declared.

    Read off the records instead of taken from the report's inputs because the two arms of this
    campaign run at different times, from different source revisions, under deliberately different
    prompts. Anything the report is told once can only describe one arm, and describing both with
    it is how a paired delta between two unknown prompt versions reads as a measurement."""
    by_arm: dict[str, set[tuple[str | None, str | None]]] = {}
    for record in records:
        by_arm.setdefault(record.arm, set()).add(
            (record.prompt_snapshot_identity, record.prompt_snapshot_digest)
        )
    return {
        arm: [
            {"identity": identity, "digest": digest}
            for identity, digest in sorted(
                observed, key=lambda pair: (pair[0] or "", pair[1] or "")
            )
        ]
        for arm, observed in sorted(by_arm.items())
    }


def _paired_comparison_withheld(
    snapshots_by_arm: dict[str, list[dict[str, str | None]]],
    expected_prompt_snapshots: dict[str, dict[str, str]] | None,
) -> tuple[str, ...]:
    """Every reason these arms cannot be subtracted from one another.

    A prompt campaign's delta means "this prompt change moved the score by this much", which
    requires each arm to have run one known prompt version throughout, *and* that version to be
    the one the comparison was declared over. None of that is checkable from the deltas
    themselves: rows with no snapshot look exactly like rows with the right one, an arm that
    changed prompts mid-run averages two versions into a single column, and an arm pointed at the
    wrong snapshot file produces rows that are internally consistent, genuinely verified, and
    answer a different question than the one being asked.

    That last case is why observed uniformity is not sufficient. Running the candidate arm against
    the control snapshot — one mistyped path — yields two arms that agree with themselves, agree
    with each other, and subtract to approximately zero. Read as a measurement it says the rewrite
    changed nothing, which is indistinguishable from the rewrite never having been run. So each
    arm is matched against the snapshot it was *declared* to run, not merely against itself.

    Reasons rather than a bool, matching the headline gate: the operator needs to see which arm is
    the problem, and a report that only says "withheld" sends them back to the raw rows."""
    reasons: list[str] = []
    for arm in ("baseline", "candidate"):
        snapshots = snapshots_by_arm.get(arm)
        # An arm that did not run produces no pairs at all, which is its own, already visible
        # outcome. Naming it here would report a missing arm as a provenance failure.
        if not snapshots:
            continue
        if any(snapshot["digest"] is None for snapshot in snapshots):
            reasons.append(f"{arm} arm has rows that recorded no prompt snapshot")
        declared = [snapshot for snapshot in snapshots if snapshot["digest"] is not None]
        if len(declared) > 1:
            named = ", ".join(
                f"{snapshot['identity']} ({str(snapshot['digest'])[:12]})" for snapshot in declared
            )
            reasons.append(f"{arm} arm mixes prompt snapshots: {named}")
        expected = (expected_prompt_snapshots or {}).get(arm)
        if expected is None:
            reasons.append(f"{arm} arm ran without a declared prompt snapshot to check against")
            continue
        for snapshot in declared:
            if (
                snapshot["identity"] == expected["identity"]
                and snapshot["digest"] == expected["digest"]
            ):
                continue
            reasons.append(
                f"{arm} arm ran {snapshot['identity']} ({str(snapshot['digest'])[:12]}) "
                f"where {expected['identity']} ({expected['digest'][:12]}) was declared"
            )
    return tuple(reasons)


def build_report(
    records: list[EvaluationRecord],
    *,
    detailed_acceptance: bool = False,
    expected_prompt_snapshots: dict[str, dict[str, str]] | None = None,
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
    mismatched_charges = [
        record
        for record in records
        if record.charge_accounting_consistent is False
        or (
            record.charge_accounting_consistent is not None
            and record.cached_input_clamps != record.raw_cached_input_anomalies
        )
    ]
    if mismatched_charges:
        raise ValueError("cached-input clamp counts disagree with raw usage anomaly counts")
    # A missing mode is a legacy row, not an absence of evidence; see core.LEGACY_CLOCK_MODE.
    clock_modes = {record.clock_mode or LEGACY_CLOCK_MODE for record in records}
    if len(clock_modes) > 1:
        raise ValueError(f"report mixes incomparable clock modes: {sorted(clock_modes)}")
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
    prompt_snapshots_by_arm = _prompt_snapshots_by_arm(records)
    paired_comparison_withheld = _paired_comparison_withheld(
        prompt_snapshots_by_arm, expected_prompt_snapshots
    )
    run_pairs = _paired_runs(records)
    pairs = _cluster_pairs(run_pairs)
    deltas = [float(pair["score_delta"]) for pair in pairs if pair["score_delta"] is not None]
    exploratory_mean_delta = sum(deltas) / len(deltas) if deltas else None
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
        # The expansion rule subtracts the two arms exactly as the headline delta does, so it is
        # unsound under the same conditions. Without this it would keep returning a verdict — and
        # "expansion not required" is the expensive direction to get wrong.
        complete = (
            not paired_comparison_withheld
            and len(profile_pairs) == 5
            and all(
                isinstance(pair["baseline_score"], int | float)
                and isinstance(pair["candidate_score"], int | float)
                for pair in profile_pairs
            )
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
    diagnostic_records = [record for record in records if record.diagnostics is not None]
    incomplete_diagnostics = sum(
        bool(record.diagnostics and record.diagnostics.get("error"))
        for record in diagnostic_records
    )
    observations: list[dict[str, Any]] = []
    observed: set[str] = set()
    for record in records:
        if record.suite == "acceptance" and not detailed_acceptance:
            continue
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
            # Two keys, not one, because they answer two different questions: what each arm
            # was *declared* to run, and what each arm's rows say it *did* run. Collapsing them
            # is how a report states as fact something it only assumed. A third key once held a
            # single report-level snapshot; it was removed rather than deprecated because a
            # report-level value cannot describe two arms that run different prompts by
            # construction, and a null one reads as "unknown" rather than "not applicable".
            "expected_prompt_snapshots_by_arm": expected_prompt_snapshots or {},
            "prompt_snapshots_by_arm": prompt_snapshots_by_arm,
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
            "charge_models": [
                {"identity": json.loads(identity), "digest": digest}
                for identity, digest in sorted(
                    {
                        (
                            json.dumps(record.charge_model_identity, sort_keys=True),
                            record.charge_model_digest,
                        )
                        for record in records
                        if record.charge_model_identity is not None
                        and record.charge_model_digest is not None
                    }
                )
            ],
        },
        "cases": [record.to_dict(detailed_acceptance=detailed_acceptance) for record in records],
        "aggregates": {
            "pass_at_1": _pass_at_1(records),
            "paired_run_deltas": run_pairs,
            "paired_deltas": pairs,
            # The campaign's result, or nothing. The per-pair rows above stay readable either way
            # — they are what an operator needs to diagnose a withheld comparison — but the two
            # numbers anyone would quote are unavailable rather than approximate when the arms
            # cannot say which prompts produced them.
            "mean_paired_score_delta": None
            if paired_comparison_withheld
            else exploratory_mean_delta,
            "paired_delta_bootstrap_95_interval": (
                None if paired_comparison_withheld else _bootstrap_interval(deltas)
            ),
            # The same mean, named for what it is, so a run in progress stays readable. Never
            # promote this into a result.
            "exploratory_mean_paired_score_delta": exploratory_mean_delta,
            "paired_comparison_withheld": paired_comparison_withheld,
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
            "simulated_clock": {
                "charged_seconds": sum(record.charged_seconds for record in records),
                "clock_modes": sorted(clock_modes),
                "inference_charge_policies": sorted(
                    {
                        record.inference_charge_policy
                        for record in records
                        if record.inference_charge_policy is not None
                    }
                ),
                "llm_wall_seconds": sum(record.llm_wall_seconds for record in records),
                "llm_wall_union_seconds": sum(record.llm_wall_union_seconds for record in records),
                "llm_wall_overlap_seconds": sum(
                    max(0.0, record.llm_wall_seconds - record.llm_wall_union_seconds)
                    for record in records
                ),
                # One unavailable run makes the group's counterfactual unavailable too: summing
                # the rest would silently report a total over fewer runs than the row claims.
                "llm_charged_union_seconds": _optional_sum(
                    record.llm_charged_union_seconds for record in records
                ),
                "llm_charged_overlap_seconds": _optional_sum(
                    None
                    if record.llm_charged_union_seconds is None
                    else max(0.0, record.charged_seconds - record.llm_charged_union_seconds)
                    for record in records
                ),
                "llm_round_trips": sum(record.llm_round_trips for record in records),
                "llm_max_in_flight": max(
                    (record.llm_max_in_flight for record in records), default=0
                ),
                "llm_overlapped_round_trips": sum(
                    record.llm_overlapped_round_trips for record in records
                ),
                "cached_input_clamps": sum(record.cached_input_clamps for record in records),
                "raw_cached_input_anomalies": sum(
                    record.raw_cached_input_anomalies for record in records
                ),
            },
            "acceptance_expansion": expansion,
            "diagnostics": {
                "records": len(diagnostic_records),
                "incomplete": incomplete_diagnostics,
            },
        },
    }


def default_provenance(eval_root: Path) -> dict[str, Any]:
    manifests = load_manifests(eval_root / "campaigns" / "prompt" / "manifests")
    profiles = load_profiles(eval_root / "profiles.json")
    return {
        "manifest_digests": {name: manifest.digest for name, manifest in manifests.items()},
        "selected_profiles": [profile.to_dict() for profile in profiles.values()],
    }
