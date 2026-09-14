"""Audit frozen latency charges against real calls from both benchmark arms.

The frozen coefficients are an accounting rule, not a promise that provider latency stands still.
This command detects the narrower failure an endpoint key cannot: service moving behind an
unchanged provider/model/routing identity. It never re-fits or edits the frozen artifact.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from examples.gaia2.evaluation.core import (
    MODEL_SNAPSHOTS,
    ChargeCoefficients,
    ChargeModelSheet,
    ModelProfile,
    load_profiles,
)

EVAL_ROOT = Path(__file__).resolve().parent / "evaluation"
SIGNED_BIAS_LIMIT = 0.15
BETWEEN_ARM_BIAS_GAP_LIMIT = 0.10


@dataclass(frozen=True)
class ArmDrift:
    arm: Literal["sora", "react"]
    total_calls: int
    usable_calls: int
    excluded: Mapping[str, int]
    observed_seconds: float
    predicted_seconds: float
    signed_bias: float
    mean_absolute_percentage_error: float


@dataclass(frozen=True)
class ChargeDriftReport:
    sora: ArmDrift
    react: ArmDrift

    @property
    def bias_gap(self) -> float:
        return abs(self.sora.signed_bias - self.react.signed_bias)

    @property
    def passed(self) -> bool:
        return (
            abs(self.sora.signed_bias) <= SIGNED_BIAS_LIMIT
            and abs(self.react.signed_bias) <= SIGNED_BIAS_LIMIT
            and self.bias_gap <= BETWEEN_ARM_BIAS_GAP_LIMIT
        )


def _integer(row: Mapping[str, Any], name: str, *, positive: bool = False) -> int | None:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < (1 if positive else 0):
        return None
    return value


def _measure_arm(
    rows: Iterable[Mapping[str, Any]],
    *,
    arm: Literal["sora", "react"],
    profile: ModelProfile,
    coefficients: ChargeCoefficients,
) -> ArmDrift:
    materialized = list(rows)
    accepted_models = {profile.model}
    snapshot = MODEL_SNAPSHOTS.get(profile.model)
    if snapshot is not None:
        accepted_models.add(snapshot)
    excluded: Counter[str] = Counter()
    observations: list[tuple[float, float]] = []
    for index, row in enumerate(materialized, start=1):
        recorded_arm = row.get("arm")
        if recorded_arm != arm:
            raise ValueError(f"row {index}: expected {arm} arm, found {recorded_arm!r}")
        model = row.get("model")
        if not isinstance(model, str) or model not in accepted_models:
            raise ValueError(
                f"row {index}: model {model!r} does not match profile {profile.name} "
                f"({sorted(accepted_models)})"
            )
        if row.get("usage_captured") is not True:
            excluded["usage_not_captured"] += 1
            continue
        total_input = _integer(row, "input_tokens")
        cached_input = _integer(row, "cached_input_tokens")
        output = _integer(row, "output_tokens")
        round_trips = _integer(row, "round_trips", positive=True)
        seconds = row.get("seconds")
        if (
            total_input is None
            or cached_input is None
            or output is None
            or round_trips is None
            or isinstance(seconds, bool)
            or not isinstance(seconds, int | float)
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            excluded["incomplete_measurement"] += 1
            continue
        if cached_input > total_input:
            excluded["cached_input_exceeds_total"] += 1
            continue
        predicted = (
            round_trips * coefficients.a0_seconds
            + (total_input - cached_input) * coefficients.seconds_per_uncached_input_token
            + cached_input * coefficients.seconds_per_cached_input_token
            + output * coefficients.seconds_per_output_token
        )
        observations.append((predicted, float(seconds)))
    if not observations:
        detail = ", ".join(f"{key}={value}" for key, value in sorted(excluded.items()))
        raise ValueError(f"no usable {arm} calls ({detail or 'empty file'})")
    observed = sum(actual for _predicted, actual in observations)
    predicted = sum(estimate for estimate, _actual in observations)
    return ArmDrift(
        arm=arm,
        total_calls=len(materialized),
        usable_calls=len(observations),
        excluded=dict(sorted(excluded.items())),
        observed_seconds=observed,
        predicted_seconds=predicted,
        signed_bias=(predicted - observed) / observed,
        mean_absolute_percentage_error=sum(
            abs(estimate - actual) / actual for estimate, actual in observations
        )
        / len(observations),
    )


def check_charge_drift(
    sora_rows: Iterable[Mapping[str, Any]],
    react_rows: Iterable[Mapping[str, Any]],
    *,
    profile: ModelProfile,
    coefficients: ChargeCoefficients,
) -> ChargeDriftReport:
    return ChargeDriftReport(
        sora=_measure_arm(sora_rows, arm="sora", profile=profile, coefficients=coefficients),
        react=_measure_arm(react_rows, arm="react", profile=profile, coefficients=coefficients),
    )


def read_calls(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: call row must be an object")
        rows.append(row)
    return rows


def format_report(report: ChargeDriftReport) -> str:
    lines: list[str] = []
    for arm in (report.sora, report.react):
        excluded = ", ".join(f"{key}={value}" for key, value in arm.excluded.items()) or "none"
        lines.append(
            f"{arm.arm:>5}: {arm.usable_calls}/{arm.total_calls} usable, "
            f"observed {arm.observed_seconds:.3f}s, predicted {arm.predicted_seconds:.3f}s, "
            f"signed bias {arm.signed_bias:+.1%}, MAPE {arm.mean_absolute_percentage_error:.1%}; "
            f"excluded: {excluded}"
        )
    lines.append(
        f"bias gap: {report.bias_gap:.1%} "
        f"(limits: |arm bias| <= {SIGNED_BIAS_LIMIT:.0%}, gap <= "
        f"{BETWEEN_ARM_BIAS_GAP_LIMIT:.0%}) -> {'PASS' if report.passed else 'FAIL'}"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", required=True, help="profile name from profiles.json")
    parser.add_argument(
        "--sora",
        type=Path,
        action="append",
        required=True,
        help="S-ORA llm_calls.jsonl; repeat for more capability files",
    )
    parser.add_argument(
        "--react",
        type=Path,
        action="append",
        required=True,
        help="ReAct llm_calls.jsonl; repeat for more capability files",
    )
    parser.add_argument("--profiles-path", type=Path, default=EVAL_ROOT / "profiles.json")
    parser.add_argument("--charge-model", type=Path, default=EVAL_ROOT / "charge_model.json")
    return parser


def _main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profiles = load_profiles(args.profiles_path)
    if args.profile not in profiles:
        raise SystemExit(f"unknown profile {args.profile!r}; have {sorted(profiles)}")
    profile = profiles[args.profile]
    coefficients = ChargeModelSheet.load(args.charge_model).for_profile(profile)
    report = check_charge_drift(
        (row for path in args.sora for row in read_calls(path)),
        (row for path in args.react for row in read_calls(path)),
        profile=profile,
        coefficients=coefficients,
    )
    print(format_report(report))
    return 0 if report.passed else 1


def main() -> None:
    raise SystemExit(_main())


if __name__ == "__main__":
    main()
