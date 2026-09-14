"""Audit both arms against the token box measured for the frozen charge model.

The fitted coefficients are linear, but their evidence is not unlimited: using them outside the
grid's measured regressor box is unmeasured extrapolation, not merely an estimate with wider
uncertainty. This command reads real call logs and the frozen ``ChargeModelSheet``; it never
re-runs the untracked fit or changes a coefficient.
"""

from __future__ import annotations

import argparse
import json
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
    decode_tokens,
    load_profiles,
)
from examples.gaia2.latency_grid import (
    CACHED_OUTPUT_LEVELS,
    CACHED_PREFIX_LEVELS,
    CACHED_SUFFIX_TOKENS,
    CALIBRATION_OUTPUT_TOKENS,
    INPUT_LEVELS,
    OUTPUT_LEVELS,
)

EVAL_ROOT = Path(__file__).resolve().parent / "evaluation"

UNCACHED_INPUT_BOUNDS = (
    min(*INPUT_LEVELS, CACHED_SUFFIX_TOKENS),
    max(INPUT_LEVELS),
)
CACHED_INPUT_BOUNDS = (0, max(CACHED_PREFIX_LEVELS))
DECODE_BOUNDS = (
    min(CALIBRATION_OUTPUT_TOKENS, *OUTPUT_LEVELS, *CACHED_OUTPUT_LEVELS),
    max(CALIBRATION_OUTPUT_TOKENS, *OUTPUT_LEVELS, *CACHED_OUTPUT_LEVELS),
)


@dataclass(frozen=True)
class AxisCoverage:
    axis: str
    lower: int
    upper: int
    usable_calls: int
    outside_calls: int
    total_tokens: int
    outside_tokens: int
    minimum: int | None
    maximum: int | None

    @property
    def bounds(self) -> tuple[int, int]:
        return self.lower, self.upper

    @property
    def inconclusive(self) -> bool:
        return self.usable_calls == 0

    @property
    def outside_call_share(self) -> float:
        return self.outside_calls / self.usable_calls if self.usable_calls else 0.0

    @property
    def outside_token_share(self) -> float:
        return self.outside_tokens / self.total_tokens if self.total_tokens else 0.0


@dataclass(frozen=True)
class ArmRange:
    arm: Literal["sora", "react"]
    total_calls: int
    uncertifiable_round_trip_aggregates: int
    excluded: Mapping[str, int]
    axes: Mapping[str, AxisCoverage]


@dataclass(frozen=True)
class ChargeRangeReport:
    profile: str
    model: str
    decode_source: str
    sora: ArmRange
    react: ArmRange

    @property
    def passed(self) -> bool:
        axes = (*self.sora.axes.values(), *self.react.axes.values())
        return (
            self.sora.uncertifiable_round_trip_aggregates == 0
            and self.react.uncertifiable_round_trip_aggregates == 0
            and all(not axis.inconclusive and axis.outside_calls == 0 for axis in axes)
        )


def _integer(row: Mapping[str, Any], name: str) -> int | None:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _coverage(axis: str, bounds: tuple[int, int], values: list[int]) -> AxisCoverage:
    lower, upper = bounds
    outside = [value for value in values if value < lower or value > upper]
    return AxisCoverage(
        axis=axis,
        lower=lower,
        upper=upper,
        usable_calls=len(values),
        outside_calls=len(outside),
        total_tokens=sum(values),
        outside_tokens=sum(outside),
        minimum=min(values) if values else None,
        maximum=max(values) if values else None,
    )


def _measure_arm(
    rows: Iterable[Mapping[str, Any]],
    *,
    arm: Literal["sora", "react"],
    profile: ModelProfile,
    coefficients: ChargeCoefficients,
) -> ArmRange:
    materialized = list(rows)
    accepted_models = {profile.model}
    snapshot = MODEL_SNAPSHOTS.get(profile.model)
    if snapshot is not None:
        accepted_models.add(snapshot)
    values: dict[str, list[int]] = {
        "uncached_input_tokens": [],
        "cached_input_tokens": [],
        "decode_tokens": [],
    }
    excluded: Counter[str] = Counter()
    uncertifiable_round_trip_aggregates = 0
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
        round_trips = _integer(row, "round_trips")
        if round_trips is None or round_trips == 0:
            raise ValueError(f"row {index}: round_trips must be a positive integer")
        if round_trips > 1:
            # S-ORA records parser repair as one logical row whose token fields are sums. The
            # grid bounds describe each provider crossing, so an aggregate cannot certify that
            # every crossing stayed inside even when its sum fits bounds scaled by round_trips.
            excluded["multi_round_trip_aggregate"] += 1
            uncertifiable_round_trip_aggregates += 1
            continue
        if row.get("usage_captured") is not True:
            excluded["usage_not_captured"] += 1
            continue

        total_input = _integer(row, "input_tokens")
        cached_input = _integer(row, "cached_input_tokens")
        if total_input is None or cached_input is None:
            excluded["input_split_not_reported"] += 1
        elif cached_input > total_input:
            excluded["cached_input_exceeds_total"] += 1
        else:
            values["uncached_input_tokens"].append(total_input - cached_input)
            values["cached_input_tokens"].append(cached_input)

        completion = _integer(row, "output_tokens")
        reasoning_raw = row.get("reasoning_tokens")
        reasoning = None if reasoning_raw is None else _integer(row, "reasoning_tokens")
        if reasoning_raw is not None and reasoning is None:
            excluded["decode_not_reported"] += 1
            continue
        decoded = decode_tokens(
            completion,
            reasoning,
            includes_reasoning=coefficients.decode_includes_reasoning,
        )
        if decoded is None:
            excluded["decode_not_reported"] += 1
        else:
            values["decode_tokens"].append(decoded)

    return ArmRange(
        arm=arm,
        total_calls=len(materialized),
        uncertifiable_round_trip_aggregates=uncertifiable_round_trip_aggregates,
        excluded=dict(sorted(excluded.items())),
        axes={
            "uncached_input_tokens": _coverage(
                "uncached_input_tokens", UNCACHED_INPUT_BOUNDS, values["uncached_input_tokens"]
            ),
            "cached_input_tokens": _coverage(
                "cached_input_tokens", CACHED_INPUT_BOUNDS, values["cached_input_tokens"]
            ),
            "decode_tokens": _coverage("decode_tokens", DECODE_BOUNDS, values["decode_tokens"]),
        },
    )


def check_charge_range(
    sora_rows: Iterable[Mapping[str, Any]],
    react_rows: Iterable[Mapping[str, Any]],
    *,
    profile: ModelProfile,
    coefficients: ChargeCoefficients,
) -> ChargeRangeReport:
    decode_source = (
        "completion_tokens"
        if coefficients.decode_includes_reasoning
        else "completion_tokens + reasoning_tokens"
    )
    return ChargeRangeReport(
        profile=profile.name,
        model=coefficients.model,
        decode_source=decode_source,
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


def format_report(report: ChargeRangeReport) -> str:
    lines = [
        f"profile {report.profile} -> {report.model}",
        "measured charge box:",
        f"  uncached_input_tokens: [{UNCACHED_INPUT_BOUNDS[0]}, {UNCACHED_INPUT_BOUNDS[1]}]",
        f"  cached_input_tokens: [{CACHED_INPUT_BOUNDS[0]}, {CACHED_INPUT_BOUNDS[1]}]",
        f"  decode_tokens ({report.decode_source}): [{DECODE_BOUNDS[0]}, {DECODE_BOUNDS[1]}]",
        "Outside this box is unmeasured extrapolation, not merely a more uncertain estimate.",
    ]
    for arm in (report.sora, report.react):
        excluded = ", ".join(f"{key}={value}" for key, value in arm.excluded.items()) or "none"
        lines.append(f"{arm.arm}: {arm.total_calls} rows; excluded: {excluded}")
        if arm.uncertifiable_round_trip_aggregates:
            lines.append(
                "  round trips: "
                f"{arm.uncertifiable_round_trip_aggregates} aggregate rows UNCERTIFIABLE "
                "without per-round-trip usage"
            )
        for axis in arm.axes.values():
            if axis.inconclusive:
                lines.append(f"  {axis.axis}: INCONCLUSIVE (no usable rows)")
                continue
            verdict = "UNMEASURED EXTRAPOLATION" if axis.outside_calls else "inside measured box"
            lines.append(
                f"  {axis.axis}: {axis.outside_calls}/{axis.usable_calls} calls outside "
                f"({axis.outside_call_share:.1%}); {axis.outside_tokens}/{axis.total_tokens} "
                f"token mass outside ({axis.outside_token_share:.1%}); "
                f"observed [{axis.minimum}, {axis.maximum}] -> {verdict}"
            )
    lines.append("PASS" if report.passed else "FAIL")
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
    report = check_charge_range(
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
