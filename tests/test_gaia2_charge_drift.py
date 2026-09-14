from __future__ import annotations

import json
from pathlib import Path

import pytest
from examples.gaia2.charge_drift import (
    BETWEEN_ARM_BIAS_GAP_LIMIT,
    SIGNED_BIAS_LIMIT,
    _main,
    check_charge_drift,
)
from examples.gaia2.evaluation.core import (
    ChargeCoefficients,
    ChargeModelSheet,
    ModelProfile,
    load_profiles,
)

EVAL_ROOT = Path(__file__).parents[1] / "examples" / "gaia2" / "evaluation"


def _row(
    arm: str,
    *,
    seconds: float,
    input_tokens: int = 10_000,
    cached_input_tokens: int = 5_000,
    output_tokens: int = 500,
    round_trips: int = 1,
    model: str = "moonshotai/kimi-k2.5",
) -> dict[str, object]:
    return {
        "arm": arm,
        "model": model,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "seconds": seconds,
        "round_trips": round_trips,
        "usage_captured": True,
    }


def _coefficient_and_profile() -> tuple[ChargeCoefficients, ModelProfile]:
    profile = load_profiles(EVAL_ROOT / "profiles.json")["kimi-k2.5-prompt"]
    coefficients = ChargeModelSheet.load(EVAL_ROOT / "charge_model.json").for_profile(profile)
    return coefficients, profile


def test_drift_check_prices_aggregate_round_trips_and_reports_both_arms() -> None:
    coefficients, profile = _coefficient_and_profile()
    expected = (
        2 * coefficients.a0_seconds
        + 5_000 * coefficients.seconds_per_uncached_input_token
        + 5_000 * coefficients.seconds_per_cached_input_token
        + 500 * coefficients.seconds_per_output_token
    )
    rows = [_row("sora", seconds=expected, round_trips=2)]
    baseline = [_row("react", seconds=expected, round_trips=2)]

    report = check_charge_drift(rows, baseline, profile=profile, coefficients=coefficients)

    assert report.passed
    assert report.sora.usable_calls == report.react.usable_calls == 1
    assert report.sora.signed_bias == pytest.approx(0.0)
    assert report.react.mean_absolute_percentage_error == pytest.approx(0.0)


def test_drift_check_rejects_absolute_and_between_arm_bias() -> None:
    coefficients, profile = _coefficient_and_profile()
    predicted = coefficients.charged_seconds(5_000, 5_000, 500)
    report = check_charge_drift(
        [_row("sora", seconds=predicted / (1.0 + SIGNED_BIAS_LIMIT + 0.01))],
        [_row("react", seconds=predicted)],
        profile=profile,
        coefficients=coefficients,
    )

    assert not report.passed
    assert report.sora.signed_bias > SIGNED_BIAS_LIMIT
    assert report.bias_gap > BETWEEN_ARM_BIAS_GAP_LIMIT


def test_drift_check_fails_closed_on_missing_cache_coverage() -> None:
    coefficients, profile = _coefficient_and_profile()
    missing_cache = _row("sora", seconds=2.0)
    missing_cache["cached_input_tokens"] = None

    with pytest.raises(ValueError, match="no usable sora calls"):
        check_charge_drift(
            [missing_cache],
            [_row("react", seconds=2.0)],
            profile=profile,
            coefficients=coefficients,
        )


def test_drift_check_rejects_wrong_arm_or_model() -> None:
    coefficients, profile = _coefficient_and_profile()
    with pytest.raises(ValueError, match="expected sora"):
        check_charge_drift(
            [_row("react", seconds=2.0)],
            [_row("react", seconds=2.0)],
            profile=profile,
            coefficients=coefficients,
        )
    with pytest.raises(ValueError, match="does not match profile"):
        check_charge_drift(
            [_row("sora", seconds=2.0, model="wrong/model")],
            [_row("react", seconds=2.0)],
            profile=profile,
            coefficients=coefficients,
        )


def test_drift_check_cli_is_an_offline_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    coefficients, _profile = _coefficient_and_profile()
    predicted = coefficients.charged_seconds(5_000, 5_000, 500)
    sora = tmp_path / "sora.jsonl"
    react = tmp_path / "react.jsonl"
    sora.write_text(json.dumps(_row("sora", seconds=predicted)) + "\n")
    react.write_text(json.dumps(_row("react", seconds=predicted)) + "\n")

    status = _main(
        [
            "--profile",
            "kimi-k2.5-prompt",
            "--sora",
            str(sora),
            "--react",
            str(react),
        ]
    )

    assert status == 0
    output = capsys.readouterr().out
    assert "sora" in output and "react" in output
    assert "signed bias" in output and "PASS" in output
