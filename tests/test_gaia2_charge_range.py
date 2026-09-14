from __future__ import annotations

import json
from pathlib import Path

import pytest
from examples.gaia2.charge_range import _main, check_charge_range, format_report
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
    input_tokens: int = 10_000,
    cached_input_tokens: int | None = 5_000,
    output_tokens: int = 500,
    reasoning_tokens: int | None = 100,
    model: str = "moonshotai/kimi-k2.5",
    round_trips: int = 1,
    usage_captured: bool = True,
) -> dict[str, object]:
    return {
        "arm": arm,
        "model": model,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "round_trips": round_trips,
        "usage_captured": usage_captured,
    }


def _coefficient_and_profile() -> tuple[ChargeCoefficients, ModelProfile]:
    profile = load_profiles(EVAL_ROOT / "profiles.json")["kimi-k2.5-prompt"]
    coefficients = ChargeModelSheet.load(EVAL_ROOT / "charge_model.json").for_profile(profile)
    return coefficients, profile


@pytest.mark.parametrize("profile_name", ["gpt-5.4-high-paper", "kimi-k2.5-prompt"])
def test_range_check_uses_the_same_decode_box_for_both_arms_and_endpoints(
    profile_name: str,
) -> None:
    profile = load_profiles(EVAL_ROOT / "profiles.json")[profile_name]
    coefficients = ChargeModelSheet.load(EVAL_ROOT / "charge_model.json").for_profile(profile)
    report = check_charge_range(
        [_row("sora", model=profile.model)],
        [_row("react", model=profile.model)],
        profile=profile,
        coefficients=coefficients,
    )

    assert report.passed
    assert report.sora.axes["uncached_input_tokens"].bounds == (1_000, 64_000)
    assert report.sora.axes["cached_input_tokens"].bounds == (0, 64_000)
    assert report.sora.axes["decode_tokens"].bounds == (16, 8_192)
    assert report.react.axes["decode_tokens"].bounds == (16, 8_192)
    assert report.decode_source == "completion_tokens"


def test_contained_reasoning_is_not_added_to_completion_tokens() -> None:
    coefficients, profile = _coefficient_and_profile()
    report = check_charge_range(
        [_row("sora", output_tokens=8_192, reasoning_tokens=7_000)],
        [_row("react", output_tokens=8_192, reasoning_tokens=7_000)],
        profile=profile,
        coefficients=coefficients,
    )

    assert report.passed
    assert report.sora.axes["decode_tokens"].maximum == 8_192


def test_range_check_reports_calls_and_token_mass_outside_each_axis() -> None:
    coefficients, profile = _coefficient_and_profile()
    sora_rows = [
        _row("sora", input_tokens=90_000, cached_input_tokens=0, output_tokens=9_000),
        _row("sora", input_tokens=1_000, cached_input_tokens=0, output_tokens=16),
    ]
    report = check_charge_range(
        sora_rows,
        [_row("react")],
        profile=profile,
        coefficients=coefficients,
    )

    assert not report.passed
    uncached = report.sora.axes["uncached_input_tokens"]
    assert uncached.outside_calls == 1
    assert uncached.outside_call_share == pytest.approx(0.5)
    assert uncached.outside_token_share == pytest.approx(90_000 / 91_000)
    assert report.sora.axes["decode_tokens"].outside_calls == 1


def test_missing_usage_is_inconclusive_instead_of_in_range() -> None:
    coefficients, profile = _coefficient_and_profile()
    report = check_charge_range(
        [_row("sora", usage_captured=False)],
        [_row("react")],
        profile=profile,
        coefficients=coefficients,
    )

    assert not report.passed
    assert all(axis.inconclusive for axis in report.sora.axes.values())


def test_multi_round_trip_aggregate_is_uncertifiable_and_fails_closed() -> None:
    coefficients, profile = _coefficient_and_profile()
    report = check_charge_range(
        [_row("sora"), _row("sora", round_trips=2, input_tokens=20_000)],
        [_row("react")],
        profile=profile,
        coefficients=coefficients,
    )

    assert not report.passed
    assert report.sora.uncertifiable_round_trip_aggregates == 1
    assert report.sora.excluded == {"multi_round_trip_aggregate": 1}
    assert report.sora.axes["uncached_input_tokens"].usable_calls == 1
    assert "aggregate rows UNCERTIFIABLE without per-round-trip usage" in format_report(report)


def test_range_check_rejects_wrong_arm_or_model() -> None:
    coefficients, profile = _coefficient_and_profile()
    with pytest.raises(ValueError, match="expected sora"):
        check_charge_range(
            [_row("react")],
            [_row("react")],
            profile=profile,
            coefficients=coefficients,
        )
    with pytest.raises(ValueError, match="does not match profile"):
        check_charge_range(
            [_row("sora", model="wrong/model")],
            [_row("react")],
            profile=profile,
            coefficients=coefficients,
        )


def test_range_check_cli_loads_the_frozen_sheet_and_names_unmeasured_extrapolation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sora = tmp_path / "sora.jsonl"
    react = tmp_path / "react.jsonl"
    sora.write_text(json.dumps(_row("sora")) + "\n")
    react.write_text(json.dumps(_row("react")) + "\n")

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
    assert "decode_tokens (completion_tokens): [16, 8192]" in output
    assert "Outside this box is unmeasured extrapolation" in output
    assert "sora" in output and "react" in output and "PASS" in output
