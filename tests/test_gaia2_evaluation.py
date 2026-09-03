from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from examples.gaia2.evaluation.campaigns.prompt.contracts import run_contract_suite
from examples.gaia2.evaluation.campaigns.prompt.neutral import NEUTRAL_CASES, run_neutral_suite
from examples.gaia2.evaluation.campaigns.prompt.reporting import build_report
from examples.gaia2.evaluation.campaigns.prompt.snapshot import (
    build_prompt_snapshot,
    load_frozen_snapshot,
)
from examples.gaia2.evaluation.campaigns.prompt.synthetic import (
    SyntheticInvocation,
    score_live_case,
)
from examples.gaia2.evaluation.cli import (
    _append_checkpoint,
    _call_records_and_cost,
    _headless_neutral_done,
    _live_neutral_record,
    _parser,
    _run_command,
)
from examples.gaia2.evaluation.core import (
    BudgetPolicy,
    CallUsage,
    EvaluationRecord,
    ManifestLockedError,
    ModelProfile,
    PriceSheet,
    RunSelection,
    build_run_matrix,
    calculate_call_cost,
    decide_acceptance_expansion,
    load_manifests,
    load_profiles,
    record_from_dict,
    resolve_scenario,
)

ROOT = Path(__file__).parents[1]
EVAL_ROOT = ROOT / "examples" / "gaia2" / "evaluation"
PROMPT_ROOT = EVAL_ROOT / "campaigns" / "prompt"


def test_initial_profiles_freeze_exact_models_and_behavior_settings() -> None:
    profiles = load_profiles(EVAL_ROOT / "profiles.json")
    assert set(profiles) == {
        "gpt-5.4-medium-prompt",
        "gpt-5.4-high-paper",
        "kimi-k2.5-prompt",
    }
    medium = profiles["gpt-5.4-medium-prompt"]
    assert medium.model == "gpt-5.4-2026-03-05"
    assert medium.campaigns == ("prompt",)
    assert medium.settings["reasoning_effort"].value == "medium"
    assert medium.settings["temperature"].status == "intentionally_omitted"
    assert medium.settings["temperature"].value is None
    assert "temperature" not in medium.client_settings()
    assert medium.settings["max_output_tokens"].value == 16384
    high = profiles["gpt-5.4-high-paper"]
    assert high.campaigns == ("prompt", "aamas2027")
    assert high.settings["reasoning_effort"].value == "high"
    assert high.settings["temperature"].status == "intentionally_omitted"
    assert high.settings["temperature"].value is None
    assert "temperature" not in high.client_settings()
    kimi = profiles["kimi-k2.5-prompt"]
    assert kimi.model == "moonshotai/kimi-k2.5"
    assert kimi.campaigns == ("prompt",)
    assert kimi.settings["temperature"].value == 0.5
    assert kimi.settings["reasoning"].value == {"enabled": True}
    assert kimi.settings["provider_routing"].value == {
        "only": ["deepinfra"],
        "order": ["deepinfra"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert kimi.settings["router_metadata"].value is True
    assert kimi.settings["max_output_tokens"].value == 16384
    assert all(profile.settings["seed"].value is None for profile in profiles.values())
    assert set(medium.reported_fields()) == {
        "provider",
        "client",
        "model",
        "endpoint",
        "credential_env",
        "campaigns",
        "reasoning_effort",
        "reasoning",
        "temperature",
        "top_p",
        "seed",
        "verbosity",
        "max_output_tokens",
        "service_tier",
        "provider_routing",
        "router_metadata",
        "stream",
        "stall_timeout",
        "sdk_max_retries",
        "instrument",
    }
    assert {field["status"] for field in medium.reported_fields().values()} <= {
        "sent",
        "intentionally_omitted",
        "provider_observed",
    }


def test_profile_validation_rejects_unknown_setting_status() -> None:
    raw = {
        "name": "bad",
        "provider": "openai",
        "client": "sora.adapters.openai_llm.OpenAICompatLLMClient",
        "model": "m",
        "endpoint": "https://example.invalid/v1",
        "credential_env": "KEY",
        "settings": {"seed": {"status": "maybe", "value": None}},
        "stream": True,
        "stall_timeout": 90,
        "sdk_max_retries": 0,
        "instrument": True,
    }
    with pytest.raises(ValueError, match="status"):
        ModelProfile.from_dict(raw)


def test_manifests_are_ids_only_unique_and_stratified() -> None:
    manifests = load_manifests(PROMPT_ROOT / "manifests")
    assert set(manifests) == {"familiar", "development", "acceptance"}
    assert all(len(manifest.cases) == 5 for manifest in manifests.values())
    assert all(
        {case.capability for case in manifest.cases}
        == {"search", "execution", "adaptability", "ambiguity", "time"}
        for manifest in manifests.values()
    )
    ids = [case.case_id for manifest in manifests.values() for case in manifest.cases]
    assert len(ids) == len(set(ids))
    manifest_text = "".join(path.read_text() for path in (PROMPT_ROOT / "manifests").glob("*.json"))
    assert "scenario_universe" not in manifest_text
    assert ".json" not in manifest_text


def test_scenario_resolution_recognizes_corrected_smoke_names_and_locks_acceptance(
    tmp_path: Path,
) -> None:
    search = tmp_path / "search"
    adaptability = tmp_path / "adaptability"
    search.mkdir()
    adaptability.mkdir()
    search_case = search / "smoke-scenario_universe_28_4sn4lc.json"
    adaptability_case = adaptability / "smoke-scenario_universe_21_5e0gvz.json"
    acceptance_case = search / "acc-scenario_universe_21_mq2acb.json"
    for path in (search_case, adaptability_case, acceptance_case):
        path.write_text("{}")
    assert resolve_scenario(tmp_path, "familiar", "search", "28_4sn4lc") == search_case
    assert resolve_scenario(tmp_path, "familiar", "adaptability", "21_5e0gvz") == adaptability_case
    with pytest.raises(ManifestLockedError):
        resolve_scenario(tmp_path, "acceptance", "search", "21_mq2acb")
    assert (
        resolve_scenario(
            tmp_path,
            "acceptance",
            "search",
            "21_mq2acb",
            ack_locked_acceptance=True,
        )
        == acceptance_case
    )


def test_budget_matrix_enforces_gaia_run_and_dollar_ceilings() -> None:
    selection = RunSelection(
        profiles=("gpt-5.4-medium-prompt",),
        suites=("familiar", "development"),
        arm="baseline",
        repeats=3,
    )
    matrix = build_run_matrix(selection, BudgetPolicy())
    assert len(matrix.entries) == 30
    assert matrix.gaia_runs == 30
    assert matrix.agent_reserve == 90.0
    assert matrix.judge_reserve == 15.0
    assert matrix.total_reserve == 105.0
    with pytest.raises(ValueError, match="Gaia run ceiling"):
        build_run_matrix(
            RunSelection(
                profiles=("gpt-5.4-medium-prompt",),
                suites=("familiar", "development", "acceptance"),
                arm="baseline",
                repeats=5,
            ),
            BudgetPolicy(),
        )
    with pytest.raises(ValueError, match="spend ceiling"):
        build_run_matrix(selection, BudgetPolicy(max_total_spend=100.0))


def test_repeats_apply_only_to_live_gaia_suites() -> None:
    matrix = build_run_matrix(
        RunSelection(
            profiles=("gpt-5.4-medium-prompt",),
            suites=("contract", "neutral", "familiar"),
            arm="candidate",
            repeats=3,
        ),
        BudgetPolicy(),
    )
    assert sum(entry.suite == "contract" for entry in matrix.entries) == 1
    assert sum(entry.suite == "neutral" for entry in matrix.entries) == 16
    assert sum(entry.suite == "familiar" for entry in matrix.entries) == 15


def test_budget_matrix_enforces_cumulative_checkpoint_run_and_spend_ceilings() -> None:
    prior = [
        EvaluationRecord.example(
            arm="baseline", suite="familiar", case_id=f"prior-{index}", score=1.0
        )
        for index in range(31)
    ]
    selection = RunSelection(
        profiles=("gpt-5.4-medium-prompt", "kimi-k2.5-prompt"),
        suites=("familiar",),
        arm="candidate",
        repeats=3,
    )
    with pytest.raises(ValueError, match="cumulative Gaia run ceiling"):
        build_run_matrix(selection, BudgetPolicy(), prior_records=prior)

    costly = [
        EvaluationRecord.example(arm="baseline", suite="familiar", case_id="prior", score=1.0)
    ]
    costly[0] = EvaluationRecord(**(costly[0].to_dict() | {"agent_cost": 179.0}))
    with pytest.raises(ValueError, match="cumulative spend ceiling"):
        build_run_matrix(
            RunSelection(
                profiles=("gpt-5.4-medium-prompt",),
                suites=("familiar",),
                arm="candidate",
            ),
            BudgetPolicy(),
            prior_records=costly,
        )


def test_cost_calculation_separates_cache_and_applies_long_context_tiers() -> None:
    sheet = PriceSheet.load(EVAL_ROOT / "price_sheets" / "2026-09-02.json")
    short = calculate_call_cost(
        sheet,
        "gpt-5.4-2026-03-05",
        CallUsage(input_tokens=100_000, cached_input_tokens=40_000, output_tokens=10_000),
    )
    assert short.agent_cost == pytest.approx(0.31)
    assert short.upper_bound is False
    long = calculate_call_cost(
        sheet,
        "gpt-5.4-2026-03-05",
        CallUsage(input_tokens=300_000, cached_input_tokens=0, output_tokens=20_000),
    )
    assert long.agent_cost == pytest.approx(1.95)
    unknown_cache = calculate_call_cost(
        sheet,
        "moonshotai/kimi-k2.5",
        CallUsage(input_tokens=100_000, cached_input_tokens=None, output_tokens=10_000),
    )
    assert unknown_cache.upper_bound is True
    assert unknown_cache.agent_cost == pytest.approx(0.0675)


def test_expansion_decision_is_explicit_and_deferred_without_fresh_payloads() -> None:
    result = decide_acceptance_expansion(
        baseline_scores=[0.4, 0.8, 0.6, 0.4, 0.8],
        candidate_scores=[0.6, 0.6, 0.6, 0.4, 0.8],
        safety_sensitive=False,
        reduces_tool_catalog=False,
        fresh_payloads_available=False,
    )
    assert result.required is True
    assert "mixed paired outcomes" in result.reasons
    assert "near-zero mean paired delta" in result.reasons
    assert result.status == "deferred"


def test_contract_and_neutral_suites_are_deterministic_and_complete() -> None:
    contract = run_contract_suite()
    assert contract.failed == 0
    assert contract.passed >= 20
    assert len(NEUTRAL_CASES) == 16
    assert {case.topic for case in NEUTRAL_CASES} == {
        "lookup",
        "joins",
        "dates",
        "fan-out",
        "communication-authorization",
        "replanning",
        "multiple-windows",
        "malformed-output",
    }
    neutral = run_neutral_suite()
    assert neutral.failed == 0
    assert neutral.passed == 16


def test_frozen_snapshot_has_all_seven_exact_prompts_and_matches_runtime() -> None:
    frozen = load_frozen_snapshot(PROMPT_ROOT / "baseline.json")
    rendered = build_prompt_snapshot(source_revision=frozen["provenance"]["source_revision"])
    assert {row["semantic_label"] for row in frozen["prompts"]} == {
        "plan",
        "ground",
        "select",
        "revalidate",
        "condition",
        "retirement",
        "relevance",
    }
    assert rendered["prompts"] == frozen["prompts"]
    assert {profile["name"] for profile in frozen["evaluation_profiles"]} == {
        "gpt-5.4-medium-prompt",
        "gpt-5.4-high-paper",
        "kimi-k2.5-prompt",
    }
    assert frozen["notes"]["campaigns"] == ["prompt", "aamas2027"]
    for row in frozen["prompts"]:
        assert len(row["system_sha256"]) == len(row["user_sha256"]) == 64
        assert row["system"] and row["user"]
    serialized = json.dumps(frozen).lower()
    assert "scenario_universe" not in serialized
    assert '"oracle":' not in serialized
    assert "sk-" not in serialized


def test_report_redacts_acceptance_details_and_marks_new_safety_violation_hard() -> None:
    baseline = EvaluationRecord.example(
        arm="baseline", suite="acceptance", case_id="secret", score=1.0
    )
    candidate = EvaluationRecord.example(
        arm="candidate",
        suite="acceptance",
        case_id="secret",
        score=0.8,
        safety_violations=1,
        prompt="locked prompt",
        oracle="locked oracle",
        trajectory={"private": True},
    )
    report = build_report([baseline, candidate], detailed_acceptance=False)
    assert report["schema_version"] == 1
    assert report["campaign"] == "prompt"
    assert report["aggregates"]["hard_failure_count"] == 1
    serialized = json.dumps(report)
    assert "locked prompt" not in serialized
    assert "locked oracle" not in serialized
    assert '"private": true' not in serialized
    assert report["provenance"]["statistical_bootstrap_seed"] == 20260831


def test_report_collects_runtime_model_and_sdk_observations() -> None:
    record = EvaluationRecord(
        arm="baseline",
        profile="example",
        suite="neutral",
        capability="lookup",
        case_id="lookup-ordinary",
        repeat=0,
        score=1.0,
        passed=True,
        call_records=(
            {
                "observed_models": ["observed-model"],
                "sdk_observations": [{"name": "openai", "version": "2.0"}],
                "provider_observations": [
                    {"requested": "moonshotai/kimi-k2.5", "summary": "selected=DeepInfra"}
                ],
            },
        ),
    )
    report = build_report([record])
    assert report["provenance"]["actual_sdk_model_observations"] == [
        {"kind": "model", "value": "observed-model"},
        {"kind": "sdk:openai", "value": "2.0"},
        {
            "kind": "provider",
            "value": {
                "requested": "moonshotai/kimi-k2.5",
                "summary": "selected=DeepInfra",
            },
        },
    ]


def test_report_clusters_repeats_by_scenario_for_paired_statistics() -> None:
    records: list[EvaluationRecord] = []
    for repeat in range(3):
        records.extend(
            [
                EvaluationRecord.example(
                    arm="baseline", suite="development", case_id="many", score=0.0
                ),
                EvaluationRecord.example(
                    arm="candidate", suite="development", case_id="many", score=1.0
                ),
            ]
        )
        records[-2] = EvaluationRecord(**(records[-2].to_dict() | {"repeat": repeat}))
        records[-1] = EvaluationRecord(**(records[-1].to_dict() | {"repeat": repeat}))
    records.extend(
        [
            EvaluationRecord.example(arm="baseline", suite="development", case_id="one", score=1.0),
            EvaluationRecord.example(
                arm="candidate", suite="development", case_id="one", score=0.0
            ),
        ]
    )

    report = build_report(records)
    aggregates = report["aggregates"]
    assert len(aggregates["paired_run_deltas"]) == 4
    assert len(aggregates["paired_deltas"]) == 2
    assert aggregates["mean_paired_score_delta"] == pytest.approx(0.0)


def test_acceptance_expansion_uses_five_scenario_clusters_not_fifteen_repeats() -> None:
    records: list[EvaluationRecord] = []
    for case in range(5):
        for repeat in range(3):
            baseline = 0.0 if case == 0 else 1.0
            candidate = 1.0 if case == 0 else (0.0 if case == 1 else 1.0)
            records.append(
                EvaluationRecord(
                    arm="baseline",
                    profile="example",
                    suite="acceptance",
                    capability="search",
                    case_id=f"case-{case}",
                    repeat=repeat,
                    score=baseline,
                    passed=bool(baseline),
                )
            )
            records.append(
                EvaluationRecord(
                    arm="candidate",
                    profile="example",
                    suite="acceptance",
                    capability="search",
                    case_id=f"case-{case}",
                    repeat=repeat,
                    score=candidate,
                    passed=bool(candidate),
                )
            )

    expansion = build_report(records)["aggregates"]["acceptance_expansion"]
    assert expansion["required"] is True
    assert "mixed paired outcomes" in expansion["reasons"]


def test_acceptance_expansion_is_computed_for_each_complete_profile() -> None:
    records: list[EvaluationRecord] = []
    for profile in ("primary", "cross-family"):
        for case in range(5):
            for arm, score in (("baseline", 0.0), ("candidate", 1.0)):
                records.append(
                    EvaluationRecord(
                        arm=arm,
                        profile=profile,
                        suite="acceptance",
                        capability="search",
                        case_id=f"case-{case}",
                        repeat=0,
                        score=score,
                        passed=bool(score),
                    )
                )

    expansion = build_report(records)["aggregates"]["acceptance_expansion"]
    assert set(expansion["by_profile"]) == {"primary", "cross-family"}
    assert all(row["status"] != "not_evaluable" for row in expansion["by_profile"].values())


def test_acceptance_expansion_marks_an_incomplete_or_unscored_profile_not_evaluable() -> None:
    records: list[EvaluationRecord] = []
    for case in range(5):
        records.extend(
            [
                EvaluationRecord(
                    arm="baseline",
                    profile="incomplete",
                    suite="acceptance",
                    capability="search",
                    case_id=f"case-{case}",
                    repeat=0,
                    score=1.0,
                    passed=True,
                ),
                EvaluationRecord(
                    arm="candidate",
                    profile="incomplete",
                    suite="acceptance",
                    capability="search",
                    case_id=f"case-{case}",
                    repeat=0,
                    score=None if case == 4 else 1.0,
                    passed=None if case == 4 else True,
                ),
            ]
        )

    expansion = build_report(records)["aggregates"]["acceptance_expansion"]
    assert expansion["by_profile"]["incomplete"]["status"] == "not_evaluable"


def _sent(text: str) -> list[tuple[str, dict[str, str]]]:
    return [("user", {"text": text})]


@pytest.mark.parametrize(
    ("case_id", "invocations", "messages"),
    [
        ("lookup-ordinary", [SyntheticInvocation("lookup", {"query": "red"})], _sent("blue-1")),
        ("joins-ordinary", [SyntheticInvocation("list_people", {})], _sent("Ada")),
        ("fanout-adversarial", [SyntheticInvocation("list_people", {})], _sent("Ada")),
        (
            "windows-adversarial",
            [SyntheticInvocation("lookup", {"query": "blue"})],
            _sent("blue-1"),
        ),
        (
            "malformed-adversarial",
            [SyntheticInvocation("lookup", {"query": "blue"})],
            _sent("done"),
        ),
    ],
)
def test_live_neutral_scoring_rejects_partial_or_wrong_trajectories(
    case_id: str,
    invocations: list[SyntheticInvocation],
    messages: list[tuple[str, dict[str, str]]],
) -> None:
    passed, _, _ = score_live_case(case_id, invocations, messages)
    assert passed is False


def test_live_neutral_scoring_checks_operation_order_params_and_emitted_result() -> None:
    passed, violations, writes = score_live_case(
        "joins-adversarial",
        [
            SyntheticInvocation("lookup", {"query": "blue"}),
            SyntheticInvocation("list_people", {}),
        ],
        _sent("Ada"),
    )
    assert (passed, violations, writes) == (True, 0, 0)

    replanned, _, _ = score_live_case(
        "replanning-adversarial",
        [
            SyntheticInvocation("lookup", {"query": "blue"}),
            SyntheticInvocation("details", {"id": "blue-1"}),
        ],
        _sent("blue-1"),
    )
    assert replanned is True


def test_costing_applies_context_tiers_to_each_provider_round_trip() -> None:
    from sora.llm import LLMRoundTripUsage

    profile = load_profiles(EVAL_ROOT / "profiles.json")["gpt-5.4-medium-prompt"]
    sheet = PriceSheet.load(EVAL_ROOT / "price_sheets" / "2026-09-02.json")
    call = SimpleNamespace(
        round_trips=2,
        usages=(
            LLMRoundTripUsage(150_000, 0, 0, 0, True),
            LLMRoundTripUsage(150_000, 0, 0, 0, True),
        ),
        semantic_label="plan",
        prompt_version="1",
        prompt_hashes=(),
        system_prompt_sha256=None,
        user_prompt_sha256=None,
        finish_reasons=(),
        cached_input_tokens=0,
        input_tokens=300_000,
        cache_unknown_input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        reasoning_tokens_exact=True,
        observed_models=(),
        sdk_observations=(),
        provider_observations=(),
        latency_seconds=0.0,
        discarded=False,
    )
    rows, total, _ = _call_records_and_cost(SimpleNamespace(inferences=(call,)), profile, sheet)
    assert total == pytest.approx(0.75)
    assert [row["cost"]["tier_max_input_tokens"] for row in rows[0]["round_trip_usage"]] == [
        272000,
        272000,
    ]


def test_costing_marks_a_completed_round_trip_without_usage_unknown() -> None:
    profile = load_profiles(EVAL_ROOT / "profiles.json")["gpt-5.4-medium-prompt"]
    sheet = PriceSheet.load(EVAL_ROOT / "price_sheets" / "2026-09-02.json")
    call = SimpleNamespace(
        round_trips=1,
        usages=(),
        semantic_label="plan",
        prompt_version="1",
        prompt_hashes=(),
        system_prompt_sha256=None,
        user_prompt_sha256=None,
        finish_reasons=(),
        cached_input_tokens=0,
        input_tokens=0,
        cache_unknown_input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        reasoning_tokens_exact=None,
        observed_models=(),
        sdk_observations=(),
        provider_observations=(),
        latency_seconds=0.0,
        discarded=False,
    )
    rows, total, _ = _call_records_and_cost(SimpleNamespace(inferences=(call,)), profile, sheet)
    assert total is None
    assert rows[0]["cost"] is None
    assert rows[0]["usage_available"] is False


def test_report_reserves_budget_when_exact_agent_cost_is_unknown() -> None:
    record = EvaluationRecord(
        arm="candidate",
        profile="example",
        suite="neutral",
        capability="lookup",
        case_id="lookup-ordinary",
        repeat=0,
        score=None,
        passed=None,
        agent_cost=None,
        agent_cost_reserve=3.0,
    )
    budget = build_report([record])["aggregates"]["budget_usage"]
    assert budget["exact_agent_cost"] == 0.0
    assert budget["unknown_agent_cost_reserve"] == 3.0
    assert budget["agent_cost_unavailable_count"] == 1
    assert budget["total"] == 3.0


def test_live_neutral_reports_admitted_logical_calls_not_provider_round_trips(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from examples.gaia2.evaluation.campaigns.prompt.synthetic import SyntheticTool

    tool = SyntheticTool()
    tool.invocations.append(SyntheticInvocation("lookup", {"query": "blue"}))
    report = SimpleNamespace(
        calls=2,
        inferences=(),
        terminal_parse_failures=0,
        latency_seconds=0.0,
        input_tokens=0,
        cached_input_tokens=0,
        cache_unknown_input_tokens=0,
        output_tokens=0,
        thinking_tokens=0,
    )
    agent = SimpleNamespace(
        procedural=SimpleNamespace(logical_calls_admitted=1),
        registry=SimpleNamespace(get=lambda _tool_id: tool),
        communication=SimpleNamespace(sent=_sent("blue-1")),
        working=SimpleNamespace(activities={}),
        cycle=SimpleNamespace(external_action_count=1),
    )

    class _Session:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.llm_report = report

        async def run(self) -> None:
            return None

    monkeypatch.setattr("sora.bootstrap.build_agent", lambda _config: agent)
    monkeypatch.setattr("sora.cli.TerminalSession", _Session)
    profile = load_profiles(EVAL_ROOT / "profiles.json")["gpt-5.4-medium-prompt"]
    sheet = PriceSheet.load(EVAL_ROOT / "price_sheets" / "2026-09-02.json")
    entry = SimpleNamespace(
        arm="candidate",
        profile=profile.name,
        suite="neutral",
        case_id="lookup-ordinary",
        repeat=0,
        reserved_agent_cost=0.5,
    )

    record = _live_neutral_record(
        entry,
        config_path=tmp_path / "agent.yaml",
        profile=profile,
        sheet=sheet,
    )
    assert record.agent_llm_calls == 1
    assert record.provider_round_trips == 0


def test_headless_neutral_stops_when_only_input_waits_remain() -> None:
    from sora.activity import ActivityState
    from sora.types import InputWait

    waiting = SimpleNamespace(
        state=ActivityState.BLOCKED,
        blocked_on=InputWait(prompt="How should I proceed?"),
    )
    terminated = SimpleNamespace(state=ActivityState.TERMINATED, blocked_on=None)
    ready = SimpleNamespace(state=ActivityState.READY, blocked_on=None)

    assert _headless_neutral_done(SimpleNamespace(working=SimpleNamespace(activities={}))) is False
    assert (
        _headless_neutral_done(
            SimpleNamespace(working=SimpleNamespace(activities={0: terminated, 1: waiting}))
        )
        is True
    )
    assert (
        _headless_neutral_done(
            SimpleNamespace(working=SimpleNamespace(activities={0: ready, 1: waiting}))
        )
        is False
    )


def test_live_neutral_checkpoints_provider_failure_as_infrastructure_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from examples.gaia2.evaluation.campaigns.prompt.synthetic import SyntheticTool

    from sora.activity import ActivityState
    from sora.types import InputWait

    error = (
        "BadRequestError(\"Error code: 400 - {'error': {'message': "
        "\"Unsupported value: 'temperature' does not support 0.5 with this model. "
        'Only the default (1) value is supported."}}")'
    )
    inference = SimpleNamespace(
        round_trips=1,
        usages=(),
        semantic_label="plan",
        prompt_version="1",
        prompt_hashes=(),
        system_prompt_sha256="system-hash",
        user_prompt_sha256="user-hash",
        finish_reasons=(),
        cached_input_tokens=0,
        input_tokens=0,
        cache_unknown_input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        reasoning_tokens_exact=None,
        observed_models=(),
        sdk_observations=(),
        provider_observations=(),
        latency_seconds=0.0,
        discarded=False,
        terminal_parse_failures=0,
        outcome="error",
        error=error,
    )
    report = SimpleNamespace(
        calls=1,
        inferences=(inference,),
        terminal_parse_failures=0,
        latency_seconds=0.0,
        input_tokens=0,
        cached_input_tokens=0,
        cache_unknown_input_tokens=0,
        output_tokens=0,
        thinking_tokens=0,
    )
    activity = SimpleNamespace(
        state=ActivityState.BLOCKED,
        blocked_on=InputWait(prompt="How should I proceed?"),
        replan_count=1,
    )
    tool = SyntheticTool()
    agent = SimpleNamespace(
        procedural=SimpleNamespace(logical_calls_admitted=1),
        registry=SimpleNamespace(get=lambda _tool_id: tool),
        communication=SimpleNamespace(sent=[]),
        working=SimpleNamespace(activities={"activity": activity}),
        cycle=SimpleNamespace(external_action_count=0),
    )

    class _Session:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            self.llm_report = report
            self._stop_when = cast(Callable[[], bool], kwargs["stop_when"])

        async def run(self) -> None:
            assert self._stop_when()

    monkeypatch.setattr("sora.bootstrap.build_agent", lambda _config: agent)
    monkeypatch.setattr("sora.cli.TerminalSession", _Session)
    profile = load_profiles(EVAL_ROOT / "profiles.json")["gpt-5.4-medium-prompt"]
    sheet = PriceSheet.load(EVAL_ROOT / "price_sheets" / "2026-09-02.json")
    entry = SimpleNamespace(
        arm="baseline",
        profile=profile.name,
        suite="neutral",
        case_id="lookup-ordinary",
        repeat=0,
        reserved_agent_cost=0.5,
    )

    record = _live_neutral_record(
        entry,
        config_path=tmp_path / "agent.yaml",
        profile=profile,
        sheet=sheet,
    )

    assert record.score is None
    assert record.passed is None
    assert record.terminal_cause == "infrastructure_error"
    assert record.status == f"error: {error}"
    assert record.call_records[0]["outcome"] == "error"
    assert record.call_records[0]["error"] == error


def test_completed_checkpoint_resume_needs_no_removed_provider_credential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile_name = "gpt-5.4-medium-prompt"
    manifests = load_manifests(PROMPT_ROOT / "manifests")
    matrix = build_run_matrix(
        RunSelection(
            profiles=(profile_name,),
            suites=("development",),
            arm="baseline",
        ),
        BudgetPolicy(),
        manifests,
    )
    checkpoint = tmp_path / "checkpoint.jsonl"
    for entry in matrix.entries:
        _append_checkpoint(
            checkpoint,
            entry.key,
            EvaluationRecord(
                arm=entry.arm,
                profile=entry.profile,
                suite=entry.suite,
                capability=entry.capability,
                case_id=entry.case_id,
                repeat=entry.repeat,
                score=1.0,
                passed=True,
            ),
        )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = _parser().parse_args(
        [
            "prompt",
            "run",
            "--profile",
            profile_name,
            "--suite",
            "development",
            "--arm",
            "baseline",
            "--output-dir",
            str(tmp_path),
            "--price-sheet",
            str(EVAL_ROOT / "price_sheets" / "2026-09-02.json"),
            "--confirm-budget",
            "180",
        ]
    )
    assert _run_command(args) == 0


def test_evaluation_cli_and_readme_name_both_campaigns() -> None:
    args = _parser().parse_args(["prompt", "check"])
    assert args.campaign == "prompt"
    readme = (EVAL_ROOT / "README.md").read_text()
    assert "python -m examples.gaia2.evaluation prompt check" in readme
    assert "aamas2027" in readme


def test_prompt_campaign_defaults_to_200_logical_agent_calls_not_cycles() -> None:
    args = _parser().parse_args(
        [
            "prompt",
            "run",
            "--profile",
            "gpt-5.4-medium-prompt",
            "--suite",
            "development",
            "--arm",
            "baseline",
            "--output-dir",
            "/tmp/eval",
            "--price-sheet",
            str(EVAL_ROOT / "price_sheets" / "2026-09-02.json"),
            "--confirm-budget",
            "180",
        ]
    )
    assert args.max_agent_llm_calls == 200
    assert not hasattr(args, "max_steps")


def test_evaluation_record_reports_architecture_units_without_claiming_cycles_are_steps() -> None:
    record = EvaluationRecord(
        arm="candidate",
        profile="example",
        suite="development",
        capability="execution",
        case_id="case",
        repeat=0,
        score=1.0,
        passed=True,
        agent_llm_calls=7,
        agent_llm_call_limit=200,
        provider_round_trips=8,
        external_actions=12,
        decision_cycles=400,
    )

    row = record.to_dict()
    assert row["agent_llm_calls"] == 7
    assert row["agent_llm_call_limit"] == 200
    assert row["provider_round_trips"] == 8
    assert row["external_actions"] == 12
    assert row["decision_cycles"] == 400
    assert "step_unit" not in row
    assert "calls" not in row
    assert "round_trips" not in row
    assert build_report([record])["provenance"]["agent_llm_call_limits"] == [200]


def test_pre_correction_checkpoint_call_fields_migrate_into_report_schema() -> None:
    raw = EvaluationRecord.example(
        arm="baseline", suite="development", case_id="legacy", score=1.0
    ).to_dict()
    raw.pop("agent_llm_calls")
    raw.pop("provider_round_trips")
    raw.pop("external_actions")
    raw["calls"] = 9
    raw["round_trips"] = 10
    raw["step_unit"] = "decision_cycle"

    record = record_from_dict(raw)

    assert record.agent_llm_calls == 9
    assert record.provider_round_trips == 10
    assert record.external_actions == 0
    assert "step_unit" not in record.to_dict()
