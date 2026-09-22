"""``examples/gaia2/batch.py`` + ``_runner.py`` — pure formatting/aggregation and the turn-aware
stop predicate, tested directly (no ARE, no model tokens). The parts that touch ARE or spend tokens
(scenario iteration, judge, trace export) are exercised by the operator-run correctness gate, not
here.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from examples.gaia2 import _runner, batch
from examples.gaia2._runner import (
    _awaiting_input,
    _context_overflow,
    _make_stop_when,
    _terminal_cause,
)
from examples.gaia2.batch import (
    ARTIFACT_ATTESTATION,
    ARTIFACT_CHECKSUMS,
    CONTAMINATION_MARKER,
    SweepManifest,
    _arm_root,
    _check_operating_point,
    _jsonl_record,
    _load_sweep_manifest,
    _parse_args,
    _pass_at_1,
    _pinned_hf_scenarios,
    _profile_for_config,
    _resolve_model_label,
    _resolve_react_label,
    _run_capability,
    _run_one_scenario,
    _score_status,
    _select_manifest_scenarios,
    _verdict_parse,
    _verify_counterpart_pairing,
    aggregate,
    main,
    refuse_contaminated_root,
    verify_artifact_attestation,
    write_artifact_attestation,
)

from sora.activity import ActivityState
from sora.types import ConditionWait, InputWait, SignalWait

# -- _score_status: the four ARE-parity cases ----------------------------------------------------


def test_score_status_success() -> None:
    assert _score_status(True, None) == (1.0, "success")


def test_score_status_failure() -> None:
    assert _score_status(False, None) == (0.0, "failed")


def test_score_status_exception() -> None:
    assert _score_status(None, RuntimeError("boom")) == (None, "exception")


def test_score_status_no_validation() -> None:
    assert _score_status(None, None) == (None, "no_validation")


# -- _resolve_model_label: the trace label can't disagree with the configured model ---------------


def _write_agent_yaml(root: Path, model: str | None) -> str:
    llm = f"  llm:\n    model: {model}\n" if model is not None else ""
    path = root / "agent.yaml"
    path.write_text(
        "agent:\n"
        "  name: t\n"
        "  strategies:\n"
        "    reason: sora.strategies.DefaultReasonStrategy\n" + llm,
        encoding="utf-8",
    )
    return str(path)


def test_model_label_defaults_to_the_configured_model(tmp_path: Path) -> None:
    config = _write_agent_yaml(tmp_path, "claude-opus-4-8")
    assert _resolve_model_label(None, config) == "claude-opus-4-8"


def test_model_label_kept_when_it_names_the_configured_model(tmp_path: Path) -> None:
    config = _write_agent_yaml(tmp_path, "claude-opus-4-8")
    assert _resolve_model_label("S-ORA/claude-opus-4-8", config) == "S-ORA/claude-opus-4-8"


def test_model_label_naming_another_model_is_refused(tmp_path: Path) -> None:
    # The failure this exists for: a config left on some other model, swept under an Opus label.
    config = _write_agent_yaml(tmp_path, "gpt-5.5")
    with pytest.raises(SystemExit, match="gpt-5.5"):
        _resolve_model_label("S-ORA/claude-opus-4-8", config)


def test_model_label_untouched_when_config_names_no_model(tmp_path: Path) -> None:
    config = _write_agent_yaml(tmp_path, None)
    assert _resolve_model_label("S-ORA/whatever", config) == "S-ORA/whatever"
    assert _resolve_model_label(None, config) is None


# -- _jsonl_record: matches ARE's _export_benchmark_result_jsonl shape ----------------------------


def test_jsonl_record_success_strips_none_metadata_keeps_false_has_exception() -> None:
    rec = _jsonl_record(
        scenario_id="s1",
        run_number=0,
        success=True,
        rationale=None,
        exception=None,
        trace_id="/traces/s1.json",
    )
    assert rec == {
        "task_id": "s1",
        "trace_id": "/traces/s1.json",
        "score": 1.0,
        "metadata": {
            "scenario_id": "s1",
            "run_number": 0,
            "status": "success",
            "has_exception": False,  # a False value is kept; only None values are stripped
        },
    }


def test_jsonl_record_marks_a_run_ARE_s_clock_ended() -> None:
    # A truncated run's score is not a measurement of the agent: past scenario.duration the
    # environment stops and later turns are never delivered, so the judge's zero and the gate's
    # "missing" list describe turns the agent never saw. Recorded so an aggregate can exclude it
    # rather than average in a number about the host machine's speed.
    rec = _jsonl_record(
        scenario_id="s5",
        run_number=0,
        success=False,
        rationale="Validation called at turn -1 but nb_turns is 2",
        exception=None,
        trace_id=None,
        timeline_expired=True,
    )
    assert rec["metadata"]["timeline_expired"] is True
    assert rec["score"] == 0.0  # still scored as ARE scored it; the flag qualifies it, not replaces


def test_jsonl_record_states_how_judge_verdicts_were_parsed() -> None:
    """Scoring provenance, not an optional diagnostic. The default relaxes ARE's case-sensitive
    verdict parse, so a sweep's scores come from a patched judge and cannot be compared with one
    produced by stock ARE — a record that does not say which it was is uninterpretable on its own.
    """
    rec = _jsonl_record(
        scenario_id="s1",
        run_number=0,
        success=True,
        rationale=None,
        exception=None,
        trace_id=None,
        verdict_parse="case-insensitive",
    )
    assert rec["metadata"]["verdict_parse"] == "case-insensitive"

    strict = _jsonl_record(
        scenario_id="s1",
        run_number=0,
        success=True,
        rationale=None,
        exception=None,
        trace_id=None,
        verdict_parse="stock",
    )
    assert strict["metadata"]["verdict_parse"] == "stock"


def test_verdict_parse_follows_the_flag_and_is_absent_without_a_judge() -> None:
    # Nothing scored an unscored run, so there were no verdicts to parse either way.
    assert _verdict_parse(Namespace(judge_model=None, strict_verdict_case=False)) is None
    assert (
        _verdict_parse(Namespace(judge_model="m", strict_verdict_case=False)) == "case-insensitive"
    )
    assert _verdict_parse(Namespace(judge_model="m", strict_verdict_case=True)) == "stock"


def test_jsonl_record_exception_carries_type_and_message() -> None:
    rec = _jsonl_record(
        scenario_id="s2",
        run_number=1,
        success=None,
        rationale="graph mismatch",
        exception=ValueError("bad graph"),
        trace_id=None,
    )
    assert rec["score"] is None
    assert rec["trace_id"] is None  # top-level trace_id is kept even when None (ARE parity)
    assert rec["metadata"]["status"] == "exception"
    assert rec["metadata"]["has_exception"] is True
    assert rec["metadata"]["exception_type"] == "ValueError"
    assert rec["metadata"]["exception_message"] == "bad graph"
    assert rec["metadata"]["rationale"] == "graph mismatch"


def test_jsonl_record_reply_count_mismatch_carries_its_evidence() -> None:
    """Replies to the user are tallied apart from the domain tools (the judge tolerates a few
    extra), so a turn that made exactly the right tool calls and one reply too many has an empty
    `surplus` AND an empty `missing`. Recorded with only those two, the line said a turn mismatched
    and gave nothing that mismatched — unreadable, and indistinguishable from a bug in the check."""
    from sora.adapters.are_sim import TurnWriteCounts, WriteCountCheck

    turn = TurnWriteCounts(
        turn=2,
        agent={"EmailClientApp__send_email": 1},
        oracle={"EmailClientApp__send_email": 1},
        agent_user_replies=3,
        oracle_user_replies=1,
        extra_user_replies_allowed=1,
    )
    assert not turn.passed and not turn.surplus and not turn.missing  # the shape being guarded

    rec = _jsonl_record(
        scenario_id="s3",
        run_number=0,
        success=False,
        rationale=None,
        exception=None,
        trace_id=None,
        write_counts=WriteCountCheck(turns=(turn,)),
    )
    assert rec["metadata"]["write_count_mismatch"] == [
        {
            "turn": 2,
            "surplus": {},
            "missing": {},
            "user_replies": {"agent": 3, "oracle": 1, "extra_allowed": 1},
        }
    ]


def test_jsonl_record_tool_count_mismatch_omits_the_reply_counts() -> None:
    # The reply tally is evidence only when it is the thing that failed; a surplus tool call
    # already says what went wrong, and every mismatch record staying lean is what keeps a sweep's
    # output.jsonl readable.
    from sora.adapters.are_sim import TurnWriteCounts, WriteCountCheck

    turn = TurnWriteCounts(
        turn=0,
        agent={"EmailClientApp__reply_to_email": 1},
        oracle={},
        agent_user_replies=1,
        oracle_user_replies=1,
        extra_user_replies_allowed=1,
    )
    rec = _jsonl_record(
        scenario_id="s4",
        run_number=0,
        success=False,
        rationale=None,
        exception=None,
        trace_id=None,
        write_counts=WriteCountCheck(turns=(turn,)),
    )
    assert rec["metadata"]["write_count_mismatch"] == [
        {"turn": 0, "surplus": {"EmailClientApp__reply_to_email": 1}, "missing": {}}
    ]


# -- pass@1 + aggregate ---------------------------------------------------------------------------


def test_pass_at_1_excludes_unscored_records() -> None:
    records: list[dict[str, Any]] = [
        {"score": 1.0},
        {"score": 0.0},
        {"score": 1.0},
        {"score": None},  # unscored — excluded from both the mean and the denominator
        {"score": 0.0, "metadata": {"timeline_expired": True}},
    ]
    pass_at_1, scored, total = _pass_at_1(records)
    assert pass_at_1 == 2 / 3
    assert (scored, total) == (3, 5)


def test_pass_at_1_all_unscored_is_none() -> None:
    assert _pass_at_1([{"score": None}, {"score": None}]) == (None, 0, 2)


def _write_config(root: Path, config: str, scores: list[float | None]) -> None:
    cfg_dir = root / "standard" / config
    cfg_dir.mkdir(parents=True)
    with (cfg_dir / "output.jsonl").open("w", encoding="utf-8") as f:
        for i, s in enumerate(scores):
            f.write(json.dumps({"task_id": f"{config}-{i}", "score": s, "metadata": {}}) + "\n")


def test_aggregate_equal_weights_core_capabilities(tmp_path: Path) -> None:
    # execution 100%, search 50%; a non-core config (mini) is reported but excluded from `overall`.
    _write_config(tmp_path, "execution", [1.0, 1.0])
    _write_config(tmp_path, "search", [1.0, 0.0])
    _write_config(tmp_path, "mini", [0.0, 0.0])

    summary = aggregate(str(tmp_path))

    assert summary["configs"]["execution"]["pass_at_1"] == 1.0
    assert summary["configs"]["search"]["pass_at_1"] == 0.5
    assert summary["configs"]["mini"]["pass_at_1"] == 0.0
    # equal-weight over the two *core* configs present: (1.0 + 0.5) / 2 = 0.75 — mini excluded.
    assert summary["exploratory_mean"] == 0.75
    # ... but two of five capabilities is not a Gaia2 headline, whatever the mean of them is.
    assert summary["overall"] is None
    assert any("capabilities not run" in reason for reason in summary["headline_withheld"])


def test_aggregate_empty_dir_is_safe(tmp_path: Path) -> None:
    summary = aggregate(str(tmp_path))
    assert summary == {
        "configs": {},
        "overall": None,
        "exploratory_mean": None,
        "headline_withheld": (
            "capabilities not run: execution, search, adaptability, time, ambiguity",
            "scenario coverage unverified: no --scenario-manifest given",
        ),
        "overall_clock_modes": [],
        "overall_scenario_manifest_digests": [],
    }


def test_aggregate_reports_charge_provenance_and_surfaces_anomaly_disagreement(
    tmp_path: Path,
) -> None:
    cfg_dir = tmp_path / "standard" / "time"
    cfg_dir.mkdir(parents=True)
    row: dict[str, Any] = {
        "task_id": "clock",
        "score": 1.0,
        "metadata": {
            "charged_seconds": 7.5,
            "charge_model_identity": {"provider": "openai", "model": "gpt"},
            "charge_model_digest": "digest",
            "cached_input_clamps": 1,
            "raw_cached_input_anomalies": 1,
            "charge_accounting_consistent": True,
            "inference_charge_policy": "serialized_sum",
            "llm_wall_seconds": 9.0,
            "llm_wall_union_seconds": 7.0,
            "llm_charged_union_seconds": 5.5,
            "llm_round_trips": 4,
            "llm_max_in_flight": 2,
            "llm_overlapped_round_trips": 3,
            "scenario_manifest_digest": "manifest-digest",
        },
    }
    path = cfg_dir / "output.jsonl"
    path.write_text(json.dumps(row) + "\n")

    report = aggregate(str(tmp_path))["configs"]["time"]
    assert report["charged_seconds"] == 7.5
    assert report["llm_wall_seconds"] == 9.0
    assert report["llm_wall_union_seconds"] == 7.0
    assert report["llm_wall_overlap_seconds"] == 2.0
    assert report["llm_charged_union_seconds"] == 5.5
    assert report["llm_charged_overlap_seconds"] == 2.0
    assert report["llm_round_trips"] == 4
    assert report["llm_max_in_flight"] == 2
    assert report["llm_overlapped_round_trips"] == 3
    assert report["scenario_manifest_digests"] == ["manifest-digest"]
    assert report["mixed_scenario_manifests"] is False
    assert report["charge_accounting_mismatches"] == 0
    assert report["inference_charge_policies"] == ["serialized_sum"]
    assert report["charge_models"] == [
        {
            "identity": {"provider": "openai", "model": "gpt"},
            "digest": "digest",
        }
    ]

    row["metadata"]["raw_cached_input_anomalies"] = 0
    row["metadata"]["charge_accounting_consistent"] = False
    path.write_text(json.dumps(row) + "\n")
    report = aggregate(str(tmp_path))["configs"]["time"]
    assert report["charge_accounting_mismatches"] == 1


def test_charge_accounting_mismatch_is_recorded_without_aborting_scenario(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    charge = SimpleNamespace(cached_input_clamps=1, identity={"model": "m"})
    charge_model = SimpleNamespace(charge_for=lambda _profile: charge, digest="digest")
    result = SimpleNamespace(
        outcome=SimpleNamespace(success=None, rationale=None),
        environment=None,
        exception=None,
        awaiting_input=[],
        write_counts=None,
        judge_recording=None,
        timeline_expired=False,
        harness_truncation=None,
        terminal_cause="unscored_completion",
        charged_seconds=2.0,
        charge_model_identity=charge.identity,
        charge_model_digest="digest",
        cached_input_clamps=1,
        raw_cached_input_anomalies=0,
        charge_accounting_consistent=False,
        clock_mode="token_charged",
        inference_charge_policy="serialized_sum",
        llm_wall_seconds=3.0,
        llm_wall_union_seconds=2.0,
        llm_charged_union_seconds=1.5,
        llm_round_trips=2,
        llm_max_in_flight=2,
        llm_overlapped_round_trips=2,
    )
    monkeypatch.setattr("examples.gaia2._runner.run_scenario", lambda *_a, **_k: result)
    monkeypatch.setattr("sora.adapters.are_sim.populate_oracle_events", lambda _scenario: None)
    args = Namespace(
        charge_model=charge_model,
        model_profile=object(),
        judge_model=None,
        judge_provider=None,
        judge_endpoint=None,
        strict_verdict_case=False,
        init_turns=False,
        arm="sora",
        config="agent.yaml",
        verbose=False,
        max_wall_seconds=10.0,
        model="m",
        wall_clock=False,
    )

    row = _run_one_scenario(
        SimpleNamespace(scenario_id="s"), 0, args, str(tmp_path), llm_calls=None
    )

    assert row["metadata"]["charge_accounting_consistent"] is False
    assert row["metadata"]["llm_round_trips"] == 2
    assert row["metadata"]["llm_max_in_flight"] == 2
    assert row["metadata"]["llm_overlapped_round_trips"] == 2

    config_dir = tmp_path / "standard" / "time"
    config_dir.mkdir(parents=True)
    (config_dir / "output.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    reported = aggregate(str(tmp_path))["configs"]["time"]
    assert reported["llm_round_trips"] == 2
    assert reported["llm_max_in_flight"] == 2
    assert reported["llm_overlapped_round_trips"] == 2


# -- _make_stop_when: the turn-aware predicate ----------------------------------------------------

# The predicate takes an absolute monotonic instant, so a test that is not about the cap
# names one no clock in these tests reaches -- including the fake clocks they install.
_FAR_DEADLINE = 1e9


def _agent_with(
    states: list[ActivityState],
    blocked_on: list[Any] | None = None,
    *,
    cycle_count: int = 0,
    logical_call_limit_exceeded: bool = False,
) -> SimpleNamespace:
    waits: list[Any] = blocked_on if blocked_on is not None else [None] * len(states)
    activities = {
        i: SimpleNamespace(state=st, blocked_on=w)
        for i, (st, w) in enumerate(zip(states, waits, strict=True))
    }
    return SimpleNamespace(
        working=SimpleNamespace(activities=activities),
        cycle=SimpleNamespace(cycle_count=cycle_count),
        procedural=SimpleNamespace(
            logical_call_limit_exceeded=logical_call_limit_exceeded,
        ),
    )


def _sim(
    *,
    running: bool,
    paused: bool = False,
    all_turns_answered: bool = False,
) -> SimpleNamespace:
    """The predicate reads both flags, so a fake has to answer both. They are not exclusive: a
    paused environment is one ARE holds mid-turn around a judge call, and ``is_running()`` counts
    that as live — the pair a real ``AreSimulation`` reports while a judge is in flight is
    ``running=True, paused=True``."""
    return SimpleNamespace(
        is_running=lambda: running,
        is_paused=lambda: paused,
        all_turns_answered=lambda: all_turns_answered,
    )


def test_stop_when_none_when_exit_when_idle_set() -> None:
    # Opting into the quiet-window heuristic means no custom predicate.
    assert _make_stop_when(SimpleNamespace(), SimpleNamespace(), 8.0, _FAR_DEADLINE) is None


def test_stop_when_rides_through_live_timeline() -> None:
    sim = _sim(running=True)
    agent = _agent_with([ActivityState.TERMINATED])  # even fully idle, a live timeline keeps going
    predicate = _make_stop_when(sim, agent, None, _FAR_DEADLINE)
    assert predicate is not None
    assert predicate() is False


def test_stop_when_ends_live_timeline_after_the_final_reply() -> None:
    sim = _sim(running=True, all_turns_answered=True)
    agent = _agent_with([ActivityState.TERMINATED])
    predicate = _make_stop_when(sim, agent, None, _FAR_DEADLINE)
    assert predicate is not None

    assert predicate() is True
    assert predicate.reason == "verification_completion"


def test_stop_when_waits_for_final_work_to_settle_after_the_final_reply() -> None:
    sim = _sim(running=True, all_turns_answered=True)
    agent = _agent_with([ActivityState.READY])
    predicate = _make_stop_when(sim, agent, None, _FAR_DEADLINE)
    assert predicate is not None

    assert predicate() is False


def test_stop_when_stops_once_timeline_done_and_all_terminated() -> None:
    sim = _sim(running=False)
    predicate = _make_stop_when(sim, _agent_with([ActivityState.TERMINATED]), None, _FAR_DEADLINE)
    assert predicate is not None
    assert predicate() is True
    # Not because the cap elapsed: the run completed.
    assert predicate.reason == "verification_completion"


def test_stop_when_waits_for_in_flight_activity_after_timeline_done() -> None:
    sim = _sim(running=False)
    agent = _agent_with([ActivityState.TERMINATED, ActivityState.READY])
    predicate = _make_stop_when(sim, agent, None, _FAR_DEADLINE)
    assert predicate is not None
    assert predicate() is False  # timeline ended but the agent still has work in flight


def test_stop_when_wall_clock_cap_fires() -> None:
    sim = _sim(running=True)  # timeline still live, but the cap wins
    predicate = _make_stop_when(sim, _agent_with([ActivityState.READY]), None, -1.0)
    assert predicate is not None
    assert predicate() is True
    assert predicate.reason == "timeout"


def test_stop_when_enforces_the_explicit_logical_agent_call_limit() -> None:
    predicate = _make_stop_when(
        _sim(running=True),
        _agent_with([ActivityState.READY], logical_call_limit_exceeded=True),
        None,
        _FAR_DEADLINE,
    )
    assert predicate is not None
    assert predicate() is True
    assert predicate.reason == "llm_call_limit"


def test_stop_when_does_not_treat_decision_cycles_as_benchmark_steps() -> None:
    predicate = _make_stop_when(
        _sim(running=True),
        _agent_with([ActivityState.READY], cycle_count=10_000),
        None,
        _FAR_DEADLINE,
    )
    assert predicate is not None
    assert predicate() is False


def test_context_overflow_is_classified_without_provider_specific_exception_types() -> None:
    assert _context_overflow(RuntimeError("maximum context length exceeded")) is True
    assert _context_overflow(RuntimeError("connection reset")) is False


def test_terminal_cause_does_not_call_an_unscored_timeline_stop_verification() -> None:
    assert _terminal_cause(None, False, "verification_completion", None) == "unscored_completion"
    assert _terminal_cause(None, False, "verification_completion", False) == (
        "verification_completion"
    )


def test_terminal_cause_reads_provider_failures_captured_off_cycle() -> None:
    assert (
        _terminal_cause(
            None,
            False,
            "verification_completion",
            None,
            inference_errors=("BadRequestError('maximum context length exceeded')",),
        )
        == "context_overflow"
    )
    assert (
        _terminal_cause(
            None,
            False,
            "verification_completion",
            None,
            inference_errors=("ConnectionError('connection reset')",),
        )
        == "infrastructure_error"
    )


def test_terminal_cause_reads_ares_logged_iteration_exhaustion() -> None:
    assert (
        _terminal_cause(
            None,
            False,
            None,
            False,
            max_iterations_reached=True,
        )
        == "llm_call_limit"
    )


def test_terminal_cause_keeps_a_raised_error_ahead_of_iteration_exhaustion() -> None:
    assert (
        _terminal_cause(
            RuntimeError("runner crashed"),
            False,
            None,
            None,
            max_iterations_reached=True,
        )
        == "infrastructure_error"
    )


# -- the pause cap: bound a stalled judge without bounding a slow scenario ------------------------


def test_stop_when_rides_through_a_judge_pause() -> None:
    """The run-6 shape, pinned: ARE pauses the timeline around each per-turn judge call, so the
    environment is mid-turn while every activity sits on the conditions its exhausted plan body
    left behind — which past the end of a timeline is exactly what "finished" looks like. The run
    has to sit through the pause instead, or it is torn down inside the judge's own window and
    every later turn is lost."""
    sim = _sim(running=True, paused=True)
    agent = _agent_with([ActivityState.BLOCKED], [ConditionWait()])
    predicate = _make_stop_when(sim, agent, None, _FAR_DEADLINE)
    assert predicate is not None
    assert predicate() is False


def test_stop_when_a_pause_that_never_ends_is_a_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    """A judge that never answers leaves the environment paused for good — ARE's pause/resume
    bracket is not exception-safe. Without a cap the run sits out its whole wall clock, which on a
    sweep is the difference between three minutes and twenty per hung scenario."""
    clock = 1000.0
    monkeypatch.setattr("examples.gaia2._runner.time.monotonic", lambda: clock)
    sim = _sim(running=True, paused=True)
    predicate = _make_stop_when(sim, _agent_with([ActivityState.READY]), None, _FAR_DEADLINE)
    assert predicate is not None
    assert predicate() is False  # the first poll only starts the pause clock

    clock += _runner.MAX_PAUSE_SECONDS - 1
    assert predicate() is False  # still inside the allowance
    clock += 1
    assert predicate() is True


def test_stop_when_pause_allowance_resets_between_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap is per pause, not cumulative: a scenario pauses once per turn, and summing those
    would fail a perfectly healthy multi-turn run."""
    clock = 1000.0
    monkeypatch.setattr("examples.gaia2._runner.time.monotonic", lambda: clock)
    paused = True
    sim = SimpleNamespace(is_running=lambda: True, is_paused=lambda: paused)
    predicate = _make_stop_when(sim, _agent_with([ActivityState.READY]), None, _FAR_DEADLINE)
    assert predicate is not None
    assert predicate() is False

    clock += _runner.MAX_PAUSE_SECONDS - 1  # a long, but answered, judge call
    paused = False
    assert predicate() is False  # resumed: the allowance is spent, not banked

    paused = True  # the next turn's judge call starts its own allowance
    assert predicate() is False
    clock += _runner.MAX_PAUSE_SECONDS - 1
    assert predicate() is False


# -- _wall_deadline: one cap over the same phases on both arms ------------------------------------


def _fake_sim(environments: list[Any]) -> Any:
    """A simulation whose ``environment()`` yields each entry in turn, then repeats the last."""

    def environment() -> Any:
        return environments[0] if len(environments) == 1 else environments.pop(0)

    return SimpleNamespace(environment=environment)


def test_wall_deadline_stops_the_environment_once_the_cap_passes() -> None:
    stopped = threading.Event()
    env = SimpleNamespace(stop=stopped.set)

    with _runner._wall_deadline(_fake_sim([env]), time.monotonic() - 1.0) as expired:
        assert stopped.wait(2.0), "the deadline never reached the environment"

    assert expired() is True


def test_wall_deadline_waits_for_an_environment_that_does_not_exist_yet() -> None:
    """The cap must not retire when it fires before the environment is published.

    Returning at that point leaves the timeout flagged but never enforced, so a run whose setup is
    what overran would then proceed uncapped -- the failure this guards is a silent one.
    """
    stopped = threading.Event()
    env = SimpleNamespace(stop=stopped.set)
    # Two polls see nothing; only the third finds a constructed environment.
    sim = _fake_sim([None, None, env])

    with _runner._wall_deadline(sim, time.monotonic() - 1.0):
        assert stopped.wait(2.0), "the watchdog retired before the environment appeared"


def test_wall_deadline_leaves_a_run_that_finishes_in_time_alone() -> None:
    stopped = threading.Event()
    env = SimpleNamespace(stop=stopped.set)

    with _runner._wall_deadline(_fake_sim([env]), time.monotonic() + 600.0) as expired:
        assert expired() is False

    assert not stopped.is_set()


def test_wall_deadline_survives_an_environment_probe_that_raises() -> None:
    """Construction can fail or tear down mid-probe; the watchdog must not die of it."""

    def environment() -> Any:
        raise RuntimeError("not constructed")

    with _runner._wall_deadline(SimpleNamespace(environment=environment), time.monotonic() - 1.0):
        time.sleep(0.2)


# -- --init-turns: turn wiring without a judge ----------------------------------------------------
#
# The flag's *effect* on a scenario graph (turn triggers built, later-turn events re-anchored off
# their OracleEvents) needs a real ARE scenario and is covered by the operator-run gate. What is
# pinned here is the driver wiring, which is where the flag can silently become a no-op: that it
# reaches ``initialize_turns``, that a judge still wins when one is asked for, and that asking for
# both is refused rather than quietly resolved in the judge's favour. ``run_benchmark.main`` imports
# its ARE seams lazily *inside* ``main``, so patching the module attributes reaches the real branch
# without ARE installed and without a scenario or a run.


def _patch_seams(monkeypatch: Any) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr("sora.adapters.are_sim.load_scenario", lambda ref: "SCENARIO")
    monkeypatch.setattr(
        "sora.adapters.are_sim.initialize_turns", lambda s: calls.append("init"), raising=False
    )
    monkeypatch.setattr("sora.adapters.are_sim.attach_judge", lambda *a, **k: calls.append("judge"))
    monkeypatch.setattr(
        "sora.adapters.are_sim.populate_oracle_events",
        lambda s: calls.append("oracle"),
        raising=False,
    )
    monkeypatch.setattr(
        "examples.gaia2._runner.run_scenario",
        lambda *a, **k: SimpleNamespace(
            outcome=SimpleNamespace(success=None, rationale=None),
            exception=None,
            write_counts=None,
            judge_recording=None,
        ),
    )
    return calls


def test_init_turns_wires_turns_without_a_judge(monkeypatch: Any) -> None:
    # Oracle replay first: it soft_resets the apps, so it has to precede the turn wiring (ARE's own
    # ordering) — the order asserted here is load-bearing, not incidental.
    from examples.gaia2.run_benchmark import main

    calls = _patch_seams(monkeypatch)
    main(["--scenario", "s.json", "--init-turns"])
    assert calls == ["oracle", "init"]


def test_judge_model_attaches_the_judge_not_the_bare_turn_init(monkeypatch: Any) -> None:
    from examples.gaia2.run_benchmark import main

    calls = _patch_seams(monkeypatch)
    main(["--scenario", "s.json", "--judge-model", "some-model"])
    assert calls == ["judge"]


def test_a_judge_run_does_not_replay_the_oracle_twice(monkeypatch: Any) -> None:
    """attach_judge already populates the oracle log as a side effect of preprocessing, so the
    standalone replay must not also run — it would be pure duplicated work."""
    from examples.gaia2.run_benchmark import main

    calls = _patch_seams(monkeypatch)
    main(["--scenario", "s.json", "--judge-model", "some-model"])
    assert "oracle" not in calls


def test_a_plain_run_still_replays_the_oracle_for_the_write_count_gate(monkeypatch: Any) -> None:
    """No judge and no --init-turns still replays the oracle: it is deterministic and modelless,
    and it is the only thing that lets an unscored run report ARE's tool-call-count gate — the
    check that would have caught run 4's surplus reply_to_email. Turn wiring stays opt-in."""
    from examples.gaia2.run_benchmark import main

    calls = _patch_seams(monkeypatch)
    main(["--scenario", "s.json"])
    assert calls == ["oracle"]


def test_single_scenario_driver_wires_one_frozen_charge(monkeypatch: Any) -> None:
    from examples.gaia2.run_benchmark import main

    _patch_seams(monkeypatch)
    captured: dict[str, Any] = {}

    def run(*_args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(
            outcome=SimpleNamespace(success=None, rationale=None),
            exception=None,
            write_counts=None,
            judge_recording=None,
            charged_seconds=0.0,
            raw_cached_input_anomalies=0,
        )

    monkeypatch.setattr("examples.gaia2._runner.run_scenario", run)
    main(["--scenario", "s.json"])

    charge = captured["charge"]
    assert captured["charge_model_identity"] == charge.identity
    assert captured["charge_model_digest"]
    assert charge.cached_input_clamps == 0


def test_single_scenario_driver_can_run_on_wall_clock(monkeypatch: Any) -> None:
    from examples.gaia2.run_benchmark import main

    _patch_seams(monkeypatch)
    captured: dict[str, Any] = {}

    def run(*_args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(
            outcome=SimpleNamespace(success=None, rationale=None),
            exception=None,
            write_counts=None,
            judge_recording=None,
            charged_seconds=0.0,
            raw_cached_input_anomalies=0,
        )

    monkeypatch.setattr("examples.gaia2._runner.run_scenario", run)
    main(["--scenario", "s.json", "--wall-clock"])

    assert captured["charge"] is None
    assert captured["charge_model_identity"]["model"] == "gpt-5.4-2026-03-05"
    assert captured["charge_model_digest"] is not None


def test_a_failed_oracle_replay_does_not_abort_the_run(monkeypatch: Any) -> None:
    """The replay exists only to report ARE's write-count gate, which is extra information about a
    run that is otherwise perfectly runnable — and this is the unscored path, where nobody asked to
    be scored at all. A failure there must cost the gate, not the run."""
    from examples.gaia2.run_benchmark import main

    calls = _patch_seams(monkeypatch)

    def _boom(_scenario: Any) -> None:
        calls.append("oracle")
        raise RuntimeError("oracle replay failed: [ValueError('nope')]")

    monkeypatch.setattr("sora.adapters.are_sim.populate_oracle_events", _boom, raising=False)
    main(["--scenario", "s.json", "--init-turns"])
    assert calls == ["oracle", "init"]  # the turn wiring and the run itself still happened


def test_init_turns_with_judge_model_is_refused(monkeypatch: Any) -> None:
    """Both route through ARE's (idempotent) turn init, so the combination would leave the judge as
    the gate — the opposite of what --init-turns asks for. It must fail, not silently pick one."""
    import pytest
    from examples.gaia2.run_benchmark import main

    _patch_seams(monkeypatch)
    with pytest.raises(SystemExit):
        main(["--scenario", "s.json", "--judge-model", "m", "--init-turns"])


def test_batch_refuses_init_turns_with_judge_model() -> None:
    import pytest
    from examples.gaia2.batch import main

    with pytest.raises(SystemExit):
        main(["--capability", "adaptability", "--judge-model", "m", "--init-turns"])


# -- a run that ends on a question ----------------------------------------------------------------
#
# An activity parked on InputWait — the replan breaker, the sub-goal recursion breaker, or a user
# stop — never reaches TERMINATED, so before this the predicate simply never fired and the run sat
# out its whole wall clock without saying why.


def _asking(prompt: str = "Stuck on 'x': ... How should I proceed?") -> SimpleNamespace:
    return SimpleNamespace(state=ActivityState.BLOCKED, blocked_on=InputWait(prompt=prompt))


def test_stop_when_stops_on_a_question_nobody_is_left_to_answer() -> None:
    sim = _sim(running=False)  # timeline over: no further user turn is coming
    agent = _agent_with([ActivityState.BLOCKED], [InputWait(prompt="How should I proceed?")])
    predicate = _make_stop_when(sim, agent, None, _FAR_DEADLINE)
    assert predicate is not None
    assert predicate() is True


def test_stop_when_lets_a_live_timeline_answer_the_question() -> None:
    """The guard order matters: while turns are still arriving one of them can resume the activity
    (Observe clears an InputWait on a user Message), so cutting the run here throws away a
    recoverable state rather than saving wall clock."""
    sim = _sim(running=True)
    agent = _agent_with([ActivityState.BLOCKED], [InputWait(prompt="How should I proceed?")])
    predicate = _make_stop_when(sim, agent, None, _FAR_DEADLINE)
    assert predicate is not None
    assert predicate() is False


def test_stop_when_still_waits_on_an_activity_blocked_for_a_signal() -> None:
    """Only a *question* ends the run. A SignalWait resolves from tool state, which can still
    settle after the timeline stops, so the narrower InputWait test is the deliberate one."""
    sim = _sim(running=False)
    agent = _agent_with([ActivityState.BLOCKED], [SignalWait(signal_name="job_done")])
    predicate = _make_stop_when(sim, agent, None, _FAR_DEADLINE)
    assert predicate is not None
    assert predicate() is False


def test_stop_when_a_question_does_not_excuse_work_still_in_flight() -> None:
    sim = _sim(running=False)
    agent = _agent_with(
        [ActivityState.READY, ActivityState.BLOCKED],
        [None, InputWait(prompt="How should I proceed?")],
    )
    predicate = _make_stop_when(sim, agent, None, _FAR_DEADLINE)
    assert predicate is not None
    assert predicate() is False  # the other activity can still make progress


def test_awaiting_input_collects_the_prompts_not_just_a_flag() -> None:
    """The prompt names the specific defects that led to the halt — the whole reason for recording
    this rather than a bare "it stopped early"."""
    agent = SimpleNamespace(
        working=SimpleNamespace(
            activities={
                0: SimpleNamespace(state=ActivityState.TERMINATED, blocked_on=None),
                1: _asking("Stuck on 'book a day': no such parameter 'limit'."),
            }
        )
    )
    assert _awaiting_input(agent) == ["Stuck on 'book a day': no such parameter 'limit'."]


def test_jsonl_record_carries_a_pending_question() -> None:
    rec = _jsonl_record(
        scenario_id="s1",
        run_number=0,
        success=False,
        rationale="did not send the email",
        exception=None,
        trace_id="t1",
        awaiting_input=["Stuck on 'book a day': no such parameter 'limit'."],
    )
    assert rec["metadata"]["awaiting_input"] == [
        "Stuck on 'book a day': no such parameter 'limit'."
    ]


def test_jsonl_record_omits_the_key_for_an_ordinary_run() -> None:
    """Every non-halted record stays byte-identical to ARE's own shape — the key is stripped, not
    emitted empty, so nothing downstream sees a new field it did not have before."""
    rec = _jsonl_record(
        scenario_id="s1",
        run_number=0,
        success=True,
        rationale=None,
        exception=None,
        trace_id="t1",
        awaiting_input=[],
    )
    assert "awaiting_input" not in rec["metadata"]


# -- exact-ID sweep manifests --------------------------------------------------------------------


def _manifest_json(cases: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "shared-time",
        "dataset": "meta-agents-research-environments/gaia2",
        "revision": "deadbeef",
        "split": "validation",
        "cases": cases
        if cases is not None
        else [
            {"capability": "time", "id": "scenario-b"},
            {"capability": "time", "id": "scenario-a"},
        ],
    }


def test_sweep_manifest_pins_revision_order_and_canonical_digest(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    raw = _manifest_json()
    first.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    second.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")

    manifest = _load_sweep_manifest(first)

    assert manifest.revision == "deadbeef"
    assert manifest.scenario_ids("time") == ("scenario-b", "scenario-a")
    assert manifest.digest == _load_sweep_manifest(second).digest


def test_paper_manifest_is_the_complete_five_capability_gaia2_mini_selection() -> None:
    manifest = _load_sweep_manifest(
        Path("examples/gaia2/evaluation/campaigns/paper2027/mini-validation.json")
    )

    assert manifest.name == "paper2027-gaia2-mini-validation"
    assert manifest.revision == "78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99"
    assert manifest.digest == "cc6ebb08d388cc0e4dee635c0a031493c280972c033ba0712577413757515e3b"
    assert len(manifest.cases) == 160
    assert {
        capability: len(manifest.scenario_ids(capability))
        for capability in ("execution", "search", "adaptability", "time", "ambiguity")
    } == {
        "execution": 32,
        "search": 32,
        "adaptability": 32,
        "time": 32,
        "ambiguity": 32,
    }


def test_pinned_hf_loader_uses_repository_revision_not_builder_parameter(
    monkeypatch: Any,
) -> None:
    import datasets  # type: ignore[import-untyped]
    from are.simulation.benchmark import scenario_loader

    captured: dict[str, Any] = {}

    def fake_load_dataset(dataset: str, **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
        captured["dataset"] = dataset
        captured.update(kwargs)
        return {"validation": [{"scenario_id": "scenario-a", "data": "{}", "run_number": 7}]}

    scenario = SimpleNamespace(scenario_id="scenario-a", run_number=None)
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        scenario_loader,
        "load_scenario",
        lambda *_args, **_kwargs: (scenario, []),
    )

    rows = list(
        _pinned_hf_scenarios(
            dataset="dataset",
            capability="time",
            split="validation",
            revision="commit-sha",
            limit=None,
        )
    )

    assert captured == {
        "dataset": "dataset",
        "name": "time",
        "revision": "commit-sha",
        "streaming": True,
    }
    assert rows == [(scenario, [])]
    assert scenario.run_number == 7


def test_sweep_manifest_rejects_duplicate_scenario_ids(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        json.dumps(
            _manifest_json(
                [
                    {"capability": "time", "id": "same"},
                    {"capability": "time", "id": "same"},
                ]
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="must be unique"):
        _load_sweep_manifest(path)


def test_manifest_selection_is_exact_and_manifest_ordered() -> None:
    manifest = SweepManifest(
        name="shared-time",
        dataset="dataset",
        revision="revision",
        split="validation",
        cases=(("time", "scenario-b"), ("time", "scenario-a")),
        digest="digest",
    )
    loaded = [
        (SimpleNamespace(scenario_id="scenario-a"), "events-a"),
        (SimpleNamespace(scenario_id="unselected"), "events-extra"),
        (SimpleNamespace(scenario_id="scenario-b"), "events-b"),
    ]

    selected = _select_manifest_scenarios(loaded, manifest, "time")

    assert [scenario.scenario_id for scenario, _events in selected] == [
        "scenario-b",
        "scenario-a",
    ]


def test_manifest_selection_fails_before_running_when_an_id_is_missing() -> None:
    manifest = SweepManifest(
        name="shared-time",
        dataset="dataset",
        revision="revision",
        split="validation",
        cases=(("time", "missing"),),
        digest="digest",
    )

    with pytest.raises(RuntimeError, match="missing"):
        _select_manifest_scenarios([], manifest, "time")


def test_counterpart_arm_must_have_same_complete_manifest_matrix(tmp_path: Path) -> None:
    counterpart = tmp_path / "standard" / "time"
    counterpart.mkdir(parents=True)
    row = _jsonl_record(
        scenario_id="scenario-a",
        run_number=0,
        success=True,
        rationale=None,
        exception=None,
        trace_id="trace",
        scenario_manifest_digest="other",
    )
    (counterpart / "output.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = SweepManifest(
        name="shared-time",
        dataset="dataset",
        revision="revision",
        split="validation",
        cases=(("time", "scenario-a"),),
        digest="expected",
    )
    args = Namespace(output_dir=str(tmp_path), arm="react", capability="time", num_runs=1)

    with pytest.raises(RuntimeError, match="does not carry"):
        _verify_counterpart_pairing(args, manifest)


_PAIRED_PROFILE = "kimi-k2.5-prompt"


def _paired_counterpart(
    tmp_path: Path, *, clock_mode: str, model_profile: str | None = _PAIRED_PROFILE
) -> tuple[Any, SweepManifest]:
    """One already-written ReAct arm, and the `args` an S-ORA arm would run against it.

    A real frozen profile rather than a stub: the clock mode this arm intends is derived from the
    charge sheet the run itself would consult, and a stub would exercise a different derivation
    than the one being guarded."""
    counterpart = tmp_path / "react" / "standard" / "time"
    counterpart.mkdir(parents=True)
    row = _jsonl_record(
        scenario_id="scenario-a",
        run_number=0,
        success=True,
        rationale=None,
        exception=None,
        trace_id="trace",
        scenario_manifest_digest="expected",
        clock_mode=clock_mode,
        model_profile=model_profile,
    )
    (counterpart / "output.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = SweepManifest(
        name="shared-time",
        dataset="dataset",
        revision="revision",
        split="validation",
        cases=(("time", "scenario-a"),),
        digest="expected",
    )
    args = Namespace(
        output_dir=str(tmp_path),
        arm="sora",
        capability="time",
        num_runs=1,
        generation_free=True,
        wall_clock=False,
        charge_model=None,
        model_profile=_profile(_PAIRED_PROFILE),
    )
    return args, manifest


def test_a_pair_split_across_two_clock_conventions_is_refused_before_it_is_paid_for(
    tmp_path: Path,
) -> None:
    """Both arms can be internally homogeneous, pass every per-file guard, and still not be a
    comparison: a generation-free arm beside a token-charged one differs in the timing convention
    as well as the architecture, and the difference that gets reported is the sum of the two."""
    args, manifest = _paired_counterpart(tmp_path, clock_mode="token_charged")

    with pytest.raises(RuntimeError, match="different clock convention"):
        _verify_counterpart_pairing(args, manifest)


def test_a_pair_split_across_two_operating_points_is_refused(tmp_path: Path) -> None:
    """The profile carries the endpoint, and decode rate is an endpoint property — so two arms on
    nominally the same model can still be timed against different hardware."""
    args, manifest = _paired_counterpart(
        tmp_path, clock_mode="generation_free", model_profile="some-other-profile"
    )

    with pytest.raises(RuntimeError, match="different operating point"):
        _verify_counterpart_pairing(args, manifest)


def test_a_counterpart_recording_no_provenance_is_refused_rather_than_assumed_to_match(
    tmp_path: Path,
) -> None:
    """Both arms are written by this harness, so an absent field means the counterpart predates
    the provenance — not that it happens to agree."""
    args, manifest = _paired_counterpart(tmp_path, clock_mode="generation_free", model_profile=None)

    with pytest.raises(RuntimeError, match="different operating point"):
        _verify_counterpart_pairing(args, manifest)


def test_a_matching_counterpart_passes(tmp_path: Path) -> None:
    args, manifest = _paired_counterpart(tmp_path, clock_mode="generation_free")

    _verify_counterpart_pairing(args, manifest)


# -- the two arms share a sweep root without sharing its files -------------------------------------


def test_the_sora_arm_keeps_the_layout_the_uploader_walks() -> None:
    """Not cosmetic: ARE's standalone upload script walks ``{root}/standard/{config}/output.jsonl``
    and ``tests/test_gaia2_upload_compat.py`` locks that round-trip. Pushing the default arm down a
    level to make the two symmetric would break every existing sweep tree and the submission path
    with it."""
    assert _arm_root("/out", "sora") == "/out"


def test_the_react_arm_gets_its_own_subtree() -> None:
    """Both arms write ``output.jsonl`` and ``llm_calls.jsonl`` per capability, and re-running a
    capability truncates them on purpose. Into one root, the second arm would silently destroy the
    first arm's results — including the per-call rows the cost comparison is computed from."""
    assert _arm_root("/out", "react") == "/out/react"


def test_a_react_label_defaults_to_the_profiles_model() -> None:
    profile = SimpleNamespace(name="p", model="gpt-5.4", endpoint="https://e")
    assert _resolve_react_label(None, profile) == "gpt-5.4"


def test_a_react_label_naming_another_model_is_refused() -> None:
    """The profile is the only thing that selects the model on this arm, so a label that disagrees
    is a trace attributed to a model that never ran — and nothing downstream can detect it."""
    profile = SimpleNamespace(name="p", model="gpt-5.4", endpoint="https://e")
    with pytest.raises(SystemExit, match="gpt-5.4"):
        _resolve_react_label("ReAct/kimi-k2.5", profile)


def test_a_react_label_that_names_the_profiles_model_is_kept() -> None:
    profile = SimpleNamespace(name="p", model="gpt-5.4", endpoint="https://e")
    assert _resolve_react_label("ReAct/gpt-5.4", profile) == "ReAct/gpt-5.4"


def test_the_arm_defaults_to_sora() -> None:
    assert _parse_args(["--capability", "execution"]).arm == "sora"


def test_batch_exposes_wall_clock_robustness_mode() -> None:
    args = _parse_args(["--capability", "time", "--wall-clock"])
    assert args.wall_clock is True


def test_scenario_manifest_and_loader_order_limit_are_mutually_exclusive(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(_manifest_json([{"capability": "time", "id": "scenario-a"}])),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="mutually exclusive"):
        main(
            [
                "--capability",
                "time",
                "--scenario-manifest",
                str(manifest),
                "--limit",
                "1",
            ]
        )


def test_scenario_manifest_supplies_pinned_revision_and_digest(
    tmp_path: Path, monkeypatch: Any
) -> None:
    captured: dict[str, Any] = {}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest_json([{"capability": "mini", "id": "scenario-a"}])),
        encoding="utf-8",
    )
    derived = _profile("gpt-5.4-medium-prompt")
    monkeypatch.setattr("examples.gaia2.batch._profile_for_config", lambda *_a: derived)
    monkeypatch.setattr(
        "examples.gaia2.batch._resolve_model_label", lambda _label, _config: "local-model"
    )
    monkeypatch.setattr(
        "examples.gaia2.batch._run_capability", lambda args: captured.update(vars(args)) or []
    )
    monkeypatch.setattr(
        "examples.gaia2.batch.aggregate",
        lambda _path, manifest=None, **_: {"configs": {}, "overall": None},
    )
    monkeypatch.setattr("examples.gaia2.batch._print_report", lambda _summary: None)
    monkeypatch.setattr("examples.gaia2._local_fs.ensure_local_fallback_fs", lambda: None)

    main(["--capability", "mini", "--scenario-manifest", str(manifest_path)])

    assert captured["hf_revision"] == "deadbeef"
    assert captured["sweep_manifest"].digest == _load_sweep_manifest(manifest_path).digest


def test_wall_clock_sora_still_derives_the_frozen_profile(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    derived = _profile("gpt-5.4-medium-prompt")
    monkeypatch.setattr("examples.gaia2.batch._profile_for_config", lambda *_a: derived)
    monkeypatch.setattr(
        "examples.gaia2.batch._resolve_model_label", lambda _label, _config: "local-model"
    )
    monkeypatch.setattr(
        "examples.gaia2.batch._run_capability", lambda args: captured.update(vars(args)) or []
    )
    monkeypatch.setattr(
        "examples.gaia2.batch.aggregate",
        lambda _path, manifest=None, **_: {"configs": {}, "overall": None},
    )
    monkeypatch.setattr("examples.gaia2.batch._print_report", lambda _summary: None)
    monkeypatch.setattr("examples.gaia2._local_fs.ensure_local_fallback_fs", lambda: None)

    main(
        [
            "--capability",
            "mini",
            "--config",
            "examples/gaia2/agent.yaml",
            "--wall-clock",
        ]
    )

    assert captured["model_profile"] is derived
    assert captured["wall_clock"] is True


def test_allow_unfrozen_config_implies_wall_clock_without_profile_derivation(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "examples.gaia2.batch._profile_for_config",
        lambda *_a: pytest.fail("an explicitly unfrozen config has no frozen profile"),
    )
    monkeypatch.setattr(
        "examples.gaia2.batch._resolve_model_label", lambda _label, _config: "local-model"
    )
    monkeypatch.setattr(
        "examples.gaia2.batch._run_capability", lambda args: captured.update(vars(args)) or []
    )
    monkeypatch.setattr(
        "examples.gaia2.batch.aggregate",
        lambda _path, manifest=None, **_: {"configs": {}, "overall": None},
    )
    monkeypatch.setattr("examples.gaia2.batch._print_report", lambda _summary: None)
    monkeypatch.setattr("examples.gaia2._local_fs.ensure_local_fallback_fs", lambda: None)

    main(
        [
            "--capability",
            "mini",
            "--config",
            "examples/gaia2/agent.dev.yaml",
            "--allow-unfrozen-config",
        ]
    )

    assert captured["model_profile"] is None
    assert captured["wall_clock"] is True


def test_charge_profile_overrides_automatic_sora_derivation(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "examples.gaia2.batch._profile_for_config",
        lambda *_a: pytest.fail("explicit override must bypass derivation"),
    )
    monkeypatch.setattr(
        "examples.gaia2.batch._resolve_model_label", lambda _label, _config: "local-model"
    )
    monkeypatch.setattr(
        "examples.gaia2.batch._run_capability", lambda args: captured.update(vars(args)) or []
    )
    monkeypatch.setattr(
        "examples.gaia2.batch.aggregate",
        lambda _path, manifest=None, **_: {"configs": {}, "overall": None},
    )
    monkeypatch.setattr("examples.gaia2.batch._print_report", lambda _summary: None)
    monkeypatch.setattr("examples.gaia2._local_fs.ensure_local_fallback_fs", lambda: None)

    main(
        [
            "--capability",
            "mini",
            "--config",
            "examples/gaia2/agent.yaml",
            "--charge-profile",
            "gpt-5.4-medium-prompt",
        ]
    )

    assert captured["model_profile"].name == "gpt-5.4-medium-prompt"


def test_charge_profile_must_match_the_sora_config() -> None:
    with pytest.raises(SystemExit, match="reasoning_effort"):
        main(
            [
                "--capability",
                "mini",
                "--config",
                "examples/gaia2/agent.yaml",
                "--charge-profile",
                "gpt-5.4-high-paper",
            ]
        )


@pytest.mark.parametrize("other", ["--wall-clock", "--allow-unfrozen-config"])
def test_charge_profile_rejects_conflicting_profile_modes(other: str) -> None:
    with pytest.raises(SystemExit, match="mutually exclusive"):
        main(
            [
                "--capability",
                "mini",
                "--charge-profile",
                "gpt-5.4-medium-prompt",
                other,
            ]
        )


def test_aggregate_suppresses_pass_at_1_for_mixed_clock_modes(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "standard" / "time"
    cfg_dir.mkdir(parents=True)
    rows = [
        {"task_id": "a", "score": 1.0, "metadata": {"clock_mode": "token_charged"}},
        {"task_id": "b", "score": 0.0, "metadata": {"clock_mode": "wall"}},
    ]
    (cfg_dir / "output.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = aggregate(str(tmp_path))["configs"]["time"]

    assert report["mixed_clock_modes"] is True
    assert report["pass_at_1"] is None


def test_aggregate_refuses_a_legacy_row_mixed_with_a_charged_one(tmp_path: Path) -> None:
    """A row predating the clock field must not read as "nothing to compare".

    Filtering the missing mode out of the set made a legacy run invisible to the check, so it sat
    beside a charged run and the pair still looked homogeneous.
    """
    cfg_dir = tmp_path / "standard" / "time"
    cfg_dir.mkdir(parents=True)
    rows = [
        {"task_id": "a", "score": 1.0, "metadata": {"clock_mode": "token_charged"}},
        {"task_id": "b", "score": 0.0, "metadata": {}},  # pre-W2 row: no clock_mode at all
    ]
    (cfg_dir / "output.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = aggregate(str(tmp_path))["configs"]["time"]

    assert report["clock_modes"] == ["legacy", "token_charged"]
    assert report["mixed_clock_modes"] is True
    assert report["pass_at_1"] is None


def test_aggregate_still_scores_a_uniformly_legacy_file(tmp_path: Path) -> None:
    """Naming the legacy mode must not make old, self-consistent reports unreadable."""
    cfg_dir = tmp_path / "standard" / "time"
    cfg_dir.mkdir(parents=True)
    rows = [
        {"task_id": "a", "score": 1.0, "metadata": {}},
        {"task_id": "b", "score": 0.0, "metadata": {}},
    ]
    (cfg_dir / "output.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = aggregate(str(tmp_path))["configs"]["time"]

    assert report["clock_modes"] == ["legacy"]
    assert report["mixed_clock_modes"] is False
    assert report["pass_at_1"] == pytest.approx(0.5)


def test_aggregate_propagates_an_unavailable_charged_union(tmp_path: Path) -> None:
    """One run without a coherent time axis makes the group's counterfactual unavailable."""
    cfg_dir = tmp_path / "standard" / "time"
    cfg_dir.mkdir(parents=True)
    rows = [
        {
            "task_id": "a",
            "score": 1.0,
            "metadata": {"clock_mode": "token_charged", "llm_charged_union_seconds": 4.0},
        },
        {
            "task_id": "b",
            "score": 1.0,
            "metadata": {"clock_mode": "token_charged", "llm_charged_union_seconds": None},
        },
    ]
    (cfg_dir / "output.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = aggregate(str(tmp_path))["configs"]["time"]

    assert report["llm_charged_union_seconds"] is None
    assert report["llm_charged_overlap_seconds"] is None
    # The mixture is in one diagnostic, not in the timing policy, so the score still stands.
    assert report["pass_at_1"] == pytest.approx(1.0)


def test_aggregate_suppresses_pass_at_1_for_mixed_scenario_manifests(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "standard" / "time"
    cfg_dir.mkdir(parents=True)
    rows = [
        {
            "task_id": "a",
            "score": 1.0,
            "metadata": {"scenario_manifest_digest": "manifest-a"},
        },
        {
            "task_id": "b",
            "score": 0.0,
            "metadata": {"scenario_manifest_digest": "manifest-b"},
        },
    ]
    (cfg_dir / "output.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = aggregate(str(tmp_path))["configs"]["time"]

    assert report["mixed_scenario_manifests"] is True
    assert report["pass_at_1"] is None


def test_empty_dataset_refuses_before_truncating_sweep_artifacts(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from are.simulation.benchmark import scenario_loader

    config_dir = tmp_path / "standard" / "mini"
    config_dir.mkdir(parents=True)
    output = config_dir / "output.jsonl"
    calls = config_dir / "llm_calls.jsonl"
    output.write_text("paid output\n")
    calls.write_text("paid calls\n")
    monkeypatch.setattr(scenario_loader, "setup_scenarios_iterator", lambda **_kwargs: iter(()))
    args = Namespace(
        output_dir=str(tmp_path),
        arm="sora",
        capability="mini",
        split="validation",
        hf_dataset="dataset",
        limit=1,
        num_runs=1,
        judge_model=None,
        no_judge_recording=False,
        # The point of this test is that the *loader* refuses before truncating. Without this the
        # attestation guard refuses first, on the deliberately unattested leftovers below, and the
        # assertion would hold for a reason the test is not about.
        overwrite_unattested=True,
    )

    with pytest.raises(RuntimeError, match="zero scenarios"):
        _run_capability(args)

    assert output.read_text() == "paid output\n"
    assert calls.read_text() == "paid calls\n"


def test_manifest_is_fully_resolved_before_truncating_sweep_artifacts(
    tmp_path: Path, monkeypatch: Any
) -> None:
    config_dir = tmp_path / "standard" / "time"
    config_dir.mkdir(parents=True)
    output = config_dir / "output.jsonl"
    calls = config_dir / "llm_calls.jsonl"
    output.write_text("paid output\n", encoding="utf-8")
    calls.write_text("paid calls\n", encoding="utf-8")
    monkeypatch.setattr(
        "examples.gaia2.batch._pinned_hf_scenarios",
        lambda **_kwargs: iter([(SimpleNamespace(scenario_id="present"), object())]),
    )
    manifest = SweepManifest(
        name="shared-time",
        dataset="dataset",
        revision="revision",
        split="validation",
        cases=(("time", "missing"),),
        digest="digest",
    )
    args = Namespace(
        output_dir=str(tmp_path),
        arm="sora",
        capability="time",
        split="validation",
        hf_dataset="dataset",
        hf_revision="revision",
        limit=None,
        num_runs=1,
        judge_model=None,
        no_judge_recording=False,
        sweep_manifest=manifest,
        # As above: the assertion is that the *manifest* refuses before truncating, so the
        # attestation guard must not be the thing that refuses on these unattested leftovers.
        overwrite_unattested=True,
    )

    with pytest.raises(RuntimeError, match="missing"):
        _run_capability(args)

    assert output.read_text(encoding="utf-8") == "paid output\n"
    assert calls.read_text(encoding="utf-8") == "paid calls\n"


def test_the_react_arm_without_a_profile_is_refused() -> None:
    """Nothing else selects the model on that arm, so this cannot be defaulted."""
    with pytest.raises(SystemExit, match="--profile"):
        main(["--capability", "execution", "--arm", "react"])


def test_a_profile_on_the_sora_arm_is_refused_rather_than_ignored() -> None:
    """A two-arm sweep script that passes --profile to both would otherwise run S-ORA at whatever
    agent.yaml says while its command line claims the profile's model."""
    with pytest.raises(SystemExit, match="--arm react"):
        main(["--capability", "execution", "--arm", "sora", "--profile", "gpt-5.4-high-paper"])


def test_report_only_reads_the_arm_it_is_asked_for(tmp_path: Path) -> None:
    """The same --output-dir holds both arms; a report that ignored --arm would print the default
    arm's numbers under the other arm's name."""
    for arm_dir, score in ((tmp_path, 1.0), (tmp_path / "react", 0.0)):
        d = arm_dir / "standard" / "execution"
        d.mkdir(parents=True)
        (d / "output.jsonl").write_text(
            json.dumps({"task_id": "s1", "trace_id": "t", "score": score, "metadata": {}}) + "\n"
        )
    assert aggregate(_arm_root(str(tmp_path), "sora"))["exploratory_mean"] == 1.0
    assert aggregate(_arm_root(str(tmp_path), "react"))["exploratory_mean"] == 0.0


# -- the two arms have to be the same experiment ------------------------------------------------


def _profile(name: str) -> Any:
    from examples.gaia2.evaluation.core import load_profiles
    from examples.gaia2.latency_grid import EVAL_ROOT

    return load_profiles(EVAL_ROOT / "profiles.json")[name]


def test_the_shipped_config_and_its_profile_describe_one_operating_point() -> None:
    """The pair the docs tell people to run. This is the test that reddens if either side moves."""
    _check_operating_point(_profile("gpt-5.4-medium-prompt"), "examples/gaia2/agent.yaml")


def test_a_profile_at_another_reasoning_effort_is_refused() -> None:
    """Same model, same cap, different thinking budget — two pass@1 columns that look comparable,
    carry the same model label, and are not. Nothing downstream can tell them apart afterwards,
    which is why this is refused up front rather than reported on the row."""
    with pytest.raises(SystemExit, match="reasoning_effort"):
        _check_operating_point(_profile("gpt-5.4-high-paper"), "examples/gaia2/agent.yaml")


def test_a_profile_at_another_model_is_refused_by_the_same_check() -> None:
    with pytest.raises(SystemExit, match="model"):
        _check_operating_point(_profile("kimi-k2.5-prompt"), "examples/gaia2/agent.yaml")


def test_a_profile_rerouted_to_another_endpoint_is_refused(tmp_path: Path) -> None:
    """Same model, same effort, same cap — served by somewhere else. A model name is not a serving
    path: a different ``base_url``, or the same one behind different ``provider_routing``, is a
    different experiment wearing the shipped experiment's label. The check compares the union of
    what the two sides declare rather than a list of interesting keys, so this is caught without
    anyone having remembered to add routing to it."""
    from dataclasses import replace

    from examples.gaia2.evaluation.core import SettingValue

    base = _profile("gpt-5.4-medium-prompt")
    settings = dict(base.settings)
    settings["provider_routing"] = SettingValue(status="sent", value={"only": ["elsewhere"]})
    rerouted = replace(
        base, name="rerouted", endpoint="https://example.invalid/v1", settings=settings
    )
    with pytest.raises(SystemExit) as refusal:
        _check_operating_point(rerouted, "examples/gaia2/agent.yaml")
    assert "base_url" in str(refusal.value)
    assert "provider_routing" in str(refusal.value)


def test_transport_and_bookkeeping_do_not_make_two_arms_incomparable() -> None:
    """The other half of a union-based comparison: it must not refuse over keys that decide how a
    call is carried or whether it is recorded. Left out, a widened check would redden the shipped
    pair — ``max_logical_calls`` is in agent.yaml and in no profile at all."""
    base = _profile("gpt-5.4-medium-prompt")
    assert {"stall_timeout", "max_retries", "instrument"} <= set(base.client_settings())
    _check_operating_point(base, "examples/gaia2/agent.yaml")


def test_a_config_that_is_not_there_is_nothing_to_disagree_with(tmp_path: Path) -> None:
    """A react-only run has no S-ORA side to be comparable with; refusing it would be refusing a
    run that the check has nothing to say about."""
    _check_operating_point(_profile("kimi-k2.5-prompt"), str(tmp_path / "absent.yaml"))


def test_sora_profile_derivation_names_a_missing_config_directly(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="does not exist"):
        _profile_for_config(
            {"kimi-k2.5-prompt": _profile("kimi-k2.5-prompt")},
            str(tmp_path / "absent.yaml"),
        )


def test_the_react_arm_checks_the_operating_point_before_spending() -> None:
    with pytest.raises(SystemExit, match="operating point"):
        main(
            [
                "--capability",
                "execution",
                "--arm",
                "react",
                "--profile",
                "gpt-5.4-high-paper",
            ]
        )


# -- run provenance echo --------------------------------------------------------------------------
#
# What produced a run but is not observable in its trace. A strategy setting changes which cycles
# spend a model call and never says so in any record, and a config is routinely edited locally and
# never committed — so once the process exits, its effective settings can exist nowhere at all.


def test_config_echo_records_the_config_verbatim_with_its_digest(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    config.write_text(
        "agent:\n"
        "  strategies:\n"
        "    context_adaptation: none  # disabled to fit the judge's tolerance\n",
        encoding="utf-8",
    )

    echo = _runner._config_echo(
        str(config),
        charge=None,
        charge_model_identity=None,
        charge_model_digest=None,
        max_wall_seconds=1200.0,
        scenario_id="scenario_universe_27",
    )

    # Verbatim, not a parsed summary: the comment is usually where the reason for a value lives.
    assert "context_adaptation: none  # disabled to fit the judge's tolerance" in echo
    assert "scenario_universe_27" in echo
    assert str(config.resolve()) in echo
    # Two runs that disagree are distinguishable without re-reading a file that has since moved on.
    assert hashlib.sha256(config.read_bytes()).hexdigest() in echo
    assert "wall (robustness mode" in echo


def test_config_echo_names_the_charged_clock_and_its_charge_model(tmp_path: Path) -> None:
    # A charged run's timings are meaningless against a wall-clock one, and the trace does not say
    # which it was — so the echo has to, and against which frozen coefficients.
    config = tmp_path / "agent.yaml"
    config.write_text("agent:\n  name: x\n", encoding="utf-8")

    echo = _runner._config_echo(
        str(config),
        charge=cast(Any, object()),
        charge_model_identity={"model": "gpt-5.4-2026-03-05"},
        charge_model_digest="c91e3535",
        max_wall_seconds=3600.0,
        scenario_id=None,
    )

    assert "charged (simulated time frozen" in echo
    assert "c91e3535" in echo
    assert "model=gpt-5.4-2026-03-05" in echo


def test_config_echo_redacts_an_inlined_credential(tmp_path: Path) -> None:
    # The echo is meant to be pasted into an issue and diffed across runs. A config that inlines a
    # key rather than naming an env var must not make the log the thing that leaks it.
    config = tmp_path / "agent.yaml"
    config.write_text(
        "agent:\n  llm:\n    api_key: sk-live-SECRET\n    api_key_env: OPENAI_API_KEY\n",
        encoding="utf-8",
    )

    echo = _runner._config_echo(
        str(config),
        charge=None,
        charge_model_identity=None,
        charge_model_digest=None,
        max_wall_seconds=1200.0,
        scenario_id=None,
    )

    assert "sk-live-SECRET" not in echo
    assert "(redacted)" in echo
    assert "api_key_env: OPENAI_API_KEY" in echo  # the env *name* is not a secret, and is wanted


# -- watchdog truncation -------------------------------------------------------------------------


def test_a_watchdog_truncated_run_is_excluded_from_pass_at_1() -> None:
    """The harness stopped this trajectory mid-flight, so its score measures how far the agent got
    before the cap rather than how well it did.

    Excluded for the same reason an expired timeline is. Kept as a separate field because both end
    the run through ``Environment.stop()`` and are otherwise indistinguishable downstream — and
    they have different causes and different fixes.

    *Every* watchdog is excluded, not only the wall-clock cap: a stalled judge truncates the
    trajectory exactly as much, and the two were briefly handled differently on the two arms — one
    scored a stalled-judge run, the other excluded it under a label that said "wall clock"."""
    records = [
        {"score": 1.0, "metadata": {}},
        {"score": 0.0, "metadata": {"harness_truncation": "wall_clock"}},
        {"score": 0.0, "metadata": {"harness_truncation": "judge_stall"}},
        {"score": 0.0, "metadata": {"timeline_expired": True}},
    ]
    pass_at_1, scored, total = batch._pass_at_1(records)
    assert (pass_at_1, scored, total) == (1.0, 1, 4)


def test_the_record_carries_why_the_run_stopped_and_the_cap_it_ran_under() -> None:
    """``terminal_cause`` was computed on both arms and then dropped on the way into the artifact,
    which left a truncated run looking exactly like one that finished. The cap belongs beside it:
    under a frozen clock it is the only bound on a trajectory's length, so a row that does not
    carry it cannot say whether its own truncation was plausible."""
    record = batch._jsonl_record(
        scenario_id="s",
        run_number=0,
        success=None,
        rationale=None,
        exception=None,
        trace_id=None,
        harness_truncation="judge_stall",
        terminal_cause="timeout",
        max_wall_seconds=3600.0,
    )
    # `terminal_cause` cannot carry this by itself: it reports both watchdogs *and* an expired
    # timeline as "timeout", which is why the run needs a field that names which one fired.
    assert record["metadata"]["harness_truncation"] == "judge_stall"
    assert record["metadata"]["terminal_cause"] == "timeout"
    assert record["metadata"]["max_wall_seconds"] == 3600.0


def test_an_untruncated_record_keeps_ares_own_shape() -> None:
    """The field names a watchdog only when one fired, so an ordinary row carries no truncation."""
    record = batch._jsonl_record(
        scenario_id="s",
        run_number=0,
        success=True,
        rationale=None,
        exception=None,
        trace_id=None,
        terminal_cause="verification_completion",
    )
    assert "harness_truncation" not in record["metadata"]


# -- the generation-free watchdog default --------------------------------------------------------


def test_generation_free_raises_the_watchdog_default_and_announces_it(
    capsys: Any, monkeypatch: Any
) -> None:
    """Freezing the clock removes generation from the scenario's budget but not from the
    operator's: real per-scenario time becomes the timeline *plus* the whole of generation, where a
    wall-clock run is bounded by the timeline alone. Measured S-ORA scenarios project past 1200s
    under that arithmetic while every ReAct one stays under it, so leaving the cap alone would
    truncate one arm and not the other. Raised explicitly and announced rather than changed
    globally in silence."""
    monkeypatch.setattr(batch, "aggregate", lambda _root, manifest=None, **_: {})
    monkeypatch.setattr(batch, "_print_report", lambda _report: None)

    batch.main(["--report-only", "out", "--generation-free"])
    assert "using 3600s" in capsys.readouterr().out

    batch.main(["--report-only", "out"])
    assert "using 1200s" in capsys.readouterr().out


def test_the_two_unfrozen_clocks_cannot_be_selected_together() -> None:
    """They are different conventions, not a flag and its intensifier."""
    with pytest.raises(SystemExit, match="pick one"):
        batch.main(["--report-only", "out", "--wall-clock", "--generation-free"])


# -- clock modes may not be spliced ----------------------------------------------------------------


def _capability(
    tmp_path: Any,
    name: str,
    clock_mode: str,
    score: float,
    *,
    scenario_id: str | None = None,
    manifest_digest: str | None = None,
    prompt_snapshot_digest: str | None = None,
) -> None:
    import os

    d = os.path.join(str(tmp_path), "standard", name)
    os.makedirs(d, exist_ok=True)
    metadata: dict[str, Any] = {"clock_mode": clock_mode}
    if scenario_id is not None:
        metadata["scenario_id"] = scenario_id
    if manifest_digest is not None:
        metadata["scenario_manifest_digest"] = manifest_digest
    if prompt_snapshot_digest is not None:
        metadata["prompt_snapshot_digest"] = prompt_snapshot_digest
    with open(os.path.join(d, "output.jsonl"), "w") as fh:
        fh.write(json.dumps({"score": score, "metadata": metadata}) + "\n")


_SWEEP_DIGEST = "deadbeefcafe0000"


def _complete_sweep(tmp_path: Any, clock_mode: str = "generation_free") -> SweepManifest:
    """Five core capabilities, one scenario each, one clock, one manifest — the only shape that
    earns a headline."""
    for name in batch._CORE_CAPABILITIES:
        _capability(
            tmp_path,
            name,
            clock_mode,
            1.0,
            scenario_id=f"scenario-{name}",
            manifest_digest=_SWEEP_DIGEST,
        )
        # A pinned sweep is attested as it is written, so the fixture for the one shape that earns
        # a headline has to be attested too — otherwise these tests would assert the headline
        # against a directory the real gate refuses.
        batch.write_artifact_attestation(
            str(Path(tmp_path) / "standard" / name),
            arm="sora",
            capability=name,
            records=[],
        )
    return SweepManifest(
        name="test-mini",
        dataset="d",
        revision="r",
        split="validation",
        cases=tuple((name, f"scenario-{name}") for name in batch._CORE_CAPABILITIES),
        digest=_SWEEP_DIGEST,
    )


def test_a_complete_homogeneous_manifest_backed_sweep_earns_the_headline(tmp_path: Any) -> None:
    manifest = _complete_sweep(tmp_path)
    summary = batch.aggregate(str(tmp_path), manifest=manifest)
    assert summary["headline_withheld"] == ()
    assert summary["overall"] == 1.0


def test_the_headline_is_withheld_when_a_capability_did_not_run(tmp_path: Any) -> None:
    """The mean used to be taken over whatever was on disk, so a capability that failed to run
    simply left the average — silently moving the headline with nothing in the number to say so."""
    manifest = _complete_sweep(tmp_path)
    (tmp_path / "standard" / "time" / "output.jsonl").unlink()
    summary = batch.aggregate(str(tmp_path), manifest=manifest)
    assert summary["overall"] is None
    assert "capabilities not run: time" in summary["headline_withheld"]
    # Still readable as an in-progress sweep — under a name that cannot be mistaken for a result.
    assert summary["exploratory_mean"] == 1.0


def test_a_suppressed_capability_withholds_the_headline_rather_than_leaving_the_mean(
    tmp_path: Any,
) -> None:
    """The sharpest version of the same hole: a capability whose own rows mix clock modes has its
    pass@1 suppressed, and that used to *remove* it from the average — so detecting the mixture
    made the headline more available instead of less."""
    manifest = _complete_sweep(tmp_path)
    path = tmp_path / "standard" / "time" / "output.jsonl"
    with path.open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "score": 0.0,
                    "metadata": {
                        "clock_mode": "generation_free",
                        "scenario_id": "scenario-time",
                        "scenario_manifest_digest": "a-different-manifest",
                    },
                }
            )
            + "\n"
        )
    summary = batch.aggregate(str(tmp_path), manifest=manifest)
    assert summary["configs"]["time"]["pass_at_1"] is None
    assert summary["overall"] is None
    assert "capabilities with no comparable pass@1: time" in summary["headline_withheld"]


def test_the_headline_is_withheld_when_capabilities_ran_different_scenario_selections(
    tmp_path: Any,
) -> None:
    manifest = _complete_sweep(tmp_path)
    _capability(
        tmp_path,
        "time",
        "generation_free",
        1.0,
        scenario_id="scenario-time",
        manifest_digest="another-manifest",
    )
    summary = batch.aggregate(str(tmp_path), manifest=manifest)
    assert summary["overall"] is None
    assert any("different scenario manifests" in reason for reason in summary["headline_withheld"])


def test_the_headline_is_withheld_when_capabilities_ran_different_prompt_snapshots(
    tmp_path: Any,
) -> None:
    """A prompt rewrite between two capabilities of one sweep changes nothing else: same model,
    same manifest, same clock. Without this gate the mean spans two agents and says so nowhere."""
    manifest = _complete_sweep(tmp_path)
    _capability(
        tmp_path,
        "time",
        "generation_free",
        1.0,
        scenario_id="scenario-time",
        manifest_digest=_SWEEP_DIGEST,
        prompt_snapshot_digest="a" * 64,
    )
    _capability(
        tmp_path,
        "search",
        "generation_free",
        1.0,
        scenario_id="scenario-search",
        manifest_digest=_SWEEP_DIGEST,
        prompt_snapshot_digest="b" * 64,
    )
    summary = batch.aggregate(str(tmp_path), manifest=manifest)
    assert summary["overall"] is None
    assert any("different prompt snapshots" in reason for reason in summary["headline_withheld"]), (
        summary["headline_withheld"]
    )


def test_the_headline_survives_an_arm_that_records_no_prompt_snapshot(tmp_path: Any) -> None:
    """ARE's own ReAct agent does not read this runtime's prompts, so its rows carry no digest.
    Requiring one would withhold every baseline-arm headline for a dependency the arm lacks."""
    manifest = _complete_sweep(tmp_path)
    summary = batch.aggregate(str(tmp_path), manifest=manifest)
    assert summary["headline_withheld"] == ()
    assert summary["overall"] == pytest.approx(1.0)


def test_the_headline_is_withheld_when_the_arm_that_reads_prompts_records_none(
    tmp_path: Any,
) -> None:
    """A sweep that recorded no provenance anywhere is perfectly homogeneous, so the mixture check
    passes it. It is also the weakest evidence of the two, and missing provenance is not evidence
    of agreement — the same rule the manifest gate already applies."""
    manifest = _complete_sweep(tmp_path)
    summary = batch.aggregate(str(tmp_path), manifest=manifest, expects_prompt_provenance=True)
    assert summary["overall"] is None
    assert any(
        "recorded no prompt snapshot" in reason for reason in summary["headline_withheld"]
    ), summary["headline_withheld"]


def test_the_headline_is_withheld_when_only_some_capabilities_record_prompts(
    tmp_path: Any,
) -> None:
    """The partial case: one capability names its prompts and another does not. Filtering the
    absences away left a single digest behind, which read as agreement."""
    manifest = _complete_sweep(tmp_path)
    _capability(
        tmp_path,
        "time",
        "generation_free",
        1.0,
        scenario_id="scenario-time",
        manifest_digest=_SWEEP_DIGEST,
        prompt_snapshot_digest="a" * 64,
    )
    summary = batch.aggregate(str(tmp_path), manifest=manifest, expects_prompt_provenance=True)
    assert summary["overall"] is None
    assert any(
        "recorded no prompt snapshot" in reason for reason in summary["headline_withheld"]
    ), summary["headline_withheld"]


def test_the_headline_is_withheld_when_a_pinned_scenario_never_scored(tmp_path: Any) -> None:
    """Coverage is the one gate the rows cannot answer on their own: they record what ran, never
    what was supposed to. A scenario that errored out of every run leaves a shorter denominator
    and no other trace."""
    manifest = _complete_sweep(tmp_path)
    manifest = SweepManifest(
        name=manifest.name,
        dataset=manifest.dataset,
        revision=manifest.revision,
        split=manifest.split,
        cases=manifest.cases + (("time", "scenario-time-second"),),
        digest=manifest.digest,
    )
    summary = batch.aggregate(str(tmp_path), manifest=manifest)
    assert summary["overall"] is None
    assert (
        "1 manifest scenarios not scored: time/scenario-time-second"
        in (summary["headline_withheld"])
    )


def test_a_truncated_run_leaves_its_scenario_uncovered_rather_than_scored(tmp_path: Any) -> None:
    """Coverage means the same thing by "scored" as pass@1 does. A watchdog-truncated run is
    excluded from the score, so it cannot also count as having covered its scenario — otherwise a
    capability whose every run was cut off would read as complete."""
    manifest = _complete_sweep(tmp_path)
    _capability(
        tmp_path,
        "time",
        "generation_free",
        1.0,
        scenario_id="scenario-time",
        manifest_digest=_SWEEP_DIGEST,
    )
    path = tmp_path / "standard" / "time" / "output.jsonl"
    path.write_text(
        json.dumps(
            {
                "score": 1.0,
                "metadata": {
                    "clock_mode": "generation_free",
                    "scenario_id": "scenario-time",
                    "scenario_manifest_digest": _SWEEP_DIGEST,
                    "harness_truncation": "judge_stall",
                },
            }
        )
        + "\n"
    )
    summary = batch.aggregate(str(tmp_path), manifest=manifest)
    assert summary["overall"] is None
    assert any("not scored: time/scenario-time" in r for r in summary["headline_withheld"])


def test_a_sweep_recording_no_manifest_digest_cannot_earn_the_headline(tmp_path: Any) -> None:
    """The check used to be conditional on a digest being recorded at all, so the one shape with no
    provenance whatsoever — five fully scored capabilities, one clock, every pinned scenario
    covered, and nothing tying any of it to the manifest being reported against — passed every
    other condition and took the headline."""
    manifest = _complete_sweep(tmp_path)
    for name in batch._CORE_CAPABILITIES:
        _capability(tmp_path, name, "generation_free", 1.0, scenario_id=f"scenario-{name}")

    summary = batch.aggregate(str(tmp_path), manifest=manifest)

    assert summary["overall"] is None
    assert any("no digest recorded" in reason for reason in summary["headline_withheld"]), summary[
        "headline_withheld"
    ]


def test_the_headline_is_withheld_without_a_manifest_to_check_coverage_against(
    tmp_path: Any,
) -> None:
    """A complete-looking sweep is still only complete relative to a pinned selection. Without one
    the denominator is unknown, and a headline over an unknown denominator is the failure the whole
    gate exists for."""
    _complete_sweep(tmp_path)
    summary = batch.aggregate(str(tmp_path))
    assert summary["overall"] is None
    assert summary["headline_withheld"] == (
        "scenario coverage unverified: no --scenario-manifest given",
    )
    assert summary["exploratory_mean"] == 1.0


def test_the_headline_refuses_to_average_capabilities_that_ran_on_different_clocks(
    tmp_path: Any,
) -> None:
    """Each capability file was already checked for mixing internally — but the headline is a mean
    *across* files, and nothing stopped four generation-free capabilities and one token-charged
    Time from being promoted into a single five-capability number. That is the exact splice the
    per-file check exists to refuse, one level up, and it is the likeliest way it would happen: a
    charged Time re-run is the one sensitivity worth doing separately."""
    _capability(tmp_path, "execution", "generation_free", 1.0)
    _capability(tmp_path, "time", "token_charged", 0.0)
    summary = batch.aggregate(str(tmp_path))
    assert summary["overall"] is None
    assert (
        "capabilities ran under different clock modes: generation_free, token_charged"
        in summary["headline_withheld"]
    )
    assert summary["overall_clock_modes"] == ["generation_free", "token_charged"]
    # Per-capability pass@1 survives: each file is internally homogeneous and still readable.
    assert summary["configs"]["execution"]["pass_at_1"] == 1.0


def test_one_clock_across_capabilities_is_not_itself_a_reason_to_withhold(tmp_path: Any) -> None:
    manifest = _complete_sweep(tmp_path, clock_mode="generation_free")
    summary = batch.aggregate(str(tmp_path), manifest=manifest)
    assert summary["overall_clock_modes"] == ["generation_free"]
    assert not any("clock modes" in reason for reason in summary["headline_withheld"])


def test_a_generation_free_run_reports_no_charge_accounting_mismatch(tmp_path: Any) -> None:
    """The fallback comparison exists to police *this harness's* clamp bookkeeping against the
    provider's raw usage figures. A generation-free run applies no clamp and computes no charge, so
    the comparison has nothing to police — and a raw usage anomaly would otherwise be reported as
    an accounting mismatch, which names the wrong component."""
    import os

    d = os.path.join(str(tmp_path), "standard", "execution")
    os.makedirs(d)
    with open(os.path.join(d, "output.jsonl"), "w") as fh:
        fh.write(
            json.dumps(
                {
                    "score": 1.0,
                    "metadata": {
                        "clock_mode": "generation_free",
                        "cached_input_clamps": 0,
                        "raw_cached_input_anomalies": 3,
                    },
                }
            )
            + "\n"
        )
    summary = batch.aggregate(str(tmp_path))
    assert summary["configs"]["execution"]["charge_accounting_mismatches"] == 0


def test_unfrozen_development_mode_cannot_also_ask_for_a_frozen_clock() -> None:
    """--allow-unfrozen-config *selects* the wall clock, and it does so after the flag-conflict
    check has already run. Left unchecked the pair is accepted, the run announces wall-clock
    development mode, and then freezes anyway — an artifact that contradicts its own banner."""
    with pytest.raises(SystemExit, match="cannot be combined"):
        batch.main(
            [
                "--arm",
                "sora",
                "--config",
                "examples/gaia2/agent.yaml",
                "--capability",
                "execution",
                "--allow-unfrozen-config",
                "--generation-free",
            ]
        )


# --- paid-artifact preservation ---------------------------------------------


def _attested_capability(root: Path, capability: str, *, digest: str = "manifest-digest") -> Path:
    """One scored row plus its attestation, laid out as a finished run would leave them."""
    cfg_dir = root / "standard" / capability
    cfg_dir.mkdir(parents=True)
    record = {
        "task_id": f"{capability}-0",
        "score": 1.0,
        "metadata": {"scenario_manifest_digest": digest, "clock_mode": "generation_free"},
    }
    (cfg_dir / "output.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (cfg_dir / "llm_calls.jsonl").write_text('{"arm": "sora"}\n', encoding="utf-8")
    write_artifact_attestation(str(cfg_dir), arm="sora", capability=capability, records=[record])
    return cfg_dir


def test_attestation_covers_every_artifact_and_verifies_clean(tmp_path: Path) -> None:
    cfg_dir = _attested_capability(tmp_path, "time")

    attested = json.loads((cfg_dir / ARTIFACT_ATTESTATION).read_text(encoding="utf-8"))
    assert [entry["path"] for entry in attested["files"]] == [
        "llm_calls.jsonl",
        "output.jsonl",
    ]
    # The checksum file is coreutils format so a preserved run can be verified without this repo.
    lines = (cfg_dir / ARTIFACT_CHECKSUMS).read_text(encoding="utf-8").splitlines()
    assert lines == [f"{entry['sha256']}  ./{entry['path']}" for entry in attested["files"]]
    # The rows' own provenance travels with the checksums: this is what ties a compact report to
    # bytes rather than to a path.
    assert attested["records"] == {
        "record_count": 1,
        "scenario_manifest_digests": ["manifest-digest"],
        "prompt_snapshot_digests": [None],
        "charge_model_digests": [None],
        "clock_modes": ["generation_free"],
    }

    assert verify_artifact_attestation(str(cfg_dir))[1] == ()


def test_attestation_detects_a_changed_added_or_removed_artifact(tmp_path: Path) -> None:
    cfg_dir = _attested_capability(tmp_path, "time")

    (cfg_dir / "output.jsonl").write_text('{"task_id": "time-0", "score": 0.0}\n', encoding="utf-8")
    assert "output.jsonl changed since it was attested" in " ".join(
        verify_artifact_attestation(str(cfg_dir))[1]
    )

    _attested_capability(tmp_path, "search")
    (tmp_path / "standard" / "search" / "extra.json").write_text("{}", encoding="utf-8")
    assert "unattested file present: extra.json" in " ".join(
        verify_artifact_attestation(str(tmp_path / "standard" / "search"))[1]
    )

    _attested_capability(tmp_path, "execution")
    (tmp_path / "standard" / "execution" / "llm_calls.jsonl").unlink()
    assert "attested file missing: llm_calls.jsonl" in " ".join(
        verify_artifact_attestation(str(tmp_path / "standard" / "execution"))[1]
    )


def test_attestation_digest_covers_the_file_list_it_was_written_with(tmp_path: Path) -> None:
    # A rewritten `files` list that agrees with the directory is still caught: the one-value digest
    # was taken over the list the run wrote, so editing both halves consistently does not restore
    # agreement with it.
    cfg_dir = _attested_capability(tmp_path, "time")
    path = cfg_dir / ARTIFACT_ATTESTATION
    attested = json.loads(path.read_text(encoding="utf-8"))
    (cfg_dir / "llm_calls.jsonl").unlink()
    attested["files"] = [e for e in attested["files"] if e["path"] != "llm_calls.jsonl"]
    path.write_text(json.dumps(attested), encoding="utf-8")

    assert "artifact set digest does not cover its own file list" in " ".join(
        verify_artifact_attestation(str(cfg_dir))[1]
    )


def test_a_contamination_marker_above_the_directory_is_never_overridable(tmp_path: Path) -> None:
    (tmp_path / CONTAMINATION_MARKER).write_text(
        "react sweep shared an output root\nsecond line\n", encoding="utf-8"
    )
    cfg_dir = tmp_path / "standard" / "time"
    cfg_dir.mkdir(parents=True)

    for allow in (False, True):
        with pytest.raises(SystemExit, match="react sweep shared an output root"):
            refuse_contaminated_root(str(cfg_dir), allow_unattested=allow)


def test_unattested_leftovers_are_refused_but_overridable(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "standard" / "time"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "output.jsonl").write_text('{"score": 1.0}\n', encoding="utf-8")

    with pytest.raises(SystemExit, match="unknown provenance"):
        refuse_contaminated_root(str(cfg_dir))

    refuse_contaminated_root(str(cfg_dir), allow_unattested=True)


def test_an_empty_or_attested_directory_is_accepted(tmp_path: Path) -> None:
    # Nothing to destroy, and a directory whose bytes are exactly what the last run closed: both
    # are ordinary re-runs, and refusing either would make the guard unusable.
    refuse_contaminated_root(str(tmp_path / "standard" / "time"))
    refuse_contaminated_root(str(_attested_capability(tmp_path, "time")))


def test_report_carries_the_artifact_set_and_withholds_a_pinned_headline_when_it_moved(
    tmp_path: Path,
) -> None:
    manifest = SweepManifest(
        name="pinned",
        dataset="dataset",
        revision="revision",
        split="validation",
        cases=tuple(("time", "time-0") for _ in (0,)),
        digest="manifest-digest",
    )
    cfg_dir = _attested_capability(tmp_path, "time")
    attested = json.loads((cfg_dir / ARTIFACT_ATTESTATION).read_text(encoding="utf-8"))

    summary = aggregate(str(tmp_path), manifest)
    assert summary["configs"]["time"]["artifact_set_sha256"] == attested["artifact_set_sha256"]
    assert summary["configs"]["time"]["artifact_attestation_reasons"] == []
    assert not any("not attested" in reason for reason in summary["headline_withheld"])

    (cfg_dir / "output.jsonl").write_text(
        json.dumps({"task_id": "time-0", "score": 0.0, "metadata": {}}) + "\n", encoding="utf-8"
    )
    moved = aggregate(str(tmp_path), manifest)
    assert moved["configs"]["time"]["artifact_set_sha256"] is None
    assert any("not attested" in reason for reason in moved["headline_withheld"])


def test_an_unpinned_sweep_is_not_required_to_be_attested(tmp_path: Path) -> None:
    # Smoke and report-only runs are the one use where unattested bytes are the point; requiring
    # attestation everywhere would refuse them. Their headline is already withheld for coverage.
    _write_config(tmp_path, "time", [1.0])

    summary = aggregate(str(tmp_path))

    assert summary["configs"]["time"]["artifact_set_sha256"] is None
    assert not any("not attested" in reason for reason in summary["headline_withheld"])
