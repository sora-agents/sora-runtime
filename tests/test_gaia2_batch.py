"""``examples/gaia2/batch.py`` + ``_runner.py`` — pure formatting/aggregation and the turn-aware
stop predicate, tested directly (no ARE, no model tokens). The parts that touch ARE or spend tokens
(scenario iteration, judge, trace export) are exercised by the operator-run correctness gate, not
here.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from examples.gaia2 import _runner
from examples.gaia2._runner import _awaiting_input, _make_stop_when
from examples.gaia2.batch import (
    _jsonl_record,
    _pass_at_1,
    _resolve_model_label,
    _score_status,
    aggregate,
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
    ]
    pass_at_1, scored, total = _pass_at_1(records)
    assert pass_at_1 == 2 / 3
    assert (scored, total) == (3, 4)


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
    assert summary["overall"] == 0.75


def test_aggregate_empty_dir_is_safe(tmp_path: Path) -> None:
    summary = aggregate(str(tmp_path))
    assert summary == {"configs": {}, "overall": None}


# -- _make_stop_when: the turn-aware predicate ----------------------------------------------------


def _agent_with(
    states: list[ActivityState], blocked_on: list[Any] | None = None
) -> SimpleNamespace:
    waits: list[Any] = blocked_on if blocked_on is not None else [None] * len(states)
    activities = {
        i: SimpleNamespace(state=st, blocked_on=w)
        for i, (st, w) in enumerate(zip(states, waits, strict=True))
    }
    return SimpleNamespace(working=SimpleNamespace(activities=activities))


def _sim(*, running: bool, paused: bool = False) -> SimpleNamespace:
    """The predicate reads both flags, so a fake has to answer both. They are not exclusive: a
    paused environment is one ARE holds mid-turn around a judge call, and ``is_running()`` counts
    that as live — the pair a real ``AreSimulation`` reports while a judge is in flight is
    ``running=True, paused=True``."""
    return SimpleNamespace(is_running=lambda: running, is_paused=lambda: paused)


def test_stop_when_none_when_exit_when_idle_set() -> None:
    # Opting into the quiet-window heuristic means no custom predicate.
    assert _make_stop_when(SimpleNamespace(), SimpleNamespace(), 8.0, 1200.0) is None


def test_stop_when_rides_through_live_timeline() -> None:
    sim = _sim(running=True)
    agent = _agent_with([ActivityState.TERMINATED])  # even fully idle, a live timeline keeps going
    predicate = _make_stop_when(sim, agent, None, 1200.0)
    assert predicate is not None
    assert predicate() is False


def test_stop_when_stops_once_timeline_done_and_all_terminated() -> None:
    sim = _sim(running=False)
    predicate = _make_stop_when(sim, _agent_with([ActivityState.TERMINATED]), None, 1200.0)
    assert predicate is not None
    assert predicate() is True


def test_stop_when_waits_for_in_flight_activity_after_timeline_done() -> None:
    sim = _sim(running=False)
    agent = _agent_with([ActivityState.TERMINATED, ActivityState.READY])
    predicate = _make_stop_when(sim, agent, None, 1200.0)
    assert predicate is not None
    assert predicate() is False  # timeline ended but the agent still has work in flight


def test_stop_when_wall_clock_cap_fires() -> None:
    sim = _sim(running=True)  # timeline still live, but the cap wins
    predicate = _make_stop_when(sim, _agent_with([ActivityState.READY]), None, -1.0)
    assert predicate is not None
    assert predicate() is True


# -- the pause cap: bound a stalled judge without bounding a slow scenario ------------------------


def test_stop_when_rides_through_a_judge_pause() -> None:
    """The run-6 shape, pinned: ARE pauses the timeline around each per-turn judge call, so the
    environment is mid-turn while every activity sits on the conditions its exhausted plan body
    left behind — which past the end of a timeline is exactly what "finished" looks like. The run
    has to sit through the pause instead, or it is torn down inside the judge's own window and
    every later turn is lost."""
    sim = _sim(running=True, paused=True)
    agent = _agent_with([ActivityState.BLOCKED], [ConditionWait()])
    predicate = _make_stop_when(sim, agent, None, 1200.0)
    assert predicate is not None
    assert predicate() is False


def test_stop_when_a_pause_that_never_ends_is_a_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    """A judge that never answers leaves the environment paused for good — ARE's pause/resume
    bracket is not exception-safe. Without a cap the run sits out its whole wall clock, which on a
    sweep is the difference between three minutes and twenty per hung scenario."""
    clock = 1000.0
    monkeypatch.setattr("examples.gaia2._runner.time.monotonic", lambda: clock)
    sim = _sim(running=True, paused=True)
    predicate = _make_stop_when(sim, _agent_with([ActivityState.READY]), None, 1200.0)
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
    predicate = _make_stop_when(sim, _agent_with([ActivityState.READY]), None, 1200.0)
    assert predicate is not None
    assert predicate() is False

    clock += _runner.MAX_PAUSE_SECONDS - 1  # a long, but answered, judge call
    paused = False
    assert predicate() is False  # resumed: the allowance is spent, not banked

    paused = True  # the next turn's judge call starts its own allowance
    assert predicate() is False
    clock += _runner.MAX_PAUSE_SECONDS - 1
    assert predicate() is False


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
    predicate = _make_stop_when(sim, agent, None, 1200.0)
    assert predicate is not None
    assert predicate() is True


def test_stop_when_lets_a_live_timeline_answer_the_question() -> None:
    """The guard order matters: while turns are still arriving one of them can resume the activity
    (Observe clears an InputWait on a user Message), so cutting the run here throws away a
    recoverable state rather than saving wall clock."""
    sim = _sim(running=True)
    agent = _agent_with([ActivityState.BLOCKED], [InputWait(prompt="How should I proceed?")])
    predicate = _make_stop_when(sim, agent, None, 1200.0)
    assert predicate is not None
    assert predicate() is False


def test_stop_when_still_waits_on_an_activity_blocked_for_a_signal() -> None:
    """Only a *question* ends the run. A SignalWait resolves from tool state, which can still
    settle after the timeline stops, so the narrower InputWait test is the deliberate one."""
    sim = _sim(running=False)
    agent = _agent_with([ActivityState.BLOCKED], [SignalWait(signal_name="job_done")])
    predicate = _make_stop_when(sim, agent, None, 1200.0)
    assert predicate is not None
    assert predicate() is False


def test_stop_when_a_question_does_not_excuse_work_still_in_flight() -> None:
    sim = _sim(running=False)
    agent = _agent_with(
        [ActivityState.READY, ActivityState.BLOCKED],
        [None, InputWait(prompt="How should I proceed?")],
    )
    predicate = _make_stop_when(sim, agent, None, 1200.0)
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
