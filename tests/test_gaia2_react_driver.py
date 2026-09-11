"""The ReAct arm's wiring: the engine ARE builds, the bracket it gets, and the config it runs at.

Nothing here runs a scenario or touches a network. What is worth pinning is the *injection* — that
ARE's own builders hand back a metered engine carrying this scenario's identity, and that the
agent's ``pause_env`` reaches the engine's bracket — because every one of those is a silent
degradation rather than a failure when it comes unstuck: an un-metered engine writes no rows, and an
unwired bracket undercounts a retried step without erroring.
"""

from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# `examples.gaia2.react_driver` imports ARE transitively, so the guard has to cover this
# module's own imports too: ARE is not a declared dependency and CI never installs it.
pytest.importorskip("are.simulation.agents.agent_builder")

from are.simulation.agents.agent_builder import AgentBuilder  # noqa: E402
from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig  # noqa: E402
from are.simulation.agents.default_agent.are_simulation_main import ARESimulationAgent  # noqa: E402
from are.simulation.time_manager import TimeManager  # noqa: E402
from examples.gaia2.evaluation.core import load_profiles  # noqa: E402
from examples.gaia2.latency_grid import EVAL_ROOT  # noqa: E402
from examples.gaia2.llm_calls import LLMCallWriter  # noqa: E402
from examples.gaia2.react_driver import (  # noqa: E402
    CapturingScenarioRunner,
    MeteredAgentBuilder,
    MeteredEngineBuilder,
    _timeline_expired,
    build_runner,
    run_react_on_scenario,
    runner_config,
    wire_bracket,
    wire_run_end,
)
from examples.gaia2.react_engine import MeteredLiteLLMEngine  # noqa: E402


@pytest.fixture
def profile() -> Any:
    return load_profiles(EVAL_ROOT / "profiles.json")["gpt-5.4-high-paper"]


def _agent(engine: Any, pause_env: Any) -> ARESimulationAgent:
    """A real ``ARESimulationAgent``, not a stand-in: the wiring is an attribute assignment on this
    class, and a duck-typed double would pass a test the installed ARE could still fail."""
    return ARESimulationAgent(
        log_callback=lambda _log: None,
        pause_env=pause_env,
        resume_env=lambda _offset: None,
        llm_engine=engine,
        base_agent=SimpleNamespace(),
        time_manager=TimeManager(),
    )


# -- the engine ARE builds ---------------------------------------------------------------------


def test_are_builds_the_arm_at_the_profiles_operating_point(profile: Any, tmp_path: Path) -> None:
    """ARE's config carries a model name and nothing else. Everything that makes the two arms the
    same experiment — reasoning setting, output cap, routing, streaming — comes from the profile."""
    writer = LLMCallWriter(tmp_path / "calls.jsonl")
    builder = MeteredEngineBuilder(profile, writer=writer, scenario_id="scenario-7", run_number=2)
    engine = builder.create_engine(LLMEngineConfig(model_name=profile.model))
    assert isinstance(engine, MeteredLiteLLMEngine)
    assert engine.scenario_id == "scenario-7" and engine.run_number == 2
    assert engine.writer is writer
    assert engine.stream is profile.stream
    assert engine.settings.request_kwargs["reasoning_effort"] == "high"
    assert engine.settings.request_kwargs["max_completion_tokens"] == 16_384
    assert builder.engines == [engine]


def test_a_runner_pointed_at_another_model_fails_before_it_spends(profile: Any) -> None:
    """A mis-wired experiment is worth a crash, not a run: the rows would look ordinary and the
    comparison would silently be against a different model."""
    builder = MeteredEngineBuilder(profile)
    with pytest.raises(ValueError, match="but the profile"):
        builder.create_engine(LLMEngineConfig(model_name="some-other-model"))


# -- the bracket -------------------------------------------------------------------------------


def test_the_agents_pause_signal_reaches_the_engines_bracket(profile: Any) -> None:
    """ARE calls ``pause_env`` once per ``step()``. That edge is the only thing that tells the
    engine a retried step's round-trips belong together."""
    calls: list[str] = []
    engine = MeteredLiteLLMEngine.from_profile(profile)
    agent = wire_bracket(_agent(engine, lambda: calls.append("are")))
    assert engine.bracketed is False
    agent.pause_env()
    agent.pause_env()
    assert engine.brackets == 2
    assert engine.bracketed is True
    # ARE's own callback still runs, and still runs after ours: the environment must actually pause.
    assert calls == ["are", "are"]


def test_the_profiles_settings_are_declared_forwardable_not_droppable(profile: Any) -> None:
    """LiteLLM validates a request against a per-model parameter list it ships, so a model newer
    than the installed version has its ``reasoning_effort`` rejected before the wire — which a live
    pilot found as every step of a run failing without a request ever being sent. The fix has to be
    the one that *forwards*: ``drop_params`` would delete the setting silently and run the baseline
    at the provider's default effort while S-ORA runs at the profile's, which is precisely the
    operating-point drift the sweep refuses to start with."""
    engine = MeteredLiteLLMEngine.from_profile(profile)
    allowed = engine.settings.request_kwargs["allowed_openai_params"]
    assert "reasoning_effort" in allowed
    assert "max_completion_tokens" in allowed
    # The envelopes LiteLLM passes through untouched have no business in a list of parameters it
    # is being told to allow, and the setting itself still has to be on the request.
    assert "extra_body" not in allowed
    assert "extra_headers" not in allowed
    assert engine.settings.request_kwargs["reasoning_effort"] == "high"


def test_an_unmetered_agent_is_left_alone(profile: Any) -> None:
    """A stock engine has no bracket to wire, and an agent built without a pause callback has no
    signal to wrap. Neither is an error — an unbracketed run undercounts, it does not break."""
    original = lambda: None  # noqa: E731 — identity is the assertion
    agent = wire_bracket(_agent(SimpleNamespace(model_name="stock"), original))
    assert agent.pause_env is original
    engine = MeteredLiteLLMEngine.from_profile(profile)
    assert wire_bracket(_agent(engine, None)).pause_env is None


def test_the_agent_builder_wires_what_are_hands_back(
    profile: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The subclass exists only for this. ARE's ``build`` is stubbed at the *parent*, so the method
    under test is the real one — if it ever stops returning the agent, or stops calling through,
    this says so rather than leaving a quiet undercount."""
    engine = MeteredLiteLLMEngine.from_profile(profile)
    built = _agent(engine, lambda: None)
    seen: dict[str, Any] = {}

    def fake_build(self: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return built

    monkeypatch.setattr(AgentBuilder, "build", fake_build)
    builder = MeteredAgentBuilder(MeteredEngineBuilder(profile))
    returned = builder.build(agent_config=object(), env=object())

    assert returned is built  # ARE's agent, not a wrapper around it
    assert set(seen) == {"agent_config", "env"}  # arguments passed through untouched
    built.pause_env()
    assert engine.brackets == 1


# -- the run configuration ---------------------------------------------------------------------


def test_the_arm_runs_ares_published_agent_configuration(profile: Any) -> None:
    config = runner_config(profile)
    assert config.agent == "default"
    assert config.model == profile.model and config.endpoint == profile.endpoint
    assert config.oracle is False
    # What makes `completion_duration` advance the simulated clock — the whole clock integration.
    assert config.simulated_generation_time_mode == "measured"


def test_max_turns_is_left_for_the_scenario_to_set(profile: Any) -> None:
    """ARE's config defaults it to 1, which would end a multi-turn scenario after one exchange —
    but ``run_scenario`` overrides it from ``scenario.nb_turns``, which every Gaia2 scenario has.
    Pinning it here would cap the run at whatever this file guessed."""
    assert "max_turns" not in runner_config(profile).model_fields_set


def test_the_runner_is_ares_own_with_only_the_agent_swapped(profile: Any, tmp_path: Path) -> None:
    writer = LLMCallWriter(tmp_path / "calls.jsonl")
    runner, engine_builder = build_runner(profile, writer=writer, scenario_id="s1", run_number=0)
    assert isinstance(runner.agent_builder, MeteredAgentBuilder)
    assert runner.agent_builder.llm_engine_builder is engine_builder
    assert engine_builder.scenario_id == "s1" and engine_builder.run_number == 0


# -- end to end, with only the transport faked -------------------------------------------------

_SCENARIO = (
    Path(__file__).resolve().parents[1]
    / "examples/gaia2/scenarios/execution/smoke-scenario_universe_25_vetd7u.json"
)
# One valid ReAct step: ARE parses the action, finds the tool, and the agent's turn ends. An
# invalid tool name would loop to the iteration cap instead and turn this into a slow test of
# ARE's error handling.
_REPLY = (
    "Thought: I will answer the user.\n"
    'Action:\n{\n  "action": "AgentUserInterface__send_message_to_user",'
    '\n  "action_input": {"content": "done"}\n}'
)


def test_a_whole_scenario_run_lands_rows_carrying_its_identity(
    profile: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real ARE environment, the real agent loop, the real scenario — only the wire is faked.

    Worth the couple of seconds it costs: every unit test above pins one seam, and the failure this
    catches is the one that shows up only when they are composed. `ScenarioRunner` refuses an
    uninitialized scenario, and nothing in the seams says so — an unscored run that skipped
    preprocessing produced a `success=None` result and zero rows, which reads exactly like a model
    that did nothing.
    """
    import are.simulation.agents.llm.litellm.litellm_engine as litellm_engine
    from examples.gaia2.react_driver import run_react_scenario
    from litellm.types.utils import Choices, Message, ModelResponse, Usage

    def fake_completion(**kwargs: Any) -> Any:
        assert not kwargs.get("stream"), "profile said not to stream"
        assert kwargs["reasoning_effort"] == "high"  # the profile reached the wire
        return ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content=_REPLY, role="assistant"),
                )
            ],
            usage=Usage(prompt_tokens=1200, completion_tokens=40),
        )

    monkeypatch.setattr(litellm_engine, "completion", fake_completion)
    # Non-streaming keeps the fake to one object; the streamed path has its own tests.
    unstreamed = replace(profile, stream=False)

    writer = LLMCallWriter(tmp_path / "calls.jsonl")
    try:
        result = run_react_scenario(
            str(_SCENARIO),
            unstreamed,
            writer=writer,
            run_number=3,
            scenario_duration=20,
            log_fn=lambda _msg: None,
        )
    finally:
        writer.close()

    assert result is not None
    rows = [json.loads(line) for line in writer.path.read_text().splitlines() if line.strip()]
    assert rows, "a completed run wrote no model calls"
    assert {row["arm"] for row in rows} == {"react"}
    assert {row["scenario_id"] for row in rows} == {"scenario_universe_25_vetd7u"}
    assert {row["run_number"] for row in rows} == {3}
    assert all(row["input_tokens"] == 1200 and row["output_tokens"] == 40 for row in rows)
    assert all(row["usage_captured"] is True for row in rows)
    # The bracket reached the engine through ARE's own pause callback, so a retried step's
    # round-trips would group rather than each being counted as its own step.
    assert all(row["bracket_id"] for row in rows)


# -- reading ARE's result without biasing the comparison -------------------------------------------


def _canned(runner_result: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``runner.run`` return a given ``ScenarioValidationResult`` without running anything.

    Patched at ``ScenarioRunner``, the parent, so the method under test is the real inherited
    one."""
    from are.simulation.scenario_runner import ScenarioRunner

    monkeypatch.setattr(ScenarioRunner, "run", lambda self, config, scenario: runner_result)


def _validation(**kwargs: Any) -> Any:
    from are.simulation.scenarios.scenario import ScenarioValidationResult

    return ScenarioValidationResult(**kwargs)


def test_an_unjudged_run_stays_unscored(profile: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """ARE's runner always calls ``scenario.validate``, and with no judge attached that falls
    through to the base implementation, which returns ``env.state != FAILED`` — True for any run
    that merely did not crash. Recorded as a score, that is a free pass for every unjudged run on
    this arm, and pass@1 would average it in against the other arm's honest None."""
    _canned(_validation(success=True, rationale="did not crash"), monkeypatch)
    result = run_react_on_scenario(
        SimpleNamespace(scenario_id="s1", judge=None, initialize=lambda: None),
        profile,
        log_fn=lambda _m: None,
    )
    assert result.outcome.success is None


def test_a_judged_runs_verdict_passes_through(
    profile: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _canned(_validation(success=True, rationale="all events matched"), monkeypatch)
    result = run_react_on_scenario(
        SimpleNamespace(scenario_id="s1", judge=object(), initialize=lambda: None),
        profile,
        log_fn=lambda _m: None,
    )
    assert result.outcome.success is True
    assert result.outcome.rationale == "all events matched"


def test_a_crash_is_recorded_as_an_exception_not_a_failure(
    profile: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ARE's ``run()`` collapses ``success=None`` plus an exception into ``success=False``. pass@1
    *excludes* an exception row and *counts* a failure row, so left collapsed an errored ReAct run
    would be scored as a genuine miss while the equivalent S-ORA run was dropped from the
    denominator — a bias with no sign anywhere in the artifacts."""
    boom = RuntimeError("engine died")
    _canned(_validation(success=False, exception=boom), monkeypatch)
    result = run_react_on_scenario(
        SimpleNamespace(scenario_id="s1", judge=object(), initialize=lambda: None),
        profile,
        log_fn=lambda _m: None,
    )
    assert result.outcome.success is None
    assert result.exception is boom


def test_the_runner_keeps_the_environment_are_built(
    profile: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_run`` builds the environment, uses it and drops it — but the two diagnostics that decide
    how a row is *read* (timeline expiry, the tool-call gate) are computed from it."""
    from are.simulation.scenario_runner import ScenarioRunner

    env = object()
    monkeypatch.setattr(
        ScenarioRunner, "_run_with_agent", lambda self, *a, **kw: _validation(success=True)
    )
    runner, _ = build_runner(profile)
    assert runner.environment is None
    runner._run_with_agent("s1", SimpleNamespace(), env)
    assert runner.environment is env


def test_the_capturing_runner_is_ares_own() -> None:
    from are.simulation.scenario_runner import ScenarioRunner

    assert issubclass(CapturingScenarioRunner, ScenarioRunner)


def test_timeline_expiry_mirrors_ares_own_exit_test() -> None:
    """ARE's loop runs ``while time_passed() <= duration``, so expiry is strictly greater-than. The
    row it marks is excluded from pass@1 — the world ended under the agent rather than the agent
    choosing badly — and the slower arm is the one that hits it."""
    assert _timeline_expired(
        SimpleNamespace(duration=100, time_manager=SimpleNamespace(time_passed=lambda: 101))
    )
    assert not _timeline_expired(
        SimpleNamespace(duration=100, time_manager=SimpleNamespace(time_passed=lambda: 100))
    )
    # ARE reads a None duration as "run indefinitely"; there is nothing to expire.
    assert not _timeline_expired(SimpleNamespace(duration=None))
    assert not _timeline_expired(None)


def test_a_broken_expiry_probe_never_costs_the_run_its_result() -> None:
    """It reinterprets every field beside it, so losing the whole result to it would be worse."""

    def boom() -> float:
        raise RuntimeError("no clock")

    assert not _timeline_expired(
        SimpleNamespace(duration=100, time_manager=SimpleNamespace(time_passed=boom))
    )


def test_expiry_is_latched_when_the_agent_stops_not_when_the_row_is_built(profile: Any) -> None:
    """ARE pauses nothing at shutdown, and validation runs *after* the agent returns.

    ``Environment.stop()`` sets the stop event and the state but leaves ``TimeManager`` on a wall
    clock, so a verdict read while the row is being assembled drifts True on any run that finished
    close to its budget — dropping a *successful* run out of pass@1. Latching at the moment the
    agent stopped makes the answer independent of how long everything afterwards took."""
    clock = iter([90.0, 5_000.0])
    env = SimpleNamespace(
        duration=100, time_manager=SimpleNamespace(time_passed=lambda: next(clock))
    )
    runner, _ = build_runner(profile)
    runner.environment = env

    runner.latch_expiry()  # the agent's loop has just returned: 90s of a 100s budget

    # Everything after that — validate(), export, the write-count check — runs on a clock that is
    # still moving, and must not be able to change the verdict.
    assert runner.timeline_expired() is False
    assert runner.timeline_expired() is False


def test_the_row_carries_the_latched_verdict_not_a_late_reading(
    profile: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row is assembled after the write-count check and the per-engine logging, and on a real
    run after ARE's own validate() and export as well — all on a clock that never stopped."""
    from are.simulation.scenario_runner import ScenarioRunner

    clock = iter([80.0, 9_000.0])
    env = SimpleNamespace(
        duration=100, time_manager=SimpleNamespace(time_passed=lambda: next(clock))
    )

    def run(self: Any, config: Any, scenario: Any) -> Any:
        self.environment = env
        self.latch_expiry()  # where ARE's agent hands control back
        return _validation(success=True)

    monkeypatch.setattr(ScenarioRunner, "run", run)
    result = run_react_on_scenario(
        SimpleNamespace(scenario_id="s1", judge=object(), initialize=lambda: None),
        profile,
        log_fn=lambda _m: None,
    )
    assert result.timeline_expired is False


def test_an_unlatched_runner_still_answers(profile: Any) -> None:
    """A runner nothing ever latched (no agent built, an aborted run) falls back to a live probe
    rather than silently reporting False."""
    runner, _ = build_runner(profile)
    runner.environment = SimpleNamespace(
        duration=100, time_manager=SimpleNamespace(time_passed=lambda: 101)
    )
    assert runner.timeline_expired() is True


def test_the_agents_own_run_is_what_latches_it() -> None:
    """The latch has to sit on the agent, because that is the only object that knows when the loop
    ended; ARE's runner reports it nowhere and validates before returning."""
    latched: list[float] = []
    passed = 10.0

    def run_scenario(*_args: Any, **_kwargs: Any) -> str:
        nonlocal passed
        passed = 90.0  # the agent worked for a while, and finished inside the budget
        return "output"

    agent = SimpleNamespace(run_scenario=run_scenario)
    wire_run_end(agent, lambda: latched.append(passed))
    assert agent.run_scenario() == "output"
    assert latched == [90.0]

    # A crashed run still has a real answer, so the latch fires on the raising path too.
    latched.clear()
    boom = SimpleNamespace(run_scenario=_raise)
    wire_run_end(boom, lambda: latched.append(passed))
    with pytest.raises(RuntimeError):
        boom.run_scenario()
    assert latched == [90.0]


def test_the_agent_are_builds_comes_back_latching(
    profile: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole chain, since each half is useless alone: the builder is the only object that sees
    the agent, the runner is the only one that holds the verdict, and one line joins them."""
    monkeypatch.setattr(
        AgentBuilder, "build", lambda self, **_kw: SimpleNamespace(run_scenario=lambda: "done")
    )
    runner, _ = build_runner(profile)
    runner.environment = SimpleNamespace(
        duration=100, time_manager=SimpleNamespace(time_passed=lambda: 42.0)
    )
    agent = runner.agent_builder.build(agent_config=None, env=None)

    assert runner._expired is None  # nothing latched before the agent ran
    assert agent.run_scenario() == "done"
    assert runner._expired is False


def test_an_agent_with_nothing_to_wrap_is_returned_untouched() -> None:
    agent = SimpleNamespace()
    assert wire_run_end(agent, lambda: None) is agent
    wrapped = SimpleNamespace(run_scenario=lambda: None)
    assert wire_run_end(wrapped, None) is wrapped


def _raise(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("agent died")


# -- the batch harness's react branch, composed --------------------------------------------------


def test_the_batch_harness_runs_the_react_arm_and_records_it(
    profile: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One scenario all the way through ``batch._run_one_scenario`` on the react arm.

    Every test above pins one seam, and the arm's first end-to-end run failed at none of them: an
    unpreprocessed scenario made ARE's runner refuse, which surfaced as a result with no score and
    no calls. The batch branch adds its own way to reach that state — ``--init-turns`` is optional
    on the S-ORA arm — so the composition is worth a test of its own. Only the wire is faked.
    """
    import are.simulation.agents.llm.litellm.litellm_engine as litellm_engine
    from examples.gaia2.batch import _run_one_scenario
    from litellm.types.utils import Choices, Message, ModelResponse, Usage

    from sora.adapters.are_sim import load_scenario

    def fake_completion(**kwargs: Any) -> Any:
        return ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content=_REPLY, role="assistant"),
                )
            ],
            usage=Usage(prompt_tokens=900, completion_tokens=20),
        )

    monkeypatch.setattr(litellm_engine, "completion", fake_completion)
    scenario = load_scenario(str(_SCENARIO))
    scenario.duration = 20
    scenario.run_number = 2

    writer = LLMCallWriter(tmp_path / "calls.jsonl")
    args = Namespace(
        arm="react",
        model_profile=replace(profile, stream=False),
        model="ReAct/gpt-5.4",
        config="examples/gaia2/agent.yaml",
        judge_model=None,
        judge_provider=None,
        judge_endpoint=None,
        strict_verdict_case=False,
        init_turns=False,  # not set, and the react branch must preprocess anyway
        verbose=False,
        max_wall_seconds=60.0,
    )
    try:
        record = _run_one_scenario(scenario, 2, args, str(tmp_path), llm_calls=writer)
    finally:
        writer.close()

    assert record["task_id"] == "scenario_universe_25_vetd7u"
    # Unscored, not "failed": no judge was attached, so ARE's base validate() saying "the env did
    # not fail" must not become a score.
    assert record["score"] is None
    assert record["metadata"]["status"] == "no_validation"
    assert "exception_type" not in record["metadata"], record["metadata"].get("exception_message")
    # The trace was exported under the arm that produced it, and the run actually made calls —
    # zero rows is what the preprocessing failure looked like.
    assert record["trace_id"] and Path(record["trace_id"]).exists()
    rows = [json.loads(line) for line in writer.path.read_text().splitlines() if line.strip()]
    assert rows and {row["arm"] for row in rows} == {"react"}
    assert {row["run_number"] for row in rows} == {2}


def test_the_driver_initializes_the_scenario_itself(
    profile: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A freshly loaded scenario runs, without the caller having preprocessed it.

    ARE's runner refuses an uninitialized scenario, and which call paths have already satisfied
    that is invisible from outside: ``load_scenario`` does not initialize, while ``attach_judge``
    and ``populate_oracle_events`` both do it as a side effect. A caller that guessed wrong got a
    result with no score and no model calls — a model that appears to have sat idle."""
    import are.simulation.agents.llm.litellm.litellm_engine as litellm_engine
    from litellm.types.utils import Choices, Message, ModelResponse, Usage

    from sora.adapters.are_sim import load_scenario

    monkeypatch.setattr(
        litellm_engine,
        "completion",
        lambda **kw: ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop", index=0, message=Message(content=_REPLY, role="assistant")
                )
            ],
            usage=Usage(prompt_tokens=800, completion_tokens=10),
        ),
    )
    scenario = load_scenario(str(_SCENARIO))
    scenario.duration = 20
    writer = LLMCallWriter(tmp_path / "calls.jsonl")
    try:
        result = run_react_on_scenario(
            scenario, replace(profile, stream=False), writer=writer, log_fn=lambda _m: None
        )
    finally:
        writer.close()

    assert result.exception is None
    assert writer.written, "the run made no model calls — ARE refused the scenario"


def test_the_judges_answers_are_collected_on_this_arm_too(
    profile: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one artifact a sweep cannot recover: ARE's graph judge keeps a bare boolean per judged
    event, so a scored run swept without the raw answers can never be re-scored at any price. The
    bracket has to span the whole run, not a later validate() — most events of a multi-turn
    scenario are judged at the per-turn release gate, mid-run."""
    from sora.adapters.are_judge import arm_judge_recording

    arm_judge_recording()
    _canned(_validation(success=True, rationale="ok"), monkeypatch)
    result = run_react_on_scenario(
        SimpleNamespace(scenario_id="s1", run_number=0, judge=object(), initialize=lambda: None),
        profile,
        record_judge=True,
        verdict_parse="case-insensitive",
        log_fn=lambda _m: None,
    )
    assert result.judge_recording is not None
    assert result.judge_recording.scenario_id == "s1"
    assert result.judge_recording.verdict_parse == "case-insensitive"


def test_the_recording_is_filed_under_the_run_number_it_was_given(
    profile: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batch harness sets both the argument and the scenario attribute and they agree. A direct
    caller passes only the argument — and a recording filed under a stale scenario attribute joins
    to a different run's model-call rows, which is what re-scoring reads them side by side for."""
    from sora.adapters.are_judge import arm_judge_recording

    arm_judge_recording()
    _canned(_validation(success=True, rationale="ok"), monkeypatch)
    result = run_react_on_scenario(
        SimpleNamespace(scenario_id="s1", run_number=0, judge=object(), initialize=lambda: None),
        profile,
        run_number=2,
        record_judge=True,
        log_fn=lambda _m: None,
    )
    assert result.judge_recording is not None
    assert result.judge_recording.run_number == 2


def test_a_recording_falls_back_to_the_scenarios_run_number() -> None:
    """Both arms resolve it the same way, so a rescore reading them side by side sees one rule."""
    from examples.gaia2._runner import _run_number_of

    assert _run_number_of(SimpleNamespace(run_number=0), 2) == 2
    assert _run_number_of(SimpleNamespace(run_number=1), None) == 1
    assert _run_number_of(SimpleNamespace(), None) is None
    assert _run_number_of(SimpleNamespace(run_number="1"), None) is None


def test_no_recording_is_kept_when_it_was_not_asked_for(
    profile: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _canned(_validation(success=True), monkeypatch)
    result = run_react_on_scenario(
        SimpleNamespace(scenario_id="s1", judge=object(), initialize=lambda: None),
        profile,
        log_fn=lambda _m: None,
    )
    assert result.judge_recording is None
