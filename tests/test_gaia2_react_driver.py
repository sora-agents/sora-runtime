"""The ReAct arm's wiring: the engine ARE builds, the bracket it gets, and the config it runs at.

Nothing here runs a scenario or touches a network. What is worth pinning is the *injection* — that
ARE's own builders hand back a metered engine carrying this scenario's identity, and that the
agent's ``pause_env`` reaches the engine's bracket — because every one of those is a silent
degradation rather than a failure when it comes unstuck: an un-metered engine writes no rows, and an
unwired bracket undercounts a retried step without erroring.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from are.simulation.agents.agent_builder import AgentBuilder
from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig
from are.simulation.agents.default_agent.are_simulation_main import ARESimulationAgent
from are.simulation.time_manager import TimeManager
from examples.gaia2.evaluation.core import load_profiles
from examples.gaia2.latency_grid import EVAL_ROOT
from examples.gaia2.llm_calls import LLMCallWriter
from examples.gaia2.react_driver import (
    MeteredAgentBuilder,
    MeteredEngineBuilder,
    build_runner,
    runner_config,
    wire_bracket,
)
from examples.gaia2.react_engine import MeteredLiteLLMEngine


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
            log=lambda _msg: None,
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
