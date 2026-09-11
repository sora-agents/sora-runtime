"""The ReAct baseline arm: one ARE scenario, run by ARE's own agent, metered per model call.

This is the *other* arm of the paired comparison. The S-ORA arm has ``run_benchmark.py``; this file
is its counterpart, and its whole job is to hand ARE's published agent the same scenario, the same
judge and the same model operating point, while recording one ``LLMCallRecord`` per round-trip so
both arms can be charged through one code path.

What it deliberately does *not* do is re-implement ARE's run. ``ScenarioRunner`` owns the
environment, the oracle mode, the turn wiring and the trace export, and the baseline is only a
baseline if that stays ARE's code. Everything here is injected through seams ARE already exposes:

- ``LLMEngineBuilder._create_concrete_engine`` — documented upstream as overridable — returns a
  :class:`MeteredLiteLLMEngine` built from the profile instead of a stock ``LiteLLMEngine``.
- ``AgentBuilder.build`` is subclassed only to wrap the agent's ``pause_env`` after ARE has
  constructed it. ARE calls ``pause_env`` exactly once at the top of every ``step()``, which is the
  only signal an engine gets that a *step* (rather than a round-trip) has begun, and a step that
  retries on malformed output otherwise loses the earlier round-trips' tokens. The attribute is
  reassigned before the agent runs, and ``ARESimulationAgent.initialize`` forwards it to the ReAct
  agent from there — so nothing is monkey-patched and no ARE source is copied.

Four things worth knowing before reading a number out of this
-------------------------------------------------------------
1. **The judge is attached here, not by ARE.** ``attach_judge`` is the same call the S-ORA arm
   makes, including ``relax_judge_verdict_case`` — ARE's engines lowercase ``True``/``False`` on the
   way out of every ``chat_completion``, so its ``[[True]]``-family checkers cannot return a verdict
   for any model, and an unparsed verdict reads as a rejection that also withholds later turns. Both
   arms must be scored by the same judge wiring or the comparison is between scorers.
2. **``max_turns`` comes from the scenario, not from the config.** ``ScenarioRunnerConfig`` defaults
   it to 1, which would stop a multi-turn scenario after the first exchange — but
   ``ARESimulationAgent.run_scenario`` overrides it with ``scenario.nb_turns`` whenever the scenario
   carries one, which every Gaia2 scenario does. It is left alone here for that reason.
3. **The judge's own calls pass through the same capture wrapper and are not recorded.** The
   wrapper applies the metered engine's settings from a thread-local that is set only around
   ``MeteredLiteLLMEngine.chat_completion``; a judge call finds it unset, so it runs as ARE built it
   and writes no row. Only the agent's calls are the agent's cost.
4. **Preprocessing is mandatory, and skipping it looks like an idle agent.** ``ScenarioRunner``
   refuses an uninitialized scenario, and ``load_scenario`` does not initialize — ARE's own path
   reaches it through ``preprocess_scenario``, with a judge configured or without one. Both
   branches below therefore preprocess; the judge only decides whether each turn's release is
   gated on a verdict. A run that skipped it returns ``success=None`` with zero recorded calls,
   which is indistinguishable at a glance from a model that did nothing.

Usage:

    python -m examples.gaia2.react_driver \
        --scenario /path/to/gaia2_scenario.json \
        --profile gpt-5.4-high-paper \
        --judge-model claude-sonnet-5 --judge-provider anthropic \
        --llm-calls react_calls.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from are.simulation.agents.agent_builder import AgentBuilder
from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig
from are.simulation.agents.llm.llm_engine_builder import LLMEngineBuilder
from are.simulation.scenario_runner import ScenarioRunner
from are.simulation.scenarios.config import ScenarioRunnerConfig

from examples.gaia2.evaluation.core import ModelProfile, load_profiles
from examples.gaia2.llm_calls import LLMCallWriter
from examples.gaia2.react_engine import ChargeModel, MeteredLiteLLMEngine

EVAL_ROOT = Path(__file__).resolve().parent / "evaluation"


class MeteredEngineBuilder(LLMEngineBuilder):  # type: ignore[misc]  # ARE is untyped
    """Builds the arm's engine from a :class:`ModelProfile` rather than from ARE's config.

    ARE hands ``_create_concrete_engine`` an ``LLMEngineConfig`` carrying a model name, a provider
    and an endpoint, and nothing else — no reasoning setting, no output cap, no routing. Those live
    on the profile, which is also what the S-ORA arm runs at, so the profile wins here and the
    config is used only to notice a disagreement early: a runner pointed at a different model than
    the profile is a mis-wired experiment, and it should fail before it spends anything."""

    def __init__(
        self,
        profile: ModelProfile,
        *,
        writer: LLMCallWriter | None = None,
        charge: ChargeModel | None = None,
        scenario_id: str | None = None,
        run_number: int | None = None,
    ) -> None:
        self.profile = profile
        self.writer = writer
        self.charge = charge
        self.scenario_id = scenario_id
        self.run_number = run_number
        self.engines: list[MeteredLiteLLMEngine] = []

    def _create_concrete_engine(self, engine_config: LLMEngineConfig) -> Any:
        configured = getattr(engine_config, "model_name", None)
        if configured and configured != self.profile.model:
            raise ValueError(
                f"runner is configured for model {configured!r} but the profile "
                f"{self.profile.name!r} is {self.profile.model!r}"
            )
        engine = MeteredLiteLLMEngine.from_profile(
            self.profile,
            writer=self.writer,
            charge=self.charge,
            scenario_id=self.scenario_id,
            run_number=self.run_number,
        )
        # Kept so a caller can read the round-trip and bracket counts after the run; ARE returns
        # only a validation result, and the engine is otherwise reachable through nothing.
        self.engines.append(engine)
        return engine


def wire_bracket(agent: Any) -> Any:
    """Point the agent's ``pause_env`` at its metered engine's bracket, and hand the agent back.

    Separate from the builder because it is the whole ARE-facing contract in one place: ARE calls
    ``pause_env`` once per ``step()``, the engine needs that edge to group a retried step's
    round-trips, and the assignment has to land after construction and before the run. An agent
    whose engine is not metered, or that was built with no pause callback, is returned untouched —
    an unbracketed run degrades to per-round-trip metadata, which undercounts and never
    double-charges."""
    engine = getattr(agent, "llm_engine", None)
    pause_env = getattr(agent, "pause_env", None)
    if isinstance(engine, MeteredLiteLLMEngine) and pause_env is not None:
        agent.pause_env = engine.wrap_pause_env(pause_env)
    return agent


class MeteredAgentBuilder(AgentBuilder):  # type: ignore[misc]  # ARE is untyped
    """ARE's agent, with the bracket wired. The build itself stays ARE's."""

    def build(self, *args: Any, **kwargs: Any) -> Any:
        return wire_bracket(super().build(*args, **kwargs))


def build_runner(
    profile: ModelProfile,
    *,
    writer: LLMCallWriter | None = None,
    charge: ChargeModel | None = None,
    scenario_id: str | None = None,
    run_number: int | None = None,
) -> tuple[ScenarioRunner, MeteredEngineBuilder]:
    """A ``ScenarioRunner`` whose agent is metered, plus the engine builder to read counts off."""
    engine_builder = MeteredEngineBuilder(
        profile,
        writer=writer,
        charge=charge,
        scenario_id=scenario_id,
        run_number=run_number,
    )
    return ScenarioRunner(agent_builder=MeteredAgentBuilder(engine_builder)), engine_builder


def runner_config(profile: ModelProfile, **overrides: Any) -> ScenarioRunnerConfig:
    """The published Gaia2 agent configuration, pointed at one profile.

    ``simulated_generation_time_mode="measured"`` is ARE's own default and is what makes the
    metadata's ``completion_duration`` advance the simulated clock — the single integration point
    for freezing the environment during generation. ``max_turns`` is deliberately not set: the
    scenario's ``nb_turns`` overrides it anyway, and pinning it here would cap a multi-turn
    scenario at whatever this file guessed."""
    return ScenarioRunnerConfig(
        model=profile.model,
        model_provider=profile.provider,
        endpoint=profile.endpoint,
        agent="default",
        oracle=False,
        simulated_generation_time_mode="measured",
        **overrides,
    )


def run_react_scenario(
    scenario_ref: str,
    profile: ModelProfile,
    *,
    writer: LLMCallWriter | None = None,
    charge: ChargeModel | None = None,
    run_number: int | None = None,
    judge_model: str | None = None,
    judge_provider: str | None = None,
    judge_endpoint: str | None = None,
    scenario_duration: float | None = None,
    output_dir: str | None = None,
    export: bool = False,
    log: Any = print,
) -> Any:
    """Run one scenario against ARE's agent and return its ``ScenarioValidationResult``.

    The order matters and is the same order the S-ORA arm uses: load, then judge (or turn wiring),
    then run. ``attach_judge`` replays the oracle events to build the graph the judge scores
    against and installs the per-turn release gate, so it has to precede the environment start that
    ``ScenarioRunner`` performs."""
    from sora.adapters.are_sim import attach_judge, initialize_turns, load_scenario

    scenario = load_scenario(scenario_ref)
    scenario_id = getattr(scenario, "scenario_id", None)
    if scenario_duration is not None:
        # A real-time allowance for the whole run, not simulated time: ARE's loop sleeps a real
        # second per tick. Raising it does not shift the scripted schedule (every Gaia2 delay is
        # relative to the event that fires it) but it does move `get_current_time`, so a number
        # produced under an override is not comparable to a published one. Announced for that
        # reason.
        log(f"scenario duration: {scenario.duration}s -> {scenario_duration}s (overridden)")
        scenario.duration = scenario_duration
    # Preprocessing is not optional on this arm, and it is easy to assume it is. `ScenarioRunner`
    # refuses an uninitialized scenario outright, and ARE's own non-oracle path always preprocesses
    # — with a judge when one is configured and without one otherwise. Both branches below are that
    # same call; the judge is what decides whether each turn's release is gated on a verdict.
    if judge_model is not None:
        attach_judge(
            scenario,
            model=judge_model,
            provider=judge_provider,
            endpoint=judge_endpoint,
        )
    else:
        # Unscored, and every turn still delivered: ARE's dummy trigger always releases the next
        # one. The later turns hang off OracleEvents an agent-mode environment would otherwise
        # ignore, so without this the run silently stops after turn 1.
        initialize_turns(scenario)

    runner, engine_builder = build_runner(
        profile,
        writer=writer,
        charge=charge,
        scenario_id=scenario_id,
        run_number=run_number,
    )
    config = runner_config(
        profile,
        output_dir=output_dir,
        export=export,
    )
    result = runner.run(config, scenario)
    for engine in engine_builder.engines:
        log(
            f"    {engine.round_trips} model round-trips over {engine.brackets} steps"
            + ("" if engine.bracketed else "  (unbracketed: retries counted as steps)")
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", required=True, help="path to a Gaia2 scenario JSON")
    parser.add_argument("--profile", required=True, help="profile name from profiles.json")
    parser.add_argument("--profiles-path", type=Path, default=EVAL_ROOT / "profiles.json")
    parser.add_argument("--llm-calls", type=Path, default=None, help="JSONL to append call rows to")
    parser.add_argument("--run-number", type=int, default=None)
    parser.add_argument("--judge-model", default=None, help="unscored without this")
    parser.add_argument("--judge-provider", default=None)
    parser.add_argument("--judge-endpoint", default=None)
    parser.add_argument("--scenario-duration", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--export", action="store_true", help="write ARE's HF trace")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profiles = load_profiles(args.profiles_path)
    if args.profile not in profiles:
        raise SystemExit(f"unknown profile {args.profile!r}; have {sorted(profiles)}")
    profile = profiles[args.profile]
    print(f"profile {profile.name} -> {profile.model} at {profile.endpoint}  (arm: react)")
    writer = LLMCallWriter(args.llm_calls) if args.llm_calls else None
    try:
        result = run_react_scenario(
            args.scenario,
            profile,
            writer=writer,
            run_number=args.run_number,
            judge_model=args.judge_model,
            judge_provider=args.judge_provider,
            judge_endpoint=args.judge_endpoint,
            scenario_duration=args.scenario_duration,
            output_dir=args.output_dir,
            export=args.export,
        )
    finally:
        if writer is not None:
            writer.close()
            print(f"    wrote {writer.written} model calls to {writer.path}")
    success = getattr(result, "success", None)
    print(f"{'PASS' if success else 'FAIL' if success is False else 'UNSCORED'}: {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
