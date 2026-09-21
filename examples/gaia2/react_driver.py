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
4. **Initialization is mandatory, and skipping it looks like an idle agent.** ``ScenarioRunner``
   refuses an uninitialized scenario, and ``load_scenario`` does not initialize — ARE's own path
   reaches it through ``preprocess_scenario``, while ``populate_oracle_events`` does it as a side
   effect, so whether a given caller has already satisfied it is invisible from outside. A run
   that skipped it returns ``success=None`` with zero recorded calls, which is indistinguishable
   at a glance from a model that did nothing, so :func:`run_react_on_scenario` meets the
   requirement itself (idempotently) rather than leaving it to each caller. Preprocessing proper
   is a separate matter: it is what decides whether turns 2..n are delivered, and the judge is
   what decides whether each release is gated on a verdict.

Usage:

    python -m examples.gaia2.react_driver \
        --scenario /path/to/gaia2_scenario.json \
        --profile gpt-5.4-high-paper \
        --judge-model claude-sonnet-5 --judge-provider anthropic \
        --llm-calls react_calls.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from are.simulation.agents.agent_builder import AgentBuilder
from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig
from are.simulation.agents.llm.llm_engine_builder import LLMEngineBuilder
from are.simulation.scenario_runner import ScenarioRunner
from are.simulation.scenarios.config import ScenarioRunnerConfig

from examples.gaia2._runner import RunResult, _run_number_of, _terminal_cause
from examples.gaia2.evaluation.core import (
    MODEL_SNAPSHOTS,
    ModelProfile,
    harness_truncation,
    load_profiles,
    resolve_clock_mode,
    resolve_max_wall_seconds,
)
from examples.gaia2.evaluation.core import (
    served_snapshot as _served_snapshot,
)
from examples.gaia2.llm_calls import (
    LLMCallWriter,
    charged_time_union_seconds,
    round_trip_concurrency,
    wall_time_union_seconds,
)
from examples.gaia2.react_engine import ChargeModel, MeteredLiteLLMEngine

log = logging.getLogger(__name__)

EVAL_ROOT = Path(__file__).resolve().parent / "evaluation"
GAIA_LOGICAL_AGENT_LLM_CALL_LIMIT = 200


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
        generation_free: bool = False,
        scenario_id: str | None = None,
        run_number: int | None = None,
    ) -> None:
        self.profile = profile
        self.writer = writer
        self.charge = charge
        self.generation_free = generation_free
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
            generation_free=self.generation_free,
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
        environment = getattr(pause_env, "__self__", None)
        pause_generation = getattr(environment, "pause_generation", None)
        resume_generation = getattr(environment, "resume_generation", None)
        if callable(pause_generation) and callable(resume_generation):
            active_token: int | None = None

            def pause_charged_generation() -> None:
                nonlocal active_token
                active_token = int(pause_generation())
                charge_time = getattr(environment, "generation_charge_time", None)
                engine.begin_bracket(
                    float(charge_time(active_token))
                    if active_token != 0 and callable(charge_time)
                    else None
                )

            def resume_charged_generation(offset: float) -> None:
                nonlocal active_token
                if active_token is None:
                    return
                token, active_token = active_token, None
                # On an exception ARE's finally resumes with zero because no metadata returned.
                # The engine still recorded and priced that crossing. Ignore ARE's argument on
                # both paths so measured wall time can never leak back into the frozen clock.
                del offset
                resume_generation(token, engine.bracket_charge)

            agent.pause_env = pause_charged_generation
            agent.resume_env = resume_charged_generation
        else:
            agent.pause_env = engine.wrap_pause_env(pause_env)
    return agent


def wire_run_end(agent: Any, latch: Any) -> Any:
    """Call ``latch`` when the agent's loop returns, and hand the agent back.

    This is the only moment at which "did the world end under the agent?" has a stable answer.
    The harness clock resumes after ``Environment.stop()``, and the very next thing ARE does after
    the agent returns is ``scenario.validate()``, whose judge pass can take minutes. A verdict read
    off the clock afterwards would drift True on any run that finished close enough to the budget,
    which is a *successful* run being dropped from pass@1. The S-ORA arm latches at its own
    shutdown for exactly this reason; this is where that instant is on ARE's side. Latched on the
    raising path too, since a crashed run still has a real answer."""
    run_scenario = getattr(agent, "run_scenario", None)
    if latch is None or run_scenario is None:
        return agent

    def latching(*args: Any, **kwargs: Any) -> Any:
        try:
            return run_scenario(*args, **kwargs)
        finally:
            latch()

    agent.run_scenario = latching
    return agent


class MeteredAgentBuilder(AgentBuilder):  # type: ignore[misc]  # ARE is untyped
    """ARE's agent, with the experiment's cap and metering hooks applied after its own build.

    ARE defaults both the wrapper and its inner ReAct loop to 80 iterations. The inner counter is
    the logical-call admission limit: it includes errored attempts, survives across scenario turns,
    and produces ARE's judged max-iterations message when exhausted. Set both copies after
    delegation because ARE's builder overwrites the inner value from the wrapper default while it
    constructs the agent.
    """

    on_run_end: Any = None
    agent: Any = None

    def build(self, *args: Any, **kwargs: Any) -> Any:
        agent = super().build(*args, **kwargs)
        agent.max_iterations = GAIA_LOGICAL_AGENT_LLM_CALL_LIMIT
        agent.react_agent.max_iterations = GAIA_LOGICAL_AGENT_LLM_CALL_LIMIT
        self.agent = agent
        return wire_run_end(wire_bracket(agent), self.on_run_end)

    def max_iterations_reached(self) -> bool:
        """Read ARE's terminal marker, which is logged rather than raised."""
        react_agent = getattr(self.agent, "react_agent", None)
        get_logs = getattr(react_agent, "get_agent_logs", None)
        if not callable(get_logs):
            return False
        try:
            return any(
                getattr(entry, "error", None) == "MaxIterationsAgentError" for entry in get_logs()
            )
        except Exception:  # a diagnostic must never cost the run its real result
            log.warning("max-iterations probe failed", exc_info=True)
            return False


class CapturingScenarioRunner(ScenarioRunner):  # type: ignore[misc]  # ARE is untyped
    """ARE's runner, keeping a reference to the ``Environment`` it built and the moment it ended.

    ``_run`` constructs the environment, runs the agent against it, exports, stops it and returns
    only a ``ScenarioValidationResult`` — the environment itself is unreachable from outside. Two
    of this harness's per-run diagnostics need it, and *both* of them decide how a row is read
    rather than decorating it:

    * ``timeline_expired`` — ARE's event loop is wall-clock paced, so ``scenario.duration`` is a
      real-time budget and a slow arm can have the world end under it mid-run. A row that expired
      is excluded from pass@1 rather than counted as a miss. The S-ORA arm records it; an arm that
      could not would have its timeouts scored as genuine failures while the other arm's were
      dropped — a bias in favour of whichever arm reports it, which is exactly the wrong direction
      here, since the slower arm is the one that expires. Latched when the agent's loop returns
      rather than read when the row is built, because the clock keeps running afterwards; see
      :func:`wire_run_end`.
    * ``write_count_check`` — ARE's tool-call-count gate, recomputed offline for no tokens, and the
      only pass/fail signal an unscored sweep has.

    Capturing it changes nothing about how ARE runs the scenario — it records what ARE would have
    discarded, and samples a clock. The alternative is reimplementing ``_run``, which would fork
    the part of ARE the baseline exists to keep."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.environment: Any = None
        self._expired: bool | None = None
        self.wall_timed_out = False
        self.judge_timed_out = False

    def _run(self, config: Any, scenario: Any) -> Any:
        # ScenarioRunner imported Environment into its module namespace. Patch that construction
        # seam for this call only; copying ARE's _run would fork the baseline implementation.
        from unittest.mock import patch

        from examples.gaia2.simulated_clock import ChargedEnvironment

        with patch("are.simulation.scenario_runner.Environment", ChargedEnvironment):
            return super()._run(config, scenario)

    def _run_with_agent(self, scenario_id: str, scenario: Any, env: Any, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        self.environment = env
        try:
            return super()._run_with_agent(scenario_id, scenario, env, *args, **kwargs)
        finally:
            # Normally already latched, by the agent wrapper, before ARE validated. This is the
            # backstop for an agent that never went through `wire_run_end` — still ahead of the
            # export and of anything this harness does afterwards.
            self.latch_expiry()

    def latch_expiry(self) -> None:
        """Sample the expiry verdict once, and keep the first answer; see :func:`wire_run_end`."""
        if self._expired is None:
            self._expired = _timeline_expired(self.environment)

    def timeline_expired(self) -> bool:
        """The latched verdict, or a live probe when nothing ever latched one."""
        if self._expired is not None:
            return self._expired
        return _timeline_expired(self.environment)


def build_runner(
    profile: ModelProfile,
    *,
    writer: LLMCallWriter | None = None,
    charge: ChargeModel | None = None,
    generation_free: bool = False,
    scenario_id: str | None = None,
    run_number: int | None = None,
) -> tuple[CapturingScenarioRunner, MeteredEngineBuilder]:
    """A ``ScenarioRunner`` whose agent is metered, plus the engine builder to read counts off."""
    engine_builder = MeteredEngineBuilder(
        profile,
        writer=writer,
        charge=charge,
        generation_free=generation_free,
        scenario_id=scenario_id,
        run_number=run_number,
    )
    agent_builder = MeteredAgentBuilder(engine_builder)
    runner = CapturingScenarioRunner(agent_builder=agent_builder)
    # The builder is what sees the agent ARE builds, and the runner is what holds the verdict; this
    # is the one line that joins them, and without it expiry would be read minutes late.
    agent_builder.on_run_end = runner.latch_expiry
    return runner, engine_builder


def runner_config(profile: ModelProfile, **overrides: Any) -> ScenarioRunnerConfig:
    """The published Gaia2 agent configuration, pointed at one profile.

    ``simulated_generation_time_mode="measured"`` keeps ARE's native pause/resume path active; the
    harness wrapper discards its computed offset and substitutes the frozen bracket charge.
    ``max_turns`` is deliberately not set: the
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
    generation_free: bool = False,
    run_number: int | None = None,
    judge_model: str | None = None,
    judge_provider: str | None = None,
    judge_endpoint: str | None = None,
    scenario_duration: float | None = None,
    output_dir: str | None = None,
    export: bool = False,
    max_wall_seconds: float = 1200.0,
    charge_model_identity: dict[str, Any] | None = None,
    charge_model_digest: str | None = None,
    log_fn: Any = print,
) -> RunResult:
    """Load one scenario, preprocess it, and run it against ARE's agent.

    The order matters and is the same order the S-ORA arm uses: load, then judge (or turn wiring),
    then run. ``attach_judge`` replays the oracle events to build the graph the judge scores
    against and installs the per-turn release gate, so it has to precede the environment start that
    ``ScenarioRunner`` performs."""
    from sora.adapters.are_sim import attach_judge, initialize_turns, load_scenario

    scenario = load_scenario(scenario_ref)
    if scenario_duration is not None:
        # A real-time allowance for the whole run, not simulated time: ARE's loop sleeps a real
        # second per tick. Raising it does not shift the scripted schedule (every Gaia2 delay is
        # relative to the event that fires it) but it does move `get_current_time`, so a number
        # produced under an override is not comparable to a published one. Announced for that
        # reason.
        log_fn(f"scenario duration: {scenario.duration}s -> {scenario_duration}s (overridden)")
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

    return run_react_on_scenario(
        scenario,
        profile,
        writer=writer,
        charge=charge,
        generation_free=generation_free,
        run_number=run_number,
        output_dir=output_dir,
        export=export,
        max_wall_seconds=max_wall_seconds,
        charge_model_identity=charge_model_identity,
        charge_model_digest=charge_model_digest,
        log_fn=log_fn,
    )


def _timeline_expired(env: Any) -> bool:
    """Whether ARE's clock, not the agent, ended the run — read off the environment ARE built.

    Mirrors the loop's own exit test (``while time_passed() <= duration``) rather than approximating
    it, and is the same condition ``AreSimulation.timeline_expired`` reports on the S-ORA arm. Never
    raises: it reinterprets every field beside it, so losing it to a probe failure would be worse
    than losing any one of them. *When* it is asked matters as much as what it asks — ARE pauses
    nothing at shutdown — so callers go through ``CapturingScenarioRunner.timeline_expired``, which
    answers from the instant the agent stopped."""
    if env is None:
        return False
    duration = getattr(env, "duration", None)
    if duration is None:  # ARE reads None as "run indefinitely" — nothing to expire
        return False
    try:
        return bool(env.time_manager.time_passed() > duration)
    except Exception:  # a diagnostic must never cost the run its real result
        log.warning("timeline-expiry probe failed", exc_info=True)
        return False


def run_react_on_scenario(
    scenario: Any,
    profile: ModelProfile,
    *,
    writer: LLMCallWriter | None = None,
    charge: ChargeModel | None = None,
    generation_free: bool = False,
    run_number: int | None = None,
    output_dir: str | None = None,
    export: bool = False,
    record_judge: bool = False,
    verdict_parse: str | None = None,
    max_wall_seconds: float = 1200.0,
    charge_model_identity: dict[str, Any] | None = None,
    charge_model_digest: str | None = None,
    log_fn: Any = print,
) -> RunResult:
    """Run ARE's agent against one **already preprocessed** scenario, as a :class:`RunResult`.

    This is the batch-facing entry point, and it returns the S-ORA arm's result type on purpose: a
    sweep that writes one row shape for both arms cannot grow a per-arm discrepancy in how a row is
    scored, aggregated or excluded. The S-ORA-only fields (``llm_report``, ``replan_count``,
    ``prop_reads``, ``decision_cycles``, ...) are left at their defaults rather than filled with a
    plausible-looking ReAct analogue — ``agent_llm_calls`` counts *logical* calls on that arm and
    would silently become a round-trip count here, which is a different measurement. The per-call
    rows in ``llm_calls.jsonl`` are the comparable record.

    Two normalizations of ARE's result, both of which would otherwise bias the comparison:

    * **An unscored run must stay unscored.** ``ScenarioRunner`` always calls ``scenario.validate``,
      and without an attached judge that falls through to ARE's base implementation, which returns
      ``env.state != FAILED`` — True for any run that merely did not crash. Reported as a score,
      that is a free pass for every unjudged ReAct run. The S-ORA arm guards on the same condition.
    * **A crash is not a failure.** ARE's ``run()`` collapses ``success=None`` plus an exception
      into ``success=False``. The S-ORA arm records a crash as an ``exception`` row, which pass@1
      *excludes*; left collapsed, an errored ReAct run would be counted as a genuine miss while
      the equivalent S-ORA run was dropped from the denominator."""
    # ARE's runner refuses an uninitialized scenario outright, and a caller has no reason to know
    # that: `load_scenario` does not initialize, while `attach_judge` and `populate_oracle_events`
    # both do it as a side effect, so whether a given call path has satisfied it is invisible.
    # Idempotent (guarded by `Scenario._initialized`), so meeting the requirement here costs a
    # no-op on every path that already did. Kept out of the callers on purpose: it is this
    # function's precondition, and the S-ORA arm — which starts its own environment — has no
    # equivalent, so leaving it to the harness would mean an arm-shaped rule in shared code.
    scenario.initialize()
    runner, engine_builder = build_runner(
        profile,
        writer=writer,
        charge=charge,
        generation_free=generation_free,
        scenario_id=getattr(scenario, "scenario_id", None),
        run_number=run_number,
    )
    config = runner_config(profile, output_dir=output_dir, export=export)

    with contextlib.ExitStack() as bracket:
        collector: Any = None
        if record_judge:
            from sora.adapters.are_judge import record_judge_events

            # Brackets the whole run, not a later validate(): ARE scores each turn's release gate
            # mid-run, so most judged events of a multi-turn scenario are decided inside run().
            collector = bracket.enter_context(record_judge_events())

        stop_watchdog = threading.Event()

        def watchdog() -> None:
            deadline = time.monotonic() + max_wall_seconds
            judge_paused_since: float | None = None
            while not stop_watchdog.wait(0.05):
                now = time.monotonic()
                env = runner.environment
                judge_paused = (
                    bool(getattr(env, "judge_paused", False)) if env is not None else False
                )
                judge_paused_since = (
                    now if judge_paused and judge_paused_since is None else judge_paused_since
                )
                if (
                    judge_paused
                    and judge_paused_since is not None
                    and now - judge_paused_since >= 180.0
                ):
                    runner.judge_timed_out = True
                if now >= deadline:
                    runner.wall_timed_out = True
                if runner.judge_timed_out or runner.wall_timed_out:
                    if env is not None:
                        env.stop()
                        return
                    # The deadline fired before `ScenarioRunner` published an environment, so
                    # there is nothing to stop yet. Keep watching instead of retiring: returning
                    # here leaves the timeout flagged but never enforced, and a run whose setup
                    # is what overran would then proceed uncapped. `stop_watchdog` ends the loop
                    # when the run finishes on its own.
                    continue
                if not judge_paused:
                    judge_paused_since = None

        watcher = threading.Thread(target=watchdog, name="Gaia2WallWatchdog", daemon=True)
        watcher.start()
        started = time.monotonic()
        try:
            result = runner.run(config, scenario)
        finally:
            stop_watchdog.set()
            watcher.join(timeout=1.0)
        duration = time.monotonic() - started

    env = runner.environment
    exc = getattr(result, "exception", None)
    success = getattr(result, "success", None)
    if exc is not None:
        success = None  # un-collapse ARE's exception-as-failure; see the docstring
    elif getattr(scenario, "judge", None) is None:
        success = None  # unjudged: ARE's base validate() would report a bare "did not crash"

    counts: Any = None
    try:
        from sora.adapters.are_sim import write_count_check

        counts = write_count_check(scenario, env) if env is not None else None
    except Exception:  # a diagnostic must never cost the run its real result
        log.warning("write-count check failed", exc_info=True)

    for engine in engine_builder.engines:
        log_fn(
            f"    {engine.round_trips} model round-trips over {engine.brackets} steps"
            + ("" if engine.bracketed else "  (unbracketed: retries counted as steps)")
        )

    from sora.adapters.are_sim import ValidationOutcome

    expired = runner.timeline_expired()
    typed_exc = exc if isinstance(exc, Exception) else None
    terminal_cause = _terminal_cause(
        typed_exc,
        expired,
        "timeout" if runner.wall_timed_out or runner.judge_timed_out else None,
        success if isinstance(success, bool) else None,
        max_iterations_reached=runner.agent_builder.max_iterations_reached(),
    )

    charged_seconds = sum(engine.charged_seconds for engine in engine_builder.engines)
    # The callable is one per scenario and sees every trip. Prefer its total when available so the
    # result and the independently written per-call records expose the same accumulator.
    charged_seconds = float(getattr(charge, "charged_seconds", charged_seconds))
    all_windows = tuple(
        window for engine in engine_builder.engines for window in engine.round_trip_windows
    )
    concurrency = round_trip_concurrency(all_windows)
    return RunResult(
        outcome=ValidationOutcome(success=success, rationale=getattr(result, "rationale", None)),
        environment=env,
        duration=duration,
        exception=typed_exc,
        write_counts=counts,
        timeline_expired=expired,
        harness_truncation=harness_truncation(
            wall_timed_out=runner.wall_timed_out, judge_timed_out=runner.judge_timed_out
        ),
        terminal_cause=terminal_cause,
        judge_recording=(
            None
            if collector is None
            else collector.snapshot(
                scenario_id=getattr(scenario, "scenario_id", None),
                run_number=_run_number_of(scenario, run_number),
                verdict_parse=verdict_parse,
            )
        ),
        charged_seconds=charged_seconds,
        charge_model_identity=charge_model_identity,
        charge_model_digest=charge_model_digest,
        cached_input_clamps=int(getattr(charge, "cached_input_clamps", 0)),
        raw_cached_input_anomalies=sum(
            engine.raw_cached_input_anomalies for engine in engine_builder.engines
        ),
        charge_accounting_consistent=(
            None
            if charge is None
            else int(getattr(charge, "cached_input_clamps", 0))
            == sum(engine.raw_cached_input_anomalies for engine in engine_builder.engines)
        ),
        clock_mode=resolve_clock_mode(charge, generation_free),
        inference_charge_policy="serialized_sum" if charge is not None else None,
        llm_wall_seconds=sum(engine.wall_seconds for engine in engine_builder.engines),
        # ReAct is serial today, but calculate the same sensitivity rather than assuming that
        # implementation detail forever. Engines share one host monotonic-clock domain.
        llm_wall_union_seconds=wall_time_union_seconds(all_windows),
        llm_charged_union_seconds=charged_time_union_seconds(all_windows),
        llm_round_trips=concurrency.round_trips,
        llm_max_in_flight=concurrency.max_in_flight,
        llm_overlapped_round_trips=concurrency.overlapped_round_trips,
    )


PREFLIGHT_MESSAGES = [{"role": "user", "content": "Reply with the single word: ok"}]

# Keyed by the request kwarg each probe replaces, so coverage is measured against what
# `request_kwargs()` actually sends rather than against the profile's settings — those also carry
# identity and transport, which no request parameter corresponds to. Only values whose refusal has
# been *observed* belong here: a provider that silently clamps an out-of-range value instead of
# refusing it would be reported as having dropped the setting, which is a false alarm on the one
# check whose whole job is to be trusted. Everything sent without a probe is named as unprobed
# rather than passed over in silence.
WIRE_PROBES: dict[str, dict[str, Any]] = {
    # OpenAI answers with its own enumeration of the values it accepts.
    "reasoning_effort": {"reasoning_effort": "supreme"},
    # OpenRouter answers with the providers that really serve the model — which is also the
    # cheapest way to find the alternatives when a pin dies. One probe covers the whole envelope:
    # `reasoning` rides in the same dict, and LiteLLM either forwards `extra_body` or it does not.
    "extra_body": {
        "extra_body": {
            "provider": {"only": ["sora-preflight-no-such-provider"], "allow_fallbacks": False}
        }
    },
    # OpenAI enumerates its tiers the same way. Only OpenAI validates this one: OpenRouter answers
    # 200 and echoes `service_tier: null`, which is why no profile of its sends it.
    "service_tier": {"service_tier": "supreme"},
}


def _provider_refused(exc: BaseException) -> tuple[bool, str]:
    """Whether the far end refused, walking the ``__cause__`` chain ARE's wrapper adds.

    ``UnsupportedParamsError`` is the one refusal that is *not* proof: LiteLLM raises it from its
    own per-model table without sending anything, which is the failure ``allowed_openai_params``
    exists to prevent. Reported apart from a clean crossing for that reason."""
    from litellm.exceptions import UnsupportedParamsError

    seen: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        seen.append(f"{type(current).__name__}: {current}")
        if isinstance(current, UnsupportedParamsError):
            return False, (
                "refused by LiteLLM before the wire (allowed_openai_params no longer covers it)"
            )
        current = current.__cause__
    return True, seen[-1].split("\n")[0][:200]


class _Collected:
    """Stands in for :class:`LLMCallWriter`; the engine only ever calls ``write``."""

    def __init__(self) -> None:
        self.rows: list[Any] = []

    def write(self, record: Any) -> None:
        self.rows.append(record)


def _asks_for_reasoning(request: Mapping[str, Any]) -> bool:
    """Whether this profile asked the model to think, by either of the two routes in use.

    ``reasoning_effort`` is a validated OpenAI parameter; OpenRouter's ``reasoning`` block rides
    ``extra_body`` instead, so a profile can request reasoning without naming a level at all."""
    if request.get("reasoning_effort"):
        return True
    extra = request.get("extra_body")
    return bool(isinstance(extra, dict) and extra.get("reasoning"))


def preflight(
    profile: ModelProfile,
    *,
    factory: Any = None,
    log: Any = print,
    served_snapshot: Any = None,
) -> int:
    """Send one live call on this profile's route, then prove its settings crossed the wire.

    The ReAct arm reaches the provider through LiteLLM rather than through an SDK client, so it
    resolves ``provider``/``model``/``endpoint`` by its own rules — rules no test that fakes the
    wire exercises, and that a repin can invalidate silently. The grid has the same guard for the
    other arm's client; this is that guard for this one, with the same contract: a non-zero exit,
    so a script chaining preflight into a sweep stops here rather than discovering it partway
    through a paid run.

    A returning call is *not* on its own evidence that the arm runs at the profile's operating
    point. The failure mode here drops a parameter rather than raising — ``drop_params`` is
    literally what LiteLLM recommends — and a dropped ``reasoning_effort`` still comes back with a
    plausible answer at the provider's default, which is exactly the arm-to-arm drift the sweep's
    own guard refuses to start with. So each probe sends a value only the far end can refuse: being
    refused is the pass, and answering is the failure."""
    factory = factory or MeteredLiteLLMEngine.from_profile
    served_snapshot = served_snapshot or _served_snapshot
    failures: list[str] = []
    log(f"preflight {profile.name} -> {profile.provider}/{profile.model} at {profile.endpoint}")

    collected = _Collected()
    engine = factory(profile, writer=collected)
    started = time.perf_counter()
    try:
        text, _ = engine.chat_completion(list(PREFLIGHT_MESSAGES))
    except Exception as exc:  # noqa: BLE001 — every refusal is reported, none is fatal here
        log(f"  REFUSED route: {type(exc).__name__}: {str(exc)[:200]}")
        # Nothing below can be interpreted without a route, so this is the whole answer.
        return 1
    row = collected.rows[-1] if collected.rows else None
    seconds = time.perf_counter() - started
    log(
        f"  route OK in {seconds:.1f}s: text={text[:40]!r} "
        f"in={getattr(row, 'input_tokens', None)} out={getattr(row, 'output_tokens', None)} "
        f"reasoning={getattr(row, 'reasoning_tokens', None)} "
        f"usage_captured={getattr(row, 'usage_captured', None)}"
    )
    request = profile.request_kwargs()
    if not (text or "").strip():
        failures.append("route: answered with no text")
    if row is None or not row.usage_captured:
        failures.append(
            "route: no usage reported, so every row of this arm would bill the fixed term alone"
        )
    elif not (row.output_tokens or 0):
        failures.append("route: accepted, no output tokens reported")
    elif _asks_for_reasoning(request) and not (row.reasoning_tokens or 0):
        # The probes below can show that a setting reached the provider; only this shows the
        # provider acted on it. It is the one honoring question a `reasoning: {enabled: true}`
        # block can be asked, since that form declares no level to compare against. Both shipped
        # endpoints itemize reasoning tokens today, so a zero here means one of two things and the
        # next step differs: the endpoint stopped reasoning, or a repin landed somewhere that does
        # not report it separately. Check the raw usage block before believing either.
        failures.append(
            "route: reasoning requested, zero reasoning tokens reported — "
            "either the endpoint ignored it or it does not itemize it"
        )

    for setting, override in WIRE_PROBES.items():
        if setting not in request:
            continue
        probe = factory(profile)
        probe.settings = replace(
            probe.settings, request_kwargs={**probe.settings.request_kwargs, **override}
        )
        try:
            probe.chat_completion(list(PREFLIGHT_MESSAGES))
        except Exception as exc:  # noqa: BLE001 — the refusal is the result being measured
            crossed, detail = _provider_refused(exc)
            log(f"  {'wire OK ' if crossed else 'DROPPED '}{setting}: {detail}")
            if not crossed:
                failures.append(f"{setting}: {detail}")
        else:
            log(f"  DROPPED {setting}: a value the provider must refuse was answered instead")
            failures.append(
                f"{setting}: dropped before the wire — the arm runs at the provider's default"
            )
    expected_snapshot = MODEL_SNAPSHOTS.get(profile.model)
    if expected_snapshot is not None:
        served = served_snapshot(profile)
        if served is None:
            log(f"  model snapshot UNVERIFIED: {profile.model} (catalogue unreadable)")
        elif served != expected_snapshot:
            log(f"  model snapshot MOVED: {profile.model} now serves {served}")
            failures.append(
                f"model: {profile.model} is an alias and now resolves to {served}, not "
                f"{expected_snapshot} — every latency number measured under it was measured on a "
                f"different model"
            )
        else:
            log(f"  model snapshot OK: {profile.model} -> {served}")

    unprobed = sorted(set(request) - set(WIRE_PROBES))
    if unprobed:
        log(
            "  unprobed settings (sent, no observed refusal to test them with): "
            + ", ".join(unprobed)
        )

    for failure in failures:
        log(f"  FAIL {failure}")
    log(f"preflight {'FAILED' if failures else 'passed'} for {profile.name}")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario",
        default=None,
        help="path to a Gaia2 scenario JSON; not needed with --preflight",
    )
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
    # Defaulted per clock mode rather than pinned here, so this CLI cannot offer
    # --generation-free while keeping a cap that truncates the arm the raise is for.
    parser.add_argument("--max-wall-seconds", type=float, default=None)
    parser.add_argument(
        "--wall-clock",
        action="store_true",
        help=(
            "Use measured model latency instead of the frozen token charge. This robustness "
            "mode is not timing-comparable to token-charged runs."
        ),
    )
    parser.add_argument(
        "--generation-free",
        action="store_true",
        help=(
            "Freeze the scenario around every model crossing and resume it by zero, so "
            "generation costs the environment nothing. The legacy ARE in-process convention. "
            "Excludes --wall-clock, and is not timing-comparable to token-charged runs."
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="confirm the profile's route resolves and its settings reach the provider, then exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profiles = load_profiles(args.profiles_path)
    if args.profile not in profiles:
        raise SystemExit(f"unknown profile {args.profile!r}; have {sorted(profiles)}")
    profile = profiles[args.profile]
    if args.preflight:
        return preflight(profile)
    if not args.scenario:
        raise SystemExit("--scenario is required unless --preflight is given")
    print(f"profile {profile.name} -> {profile.model} at {profile.endpoint}  (arm: react)")
    writer = LLMCallWriter(args.llm_calls) if args.llm_calls else None
    from examples.gaia2.evaluation.core import ChargeModelSheet

    charge_sheet = ChargeModelSheet.load(EVAL_ROOT / "charge_model.json")
    profile_charge = charge_sheet.charge_for(profile)
    if args.wall_clock and args.generation_free:
        raise SystemExit("--wall-clock and --generation-free are different clocks; pick one")
    if args.max_wall_seconds is None:
        args.max_wall_seconds = resolve_max_wall_seconds(None, args.generation_free)
        print(
            f"watchdog: --max-wall-seconds not given; using {args.max_wall_seconds:g}s "
            f"({'generation-free' if args.generation_free else 'default'})"
        )
    charge = None if (args.wall_clock or args.generation_free) else profile_charge
    try:
        result = run_react_scenario(
            args.scenario,
            profile,
            writer=writer,
            charge=charge,
            generation_free=args.generation_free,
            run_number=args.run_number,
            judge_model=args.judge_model,
            judge_provider=args.judge_provider,
            judge_endpoint=args.judge_endpoint,
            scenario_duration=args.scenario_duration,
            output_dir=args.output_dir,
            export=args.export,
            max_wall_seconds=args.max_wall_seconds,
            charge_model_identity=profile_charge.identity,
            charge_model_digest=charge_sheet.digest,
        )
    finally:
        if writer is not None:
            writer.close()
            print(f"    wrote {writer.written} model calls to {writer.path}")
    if result.charge_accounting_consistent is False:
        print(
            "warning: charge clamp count disagrees with independently counted raw usage "
            f"anomalies ({result.cached_input_clamps} != "
            f"{result.raw_cached_input_anomalies})"
        )
    success = result.outcome.success
    verdict = "PASS" if success else "FAIL" if success is False else "UNSCORED"
    print(f"{verdict}: {result.outcome.rationale or result.exception or ''}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
