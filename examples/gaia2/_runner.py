"""Shared per-scenario run core for the Gaia2 drivers.

Both the single-scenario correctness gate (``run_benchmark.py``) and the batch harness
(``batch.py``) need the same thing: take one already-loaded ARE scenario (judge attached if
scoring), run S-ORA through its final reply, and score it. That logic — the turn-aware ``stop_when``
predicate especially — lives here once so the two entry points can't drift.

ARE (and the LLM client) are optional dependency groups, so every import of them is lazy, done
inside the functions rather than at module top; the pure ``dataclass`` below is importable without
them, which keeps ``batch.py``'s formatting/aggregation helpers unit-testable without the ``are``
extra installed.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from examples.gaia2.llm_calls import (
    Charge,
    ClockedSoraLLMClient,
    LLMCallWriter,
    SoraCallRecorder,
)

log = logging.getLogger(__name__)

# How long ARE may hold the timeline paused before the run is treated as stalled. The pause bracket
# around a per-turn judge call is the only thing that pauses a benchmark run, and a judge that is
# answering at all answers in seconds; three minutes is slack for a slow endpoint, not for a call
# that will never return. Deliberately not a CLI flag: it separates "the judge died" from "this
# scenario is slow", which is a property of the judge, not of the run being scored.
MAX_PAUSE_SECONDS = 180.0


@dataclass
class RunResult:
    """One scenario run's outcome. ``environment`` is the live ARE ``Environment`` (handed to the
    trace exporter); ``exception`` is set when the run *or* ``validate()`` raised, so a batch can
    record an ``exception`` result for this one scenario and carry on instead of aborting the sweep.
    ``outcome`` is a ``sora.adapters.are_sim.ValidationOutcome`` (``success=None`` when unscored or
    when the run failed before scoring). ``awaiting_input`` holds the prompts of any activity the
    run ended on a question from — empty for every ordinary run. ``write_counts`` is a
    ``sora.adapters.are_sim.WriteCountCheck`` when an oracle log was available (None otherwise) —
    ARE's tool-call-count gate recomputed offline, which costs no model tokens and so is filled in
    for *unscored* runs too. ``timeline_expired`` is True when ARE's event loop ran out of
    ``scenario.duration`` — a real-time budget, since the loop is wall-clock paced — before the run
    finished; every result below it then describes a world that stopped early, not an agent that
    chose badly. ``judge_recording`` is a ``sora.adapters.are_judge.JudgeRecording`` when the run
    asked for one — the judge's raw answers, which ARE itself discards, and the only thing that
    makes the run re-scorable after the fact."""

    outcome: Any
    environment: Any
    duration: float
    exception: Exception | None = None
    awaiting_input: list[str] = field(default_factory=list)
    write_counts: Any = None
    timeline_expired: bool = False
    llm_report: Any = None
    replan_count: int = 0
    terminal_cause: str | None = None
    agent_llm_calls: int = 0
    external_actions: int = 0
    decision_cycles: int = 0
    # `$prop`-satisfied reads: how many times a reference resolved off the observed property
    # snapshot, and over how many distinct properties. Each is a collection read that cost no model
    # call and no tool call, where ARE's own step-loop agent — which never sees app state — has to
    # invoke an operation and feed the result back through the model. Reported so the cycle's call
    # saving can be split between plan amortization and this, instead of being credited wholly to
    # the first. `prop_reads_by_property` keeps the per-property breakdown, since the distinct-key
    # count is the one comparable to a tool-call count.
    prop_reads: int = 0
    prop_reads_by_property: dict[str, int] = field(default_factory=dict)
    judge_recording: Any = None
    charged_seconds: float = 0.0
    charge_model_identity: dict[str, Any] | None = None
    charge_model_digest: str | None = None
    cached_input_clamps: int = 0
    raw_cached_input_anomalies: int = 0
    charge_accounting_consistent: bool | None = None
    clock_mode: str = "wall"
    inference_charge_policy: str | None = None
    llm_wall_seconds: float = 0.0
    llm_wall_union_seconds: float = 0.0
    # None when this run's crossings did not share one time axis, so the parallel-charge
    # counterfactual is unavailable rather than zero.
    llm_charged_union_seconds: float | None = 0.0
    llm_round_trips: int = 0
    llm_max_in_flight: int = 0
    llm_overlapped_round_trips: int = 0


StopReason = Literal["verification_completion", "llm_call_limit", "timeout"]


@dataclass
class _StopController:
    simulation: Any
    agent: Any
    deadline: float
    paused_since: float | None = None
    reason: StopReason | None = None

    def __call__(self) -> bool:
        now = time.monotonic()
        if self.agent.procedural.logical_call_limit_exceeded:
            self.reason = "llm_call_limit"
            return True
        if now >= self.deadline:
            self.reason = "timeout"
            return True
        judge_pause_probe = getattr(self.simulation, "is_judge_paused", None)
        judge_paused = (
            bool(judge_pause_probe())
            if callable(judge_pause_probe)
            else bool(self.simulation.is_paused())
        )
        if judge_paused:
            self.paused_since = now if self.paused_since is None else self.paused_since
            if now - self.paused_since >= MAX_PAUSE_SECONDS:
                self.reason = "timeout"
                return True
            return False
        self.paused_since = None
        from sora.activity import ActivityState
        from sora.types import ConditionWait, InputWait

        activities = list(self.agent.working.activities.values())
        if not activities:
            return False
        done = all(
            activity.state is ActivityState.TERMINATED
            or (
                activity.state is ActivityState.BLOCKED
                and isinstance(activity.blocked_on, InputWait | ConditionWait)
            )
            for activity in activities
        )
        final_reply_probe = getattr(self.simulation, "all_turns_answered", None)
        final_reply_sent = bool(final_reply_probe()) if callable(final_reply_probe) else False
        if self.simulation.is_running() and not (done and final_reply_sent):
            return False
        if done:
            self.reason = "verification_completion"
        return done


def _awaiting_input(agent: Any) -> list[str]:
    """The await-input prompts of every activity currently asking a question — the replan breaker,
    the sub-goal recursion breaker, or a user stop, all of which park on ``InputWait``. Read as a
    list rather than a bool because the prompt is the whole value: it names the specific defects
    that led there, which is what tells a swept run apart from one that merely scored badly."""
    from sora.activity import ActivityState
    from sora.types import InputWait

    return [
        a.blocked_on.prompt or ""
        for a in agent.working.activities.values()
        if a.state is ActivityState.BLOCKED and isinstance(a.blocked_on, InputWait)
    ]


def _run_number_of(scenario: Any, run_number: int | None) -> int | None:
    """Which run a judge recording belongs to, preferring the number the caller asked for.

    The batch harness sets both and they agree; a direct caller passes only the argument, and a
    recording filed under a stale scenario attribute joins to the wrong run's model-call rows —
    or to none at all. Both arms resolve it the same way, since rescoring reads them side by
    side."""
    if run_number is not None:
        return run_number
    value = getattr(scenario, "run_number", None)
    return int(value) if isinstance(value, int) else None


def _timeline_expired(simulation: Any) -> bool:
    """Whether ARE's own clock, not the agent, ended the run. Tolerates a simulation that predates
    the probe (a fake in a test) rather than requiring it on the ``Simulation`` Protocol."""
    probe = getattr(simulation, "timeline_expired", None)
    if probe is None:
        return False
    try:
        return bool(probe())
    except Exception:  # a diagnostic must never cost the run its real result
        log.warning("timeline-expiry probe failed", exc_info=True)
        return False


def _make_stop_when(
    simulation: Any,
    agent: Any,
    exit_when_idle: float | None,
    deadline: float,
) -> _StopController | None:
    """The turn-aware done predicate (see ``run_benchmark.py``'s module docstring). Returns None
    when the caller opts into ``TerminalSession``'s own quiet-window heuristic (``exit_when_idle``
    set), letting the session drive its old single-turn behavior unchanged.

    The deadline is passed in rather than computed here because it has to govern scoring as well,
    and this predicate is only polled while the decision cycle is running."""
    if exit_when_idle is not None:
        return None
    return _StopController(simulation=simulation, agent=agent, deadline=deadline)


@contextlib.contextmanager
def _wall_deadline(simulation: Any, deadline: float) -> Iterator[Callable[[], bool]]:
    """Enforce one wall deadline across everything inside the block, not just the agent phase.

    Both arms must be capped over the same phases or the cap is not a shared condition: ARE's
    ``ScenarioRunner`` performs ``validate()`` inside the call the ReAct watchdog wraps, while
    here the agent loop and scoring are separate statements, and ``_StopController`` is only
    consulted from inside the decision cycle.  A judge pass can take minutes, so without this the
    S-ORA arm is effectively uncapped over exactly the phase the other arm caps.

    The lever is the one ARE gives either arm — stopping the environment — so a judge blocked on
    the simulation is released the same way.  Artifact serialization stays outside the block: it
    is bookkeeping, and killing it would lose the run's record rather than bound its cost."""
    expired = threading.Event()
    finished = threading.Event()

    def watch() -> None:
        while not finished.wait(0.05):
            if time.monotonic() < deadline:
                continue
            expired.set()
            env = None
            try:
                env = simulation.environment()
            except Exception:  # not constructed yet, or already torn down
                env = None
            if env is not None:
                with contextlib.suppress(Exception):
                    env.stop()
                return
            # Deadline reached before the environment exists: there is nothing to stop yet, so
            # keep watching rather than retiring — otherwise a slow construction outlives its own
            # deadline unbounded.  The block's exit releases this thread either way.

    watcher = threading.Thread(target=watch, name="Gaia2SoraWallDeadline", daemon=True)
    watcher.start()
    try:
        yield expired.is_set
    finally:
        finished.set()
        watcher.join(timeout=1.0)


def _context_overflow(exc: Exception | str | None) -> bool:
    if exc is None:
        return False
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("context overflow", "context length", "context window", "maximum context")
    )


def _terminal_cause(
    exc: Exception | None,
    timeline_expired: bool,
    stop_reason: StopReason | None,
    success: bool | None,
    *,
    inference_errors: tuple[str, ...] = (),
    max_iterations_reached: bool = False,
) -> str:
    if _context_overflow(exc) or any(_context_overflow(error) for error in inference_errors):
        return "context_overflow"
    if exc is not None or inference_errors:
        return "infrastructure_error"
    if stop_reason == "llm_call_limit" or max_iterations_reached:
        return "llm_call_limit"
    if timeline_expired or stop_reason == "timeout":
        return "timeout"
    if isinstance(success, bool):
        return "verification_completion"
    return "unscored_completion"


def _terminal_inference_errors(llm_report: Any) -> tuple[str, ...]:
    """Errors in the unresolved suffix of off-cycle inference outcomes.

    A provider failure is delivered through ``InferenceResult`` and therefore never escapes
    ``TerminalSession.run``. A later successful inference means the runtime recovered; only errors
    after the last success describe why the run finally stopped.
    """
    if llm_report is None:
        return ()
    errors: list[str] = []
    for inference in reversed(llm_report.inferences):
        outcome = getattr(inference, "outcome", None)
        if outcome == "success":
            break
        error = getattr(inference, "error", None)
        if outcome == "error" and isinstance(error, str):
            errors.append(error)
    return tuple(reversed(errors))


@contextlib.contextmanager
def _recording_llm_calls(
    recorder: SoraCallRecorder,
) -> Iterator[None]:
    """Attach the per-call recorder for one run and detach it again — a sweep runs many scenarios
    in one process, and a handler left on the logger would keep attributing later runs to this
    scenario's id."""
    sora_log = logging.getLogger("sora")
    # The session raises this to DEBUG for its own presenter, so in practice it is already open —
    # but a level left at the root default would drop every per-call record *before* any handler
    # saw it, and a silently empty file is the one failure this artifact cannot afford.
    previous_level = sora_log.level
    if not sora_log.isEnabledFor(logging.INFO):
        sora_log.setLevel(logging.INFO)
    sora_log.addHandler(recorder)
    try:
        yield
    finally:
        sora_log.removeHandler(recorder)
        sora_log.setLevel(previous_level)
        # Closing is what writes the rows: a logical call can span more than one round trip
        # (parser repair) and nothing in the stream marks the last one, so the recorder can only
        # settle its rows once the scenario's stream has ended. In a `finally` because an aborted
        # scenario still paid for the calls it made.
        recorder.close()


# Keys whose *value* would be a live credential if someone inlined one instead of using
# `api_key_env`. The echo exists to be pasted into issues and diffed across runs, so it must not be
# the thing that leaks a key out of a config file.
_SECRET_KEYS = ("api_key", "token", "secret", "password")


def _config_echo(
    config: str,
    *,
    charge: Charge | None,
    charge_model_identity: dict[str, Any] | None,
    charge_model_digest: str | None,
    max_wall_seconds: float,
    scenario_id: str | None,
) -> str:
    """The run's own provenance, written at the top of its ``--log-file``.

    A trace records what the agent did, never what it was configured to do, and the two are not
    recoverable from each other: a strategy setting that changes which cycles spend a model call
    leaves behind only its consequences. Nor does git close the gap — a config is routinely edited
    locally and never committed, so a run's effective settings can exist nowhere but the process
    that has already exited. Echoing the file verbatim (not a parsed summary) keeps comments and
    commented-out alternatives, which is usually where the reason for a setting lives.
    """
    path = Path(config)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:  # unreadable config is a run-level problem, not a logging one
        raw = f"(could not read: {exc})"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    redacted = []
    for line in raw.splitlines():
        head, sep, value = line.partition(":")
        key = head.strip().lstrip("- ").strip()
        if sep and value.strip() and key in _SECRET_KEYS:
            redacted.append(f"{head}: (redacted)")
        else:
            redacted.append(line)
    lines = [
        "=== run provenance " + "=" * 60,
        f"scenario id      : {scenario_id or '(unknown)'}",
        f"config path      : {path.resolve()}",
        f"config sha256    : {digest}",
        f"max wall seconds : {max_wall_seconds:g}",
    ]
    if charge is None:
        lines.append("scenario clock   : wall (robustness mode; not token-charged)")
    else:
        lines.append("scenario clock   : charged (simulated time frozen across model calls)")
        lines.append(f"charge digest    : {charge_model_digest or '(none)'}")
        if charge_model_identity:
            identity = ", ".join(f"{k}={v}" for k, v in sorted(charge_model_identity.items()))
            lines.append(f"charge identity  : {identity}")
    lines.append("--- agent config (verbatim) " + "-" * 52)
    lines.extend(redacted)
    lines.append("=" * 79)
    lines.append("")
    return "\n".join(lines)


def run_scenario(
    scenario: Any,
    *,
    config: str,
    verbose: bool = False,
    log_file: str | None = None,
    max_wall_seconds: float = 1200.0,
    exit_when_idle: float | None = None,
    read_stdin: bool = True,
    record_judge: bool = False,
    verdict_parse: str | None = None,
    llm_calls: LLMCallWriter | None = None,
    scenario_id: str | None = None,
    run_number: int | None = None,
    charge: Charge | None = None,
    charge_model_identity: dict[str, Any] | None = None,
    charge_model_digest: str | None = None,
) -> RunResult:
    """Run S-ORA against one loaded scenario to completion, then score it. Attach the judge (via
    ``are_sim.attach_judge``) *before* calling this if a real score is wanted; without it the run is
    unscored (``outcome.success is None``). A run-time crash or a ``validate()`` error is captured
    on ``RunResult.exception`` rather than raised, so a batch loop can record it and move on;
    ``KeyboardInterrupt`` still propagates so an operator can abort.

    ``record_judge`` collects the judge's raw answers into ``RunResult.judge_recording`` (arm the
    patch with ``are_judge.arm_judge_recording`` first). The bracket spans the run *and*
    ``validate()``, not just the latter: under online validation the judge is also each turn's
    release gate, so on a multi-turn scenario most judged events are decided mid-run.
    ``verdict_parse`` names how those verdicts were read, and is stored with the recording — a
    recording that does not say which parse produced its verdicts cannot be checked against the run
    it came from.

    ``llm_calls``, when given, receives one row per model call (``examples.gaia2.llm_calls``) —
    tokens, cache reads and measured latency, which the charge model is fitted and validated
    against. It is a separate channel from ``llm_report``: that one summarizes this run, this one
    is the per-call record a later fit reads, and it is written for the *unscored* runs too."""
    from examples.gaia2.simulated_clock import ChargedEnvironment
    from sora.adapters.are_sim import AreSimulation, ValidationOutcome, write_count_check
    from sora.bootstrap import build_agent
    from sora.cli import TerminalSession

    simulation = AreSimulation(scenario, environment_factory=ChargedEnvironment)
    agent = build_agent(config, simulation=simulation)
    # One instant, shared by the in-cycle predicate and the scoring-phase watchdog below, so the
    # cap covers the same phases the ReAct arm's watchdog covers.
    wall_deadline = time.monotonic() + max_wall_seconds
    stop_when = _make_stop_when(
        simulation,
        agent,
        exit_when_idle,
        wall_deadline,
    )

    session = TerminalSession(
        agent,
        verbose=verbose,
        initial_task=None,  # the Gaia2 scenario delivers its own task via the AUI timeline
        exit_when_idle=exit_when_idle,
        stop_when=stop_when,
        read_stdin=read_stdin,
        log_file=log_file,
        log_preamble=_config_echo(
            config,
            charge=charge,
            charge_model_identity=charge_model_identity,
            charge_model_digest=charge_model_digest,
            max_wall_seconds=max_wall_seconds,
            scenario_id=scenario_id,
        ),
    )

    # The bracket spans the run AND validate(): under online validation the judge is also each
    # turn's release gate, so on a multi-turn scenario most judged events are decided mid-run, and
    # a bracket around validate() alone would record only the last one. Nothing here is entered
    # when `record_judge` is off, so an unrecorded run pays nothing.
    recorder: SoraCallRecorder | None = None
    original_llm: Any = None
    with contextlib.ExitStack() as bracket:
        # On the `sora` logger, the same stream `LLMMeter` and the CLI presenter read; the session
        # attaches its own handlers independently. Keep the recorder even without a writer or a
        # charge model: wall-clock robustness runs still need their overlap sensitivity in the
        # result, and collecting it should not require a JSONL destination.
        recorder = SoraCallRecorder(
            llm_calls,
            model=getattr(agent.procedural, "model", None),
            scenario_id=scenario_id,
            run_number=run_number,
            charge=charge,
        )
        bracket.enter_context(_recording_llm_calls(recorder))
        # Harness-only decoration: bootstrap still owns construction, while the benchmark times
        # the completed client for this run alone. Every current agent-model entry point — the five
        # activity actions plus off-cycle relevance and retirement judgement — resolves through
        # this one ProceduralMemory client. That single seam is what makes the recorder complete;
        # a future strategy that owns a separate client must be instrumented here too. With a
        # charge model, the same wrapper freezes the scenario; in wall-clock robustness mode it
        # only records windows.
        original_llm = agent.procedural._llm
        if original_llm is None:
            if charge is not None:
                raise ValueError("a charged Gaia2 run requires an LLM client")
        else:
            agent.procedural._llm = ClockedSoraLLMClient(
                original_llm,
                clock=simulation if charge is not None else None,
                recorder=recorder,
            )
            bracket.callback(setattr, agent.procedural, "_llm", original_llm)
        collector: Any = None
        if record_judge:
            from sora.adapters.are_judge import record_judge_events

            collector = bracket.enter_context(record_judge_events())

        wall_expired = bracket.enter_context(_wall_deadline(simulation, wall_deadline))

        exc: Exception | None = None
        started = time.monotonic()
        try:
            asyncio.run(session.run())
        except Exception as e:  # a run-time crash: record it, sweep continues (KI still propagates)
            exc = e
        duration = time.monotonic() - started
        # Sampled here, immediately after the session's teardown, rather than in the
        # RunResult below: everything between is scoring work (a judge pass over the oracle
        # graph, which can take minutes), and this reads a wall clock. `AreSimulation`
        # latches the verdict at stop() so the position no longer matters for the real
        # adapter, but a fake or a future Simulation that does not latch still gets a value
        # measured against the run rather than against the judge.
        expired = _timeline_expired(simulation)

        outcome: Any = ValidationOutcome(success=None)
        if exc is None and getattr(scenario, "judge", None) is not None:
            # Only trust validate() when a scoring judge is actually attached (attach_judge
            # → ARE's preprocess_scenario sets scenario.judge). Without one, the scenario
            # falls back to ARE's base Scenario.validate, which returns ``env.state !=
            # FAILED`` — a spurious True for any run that merely didn't crash, meaningless
            # as a score. So a judge-less run stays unscored (success=None) rather than
            # reporting a false PASS.
            try:
                outcome = simulation.validate()
            # A judge/oracle failure surfaces here, not as a silent unscored run.
            except Exception as e:
                exc = e

        # Deliberately outside the judge guard and after validate(): it needs no judge and no
        # tokens, so an unscored dev run — where it is the only pass/fail signal there is —
        # gets it too. Never lets a reporting failure cost the run's real result.
        counts: Any = None
        try:
            counts = write_count_check(scenario, simulation.environment())
        except Exception:  # a diagnostic must never cost the run its real result
            log.warning("write-count check failed", exc_info=True)

        # A deadline that fired during scoring is still a timeout: the run has no result it
        # could have reached, and recording it as anything else would hide a capped judge pass.
        stop_reason: StopReason | None = stop_when.reason if stop_when is not None else None
        if stop_reason is None and wall_expired():
            stop_reason = "timeout"

        terminal_cause = _terminal_cause(
            exc,
            expired,
            stop_reason,
            outcome.success if isinstance(outcome.success, bool) else None,
            inference_errors=_terminal_inference_errors(session.llm_report),
        )

        return RunResult(
            outcome=outcome,
            environment=simulation.environment(),
            duration=duration,
            exception=exc,
            # Read after the session returns, so this reflects where the run actually
            # stopped. Not an error: the agent halting to ask rather than looping is the
            # designed behavior, and the scenario is still scored normally — this only
            # records *why* it stopped short.
            awaiting_input=_awaiting_input(agent),
            write_counts=counts,
            # Sampled above, right after teardown; never allowed to raise, because it
            # reinterprets every field beside it and losing it to a probe failure would be
            # worse than losing any single one of them.
            timeline_expired=expired,
            llm_report=session.llm_report,
            replan_count=sum(
                getattr(activity, "replan_count", 0)
                for activity in agent.working.activities.values()
            ),
            terminal_cause=terminal_cause,
            agent_llm_calls=agent.procedural.logical_calls_admitted,
            external_actions=agent.cycle.external_action_count,
            decision_cycles=agent.cycle.cycle_count,
            prop_reads=sum(agent.working.prop_reads.values()),
            prop_reads_by_property={
                f"{source}.{name}": count
                for (source, name), count in sorted(agent.working.prop_reads.items())
            },
            # Taken inside the bracket, after validate(), so it holds every event the judge decided
            # — the mid-run turn gates as well as the final pass. None when recording was off.
            judge_recording=(
                None
                if collector is None
                else collector.snapshot(
                    scenario_id=getattr(scenario, "scenario_id", None),
                    run_number=_run_number_of(scenario, run_number),
                    verdict_parse=verdict_parse,
                )
            ),
            charged_seconds=recorder.charged_seconds if recorder is not None else 0.0,
            charge_model_identity=charge_model_identity,
            charge_model_digest=charge_model_digest,
            cached_input_clamps=int(getattr(charge, "cached_input_clamps", 0)),
            raw_cached_input_anomalies=(
                recorder.raw_cached_input_anomalies if recorder is not None else 0
            ),
            charge_accounting_consistent=(
                None
                if charge is None
                else int(getattr(charge, "cached_input_clamps", 0))
                == (recorder.raw_cached_input_anomalies if recorder is not None else 0)
            ),
            clock_mode="token_charged" if charge is not None else "wall",
            inference_charge_policy="serialized_sum" if charge is not None else None,
            llm_wall_seconds=recorder.wall_seconds if recorder is not None else 0.0,
            llm_wall_union_seconds=(recorder.wall_union_seconds if recorder is not None else 0.0),
            llm_charged_union_seconds=(
                recorder.charged_union_seconds if recorder is not None else 0.0
            ),
            llm_round_trips=(recorder.concurrency.round_trips if recorder is not None else 0),
            llm_max_in_flight=(recorder.concurrency.max_in_flight if recorder is not None else 0),
            llm_overlapped_round_trips=(
                recorder.concurrency.overlapped_round_trips if recorder is not None else 0
            ),
        )
