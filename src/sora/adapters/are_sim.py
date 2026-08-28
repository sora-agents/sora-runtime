"""In-process ARE ``WorkspaceAdapter`` + ``MessageTransport`` — talk to live ARE apps directly.

Unlike ``are_mcp`` (which reaches ARE over an MCP subprocess serving a *static* app snapshot), this
runs the ARE ``Environment`` **event loop** in the same process (a bg thread) so a scenario's
timeline actually fires — mid-run email injections, follow-up user messages, task delivery — and
bridges the two off-cycle event channels into S-ORA directly, as method calls on shared objects:

  * app state changes  -> ``state_changed`` Signal into the focused tool's ``signal_sink``
    (poll-on-observe: the tool re-reads ``app.get_state()`` each Observe and diffs — see
    ``_AreTool.observe``).  This is what MCP could not push off-request (ARE's MCP server only emits
    ``resource_updated`` from inside a write-tool request), so we go in-process instead.
  * ``AgentUserInterface`` USER messages  -> ``MessageTransport`` (``AreTransport`` over the AUI).

``AreSimulation`` owns the ``Environment``/scenario lifecycle and is the single object both seams
share (see the new ADR). The adapter's *workspace* owns start/stop (start on ``discover``, stop on
``close``), exactly as ``_McpWorkspace`` owns its subprocess. ``ARE`` (``are.simulation.*``) is an
optional dependency-group, so every import of it is lazy; the adapter/transport depend only
on a small duck-typed app/AUI interface (``app_name``/``get_tools``/``get_state`` and AUI
``get_last_unread_messages``/``send_message_to_user``/``send_message_to_agent``), which fakes
satisfy, so S-ORA-side logic stays testable without ARE (see ADR-0003).
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import logging
import threading
import time
import typing
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, Union

from sora.manual import (
    Manual,
    ObservablePropertySpecification,
    OperationSpecification,
    SignalSpecification,
    merge_manuals,
)
from sora.perception import Message
from sora.types import ObservableProperty, OperationAck, Signal, diff_values

if TYPE_CHECKING:
    from sora.environment import DomainClock, Tool, Workspace, WorkspaceOrigin
    from sora.manual import ManualSource, ToolRecord, WorkspaceRecord

_T = TypeVar("_T")

log = logging.getLogger("sora.adapters.are_sim")

_AUI_APP = "AgentUserInterface"  # ARE's user-message app; routed via the transport, not as a tool

# ARE mutates app state on its own event-loop thread with no lock we can share (see AreSimulation),
# so a ``get_state()`` that iterates a dict the event loop is concurrently growing can raise
# "changed size during iteration". Mutation happens in sub-second bursts, so an immediate re-read
# sees a settled snapshot — retry a few times before giving up.
_STATE_READ_ATTEMPTS = 3


class Simulation(Protocol):
    """The runtime surface the adapter/transport use, decoupled from ARE so fakes satisfy it. The
    concrete ``AreSimulation`` implements it over a live ARE ``Environment``; a test fake implements
    it over plain app/AUI objects."""

    @property
    def aui(self) -> Any:  # the live AgentUserInterface app, or None
        ...

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def is_running(self) -> bool: ...  # scenario timeline still advancing (the eval done-signal)
    # DOMAIN time — what the SCENARIO says it is, which is not what `time.time()` says (ADR-0027
    # §5). It belongs on this Protocol rather than behind `environment()` because the workspace's
    # `DomainClock` is part of the runtime surface, and a fake simulation has to be able to supply
    # one.
    def now(self) -> datetime: ...
    def apps(self) -> list[Any]: ...
    def run(self, fn: Callable[[], _T]) -> _T: ...  # serialize S-ORA's own concurrent app calls


@dataclass(frozen=True)
class ValidationOutcome:
    """``AreSimulation.validate()``'s result — decouples callers (``report.py``) from ARE's own
    ``ScenarioValidationResult`` shape, so a fake simulation in a test can produce one without
    depending on ARE. ``rationale`` is None when the scenario's ``validate()`` didn't supply one."""

    success: bool | None  # None = unscored: ARE's judge produced no verdict (a caller with no judge
    # attached decides that upstream — ARE's *base* validate() returns a bool, not None)
    rationale: str | None = None


class AreSimulation:
    """Owns the ARE ``Environment`` + scenario lifecycle — the shared object the in-process adapter
    and transport both reference. ``start`` runs the scenario's event loop on a background thread
    (``env.run(..., wait_for_end=False)``). The ``Lock`` serializes S-ORA's *own* concurrent app
    calls (e.g. an ``invoke`` on a worker thread vs an ``observe`` on the cycle thread); it does
    **not** — and cannot — serialize against ARE's event-loop thread, which mutates app state with
    no lock we can share, so reads tolerate a transient concurrent-modification error by retry (see
    ``_AreTool._read_state`` / ``_STATE_READ_ATTEMPTS``). The agent replies without blocking
    (``aui.wait_for_user_response = False``) — a follow-up user message arrives via the timeline and
    is picked up by ``AreTransport.receive``."""

    def __init__(self, scenario: Any, *, config: Any | None = None) -> None:
        self._scenario = scenario
        self._config = config
        self._env: Any | None = None
        self._lock = threading.Lock()
        self._started = False
        # Latched at stop(): ARE's clock keeps running after the environment is torn down, so the
        # expiry verdict has to be taken at the moment the run ended, not whenever it is read.
        self._expired: bool | None = None

    def start(self) -> None:
        if self._started:
            return
        from are.simulation.environment import Environment, EnvironmentConfig

        if not getattr(self._scenario, "_initialized", False):
            self._scenario.initialize()
        # `Environment.run` copies `duration` and `time_increment_in_seconds` off the scenario,
        # but NOT `start_time` — that is read from the config alone
        # (`time_manager.reset(start_time=self.start_time)`), and `EnvironmentConfig` defaults it to
        # None, which the Environment reads as 0. A scenario therefore runs with its simulated clock
        # at the Unix epoch, counting real seconds up from 1970-01-01, while its data and its oracle
        # sit in the scenario's own year: `get_current_time` told an agent it was Thursday
        # 1 Jan 1970 for a scenario starting Tuesday 2024-10-15, so every date the agent derived
        # ("this upcoming Saturday") was computed against the wrong epoch — a silent wrong answer,
        # not an error. ARE's own ScenarioRunner sets it from the scenario; mirror that.
        # An explicit start_time on a caller-supplied config wins, and the config is copied rather
        # than mutated because it belongs to the caller.
        config = self._config or EnvironmentConfig()
        start_time = getattr(self._scenario, "start_time", None)
        if start_time and config.start_time is None:
            config = dataclasses.replace(config, start_time=start_time)
        self._env = Environment(config=config)
        # wait_for_end=False: registers apps, schedules the timeline, starts the event-loop thread,
        # and returns — the agent then drives its cycle against the live, ticking world.
        self._env.run(self._scenario, wait_for_end=False)
        aui = self.aui
        if aui is not None:
            aui.wait_for_user_response = False
        self._started = True
        self._expired = None

    def stop(self) -> None:
        if self._env is not None and self._started:
            # Latch the expiry verdict *before* tearing the environment down. ARE's clock is a wall
            # clock that nothing pauses -- `Environment.stop()` sets the stop event and the state
            # but leaves `TimeManager` running -- so `time_passed()` keeps advancing afterwards. A
            # verdict computed at read time would therefore drift True on any run whose result is
            # read slowly enough, and the caller reads it after `validate()`, whose judge pass can
            # itself take minutes. Sampling here makes the answer independent of when it is asked.
            self._expired = self._probe_expired()
            self._env.stop()
        self._started = False

    def is_running(self) -> bool:
        """True while the scenario still has timeline left to play — ARE ``RUNNING`` or ``PAUSED``.
        Flips to False only when the timeline completes (``scenario.duration`` reached) or fails:
        ARE's ``_event_loop`` sets ``STOPPED``/``FAILED`` on loop exit, *after* every scheduled user
        turn and the per-turn judge ``ConditionCheckEvent``s have fired. So an eval runner that
        waits for this to go False before calling ``validate()`` scores a fully-played-out scenario,
        riding through the idle gaps between turns that a quiet-window heuristic would exit on
        prematurely.

        ``PAUSED`` counts as live, which is *not* what ARE's own ``Environment.is_running()`` says
        (that one is ``state == RUNNING`` exactly). ARE's per-turn gate wraps the judge in
        ``env.pause()`` / ``env.resume()`` (``scenarios/utils/turn_conditions.py``), so with a
        model-backed judge attached the environment sits in ``PAUSED`` for however long that call
        takes. Delegating raw would report "not running" for that whole window, and a runner polling
        this alongside "all activities blocked" would tear the run down mid-judgement — losing every
        turn after the first. ARE documents ``PAUSED`` as "can be restarted"; it is mid-flight, not
        finished. A judge that actually rejects a turn calls ``env.stop()``, so the run still ends
        promptly on a genuine failure."""
        if self._env is None or not self._started:
            return False
        from are.simulation.types import EnvironmentState

        return self._env.state in (EnvironmentState.RUNNING, EnvironmentState.PAUSED)

    def is_paused(self) -> bool:
        """True while ARE holds the timeline paused — in practice, while a per-turn judge call is
        in flight (``turn_conditions.wrapped_condition`` brackets it in ``pause()``/``resume()``).

        Exposed so eval tooling can bound *that wait specifically* rather than only the whole run:
        a healthy judge answers in seconds, so a pause lasting minutes is a stalled call, and the
        two are indistinguishable through a single wall clock. Note ARE's bracket is not
        exception-safe — ``resume()`` is not in a ``finally`` — so a judge that raises leaves the
        environment paused for good. Like ``environment()``, deliberately not on the ``Simulation``
        Protocol: it says nothing to the adapter/transport, and only concrete eval code reads it."""
        if self._env is None or not self._started:
            return False
        from are.simulation.types import EnvironmentState

        return bool(self._env.state == EnvironmentState.PAUSED)

    def timeline_expired(self) -> bool:
        """True when ARE's event loop exited because ``scenario.duration`` was reached, rather than
        because the scenario played out or something stopped it.

        This distinction is not cosmetic. ARE's loop is **wall-clock paced** — one ``time.sleep(1)``
        per tick, advancing simulated time by ``time_increment_in_seconds`` — so a scenario's
        ``duration`` (1000s by default for a JSON benchmark scenario) is a real-time budget for the
        agent, not a property of the scripted world. A model slow enough to spend that budget on
        inference has the environment die underneath it mid-turn: later turns are never delivered,
        because a Gaia2 turn is released by a ``ConditionCheckEvent`` that only the live event loop
        ticks. The run then presents exactly like a competent agent that chose to do nothing —
        ``validate()`` reports the turn index never advanced, and the write-count gate reports every
        one of the missing turn's oracle calls as missing. Reading that as an agent failure is
        wrong, and nothing else in the result distinguishes the two, which is why this is surfaced
        as its own signal. Like ``is_paused``, deliberately not on the ``Simulation`` Protocol: it
        says nothing to the adapter/transport and only concrete eval code reads it.

        Unlike ``is_running``/``is_paused`` this deliberately does **not** guard on ``_started``.
        Those two report *live* state and are rightly False once the run is over; this is a
        post-mortem, and the only moment anyone asks it is after the run is over. The shipped
        shutdown path clears ``_started`` before any caller gets to read it -- the session's
        teardown leaves every joined workspace, which closes the ARE workspace, which calls
        ``stop()`` -- so guarding here would make the answer unconditionally False in production
        while still reading True in a test that never stopped the simulation."""
        if self._expired is not None:  # latched at stop(); see there
            return self._expired
        return self._probe_expired()

    def _probe_expired(self) -> bool:
        env = self._env
        if env is None:
            return False
        duration = getattr(env, "duration", None)
        if duration is None:  # ARE reads None as "run indefinitely" — nothing to expire
            return False
        try:
            passed = env.time_manager.time_passed()
        except Exception:  # a diagnostic must never cost the run its real result
            return False
        # Mirrors the loop's own exit test (`while time_passed() <= duration`), so this reports the
        # condition ARE actually stopped on rather than an approximation of it.
        return bool(passed > duration)

    def now(self) -> datetime:
        """The scenario's own clock, not the host's (ADR-0027 §5).

        `TimeManager.time()` is `start_time + time_passed()`, and `start()` above is what makes
        `start_time` the scenario's rather than the epoch — so this is the same instant the apps'
        own `get_current_time` reports, reached without spending an external action on an invoke.
        Before `start()` there is no environment and therefore no domain time: host wall-clock is
        the only remaining answer and it is the wrong one, so this raises rather than returning it.

        Epoch seconds in, an aware instant out. ARE keeps time as a bare float, which names an
        instant unambiguously; the timezone attached here is UTC because a rendering has to pick
        one, and comparisons happen on the instant either way."""
        assert self._env is not None, "start() the simulation before reading its clock"
        return datetime.fromtimestamp(self._env.time_manager.time(), tz=UTC)

    def apps(self) -> list[Any]:
        return list(getattr(self._scenario, "apps", None) or [])

    def environment(self) -> Any:
        """The live ARE ``Environment`` (or None before ``start()``). Exposed for eval tooling that
        exports a completed run's trace — ARE's ``JsonScenarioExporter`` needs the Environment. It's
        deliberately *not* on the ``Simulation`` Protocol, which stays the minimal runtime surface
        the adapter/transport share; only concrete eval code touches this."""
        return self._env

    @property
    def aui(self) -> Any:
        return next((a for a in self.apps() if a.app_name() == _AUI_APP), None)

    def run(self, fn: Callable[[], _T]) -> _T:
        # Serializes S-ORA's own concurrent calls only; ARE's event-loop thread does not take this
        # lock (it's ARE-internal), so it does not guard app reads against that thread — see the
        # class docstring and _AreTool._read_state's retry.
        with self._lock:
            return fn()

    def validate(self) -> ValidationOutcome:
        """Oracle scoring: run the scenario's validators against the final environment state."""
        assert self._env is not None, "start() the simulation before validating"
        result = self._scenario.validate(self._env)
        # A judge/validator that *errored* reports success=None with an in-band exception (ARE puts
        # it on the result rather than raising). Re-raise it so a caller's try/except records the
        # run as an 'exception', not a silent unscored one — dropping it here would make a judge
        # crash indistinguishable from 'no judge attached'.
        if result.success is None and result.exception is not None:
            raise result.exception
        # Preserve None (unscored/vacuous) rather than coercing to False — a run with no judge
        # attached is distinct from a genuine FAIL, and the eval reporter surfaces that difference.
        return ValidationOutcome(success=result.success, rationale=result.rationale)


def load_scenario(ref: str) -> Any:
    """Resolve a scenario reference to an ARE ``Scenario`` — the "any ARE scenario" seam. ``ref`` is
    either a ``.json`` benchmark scenario path or a dotted path to a ``Scenario`` subclass (or a
    ready instance). Instances are returned as-is (``AreSimulation.start`` initializes them)."""
    if ref.endswith(".json"):
        from are.simulation.benchmark.scenario_loader import load_scenario as _are_load

        scenario, _ = _are_load(
            Path(ref).read_text(encoding="utf-8"), ref, load_completed_events=False
        )
        if scenario is None:
            raise ValueError(f"failed to load ARE scenario from {ref!r}")
        return scenario
    from sora.bootstrap import import_object  # lazy: avoid a bootstrap<->are_sim import cycle

    obj = import_object(ref)
    return obj() if isinstance(obj, type) else obj


def relax_judge_verdict_case() -> bool:
    """Work around an upstream ARE defect: its soft checkers parse the judge model's verdict with a
    **case-sensitive** substring test, so a judge that answers ``[[true]]`` instead of ``[[True]]``
    records no vote at all.

    ``LLMChecker.__call__`` (``are/simulation/validation/utils/llm_utils.py``) collects votes with
    ``if self.success_str in response`` / ``elif self.failure_str in response`` and returns ``None``
    when neither matched. ``SoftToolJudge.compare`` then calls it as ``if not checker_fn(...)``, and
    ``None`` is falsy — so an *unparsed* verdict rejects the event on exactly the same code path a
    genuine ``False`` would, with no distinguishing signal downstream. The judge's real answer is
    discarded.

    That is not merely a scoring error. ``turn_condition_wrapper`` gates each turn's release on the
    same verdict and calls ``env.stop()`` when it is falsy, so one discarded verdict on turn 0 ends
    the scenario at the agent's first reply and every later turn's oracle writes are then recorded
    as work the agent "did not perform" — work it was never given the chance to do.

    The casing never comes from the model, and no judge can avoid it. Both engines ARE ships end
    ``chat_completion`` with ``res.replace("False", "false").replace("True", "true")``
    (``agents/llm/litellm/litellm_engine.py``, ``agents/llm/hf/hf_engine.py``) — a JSON-shaped
    normalization of *agent* output that also rewrites every judge response on its way to the
    checker. ``create_judge_engine`` returns a ``LiteLLMEngine`` unconditionally, so on the shipped
    path ``"[[True]]" in response`` is **unsatisfiable**: the checker is handed ``[[true]]`` no
    matter which model answered. ``[[False]]`` is mangled identically, so the failure branch is dead
    too and the checker's only reachable return is ``None``. Verified offline through the real
    ``LLMChecker`` with LiteLLM's ``mock_response`` — model output ``[[True]]`` in, ``None`` out.

    So every ``[[True]]``-family checker — ``signature_checker``, ``tone_checker``,
    ``sanity_checker``, ``cab_checker`` — is structurally incapable of returning a verdict, while
    the ``[[Success]]`` family is untouched by the replace and works. Per
    ``validation/constants.py``'s ``PER_TOOL_TO_SOFT_CHECKER_TYPES`` that puts a dead checker in the
    loop for ``send_email``, ``reply_to_email``, ``send_message``, ``send_message_to_user`` and
    ``order_ride``; and ``SoftToolJudge.compare`` reaches that loop only when ``equality_checker``
    (exact match after normalization) has already failed — so a free-text body that differs from the
    oracle's at all is routed to a judge that cannot say yes.

    This relaxes **only** the marker comparison to case-insensitive; prompts, checkers, vote
    tallying and every other rule are untouched, so it restores the parse ARE's own prompts and unit
    tests intend rather than loosening the bar. Idempotent, and returns False if ARE is absent or
    already patched. Remove once ARE fixes this upstream — at the parse or at the engine — and note
    that a run scored with it applied is not the same artifact as one scored under stock ARE.
    """
    try:
        from are.simulation.validation.utils.llm_utils import LLMChecker
    except ImportError:  # ARE not installed — nothing to patch
        return False
    if getattr(LLMChecker, "_sora_case_insensitive_verdicts", False):
        return False

    def __call__(self: Any, user_prompt_args: dict[str, str]) -> bool | None:
        # Mirrors ARE's own implementation, with `.lower()` on both sides of the two membership
        # tests. Kept as a full replacement rather than a wrapper because the comparison sits in the
        # middle of the voting loop, with no seam to intercept the response on its way out.
        votes: list[bool] = []
        success = self.success_str.lower()
        failure = self.failure_str.lower()
        for _ in range(self.num_votes):
            response = self.judge(user_prompt_args)
            if response is None:
                continue
            lowered = response.lower()
            if success in lowered:
                votes.append(True)
            elif failure in lowered:
                votes.append(False)
        if len(votes) == 0:
            return None
        return sum(votes) >= len(votes) / 2

    LLMChecker.__call__ = __call__
    LLMChecker._sora_case_insensitive_verdicts = True
    log.info(
        "patched ARE LLMChecker: judge verdict markers compared case-insensitively "
        "(upstream defect — a [[true]] verdict is otherwise discarded and read as a rejection)"
    )
    return True


def attach_judge(
    scenario: Any,
    *,
    model: str | None = None,
    provider: str | None = None,
    endpoint: str | None = None,
    offline_validation: bool = False,
    relax_verdict_case: bool = True,
) -> None:
    """Attach ARE's GraphPerEvent judge so ``AreSimulation.validate()`` scores a benchmark scenario
    against its oracle event graph (the Gaia2 scoring path) instead of the ``success=None`` no-op.

    Runs ARE's ``preprocess_scenario``, which itself executes the scenario's ``OracleEvent``s in
    oracle mode to populate ``oracle_run_event_log`` and then sets ``scenario.judge`` /
    ``scenario.validate``. Call *after* ``load_scenario`` and *before* ``AreSimulation.start()``.
    Only meaningful for scenarios that carry ``OracleEvent``s (Gaia2 JSON does). ``model=None`` uses
    ARE's default judge model. This call is itself offline — oracle-mode replay of the OracleEvents
    is deterministic and modelless — but the judge model is *not* contacted only at ``validate()``:
    under online validation (``offline_validation=False``, the default) ARE installs
    ``judge.trigger_condition`` as each turn's release gate, so the judge is also called mid-run at
    every turn boundary, and a verdict of "turn failed" stops the environment and withholds the
    remaining turns. Use ``initialize_turns`` when the later turns are wanted without that gate.

    ``relax_verdict_case`` (default on) applies ``relax_judge_verdict_case`` first — see there for
    why. Not a per-model quirk: ARE's own engines lowercase ``True``/``False`` in *every* response
    on their way out of ``chat_completion``, so the ``[[True]]``-family checkers cannot return a
    verdict for **any** judge model, and the unparsed result is read as a rejection — which both
    mis-scores the event and, because the same verdict gates turn release, silently truncates the
    scenario. It is on by default because a run scored without it is not interpretable, and it logs
    when it fires so no run is patched silently. Pass False to reproduce stock ARE behavior
    exactly."""
    from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig
    from are.simulation.scenarios.scenario_imported_from_json.utils import preprocess_scenario
    from are.simulation.validation.configs import GraphPerEventJudgeConfig, create_judge_engine

    if relax_verdict_case:
        relax_judge_verdict_case()

    engine_config = (
        LLMEngineConfig(model_name=model, provider=provider, endpoint=endpoint)
        if model is not None
        else None  # None -> ARE's default judge model/provider
    )
    preprocess_scenario(
        scenario,
        judge_config=GraphPerEventJudgeConfig(engine=create_judge_engine(engine_config)),
        offline_validation=offline_validation,
    )


def initialize_turns(scenario: Any) -> None:
    """Wire a multi-turn benchmark scenario's turn triggers *without* attaching a judge, so every
    turn is delivered unconditionally. The run stays unscored (``scenario.judge`` is never set, so
    ``AreSimulation.validate()`` keeps returning the ``success=None`` no-op) — this buys the later
    turns, not a verdict.

    Why this exists as its own entry point: a Gaia2 scenario's later turns do not fire on their own.
    Their environment events hang off ``OracleEvent``s, which an agent-mode environment ignores
    without releasing successors, so a scenario run straight from ``load_scenario`` silently stops
    after turn 1. ARE re-anchors those events onto a per-turn ``ConditionCheckEvent`` only inside
    ``preprocess_scenario`` — which ``attach_judge`` reaches, but only by also making the judge the
    gate that decides whether each turn is released at all (see ``attach_judge``). Passing no judge
    takes ARE's own dummy trigger instead: it always releases the next turn, so the later turns
    arrive whether or not the agent got the earlier ones right.

    That is what development wants. Exercising behaviour that only a *later* turn provokes — the
    agent revising a plan when new information contradicts it — otherwise requires paying for a
    judge good enough that its verdict on turn 1 can be trusted not to end the run first, which
    couples the behaviour under test to judge quality for no benefit. Call *after* ``load_scenario``
    and *before* ``AreSimulation.start()``, and not together with ``attach_judge`` (either one
    initializes the turns; the second call is a no-op and the judge would still be the gate)."""
    from are.simulation.scenarios.scenario_imported_from_json.utils import preprocess_scenario

    preprocess_scenario(scenario, judge_config=None)


def populate_oracle_events(scenario: Any) -> None:
    """Replay the scenario's ``OracleEvent``s in ARE's oracle mode to populate
    ``oracle_run_event_log`` (and the turn map) *without* attaching a judge — the input
    ``write_count_check`` needs.

    ``preprocess_scenario`` already does this, but only as a side effect of being given a
    ``judge_config``, and taking that path costs a scoring judge: it also installs
    ``judge.validate`` as the per-turn release gate, so every turn boundary contacts the judge model
    (see ``attach_judge``). The replay itself is deterministic and modelless, so it is separable,
    and separating it is what lets an *unscored* dev run still be told whether it would have
    cleared ARE's tool-call-count gate.

    Call **before** ``initialize_turns``/``attach_judge`` and before ``AreSimulation.start()``:
    this mirrors ARE's own ordering (initialize -> oracle replay -> ``soft_reset``), and that
    ``soft_reset`` hands the agent run a clean environment afterwards. A no-op when the log is
    there, so pairing it with ``attach_judge`` wastes nothing; also a no-op for a scenario with no
    ``OracleEvent``s, which simply has no oracle to compare against.

    Raises when the replay itself fails, leaving the scenario as it was found — the apps are
    restored either way — so a caller for whom the gate is only a diagnostic can catch it and run
    without one rather than lose the run."""
    from are.simulation.environment import Environment, EnvironmentConfig
    from are.simulation.scenarios.scenario_imported_from_json.benchmark_scenario import (
        build_event_id_to_turn_idx,
    )
    from are.simulation.types import OracleEvent

    if getattr(scenario, "oracle_run_event_log", None) is not None:
        return
    # initialize() first, then look for OracleEvents — ARE's own ordering, and load-bearing: a
    # freshly loaded scenario's events are only built during initialize(), so testing before it
    # finds none and silently skips the replay.
    scenario.initialize()  # idempotent (guarded by Scenario._initialized)
    if not any(isinstance(e, OracleEvent) for e in scenario.events):
        return
    env = Environment(
        EnvironmentConfig(oracle_mode=True, queue_based_loop=True, start_time=scenario.start_time)
    )
    try:
        env.run(scenario)
        oracle_log = env.event_log.list_view()
    finally:
        # The replay drives the scenario's *live* apps, so its writes land in the very state the
        # agent is about to be run against, and ``soft_reset`` is what takes them back out again.
        # It has to run on the failure path too: a replay that raises partway (or one whose events
        # failed, checked below) would otherwise hand the agent an environment already carrying
        # the oracle's emails and calendar entries, with nothing anywhere to say so — a corrupted
        # run that still looks like a run. Restoring unconditionally is also what lets a caller
        # treat this whole check as optional and continue without it. ``stop`` only sets a flag
        # in ARE, so it is safe whether or not the loop ever started.
        env.stop()
        scenario.soft_reset()
    if any(e.failed() for e in oracle_log):
        raise RuntimeError(
            f"oracle replay failed: {[e.metadata.exception for e in oracle_log if e.failed()]}"
        )
    scenario.oracle_run_event_log = oracle_log
    if getattr(scenario, "event_id_to_turn_idx", None) is None:
        build_event_id_to_turn_idx(scenario=scenario)


_AUI_TOOL = "AgentUserInterface__send_message_to_user"


@dataclass(frozen=True)
class TurnWriteCounts:
    """One turn's agent-vs-oracle tally of *write* tool calls, and whether it clears ARE's gate.

    ``agent``/``oracle`` exclude ``send_message_to_user``, which is counted separately because the
    judge treats it differently: surplus replies *to the user* are tolerated up to
    ``extra_user_replies_allowed``, while any surplus call to a domain tool fails outright."""

    turn: int
    agent: Mapping[str, int]
    oracle: Mapping[str, int]
    agent_user_replies: int
    oracle_user_replies: int
    extra_user_replies_allowed: int

    @property
    def surplus(self) -> dict[str, int]:
        """Calls the agent made more often than the oracle — the usual reason a good-looking run
        fails, since an action nobody asked for is invisible in the trajectory."""
        return {
            name: n - self.oracle.get(name, 0)
            for name, n in self.agent.items()
            if n > self.oracle.get(name, 0)
        }

    @property
    def missing(self) -> dict[str, int]:
        return {
            name: n - self.agent.get(name, 0)
            for name, n in self.oracle.items()
            if n > self.agent.get(name, 0)
        }

    @property
    def replies_within_band(self) -> bool:
        """Whether the replies *to the user* clear the judge's band. Named because it is the one
        failing dimension ``surplus``/``missing`` cannot express — both are empty when the domain
        tallies match exactly and only the reply count is off, so a reader handed just those two
        sees a mismatch with no evidence attached. Anything reporting a failed turn has to be able
        to ask this separately."""
        return (
            self.oracle_user_replies
            <= self.agent_user_replies
            <= self.oracle_user_replies + self.extra_user_replies_allowed
        )

    @property
    def passed(self) -> bool:
        return dict(self.agent) == dict(self.oracle) and self.replies_within_band


@dataclass(frozen=True)
class WriteCountCheck:
    """ARE's ``preliminary_checks`` gate, recomputed offline. Not a score: clearing it only means
    the run reaches the judge's per-event matching, which can still fail it. Failing it, though, is
    conclusive — it is a hard gate ARE applies *before* any model-backed matching, so no verdict on
    the trajectory's quality can rescue a run whose write counts differ."""

    turns: tuple[TurnWriteCounts, ...]

    @property
    def passed(self) -> bool:
        return all(t.passed for t in self.turns)

    def summary(self) -> str:
        lines = []
        for t in self.turns:
            verdict = "ok" if t.passed else "MISMATCH"
            lines.append(f"  turn {t.turn}: {verdict}")
            if not t.passed:
                if t.surplus:
                    lines.append(f"    surplus (agent did, oracle did not): {t.surplus}")
                if t.missing:
                    lines.append(f"    missing (oracle did, agent did not): {t.missing}")
                if not t.replies_within_band:
                    lines.append(
                        f"    user replies: agent {t.agent_user_replies}, oracle "
                        f"{t.oracle_user_replies} (+{t.extra_user_replies_allowed} allowed)"
                    )
        head = "write-count gate: PASS" if self.passed else "write-count gate: FAIL"
        return "\n".join([head, *lines])


def _detached_events(events: list[Any]) -> list[Any]:
    """Shallow copies of ARE ``CompletedEvent``s, safe to hand to an ``EventFilter``.

    ARE's filters don't only filter: ``EventFilter.__call__`` first runs ``preprocess_event``, which
    *mutates* the event it is given — an agent ``add_email`` is relabeled ``EventType.ENV``, a
    ``CabApp.get_quotation`` becomes a READ, ``makedirs`` is renamed. Those are validation-time
    carve-outs that keep the judge's tallies honest, and under ARE's own runner they are invisible
    because everything downstream of the judge already expects them.

    Here they would not be: this check runs on *unscored* runs too, where no judge would otherwise
    have touched anything, and the HF trace is exported from these same objects afterwards — the
    exporter writes ``event.event_type.name``, so a relabeled agent write would be published as an
    environment event. Copying the event (and its ``action``, which carries ``operation_type``)
    leaves the tallies identical and the exported trace untouched."""
    detached = []
    for event in events:
        clone = copy.copy(event)
        if getattr(clone, "action", None) is not None:
            clone.action = copy.copy(clone.action)
        detached.append(clone)
    return detached


def write_count_check(
    scenario: Any,
    environment: Any,
    *,
    extra_user_replies_allowed: int = 1,
) -> WriteCountCheck | None:
    """Recompute ARE's tool-call-count gate for a finished run — **no judge model, no tokens**.

    ``GraphPerEventJudge.preliminary_checks`` requires the agent's write actions to match the
    oracle's as an exact ``Counter``, per turn, and applies it before any per-event LLM matching. So
    a run can perform every oracle action correctly and still score zero on one unrequested write,
    with nothing in the trajectory to show for it. This reproduces that arithmetic from the same ARE
    helpers the judge uses (``AgentEventFilter`` — non-failed AGENT writes only — plus ARE's own
    turn splitting at each ``send_message_to_user``), which is what keeps it from drifting.

    Returns ``None`` when the scenario carries no oracle log (not a benchmark scenario, or
    ``populate_oracle_events``/``attach_judge`` was never called). ``extra_user_replies_allowed``
    mirrors ``GraphPerEventJudgeConfig.extra_send_message_to_user_allowed`` (ARE's default is 1)."""
    from collections import Counter
    from types import SimpleNamespace

    from are.simulation.types import EventLog
    from are.simulation.validation.utils.event_utils import AgentEventFilter
    from are.simulation.validation.utils.scenario_utils import (
        extract_agent_events,
        extract_oracle_events,
    )

    if getattr(scenario, "oracle_run_event_log", None) is None:
        return None
    nb_turns = getattr(scenario, "nb_turns", None)
    if not nb_turns:
        return None

    # Both extractors are handed a stand-in holding *detached* events (see `_detached_events`), so
    # the filter's preprocessing can't reach the live environment the trace is exported from, nor
    # the scenario's oracle log. Each reads only these attributes; a missing one still surfaces as
    # ARE's own error, since the stand-in carries the same None through.
    agent_env = SimpleNamespace(
        event_log=EventLog.from_list_view(_detached_events(environment.event_log.list_view()))
    )
    oracle_view = SimpleNamespace(
        oracle_run_event_log=_detached_events(list(scenario.oracle_run_event_log)),
        event_id_to_turn_idx=getattr(scenario, "event_id_to_turn_idx", None),
        events=getattr(scenario, "events", []),
    )

    def split(events: list[Any]) -> tuple[dict[str, int], int]:
        counter = Counter(e.tool_name for e in events)
        replies = counter.pop(_AUI_TOOL, 0)
        return dict(counter), replies

    turns: list[TurnWriteCounts] = []
    for turn_idx in range(nb_turns):
        # One filter instance per call, as ARE does: `filter` also *preprocesses* each event
        # (reclassifying a couple of app-specific calls), so it must be the judge's own.
        oracle_events, _ = extract_oracle_events(oracle_view, AgentEventFilter(), turn_idx)
        try:
            agent_events = extract_agent_events(agent_env, AgentEventFilter(), turn_idx)
        except AssertionError:
            # ARE asserts the turn exists in the agent's own log. It won't when the agent sent
            # fewer `send_message_to_user` replies than the scenario has turns — a real failure
            # (it never reported back), and an empty tally is the truthful tally for that turn.
            agent_events = []
        agent, agent_replies = split(agent_events)
        oracle, oracle_replies = split(oracle_events)
        turns.append(
            TurnWriteCounts(
                turn=turn_idx,
                agent=agent,
                oracle=oracle,
                agent_user_replies=agent_replies,
                oracle_user_replies=oracle_replies,
                extra_user_replies_allowed=extra_user_replies_allowed,
            )
        )
    if not any(t.oracle or t.oracle_user_replies for t in turns):
        # The oracle asked for *nothing* anywhere in the run. Almost always extraction silently
        # yielding nothing rather than a real scenario — e.g. events whose actions never got their
        # WRITE classification, which AgentEventFilter then drops wholesale. Comparing two empty
        # tallies would report a confident PASS built on no evidence, and this check is precisely
        # what an unscored run trusts to tell it the run was clean. Unknown, not clean.
        return None
    return WriteCountCheck(turns=tuple(turns))


# -- app -> S-ORA usage-interface extraction ------------------------------------------------------

_JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}

_ARE_SEP = "__"  # ARE namespaces an app's tools as <App>__<operation> (as the flat MCP names do)


def _op_name(app: Any, app_tool: Any) -> str:
    """The bare operation name, stripping ARE's ``<App>__`` prefix so ops read as ``list_emails``
    (matching the ``are_mcp`` adapter and the hand-authored manuals), not ``EmailClientApp__…``."""
    prefix = f"{app.app_name()}{_ARE_SEP}"
    name: str = app_tool.name
    return name.removeprefix(prefix)


def _json_atom(t: str) -> dict[str, Any]:
    """One non-union ARE type string -> a JSON-Schema fragment. Recursive on ``list[...]`` so the
    item type is faithful too (``list[str]`` -> array of string, not array of anything)."""
    if t.startswith("list[") and t.endswith("]"):
        return {"type": "array", "items": _json_atom(t[len("list[") : -1].strip())}
    if t == "list":
        return {"type": "array"}
    if t == "dict" or t.startswith("dict["):
        return {"type": "object"}
    return {"type": _JSON_TYPES.get(t, "string")}


def _json_type(arg_type: Any) -> dict[str, Any]:
    """Map an ARE ``AppTool`` arg-type *string* (``str``, ``int``, ``list[str]``,
    ``list[str] | None``, ``int | float | None``, ``dict[str, Any]``, ...) to a JSON-Schema type
    fragment. Every arg the grounding model fills has to be represented faithfully: collapsing
    ``list[str]`` to ``string`` is what led the model to fill ``attendees`` with ``"Alice, Bob"`` —
    which ARE's own runtime type-check then rejects (``must be of type list[str] | None, got str``).
    Unions are split on ``|`` (``None`` dropped): a lone member maps directly; an all-numeric union
    (``int | float``) becomes JSON ``number`` (which admits both); any other heterogeneous union
    has no single faithful JSON type, so it falls back to ``string``. (Assumes ARE's flat vocabulary
    — no ``|`` nested inside brackets, which its apps never emit.)"""
    if not isinstance(arg_type, str):
        return {"type": "string"}
    members = [m.strip() for m in arg_type.split("|")]
    members = [m for m in members if m and m != "None"]
    if len(members) == 1:
        return _json_atom(members[0])
    if {_json_atom(m).get("type") for m in members} <= {"integer", "number"}:
        return {"type": "number"}
    return {"type": "string"}


def _params_schema(app_tool: Any) -> dict[str, Any]:
    """A JSON-Schema object for an ARE ``AppTool``'s args (same shape the ARE MCP server uses)."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for arg in app_tool.args:
        properties[arg.name] = {**_json_type(arg.arg_type), "description": arg.description or ""}
        if not getattr(arg, "has_default", False):
            required.append(arg.name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# How deep to expand a returned record's nested records into a JSON-Schema shape. The deepest real
# ARE shape is four levels: an envelope whose payload is a list of records, one of whose own fields
# is a list (CalendarEventsResult -> events: list[CalendarEvent] -> attendees: list[str]; likewise
# ReturnedEmails -> emails: list[Email] -> recipients: list[str]). The cap was 3, which clipped the
# innermost list's element type on exactly those ops — and `attendees` is precisely the field a plan
# needs to path into, so the elision was not cosmetic. 4 expands every real shape and still bounds
# the walk: the cap is defensive against a foreign/future annotation that is *self*-referential
# (`children: list[Node]`, resolved back to the live class by ``get_type_hints``), which would
# otherwise recurse into a ``RecursionError`` — one more level costs nothing there. Raise this only
# on the same evidence: a real returned shape that a plan must path into and cannot. When the cap
# does elide a nested shape, ``_type_to_schema`` emits a DEBUG log — a lead when a planner ``$from``
# path won't resolve because the shape below the cap was dropped.
_MAX_RETURN_DEPTH = 4

_PRIMITIVE_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _log_depth_cap(tp: Any) -> None:
    """A returned type expandable past ``_MAX_RETURN_DEPTH`` was clipped to a leaf. Benign — an
    over-deep or (unexpectedly) self-referential annotation is bounded rather than crashing — so
    this is DEBUG, not a warning: no ARE app record reaches the cap today, and the record shape
    stays valid, just shallower than the source type."""
    log.debug(
        "return-type introspection hit depth cap %d at %r; nested shape below it is elided "
        "(a planner $from path can't index past this point)",
        _MAX_RETURN_DEPTH,
        tp,
    )


def _record_fields(tp: Any) -> dict[str, Any] | None:
    """``{field name: annotation}`` when ``tp`` is a *record* — a dataclass or a ``TypedDict`` —
    else ``None``. Both spellings carry the same thing for a planner's purposes (named fields to
    path a ``$from`` into) and differ only in how the field list is reached, so they share one
    branch in ``_type_to_schema``. ARE uses both, and the split is not incidental: plain records are
    dataclasses (``Email``, ``CalendarEvent``) while every *paginated envelope* is a TypedDict
    (``CalendarEventsResult``, ``ProductListResult``). Recognizing only dataclasses therefore left
    exactly the windowed list reads — the ops whose payload a planner most needs to path into —
    declaring a bare ``string``. Annotations are resolved where possible, since ARE's app modules
    use ``from __future__ import annotations``; a dataclass falls back to its raw (string) field
    annotation, which the string mapper still reads."""
    if not isinstance(tp, type):
        return None
    is_record = dataclasses.is_dataclass(tp) or typing.is_typeddict(tp)
    if not is_record:
        return None
    try:
        hints = typing.get_type_hints(tp)
    except Exception:
        hints = {}
    if dataclasses.is_dataclass(tp):
        return {f.name: hints.get(f.name, f.type) for f in dataclasses.fields(tp)}
    # A TypedDict's own ``__annotations__`` omits inherited keys; ``get_type_hints`` flattens the
    # bases, so it is the primary and the raw dict only the fallback.
    return hints or dict(getattr(tp, "__annotations__", {}))


def _type_to_schema(tp: Any, depth: int = 0) -> dict[str, Any]:
    """Best-effort JSON-Schema fragment for a Python return annotation — deep enough that a planner
    can author a ``$from`` path into it (list nesting + a record's field names), no deeper. Handles
    ``list[X]`` (and tuple/set), ``X | None`` unions, records (dataclass or ``TypedDict``, one level
    of fields — see ``_record_fields``), and the JSON primitives; anything else (or past the depth
    cap) degrades to a bare ``string`` rather than raising. A *string* annotation (an unresolved
    ``from __future__ import annotations`` hint) reuses the union-aware arg-type string mapper (so
    ``'int | None'`` maps to integer, not the ``string`` a single-atom mapper would give)."""
    if isinstance(tp, str):
        return _json_type(tp)
    origin = typing.get_origin(tp)
    if origin in (Union, UnionType):
        members = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(members) == 1:
            return _type_to_schema(members[0], depth)
        # An all-numeric union (``int | float``) has a single faithful JSON type; anything else
        # heterogeneous doesn't — same rule the arg-type mapper (`_json_type`) applies.
        if members and {_type_to_schema(m, depth).get("type") for m in members} <= {
            "integer",
            "number",
        }:
            return {"type": "number"}
        return {"type": "string"}
    if origin in (list, set, frozenset) or tp in (list, set, frozenset):
        args = typing.get_args(tp)
        schema: dict[str, Any] = {"type": "array"}
        if args and args[0] is not ...:
            if depth < _MAX_RETURN_DEPTH:
                schema["items"] = _type_to_schema(args[0], depth + 1)
            else:
                _log_depth_cap(tp)
        return schema
    if origin is tuple or tp is tuple:
        return {"type": "array"}
    fields = _record_fields(tp)
    if fields is not None:
        if depth >= _MAX_RETURN_DEPTH:
            _log_depth_cap(tp)
            return {"type": "string"}
        properties = {
            name: _type_to_schema(annotation, depth + 1) for name, annotation in fields.items()
        }
        return {"type": "object", "properties": properties}
    if tp in _PRIMITIVE_JSON:
        return {"type": _PRIMITIVE_JSON[tp]}
    if tp is dict or origin is dict:
        return {"type": "object"}
    return {"type": "string"}


def _resolved_return_type(app_tool: Any) -> Any:
    """The operation's return annotation as a *resolved* type, not a string. ARE's app modules use
    ``from __future__ import annotations``, so ``AppTool.return_type`` is the raw annotation string
    (``'list[Email]'``) — useless for field-name introspection. Resolve it against the underlying
    function's own module globals (where ``Email``/``ReturnedEmails`` are defined) via
    ``get_type_hints``; fall back to the raw ``return_type`` (a real type in a fake, or a string we
    can only shallow-map) when there's no function or resolution fails."""
    func = getattr(app_tool, "function", None)
    if func is not None:
        try:
            resolved = typing.get_type_hints(func).get("return")
        except Exception:
            resolved = None
        if resolved is not None:
            return resolved
    return getattr(app_tool, "return_type", None)


def _returns_schema(app_tool: Any) -> dict[str, Any] | None:
    """The operation's declared result shape, or None. Synthesized from the resolved return type
    (see ``_resolved_return_type``) and seeded with the ``return_description`` prose so a planner
    sees both the shape to index a ``$from`` path into and what it means. A ``-> None`` op (resolved
    to ``NoneType``) has no result to reference, so it declares no shape — otherwise it would render
    a fictitious leaf a planner could bind an empty-path ``$from`` against."""
    tp = _resolved_return_type(app_tool)
    if tp is None or tp is type(None):
        return None
    schema = _type_to_schema(tp)
    description = getattr(app_tool, "return_description", None)
    return {**schema, "description": description} if description else schema


def _side_effecting(app_tool: Any) -> bool | None:
    """ARE's ``AppTool.write_operation`` (a bool set by the ``@app_tool`` decorator) mapped onto
    ``OperationSpecification.side_effecting`` — the native read/write signal, so no name heuristic.
    Absent/non-bool -> ``None`` (unknown; the checkpoint treats it as a write)."""
    write_operation = getattr(app_tool, "write_operation", None)
    return write_operation if isinstance(write_operation, bool) else None


def _operation_specs(app: Any) -> list[OperationSpecification]:
    return [
        OperationSpecification(
            name=_op_name(app, at),
            description=getattr(at, "function_description", None) or "",
            parameters=_params_schema(at),
            returns=_returns_schema(at),
            side_effecting=_side_effecting(at),
        )
        for at in app.get_tools()
    ]


def _to_serializable(value: Any) -> Any:
    """Make an app op's result JSON-friendly so a later step can ground on its fields (e.g. an
    ``email_id`` from ``list_emails``). Falls back to the raw value when ARE isn't importable (fakes
    already return plain data)."""
    try:
        from are.simulation.utils import make_serializable

        return make_serializable(value)
    except Exception:
        return value


# -- Tool / Workspace / Adapter -------------------------------------------------------------------


class _AreTool:
    """One live tool over one ARE app. ``invoke`` calls the app op (off-thread, lock-guarded
    against the Environment thread); ``observe`` polls ``get_state`` and emits ``state_changed`` on
    diff into the sink handed at ``focus`` — the in-process analogue of MCP's resource-update push,
    tied to the cycle's own Observe cadence so it's deterministic."""

    def __init__(
        self, *, tool_id: str, manual: Manual, app: Any, ops: dict[str, Any], simulation: Simulation
    ) -> None:
        self.id = tool_id
        self.manual = manual
        self.address: str | None = None
        self._app = app
        self._ops = ops
        self._sim = simulation
        self._sink: Any | None = None
        self._state: Any = None  # last observed state, for the diff

    async def invoke(self, operation_name: str, **params: Any) -> OperationAck:
        app_tool = self._ops.get(operation_name)
        if app_tool is None:
            return OperationAck(ok=False, result=f"unknown operation {operation_name!r}")
        try:
            result = await asyncio.to_thread(self._sim.run, lambda: app_tool(**params))
            return OperationAck(ok=True, result=_to_serializable(result))
        except Exception as exc:  # an app op raising is a failed ack, not a runtime crash
            return OperationAck(ok=False, result=str(exc))

    async def focus(self, sink: Any) -> None:
        self._sink = sink
        self._state = self._read_state()

    async def unfocus(self) -> None:
        self._sink = None
        self._state = None

    def observe(self) -> list[ObservableProperty]:
        # The new state is recorded *before* the push, not after. A pushed signal is screened
        # synchronously by the InterruptPolicy, which runs upstream of the once-per-cycle property
        # snapshot — so a push-time consumer has to read the current value off this tool, and this
        # tool must therefore already hold it. Assigning first also makes a re-entrant observe()
        # from inside that screen a no-op (state == self._state -> no second push) rather than a
        # recursion.
        state = self._read_state()
        previous = self._state
        changed = state != previous
        self._state = state
        if self._sink is not None and changed:
            # Thin: the event, not the state. The snapshot is published as the `state` observable
            # property below; duplicating it into the signal would only reproduce it in every
            # prompt that renders wm.signals. But thin is not contentless — the payload also names
            # WHERE it moved, which is the one thing a replace-by-key snapshot cannot express and
            # so duplicates nothing. This diff is nearly free: the comparison above already walks
            # both snapshots to decide `changed` at all, and today throws the result away.
            self._sink.push(
                self.id,
                Signal(
                    "state_changed",
                    {
                        "app": self._app.app_name(),
                        "changes": diff_values(previous, state),
                    },
                ),
            )
        # `self._state`, not the local: the push screen re-enters observe(), and if ARE's thread
        # mutated state in between, that nested call already advanced `self._state` past the local.
        # Returning the local would snapshot the pre-change world into working memory for one tick.
        return [ObservableProperty(name="state", value=self._state)]

    def _read_state(self) -> Any:
        # ARE's event-loop thread can mutate app state mid-read (no shared lock), so a get_state()
        # iterating a dict it's concurrently growing may raise RuntimeError. Retry the snapshot —
        # mutation is bursty, so an immediate re-read almost always settles (_STATE_READ_ATTEMPTS).
        last: RuntimeError | None = None
        for _ in range(_STATE_READ_ATTEMPTS):
            try:
                # Same normalization invoke() applies to an op result. ARE builds app state with
                # asdict(), which leaves Enum members in place: a mechanical `eq` against "Female"
                # then fails against <Gender.FEMALE: 'Female'>, and the value defeats JSON rendering
                # in prompts. Observed state is ground for the same comparisons an op result is, so
                # it has to arrive in the same shape.
                return _to_serializable(self._sim.run(self._app.get_state))
            except RuntimeError as exc:  # concurrent modification by the ARE event-loop thread
                last = exc
        assert last is not None
        raise last


class _SimulationClock:  # satisfies sora.environment.DomainClock
    """Reads domain time off the running simulation. A thin object rather than the ``Simulation``
    itself so the workspace exposes exactly one method to the runtime, and so a simulation that has
    not started yet fails where the clock is *read* rather than where the workspace is built."""

    def __init__(self, simulation: Simulation) -> None:
        self._sim = simulation

    def now(self) -> datetime:
        return self._sim.now()


class _AreWorkspace:
    def __init__(
        self, ws_id: str, origin: WorkspaceOrigin, tools: list[Tool], simulation: Simulation
    ) -> None:
        self.id = ws_id
        self.origin = origin
        # The one adapter that genuinely needs this seam: its environment's clock starts at the
        # scenario's own start_time and advances on the simulation's terms, so answering an `until`
        # from host wall-clock here is the 1 Jan 1970 bug ADR-0027 §5 records.
        self.clock: DomainClock | None = _SimulationClock(simulation)
        self._tools = tools
        self._sim = simulation

    def tools(self) -> list[Tool]:
        return self._tools

    async def close(self) -> None:
        await asyncio.to_thread(self._sim.stop)  # stops the Environment event-loop thread


class AreInProcessWorkspaceAdapter:
    """Imports the live ARE apps of a running ``AreSimulation`` as S-ORA tools (one per app, its
    ops from ``app.get_tools()``, plus a ``state`` observable + ``state_changed`` signal). The
    ``AgentUserInterface`` app is deliberately excluded — user messages are a transport concern
    (``AreTransport``), not a tool. The workspace owns the Environment lifecycle: ``discover``
    it, ``close`` stops it."""

    name = "are-sim"  # matches WorkspaceOrigin.adapter

    def __init__(
        self,
        *,
        workspace_id: str,
        origin: WorkspaceOrigin,
        simulation: Simulation,
        manual_source: ManualSource | None = None,
    ) -> None:
        self._workspace_id = workspace_id
        self._origin = origin
        self._sim = simulation
        self._manual_source = manual_source

    async def discover(self) -> list[Workspace]:
        await asyncio.to_thread(self._sim.start)
        tools = [await self._build_tool(app) for app in self._tool_apps()]
        return [_AreWorkspace(self._workspace_id, self._origin, tools, self._sim)]

    async def connect(
        self,
        workspace_record: WorkspaceRecord,
        tool_records: list[ToolRecord],
        manuals: dict[str, Manual],
    ) -> Workspace:
        # In-process apps live in the current simulation, so rebuild directly from them (no snapshot
        # reconstruction needed — the process holds the live objects).
        await asyncio.to_thread(self._sim.start)
        by_name = {app.app_name(): app for app in self._tool_apps()}
        tools: list[Tool] = []
        for record in tool_records:
            app = by_name.get(record.manual_id)
            if app is not None:
                tools.append(self._make_tool(record.id, app, manuals[record.manual_id]))
        return _AreWorkspace(workspace_record.id, workspace_record.origin, tools, self._sim)

    def _tool_apps(self) -> list[Any]:
        return [a for a in self._sim.apps() if a.app_name() != _AUI_APP]

    async def _build_tool(self, app: Any) -> Tool:
        manual = await self._paired_manual(app.app_name(), self._synth_manual(app))
        return self._make_tool(self._derive_tool_id(app.app_name()), app, manual)

    def _make_tool(self, tool_id: str, app: Any, manual: Manual) -> Tool:
        ops = {_op_name(app, at): at for at in app.get_tools()}
        return _AreTool(tool_id=tool_id, manual=manual, app=app, ops=ops, simulation=self._sim)

    def _synth_manual(self, app: Any) -> Manual:
        name = app.app_name()
        return Manual(
            id=name,
            metadata={"source": self.name, "app": name},
            description=f"ARE app {name}, in-process",
            observable_properties=[
                ObservablePropertySpecification(name="state", description="", schema={})
            ],
            signals=[SignalSpecification(name="state_changed", description="", schema={})],
            operations=_operation_specs(app),
            raw_text=None,
        )

    async def _paired_manual(self, manual_id: str, adapter_manual: Manual) -> Manual:
        if self._manual_source is None:
            return adapter_manual
        authored = await self._manual_source.get(manual_id)
        return adapter_manual if authored is None else merge_manuals(adapter_manual, authored)

    def _derive_tool_id(self, seed: str) -> str:
        # ADR-0014: globally unique, adapter-derived, deterministic (origin address + app name).
        return f"{self._origin.address}/{seed}"


class AreTransport:
    """``MessageTransport`` over the scenario's ``AgentUserInterface``. ``receive`` drains unread
    USER messages (the task + timeline follow-ups) as ``Message``s; ``send`` posts the agent's reply
    via ``send_message_to_user``; ``submit`` injects an ad hoc user message (a typed CLI line, a
    ``/stop`` resume) via ``send_message_to_agent``, which surfaces on the next ``receive`` drain
    like any timeline message. Shares the running ``AreSimulation`` with the adapter."""

    def __init__(self, simulation: Simulation) -> None:
        self._sim = simulation
        # Mirrors InProcessTransport.sent (an outbound log for tests/inspection) so a presentation
        # layer like TerminalSession can stream a reply the same way regardless of transport kind.
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def submit(self, message: Message) -> None:
        # The user side of the AUI: a message *from* the user *to* the agent. Routed through
        # sim.run (like send) so the write is registered on the Environment's own event loop, then
        # picked up by the next receive() drain — same path as the scenario's timeline messages, so
        # nothing downstream distinguishes an ad hoc line from a scripted one.
        aui = self._sim.aui
        if aui is None:
            return
        content = message.content
        text = content.get("text", "") if isinstance(content, dict) else str(content)
        self._sim.run(lambda: aui.send_message_to_agent(text))

    async def send(self, to: str, content: dict[str, Any]) -> None:
        aui = self._sim.aui
        if aui is None:
            return
        # Record only what was actually delivered — a presentation layer like TerminalSession
        # polls `.sent` and streams it as the agent's reply, so logging content that never
        # reached the AUI would show a message the user never actually got.
        self.sent.append((to, content))
        text = content.get("text", "") if isinstance(content, dict) else str(content)
        await asyncio.to_thread(self._sim.run, lambda: aui.send_message_to_user(text))

    def receive(self) -> AsyncIterator[Message]:
        async def _drain() -> AsyncIterator[Message]:
            aui = self._sim.aui
            if aui is None:
                return
            for m in self._sim.run(aui.get_last_unread_messages):
                # ARE message timestamps are sim-relative time; the t0 task message legitimately
                # has timestamp 0.0, so distinguish an absent timestamp (None) from a falsy 0.0
                # rather than `... or time.time()`, which would stamp wall-clock over a real 0.0.
                ts = getattr(m, "timestamp", None)
                yield Message(
                    sender="user",
                    content={"text": m.content},
                    received_at=time.time() if ts is None else ts,
                )

        return _drain()
