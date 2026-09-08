"""Record ARE's judge responses during a scored run, and re-score a stored run offline.

ARE's graph judge keeps a bare boolean per judged event, and nothing downstream keeps the model's
answer, so **a completed sweep cannot be re-scored afterwards at any price**: a run that was not
recorded while it happened can only be re-judged by paying for a fresh, nondeterministic judge pass
that is not paired with the original one event-for-event. That makes recording the one piece of
scoring plumbing whose window closes rather than slips, which is why it lives here rather than in a
throwaway script.

It matters beyond replication. ``relax_judge_verdict_case`` (``are_sim``) documents the upstream
defect this exists to survive: ARE's engines lowercase ``True``/``False`` in transit, so the
``[[True]]``-family soft checkers are structurally unable to return a verdict, and an *unparsed*
answer rejects the event on exactly the same falsy path a genuine rejection takes. Reading a
recorded run under both parses is what separates "the agent got it wrong" from "the scorer could
not say yes" — and, because the same verdict gates turn release, from "the scenario was truncated".

Two halves, deliberately separable:

* :func:`arm_judge_recording` + :func:`record_judge_events` — an **observational** patch over ARE's
  ``SoftToolJudge.compare`` and ``LLMChecker.judge``. It returns every original result untouched and
  swallows its own failures, so recording can never change a trajectory or cost a run its score.
* :func:`rescore` / :func:`compare_parses` — pure, deterministic, and ARE-free: they re-apply the
  checker vote tally and ARE's equality-then-conjunction rule to a stored recording under a named
  verdict parse. Importable, and testable, without the ``are`` extra installed.

The re-scorer re-implements ARE's rule rather than calling into it, so every result carries its own
falsification: ``RescoreResult.disagreements_with_recorded`` re-scores under the parse the run
*actually* used and names the events where the re-implementation disagrees with the boolean ARE
returned. On the live parse that tuple must be empty; anything else is a bug here, not a judge that
changed its mind.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

log = logging.getLogger(__name__)

# The two ways a judge response can be read. "stock" is ARE as shipped — a case-SENSITIVE substring
# test against the marker, which its own engines make unsatisfiable for the `[[True]]` family.
# "case-insensitive" is what `relax_judge_verdict_case` installs. Named rather than boolean because
# these names are written into every scored record and read back by the re-scorer.
VerdictParse = Literal["stock", "case-insensitive"]
VERDICT_PARSES: tuple[VerdictParse, ...] = ("stock", "case-insensitive")


# -- the stored shape -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckerCall:
    """One soft checker's contribution to one judged event.

    ``responses`` holds the model's raw text once per vote, in order, with ``None`` where the engine
    returned nothing — never a parsed verdict, because a parse is exactly what a re-score varies.
    ``success_str``/``failure_str`` travel with it: the markers are per-checker, so a recording that
    dropped them could not be re-parsed without ARE installed.
    """

    checker: str | None  # ARE's own checker name; None when it could not be attributed
    success_str: str
    failure_str: str
    responses: tuple[str | None, ...]
    prompt_args: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgedEvent:
    """One agent-event/oracle-event pair as the judge saw it.

    ``equality`` is ``equality_checker``'s outcome — the fast path, and the thing that decides
    whether any model was consulted at all. ``verdict`` is the boolean ARE itself returned, kept so
    a re-score can be checked against the run it re-scores rather than trusted.
    """

    tool_name: str | None
    agent_args: dict[str, Any]
    oracle_args: dict[str, Any]
    equality: bool | None
    verdict: bool | None
    checkers: tuple[CheckerCall, ...] = ()
    event_id: str | None = None
    event_time: float | None = None


@dataclass(frozen=True)
class JudgeRecording:
    """Every judged event of one scenario run, plus which parse produced its live verdicts."""

    events: tuple[JudgedEvent, ...] = ()
    scenario_id: str | None = None
    run_number: int | None = None
    verdict_parse: VerdictParse | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "run_number": self.run_number,
            "verdict_parse": self.verdict_parse,
            "events": [
                {
                    "tool_name": e.tool_name,
                    "agent_args": e.agent_args,
                    "oracle_args": e.oracle_args,
                    "equality": e.equality,
                    "verdict": e.verdict,
                    "event_id": e.event_id,
                    "event_time": e.event_time,
                    "checkers": [
                        {
                            "checker": c.checker,
                            "success_str": c.success_str,
                            "failure_str": c.failure_str,
                            "responses": list(c.responses),
                            "prompt_args": c.prompt_args,
                        }
                        for c in e.checkers
                    ],
                }
                for e in self.events
            ],
        }


def recording_from_dict(payload: dict[str, Any]) -> JudgeRecording:
    return JudgeRecording(
        scenario_id=payload.get("scenario_id"),
        run_number=payload.get("run_number"),
        verdict_parse=payload.get("verdict_parse"),
        events=tuple(
            JudgedEvent(
                tool_name=e.get("tool_name"),
                agent_args=e.get("agent_args") or {},
                oracle_args=e.get("oracle_args") or {},
                equality=e.get("equality"),
                verdict=e.get("verdict"),
                event_id=e.get("event_id"),
                event_time=e.get("event_time"),
                checkers=tuple(
                    CheckerCall(
                        checker=c.get("checker"),
                        success_str=c.get("success_str", ""),
                        failure_str=c.get("failure_str", ""),
                        responses=tuple(c.get("responses") or ()),
                        prompt_args=c.get("prompt_args") or {},
                    )
                    for c in e.get("checkers") or ()
                ),
            )
            for e in payload.get("events") or ()
        ),
    )


# -- recording ------------------------------------------------------------------------------------


@dataclass
class _OpenEvent:
    """A judged event under construction, while ``compare`` is on the stack."""

    tool_name: str | None = None
    agent_args: dict[str, Any] = field(default_factory=dict)
    oracle_args: dict[str, Any] = field(default_factory=dict)
    equality: bool | None = None
    event_id: str | None = None
    event_time: float | None = None
    checker_names: dict[int, str] = field(default_factory=dict)  # id(checker) -> ARE's own name
    calls: list[tuple[int, CheckerCall]] = field(default_factory=list)


class JudgeCollector:
    """Collects judged events while it is the active recording. See :func:`record_judge_events`."""

    def __init__(self) -> None:
        self._events: list[JudgedEvent] = []
        self._open: list[_OpenEvent] = []
        self._lock = threading.Lock()

    # -- called from the patches (never raises into ARE; see _guard) ------------------------------

    def open_event(self, frame: _OpenEvent) -> None:
        with self._lock:
            self._open.append(frame)

    def close_event(self, frame: _OpenEvent, verdict: Any) -> None:
        with self._lock:
            if frame in self._open:
                self._open.remove(frame)
            self._events.append(
                JudgedEvent(
                    tool_name=frame.tool_name,
                    agent_args=frame.agent_args,
                    oracle_args=frame.oracle_args,
                    equality=verdict_as_bool(frame.equality),
                    verdict=verdict_as_bool(verdict),
                    # Grouped by checker identity so the votes of a checker ARE calls repeatedly
                    # land in one entry, in call order, rather than as N single-vote entries.
                    checkers=tuple(_merge_calls(frame.calls)),
                    event_id=frame.event_id,
                    event_time=frame.event_time,
                )
            )

    def record_response(self, checker: Any, prompt_args: Any, response: Any) -> None:
        """Attribute one raw model answer to the innermost open event, or drop it.

        Dropping is deliberate: ARE also drives the judge outside any bracketed run (oracle
        preprocessing, an ad-hoc replay), and an unattributable answer is worth less than the
        guarantee that recording never fails a judge call.
        """
        with self._lock:
            if not self._open:
                return
            frame = self._open[-1]
            frame.calls.append(
                (
                    id(checker),
                    CheckerCall(
                        checker=frame.checker_names.get(id(checker)),
                        success_str=str(getattr(checker, "success_str", "")),
                        failure_str=str(getattr(checker, "failure_str", "")),
                        responses=(response if isinstance(response, str) else None,),
                        prompt_args=(
                            {str(k): str(v) for k, v in prompt_args.items()}
                            if isinstance(prompt_args, dict)
                            else {}
                        ),
                    ),
                )
            )

    # -- called by the harness ---------------------------------------------------------------------

    def snapshot(
        self,
        *,
        scenario_id: str | None = None,
        run_number: int | None = None,
        verdict_parse: VerdictParse | None = None,
    ) -> JudgeRecording:
        with self._lock:
            events = tuple(self._events)
        return JudgeRecording(
            events=events,
            scenario_id=scenario_id,
            run_number=run_number,
            verdict_parse=verdict_parse,
        )


def _merge_calls(calls: list[tuple[int, CheckerCall]]) -> list[CheckerCall]:
    merged: dict[int, CheckerCall] = {}
    order: list[int] = []
    for key, call in calls:
        existing = merged.get(key)
        if existing is None:
            merged[key] = call
            order.append(key)
        else:
            merged[key] = CheckerCall(
                checker=existing.checker or call.checker,
                success_str=existing.success_str,
                failure_str=existing.failure_str,
                responses=existing.responses + call.responses,
                prompt_args=existing.prompt_args or call.prompt_args,
            )
    return [merged[key] for key in order]


def verdict_as_bool(value: Any) -> bool | None:
    """ARE's checkers return ``True``/``False``/``None``; anything else is not a verdict."""
    return value if isinstance(value, bool) else None


# The active collector. A module-level slot rather than a thread-local: ARE validates a scenario's
# events from its own environment thread, and a run brackets the whole of that, so the collector has
# to be visible from wherever the judge happens to be driven.
_active: JudgeCollector | None = None
_active_lock = threading.Lock()


@contextmanager
def record_judge_events() -> Iterator[JudgeCollector]:
    """Collect every judged event that ARE decides inside this block.

    Must bracket the *whole* run, not just ``validate()``: under online validation the judge is also
    the per-turn release gate, so most of a multi-turn scenario's events are judged mid-run.
    """
    global _active
    collector = JudgeCollector()
    with _active_lock:
        previous, _active = _active, collector
    try:
        yield collector
    finally:
        with _active_lock:
            _active = previous


def _current() -> JudgeCollector | None:
    with _active_lock:
        return _active


def judge_recording_armed() -> bool:
    """Whether the patch is in force. Distinct from :func:`arm_judge_recording`'s return, which is
    False both when ARE is missing and when arming already happened — a caller deciding whether it
    is safe to spend a sweep's tokens needs the state, not the transition."""
    try:
        from are.simulation.validation.tool_judge import SoftToolJudge
    except ImportError:
        return False
    return bool(getattr(SoftToolJudge, "_sora_records_judge_responses", False))


def arm_judge_recording() -> bool:
    """Patch ARE so judge responses are recorded. Idempotent; returns False if already armed, or if
    ARE is not installed — never raises, so a caller can *decide* what an unrecordable run means.

    One wrap point on the class — ``SoftToolJudge.compare``, which brackets exactly one judged
    event. Everything else is shadowed *on the instance* for the duration of that call, which is
    what keeps the patch from depending on details of ARE it cannot verify:

    * ``equality_checker`` — the fast path's outcome, and the agent/oracle args exactly as the judge
      itself selected them. Instance-level, so it works whether ARE binds a method or a plain
      callable, and whatever its signature.
    * each of ``soft_checkers``' own ``judge`` — the model's raw text, once per vote. Also
      instance-level, and for the same reason: whether ``judge`` is a class method or an attribute
      assigned in ``__init__`` is not something this code should have to know. It composes with
      ``relax_judge_verdict_case``, which replaces ``LLMChecker.__call__``: that method calls
      ``self.judge``, so the relaxed tally reads the shadowed one and each vote is still recorded
      exactly once.

    Every wrapper calls through and records afterwards, inside a guard: a recording failure logs
    once and is dropped, because a diagnostic must never cost the run its real result. A judge call
    outside any ``compare`` — oracle preprocessing, an ad-hoc replay — is simply not recorded, which
    is the right answer: it belongs to no judged event.

    Concurrency: ARE validates a scenario's events sequentially, so the instance-level shadows are
    safe there. Two ``compare`` calls overlapping on the *same* judge instance would interleave
    them; nothing in ARE's own runner does that.
    """
    try:
        from are.simulation.validation.tool_judge import SoftToolJudge
    except ImportError:  # ARE not installed — nothing to patch
        return False
    if getattr(SoftToolJudge, "_sora_records_judge_responses", False):
        return False

    original_compare = SoftToolJudge.compare

    def compare(self: Any, *args: Any, **kwargs: Any) -> Any:
        collector = _current()
        if collector is None:
            return original_compare(self, *args, **kwargs)
        frame = _OpenEvent()
        with _guard():
            _describe_compare(self, args, kwargs, frame)
        collector.open_event(frame)
        undo = [_shadow_equality(self, frame), _shadow_judges(self, collector)]
        try:
            verdict = original_compare(self, *args, **kwargs)
        finally:
            for restore in undo:
                restore()
        with _guard():
            collector.close_event(frame, verdict)
        return verdict

    SoftToolJudge.compare = compare
    SoftToolJudge._sora_records_judge_responses = True
    log.info("patched ARE judge: raw checker responses recorded for offline re-scoring")
    return True


@contextmanager
def _guard() -> Iterator[None]:
    """Recording is a diagnostic wrapped around a scored run; it never takes the run down."""
    try:
        yield
    except Exception:  # noqa: BLE001 — deliberate: any failure here is worth less than the run
        log.warning("judge-response recording failed for one event", exc_info=True)


def _describe_compare(
    judge: Any, args: tuple[Any, ...], kwargs: dict[str, Any], frame: _OpenEvent
) -> None:
    """Fill in what can be read off the judge and its call without knowing ``compare``'s signature.

    Two independent sources, because either alone has a hole: the *events* carry the tool name and
    ids even on the equality fast path (where no checker ever runs), while ``equality_checker``'s
    own arguments are the args ARE actually compared, after its per-tool selection.
    """
    frame.tool_name = _text(getattr(getattr(judge, "config", None), "tool_name", None))
    frame.checker_names = {id(checker): str(name) for name, checker in _llm_checkers(judge).items()}
    events = [value for value in (*args, *kwargs.values()) if _is_event(value)]
    if events:
        frame.agent_args = _args_of(events[0])
        frame.event_id = _text(getattr(events[0], "event_id", None))
        time_value = getattr(events[0], "event_time", None)
        frame.event_time = float(time_value) if isinstance(time_value, int | float) else None
        frame.tool_name = _text(getattr(events[0], "tool_name", None)) or frame.tool_name
    if len(events) > 1:
        frame.oracle_args = _args_of(events[1])


def _shadow_equality(judge: Any, frame: _OpenEvent) -> Any:
    """Temporarily wrap ``judge.equality_checker`` to capture its outcome and its inputs.

    Instance-level, so it intercepts the attribute lookup regardless of whether ARE binds a method
    or a plain callable, and regardless of the checker's signature. Returns the undo.
    """
    try:
        original = judge.equality_checker
    except Exception:  # noqa: BLE001 — no equality checker to shadow; the events still describe it
        return lambda: None
    had_own = "equality_checker" in vars(judge)

    def equality_checker(*args: Any, **kwargs: Any) -> Any:
        outcome = original(*args, **kwargs)
        with _guard():
            frame.equality = verdict_as_bool(outcome)
            # The args ARE compared, which are the selected subset — preferred over the raw event
            # args recorded above precisely because they are what the verdict was about.
            agent = kwargs.get("agent_args")
            oracle = kwargs.get("oracle_args")
            if isinstance(agent, dict):
                frame.agent_args = dict(agent)
            if isinstance(oracle, dict):
                frame.oracle_args = dict(oracle)
        return outcome

    try:
        judge.equality_checker = equality_checker
    except Exception:  # noqa: BLE001 — a frozen judge: fall back to the events alone
        return lambda: None

    def undo() -> None:
        with _guard():
            if had_own:
                judge.equality_checker = original
            else:
                del judge.equality_checker

    return undo


def _llm_checkers(judge: Any) -> dict[Any, Any]:
    """The judge's ``LLMChecker`` objects — the only place a raw model answer passes through.

    Deliberately *not* ``soft_checkers``: that dict maps a checker name to a **bound method of the
    judge** (``self.email_checker``), which merely forwards to the checker held here. Shadowing
    there records nothing and raises nothing, which is how a run once came back rejected with no
    reason attached.
    """
    checkers = getattr(judge, "llm_checkers", None)
    return checkers if isinstance(checkers, dict) else {}


def _shadow_judges(judge: Any, collector: JudgeCollector) -> Any:
    """Temporarily wrap every checker's ``judge`` so each vote's raw text is recorded.

    Shadowed on the checker instances rather than patched on ``LLMChecker``, so it does not matter
    whether ARE binds ``judge`` as a method or assigns it per instance. Returns the undo.
    """
    checkers = _llm_checkers(judge)
    if not checkers:
        return lambda: None
    restores: list[Any] = []
    for checker in checkers.values():
        try:
            original = checker.judge
        except Exception:  # noqa: BLE001 — a checker with no judge seam: nothing to record
            continue

        def wrapped(
            *args: Any, _checker: Any = checker, _original: Any = original, **kwargs: Any
        ) -> Any:
            response = _original(*args, **kwargs)
            with _guard():
                prompt_args = args[0] if args else kwargs.get("user_prompt_args")
                collector.record_response(_checker, prompt_args, response)
            return response

        # Read before the assignment, which would otherwise always make it look instance-owned.
        had_own = "judge" in vars(checker)
        try:
            checker.judge = wrapped
        except Exception:  # noqa: BLE001 — a frozen checker: its votes go unrecorded, the run lives
            continue
        restores.append((checker, original, had_own))

    def undo() -> None:
        with _guard():
            for checker, original, had_own in restores:
                if had_own:
                    checker.judge = original
                else:
                    del checker.judge

    return undo


def _is_event(value: Any) -> bool:
    return callable(getattr(value, "get_args", None)) and hasattr(value, "tool_name")


def _args_of(event: Any) -> dict[str, Any]:
    args = event.get_args()
    return dict(args) if isinstance(args, dict) else {}


def _text(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value else None


# -- offline re-scoring ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RescoreResult:
    """One recording re-scored under one parse."""

    parse: VerdictParse
    verdicts: tuple[bool, ...]
    # Indices where this re-score disagrees with the boolean ARE recorded. Only meaningful when
    # `parse` is the parse the run actually used — there it must be empty, and anything else is a
    # defect in this re-implementation rather than evidence about the run.
    disagreements_with_recorded: tuple[int, ...] = ()

    @property
    def score(self) -> float | None:
        """Fraction of judged events that pass. None when the recording holds no events — an
        unjudged run has no score, which is not the same as scoring zero."""
        if not self.verdicts:
            return None
        return sum(1 for verdict in self.verdicts if verdict) / len(self.verdicts)


@dataclass(frozen=True)
class ParseComparison:
    """The same recording re-scored under both parses, and where they part company."""

    recording: JudgeRecording
    results: dict[VerdictParse, RescoreResult]
    divergent_events: tuple[int, ...]
    # Whether at least one divergence is on an event `equality_checker` missed. This is the
    # criterion that matters: divergence *only* on the fast path would mean the re-parse never
    # exercised the patched code, which is indistinguishable from agreement unless it is asked
    # about separately.
    divergent_off_equality_fast_path: bool

    def score(self, parse: VerdictParse) -> float | None:
        return self.results[parse].score


def parse_verdict(
    response: str | None, success_str: str, failure_str: str, *, case_insensitive: bool
) -> bool | None:
    """ARE's marker test: substring, success wins ties, ``None`` when neither marker appears."""
    if response is None:
        return None
    haystack = response.lower() if case_insensitive else response
    success = success_str.lower() if case_insensitive else success_str
    failure = failure_str.lower() if case_insensitive else failure_str
    if success and success in haystack:
        return True
    if failure and failure in haystack:
        return False
    return None


def checker_verdict(call: CheckerCall, *, parse: VerdictParse) -> bool | None:
    """One checker's verdict: majority over the votes that parsed, ``None`` when none did — the
    tally in ``LLMChecker.__call__``, reproduced so a recording can be read without ARE."""
    case_insensitive = parse == "case-insensitive"
    votes = [
        vote
        for vote in (
            parse_verdict(
                response, call.success_str, call.failure_str, case_insensitive=case_insensitive
            )
            for response in call.responses
        )
        if vote is not None
    ]
    if not votes:
        return None
    return sum(votes) >= len(votes) / 2


def event_verdict(event: JudgedEvent, *, parse: VerdictParse) -> bool:
    """ARE's ``SoftToolJudge.compare`` rule: the equality fast path, else a strict conjunction over
    the soft checkers in which an unparsed ``None`` rejects exactly like a ``False``.

    Equality failing with no soft checker to appeal to is a rejection, not a vacuous pass — that is
    the arrangement for a tool with no soft checkers at all, where an inexact argument is simply
    wrong.
    """
    if event.equality is True:
        return True
    if not event.checkers:
        return False
    return all(checker_verdict(call, parse=parse) is True for call in event.checkers)


def rescore(recording: JudgeRecording, *, parse: VerdictParse) -> RescoreResult:
    verdicts = tuple(event_verdict(event, parse=parse) for event in recording.events)
    return RescoreResult(
        parse=parse,
        verdicts=verdicts,
        disagreements_with_recorded=tuple(
            index
            for index, (event, verdict) in enumerate(zip(recording.events, verdicts, strict=True))
            if isinstance(event.verdict, bool) and event.verdict is not verdict
        ),
    )


def compare_parses(recording: JudgeRecording) -> ParseComparison:
    results = {parse: rescore(recording, parse=parse) for parse in VERDICT_PARSES}
    stock = results["stock"].verdicts
    relaxed = results["case-insensitive"].verdicts
    divergent = tuple(
        index for index, (a, b) in enumerate(zip(stock, relaxed, strict=True)) if a != b
    )
    return ParseComparison(
        recording=recording,
        results=results,
        divergent_events=divergent,
        divergent_off_equality_fast_path=any(
            recording.events[index].equality is not True for index in divergent
        ),
    )
