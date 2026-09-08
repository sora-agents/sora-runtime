"""Judge-response recording and the offline re-parse.

ARE's graph judge keeps a bare boolean per judged event, so a completed sweep cannot be re-scored
afterwards at any price. These tests pin two separable halves:

* the **recorder** — an observational patch over ARE's ``SoftToolJudge`` / ``LLMChecker`` that
  captures, per judged event, the checker inputs, the ``equality_checker`` outcome and every raw
  model response. ARE is an optional extra, so the classes it patches are stubbed in
  ``sys.modules`` (the same technique ``test_are_sim.py`` uses for the verdict-case patch) — which
  also pins *our* contract rather than whatever ARE happens to ship.
* the **re-scorer** — pure, ARE-free, and the half the recording exists for: it replays a stored
  recording under either verdict parse and reports where the two disagree.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from sora.adapters.are_judge import (
    CheckerCall,
    JudgedEvent,
    JudgeRecording,
    arm_judge_recording,
    compare_parses,
    record_judge_events,
    recording_from_dict,
    rescore,
)

# ------------------------------------------------------------------------------------------------
# Stubs: the ARE surface the recorder wraps, mirroring the shapes documented on
# `relax_judge_verdict_case` and exercised by the offline replay of a single event pair.
# ------------------------------------------------------------------------------------------------


class _StubLLMChecker:
    """ARE's ``LLMChecker``: ``judge()`` calls the model, ``__call__`` tallies votes over
    ``num_votes`` responses with a case-SENSITIVE membership test."""

    def __init__(
        self,
        responses: list[str | None],
        *,
        success: str = "[[True]]",
        failure: str = "[[False]]",
    ) -> None:
        self.success_str = success
        self.failure_str = failure
        self.num_votes = len(responses)
        self._responses = list(responses)
        self._n = 0
        self.seen_prompt_args: list[dict[str, str]] = []

    def judge(self, user_prompt_args: dict[str, str]) -> str | None:
        self.seen_prompt_args.append(dict(user_prompt_args))
        response = self._responses[self._n % len(self._responses)]
        self._n += 1
        return response

    def __call__(self, user_prompt_args: dict[str, str]) -> bool | None:
        votes: list[bool] = []
        for _ in range(self.num_votes):
            response = self.judge(user_prompt_args)
            if response is None:
                continue
            if self.success_str in response:
                votes.append(True)
            elif self.failure_str in response:
                votes.append(False)
        if not votes:
            return None
        return sum(votes) >= len(votes) / 2


class _StubSoftToolJudge:
    """ARE's ``SoftToolJudge``: ``compare`` takes the fast path when ``equality_checker`` succeeds
    and otherwise applies every soft checker as a strict conjunction (``if not checker_fn(...)``,
    so an unparsed ``None`` rejects exactly like a ``False``)."""

    def __init__(self, tool_name: str, checkers: dict[str, _StubLLMChecker], equal: bool) -> None:
        self.config = SimpleNamespace(tool_name=tool_name)
        # Mirrors ARE's own split, which a recorder is easy to get wrong: the raw answers pass
        # through the `LLMChecker` objects in `llm_checkers`, while `soft_checkers` maps the same
        # names to *bound methods of the judge* that merely forward to them.
        self.llm_checkers: dict[str, _StubLLMChecker] = checkers
        self.soft_checkers: dict[str, Any] = {
            name: (
                lambda _c=checker, agent_args=None, **kwargs: _c(
                    {"agent_action_call": str(agent_args)}
                )
            )
            for name, checker in checkers.items()
        }
        self._equal = equal

    def equality_checker(self, agent_args: dict[str, Any], oracle_args: dict[str, Any]) -> bool:
        del agent_args, oracle_args
        return self._equal

    def compare(self, agent_event: Any, oracle_event: Any, **kwargs: Any) -> bool:
        del kwargs
        agent_args = agent_event.get_args()
        oracle_args = oracle_event.get_args()
        if self.equality_checker(agent_args=agent_args, oracle_args=oracle_args):
            return True
        for checker_fn in self.soft_checkers.values():
            if not checker_fn(agent_args=agent_args):
                return False
        return True


def _event(tool: str, args: dict[str, Any], event_id: str = "e1") -> Any:
    return SimpleNamespace(
        tool_name=tool, get_args=lambda: dict(args), event_time=1728975600.0, event_id=event_id
    )


def _stub_are(monkeypatch: pytest.MonkeyPatch) -> tuple[type, type]:
    """Install fresh stub ``LLMChecker``/``SoftToolJudge`` classes under ARE's import paths."""

    class LLMChecker(_StubLLMChecker):
        pass

    class SoftToolJudge(_StubSoftToolJudge):
        pass

    monkeypatch.setitem(
        sys.modules,
        "are.simulation.validation.utils.llm_utils",
        SimpleNamespace(LLMChecker=LLMChecker),
    )
    monkeypatch.setitem(
        sys.modules,
        "are.simulation.validation.tool_judge",
        SimpleNamespace(SoftToolJudge=SoftToolJudge),
    )
    return LLMChecker, SoftToolJudge


# ------------------------------------------------------------------------------------------------
# The recorder
# ------------------------------------------------------------------------------------------------


def test_arming_without_are_installed_reports_false(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry makes the import raise, which is what an uninstalled `are` extra looks like.
    # Arming must say so rather than raise, so a caller can refuse a scored sweep instead of
    # running one it can never re-score. Forced rather than assumed: `are` is installed in some
    # environments and absent in others, and this contract holds either way.
    monkeypatch.setitem(sys.modules, "are.simulation.validation.tool_judge", None)
    assert arm_judge_recording() is False


def test_records_checker_inputs_the_equality_outcome_and_every_raw_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker_cls, judge_cls = _stub_are(monkeypatch)
    assert arm_judge_recording() is True

    tone = checker_cls(["The tone is fine. Evaluation: [[true]]"])
    judge = judge_cls("EmailClientApp__send_email", {"tone_checker": tone}, equal=False)

    agent_args = {"subject": "Film Production Day appointment details"}
    oracle_args = {"subject": "Film Production Day"}
    with record_judge_events() as collector:
        verdict = judge.compare(
            _event("EmailClientApp__send_email", agent_args),
            _event("EmailClientApp__send_email", oracle_args),
        )

    # Purely observational: the live verdict is untouched by recording.
    assert verdict is False

    recording = collector.snapshot(scenario_id="s1", run_number=0, verdict_parse="stock")
    assert len(recording.events) == 1
    event = recording.events[0]
    assert event.tool_name == "EmailClientApp__send_email"
    assert event.agent_args == agent_args
    assert event.oracle_args == oracle_args
    assert event.equality is False
    assert event.verdict is False
    assert [c.checker for c in event.checkers] == ["tone_checker"]
    assert event.checkers[0].responses == ("The tone is fine. Evaluation: [[true]]",)
    assert event.checkers[0].success_str == "[[True]]"
    assert event.checkers[0].prompt_args == {"agent_action_call": str(agent_args)}


def test_equality_fast_path_records_an_event_with_no_checker_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fast path spends no model tokens, but the event still has to appear: a re-parse that
    # silently dropped it would report a different denominator than the run it re-scores.
    _, judge_cls = _stub_are(monkeypatch)
    arm_judge_recording()
    judge = judge_cls("EmailClientApp__send_email", {}, equal=True)
    with record_judge_events() as collector:
        assert judge.compare(_event("t", {"a": 1}), _event("t", {"a": 1})) is True
    event = collector.snapshot().events[0]
    assert event.equality is True
    assert event.verdict is True
    assert event.checkers == ()


def test_arming_is_idempotent_and_composes_with_the_verdict_case_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sora.adapters.are_sim import relax_judge_verdict_case

    checker_cls, judge_cls = _stub_are(monkeypatch)
    assert arm_judge_recording() is True
    assert arm_judge_recording() is False  # already armed
    # The two touch different seams — the recorder shadows each checker's `judge`, the relaxation
    # replaces `LLMChecker.__call__`, which calls it — so they compose in either order and a vote
    # is still recorded exactly once.
    assert relax_judge_verdict_case() is True

    tone = checker_cls(["Evaluation: [[true]]"])
    judge = judge_cls("t", {"tone_checker": tone}, equal=False)
    with record_judge_events() as collector:
        # The relaxed parse now reads the lowercase verdict, so the event passes...
        assert judge.compare(_event("t", {"a": 1}), _event("t", {"a": 2})) is True
    # ...and the raw response is recorded exactly once per vote regardless.
    assert collector.snapshot().events[0].checkers[0].responses == ("Evaluation: [[true]]",)


def test_records_a_checker_whose_judge_is_bound_per_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Whether ARE defines `judge` on the class or assigns it in `__init__` is not something the
    # recorder can verify from here, so it must not depend on the answer: the checker's `judge` is
    # shadowed on the instance, which intercepts either binding.
    checker_cls, judge_cls = _stub_are(monkeypatch)
    arm_judge_recording()
    tone = checker_cls(["[[true]]"])
    tone.judge = lambda prompt_args: "instance-bound: [[true]]"
    judge = judge_cls("t", {"tone_checker": tone}, equal=False)
    with record_judge_events() as collector:
        judge.compare(_event("t", {"a": 1}), _event("t", {"a": 2}))
    assert collector.snapshot().events[0].checkers[0].responses == ("instance-bound: [[true]]",)
    # ...and the instance binding is put back exactly as it was found.
    assert tone.judge({}) == "instance-bound: [[true]]"


def test_a_judge_call_outside_an_active_collector_is_dropped_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ARE also runs the judge outside any run we bracket (oracle preprocessing, ad-hoc replays).
    # Recording must never become a reason a judge call fails.
    _, judge_cls = _stub_are(monkeypatch)
    arm_judge_recording()
    judge = judge_cls("t", {}, equal=True)
    assert judge.compare(_event("t", {}), _event("t", {})) is True


def test_recording_round_trips_through_json(monkeypatch: pytest.MonkeyPatch) -> None:
    checker_cls, judge_cls = _stub_are(monkeypatch)
    arm_judge_recording()
    judge = judge_cls("t", {"tone_checker": checker_cls(["[[true]]"])}, equal=False)
    with record_judge_events() as collector:
        judge.compare(_event("t", {"a": 1}), _event("t", {"a": 2}))
    recording = collector.snapshot(scenario_id="s1", run_number=2, verdict_parse="case-insensitive")
    assert recording_from_dict(json.loads(json.dumps(recording.to_dict()))) == recording


# ------------------------------------------------------------------------------------------------
# The offline re-parse
# ------------------------------------------------------------------------------------------------


def _lowercase_pass_event(**overrides: Any) -> JudgedEvent:
    """One judged event whose only soft checker answered ``[[true]]`` — the shape ARE's own engine
    produces for every judge model, and the one the stock parse discards."""
    base: dict[str, Any] = {
        "tool_name": "EmailClientApp__send_email",
        "agent_args": {"subject": "paraphrased"},
        "oracle_args": {"subject": "oracle"},
        "equality": False,
        "verdict": False,
        "checkers": (
            CheckerCall(
                checker="tone_checker",
                success_str="[[True]]",
                failure_str="[[False]]",
                responses=("The tone is fine. Evaluation: [[true]]",),
                prompt_args={},
            ),
        ),
    }
    base.update(overrides)
    return JudgedEvent(**base)


def test_stock_parse_discards_a_lowercase_verdict_and_the_relaxed_one_reads_it() -> None:
    recording = JudgeRecording(events=(_lowercase_pass_event(),))
    assert rescore(recording, parse="stock").verdicts == (False,)
    assert rescore(recording, parse="case-insensitive").verdicts == (True,)


def test_a_genuine_rejection_still_rejects_under_both_parses() -> None:
    event = _lowercase_pass_event(
        checkers=(
            CheckerCall(
                checker="tone_checker",
                success_str="[[True]]",
                failure_str="[[False]]",
                responses=("The signature names someone else. Evaluation: [[false]]",),
                prompt_args={},
            ),
        )
    )
    recording = JudgeRecording(events=(event,))
    assert rescore(recording, parse="stock").verdicts == (False,)
    assert rescore(recording, parse="case-insensitive").verdicts == (False,)


def test_the_equality_fast_path_passes_under_both_parses() -> None:
    recording = JudgeRecording(
        events=(_lowercase_pass_event(equality=True, verdict=True, checkers=()),)
    )
    assert rescore(recording, parse="stock").verdicts == (True,)
    assert rescore(recording, parse="case-insensitive").verdicts == (True,)


def test_no_soft_checker_and_no_equality_match_rejects() -> None:
    # Equality failed and there is nothing else to appeal to — the event is a miss, not a vacuous
    # pass, which is what the recorded live verdict says too.
    recording = JudgeRecording(events=(_lowercase_pass_event(checkers=()),))
    assert rescore(recording, parse="stock").verdicts == (False,)


def test_soft_checkers_are_a_strict_conjunction() -> None:
    good = CheckerCall("tone_checker", "[[True]]", "[[False]]", ("[[true]]",), {})
    bad = CheckerCall("signature_checker", "[[True]]", "[[False]]", ("[[false]]",), {})
    recording = JudgeRecording(events=(_lowercase_pass_event(checkers=(good, bad)),))
    assert rescore(recording, parse="case-insensitive").verdicts == (False,)


def test_votes_are_tallied_by_majority_like_ares_own_checker() -> None:
    call = CheckerCall(
        "tone_checker", "[[True]]", "[[False]]", ("[[true]]", "[[true]]", "[[false]]"), {}
    )
    recording = JudgeRecording(events=(_lowercase_pass_event(checkers=(call,)),))
    assert rescore(recording, parse="case-insensitive").verdicts == (True,)


def test_rescoring_under_the_live_parse_reproduces_the_recorded_verdict() -> None:
    # The re-scorer's own self-check: it re-implements ARE's conjunction, so on the parse the run
    # actually used it must agree with what ARE returned. A disagreement means the model is wrong,
    # not that the judge changed its mind.
    recording = JudgeRecording(events=(_lowercase_pass_event(),), verdict_parse="stock")
    assert rescore(recording, parse="stock").disagreements_with_recorded == ()
    result = rescore(recording, parse="case-insensitive")
    assert result.disagreements_with_recorded == (0,)


def test_compare_parses_reports_divergence_only_off_the_equality_fast_path() -> None:
    diverging = _lowercase_pass_event()
    fast_path = _lowercase_pass_event(equality=True, verdict=True, checkers=())
    diff = compare_parses(JudgeRecording(events=(fast_path, diverging)))
    assert diff.divergent_events == (1,)
    assert diff.divergent_off_equality_fast_path is True
    assert diff.score("stock") == 0.5
    assert diff.score("case-insensitive") == 1.0


def test_compare_parses_on_an_agreeing_recording_reports_no_divergence() -> None:
    # The exit criterion's failure mode: a pipeline that records faithfully but never exercises the
    # patched path looks exactly like agreement, so agreement must be reported as such.
    fast_path = _lowercase_pass_event(equality=True, verdict=True, checkers=())
    diff = compare_parses(JudgeRecording(events=(fast_path,)))
    assert diff.divergent_events == ()
    assert diff.divergent_off_equality_fast_path is False
