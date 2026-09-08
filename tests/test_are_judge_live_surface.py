"""The recorder against the *real* ARE judge, driven by a fake engine instead of a model.

The rest of the recorder's tests stub ARE's surface, which pins the wiring but cannot catch a
wrong guess about that surface — and one wrong guess (shadowing ``judge`` on ``soft_checkers``,
whose values are bound methods of the judge, not the ``LLMChecker`` objects that hold the raw
responses) recorded zero checker answers on the first real run while looking healthy. These tests
build ARE's own ``SoftToolJudge`` and assert against it, so that class of mistake fails here rather
than after a paid sweep. No network: the judge's engine is a callable, so a fake one is enough.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("are.simulation.validation.tool_judge")

from are.simulation.validation.configs import SoftToolJudgeConfig  # noqa: E402
from are.simulation.validation.constants import (  # noqa: E402
    PER_TOOL_ARG_TO_CHERCKER_TYPE,
    PER_TOOL_TO_SOFT_CHECKER_TYPES,
)
from are.simulation.validation.tool_judge import SoftToolJudge  # noqa: E402

from sora.adapters.are_judge import (  # noqa: E402
    arm_judge_recording,
    compare_parses,
    record_judge_events,
    rescore,
)

TOOL = "EmailClientApp__send_email"

ORACLE_CONTENT = (
    "Dear Elara, \n\nThese are the two cheapest properties with low violent crime rates: \n"
    "- Mechelen Mansion ($2,000)\n- Mechelen Maison ($2,000)\n\nBest regards,\nLiesbeth"
)
AGENT_CONTENT = (
    "Hi Elara,\n\nHere are the newly saved properties with low violent crime rates that I wanted "
    "to share with you:\n- Mechelen Maison — 2000\n- Mechelen Mansion — 2000\n\nBest,\nLiesbeth"
)


class _Event:
    """The judge reads only ``tool_name`` / ``get_args()`` / ``event_time`` off an event."""

    def __init__(self, args: dict[str, Any], event_id: str) -> None:
        self._args = args
        self.tool_name = TOOL
        self.event_id = event_id
        self.event_time = 1728975670.656627

    def get_args(self) -> dict[str, Any]:
        return dict(self._args)


def _events() -> tuple[_Event, _Event]:
    agent = _Event(
        {
            "recipients": ["elara.vandenberghe@gaia2mail.com"],
            "subject": "Properties List",
            "content": AGENT_CONTENT,
            "cc": [],
            "attachment_paths": [],
        },
        "AGENT-Emails.send_email-9ec278c0",
    )
    oracle = _Event(
        {
            "recipients": ["elara.vandenberghe@gaia2mail.com"],
            "subject": "Properties List",
            "content": ORACLE_CONTENT,
            "cc": [],
            "attachment_paths": [],
        },
        "OracleEvent-AGENT-ac421119",
    )
    return agent, oracle


def _judge(engine: Any) -> SoftToolJudge:
    return SoftToolJudge(
        SoftToolJudgeConfig(
            tool_name=TOOL,
            arg_to_checker_type=PER_TOOL_ARG_TO_CHERCKER_TYPE[TOOL],
            engine=engine,
            soft_checker_types=list(PER_TOOL_TO_SOFT_CHECKER_TYPES[TOOL]),
        )
    )


def _engine(reply: str) -> Any:
    """ARE's engine contract: ``engine(messages, **kwargs) -> (response, usage)``."""
    calls: list[list[dict[str, str]]] = []

    def engine(messages: list[dict[str, str]], **kwargs: Any) -> tuple[str, Any]:
        calls.append(messages)
        return reply, None

    engine.calls = calls  # type: ignore[attr-defined]
    return engine


def test_records_every_checker_answer_from_the_real_judge() -> None:
    """The regression the first live run exposed: rejected, with no reason recorded."""
    assert arm_judge_recording() or True  # idempotent; the class patch may already be in place
    judge = _judge(_engine("- Reasoning: differs.\n- Evaluation: [[Failure]]"))
    agent, oracle = _events()

    with record_judge_events() as collector:
        verdict = judge.compare(agent, oracle)
    recording = collector.snapshot(scenario_id="live-surface")

    assert verdict is False
    assert len(recording.events) == 1
    event = recording.events[0]
    assert event.equality is False, "the two emails differ, so the fast path must not settle it"
    assert event.verdict is False
    # The point of the whole exercise: the model's answer, not just the boolean it collapsed to.
    assert event.checkers, "no checker answers recorded — the raw responses are lost again"
    assert all(call.responses for call in event.checkers)
    assert any("[[Failure]]" in (call.responses[0] or "") for call in event.checkers)
    # ARE's own checker names, so a failure can be attributed to a checker rather than to the run.
    assert {call.checker for call in event.checkers} <= set(PER_TOOL_TO_SOFT_CHECKER_TYPES) | {
        checker.value for checker in PER_TOOL_TO_SOFT_CHECKER_TYPES[TOOL]
    }
    assert any(call.checker is not None for call in event.checkers)


def test_records_the_args_the_judge_actually_compared() -> None:
    judge = _judge(_engine("- Evaluation: [[Failure]]"))
    agent, oracle = _events()

    with record_judge_events() as collector:
        judge.compare(agent, oracle)
    event = collector.snapshot().events[0]

    # send_email's llm-checked args are subject and content; recipients go to the hard judge.
    assert set(event.agent_args) == {"subject", "content"}
    assert event.agent_args["content"] == AGENT_CONTENT
    assert event.oracle_args["content"] == ORACLE_CONTENT
    assert event.tool_name == TOOL


def _engine_family(*, approve: bool, mangle: bool) -> Any:
    """A model answering each checker in *its own* marker family.

    ``send_email``'s chain mixes the two: signature and tone want ``[[True]]``/``[[False]]``,
    email wants ``[[Success]]``/``[[Failure]]``. ``mangle`` reproduces what ARE's own engines do
    on the way back — ``.replace("True", "true").replace("False", "false")`` — which touches only
    the first family. That asymmetry is why an approving model still rejects under stock ARE.
    """

    def engine(messages: list[dict[str, str]], **kwargs: Any) -> tuple[str, Any]:
        true_family = "[[True]]" in messages[0]["content"]
        if true_family:
            verdict = "[[True]]" if approve else "[[False]]"
            if mangle:
                verdict = verdict.replace("True", "true").replace("False", "false")
        else:
            verdict = "[[Success]]" if approve else "[[Failure]]"
        return f"- Reasoning: because.\n- Evaluation: {verdict}", None

    return engine


def test_lowercased_verdict_diverges_between_the_two_parses() -> None:
    """The divergence the re-scorer exists to surface, on the real checker chain.

    Every checker approves; stock ARE still rejects, because it cannot read two of the three
    answers. That gap is invisible in ARE's own output — one boolean, indistinguishable from an
    agent that got it wrong."""
    judge = _judge(_engine_family(approve=True, mangle=True))
    agent, oracle = _events()

    with record_judge_events() as collector:
        verdict = judge.compare(agent, oracle)
    # Unpatched ARE ran this one, so its own parse is the stock one.
    recording = collector.snapshot(verdict_parse="stock")

    assert verdict is False, "stock ARE rejects an approving model on the [[True]] checkers"
    comparison = compare_parses(recording)
    assert comparison.divergent_events == (0,)
    assert comparison.divergent_off_equality_fast_path, (
        "the divergence must land on an event the equality checker missed, or the checker path "
        "was never exercised"
    )
    assert rescore(recording, parse="stock").score == 0.0
    assert rescore(recording, parse="case-insensitive").score == 1.0
    # And the re-score of the parse ARE actually used still reproduces ARE.
    assert rescore(recording, parse="stock").disagreements_with_recorded == ()


def test_rescore_reproduces_the_real_judge_verdict() -> None:
    """The self-audit, against ARE itself rather than against our model of it."""
    for approve in (True, False):
        # Unmangled markers, so ARE's own parse can read every answer and the verdict turns on
        # what the model said rather than on the defect.
        judge = _judge(_engine_family(approve=approve, mangle=False))
        agent, oracle = _events()
        with record_judge_events() as collector:
            verdict = judge.compare(agent, oracle)
        recording = collector.snapshot(verdict_parse="stock")

        assert verdict is approve
        result = rescore(recording, parse="stock")
        assert result.disagreements_with_recorded == (), (
            "the re-scorer no longer reproduces ARE's own rule"
        )
