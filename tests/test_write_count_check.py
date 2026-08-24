"""ARE's tool-call-count gate, recomputed offline — ``sora.adapters.are_sim.write_count_check``.

Why this exists at all: ``GraphPerEventJudge`` applies ``preliminary_checks`` as a HARD gate before
any per-event model matching, and it demands exact ``Counter`` equality over non-failed AGENT WRITE
events, per turn. A run can therefore perform every oracle action correctly and still score zero on
one unrequested write, with nothing in the trajectory to show for it — which is exactly what
happened on the aug24-run4 adaptability run. Recomputing the gate costs no judge model and no
tokens, so it is the only pass/fail signal an *unscored* dev run has, and the regression test for
prompt changes whose effect the unit suite cannot otherwise see.

The tally arithmetic is tested directly here — no ARE import, so these run on a bare install. The
other half (real oracle replay through ARE's own filter and turn splitting) needs the ``are`` extra
and lives in ``test_write_count_check_are.py``.
"""

from __future__ import annotations

from sora.adapters.are_sim import TurnWriteCounts, WriteCountCheck

_DELETE = "Calendar__delete_calendar_event"
_ADD = "Calendar__add_calendar_event"
_SEND = "EmailClientV2__send_email"
_REPLY = "EmailClientV2__reply_to_email"


def _turn(
    agent: dict[str, int],
    oracle: dict[str, int],
    *,
    turn: int = 0,
    agent_replies: int = 0,
    oracle_replies: int = 0,
    allowed: int = 1,
) -> TurnWriteCounts:
    return TurnWriteCounts(
        turn=turn,
        agent=agent,
        oracle=oracle,
        agent_user_replies=agent_replies,
        oracle_user_replies=oracle_replies,
        extra_user_replies_allowed=allowed,
    )


def test_identical_counts_pass() -> None:
    assert _turn({_DELETE: 6, _ADD: 1}, {_DELETE: 6, _ADD: 1}).passed


def test_one_unrequested_write_fails_the_turn() -> None:
    # The aug24-run4 failure in miniature: every oracle action performed correctly, plus a courtesy
    # reply nobody asked for. The judge never gets to see that the rest was right.
    counts = _turn({_DELETE: 6, _ADD: 1, _REPLY: 1}, {_DELETE: 6, _ADD: 1})

    assert not counts.passed
    assert counts.surplus == {_REPLY: 1}
    assert counts.missing == {}


def test_a_missing_write_fails_and_is_named() -> None:
    counts = _turn({_DELETE: 5}, {_DELETE: 6, _ADD: 1})

    assert not counts.passed
    assert counts.missing == {_DELETE: 1, _ADD: 1}
    assert counts.surplus == {}


def test_surplus_user_replies_are_tolerated_but_domain_writes_are_not() -> None:
    # The asymmetry is ARE's, and it is the whole reason send_message_to_user is counted apart:
    # an extra word to the user is forgiven, an extra action in the user's name is not.
    assert _turn({}, {}, agent_replies=2, oracle_replies=1, allowed=1).passed
    assert not _turn({}, {}, agent_replies=3, oracle_replies=1, allowed=1).passed
    assert not _turn({_SEND: 1}, {}, agent_replies=1, oracle_replies=1).passed


def test_never_reporting_back_fails_even_with_perfect_writes() -> None:
    # Below the oracle's reply count, not above it: the agent did the work and never told the user.
    assert not _turn({_ADD: 1}, {_ADD: 1}, agent_replies=0, oracle_replies=1).passed


def test_the_check_fails_if_any_turn_fails() -> None:
    # Run 4's shape exactly: turn 0 clean, turn 1 carrying the surplus reply. A per-turn gate means
    # a flawless first turn cannot compensate, which is why the run looked fine and scored zero.
    clean = _turn(
        {_DELETE: 1, _ADD: 1, _SEND: 1},
        {_DELETE: 1, _ADD: 1, _SEND: 1},
        agent_replies=1,
        oracle_replies=1,
    )
    dirty = _turn({_DELETE: 6, _ADD: 1, _REPLY: 1}, {_DELETE: 6, _ADD: 1}, turn=1)
    check = WriteCountCheck(turns=(clean, dirty))

    assert clean.passed
    assert not check.passed
    assert "FAIL" in check.summary()
    assert _REPLY in check.summary()  # the offending call is named, not just the verdict


def test_summary_of_a_passing_check_names_no_diff() -> None:
    check = WriteCountCheck(turns=(_turn({_ADD: 1}, {_ADD: 1}),))

    summary = check.summary()
    assert check.passed
    assert "PASS" in summary
    assert "surplus" not in summary and "missing" not in summary
