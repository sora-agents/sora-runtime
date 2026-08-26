"""``write_count_check`` driven through the real ARE machinery — real oracle replay, ARE's own
``AgentEventFilter`` and turn splitting. Needs the ``are`` extra; still costs no model tokens,
because oracle mode is deterministic.

The scenario is built *in code* rather than loaded from a Gaia2 JSON. The dataset is Meta's and
gated by the HuggingFace terms, so ``examples/gaia2/scenarios/`` is gitignored and no fixture can
ship with the repo — a test pinned to a fetched scenario silently skips everywhere except the
machine that fetched it, which is indistinguishable from having no test. An in-code scenario
exercises the same parts that could actually drift, and unlike a fetched one it can be perturbed,
so the *failing* direction gets covered too — the direction that matters, since the check exists to
catch a run that did everything right plus one thing nobody asked for.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("are.simulation.environment")

from are.simulation.apps.agent_user_interface import AgentUserInterface  # noqa: E402
from are.simulation.apps.calendar import CalendarApp  # noqa: E402
from are.simulation.scenarios.scenario import Scenario  # noqa: E402
from are.simulation.types import EventRegisterer  # noqa: E402

from sora.adapters.are_sim import populate_oracle_events, write_count_check  # noqa: E402

pytestmark = pytest.mark.integration

_ADD = "CalendarApp__add_calendar_event"
_DELETE = "CalendarApp__delete_calendar_event"


class _TwoTurnScenario(Scenario):  # type: ignore[misc]
    """Two turns, one write each, split by a ``send_message_to_user``; ``surplus`` adds one
    unrequested write to turn 1 — aug24-run4's failure in miniature.

    Events are built inside ``EventRegisterer.capture_mode()``, ARE's own idiom for hand-authored
    scenarios, and that is load-bearing: calling an app method under capture mode returns an Event
    whose Action carries the method's registered ``operation_type``. Built with
    ``Event.from_function`` instead, the Action keeps ``OperationType``'s READ default and
    ``AgentEventFilter`` drops every event — see the unclassified-scenario test below.
    """

    start_time = 0
    duration = 8
    surplus = False

    def init_and_populate_apps(self, *args: object, **kwargs: object) -> None:
        self.calendar = CalendarApp()
        self.aui = AgentUserInterface()
        # Seeded directly rather than as an event, so turn 1 has something real to delete.
        self.seeded = self.calendar.add_calendar_event(
            title="Standup",
            start_datetime="2024-01-03 09:00:00",
            end_datetime="2024-01-03 09:15:00",
        )
        self.apps = [self.calendar, self.aui]

    def build_events_flow(self) -> None:
        with EventRegisterer.capture_mode():
            ask = self.aui.send_message_to_agent(content="Book Monday.").with_id("ask")
            add = (
                self.calendar.add_calendar_event(
                    title="Slot",
                    start_datetime="2024-01-08 09:00:00",
                    end_datetime="2024-01-08 10:00:00",
                )
                .oracle()
                .with_id("add")
                .depends_on(ask, delay_seconds=1)
            )
            reply = (
                self.aui.send_message_to_user(content="Booked.")
                .oracle()
                .with_id("reply")
                .depends_on(add, delay_seconds=1)
            )
            ask2 = (
                self.aui.send_message_to_agent(content="Cancel it.")
                .with_id("ask2")
                .depends_on(reply, delay_seconds=1)
            )
            remove = (
                self.calendar.delete_calendar_event(event_id=self.seeded)
                .oracle()
                .with_id("remove")
                .depends_on(ask2, delay_seconds=1)
            )
            reply2 = (
                self.aui.send_message_to_user(content="Cancelled.")
                .oracle()
                .with_id("reply2")
                .depends_on(remove, delay_seconds=1)
            )
            self.events = [ask, add, reply, ask2, remove, reply2]
            if self.surplus:
                self.events.append(
                    self.calendar.add_calendar_event(
                        title="Unasked",
                        start_datetime="2024-01-09 09:00:00",
                        end_datetime="2024-01-09 10:00:00",
                    )
                    .oracle()
                    .with_id("extra")
                    .depends_on(ask2, delay_seconds=1)
                )


def _replay(scenario: Any) -> Any:
    """Run a scenario's oracle events and hand back the environment. Oracle events execute as
    ``EventType.AGENT``, so a replay stands in for a perfect agent run — which is what lets the
    agent side of the comparison be produced without a model in the loop."""
    from are.simulation.environment import Environment, EnvironmentConfig

    scenario.initialize()
    env = Environment(
        EnvironmentConfig(oracle_mode=True, queue_based_loop=True, start_time=scenario.start_time)
    )
    env.run(scenario)
    env.stop()
    return env


def test_a_perfect_run_clears_the_gate() -> None:
    """A run that did exactly what the oracle asked must pass every turn. If this fails, the check
    is rejecting correct runs — worse than not having it."""
    scenario = _TwoTurnScenario()
    populate_oracle_events(scenario)

    assert scenario.oracle_run_event_log is not None
    # ARE split the flow at the send_message_to_user, the same boundary the judge uses.
    assert scenario.nb_turns == 2

    check = write_count_check(scenario, _replay(_TwoTurnScenario()))

    assert check is not None
    assert check.passed, check.summary()
    # Non-vacuous: the writes really were classified and counted, not filtered away to {} == {}.
    assert check.turns[0].oracle == {_ADD: 1}
    assert check.turns[1].oracle == {_DELETE: 1}


def test_one_unrequested_write_fails_the_gate() -> None:
    """The whole reason the check exists: every oracle action performed correctly, plus one write
    nobody asked for, and ARE scores the scenario zero with nothing in the trajectory to show
    why."""
    scenario = _TwoTurnScenario()
    populate_oracle_events(scenario)
    run = _TwoTurnScenario()
    run.surplus = True

    check = write_count_check(scenario, _replay(run))

    assert check is not None
    assert not check.passed
    assert check.turns[0].passed  # a clean first turn cannot compensate — the gate is per-turn
    assert check.turns[1].surplus == {_ADD: 1}
    assert _ADD in check.summary()  # the offending call is named, not just the verdict


def test_an_unclassified_scenario_is_unknown_rather_than_a_pass() -> None:
    """Events built outside capture_mode keep ``OperationType``'s READ default, so
    ``AgentEventFilter`` drops all of them. Two empty tallies compare equal, so the natural
    implementation reports a confident PASS on no evidence — exactly the false green an unscored
    run would trust, since there the check is the only pass/fail signal there is."""
    from are.simulation.types import Event

    class _Unclassified(_TwoTurnScenario):
        def build_events_flow(self) -> None:
            ask = Event.from_function(
                self.aui.send_message_to_agent, content="Book Monday."
            ).depends_on(None, delay_seconds=0)
            add = (
                Event.from_function(
                    self.calendar.add_calendar_event,
                    title="Slot",
                    start_datetime="2024-01-08 09:00:00",
                    end_datetime="2024-01-08 10:00:00",
                )
                .oracle()
                .depends_on(ask, delay_seconds=1)
            )
            reply = (
                Event.from_function(self.aui.send_message_to_user, content="Booked.")
                .oracle()
                .depends_on(add, delay_seconds=1)
            )
            self.events = [ask, add, reply]

    scenario = _Unclassified()
    populate_oracle_events(scenario)

    assert write_count_check(scenario, _replay(_Unclassified())) is None


class _EmailScenario(Scenario):  # type: ignore[misc]
    """One turn whose single write is an ``add_email`` — the one call ARE's ``EventFilter``
    *rewrites* rather than merely reads (``preprocess_event`` relabels an agent ``add_email`` on
    EmailClientApp/V2 to ``EventType.ENV``, so it stops counting as an agent write)."""

    start_time = 0
    duration = 8

    def init_and_populate_apps(self, *args: object, **kwargs: object) -> None:
        from are.simulation.apps.email_client import EmailClientApp

        self.email = EmailClientApp()
        self.aui = AgentUserInterface()
        self.apps = [self.email, self.aui]

    def build_events_flow(self) -> None:
        from are.simulation.apps.email_client import Email

        with EventRegisterer.capture_mode():
            ask = self.aui.send_message_to_agent(content="File this.").with_id("ask")
            add = (
                self.email.add_email(
                    email=Email(sender="a@b.com", recipients=["c@d.com"], subject="Hi")
                )
                .oracle()
                .with_id("add")
                .depends_on(ask, delay_seconds=1)
            )
            reply = (
                self.aui.send_message_to_user(content="Filed.")
                .oracle()
                .with_id("reply")
                .depends_on(add, delay_seconds=1)
            )
            self.events = [ask, add, reply]


def test_the_check_leaves_the_environment_it_read_unmodified() -> None:
    """The check must not disturb the run it inspects: the same event objects are handed to ARE's
    trace exporter afterwards, and the exporter publishes ``event_type.name``. ARE's filter mutates
    what it is given, so a naive call relabels the agent's ``add_email`` to ENV *in the live
    environment* — the exported trace would then attribute an agent write to the environment. This
    bites unscored runs especially: no judge ran, so nothing else would have touched these events.
    """
    from are.simulation.types import EventType

    scenario = _EmailScenario()
    populate_oracle_events(scenario)
    env = _replay(_EmailScenario())
    agent_writes = [
        e
        for e in env.event_log.list_view()
        if e.event_type is EventType.AGENT and e.tool_name.endswith("__add_email")
    ]
    assert agent_writes, "precondition: the replay produced an agent add_email to be relabeled"

    check = write_count_check(scenario, env)

    assert check is not None
    assert all(e.event_type is EventType.AGENT for e in agent_writes)


def test_a_failed_oracle_replay_leaves_the_apps_as_it_found_them() -> None:
    """The replay drives the scenario's *live* apps and takes its writes back out with
    ``soft_reset``. When it fails partway, that undo is exactly what still has to happen: whoever
    called this may well continue without the gate (it is only a diagnostic), and the agent would
    then start from an environment already carrying whatever the oracle managed to write — a run
    corrupted before its first tick, with nothing anywhere to say so."""

    class _HalfFailing(_TwoTurnScenario):
        """First oracle write lands, second fails: a replay that gets far enough to dirty the
        apps and then cannot finish."""

        def build_events_flow(self) -> None:
            with EventRegisterer.capture_mode():
                ask = self.aui.send_message_to_agent(content="Book Monday.").with_id("ask")
                add = (
                    self.calendar.add_calendar_event(
                        title="Slot",
                        start_datetime="2024-01-08 09:00:00",
                        end_datetime="2024-01-08 10:00:00",
                    )
                    .oracle()
                    .with_id("add")
                    .depends_on(ask, delay_seconds=1)
                )
                doomed = (
                    self.calendar.delete_calendar_event(event_id="no-such-event")
                    .oracle()
                    .with_id("doomed")
                    .depends_on(add, delay_seconds=1)
                )
                self.events = [ask, add, doomed]

    scenario = _HalfFailing()

    with pytest.raises(RuntimeError, match="oracle replay failed"):
        populate_oracle_events(scenario)

    titles = {e.title for e in scenario.calendar.events.values()}
    assert titles == {"Standup"}  # the seeded event only — "Slot" was taken back out
    # And no half-built log was published — probed the way the check itself probes it, since a
    # scenario that never got one does not carry the attribute at all.
    assert getattr(scenario, "oracle_run_event_log", None) is None


def test_the_check_is_absent_rather_than_wrong_without_an_oracle() -> None:
    """No oracle log -> None, not a vacuous PASS. A scenario that was never preprocessed has
    nothing to compare against, and reporting "gate cleared" there would be a false green."""
    assert write_count_check(_TwoTurnScenario(), None) is None
