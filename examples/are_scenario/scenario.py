"""The default in-code ARE scenario for the in-process showcase — a *dynamic* one.

Unlike the seeded static MCP demo (``examples/gaia2/email_calendar``), this runs against the ARE
``Environment`` event loop, so its timeline actually fires: the task is delivered through the
``AgentUserInterface`` at t0, then a follow-up email lands at delay ``T`` and *changes the answer*
(Monday -> Tuesday). That mid-run change is what surfaces to the agent as a ``state_changed`` signal
and drives a replan — the thing the static MCP world cannot do.

It's just the default when ``run.py`` is invoked with no ``--scenario``. Point ``--scenario`` at any
dotted ``Scenario`` subclass or a Gaia2 ``.json`` file to run that instead — the in-process path is
scenario-agnostic.
"""

from __future__ import annotations

from are.simulation.apps.agent_user_interface import AgentUserInterface
from are.simulation.apps.calendar import CalendarApp
from are.simulation.apps.email_client import Email, EmailClientApp, EmailFolderName
from are.simulation.scenarios.scenario import Scenario
from are.simulation.types import Event

USER_ADDRESS = "me@corp.com"


class EmailScheduleScenario(Scenario):  # type: ignore[misc]  # ARE is an untyped dependency-group
    start_time = 0
    duration = 60

    def init_and_populate_apps(self, *args: object, **kwargs: object) -> None:
        self.email = EmailClientApp()
        self.calendar = CalendarApp()
        self.aui = AgentUserInterface()
        self.email.add_email(
            Email(
                sender="alice@corp.com",
                recipients=[USER_ADDRESS],
                subject="Team sync next Monday?",
                content=(
                    "Hi! Can you set up a 30-minute team sync with Bob and Carol next Monday? "
                    "Any time that works is fine. Thanks! — Alice"
                ),
            ),
            folder_name=EmailFolderName.INBOX,
        )
        self.apps = [self.email, self.calendar, self.aui]

    def build_events_flow(self) -> None:
        # t0: the user (simulation) hands the agent its task through the AUI.
        task = Event.from_function(
            self.aui.send_message_to_agent,
            content="Please schedule the team sync Alice emailed about, then reply to her.",
        ).depends_on(None, delay_seconds=0)
        # mid-run: Alice changes the day. This lands off the agent's own action -> state_changed.
        follow_up = Event.from_function(
            self.email.add_email,
            email=Email(
                sender="alice@corp.com",
                recipients=[USER_ADDRESS],
                subject="Re: Team sync next Monday?",
                content="Small change — could we do Tuesday instead of Monday? Thanks!",
            ),
            folder_name=EmailFolderName.INBOX,
        ).depends_on(None, delay_seconds=8)
        self.events = [task, follow_up]
