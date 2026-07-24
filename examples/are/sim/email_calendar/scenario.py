"""The default in-code ARE scenario for the in-process showcase — a *dynamic* one.

Unlike the seeded static MCP demo (``examples/are/mcp/email_calendar``), this runs against the ARE
``Environment`` event loop, so its timeline actually fires: the task is delivered through the
``AgentUserInterface`` at t0, then a follow-up email lands at delay ``T`` and *changes the answer*
(Monday -> Tuesday). That mid-run change is what surfaces to the agent as a ``state_changed`` signal
and drives a replan — the thing the static MCP world cannot do.

It's the bundled illustrative scenario for this showcase, referenced via ``sora run``'s
``--scenario examples.are.sim.email_calendar.scenario.EmailScheduleScenario`` (there is no
default — ``agent.yaml`` names no scenario). Point ``--scenario`` at any other dotted ``Scenario``
subclass or a Gaia2 ``.json`` file to run that instead — the in-process path is scenario-agnostic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from are.simulation.apps.agent_user_interface import AgentUserInterface
from are.simulation.apps.calendar import CalendarApp
from are.simulation.apps.email_client import Email, EmailClientApp, EmailFolderName
from are.simulation.scenarios.scenario import Scenario
from are.simulation.scenarios.validation_result import ScenarioValidationResult
from are.simulation.types import Event

USER_ADDRESS = "me@corp.com"
ALICE_ADDRESS = "alice@corp.com"
# Fixed (not the default random uuid) so build_events_flow's oracle reply and validate() can both
# reference Alice's original email deterministically.
ALICE_REQUEST_EMAIL_ID = "alice_team_sync_request"


class EmailScheduleScenario(Scenario):  # type: ignore[misc]  # ARE is an untyped dependency-group
    start_time = 0
    duration = 60

    def init_and_populate_apps(self, *args: object, **kwargs: object) -> None:
        self.email = EmailClientApp()
        self.calendar = CalendarApp()
        self.aui = AgentUserInterface()
        self.email.add_email(
            Email(
                email_id=ALICE_REQUEST_EMAIL_ID,
                sender=ALICE_ADDRESS,
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
                sender=ALICE_ADDRESS,
                recipients=[USER_ADDRESS],
                subject="Re: Team sync next Monday?",
                content="Small change — could we do Tuesday instead of Monday? Thanks!",
            ),
            folder_name=EmailFolderName.INBOX,
        ).depends_on(None, delay_seconds=8)

        # Oracle events model the ideal response, for ARE's own oracle-mode tooling
        # (run_and_validate) — not for validate() below, which never checks whether these
        # specifically fired. Their literal date is illustrative only: this scenario has no
        # calendar-anchored start_time (0 is a nominal counter, not a real date), so there's no
        # single "correct" day for validate() to check the agent's own choice against — see
        # validate()'s weekday check.
        oracle_calendar_event = (
            Event.from_function(
                self.calendar.add_calendar_event,
                title="Team sync",
                start_datetime="2025-01-07 09:00:00",  # a Tuesday
                end_datetime="2025-01-07 09:30:00",
                attendees=["Bob", "Carol"],
            )
            .oracle()
            .depends_on(follow_up, delay_seconds=2)
        )
        oracle_reply_event = (
            Event.from_function(
                self.email.reply_to_email,
                email_id=ALICE_REQUEST_EMAIL_ID,
                content=(
                    "Hi Alice, I've set up the sync with Bob and Carol for Tuesday. See you there!"
                ),
            )
            .oracle()
            .depends_on(oracle_calendar_event, delay_seconds=2)
        )

        self.events = [task, follow_up, oracle_calendar_event, oracle_reply_event]

    def validate(self, env: Any) -> ScenarioValidationResult:
        """Checked against final app state, not the event log (the agent may reasonably reply to
        either Alice's original email or her follow-up, and the follow-up's email_id is
        auto-generated, so validate() can't hardcode which one). No anchored calendar date to
        check literally (see build_events_flow) — the weekday is the signal that the agent used
        the follow-up's correction rather than the stale original Monday request.

        The gate is deliberately narrow: a Tuesday event with Bob and Carol as attendees, plus a
        reply to Alice. Title wording and exact meeting length are Alice's incidental phrasing, not
        what this scenario is testing (whether the mid-run correction actually drove a replan) — so
        they're reported for visibility but don't block success. Attendees are matched by substring,
        not exact name, since "Bob Smith" is as valid an answer as "Bob"."""
        try:
            calendar_app = env.get_app("CalendarApp")
            # Candidates, not a single boolean: an event can have the right attendees and still
            # miss on weekday, or nothing may have the right attendees at all — the rationale needs
            # to say which, instead of collapsing both into one False.
            candidates = []
            for calendar_event in calendar_app.events.values():
                attendees_text = " ".join(calendar_event.attendees).lower()
                if "bob" not in attendees_text or "carol" not in attendees_text:
                    continue
                start = datetime.fromtimestamp(calendar_event.start_datetime, tz=UTC)
                duration_minutes = (
                    calendar_event.end_datetime - calendar_event.start_datetime
                ) / 60
                candidates.append((calendar_event, start, duration_minutes))

            scheduled_on_tuesday = any(start.weekday() == 1 for _, start, _ in candidates)

            email_app = env.get_app("EmailClientApp")
            sent_emails = email_app.folders[EmailFolderName.SENT].emails
            replied_to_alice = any(ALICE_ADDRESS in email.recipients for email in sent_emails)

            success = scheduled_on_tuesday and replied_to_alice
            if candidates:
                detail = "; ".join(
                    f"{event.title!r} start={start.isoformat()} ({start.strftime('%A')}) "
                    f"duration={duration_minutes:g}min"
                    for event, start, duration_minutes in candidates
                )
            else:
                titles = [e.title for e in calendar_app.events.values()]
                detail = f"no event with bob+carol as attendees; calendar has {titles!r}"
            rationale = (
                f"scheduled_on_tuesday={scheduled_on_tuesday}, replied_to_alice={replied_to_alice}"
                f" [{detail}]"
            )
            return ScenarioValidationResult(success=success, rationale=rationale)
        except Exception as e:
            return ScenarioValidationResult(success=False, exception=e)
