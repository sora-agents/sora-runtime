"""A *seeded* ARE MCP server for the gaia2 email/calendar showcase.

The stock ARE MCP server (`python -m are.simulation.apps.mcp.server...`) starts its apps **empty**:
`list_emails` returns nothing, so there's no email to schedule from or reply to. ARE's scenarios do
populate a world, but they deliver emails/tasks through the simulation's *event timeline*, which the
MCP server does not run — it only exposes a static snapshot of app *initial* state.

So this launcher seeds the initial state directly: it builds an `EmailClientApp` with an inbox email
from Alice (the task) plus a `CalendarApp`, and serves those live app instances over stdio MCP via
`ARESimulationMCPServer(apps=...)`. S-ORA's existing MCP adapter connects to it unchanged — now
`list_emails` returns a real email with a real `email_id`, which the Reason phase grounds
`reply_to_email` against. No simulation engine, so this is a *static* world (no mid-run follow-up
email / signal-driven replanning against real ARE — that needs `Environment.run`, a separate,
larger integration); it is enough to showcase the plan → ground → act loop end to end.

Launched as a subprocess by the adapter (see agent.yaml); not imported by the test suite. Requires
the ARE package: `uv sync --all-extras --group are`.
"""

from __future__ import annotations

from are.simulation.apps.calendar import CalendarApp
from are.simulation.apps.email_client import Email, EmailClientApp, EmailFolderName
from are.simulation.apps.mcp.server.are_simulation_mcp_server import ARESimulationMCPServer

USER_ADDRESS = "me@corp.com"


def build_seeded_apps() -> list[object]:
    """The showcase world: an inbox with Alice's scheduling request, and an (empty) calendar the
    agent checks for availability. Add more emails here to make `reply_to_email` pick the right one
    (a richer grounding demo); one keeps the happy path deterministic."""
    email = EmailClientApp()
    email.add_email(
        Email(
            sender="alice@corp.com",
            recipients=[USER_ADDRESS],
            subject="Team sync next Monday?",
            content=(
                "Hi! Can you set up a 30-minute team sync with Bob and Carol next Monday? "
                "Any time that works for everyone is fine. Thanks! — Alice"
            ),
        ),
        folder_name=EmailFolderName.INBOX,
    )
    return [email, CalendarApp()]


def main() -> None:
    server = ARESimulationMCPServer(apps=build_seeded_apps(), server_name="gaia2-email-calendar")
    server.run("stdio")  # blocks, serving the seeded apps until the client disconnects


if __name__ == "__main__":
    main()
