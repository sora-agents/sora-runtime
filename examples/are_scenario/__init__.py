"""In-process ARE showcase — run a *dynamic* ARE scenario end to end.

Runs the ARE ``Environment`` event loop in-process (via ``sora.adapters.are_sim``), so a scenario's
timeline fires for real: the task arrives through the ``AgentUserInterface``, a mid-run follow-up
email triggers signal-driven replanning, and the run can be scored with the scenario's validators.

The config (``agent.yaml``) is **generic** — it names the ``are-sim`` workspace and ``are``
transport but *not* a scenario. The scenario is a runtime input: ``run.py --scenario
<dotted-or-json>`` (default ``scenario.EmailScheduleScenario``). Contrast the seeded static MCP demo
``examples/gaia2/email_calendar`` — that stays as the simple, single-shot MCP example.
"""

from examples.are_scenario.scenario import EmailScheduleScenario

__all__ = ["EmailScheduleScenario"]
