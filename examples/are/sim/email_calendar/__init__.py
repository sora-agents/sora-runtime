"""In-process ARE showcase — run a *dynamic* ARE scenario end to end.

Runs the ARE ``Environment`` event loop in-process (via ``sora.adapters.are_sim``), so a scenario's
timeline fires for real: the task arrives through the ``AgentUserInterface``, a mid-run follow-up
email triggers signal-driven replanning, and the run can be scored with the scenario's validators.

The config (``agent.yaml``) is **generic** — it names the ``are-sim`` workspace and ``are``
transport but *not* a scenario. The scenario is a runtime input: ``sora run ... --scenario
<dotted-or-json>`` (default ``scenario.EmailScheduleScenario``). Contrast the seeded static MCP demo
``examples/are/mcp/email_calendar`` — that stays as the simple, single-shot MCP example.

Deliberately no package-level re-export: unlike ``scenario.py``, ``strategies.py``/``report.py``
don't need the real ``are`` package, and nothing resolves ``EmailScheduleScenario`` except by its
full dotted path (``examples.are.sim.email_calendar.scenario.EmailScheduleScenario``, via
``import_object``)
— re-exporting it here would force every sibling-module import through ``scenario.py``'s hard ARE
dependency, breaking collection of fakes-only tests when the optional ``are`` group isn't installed.
"""
