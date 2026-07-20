"""Gaia2 ``email_calendar`` scenario: schedule a meeting from an email and reply.

Reproduces the driving ARE scenario as running code — the four-step plan (read email -> check
calendar -> create event -> reply), procedural-memory reuse across runs, and signal-driven
replanning on a mid-scenario follow-up email. ``agent.yaml`` is the config; ``run`` is the on-demand
runner (real ARE MCP server + real Claude).
"""

from examples.gaia2.email_calendar.strategy import ScheduleFromEmailStrategy

__all__ = ["ScheduleFromEmailStrategy"]
