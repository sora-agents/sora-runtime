"""Runtime reporting and user-input transitions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sora.activity import Activity, ActivityState
from sora.types import (
    InputWait,
)

if TYPE_CHECKING:
    from sora.cycle import DecisionCycle
    from sora.perception import Message

log = logging.getLogger("sora.strategies")


async def _report_to_user(cycle: DecisionCycle, text: str) -> None:
    """Say something to the user on the agent's own channel, from the runtime rather than a plan.

    The same transport call `runtime-io`'s `send_message_to_user` makes — used directly because
    there is no plan left to route through at the points that need it (an activity being abandoned,
    or parked on a question). Failures are logged, never raised: this runs on paths that are already
    reporting bad news, and a dead transport must not replace one failure with another.
    """
    try:
        await cycle.communication.send("user", {"text": text})
    except Exception:  # noqa: BLE001 — a transport failure must not mask what we were reporting
        log.exception("could not deliver a runtime message to the user: %s", text)


async def _await_input(cycle: DecisionCycle, activity: Activity, prompt: str) -> None:
    """Park an activity on the user's next instruction *and actually ask the question*.

    A breaker that sets `blocked_on` without delivering `prompt` stops the agent on a question no
    one can hear: `_resume_on_input` waits for a Message that the user has no reason to send. The
    two halves belong together, so every breaker goes through here rather than setting the fields
    itself. Deliberately not used for the hard-interrupt pause — the user caused that one and does
    not need to be told they did it.
    """
    activity.state = ActivityState.BLOCKED
    activity.blocked_on = InputWait(prompt=prompt)
    await _report_to_user(cycle, prompt)


def _truncate(value: Any, limit: int = 300) -> str:
    """One-line, length-capped rendering of an operation result for a log line (a tool error can be
    a long multi-line traceback)."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit] + "…"


def _goal_from_message(message: Message) -> str:
    """The default's deterministic goal derivation from a message — no interpretation, no model
    call: the message's own text if it carries a conventional ``text`` field, else the whole content
    rendered. A model-backed Situate would derive a richer goal instead."""
    text = message.content.get("text")
    return text if isinstance(text, str) else str(message.content)
