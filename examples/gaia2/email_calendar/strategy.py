"""``ScheduleFromEmailStrategy`` — the one custom strategy this example needs.

The four-step plan itself is produced by the runtime's model-backed ``DefaultReasonStrategy`` (E2):
retrieve a cached plan for the goal, else infer one from the joined tools, then advance it step by
step. This strategy adds exactly one behavior on top — **signal-driven replanning** (EXAMPLES.md's
"ARE's dynamic events as reactive interrupts"): when a state-change ``Signal`` from a focused tool
lands *mid-plan* (ARE emits ``resource_updated`` after a scheduled event injects a follow-up email),
the in-flight plan is invalidated so the next ``reason()`` re-plans from the updated working
memory — same four-step shape, corrected date. Everything else delegates to the default, so this
stays a thin wrapper, not a re-implementation of planning.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sora.strategies import DefaultReasonStrategy

if TYPE_CHECKING:
    from sora.activity import Activity
    from sora.cycle import DecisionCycle
    from sora.memory import WorkingMemory
    from sora.strategies import TickResult

log = logging.getLogger("examples.gaia2.email_calendar")

# Strategy-owned bookkeeping stored on activity.context (the runtime never writes there): how many
# signal percepts this strategy has already reacted to, so a *new* one — not an already-seen one —
# triggers a replan. wm.signals is never shrunk by a matched completion signal (see ADR-0019) — this
# example's manuals declare none anyway — so the count only grows in practice, bounded only by the
# blocked-state machinery's fixed retention cap (a burst this example's scale never approaches).
_REACTED_SIGNALS = "_reacted_signals"


class ScheduleFromEmailStrategy:
    """Model-backed Reason (the E2 default) plus signal-driven plan invalidation."""

    def __init__(self) -> None:
        self._default = DefaultReasonStrategy()

    async def reason(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        signal_count = len(wm.signals)
        reacted = activity.context.get(_REACTED_SIGNALS, 0)
        if activity.plan is not None and signal_count > reacted:
            # A fresh state-change signal arrived while a plan was in flight -> invalidate it so the
            # default re-plans from the now-updated working memory instead of blindly advancing.
            log.info("reason: state-change signal mid-plan -> re-planning %r", activity.goal)
            activity.plan = None
            activity.step_index = 0
        activity.context[_REACTED_SIGNALS] = signal_count
        return await self._default.reason(activity, wm, cycle, result)
