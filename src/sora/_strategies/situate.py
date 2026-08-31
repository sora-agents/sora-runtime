"""Default Situate strategy and activity selection."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from sora._strategies.contracts import (
    ActivitySelectionStrategy,
    TickResult,
)
from sora._strategies.interaction import (
    _goal_from_message,
)
from sora.action import (
    CreateActivityAction,
    FilterPerceptionsAction,
    LoadManualAction,
    UnloadManualAction,
)
from sora.activity import Activity, ActivityState

if TYPE_CHECKING:
    from sora.cycle import DecisionCycle
    from sora.memory import WorkingMemory


class RoundRobinActivitySelection:
    """Deterministic anti-starvation default: rotate through the ready set by carrying a cursor
    (last-selected activity id) across cycles. Cold start (or when the last pick is no longer ready)
    falls back to ready[0] — the oldest — so behavior matches a static priority-by-age default until
    an activity lingers READY, at which point selection rotates instead of pinning it. Genuine
    cross-cycle state (unlike a stateless default), feasible because the strategy instance persists
    for the agent's lifetime — cf. DefaultReflectStrategy's task set."""

    def __init__(self) -> None:
        self._last_id: str | None = None

    async def select(
        self, ready: list[Activity], wm: WorkingMemory, cycle: DecisionCycle
    ) -> Activity | None:
        if not ready:
            return None
        ids = [a.id for a in ready]
        # Rotate off the last pick; wrap via modulo. Single-ready -> (0+1)%1 == 0 re-picks it (no
        # starvation possible). Last pick gone from the ready set -> restart at the oldest.
        nxt = (ids.index(self._last_id) + 1) % len(ids) if self._last_id in ids else 0
        chosen = ready[nxt]
        self._last_id = chosen.id
        return chosen


class DefaultSituateStrategy:
    """The runtime's built-in default — mechanical, no LLM. Always runs: it adjusts working memory
    for the joined workspaces every cycle (even for an already-selected activity), then selects only
    if result.activity is still None. Creates an activity from any unhandled message (deduped by
    derived goal) via the internal _create_activity_ action, and adjusts wm via the internal
    working-memory actions — loads joined tools' manuals (_load_), unloads manuals no longer backed
    by a joined tool (_unload_), and filters observable-property percepts to the *attended* tools
    (_filter_). _filter_ only prunes properties (a re-observed snapshot, safe to drop); signals are
    retained regardless of source — they're fire-and-forget, and their retention and eviction is
    consumption-driven, owned by the blocked-state machinery, not this prune. Deciding what to
    attend to is *not* done here: it is a subscription to the environment, reconciled in Observe
    against the live intentions, and a plan can still override either way with an
    explicit `focus`/`unfocus` step dispatched as the cycle's one external action (at Act). Which
    ready activity runs is delegated to a pluggable ActivitySelectionStrategy (default
    RoundRobinActivitySelection — fair rotation over the ready set), so a richer scheduler can be
    swapped in without re-authoring the mechanical activity-creation and wm-adjustment above."""

    def __init__(self, selection: ActivitySelectionStrategy | None = None) -> None:
        self._activity_selection = selection or RoundRobinActivitySelection()

    async def situate(
        self,
        activities: list[Activity],
        wm: WorkingMemory,
        cycle: DecisionCycle,
        result: TickResult,
    ) -> TickResult:
        await self._create_activities_from_messages(wm, cycle)
        await self._adjust_working_memory(wm, cycle)
        if result.activity is not None:
            return result  # a pre-set selection is respected, not overridden
        # Recompute from wm (not the passed snapshot) so a just-created activity is selectable now.
        # wm.activities preserves insertion (creation) order and is never reordered, so the ready
        # list is oldest-first; the pick itself is delegated to the selection sub-strategy.
        ready = [a for a in wm.activities.values() if a.state is ActivityState.READY]
        selected = await self._activity_selection.select(ready, wm, cycle)
        return result if selected is None else replace(result, activity=selected)

    @staticmethod
    async def _create_activities_from_messages(wm: WorkingMemory, cycle: DecisionCycle) -> None:
        if wm.messages_cursor >= len(wm.messages):
            return  # nothing new since last processed -> the internal action isn't required
        create = cycle.actions.internal(CreateActivityAction.name)
        goals = {a.goal for a in wm.activities.values()}
        for message in wm.messages[wm.messages_cursor :]:  # only messages not yet routed/claimed
            goal = _goal_from_message(message)
            if goal not in goals:  # an unhandled message maps to no existing activity (by goal)
                await create.execute(cycle, goal=goal)
                goals.add(goal)
        wm.messages_cursor = len(wm.messages)  # claim the batch -> each message handled once

    @staticmethod
    async def _adjust_working_memory(wm: WorkingMemory, cycle: DecisionCycle) -> None:
        tools = wm.registry.all_tools()
        manual_ids = {tool.manual.id for tool in tools}
        # Manuals track the joined workspaces; percepts track the narrower ATTENDED set. Only a
        # focused tool is re-observed, so the moment attention narrows a released tool's
        # snapshot is frozen and misleading — this is the housekeeping backstop that drops it even
        # if `release` did not. Signals ignore this set: _filter_ never drops them.
        relevant_ids = set(wm.focused_tools)
        load = cycle.actions.internal(LoadManualAction.name)
        unload = cycle.actions.internal(UnloadManualAction.name)
        filter_ = cycle.actions.internal(FilterPerceptionsAction.name)
        for manual_id in manual_ids - wm.loaded_manuals.keys():
            await load.execute(cycle, manual_id=manual_id)
        for manual_id in wm.loaded_manuals.keys() - manual_ids:
            await unload.execute(cycle, manual_id=manual_id)
        await filter_.execute(cycle, tool_ids=relevant_ids)
