"""One pluggable strategy per phase, threaded through a shared TickResult."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

from sora.action import (
    CreateActivityAction,
    FilterPerceptionsAction,
    LoadManualAction,
    UnloadManualAction,
)
from sora.activity import ActivityState
from sora.perception import Percept, PerceptKind
from sora.types import OPERATION_NAME, TOOL_ID, OperationInvocation, Step

if TYPE_CHECKING:
    from sora.activity import Activity
    from sora.cycle import DecisionCycle
    from sora.manual import Manual
    from sora.memory import WorkingMemory
    from sora.perception import Message


@dataclass(frozen=True)
class TickResult:
    """The decision surface for one cycle. Every phase strategy receives and returns one of these.
    Whatever's still None, DecisionCycle fills in by calling the next phase's own strategy — so a
    fully-decomposed configuration produces one field at a time, and a fused Situate can fill in
    step/invocation too, deciding the rest of the cycle in one call. Lives only for the duration of
    one tick() call — nothing persists across cycles, so there's no cache to key or invalidate.

    A freeform per-tick scratchpad for multi-call strategy configurations (e.g. a fused Situate
    passing notes to a separate, focused Act) is a foreseen addition, deferred until the first such
    configuration actually exists."""

    activity: Activity | None = None
    step: Step | None = None  # this cycle's decision — not the whole (possibly multi-step) Plan
    invocation: OperationInvocation | None = None


class ObserveStrategy(Protocol):
    async def observe(self, cycle: DecisionCycle) -> TickResult:
        """Mutates cycle.working (perceptions, messages) as a side effect — same as the default
        below. Default: mechanical, no model call, returns an empty TickResult(). An LLM-backed
        Observe is for interpreting raw perception itself (e.g. describing a camera snapshot), not
        for deciding the cycle — decision-chain fusion starts at Situate, not here."""
        ...


class ReflectStrategy(Protocol):
    async def reflect(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        """Decides whether this activity just completed or failed — deterministic or model-backed,
        depending on the application — and if so, summarizes and stores to episodic memory. On
        success, also stores activity.plan via cycle.procedural.store() so future activities with a
        similar goal can reuse it; on failure, it isn't stored. The completion judgment is
        synchronous — it must land before Situate selects, so a just-completed activity is never
        re-selected the same cycle — while the summarize/store side effects are dispatched
        asynchronously and never block the cycle; several activities may terminate in the same
        cycle. Passes `result` through, optionally adding to it. Default: performs the completion
        check and the store-on-success, leaves TickResult's other fields untouched. `cycle` is what
        makes these memory calls possible at all — previously missing from this Protocol despite
        the calls it was already documented as making."""
        ...


class SituateStrategy(Protocol):
    async def situate(
        self,
        activities: list[Activity],
        wm: WorkingMemory,
        cycle: DecisionCycle,
        result: TickResult,
    ) -> TickResult:
        """Selects the next activity and adjusts wm for it. Always runs — unlike Reason/Act it is
        not gated on its own output field, because adjusting wm (selecting tools, loading/unloading
        manuals, filtering percepts) must reflect this cycle's fresh percepts even for an
        already-selected activity. Selects only if result.activity is still None; a pre-set
        selection (uncommon — e.g. an Observe that pins the activity handling a critical signal) is
        respected and situated, not overridden. Also responsible for activity creation: if
        wm.messages includes a new goal delegation, invokes the internal _create_activity_ action
        (via cycle) before selecting. Head of the decision chain (Situate -> Reason -> Act) and the
        intended entry point for fusing the remaining phases into one model call — it runs after
        this cycle's percepts and messages are already in working memory. May additionally fill in
        step/invocation, short-circuiting Reason/Act (those forward-fusion gates remain; only
        Situate's own activity gate is removed)."""
        ...


class ActivitySelectionStrategy(Protocol):
    async def select(
        self, ready: list[Activity], wm: WorkingMemory, cycle: DecisionCycle
    ) -> Activity | None:
        """Picks the activity to progress this cycle from the ready set (empty -> None). A
        scheduling policy, not a phase: it decides *which* ready activity runs, nothing else — the
        caller (Situate) folds the pick into TickResult; fusing step/invocation stays a full
        SituateStrategy concern. `async` + the `cycle` handle are for a richer policy (priority,
        aging, deadlines, or an LLM-based scheduler) that consults memory or a model; the mechanical
        default consults neither."""
        ...


class ReasonStrategy(Protocol):  # pluggable; default targets 1 LLM call/cycle
    async def reason(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        """Only called if result.step is still None. Typical implementation: if activity.plan is
        already set and still valid, just read activity.plan.steps[activity.step_index] and
        advance the index — no model call. Otherwise, retrieve a cached Plan via
        cycle.procedural.retrieve() or infer a new one (the expensive path), reset step_index to
        0, and use its first Step. Deciding when a plan counts as invalidated is entirely up to
        the implementation. May additionally fill in invocation, short-circuiting Act — this is
        where the historical 'tool hallucination' risk lives if it does."""
        ...


class ActStrategy(Protocol):
    async def bind(
        self, step: Step, manual: Manual | None, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        """Only called if result.invocation is still None. This is *parameter binding*: grounding
        an abstract Step into a concrete, schema-conformant OperationInvocation (the tool-
        hallucination-prone step — where "email the boss" becomes validated `{to, subject, ...}`).
        Distinct from a *protocol binding* (WoT forms/security, an MCP session), which is how the
        adapter's Tool actually reaches the instance and never surfaces here — see ADR-0015. `cycle`
        is available for implementations that cache bindings (e.g. belief-state -> params) rather
        than re-deriving one every time."""
        ...


@dataclass(frozen=True)
class Strategies:  # bundles the five, so DecisionCycle.__init__ doesn't take five loose params
    observe: ObserveStrategy
    reflect: ReflectStrategy
    situate: SituateStrategy
    reason: ReasonStrategy
    act: ActStrategy


class DefaultObserveStrategy:
    """The runtime's built-in default — purely mechanical, no LLM."""

    async def observe(self, cycle: DecisionCycle) -> TickResult:
        wm = cycle.working
        self._snapshot_properties(wm)
        async for source, signal in cycle.signal_sink.drain():
            wm.perceptions.append(Percept(source, PerceptKind.SIGNAL, signal, time.time()))
        async for invocation_id, ack in cycle.result_sink.drain():
            # Unambiguous 1:1 match: the invoke's own result resolves its activity automatically,
            # never as a Percept, no strategy involved (see Activities in README).
            for activity in wm.activities.values():
                if activity.pending_operation and activity.pending_operation.id == invocation_id:
                    activity.last_operation = ack
                    activity.pending_operation = None
                    activity.state = ActivityState.READY
                    break
        async for message in cycle.communication.receive():
            wm.messages.append(message)
        return TickResult()

    @staticmethod
    def _snapshot_properties(wm: WorkingMemory) -> None:
        """Represent observable properties as a replace-by-(source, name) snapshot: one percept per
        property, last value wins. A property is persistent, re-observed state, so re-observing the
        same (source, name) overwrites its percept in place rather than appending — otherwise
        wm.perceptions would grow unbounded with stale duplicate values every cycle. Signals are the
        opposite (transient, fire-and-forget) and keep append semantics, handled in observe()."""
        index = {
            (p.source, p.payload.name): i
            for i, p in enumerate(wm.perceptions)
            if p.kind is PerceptKind.PROPERTY
        }
        for tool in wm.focused_tools.values():
            for prop in tool.observe():
                percept = Percept(tool.id, PerceptKind.PROPERTY, prop, time.time())
                key = (tool.id, prop.name)
                if key in index:
                    wm.perceptions[index[key]] = percept  # replace in place, position preserved
                else:
                    index[key] = len(wm.perceptions)
                    wm.perceptions.append(percept)


class DefaultReflectStrategy:
    """The runtime's built-in default — purely mechanical, no LLM.

    Judges each activity completed or failed by two deterministic rules, and on a terminal outcome
    records the experience: the state transition is synchronous (so Situate, which runs later this
    cycle and selects only READY activities, never re-selects a just-terminated one), while the
    episodic/procedural writes are dispatched as background tasks and never block the cycle (several
    activities may terminate in the same cycle). Strong refs to the in-flight tasks are held so they
    aren't GC'd mid-write — the same pattern as InvokeAction.

    The two rules are deliberately asymmetric. **Failure** fires on any resolved-but-not-ok
    ``last_operation``, independent of the plan: a failed operation is definite negative evidence,
    so the activity terminates even mid-plan. **Completion** requires positive evidence that all
    planned work is done — a plan present and fully consumed (``step_index >= len(plan.steps)``) —
    so a plan-less activity is never auto-completed here (what a plan-following Reason, and any
    application driving activities without a plan, relies on). Only completion stores the plan to
    procedural memory; a failed plan isn't something future activities should retrieve as procedrual
    knowledge."""

    def __init__(self) -> None:
        # Hold strong refs to in-flight background stores so they aren't GC'd mid-write.
        self._tasks: set[asyncio.Task[None]] = set()

    async def reflect(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        # Only READY activities are judged: RUNNING has an operation still in flight (nothing to
        # judge yet), BLOCKED is waiting on a signal, and TERMINATED was already recorded — skipping
        # it is what makes reflect() idempotent across the cycles it runs on every activity.
        if activity.state is not ActivityState.READY:
            return result
        last = activity.last_operation
        if last is not None and not last.ok:
            activity.state = ActivityState.TERMINATED  # synchronous — Situate sees it this cycle
            self._dispatch(self._record_failure(cycle, activity))
        elif activity.plan is not None and activity.step_index >= len(activity.plan.steps):
            activity.state = ActivityState.TERMINATED
            self._dispatch(self._record_success(cycle, activity))
        # Reflect never fills in the decision fields (activity/step/invocation) — it threads
        # `result` through untouched.
        return result

    def _dispatch(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _record_success(self, cycle: DecisionCycle, activity: Activity) -> None:
        await cycle.episodic.learn(activity, _summarize(activity, succeeded=True), succeeded=True)
        if activity.plan is not None:
            await cycle.procedural.store(activity.plan)

    async def _record_failure(self, cycle: DecisionCycle, activity: Activity) -> None:
        await cycle.episodic.learn(activity, _summarize(activity, succeeded=False), succeeded=False)


def _summarize(activity: Activity, *, succeeded: bool) -> str:
    """A deterministic, no-LLM episode summary. A model-backed ReflectStrategy would substitute a
    richer natural-language summary here; the mechanical default just states outcome and goal."""
    outcome = "completed" if succeeded else "failed"
    return f"{outcome}: {activity.goal}"


def _goal_from_message(message: Message) -> str:
    """The default's deterministic goal derivation from a message — no interpretation, no model
    call: the message's own text if it carries a conventional ``text`` field, else the whole content
    rendered. A model-backed Situate would derive a richer goal instead."""
    text = message.content.get("text")
    return text if isinstance(text, str) else str(message.content)


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
    by a joined tool (_unload_), and filters observable-property percepts to the joined workspaces'
    tools (_filter_). _filter_ only prunes properties (a re-observed snapshot, safe to drop);
    signals are retained regardless of source — they're fire-and-forget, and their retention and
    eviction is consumption-driven, owned by the blocked-state machinery, not this prune. Focusing
    tools is *not* done here: _focus_ is an external action, and the cycle dispatches at most one
    external action per cycle (at Act), so a richer strategy emits focus as a plan step. Which ready
    activity runs is delegated to a pluggable ActivitySelectionStrategy (default
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
        if not wm.messages:
            return  # nothing to handle -> the internal action is never required this cycle
        create = cycle.actions.internal(CreateActivityAction.name)
        goals = {a.goal for a in wm.activities.values()}
        for message in wm.messages:
            goal = _goal_from_message(message)
            if goal not in goals:  # an unhandled message maps to no existing activity (by goal)
                await create.execute(cycle, goal=goal)
                goals.add(goal)

    @staticmethod
    async def _adjust_working_memory(wm: WorkingMemory, cycle: DecisionCycle) -> None:
        tools = wm.registry.all_tools()
        manual_ids = {tool.manual.id for tool in tools}
        # Relevant = the tools of the joined workspaces. focused_tools is a subset — per A&A you can
        # only focus a tool discovered by joining its workspace (FocusAction resolves it through the
        # registry) — so it adds nothing here. Signals ignore this set: _filter_ never drops them.
        relevant_ids = {tool.id for tool in tools}
        load = cycle.actions.internal(LoadManualAction.name)
        unload = cycle.actions.internal(UnloadManualAction.name)
        filter_ = cycle.actions.internal(FilterPerceptionsAction.name)
        for manual_id in manual_ids - wm.loaded_manuals.keys():
            await load.execute(cycle, manual_id=manual_id)
        for manual_id in wm.loaded_manuals.keys() - manual_ids:
            await unload.execute(cycle, manual_id=manual_id)
        await filter_.execute(cycle, tool_ids=relevant_ids)


class DefaultReasonStrategy:
    """The runtime's Reason default. Reason is the one phase with no *mechanical* default —
    planning inherently needs a model — so this is deterministic orchestration around the single
    model call, which is isolated in ``ProceduralMemory.infer``:

    * an activity that already has a plan with steps left is the cheap path — read the current step
      and advance ``step_index``: no model call, no procedural lookup;
    * an activity with no plan gets one by *reuse* first (``procedural.retrieve`` — a plan some past
      activity with this goal actually followed) and only *infers* a fresh one on a miss, passing
      the currently-joined tools (id -> Manual) as the planning catalog (the strategy holds the live
      registry; a memory module never reaches into the environment itself);
    * an exhausted plan yields no step — the cycle returns, and Reflect terminates the activity the
      next cycle on the same "plan present and fully consumed" rule (so this branch is normally only
      reached by a just-inferred empty plan).

    Mutates the activity (plan/step_index) in place, like the other phase defaults. The model itself
    lives behind ``ProceduralMemory.infer``; this strategy makes zero model calls on the cheap
    path."""

    async def reason(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        if activity.plan is None:
            plan = await cycle.procedural.retrieve(activity)  # reuse across runs (cheap)
            if plan is None:
                catalog = {tool.id: tool.manual for tool in wm.registry.all_tools()}
                plan = await cycle.procedural.infer(activity, catalog)  # the model call
            activity.plan = plan
            activity.step_index = 0
        if activity.step_index >= len(activity.plan.steps):
            return result  # exhausted -> no step this cycle
        step = activity.plan.steps[activity.step_index]
        activity.step_index += 1
        return replace(result, activity=activity, step=step)


class DefaultActStrategy:
    """The mechanical, no-LLM default for *parameter binding* (not protocol binding — see
    ActStrategy): bind an ``invoke`` Step straight to an OperationInvocation, splitting the
    tool_id/operation_name routing keys out of the operation's own params. A model-backed
    ActStrategy would instead ground under-specified params against the manual's schema here; the
    default assumes the Step already carries concrete params, so binding is just the key-split."""

    async def bind(
        self, step: Step, manual: Manual | None, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        params = {k: v for k, v in step.params.items() if k not in (TOOL_ID, OPERATION_NAME)}
        invocation = OperationInvocation(
            tool_id=step.params[TOOL_ID],
            operation_name=step.params[OPERATION_NAME],
            params=params,
        )
        return replace(result, invocation=invocation)
