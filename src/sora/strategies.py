"""One pluggable strategy per phase, threaded through a shared TickResult."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Coroutine
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

from sora.action import (
    CreateActivityAction,
    FilterPerceptionsAction,
    InvokeAction,
    LoadManualAction,
    ResumeAction,
    SuspendAction,
    UnloadManualAction,
)
from sora.activity import ActivityState
from sora.perception import Percept
from sora.types import (
    OPERATION_NAME,
    TOOL_ID,
    CompletedOperation,
    OperationInvocation,
    SignalWait,
    Step,
)

# Bound on retained signals: they're consumption-evicted (a matched signal leaves when its activity
# resumes), but an *orphan* — one that arrives before its waiter, or that nothing ever waits on —
# can't be dropped eagerly without losing the early-arrival window (a completion signal that beats
# its op's ack must survive to the cycle that suspends). Cap the append log so orphans can't grow it
# without bound; the newest win. Deliberately simple, revisited when a real multi-waiter scenario
# needs age- or ownership-based eviction.
_SIGNAL_RETENTION = 256

if TYPE_CHECKING:
    from sora.activity import Activity
    from sora.cycle import DecisionCycle
    from sora.manual import Manual
    from sora.memory import WorkingMemory
    from sora.perception import Message

log = logging.getLogger("sora.strategies")


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
            wm.signals.append(Percept(source, signal, time.time()))
            log.info("observe: signal %s from %s", signal.name, source)
        just_resolved: list[tuple[Activity, OperationInvocation]] = []
        async for invocation_id, ack in cycle.result_sink.drain():
            # Unambiguous 1:1 match: the invoke's own result resolves its activity automatically to
            # READY — manual-agnostic, no strategy involved. The *second* kind of waiting (block on
            # a declared completion signal) is layered on top below, never fused into this resolve.
            for activity in wm.activities.values():
                if activity.pending_operation and activity.pending_operation.id == invocation_id:
                    invocation = activity.pending_operation.invocation
                    op = invocation.operation_name
                    activity.last_operation = ack
                    activity.history.append(
                        CompletedOperation(invocation, ack)
                    )  # belief to ground on
                    activity.pending_operation = None
                    activity.state = ActivityState.READY
                    just_resolved.append((activity, invocation))
                    if ack.ok:
                        log.info("observe: resolved %s -> ok", op)
                    else:
                        # Surface *why*: a failed op terminates the activity in Reflect, and without
                        # the error the trace just says failed with no cause (e.g. a schema error).
                        log.warning("observe: resolved %s -> FAILED: %s", op, _truncate(ack.result))
                    break
        await self._suspend_on_completion_signal(cycle, just_resolved)
        await self._resume_on_signal(cycle)
        # Trim last: a signal that just arrived this tick must survive to be matched by the two
        # passes above before it's ever subject to eviction (bound orphan growth; newest win).
        if len(wm.signals) > _SIGNAL_RETENTION:
            del wm.signals[:-_SIGNAL_RETENTION]
        async for message in cycle.communication.receive():
            wm.messages.append(message)
            log.info("observe: message from %s: %r", message.sender, _goal_from_message(message))
        return TickResult()

    async def _suspend_on_completion_signal(
        self, cycle: DecisionCycle, just_resolved: list[tuple[Activity, OperationInvocation]]
    ) -> None:
        """For each activity whose op just resolved: if the op's manual declares a completion signal
        that hasn't already arrived, suspend the activity until it does. Layered on the automatic
        RUNNING->READY resolve above (a failed op still terminates in Reflect; only a successful,
        signal-declaring op suspends). If the signal already arrived (it beat the ack), stay READY
        without blocking — the two waits compose, they don't deadlock. The signal itself is never
        consumed here: it's left in `wm.signals` for `_resume_on_signal` (or any other activity
        blocked on the same wait, or a strategy reading `wm.signals` directly) to still see it."""
        wm = cycle.working
        suspend = cycle.actions.internal(SuspendAction.name)
        for activity, invocation in just_resolved:
            last = activity.last_operation
            if last is None or not last.ok:  # a failure isn't a completion to wait past
                continue
            completion = self._completion_signal(wm, invocation)
            if completion is None:
                continue
            wait = SignalWait(signal_name=completion, source=invocation.tool_id)
            if self._match_signal(wm, wait) is not None:  # early signal: already satisfied
                log.info("observe: completion signal %s already present", completion)
                continue
            await suspend.execute(cycle, activity_id=activity.id, wait=wait)

    async def _resume_on_signal(self, cycle: DecisionCycle) -> None:
        """For each BLOCKED activity, if an observed signal satisfies its wait, resume it. The
        matched signal is left in `wm.signals` rather than evicted — it's a shared, append-only log
        any other blocked activity (waiting on the identical name+source) or strategy reading it
        directly may still need; only the fixed retention cap (see observe()) ever evicts it."""
        wm = cycle.working
        resume = cycle.actions.internal(ResumeAction.name)
        for activity in wm.activities.values():
            if activity.blocked_on is None:
                continue
            if self._match_signal(wm, activity.blocked_on) is not None:
                await resume.execute(cycle, activity_id=activity.id)

    @staticmethod
    def _completion_signal(wm: WorkingMemory, invocation: OperationInvocation) -> str | None:
        """The completion signal the invoked op declares in its manual, or None (unknown tool,
        unknown op, or a synchronous op)."""
        try:
            tool = wm.registry.get(invocation.tool_id)
        except KeyError:
            return None  # tool left since the invoke — nothing to wait on
        op = tool.manual.operation(invocation.operation_name)
        return op.completion_signal if op is not None else None

    @staticmethod
    def _match_signal(wm: WorkingMemory, wait: SignalWait) -> Percept | None:
        """The first stored signal satisfying `wait` (name equality, plus source when scoped), or
        None. Mechanical — no LLM judgment — since the wait is a manual-declared signal name."""
        for percept in wm.signals:
            if percept.payload.name == wait.signal_name and (
                wait.source is None or percept.source == wait.source
            ):
                return percept
        return None

    @staticmethod
    def _snapshot_properties(wm: WorkingMemory) -> None:
        """Represent observable properties as a replace-by-(source, name) snapshot: one percept per
        property, last value wins. A property is persistent, re-observed state, so re-observing the
        same (source, name) overwrites its entry in the keyed store rather than accumulating — the
        store *is* the snapshot. Signals are the opposite (transient, fire-and-forget) and keep
        append semantics in their own list, handled in observe()."""
        for tool in wm.focused_tools.values():
            for prop in tool.observe():
                wm.properties[(tool.id, prop.name)] = Percept(tool.id, prop, time.time())


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
            log.info("reflect: activity %s failed; storing episode", activity.id)
            self._dispatch(self._record_failure(cycle, activity))
        elif activity.plan is not None and activity.step_index >= len(activity.plan.steps):
            activity.state = ActivityState.TERMINATED
            log.info("reflect: activity %s completed; storing episode + plan", activity.id)
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


# --- parameter grounding: references + a deterministic resolver ----------------------------------
# A plan is a reusable *skeleton*; a param whose value depends on a prior step's result can't be a
# literal at plan time, so the planner emits a *reference* the Reason phase grounds each run against
# the activity's execution history. Two forms (see ADR-0017):
#   hard: {"$from": "<operation_name>", "path": "<dotted path>"}  -> resolved deterministically
#   soft: {"$decide": "<natural-language description>"}           -> always escalates to the model
_REF_FROM = "$from"
_REF_PATH = "path"
_REF_DECIDE = "$decide"
_MISSING = object()  # sentinel: no matching history entry (distinct from a genuine None result)


def _is_reference(value: Any) -> bool:
    return isinstance(value, dict) and (_REF_FROM in value or _REF_DECIDE in value)


def _latest_result(history: list[CompletedOperation], operation_name: str) -> Any:
    """The result of the most recent completed operation with this name, or _MISSING."""
    for completed in reversed(history):
        if completed.invocation.operation_name == operation_name:
            return completed.ack.result
    return _MISSING


def _walk_path(value: Any, path: str) -> Any:
    """Walk a dotted path into a nested result — a numeric segment indexes a list, else a dict."""
    for segment in filter(None, path.split(".")):
        value = value[int(segment)] if segment.isdigit() else value[segment]
    return value


def _manual_for(wm: WorkingMemory, tool_id: str | None) -> Manual | None:
    """The joined tool's manual (the operation schema the model grounds against), or None."""
    if tool_id is None:
        return None
    try:
        return wm.registry.get(tool_id).manual
    except KeyError:
        return None


def resolve_references(
    op_params: dict[str, Any], history: list[CompletedOperation]
) -> tuple[dict[str, Any], list[str]]:
    """Resolve a step's operation params against execution history. Non-reference values pass
    through; a hard reference is resolved deterministically; anything that can't be resolved
    mechanically (soft ref, missing step, bad path) is left in place and its key returned in
    ``unresolved`` for the caller to escalate. Never raises an exception on a bad path — that's
    an escalation signal, not an error."""
    resolved = dict(op_params)
    unresolved: list[str] = []
    for key, value in op_params.items():
        if not _is_reference(value):
            continue
        if _REF_DECIDE in value:
            unresolved.append(key)
            continue
        result = _latest_result(history, value.get(_REF_FROM))
        if result is _MISSING:
            unresolved.append(key)
            continue
        try:
            resolved[key] = _walk_path(result, value.get(_REF_PATH, ""))
        except (KeyError, IndexError, TypeError, ValueError):
            unresolved.append(key)
    return resolved, unresolved


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
                log.info("reason: inferring a plan for %r (%d tools)", activity.goal, len(catalog))
                plan = await cycle.procedural.infer(activity, catalog)  # the model call
                log.info("reason: inferred plan with %d steps", len(plan.steps))
            else:
                log.info(
                    "reason: reusing cached plan (%d steps) for %r", len(plan.steps), activity.goal
                )
            activity.plan = plan
            activity.step_index = 0
        if activity.step_index >= len(activity.plan.steps):
            return result  # exhausted -> no step this cycle
        step = activity.plan.steps[activity.step_index]
        activity.step_index += 1
        grounded = await self._ground(step, activity, wm, cycle)
        return replace(result, activity=activity, step=grounded)

    async def _ground(
        self, step: Step, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle
    ) -> Step:
        """Ground a step's operation params against this run's execution history for *this cycle*,
        leaving the stored plan's references intact so procedural reuse keeps a reusable skeleton.
        Deciding a param value is a *reasoning* act, so it lives here, not in Act (which stays
        mechanistic). Hybrid: resolve references deterministically; escalate to one model call
        (``procedural.ground``) only for what can't be resolved mechanically. A step with
        no references is a pure no-op — the cheap path makes no model call."""
        if step.next_action != InvokeAction.name:
            return step  # only invoke steps carry an operation param bag to ground
        routing = {k: v for k, v in step.params.items() if k in (TOOL_ID, OPERATION_NAME)}
        op_params = {k: v for k, v in step.params.items() if k not in (TOOL_ID, OPERATION_NAME)}
        resolved, unresolved = resolve_references(op_params, activity.history)
        if unresolved:
            log.info(
                "reason: grounding %s params %s via the model", routing[OPERATION_NAME], unresolved
            )
            manual = _manual_for(wm, routing.get(TOOL_ID))
            resolved = await cycle.procedural.ground(
                activity, routing[OPERATION_NAME], manual, resolved
            )
        if resolved == op_params:
            return step  # no references -> unchanged, reuse the original Step
        return replace(step, params={**routing, **resolved})


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
