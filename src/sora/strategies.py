"""One pluggable strategy per phase, threaded through a shared TickResult."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Coroutine
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

from sora.action import (
    CreateActivityAction,
    FilterPerceptionsAction,
    GroundAction,
    InferAction,
    InvokeAction,
    LoadManualAction,
    ResumeAction,
    SendAction,
    SuspendAction,
    UnloadManualAction,
)
from sora.activity import ActivityState
from sora.llm import log_llm_discarded
from sora.memory import PerceptSnapshot, step_from_raw
from sora.perception import Percept
from sora.types import (
    OPERATION_NAME,
    SUBGOAL,
    TOOL_ID,
    USER_STOP,
    CompletedOperation,
    InputWait,
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
    from sora.types import InterruptRequest, Signal

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
        depending on the application — and if so, summarizes and stores to episodic memory. (The
        default does NOT auto-cache the completed plan to procedural memory — replaying a stored
        plan verbatim is unsound; distilling reusable procedures from episodes is future work.) The
        completion judgment is
        synchronous — it must land before Situate selects, so a just-completed activity is never
        re-selected the same cycle — while the summarize/store side effects are dispatched
        asynchronously and never block the cycle; several activities may terminate in the same
        cycle. Passes `result` through, optionally adding to it. Default: performs the completion
        check and the store-on-success, leaves TickResult's other fields untouched. `cycle` is what
        makes these memory calls possible at all — previously missing from this Protocol despite
        the calls it was already documented as making."""
        ...

    def failed(self, activity: Activity) -> bool:
        """This strategy's own judgment of whether `activity` has failed — a *judgment call*, not
        a fact recorded on `Activity` itself, since a different ReflectStrategy may define failure
        differently (e.g. from a signal, or a partial-success rule) than the default's "resolved
        operation, not ok" rule. Exposed on the Protocol (not just the default) so callers outside
        the decision cycle — a reporting hook, a test assertion — go through whichever strategy is
        actually configured rather than re-deriving the rule themselves."""
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


class InterruptPolicy(Protocol):
    def decide(self, source: str, signal: Signal, wm: WorkingMemory) -> InterruptRequest | None:
        """Consulted synchronously the instant a signal is pushed to signal_sink — before the once-
        per-cycle Observe drain. Return an InterruptRequest to preempt the current phase (a hard
        interrupt), or None to let the signal flow cooperatively (reacted to at the next cycle
        boundary). Sync because push is sync. A stateful policy may diff the signal vs remembered
        state (e.g. a set of inbox ids) to fire only on a new external event and filter the
        agent's own writes — the only distinguishable-external test available until read-write/
        efference tagging lands. Default: NeverInterruptPolicy — no signal ever preempts."""
        ...


class NeverInterruptPolicy:
    """The runtime default: a pushed signal never becomes an interrupt, so the cooperative signal
    path (drained in Observe, resumes a BLOCKED activity) is unchanged. With no runtime way yet to
    tell the agent's own writes from external events, preempting on a signal would risk a self-write
    loop; opting in is a deliberate, application-supplied policy."""

    def decide(self, source: str, signal: Signal, wm: WorkingMemory) -> InterruptRequest | None:
        return None


class InterruptHandler(Protocol):
    async def handle(
        self, request: InterruptRequest, wm: WorkingMemory, cycle: DecisionCycle
    ) -> bool:
        """Runs after tick() aborts on a pending interrupt — the process-scheduling 'interrupt
        handler'. The interrupted activity's context is already saved (durable on Activity; the per-
        tick TickResult was discarded, immune to interrupt staleness per ADR-0011), so this only
        decides the follow-up: map each targeted activity onto an existing state — READY (resume, or
        replan by clearing plan/step_index), BLOCKED via InputWait (await the user's next
        instruction), or TERMINATED (drop) — then the ActivitySelectionStrategy picks
        next. Never abandons an in-flight *external* op (side effects): a RUNNING activity is left
        RUNNING and revisited at the next checkpoint once its ack resolves. Returns True once every
        targeted activity is routed (the request is discharged and cleared), False when some are
        still RUNNING and the request must be revisited next tick."""
        ...


class DefaultInterruptHandler:
    """The runtime default: a user stop. Pauses each targeted, schedulable (READY) activity to a
    resumable point via an InputWait, so the agent halts current work but stays alive; a later user
    Message resumes it (DefaultObserveStrategy._resume_on_input). An activity mid-external-op
    (RUNNING on a pending_operation) is left to finish and routed on a later checkpoint, so a
    physical side effect always runs to completion. An activity RUNNING only on an off-cycle
    infer/ground (pending_inference) has no side effect to protect, so it is paused *now* — its
    inference invalidated (discarded on resolve) — rather than waiting out an unbounded model call
    (ADR-0021). target=None is agent-wide; a named target pauses just that activity.

    It is a *user-stop* handler, not a general router: it recognizes only the USER_STOP signal and
    treats any other interrupt as unrouted — pausing to await human instruction is the fail-safe
    fallback (halt-and-ask, never barrel ahead), logged at warning level since no handler claimed
    it. There is no general runtime answer for an arbitrary interrupt signal; that is why a custom
    InterruptPolicy should ship a paired InterruptHandler for its own signals (see ADR-0020). With
    the default components only a CLI /stop ever reaches here."""

    async def handle(
        self, request: InterruptRequest, wm: WorkingMemory, cycle: DecisionCycle
    ) -> bool:
        if request.signal.name != USER_STOP:
            # Unrouted interrupt: a custom policy raised it but no paired handler claimed it. Fall
            # back to the same halt-to-await-input as a user stop (fail-safe), but log it — a silent
            # strand (an activity blocked on input no user message ever arrives to satisfy, e.g.
            # headless) is otherwise invisible. Still routed, not dropped: swallowing an interrupt a
            # policy deliberately raised would be as surprising as stranding on it.
            log.warning(
                "no interrupt handler routed signal %r; pausing targeted activities to await input "
                "as a fallback (a custom InterruptPolicy should ship a paired InterruptHandler)",
                request.signal.name,
            )
        if request.target is None:
            targets = list(wm.activities.values())
        else:
            target = wm.activities.get(request.target)
            targets = [target] if target is not None else []
        pending = False
        for activity in targets:
            if activity.state is ActivityState.RUNNING and activity.pending_operation is not None:
                pending = True  # external op in flight: let it finish, route on a later checkpoint
            elif activity.state in (ActivityState.READY, ActivityState.RUNNING):
                # READY, or RUNNING only on a side-effect-free inference: drop the inference (no
                # reason to wait out an unbounded model call) and pause to await the user's next
                # instruction.
                activity.discard_inference()
                activity.state = ActivityState.BLOCKED
                activity.blocked_on = InputWait(prompt=request.signal.name)
        return not pending


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
                # Guarded on RUNNING: a late ack for an activity a hard interrupt already routed
                # away (paused to BLOCKED/InputWait, or dropped to TERMINATED) must not resurrect it
                # to READY. The in-flight external op was allowed to finish; its result is just no
                # longer awaited. A RUNNING activity still resolves normally (the interrupt is
                # honored on the checkpoint *after* this resolve — see DefaultInterruptHandler).
                if (
                    activity.pending_operation
                    and activity.pending_operation.id == invocation_id
                    and activity.state is ActivityState.RUNNING
                ):
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
                        log.debug("observe: %s result\n%r", op, ack.result)
                    else:
                        # Surface *why*: a failed op terminates the activity in Reflect, and without
                        # the error the trace just says failed with no cause (e.g. a schema error).
                        log.warning("observe: resolved %s -> FAILED: %s", op, _truncate(ack.result))
                    break
        await self._resolve_inferences(cycle)
        await self._suspend_on_completion_signal(cycle, just_resolved)
        await self._resume_on_signal(cycle)
        # Trim last: a signal that just arrived this tick must survive to be matched by the two
        # passes above before it's ever subject to eviction (bound orphan growth; newest win).
        if len(wm.signals) > _SIGNAL_RETENTION:
            del wm.signals[:-_SIGNAL_RETENTION]
        received_message = False
        async for message in cycle.communication.receive():
            wm.messages.append(message)
            received_message = True
            log.info("observe: message from %s: %r", message.sender, _goal_from_message(message))
        if received_message and self._resume_on_input(wm):
            # A resume consumed this batch as reconsideration input for the resumed activity —
            # claim it so Situate won't also mint a ghost activity from the same follow-up (the
            # double-duty bug). A normal message (no InputWait) is turned into a goal as before.
            wm.messages_cursor = len(wm.messages)
        return TickResult()

    @staticmethod
    async def _resolve_inferences(cycle: DecisionCycle) -> None:
        """Drain the off-cycle infer()/ground() results and apply each to the activity still RUNNING
        on it. An unambiguous 1:1 match on pending_inference.id, resolved to READY — never a Percept
        (deliberation output, not observed state — ADR-0019/0021), so it never touches the
        perception path. `kind == "plan"` lands the Plan and resets step_index; `"ground"` parks
        the resolved params on grounded_params for Reason's next pass to consume. A result carrying
        an `error` (the model call raised) terminates the activity instead of stranding it RUNNING
        forever — the failure surfaces, cycle-synchronized, the way a failed op does. Stale results
        are discarded by the same guard the external-op late-ack uses: a result whose id no longer
        matches the live pending_inference (an interrupt handler re-routed or re-inferred the
        activity), or whose activity is no longer RUNNING, is dropped — the background call ran to
        completion (an LLM call can't be cut mid-generation) but its result is no longer wanted."""
        wm = cycle.working
        async for inf_id, res in cycle.inference_sink.drain():
            for activity in wm.activities.values():
                if (
                    activity.pending_inference is not None
                    and activity.pending_inference.id == inf_id
                    and activity.state is ActivityState.RUNNING
                ):
                    kind = activity.pending_inference.kind
                    activity.pending_inference = None
                    if res.error is not None:
                        activity.grounded_params = None
                        activity.state = ActivityState.TERMINATED
                        log.error(
                            "observe: %s for activity %s failed (%s) -> terminated",
                            kind,
                            activity.id,
                            res.error,
                        )
                    elif kind == "plan":
                        activity.plan = res.value  # type: ignore[assignment]  # kind=="plan" => Plan
                        activity.step_index = 0
                        activity.state = ActivityState.READY
                        log.info("observe: resolved inferred plan for activity %s", activity.id)
                    elif kind == "subgoal":
                        # A mid-plan sub-goal's synthesized sub-plan: push the parent frame (its
                        # plan + the sub-goal's step_index) and enter the sub-plan, so Reason
                        # advances it and pops back to the parent when it exhausts (ADR-0022). It
                        # lands like a top-level plan, only onto a stacked frame not the activity.
                        frame = (activity.plan, activity.step_index)
                        activity.parent_frames.append(frame)  # type: ignore[arg-type]  # plan set mid-plan
                        activity.plan = res.value  # type: ignore[assignment]  # kind=="subgoal" => Plan
                        activity.step_index = 0
                        activity.state = ActivityState.READY
                        log.info("observe: entered sub-plan for activity %s", activity.id)
                    else:
                        activity.grounded_params = res.value  # type: ignore[assignment]  # => dict
                        activity.state = ActivityState.READY
                        log.info("observe: resolved grounded params for activity %s", activity.id)
                    break
            else:
                # No live activity claimed this result: it was invalidated (an interrupt re-routed
                # the activity) or superseded (a re-inference gave a new id). The background model
                # call ran to completion and was already metered, so its cost is real but wasted —
                # tell the meter to move it to the wasted bucket (a no-op when uninstrumented).
                log_llm_discarded(inf_id)

    @staticmethod
    def _resume_on_input(wm: WorkingMemory) -> bool:
        """A user Message satisfies an InputWait: any activity a hard interrupt paused (a user stop)
        returns to READY, so the decision cycle can reconsider it with the new instruction now in
        working memory. The mirror of _resume_on_signal, but the awaited stimulus is inbound user
        input rather than a tool signal — mechanical, no judgment. The plan is *cleared* on resume
        (not kept) so Reason re-infers with the follow-up message and the executed history visible —
        a bare resume would keep advancing the stale plan and never see the instruction. Returns
        whether any activity was resumed, so the caller can claim the message batch as
        reconsideration input rather than letting Situate mint a ghost activity from it."""
        resumed = False
        for activity in wm.activities.values():
            if isinstance(activity.blocked_on, InputWait):
                activity.blocked_on = None
                # Route the plan-drop through the single funnel (clears plan/step_index, the whole
                # intention stack, and any parked/in-flight deliberation) so Reason re-infers with
                # the follow-up instruction and executed history visible — a bare resume would keep
                # advancing the stale plan and never see it. blocked_on and the unconditional READY
                # stay here: resume-specific, not plan invalidation (cf. the _resume_ action).
                activity.reset_for_replan()
                activity.state = ActivityState.READY
                resumed = True
        return resumed

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
            # Only a SignalWait is satisfied by an observed signal; an InputWait waits on a user
            # Message and is resumed in _resume_on_input, not here.
            if not isinstance(activity.blocked_on, SignalWait):
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
    application driving activities without a plan, relies on). Both outcomes record an episode;
    neither auto-caches the plan to procedural memory — replaying a stored plan verbatim is unsound,
    so plan storage is disabled until reusable procedures are distilled from episodes (Reason still
    consults ``procedural.retrieve``, which simply finds nothing until then)."""

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
        if self.failed(activity):
            activity.state = ActivityState.TERMINATED  # synchronous — Situate sees it this cycle
            log.info("reflect: activity %s failed; storing episode", activity.id)
            self._dispatch(self._record_failure(cycle, activity))
        elif (
            activity.plan is not None
            and activity.step_index >= len(activity.plan.steps)
            and not activity.parent_frames
        ):
            # Complete only when the *top-level* plan is exhausted: a just-exhausted sub-plan still
            # has parent frames to pop (Reason does that next cycle), so it isn't done (ADR-0022).
            activity.state = ActivityState.TERMINATED
            log.info("reflect: activity %s completed; storing episode", activity.id)
            self._dispatch(self._record_success(cycle, activity))
        # Reflect never fills in the decision fields (activity/step/invocation) — it threads
        # `result` through untouched.
        return result

    def failed(self, activity: Activity) -> bool:
        """The default rule: a resolved-but-not-ok last_operation is definite negative evidence,
        independent of the plan (see the class docstring's "asymmetric" rules)."""
        return activity.last_operation is not None and not activity.last_operation.ok

    def _dispatch(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _record_success(self, cycle: DecisionCycle, activity: Activity) -> None:
        # Records the episode only. The completed plan is deliberately NOT stored to procedural
        # memory: auto-caching a plan and replaying it verbatim is unsound (a corrected or
        # observation-coupled plan is not reusable). Distilling reusable procedures from episodes is
        # future work; cycle.procedural.store stays available for that deliberate step.
        await cycle.episodic.learn(activity, _summarize(activity, succeeded=True), succeeded=True)

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
# The named-binding read token (ADR-0022). Distinct from $from (which reads Activity.history): $bind
# reads a named binding — here, the current loop element of a mechanical sub-goal, substituted
# eagerly at fan-out time. (Binding names from a plan's context guard are a later addition.)
_REF_BIND = "$bind"
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


# --- sub-goals: mechanical fan-out over a collection (ADR-0022) -----------------------------------
# A `subgoal` Step with mode="mechanical" is expanded in Reason into one concrete step per element
# of a run-time collection — the count is len(data), not a model guess (the RentAFlat "for each"
# fix). _SUBGOAL_RUNNING / _SUBGOAL_SPLICED are the two outcomes _subgoal reports to reason():
# a deliberative sub-goal fired _infer_ and is RUNNING (return, no step); a mechanical one spliced
# its expansion into the plan in place (re-loop and read the first expanded step); a deliberative
# one the loop-guard refused pauses the activity to await input (no step) -> _SUBGOAL_HALTED.
_SUBGOAL_RUNNING = object()
_SUBGOAL_SPLICED = object()
_SUBGOAL_HALTED = object()

# Circuit breaker for runaway deliberative sub-goal recursion (ADR-0022's deferred overflow valve,
# pulled forward). Synthesis-as-selection has no termination guarantee an *authored* plan library
# has: the model can satisfy "plan for goal G" by emitting a plan whose body is another deliberative
# sub-goal for ~G, deferring instead of reducing, and recurse until a budget (or credit) runs out.
# Two mechanical detectors, tripped before the _infer_ spend: a depth cap on the intention stack,
# and goal-similarity against the ancestor sub-goals — a new sub-goal that closely repeats one still
# on the stack is not reducing. Tripping pauses to await-input (ADR-0020) rather than terminating,
# so a deep-but-legitimate task can be redirected, not killed. Both are coarse backstops; the real
# fix is making the common map/filter/distinct shapes expressible without deliberation at all.
_MAX_SUBGOAL_DEPTH = 8
_SUBGOAL_GOAL_SIMILARITY = 0.7


def _goal_token_similarity(a: str, b: str) -> float:
    """Order-independent token Jaccard over two goal strings — 1.0 identical, 0.0 disjoint. Cheap
    and deterministic (no model call): enough to catch a sub-goal that re-states an ancestor's goal
    in reworded form, which is how the non-reducing recursion manifests."""
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _ancestor_subgoal_goals(activity: Activity) -> list[str]:
    """The goals of the deliberative sub-goals still suspended on the intention stack — each parent
    frame's ``(plan, idx)`` points back at the ``subgoal`` step that pushed it. The root
    ``activity.goal`` is deliberately excluded: the first decomposition legitimately shares its
    vocabulary, so comparing against it would false-trip a single, valid refinement."""
    goals: list[str] = []
    for plan, idx in activity.parent_frames:
        if 0 <= idx < len(plan.steps):
            goal = plan.steps[idx].params.get("goal")
            if isinstance(goal, str):
                goals.append(goal)
    return goals


def _resolve_collection(ref: Any, history: list[CompletedOperation]) -> list[Any] | None:
    """The list a mechanical sub-goal iterates: a ``$from`` reference resolved against history (the
    common case — the collection is a prior step's result), or a literal list. ``None`` when it
    can't be resolved to a list (missing step, bad path, or a non-list value) — the caller treats
    that as an empty fan-out, mirroring ``resolve_references``' never-raise contract."""
    if not _is_reference(ref):
        return ref if isinstance(ref, list) else None
    result = _latest_result(history, ref.get(_REF_FROM))
    if result is _MISSING:
        return None
    try:
        value = _walk_path(result, ref.get(_REF_PATH, ""))
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return value if isinstance(value, list) else None


def _substitute_bindings(obj: Any, name: str, element: Any) -> Any:
    """Replace every ``{"$bind": name, "path": ...}`` in a template with the value at that path of
    the current loop ``element``, recursively. Only the named binding is substituted;
    ``$from``/``$decide`` references (and a ``$bind`` for a different name) pass through untouched,
    to be grounded later by the ordinary Reason path. A path that doesn't resolve substitutes a
    ``None`` — which the Act required-param guard skips, not a literal ``$bind`` dict reaching the
    tool."""
    if isinstance(obj, dict):
        if obj.get(_REF_BIND) == name:
            try:
                return _walk_path(element, obj.get(_REF_PATH, ""))
            except (KeyError, IndexError, TypeError, ValueError):
                log.warning("subgoal: $bind path %r did not resolve against %r", obj, element)
                return None
        return {k: _substitute_bindings(v, name, element) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_bindings(v, name, element) for v in obj]
    return obj


def _expand_mechanical(step: Step, history: list[CompletedOperation]) -> list[Step]:
    """Fan a mechanical sub-goal out to one concrete ``Step`` per element of its ``in`` collection,
    the element substituted for ``{"$bind": "<as>"}`` in its ``template``. Empty (or an unresolvable
    collection) -> no steps, so the sub-goal simply vanishes from the plan."""
    ref = step.params.get("in")
    elements = _resolve_collection(ref, history)
    if elements is None:
        # Unresolvable (vs. a genuinely empty list): the `in` reference points at an op that never
        # ran or a value that isn't a list — most likely a plan bug (no narrowing search ran first),
        # so surface it rather than silently doing nothing, as an empty list legitimately would.
        log.warning("subgoal: collection %r did not resolve to a list; expanding to nothing", ref)
        return []
    if not elements:
        return []
    loop_var = step.params.get("as", "")
    template = step.params.get("template", {})
    return [
        step_from_raw(_substitute_bindings(template, loop_var, element)) for element in elements
    ]


class DefaultReasonStrategy:
    """The runtime's Reason default. Reason is the one phase with no *mechanical* default —
    planning inherently needs a model — so this is deterministic orchestration around the single
    model call, which is isolated in ``ProceduralMemory.infer``:

    * an activity that already has a plan with steps left is the cheap path — read the current step
      and advance ``step_index``: no model call, no procedural lookup;
    * an activity with no plan gets one by *reuse* first (``procedural.retrieve``) and only *infers*
      a fresh one on a miss — firing the ``_infer_`` internal action, which moves the activity to
      RUNNING and yields no step this cycle; the plan lands a later cycle via ``inference_sink`` and
      Reason then advances it. Reuse is currently always a miss — the default Reflect no longer
      stores completed plans (auto-caching a plan for verbatim replay is unsound), so every activity
      infers until reusable procedures are distilled from episodes. Infer passes the currently-
      joined tools (id -> Manual) as the planning catalog and a ``PerceptSnapshot`` of
      ``wm.properties``/``wm.signals`` as the agent's known world state (the strategy holds `wm`; a
      memory module never reaches into the environment or working memory itself — it only sees
      what's extracted and handed to it);
    * an exhausted plan yields no step — the cycle returns, and Reflect terminates the activity the
      next cycle on the same "plan present and fully consumed" rule (so this branch is normally only
      reached by a just-inferred empty plan).

    Mutates the activity (plan/step_index) in place, like the other phase defaults. Both model calls
    run **off-cycle** as the ``_infer_``/``_ground_`` internal actions (isolated in
    ``ProceduralMemory.infer``/``ground``): the cheap path — advancing an existing plan's step_index
    and mechanically resolving its params — makes zero model calls and stays same-cycle; only the
    escalations park the activity in RUNNING and resolve a later cycle. No phase blocks on a model
    call, so there is nothing to race or abandon (ADR-0021)."""

    async def reason(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        if activity.plan is None:
            plan = await cycle.procedural.retrieve(activity)  # reuse across runs (cheap)
            if plan is None:
                # Miss -> fire _infer_ off-cycle: it moves the activity to RUNNING and returns at
                # once, so the cycle never blocks on the model. The plan lands a later cycle via
                # inference_sink (Observe attaches it and resets step_index); this Reason yields no
                # step now. Situate won't reselect a RUNNING activity, so no re-fire meanwhile.
                catalog = {tool.id: tool.manual for tool in wm.registry.all_tools()}
                observed = PerceptSnapshot(list(wm.properties.values()), list(wm.signals))
                infer = cycle.actions.internal(InferAction.name)
                await infer.execute(
                    cycle,
                    activity_id=activity.id,
                    tools=catalog,
                    observed=observed,
                    messages=list(wm.messages),  # recent user instructions, snapshot at fire time
                )
                return result  # RUNNING on the inference; plan lands a later cycle
            log.info(
                "reason: reusing cached plan (%d steps) for %r", len(plan.steps), activity.goal
            )
            activity.plan = plan
            activity.step_index = 0
        while True:
            plan = activity.plan
            assert plan is not None  # set above, and by every branch that continues this loop
            if activity.step_index >= len(plan.steps):
                if activity.parent_frames:
                    # Sub-plan exhausted: pop the frame and resume the parent at the step *after*
                    # its sub-goal, then loop to read it (or pop again if that frame is exhausted).
                    parent_plan, parent_index = activity.parent_frames.pop()
                    activity.plan = parent_plan
                    activity.step_index = parent_index + 1
                    continue
                return result  # top-level plan exhausted -> no step this cycle
            step = plan.steps[activity.step_index]
            if step.next_action == SUBGOAL:
                outcome = await self._subgoal(step, activity, wm, cycle)
                if outcome is _SUBGOAL_SPLICED:
                    continue  # mechanical: spliced the expansion in place -> read the first step
                return result  # deliberative fired _infer_ (RUNNING), or the guard halted (BLOCKED)
            grounded = await self._ground(step, activity, wm, cycle)
            if grounded is None:
                return result  # escalated via _ground_ (RUNNING); the step lands a later cycle
            # Advance only once a concrete step is emitted, so a step awaiting its grounding
            # escalation isn't skipped — step_index stays put across RUNNING cycles until it lands.
            activity.step_index += 1
            return replace(result, activity=activity, step=grounded)

    async def _subgoal(
        self, step: Step, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle
    ) -> object:
        """Handle a ``subgoal`` step (ADR-0022). **Deliberative** -> fire ``_infer_`` for the
        sub-goal's own goal (activity goes RUNNING; the sub-plan lands a later cycle and Observe
        enters it on a pushed frame), reporting ``_SUBGOAL_RUNNING``. **Mechanical** -> fan the
        sub-goal out over its collection and splice the expansion into the plan *in place* — a
        per-run copy via ``replace``, so the stored skeleton keeps its ``subgoal`` step — leaving
        ``step_index`` on the first expanded step and reporting ``_SUBGOAL_SPLICED`` so ``reason``
        re-reads it. An empty/unresolvable collection expands to nothing (the sub-goal vanishes)."""
        mode = step.params.get("mode", "deliberative")
        if mode == "deliberative":
            goal = step.params["goal"]
            halt = self._deliberation_would_loop(activity, goal)
            if halt is not None:
                # Refuse to recurse: pause to await the user's guidance instead of spending another
                # (and another...) _infer_ on a sub-goal that isn't reducing. Set BLOCKED directly,
                # as the interrupt handler does for its InputWait — no _suspend_ (that's for a
                # manual-declared SignalWait), and no model call for the prompt (kept mechanical).
                log.warning(
                    "reason: halting sub-goal recursion for activity %s: %s", activity.id, halt
                )
                activity.state = ActivityState.BLOCKED
                activity.blocked_on = InputWait(
                    prompt=f"Stuck on {goal!r}: {halt}. How should I proceed?"
                )
                return _SUBGOAL_HALTED
            catalog = {tool.id: tool.manual for tool in wm.registry.all_tools()}
            observed = PerceptSnapshot(list(wm.properties.values()), list(wm.signals))
            infer = cycle.actions.internal(InferAction.name)
            await infer.execute(
                cycle,
                activity_id=activity.id,
                tools=catalog,
                observed=observed,
                messages=list(wm.messages),
                kind="subgoal",
                goal=goal,
            )
            return _SUBGOAL_RUNNING
        plan = activity.plan
        assert plan is not None  # reason() only dispatches a step off a set plan
        i = activity.step_index
        expanded = _expand_mechanical(step, activity.history)
        activity.plan = replace(plan, steps=plan.steps[:i] + expanded + plan.steps[i + 1 :])
        log.info(
            "reason: sub-goal %r fanned out to %d step(s)", step.params.get("goal"), len(expanded)
        )
        return _SUBGOAL_SPLICED

    def _deliberation_would_loop(self, activity: Activity, goal: str) -> str | None:
        """Whether firing a deliberative sub-goal for ``goal`` now would be runaway recursion rather
        than progress — the reason string if so (for the log and the await-input prompt), else
        ``None``. Two mechanical checks: the intention stack is already ``_MAX_SUBGOAL_DEPTH`` deep,
        or ``goal`` closely repeats a sub-goal still suspended above it (not reducing)."""
        depth = len(activity.parent_frames)
        if depth >= _MAX_SUBGOAL_DEPTH:
            return f"sub-goal recursion reached the depth cap ({depth} >= {_MAX_SUBGOAL_DEPTH})"
        for ancestor in _ancestor_subgoal_goals(activity):
            if _goal_token_similarity(goal, ancestor) >= _SUBGOAL_GOAL_SIMILARITY:
                return "sub-goal goal repeats an ancestor's without reducing to concrete actions"
        return None

    async def _ground(
        self,
        step: Step,
        activity: Activity,
        wm: WorkingMemory,
        cycle: DecisionCycle,
    ) -> Step | None:
        """Ground a step's reference-bearing params against this run's execution history for *this
        cycle* plus the agent's currently observed properties/signals, leaving the stored plan's
        references intact so procedural reuse keeps a reusable skeleton. Deciding a param value is
        a *reasoning* act, so it lives here, not in Act (which stays mechanistic). Hybrid: resolve
        references deterministically; escalate to one **off-cycle** model call (the ``_ground_``
        internal action) only for what can't be resolved mechanically. Returns the concrete Step
        when grounding is complete (no references, references resolved mechanically, or a prior
        ``_ground_`` escalation's params consumed from ``activity.grounded_params``), or ``None``
        when it just fired ``_ground_`` — the activity is now RUNNING and the step lands a later
        cycle. A step with no references is a pure no-op — the cheap path makes no model call. Only
        ``invoke`` (an operation's params) and ``send`` (its ``content``) carry a groundable bag
        today; ``focus``/``unfocus`` carry only a bare ``tool_id``, nothing to ground."""
        if step.next_action == InvokeAction.name:
            routing = {k: v for k, v in step.params.items() if k in (TOOL_ID, OPERATION_NAME)}
            op_params = {k: v for k, v in step.params.items() if k not in (TOOL_ID, OPERATION_NAME)}
            manual = _manual_for(wm, routing.get(TOOL_ID))
            resolved = await self._resolve(
                activity, wm, cycle, op_params, routing[OPERATION_NAME], manual
            )
            if resolved is None:
                return None  # escalated to _ground_; RUNNING now
            if resolved == op_params:
                return step  # no references -> unchanged, reuse the original Step
            return replace(step, params={**routing, **resolved})
        if step.next_action == SendAction.name:
            content = step.params.get("content")
            if not isinstance(content, dict):
                return step  # nothing groundable
            resolved = await self._resolve(activity, wm, cycle, content, SendAction.name, None)
            if resolved is None:
                return None
            if resolved == content:
                return step
            return replace(step, params={**step.params, "content": resolved})
        return step

    @staticmethod
    async def _resolve(
        activity: Activity,
        wm: WorkingMemory,
        cycle: DecisionCycle,
        params: dict[str, Any],
        operation_name: str,
        manual: Manual | None,
    ) -> dict[str, Any] | None:
        """Mechanically resolve ``params`` against history; if anything can't be, escalate to the
        off-cycle ``_ground_`` action and return ``None``. When a prior escalation already resolved
        (``activity.grounded_params`` set by Observe), consume and return those instead — the 1:1
        counterpart to the plan landing on ``activity.plan``. step_index hasn't advanced across the
        RUNNING cycles, so the parked params belong to exactly this step."""
        if activity.grounded_params is not None:
            resolved = activity.grounded_params  # the escalation resolved; consume it
            activity.grounded_params = None
            return resolved
        resolved, unresolved = resolve_references(params, activity.history)
        if not unresolved:
            return resolved  # cheap path — resolved mechanically, no model call
        observed = PerceptSnapshot(list(wm.properties.values()), list(wm.signals))
        ground = cycle.actions.internal(GroundAction.name)
        await ground.execute(
            cycle,
            activity_id=activity.id,
            operation_name=operation_name,
            manual=manual,
            partial_params=resolved,
            observed=observed,
        )
        return None  # RUNNING on the grounding escalation; params land a later cycle


class DefaultActStrategy:
    """The mechanical, no-LLM default for *parameter binding* (not protocol binding — see
    ActStrategy): bind an ``invoke`` Step straight to an OperationInvocation, splitting the
    tool_id/operation_name routing keys out of the operation's own params. A model-backed
    ActStrategy would instead ground under-specified params against the manual's schema here; the
    default assumes the Step already carries concrete params, so binding is just the key-split.

    One mechanical guard sits here (still no judgment, so Act stays mechanistic per ADR-0017): a
    **required** param that resolves to null is a schema violation, so the invoke is *skipped* — no
    invocation is emitted and the cycle dispatches nothing this step (`_act`). Grounding (Reason)
    has already run by now, so a null at bind time is a value the model declined or could not fill,
    not an un-grounded reference; dispatching the operation anyway is the historic blind-`delete`
    mis-action, degraded to a probabilistic one and previously held off only by a prompt fragment.
    The guard needs the operation's schema (`OperationSpecification.parameters`, adapter-
    synthesized) to know which params are required; with no manual/spec/declared ``required`` it
    cannot tell, so it does not fire and binds as before — the same structured-spec dependency the
    thread-reading Manual relocation has. It narrows, not eliminates, a null-invoke: an *optional*
    null still passes through by design (many operations take legitimately-optional params)."""

    async def bind(
        self, step: Step, manual: Manual | None, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        params = {k: v for k, v in step.params.items() if k not in (TOOL_ID, OPERATION_NAME)}
        operation_name = step.params[OPERATION_NAME]
        null_required = _null_required_params(manual, operation_name, params)
        if null_required:
            log.warning(
                "act: skipping invoke %s.%s — required param(s) %s resolved to null",
                step.params[TOOL_ID],
                operation_name,
                null_required,
            )
            return result  # no invocation -> _act dispatches nothing this step (skip-and-continue)
        invocation = OperationInvocation(
            tool_id=step.params[TOOL_ID],
            operation_name=operation_name,
            params=params,
        )
        return replace(result, invocation=invocation)


def _null_required_params(
    manual: Manual | None, operation_name: str, params: dict[str, Any]
) -> list[str]:
    """The operation's *required* params (per its schema) that resolve to null in ``params`` —
    either an explicit None or absent entirely (a required key the step never supplied). Empty when
    the schema is unavailable (no manual, the manual doesn't describe this op, or it declares no
    ``required``): required-ness is then unknowable, so the guard can't fire and binding goes on."""
    spec = manual.operation(operation_name) if manual is not None else None
    if spec is None:
        return []
    required = spec.parameters.get("required", [])
    return [key for key in required if params.get(key) is None]
