"""Default Observe strategy, attention policy, and perception retention."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from sora._strategies.conditions import (
    ConditionRetirement,
    _eligible_conditions,
    _match_signal,
)
from sora._strategies.contracts import (
    FocusPolicy,
    TickResult,
)
from sora._strategies.inference import (
    DEFAULT_INFERENCE_DEADLINE,
    _expire_stalled_inferences,
    _resolve_inferences,
)
from sora._strategies.interaction import (
    _goal_from_message,
    _truncate,
)
from sora.action import (
    ResumeAction,
    SuspendAction,
    UnfocusAction,
    attend,
    release,
)
from sora.activity import Activity, ActivityState
from sora.memory import (
    PerceptSnapshot,
)
from sora.perception import Percept
from sora.references import (
    _REF_PROP,
)
from sora.types import (
    TOOL_ID,
    CompletedOperation,
    ConditionWait,
    InputWait,
    OperationInvocation,
    PropertyChange,
    SignalWait,
    changes_of,
    diff_values,
    path_matches,
)

if TYPE_CHECKING:
    from sora.cycle import DecisionCycle
    from sora.memory import WorkingMemory

log = logging.getLogger("sora.strategies")

# Bound on retained signals. Nothing is ever evicted for being *matched* — `wm.signals` is a shared
# broadcast log, so one waiter's satisfaction must not remove an event another waiter (or a strategy
# reading the log directly) still needs, and a signal that arrives before its waiter has to survive
# to the cycle that suspends. The retention cap is therefore the only eviction, and it runs after
# both the suspend and resume passes so a signal that arrived this tick is matched before it can be
# a candidate. The newest win. Deliberately simple; note the cap now bounds something with more
# consequence than it used to — losing a signal loses not just the fact that an event happened but
# the only record of *where*, which no later snapshot can reconstruct.
_SIGNAL_RETENTION = 256


# Bound on retained derived changes. Deliberately larger than `_SIGNAL_RETENTION` and deliberately
# a separate number: the two logs are sized against different windows. A signal accrues per
# environment event, a derived change per observation cycle in which anything moved -- so on a tool
# whose property churns, this log fills at tick rate (~11/s on a live scenario) rather than at event
# rate. What an entry must survive is the gap between its append and the cycle the watching activity
# is free to judge it, which is bounded by how long an inference can occupy that activity
# (`DEFAULT_INFERENCE_DEADLINE`). 1024 covers ~90s of worst-case single-property churn, which clears
# a slow thinking-model call; sizing it to the full deadline would not help, since under that much
# churn the newest-win policy retains the churn and evicts the real change regardless -- that regime
# needs the property excluded from derivation, not a bigger buffer.
_DERIVED_RETENTION = 1024


# Cycles between per-tool observation summaries (DEBUG only). The question this answers is the one a
# `[cycle N]` trace structurally cannot: cycle numbers alone say how many ticks ran, never WHEN, so
# a run whose ticks were bunched at the end of a window is indistinguishable from one that ticked
# evenly through it — and those two readings of a missed change (never looked / looked and lost it)
# call for completely different fixes. Counts plus elapsed wall time separate them. Periodic rather
# than per-cycle because Observe runs at ~11/s on a live scenario, where a line per tick buries the
# trace it is meant to explain.
_OBSERVATION_HEARTBEAT = 100


# How long the retire-only condition sweep waits before re-judging one activity's quiet conditions,
# and how far it backs off after a judgement that retired nothing (the interval doubles per miss,
# topping out at 2**_RETIREMENT_BACKOFF times the base). Backoff is not a refinement here but the
# thing that makes the sweep affordable: an activity BLOCKED on a condition makes *every* tick idle,
# so the eligibility that schedules this never lapses on its own, and a fixed interval would keep
# paying for the same "still waiting" answer for as long as the agent lives. A miss counter resets
# whenever a condition is declared or retired, so a set that actually moves is checked promptly
# again. What the backoff does not solve — an activity waiting on a healthy-looking condition that
# has been quiet for a very long time — is a question for the user, and deliberately not answered
# here (ADR-0027's last open consequence).
#
# Wall-clock on purpose, and this is emphatically NOT the clock an `until` is answered against.
# This paces how often the runtime *asks*, which is infrastructure timing like the inference
# watchdog above; the `until` itself is a question about **domain** time, which reaches the runtime
# through the workspace and never through `time.time()` (ADR-0027 §5). A tick count would be the
# wrong unit for the pacing anyway: tick rate varies by two orders of magnitude between an
# interactive session and a simulation loop.
DEFAULT_RETIREMENT_INTERVAL = 30.0


def _prop_tool_ids(params: Any, joined: set[str]) -> set[str]:
    """Tool ids named by any ``$prop`` reference nested anywhere inside a step's params.

    A ``$prop`` is the one way a step reads a tool it never invokes, and it lives *inside* the
    param bag — under a data-op's ``in``, a sub-goal's collection — not under ``TOOL_ID``. Missing
    it releases the tool mid-plan and the reference then resolves to nothing, which is the silent
    blindness intention-scoped focus exists to remove. Recursive for that reason: depth is
    not bounded by the schema.

    The tool id is found by matching against the *joined* id set rather than by splitting on the
    first dot, for the reason `_property_ref` documents at length: neither half of a property key
    is dot-free. Longest match wins. A bare property name (``{"$prop": "state"}``) names no tool
    and contributes nothing — it stays resolvable only while something else attends its tool.
    """
    found: set[str] = set()
    if isinstance(params, dict):
        ref = params.get(_REF_PROP)
        if isinstance(ref, str):
            candidates = [t for t in joined if ref == t or ref.startswith(f"{t}.")]
            if candidates:
                found.add(max(candidates, key=len))
        for value in params.values():
            found |= _prop_tool_ids(value, joined)
    elif isinstance(params, (list, tuple)):
        for item in params:
            found |= _prop_tool_ids(item, joined)
    return found


def _wait_sources(wait: SignalWait | InputWait | ConditionWait | None) -> set[str]:
    if isinstance(wait, SignalWait):
        return {wait.source} if wait.source else set()
    if isinstance(wait, ConditionWait):
        return {w.source for w in wait.watches if w.source}
    return set()  # InputWait waits on the user, not on a tool


def referenced_tools(activity: Activity, joined: set[str]) -> set[str] | None:
    """The tools one activity's live intentions reference — or ``None`` for "the whole world".

    ``None`` is not "nothing": an activity with no plan is about to be planned, and planning is
    deliberately broad (the planner picks ``$prop`` over paginated scanning by reading property
    *shapes* in the plan prompt, so narrowing it there would go dark on ADR-0023 discovery). Since
    `reset_for_replan` clears `plan`, that same clause covers the whole replan window — which is
    why attention needs no grace period and no history retention.

    Shared by the attention policy (agent-level union) and by the Stage-2 per-activity prompt view,
    so the two layers cannot disagree about what an activity's tools are.
    """
    if activity.state is ActivityState.TERMINATED:
        return set()
    if activity.plan is None:
        return None
    ids: set[str] = set()
    for plan in [activity.plan, *(frame[0] for frame in activity.parent_frames)]:
        for step in plan.steps:
            # An `unfocus` step names a tool in order to STOP attending to it; counting it here
            # would attend the tool right back and make the step a permanent no-op. Note this only
            # gets the tool no OTHER step names: the scan covers every step of a live plan, not the
            # un-run tail (that is what removes the need for history retention), so an `unfocus`
            # following an `invoke` of the same tool still reads as referenced here. That is not the
            # last word — once the `unfocus` step actually runs, `WorkingMemory.suppressed_tools`
            # holds the release against this derived floor, so the explicit act wins.
            if step.next_action != UnfocusAction.name:
                tool_id = step.params.get(TOOL_ID)
                if isinstance(tool_id, str):
                    ids.add(tool_id)
            ids |= _prop_tool_ids(step.params, joined)
        # A declared condition watches a tool the body may never touch (waiting on a reply in a
        # messaging app while every step touches email) — not redundant with the step scan.
        for condition in plan.pending:
            if condition.watch.source:
                ids.add(condition.watch.source)
    for state in activity.pending_conditions:
        if state.condition.watch.source:
            ids.add(state.condition.watch.source)
    if activity.pending_operation is not None:
        ids.add(activity.pending_operation.invocation.tool_id)
    ids |= _wait_sources(activity.blocked_on)
    return ids


class IntentionScopedFocus:
    """Attend exactly the tools the agent's live *intentions* reference — the narrowing policy,
    opt-in rather than default.

    BDI vocabulary on purpose — this set is derived from committed plans, not from candidate goals,
    so it is intention-scoped and not desire-scoped. The union over live activities is an eagerly
    evaluated refcount: a tool leaves focus precisely when no live plan names it any more. Kept as
    a recomputation rather than incremental leases because a lease has to be maintained at every
    plan mutation site, and a missed decrement leaks cost silently while a double decrement blinds
    a tool that is still in use — going silently blind, the failure this policy exists to
    remove. Analysis and measurement: `docs/architecture/notes/
    attention-scoped-to-live-intentions.md`.

    Broad while nothing is planned (no activities at all, or any activity awaiting a plan), narrow
    while executing.
    """

    def attend(self, wm: WorkingMemory) -> set[str]:
        joined = {tool.id for tool in wm.registry.all_tools()}
        live = [a for a in wm.activities.values() if a.state is not ActivityState.TERMINATED]
        if not live:
            return joined  # an idle agent stays observant (ADR-0026's judge reads the world)
        attended: set[str] = set()
        for activity in live:
            referenced = referenced_tools(activity, joined)
            if referenced is None:
                return joined
            attended |= referenced
        # Intersect last, so an id left over from a departed workspace can never be re-attended.
        return attended & joined


class FocusAllJoined:
    """Attend every joined tool, ignoring activities — **the default**.

    Not a relic of the auto-focus-on-join fallback: attention is still reconciled every
    Observe by the same mechanism, so perception never hinges on the model emitting a
    `focus` step. What this policy declines to do is *narrow*. Measured, narrowing is worth
    ~825 prompt tokens per model call and **zero** judge calls on the shipped configs — the
    condition judge already gates on `watch.source`, and the relevance judge is opt-in and
    off — while it introduces churn across replans, where re-attaching re-baselines an
    adapter and a change occurring in the gap is absorbed rather than reported. That failure
    is silent, so it is not worth cents against a benchmark result.

    `IntentionScopedFocus` is the opt-in for a dynamic, many-workspace run, where the
    broad set grows with the environment while the narrow one stays constant."""

    def attend(self, wm: WorkingMemory) -> set[str]:
        return {tool.id for tool in wm.registry.all_tools()}


def scoped_snapshot(wm: WorkingMemory, activity: Activity) -> PerceptSnapshot:
    """The agent's percepts narrowed to one activity's own tools — the second attention layer.

    Attention is agent-level (a union over live activities, because focusing is a subscription and
    one subscription serves everyone). A *prompt* is not: a model call fired for one activity has
    no use for a sibling activity's tools, and with several activities in flight the union is again
    the thing that grows. This closes that gap at the `PerceptSnapshot` boundary every call already
    takes, which makes it **non-destructive**: `wm.properties` is untouched, so the ADR-0024 change
    gate keeps hashing a stable agent-level world and the idle-tick relevance judge keeps reading
    it. A destructive per-activity `_filter_` would instead make the shared store depend on which
    activity the scheduler happened to pick, and move the change signature when nothing moved.

    Deliberately *not* applied everywhere. Planning stays broad (`_infer_` reads property shapes to
    choose `$prop` over paginated scanning), both judges stay broad (they dereference change ids
    against these same properties, and a watch source is routinely a tool no step names), and plan
    revalidation stays broad — its gate fires on *any* perception change, so narrowing it to the
    plan's own tools would blind the judge to the very change that woke it. It applies where the
    question genuinely concerns one step of one activity: `_ground_` and `_select_`.
    """
    # Follows the FocusPolicy rather than deciding for itself. An agent on the broad default has
    # declined to narrow *because* a wrongly narrowed view fails silently — the model simply reasons
    # without the property and nothing reports a miss — so narrowing its prompts anyway would
    # reintroduce exactly the risk the policy choice was made to avoid, one layer down and
    # invisibly. This layer only sharpens a narrowing the agent already opted into: from the
    # agent-level union to the one activity the call is actually about.
    if not wm.attention_narrowed:
        return PerceptSnapshot(list(wm.properties.values()), list(wm.signals))
    joined = {tool.id for tool in wm.registry.all_tools()}
    referenced = referenced_tools(activity, joined)
    if referenced is None:  # unplanned -> the same breadth attention gives it
        return PerceptSnapshot(list(wm.properties.values()), list(wm.signals))
    return PerceptSnapshot(
        [percept for key, percept in wm.properties.items() if key[0] in referenced],
        [percept for percept in wm.signals if percept.source in referenced],
    )


class DefaultObserveStrategy:
    """The runtime's built-in default — purely mechanical, no LLM.

    Owns two reconciliations of activity state against the world, both before any perception is
    read: what the agent *attends to* and what it is *waiting for* (`_suspend_`/
    `_resume_`, ADR-0019). Attention belongs here rather than in Situate because focusing is a
    subscription to the environment, not a view over memory — it decides what can be perceived at
    all, so deciding it after Observe would leave it structurally one tick behind."""

    def __init__(
        self,
        focus: FocusPolicy | None = None,
        *,
        inference_deadline: float | None = DEFAULT_INFERENCE_DEADLINE,
        retirement_interval: float | None = DEFAULT_RETIREMENT_INTERVAL,
    ) -> None:
        # Defaults to attending everything joined. Narrowing to live intentions is a
        # deliberate opt-in (`focus=IntentionScopedFocus()`), not the default: measured, it
        # is worth ~825 prompt tokens per model call and zero judge calls, against a
        # re-baselining risk that fails silently. A benchmark number is worth more than
        # that saving. See the attention design note.
        self._focus = focus or FocusAllJoined()
        # None disables the watchdog entirely — for a deliberately unbounded interactive session,
        # or a debugger sitting on a breakpoint inside the client.
        self._inference_deadline = inference_deadline
        self._retirement = ConditionRetirement(retirement_interval)
        # DEBUG-only diagnostic state, reset at each heartbeat; see _OBSERVATION_HEARTBEAT.
        self._observed: dict[str, int] = {}
        self._heartbeat_cycles = 0
        self._heartbeat_wall = time.time()

    async def observe(self, cycle: DecisionCycle) -> TickResult:
        wm = cycle.working
        # Before the snapshot, so a tool a plan just started referencing is perceived on THIS tick
        # rather than the next one.
        attention_moved = await self._reconcile_attention(cycle)
        self._snapshot_properties(wm)
        if attention_moved:
            self._rebaseline(cycle)
        signals_before = wm.signals_appended
        async for source, signal in cycle.signal_sink.drain():
            wm.signals.append(Percept(source, signal, time.time()))
            # Bumped with the append, never with the eviction: this is what a per-waiter high-water
            # mark is measured against, so it has to survive the retention trim below.
            wm.signals_appended += 1
            log.info("observe: signal %s from %s", signal.name, source)
        self._count_observations(wm, cycle)
        # After the drain, because this reads the signals that just landed to avoid deriving a
        # duplicate of a change the adapter did announce.
        self._derive_property_changes(wm, wm.signals_appended - signals_before)
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
        # Drain first: a provider result may already be queued even though Observe starts after its
        # deadline. It is a real resolution, so do not append a synthetic timeout behind it. Then
        # expire requests that remain pending and drain again so a true absence still resolves in
        # this same Observe pass.
        await _resolve_inferences(cycle)
        _expire_stalled_inferences(cycle, self._inference_deadline)
        await _resolve_inferences(cycle)
        await self._suspend_on_completion_signal(cycle, just_resolved)
        await self._resume_on_signal(cycle)
        # After the resume pass, so an activity a signal just woke counts as one that can advance
        # and stands the sweep down — and so a condition the gate is about to re-judge for free is
        # never also paid for here.
        # Ahead of the judged sweep, and unthrottled by it: a window the clock has already closed
        # costs nothing to notice, so it must neither buy a model call nor wait behind that sweep's
        # backoff (which can hold one activity off for many minutes — long enough to over-run a
        # four-minute maintenance window by four times its own length).
        await self._retirement.retire_expired(cycle)
        await self._retirement.retire_quiet(cycle)
        # Trim last: a signal that just arrived this tick must survive to be matched by the two
        # passes above before it's ever subject to eviction (bound orphan growth; newest win).
        if len(wm.signals) > _SIGNAL_RETENTION:
            del wm.signals[:-_SIGNAL_RETENTION]
        if len(wm.property_changes) > _DERIVED_RETENTION:
            del wm.property_changes[:-_DERIVED_RETENTION]
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

    def _count_observations(self, wm: WorkingMemory, cycle: DecisionCycle) -> None:
        """Per-tool observation counts and the wall-clock rate, summarized every
        `_OBSERVATION_HEARTBEAT` cycles.

        Purely diagnostic — it decides nothing and is skipped entirely unless someone is listening
        at DEBUG, because the counting runs on the hot path. A tool appears here once per cycle it
        was focused for, which is exactly once per cycle its properties were read, so a gap in the
        count is a gap in perception and not merely a quiet log.
        """
        if not log.isEnabledFor(logging.DEBUG):
            return
        for tool_id in wm.focused_tools:
            self._observed[tool_id] = self._observed.get(tool_id, 0) + 1
        self._heartbeat_cycles += 1
        if self._heartbeat_cycles < _OBSERVATION_HEARTBEAT:
            return
        elapsed = time.time() - self._heartbeat_wall
        log.debug(
            "observe: %d cycles in %.1fs (%.1f/s) through cycle %d; observed %s",
            self._heartbeat_cycles,
            elapsed,
            self._heartbeat_cycles / elapsed if elapsed > 0 else 0.0,
            cycle.cycle_count,
            ", ".join(f"{tool}={n}" for tool, n in sorted(self._observed.items())),
        )
        self._observed.clear()
        self._heartbeat_cycles = 0
        self._heartbeat_wall = time.time()

    @staticmethod
    def _derive_property_changes(wm: WorkingMemory, arrived: int) -> None:
        """Recover the changes no adapter announced, by diffing each re-observed property.

        A signal is transient: pushed once, and if it never lands there is nothing left to find,
        because the property snapshot answers "what is true now" and never "what just moved". On the
        run that motivated this, six calendar events were added, four `state_changed` signals
        arrived, and the two the agent never heard about were sitting in its own snapshot the whole
        time — visible, and read by nothing. Diffing against the value last seen reconstructs
        exactly the missing half, the way AgentSpeak's belief revision derives belief-change
        events by comparing new percepts against the belief base.

        Deliberately a *safety net over* the adapter's signal, not a replacement for it. The signal
        stays the fast path (it can say things a diff cannot infer, and it arrives without waiting
        for a snapshot to move), and an adapter with no signal facility at all now gets change
        detection for free.

        Two rules do the real work:

        A property seen for the FIRST time reports nothing. It has not changed; it has only just
        become visible, and treating its whole value as an addition would open every gate watching
        it the instant attention arrived — exactly what `tool.focus()` establishing an adapter-side
        baseline already avoids on the signal path.

        A change the adapter DID signal this tick is not derived again. That is the cost control:
        an adapter refreshes the property and pushes the signal in one breath (ADR-0004 requires
        that order), so both land in the same Observe, and a derived twin would double every
        watching condition's judge calls while carrying nothing new. Dedup is per (source, path)
        rather than per source, so an adapter that reported one change and dropped another in the
        same tick still has the dropped one recovered. `arrived` is how many signals this Observe
        just drained, which is what scopes that check to the current tick.
        """
        signalled: set[tuple[str, str]] = set()
        for percept in wm.signals[len(wm.signals) - arrived :] if arrived else []:
            for change in changes_of(percept.payload):
                signalled.add((percept.source, change.path))
        for (source, name), percept in wm.properties.items():
            current = percept.payload.value
            key = (source, name)
            if key not in wm.property_baseline:
                wm.property_baseline[key] = current
                continue
            previous = wm.property_baseline[key]
            if previous == current:
                continue
            wm.property_baseline[key] = current
            # `path_matches` both ways, as everywhere else: a signal reported above or below the
            # derived path still covers it, so a coarse announcement suppresses the fine twin.
            changes = tuple(
                change
                for change in diff_values(previous, current)
                if not any(
                    src == source and path_matches(path, [change]) for src, path in signalled
                )
            )
            if not changes:
                continue
            wm.property_changes.append(
                Percept(source, PropertyChange(name=name, changes=changes), time.time())
            )
            wm.property_changes_appended += 1
            log.info(
                "observe: derived change on %s.%s from %s (no signal)",
                source,
                name,
                ", ".join(change.path or "<root>" for change in changes),
            )

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
                # The instruction just received is the new direction, so the attempts that led here
                # no longer bear on the next plan. Without this the breaker re-trips on the resumed
                # activity's first Reason pass and the halt becomes permanent instead of a question
                # — including for the reset_for_replan immediately above, which appends to the very
                # trail being read. Cleared after it, so the resume's own entry goes too.
                activity.clear_replan_trail()
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
            if _match_signal(wm, wait) is not None:  # early signal: already satisfied
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
            # An InputWait waits on a user Message and is resumed in _resume_on_input, not here.
            if isinstance(activity.blocked_on, ConditionWait):
                # A gate opening only makes a condition ELIGIBLE — whether it actually holds is a
                # Reason judgment, because matching prose against an email body is irreducibly
                # semantic. Resuming is how the activity gets selected so Reason can make it.
                # Gated on an UNJUDGED match (each condition's own mark): resuming on a signal
                # every condition has already dismissed would re-block immediately and spin.
                if _eligible_conditions(activity, wm):
                    log.info("observe: activity %s has an eligible pending condition", activity.id)
                    await resume.execute(cycle, activity_id=activity.id)
                continue
            if not isinstance(activity.blocked_on, SignalWait):
                continue
            if _match_signal(wm, activity.blocked_on) is not None:
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

    async def _reconcile_attention(self, cycle: DecisionCycle) -> bool:
        """Diff the policy's target set against what is actually focused, and close the gap.

        Returns whether the attended set actually moved, so the caller can re-anchor the
        reconsideration gate against the new observation window.

        A mechanical internal effect, the same class as the auto-focus this replaces — not an
        external-action dispatch — so one-external-action-per-cycle (ADR-0009) is untouched.
        Ordering is safe in the other direction too: `tool.focus()` establishes the adapter's
        change baseline, so the `observe()` immediately below compares equal and a newly attended
        tool emits no spurious `state_changed` on its first tick.
        """
        wm = cycle.working
        joined = {tool.id for tool in wm.registry.all_tools()}
        target = self._focus.attend(wm)
        # Recorded from the policy's own target, *before* suppression is subtracted: "is this agent
        # narrowing?" is a question about the policy. Deriving it from `focused_tools` instead would
        # conflate a narrowing policy with an explicit `_unfocus_` or a tool that failed to resolve,
        # and would switch on the per-activity prompt view for an agent that chose the broad policy.
        wm.attention_narrowed = not joined <= target
        # An explicit `_unfocus_` outranks the policy. The policy is a derived floor recomputed from
        # scratch each tick; the suppression is a decision the agent took and nothing here undoes.
        attended = target - wm.suppressed_tools
        before = set(wm.focused_tools)
        for tool_id in wm.focused_tools.keys() - attended:
            await release(cycle, tool_id)
        for tool_id in attended - wm.focused_tools.keys():
            try:
                tool = wm.registry.get(tool_id)
            except KeyError:  # named by a plan but not (or no longer) joined
                continue
            await attend(cycle, tool)
        if before != wm.focused_tools.keys():  # only on a real transition — this is THE diagnostic
            log.info("observe: attending %s", ", ".join(sorted(wm.focused_tools)) or "(nothing)")
            return True
        return False

    @staticmethod
    def _rebaseline(cycle: DecisionCycle) -> None:
        """Re-anchor the reconsideration gate after the *observation window* moved, rather than the
        world (ADR-0024).

        The gate hashes the property store, so attending or releasing a tool moves the signature
        with nothing in the environment having changed — and the first checkpoint after a plan lands
        (narrowing) or a workspace is joined (broadening) would spend a revalidation call on the
        agent's own attention. Re-anchoring reads that as "the window moved", which is what it is.

        The cost is a one-tick blind spot: a genuine change landing on the same tick as an attention
        transition is absorbed into the new baseline instead of firing. That is bounded — a
        transition happens when a plan lands, on join/leave, and on an explicit focus/unfocus, not
        continuously — and the alternative is a false positive on every one of those.
        """
        signature = cycle.change_gate.signature(cycle.working)
        for activity in cycle.working.activities.values():
            if activity.reconsider_baseline is not None:
                activity.reconsider_baseline = signature

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
