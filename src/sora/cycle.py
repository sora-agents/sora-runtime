"""The decision cycle and the agent that runs it."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

from sora.action import JoinAction
from sora.activity import ActivityState
from sora.perception import NotificationQueueSink
from sora.strategies import DefaultInterruptHandler, NeverInterruptPolicy
from sora.types import TOOL_ID, WAIT, InterruptRequest

log = logging.getLogger("sora.cycle")

if TYPE_CHECKING:
    from sora.action import ActionRegistry
    from sora.activity import Activity
    from sora.environment import EnvironmentRegistry
    from sora.memory import EpisodicMemory, ProceduralMemory, SemanticMemory, WorkingMemory
    from sora.strategies import InterruptHandler, InterruptPolicy, Strategies, TickResult
    from sora.transport import MessageTransport
    from sora.types import InferenceResult, OperationAck, Signal, Step


class DecisionCycle:
    def __init__(
        self,
        strategies: Strategies,
        communication: MessageTransport,
        actions: ActionRegistry,
        registry: EnvironmentRegistry,
        working: WorkingMemory,
        semantic: SemanticMemory,
        procedural: ProceduralMemory,
        episodic: EpisodicMemory,
        interrupt_handler: InterruptHandler | None = None,
        interrupt_policy: InterruptPolicy | None = None,
    ) -> None:
        self.strategies = strategies
        self.communication = communication
        self.actions = actions
        # The two hard-interrupt seams. interrupt_handler decides an interrupted activity's next
        # state (default: a user stop pauses it to await input); interrupt_policy screens pushed
        # signals for ones that should preempt the current phase (default: none do — the cooperative
        # signal path is unchanged). Both default so existing construction sites (and single-agent
        # bootstraps that never set them) keep working.
        self.interrupt_handler: InterruptHandler = interrupt_handler or DefaultInterruptHandler()
        self.interrupt_policy: InterruptPolicy = interrupt_policy or NeverInterruptPolicy()
        # The mutation-capable handle, passed to external actions at dispatch. WorkingMemory holds
        # this same shared instance read-only (as EnvironmentView) for strategies to reason over.
        self.registry = registry
        self.working = working
        self.semantic = semantic
        self.procedural = procedural
        self.episodic = episodic
        # These sinks live here rather than on WorkingMemory: they're the bridge from
        # asynchronous, off-cycle events into this engine's tick()/interrupt() — not settled
        # state. signal_sink specifically has to be co-located with interrupt() below, since a
        # pushed Signal can preempt the current phase; that control-flow role, not "where it
        # eventually lands as a percept," is why it isn't a WorkingMemory field. result_sink carries
        # invoke() acks; inference_sink carries off-cycle infer()/ground() results (an
        # InferenceResult, never a Percept — ADR-0019/0021).
        self.signal_sink: NotificationQueueSink[Signal] = NotificationQueueSink()
        self.result_sink: NotificationQueueSink[OperationAck] = NotificationQueueSink()
        self.inference_sink: NotificationQueueSink[InferenceResult] = NotificationQueueSink()
        # Screen every signal at push time (before the cooperative Observe drain) so an
        # InterruptPolicy can raise a hard interrupt the instant a qualifying signal arrives. Only
        # signal_sink gets the hook; result_sink stays plain enqueue-only.
        self.signal_sink.on_push = self._screen_signal
        # A pending hard interrupt (None when idle) and the edge that wakes a waiting cycle.
        # Set together by interrupt()/_screen_signal; the request clears once the handler routes
        # it, the wake edge is consumed by wait_between_ticks. See interrupt() / _preempted().
        self._interrupt: InterruptRequest | None = None
        self._wake = asyncio.Event()
        # Monotonic count of ticks run, for observability (the README's `[cycle N]` trace). Read via
        # cycle_count; a richer per-phase presenter (the --verbose CLI) is deferred to CLI polish.
        self._cycle_count = 0

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    async def tick(self) -> None:
        """One Observe -> Reflect -> Situate -> Reason -> Act pass, threading a TickResult through
        all five phases and calling each phase's own strategy only for whatever's still missing —
        so a field an earlier phase already filled short-circuits the later phase.
        working/semantic/procedural/episodic/communication/registry are all shared with Agent,
        constructed once and passed to both — see sora/bootstrap.py. (Dispatch in _act() uses
        self.registry — the mutation-capable handle — not working.registry, which is read-only.)"""
        self._cycle_count += 1
        log.debug("[cycle %d] begin", self._cycle_count)
        result = await self.strategies.observe.observe(self)
        # Checkpoint after every phase boundary: if a hard interrupt is pending (raised by
        # interrupt() or an InterruptPolicy screening a pushed signal), run the handler and abort
        # rest of this tick. The abandoned TickResult carries no staleness — it never outlives one
        # tick() (ADR-0011) — and Act (the cycle's single external action) is never reached, so an
        # interrupted tick commits nothing external. Reason's model calls run off-cycle as the
        # _infer_/_ground_ internal actions (never blocking the tick), so no phase sits inside a
        # model call for a checkpoint to interrupt — the phase-boundary checkpoints alone meet the
        # reactive target (ADR-0021).
        if await self._preempted():
            return
        for activity in list(self.working.activities.values()):
            result = await self.strategies.reflect.reflect(activity, self.working, self, result)
        if await self._preempted():
            return
        # Situate always runs: it re-adjusts wm for the (possibly already-selected) activity every
        # cycle, and selects only if result.activity is still None. Unlike the step/invocation gates
        # below — genuine forward-fusion short-circuits — Situate is not gated on its own field.
        ready = [a for a in self.working.activities.values() if a.state is ActivityState.READY]
        result = await self.strategies.situate.situate(ready, self.working, self, result)
        if await self._preempted():
            return
        selected = result.activity
        if selected is None:
            return  # nothing selectable this cycle — at most one action, never a mandatory one
        if result.step is None:
            result = await self.strategies.reason.reason(selected, self.working, self, result)
            if await self._preempted():
                return
        step = result.step
        if step is None:
            return
        await self._act(selected, step, result)

    async def _preempted(self) -> bool:
        """A phase-boundary checkpoint. If no interrupt is pending, return False and let the tick
        continue. Otherwise run the handler — which routes each targeted activity onto an existing
        state (the saved-context / interrupt-handler / reschedule shape) — and return True so the
        caller aborts this tick. The request is cleared only once the handler reports it fully
        discharged; while some targeted activity is still RUNNING (an external op in flight, left to
        finish) it stays pending and is revisited on the next checkpoint after that op resolves."""
        if self._interrupt is None:
            return False
        discharged = await self.interrupt_handler.handle(self._interrupt, self.working, self)
        if discharged:
            self._interrupt = None
        return True

    def _screen_signal(self, source: str, signal: Signal) -> None:
        """signal_sink.on_push: consulted synchronously as each signal is pushed, before the
        cooperative Observe drain. If the InterruptPolicy elects to preempt, record the request and
        wake the cycle; otherwise the signal just flows to the drain as before."""
        request = self.interrupt_policy.decide(source, signal, self.working)
        if request is not None:
            self._interrupt = request
            self._wake.set()

    async def wait_between_ticks(self, interval: float) -> None:
        """Sleep up to `interval` between ticks, but wake immediately if interrupt() (or a signal
        policy) fired — so a hard interrupt starts the next tick without waiting out the idle wait
        (the reactive target), instead of a bare asyncio.sleep. The edge is consumed (cleared) here,
        so it only shortens the one following sleep."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._wake.wait(), timeout=interval)
        self._wake.clear()

    async def _act(self, selected: Activity, step: Step, result: TickResult) -> None:
        """Act's bind-then-dispatch boundary. "Bind" here is *parameter binding* — grounding the
        abstract step into a concrete OperationInvocation (not a protocol binding, which is the
        adapter's Tool concern) — done iff its *action* declares it needs binding; then dispatch
        this cycle's single external action, with the bound invocation's routing keys + params when
        present, else the raw step params.

        WAIT is the cycle's own no-op sentinel, not a registered ExternalAction, so it's guarded
        first — before the registry lookup that would otherwise KeyError on it. Which steps need
        binding is the action's property (`requires_binding`), not a hardcoded `next_action` branch:
        that keeps the generic cycle uncoupled from any one action's name and lets a custom binding
        action bind too."""
        if step.next_action == WAIT:
            return
        action = self.actions.external(step.next_action)
        if result.invocation is None and action.requires_binding:
            tool = self.registry.get(step.params[TOOL_ID])
            result = await self.strategies.act.bind(step, tool.manual, self, result)
        invocation = result.invocation
        if invocation is not None:
            await action.execute(
                self.registry,
                self,
                activity_id=selected.id,
                tool_id=invocation.tool_id,
                operation_name=invocation.operation_name,
                **invocation.params,
            )
        else:
            await action.execute(self.registry, self, activity_id=selected.id, **step.params)

    async def interrupt(self, signal: Signal, *, target: str | None = None) -> None:
        """Raise a hard interrupt: preempt the current phase for an authoritative event (the 10ms
        reactive target). Records the request and wakes the loop; the next phase-boundary checkpoint
        runs the handler and aborts the tick (no phase blocks on a model call — infer/ground run
        off-cycle — so the checkpoints alone meet the target). `signal` carries the reason the
        handler reads; `target` names the activity to preempt, None = agent-wide. The one wired
        caller is a user stop from the CLI — a cooperative signal that merely matches a wait resumes
        in Observe and never comes here (an InterruptPolicy promotes a pushed signal to this path).
        `async` for a uniform call surface, though the body doesn't await."""
        self._interrupt = InterruptRequest(signal, target)
        self._wake.set()


class Agent:
    """Owns the pieces that are conceptually the agent's own — the shared EnvironmentRegistry,
    memory, transport — built from the same shared instances as DecisionCycle, so e.g.
    agent.registry.restore(records, agent.semantic) never needs to reach through agent.cycle.
    (agent.registry is the mutation-capable handle; the same instance is exposed read-only as
    agent.working.registry.)"""

    def __init__(
        self,
        cycle: DecisionCycle,
        registry: EnvironmentRegistry,
        working: WorkingMemory,
        semantic: SemanticMemory,
        procedural: ProceduralMemory,
        episodic: EpisodicMemory,
        communication: MessageTransport,
        *,
        tick_interval: float = 0.05,
    ) -> None:
        self.cycle = cycle
        self.registry = registry
        self.working = working
        self.semantic = semantic
        self.procedural = procedural
        self.episodic = episodic
        self.communication = communication
        # Seconds slept between ticks — the pace at which the agent yields to off-cycle I/O (an
        # invoke resolving, an inbound message). Small, not zero, so a mostly-idle agent doesn't
        # busy-spin; tests pass 0 to run the loop as fast as the event loop allows.
        self._tick_interval = tick_interval
        self._stopped = False
        self._started = False

    async def run(self) -> None:
        """Join the configured workspaces once (startup), then drive the decision cycle until
        stop() is called. The join happens here — not in bootstrap — because it is async I/O and
        bootstrap stays synchronous; it is what makes the configured tools already available on the
        first cycle. Both the startup join and the loop run inside the try, so the finally leaves
        (closes MCP sessions/subprocesses) whatever managed to join — even a *partial* startup join
        that then failed (otherwise an already-joined workspace's subprocess would leak). Leaving in
        the finally, after the loop exits, also avoids racing an in-flight tick — unlike leaving
        from stop()."""
        try:
            await self._start()
            while not self._stopped:
                await self.cycle.tick()
                # Interruptible idle wait, not a bare sleep: a hard interrupt (user stop, or a
                # signal a policy elects to preempt on) wakes the loop at once for the next tick.
                await self.cycle.wait_between_ticks(self._tick_interval)
        finally:
            for workspace in list(self.registry.joined_workspaces()):
                await self.registry.leave(workspace.id)

    async def _start(self) -> None:
        if self._started:
            return
        self._started = True
        # Join through the predefined _join_ action (not registry.join directly) so the connection
        # *and* its persistence — WorkspaceRecord/ToolRecord/manuals into SemanticMemory — happen
        # together, exactly as a mid-run _join_ would; that's what lets the default Situate's _load_
        # find each tool's manual, and sets up restore() across runs. activity_id is absorbed (join
        # doesn't use it). Idempotent: origins already joined are skipped.
        join = self.cycle.actions.external(JoinAction.name)
        already_joined = {ws.origin for ws in self.registry.joined_workspaces()}
        for origin in self.registry.configured_origins():
            if origin not in already_joined:
                log.info("startup: joining workspace %s (%s)", origin.address, origin.adapter)
                await join.execute(self.registry, self.cycle, activity_id="", origin=origin)

    async def stop(self) -> None:
        self._stopped = True
