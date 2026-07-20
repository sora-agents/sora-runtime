"""The decision cycle and the agent that runs it."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from sora.action import JoinAction
from sora.activity import ActivityState
from sora.perception import NotificationQueueSink
from sora.types import TOOL_ID, WAIT

log = logging.getLogger("sora.cycle")

if TYPE_CHECKING:
    from sora.action import ActionRegistry
    from sora.activity import Activity
    from sora.environment import EnvironmentRegistry
    from sora.memory import EpisodicMemory, ProceduralMemory, SemanticMemory, WorkingMemory
    from sora.strategies import Strategies, TickResult
    from sora.transport import MessageTransport
    from sora.types import OperationAck, Signal, Step


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
    ) -> None:
        self.strategies = strategies
        self.communication = communication
        self.actions = actions
        # The mutation-capable handle, passed to external actions at dispatch. WorkingMemory holds
        # this same shared instance read-only (as EnvironmentView) for strategies to reason over.
        self.registry = registry
        self.working = working
        self.semantic = semantic
        self.procedural = procedural
        self.episodic = episodic
        # Both sinks live here rather than on WorkingMemory: they're the bridge from
        # asynchronous, off-cycle events into this engine's tick()/interrupt() — not settled
        # state. signal_sink specifically has to be co-located with interrupt() below, since a
        # pushed Signal can preempt the current phase; that control-flow role, not "where it
        # eventually lands as a percept," is why it isn't a WorkingMemory field.
        self.signal_sink: NotificationQueueSink[Signal] = NotificationQueueSink()
        self.result_sink: NotificationQueueSink[OperationAck] = NotificationQueueSink()
        # Monotonic count of ticks run, for observability (the README's `[cycle N]` trace). Read via
        # cycle_count; a richer per-phase presenter (the --verbose CLI) is deferred to CLI polish.
        self._cycle_count = 0

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    async def tick(self) -> None:
        """One Observe -> Reflect -> Situate -> Reason -> Act pass, threading a TickResult through
        all five phases and calling each phase's own strategy only for whatever's still missing —
        so a fully-fused Observe (or Reflect) call can skip the rest of the cycle entirely.
        working/semantic/procedural/episodic/communication/registry are all shared with Agent,
        constructed once and passed to both — see sora/bootstrap.py. (Dispatch in _act() uses
        self.registry — the mutation-capable handle — not working.registry, which is read-only.)"""
        self._cycle_count += 1
        log.debug("[cycle %d] begin", self._cycle_count)
        result = await self.strategies.observe.observe(self)
        for activity in list(self.working.activities.values()):
            result = await self.strategies.reflect.reflect(activity, self.working, self, result)
        # Situate always runs: it re-adjusts wm for the (possibly already-selected) activity every
        # cycle, and selects only if result.activity is still None. Unlike the step/invocation gates
        # below — genuine forward-fusion short-circuits — Situate is not gated on its own field.
        ready = [a for a in self.working.activities.values() if a.state is ActivityState.READY]
        result = await self.strategies.situate.situate(ready, self.working, self, result)
        selected = result.activity
        if selected is None:
            return  # nothing selectable this cycle — at most one action, never a mandatory one
        if result.step is None:
            result = await self.strategies.reason.reason(selected, self.working, self, result)
        step = result.step
        if step is None:
            return
        await self._act(selected, step, result)

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

    async def interrupt(self, signal: Signal) -> None:
        """Preempts the current phase for a high-priority event (10ms target)."""
        raise NotImplementedError


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
                await asyncio.sleep(self._tick_interval)
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
