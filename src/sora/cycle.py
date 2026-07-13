"""The decision cycle and the agent that runs it."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sora.action import InvokeAction
from sora.activity import ActivityState
from sora.perception import NotificationQueueSink
from sora.types import TOOL_ID, WAIT

if TYPE_CHECKING:
    from sora.action import ActionRegistry
    from sora.environment import EnvironmentRegistry
    from sora.memory import EpisodicMemory, ProceduralMemory, SemanticMemory, WorkingMemory
    from sora.strategies import Strategies
    from sora.transport import MessageTransport
    from sora.types import OperationAck, Signal


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

    async def tick(self) -> None:
        """One Observe -> Reflect -> Situate -> Reason -> Act pass, threading a TickResult through
        all five phases and calling each phase's own strategy only for whatever's still missing —
        so a fully-fused Observe (or Reflect) call can skip the rest of the cycle entirely.
        working/semantic/procedural/episodic/communication/registry are all shared with Agent,
        constructed once and passed to both — see sora/bootstrap.py."""
        registry = self.registry  # mutable handle for dispatch (working.registry is read-only)
        result = await self.strategies.observe.observe(self)
        for activity in list(self.working.activities.values()):
            result = await self.strategies.reflect.reflect(activity, self.working, self, result)
        if result.activity is None:
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
        # NOTE: this "invoke needs binding" special-case (and the WAIT no-op below) is a stopgap —
        # whether a step needs Act-binding should be a property of the action, not a hardcoded
        # branch in the generic cycle. Until that lands, at least name the constants, not literals.
        if result.invocation is None and step.next_action == InvokeAction.name:
            tool = registry.get(step.params[TOOL_ID])
            result = await self.strategies.act.bind(step, tool.manual, self, result)
        # Dispatch this cycle's single external action (if any). WAIT is a no-op placeholder.
        if step.next_action == WAIT:
            return
        action = self.actions.external(step.next_action)
        invocation = result.invocation
        if invocation is not None:
            await action.execute(
                registry,
                self,
                activity_id=selected.id,
                tool_id=invocation.tool_id,
                operation_name=invocation.operation_name,
                **invocation.params,
            )
        else:
            await action.execute(registry, self, activity_id=selected.id, **step.params)

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
    ) -> None:
        self.cycle = cycle
        self.registry = registry
        self.working = working
        self.semantic = semantic
        self.procedural = procedural
        self.episodic = episodic
        self.communication = communication

    async def run(self) -> None:
        """Loop: await self.cycle.tick()"""
        raise NotImplementedError

    async def stop(self) -> None: ...
