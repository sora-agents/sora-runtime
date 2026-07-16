"""The decision cycle and the agent that runs it."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sora.activity import ActivityState
from sora.perception import NotificationQueueSink
from sora.types import TOOL_ID, WAIT

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

    async def tick(self) -> None:
        """One Observe -> Reflect -> Situate -> Reason -> Act pass, threading a TickResult through
        all five phases and calling each phase's own strategy only for whatever's still missing —
        so a fully-fused Observe (or Reflect) call can skip the rest of the cycle entirely.
        working/semantic/procedural/episodic/communication/registry are all shared with Agent,
        constructed once and passed to both — see sora/bootstrap.py. (Dispatch in _act() uses
        self.registry — the mutation-capable handle — not working.registry, which is read-only.)"""
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
