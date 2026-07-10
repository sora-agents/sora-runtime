"""The decision cycle and the agent that runs it."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sora.perception import NotificationQueueSink

if TYPE_CHECKING:
    from sora.action import ActionRegistry
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
        working: WorkingMemory,
        semantic: SemanticMemory,
        procedural: ProceduralMemory,
        episodic: EpisodicMemory,
    ) -> None:
        self.strategies = strategies
        self.communication = communication
        self.actions = actions
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
        raise NotImplementedError

    async def interrupt(self, signal: Signal) -> None:
        """Preempts the current phase for a high-priority event (10ms target)."""
        raise NotImplementedError


class Agent:
    """Owns the pieces that are conceptually the agent's own — memory, transport — built from the
    same shared instances as DecisionCycle, so e.g. agent.working.registry.restore(records,
    agent.semantic) never needs to reach through agent.cycle."""

    def __init__(
        self,
        cycle: DecisionCycle,
        working: WorkingMemory,
        semantic: SemanticMemory,
        procedural: ProceduralMemory,
        episodic: EpisodicMemory,
        communication: MessageTransport,
    ) -> None: ...

    async def run(self) -> None:
        """Loop: await self.cycle.tick()"""
        raise NotImplementedError

    async def stop(self) -> None: ...
