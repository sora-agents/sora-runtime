"""Activity: the sole first-class unit of work (see ADR-0002)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sora.types import CompletedOperation, OperationAck, PendingOperation, Plan


class ActivityState(Enum):
    RUNNING = "running"
    BLOCKED = "blocked"
    READY = "ready"
    TERMINATED = "terminated"


@dataclass
class Activity:
    id: str
    goal: str
    context: dict[str, Any]
    state: ActivityState = ActivityState.READY
    plan: Plan | None = None  # once set, Reason can just advance it instead of (re)planning
    step_index: int = 0
    pending_operation: PendingOperation | None = None  # set while RUNNING; cleared on resolve
    last_operation: OperationAck | None = None  # most recently resolved result, for Reason to read
    # Append-only trace of resolved operations this activity ran — a later step grounds its params
    # against it (last_operation keeps only the newest, overwritten each step). Transient:
    # not persisted, and episodic learn() captures selectively, not a blind asdict(activity).
    history: list[CompletedOperation] = field(default_factory=list)
    # context is exclusively for strategy-author data — the runtime itself never writes into it,
    # which is what keeps pending_operation/last_operation as dedicated fields instead of context
    # keys with a naming convention: no shared namespace means no collision to avoid in the first
    # place
