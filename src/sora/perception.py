"""Percepts, messages, and the sink types that feed them into working memory."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from sora.types import Signal


@dataclass(frozen=True)
class Percept:
    source: str  # tool id
    kind: str  # "property" | "signal" — genuine environment stimuli only; an invoked operation's
    payload: Any  # own result is not a Percept (see Activity.pending_operation/last_operation)
    observed_at: float  # and neither are agent messages (see WorkingMemory.messages)


@dataclass(frozen=True)
class Message:
    sender: str
    content: dict[str, Any]
    received_at: float


class SignalSink(Protocol):
    """Narrow, write-only interface: tools push here, they never see WorkingMemory or
    DecisionCycle."""

    def push(self, source: str, signal: Signal) -> None: ...


class NotificationQueueSink[T]:  # was QueueSink — too generic a name to keep
    """Generic FIFO sink: producers push, _observe() drains once per cycle. Concrete backing for
    SignalSink (tool-facing) and for the runtime-internal channel that carries invoke() results —
    both are, structurally, queues of asynchronous notifications awaiting delivery as percepts."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[str, T]] = asyncio.Queue()

    def push(self, source: str, item: T) -> None: ...

    async def drain(self) -> AsyncIterator[tuple[str, T]]:
        raise NotImplementedError
