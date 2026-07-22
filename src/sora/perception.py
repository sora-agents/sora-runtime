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
    # A single observed environment stimulus — genuine environment stimuli only; an invoked
    # operation's own result is not a Percept (see Activity.pending_operation/last_operation), and
    # neither are agent messages (see WorkingMemory.messages). Properties and signals share this
    # envelope but live in *separate* WorkingMemory stores (properties: a replace-by-(source, name)
    # snapshot; signals: an append log) — the store discriminates them, so there is no `kind` field.
    source: str  # tool id
    payload: Any  # an ObservableProperty (in WorkingMemory.properties) or a Signal (in .signals)
    observed_at: float


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

    def push(self, source: str, item: T) -> None:
        self._queue.put_nowait((source, item))

    async def drain(self) -> AsyncIterator[tuple[str, T]]:
        """Yield everything queued *right now* and stop — one drain per cycle, never blocking on a
        future push. `range(self._queue.qsize())` reads the depth exactly once, when the for-loop
        builds the range, fixing the iteration count to the number of items present at that instant.
        Anything a consumer pushes back while iterating (e.g. a re-queued signal) grows the queue
        but not the range, so it waits for the next drain rather than starving this cycle.
        (`get_nowait()` can't underflow here: this is the single consumer, and no other drain of the
        same sink runs concurrently, so the count can only be reduced by this loop.)"""
        for _ in range(self._queue.qsize()):
            yield self._queue.get_nowait()
