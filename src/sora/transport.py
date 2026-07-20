"""Pluggable message transport: A2A, HTTP, in-process."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from sora.perception import Message


class MessageTransport(Protocol):  # pluggable: A2A, HTTP, in-process
    async def send(self, to: str, content: dict[str, Any]) -> None: ...

    # NOTE: non-async `def` returning an AsyncIterator, so `async for m in
    # transport.receive()` works as the README's DefaultObserveStrategy already writes it. The API
    # sketch declared this `async def receive(...) -> AsyncIterator[Message]`, which is
    # self-inconsistent with that call site (you'd have to `await` it first). See
    # docs/phase-2-findings.md; a README diff is proposed, not yet applied.
    def receive(self) -> AsyncIterator[Message]: ...


class InProcessTransport:
    """The default transport for a single-agent runtime: an in-process inbox with no network. The
    runtime drains ``receive()`` once per Observe; whoever holds the agent (the CLI, the showcase
    runner, a test) delivers inbound goals by calling ``submit()`` — that's how the scenario's
    initial task prompt reaches ``working_memory.messages``. ``send()`` records outbound content
    rather than routing it anywhere; a genuine peer-to-peer transport (A2A/HTTP) is the two-agent
    case and is deferred.

    ``receive()`` is a non-async ``def`` returning an async generator (mirroring the Protocol), so
    ``async for m in transport.receive()`` reads exactly what is queued *now* and stops — never
    blocking the cycle on a future ``submit()``."""

    def __init__(self) -> None:
        self._inbox: asyncio.Queue[Message] = asyncio.Queue()
        self.sent: list[tuple[str, dict[str, Any]]] = []  # outbound log, for tests/inspection

    async def send(self, to: str, content: dict[str, Any]) -> None:
        self.sent.append((to, content))

    def submit(self, message: Message) -> None:
        """Deliver an inbound message (e.g. the user's/scenario's goal) for the next Observe."""
        self._inbox.put_nowait(message)

    def receive(self) -> AsyncIterator[Message]:
        async def _drain() -> AsyncIterator[Message]:
            # Snapshot the depth once, same non-starvation contract as NotificationQueueSink.drain.
            for _ in range(self._inbox.qsize()):
                yield self._inbox.get_nowait()

        return _drain()
