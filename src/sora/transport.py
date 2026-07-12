"""Pluggable message transport: A2A, HTTP, in-process."""

from __future__ import annotations

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
