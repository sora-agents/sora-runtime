"""Pluggable message transport: A2A, HTTP, in-process."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from sora.perception import Message


class MessageTransport(Protocol):  # pluggable: A2A, HTTP, in-process
    async def send(self, to: str, content: dict[str, Any]) -> None: ...

    async def receive(self) -> AsyncIterator[Message]: ...
