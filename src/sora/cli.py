"""The runtime's minimal terminal interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sora.cycle import Agent


class TerminalSession:
    """Streams cycle output to stdout; queues stdin as Message(sender="user", ...) — not a
    Percept, since terminal input is user communication, not environment stimuli. No UI beyond
    this."""

    def __init__(self, agent: Agent, verbose: bool = False) -> None: ...

    async def run(self) -> None: ...


def main() -> None:
    # Not in the README sketch — added so `[project.scripts] sora = "sora.cli:main"` resolves to a
    # real callable. Real body (parse args, call build_agent, run a TerminalSession) lands in
    # Phase 3 step 15 ("CLI polish").
    raise NotImplementedError
