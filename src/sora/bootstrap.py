"""Internal; developers implement protocols, they don't call this directly."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sora.cycle import Agent


def build_agent(config_path: str) -> Agent:
    """What `sora run` calls before handing off to TerminalSession. This is the one place all the
    wiring actually happens (which memory backend, which transport, which adapters, DecisionCycle
    <-> Agent sharing the same instances) — a developer implementing an agent never writes this."""
    raise NotImplementedError
