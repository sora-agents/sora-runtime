"""Internal; developers implement protocols, they don't call this directly."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sora.cycle import Agent


def load_dotenv(path: str | Path = ".env") -> None:
    """Load ``KEY=value`` lines from a local ``.env`` into ``os.environ`` — a convenience so a local
    ``ANTHROPIC_API_KEY`` (and any other config) is picked up without an explicit ``export``.

    **Real environment variables take precedence**: an existing variable is never overwritten
    (``os.environ.setdefault``), so the process environment always wins over the file — the standard
    precedence. A missing file is a no-op. Blank lines and ``#`` comments are skipped, an optional
    ``export`` prefix is tolerated, and matching single/double quotes around the value are stripped.
    No dependency — the core stays dependency-free; copy ``.env.example`` to ``.env`` to start.
    """
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)  # real env wins — never overwrite


def build_agent(config_path: str) -> Agent:
    """What `sora run` calls before handing off to TerminalSession. This is the one place all the
    wiring actually happens (which memory backend, which transport, which adapters, DecisionCycle
    <-> Agent sharing the same instances) — a developer implementing an agent never writes this."""
    load_dotenv()  # convenience: pick up ANTHROPIC_API_KEY etc. from a local .env if present
    raise NotImplementedError
