"""Templates + file generation for `sora init` — scaffolds a minimal, immediately-runnable example
project demonstrating the pieces a real integration wires together: `agent.yaml`, a hand-written
`WorkspaceAdapter`, and a hand-authored manual. No external MCP server is involved (none exists for
a "clock" tool to depend on) — the scaffolded `clock_tool.py` is a small, self-contained
`WorkspaceAdapter`/`Workspace`/`Tool` trio standing in for one, so the example runs with zero extra
setup beyond an LLM key."""

from __future__ import annotations

from pathlib import Path

_PYPROJECT_TEMPLATE = """\
[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.12"
# sora-runtime isn't published to PyPI yet, so the dependency points straight at the git repo (no
# release tag exists yet either, so this tracks `main` -- switch to a bare "sora-runtime[llm]"
# version constraint once a real release exists).
dependencies = ["sora-runtime[llm] @ git+https://github.com/sora-agents/sora-runtime.git"]
"""

_AGENT_YAML_TEMPLATE = """\
agent:
  name: {name}
  strategies:
    reason: sora.strategies.DefaultReasonStrategy
  memory:
    semantic: file://./.sora/memory/semantic
    procedural: file://./.sora/memory/procedural
    episodic: file://./.sora/memory/episodic
  llm:
    model: claude-opus-4-8
  workspaces:
    - origin:
        adapter: local
        address: "local:clock"
      workspace_id: clock
      factory: clock_tool.build_adapter
"""

_MANUAL_CLOCK_MD = """\
# Tool Metadata
category: Utilities / Time
id: clock

# Functional Description
A simple clock that reports the current wall-clock time on request.

# Observable Properties
(none)

# Signals
(none)

# Operations
- get_time(): returns the current time as an ISO-8601 string.

# Usage Protocols & Safety
No constraints. get_time is read-only and completes synchronously.
"""

_CLOCK_TOOL_PY = '''\
"""A minimal, hand-written clock tool -- no external MCP server needed.

Demonstrates the WorkspaceAdapter/Workspace/Tool seam directly, the way an adapter (MCP, WoT, ...)
imports an externally-defined tool. By design, the S-ORA runtime does not author tools: this is
an illustrative exception `sora init` writes for you, not a pattern the runtime ships. A real
integration typically imports tools through an adapter instead of hand-writing one like this.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sora.environment import Tool, Workspace, WorkspaceOrigin
from sora.manual import Manual, MarkdownManualParser, ToolRecord, WorkspaceRecord
from sora.perception import SignalSink
from sora.types import ObservableProperty, OperationAck

_MANUAL = MarkdownManualParser().parse(
    (Path(__file__).parent / "manuals" / "clock.md").read_text(encoding="utf-8")
)


class ClockTool:  # satisfies sora.environment.Tool
    id = "clock"
    manual: Manual = _MANUAL
    address: str | None = None

    async def invoke(self, operation_name: str, **params: Any) -> OperationAck:
        if operation_name != "get_time":
            return OperationAck(ok=False, result=f"unknown operation {operation_name!r}")
        return OperationAck(ok=True, result=datetime.now(timezone.utc).isoformat())

    async def focus(self, sink: SignalSink) -> None:
        pass  # the clock has no signals

    async def unfocus(self) -> None:
        pass

    def observe(self) -> list[ObservableProperty]:
        return []  # the clock has no observable properties


class ClockWorkspace:  # satisfies sora.environment.Workspace
    def __init__(self, ws_id: str, origin: WorkspaceOrigin) -> None:
        self.id = ws_id
        self.origin = origin
        self._tools: list[Tool] = [ClockTool()]

    def tools(self) -> list[Tool]:
        return self._tools

    async def close(self) -> None:
        pass


class ClockAdapter:  # satisfies sora.environment.WorkspaceAdapter
    name = "local"

    def __init__(self, origin: WorkspaceOrigin) -> None:
        self._origin = origin

    async def discover(self) -> list[Workspace]:
        return [ClockWorkspace("clock", self._origin)]

    async def connect(
        self,
        workspace_record: WorkspaceRecord,
        tool_records: list[ToolRecord],
        manuals: dict[str, Manual],
    ) -> Workspace:
        return ClockWorkspace(workspace_record.id, workspace_record.origin)


def build_adapter(origin: WorkspaceOrigin) -> ClockAdapter:
    return ClockAdapter(origin)
'''


def write_project(project_dir: Path) -> None:
    """Scaffold a minimal, immediately-runnable example agent into ``project_dir``:
    ``pyproject.toml``, ``agent.yaml``, ``manuals/clock.md``, and ``clock_tool.py``. Refuses to
    touch an already-existing target (file or directory, empty or not) — no merge/overwrite."""
    if project_dir.exists():
        raise FileExistsError(f"{project_dir} already exists -- sora init won't overwrite it")
    name = project_dir.name
    project_dir.mkdir(parents=True)
    (project_dir / "pyproject.toml").write_text(
        _PYPROJECT_TEMPLATE.format(name=name), encoding="utf-8"
    )
    (project_dir / "agent.yaml").write_text(
        _AGENT_YAML_TEMPLATE.format(name=name), encoding="utf-8"
    )
    manuals_dir = project_dir / "manuals"
    manuals_dir.mkdir()
    (manuals_dir / "clock.md").write_text(_MANUAL_CLOCK_MD, encoding="utf-8")
    (project_dir / "clock_tool.py").write_text(_CLOCK_TOOL_PY, encoding="utf-8")
