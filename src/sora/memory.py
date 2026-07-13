"""Working, semantic, procedural, and episodic memory modules."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import quote

if TYPE_CHECKING:
    from sora.activity import Activity
    from sora.environment import EnvironmentView, Tool
    from sora.manual import Manual, ToolRecord, WorkspaceRecord
    from sora.perception import Message, Percept
    from sora.types import Plan


class MemoryBackend(Protocol):  # pluggable: file, DB, vector store
    async def get(self, key: str) -> Any: ...

    async def put(self, key: str, value: Any) -> None: ...

    async def query(self, **filters: Any) -> list[Any]: ...


class FileMemoryBackend:
    """The default persistent MemoryBackend: one JSON file per key under a root directory.

    Deals only in JSON-serializable values — the memory modules (semantic/procedural/episodic)
    convert their dataclasses to/from plain dict/list/scalar before touching this. Keeping the
    backend generic is what makes a database/vector-store backend a true drop-in: it never learns
    about sora's specific types.

    Reading re-parses from disk, so a returned value is always a fresh copy — a caller can mutate
    it without corrupting the store (unlike an in-memory dict backend that hands back live refs).
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _path(self, key: str) -> Path:
        # quote(safe="") encodes '/', ':', etc. so URI / <App>__<op> ids map to safe filenames.
        return self._root / f"{quote(key, safe='')}.json"

    async def get(self, key: str) -> Any:
        return await asyncio.to_thread(self._read, self._path(key))

    async def put(self, key: str, value: Any) -> None:
        await asyncio.to_thread(self._write, key, value)

    async def query(self, **filters: Any) -> list[Any]:
        return await asyncio.to_thread(self._scan, filters)

    @staticmethod
    def _read(path: Path) -> Any:
        try:
            with path.open(encoding="utf-8") as f:
                return json.load(f)["value"]
        except FileNotFoundError:
            return None

    def _write(self, key: str, value: Any) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        # Atomic: write a temp file in the same dir, then rename over the target — a crash
        # mid-write never leaves a half-written .json that a later query() would choke on.
        fd, tmp = tempfile.mkstemp(dir=self._root, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"key": key, "value": value}, f)  # self-describing envelope
            os.replace(tmp, self._path(key))
        except BaseException:
            os.unlink(tmp)
            raise

    def _scan(self, filters: dict[str, Any]) -> list[Any]:
        if not self._root.exists():
            return []
        results = []
        for path in sorted(self._root.glob("*.json")):  # *.tmp files are excluded by the glob
            value = self._read(path)
            if not filters or (
                isinstance(value, dict) and all(value.get(k) == v for k, v in filters.items())
            ):
                results.append(value)
        return results


@dataclass
class WorkingMemory:  # transient, in-process, fast
    # Read-only view of the live joined workspaces/tools: the agent reasons over what it's currently
    # connected to; the durable WorkspaceRecord/ToolRecord knowledge stays in SemanticMemory. The
    # concrete instance behind this is the shared EnvironmentRegistry (mutable on DecisionCycle).
    registry: EnvironmentView
    activities: dict[str, Activity] = field(default_factory=dict)
    # stimuli from the environment: properties and signals only
    perceptions: list[Percept] = field(default_factory=list)
    # inbound agent-to-agent communication — kept distinct
    messages: list[Message] = field(default_factory=list)
    focused_tools: dict[str, Tool] = field(default_factory=dict)


class SemanticMemory:  # knowledge about the world: tool types, workspaces, instances
    def __init__(self, backend: MemoryBackend) -> None: ...

    async def retrieve_manual(self, manual_id: str) -> Manual | None:
        raise NotImplementedError

    async def store_manual(self, manual: Manual) -> None:
        raise NotImplementedError

    async def retrieve_workspace_record(self, workspace_id: str) -> WorkspaceRecord | None:
        raise NotImplementedError

    async def store_workspace_record(self, record: WorkspaceRecord) -> None:
        raise NotImplementedError

    async def list_workspace_records(self) -> list[WorkspaceRecord]:
        raise NotImplementedError

    async def retrieve_tool_record(self, tool_id: str) -> ToolRecord | None:
        raise NotImplementedError

    async def store_tool_record(self, record: ToolRecord) -> None:
        raise NotImplementedError

    async def list_tool_records(self) -> list[ToolRecord]:  # reconstitute known instances at boot
        raise NotImplementedError


class ProceduralMemory:
    def __init__(self, backend: MemoryBackend) -> None: ...

    async def retrieve(self, activity: Activity) -> Plan | None:
        """Looks up a cached Plan matching this activity's goal — e.g. exact match or embedding
        similarity, backend-dependent. The cheap path: skips infer() entirely when it hits."""
        raise NotImplementedError

    async def infer(self, activity: Activity) -> Plan:
        """Produces a new multi-step Plan when no cached one fits — the expensive path,
        potentially an LLM call producing a whole sequence of Steps at once, not just the next
        one."""
        raise NotImplementedError

    async def store(self, plan: Plan) -> None:
        """Persists a Plan that was actually followed to completion, so future retrieve() calls
        for similar goals can reuse it. Called by ReflectStrategy on success only — a failed plan
        isn't something future activities should retrieve by default."""
        raise NotImplementedError


class EpisodicMemory:
    def __init__(self, backend: MemoryBackend) -> None: ...

    async def learn(self, activity: Activity, summary: str) -> None:
        raise NotImplementedError

    async def consult(self, activity: Activity) -> list[Any]:
        raise NotImplementedError
