"""Working, semantic, procedural, and episodic memory modules."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import quote

# Imported at runtime (not just for typing): SemanticMemory reconstructs these dataclasses from
# the plain dicts the backend hands back. manual.py / environment.py only import their sora deps
# under TYPE_CHECKING, so importing them here introduces no cycle.
from sora.environment import WorkspaceOrigin
from sora.manual import (
    Manual,
    ObservablePropertySpecification,
    OperationSpecification,
    SignalSpecification,
    ToolRecord,
    WorkspaceRecord,
)

if TYPE_CHECKING:
    from sora.activity import Activity
    from sora.environment import EnvironmentView, Tool
    from sora.perception import Message, Percept
    from sora.types import Plan


class MemoryBackend(Protocol):  # pluggable: file, DB, vector store
    async def get(self, key: str) -> Any: ...

    async def put(self, key: str, value: Any) -> None: ...

    async def query(self, **filters: Any) -> list[Any]:
        """Returns stored values whose top-level fields match every filter (conjunctive exact
        equality); non-dict values and any value missing/mismatching a filter are excluded, and no
        filters returns everything. Callers (e.g. EpisodicMemory.consult) rely on this equality
        contract, so a non-file backend must honor it rather than substitute fuzzy matching."""
        ...


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


# Discriminators for the three record kinds sharing one backend. They serve double duty: as a
# storage-key prefix (so the three independent id-spaces can't clobber each other's files) and as
# a stored `kind` field (so a query() lists just one kind — see FileMemoryBackend.query).
_MANUAL = "manual"
_WORKSPACE_RECORD = "workspace_record"
_TOOL_RECORD = "tool_record"


class SemanticMemory:  # knowledge about the world: tool types, workspaces, instances
    """Durable store for manuals and workspace/tool records. Owns the dataclass<->dict
    (de)serialization so the backend stays a generic key->JSON store: it converts to plain
    dicts on the way in and rebuilds typed instances on the way out."""

    def __init__(self, backend: MemoryBackend) -> None:
        self._backend = backend

    async def retrieve_manual(self, manual_id: str) -> Manual | None:
        value = await self._backend.get(f"{_MANUAL}:{manual_id}")
        return None if value is None else _manual_from_dict(value)

    async def store_manual(self, manual: Manual) -> None:
        await self._backend.put(f"{_MANUAL}:{manual.id}", {"kind": _MANUAL, **asdict(manual)})

    async def retrieve_workspace_record(self, workspace_id: str) -> WorkspaceRecord | None:
        value = await self._backend.get(f"{_WORKSPACE_RECORD}:{workspace_id}")
        return None if value is None else _workspace_record_from_dict(value)

    async def store_workspace_record(self, record: WorkspaceRecord) -> None:
        key = f"{_WORKSPACE_RECORD}:{record.id}"
        await self._backend.put(key, {"kind": _WORKSPACE_RECORD, **asdict(record)})

    async def list_workspace_records(self) -> list[WorkspaceRecord]:
        values = await self._backend.query(kind=_WORKSPACE_RECORD)
        return [_workspace_record_from_dict(v) for v in values]

    async def retrieve_tool_record(self, tool_id: str) -> ToolRecord | None:
        value = await self._backend.get(f"{_TOOL_RECORD}:{tool_id}")
        return None if value is None else _tool_record_from_dict(value)

    async def store_tool_record(self, record: ToolRecord) -> None:
        key = f"{_TOOL_RECORD}:{record.id}"
        await self._backend.put(key, {"kind": _TOOL_RECORD, **asdict(record)})

    async def list_tool_records(self) -> list[ToolRecord]:  # reconstitute known instances at boot
        values = await self._backend.query(kind=_TOOL_RECORD)
        return [_tool_record_from_dict(v) for v in values]


# --- deserialization: rebuild typed instances from the plain dicts the backend returns ---------
# asdict() flattens nested dataclasses to dicts on the way in; these undo exactly that, dropping the
# stored `kind` discriminator (which isn't a dataclass field). A fresh instance per call is what
# gives callers copy isolation even for the mutable lists inside a (frozen) Manual.


def _manual_from_dict(d: dict[str, Any]) -> Manual:
    return Manual(
        id=d["id"],
        metadata=d["metadata"],
        description=d["description"],
        observable_properties=[
            ObservablePropertySpecification(**p) for p in d["observable_properties"]
        ],
        signals=[SignalSpecification(**s) for s in d["signals"]],
        operations=[OperationSpecification(**o) for o in d["operations"]],
        usage_protocols=d["usage_protocols"],
    )


def _workspace_record_from_dict(d: dict[str, Any]) -> WorkspaceRecord:
    return WorkspaceRecord(
        id=d["id"],
        origin=WorkspaceOrigin(**d["origin"]),
        discovered_at=d["discovered_at"],
        last_seen_at=d["last_seen_at"],
    )


def _tool_record_from_dict(d: dict[str, Any]) -> ToolRecord:
    return ToolRecord(
        id=d["id"],
        manual_id=d["manual_id"],
        workspace_id=d["workspace_id"],
        address=d["address"],
        discovered_at=d["discovered_at"],
        last_seen_at=d["last_seen_at"],
    )


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
    """Records a summary of each completed activity and retrieves the ones relevant to a new
    activity. Relevance is goal-equality — the same cheap, deterministic proxy ProceduralMemory
    uses; embedding/LLM similarity is deferred so the default stays reproducible."""

    def __init__(self, backend: MemoryBackend) -> None:
        self._backend = backend

    async def learn(self, activity: Activity, summary: str) -> None:
        # One episode per activity, keyed by its id: re-learning the same activity overwrites rather
        # than accumulating duplicates. goal is stored top-level so consult can filter on it through
        # the backend's exact-match query().
        await self._backend.put(
            activity.id,
            {"activity_id": activity.id, "goal": activity.goal, "summary": summary},
        )

    async def consult(self, activity: Activity) -> list[Any]:
        # query() re-reads from disk, so results are fresh copies a caller can mutate without
        # corrupting the store.
        return await self._backend.query(goal=activity.goal)
