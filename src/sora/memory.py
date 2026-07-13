"""Working, semantic, procedural, and episodic memory modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

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
