"""Tool manuals and the durable records of discovered workspaces/tool instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from sora.environment import WorkspaceOrigin


@dataclass(frozen=True)
class OperationSpecification:  # was Operation — renamed for symmetry with the two specs below
    name: str
    description: str
    parameters: dict[str, Any]  # JSON-Schema-shaped


@dataclass(frozen=True)
class ObservablePropertySpecification:
    name: str
    description: str
    schema: dict[str, Any]  # JSON-Schema-shaped, matching e.g. a WoT property affordance


@dataclass(frozen=True)
class SignalSpecification:
    name: str
    description: str
    schema: dict[str, Any]  # JSON-Schema-shaped, matching e.g. a WoT event affordance


@dataclass(frozen=True)
class Manual:
    id: str  # type identifier — NOT a tool instance id; shared across instances
    metadata: dict[str, Any]
    description: str
    observable_properties: list[ObservablePropertySpecification]
    signals: list[SignalSpecification]
    operations: list[OperationSpecification]
    usage_protocols: str


class ManualParser(Protocol):  # Markdown by default, XML pluggable
    def parse(self, raw: str) -> Manual: ...


@dataclass(frozen=True)
class WorkspaceRecord:
    """A WorkspaceOrigin that's actually been connected to, plus the identity/bookkeeping only
    assigned once that connection exists. Not duplicated onto every ToolRecord that references it;
    individual tools may still override the address (see ToolRecord.address)."""

    id: str  # matches Workspace.id once live
    origin: WorkspaceOrigin
    discovered_at: float
    last_seen_at: float


@dataclass(frozen=True)
class ToolRecord:
    """Durable record of a discovered tool instance — many records can share one manual_id,
    and every record from the same connection shares one workspace_id."""

    id: str  # instance id, matches Tool.id once live
    manual_id: str
    workspace_id: str  # references WorkspaceRecord.id
    address: str | None  # overrides WorkspaceRecord.origin.address; e.g. a device's own endpoint
    discovered_at: float
    last_seen_at: float
