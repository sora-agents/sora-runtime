"""Tool manuals and the durable records of discovered workspaces/tool instances."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from sora.environment import WorkspaceOrigin


class ManualSection(StrEnum):
    """The six canonical `#`-headed sections of a tool manual (README's manual structure). Each
    member's value is the exact heading text, so it is both the key `Manual.section()` slices by and
    the heading the parser matches — one source of truth, so no section-title string literal can
    drift or be mistyped across the parser and its consumers."""

    METADATA = "Tool Metadata"
    DESCRIPTION = "Functional Description"
    OBSERVABLE_PROPERTIES = "Observable Properties"
    SIGNALS = "Signals"
    OPERATIONS = "Operations"
    USAGE_AND_SAFETY = "Usage Protocols & Safety"


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
    # The structured usage-interface specs. The adapter provenance channel fills these from a native
    # description (MCP schema / WoT TD affordance); the hand-authored Markdown channel leaves them
    # empty and carries its content in raw_text instead (see ADR-0015).
    observable_properties: list[ObservablePropertySpecification]
    signals: list[SignalSpecification]
    operations: list[OperationSpecification]
    raw_text: str | None = None  # verbatim authored source (Markdown channel); None if synthesized

    def section(self, name: str) -> str | None:
        """A `#`-headed section of the authored manual, sliced lazily from raw_text; None if there
        is no raw_text or no such section. Pass a `ManualSection` for a canonical section (the
        common case, typo-proof); a bare string still works for slicing any heading."""
        if self.raw_text is None:
            return None
        body = _split_sections(self.raw_text).get(name)
        return None if body is None else body.strip()


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


# ------------------------------------------------------------------------------------------------
# Markdown parser for the clean manual format. It produces a Manual *envelope* — id, metadata,
# description, and the verbatim raw_text — and does NOT lift the bullet-list sections into the
# structured spec fields: that field extraction proved brittle and no consumer reads it, so section
# prose is served lazily from raw_text via Manual.section (see ADR-0015). The other Manual producer
# is a WorkspaceAdapter synthesizing from a native description; the two reconcile by Manual.id.
# ------------------------------------------------------------------------------------------------
class ManualParseError(ValueError):
    """Raised when Markdown can't become a Manual — notably a manual with no derivable id."""


def _split_sections(raw: str) -> dict[str, str]:
    """Split the manual into its `# `-headed sections (text before the first heading is ignored)."""
    sections: dict[str, str] = {}
    heading: str | None = None
    body: list[str] = []
    for line in raw.splitlines():
        if line.startswith("# "):
            if heading is not None:
                sections[heading] = "\n".join(body)
            heading, body = line[2:].strip(), []
        elif heading is not None:
            body.append(line)
    if heading is not None:
        sections[heading] = "\n".join(body)
    return sections


def _parse_metadata(block: str) -> tuple[str, dict[str, Any]]:
    manual_id = ""
    metadata: dict[str, Any] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "id":
            manual_id = value
        else:
            metadata[key] = value
    if not manual_id:
        raise ManualParseError(f"manual has no `id:` in its `# {ManualSection.METADATA}` section")
    return manual_id, metadata


class MarkdownManualParser:  # satisfies the ManualParser Protocol (Markdown is the default format)
    def parse(self, raw: str) -> Manual:
        sections = _split_sections(raw)
        if ManualSection.METADATA not in sections:
            raise ManualParseError(f"manual is missing its `# {ManualSection.METADATA}` section")
        manual_id, metadata = _parse_metadata(sections[ManualSection.METADATA])
        return Manual(
            id=manual_id,
            metadata=metadata,
            description=sections.get(ManualSection.DESCRIPTION, "").strip(),
            observable_properties=[],
            signals=[],
            operations=[],
            raw_text=raw,
        )
