"""Tool manuals and the durable records of discovered workspaces/tool instances."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import yaml

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


class ManualMergeError(ValueError):
    """Raised when the two provenance channels for one Manual.id can't be reconciled — a mismatched
    id (the reconciliation key — probably a wrong pairing), or an authored interface that diverges
    from the adapter's structured specs (see merge_manuals / ADR-0018)."""


def merge_manuals(adapter: Manual, authored: Manual) -> Manual:
    """Reconcile a Manual's two provenance channels (ADR-0015/ADR-0018), keyed by Manual.id.

    Policy: the *adapter* channel owns the structured usage interface (operations / observable
    properties / signals, with their JSON Schema — native and authoritative); the hand-authored
    *Markdown* channel owns the prose (`raw_text`, and `description` when it carries one). Metadata
    is a union with the authored side winning on conflict (the adapter's `source` survives, since an
    authored manual won't set it).

    Fail loud, not silently merge, on a genuine mismatch: the ids must match (else it's a wrong
    pairing), and where the authored manual *declares* a structured interface for an affordance kind
    (an optional per-section block — see MarkdownManualParser), its names must match the adapter's
    and every required key it names must exist in the adapter's schema. An affordance kind the
    authored manual leaves undeclared is not validated — adapter specs pass through (per-kind
    opt-in)."""
    if adapter.id != authored.id:
        raise ManualMergeError(
            f"cannot merge manuals with different ids: adapter {adapter.id!r} vs authored "
            f"{authored.id!r} (the id is the reconciliation key — likely a wrong pairing)"
        )
    _validate_interface(adapter.operations, authored.operations, "operation", adapter.id)
    _validate_interface(
        adapter.observable_properties,
        authored.observable_properties,
        "observable property",
        adapter.id,
    )
    _validate_interface(adapter.signals, authored.signals, "signal", adapter.id)
    return Manual(
        id=adapter.id,
        metadata={**adapter.metadata, **authored.metadata},
        description=authored.description or adapter.description,
        # Structured specs are the adapter channel's — full, native schema. The authored side's
        # partial specs were only for validation above and are discarded here.
        observable_properties=adapter.observable_properties,
        signals=adapter.signals,
        operations=adapter.operations,
        raw_text=adapter.raw_text if authored.raw_text is None else authored.raw_text,
    )


def _spec_schema(spec: Any) -> dict[str, Any]:
    """The JSON-Schema-shaped dict a spec carries — `parameters` on an operation, `schema` on an
    observable property / signal. One accessor so `_validate_interface` stays kind-agnostic."""
    return spec.parameters if isinstance(spec, OperationSpecification) else spec.schema


def _validate_interface(
    adapter_specs: list[Any], authored_specs: list[Any], kind: str, manual_id: str
) -> None:
    if not authored_specs:  # per-kind opt-in: nothing authored for this kind -> nothing to validate
        return
    adapter_by_name = {s.name: s for s in adapter_specs}
    authored_names = {s.name for s in authored_specs}
    adapter_names = set(adapter_by_name)
    if authored_names != adapter_names:
        raise ManualMergeError(
            f"authored manual {manual_id!r} {kind} names diverge from the adapter's: "
            f"authored-only={sorted(authored_names - adapter_names)}, "
            f"adapter-only={sorted(adapter_names - authored_names)}"
        )
    for spec in authored_specs:
        # Only cross-check required keys the adapter actually describes: a missing `properties` key
        # means the adapter carries no field schema for this affordance (e.g. ARE synthesizes an
        # empty schema for observable properties), so we can validate names but not fields.
        adapter_props = _spec_schema(adapter_by_name[spec.name]).get("properties")
        if adapter_props is None:
            continue
        missing = [k for k in _spec_schema(spec).get("required", []) if k not in adapter_props]
        if missing:
            raise ManualMergeError(
                f"authored manual {manual_id!r} {kind} {spec.name!r} requires {sorted(missing)}, "
                f"absent from the adapter schema (keys: {sorted(adapter_props)})"
            )


class ManualParser(Protocol):  # Markdown by default, XML pluggable
    def parse(self, raw: str) -> Manual: ...


class ManualSource(Protocol):
    """A resolver from a type-level Manual.id to a hand-authored Manual, injected into an adapter so
    it can pair authored semantics with the structured specs it synthesizes (ADR-0018). Owning
    fetch+parse (returns a parsed Manual, not raw text) keeps a remote manual catalogue serving a
    non-Markdown format a drop-in for the local DirectoryManualSource."""

    async def get(self, manual_id: str) -> Manual | None: ...


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
# description, and the verbatim raw_text — and does NOT regex-lift the prose bullet lists into the
# structured spec fields: that extraction proved brittle and section prose is served lazily from
# raw_text via Manual.section (see ADR-0015). The one exception is an *explicit, optional* fenced
# interface block a section may carry (```yaml ... ```): a names-level declaration (operation /
# property / signal names + their required keys) the parser lifts into the structured spec fields so
# merge_manuals can cross-validate it against an adapter's native schema (ADR-0018). No block ->
# spec list stays empty. The other Manual producer is a WorkspaceAdapter synthesizing from a native
# description; the two reconcile by Manual.id.
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


def _extract_fenced_block(section_body: str | None) -> str | None:
    """The first fenced (``` ... ```) code block inside a section body, or None. The info string
    (```yaml / ```json / bare) is ignored — YAML is a JSON superset, so one parser reads either."""
    if section_body is None:
        return None
    collected: list[str] = []
    inside = False
    for line in section_body.splitlines():
        if line.strip().startswith("```"):
            if inside:
                return "\n".join(collected)
            inside = True
            continue
        if inside:
            collected.append(line)
    return None


def _interface_entries(section_body: str | None) -> list[tuple[str, list[str]]]:
    """Parse a section's optional interface block into (name, required-keys) pairs; [] if no block.
    The block is a list of ``{name: <str>, required: [<str>, ...]}`` (required optional)."""
    block = _extract_fenced_block(section_body)
    if block is None:
        return []
    try:
        entries = yaml.safe_load(block)
        return [(item["name"], list(item.get("required", []) or [])) for item in entries]
    except (yaml.YAMLError, TypeError, KeyError) as exc:
        raise ManualParseError(f"malformed interface block: {exc!r}\n---\n{block}") from exc


def _required_schema(required: list[str]) -> dict[str, Any]:
    """A minimal JSON-Schema-shaped dict capturing just the required keys — enough for merge_manuals
    to cross-validate against an adapter's full schema; the adapter supplies the real shapes."""
    return {"properties": {k: {} for k in required}, "required": list(required)}


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
            observable_properties=[
                ObservablePropertySpecification(name=n, description="", schema=_required_schema(r))
                for n, r in _interface_entries(sections.get(ManualSection.OBSERVABLE_PROPERTIES))
            ],
            signals=[
                SignalSpecification(name=n, description="", schema=_required_schema(r))
                for n, r in _interface_entries(sections.get(ManualSection.SIGNALS))
            ],
            operations=[
                OperationSpecification(name=n, description="", parameters=_required_schema(r))
                for n, r in _interface_entries(sections.get(ManualSection.OPERATIONS))
            ],
            raw_text=raw,
        )


class DirectoryManualSource:  # satisfies the ManualSource Protocol
    """Serves hand-authored Manuals from a directory of ``*.md`` files, indexed by their parsed
    Manual.id (not filename — the id is the reconciliation key). Parses every file once, lazily, on
    first lookup. A missing directory yields an empty index (no authored manuals), not an error."""

    def __init__(self, root: str | Path, parser: ManualParser | None = None) -> None:
        self._root = Path(root)
        self._parser = parser or MarkdownManualParser()
        self._index: dict[str, Manual] | None = None

    async def get(self, manual_id: str) -> Manual | None:
        if self._index is None:
            self._index = await asyncio.to_thread(self._build_index)
        return self._index.get(manual_id)

    def _build_index(self) -> dict[str, Manual]:
        if not self._root.exists():
            return {}
        index: dict[str, Manual] = {}
        for path in sorted(self._root.glob("*.md")):
            manual = self._parser.parse(path.read_text(encoding="utf-8"))
            index[manual.id] = manual
        return index
