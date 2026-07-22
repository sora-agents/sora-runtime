"""Contract for the clean-format Markdown ``ManualParser``.

``MarkdownManualParser`` turns S-ORA's clean manual format (the ``#``-sectioned Markdown authored by
hand and mirrored in EXAMPLES.md) into a ``Manual`` **envelope**: it fills ``id``, ``metadata``,
``description``, and the verbatim ``raw_text``, and leaves the structured spec lists
(``observable_properties``/``signals``/``operations``) empty — those are the adapter channel's to
fill from a native description (see ADR-0015). The hand-authored channel does *not* parse Markdown
bullet lists into typed fields; that lifting proved brittle and no consumer reads it. Section-level
prose (Operations, Usage & Safety, …) is reachable as a lazy view over ``raw_text`` via
``Manual.section(...)``.

This suite pins that contract against the committed fixtures in ``tests/fixtures/manuals/`` — the
parser must accept them exactly as written, since they are byte-synced to EXAMPLES.md (see
``scripts/check_manuals_sync.py``).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sora.manual import (
    DirectoryManualSource,
    Manual,
    ManualMergeError,
    ManualParseError,
    ManualSection,
    MarkdownManualParser,
    ObservablePropertySpecification,
    OperationSpecification,
    merge_manuals,
)

MANUALS = Path(__file__).parent / "fixtures" / "manuals"


def load(name: str) -> str:
    return (MANUALS / name).read_text()


# ------------------------------------------------------------------------------------------------
# Metadata & the mandatory id
# ------------------------------------------------------------------------------------------------
def test_metadata_id_and_dict() -> None:
    m = MarkdownManualParser().parse(load("water-pump.md"))
    assert m.id == "hydraulic_control"
    assert m.metadata == {
        "category": "Critical Infrastructure / Fluid Dynamics",
        "version": "4.0.0",
    }


def test_metadata_excludes_id() -> None:
    m = MarkdownManualParser().parse(load("clock.md"))
    assert m.id == "clock"
    assert m.metadata == {"category": "Utilities / Time"}


def test_robotic_arm_metadata_records_wot_td_mapping() -> None:
    m = MarkdownManualParser().parse(load("robotic-arm.md"))
    assert m.id == "robotic-arm"
    assert m.metadata["wot_td"] == "urn:cherrybot"


def test_missing_metadata_is_rejected() -> None:
    with pytest.raises(ManualParseError):
        MarkdownManualParser().parse(load("invalid/missing-metadata.md"))


# ------------------------------------------------------------------------------------------------
# Envelope: description + verbatim raw_text; structured lists stay empty
# ------------------------------------------------------------------------------------------------
def test_description_from_functional_description() -> None:
    m = MarkdownManualParser().parse(load("water-pump.md"))
    assert m.description.startswith("Manages the generation of hydraulic pressure")
    assert "reactor_core cooling" in m.description


def test_raw_text_is_verbatim() -> None:
    raw = load("water-pump.md")
    m = MarkdownManualParser().parse(raw)
    assert m.raw_text == raw  # byte-identical, including the trailing newline


def test_structured_lists_are_empty_from_markdown_channel() -> None:
    for name in ["water-pump.md", "clock.md", "blinds.md", "video-stream.md", "robotic-arm.md"]:
        m = MarkdownManualParser().parse(load(name))
        assert m.observable_properties == []
        assert m.signals == []
        assert m.operations == []


# ------------------------------------------------------------------------------------------------
# Lazy section() view over raw_text
# ------------------------------------------------------------------------------------------------
def test_section_returns_operations_prose() -> None:
    m = MarkdownManualParser().parse(load("robotic-arm.md"))
    ops = m.section(ManualSection.OPERATIONS)
    assert ops is not None
    assert "move_to" in ops and "open_gripper" in ops and "close_gripper" in ops
    # section text is scoped to that section only
    assert ManualSection.METADATA not in ops
    assert "target_reached" in ops  # the Behavior sub-bullet is part of the Operations prose


def test_section_returns_usage_and_safety() -> None:
    m = MarkdownManualParser().parse(load("water-pump.md"))
    usage = m.section(ManualSection.USAGE_AND_SAFETY)
    assert usage is not None
    assert "water hammer" in usage
    assert "Sequence:" in usage


def test_section_of_none_sentinel_is_empty_text() -> None:
    m = MarkdownManualParser().parse(load("clock.md"))
    props = m.section(ManualSection.OBSERVABLE_PROPERTIES)
    assert props is not None
    assert props.strip() == "(none)"


def test_section_absent_returns_none() -> None:
    # section() is a generic slicer: a non-canonical heading (a bare string) just returns None.
    m = MarkdownManualParser().parse(load("clock.md"))
    assert m.section("Nonexistent Section") is None


def test_section_without_raw_text_returns_none() -> None:
    # A Manual with no authored source (e.g. adapter-synthesized) has nothing to slice.
    m = MarkdownManualParser().parse(load("clock.md"))
    assert replace(m, raw_text=None).section(ManualSection.OPERATIONS) is None


# ------------------------------------------------------------------------------------------------
# Whole-manual sanity across every committed clean fixture
# ------------------------------------------------------------------------------------------------
def test_all_clean_fixtures_parse_to_envelopes() -> None:
    for name in ["water-pump.md", "clock.md", "blinds.md", "video-stream.md", "robotic-arm.md"]:
        raw = load(name)
        m = MarkdownManualParser().parse(raw)
        assert isinstance(m.id, str) and m.id
        assert m.raw_text == raw
        assert m.description  # every fixture has a Functional Description


# ------------------------------------------------------------------------------------------------
# Authored interface block (ADR-0018) — an optional per-section fenced ```yaml block declaring
# names-level structure (operation/property/signal names + their required keys), lifted into the
# structured spec fields the adapter channel otherwise owns exclusively.
# ------------------------------------------------------------------------------------------------
_INTERFACE_MANUAL = """# Tool Metadata
id: thermostat

# Functional Description
Controls ambient temperature.

# Observable Properties
```yaml
- name: temperature
  required: [unit]
```
- temperature (integer): current reading in the configured unit.

# Signals
(none)

# Operations
```yaml
- name: set_temperature
  required: [value]
- name: set_mode
```
- set_temperature(value): sets the target temperature.
- set_mode(mode): switches operating mode.

# Usage Protocols & Safety
No special precautions.
"""


def test_interface_block_lifts_operation_names_and_required_keys() -> None:
    m = MarkdownManualParser().parse(_INTERFACE_MANUAL)
    ops = {op.name: op.parameters for op in m.operations}
    assert set(ops) == {"set_temperature", "set_mode"}
    assert ops["set_temperature"] == {"properties": {"value": {}}, "required": ["value"]}
    assert ops["set_mode"] == {"properties": {}, "required": []}


def test_interface_block_lifts_observable_property_names_and_required_keys() -> None:
    m = MarkdownManualParser().parse(_INTERFACE_MANUAL)
    assert [p.name for p in m.observable_properties] == ["temperature"]
    assert m.observable_properties[0].schema == {"properties": {"unit": {}}, "required": ["unit"]}


def test_signals_section_without_a_block_stays_empty() -> None:
    m = MarkdownManualParser().parse(_INTERFACE_MANUAL)
    assert m.signals == []  # "(none)" prose, no fenced block


def test_no_interface_block_leaves_structured_specs_empty() -> None:
    m = MarkdownManualParser().parse(load("water-pump.md"))  # no fenced yaml block in this fixture
    assert m.operations == []
    assert m.observable_properties == []
    assert m.signals == []


def test_raw_text_stays_verbatim_including_interface_block() -> None:
    m = MarkdownManualParser().parse(_INTERFACE_MANUAL)
    assert m.raw_text == _INTERFACE_MANUAL
    ops_prose = m.section(ManualSection.OPERATIONS)
    assert ops_prose is not None and "```yaml" in ops_prose  # the prose view keeps the block too


def test_malformed_interface_block_raises_parse_error() -> None:
    bad = _INTERFACE_MANUAL.replace(
        "- name: set_temperature\n  required: [value]", "not: [valid, yaml, {"
    )
    with pytest.raises(ManualParseError, match="malformed interface block"):
        MarkdownManualParser().parse(bad)


# The operations interface block may additionally declare `completes_on: <signal>` — the domain
# signal that marks a long-running op's real completion. Lifted into OperationSpecification so the
# blocked-state machinery can mechanically suspend/resume; absent means a synchronous op (None).
_COMPLETION_MANUAL = """# Tool Metadata
id: robotic-arm

# Functional Description
A robotic arm.

# Operations
```yaml
- name: move_to
  required: [speed, target]
  completes_on: target_reached
- name: open_gripper
```
- move_to(speed, target): long-running motion; completion signalled by target_reached.
- open_gripper(): opens the gripper.

# Usage Protocols & Safety
Suspend after move_to until target_reached.
"""


def test_interface_block_lifts_operation_completion_signal() -> None:
    m = MarkdownManualParser().parse(_COMPLETION_MANUAL)
    ops = {op.name: op.completion_signal for op in m.operations}
    assert ops == {"move_to": "target_reached", "open_gripper": None}


# ------------------------------------------------------------------------------------------------
# merge_manuals — reconciling the adapter and hand-authored provenance channels (ADR-0015/ADR-0018)
# ------------------------------------------------------------------------------------------------
def _adapter_manual(
    *,
    manual_id: str = "pump",
    operations: list[OperationSpecification] | None = None,
    observable_properties: list[ObservablePropertySpecification] | None = None,
) -> Manual:
    return Manual(
        id=manual_id,
        metadata={"source": "mcp"},
        description="adapter description",
        observable_properties=observable_properties or [],
        signals=[],
        operations=operations
        if operations is not None
        else [
            OperationSpecification(
                name="open_valve",
                description="",
                parameters={"properties": {"force": {}}, "required": ["force"]},
            )
        ],
        raw_text=None,
    )


def _authored_manual(
    *,
    manual_id: str = "pump",
    description: str = "",
    raw_text: str | None = "authored raw",
    metadata: dict[str, object] | None = None,
    operations: list[OperationSpecification] | None = None,
    observable_properties: list[ObservablePropertySpecification] | None = None,
) -> Manual:
    return Manual(
        id=manual_id,
        metadata=metadata or {},
        description=description,
        observable_properties=observable_properties or [],
        signals=[],
        operations=operations or [],
        raw_text=raw_text,
    )


def test_merge_adapter_owns_structured_specs_authored_owns_raw_text() -> None:
    adapter = _adapter_manual()
    authored = _authored_manual(description="authored description", metadata={"category": "Fluids"})
    merged = merge_manuals(adapter, authored)
    assert merged.id == "pump"
    assert merged.operations == adapter.operations
    assert merged.raw_text == "authored raw"
    assert merged.description == "authored description"
    assert merged.metadata == {
        "source": "mcp",
        "category": "Fluids",
    }  # union, authored wins conflict


def test_merge_falls_back_to_adapter_description_when_authored_is_empty() -> None:
    merged = merge_manuals(_adapter_manual(), _authored_manual(description=""))
    assert merged.description == "adapter description"


def test_merge_falls_back_to_adapter_raw_text_when_authored_has_none() -> None:
    adapter = replace(_adapter_manual(), raw_text="adapter raw")
    merged = merge_manuals(adapter, _authored_manual(raw_text=None))
    assert merged.raw_text == "adapter raw"


def test_merge_rejects_mismatched_ids() -> None:
    with pytest.raises(ManualMergeError, match="different ids"):
        merge_manuals(_adapter_manual(manual_id="pump"), _authored_manual(manual_id="valve"))


def test_merge_passes_through_kind_left_undeclared() -> None:
    adapter = _adapter_manual(
        observable_properties=[
            ObservablePropertySpecification(name="pressure", description="", schema={})
        ]
    )
    merged = merge_manuals(
        adapter, _authored_manual()
    )  # authored declares no properties -> opt-in skip
    assert merged.observable_properties == adapter.observable_properties


def test_merge_validates_declared_operation_names_match() -> None:
    adapter = _adapter_manual()  # only "open_valve"
    authored = _authored_manual(
        operations=[OperationSpecification(name="close_valve", description="", parameters={})]
    )
    with pytest.raises(ManualMergeError, match="operation names diverge"):
        merge_manuals(adapter, authored)


def test_merge_validates_required_keys_present_in_adapter_schema() -> None:
    adapter = _adapter_manual(
        operations=[
            OperationSpecification(
                name="open_valve",
                description="",
                parameters={"properties": {"force": {}}, "required": []},
            )
        ]
    )
    authored = _authored_manual(
        operations=[
            OperationSpecification(
                name="open_valve", description="", parameters={"required": ["torque"]}
            )
        ]
    )
    with pytest.raises(ManualMergeError, match="requires"):
        merge_manuals(adapter, authored)


def test_merge_keeps_authored_completion_signal_over_adapter() -> None:
    # completion_signal is author-owned semantics a native description can't express, so the merged
    # operation carries the authored value even though the adapter otherwise owns operations.
    adapter = _adapter_manual(
        operations=[OperationSpecification(name="open_valve", description="adapter", parameters={})]
    )
    authored = _authored_manual(
        operations=[
            OperationSpecification(
                name="open_valve", description="", parameters={}, completion_signal="valve_settled"
            )
        ]
    )
    merged = merge_manuals(adapter, authored)
    assert [(op.description, op.completion_signal) for op in merged.operations] == [
        ("adapter", "valve_settled")  # adapter owns description/params; authored owns completion
    ]


def test_merge_skips_required_check_when_adapter_schema_has_no_properties_key() -> None:
    # ARE's synthesized observable properties carry an empty {} schema — no "properties" key at
    # all — so required-key checking can't run, but the name still validates.
    adapter = _adapter_manual(
        observable_properties=[
            ObservablePropertySpecification(name="state", description="", schema={})
        ]
    )
    authored = _authored_manual(
        observable_properties=[
            ObservablePropertySpecification(
                name="state", description="", schema={"required": ["unread"]}
            )
        ]
    )
    merged = merge_manuals(adapter, authored)  # does not raise
    assert merged.observable_properties == adapter.observable_properties


# ------------------------------------------------------------------------------------------------
# DirectoryManualSource — the default ManualSource: *.md in a dir, indexed by parsed Manual.id
# ------------------------------------------------------------------------------------------------
async def test_directory_manual_source_resolves_by_parsed_id(tmp_path: Path) -> None:
    (tmp_path / "pump.md").write_text(load("water-pump.md"), encoding="utf-8")
    source = DirectoryManualSource(tmp_path)
    manual = await source.get("hydraulic_control")  # the fixture's parsed id, not the filename
    assert manual is not None
    assert manual.id == "hydraulic_control"


async def test_directory_manual_source_missing_id_returns_none(tmp_path: Path) -> None:
    (tmp_path / "pump.md").write_text(load("water-pump.md"), encoding="utf-8")
    assert await DirectoryManualSource(tmp_path).get("no-such-id") is None


async def test_directory_manual_source_missing_directory_returns_none(tmp_path: Path) -> None:
    source = DirectoryManualSource(tmp_path / "does-not-exist")
    assert await source.get("anything") is None


async def test_directory_manual_source_builds_index_once(tmp_path: Path) -> None:
    (tmp_path / "pump.md").write_text(load("water-pump.md"), encoding="utf-8")
    source = DirectoryManualSource(tmp_path)
    await source.get("hydraulic_control")
    (tmp_path / "clock.md").write_text(
        load("clock.md"), encoding="utf-8"
    )  # added after first lookup
    assert await source.get("clock") is None  # index already built — lazily, only once
