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

from sora.manual import ManualParseError, ManualSection, MarkdownManualParser

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
