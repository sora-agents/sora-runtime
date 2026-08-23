"""Located change summaries and path-scoped waits (ADR-0019).

A signal says *that* a tool changed; ``properties`` is a replace-by-key snapshot and by construction
holds no delta, so a contentless signal forces every waiter to re-derive a change against a store
that keeps no previous value to diff against. ``Change`` carries *where* it moved and the identities
involved — never the values, which stay in the property — and ``SignalWait.path`` scopes a wait to
part of it.

The two properties worth pinning hardest are the ones a future change is most likely to "simplify"
into a bug: the prefix match is **bidirectional** (a coarse change reported above the watched path
must still wake waiters beneath it, or a degrading adapter silently starves them), and each waiter
carries its **own** high-water mark over a monotonic counter that the retention cap never rewinds.
"""

from __future__ import annotations

from sora.types import Change, Signal, changes_of, path_matches

# --------------------------------------------------------------------------------------------------
# path_matches — the bidirectional prefix
# --------------------------------------------------------------------------------------------------


def test_unscoped_wait_matches_any_change() -> None:
    # path=None is today's behavior and stays the default, so existing completion-signal waits are
    # unaffected by this feature existing.
    assert path_matches(None, [Change(path="folders.INBOX.emails")])


def test_change_inside_the_watched_subtree_matches() -> None:
    assert path_matches("folders.INBOX", [Change(path="folders.INBOX.emails", added=("e1",))])


def test_coarser_change_above_the_watched_path_also_matches() -> None:
    # The direction that keeps a DEGRADING adapter correct. An adapter that can only report "the
    # Emails app changed" must still wake a waiter watching folders.INBOX.emails beneath it —
    # otherwise the waiter starves precisely when the adapter is least capable. A redundant
    # evaluation is recoverable; a missed wake is the failure this mechanism exists to prevent.
    assert path_matches("folders.INBOX.emails", [Change(path="folders")])


def test_sibling_path_does_not_match() -> None:
    # The discrimination that replaces an ARE-specific efference filter: the agent's own send_email
    # lands in SENT, an inbound reply in INBOX. Same signal name, same source, told apart by where
    # they landed — with no reasoning about which changes the agent caused itself.
    assert not path_matches("folders.INBOX.emails", [Change(path="folders.SENT.emails")])


def test_a_sibling_sharing_a_leading_substring_does_not_match() -> None:
    # Prefix on SEGMENTS, not on characters. Siblings are not ancestors of one another, however much
    # of a leading substring they share — and the ids these paths are built from routinely do share
    # one: an ARE contacts app runs contact_1..contact_125, so `contact_1` is a character-prefix of
    # eleven other records. A raw startswith wakes a wait on every one of them.
    assert not path_matches("folders.INBOX", [Change(path="folders.INBOX_ARCHIVE")])
    assert not path_matches("contacts.contact_1", [Change(path="contacts.contact_10.emails")])
    assert not path_matches("contacts.contact_10.emails", [Change(path="contacts.contact_1")])


def test_a_segment_boundary_is_what_makes_a_prefix_a_parent() -> None:
    # The pair the test above is the negative of: the same characters, cut at a segment boundary.
    assert path_matches("folders.INBOX", [Change(path="folders.INBOX.emails")])
    assert path_matches("folders.INBOX", [Change(path="folders.INBOX")])


def test_coarse_change_with_empty_path_matches_everything() -> None:
    # The coarsest degradation: "something moved, I can't say where." Must not be read as "nothing
    # moved" — that would turn an uninformative adapter into a silently broken one.
    assert path_matches("folders.INBOX.emails", [Change(path="")])


def test_signal_carrying_no_changes_matches_a_scoped_wait() -> None:
    # An adapter that reports no changes at all is indistinguishable from one reporting a change
    # everywhere, so the safe reading is the wide one. This is what keeps a path-scoped wait working
    # against an adapter that has not been taught to emit Change at all.
    assert path_matches("folders.INBOX.emails", [])


def test_any_matching_change_in_a_batch_is_enough() -> None:
    changes = [Change(path="folders.SENT.emails"), Change(path="folders.INBOX.emails")]
    assert path_matches("folders.INBOX.emails", changes)


# --------------------------------------------------------------------------------------------------
# changes_of — tolerating what a serialization boundary does to a payload
# --------------------------------------------------------------------------------------------------


def test_changes_of_reads_change_objects() -> None:
    signal = Signal("state_changed", {"changes": [Change(path="a", added=("x",))]})
    assert changes_of(signal) == [Change(path="a", added=("x",))]


def test_changes_of_rebuilds_dicts_from_a_json_round_trip() -> None:
    # A payload can arrive as plain JSON (a persisted percept, a JSON-shaped adapter), so a Change
    # may show up as a dict with its tuples flattened to lists. Normalizing here is what lets
    # path_matches stay a simple comparison instead of every caller re-deriving the shape.
    signal = Signal("state_changed", {"changes": [{"path": "a", "added": ["x", "y"]}]})
    assert changes_of(signal) == [Change(path="a", added=("x", "y"))]


def test_changes_of_degrades_a_malformed_entry_rather_than_raising() -> None:
    # A malformed delta must not be able to break a wait that would otherwise have matched: the
    # entry degrades to the coarse "something moved" form, which errs toward waking the waiter.
    signal = Signal("state_changed", {"changes": ["not-a-change"]})
    assert changes_of(signal) == [Change()]
    assert path_matches("anything", changes_of(signal))


def test_changes_of_tolerates_a_missing_or_wrong_shaped_key() -> None:
    assert changes_of(Signal("state_changed", {})) == []
    assert changes_of(Signal("state_changed", {"changes": "nope"})) == []


def test_change_carries_identities_not_values() -> None:
    # The thin-signal rule, pinned structurally: a Change has nowhere to put a value even if an
    # adapter wanted to. The snapshot stays in wm.properties and this says where to look inside it.
    assert {f for f in Change.__dataclass_fields__} == {"path", "added", "removed", "updated"}
