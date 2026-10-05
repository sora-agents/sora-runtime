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

import pytest

from sora.memory import render_changes
from sora.perception import Percept
from sora.types import (
    Change,
    ObservableProperty,
    Signal,
    changes_of,
    diff_values,
    identities,
    path_matches,
    watch_matches,
)

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


# --------------------------------------------------------------------------------------------------
# watch_matches — the direction scope on top of the path scope
# --------------------------------------------------------------------------------------------------


def test_direction_separates_the_agents_own_write_from_the_worlds() -> None:
    """The whole reason `kind` exists. An agent watching a collection for additions and deleting
    from that same collection lands its own write on the exact path it watches, so `path` alone
    cannot tell the two apart — and the gate it opens costs a model call to conclude nothing."""
    own_delete = [Change(path="events", removed=("e1",))]
    world_add = [Change(path="events", added=("e2",))]
    assert path_matches("events", own_delete)  # indistinguishable on path alone
    assert not watch_matches("events", "added", own_delete)
    assert watch_matches("events", "added", world_add)


def test_direction_is_conjunctive_with_the_path_on_one_change() -> None:
    """Checking the two scopes independently would let an addition somewhere else pair up with a
    deletion on the watched path — exactly the self-write the direction scope exists to exclude."""
    changes = [Change(path="events", removed=("e1",)), Change(path="contacts", added=("c1",))]
    assert not watch_matches("events", "added", changes)


def test_direction_degrades_open_on_a_coarse_change() -> None:
    """`Change`'s contract is that adapters degrade rather than fail — a WoT observation or an MCP
    resources/updated reports "something under here moved" and no more. Narrowing that away would
    make a kind-scoped watch permanently deaf on those adapters; a redundant evaluation costs one
    call, a missed wake is the failure the whole mechanism exists to prevent."""
    assert watch_matches("events", "added", [Change(path="events")])
    assert watch_matches("events", "added", [Change()])
    assert watch_matches("events", "added", [])


def test_an_unscoped_watch_is_unchanged_by_the_new_field() -> None:
    # Every completion-signal wait leaves kind unset, so this is the path they all take.
    changes = [Change(path="events", removed=("e1",))]
    assert watch_matches("events", None, changes)
    assert watch_matches(None, None, changes)


# --------------------------------------------------------------------------------------------------
# identities — which field identifies a record, and what a wrong answer costs
# --------------------------------------------------------------------------------------------------


def test_a_record_keyed_on_message_id_is_identified() -> None:
    """Message records need their own recognized identity field. Records keyed on `message_id` fell
    through `id`/`uid`/`event_id`/`email_id`, so a conversation that gained a reply reported the
    coarse form, the dereference had no id to look up, and a judgement asking whether anyone
    declined an invitation was handed a path with no values at all."""
    before = [{"message_id": "m1", "sender_id": "+1", "content": "are you coming?"}]
    after = before + [{"message_id": "m2", "sender_id": "+2", "content": "Sorry, I can't join."}]
    assert identities(after) is not None
    assert diff_values(before, after) == [Change(path="", added=("m2",), updated=())]


def test_a_foreign_key_does_not_identify_a_record() -> None:
    """`sender_id` sits next to `message_id` and sorts before it, so name order alone picks the
    wrong field. Keying on it collapses two messages from one sender onto a single entry — turning
    an appended reply into an `updated` of the earlier one, which reads as "he changed his mind"
    rather than "a second person answered"."""
    items = [
        {"message_id": "m1", "sender_id": "+1", "content": "first"},
        {"message_id": "m2", "sender_id": "+1", "content": "second"},
    ]
    assert identities(items) == {"m1": items[0], "m2": items[1]}


def test_a_non_unique_identity_degrades_to_coarse_rather_than_collapsing() -> None:
    # Keying on a duplicated field silently drops a record. Reporting "something under here moved"
    # is the honest answer, and `Change`'s contract already requires consumers to accept it.
    assert identities([{"id": "dup", "v": 1}, {"id": "dup", "v": 2}]) is None


@pytest.mark.parametrize("calendar_id", ["calendar-one", "calendar-two"])
def test_event_identity_is_stable_when_appending_across_or_within_calendars(
    calendar_id: str,
) -> None:
    before = [{"calendar_id": "calendar-one", "event_id": "event-one"}]
    after = before + [{"calendar_id": calendar_id, "event_id": "event-two"}]
    assert identities(before) == {"event-one": before[0]}
    assert identities(after) == {"event-one": before[0], "event-two": after[1]}
    assert diff_values(before, after) == [Change(added=("event-two",))]


def test_a_unique_foreign_key_without_a_record_id_degrades_to_coarse() -> None:
    before = [{"sender_id": "sender-one", "content": "first"}]
    after = before + [{"sender_id": "sender-two", "content": "second"}]
    assert identities(after) is None
    assert diff_values(before, after) == [Change()]


def test_a_duplicate_record_id_does_not_fall_back_to_a_unique_foreign_key() -> None:
    items = [
        {"event_id": "event-one", "calendar_id": "calendar-one"},
        {"event_id": "event-one", "calendar_id": "calendar-two"},
    ]
    assert identities(items) is None


def test_a_list_of_scalars_still_has_no_identity() -> None:
    # Scalars have no stable record key: invented positional ids change meaning on every insert.
    assert identities(["a", "b"]) is None


# --------------------------------------------------------------------------------------------------
# render_changes — the dereference, including where the change could not name its items
# --------------------------------------------------------------------------------------------------


def _property(source: str, name: str, value: object) -> Percept:
    return Percept(source=source, payload=ObservableProperty(name=name, value=value), observed_at=0)


def test_a_coarse_change_on_a_list_shows_its_tail() -> None:
    """The structural backstop. An adapter that cannot identify its items (an MCP
    `resources/updated` carries only a URI) must still leave the judgement something to read, and
    for a list an append is what a coarse change overwhelmingly is."""
    value = {
        "conversations": {
            "c1": {
                "messages": [{"content": "are you coming?"}, {"content": "Sorry, I can't join."}]
            }
        }
    }
    rendered = render_changes(
        [("insim:are/Messages", Change(path="conversations.c1.messages"))],
        [_property("insim:are/Messages", "state", value)],
    )
    assert "Sorry, I can't join." in rendered
    assert "could not identify which items moved" in rendered
    assert "(most recent)" in rendered


def test_a_coarse_change_on_a_dict_renders_no_records() -> None:
    """Deliberately the one shape left unanswered: with no key added or removed there is no tail to
    point at, and dumping the map is the shape sketch this dereference exists to replace."""
    value = {"conversations": {f"c{i}": {"topic": f"t{i}"} for i in range(50)}}
    rendered = render_changes(
        [("insim:are/Messages", Change(path="conversations"))],
        [_property("insim:are/Messages", "state", value)],
    )
    assert "read from the current snapshot" not in rendered
    assert "could not identify which items moved" in rendered


def test_a_coarse_change_on_a_leaf_shows_its_value() -> None:
    # A leaf that moved has no sub-structure to name, and its value is both the whole answer and
    # cheap to render.
    rendered = render_changes(
        [("insim:are/City", Change(path="crime_rate"))],
        [_property("insim:are/City", "state", {"crime_rate": 0.42})],
    )
    assert "0.42" in rendered


@pytest.mark.parametrize(
    "survivors", [[{"message_id": "kept", "content": "unchanged"}], "unchanged"]
)
def test_precise_removal_does_not_render_surviving_records_as_changed(survivors: object) -> None:
    rendered = render_changes(
        [("tool", Change(path="messages", removed=("gone",)))],
        [_property("tool", "state", {"messages": survivors})],
    )
    assert "removed=['gone']" in rendered
    assert "unchanged" not in rendered
    assert "read from the current snapshot" not in rendered
    assert "most recent" not in rendered


def test_an_identified_change_is_unaffected_by_the_coarse_fallback() -> None:
    # The precise path stays precise: named ids are looked up, and the tail is not appended to them.
    value = {
        "messages": [{"message_id": "m1", "content": "old"}, {"message_id": "m2", "content": "new"}]
    }
    rendered = render_changes(
        [("tool", Change(path="messages", added=("m2",)))],
        [_property("tool", "state", value)],
    )
    assert "new" in rendered and "old" not in rendered
    assert "most recent" not in rendered
