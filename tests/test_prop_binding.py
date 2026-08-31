"""``$prop`` — binding a param from the observed property snapshot (ADR-0022).

The third read token, and the only one whose source is re-observed every cycle: ``$from`` reads
``Activity.history``, ``$bind`` reads the named-binding namespace, ``$prop`` reads
``WorkingMemory.properties``. These tests pin the two things that make it safe rather than merely
convenient — a *bare* property name resolves only when exactly one focused tool exposes it, and
every failure comes back as a defect that names the correction (which tool to focus, which
candidates collide) instead of silently resolving to nothing.
"""

from __future__ import annotations

from sora._strategies.subgoals import _expand_mechanical
from sora.data_ops import _resolve_collection
from sora.memory import render_properties
from sora.perception import Percept
from sora.strategies import resolve_references
from sora.types import ObservableProperty, Step

# A contacts-app state property of the shape ARE actually publishes: an {id -> record} map under a
# key, alongside a scalar. The motivating run failed with exactly this in working memory.
_CONTACTS = {
    "contacts": {
        "c1": {"first_name": "Åke", "last_name": "Lindström", "job": "Film Producer"},
        "c2": {"first_name": "Mia", "last_name": "Rossi", "job": "Architect"},
    },
    "view_limit": 10,
}


def _props(*items: tuple[str, str, object]) -> dict[tuple[str, str], Percept]:
    """The store's own shape: replace-by-(source, name), exactly as WorkingMemory holds it."""
    return {
        (source, name): Percept(source, ObservableProperty(name, value), 0.0)
        for source, name, value in items
    }


# --------------------------------------------------------------------------------------------------
# resolving a $prop param
# --------------------------------------------------------------------------------------------------


def test_qualified_prop_resolves_and_walks_path() -> None:
    props = _props(("Contacts", "state", _CONTACTS))
    params = {"n": {"$prop": "Contacts.state", "path": "view_limit"}}
    resolved, unresolved = resolve_references(params, [], None, props)
    assert resolved == {"n": 10}
    assert unresolved == []


def test_qualified_prop_splits_on_the_last_dot() -> None:
    """A tool id may itself contain dots; a property name never does."""
    props = _props(("wot:lamp.local/Lamp", "state", {"on": True}))
    params = {"x": {"$prop": "wot:lamp.local/Lamp.state", "path": "on"}}
    resolved, unresolved = resolve_references(params, [], None, props)
    assert resolved == {"x": True}
    assert unresolved == []


def test_bare_prop_resolves_when_exactly_one_tool_exposes_it() -> None:
    props = _props(("Contacts", "view_limit", 10), ("Calendar", "state", {"events": []}))
    resolved, unresolved = resolve_references({"n": {"$prop": "view_limit"}}, [], None, props)
    assert resolved == {"n": 10}
    assert unresolved == []


def test_bare_prop_is_unresolved_when_several_tools_expose_it() -> None:
    """ARE gives thirteen tools a `state` property — picking one silently is the failure mode."""
    props = _props(("Contacts", "state", _CONTACTS), ("Calendar", "state", {"events": []}))
    params = {"x": {"$prop": "state"}}
    resolved, unresolved = resolve_references(params, [], None, props)
    assert unresolved == ["x"]
    assert resolved["x"] == params["x"]  # left in place for the escalation


def test_prop_naming_an_unobserved_tool_is_unresolved() -> None:
    props = _props(("Calendar", "state", {"events": []}))
    resolved, unresolved = resolve_references({"x": {"$prop": "Contacts.state"}}, [], None, props)
    assert unresolved == ["x"]


def test_prop_with_a_bad_path_is_unresolved() -> None:
    props = _props(("Contacts", "state", _CONTACTS))
    params = {"x": {"$prop": "Contacts.state", "path": "no_such_key"}}
    _, unresolved = resolve_references(params, [], None, props)
    assert unresolved == ["x"]


def test_prop_resolves_nested_inside_a_list_param() -> None:
    props = _props(("Contacts", "state", _CONTACTS))
    params = {"attendees": [{"$prop": "Contacts.state", "path": "contacts.c1.first_name"}]}
    resolved, unresolved = resolve_references(params, [], None, props)
    assert resolved == {"attendees": ["Åke"]}
    assert unresolved == []


def test_no_properties_passed_leaves_prop_unresolved_rather_than_raising() -> None:
    """The param defaults to None so every existing caller keeps working."""
    resolved, unresolved = resolve_references({"x": {"$prop": "Contacts.state"}}, [])
    assert unresolved == ["x"]
    assert resolved["x"] == {"$prop": "Contacts.state"}


# --------------------------------------------------------------------------------------------------
# $prop as a collection — a data-op's `in`, or a mechanical sub-goal's iteration source
# --------------------------------------------------------------------------------------------------


def test_prop_map_resolves_to_a_collection_of_records() -> None:
    """The {id -> record} tier: a data-op filters the values, not the ids."""
    props = _props(("Contacts", "state", _CONTACTS))
    ref = {"$prop": "Contacts.state", "path": "contacts"}
    collection, defect = _resolve_collection(ref, [], None, props)
    assert defect is None
    assert collection is not None
    assert sorted(c["job"] for c in collection) == ["Architect", "Film Producer"]


def test_prop_collection_without_a_path_reports_the_envelope_defect() -> None:
    """Empty is an answer; unreadable is a question — and the question names the fix."""
    props = _props(("Contacts", "state", _CONTACTS))
    collection, defect = _resolve_collection({"$prop": "Contacts.state"}, [], None, props)
    assert collection is None
    assert defect is not None
    assert "add a 'path'" in defect


def test_unobserved_prop_collection_defect_names_a_correction_that_can_work() -> None:
    """Deliberately NOT "add a focus step": the runtime already attends to every tool a live plan
    names, so focusing cannot make an unlisted property appear. Prescribing it buys a replan that
    repeats the same reference — the one outcome a defect message exists to prevent."""
    props = _props(("Calendar", "state", {"events": []}))
    collection, defect = _resolve_collection({"$prop": "Contacts.state"}, [], None, props)
    assert collection is None
    assert defect is not None
    assert "focus" not in defect
    assert "join" in defect  # the reachable correction when the tool is absent entirely
    assert "Calendar.state" in defect  # the planner is told what *is* observed


def test_ambiguous_prop_collection_defect_names_the_candidates() -> None:
    props = _props(("Contacts", "state", _CONTACTS), ("Calendar", "state", {"events": []}))
    collection, defect = _resolve_collection({"$prop": "state"}, [], None, props)
    assert collection is None
    assert defect is not None
    assert "Calendar" in defect and "Contacts" in defect
    assert "qualify" in defect


def test_a_bad_path_into_an_observed_prop_is_not_reported_as_ambiguity() -> None:
    """A present property with a wrong path took the candidates branch and announced that 'state'
    "is exposed by several focused tools (Contacts)" — "several", listing one, and telling the
    planner to qualify a name it had already qualified."""
    props = _props(("Contacts", "state", _CONTACTS))

    collection, defect = _resolve_collection(
        {"$prop": "Contacts.state", "path": "nope"}, [], None, props
    )

    assert collection is None
    assert defect is not None
    assert "several focused tools" not in defect
    assert "'nope'" in defect and "'contacts'" in defect


def test_a_folded_bad_tail_reports_a_defect_rather_than_raising() -> None:
    """The split form recovers by stripping `path` and re-resolving the source — a no-op for the
    folded spelling, where the tail lives inside the token. So the retry re-raised the same
    KeyError, and with nothing above `_resolve_collection` catching it the exception left Reason,
    left tick(), and aborted the whole run instead of writing a replan brief."""
    props = _props(("Contacts", "state", _CONTACTS))

    collection, defect = _resolve_collection({"$prop": "Contacts.state.nope"}, [], None, props)

    assert collection is None
    assert defect is not None
    assert "'nope'" in defect and "'contacts'" in defect  # the segment, and what is there instead


def test_a_folded_bad_tail_deep_in_the_route_names_the_segment_that_failed() -> None:
    props = _props(("Contacts", "state", _CONTACTS))

    collection, defect = _resolve_collection(
        {"$prop": "Contacts.state.contacts.c9"}, [], None, props
    )

    assert collection is None
    assert defect is not None
    assert "'c9'" in defect
    assert "contacts is a mapping" in defect  # walked as far as it could, then said what it found


def test_a_half_folded_bad_tail_is_reported_against_the_composed_route() -> None:
    """The two spellings compose, so the defect has to be read off the composed route rather than
    off whichever half happened to carry the failing segment."""
    props = _props(("Contacts", "state", _CONTACTS))

    collection, defect = _resolve_collection(
        {"$prop": "Contacts.state.contacts", "path": "c9"}, [], None, props
    )

    assert collection is None
    assert defect is not None
    assert "'c9'" in defect


def test_a_folded_reference_to_an_unobserved_tool_is_still_the_missing_source_defect() -> None:
    """A folded token whose HEAD does not resolve is a different question from a bad tail: the tool
    is not being observed at all, so the answer is to reference one that is — not to correct the
    route into a property that was never there."""
    props = _props(("Calendar", "state", {"events": []}))

    collection, defect = _resolve_collection({"$prop": "Contacts.state.nope"}, [], None, props)

    assert collection is None
    assert defect is not None
    assert "names no observed property" in defect
    assert "Calendar.state" in defect


def test_mechanical_subgoal_fans_out_over_a_prop_collection() -> None:
    """The point of it: bulk observed state becomes len(collection) steps, with no model call."""
    props = _props(("Contacts", "state", _CONTACTS))
    step = Step(
        next_action="subgoal",
        params={
            "in": {"$prop": "Contacts.state", "path": "contacts"},
            "as": "contact",
            "template": {
                "action": "invoke",
                "tool_id": "Email",
                "operation_name": "send_email",
                "params": {"to": {"$bind": "contact", "path": "first_name"}},
            },
        },
    )
    expanded, defect = _expand_mechanical(step, [], None, props)
    assert defect is None
    assert [s.params["to"] for s in expanded] == ["Åke", "Mia"]


# --------------------------------------------------------------------------------------------------
# rendering: a fat property must show its SHAPE, or a planner cannot author a path into it
# --------------------------------------------------------------------------------------------------


def test_small_property_renders_verbatim() -> None:
    props = [Percept("Calendar", ObservableProperty("state", {"events": [], "tz": "UTC"}), 0.0)]
    assert render_properties(props) == '- Calendar.state = {"events": [], "tz": "UTC"}'


def _contact(i: int) -> dict[str, object]:
    """An ARE contact at its real width. The count matters: a fifteen-field record is wider than
    any plausible "enumerate the keys" cap, so a renderer that decides record-vs-map on key count
    alone reports it as ``{<key>: "P0"} x 15`` and drops every field name."""
    return {
        "first_name": f"P{i}",
        "last_name": f"L{i}",
        "contact_id": f"id{i}",
        "is_user": False,
        "gender": "Male",
        "age": 45,
        "nationality": "Swedish",
        "city_living": "Stockholm",
        "country": "Sweden",
        "status": "Employed",
        "job": "Film Producer",
        "description": "d" * 500,
        "phone": "+46701234567",
        "email": f"p{i}@x.com",
        "address": "Stockholm, Sweden",
    }


def test_bulk_property_renders_as_shape_with_cardinality() -> None:
    """The motivating case: 125 contacts under one property. A truncated 400 chars of the first
    record tells the planner nothing about the path it has to write."""
    contacts = {f"id{i}": _contact(i) for i in range(125)}
    props = [Percept("Contacts", ObservableProperty("state", {"contacts": contacts}), 0.0)]
    rendered = render_properties(props)
    assert "x 125" in rendered  # the count survives
    assert "first_name" in rendered and "job" in rendered  # the record shape survives
    assert "\u2026" not in rendered  # and it is a sketch, not a truncation


def test_wide_record_keeps_its_field_names() -> None:
    """A record is not a keyed collection just because it is wide. Its field names are the whole
    reason this line exists \u2014 they are what a planner writes into a $prop path or a filter."""
    rendered = render_properties([Percept("C", ObservableProperty("state", _contact(0)), 0.0)])
    assert "<key>" not in rendered
    for field in ("first_name", "city_living", "job", "email"):
        assert field in rendered


def test_small_homogeneous_dict_keeps_its_keys() -> None:
    """Homogeneous values alone must not collapse a dict: these keys are path segments a planner
    has to write verbatim, so ``{<key>: ...} x 4`` would make the property unnavigable."""
    folders = {name: {"folder_name": name, "emails": []} for name in ("INBOX", "SENT", "TRASH")}
    state = {"folders": folders, "n": "x" * 500}  # long enough to force a sketch
    props = [Percept("Emails", ObservableProperty("state", state), 0.0)]
    rendered = render_properties(props)
    assert "INBOX" in rendered and "SENT" in rendered


def test_large_homogeneous_map_still_collapses() -> None:
    """The case the collapse is for: keys that are ids carry nothing a planner can use."""
    name_to_id = {f"Person {i}": f"+4670{i:07d}" for i in range(62)}
    props = [Percept("Messages", ObservableProperty("state", {"name_to_id": name_to_id}), 0.0)]
    rendered = render_properties(props)
    assert "<key>" in rendered and "x 62" in rendered


def test_long_list_property_renders_element_shape() -> None:
    events = [{"id": f"e{i}", "title": "x" * 40} for i in range(50)]
    props = [Percept("Calendar", ObservableProperty("state", {"events": events}), 0.0)]
    rendered = render_properties(props)
    assert "x 50" in rendered
    assert "title" in rendered


def test_long_scalar_property_still_truncates() -> None:
    """No shape to describe — a long string falls back to the old behaviour, not to nothing."""
    props = [Percept("Notes", ObservableProperty("body", "z" * 900), 0.0)]
    assert render_properties(props).endswith("\u2026")


# --------------------------------------------------------------------------------------------------
# a sub-path folded into the token instead of into `path`
# --------------------------------------------------------------------------------------------------
# The canonical spelling splits the route in two, but a planner reading a catalog that addresses
# everything by dotted name writes it as one string. Both name one value; refusing the second cost a
# whole plan on the 2026-08-21 adaptability run, where the *same* model wrote the split form on one
# attempt and the folded form on the next. Resolution is longest-prefix, so an exact key still wins.


def test_folded_subpath_resolves_as_a_collection() -> None:
    """The exact reference from the failing run: the whole route inside the $prop token."""
    props = _props(("insim:are/Contacts", "state", _CONTACTS))
    ref = {"$prop": "insim:are/Contacts.state.contacts"}
    collection, defect = _resolve_collection(ref, [], None, props)
    assert defect is None
    assert collection is not None
    assert sorted(c["job"] for c in collection) == ["Architect", "Film Producer"]


def test_folded_and_split_spellings_agree() -> None:
    props = _props(("Contacts", "state", _CONTACTS))
    folded, _ = resolve_references(
        {"x": {"$prop": "Contacts.state.contacts.c1.job"}}, [], None, props
    )
    split, _ = resolve_references(
        {"x": {"$prop": "Contacts.state", "path": "contacts.c1.job"}}, [], None, props
    )
    assert folded == split == {"x": "Film Producer"}


def test_half_folded_reference_composes_token_route_then_path() -> None:
    """A folded segment is walked before an explicit `path`, not instead of it."""
    props = _props(("Contacts", "state", _CONTACTS))
    params = {"x": {"$prop": "Contacts.state.contacts", "path": "c2.last_name"}}
    resolved, unresolved = resolve_references(params, [], None, props)
    assert resolved == {"x": "Rossi"}
    assert unresolved == []


def test_exact_property_key_wins_over_a_folded_reading() -> None:
    """Longest reference first: where the whole string IS a key, that key is used and nothing is
    treated as a path, so a real property is never shadowed by a folded reading of itself."""
    props = _props(
        ("Tool", "state", {"contacts": "the folded route"}),
        ("Tool.state", "contacts", "the property"),
    )
    resolved, _ = resolve_references({"x": {"$prop": "Tool.state.contacts"}}, [], None, props)
    assert resolved == {"x": "the property"}


def test_a_property_name_containing_a_dot_resolves() -> None:
    """The split is found against the live key set, not by parsing the string, so nothing here
    depends on a property name being dot-free. The runtime does not author names (ADR-0003): a
    WoT or MCP adapter may well import a property called `sensor.temp`, and a last-dot split would
    look for a tool `wot:x/Y.sensor` and silently find nothing."""
    props = _props(("wot:lamp.local/Lamp", "sensor.temp", 21.5))
    resolved, unresolved = resolve_references(
        {"x": {"$prop": "wot:lamp.local/Lamp.sensor.temp"}}, [], None, props
    )
    assert resolved == {"x": 21.5}
    assert unresolved == []


def test_a_dotted_property_name_also_takes_a_folded_subpath() -> None:
    """Both tolerances at once: a dotted name AND a route folded in after it."""
    props = _props(("wot:lamp.local/Lamp", "sensor.temp", {"celsius": 21.5}))
    params = {"x": {"$prop": "wot:lamp.local/Lamp.sensor.temp.celsius"}}
    resolved, unresolved = resolve_references(params, [], None, props)
    assert resolved == {"x": 21.5}
    assert unresolved == []


def test_two_keys_joining_to_the_same_string_are_ambiguous_not_guessed() -> None:
    """`("A", "b.c")` and `("A.b", "c")` both spell `A.b.c`. Neither is more specific, so the
    reference resolves to nothing rather than to whichever the dict happens to yield first."""
    props = _props(("A", "b.c", "one"), ("A.b", "c", "the other"))
    _, unresolved = resolve_references({"x": {"$prop": "A.b.c"}}, [], None, props)
    assert unresolved == ["x"]


def test_a_qualified_match_beats_a_bare_one_of_the_same_length() -> None:
    """Naming the tool is the more specific claim, so it wins where both could apply."""
    props = _props(("A", "b", "qualified"), ("Other", "A.b", "bare"))
    resolved, _ = resolve_references({"x": {"$prop": "A.b"}}, [], None, props)
    assert resolved == {"x": "qualified"}


def test_folded_reference_to_an_unobserved_tool_is_still_unresolved() -> None:
    """Tolerance is for spelling, not for absence: nothing matches, so nothing resolves."""
    props = _props(("Calendar", "state", {"events": []}))
    resolved, unresolved = resolve_references(
        {"x": {"$prop": "Contacts.state.contacts"}}, [], None, props
    )
    assert unresolved == ["x"]


def test_folded_reference_with_a_bad_tail_is_unresolved() -> None:
    props = _props(("Contacts", "state", _CONTACTS))
    _, unresolved = resolve_references(
        {"x": {"$prop": "Contacts.state.no_such_key"}}, [], None, props
    )
    assert unresolved == ["x"]
