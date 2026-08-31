"""Structured collection and predicate resolution shared by planning and actions."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from sora.perception import Percept
from sora.references import (
    _AMBIGUOUS,
    _MISSING,
    _REF_DECIDE,
    _REF_PATH,
    _REF_PROP,
    _is_reference,
    _property_ref,
    _resolve_ref,
)
from sora.types import (
    CompletedOperation,
    walk_path,
)

if TYPE_CHECKING:
    pass

log = logging.getLogger("sora.strategies")


def pluck(element: Any, path: str | None) -> Any:
    """The value at ``path`` of ``element`` (the shared dotted-path grammar), or ``None`` on a bad
    path — a missing key is a non-match/absent value for a data-op, never a crash. Public because
    it is the one path-projection helper shared across the data-op layer: mechanical ``filter``
    evaluation (``_matches`` here) and cross-collection membership projection (Reason, in
    ``strategies``) must read a field the same way."""
    if not path:
        return element
    try:
        return walk_path(element, path)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _dedup_key(value: Any) -> str:
    """A stable, hashable signature for a (possibly unhashable, e.g. dict) value, for _distinct_."""
    return json.dumps(value, sort_keys=True, default=str)


def _overlaps(element: Any, where: dict[str, Any]) -> bool:
    """Does the element's own ``[start_path, end_path]`` interval meet any interval in ``against``?

    The two-sided sibling of ``between``: ``between`` compares one value against one fixed pair,
    this compares one *pair* against a whole collection of them. ``against`` arrives already
    resolved and projected by Reason into ``[start, end]`` pairs, exactly as an ``in`` membership
    set arrives projected to keys — so this stays a literal comparison with no reference
    resolution, and no per-member alias to scope.

    Half-open by default (``boundaries: "exclusive"``): two intervals that merely touch at a
    boundary do NOT overlap, which is what a calendar means by a conflict. ``"inclusive"`` makes a
    shared endpoint count. Nothing here is calendar-specific — intervals are ordered values of any
    comparable type, and ISO-8601 timestamps happen to compare correctly as strings.

    Every unusable input is a non-match rather than a crash or a blanket keep, matching the ordered
    ops: an element missing either end, a malformed pair, a null bound, an incomparable type. That
    direction is deliberate — this predicate's output feeds delete fan-outs, where failing *open*
    would act on the whole collection. An ``against`` that could not be read at all is caught in
    Reason as a plan defect (``_operand_defect``), since silently matching nothing is a confident
    wrong answer about the world."""
    start = pluck(element, where.get("start_path", ""))
    end = pluck(element, where.get("end_path", ""))
    intervals = where.get("against")
    if start is None or end is None or not isinstance(intervals, (list, tuple)):
        return False
    inclusive = where.get("boundaries") == "inclusive"
    for interval in intervals:
        if not (isinstance(interval, (list, tuple)) and len(interval) == 2):
            continue
        other_start, other_end = interval
        if other_start is None or other_end is None:
            continue
        try:
            if (
                (start <= other_end and end >= other_start)
                if inclusive
                else (start < other_end and end > other_start)
            ):
                return True
        except TypeError:
            continue  # incomparable types -> non-match, never a crash (like lt/le/gt/ge)
    return False


def _matches(element: Any, where: Any) -> bool:
    """Evaluate a mechanical ``filter`` predicate against one element: ``{"path", "op", "value"}``
    with op in eq/ne/lt/le/gt/ge/between/in/not_in/overlaps. ``in``/``not_in`` test membership of
    the element's ``path`` value in ``value`` (a literal list, or — resolved upstream in Reason —
    the projected keys of another collection named by a reference); ``overlaps`` tests the
    element's own interval against a collection of them (see ``_overlaps``). A ``$decide``
    predicate never gets here (FilterAction escalates it). No predicate keeps everything. A
    membership set that isn't a list is treated as empty: ``in`` matches nothing, ``not_in`` keeps
    everything (fails open, so a malformed exclusion set never silently drops the whole
    collection).

    A predicate may instead COMPOSE others under ``all`` (conjunction) or ``any`` (disjunction),
    recursively. Composition is what makes the mechanical path reach predicates that previously had
    to escalate whole: the real ones are rarely a single clause — "not one of the newly added
    events AND overlapping one of them" is two — and one un-mechanical clause used to drag the
    entire predicate to a model call over the whole collection. A malformed or EMPTY clause list
    matches nothing rather than vacuously everything: ``all([])`` is true in logic, but this
    predicate's consumers fan out over what it keeps, so the failure that acts on the whole
    collection is the one worth refusing. Reason reports either as a plan defect
    (``_composition_defect``) rather than leaving it as a silent empty result."""
    if not isinstance(where, dict):
        return False
    if _COMPOSE_ALL in where:
        clauses = where[_COMPOSE_ALL]
        if not isinstance(clauses, list) or not clauses:
            return False
        return all(_matches(element, clause) for clause in clauses)
    if _COMPOSE_ANY in where:
        clauses = where[_COMPOSE_ANY]
        if not isinstance(clauses, list) or not clauses:
            return False
        return any(_matches(element, clause) for clause in clauses)
    op = where.get("op", "eq")
    if op == "overlaps":
        return _overlaps(element, where)
    actual = pluck(element, where.get("path", ""))
    value = where.get("value")
    if op == "eq":
        return bool(actual == value)
    if op == "ne":
        return bool(actual != value)
    if op == "in":
        return isinstance(value, (list, tuple)) and actual in value
    if op == "not_in":
        return not (isinstance(value, (list, tuple)) and actual in value)
    if op == "between":
        if not (isinstance(value, (list, tuple)) and len(value) == 2):
            return False
        lo, hi = value
        if actual is None:
            return False
        try:
            return bool(lo <= actual <= hi)
        except TypeError:
            return False  # incomparable types -> non-match, never a crash (like lt/le/gt/ge)
    if actual is None:
        return False  # ordered comparisons need a present value
    try:
        if op == "lt":
            return bool(actual < value)
        if op == "le":
            return bool(actual <= value)
        if op == "gt":
            return bool(actual > value)
        if op == "ge":
            return bool(actual >= value)
    except TypeError:
        return False
    log.warning("filter: unknown predicate op %r -> excluding element", op)
    return False


# The names a windowed list operation uses for the metadata it returns *beside* its payload. Closed
# and deliberately short: it is the only thing separating a paginated envelope from a record that
# happens to carry one list field, so every addition widens what gets read as a collection. A name
# belongs here only if no tool would plausibly use it for a record's own field.
_PAGE_META = frozenset(
    {
        "count",
        "cursor",
        "has_more",
        "limit",
        "next_cursor",
        "next_offset",
        "offset",
        "page",
        "page_size",
        "per_page",
        "range",
        "total",
        "total_count",
        "view_limit",
    }
)


def _paginated_payload(value: dict[str, Any]) -> list[Any] | None:
    """The payload list out of ``{"events": [...], "range": ..., "total": ...}``, or ``None`` when
    this is not that shape: exactly one list-valued key, every other key a ``_PAGE_META`` scalar.
    The vocabulary check is the whole load-bearing part — without it a one-list-field record
    qualifies (see ``_as_collection`` tier 3)."""
    payload: list[Any] | None = None
    for key, item in value.items():
        if isinstance(item, list):
            if payload is not None:
                return None  # two candidate payloads: which one was meant is not mechanical
            payload = item
        elif isinstance(item, dict) or key not in _PAGE_META:
            return None
    return payload


def _as_collection(value: Any) -> list[Any] | None:
    """Coerce a resolved value to the list a fan-out/pipeline iterates, in deterministic tiers so a
    plan author never has to hand-shape a tool's return:

    1. **list** -> itself.
    2. **single-key envelope** (a lone key wrapping the payload, e.g. ``{"apartments": {id -> r}}``
       or ``{"results": [...]}``): unwrap and recurse into the one value, *iff* that value is itself
       a collection. A single-element ``{id -> record}`` map whose record has *any* scalar field
       falls through here (the recursion refuses a mixed/scalar record) and tier 3 catches it. The
       residual ambiguity is any single-element map ``{"a1": {record}}`` whose lone record's fields
       are *all* mapping-valued: that record is itself indistinguishable from an ``{id -> record}``
       map (a one-field record ``{"a1": {"photos": [...]}}`` and a many-field one
       ``{"a1": {"loc": {...}, "meta": {...}}}`` both recurse to a collection), so it is unwrapped
       into the record's field-*values* rather than kept as one record — a genuine
       misclassification for such a shape. This is **undecidable** at this layer
       (``{K: {k1: {...}, k2: {...}}}`` is structurally identical whether ``K`` is an id or a
       wrapper name); the unwrap is chosen because ARE's records always carry scalar fields (so they
       never reach it) and its real envelopes are plural ``{id -> record}`` maps that must unwrap.
       The principled fix for a tool that returns all-mapping-field records is the deferred
       model-escalated extraction below, not another shape heuristic — every mechanical tie-break
       here only shifts *which* shape misfires.
    3. **paginated envelope** (a lone list-valued key beside pagination metadata, e.g. ARE's
       ``get_calendar_events_from_to`` -> ``{"events": [...], "range": "(0, 1)", "total": 1}``):
       take the list. Unlike tier 2 this cannot be decided structurally — a record with one list
       field and scalar siblings (``{"event_id": ..., "title": ..., "attendees": [...]}``, an ARE
       calendar event) is *shape-identical* to the envelope, and reading that as a collection of
       attendees would be worse than refusing. So the siblings must additionally all be scalars
       drawn from a closed pagination vocabulary (``_PAGE_META``), which a record's own field names
       are not in. Narrow on purpose: it buys the one shape ARE's windowed list operations actually
       return, and nothing else.
    4. **``{id -> record}`` mapping** (ARE's ``list_all_apartments`` / ``search_apartments`` /
       ``list_saved_apartments``): every value a mapping -> iterate the *values*; the id is carried
       inside each record, so values-iteration is lossless.
    5. anything else -> ``None``: a single record's fields, an ``{id -> scalar}`` map, a scalar. NOT
       mechanically a collection, so the caller logs the shape rather than fanning out over garbage.

    An empty dict is an empty collection (``[]``), not "unresolvable". (Model-escalated extraction
    for shapes these tiers still refuse is deferred.)"""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if not value:
            return []  # an empty {id -> record} map is an empty collection, not "unresolvable"
        if len(value) == 1:
            inner = _as_collection(next(iter(value.values())))
            if inner is not None:
                return inner  # single-key envelope: the lone value is the real collection
        paginated = _paginated_payload(value)
        if paginated is not None:
            return paginated
        if all(isinstance(v, dict) for v in value.values()):
            return list(value.values())
    return None  # single record / envelope-of-scalars / id->scalar / scalar: refuse to guess


def _collection_defect(
    ref: Any,
    value: Any,
    history: list[CompletedOperation],
    properties: dict[tuple[str, str], Percept] | None = None,
) -> str:
    """Why a collection reference could not be read, phrased for the *planner* rather than the log:
    it goes into the replan brief, so it has to say what to write instead. Both cases have a
    concrete correction available, and naming it is the difference between a retry that differs and
    one that repeats — the same reason the undeclared-parameter defect lists the accepted names."""
    if isinstance(ref, dict) and _REF_PROP in ref and value is _MISSING:
        name = str(ref[_REF_PROP])
        props = properties or {}
        candidates = sorted({source for (source, prop) in props if prop == name})
        if candidates:
            return (
                f"{name!r} is exposed by several focused tools ({', '.join(candidates)}) — "
                "qualify it as '<tool_id>.<property_name>' so it names exactly one."
            )
        observed = ", ".join(sorted(f"{s}.{p}" for (s, p) in props))
        # Deliberately does NOT prescribe a 'focus' step. The runtime already attends to every tool
        # a live plan names, so a focus step cannot conjure a property that is not in this list —
        # the name is wrong, or its tool's workspace was never joined. Sending the planner to add a
        # focus step buys a replan that repeats the same reference, which is the one outcome a
        # defect message exists to prevent.
        return (
            f"{name!r} names no observed property; currently observed: "
            f"{observed or 'none — no tool is being observed'}. Reference one of those, qualified "
            "as '<tool_id>.<property_name>'. If the property you want belongs to a tool that is "
            "not listed, its workspace has not been joined — plan a 'join' step first, or reach "
            "the value through an operation instead."
        )
    if value is _MISSING:
        ran = sorted({c.invocation.operation_name for c in history})
        available = ", ".join(ran) if ran else "none yet — no operation has run"
        return (
            f"{ref!r} names no result the plan has produced; operations run so far: {available}. "
            "Reference a collection only after a step has produced it."
        )
    if isinstance(value, dict):
        keys = ", ".join(repr(k) for k in list(value)[:8])
        return (
            f"{ref!r} resolved to a dict with keys {keys}, not a list the runtime can iterate — "
            "add a 'path' naming the field that holds the list."
        )
    return (
        f"{ref!r} resolved to a {type(value).__name__}, not a list the runtime can iterate — "
        "reference something that is a collection, or narrow to one with a data-op first."
    )


def _path_defect(
    ref: dict[str, Any],
    history: list[CompletedOperation],
    bindings: dict[str, Any],
    properties: dict[tuple[str, str], Percept] | None = None,
) -> str:
    """Why a ``path`` failed against a source that *is* present — kept apart from the missing-source
    defect on purpose. Collapsing the two told the planner ``{'$from': 'search_events', 'path':
    'events'} names no result the plan has produced; operations run so far: search_events`` — a
    brief that contradicts itself and aims the repair at a step that had just run. The planner
    rewrote the same reference, the trail saw the same defect twice, and the activity halted on a
    question the user could not usefully answer. So: say the source ran, name the segment that did
    not fit, and show what is actually there to name instead.

    A ``$prop`` is split by `_property_ref` rather than by dropping ``path``, because its route can
    be *folded into the token* — for `{"$prop": "Contacts.state.nope"}` there is no ``path`` key to
    drop, so re-resolving the stripped ref just re-ran the same failing walk and re-raised out of
    this function, out of `_resolve_collection`, and out of `tick()`: the plan-defect layer aborting
    the run it exists to recover. Splitting head from route composes the folded and explicit halves
    the same way `_resolve_ref` does, so either spelling reports the same segment."""
    if _REF_PROP in ref:
        source, residual = _property_ref(properties or {}, str(ref[_REF_PROP]))
        if source is _MISSING or source is _AMBIGUOUS:
            # The HEAD did not resolve: an unobserved or ambiguous property is a different question
            # from a bad route, and _collection_defect is the one that names focusing/qualifying.
            return _collection_defect(ref, _MISSING, history, properties)
        path = ".".join(p for p in (residual, str(ref.get(_REF_PATH, ""))) if p)
        return _walk_defect(ref, source, path)
    source = _resolve_ref(
        {k: v for k, v in ref.items() if k != _REF_PATH}, history, bindings, properties
    )
    path = str(ref.get(_REF_PATH, ""))
    return _walk_defect(ref, source, path)


def _walk_defect(ref: dict[str, Any], source: Any, path: str) -> str:
    """Walk `path` into an already-resolved `source` and describe where it stopped fitting."""
    value: Any = source
    walked: list[str] = []
    failed = path
    for segment in filter(None, path.split(".")):
        try:
            value = value[int(segment)] if segment.isdigit() else value[segment]
        except (KeyError, IndexError, TypeError, ValueError):
            failed = segment
            break
        walked.append(segment)
    at = ".".join(walked) or "the result itself"
    if isinstance(value, dict):
        keys = ", ".join(repr(k) for k in list(value)[:8]) or "no keys"
        holds = f"{at} is a mapping with keys {keys}"
    elif isinstance(value, list):
        holds = f"{at} is a list of {len(value)} item(s) — only a numeric segment indexes it"
    else:
        holds = f"{at} is a {type(value).__name__}"
    return (
        f"{ref!r} names a source that IS present and does not need to run again, but its 'path' "
        f"does not fit that result: {failed!r} is not readable there, because {holds}. Correct the "
        "'path' to a field that is, or drop 'path' if the source is already the collection."
    )


def _resolve_collection(
    ref: Any,
    history: list[CompletedOperation],
    bindings: dict[str, Any] | None = None,
    properties: dict[tuple[str, str], Percept] | None = None,
) -> tuple[list[Any] | None, str | None]:
    """The collection a mechanical sub-goal iterates or a data-op transforms: a ``$from`` reference
    resolved against history, a ``$bind`` reference resolved against named bindings (a prior data-op
    output), or a literal list. A resolved mapping iterates its values (see ``_as_collection``).
    Returns ``(collection, defect)``. A ``defect`` is set exactly when the reference could not be
    *read* — a missing source, a bad path, or a resolved value of a shape these tiers refuse — and
    the caller replans on it rather than proceeding. That distinction is the whole point of the
    pair: a collection that is legitimately empty and a collection the runtime could not read both
    used to come back as "nothing to do", so a sub-goal over an unreadable reference fanned out to
    zero steps and the plan sailed past it as though the work were done. An observed run dropped
    three calendar cancellations that way without a single error surfacing. Empty is an answer;
    unreadable is a question, and only the second is a plan defect worth another inference.

    ``(None, None)`` means soft, not failed: a ``$decide`` collection is resolved off-cycle, so
    there is nothing to read here yet and nothing to blame the plan for."""
    if _is_reference(ref):
        if _REF_DECIDE in ref:
            return None, None  # a $decide collection is soft — resolved off-cycle, not a defect
        try:
            value: Any = _resolve_ref(ref, history, bindings or {}, properties)
        except (KeyError, IndexError, TypeError, ValueError):
            return None, _path_defect(ref, history, bindings or {}, properties)
        if value is _MISSING:
            return None, _collection_defect(ref, value, history, properties)
    else:
        value = ref  # a literal (already a list, mapping, or a plan author's mistake)
    collection = _as_collection(value)
    if collection is None:
        return None, _collection_defect(ref, value, history, properties)
    return collection, None


def _enrich_with_params(result: Any, params: dict[str, Any]) -> Any:
    """``collect`` carries each fanned-out call's input params alongside its result, so a downstream
    ``filter``/membership can correlate a result back to the input that produced it — a crime rate
    back to its ``zip_code`` when ``get_crime_rate`` doesn't echo the zip. A dict result is enriched
    in place, the params filling in only the keys the result doesn't already carry (the
    authoritative return wins on collision); a non-dict result is wrapped as
    ``{**params, "result": <value>}`` so the key stays reachable. Empty params (or none) -> the
    result untouched."""
    if not params:
        return result
    if isinstance(result, dict):
        return {**params, **result}
    return {**params, "result": result}


_ORDERED_OPS = ("lt", "le", "gt", "ge")  # the ops whose operand must be a single comparable value


def _shape_of(value: Any) -> str:
    """A resolved value's shape, phrased for a plan defect — what the planner needs to see is what
    it got instead of a comparable value, not the value itself (which can be a whole record)."""
    if value is None:
        return "None"
    if isinstance(value, dict):
        keys = ", ".join(repr(k) for k in list(value)[:8]) or "no keys"
        return f"a mapping with keys {keys}"
    if isinstance(value, (list, tuple)):
        return f"a list of {len(value)} item(s)"
    return f"a {type(value).__name__}"


def _resolve_operand_items(
    items: list[Any] | tuple[Any, ...],
    history: list[CompletedOperation],
    bindings: dict[str, Any],
    properties: dict[tuple[str, str], Percept] | None = None,
) -> tuple[list[Any], str | None]:
    """Resolve references written *inside* a list operand, element-wise.

    A list is not itself a reference, so ``_is_reference`` says no to it and the whole thing used
    to pass through as a literal. But the natural way to write a ``between`` whose ends are each
    computed is exactly a list of references — ``[{"$bind": "lo"}, {"$bind": "hi"}]`` — since a
    reference in the *whole-value* position would have to resolve to the pair already assembled,
    which no single earlier step produces. Written the natural way, the reference dicts reached
    ``_matches`` intact, every ``lo <= actual`` raised ``TypeError``, that was caught as a
    non-match, and the filter kept nothing at all while reporting an ordinary empty result. The
    same applies to a literal ``in`` set with a reference among its members.

    Resolving per element makes both spellings mean what they read as. A defect here is a defect
    for the whole predicate: a pair with one unreadable end is no more comparable than one with
    two."""
    resolved: list[Any] = []
    for item in items:
        if not _is_reference(item):
            resolved.append(item)
            continue
        if _REF_DECIDE in item:
            return [], (
                f"a filter predicate's 'value' cannot contain a $decide reference "
                f"({item[_REF_DECIDE]!r}) — the comparison runs mechanically, so every part of "
                "the operand has to be known before it. Compute it in an earlier step and "
                "reference that binding, or make the whole 'where' a $decide predicate."
            )
        try:
            value = _resolve_ref(item, history, bindings, properties)
        except (KeyError, IndexError, TypeError, ValueError):
            return [], _path_defect(item, history, bindings, properties)
        if value is _MISSING:
            return [], _collection_defect(item, value, history, properties)
        resolved.append(value)
    return resolved, None


def _operand_defect(where: dict[str, Any], written: Any, operand: Any) -> str | None:
    """Why a predicate operand that *read* cleanly still cannot be compared against.

    Resolving is not the same as being usable, and the gap between the two is silent: an operand
    that lands on ``None``, a whole record, or a list makes every ordered comparison raise
    ``TypeError`` inside ``_matches``, where it is caught as a non-match by design — so no element
    survives and the step writes an empty binding that reads downstream as a fact about the world.
    ``between`` is the same trap one level up: it compares against exactly a two-element pair and
    treats anything else as a blanket non-match. In both cases the filter is not selecting badly,
    it is not selecting at all, so this is reported as a plan defect rather than an answer.

    ``eq``/``ne`` are deliberately excluded. An operand of any shape can genuinely match there — a
    field that really is null, an object compared whole — so refusing one would refuse a
    legitimate predicate to guard against a mistake that isn't provable from the shape alone.
    ``in``/``not_in`` are excluded too: their operand is a collection, already checked as one."""
    op = where.get("op", "eq")
    if op == "between":
        if not (isinstance(operand, (list, tuple)) and len(operand) == 2):
            return (
                f"the 'between' predicate's value {written!r} is {_shape_of(operand)}, not the "
                "[lo, hi] pair 'between' compares against — every element then fails the "
                "comparison and the filter keeps nothing at all. Give it two ends: a reference "
                "that resolves to a pair, or the two bounds written out as [<lo>, <hi>]."
            )
        if any(end is None for end in operand):
            return (
                f"the 'between' predicate's value {written!r} resolved to {list(operand)!r}, a "
                "pair with a missing end — a null bound excludes every element. Produce both "
                "bounds before comparing against them, or use 'lt'/'gt' against the one that "
                "exists."
            )
        return None
    if op in _ORDERED_OPS and (operand is None or isinstance(operand, (dict, list, tuple))):
        return (
            f"the {op!r} predicate's value {written!r} is {_shape_of(operand)}, not a single "
            f"value to compare against — {op!r} then fails for every element and the filter "
            "keeps nothing at all. Add a 'path' naming the field that holds the threshold, or "
            "compute the threshold in an earlier step and reference that binding."
        )
    return None


# Mirrors sora.action's predicate grammar — the evaluator there walks these, the resolver here
# fills them in. Kept as constants on both sides rather than imported, for the same reason
# `_REF_DECIDE` is: the two modules share a wire format, not an implementation.
_COMPOSE_ALL = "all"

_COMPOSE_ANY = "any"

_OP_OVERLAPS = "overlaps"


def _composition_defect(key: str, clauses: Any) -> str | None:
    """Why an ``all``/``any`` composition cannot be walked at all.

    Both failures are silent and both are dangerous in the same direction. A non-list has no
    clauses to evaluate; an EMPTY list is worse, because an empty conjunction is vacuously *true*
    in logic — written naively it would keep the whole collection, and a filter's output is
    routinely fanned out into one external action per item. The evaluator therefore matches nothing
    for either, and this reports it rather than let the step write an empty binding that reads
    downstream as a fact about the world."""
    if not isinstance(clauses, list):
        return (
            f"the {key!r} predicate's clauses are {_shape_of(clauses)}, not a list of predicates "
            f"to combine — nothing can be evaluated and the filter keeps nothing at all. Write "
            f"{key!r} as a list, each entry its own clause."
        )
    if not clauses:
        return (
            f"the {key!r} predicate has an empty clause list, so it selects nothing. If there is "
            "only one condition, write that comparison directly instead of composing it; if a "
            "clause was meant to come from an earlier step, produce it first."
        )
    return None


def _resolve_overlaps_against(
    where: dict[str, Any],
    history: list[CompletedOperation],
    bindings: dict[str, Any],
    properties: dict[tuple[str, str], Percept] | None = None,
) -> tuple[Any, str | None]:
    """Resolve an ``overlaps`` clause's ``against`` into plain ``[start, end]`` pairs.

    The third projection shape, alongside the membership set and the bare operand — and the reason
    ``overlaps`` needs no per-member alias grammar. ``against`` names a *collection*, so it resolves
    once here exactly as an ``in`` set does, and the two paths that read each member's interval are
    applied at resolution time. What reaches ``_matches`` is a literal list of pairs, which keeps
    the "resolve once in Reason, compare literals in the evaluator" invariant that every other op
    already relies on; deferring a reference to evaluation time instead would need a second
    resolution regime for one op's benefit.

    An unreadable ``against`` fails *closed* (nothing overlaps nothing), so it is reported rather
    than swallowed, like every other operand here."""
    against = where.get("against")
    if _is_reference(against):
        if _REF_DECIDE in against:
            return where, (
                f"an 'overlaps' predicate's 'against' cannot be a $decide reference "
                f"({against[_REF_DECIDE]!r}) — the comparison runs mechanically, so the intervals "
                "have to be known before it. Produce that collection in an earlier step and "
                "reference the binding it wrote."
            )
        resolved_members, defect = _resolve_collection(against, history, bindings, properties)
        if defect is not None:
            return where, defect
        members = resolved_members or []
    elif isinstance(against, list):
        members = against
    else:
        return where, (
            f"the 'overlaps' predicate's 'against' is {_shape_of(against)}, not a collection of "
            "intervals to compare with — the comparison then fails for every element and the "
            "filter keeps nothing at all. Give it a reference to the collection whose items carry "
            "the other interval, plus 'against_start_path'/'against_end_path' naming its two ends."
        )
    start_path = where.get("against_start_path", "")
    end_path = where.get("against_end_path", "")
    projected = [[pluck(m, start_path), pluck(m, end_path)] for m in members]
    # The sibling of the membership-set warning below: a member whose ends do not project to
    # comparable values can never overlap anything, so it drops out of the comparison silently and
    # the filter quietly narrows more than the plan asked. The usual cause is a path naming a field
    # the records don't carry (-> None) or pointing at a nested object. An empty collection stays
    # silent — `_resolve_collection` already logged why, and genuinely nothing-was-added is a real
    # and common answer here.
    if members and any(
        end is None or isinstance(end, (dict, list)) for pair in projected for end in pair
    ):
        bad = sum(
            1 for pair in projected if any(e is None or isinstance(e, (dict, list)) for e in pair)
        )
        log.warning(
            "filter: 'overlaps' against-collection has %d/%d member(s) with an end that projected "
            "to a non-scalar or None (against_start_path=%r, against_end_path=%r); those can never "
            "overlap anything — check the two paths",
            bad,
            len(projected),
            start_path,
            end_path,
        )
    resolved = {
        k: v for k, v in where.items() if k not in ("against_start_path", "against_end_path")
    }
    resolved["against"] = projected
    return resolved, None


def _resolve_predicate_value(
    params: dict[str, Any],
    history: list[CompletedOperation],
    bindings: dict[str, Any],
    properties: dict[tuple[str, str], Percept] | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Resolve a ``filter`` predicate's ``value`` when it is a reference rather than a literal, so
    ``_matches`` stays a pure literal comparison and the resolution lives here in Reason, next to
    the ``in``-collection resolution (ADR-0023 extension). Two shapes, because the ops read
    ``value`` differently:

    * ``in``/``not_in`` compare against a *set*: the reference resolves as a collection and is
      projected by ``value_path`` (default: the elements themselves, for a reference that already
      resolves to a list of scalars) into a list of comparable keys.
    * every other op compares against the value *itself*: the reference resolves to whatever it
      names — a scalar for ``eq``/``ne``/``lt``/``le``/``gt``/``ge``, the pair for ``between`` —
      and is **not** projected, since here the value is the operand rather than a collection to
      key on. This is the threshold shape ADR-0023's own reduce-then-compare pipeline is written
      in, and it was unreachable while resolution was gated on the membership ops: the raw
      reference dict reached ``_matches``, every comparison against it raised ``TypeError``, that
      was caught as a non-match, and the filter silently kept *nothing*.

    A reference may also sit *inside* a list operand rather than being the whole of it — the only
    way to write a ``between`` whose two ends come from two different steps — so those are resolved
    element-wise first (``_resolve_operand_items``); a list is not a reference, so without that the
    pair reached ``_matches`` with its reference dicts intact and kept nothing.

    A whole-predicate ``$decide`` (which ``FilterAction`` escalates intact) passes through
    untouched. A literal ``value`` has nothing to resolve, but is still shape-checked like a
    resolved one: whether an uncomparable operand was written out or arrived through a reference
    makes no difference to the filter it kills. Returns the resolved params and a ``defect``, set
    when the reference could not be read, or when it read cleanly but landed on a shape the op
    cannot compare against (``_operand_defect``) — reported rather than swallowed for the same
    reason the fan-out reports it:
    an unreadable predicate value fails *open* in some direction for every op (``in`` matches
    nothing, ``not_in`` keeps everything, an unreadable threshold excludes everything), so the
    filter confidently does the wrong thing to the whole collection. A resolved-but-unusable
    projection (a ``value_path`` that plucks to ``None`` or a non-scalar for every member) is the
    neighbouring trap the warning below guards, and fails open the same way."""
    where = params.get("where")
    if not isinstance(where, dict):
        return params, None  # no predicate
    resolved, defect = _resolve_predicate_clause(where, history, bindings, properties)
    if defect is not None:
        return params, defect
    return ({**params, "where": resolved} if resolved is not where else params), None


def _resolve_predicate_clause(
    where: Any,
    history: list[CompletedOperation],
    bindings: dict[str, Any],
    properties: dict[tuple[str, str], Percept] | None = None,
) -> tuple[Any, str | None]:
    """One clause of a predicate, resolved — the recursive worker behind
    ``_resolve_predicate_value``, and where that function's documented shapes are actually applied.

    Recursive because a predicate composes. ``all``/``any`` hold child clauses, each resolving its
    own operand exactly as a lone clause would, and a defect anywhere is a defect for the *whole*
    predicate: a conjunction with one dead clause selects nothing, and a disjunction with one
    silently drops whatever that clause was meant to catch. Composition resolves nothing itself —
    it is structure the evaluator walks — so all that is checked of it here is that it can be
    walked."""
    if not isinstance(where, dict):
        return where, (
            f"a composed predicate contains {_shape_of(where)}, not an object describing a "
            "predicate clause — the malformed clause selects nothing. Write each 'all'/'any' "
            "entry as its own predicate object."
        )
    if _REF_DECIDE in where:
        return where, None  # a soft clause is escalated whole rather than resolved mechanically
    for key in (_COMPOSE_ALL, _COMPOSE_ANY):
        if key not in where:
            continue
        clauses = where[key]
        if (composition_defect := _composition_defect(key, clauses)) is not None:
            return where, composition_defect
        resolved_clauses: list[Any] = []
        for clause in clauses:
            got, clause_defect = _resolve_predicate_clause(clause, history, bindings, properties)
            if clause_defect is not None:
                return where, clause_defect
            resolved_clauses.append(got)
        return {**where, key: resolved_clauses}, None
    if where.get("op") == _OP_OVERLAPS:
        return _resolve_overlaps_against(where, history, bindings, properties)
    written = where.get("value")  # what the plan wrote, kept for the defect messages
    value: Any = written
    if isinstance(value, (list, tuple)) and any(_is_reference(v) for v in value):
        items, list_defect = _resolve_operand_items(value, history, bindings, properties)
        if list_defect is not None:
            return where, list_defect
        where = {**where, "value": items}
        value = items
    if not _is_reference(value):
        # A literal, or a list whose members just resolved: nothing left to resolve, but the shape
        # still has to be one the op can compare against — an uncomparable operand is as dead
        # written out as it is referenced.
        return where, _operand_defect(where, written, value)
    if _REF_DECIDE in value:
        # A soft reference in the *operand* position, which no prompt documents and no op can
        # evaluate: the comparison itself is mechanical. It used to resolve to an empty membership
        # set (silently fails open) or a raw dict (silently excludes everything), so say what to
        # write instead rather than let either happen.
        return where, (
            f"a filter predicate's 'value' cannot be a $decide reference "
            f"({value[_REF_DECIDE]!r}) — the comparison runs mechanically, so the operand has to "
            "be known before it. Compute it in an earlier step and reference that binding, or "
            "make the whole 'where' a $decide predicate."
        )
    if where.get("op") not in ("in", "not_in"):
        try:
            operand: Any = _resolve_ref(value, history, bindings, properties)
        except (KeyError, IndexError, TypeError, ValueError):
            return where, _path_defect(value, history, bindings, properties)
        if operand is _MISSING:
            return where, _collection_defect(value, operand, history, properties)
        if (shape_defect := _operand_defect(where, written, operand)) is not None:
            return where, shape_defect
        return {**where, "value": operand}, None
    resolved_members, defect = _resolve_collection(value, history, bindings, properties)
    if defect is not None:
        return where, defect
    members = resolved_members or []
    projected = [pluck(m, where.get("value_path", "")) for m in members]
    # A membership set is compared element-by-element against a scalar key, so only scalar members
    # can ever match. A non-scalar (dict/list) or ``None`` projection is dead weight: `in` silently
    # drops it, `not_in` silently keeps it. The usual cause is a `value_path` that's missing (the
    # referenced collection is records, not bare keys), wrong (names a field the records don't
    # carry -> None), or points at a nested object — surface any such member rather than let the
    # filter fail open invisibly (an all-None projection is exactly the duplicate-action trap:
    # "not already saved" keeps everything). An empty set (nothing resolved) stays silent here —
    # _resolve_collection already logged *why*, and a genuinely empty exclusion list is benign.
    if members and any(p is None or isinstance(p, (dict, list)) for p in projected):
        bad = sum(1 for p in projected if p is None or isinstance(p, (dict, list)))
        log.warning(
            "filter: membership set for %r has %d/%d member(s) that projected to a non-scalar or "
            "None key (value_path=%r); those can never match — `in` drops them, `not_in` keeps "
            "them — check `value_path`",
            where.get("path"),
            bad,
            len(projected),
            where.get("value_path"),
        )
    resolved_where = {k: v for k, v in where.items() if k != "value_path"}
    resolved_where["value"] = projected
    return resolved_where, None
