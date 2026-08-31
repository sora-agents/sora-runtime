"""Reference grammar, deterministic resolution, and diagnostics."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeGuard

from sora.action import (
    FilterAction,
    InvokeAction,
)
from sora.activity import Activity
from sora.perception import Percept
from sora.types import (
    CompletedOperation,
    OperationInvocation,
    Step,
    walk_path,
)

if TYPE_CHECKING:
    from sora.manual import Manual
    from sora.memory import WorkingMemory

# The dotted-path walker lives in sora.types now (shared with the data-ops in sora.action, which
# can't import this module); keep the private alias so the many call sites below stay untouched.
_walk_path = walk_path


# --- parameter grounding: references + a deterministic resolver ----------------------------------
# A plan is a reusable *skeleton*; a param whose value depends on a prior step's result can't be a
# literal at plan time, so the planner emits a *reference* the Reason phase grounds each run against
# the activity's execution history. Two forms (see ADR-0017):
#   hard: {"$from": "<operation_name>", "path": "<dotted path>"}  -> resolved deterministically
#   soft: {"$decide": "<natural-language description>"}           -> always escalates to the model
_REF_FROM = "$from"

_REF_PATH = "path"

_REF_DECIDE = "$decide"

# The named-binding read token. Distinct from $from (which reads Activity.history): $bind reads a
# named binding — either a data-op's output binding in Activity.bindings (ADR-0023), resolved here
# at ground/fan-out time against that dict, or the current loop element of a mechanical sub-goal
# (ADR-0022), which is substituted eagerly at fan-out by _substitute_bindings and so never reaches
# this resolver. The two coexist: the loop element is gone before grounding runs, so any $bind left
# here is a binding read.
_REF_BIND = "$bind"

_REF_NAME = "$bind"  # the binding-name key inside a $bind reference (same token, read as a key)

# The observed-world-state read token. The third and last binding source (ADR-0022): $from reads
# Activity.history, $bind reads the named-binding namespace, $prop reads WorkingMemory.properties —
# the snapshot Observe refreshes each cycle for every focused tool. It resolves per step, at the
# same point $from does, because a property is re-observed state whose whole value is being current;
# binding it once at plan entry would freeze a moving value for the plan's life.
_REF_PROP = "$prop"

_MISSING = object()  # sentinel: no matching history entry (distinct from a genuine None result)

_AMBIGUOUS = object()  # sentinel: a bare property name several focused tools expose

_BAD_PATH = object()  # sentinel: the source IS present, its `path` names nothing inside it


def _is_reference(value: Any) -> TypeGuard[dict[str, Any]]:
    return isinstance(value, dict) and (
        _REF_FROM in value or _REF_DECIDE in value or _REF_BIND in value or _REF_PROP in value
    )


def _latest_result(history: list[CompletedOperation], reference: str) -> Any:
    """The result of the most recent completed operation this reference names, or _MISSING.

    Three accepted spellings, tried most-precise first, because a plan may name an operation any of
    the ways its own brief shows one. The bare ``operation_name`` is the canonical form. The
    fully-qualified ``tool_id.operation_name`` is what a planner reaches for after reading a catalog
    that addresses every operation that way, and it is the *more* specific match when two joined
    workspaces expose the same operation (ARE's Contacts and InternalContacts both have
    ``get_contacts``), so it is honored rather than merely tolerated. Last, the segment after the
    final dot, for a qualification whose prefix matches no tool actually invoked (an abbreviated or
    misremembered tool id) — the operation name never contains a dot, so this is unambiguous.

    Accepting all three is not laxity: a reference the runtime refuses resolves to nothing and, at
    a fan-out, used to vanish silently. Refusing a reference whose *intent* is unambiguous buys no
    safety and costs a whole plan."""

    def _latest(matches: Callable[[OperationInvocation], bool]) -> Any:
        for completed in reversed(history):
            if matches(completed.invocation):
                return completed.ack.result
        return _MISSING

    result = _latest(lambda inv: inv.operation_name == reference)
    if result is not _MISSING:
        return result
    result = _latest(lambda inv: f"{inv.tool_id}.{inv.operation_name}" == reference)
    if result is not _MISSING or "." not in reference:
        return result
    tail = reference.rsplit(".", 1)[-1]
    return _latest(lambda inv: inv.operation_name == tail)


def _property_ref(properties: dict[tuple[str, str], Percept], reference: str) -> tuple[Any, str]:
    """Resolve a ``$prop`` reference to ``(value, residual_path)``, ``_MISSING``, or ``_AMBIGUOUS``.

    Two spellings reach here and both name one value. The canonical one keeps the sub-path in its
    own ``path`` key; a planner that has just read a catalog addressing everything by dotted name
    folds the whole route into the token instead — ``insim:are/Contacts.state.contacts`` for
    ``{"$prop": "insim:are/Contacts.state", "path": "contacts"}``. Honoring only the first cost a
    whole plan on the 2026-08-21 adaptability run, for a spelling difference; this is the tolerance
    ``_latest_result`` already grants ``$from``, applied to the token that lacked it.

    The split is found by **matching against the live key set**, never by parsing the string. That
    matters because the store's key is ``(source, name)`` and *neither half is dot-free*: a WoT tool
    id contains them (``wot:lamp.local/Lamp``), and while today's adapters happen to mint dot-free
    property names, nothing in ``ObservablePropertySpecification`` or the adapter boundary forbids a
    property called ``sensor.temp`` — the runtime does not author names (ADR-0003), so it cannot
    assume their shape. Joining each candidate key back to ``f"{source}.{name}"`` and comparing asks
    the store what it actually holds, so a dotted property name resolves and a dotted tool id keeps
    resolving, with no rule about where the boundary "should" be.

    Longest reference first, so an exact key always beats a folded reading of the same string, and
    within one length a **qualified** match beats a bare one (naming the tool is more specific).
    Where a length is genuinely ambiguous — several tools exposing one bare name, or two distinct
    keys that join to the same string — it comes back ``_AMBIGUOUS`` rather than whichever the dict
    happened to yield first. ARE gives thirteen tools a ``state`` property, so guessing here would
    be a silent wrong answer: worse than a missing one, because it is harder to see."""
    cuts = [i for i, ch in enumerate(reference) if ch == "."]
    for cut in [len(reference), *reversed(cuts)]:
        head, residual = reference[:cut], reference[cut + 1 :]
        keys = [key for key in properties if f"{key[0]}.{key[1]}" == head]
        if not keys:  # unqualified: the planner named the property without its tool
            keys = [key for key in properties if key[1] == head]
        if len(keys) == 1:
            return properties[keys[0]].payload.value, residual
        if keys:
            return _AMBIGUOUS, ""
    return _MISSING, ""


def _resolve_ref(
    ref: dict[str, Any],
    history: list[CompletedOperation],
    bindings: dict[str, Any],
    properties: dict[tuple[str, str], Percept] | None = None,
) -> Any:
    """Resolve one *hard* reference — ``$from`` (history) or ``$bind`` (a named binding) — to its
    value, walking the ``path`` into it. ``_MISSING`` when the source is absent (no such op ran / no
    such binding). Raises on a bad path (a present source, wrong path) so the caller can distinguish
    "escalate" from "left in place". ``$decide`` is soft and never resolved here."""
    if _REF_FROM in ref:
        result = _latest_result(history, str(ref[_REF_FROM]))
        return _MISSING if result is _MISSING else _walk_path(result, ref.get(_REF_PATH, ""))
    if _REF_NAME in ref:
        name = ref[_REF_NAME]
        if name not in bindings:
            return _MISSING
        return _walk_path(bindings[name], ref.get(_REF_PATH, ""))
    if _REF_PROP in ref:
        value, residual = _property_ref(properties or {}, str(ref[_REF_PROP]))
        if value is _MISSING or value is _AMBIGUOUS:
            return _MISSING  # both escalate; _collection_defect re-reads which, to say why
        # A sub-path folded into the token is walked *before* an explicit `path`, since it names the
        # outer route; the two compose so a half-folded reference resolves the same as either form.
        folded = ".".join(p for p in (residual, str(ref.get(_REF_PATH, ""))) if p)
        return _walk_path(value, folded)
    return _MISSING


def _resolve_nested(
    value: Any,
    history: list[CompletedOperation],
    bindings: dict[str, Any],
    properties: dict[tuple[str, str], Percept] | None = None,
) -> tuple[Any, bool]:
    """Resolve every reference *anywhere* in ``value``, returning ``(resolved, fully_resolved)``.

    References nest because the plan schema makes them nest: a param typed ``list[str]`` whose one
    element is only known at run time can *only* be written ``[{"$decide": ...}]`` — a reference as
    the whole value would yield a string, not a list. The resolver used to look at top-level param
    values only, so such a reference was neither resolved nor reported unresolved, and the raw
    ``{"$decide": ...}`` dict was serialized to the tool as a literal (ARE reported it as
    ``Argument 'attendees' must be of type list[str] | None, got <class 'list'>``, naming the wrong
    culprit). It surfaced only when the *sole* reference in a step was nested — any independently
    unresolved top-level param escalated the whole step anyway and the model grounder filled it in,
    which is why it hid for so long.

    A dict that *is* a reference is resolved, not descended into; every other dict/list is rebuilt
    element-wise. Partial resolution is deliberate: a list holding one resolvable ``$from`` and one
    ``$decide`` comes back with the ``$from`` filled and the ``$decide`` left in place, so the
    escalation the caller raises hands the grounder as much settled context as possible."""
    if _is_reference(value):
        if _REF_DECIDE in value:
            return value, False  # soft — always escalates, left in place for the model
        try:
            got = _resolve_ref(value, history, bindings, properties)
        except (KeyError, IndexError, TypeError, ValueError):
            return value, False  # bad path against a present source
        return (value, False) if got is _MISSING else (got, True)
    if isinstance(value, dict):
        pairs = [
            (key, _resolve_nested(item, history, bindings, properties))
            for key, item in value.items()
        ]
        return {key: got for key, (got, _ok) in pairs}, all(ok for _key, (_got, ok) in pairs)
    if isinstance(value, list):
        items = [_resolve_nested(item, history, bindings, properties) for item in value]
        return [got for got, _ok in items], all(ok for _got, ok in items)
    return value, True


def _reference_paths(value: Any, prefix: str = "") -> list[str]:
    """Dotted paths of every reference token surviving in ``value`` — at any depth. ``[]`` is the
    healthy case. Used as Act's leak guard; see ``DefaultActStrategy``."""
    if _is_reference(value):
        return [prefix or "<root>"]
    if isinstance(value, dict):
        return [
            path
            for key, item in value.items()
            for path in _reference_paths(item, f"{prefix}.{key}" if prefix else str(key))
        ]
    if isinstance(value, list):
        return [
            path
            for index, item in enumerate(value)
            for path in _reference_paths(item, f"{prefix}.{index}" if prefix else str(index))
        ]
    return []


def _manual_for(wm: WorkingMemory, tool_id: str | None) -> Manual | None:
    """The joined tool's manual (the operation schema the model grounds against), or None."""
    if tool_id is None:
        return None
    try:
        return wm.registry.get(tool_id).manual
    except KeyError:
        return None


def resolve_references(
    op_params: dict[str, Any],
    history: list[CompletedOperation],
    bindings: dict[str, Any] | None = None,
    properties: dict[tuple[str, str], Percept] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Resolve a step's operation params against execution history and named bindings. Non-reference
    values pass through; a hard reference (``$from``/``$bind``) is resolved deterministically;
    anything that can't be resolved mechanically (soft ``$decide``, missing source, bad path) is
    left in place and its key returned in ``unresolved`` for the caller to escalate. Never raises on
    a bad path — that's an escalation signal, not an error.

    A reference is found wherever it sits — as a param's whole value, or nested inside a list or
    dict the param holds (see ``_resolve_nested``). ``unresolved`` names the *top-level key*
    whatever the depth, because that is the unit grounding escalates on (``partial_params`` is
    per-param), so one stubborn leaf re-grounds its whole param."""
    binds = bindings or {}
    resolved: dict[str, Any] = {}
    unresolved: list[str] = []
    for key, value in op_params.items():
        got, ok = _resolve_nested(value, history, binds, properties)
        resolved[key] = got
        if not ok:
            unresolved.append(key)
    return resolved, unresolved


# --- irreversibility guard: never commit a write on a plan already known to be dead ---------------
# A run showed why this is needed. A filter chain wrote `friend_contact = []` at step 3 (the planner
# had read only the first page of contacts, so nobody matched); the plan then kept going — reading
# the calendar at step 4 and DELETING the user's real appointment at step 5 — and only tripped over
# the empty binding at step 7, where it finally needed the friend's address. The plan was already
# unfinishable two steps before the irreversible act, and the evidence was sitting in `bindings`.
#
# So this is not an ordering rule: that plan *was* ordered correctly, gathering before destroying.
# It is a viability rule. Nothing rolls a delete back, and the asymmetry is stark — abandoning a
# plan that might still have worked costs one more inference, while acting on a plan that cannot
# work costs the user something real and unrecoverable. So: check before a write, and only a write.
#
# "Provably" is meant strictly. Only a binding an earlier step *already produced* and produced empty
# counts, and only where a later step reads a VALUE out of it — a collection position (`in`, a
# membership `where`) is exempt, because an empty collection there is a legitimate answer ("nothing
# to iterate", "exclude nothing"), which is the same line _data_op already draws. A name a later
# step rewrites is exempt from that point on, since it is no longer provably anything.
#
# The same proof holds for a `$from` read of an operation that already ran and came back empty, and
# a later run showed the guard missing it for want of scanning that token: a plan invoked
# `search_contacts`, got `[]`, and the runtime still committed `add_calendar_event` two steps on,
# creating the event with no attendee. Identical evidence, identical asymmetry, different spelling —
# so both references are scanned, with `refreshed` playing for operations the part `out` plays for
# bindings (a step ahead of the read that re-invokes the operation makes it no longer provably
# anything, which is what keeps a replan's second attempt at a search from being condemned by its
# first attempt's empty result).
#
# What does NOT count for a `$from`, deliberately: an operation that has not run yet (a plan
# normally reads at step 3 what it invokes at step 1 — absence here is not evidence), and a
# *present but mis-pathed* source. The latter is a real defect but a recoverable one: grounding
# reads the actual history and routinely resolves a value the path spelled wrong, so condemning
# would pre-empt a repair that works. An empty source admits no such repair — there is no value at
# any path — which is the line between the two.

# Keys whose value is a collection rather than a value read out of one: an empty binding in these
# positions is an answer, not a defect (see above). `from` is the collect data-op's operation name.
_COLLECTION_KEYS = frozenset({"in", "where", "from"})


def _is_empty(value: Any) -> bool:
    """Empty in the sense that a step reading a value out of it cannot get one. Deliberately not
    falsiness: 0 and False are perfectly good values a step can act on."""
    if value is None:
        return True
    return len(value) == 0 if isinstance(value, str | bytes | list | tuple | set | dict) else False


def _dereferenced_bindings(step: Step) -> set[str]:
    """Binding names this step reads a *value* out of, at any nesting depth.

    A mechanical sub-goal's template always contains ``{"$bind": "<as>"}`` — its own loop element,
    bound per iteration at fan-out, not a name the plan produced. Nothing stops a plan from
    spelling ``as`` the same as a real binding (``filter(out: "contacts")`` then
    ``subgoal(in: {"$bind": "contacts"}, as: "contacts")`` is a natural thing to write), and when
    that binding is empty the template's read looked like a dereference of it — so a fan-out that
    legitimately reduces to zero steps ("nothing matched, nothing to do") condemned the plan at the
    next write. The loop name is excluded explicitly rather than assumed distinct."""
    loop_var = step.params.get("as")
    names: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            name = value.get(_REF_BIND)
            if isinstance(name, str):
                if name != loop_var:
                    names.add(name)
                return
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    for key, value in step.params.items():
        if key not in _COLLECTION_KEYS:
            walk(value)
    return names


# --- attributing an empty binding to the step that produced it ------------------------------------
# An empty binding is evidence about the step that WROTE it, not the step that tripped over it, and
# the two defect channels both used to describe it from the reader's side. Grounding does so for a
# reason it cannot help: it is handed one step's params and the history, never the plan, so the most
# it can honestly report is which parameter came up short. The replanning prompt then says the
# replacement "has to differ THERE" — and THERE resolved to the reader. A run showed the cost: the
# planner rewrote the invoke that read the binding, re-emitted the identical `eq` filter that had
# written it empty, and bound empty again, three times until the replan breaker parked the activity.
#
# The runtime holds the plan that the grounder does not, so it names the producer mechanically —
# no prose parsing, no second model call. It only INFORMS: the empty collection may be the true
# answer ("that record really is absent"), and the existing framing's "tell the user instead" escape
# has to stay reachable, so nothing here instructs the planner to change that step.
#
# The producing step is deliberately identified by its action and its own params rather than by
# index. It has already RUN, so it is absent from the discarded plan's rendered tail — and
# render_superseded_plan renumbers that tail from 0 precisely so a listed step is not misread as an
# executed one. An index quoted here would point at a different step than the one meant.


def _binding_source(step: Step) -> str | None:
    """The binding a data-op step reads its input collection from, when that is where its input came
    from. Only a ``$bind`` input continues a chain of emptiness: a ``$prop``/``$from``/literal input
    is where the chain ends, and ``collect`` reads an operation rather than a collection.

    Attribution therefore stops at a ``filter`` whose own ``$prop``/``$from`` input was already
    empty, describing that step's predicate when the source it read was the cause. The outcome
    wording stays true and the replan prompt still carries the history showing the upstream
    miss, but the clause points one step downstream. Closing that needs the operation history
    (for ``$from``) and ``WorkingMemory`` (for ``$prop``) threaded into
    :func:`_empty_binding_origin`, which today is given only the activity."""
    source = step.params.get("in")
    if isinstance(source, dict):
        name = source.get(_REF_BIND)
        if isinstance(name, str):
            return name
    return None


def _executed_steps(activity: Activity) -> list[Step]:
    """The activity's already-run steps in execution order, flattened across the frame stack: each
    suspended parent's executed prefix (it ran before the sub-plan was spliced in), then the live
    frame's. Whatever wrote a binding is in here — a binding exists only because a step ran."""
    steps: list[Step] = []
    for parent, index, _ in activity.parent_frames:
        steps.extend(parent.steps[:index])
    if activity.plan is not None:
        steps.extend(activity.plan.steps[: activity.step_index])
    return steps


def _root_empty_producer(
    executed: list[Step], name: str, empty: frozenset[str]
) -> tuple[Step, bool] | None:
    """``(the step an empty binding originates in, whether the chain was walked)``, else ``None``.

    A data-op fed an input binding that was ALREADY empty is only passing the emptiness along;
    naming it would misattribute by one link, which is the same defect this exists to fix. So the
    walk continues through such producers to the first one whose own input was not empty — the
    ``$decide`` filter over an empty ``eq`` result names the ``eq``. ``seen`` guards a plan that
    writes two bindings from each other rather than trusting it cannot."""
    seen: set[str] = set()
    producer: Step | None = None
    walked = False
    while name not in seen:
        seen.add(name)
        wrote = next((s for s in reversed(executed) if s.params.get("out") == name), None)
        if wrote is None:
            break
        walked = producer is not None
        producer = wrote
        source = _binding_source(wrote)
        if source is None or source not in empty:
            break
        name = source
    return None if producer is None else (producer, walked)


def _and_list(names: list[str]) -> str:
    quoted = [repr(name) for name in names]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def _empty_binding_origin(activity: Activity, names: list[str], empty: frozenset[str]) -> str:
    """The clause naming where the empty bindings came from, or ``""`` when none can be attributed
    (a binding no step of this plan wrote — a sub-goal element, or one carried in). Silence is the
    right degradation: the defect without it is what shipped before.

    Bindings sharing one root are named in ONE clause. A `filter` then a `take` off its result are
    two dead bindings with a single cause, and describing that step twice reads as two independent
    defects — noise in a message whose whole point is to say what to write differently."""
    executed = _executed_steps(activity)
    grouped: dict[int, tuple[Step, bool, list[str]]] = {}
    for name in names:
        found = _root_empty_producer(executed, name, empty)
        if found is None:
            continue
        step, walked = found
        origin, seen, bound = grouped.get(id(step), (step, False, []))
        grouped[id(step)] = (origin, seen or walked, [*bound, name])
    clauses: list[str] = []
    for step, walked, bound in grouped.values():
        # `in`/`out` are the plumbing; what discriminates the step is the rest (a `where`, a `by`).
        params = {key: value for key, value in step.params.items() if key not in ("in", "out")}
        # A step with nothing left to show (a `distinct` over whole items) gets no empty `{}` — the
        # dash-clause is there to quote the part that missed, not to prove one was looked for.
        shown = f" — {json.dumps(params, default=str)} —" if params else ","
        outcome = (
            "matched no items in its input collection"
            if step.next_action == FilterAction.name
            else "produced an empty result"
        )
        derived = "; every binding derived from it was empty in consequence" if walked else ""
        clauses.append(
            f" The empty {_and_list(bound)} {'was' if len(bound) == 1 else 'were'} produced by an "
            f"earlier `{step.next_action}` step of this plan{shown} which {outcome}{derived}."
        )
    return "".join(clauses)


def _with_empty_binding_origin(activity: Activity, defect: str) -> str:
    """Append the origin clause to a grounder-authored defect, for the empty bindings the step it
    failed on actually reads. Computed from the plan rather than read out of the grounder's prose:
    the names are already in ``bindings``, so parsing a model's sentence for them would be a
    fragility with nothing to buy it."""
    plan = activity.plan
    if plan is None or not 0 <= activity.step_index < len(plan.steps):
        return defect
    empty = frozenset(name for name, value in activity.bindings.items() if _is_empty(value))
    dead = sorted(_dereferenced_bindings(plan.steps[activity.step_index]) & empty)
    origin = _empty_binding_origin(activity, dead, empty) if dead else ""
    if not origin:
        return defect
    # The grounder is asked for "<which parameter, and what was missing>" and answers with a
    # fragment ("product_id: matches is empty"), not a sentence — close it, or the two run together.
    head = defect.rstrip()
    return head + ("" if head.endswith((".", "!", "?", ":", ";")) else ".") + origin


# Shared tail of both defect strings: what the planner should do about it. The corrections are the
# same whichever token carried the dead reference, and naming them is what makes the retry differ.
_REPLAN_HINT = (
    "Re-plan a way to obtain that data (read the whole collection rather than one page, filter on "
    "a different field, search by another term) before any step that changes the world; the "
    "runtime stopped short of the next one."
)


def _dereferenced_operations(step: Step) -> list[dict[str, Any]]:
    """The ``$from`` reference objects this step reads a *value* out of, at any nesting depth — the
    reference itself, not just its name, since whether it can yield a value depends on its ``path``.

    Nesting matters more here than for ``$bind``: a ``$decide`` element carries its own source under
    a plain ``from`` key (``{"$decide": "...", "from": {"$from": "search", "path": "0"}}``), which
    is the shape that actually slipped a write past this guard. Note that ``_COLLECTION_KEYS``
    filters only the step's *top-level* params, so such a nested ``from`` is still walked — the two
    senses of the word do not collide."""
    refs: list[dict[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get(_REF_FROM), str):
                refs.append(value)
                return
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    for key, value in step.params.items():
        if key not in _COLLECTION_KEYS:
            walk(value)
    return refs


def _spent_operation_read(
    ref: dict[str, Any], history: list[CompletedOperation], refreshed: set[str]
) -> tuple[str, str] | None:
    """``(operation, what it yielded)`` when this reference provably cannot produce a value, else
    ``None``. See the guard's header for why "never ran" and "wrong path" are both excluded."""
    name = str(ref[_REF_FROM])
    # Tail comparison covers all three spellings _latest_result accepts; erring toward "refreshed"
    # errs toward NOT condemning, which is the safe direction for a guard that abandons plans.
    if name in refreshed or name.rsplit(".", 1)[-1] in refreshed:
        return None
    result = _latest_result(history, name)
    if result is _MISSING:
        return None  # not yet run — the step that runs it may be ahead of this read
    if _is_empty(result):
        return name, "returned an empty result"  # no path finds a value in it
    try:
        value = _walk_path(result, str(ref.get(_REF_PATH, "")))
    except (KeyError, IndexError, TypeError, ValueError):
        return None  # present but mis-pathed: grounding can still recover the value
    path = str(ref.get(_REF_PATH, ""))
    return (name, f"returned nothing at {path!r}") if _is_empty(value) else None


def _invoked_operation(step: Step) -> str | None:
    if step.next_action != InvokeAction.name:
        return None
    name = step.params.get("operation_name")
    return name if isinstance(name, str) else None


def _unsatisfiable_reference(activity: Activity) -> str | None:
    """Where the rest of the plan dereferences data that provably is not there — a binding an
    earlier step produced empty, or a ``$from`` naming an operation that already ran and came back
    empty — described as a defect for the replanning prompt; ``None`` when nothing is provably dead.
    Scans the active frame and then every suspended parent in resume order, since a sub-plan's
    caller runs later and reads the same flat `bindings` and the same history."""
    plan = activity.plan
    if plan is None:
        return None
    empty = {name for name, value in activity.bindings.items() if _is_empty(value)}
    # Captured before the forward scan starts discarding rewritten names: attribution asks what is
    # empty NOW, which is what the producer walk has to chase through.
    empty_now = frozenset(empty)
    refreshed: set[str] = set()
    frames = [(plan, activity.step_index)]
    frames += [(parent, index + 1) for parent, index, _ in reversed(activity.parent_frames)]
    for frame, start in frames:
        for index in range(start, len(frame.steps)):
            step = frame.steps[index]
            dead = sorted(_dereferenced_bindings(step) & empty)
            if dead:
                return (
                    f"step {index} ({step.next_action}) reads {', '.join(repr(n) for n in dead)}, "
                    "which an earlier step of this plan produced EMPTY — nothing matched it, so "
                    "that step cannot work and the plan cannot finish as written."
                    f"{_empty_binding_origin(activity, dead, empty_now)} {_REPLAN_HINT}"
                )
            for ref in _dereferenced_operations(step):
                spent = _spent_operation_read(ref, activity.history, refreshed)
                if spent is not None:
                    name, yielded = spent
                    return (
                        f"step {index} ({step.next_action}) reads a value out of {name!r}, which "
                        f"already ran in this run and {yielded} — so that step cannot work and the "
                        f"plan cannot finish as written. {_REPLAN_HINT}"
                    )
            out = step.params.get("out")
            if isinstance(out, str):
                empty.discard(out)  # rewritten here -> no longer provably empty further down
            invoked = _invoked_operation(step)
            if invoked is not None:
                refreshed.add(invoked)  # re-run here -> its old empty result proves nothing below
    return None
