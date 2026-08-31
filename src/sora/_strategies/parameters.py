"""Manual-schema parameter validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sora.manual import Manual


def _declared_param_names(manual: Manual | None, operation_name: str) -> list[str]:
    """The param names the operation's schema declares — for naming the alternatives in a defect
    message, so the replanning prompt says what IS accepted, not only what was not."""
    spec = manual.operation(operation_name) if manual is not None else None
    declared = spec.parameters.get("properties") if spec is not None else None
    return list(declared) if isinstance(declared, dict) else []


def _undeclared_params(
    manual: Manual | None, operation_name: str, params: dict[str, Any]
) -> list[str]:
    """Params the operation's schema does not declare — the ones an invoke would pass as unexpected
    keyword arguments. Empty when required-ness is unknowable (no manual, no spec, no declared
    ``properties``), the same structured-spec dependency ``_null_required_params`` has.

    Treats a schema without an explicit ``additionalProperties`` as *closed*, which inverts the
    JSON Schema default. Deliberate: these schemas are synthesized from real callables (an ARE
    ``AppTool`` signature, an MCP ``inputSchema``), so an undeclared key is a TypeError at the
    wire, not a tolerated extra. An adapter whose operation genuinely takes a free-form bag
    declares ``additionalProperties`` itself, and is then left alone."""
    spec = manual.operation(operation_name) if manual is not None else None
    if spec is None:
        return []
    if spec.parameters.get("additionalProperties", False) is not False:
        return []
    declared = spec.parameters.get("properties")
    if not isinstance(declared, dict):
        return []
    return sorted(key for key in params if key not in declared)


# What each JSON-Schema `type` accepts, as Python. `bool` is excluded from the numeric rows on
# purpose: it IS an `int` to Python, and a tool asking for a count that is handed True has been
# mis-planned, not satisfied. Types outside this table (unions, `null`, anything an adapter invents)
# are absent rather than empty — absent means "no opinion", which is how the check fails open.
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def _mistyped_params(
    manual: Manual | None,
    operation_name: str,
    params: dict[str, Any],
    raw: dict[str, Any],
) -> list[str]:
    """Params whose resolved value contradicts the type its schema declares, for a planner.

    The sibling of `_undeclared_params`, and it exists for the identical reason: the wire raises
    (ARE answers `Argument 'start_datetime' must be of type <class 'str'>, got <class 'float'>`), a
    failed op terminates the activity, and so a plan that is otherwise right dies on a conversion.
    It is a plan defect in the same sense too — the schema was in the catalog and the step wrote
    past it. The motivating run piped `get_calendar_event`'s epoch-float `start_datetime` straight
    into `get_calendar_events_from_to`, which declares a `YYYY-MM-DD HH:MM:SS` STRING; the run died
    there, mid-maintenance-window, on the first firing.

    Nothing is coerced. Turning 1729375200.0 into "1729375200.0" would satisfy the type and still be
    the wrong argument — the operation wants a formatted date, and only the planner can know that.
    Reporting it is the whole fix: the replan gets a defect naming the format and picks the field
    that already carries it.

    Fails open at every step where the schema is less than explicit — no manual, no spec, no
    declared `properties`, a `type` this table has no row for, a value of None (that is
    `_null_required_params`' question). A false positive here would refuse a call that WOULD have
    worked, which is strictly worse than the failure being guarded against.

    Names the reference a bad value came from, not just the parameter it landed in: the value is one
    hop from its producer and the producer is what has to change.
    """
    spec = manual.operation(operation_name) if manual is not None else None
    if spec is None:
        return []
    declared = spec.parameters.get("properties")
    if not isinstance(declared, dict):
        return []
    problems = []
    for key, value in params.items():
        schema = declared.get(key)
        if not isinstance(schema, dict) or value is None:
            continue
        declared_type = schema.get("type")
        # A union (`["string", "null"]`) is a list, and an unhashable one — so this reads the type
        # before looking it up, rather than after. A union means the schema has more than one
        # opinion, which is no single opinion to check against.
        accepted = _JSON_TYPES.get(declared_type) if isinstance(declared_type, str) else None
        if accepted is None:
            continue
        # `bool` is an `int` to Python, so the isinstance below would wave True through for a
        # declared number. Decide it first, in both directions.
        if isinstance(value, bool):
            if accepted == (bool,):
                continue
        elif declared_type == "integer" and isinstance(value, float) and value.is_integer():
            continue  # 3.0 after a JSON round trip still represents an integer; 3.14 does not
        elif isinstance(value, accepted):
            continue
        origin = raw.get(key)
        came_from = f" (from {origin!r})" if isinstance(origin, dict) else ""
        described = schema.get("description")
        wants = f", which wants {described}" if isinstance(described, str) and described else ""
        problems.append(
            f"{key!r} must be {declared_type} but got {type(value).__name__} "
            f"{value!r}{came_from}{wants}"
        )
    return problems


def _null_required_params(
    manual: Manual | None, operation_name: str, params: dict[str, Any]
) -> list[str]:
    """The operation's *required* params (per its schema) that resolve to null in ``params`` —
    either an explicit None or absent entirely (a required key the step never supplied). Empty when
    the schema is unavailable (no manual, the manual doesn't describe this op, or it declares no
    ``required``): required-ness is then unknowable, so the guard can't fire and binding goes on."""
    spec = manual.operation(operation_name) if manual is not None else None
    if spec is None:
        return []
    required = spec.parameters.get("required", [])
    return [key for key in required if params.get(key) is None]
