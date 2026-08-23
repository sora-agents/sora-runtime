"""In-process ARE ``WorkspaceAdapter`` + ``MessageTransport`` — talk to live ARE apps directly.

Unlike ``are_mcp`` (which reaches ARE over an MCP subprocess serving a *static* app snapshot), this
runs the ARE ``Environment`` **event loop** in the same process (a bg thread) so a scenario's
timeline actually fires — mid-run email injections, follow-up user messages, task delivery — and
bridges the two off-cycle event channels into S-ORA directly, as method calls on shared objects:

  * app state changes  -> ``state_changed`` Signal into the focused tool's ``signal_sink``
    (poll-on-observe: the tool re-reads ``app.get_state()`` each Observe and diffs — see
    ``_AreTool.observe``).  This is what MCP could not push off-request (ARE's MCP server only emits
    ``resource_updated`` from inside a write-tool request), so we go in-process instead.
  * ``AgentUserInterface`` USER messages  -> ``MessageTransport`` (``AreTransport`` over the AUI).

``AreSimulation`` owns the ``Environment``/scenario lifecycle and is the single object both seams
share (see the new ADR). The adapter's *workspace* owns start/stop (start on ``discover``, stop on
``close``), exactly as ``_McpWorkspace`` owns its subprocess. ``ARE`` (``are.simulation.*``) is an
optional dependency-group, so every import of it is lazy; the adapter/transport depend only
on a small duck-typed app/AUI interface (``app_name``/``get_tools``/``get_state`` and AUI
``get_last_unread_messages``/``send_message_to_user``/``send_message_to_agent``), which fakes
satisfy, so S-ORA-side logic stays testable without ARE (see ADR-0003).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
import typing
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, Union

from sora.manual import (
    Manual,
    ObservablePropertySpecification,
    OperationSpecification,
    SignalSpecification,
    merge_manuals,
)
from sora.perception import Message
from sora.types import Change, ObservableProperty, OperationAck, Signal

if TYPE_CHECKING:
    from sora.environment import Tool, Workspace, WorkspaceOrigin
    from sora.manual import ManualSource, ToolRecord, WorkspaceRecord

_T = TypeVar("_T")

log = logging.getLogger("sora.adapters.are_sim")

_AUI_APP = "AgentUserInterface"  # ARE's user-message app; routed via the transport, not as a tool

# ARE mutates app state on its own event-loop thread with no lock we can share (see AreSimulation),
# so a ``get_state()`` that iterates a dict the event loop is concurrently growing can raise
# "changed size during iteration". Mutation happens in sub-second bursts, so an immediate re-read
# sees a settled snapshot — retry a few times before giving up.
_STATE_READ_ATTEMPTS = 3

# How deep the change diff walks before reporting a subtree coarsely. ARE state nests app -> folders
# -> folder -> emails, so a handful of levels reaches the collections that actually matter; beyond
# that the extra precision is not worth walking a large structure on every observe.
_DIFF_MAX_DEPTH = 6


def _identities(value: Any) -> dict[str, Any] | None:
    """Read a container as {identity: item}, or None if it isn't one we can identify items in.

    A dict is already keyed. A list of dicts is keyed by each item's own id-ish field — which is
    what makes "this email appeared" expressible at all. A list of scalars has no identity to
    report, so it degrades to the coarse form rather than inventing positional ids that would
    change meaning whenever anything is inserted.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        keyed: dict[str, Any] = {}
        for item in value:
            if not isinstance(item, dict):
                return None
            ident = (
                item.get("id") or item.get("uid") or item.get("event_id") or item.get("email_id")
            )
            if not isinstance(ident, str | int):
                return None
            keyed[str(ident)] = item
        return keyed
    return None


def _diff_state(previous: Any, current: Any, path: str = "", depth: int = 0) -> list[Change]:
    """Locate what moved between two app-state snapshots, as identities rather than values.

    Recurses while both sides are containers whose items can be identified, so the reported path is
    as specific as the data allows — ``folders.INBOX.emails`` rather than the whole app. That
    specificity is the entire point: it is what lets a waiter tell an inbound message from the
    agent's own outbound one without any reasoning about self-causation, since they land in
    different folders.

    Degrades rather than fails at every step. An unidentifiable container, a type change, or the
    depth limit all produce a coarse ``Change`` naming the deepest path known to have moved —
    "something under here changed" — which consumers must accept.
    """
    if previous == current:
        return []
    if depth >= _DIFF_MAX_DEPTH:
        return [Change(path=path)]
    prev_items, curr_items = _identities(previous), _identities(current)
    if prev_items is None or curr_items is None:
        return [Change(path=path)]
    added = tuple(k for k in curr_items if k not in prev_items)
    removed = tuple(k for k in prev_items if k not in curr_items)
    shared = [k for k in curr_items if k in prev_items and curr_items[k] != prev_items[k]]
    changes: list[Change] = []
    if added or removed:
        changes.append(Change(path=path, added=added, removed=removed, updated=tuple(shared)))
    for key in shared:
        child = f"{path}.{key}" if path else key
        nested = _diff_state(prev_items[key], curr_items[key], child, depth + 1)
        # A leaf that changed value has no sub-structure to name; report the leaf itself as moved
        # so the path still points somewhere useful rather than vanishing from the summary.
        changes.extend(nested or [Change(path=child)])
    if not changes:
        # Both sides identifiable and same keys, but unequal — a value-only change somewhere we
        # could not localize. Report the level itself rather than nothing.
        changes.append(Change(path=path, updated=tuple(shared)))
    return changes


class Simulation(Protocol):
    """The runtime surface the adapter/transport use, decoupled from ARE so fakes satisfy it. The
    concrete ``AreSimulation`` implements it over a live ARE ``Environment``; a test fake implements
    it over plain app/AUI objects."""

    @property
    def aui(self) -> Any:  # the live AgentUserInterface app, or None
        ...

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def is_running(self) -> bool: ...  # scenario timeline still advancing (the eval done-signal)
    def apps(self) -> list[Any]: ...
    def run(self, fn: Callable[[], _T]) -> _T: ...  # serialize S-ORA's own concurrent app calls


@dataclass(frozen=True)
class ValidationOutcome:
    """``AreSimulation.validate()``'s result — decouples callers (``report.py``) from ARE's own
    ``ScenarioValidationResult`` shape, so a fake simulation in a test can produce one without
    depending on ARE. ``rationale`` is None when the scenario's ``validate()`` didn't supply one."""

    success: bool | None  # None = unscored: ARE's judge produced no verdict (a caller with no judge
    # attached decides that upstream — ARE's *base* validate() returns a bool, not None)
    rationale: str | None = None


class AreSimulation:
    """Owns the ARE ``Environment`` + scenario lifecycle — the shared object the in-process adapter
    and transport both reference. ``start`` runs the scenario's event loop on a background thread
    (``env.run(..., wait_for_end=False)``). The ``Lock`` serializes S-ORA's *own* concurrent app
    calls (e.g. an ``invoke`` on a worker thread vs an ``observe`` on the cycle thread); it does
    **not** — and cannot — serialize against ARE's event-loop thread, which mutates app state with
    no lock we can share, so reads tolerate a transient concurrent-modification error by retry (see
    ``_AreTool._read_state`` / ``_STATE_READ_ATTEMPTS``). The agent replies without blocking
    (``aui.wait_for_user_response = False``) — a follow-up user message arrives via the timeline and
    is picked up by ``AreTransport.receive``."""

    def __init__(self, scenario: Any, *, config: Any | None = None) -> None:
        self._scenario = scenario
        self._config = config
        self._env: Any | None = None
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        from are.simulation.environment import Environment, EnvironmentConfig

        if not getattr(self._scenario, "_initialized", False):
            self._scenario.initialize()
        # `Environment.run` copies `duration` and `time_increment_in_seconds` off the scenario,
        # but NOT `start_time` — that is read from the config alone
        # (`time_manager.reset(start_time=self.start_time)`), and `EnvironmentConfig` defaults it to
        # None, which the Environment reads as 0. A scenario therefore runs with its simulated clock
        # at the Unix epoch, counting real seconds up from 1970-01-01, while its data and its oracle
        # sit in the scenario's own year: `get_current_time` told an agent it was Thursday
        # 1 Jan 1970 for a scenario starting Tuesday 2024-10-15, so every date the agent derived
        # ("this upcoming Saturday") was computed against the wrong epoch — a silent wrong answer,
        # not an error. ARE's own ScenarioRunner sets it from the scenario; mirror that.
        # An explicit start_time on a caller-supplied config wins, and the config is copied rather
        # than mutated because it belongs to the caller.
        config = self._config or EnvironmentConfig()
        start_time = getattr(self._scenario, "start_time", None)
        if start_time and config.start_time is None:
            config = dataclasses.replace(config, start_time=start_time)
        self._env = Environment(config=config)
        # wait_for_end=False: registers apps, schedules the timeline, starts the event-loop thread,
        # and returns — the agent then drives its cycle against the live, ticking world.
        self._env.run(self._scenario, wait_for_end=False)
        aui = self.aui
        if aui is not None:
            aui.wait_for_user_response = False
        self._started = True

    def stop(self) -> None:
        if self._env is not None and self._started:
            self._env.stop()
        self._started = False

    def is_running(self) -> bool:
        """True while the scenario's event loop is still advancing (ARE
        ``EnvironmentState.RUNNING``). Flips to False when the timeline completes
        (``scenario.duration`` reached) or fails: ARE's ``_event_loop`` sets ``STOPPED``/``FAILED``
        on loop exit, *after* every scheduled user turn and the per-turn judge
        ``ConditionCheckEvent``s have fired. So an eval runner that waits for this to go False
        before calling ``validate()`` scores a fully-played-out scenario, riding through the idle
        gaps between turns that a quiet-window heuristic would exit on prematurely."""
        return self._env is not None and self._started and bool(self._env.is_running())

    def apps(self) -> list[Any]:
        return list(getattr(self._scenario, "apps", None) or [])

    def environment(self) -> Any:
        """The live ARE ``Environment`` (or None before ``start()``). Exposed for eval tooling that
        exports a completed run's trace — ARE's ``JsonScenarioExporter`` needs the Environment. It's
        deliberately *not* on the ``Simulation`` Protocol, which stays the minimal runtime surface
        the adapter/transport share; only concrete eval code touches this."""
        return self._env

    @property
    def aui(self) -> Any:
        return next((a for a in self.apps() if a.app_name() == _AUI_APP), None)

    def run(self, fn: Callable[[], _T]) -> _T:
        # Serializes S-ORA's own concurrent calls only; ARE's event-loop thread does not take this
        # lock (it's ARE-internal), so it does not guard app reads against that thread — see the
        # class docstring and _AreTool._read_state's retry.
        with self._lock:
            return fn()

    def validate(self) -> ValidationOutcome:
        """Oracle scoring: run the scenario's validators against the final environment state."""
        assert self._env is not None, "start() the simulation before validating"
        result = self._scenario.validate(self._env)
        # A judge/validator that *errored* reports success=None with an in-band exception (ARE puts
        # it on the result rather than raising). Re-raise it so a caller's try/except records the
        # run as an 'exception', not a silent unscored one — dropping it here would make a judge
        # crash indistinguishable from 'no judge attached'.
        if result.success is None and result.exception is not None:
            raise result.exception
        # Preserve None (unscored/vacuous) rather than coercing to False — a run with no judge
        # attached is distinct from a genuine FAIL, and the eval reporter surfaces that difference.
        return ValidationOutcome(success=result.success, rationale=result.rationale)


def load_scenario(ref: str) -> Any:
    """Resolve a scenario reference to an ARE ``Scenario`` — the "any ARE scenario" seam. ``ref`` is
    either a ``.json`` benchmark scenario path or a dotted path to a ``Scenario`` subclass (or a
    ready instance). Instances are returned as-is (``AreSimulation.start`` initializes them)."""
    if ref.endswith(".json"):
        from are.simulation.benchmark.scenario_loader import load_scenario as _are_load

        scenario, _ = _are_load(
            Path(ref).read_text(encoding="utf-8"), ref, load_completed_events=False
        )
        if scenario is None:
            raise ValueError(f"failed to load ARE scenario from {ref!r}")
        return scenario
    from sora.bootstrap import import_object  # lazy: avoid a bootstrap<->are_sim import cycle

    obj = import_object(ref)
    return obj() if isinstance(obj, type) else obj


def attach_judge(
    scenario: Any,
    *,
    model: str | None = None,
    provider: str | None = None,
    endpoint: str | None = None,
    offline_validation: bool = False,
) -> None:
    """Attach ARE's GraphPerEvent judge so ``AreSimulation.validate()`` scores a benchmark scenario
    against its oracle event graph (the Gaia2 scoring path) instead of the ``success=None`` no-op.

    Runs ARE's ``preprocess_scenario``, which itself executes the scenario's ``OracleEvent``s in
    oracle mode to populate ``oracle_run_event_log`` and then sets ``scenario.judge`` /
    ``scenario.validate``. Call *after* ``load_scenario`` and *before* ``AreSimulation.start()``.
    Only meaningful for scenarios that carry ``OracleEvent``s (Gaia2 JSON does). ``model=None`` uses
    ARE's default judge model. This call is itself offline — oracle-mode replay of the OracleEvents
    is deterministic and modelless — but the judge model is *not* contacted only at ``validate()``:
    under online validation (``offline_validation=False``, the default) ARE installs
    ``judge.trigger_condition`` as each turn's release gate, so the judge is also called mid-run at
    every turn boundary, and a verdict of "turn failed" stops the environment and withholds the
    remaining turns. Use ``initialize_turns`` when the later turns are wanted without that gate."""
    from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig
    from are.simulation.scenarios.scenario_imported_from_json.utils import preprocess_scenario
    from are.simulation.validation.configs import GraphPerEventJudgeConfig, create_judge_engine

    engine_config = (
        LLMEngineConfig(model_name=model, provider=provider, endpoint=endpoint)
        if model is not None
        else None  # None -> ARE's default judge model/provider
    )
    preprocess_scenario(
        scenario,
        judge_config=GraphPerEventJudgeConfig(engine=create_judge_engine(engine_config)),
        offline_validation=offline_validation,
    )


def initialize_turns(scenario: Any) -> None:
    """Wire a multi-turn benchmark scenario's turn triggers *without* attaching a judge, so every
    turn is delivered unconditionally. The run stays unscored (``scenario.judge`` is never set, so
    ``AreSimulation.validate()`` keeps returning the ``success=None`` no-op) — this buys the later
    turns, not a verdict.

    Why this exists as its own entry point: a Gaia2 scenario's later turns do not fire on their own.
    Their environment events hang off ``OracleEvent``s, which an agent-mode environment ignores
    without releasing successors, so a scenario run straight from ``load_scenario`` silently stops
    after turn 1. ARE re-anchors those events onto a per-turn ``ConditionCheckEvent`` only inside
    ``preprocess_scenario`` — which ``attach_judge`` reaches, but only by also making the judge the
    gate that decides whether each turn is released at all (see ``attach_judge``). Passing no judge
    takes ARE's own dummy trigger instead: it always releases the next turn, so the later turns
    arrive whether or not the agent got the earlier ones right.

    That is what development wants. Exercising behaviour that only a *later* turn provokes — the
    agent revising a plan when new information contradicts it — otherwise requires paying for a
    judge good enough that its verdict on turn 1 can be trusted not to end the run first, which
    couples the behaviour under test to judge quality for no benefit. Call *after* ``load_scenario``
    and *before* ``AreSimulation.start()``, and not together with ``attach_judge`` (either one
    initializes the turns; the second call is a no-op and the judge would still be the gate)."""
    from are.simulation.scenarios.scenario_imported_from_json.utils import preprocess_scenario

    preprocess_scenario(scenario, judge_config=None)


# -- app -> S-ORA usage-interface extraction ------------------------------------------------------

_JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}

_ARE_SEP = "__"  # ARE namespaces an app's tools as <App>__<operation> (as the flat MCP names do)


def _op_name(app: Any, app_tool: Any) -> str:
    """The bare operation name, stripping ARE's ``<App>__`` prefix so ops read as ``list_emails``
    (matching the ``are_mcp`` adapter and the hand-authored manuals), not ``EmailClientApp__…``."""
    prefix = f"{app.app_name()}{_ARE_SEP}"
    name: str = app_tool.name
    return name.removeprefix(prefix)


def _json_atom(t: str) -> dict[str, Any]:
    """One non-union ARE type string -> a JSON-Schema fragment. Recursive on ``list[...]`` so the
    item type is faithful too (``list[str]`` -> array of string, not array of anything)."""
    if t.startswith("list[") and t.endswith("]"):
        return {"type": "array", "items": _json_atom(t[len("list[") : -1].strip())}
    if t == "list":
        return {"type": "array"}
    if t == "dict" or t.startswith("dict["):
        return {"type": "object"}
    return {"type": _JSON_TYPES.get(t, "string")}


def _json_type(arg_type: Any) -> dict[str, Any]:
    """Map an ARE ``AppTool`` arg-type *string* (``str``, ``int``, ``list[str]``,
    ``list[str] | None``, ``int | float | None``, ``dict[str, Any]``, ...) to a JSON-Schema type
    fragment. Every arg the grounding model fills has to be represented faithfully: collapsing
    ``list[str]`` to ``string`` is what led the model to fill ``attendees`` with ``"Alice, Bob"`` —
    which ARE's own runtime type-check then rejects (``must be of type list[str] | None, got str``).
    Unions are split on ``|`` (``None`` dropped): a lone member maps directly; an all-numeric union
    (``int | float``) becomes JSON ``number`` (which admits both); any other heterogeneous union
    has no single faithful JSON type, so it falls back to ``string``. (Assumes ARE's flat vocabulary
    — no ``|`` nested inside brackets, which its apps never emit.)"""
    if not isinstance(arg_type, str):
        return {"type": "string"}
    members = [m.strip() for m in arg_type.split("|")]
    members = [m for m in members if m and m != "None"]
    if len(members) == 1:
        return _json_atom(members[0])
    if {_json_atom(m).get("type") for m in members} <= {"integer", "number"}:
        return {"type": "number"}
    return {"type": "string"}


def _params_schema(app_tool: Any) -> dict[str, Any]:
    """A JSON-Schema object for an ARE ``AppTool``'s args (same shape the ARE MCP server uses)."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for arg in app_tool.args:
        properties[arg.name] = {**_json_type(arg.arg_type), "description": arg.description or ""}
        if not getattr(arg, "has_default", False):
            required.append(arg.name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# How deep to expand a returned record's nested records into a JSON-Schema shape. The deepest real
# ARE shape is four levels: an envelope whose payload is a list of records, one of whose own fields
# is a list (CalendarEventsResult -> events: list[CalendarEvent] -> attendees: list[str]; likewise
# ReturnedEmails -> emails: list[Email] -> recipients: list[str]). The cap was 3, which clipped the
# innermost list's element type on exactly those ops — and `attendees` is precisely the field a plan
# needs to path into, so the elision was not cosmetic. 4 expands every real shape and still bounds
# the walk: the cap is defensive against a foreign/future annotation that is *self*-referential
# (`children: list[Node]`, resolved back to the live class by ``get_type_hints``), which would
# otherwise recurse into a ``RecursionError`` — one more level costs nothing there. Raise this only
# on the same evidence: a real returned shape that a plan must path into and cannot. When the cap
# does elide a nested shape, ``_type_to_schema`` emits a DEBUG log — a lead when a planner ``$from``
# path won't resolve because the shape below the cap was dropped.
_MAX_RETURN_DEPTH = 4

_PRIMITIVE_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _log_depth_cap(tp: Any) -> None:
    """A returned type expandable past ``_MAX_RETURN_DEPTH`` was clipped to a leaf. Benign — an
    over-deep or (unexpectedly) self-referential annotation is bounded rather than crashing — so
    this is DEBUG, not a warning: no ARE app record reaches the cap today, and the record shape
    stays valid, just shallower than the source type."""
    log.debug(
        "return-type introspection hit depth cap %d at %r; nested shape below it is elided "
        "(a planner $from path can't index past this point)",
        _MAX_RETURN_DEPTH,
        tp,
    )


def _record_fields(tp: Any) -> dict[str, Any] | None:
    """``{field name: annotation}`` when ``tp`` is a *record* — a dataclass or a ``TypedDict`` —
    else ``None``. Both spellings carry the same thing for a planner's purposes (named fields to
    path a ``$from`` into) and differ only in how the field list is reached, so they share one
    branch in ``_type_to_schema``. ARE uses both, and the split is not incidental: plain records are
    dataclasses (``Email``, ``CalendarEvent``) while every *paginated envelope* is a TypedDict
    (``CalendarEventsResult``, ``ProductListResult``). Recognizing only dataclasses therefore left
    exactly the windowed list reads — the ops whose payload a planner most needs to path into —
    declaring a bare ``string``. Annotations are resolved where possible, since ARE's app modules
    use ``from __future__ import annotations``; a dataclass falls back to its raw (string) field
    annotation, which the string mapper still reads."""
    if not isinstance(tp, type):
        return None
    is_record = dataclasses.is_dataclass(tp) or typing.is_typeddict(tp)
    if not is_record:
        return None
    try:
        hints = typing.get_type_hints(tp)
    except Exception:
        hints = {}
    if dataclasses.is_dataclass(tp):
        return {f.name: hints.get(f.name, f.type) for f in dataclasses.fields(tp)}
    # A TypedDict's own ``__annotations__`` omits inherited keys; ``get_type_hints`` flattens the
    # bases, so it is the primary and the raw dict only the fallback.
    return hints or dict(getattr(tp, "__annotations__", {}))


def _type_to_schema(tp: Any, depth: int = 0) -> dict[str, Any]:
    """Best-effort JSON-Schema fragment for a Python return annotation — deep enough that a planner
    can author a ``$from`` path into it (list nesting + a record's field names), no deeper. Handles
    ``list[X]`` (and tuple/set), ``X | None`` unions, records (dataclass or ``TypedDict``, one level
    of fields — see ``_record_fields``), and the JSON primitives; anything else (or past the depth
    cap) degrades to a bare ``string`` rather than raising. A *string* annotation (an unresolved
    ``from __future__ import annotations`` hint) reuses the union-aware arg-type string mapper (so
    ``'int | None'`` maps to integer, not the ``string`` a single-atom mapper would give)."""
    if isinstance(tp, str):
        return _json_type(tp)
    origin = typing.get_origin(tp)
    if origin in (Union, UnionType):
        members = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(members) == 1:
            return _type_to_schema(members[0], depth)
        # An all-numeric union (``int | float``) has a single faithful JSON type; anything else
        # heterogeneous doesn't — same rule the arg-type mapper (`_json_type`) applies.
        if members and {_type_to_schema(m, depth).get("type") for m in members} <= {
            "integer",
            "number",
        }:
            return {"type": "number"}
        return {"type": "string"}
    if origin in (list, set, frozenset) or tp in (list, set, frozenset):
        args = typing.get_args(tp)
        schema: dict[str, Any] = {"type": "array"}
        if args and args[0] is not ...:
            if depth < _MAX_RETURN_DEPTH:
                schema["items"] = _type_to_schema(args[0], depth + 1)
            else:
                _log_depth_cap(tp)
        return schema
    if origin is tuple or tp is tuple:
        return {"type": "array"}
    fields = _record_fields(tp)
    if fields is not None:
        if depth >= _MAX_RETURN_DEPTH:
            _log_depth_cap(tp)
            return {"type": "string"}
        properties = {
            name: _type_to_schema(annotation, depth + 1) for name, annotation in fields.items()
        }
        return {"type": "object", "properties": properties}
    if tp in _PRIMITIVE_JSON:
        return {"type": _PRIMITIVE_JSON[tp]}
    if tp is dict or origin is dict:
        return {"type": "object"}
    return {"type": "string"}


def _resolved_return_type(app_tool: Any) -> Any:
    """The operation's return annotation as a *resolved* type, not a string. ARE's app modules use
    ``from __future__ import annotations``, so ``AppTool.return_type`` is the raw annotation string
    (``'list[Email]'``) — useless for field-name introspection. Resolve it against the underlying
    function's own module globals (where ``Email``/``ReturnedEmails`` are defined) via
    ``get_type_hints``; fall back to the raw ``return_type`` (a real type in a fake, or a string we
    can only shallow-map) when there's no function or resolution fails."""
    func = getattr(app_tool, "function", None)
    if func is not None:
        try:
            resolved = typing.get_type_hints(func).get("return")
        except Exception:
            resolved = None
        if resolved is not None:
            return resolved
    return getattr(app_tool, "return_type", None)


def _returns_schema(app_tool: Any) -> dict[str, Any] | None:
    """The operation's declared result shape, or None. Synthesized from the resolved return type
    (see ``_resolved_return_type``) and seeded with the ``return_description`` prose so a planner
    sees both the shape to index a ``$from`` path into and what it means. A ``-> None`` op (resolved
    to ``NoneType``) has no result to reference, so it declares no shape — otherwise it would render
    a fictitious leaf a planner could bind an empty-path ``$from`` against."""
    tp = _resolved_return_type(app_tool)
    if tp is None or tp is type(None):
        return None
    schema = _type_to_schema(tp)
    description = getattr(app_tool, "return_description", None)
    return {**schema, "description": description} if description else schema


def _side_effecting(app_tool: Any) -> bool | None:
    """ARE's ``AppTool.write_operation`` (a bool set by the ``@app_tool`` decorator) mapped onto
    ``OperationSpecification.side_effecting`` — the native read/write signal, so no name heuristic.
    Absent/non-bool -> ``None`` (unknown; the checkpoint treats it as a write)."""
    write_operation = getattr(app_tool, "write_operation", None)
    return write_operation if isinstance(write_operation, bool) else None


def _operation_specs(app: Any) -> list[OperationSpecification]:
    return [
        OperationSpecification(
            name=_op_name(app, at),
            description=getattr(at, "function_description", None) or "",
            parameters=_params_schema(at),
            returns=_returns_schema(at),
            side_effecting=_side_effecting(at),
        )
        for at in app.get_tools()
    ]


def _to_serializable(value: Any) -> Any:
    """Make an app op's result JSON-friendly so a later step can ground on its fields (e.g. an
    ``email_id`` from ``list_emails``). Falls back to the raw value when ARE isn't importable (fakes
    already return plain data)."""
    try:
        from are.simulation.utils import make_serializable

        return make_serializable(value)
    except Exception:
        return value


# -- Tool / Workspace / Adapter -------------------------------------------------------------------


class _AreTool:
    """One live tool over one ARE app. ``invoke`` calls the app op (off-thread, lock-guarded
    against the Environment thread); ``observe`` polls ``get_state`` and emits ``state_changed`` on
    diff into the sink handed at ``focus`` — the in-process analogue of MCP's resource-update push,
    tied to the cycle's own Observe cadence so it's deterministic."""

    def __init__(
        self, *, tool_id: str, manual: Manual, app: Any, ops: dict[str, Any], simulation: Simulation
    ) -> None:
        self.id = tool_id
        self.manual = manual
        self.address: str | None = None
        self._app = app
        self._ops = ops
        self._sim = simulation
        self._sink: Any | None = None
        self._state: Any = None  # last observed state, for the diff

    async def invoke(self, operation_name: str, **params: Any) -> OperationAck:
        app_tool = self._ops.get(operation_name)
        if app_tool is None:
            return OperationAck(ok=False, result=f"unknown operation {operation_name!r}")
        try:
            result = await asyncio.to_thread(self._sim.run, lambda: app_tool(**params))
            return OperationAck(ok=True, result=_to_serializable(result))
        except Exception as exc:  # an app op raising is a failed ack, not a runtime crash
            return OperationAck(ok=False, result=str(exc))

    async def focus(self, sink: Any) -> None:
        self._sink = sink
        self._state = self._read_state()

    async def unfocus(self) -> None:
        self._sink = None
        self._state = None

    def observe(self) -> list[ObservableProperty]:
        # The new state is recorded *before* the push, not after. A pushed signal is screened
        # synchronously by the InterruptPolicy, which runs upstream of the once-per-cycle property
        # snapshot — so a push-time consumer has to read the current value off this tool, and this
        # tool must therefore already hold it. Assigning first also makes a re-entrant observe()
        # from inside that screen a no-op (state == self._state -> no second push) rather than a
        # recursion.
        state = self._read_state()
        previous = self._state
        changed = state != previous
        self._state = state
        if self._sink is not None and changed:
            # Thin: the event, not the state. The snapshot is published as the `state` observable
            # property below; duplicating it into the signal would only reproduce it in every
            # prompt that renders wm.signals. But thin is not contentless — the payload also names
            # WHERE it moved, which is the one thing a replace-by-key snapshot cannot express and
            # so duplicates nothing. This diff is nearly free: the comparison above already walks
            # both snapshots to decide `changed` at all, and today throws the result away.
            self._sink.push(
                self.id,
                Signal(
                    "state_changed",
                    {
                        "app": self._app.app_name(),
                        "changes": _diff_state(previous, state),
                    },
                ),
            )
        # `self._state`, not the local: the push screen re-enters observe(), and if ARE's thread
        # mutated state in between, that nested call already advanced `self._state` past the local.
        # Returning the local would snapshot the pre-change world into working memory for one tick.
        return [ObservableProperty(name="state", value=self._state)]

    def _read_state(self) -> Any:
        # ARE's event-loop thread can mutate app state mid-read (no shared lock), so a get_state()
        # iterating a dict it's concurrently growing may raise RuntimeError. Retry the snapshot —
        # mutation is bursty, so an immediate re-read almost always settles (_STATE_READ_ATTEMPTS).
        last: RuntimeError | None = None
        for _ in range(_STATE_READ_ATTEMPTS):
            try:
                # Same normalization invoke() applies to an op result. ARE builds app state with
                # asdict(), which leaves Enum members in place: a mechanical `eq` against "Female"
                # then fails against <Gender.FEMALE: 'Female'>, and the value defeats JSON rendering
                # in prompts. Observed state is ground for the same comparisons an op result is, so
                # it has to arrive in the same shape.
                return _to_serializable(self._sim.run(self._app.get_state))
            except RuntimeError as exc:  # concurrent modification by the ARE event-loop thread
                last = exc
        assert last is not None
        raise last


class _AreWorkspace:
    def __init__(
        self, ws_id: str, origin: WorkspaceOrigin, tools: list[Tool], simulation: Simulation
    ) -> None:
        self.id = ws_id
        self.origin = origin
        self._tools = tools
        self._sim = simulation

    def tools(self) -> list[Tool]:
        return self._tools

    async def close(self) -> None:
        await asyncio.to_thread(self._sim.stop)  # stops the Environment event-loop thread


class AreInProcessWorkspaceAdapter:
    """Imports the live ARE apps of a running ``AreSimulation`` as S-ORA tools (one per app, its
    ops from ``app.get_tools()``, plus a ``state`` observable + ``state_changed`` signal). The
    ``AgentUserInterface`` app is deliberately excluded — user messages are a transport concern
    (``AreTransport``), not a tool. The workspace owns the Environment lifecycle: ``discover``
    it, ``close`` stops it."""

    name = "are-sim"  # matches WorkspaceOrigin.adapter

    def __init__(
        self,
        *,
        workspace_id: str,
        origin: WorkspaceOrigin,
        simulation: Simulation,
        manual_source: ManualSource | None = None,
    ) -> None:
        self._workspace_id = workspace_id
        self._origin = origin
        self._sim = simulation
        self._manual_source = manual_source

    async def discover(self) -> list[Workspace]:
        await asyncio.to_thread(self._sim.start)
        tools = [await self._build_tool(app) for app in self._tool_apps()]
        return [_AreWorkspace(self._workspace_id, self._origin, tools, self._sim)]

    async def connect(
        self,
        workspace_record: WorkspaceRecord,
        tool_records: list[ToolRecord],
        manuals: dict[str, Manual],
    ) -> Workspace:
        # In-process apps live in the current simulation, so rebuild directly from them (no snapshot
        # reconstruction needed — the process holds the live objects).
        await asyncio.to_thread(self._sim.start)
        by_name = {app.app_name(): app for app in self._tool_apps()}
        tools: list[Tool] = []
        for record in tool_records:
            app = by_name.get(record.manual_id)
            if app is not None:
                tools.append(self._make_tool(record.id, app, manuals[record.manual_id]))
        return _AreWorkspace(workspace_record.id, workspace_record.origin, tools, self._sim)

    def _tool_apps(self) -> list[Any]:
        return [a for a in self._sim.apps() if a.app_name() != _AUI_APP]

    async def _build_tool(self, app: Any) -> Tool:
        manual = await self._paired_manual(app.app_name(), self._synth_manual(app))
        return self._make_tool(self._derive_tool_id(app.app_name()), app, manual)

    def _make_tool(self, tool_id: str, app: Any, manual: Manual) -> Tool:
        ops = {_op_name(app, at): at for at in app.get_tools()}
        return _AreTool(tool_id=tool_id, manual=manual, app=app, ops=ops, simulation=self._sim)

    def _synth_manual(self, app: Any) -> Manual:
        name = app.app_name()
        return Manual(
            id=name,
            metadata={"source": self.name, "app": name},
            description=f"ARE app {name}, in-process",
            observable_properties=[
                ObservablePropertySpecification(name="state", description="", schema={})
            ],
            signals=[SignalSpecification(name="state_changed", description="", schema={})],
            operations=_operation_specs(app),
            raw_text=None,
        )

    async def _paired_manual(self, manual_id: str, adapter_manual: Manual) -> Manual:
        if self._manual_source is None:
            return adapter_manual
        authored = await self._manual_source.get(manual_id)
        return adapter_manual if authored is None else merge_manuals(adapter_manual, authored)

    def _derive_tool_id(self, seed: str) -> str:
        # ADR-0014: globally unique, adapter-derived, deterministic (origin address + app name).
        return f"{self._origin.address}/{seed}"


class AreTransport:
    """``MessageTransport`` over the scenario's ``AgentUserInterface``. ``receive`` drains unread
    USER messages (the task + timeline follow-ups) as ``Message``s; ``send`` posts the agent's reply
    via ``send_message_to_user``; ``submit`` injects an ad hoc user message (a typed CLI line, a
    ``/stop`` resume) via ``send_message_to_agent``, which surfaces on the next ``receive`` drain
    like any timeline message. Shares the running ``AreSimulation`` with the adapter."""

    def __init__(self, simulation: Simulation) -> None:
        self._sim = simulation
        # Mirrors InProcessTransport.sent (an outbound log for tests/inspection) so a presentation
        # layer like TerminalSession can stream a reply the same way regardless of transport kind.
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def submit(self, message: Message) -> None:
        # The user side of the AUI: a message *from* the user *to* the agent. Routed through
        # sim.run (like send) so the write is registered on the Environment's own event loop, then
        # picked up by the next receive() drain — same path as the scenario's timeline messages, so
        # nothing downstream distinguishes an ad hoc line from a scripted one.
        aui = self._sim.aui
        if aui is None:
            return
        content = message.content
        text = content.get("text", "") if isinstance(content, dict) else str(content)
        self._sim.run(lambda: aui.send_message_to_agent(text))

    async def send(self, to: str, content: dict[str, Any]) -> None:
        aui = self._sim.aui
        if aui is None:
            return
        # Record only what was actually delivered — a presentation layer like TerminalSession
        # polls `.sent` and streams it as the agent's reply, so logging content that never
        # reached the AUI would show a message the user never actually got.
        self.sent.append((to, content))
        text = content.get("text", "") if isinstance(content, dict) else str(content)
        await asyncio.to_thread(self._sim.run, lambda: aui.send_message_to_user(text))

    def receive(self) -> AsyncIterator[Message]:
        async def _drain() -> AsyncIterator[Message]:
            aui = self._sim.aui
            if aui is None:
                return
            for m in self._sim.run(aui.get_last_unread_messages):
                # ARE message timestamps are sim-relative time; the t0 task message legitimately
                # has timestamp 0.0, so distinguish an absent timestamp (None) from a falsy 0.0
                # rather than `... or time.time()`, which would stamp wall-clock over a real 0.0.
                ts = getattr(m, "timestamp", None)
                yield Message(
                    sender="user",
                    content={"text": m.content},
                    received_at=time.time() if ts is None else ts,
                )

        return _drain()
