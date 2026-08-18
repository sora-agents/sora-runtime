"""Working, semantic, procedural, and episodic memory modules."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import quote

# Imported at runtime (not just for typing): SemanticMemory reconstructs these dataclasses from
# the plain dicts the backend hands back. manual.py / environment.py only import their sora deps
# under TYPE_CHECKING, so importing them here introduces no cycle.
from sora.action import InvokeAction, invoke_step
from sora.environment import WorkspaceOrigin
from sora.manual import (
    Manual,
    ManualSection,
    ObservablePropertySpecification,
    OperationSpecification,
    SignalSpecification,
    ToolRecord,
    WorkspaceRecord,
)
from sora.types import CompletedOperation, Plan, Step

if TYPE_CHECKING:
    from sora.activity import Activity
    from sora.environment import EnvironmentView, Tool
    from sora.llm import LLMClient
    from sora.perception import Message, Percept

log = logging.getLogger("sora.memory")


class MemoryBackend(Protocol):  # pluggable: file, DB, vector store
    async def get(self, key: str) -> Any: ...

    async def put(self, key: str, value: Any) -> None: ...

    async def query(self, **filters: Any) -> list[Any]:
        """Every stored value matching all `filters`, ordered most-relevant-first with ties broken
        deterministically — callers may treat `result[0]` as the single best match and the order as
        stable across identical calls. Ranking backends (a vector store) order by relevance;
        non-ranking ones (exact-match file storage) treat all matches as equally relevant and fall
        back to a stable key order."""


class FileMemoryBackend:
    """The default persistent MemoryBackend: one JSON file per key under a root directory.

    Deals only in JSON-serializable values — the memory modules (semantic/procedural/episodic)
    convert their dataclasses to/from plain dict/list/scalar before touching this. Keeping the
    backend generic is what makes a database/vector-store backend a true drop-in: it never learns
    about sora's specific types.

    Reading re-parses from disk, so a returned value is always a fresh copy — a caller can mutate
    it without corrupting the store (unlike an in-memory dict backend that hands back live refs).
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _path(self, key: str) -> Path:
        # quote(safe="") encodes '/', ':', etc. so URI / <App>__<op> ids map to safe filenames.
        return self._root / f"{quote(key, safe='')}.json"

    async def get(self, key: str) -> Any:
        return await asyncio.to_thread(self._read, self._path(key))

    async def put(self, key: str, value: Any) -> None:
        await asyncio.to_thread(self._write, key, value)

    async def query(self, **filters: Any) -> list[Any]:
        return await asyncio.to_thread(self._scan, filters)

    @staticmethod
    def _read(path: Path) -> Any:
        try:
            with path.open(encoding="utf-8") as f:
                return json.load(f)["value"]
        except FileNotFoundError:
            return None

    def _write(self, key: str, value: Any) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        # Atomic: write a temp file in the same dir, then rename over the target — a crash
        # mid-write never leaves a half-written .json that a later query() would choke on.
        fd, tmp = tempfile.mkstemp(dir=self._root, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"key": key, "value": value}, f)  # self-describing envelope
            os.replace(tmp, self._path(key))
        except BaseException:
            os.unlink(tmp)
            raise

    def _scan(self, filters: dict[str, Any]) -> list[Any]:
        if not self._root.exists():
            return []
        results = []
        # Exact-equality filters give no relevance ranking, so honor query()'s deterministic-
        # tiebreak clause with a stable on-disk-key order; the *.json glob excludes *.tmp files.
        for path in sorted(self._root.glob("*.json")):
            value = self._read(path)
            if not filters or (
                isinstance(value, dict) and all(value.get(k) == v for k, v in filters.items())
            ):
                results.append(value)
        return results


@dataclass
class WorkingMemory:  # transient, in-process, fast
    # Read-only view of the live joined workspaces/tools: the agent reasons over what it's currently
    # connected to; the durable WorkspaceRecord/ToolRecord knowledge stays in SemanticMemory. The
    # concrete instance behind this is the shared EnvironmentRegistry (mutable on DecisionCycle).
    registry: EnvironmentView
    activities: dict[str, Activity] = field(default_factory=dict)
    # Environment stimuli, stored by their opposite lifecycles (see ADR-0012's split rationale).
    # properties: persistent, re-observed state — a replace-by-(source, name) snapshot, one entry
    # per property, last value wins (keyed so re-observing overwrites in place). signals: transient,
    # fire-and-forget events — an append log. The same signal name can appear more than once, each a
    # distinct occurence (not a repeat of the same one). A signal is never removed just because it
    # satisfied some wait (it stays visible to any other blocked activity or strategy reading it
    # directly) — the only eviction is the retention cap bounding orphan growth.
    properties: dict[tuple[str, str], Percept] = field(default_factory=dict)
    signals: list[Percept] = field(default_factory=list)
    # inbound agent-to-agent communication — kept distinct
    messages: list[Message] = field(default_factory=list)
    # Count of messages already routed (turned into an activity goal by Situate, or claimed as
    # reconsideration input by a resume) — a consumed-cursor over the append-only log so each
    # message drives activity-creation at most once and a later tick never re-scans the whole log.
    # Storage stays uncapped (no eviction), so this index is always valid; bounding `messages` with
    # a retention cap is deferred (front-eviction would have to adjust the cursor).
    messages_cursor: int = 0
    focused_tools: dict[str, Tool] = field(default_factory=dict)
    # manuals pulled from SemanticMemory by _load_ (removed by _unload_) — distinct from
    # focused_tools: focusing a tool is an external action, loading its manual is internal.
    loaded_manuals: dict[str, Manual] = field(default_factory=dict)

    def drop_properties(self, keep: Callable[[str], bool]) -> None:
        """Prune `properties` in place to entries whose tool id satisfies `keep` — the shared
        mechanism behind `_unfocus_` (drop one tool) and `_filter_` (keep only the relevant set)."""
        for key in [k for k in self.properties if not keep(k[0])]:
            del self.properties[key]


# Discriminators for the three record kinds sharing one backend. They serve double duty: as a
# storage-key prefix (so the three independent id-spaces can't clobber each other's files) and as
# a stored `kind` field (so a query() lists just one kind — see FileMemoryBackend.query).
_MANUAL = "manual"
_WORKSPACE_RECORD = "workspace_record"
_TOOL_RECORD = "tool_record"


class SemanticMemory:  # knowledge about the world: tool types, workspaces, instances
    """Durable store for manuals and workspace/tool records. Owns the dataclass<->dict
    (de)serialization so the backend stays a generic key->JSON store: it converts to plain
    dicts on the way in and rebuilds typed instances on the way out."""

    def __init__(self, backend: MemoryBackend) -> None:
        self._backend = backend

    async def retrieve_manual(self, manual_id: str) -> Manual | None:
        value = await self._backend.get(f"{_MANUAL}:{manual_id}")
        return None if value is None else _manual_from_dict(value)

    async def store_manual(self, manual: Manual) -> None:
        await self._backend.put(f"{_MANUAL}:{manual.id}", {"kind": _MANUAL, **asdict(manual)})

    async def retrieve_workspace_record(self, workspace_id: str) -> WorkspaceRecord | None:
        value = await self._backend.get(f"{_WORKSPACE_RECORD}:{workspace_id}")
        return None if value is None else _workspace_record_from_dict(value)

    async def store_workspace_record(self, record: WorkspaceRecord) -> None:
        key = f"{_WORKSPACE_RECORD}:{record.id}"
        await self._backend.put(key, {"kind": _WORKSPACE_RECORD, **asdict(record)})

    async def list_workspace_records(self) -> list[WorkspaceRecord]:
        values = await self._backend.query(kind=_WORKSPACE_RECORD)
        return [_workspace_record_from_dict(v) for v in values]

    async def retrieve_tool_record(self, tool_id: str) -> ToolRecord | None:
        value = await self._backend.get(f"{_TOOL_RECORD}:{tool_id}")
        return None if value is None else _tool_record_from_dict(value)

    async def store_tool_record(self, record: ToolRecord) -> None:
        key = f"{_TOOL_RECORD}:{record.id}"
        await self._backend.put(key, {"kind": _TOOL_RECORD, **asdict(record)})

    async def list_tool_records(self) -> list[ToolRecord]:  # reconstitute known instances at boot
        values = await self._backend.query(kind=_TOOL_RECORD)
        return [_tool_record_from_dict(v) for v in values]


# --- deserialization: rebuild typed instances from the plain dicts the backend returns ---------
# asdict() flattens nested dataclasses to dicts on the way in; these undo exactly that, dropping the
# stored `kind` discriminator (which isn't a dataclass field). A fresh instance per call is what
# gives callers copy isolation even for the mutable lists inside a (frozen) Manual.


def _manual_from_dict(d: dict[str, Any]) -> Manual:
    return Manual(
        id=d["id"],
        metadata=d["metadata"],
        description=d["description"],
        observable_properties=[
            ObservablePropertySpecification(**p) for p in d["observable_properties"]
        ],
        signals=[SignalSpecification(**s) for s in d["signals"]],
        operations=[OperationSpecification(**o) for o in d["operations"]],
        raw_text=d.get("raw_text"),
    )


def _workspace_record_from_dict(d: dict[str, Any]) -> WorkspaceRecord:
    return WorkspaceRecord(
        id=d["id"],
        origin=WorkspaceOrigin(**d["origin"]),
        discovered_at=d["discovered_at"],
        last_seen_at=d["last_seen_at"],
    )


def _tool_record_from_dict(d: dict[str, Any]) -> ToolRecord:
    return ToolRecord(
        id=d["id"],
        manual_id=d["manual_id"],
        workspace_id=d["workspace_id"],
        address=d["address"],
        discovered_at=d["discovered_at"],
        last_seen_at=d["last_seen_at"],
    )


# --- procedural inference: prompt the LLM, convert its JSON answer into Plan/Step ----------------
# The model is asked for a strict JSON object; infer() turns it into the runtime's own vocabulary.
# This is the anti-corruption boundary: no provider/message shape leaks past here into Plan/Step.
# The *prompt* is a pluggable `PlanPrompt` (customize planning content without subclassing or moving
# planning into a ReasonStrategy); the *response contract* stays fixed — a custom prompt must still
# yield the {"steps": [...]} JSON that `_parse_plan_steps` reads.

PLAN_SYSTEM_PROMPT = (
    "You are the planning component of an autonomous agent runtime. Given a goal and the tools "
    "available to the agent, produce a short, ordered plan of concrete steps that achieves the "
    "goal using only the listed tools and operations.\n"
    'Respond with ONLY a JSON object of the form {"steps": [ ... ]} and nothing else — no prose, '
    "no markdown fences. Each step is one of:\n"
    '  {"action": "invoke", "tool_id": "<id>", "operation_name": "<op>", "params": { ... }}\n'
    '  {"action": "focus", "tool_id": "<id>"}\n'
    '  {"action": "unfocus", "tool_id": "<id>"}\n'
    '  {"action": "subgoal", "goal": "<what to achieve>", "mode": "mechanical" | "deliberative", '
    "...}\n"
    'A step with no "action" is treated as "invoke". Use only tool ids and operation names that '
    "appear in the provided tool list. Invoking an operation does not require focusing the tool "
    "first — focus a tool only to perceive its observable properties and signals, and unfocus once "
    "you no longer need them. Respect any usage protocols & safety constraints listed for a tool "
    "when choosing and ordering steps. If the goal came from the user, end the plan by invoking "
    "the user-reply tool's `send_message_to_user` operation to report the outcome — a plan that "
    "never reports back leaves the user without an answer. Put that report in its `text` param as "
    "a short natural-language sentence (or a $decide phrased from the real result), not a bare "
    "data value.\n"
    "`send_message_to_user` is the agent's OWN reply channel — the recipient is always the user, "
    "so it is NEVER how you message anyone else. When the goal asks to email/message/notify some "
    "OTHER person, that is a domain tool's own operation (e.g. an email client's `send_email`), "
    "filling recipient / subject / body from earlier results; when it must reach EACH of several "
    "recipients (e.g. email each relative), fan that invoke out with a mechanical sub-goal.\n"
    "When a parameter's value depends on the RESULT of an earlier step (e.g. an id or address you "
    "only learn by first listing/searching), you do NOT know it yet — never invent a literal. "
    "Instead reference the earlier result:\n"
    '  {"$from": "<operation_name>", "path": "<dotted path into that operation\'s result>"}, or\n'
    '  {"$decide": "<what value is needed>"} when picking the value needs judgement.\n'
    "For a $from path, read the referenced operation's declared `returns:` shape in the tool "
    "catalog and index into THAT: a numeric segment indexes a list position, a name indexes a "
    "field. So if an operation returns an array of records, the id of the first record is "
    '{"$from": "<op>", "path": "0.<id_field>"}; if it returns a single record, just "<id_field>"; '
    'if it returns a bare value, the empty path "". A path that does not match the declared shape '
    "will not resolve against the real result, so match the field names and nesting shown under "
    "`returns:` exactly (do not assume a wrapper key or a field name that isn't listed there).\n"
    "A reference must be the WHOLE value of its key, never embedded inside a larger string — "
    '{"text": "It is {"$from": ...}."} is invalid and will be sent to the user unresolved, '
    "literal braces and all. To report a not-yet-known result in prose, make the field itself a "
    '$decide reference describing the sentence to produce, e.g. {"text": {"$decide": "one '
    'sentence reporting the get_time result"}} — it is phrased from the real result at run time, '
    "not at plan time.\n"
    "Prefer a narrowing step first (e.g. search for the specific item) so a $from reference points "
    "at an unambiguous result.\n"
    "When a step must be repeated once PER ITEM of a collection you only learn at run time (save "
    "each of the found apartments, email each relative), do NOT hard-code one step per item and do "
    "NOT collapse it to a single step — you do not know how many items there will be. Emit ONE "
    '`subgoal` step instead. For a uniform repeat over a collection, use "mode": "mechanical" '
    "with:\n"
    '  "in": {"$from": "<operation_name>", "path": "<path to the array in that result>"}  the '
    "collection to iterate,\n"
    '  "as": "<name>"  a name for the current element, and\n'
    '  "template": { <a single step> }  the step to run once per element, referencing the current '
    'element as {"$bind": "<name>", "path": "<path into the element>"} wherever the element\'s '
    "value is needed. The runtime fans this out to exactly one concrete step per element (the "
    "count comes from the data, not from you), so narrow the collection first (search/filter) to "
    "exactly the items that should be acted on. For repeated work that needs fresh per-item "
    'judgement rather than a uniform template, use "mode": "deliberative" with just the "goal" — '
    "the runtime plans "
    "that sub-goal separately when it is reached.\n"
    "To NARROW or RESHAPE a collection before you act on it — keep only the qualifying items, "
    "dedupe, sort, take the top few, gather per-item results, or reduce to a single number — emit "
    "one data-op step per transform (they compose in order, one per step; do NOT do it all at "
    'once). Each reads an `in` collection ({"$from": ...}, a prior {"$bind": "<name>"}, or a '
    "literal list) and writes a named result under `out`, which later steps read as "
    '{"$bind": "<name>", "path": "..."} (the same $bind you use for a sub-goal element). The '
    "data-ops:\n"
    '  {"action": "filter", "in": ..., "out": "<name>", "where": ...}  keep matching items; '
    '`where` is either {"path": "<field>", "op": "<eq|ne|lt|le|gt|ge|between|in|not_in>", '
    '"value": <v>} (a mechanical comparison; `between` takes [lo, hi], '
    "`in`/`not_in` take a list) or "
    '{"$decide": "<predicate in words>"} when keeping an item needs judgement. For `in`/`not_in`, '
    "`value` may itself be a reference to ANOTHER collection to test membership against — "
    '{"path": "<field>", "op": "not_in", "value": {"$from": "<op>"} | {"$bind": "<name>"}, '
    '"value_path": "<field to read from each item of that collection>"}: '
    "this keeps (in) / excludes "
    "(not_in) items whose `path` value is among that other collection's `value_path` values — e.g. "
    "keep apartments NOT already saved. Omit `value_path` when the referenced "
    "collection is already "
    "a list of the bare keys,\n"
    '  {"action": "distinct", "in": ..., "out": "<name>", "by": "<field>"}  drop duplicates (omit '
    "`by` to dedupe whole items),\n"
    '  {"action": "sort", "in": ..., "out": "<name>", "by": "<field>", "desc": true|false},\n'
    '  {"action": "take", "in": ..., "out": "<name>", "n": <count>}  the first n items,\n'
    '  {"action": "collect", "from": "<operation_name>", "out": "<name>"}  gather the results of '
    "every run of that operation — use it after a mechanical sub-goal that invoked one operation "
    "per item, to turn the scattered per-item results into one list. Each collected item also "
    "carries that call's INPUT arguments, so you can filter/join on them even when the result "
    "doesn't echo them back — e.g. after get_crime_rate per zip, `collect` yields items with both "
    "the returned rate AND the zip_code it was called for, so a mechanical `between` then an "
    "`in`/`not_in` membership join on zip_code needs no $decide,\n"
    '  {"action": "reduce", "in": ..., "out": "<name>", "op": "<sum|min|max|count|mean>", '
    '"by": "<field>"}  aggregate to a single value.\n'
    "So the 'save each QUALIFYING apartment' shape is: search -> `filter` the results into a "
    '`qualifying` binding -> a mechanical sub-goal whose "in" is {"$bind": "qualifying"}. To act '
    "on values a tool produced per item (e.g. a crime rate per zip), map with a mechanical "
    "sub-goal, then `collect` its results before filtering or reducing them.\n"
    "For a plain top-N selection, `sort` + `take` is the right tool — and it stays right even when "
    "ties on the sort key are possible, AS LONG AS the goal does not dictate how to break them "
    "(any of the tied items is an acceptable pick). Do NOT reach for a $decide just because a tie "
    "could happen. ONLY when the goal SPECIFIES a tie-break or priority rule that the sort order "
    "cannot encode — one that applies among items tied on the primary key, or that depends on how "
    "many items qualify (e.g. 'the two cheapest; if their prices tie, prefer the ones with "
    "laundry, ordered alphabetically; if fewer than two have laundry, take the ones with the most "
    "amenities') — is `sort` + `take` wrong: taking first collapses the tie by the sort's "
    "incidental order and DISCARDS the other tied candidates before the specified rule can weigh "
    "them. For that case bring the candidates together with `sort` on the primary key, then apply "
    "the WHOLE rule in ONE `$decide` filter over that sorted collection — "
    '{"action": "filter", "in": {"$bind": "<sorted>"}, "out": "<name>", "where": {"$decide": '
    '"<the entire selection rule: how many to keep and every tie-break clause>"}} — so the '
    "judgement sees every tied candidate and returns exactly the chosen subset (do not `take` "
    "before it).\n"
    "Keep deliberative sub-goals RARE and SMALL. A deliberative sub-goal re-plans with the model "
    "when it is reached, so its goal must REDUCE the problem to a concrete, hard-to-template slice "
    "— it must NOT restate the task, or a large part of it, as another sub-goal. If you are about "
    'to write a deliberative "goal" that echoes the parent goal, STOP and decompose the work HERE '
    "instead — into concrete invoke steps, data-ops, and mechanical sub-goals. Repeating one tool "
    "call over a collection is a MECHANICAL sub-goal; keeping / deduping / sorting / limiting / "
    "gathering / aggregating a collection is DATA-OP steps; a value that needs judgement is "
    "$decide. Reserve a deliberative sub-goal for a small, genuinely heterogeneous continuation "
    "whose SHAPE — not merely its values — is unknown until you see run-time state (e.g. triage an "
    "ambiguous result set where the right next step depends on what was found). Prefer a single "
    "flat plan that "
    "reduces to concrete steps: the runtime REFUSES a deliberative sub-goal that merely re-states "
    "an ancestor, so a plan that leans on them instead of reducing will stall and do nothing.\n"
    "You are also given the agent's currently observed properties (persistent state, e.g. a "
    "thermostat reading) and recently observed signals (transient events, e.g. a notification) as "
    "already-known facts about the current world. Use them to decide WHAT to do — which branch to "
    "take, whether a step is still needed — and to fill parameters whose value is stable and "
    "meaningful (a temperature, a status, a name). But do NOT copy a volatile IDENTIFIER you "
    "happen to see there — an email id, event id, message or thread id, an address — into a step "
    "as a literal: such an id is specific to this run's data, so a plan that hardcodes it is not "
    "reusable and breaks the next time the same goal runs against different data. For an id, still "
    "emit a $from reference to the operation that yields it (adding the narrowing search/list step "
    "if the plan lacks one), exactly as you would if it were not currently visible."
)


def render_tools(tools: dict[str, Manual]) -> str:
    """Render the tools' three-part usage interface (A&A) for a planning prompt: operations to
    *invoke*, plus the observable properties and signals perceivable by *focusing* — surfacing the
    latter two is what motivates a focus/unfocus plan step (a tool with neither reads as
    invoke-only). Also renders any authored ``Usage Protocols & Safety`` as constraints the plan
    must respect. Public so a custom ``PlanPrompt`` can reuse it."""
    if not tools:
        return "(no tools available)"
    blocks: list[str] = []
    for tool_id, manual in tools.items():
        header = f"- tool `{tool_id}`"
        if manual.description:
            header += f": {manual.description}"
        lines = [header]
        lines += _render_operations(manual)
        lines += _render_affordances(
            "observable properties (focus to perceive)",
            "property",
            [(p.name, p.description) for p in manual.observable_properties],
            manual.section(ManualSection.OBSERVABLE_PROPERTIES),
        )
        lines += _render_affordances(
            "signals (focus to receive)",
            "signal",
            [(s.name, s.description) for s in manual.signals],
            manual.section(ManualSection.SIGNALS),
        )
        # Usage protocols & safety — the constraints the plan must respect. Prose-only (no
        # structured field: it lives only in an authored Markdown manual's raw_text — ADR-0015), so
        # it surfaces just for hand-authored manuals. The "suspend until signal Y" portion is
        # consumed by the blocked-state machinery, not the planner (that action does not exist yet).
        safety = _prose(manual.section(ManualSection.USAGE_AND_SAFETY))
        if safety is not None:
            lines += [
                "    usage protocols & safety (constraints the plan must respect):",
                f"      {safety}",
            ]
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _render_operations(manual: Manual) -> list[str]:
    """Operations block for ``render_tools``. Unlike properties/signals, each operation also renders
    its *parameter schema* (names, types, required-ness, and any format hint in the JSON-schema
    description) — without it the planner guesses param names/formats and the invoke fails against
    the real tool (e.g. ARE wants ``start_datetime`` in ``YYYY-MM-DD HH:MM:SS``, not a made-up
    ``start``). Falls back to the authored Markdown ``Operations`` section when the adapter channel
    filled no structured specs (a hand-authored manual), same rule as ``_render_affordances``."""
    if manual.operations:
        out = ["    operations (invoke):"]
        for op in manual.operations:
            head = f"      - operation `{op.name}`"
            out.append(head + (f": {op.description}" if op.description else ""))
            out += _render_params(op.parameters)
            out += _render_returns(op.returns)
        return out
    body = _prose(manual.section(ManualSection.OPERATIONS))
    return ["    operations (invoke):", f"      {body}"] if body is not None else []


def _render_params(schema: dict[str, Any]) -> list[str]:
    """Render an operation's JSON-schema ``parameters`` (object with ``properties``/``required``) as
    a compact bullet list the planner can bind against. Empty/absent schema -> nothing."""
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict) or not properties:
        return []
    required = set(schema.get("required", []) or [])
    out = ["          params:"]
    for name, spec in properties.items():
        spec = spec if isinstance(spec, dict) else {}
        kind = spec.get("type", "any")
        flag = ", required" if name in required else ""
        description = spec.get("description", "")
        out.append(
            f"            - {name} ({kind}{flag})" + (f": {description}" if description else "")
        )
    return out


def _render_returns(schema: dict[str, Any] | None) -> list[str]:
    """Render an operation's declared result shape (``OperationSpecification.returns``) so a planner
    can author a ``$from`` path into it (which field/index yields an id) instead of guessing. Empty/
    absent -> nothing (the op reads as a bare/undeclared result)."""
    if not isinstance(schema, dict) or not schema:
        return []
    out = [f"          returns: {_describe_return_shape(schema)}"]
    description = schema.get("description")
    if isinstance(description, str) and description.strip():
        out.append(f"            ({description.strip()})")
    return out


# How deep the shape rendering expands nested records. A record whose field is itself a list of
# records (ARE's ReturnedEmails.emails -> list[Email]) needs its inner field names surfaced too, or
# the planner can't author the nested `$from` path. Bounded so an arbitrary (e.g. MCP-supplied)
# outputSchema can't render an unboundedly long line.
_MAX_SHAPE_DEPTH = 4


def _describe_return_shape(schema: dict[str, Any], depth: int = 0) -> str:
    """A compact, path-oriented rendering of a JSON-Schema-shaped result: an array says what it
    holds, an object lists its field names (what a ``$from`` path indexes into), a leaf its type.
    Recurses into both an array's item shape and each object field's nested array/object shape (up
    to ``_MAX_SHAPE_DEPTH``), so a record nested inside a record still shows the field names a path
    binds against — otherwise a wrapped list-of-records (list_emails -> ReturnedEmails.emails) would
    hide the very field a ``$from`` needs."""
    kind = schema.get("type")
    if kind == "array":
        items = schema.get("items")
        if isinstance(items, dict) and items and depth < _MAX_SHAPE_DEPTH:
            inner = _describe_return_shape(items, depth + 1)
        else:
            inner = "value"
        return f"array of {inner}"
    if kind == "object":
        properties = schema.get("properties")
        if isinstance(properties, dict) and properties and depth < _MAX_SHAPE_DEPTH:
            fields = [_describe_field(name, spec, depth) for name, spec in properties.items()]
            return "object with fields: " + ", ".join(fields)
        return "object"
    return str(kind) if kind else "value"


def _describe_field(name: str, spec: Any, depth: int) -> str:
    """A field name, annotated with its nested shape in parens when the field is itself an array or
    object (a record's field the planner must index further into); leaf fields render bare."""
    if isinstance(spec, dict) and spec.get("type") in ("array", "object"):
        return f"{name} ({_describe_return_shape(spec, depth + 1)})"
    return name


def _render_affordances(
    label: str, kind: str, entries: list[tuple[str, str]], prose: str | None
) -> list[str]:
    """One affordance group (operations / observable properties / signals) for ``render_tools``:
    the structured specs if the adapter channel filled them, else the authored Markdown section as
    prose, else nothing. An empty group is omitted, not shown blank — so an invoke-only tool carries
    no properties/signals to focus, and a plain MCP tool renders just its one operation."""
    if entries:
        out = [f"    {label}:"]
        for name, description in entries:
            out.append(f"      - {kind} `{name}`" + (f": {description}" if description else ""))
        return out
    body = _prose(prose)
    return [f"    {label}:", f"      {body}"] if body is not None else []


def _prose(section: str | None) -> str | None:
    """A manual section's stripped body, or None if absent, blank, or the literal ``(none)``."""
    if section and section.strip() and section.strip().lower() != "(none)":
        return section.strip()
    return None


@dataclass(frozen=True)
class PerceptSnapshot:
    """The agent's currently-observed world state, bundled for a planning/grounding prompt —
    ``WorkingMemory.properties``/``.signals`` handed down as one value instead of two parallel
    ``list[Percept]`` parameters, so a ``PlanPrompt``/``GroundPrompt`` call can't have properties
    and signals transposed (both are otherwise the exact same static type). ``properties`` holds
    ``ObservableProperty``-payload percepts (replace-by-key, at most one per ``(source, name)``);
    ``signals`` holds ``Signal``-payload percepts (an append log — duplicates are distinct
    occurrences, never deduplicated). An empty snapshot (``PerceptSnapshot()``) means nothing
    observed yet, not an omitted section."""

    properties: list[Percept] = field(default_factory=list)
    signals: list[Percept] = field(default_factory=list)


def _render_json(value: Any) -> str:
    """JSON-render a percept value/payload for a prompt. Falls back to the ``str()`` of anything
    ``json.dumps`` can't serialize (e.g. a ``datetime`` a custom adapter pushed) rather than raising
    and crashing the whole ``infer``/``ground`` call — a degraded rendering beats no plan at all."""
    try:
        return json.dumps(value)
    except TypeError:
        return json.dumps(str(value))


def render_properties(properties: list[Percept]) -> str:
    """Render the agent's currently observed property snapshot (``WorkingMemory.properties``) for a
    planning/grounding prompt — the runtime's currently-known world state, not just the results of
    past actions (``activity.history``). Every percept here carries an ``ObservableProperty``
    payload (``name``/``value``), replaced-by-key so at most one line per ``(source, name)``. Values
    are JSON-rendered (not ``repr``) so a model told to reuse one verbatim in its own JSON answer
    copies a valid literal (``false``/``null``, not ``False``/``None``), and length-capped like
    ``render_history``. Public so a custom ``PlanPrompt``/``GroundPrompt`` can reuse it."""
    if not properties:
        return "(none observed yet)"
    return "\n".join(
        f"- {p.source}.{p.payload.name} = {_truncate(_render_json(p.payload.value))}"
        for p in properties
    )


def render_signals(signals: list[Percept]) -> str:
    """Render recently observed signals (``WorkingMemory.signals``) for a planning/grounding
    prompt. Every percept here carries a ``Signal`` payload (``name``/``payload`` dict). An
    append log — the same name from the same source may appear more than once, each a distinct
    occurrence, never deduplicated. Length-capped like ``render_history``. Public so a custom
    ``PlanPrompt``/``GroundPrompt`` can reuse it."""
    if not signals:
        return "(none observed yet)"
    return "\n".join(
        f"- {p.source}.{p.payload.name}: {_truncate(_render_json(p.payload.payload))}"
        for p in signals
    )


# The plan prompt renders a bounded recent window of user messages, not the whole log: `messages`
# storage is intentionally uncapped (so the consumed-cursor stays valid), so an unbounded render
# would grow every prompt. Newest-last; only the most recent are shown.
_MESSAGE_RENDER = 10


def render_messages(messages: list[Message]) -> str:
    """Render the recent inbound user messages (``WorkingMemory.messages``) for a planning prompt —
    the channel that carries user *instructions / steering* (a follow-up after a stop, a mid-task
    correction) into inference, distinct from an activity's goal string. An append log like
    ``render_signals``; only the most recent ``_MESSAGE_RENDER`` are shown (storage is uncapped, so
    the prompt window is capped here instead), each length-capped. ``(none)`` when empty. Public so
    a custom ``PlanPrompt`` can reuse it."""
    if not messages:
        return "(none)"
    lines = []
    for m in messages[-_MESSAGE_RENDER:]:
        text = m.content.get("text")
        rendered = text if isinstance(text, str) else _render_json(m.content)
        lines.append(f"- {m.sender}: {_truncate(rendered)}")
    return "\n".join(lines)


class PlanPrompt(Protocol):
    """Builds the ``(system, user_prompt)`` pair ``infer()`` sends to the LLM, from the activity,
    the available tools, and the agent's currently observed world state. Injected into
    ``ProceduralMemory`` so planning *content* is customizable — a domain system prompt, few-shot
    examples, a different catalog rendering — without subclassing or moving planning into a
    ``ReasonStrategy``. A plain function satisfies it; a class with ``__call__`` works too (a
    stateful builder). Whatever it produces, the model's response must still parse as the
    ``{"steps": [...]}`` contract — that half is fixed (``_parse_plan_steps``).
    """

    def __call__(
        self,
        activity: Activity,
        tools: dict[str, Manual],
        observed: PerceptSnapshot,
        messages: list[Message],
    ) -> tuple[str, str]: ...


def default_plan_prompt(
    activity: Activity,
    tools: dict[str, Manual],
    observed: PerceptSnapshot | None = None,
    messages: list[Message] | None = None,
) -> tuple[str, str]:
    """The built-in ``PlanPrompt``: the fixed ``PLAN_SYSTEM_PROMPT`` (the JSON step vocabulary) plus
    a user prompt rendering the goal, the tool catalog, the agent's currently observed world state,
    the results of operations already executed, and recent user instructions (``observed`` /
    ``messages`` are omittable — an unrelated caller isn't forced to supply them). Rendering the
    executed history lets a *replan* (e.g. after a user stop clears the plan) see what has already
    been done and not repeat a side-effecting step; rendering recent messages makes a follow-up
    instruction visible to inference rather than reaching the model only as a goal string. Reuse
    ``PLAN_SYSTEM_PROMPT`` / ``render_tools`` / ``render_properties`` / ``render_signals`` /
    ``render_history`` / ``render_messages`` when writing a custom one."""
    observed = observed or PerceptSnapshot()
    user = (
        f"Goal: {activity.goal}\n\n"
        f"Available tools and their operations:\n{render_tools(tools)}\n\n"
        f"Currently observed properties:\n{render_properties(observed.properties)}\n\n"
        f"Recently observed signals:\n{render_signals(observed.signals)}\n\n"
        f"Results of operations already executed:\n{render_history(activity.history)}\n\n"
        f"Recent instructions from the user:\n{render_messages(messages or [])}"
    )
    return PLAN_SYSTEM_PROMPT, user


def step_from_raw(raw: dict[str, Any]) -> Step:
    """Convert one raw plan-step dict (the model's JSON step shape) into a ``Step``. An ``invoke`` —
    the default when no ``action`` is given — routes through ``invoke_step`` so the tool_id and
    operation_name land under the routing keys; every other action (including a ``subgoal``, whose
    nested ``template`` dict is preserved verbatim for the fan-out to instantiate later) keeps its
    remaining keys as ``params``. Shared by ``_parse_plan_steps`` and the mechanical fan-out so both
    read the same step grammar."""
    action = raw.get("action", InvokeAction.name)
    if action == InvokeAction.name:
        return invoke_step(raw["tool_id"], raw["operation_name"], **raw.get("params", {}))
    params = {k: v for k, v in raw.items() if k != "action"}
    return Step(next_action=action, params=params)


def _parse_plan_steps(text: str) -> list[Step]:
    try:
        data = _load_json_object(text)
        return [step_from_raw(raw) for raw in data["steps"]]
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"could not parse a plan from model output: {exc!r}\n---\n{text}") from exc


def _strip_code_fences(text: str) -> str:
    """Tolerate a ```json ... ``` or bare ``` wrapper (models can add one despite the ask not to."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def _iter_json_objects(text: str) -> Iterator[str]:
    """Yield each balanced ``{...}`` substring, in order, tracking string literals/escapes so a
    brace *inside* a string doesn't unbalance the count. Used to recover the JSON when a model wraps
    its answer in prose despite the ask (adaptive thinking makes a stray lead-in sentence common).
    More than one is yielded because a prose lead-in can itself contain a brace-group (``"use the
    {tag} format: {...}"``) — the caller tries each until one parses, rather than betting the first
    balanced group is the JSON answer."""
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escape = False
        end: int | None = None
        for j in range(i, n):
            ch = text[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            return  # an unbalanced tail: no further complete object can start before it closes
        yield text[i : end + 1]
        i = end + 1  # resume past this object so the next top-level group is found, not re-nested


def _load_json_object(text: str) -> Any:
    """Parse a JSON value from model output, tolerating both a code-fence wrapper and surrounding
    prose. Fast path: parse the fence-stripped text directly (the common clean case). Fallback: try
    each balanced ``{...}`` in order and return the first that parses, so a prose-wrapped
    ``{"keep": []}`` (or plan/params object) no longer dies on a bare ``json.loads`` — and a prose
    lead-in that *itself* contains a brace-group (``"use the {tag} format: {...}"``) doesn't shadow
    the real object the way betting on the first balanced group would. Re-raises the last
    ``json.JSONDecodeError`` when nothing yields valid JSON — the callers' anti-corruption boundary
    converts it to a ``ValueError`` as before."""
    stripped = _strip_code_fences(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as first_err:
        last_err: json.JSONDecodeError = first_err
    # Fallback runs outside the except so the eventual re-raise isn't exception-chained onto the
    # fast-path error (they're the same failure viewed twice, not a handler bug).
    for span in _iter_json_objects(stripped):
        try:
            return json.loads(span)
        except json.JSONDecodeError as exc:
            last_err = exc
    raise last_err


# --- parameter grounding (the Reason-phase escalation, packaged here like infer) ------------------

GROUND_SYSTEM_PROMPT = (
    "You are grounding the parameters of a SINGLE tool operation about to be invoked. You are "
    "given the goal, the operation and its parameter schema, a partial set of parameters (some "
    "values may still be references to earlier results), the agent's currently observed properties "
    "and recently observed signals, the named data-op bindings (collections an earlier step "
    "computed), and the results of the operations already executed. Produce the final, concrete "
    "parameters: fill every value that depends on a prior result, a named binding, or an already-"
    "observed property/signal from the ACTUAL data given, and keep already-concrete values as "
    "given.\n"
    'Respond with ONLY a JSON object of the form {"params": { ... }} and nothing else — no prose, '
    "no markdown fences. Use only parameter names from the schema."
)


class GroundPrompt(Protocol):
    """Builds the ``(system, user_prompt)`` pair ``ground()`` sends to the LLM — the grounding
    counterpart to ``PlanPrompt``, injected into ``ProceduralMemory`` so grounding *content* is
    customizable. The response must parse as the fixed ``{"params": {...}}`` contract."""

    def __call__(
        self,
        activity: Activity,
        operation_name: str,
        manual: Manual | None,
        partial_params: dict[str, Any],
        observed: PerceptSnapshot,
    ) -> tuple[str, str]: ...


def render_history(history: list[CompletedOperation]) -> str:
    """Render an activity's executed operations + results for a grounding prompt. Public so a custom
    ``GroundPrompt`` can reuse it. Results are the grounder's ground truth for resolving a reference
    (the id a ``$from``/``$decide`` needs to pick), so they're rendered with a much larger cap than
    the observed-state renderers — a search returning several records must not have a later record's
    id truncated off the end (the failure that made a second email invisible to grounding)."""
    if not history:
        return "(nothing executed yet)"
    lines = []
    for completed in history:
        outcome = completed.ack.result if completed.ack.ok else f"ERROR: {completed.ack.result}"
        args = json.dumps(completed.invocation.params)
        rendered = _truncate(outcome, _HISTORY_RESULT_LIMIT)
        lines.append(f"- {completed.invocation.operation_name}({args}) -> {rendered}")
    return "\n".join(lines)


def render_bindings(bindings: dict[str, Any]) -> str:
    """Render an activity's named data-op bindings for a grounding prompt. Public so a custom
    ``GroundPrompt`` can reuse it. A ``$bind`` reference or a ``$decide`` instruction may name a
    binding an earlier data-op step produced (a filtered/sorted/collected collection), so — like
    history results — these are the grounder's ground truth for resolving a reference and share the
    same generous per-value cap. Order is preserved, so a sorted binding's front (e.g. the cheapest
    items) survives truncation of a long tail."""
    if not bindings:
        return "(none)"
    lines = []
    for name, value in bindings.items():
        rendered = _truncate(json.dumps(value, default=str), _HISTORY_RESULT_LIMIT)
        lines.append(f"- {name} = {rendered}")
    return "\n".join(lines)


def _render_operation_schema(manual: Manual | None, operation_name: str) -> str:
    """The single operation's name/description + parameter schema (reuses ``_render_params``)."""
    if manual is None:
        return f"operation `{operation_name}` (no schema available)"
    op = manual.operation(operation_name)
    if op is None:
        return f"operation `{operation_name}` (not described in the manual)"
    lines = [f"operation `{op.name}`" + (f": {op.description}" if op.description else "")]
    lines += _render_params(op.parameters)
    return "\n".join(lines)


def default_ground_prompt(
    activity: Activity,
    operation_name: str,
    manual: Manual | None,
    partial_params: dict[str, Any],
    observed: PerceptSnapshot | None = None,
) -> tuple[str, str]:
    """The built-in ``GroundPrompt``: goal + the operation schema + the partial params + the
    agent's currently observed world state + the named data-op bindings + the execution history
    (``observed`` is omittable — an unrelated caller isn't forced to supply one). Reuse
    ``GROUND_SYSTEM_PROMPT`` / ``render_history`` / ``render_bindings`` / ``render_properties`` /
    ``render_signals`` in a custom one."""
    observed = observed or PerceptSnapshot()
    user = (
        f"Goal: {activity.goal}\n\n"
        f"Operation to invoke:\n{_render_operation_schema(manual, operation_name)}\n\n"
        f"Partial parameters (resolve any references, keep concrete values):\n"
        f"{json.dumps(partial_params, indent=2)}\n\n"
        f"Currently observed properties:\n{render_properties(observed.properties)}\n\n"
        f"Recently observed signals:\n{render_signals(observed.signals)}\n\n"
        f"Named data-op bindings (a $bind reference or a $decide instruction may name one):\n"
        f"{render_bindings(activity.bindings)}\n\n"
        f"Results of operations already executed:\n{render_history(activity.history)}"
    )
    return GROUND_SYSTEM_PROMPT, user


def _parse_params(text: str) -> dict[str, Any]:
    try:
        data = _load_json_object(text)
        params = data["params"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(
            f"could not parse grounded params from model output: {exc!r}\n---\n{text}"
        ) from exc
    if not isinstance(params, dict):
        raise ValueError(f"grounded params is not a JSON object: {params!r}")
    return params


# --- revalidate: the context-adaptation plan-validity re-check — ADR-0024 ------------------------

REVALIDATE_SYSTEM_PROMPT = (
    "You are deciding whether an IN-PROGRESS plan is still VALID given the latest observations. "
    "You are given the goal, the plan's REMAINING steps, and the new observed state and messages. "
    "The plan is INVALID if the new information changes what the remaining steps should do — a "
    "follow-up that changes a detail the plan acted on, or a precondition that no longer holds. "
    "It is VALID if the remaining steps still achieve the goal; the agent's OWN prior actions do "
    "not by themselves invalidate it.\n"
    'Respond with ONLY a JSON object {"valid": true} or {"valid": false} — no prose, no fences.'
)


def _parse_verdict(text: str) -> bool:
    """Parse the revalidation's ``{"valid": bool}`` answer. A malformed/unparseable answer degrades
    to ``True`` (still valid), so a flaky call can't trigger a replan storm — mirrors ``select``'s
    fail-soft degrade (ADR-0024)."""
    try:
        obj = _load_json_object(text)
        return bool(obj["valid"])
    except (ValueError, KeyError, TypeError):
        return True


# --- select: the model-escalated data-op filter predicate ($decide) — ADR-0023 -------------------

SELECT_SYSTEM_PROMPT = (
    "You are filtering a list down to the subset that satisfies a natural-language predicate. You "
    "are given the goal, the predicate, and the list items each on its own line prefixed by its "
    "0-based index. Decide which items satisfy the predicate, judging each item ON ITS OWN DATA.\n"
    'Respond with ONLY a JSON object of the form {"keep": [<indices>]} — the 0-based indices of '
    "the items to KEEP — and nothing else, no prose, no markdown fences. Keep an item only if it "
    'clearly satisfies the predicate; if none do, respond {"keep": []}.'
)


def _parse_keep(text: str, count: int) -> list[int]:
    """Parse the ``{"keep": [<indices>]}`` selection contract into the in-range indices to keep,
    preserving the model's order. Out-of-range or non-integer entries are dropped (defensive against
    a stray index) and a repeated index collapses to its first occurrence (so the filtered subset
    never duplicates an item — a duplicate would double-act a downstream fan-out); a structurally
    malformed answer raises, the anti-corruption boundary that infer/ground share."""
    try:
        data = _load_json_object(text)
        keep = data["keep"]
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(
            f"could not parse a selection from model output: {exc!r}\n---\n{text}"
        ) from exc
    if not isinstance(keep, list):
        raise ValueError(f"selection 'keep' is not a JSON array: {keep!r}")
    seen: set[int] = set()
    kept: list[int] = []
    for i in keep:
        if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < count and i not in seen:
            seen.add(i)
            kept.append(i)
    return kept


# Operation results carry the identifiers a reference resolves against, so history is rendered with
# a generous cap (bounded, but large enough that a multi-record search result isn't cut mid-record).
# Observed properties/signals are re-observed state that would grow every prompt, so they keep the
# smaller default below — this is why the two aren't one shared limit.
_HISTORY_RESULT_LIMIT = 4000


def _truncate(value: Any, limit: int = 400) -> str:
    """One-line, length-capped rendering of a result for a prompt/log line."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit] + "…"


class ProceduralMemory:
    # Plans are keyed by their own stable id (the storage handle) and retrieved by an exact match
    # on goal (the retrieval key) via backend.query — so two plans with distinct ids but the same
    # goal coexist, and re-storing under the same id updates in place. The default is deterministic:
    # exact goal-string equality, no embedding similarity (that would come with a vector-store
    # backend, alongside infer()).
    def __init__(
        self,
        backend: MemoryBackend,
        llm: LLMClient | None = None,
        prompt: PlanPrompt = default_plan_prompt,
        ground_prompt: GroundPrompt = default_ground_prompt,
    ) -> None:
        # `llm` is the model behind infer()/ground() — procedural memory "includes implicit
        # knowledge encoded in LLM weights", both *query* against it (CoALA). `None` keeps
        # the deterministic store/retrieve half usable without a model. `prompt` / `ground_prompt`
        # are the pluggable knobs for planning / grounding *content* (custom instructions etc.).
        self._backend = backend
        self._llm = llm
        self._prompt = prompt
        self._ground_prompt = ground_prompt

    async def retrieve(self, activity: Activity) -> Plan | None:
        """Looks up a cached Plan matching this activity's goal — e.g. exact match or embedding
        similarity, backend-dependent. The cheap path: skips infer() entirely when it hits."""
        rows = await self._backend.query(goal=activity.goal)
        if not rows:
            return None
        # query()'s contract is most-relevant-first with a deterministic tiebreak, so rows[0] is the
        # canonical plan for this goal regardless of backend.
        return self._from_dict(rows[0])

    async def infer(
        self,
        activity: Activity,
        tools: dict[str, Manual],
        observed: PerceptSnapshot | None = None,
        messages: list[Message] | None = None,
    ) -> Plan:
        """Produce a new multi-step Plan when no cached one fits — the expensive path, one model
        call producing a whole sequence of Steps at once. This is querying procedural memory for
        "implicit knowledge encoded in LLM weights": it builds a prompt from the goal, the
        available ``tools`` (keyed by tool id -> its Manual, supplied by the caller that holds the
        live registry — a memory module never reaches into the environment), and the caller's
        current ``observed`` world-state snapshot (so planning isn't blind to already-observed
        properties/signals — omittable, defaulting to none observed) and any recent user
        ``messages`` (so a follow-up instruction reaches inference, not just the goal string) via
        the injected ``PlanPrompt``, calls the pluggable ``LLMClient``, converts the model's JSON to
        the runtime's own ``Plan``/``Step`` vocabulary. That conversion is the anti-corruption
        boundary; malformed output raises ``ValueError`` rather than a half-built plan. Without an
        LLM the module is store/retrieve only and this raises."""
        if self._llm is None:
            raise RuntimeError(
                "ProceduralMemory has no LLM configured; cannot infer a plan (store/retrieve "
                "still work). Pass an LLMClient to enable inference."
            )
        system, user = self._prompt(activity, tools, observed or PerceptSnapshot(), messages or [])
        log.debug("reason: system prompt\n%s\nUser prompt\n%s", system, user)
        text = await self._llm.complete(system=system, prompt=user)
        return Plan(id=uuid.uuid4().hex, goal=activity.goal, steps=_parse_plan_steps(text))

    async def ground(
        self,
        activity: Activity,
        operation_name: str,
        manual: Manual | None,
        partial_params: dict[str, Any],
        observed: PerceptSnapshot | None = None,
    ) -> dict[str, Any]:
        """Decide an operation's concrete parameters from the execution context — the escalation
        the Reason phase calls when a param reference can't be resolved *mechanically* (an ambiguous
        pick, or an unknown/mis-guessed result shape). One model call: a prompt from the goal,
        the operation schema, the partial params, the caller's current ``observed`` world-state
        snapshot (omittable, defaulting to none observed), and the activity's history (via the
        injected ``GroundPrompt``), then converts the model's JSON answer to a concrete params dict
        — the anti-corruption boundary (malformed -> ``ValueError``). Reuses the same ``LLMClient``
        seam as ``infer``; no LLM -> raises. (Grounding a step is an Act-adjacent reasoning act; it
        lives here only because procedural memory currently owns the model handle — the eventual
        home is a client injected per strategy. See ADR-0017.)"""
        if self._llm is None:
            raise RuntimeError(
                "ProceduralMemory has no LLM configured; cannot ground parameters. Pass a client."
            )
        system, user = self._ground_prompt(
            activity, operation_name, manual, partial_params, observed or PerceptSnapshot()
        )
        log.debug("reason: system prompt\n%s\nUser prompt\n%s", system, user)
        text = await self._llm.complete(system=system, prompt=user)
        return _parse_params(text)

    async def select(self, activity: Activity, collection: list[Any], predicate: str) -> list[Any]:
        """Filter ``collection`` to the subset satisfying a natural-language ``predicate`` — the
        model-escalated ``$decide`` half of the ``filter`` data-op (ADR-0023). One model call over
        the whole collection (a batching simplification of the per-element ideal): it renders each
        item with its index, asks for a ``{"keep": [<indices>]}`` answer, and returns the kept items
        in the model's order. The index contract keeps the model from re-serializing (and mangling)
        the items. Reuses the same ``LLMClient`` seam as ``infer``/``ground``; no LLM -> raises."""
        if self._llm is None:
            raise RuntimeError(
                "ProceduralMemory has no LLM configured; cannot evaluate a $decide filter. Pass a "
                "client."
            )
        items = "\n".join(
            f"{index}: {json.dumps(item, default=str)}" for index, item in enumerate(collection)
        )
        user = f"Goal: {activity.goal}\nPredicate: {predicate}\nItems:\n{items}"
        log.debug("reason: system prompt\n%s\nUser prompt\n%s", SELECT_SYSTEM_PROMPT, user)
        text = await self._llm.complete(system=SELECT_SYSTEM_PROMPT, prompt=user)
        return [collection[index] for index in _parse_keep(text, len(collection))]

    async def revalidate(
        self,
        activity: Activity,
        observed: PerceptSnapshot | None = None,
        messages: list[Message] | None = None,
    ) -> bool:
        """Re-check whether the activity's in-progress plan is still valid against the current world
        — the context-adaptation relevance judgment (ADR-0024). One model call: the goal, the plan's
        REMAINING steps, and the observed properties/signals plus recent messages, asking for
        a ``{"valid": bool}`` verdict. Reuses the same ``LLMClient`` seam as ``infer``; no
        LLM -> raises. A ``False`` verdict re-infers; best-effort, not a guarantee, and
        deliberately general (no domain-authored predicate) — it reasons about relevance itself, so
        the agent's own writes don't spuriously invalidate the plan."""
        if self._llm is None:
            raise RuntimeError(
                "ProceduralMemory has no LLM configured; cannot revalidate a plan. Pass a client."
            )
        snapshot = observed or PerceptSnapshot()
        # The full remaining tail across the sub-goal stack (ADR-0022), not just the active
        # sub-plan: the current plan's steps from step_index, then each suspended parent's steps
        # after its sub-goal (innermost frame resumes first — the top of parent_frames). Checking
        # only the sub-plan would call it "still valid" while a now-stale parent step still waits.
        remaining = (
            list(activity.plan.steps[activity.step_index :]) if activity.plan is not None else []
        )
        for parent_plan, subgoal_index in reversed(activity.parent_frames):
            remaining.extend(parent_plan.steps[subgoal_index + 1 :])
        steps_text = (
            "\n".join(
                f"{index}: {step.next_action} {json.dumps(step.params, default=str)}"
                for index, step in enumerate(remaining)
            )
            or "(none)"
        )
        user = (
            f"Goal: {activity.goal}\n"
            f"Remaining plan steps:\n{steps_text}\n"
            f"Observed properties:\n{render_properties(snapshot.properties)}\n"
            f"Observed signals:\n{render_signals(snapshot.signals)}\n"
            f"Recent messages:\n{render_messages(messages or [])}"
        )
        log.debug(
            "reason: revalidate system prompt\n%s\nUser prompt\n%s", REVALIDATE_SYSTEM_PROMPT, user
        )
        text = await self._llm.complete(system=REVALIDATE_SYSTEM_PROMPT, prompt=user)
        return _parse_verdict(text)

    async def store(self, plan: Plan) -> None:
        """Persists a Plan that was actually followed to completion, so future retrieve() calls
        for similar goals can reuse it. Called by ReflectStrategy on success only — a failed plan
        isn't something future activities should retrieve by default."""
        await self._backend.put(plan.id, asdict(plan))

    @staticmethod
    def _from_dict(data: dict[str, Any]) -> Plan:
        # Rebuild the dataclass graph the backend flattened to plain dict/list/scalar on store.
        return Plan(
            id=data["id"],
            goal=data["goal"],
            steps=[Step(**step) for step in data["steps"]],
        )


class EpisodicMemory:
    """Records a summary of each completed activity and retrieves the ones relevant to a new
    activity. Relevance is goal-equality — the same cheap, deterministic proxy ProceduralMemory
    uses; embedding/LLM similarity is deferred so the default stays reproducible."""

    def __init__(self, backend: MemoryBackend) -> None:
        self._backend = backend

    async def learn(self, activity: Activity, summary: str, *, succeeded: bool) -> None:
        # One episode per activity, keyed by its id: re-learning the same activity overwrites rather
        # than accumulating duplicates. goal is stored top-level so consult can filter on it through
        # the backend's exact-match query(). Beyond the prose summary, the episode captures a
        # self-contained record of what was attempted — outcome, the plan snapshot, step progress,
        # and the last operation result — reconstructing as much of the experience as survives on
        # the activity. `succeeded` is passed in because ActivityState.TERMINATED alone can't tell a
        # completed activity from a failed one — only the judging strategy knows. The plan is stored
        # in full even on success (where procedural memory also holds it): the episode stays legible
        # on its own, and on failure it's the only surviving copy, since procedural memory does not
        # store failed plans.
        plan = activity.plan
        await self._backend.put(
            activity.id,
            {
                "activity_id": activity.id,
                "goal": activity.goal,
                "succeeded": succeeded,
                "summary": summary,
                "step_index": activity.step_index,
                "step_count": None if plan is None else len(plan.steps),
                "last_result": (
                    None if activity.last_operation is None else asdict(activity.last_operation)
                ),
                "plan": None if plan is None else asdict(plan),
            },
        )

    async def consult(self, activity: Activity) -> list[Any]:
        # query() re-reads from disk, so results are fresh copies a caller can mutate without
        # corrupting the store.
        return await self._backend.query(goal=activity.goal)
