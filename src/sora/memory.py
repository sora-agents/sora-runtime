"""Working, semantic, procedural, and episodic memory modules."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
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
from sora.types import (
    Change,
    CompletedOperation,
    ConditionVerdict,
    PendingCondition,
    Plan,
    RelevanceCandidate,
    SignalWait,
    Step,
    SupersededPlan,
    UnresolvableGrounding,
    walk_path,
)

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
    # Monotonic count of signals EVER appended — never decremented by the retention cap, so the
    # sequence number of signals[i] is `signals_appended - len(signals) + i`. A waiter that must not
    # re-evaluate a signal it has already seen keeps its own high-water mark against this; it cannot
    # be a list index, because the cap front-evicts and would silently shift every stored index.
    # Deliberately NOT a shared consumed-cursor like messages_cursor: that shape is correct there
    # (a message must drive activity-creation at most once) and wrong here, where signals are a
    # broadcast log with many independent readers — the first reader to advance a shared cursor
    # would blind all the others. Adding it here rather than to Percept keeps a sequence number off
    # the property half of the store, where it would be meaningless.
    signals_appended: int = 0
    # inbound agent-to-agent communication — kept distinct
    messages: list[Message] = field(default_factory=list)
    # Count of messages already routed (turned into an activity goal by Situate, or claimed as
    # reconsideration input by a resume) — a consumed-cursor over the append-only log so each
    # message drives activity-creation at most once and a later tick never re-scans the whole log.
    # Storage stays uncapped (no eviction), so this index is always valid; bounding `messages` with
    # a retention cap is deferred (front-eviction would have to adjust the cursor).
    messages_cursor: int = 0
    focused_tools: dict[str, Tool] = field(default_factory=dict)
    # Tools an explicit `_unfocus_` released, held against the attention reconciler so a deliberate
    # release survives it. Attention is recomputed from scratch every Observe, so without this the
    # policy simply re-attends the tool on the next tick and `_unfocus_` is a permanent no-op —
    # which is what the plan prompt offers the planner as a way to stop watching something early.
    # Cleared by the opposite explicit act (`_focus_`) and by the tool leaving the world (`_leave_`)
    # — never by a policy, so a suppression cannot be lifted by something the agent did not decide.
    suppressed_tools: set[str] = field(default_factory=set)
    # Whether the agent's FocusPolicy is currently narrowing attention below the joined set —
    # written by Observe's reconciler from the policy's own target, read by `scoped_snapshot` so the
    # per-activity prompt view narrows only for an agent that already chose to narrow. Without it
    # the two attention layers can disagree, and the broad policy's whole point (declining to
    # narrow, because a wrongly narrowed view fails silently) is undone one layer down.
    attention_narrowed: bool = False
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

# Worked examples in this prompt are deliberately drawn from a domain nothing evaluates this
# runtime on (a museum collection catalogue). They used to be drawn from the ARE/Gaia2 apartment
# search that most runs exercise — which quietly turned a benchmark score into a partial measure of
# how well the prompt pre-solved that benchmark's own task family. The structural rules are what
# these examples exist to teach and they are domain-free; keeping the nouns off any evaluated domain
# costs nothing and keeps the number honest. When adding an example, do NOT reach for the scenario
# you happen to be debugging.
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
    "appear in the provided tool list. You do not need `focus` steps for the tools your plan "
    "already names: the runtime attends to every tool your steps invoke or reference, for as long "
    "as the plan is live. Emit `focus` only for a tool whose properties or signals you need but "
    "whose operations the plan never calls, and `unfocus` only to stop watching one early. "
    "Respect any usage protocols & safety constraints listed for a tool "
    "when choosing and ordering steps. If the goal came from the user, end the plan by invoking "
    "the user-reply tool's `send_message_to_user` operation to report the outcome — a plan that "
    "never reports back leaves the user without an answer. Put that report in its `text` param as "
    "a short natural-language sentence (or a $decide phrased from the real result), not a bare "
    "data value.\n"
    "`send_message_to_user` is the agent's OWN reply channel — the recipient is always the user, "
    "so it is NEVER how you message anyone else. When the goal asks to email/message/notify some "
    "OTHER person, that is a domain tool's own operation (e.g. an email client's `send_email`), "
    "filling recipient / subject / body from earlier results; when it must reach EACH of several "
    "recipients (e.g. notify each curator), fan that invoke out with a mechanical sub-goal.\n"
    "Never add outward communication the goal did not ask for. A step that sends something to "
    "anyone other than the user — a message, a reply, an invitation, a confirmation — belongs in "
    "the plan ONLY where the goal explicitly asks for it. Doing the requested work AND one extra "
    "courtesy note is not thoroughness: it speaks in the user's name on a matter they did not "
    "raise. (`send_message_to_user` is not an exception to route around this — it reports back to "
    "the user, who asked.)\n"
    "Some verbs only LOOK like speech. To accept, confirm, agree to, acknowledge, approve, decline "
    "or turn down an offer or proposal names a DECISION and the state change that records it — not "
    "an instruction to compose a message saying so. Plan those as the operations that change state "
    "(make the booking, cancel what it supersedes, update the record), and tell the other party "
    "only where the goal separately says to.\n"
    "When a parameter's value depends on the RESULT of an earlier step (e.g. an id or address you "
    "only learn by first listing/searching), you do NOT know it yet — never invent a literal. "
    "Instead reference the earlier result:\n"
    '  {"$from": "<operation_name>", "path": "<dotted path into that operation\'s result>"}, or\n'
    '  {"$decide": "<what value is needed>"} when picking the value needs judgement.\n'
    "A value already in the CURRENTLY OBSERVED PROPERTIES above needs no operation at all — "
    "reference it directly:\n"
    '  {"$prop": "<tool_id>.<property_name>", "path": "<dotted path into the property value>"}\n'
    "One rule: qualify the property name with its tool id. A `$prop` that names its tool is what "
    "tells the runtime to keep observing that tool while the plan runs, and it is also what keeps "
    "the reference unambiguous when several tools expose a property by that name (many publish a "
    "`state`). A bare name is accepted only when it is unambiguous among the tools already being "
    "observed — so it can go missing later even though it resolves now. Qualify it.\n"
    "Prefer $prop over paginated scanning: where a property already holds the whole collection "
    "(e.g. an app's `state`), filter THAT with a data-op in one step rather than calling a "
    "list/search operation repeatedly to page through the same data.\n"
    "To spot that case, read the shape each property is listed with above: it names the fields and "
    'gives the count, e.g. `Contacts.state = {contacts: {<key>: {..., job: "..."}} x 125}`. A '
    "count that large is the COMPLETE collection, and a list operation that returns ten at a time "
    "is a window onto this same data — so filter the property on the field you need and skip the "
    "operation entirely. For a field whose value is EXACT — an id, a status, a category, a "
    "number, a date, a flag — a search operation is a guess that can return [] even when the "
    "record is there, while filtering the property cannot.\n"
    "That flips when the value you match on is a NAME the USER phrased. `eq` matches only the "
    "stored string in full, and people name things approximately — they shorten a title, drop a "
    'subtitle or an edition, reorder words, punctuate it differently — so a goal saying "the '
    'Delft landscape" is stored as "View of Delft, oil on canvas (1661)". A mechanical `eq` on '
    "that phrase matches NOTHING, and an empty result is indistinguishable from the record not "
    "existing: the agent goes on to tell the user the thing cannot be found while it sits in the "
    "collection. So do NOT resolve a user-phrased name with `eq`.\n"
    "Use the tool's OWN search or lookup operation for that instead — whatever the catalog calls "
    "it (a `search_*` / `find_*` / `lookup_*` operation, or one taking a `query`, `name` or "
    "`keyword` parameter). Matching an approximate name against its own records is the job that "
    "operation exists to do, and it is CHEAP: one call, no collection shipped to the model. Only "
    'where the tool offers no such operation, filter the property with a {"$decide": ...} '
    "predicate that accepts the record whose stored name CONTAINS or paraphrases the user's "
    "phrase — still never a mechanical `eq`. What flips the rule is a FREE-FORM name, not who "
    "uttered the value: `eq` stays right for anything the record stores verbatim out of a fixed "
    "vocabulary — ids and keys, enumerated statuses and categories, numbers, dates, booleans, and "
    "anything copied from an earlier result or from observed state — and the user naming one of "
    "those (a city, a status) does not make it approximate.\n"
    "Expect that search to come back with SEVERAL near-matches — for an approximate name that is "
    "the normal outcome, not a failure. Narrow them afterwards on the fields the goal actually "
    "constrains (a date, a medium, a gallery), or ask the user which one they meant. Do not "
    "re-tighten to an `eq` on the name to cut the list down: that is the same mistake one step "
    "later.\n"
    "A value you must COMPUTE from an earlier result is in the same boat, and a DATE is the case "
    'that most often goes wrong. "This coming Saturday", "tomorrow", "an hour after the '
    "meeting\" all depend on a clock reading you have not taken yet: you do not know today's date "
    "at planning time, so a literal date baked into a step is a guess, and a plan that reads the "
    "clock in step 0 and then hardcodes a date in step 2 has thrown that reading away. Take the "
    "reading in an earlier step and express every value derived from it as a $decide naming the "
    "calculation, e.g. "
    '{"start_datetime": {"$decide": "the first Saturday on or after the get_current_time result, '
    'at 08:00:00, formatted YYYY-MM-DD HH:MM:SS"}}'
    " — it is then computed at run time against the real clock.\n"
    "For a $from path, read the referenced operation's declared `returns:` shape in the tool "
    "catalog and index into THAT: a numeric segment indexes a list position, a name indexes a "
    "field. So if an operation returns an array of records, the id of the first record is "
    '{"$from": "<op>", "path": "0.<id_field>"}; if it returns a single record, just "<id_field>"; '
    'if it returns a bare value, the empty path "". A path that does not match the declared shape '
    "will not resolve against the real result, so match the field names and nesting shown under "
    "`returns:` exactly (do not assume a wrapper key or a field name that isn't listed there).\n"
    "A reference must be the WHOLE value of its key, never embedded inside a larger string — "
    '{"text": "It is {"$from": ...}."} is invalid and will be sent to the user unresolved, '
    "literal braces and all. It MAY, though, stand as a whole ELEMENT of a list when the "
    'parameter is a list — {"attendees": [{"$decide": "the manager\'s full name"}]} is '
    "valid and resolves element by element; that is the only way to express a list whose members "
    "are not known until run time. "
    "To report a not-yet-known result in prose, make the field itself a "
    '$decide reference describing the sentence to produce, e.g. {"text": {"$decide": "one '
    'sentence reporting the get_time result"}} — it is phrased from the real result at run time, '
    "not at plan time.\n"
    "Where the data is reachable only through operations, prefer a narrowing step first (e.g. "
    "search for the specific item, or a date/range-bounded list operation) so a $from reference "
    "points at an unambiguous result. Check the observed properties before reaching for that: if "
    "one already holds the collection and the narrowing is MECHANICAL (a field compared against an "
    "EXACT value — a name the user phrased is not one, per the rule above), $prop plus that "
    "filter beats a search — it sees every record rather than the first "
    "page, and it costs no operation at all. When the narrowing would instead need a $decide "
    "filter, prefer an operation that takes it as PARAMETERS (a from/to range, a query, a status): "
    "a $decide filter ships EVERY item in the collection to the model, so it costs more the bigger "
    "the collection, where the operation does the same selection in one declared call. Reach for "
    "$prop plus a $decide filter only when no operation expresses that narrowing.\n"
    "When a step must be repeated once PER ITEM of a collection you only learn at run time "
    "(catalogue each of the found artifacts, notify each curator), do NOT hard-code one step per "
    "item and do "
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
    '{"$decide": "<predicate in words>"} when keeping an item needs judgement — costly, since '
    "every item goes to the model, so narrow with an operation or a mechanical comparison "
    "wherever you can. For `in`/`not_in`, "
    "`value` may itself be a reference to ANOTHER collection to test membership against — "
    '{"path": "<field>", "op": "not_in", "value": {"$from": "<op>"} | {"$bind": "<name>"}, '
    '"value_path": "<field to read from each item of that collection>"}: '
    "this keeps (in) / excludes "
    "(not_in) items whose `path` value is among that other collection's `value_path` values — e.g. "
    "keep artifacts NOT already catalogued. Omit `value_path` when the referenced "
    "collection is already "
    "a list of the bare keys. For the OTHER ops, `value` may likewise be a reference — to the "
    "value being compared against, not to a collection — so a threshold computed by an earlier "
    'step is usable directly: {"path": "score", "op": "gt", "value": {"$bind": "<mean>"}} keeps '
    "what beats a `reduce` mean, and `between` accepts either a reference resolving to [lo, hi] or "
    'the pair with a reference at each end: {"op": "between", "value": [{"$bind": "<lo>"}, '
    '{"$bind": "<hi>"}]}, for two bounds that come from two different steps. No '
    "`value_path` there: the referenced value IS the operand. Whichever way you write it, the "
    "operand has to end up being something the op can compare against — a single value for "
    "lt/le/gt/ge, a two-element pair for `between`. A reference to a whole record, or to nothing, "
    "does not compare and is rejected as a plan defect rather than silently matching no item,\n"
    '  {"action": "distinct", "in": ..., "out": "<name>", "by": "<field>"}  drop duplicates (omit '
    "`by` to dedupe whole items),\n"
    '  {"action": "sort", "in": ..., "out": "<name>", "by": "<field>", "desc": true|false},\n'
    '  {"action": "take", "in": ..., "out": "<name>", "n": <count>}  the first n items,\n'
    '  {"action": "collect", "from": "<operation_name>", "out": "<name>"}  gather the results of '
    "every run of that operation THIS plan performs — use it after a mechanical sub-goal that "
    "invoked one operation per item, to turn the scattered per-item results into one list. It "
    "reaches only this plan's own runs: results listed under 'Results of operations already "
    "executed' that a PREVIOUS, replaced plan produced are NOT collectible, and collecting them "
    "yields an empty list — if this plan needs those values, run the operation again. Each "
    "collected item also "
    "carries that call's INPUT arguments, so you can filter/join on them even when the result "
    "doesn't echo them back — e.g. after get_condition_score per gallery, `collect` yields items "
    "with both the returned score AND the gallery_id it was called for, so a mechanical `between` "
    "then an `in`/`not_in` membership join on gallery_id needs no $decide,\n"
    '  {"action": "reduce", "in": ..., "out": "<name>", "op": "<sum|min|max|count|mean>", '
    '"by": "<field>"}  aggregate to a single value.\n'
    "So the 'catalogue each QUALIFYING artifact' shape is: search -> `filter` the results into a "
    '`qualifying` binding -> a mechanical sub-goal whose "in" is {"$bind": "qualifying"}. To act '
    "on values a tool produced per item (e.g. a condition score per gallery), map with a "
    "mechanical sub-goal, then `collect` its results before filtering or reducing them.\n"
    "For a plain top-N selection, `sort` + `take` is the right tool — and it stays right even when "
    "ties on the sort key are possible, AS LONG AS the goal does not dictate how to break them "
    "(any of the tied items is an acceptable pick). Do NOT reach for a $decide just because a tie "
    "could happen. ONLY when the goal SPECIFIES a tie-break or priority rule that the sort order "
    "cannot encode — one that applies among items tied on the primary key, or that depends on how "
    "many items qualify (e.g. 'the two oldest; if their dates tie, prefer the ones with a "
    "conservation report, ordered alphabetically; if fewer than two have a report, take the ones "
    "with the most provenance records') — is `sort` + `take` wrong: taking first collapses the "
    "tie by the sort's "
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
    "if the plan lacks one), exactly as you would if it were not currently visible.\n"
    # Pending conditions. This is the one part of the contract the planner most reliably gets wrong
    # by OMISSION rather than by malformation: a run that stated three conditional clauses in its
    # own prose encoded none of them as structure, terminated on a confirmation, and had nothing
    # alive when the awaited reply arrived. Hence the explicit instruction to re-read the goal for
    # conditional language, and the worked example showing prose -> structure for that shape.
    'A plan MAY also carry {"pending": [ ... ]} alongside "steps". A pending condition says what '
    "would make this goal relevant AGAIN after the steps are done — so the activity waits instead "
    "of finishing. Re-read the goal for conditional language ('if', 'in case', 'should X happen', "
    "'let me know when', 'once they reply') and turn EACH such clause into one entry. Do not leave "
    "a condition in prose: a clause you mention but do not encode here is silently lost the moment "
    "the last step completes. Each entry is:\n"
    '  {"watch": {"signal": "<signal name>", "source": "<tool id>", "path": "<dotted path>"}, '
    '"when": "<what must have happened>", "then": "<what to do about it>", '
    '"until": "<when to stop waiting>"}\n'
    '"watch" is REQUIRED and is a cheap mechanical filter, not the judgement: name the signal and '
    "the tool that would carry the news, and use `path` to point at the part of that tool's "
    "observable state that would move — it is what stops every unrelated event from waking this "
    "goal, INCLUDING the agent's own writes, which usually land somewhere different from what it "
    "is waiting on. `when` is the actual judgement, in plain language. `then` is a goal, phrased "
    "like the original goal — the runtime plans it fresh when the moment comes, so do not write "
    "steps here. `until` bounds the wait.\n"
    'Example — goal: "Book the Rembrandt restoration slot for the 14th and tell the conservator; '
    "if she can't make it, rebook for whatever day she suggests.\" The second clause is a pending "
    "condition, not a step:\n"
    '  {"steps": [ ... book the slot, message the conservator, report to the user ... ],\n'
    '   "pending": [{"watch": {"signal": "state_changed", "source": "<messaging tool id>", '
    '"path": "folders.INBOX.messages"},\n'
    '     "when": "the conservator replies that the 14th does not work, or proposes another day",\n'
    '     "then": "Rebook the Rembrandt restoration slot for the day she proposes, clearing '
    'whatever is already booked then",\n'
    '     "until": "the restoration slot has taken place"}]}\n'
    "Emit no `pending` at all when the goal is unconditional — most goals are. A condition you "
    "cannot name a watch for does not belong here."
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


# Past this rendered length a property is sketched by shape instead of truncated — see
# _render_property_value. The caps below bound the sketch itself: how deep it descends, how many
# fields it enumerates, and how much of a scalar it shows.
_TRUNCATE_LIMIT = 400
_SHAPE_MAX_DEPTH = 3
_SHAPE_MAX_FIELDS = 24  # a record's field NAMES are the payload here, so enumerate generously
_SHAPE_SAMPLE = 5  # values compared when testing a dict for homogeneity — maps run to the hundreds
_SHAPE_MAX_SCALAR = 40  # a scalar longer than this is a placeholder, not information


def _value_signature(value: Any) -> object:
    """What makes two dict values "the same shape" for _is_keyed_collection."""
    if isinstance(value, dict):
        return frozenset(value)
    if isinstance(value, list):
        return list
    return type(value)


def _is_keyed_collection(value: dict[Any, Any]) -> bool:
    """Whether to read this dict as an ``{id -> record}`` map — describe one entry and count them —
    rather than as a record whose field names should be enumerated.

    Key count alone cannot tell the two apart, and getting it wrong is not cosmetic. An ARE contact
    has fifteen fields, wider than any plausible enumeration cap, so a count-only rule reported it
    as ``{<key>: "Astrid"} x 15`` — withholding every field name, which is the one thing a planner
    needs in order to write a ``$prop`` path or a filter predicate over the property. Real keyed
    collections are *homogeneous* (every value the same shape) where a record is not (a contact
    mixes str, bool and int), so homogeneity is the discriminator and the count is only a floor.
    That floor keeps a small homogeneous dict like ``{INBOX: ..., SENT: ...}`` enumerated, where
    the keys are path segments a planner has to write verbatim rather than opaque ids."""
    if len(value) <= _SHAPE_MAX_FIELDS:
        return False
    signatures = {_value_signature(v) for v in list(value.values())[:_SHAPE_SAMPLE]}
    return len(signatures) == 1


def _property_shape(value: Any, depth: int = 0) -> str | None:
    """A shape sketch for a container too big to show whole, or ``None`` if it should be shown
    verbatim. Bulk state — ARE publishes 125 contacts under one ``state`` property — is the case a
    length cap serves worst: a truncated quarter-kilobyte of one record tells a planner nothing
    about the *path* it must write, which is the only thing it needs from this line. So past the
    threshold, describe the container instead: its keys, or its element shape and cardinality."""
    if isinstance(value, dict):
        if not value:
            return None
        if depth >= _SHAPE_MAX_DEPTH:
            return f"{{... x {len(value)}}}"
        keys = list(value)
        if _is_keyed_collection(value):  # describe one record, count them
            return f"{{<key>: {_shape_or_value(value[keys[0]], depth)}}} x {len(value)}"
        shown = keys[:_SHAPE_MAX_FIELDS]
        inner = ", ".join(f"{k}: {_shape_or_value(value[k], depth)}" for k in shown)
        if len(keys) > len(shown):
            inner += f", ... +{len(keys) - len(shown)} more"
        return f"{{{inner}}}"
    if isinstance(value, list):
        if not value:
            return None
        return f"[{_shape_or_value(value[0], depth)}] x {len(value)}"
    return None  # a scalar is its own best rendering


def _shape_or_value(value: Any, depth: int) -> str:
    """One member of a shape sketch: its own sketch if it has one, else the scalar itself when
    that is short enough to be worth more than a placeholder (``view_limit: 10`` tells the planner
    something ``view_limit: <value>`` does not)."""
    shape = _property_shape(value, depth + 1)
    if shape is not None:
        return shape
    rendered = _render_json(value)
    return rendered if len(rendered) <= _SHAPE_MAX_SCALAR else "<value>"


def _render_property_value(value: Any) -> str:
    """Verbatim JSON while it fits; a shape sketch once it doesn't. The cutoff is the same
    ``_truncate`` limit, so small properties are unaffected and read exactly as before."""
    rendered = _render_json(value)
    if len(rendered) <= _TRUNCATE_LIMIT:
        return rendered
    shape = _property_shape(value)
    return shape if shape is not None else _truncate(rendered)


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
        f"- {p.source}.{p.payload.name} = {_render_property_value(p.payload.value)}"
        for p in properties
    )


# A located change resolves to whole records, and the same all-or-nothing rule as history applies:
# a record cut mid-field is worse than an absent one, because the judgement reads it as complete.
# Smaller than the history budget — this is a delta (what just moved), not an execution trace.
_CHANGED_RECORD_BUDGET = 12_000


def _identifies(item: Any, wanted: set[str]) -> bool:
    """Whether a record is one of the ones a ``Change`` named. An adapter reports *which* ids moved
    but never which FIELD carries the id, so matching any string field is the only general test —
    and an id is opaque enough that a collision against an unrelated field would itself be a real
    coincidence rather than a routine false positive."""
    if isinstance(item, dict):
        return any(value in wanted for value in item.values() if isinstance(value, str))
    return isinstance(item, str) and item in wanted


def _records_for_change(value: Any, change: Change) -> list[Any]:
    """The records inside one property that a ``Change``'s ids point at.

    Handles both container shapes an adapter publishes: an ``{id -> record}`` map (indexed straight)
    and a list of records (scanned for the id). ``removed`` ids are deliberately not looked up —
    they are gone from the snapshot by definition, so naming them is all that can be done.
    """
    ids = change.added + change.updated
    if not ids:  # the coarse form: "something under here moved", nothing to dereference
        return []
    try:
        container = walk_path(value, change.path)
    except (KeyError, IndexError, TypeError, ValueError):
        # The snapshot is read at judgement time, not at change time, so a path can have gone away
        # between the two. Skip it: the change line still says where to look.
        return []
    if isinstance(container, dict):
        return [container[key] for key in ids if key in container]
    if isinstance(container, list):
        wanted = set(ids)
        return [item for item in container if _identifies(item, wanted)]
    return []


def render_changes(changes: Sequence[tuple[str, Change]], properties: list[Percept]) -> str:
    """Render located changes for a judgement prompt: *where* each one landed, and then the actual
    records behind the ids, dereferenced out of the property snapshot.

    This is the division of labour ADR-0022 specifies — ``Change`` carries identities only, and "the
    values behind them come from the observed property snapshot", so the judgement "reads one named
    path instead of re-scanning a whole property". Only the dereference makes that true. Without it
    the judge got the ids and `render_properties`, which shape-sketches any property past its length
    cap: a 129-email inbox arrives as ``emails: [{... x 10}] x 129`` and the id that just landed is
    nowhere in the prompt. Asking whether someone declined an invitation, while showing neither the
    invitation nor the reply, can only produce "nothing fired".

    Duplicate changes are collapsed. Several conditions on one activity routinely declare the same
    watch, and each eligible one contributes its own copy of the identical ``Change`` — six of them
    in the run this was found in, six identical lines in the prompt.
    """
    if not changes:
        return "(the source reported no detail about what moved)"
    seen: set[tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = set()
    located: list[str] = []
    records: list[tuple[str, str | None, str]] = []
    rendered_records: set[str] = set()
    for source, change in changes:
        key = (source, change.path, change.added, change.removed, change.updated)
        if key in seen:
            continue
        seen.add(key)
        where = f"{source}.{change.path or '(whole property)'}"
        detail = "".join(
            f" {label}={list(ids)}"
            for label, ids in (
                ("added", change.added),
                ("removed", change.removed),
                ("updated", change.updated),
            )
            if ids
        )
        located.append(f"- {where}{detail}")
        for percept in properties:
            if percept.source != source:
                continue
            found = _records_for_change(percept.payload.value, change)
            for record in found:
                body = _one_line(_render_json(record))
                if body not in rendered_records:
                    rendered_records.add(body)
                    records.append((where, None, body))
            if found:
                break  # a Change names a path, not a property; the first that resolves is it
    block = "\n".join(located)
    if records:
        block += "\n\nThe records behind those ids, read from the current snapshot:\n" + "\n".join(
            _fit_to_budget(records, _CHANGED_RECORD_BUDGET)
        )
    return block


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


def _render_goal_provenance(activity: Activity) -> str:
    """Say where this plan's goal came from, when the answer changes what the plan should contain.

    For the **plan** prompt only. ``InferAction`` renders a sub-plan against a *copy* of the
    activity whose ``goal`` is the sub-goal string, so the notice below and the goal it qualifies
    agree. The live activity's ``goal`` is never rewritten when Observe enters a sub-plan, so any
    other caller would print the user's actual request and then assert it is not a request from the
    user.

    ``PLAN_SYSTEM_PROMPT`` tells the planner to end with ``send_message_to_user`` *if the goal came
    from the user* — a condition the model cannot evaluate, because a sub-goal is inferred against
    its own goal string with the parent nowhere in the prompt. Every plan therefore read as the
    user's and ended with a report, so one user turn produced a report per plan in the tree instead
    of one for the turn. A non-empty intention stack is the mechanical answer (see
    ``InferAction.execute``): it means this plan is a sub-plan, and the reply belongs to whatever
    plan the user's goal is on. Empty stack renders nothing, leaving the system prompt's condition
    to apply as before."""
    if not activity.parent_frames:
        return ""
    return (
        "This goal is one step of a larger plan — it is NOT a request from the user, and the user "
        "is not waiting on its result. The plan that owns the user's goal reports back when the "
        "whole task is done, so do NOT end this plan by invoking `send_message_to_user`: a report "
        "here reaches the user as a second, partial answer to a question they asked once."
    )


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
        f"Goal: {activity.goal}\n"
        f"{_render_goal_provenance(activity)}\n"
        f"Available tools and their operations:\n{render_tools(tools)}\n\n"
        f"Currently observed properties:\n{render_properties(observed.properties)}\n\n"
        f"Recently observed signals:\n{render_signals(observed.signals)}\n\n"
        f"Results of operations already executed:\n"
        f"{render_history(activity.history, _HISTORY_RENDER_PLAN)}\n\n"
        f"Recent instructions from the user:\n{render_messages(messages or [])}"
    )
    if activity.superseded is not None:
        # Framing lives in the section itself, not PLAN_SYSTEM_PROMPT: a custom PlanPrompt that
        # omits the section then carries no dangling instruction about one that isn't there. Worded
        # as reusable material rather than as a negative example — told "this was wrong", a planner
        # avoids the parts that were fine too, which is the whole benefit lost.
        if activity.superseded.defect is None:
            framing = (
                "\n\nA previous plan for this goal was abandoned because the world moved; the "
                "observations above supersede the assumptions it was written against. Reuse "
                "whatever still applies and re-derive whatever does not — it is stale, not wrong "
                "throughout. The intermediate values its earlier steps had bound were discarded "
                "with it, so a remaining step that reads one must be re-derived rather than "
                "carried over as-is.\n"
            )
        else:
            # The opposite brief, and it has to be explicit about scope: told only "that was
            # wrong", a planner throws out the steps that were fine too (the same trap the
            # reconsideration wording above is worded around). So: name the one broken step, say
            # plainly that re-deriving it will not help, and bless the rest. Deliberately covers
            # every defect the runtime can detect rather than one — the remedies differ (a
            # different operation, a corrected param name) but the shape of the advice does not.
            framing = (
                "\n\nA previous plan for this goal was abandoned because one of its steps could "
                f"not be carried out: {activity.superseded.defect}. That is not a stale reading "
                "that a fresh look would fix — writing that step the same way again will fail in "
                "the same place, so the replacement has to differ THERE. Depending on what the "
                "problem was: reach the value by a different operation, or by a broader query "
                "narrowed afterwards with a filter step; take parameter names only from the tool "
                "catalog above, never from an operation's prose description. If what the step "
                "needed genuinely is not available anywhere, plan to tell the user so instead of "
                "proceeding without it — never substitute a value that merely looks plausible. "
                "Everything in the plan that did not depend on that step was fine and should be "
                "reused; the intermediate values its earlier steps had bound were discarded with "
                "it, so a remaining step that reads one must be re-derived rather than carried "
                "over as-is.\n"
            )
        user += f"{framing}{render_superseded_plan(activity.superseded)}"
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


def pending_from_raw(raw: dict[str, Any]) -> PendingCondition | None:
    """Convert one raw ``pending`` entry into a ``PendingCondition``, or None if it is unusable.

    A malformed condition is **dropped, not raised**: the body is the part that does the work, and
    failing a whole plan because the model mis-shaped an optional forward-looking clause would trade
    a partial success for a total one. Dropping is also the honest degradation — it lands the run
    back on the pre-conditions behavior (terminate when the body ends) rather than on a wait that
    can never fire.

    ``watch`` is required and must name a signal, because a condition without a mechanical gate is
    the unbounded keep-alive this design rejected: it would have to be evaluated against every
    signal the agent ever sees. A missing ``when``/``then`` is equally fatal — there would be
    nothing to judge or nothing to do.
    """
    if not isinstance(raw, dict):
        return None
    watch = raw.get("watch")
    when, then = raw.get("when"), raw.get("then")
    if not isinstance(watch, dict) or not isinstance(when, str) or not isinstance(then, str):
        return None
    signal_name = watch.get("signal") or watch.get("signal_name")
    if not isinstance(signal_name, str) or not signal_name:
        return None
    if not when.strip() or not then.strip():
        return None
    until = raw.get("until")
    return PendingCondition(
        watch=SignalWait(
            signal_name=signal_name,
            source=watch.get("source"),
            path=watch.get("path"),
        ),
        when=when,
        then=then,
        until=until if isinstance(until, str) and until.strip() else None,
    )


def _parse_plan_pending(text: str) -> tuple[PendingCondition, ...]:
    """The ``pending`` half of the plan contract. Absent means none — every plan predates this
    field, and a planner that declares nothing must keep working exactly as before."""
    try:
        data = _load_json_object(text)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return ()  # the steps parse above already raised on genuinely unparseable output
    raw = data.get("pending") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return ()
    parsed = (pending_from_raw(entry) for entry in raw)
    return tuple(cond for cond in parsed if cond is not None)


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


REPARSE_FEEDBACK = (
    "\n\nYour previous answer could not be parsed as JSON and was discarded. The parser reported:\n"
    "{error}\n\nHere is exactly what you returned:\n{output}\n\n"
    "Return the SAME answer with the syntax fixed — a single well-formed JSON object and nothing "
    "else, no prose and no markdown fences. Do not change what the answer says; only make it parse."
)


async def _complete_and_parse[T](
    llm: LLMClient,
    system: str,
    user: str,
    parse: Callable[[str], T],
    *,
    what: str,
) -> T:
    """One model call through the anti-corruption boundary, with a single retry that shows the model
    its own parse error.

    The last resort, and deliberately the *third* thing tried: a clean parse costs nothing, the
    structural repair in ``_load_json_object`` costs nothing, and only a defect neither of those can
    touch is worth a second round trip — which on a local reasoning model is minutes, not
    milliseconds. One retry, not a loop: a model that cannot produce well-formed JSON twice is not
    going to on the fifth attempt, and each attempt is charged to the same inference id (the
    contextvar the caller set), so a retried inference reports its true cost rather than hiding half
    of it. A retry that also fails raises, exactly as a single failed parse did before."""
    text = await llm.complete(system=system, prompt=user)
    try:
        return parse(text)
    except ValueError as exc:
        failure = str(exc)
        log.warning("reason: %s did not parse (%s) — retrying once with the error", what, failure)
    retry = user + REPARSE_FEEDBACK.format(error=failure, output=text)
    return parse(await llm.complete(system=system, prompt=retry))


def _drop_surplus_closers(text: str) -> str | None:
    """``text`` with every structurally impossible ``}``/``]`` removed, or None if there were none.

    Narrow on purpose. A closer that appears where nothing is open, or that closes the wrong kind of
    bracket, has *no* valid reading — deleting it cannot change which document was meant, because
    there was no document while it was there. That makes this a repair rather than a guess, and it
    is the only repair taken: an *unclosed* tail is left alone, because completing it would turn a
    truncated response into a shorter-but-plausible one, and a plan silently missing its last steps
    is far more dangerous than a plan that failed to parse.

    The motivating case is a 2712-character plan that died on one stray brace at the tail of its
    `pending` block — eight valid steps and a well-formed condition discarded, and the activity
    terminated, over a character with no meaning. Every repaired result is still handed to
    ``json.loads``, so nothing is trusted merely because it was repaired."""
    stack: list[str] = []
    kept: list[str] = []
    dropped = False
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and stack[-1] == ("{" if ch == "}" else "["):
                stack.pop()
            else:
                dropped = True  # nothing open, or the wrong kind — this character cannot be meant
                continue
        kept.append(ch)
    return "".join(kept) if dropped else None


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
    repaired = _drop_surplus_closers(stripped)
    if repaired is not None:
        try:
            return json.loads(repaired)
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
    "no markdown fences. Use only parameter names from the schema.\n"
    "If a reference names data that is NOT in what you were given — the operation it names "
    "returned an empty list, the field is absent, the binding is empty — then that value does not "
    "exist yet and you must NOT invent one. Do NOT substitute a nearby value that merely looks "
    "plausible: the user's own name or address, some other contact, a guessed date. The agent "
    "ACTS on what you "
    "return and many operations are irreversible, so a wrong-but-plausible recipient is far worse "
    "than an admitted gap — a gap can be recovered from, a sent email cannot. In that case respond "
    'with ONLY {"unresolvable": "<which parameter, and what was missing from the data>"} instead, '
    "and the runtime re-plans from it.\n"
    "This applies element by element inside a LIST parameter too: if an element's reference names "
    "data that is not there, report the gap for that parameter — do NOT quietly drop the element "
    "and return a shorter list. A list that comes back shorter than it went in is silently doing "
    "less than the step asked for, which reads as success and is not; the runtime rejects it.\n"
    "That is only for missing DATA. A value you can compute or phrase from what you WERE given is "
    "resolvable, so produce it: a $decide asking for a sentence about a result that is present, or "
    "a date derived from a clock reading in the history, are ordinary work, not gaps."
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


# A rendered result is either whole or explicitly absent — never a prefix. These bound the *total*
# characters a history/bindings section contributes to a prompt; an entry that does not fit is
# replaced by a counted placeholder rather than cut, because a half-rendered record is worse than a
# missing one: it reads as complete, and grounding will happily resolve a reference against a
# record whose tail (an id, an address, a `total:` that says "there are 115 more") was silently
# removed. Bounding the prompt is a legitimate job for an arbitrary number; deciding how much of a
# tool result is true is not.
#
# 60k chars is *tighter* than what the previous per-entry cap allowed (40 entries x 4000 = 160k) —
# the old cap paid a correctness price for a bound it did not actually deliver.
_HISTORY_CHAR_BUDGET = 60_000

# Bindings are rendered alongside history in the grounding prompt, so they get their own smaller
# budget rather than sharing one: a data-op binding is a narrowed collection (already the product of
# a filter/take), so it is expected to be small, and letting it compete with history for one pooled
# budget would let a single wide binding evict the operation results grounding resolves against.
_BINDINGS_CHAR_BUDGET = 20_000


def _one_line(value: Any) -> str:
    """Whitespace-collapsed rendering of a result. Length is handled by the budget walk, not here —
    this only flattens, it never elides."""
    return " ".join(str(value).split())


def _fit_to_budget(entries: Sequence[tuple[str, str | None, str]], budget: int) -> list[str]:
    """Render ``(name, args, body)`` entries newest-last within a total character ``budget``.

    All-or-nothing per entry: one that does not fit becomes a counted placeholder naming its size,
    so the reader can tell "elided, re-invoke" from "never happened". The walk runs newest-first
    because the newest entry is what a ``$from``/``$decide`` resolves against in the common case,
    and **the newest entry is always rendered in full** even when it alone exceeds the budget — a
    budget able to elide it would reintroduce the silent-loss bug in a new costume. Output order is
    the caller's original order, so reading order is unchanged.
    """
    rendered: list[str] = []
    spent = 0
    for position, (name, args, body) in enumerate(reversed(entries)):
        line = f"- {name}({args}) -> {body}" if args is not None else f"- {name} = {body}"
        if position == 0 or spent + len(line) <= budget:
            spent += len(line)
            rendered.append(line)
            continue
        noun = "result" if args is not None else "value"
        head = f"- {name}({_truncate(args, 120)}) -> " if args is not None else f"- {name} = "
        rendered.append(f"{head}({noun} elided: {len(body)} chars — re-invoke to view)")
    return rendered[::-1]


def render_history(
    history: list[CompletedOperation],
    limit: int | None = None,
    *,
    budget: int = _HISTORY_CHAR_BUDGET,
) -> str:
    """Render an activity's executed operations + results for a grounding prompt. Public so a custom
    ``GroundPrompt`` can reuse it. Results are the grounder's ground truth for resolving a reference
    (the id a ``$from``/``$decide`` needs to pick), so an individual result is **never** cut: it is
    rendered whole or replaced by a counted placeholder, under a total ``budget`` (see
    ``_fit_to_budget``). A partially rendered record is the worst outcome here — a contact record
    cut mid-field yields a truncated email address that still looks like an email address, and the
    paging metadata saying how many records were withheld lives in the tail a cut removes.

    ``limit`` keeps only the most recent N entries, with a counted marker in place of the rest so a
    reader can tell elision from "nothing happened earlier". It defaults to *unbounded* because the
    grounding caller must not have it: grounding resolves a reference against an arbitrary past
    result, and dropping the entry holding the referent is the same class of failure as truncating
    it mid-record. Callers that judge *what to do next* rather than resolve a reference pass a
    window, since history is append-only for the life of an activity and would otherwise grow every
    prompt without bound."""
    if not history:
        return "(nothing executed yet)"
    shown = history if limit is None or len(history) <= limit else history[-limit:]
    lines = []
    if len(shown) < len(history):
        lines.append(f"(… {len(history) - len(shown)} earlier operation(s) not shown)")

    entries = []
    for completed in shown:
        outcome = completed.ack.result if completed.ack.ok else f"ERROR: {completed.ack.result}"
        args = json.dumps(completed.invocation.params)
        entries.append((completed.invocation.operation_name, args, _one_line(outcome)))
    lines.extend(_fit_to_budget(entries, budget))
    return "\n".join(lines)


def render_steps(steps: list[Step]) -> str:
    """Render plan steps one per line as ``index: action {params}``. Shared by the revalidation
    prompt and the runtime's own plan-level DEBUG log so both read alike — the same rendering, not
    the same text: an index is a position in whatever list the caller passes, and the prompt passes
    only the remaining tail (flattened across the sub-goal stack, re-indexed from 0) while the log
    passes a whole plan body. ``(none)`` for an empty list — an inferred plan can legitimately be
    empty."""
    return (
        "\n".join(
            f"{index}: {step.next_action} {json.dumps(step.params, default=str)}"
            for index, step in enumerate(steps)
        )
        or "(none)"
    )


def render_pending(pending: tuple[PendingCondition, ...]) -> str:
    """Render a plan's declared pending conditions, one block per condition, or ``""`` when there
    are none — so a caller can append it to ``render_steps`` unconditionally and a body-only plan
    renders exactly as it did before.

    Separate from ``render_steps`` rather than folded into it because the two have different
    audiences: ``render_steps`` is also the revalidation prompt's rendering and is handed a step
    *tail*, which has no plan and therefore no conditions to show. Kept in the trace at all because
    a declared condition is otherwise invisible — the body is what executes, so a plan that silently
    failed to declare a gate and one that declared a good gate produce identical logs, and telling
    those apart is the whole question when a run ends early."""
    if not pending:
        return ""
    lines = ["pending:"]
    for index, condition in enumerate(pending):
        watch = {
            "signal": condition.watch.signal_name,
            "source": condition.watch.source,
            "path": condition.watch.path,
        }
        lines.append(f"{index}: watch {json.dumps(watch, default=str)}")
        lines.append(f"   when  {condition.when}")
        lines.append(f"   then  {condition.then}")
        if condition.until is not None:
            lines.append(f"   until {condition.until}")
    return "\n".join(lines)


def render_plan(plan: Plan) -> str:
    """A whole plan body for the DEBUG trace: steps, then any declared pending conditions."""
    return "\n".join(filter(None, (render_steps(plan.steps), render_pending(plan.pending))))


def remaining_steps(
    plan: Plan | None, step_index: int, parent_frames: list[tuple[Plan, int]]
) -> list[Step]:
    """The un-run tail of a plan, flattened across the sub-goal stack (ADR-0022): the active frame
    from ``step_index``, then each suspended parent's steps after the sub-goal that pushed it —
    innermost frame first, since that is the order they resume in. Shared by the revalidation prompt
    (which asks *is all of this still valid*, and would call a sub-plan valid while a stale parent
    step still waits) and the superseded-plan rendering, so both mean the same thing by "remaining".
    Public so a custom prompt can reuse it."""
    tail = list(plan.steps[step_index:]) if plan is not None else []
    for parent_plan, subgoal_index in reversed(parent_frames):
        tail.extend(parent_plan.steps[subgoal_index + 1 :])
    return tail


def render_superseded_plan(superseded: SupersededPlan) -> str:
    """Render the discarded plan's un-run tail for the replanning prompt (ADR-0024). Only the tail:
    what already ran is rendered separately as history by the same prompt, and repeating it as
    *intent* alongside its *results* would be redundant weight. Flattened across the stack, so a
    discard taken inside a sub-plan still shows the suspended parents' un-run steps — the part of
    the loss a frame-local view would hide. Public so a custom ``PlanPrompt`` can reuse it."""
    tail = remaining_steps(superseded.plan, superseded.step_index, superseded.parent_frames)
    frames = len(superseded.parent_frames)
    nested = f", inside a sub-plan nested under {frames} suspended frame(s)" if frames else ""
    # Deliberately no absolute step number beside the listing: render_steps numbers whatever it is
    # given from 0, so quoting the discarded plan's own index here would invite reading a *listed*
    # step as one already executed. State how many ran, and where their results are, instead.
    return (
        f"It had already executed {superseded.step_index} step(s) of that plan{nested}; their "
        f"results appear above under the executed operations and are not repeated here.\n"
        f"Its remaining, unexecuted steps were (numbered afresh from 0):\n{render_steps(tail)}"
    )


def render_bindings(bindings: dict[str, Any], *, budget: int = _BINDINGS_CHAR_BUDGET) -> str:
    """Render an activity's named data-op bindings for a grounding or plan-validity prompt. Public
    so a custom ``GroundPrompt`` can reuse it. A ``$bind`` reference or a ``$decide`` instruction
    may name a binding an earlier data-op step produced (a filtered/sorted/collected collection),
    so — like history results — these are the grounder's ground truth for resolving a reference and
    get the same all-or-nothing treatment under a total ``budget``: a binding renders whole or
    becomes a counted placeholder, never a prefix. Insertion order is preserved in the output, but
    the budget walk runs in reverse, so the most recently produced binding is the one guaranteed to
    survive.

    ``revalidate`` renders them for a different reason: a data-op writes no history, so bindings are
    the only record that its steps ran at all. They are deliberately absent from the *plan* prompt,
    where they would be a lie — a replan discards them."""
    if not bindings:
        return "(none)"
    entries = [
        (name, None, _one_line(json.dumps(value, default=str))) for name, value in bindings.items()
    ]
    return "\n".join(_fit_to_budget(entries, budget))


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
        # No goal-provenance notice here, unlike default_plan_prompt: it is advice about how to end
        # a *plan* ("do NOT invoke send_message_to_user"), which grounding one operation's params
        # cannot act on, and this activity's `goal` is the top-level one even mid-sub-plan — so
        # rendering it would tell the grounder the user's own request is not from the user.
        f"Goal: {activity.goal}\n"
        f"Operation to invoke:\n{_render_operation_schema(manual, operation_name)}\n\n"
        f"Partial parameters (resolve any references, keep concrete values):\n"
        f"{json.dumps(partial_params, indent=2)}\n\n"
        f"Currently observed properties:\n{render_properties(observed.properties)}\n\n"
        f"Recently observed signals:\n{render_signals(observed.signals)}\n\n"
        f"Named data-op bindings (a $bind reference or a $decide instruction may name one):\n"
        f"{render_bindings(activity.bindings)}\n\n"
        # Deliberately unwindowed: a $from/$decide reference may name any past result, and hiding
        # the entry that holds the referent fails the same way truncating it mid-record does.
        f"Results of operations already executed:\n{render_history(activity.history)}"
    )
    return GROUND_SYSTEM_PROMPT, user


# The second legal answer of every escalation asked to RESOLVE a reference: the data that reference
# names is not present. Shared by grounding ("params") and the $decide filter predicate ("keep"),
# because it is one contract — the model reports the gap rather than inventing something to fill it.
# Checked before the primary key so a response carrying both (a model hedging) is read as the gap it
# reports: the other half of such an answer is exactly the fabrication this channel exists to
# prevent, so the gap is always the safe reading.
_UNRESOLVABLE = "unresolvable"


def _parse_params(text: str) -> dict[str, Any]:
    try:
        data = _load_json_object(text)
        if isinstance(data, dict) and data.get(_UNRESOLVABLE):
            raise UnresolvableGrounding(str(data[_UNRESOLVABLE]))
        params = data["params"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(
            f"could not parse grounded params from model output: {exc!r}\n---\n{text}"
        ) from exc
    if not isinstance(params, dict):
        raise ValueError(f"grounded params is not a JSON object: {params!r}")
    return params


def _check_no_dropped_elements(partial_params: dict[str, Any], params: dict[str, Any]) -> None:
    """Refuse a grounding that returned a *shorter* list than the step supplied.

    ``unresolvable`` is the honest answer for a reference whose data never materialized, but a list
    parameter offers a third way out the prompt cannot fully close: keep the shape, drop the
    element. That answer parses, looks like success, and gets invoked — an event created with
    ``attendees: []`` when the step asked for one attendee is not a partial success, it is a wrong
    write that nothing downstream will question. The count is a pre-image the runtime already
    holds, so checking it costs nothing and needs no model.

    Growth is fine and expected: one reference element can resolve to many values. Only shrinkage
    is a claim that something asked for is not there — which is what ``unresolvable`` is *for*, so
    it is raised here on the model's behalf and drives the same replan."""
    for name, before in partial_params.items():
        if not isinstance(before, list) or not before:
            continue
        after = params.get(name)
        kept = len(after) if isinstance(after, list) else 0
        if kept < len(before):
            raise UnresolvableGrounding(
                f"{name}: the step supplied {len(before)} element(s) and grounding returned "
                f"{kept} — the data the missing element(s) referenced is not present in this run, "
                f"so the step cannot be carried out as written"
            )


# --- revalidate: the context-adaptation plan-validity re-check — ADR-0024 ------------------------

REVALIDATE_SYSTEM_PROMPT = (
    "You are deciding whether an IN-PROGRESS plan is still VALID given the latest observations. "
    "You are given the goal, what the agent has ALREADY DONE (the operations executed so far with "
    "their results, and the intermediate values its earlier steps computed and named), the plan's "
    "REMAINING steps, and the new observed state and messages. "
    "The plan is INVALID if the new information changes what the remaining steps should do — a "
    "follow-up that changes a detail the plan acted on, or a precondition that no longer holds. "
    "It is VALID if the work already executed plus the remaining steps still achieve the goal; "
    "the agent's OWN prior actions do not by themselves invalidate it. Judge the remaining steps "
    "only: work an executed operation already accomplished, or a value already computed and named, "
    "does not have to reappear in them — a remaining step that reads such a value is satisfied by "
    "it, not evidence of a gap.\n"
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


# --- pending conditions: the batched "did any of these fire?" judgement — ADR-0022 --------------

CONDITION_SYSTEM_PROMPT = (
    "You are deciding whether an agent's declared FOLLOW-UP CONDITIONS have come true. The agent "
    "finished a task and is waiting in case something specific happens. Something changed in its "
    "environment; you decide whether that change is what it was waiting for.\n"
    "You are given the original goal, and a numbered list of conditions. Each has a `when` (what "
    "the agent is waiting for) and, optionally, an `until` (when it should stop waiting). You are "
    "also given the observed change and the current state it landed in.\n"
    "For each condition decide, independently:\n"
    "  - FIRED: the `when` has actually happened, judged from the observed state. Be strict — the "
    "change reaching the agent is only a prompt to look; most changes are not the awaited event, "
    "and a wrong `fired` makes the agent redo work nobody asked for.\n"
    "  - RETIRED: the `until` is now satisfied, so the agent should stop waiting on it. A "
    "condition with no `until` is retired only if waiting has become pointless.\n"
    "A condition can be neither (the usual answer: keep waiting), or both.\n"
    "Judge FIRED for each condition on its own, but retirement is not always independent. "
    "Conditions listed together usually come from ONE clause of the goal, and often describe "
    'MUTUALLY EXCLUSIVE branches of it ("if they propose an alternative" / "if they propose '
    'none"). Firing one branch settles the others: also retire any condition whose `when` can no '
    "longer happen given what you just judged to be true — a branch that has been overtaken is not "
    "still waiting, it is decided. Retire on that logical incompatibility only, never because a "
    "condition merely looks less likely now, and never to tidy up.\n"
    'Respond with ONLY a JSON object {"fired": [<indices>], "retired": [<indices>]} — 0-based '
    "indices into the numbered list, no prose, no fences. Use empty lists when nothing applies."
)


def _parse_condition_verdict(text: str, count: int) -> ConditionVerdict:
    """Parse the batched ``{"fired": [...], "retired": [...]}`` answer.

    Degrades to "nothing happened" on malformed output rather than raising: the activity then keeps
    waiting, which is the same state it was already in. The opposite default would let a flaky call
    invent follow-up work — the expensive, user-visible failure. Out-of-range indices are dropped
    for the same reason, so a hallucinated index cannot retire a condition that is still live.
    """
    try:
        obj = _load_json_object(text)
    except (ValueError, TypeError, AttributeError):
        return ConditionVerdict()

    def _indices(key: str) -> tuple[int, ...]:
        raw = obj.get(key) if isinstance(obj, dict) else None
        if not isinstance(raw, list):
            return ()
        out: list[int] = []
        for entry in raw:
            if isinstance(entry, bool) or not isinstance(entry, int):
                continue  # bool is an int subclass; a `true` here is not index 1
            if 0 <= entry < count and entry not in out:
                out.append(entry)
        return tuple(out)

    return ConditionVerdict(fired=_indices("fired"), retired=_indices("retired"))


# --- undeclared relevance: does a change bear on work that already finished? — ADR-0026 ----------

RELEVANCE_SYSTEM_PROMPT = (
    "You are deciding whether something that just changed in an agent's environment means a task "
    "it ALREADY FINISHED needs following up.\n"
    "You are given a numbered list of recently finished tasks (what each was trying to do, how it "
    "went) and a description of what just changed. Decide whether the change is a genuine "
    "follow-up to exactly one of those tasks — a reply to a message it sent, a cancellation of "
    "something it arranged, a rejection of something it submitted.\n"
    "Answer NO unless the connection is specific and concrete. Most changes are unrelated "
    "background activity, and the agent's own past actions often cause changes that follow from "
    "work it already completed correctly — those are not follow-ups. A wrong YES interrupts a "
    "person with a question about work that was already done properly.\n"
    'Respond with ONLY a JSON object. If nothing follows up: {"relevant": false}. Otherwise:\n'
    '  {"relevant": true, "task": <index>, "goal": "<what the agent should now do about it>", '
    '"question": "<a one-sentence question asking the user whether to do it>"}\n'
    "`goal` is phrased like a task instruction. `question` is addressed to the user, states what "
    "changed and what you propose, and must be answerable with yes or no. No prose, no fences."
)


def _parse_relevance(text: str, episodes: Sequence[Any]) -> RelevanceCandidate | None:
    """Parse the relevance judgement, or None for "nothing follows up".

    Degrades to None on anything malformed or out of range. That asymmetry is deliberate: a missed
    follow-up costs a silent gap the user can still notice and correct, while a fabricated one
    interrupts them with a question about work that was already finished properly — and this is the
    one layer whose judgement nothing downstream can mechanically check.
    """
    try:
        obj = _load_json_object(text)
    except (ValueError, TypeError, AttributeError):
        return None
    if not isinstance(obj, dict) or not obj.get("relevant"):
        return None
    index = obj.get("task")
    if isinstance(index, bool) or not isinstance(index, int):
        return None
    if not 0 <= index < len(episodes):
        return None
    goal, question = obj.get("goal"), obj.get("question")
    if not isinstance(goal, str) or not goal.strip():
        return None
    if not isinstance(question, str) or not question.strip():
        return None
    episode = episodes[index]
    episode_id = episode.get("activity_id") if isinstance(episode, dict) else None
    if not isinstance(episode_id, str):
        return None
    return RelevanceCandidate(episode_id=episode_id, goal=goal.strip(), question=question.strip())


# --- select: the model-escalated data-op filter predicate ($decide) — ADR-0023 -------------------

SELECT_SYSTEM_PROMPT = (
    "You are filtering a list down to the subset that satisfies a natural-language predicate. You "
    "are given the goal, the predicate, the agent's execution context (the results of operations "
    "already executed, the named data-op bindings, and the observed world state), and finally the "
    "list items, each on its own line prefixed by its 0-based index.\n"
    "The predicate may NAME a value from that context instead of spelling it out — 'the Saturday "
    "after the get_current_time result', 'whoever is in the shortlist binding'. Resolve every such "
    "reference against the context FIRST, down to a concrete value, and only then test the items "
    "against it. Whether an individual item is kept is still judged on ITS OWN data; the context "
    "supplies the values the predicate compares against, never a reason to keep an item.\n"
    'Respond with ONLY a JSON object of the form {"keep": [<indices>]} — the 0-based indices of '
    "the items to KEEP — and nothing else, no prose, no markdown fences. Keep an item only if it "
    'clearly satisfies the predicate; if none do, respond {"keep": []}.\n'
    "There is a second legal answer, for one specific case: the predicate names something the "
    "context does NOT contain — a result from an operation that never ran or came back empty, a "
    "binding that is not there — so you cannot work out what to compare the items against. Do NOT "
    "guess the missing value, and do NOT fall back on an empty keep-list: an empty answer means "
    "'no item qualified', the agent acts on it as a real result, and a whole clause of the task "
    "then goes silently undone. Respond with ONLY "
    '{"unresolvable": "<what the predicate names, and what was missing from the context>"} '
    "instead, and the runtime re-plans from it.\n"
    "That is only for missing CONTEXT. A predicate you CAN evaluate from what you were given is "
    "ordinary work, however much judgement it takes — including one whose honest answer is that no "
    'item qualifies. Answer {"keep": []} for that, not "unresolvable".'
)


def _parse_keep(text: str, count: int) -> list[int]:
    """Parse the ``{"keep": [<indices>]}`` selection contract into the in-range indices to keep,
    preserving the model's order. Out-of-range or non-integer entries are dropped (defensive against
    a stray index) and a repeated index collapses to its first occurrence (so the filtered subset
    never duplicates an item — a duplicate would double-act a downstream fan-out); a structurally
    malformed answer raises, the anti-corruption boundary that infer/ground share.

    The selection's *second* legal answer, ``{"unresolvable": ...}``, is checked first and raises
    ``UnresolvableGrounding`` — the same channel and the same precedence rule grounding uses, since
    an answer carrying both is a hedge whose keep-list is the guess the channel exists to stop."""
    try:
        data = _load_json_object(text)
        if isinstance(data, dict) and data.get(_UNRESOLVABLE):
            raise UnresolvableGrounding(str(data[_UNRESOLVABLE]))
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


# How many *entries* of an activity's history each prompt renders. History is append-only for the
# whole life of an activity, so without a window every prompt grows with the activity — unbounded
# in the long-lived, asynchronous setting this runtime is built for, not just in a long plan.
#
# Revalidation gets the tight window. It fires before every gate-hot write, and ADR-0024 justifies
# it as much cheaper than the re-plan it guards; letting it carry the same history as a planning
# prompt is what would erode that. It also needs the least: it answers one yes/no question about
# the remaining tail, where the recent results carry the signal and "what is already done" is
# implied by the tail's own contents. 10 matches _MESSAGE_RENDER, the existing recent-window
# precedent for a prompt section whose storage is uncapped.
_HISTORY_RENDER_REVALIDATE = 10

# Planning gets the generous one: it runs once per plan rather than per write, and a re-plan must
# not silently lose a fact the new plan depends on. 40 is _DEFAULT_MAX_SUBGOAL_DEPTH (4) frames of
# roughly ten steps each — the deepest history a single plan tree can currently produce — so a
# re-plan still sees the whole of its predecessor's work, and only an activity outliving many
# plans is trimmed at all.
_HISTORY_RENDER_PLAN = 40


def _truncate(value: Any, limit: int = _TRUNCATE_LIMIT) -> str:
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

    @property
    def model(self) -> str | None:
        """The name of the model behind infer()/ground(), for a run surface to record in its trace.
        Read off the client rather than stored here: bootstrap knows the name from config and puts
        it on the instrumentation wrapper it builds. Optional by design — `LLMClient` is one method
        wide, so a client that names no model (and a memory with no model at all) answers None."""
        model = getattr(self._llm, "model", None)
        return model if isinstance(model, str) else None

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

        def _to_plan(text: str) -> Plan:
            # Both halves parse from the same object, so they share the one retry: a plan recovered
            # without its declared conditions would look complete while quietly dropping the gate.
            return Plan(
                id=uuid.uuid4().hex,
                goal=activity.goal,
                steps=_parse_plan_steps(text),
                pending=_parse_plan_pending(text),
            )

        return await _complete_and_parse(self._llm, system, user, _to_plan, what="plan inference")

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
        params = _parse_params(text)
        _check_no_dropped_elements(partial_params, params)
        return params

    async def select(
        self,
        activity: Activity,
        collection: list[Any],
        predicate: str,
        observed: PerceptSnapshot | None = None,
    ) -> list[Any]:
        """Filter ``collection`` to the subset satisfying a natural-language ``predicate`` — the
        model-escalated ``$decide`` half of the ``filter`` data-op (ADR-0023). One model call over
        the whole collection (a batching simplification of the per-element ideal): it renders each
        item with its index, asks for a ``{"keep": [<indices>]}`` answer, and returns the kept items
        in the model's order. The index contract keeps the model from re-serializing (and mangling)
        the items. Reuses the same ``LLMClient`` seam as ``infer``/``ground``; no LLM -> raises.

        A ``$decide`` predicate is judged against **the same context ``ground`` gets** — history,
        bindings, properties, signals — and that is load-bearing rather than symmetry for its own
        sake. The planner is explicitly taught to write predicates that name an earlier result
        rather than a literal ("the first Saturday on or after the get_current_time result"), which
        is the correct instruction: a date baked in at plan time is a guess. Rendering only the
        goal, the predicate and the items therefore handed the model a reference whose referent was
        nowhere in the prompt. It cannot resolve that, and its honest answer is ``{"keep": []}`` —
        which is indistinguishable from "nothing matched", lands in a binding, and reads downstream
        as a real answer. One gaia2 run lost an entire cancellation clause that way, silently.

        History is deliberately **unwindowed**, for the reason ``default_ground_prompt`` gives: the
        predicate may name any past result, and hiding the entry holding the referent fails exactly
        the way truncating it mid-record does.
        """
        if self._llm is None:
            raise RuntimeError(
                "ProceduralMemory has no LLM configured; cannot evaluate a $decide filter. Pass a "
                "client."
            )
        snapshot = observed or PerceptSnapshot()
        items = "\n".join(
            f"{index}: {json.dumps(item, default=str)}" for index, item in enumerate(collection)
        )
        user = (
            f"Goal: {activity.goal}\n"
            f"Predicate: {predicate}\n\n"
            f"Results of operations already executed:\n{render_history(activity.history)}\n\n"
            f"Named data-op bindings (the predicate may name one):\n"
            f"{render_bindings(activity.bindings)}\n\n"
            f"Currently observed properties:\n{render_properties(snapshot.properties)}\n\n"
            f"Recently observed signals:\n{render_signals(snapshot.signals)}\n\n"
            # Items last: it is by far the largest section, and the context above is what the
            # predicate's references resolve against, so it should be read first.
            f"Items:\n{items}"
        )
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
        — the context-adaptation relevance judgment (ADR-0024). One model call: the goal, the
        operations already executed, the plan's REMAINING steps, and the observed
        properties/signals plus recent messages, asking for a ``{"valid": bool}`` verdict.
        The executed half is not decoration: a checkpoint late in a plan leaves almost nothing in
        ``remaining``, so without history the model sees a goal whose work is nowhere in evidence
        and reasonably calls a nearly-finished plan invalid — it is also what makes "the agent's
        own prior actions don't invalidate it" a judgment it can actually make rather than guess.
        The named bindings go with it: a data-op writes no history at all, so a plan that spent its
        early steps narrowing a collection would otherwise arrive here looking like it had done
        nothing, and get replanned into a copy of itself.
        Reuses the same ``LLMClient`` seam as ``infer``; no
        LLM -> raises. A ``False`` verdict re-infers; best-effort, not a guarantee, and
        deliberately general (no domain-authored predicate) — it reasons about relevance itself, so
        the agent's own writes don't spuriously invalidate the plan."""
        if self._llm is None:
            raise RuntimeError(
                "ProceduralMemory has no LLM configured; cannot revalidate a plan. Pass a client."
            )
        snapshot = observed or PerceptSnapshot()
        # The full remaining tail across the sub-goal stack, not just the active sub-plan (ADR-0022)
        # — checking only the sub-plan would call it "still valid" while a stale parent step waits.
        steps_text = render_steps(
            # Rendering never needs a frame's history_mark — that is live collect scoping — so the
            # tail is read in the same (plan, step_index) shape a SupersededPlan carries.
            remaining_steps(
                activity.plan,
                activity.step_index,
                [(plan, index) for plan, index, _ in activity.parent_frames],
            )
        )
        user = (
            f"Goal: {activity.goal}\n"
            f"Results of operations already executed:\n"
            f"{render_history(activity.history, _HISTORY_RENDER_REVALIDATE)}\n"
            f"Intermediate values already computed:\n{render_bindings(activity.bindings)}\n"
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

    async def evaluate_conditions(
        self,
        activity: Activity,
        conditions: Sequence[PendingCondition],
        changes: Sequence[tuple[str, Change]],
        observed: PerceptSnapshot | None = None,
    ) -> ConditionVerdict:
        """Judge, in ONE call, which of an activity's eligible pending conditions have fired or
        retired (ADR-0022).

        Batched by design: the alternative is a call per condition per signal, and the whole point
        of the mechanical `watch` gate is to make the model see only changes that already passed a
        cheap filter. Batching is what keeps that saving as conditions accumulate.

        The `changes` are rendered as *where* to look, and `render_changes` then dereferences their
        ids against the observed property snapshot to show the records themselves — which is the
        division of labour a located change summary exists to create: the judgement reads one named
        path instead of re-scanning a whole property. Each change is paired with the **source** that
        reported it, because a bare `Change` says which path moved but not on which tool, and both
        are needed to find the value. Note the dereference is what makes the whole layer work at
        all: the judgement is irreducibly semantic ("did he decline?"), so a prompt carrying only
        opaque ids and a shape sketch of the property cannot answer it, and answers "nothing fired".
        """
        if self._llm is None:
            raise RuntimeError(
                "ProceduralMemory has no LLM configured; cannot evaluate pending conditions."
            )
        if not conditions:
            return ConditionVerdict()
        snapshot = observed or PerceptSnapshot()
        listed = "\n".join(
            f"{i}. when: {c.when}\n   until: {c.until or '(no explicit bound)'}"
            for i, c in enumerate(conditions)
        )
        user = (
            f"Original goal: {activity.goal}\n"
            f"Conditions:\n{listed}\n"
            f"What changed:\n{render_changes(changes, snapshot.properties)}\n"
            f"Current observed properties:\n{render_properties(snapshot.properties)}\n"
            f"Recently observed signals:\n{render_signals(snapshot.signals)}"
        )
        log.debug(
            "reason: condition system prompt\n%s\nUser prompt\n%s", CONDITION_SYSTEM_PROMPT, user
        )
        text = await self._llm.complete(system=CONDITION_SYSTEM_PROMPT, prompt=user)
        return _parse_condition_verdict(text, len(conditions))

    async def judge_relevance(
        self,
        episodes: Sequence[Any],
        changes: Sequence[tuple[str, Change]],
        observed: PerceptSnapshot | None = None,
    ) -> RelevanceCandidate | None:
        """Does an observed change bear on work that already finished? (ADR-0026)

        The expensive layer, and deliberately the last resort: unlike every other match in this
        runtime there is no declared thing to compare against, so the judgement is unverifiable from
        the runtime's side. Its input is only what the declared-condition gates left unclaimed, so
        the cost shrinks as the planner learns to declare — the two are complements.

        Shares `render_changes` with `evaluate_conditions`, and for a sharper reason: that judge is
        at least handed a `when` clause naming what to look for, while this one is asked the open
        question against episode *summaries*. Answered from ids beside a shape-sketched property,
        it can only say "nothing follows up" — so each change arrives paired with the **source**
        that reported it, which is what says which property to dereference the ids against.
        """
        if self._llm is None:
            raise RuntimeError(
                "ProceduralMemory has no LLM configured; cannot judge undeclared relevance."
            )
        if not episodes:
            return None
        snapshot = observed or PerceptSnapshot()
        listed = "\n".join(
            f"{i}. goal: {e.get('goal', '(unknown)')}\n"
            f"   outcome: {'succeeded' if e.get('succeeded') else 'failed'}\n"
            f"   summary: {e.get('summary', '(none)')}"
            for i, e in enumerate(episodes)
            if isinstance(e, dict)
        )
        user = (
            f"Recently finished tasks:\n{listed}\n"
            f"What just changed:\n{render_changes(changes, snapshot.properties)}\n"
            f"Current observed properties:\n{render_properties(snapshot.properties)}"
        )
        log.debug("relevance: system prompt\n%s\nUser prompt\n%s", RELEVANCE_SYSTEM_PROMPT, user)
        text = await self._llm.complete(system=RELEVANCE_SYSTEM_PROMPT, prompt=user)
        return _parse_relevance(text, episodes)

    async def store(self, plan: Plan) -> None:
        """Persists a Plan that was actually followed to completion, so future retrieve() calls
        for similar goals can reuse it. Called by ReflectStrategy on success only — a failed plan
        isn't something future activities should retrieve by default."""
        await self._backend.put(plan.id, asdict(plan))

    @staticmethod
    def _from_dict(data: dict[str, Any]) -> Plan:
        # Rebuild the dataclass graph the backend flattened to plain dict/list/scalar on store.
        # `pending` needs rebuilding too, and its nested SignalWait with it: a JSON round-trip
        # leaves plain dicts, so skipping this would hand back a "plan" whose declared conditions
        # are dicts that no wait can ever match — the conditions would look present and silently
        # never fire. Absent (a plan stored before conditions existed) means none.
        return Plan(
            id=data["id"],
            goal=data["goal"],
            steps=[Step(**step) for step in data["steps"]],
            pending=tuple(
                PendingCondition(
                    watch=SignalWait(**cond["watch"]),
                    when=cond["when"],
                    then=cond["then"],
                    until=cond.get("until"),
                )
                for cond in (data.get("pending") or ())
            ),
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
                # When this episode closed. Stored because recency is not otherwise recoverable:
                # the backend orders non-ranking results by key (the activity id), which says
                # nothing about time, so without this "the most recently finished episodes" cannot
                # be asked for at all. Re-learning the same activity overwrites, so this is the
                # *last* close, which is what a recency query wants.
                "ended_at": time.time(),
            },
        )

    async def consult(self, activity: Activity) -> list[Any]:
        # query() re-reads from disk, so results are fresh copies a caller can mutate without
        # corrupting the store.
        return await self._backend.query(goal=activity.goal)

    async def consult_recent(self, limit: int) -> list[Any]:
        """The most recently closed episodes, newest first — the disambiguated sibling of consult().

        consult() retrieves by goal-equality, which is exactly wrong for a caller holding an
        observed *change* rather than a goal and asking which recently-closed work it might bear on
        (ADR-0026). Sorting happens here rather than in the backend because `query()`'s ordering
        contract is relevance, not recency, and a ranking backend is free to ignore both. An episode
        written before `ended_at` existed sorts oldest rather than crashing.
        """
        episodes = await self._backend.query()
        episodes.sort(key=lambda e: e.get("ended_at") or 0.0, reverse=True)
        return episodes[: max(0, limit)]
