"""Working, semantic, procedural, and episodic memory modules."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
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
from sora.types import Plan, Step

if TYPE_CHECKING:
    from sora.activity import Activity
    from sora.environment import EnvironmentView, Tool
    from sora.llm import LLMClient
    from sora.perception import Message, Percept


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
    # stimuli from the environment: properties and signals only
    perceptions: list[Percept] = field(default_factory=list)
    # inbound agent-to-agent communication — kept distinct
    messages: list[Message] = field(default_factory=list)
    focused_tools: dict[str, Tool] = field(default_factory=dict)
    # manuals pulled from SemanticMemory by _load_ (removed by _unload_) — distinct from
    # focused_tools: focusing a tool is an external action, loading its manual is internal.
    loaded_manuals: dict[str, Manual] = field(default_factory=dict)


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
    # `send` disabled until we have proper communication support:
    #    '  {"action": "send", "to": "<recipient>", "content": { ... }}\n'
    'A step with no "action" is treated as "invoke". Use only tool ids and operation names that '
    "appear in the provided tool list. Invoking an operation does not require focusing the tool "
    "first — focus a tool only to perceive its observable properties and signals, and unfocus once "
    "you no longer need them. Respect any usage protocols & safety constraints listed for a tool "
    "when choosing and ordering steps."
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
        lines += _render_affordances(
            "operations (invoke)",
            "operation",
            [(op.name, op.description) for op in manual.operations],
            manual.section(ManualSection.OPERATIONS),
        )
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


class PlanPrompt(Protocol):
    """Builds the ``(system, user_prompt)`` pair ``infer()`` sends to the LLM, from the activity and
    the available tools. Injected into ``ProceduralMemory`` so planning *content* is customizable —
    a domain system prompt, few-shot examples, a different catalog rendering — without subclassing
    or moving planning into a ``ReasonStrategy``. A plain function satisfies it; a class with
    ``__call__`` works too (a stateful builder). Whatever it produces, the model's response must
    still parse as the ``{"steps": [...]}`` contract — that half is fixed (``_parse_plan_steps``).
    """

    def __call__(self, activity: Activity, tools: dict[str, Manual]) -> tuple[str, str]: ...


def default_plan_prompt(activity: Activity, tools: dict[str, Manual]) -> tuple[str, str]:
    """The built-in ``PlanPrompt``: the fixed ``PLAN_SYSTEM_PROMPT`` (the JSON step vocabulary) plus
    a user prompt rendering the goal and the tool catalog. Reuse ``PLAN_SYSTEM_PROMPT`` /
    ``render_tools`` when writing a custom one."""
    user = f"Goal: {activity.goal}\n\nAvailable tools and their operations:\n{render_tools(tools)}"
    return PLAN_SYSTEM_PROMPT, user


def _parse_plan_steps(text: str) -> list[Step]:
    try:
        data = json.loads(_strip_code_fences(text))
        raw_steps = data["steps"]
        steps: list[Step] = []
        for raw in raw_steps:
            action = raw.get("action", InvokeAction.name)
            if action == InvokeAction.name:
                steps.append(
                    invoke_step(raw["tool_id"], raw["operation_name"], **raw.get("params", {}))
                )
            else:
                params = {k: v for k, v in raw.items() if k != "action"}
                steps.append(Step(next_action=action, params=params))
        return steps
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
    ) -> None:
        # `llm` is the model behind infer() — procedural memory "includes implicit knowledge encoded
        # in LLM weights" (README), and infer() is a *query* against it (CoALA), not a planner. None
        # keeps the deterministic store/retrieve half usable without a model (or a provider SDK).
        # `prompt` is the pluggable PlanPrompt that builds infer()'s (system, user) pair — the knob
        # for planning *content* (custom instructions, few-shot, catalog rendering).
        self._backend = backend
        self._llm = llm
        self._prompt = prompt

    async def retrieve(self, activity: Activity) -> Plan | None:
        """Looks up a cached Plan matching this activity's goal — e.g. exact match or embedding
        similarity, backend-dependent. The cheap path: skips infer() entirely when it hits."""
        rows = await self._backend.query(goal=activity.goal)
        if not rows:
            return None
        # query()'s contract is most-relevant-first with a deterministic tiebreak, so rows[0] is the
        # canonical plan for this goal regardless of backend.
        return self._from_dict(rows[0])

    async def infer(self, activity: Activity, tools: dict[str, Manual]) -> Plan:
        """Produce a new multi-step Plan when no cached one fits — the expensive path, one model
        call producing a whole sequence of Steps at once. This is querying procedural memory for
        "implicit knowledge encoded in LLM weights": it builds a prompt from the goal plus the
        available ``tools`` (keyed by tool id -> its Manual, supplied by the caller that holds the
        live registry — a memory module never reaches into the environment) via the injected
        ``PlanPrompt``, calls the pluggable ``LLMClient``, then converts the model's JSON answer to
        the runtime's own ``Plan``/``Step`` vocabulary. That text -> domain conversion is the
        anti-corruption boundary; malformed output raises ``ValueError`` rather than a half-built
        plan. Without an LLM the module is store/retrieve only and this raises."""
        if self._llm is None:
            raise RuntimeError(
                "ProceduralMemory has no LLM configured; cannot infer a plan (store/retrieve "
                "still work). Pass an LLMClient to enable inference."
            )
        system, user = self._prompt(activity, tools)  # the injected PlanPrompt
        text = await self._llm.complete(system=system, prompt=user)
        return Plan(id=uuid.uuid4().hex, goal=activity.goal, steps=_parse_plan_steps(text))

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
