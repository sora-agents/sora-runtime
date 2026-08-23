"""Internal; developers implement protocols, they don't call this directly.

This is the one place all the wiring happens: which memory backend, which transport, which adapters,
and — crucially — ``DecisionCycle`` and ``Agent`` sharing the *same* instances (ADR-0013).
A developer implementing an agent only ever writes ``agent.yaml`` plus, typically, one
``ReasonStrategy``; they never construct any of this by hand.
"""

from __future__ import annotations

import importlib
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from sora.action import default_action_registry
from sora.adapters.runtime_io import (
    RUNTIME_IO_ADAPTER,
    RUNTIME_IO_ADDRESS,
    RuntimeIOAdapter,
)
from sora.cycle import Agent, DecisionCycle
from sora.environment import EnvironmentRegistry, WorkspaceOrigin
from sora.manual import DirectoryManualSource
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from sora.strategies import ReconsiderationPolicy, Strategies
from sora.transport import InProcessTransport

if TYPE_CHECKING:
    from sora.environment import WorkspaceAdapter
    from sora.llm import LLMClient
    from sora.memory import MemoryBackend
    from sora.transport import MessageTransport

# Default strategy classes for the four phases that *have* a mechanical default. Reason is
# deliberately absent — it is the one phase with no mechanical default (planning needs a model), so
# agent.yaml must name it. Dotted paths (not direct imports) so a config can override any phase the
# same way, and so this table reads like the config it backs.
_DEFAULT_STRATEGIES = {
    "observe": "sora.strategies.DefaultObserveStrategy",
    "reflect": "sora.strategies.DefaultReflectStrategy",
    "situate": "sora.strategies.DefaultSituateStrategy",
    "act": "sora.strategies.DefaultActStrategy",
    # The two hard-interrupt seams — selectable like the phase strategies, but not part of the
    # five-phase Strategies bundle. interrupt = the follow-up handler (default: user stop pauses to
    # await input); interrupt_policy = which pushed signals preempt (default: none).
    "interrupt": "sora.strategies.DefaultInterruptHandler",
    "interrupt_policy": "sora.strategies.NeverInterruptPolicy",
    # The pre-revalidation change-gate (ADR-0024): whether the world moved since the plan was
    # baselined, orthogonal to context_adaptation's when. Default the domain-free signature gate; an
    # app names a domain gate (e.g. an INBOX-id projection) to filter the agent's own writes out.
    "change_gate": "sora.strategies.PerceptionSignatureGate",
}

# context_adaptation (ADR-0024): how eagerly to re-validate an in-progress plan against new
# perception. Selected by a level name — none | before_writes | before_each_op — or a dotted
# path to a custom ReconsiderationPolicy; the agent-facing default is before_writes.
_RECONSIDERATION_LEVELS = {
    "none": "sora.strategies.NoneReconsideration",
    "before_writes": "sora.strategies.BeforeWrites",
    "before_each_op": "sora.strategies.BeforeEachOp",
}


def _reconsideration_for(value: str) -> ReconsiderationPolicy:
    """Resolve a ``context_adaptation`` value (a level name or dotted path) to a policy."""
    policy: ReconsiderationPolicy = import_object(_RECONSIDERATION_LEVELS.get(value, value))()
    return policy


_DEFAULT_LLM_CLIENT = "sora.adapters.anthropic_llm.AnthropicLLMClient"

# Built-in workspace-adapter / transport "kinds" — the config↔code contract. Each is the string a
# shipped adapter also declares as its own ``.name`` (a duck-typed invariant guarded by a test:
# ``test_shipped_adapter_names_match_bootstrap_dispatch_kinds``). Bootstrap can't dispatch off
# ``Adapter.name`` directly because adapter imports are lazy — importing bootstrap must not require
# an optional SDK extra — so the literal is repeated here, named (not inline) so a typo is a
# NameError, not a silent dispatch miss, and a rename is one edit. Custom adapters stay open via
# ``factory:``; this is only the *built-in* set, deliberately not a closed enum (ADR-0008).
_ADAPTER_MCP = "mcp"
_ADAPTER_ARE_MCP = "are-mcp"
_ADAPTER_ARE_SIM = "are-sim"
_TRANSPORT_ARE = "are"


@dataclass(frozen=True)
class AgentConfig:
    """The parsed ``agent.yaml`` (its ``agent:`` block). Frozen so wiring can't mutate config
    mid-build. ``strategies``/``memory`` are dotted-path / URI maps resolved during ``build_agent``;
    ``workspaces`` is the raw list (each entry carries an ``origin`` plus adapter-specific keys like
    ``command``/``args``); ``llm`` is optional (absent -> no model, store/retrieve-only procedural
    memory); ``procedural`` optionally names dotted-path ``plan_prompt``/``ground_prompt`` overrides
    for ``ProceduralMemory`` (absent -> its own built-in ``default_plan_prompt``/
    ``default_ground_prompt``)."""

    name: str
    strategies: dict[str, str]
    memory: dict[str, str]
    workspaces: list[dict[str, Any]]
    transport: dict[str, Any] | None = None
    llm: dict[str, Any] | None = None
    procedural: dict[str, str] | None = None
    # Override for the deliberative sub-goal recursion breaker's depth cap (DefaultReasonStrategy).
    # A generic runtime safety limit, not domain config; raise it for a task with legitimately deep
    # sub-goals. None (agent.yaml omits it) -> the strategy's own default owns the value, so the
    # single source of truth is _DEFAULT_MAX_SUBGOAL_DEPTH there, never a second literal here.
    max_subgoal_depth: int | None = None


def load_dotenv(path: str | Path = ".env") -> None:
    """Load ``KEY=value`` lines from a local ``.env`` into ``os.environ`` — a convenience so a local
    ``ANTHROPIC_API_KEY`` (and any other config) is picked up without an explicit ``export``.

    **Real environment variables take precedence**: an existing variable is never overwritten
    (``os.environ.setdefault``), so the process environment always wins over the file — the standard
    precedence. A missing file is a no-op. Blank lines and ``#`` comments are skipped, an optional
    ``export`` prefix is tolerated, and matching single/double quotes around the value are stripped.
    No dependency — the core stays dependency-free; copy ``.env.example`` to ``.env`` to start.
    """
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)  # real env wins — never overwrite


def import_object(path: str) -> Any:
    """Resolve a dotted (``pkg.mod.Attr``) or ``module:attr`` path to the object it names. The one
    mechanism agent.yaml uses to name a strategy, an adapter factory, or an LLM client — anything on
    ``sys.path`` resolves, so ``examples.*`` and test helpers work the same as ``sora.*``."""
    module_path, _, attr = path.replace(":", ".").rpartition(".")
    if not module_path or not attr:
        raise ValueError(f"not a dotted import path: {path!r}")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def load_yaml(config_path: str | Path) -> AgentConfig:
    """Parse ``agent.yaml`` (its ``agent:`` block) into an ``AgentConfig``. Fails loud on a missing
    ``strategies.reason`` — the one strategy with no default — rather than deferring to a confusing
    KeyError deep in wiring."""
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    agent = data["agent"]
    strategies = dict(agent.get("strategies", {}))
    if "reason" not in strategies:
        raise ValueError(
            "agent.yaml: strategies.reason is required — Reason is the one phase with no "
            "mechanical default (planning needs a model). Name your ReasonStrategy there."
        )
    procedural = agent.get("procedural")
    depth = agent.get("max_subgoal_depth")  # None -> the strategy's own default (no literal here)
    return AgentConfig(
        name=agent["name"],
        strategies=strategies,
        memory=dict(agent.get("memory", {})),
        workspaces=list(agent.get("workspaces", [])),
        transport=agent.get("transport"),
        llm=agent.get("llm"),
        procedural=dict(procedural) if procedural else None,
        max_subgoal_depth=int(depth) if depth is not None else None,
    )


def backend_for(spec: str) -> MemoryBackend:
    """Build a durable ``MemoryBackend`` from its config spec. ``file://<path>`` (or a bare path) ->
    ``FileMemoryBackend``; that's the only backend the core ships — a database/vector-store backend
    is a drop-in registered the same way once it exists."""
    path = spec.removeprefix("file://") if spec.startswith("file://") else spec
    if not path:
        raise ValueError(f"empty memory backend path in spec {spec!r}")
    return FileMemoryBackend(path)


def _require_simulation(simulation: Any | None, *, subject: str) -> Any:
    """Guard that the opaque ``simulation`` object ``build_agent`` accepts at runtime is present,
    for a workspace/transport kind that needs one. This is the *generic* injection seam — any
    adapter/transport that requires a live per-run object goes through it (ARE's ``AreSimulation``
    is the only one today); the ARE-specific part stays in the caller's ``subject``. Returns it
    (narrowing away ``None``) so callers pass the result on."""
    if simulation is None:
        raise ValueError(
            f"{subject} needs the runtime `simulation` object: pass simulation=... to build_agent "
            "(a per-run input the runner supplies — e.g. the ARE runner builds it from --scenario)"
        )
    return simulation


def adapter_for(
    entry: dict[str, Any], simulation: Any | None = None
) -> tuple[WorkspaceOrigin, WorkspaceAdapter]:
    """Build one workspace's ``(origin, adapter)`` from its agent.yaml entry. Dispatch is on
    ``origin.adapter``. Adapter SDKs are optional extras, so their imports are lazy (importing
    bootstrap must not require the ``mcp`` extra).

    For ``mcp``/``are-mcp`` the *transport* is chosen by what the entry carries: a ``command``
    (with ``args``) spawns and owns a local **stdio** subprocess; otherwise ``origin.address`` is
    the URL of an **already-running remote** server (SSE by default, or streamable-HTTP via
    ``transport: streamable-http``) and nothing is deployed. An optional ``manuals:`` directory path
    is wired into a ``DirectoryManualSource``, given to the adapter so it can pair hand-authored
    manuals with the ones it synthesizes for the same id (ADR-0018); absent -> no pairing, unchanged
    adapter-only manuals.

    ``are-sim`` is the in-process ARE integration: it needs a running scenario, so ``simulation``
    (an ``AreSimulation``) is *injected* at runtime; config stays generic (no scenario key); the
    scenario is a CLI argument the runner turns into the simulation (see ``sora run --scenario``,
    ``src/sora/cli.py``).

    The ``fake`` kind — plus any custom adapter — is an escape hatch: name a factory
    ``(origin) -> WorkspaceAdapter`` via ``factory:`` resolved through ``import_object`` (the
    test/showcase seam)."""
    origin = WorkspaceOrigin(**entry["origin"])
    kind = origin.adapter
    if kind == _ADAPTER_ARE_SIM:
        sim = _require_simulation(simulation, subject="an 'are-sim' workspace")
        from sora.adapters.are_sim import AreInProcessWorkspaceAdapter

        manual_source = DirectoryManualSource(entry["manuals"]) if "manuals" in entry else None
        return origin, AreInProcessWorkspaceAdapter(
            workspace_id=entry["workspace_id"],
            origin=origin,
            simulation=sim,
            manual_source=manual_source,
        )
    if kind in (_ADAPTER_MCP, _ADAPTER_ARE_MCP):
        if kind == _ADAPTER_MCP:
            from sora.adapters.mcp import McpWorkspaceAdapter as _Adapter
        else:
            from sora.adapters.are_mcp import AreMcpWorkspaceAdapter as _Adapter
        manual_source = DirectoryManualSource(entry["manuals"]) if "manuals" in entry else None
        if "command" in entry:  # local stdio subprocess the adapter owns
            return origin, _Adapter(
                workspace_id=entry["workspace_id"],
                origin=origin,
                command=entry["command"],
                args=list(entry.get("args", [])),
                env=entry.get("env"),
                manual_source=manual_source,
            )
        # remote: connect to an already-running server at origin.address
        return origin, _Adapter(
            workspace_id=entry["workspace_id"],
            origin=origin,
            url=origin.address,
            transport=entry.get("transport"),
            manual_source=manual_source,
        )
    if "factory" in entry:
        factory = import_object(entry["factory"])
        return origin, factory(origin)
    raise ValueError(
        f"no adapter for origin.adapter={kind!r}; give a `factory:` dotted path for a custom one"
    )


def llm_for(config: AgentConfig) -> LLMClient | None:
    """Build the ``LLMClient`` behind ``ProceduralMemory.infer`` from the optional ``llm:`` block.
    Absent -> ``None`` (store/retrieve-only procedural memory, no model). Present -> the client
    named by ``client:`` (default the shipped ``AnthropicLLMClient``), with the rest of the block
    passed as kwargs — so ``model:`` is config, never hardcoded, and the key stays in the env."""
    if not config.llm:
        return None
    settings = dict(config.llm)
    client_path = settings.pop("client", _DEFAULT_LLM_CLIENT)
    client_cls = import_object(client_path)
    client = client_cls(**settings)
    # Wrap every built client so each round-trip is timed and logged (`sora.llm`) — instrumentation
    # a run surface reads back via `LLMMeter`, without the concrete client growing that concern.
    # The model id rides along on the wrapper for the same reason: config is where it is known, and
    # a run that doesn't record which model produced its trace can't be read back later.
    from sora.llm import MeteredLLMClient

    model = settings.get("model")
    return MeteredLLMClient(client, model=str(model) if model is not None else None)


def procedural_prompts_for(config: AgentConfig) -> dict[str, Any]:
    """Resolve the optional ``procedural.plan_prompt``/``procedural.ground_prompt`` dotted paths
    into ``ProceduralMemory`` constructor kwargs — a custom callable fully replaces the built-in
    default (``default_plan_prompt``/``default_ground_prompt``), it doesn't patch pieces of it.
    Absent -> an empty dict, so ``ProceduralMemory``'s own defaults apply unchanged."""
    if not config.procedural:
        return {}
    kwargs: dict[str, Any] = {}
    if "plan_prompt" in config.procedural:
        kwargs["prompt"] = import_object(config.procedural["plan_prompt"])
    if "ground_prompt" in config.procedural:
        kwargs["ground_prompt"] = import_object(config.procedural["ground_prompt"])
    return kwargs


def transport_for(config: AgentConfig, simulation: Any | None = None) -> MessageTransport:
    """The single-agent default is an in-process inbox. ``transport: { kind: are }`` selects the ARE
    in-process transport (user messages via the running scenario's ``AgentUserInterface``)
    — it shares the injected ``simulation`` with the ``are-sim`` workspace. An agent-to-agent
    transport (A2A/HTTP, driven by ``transport.peers``) is the multi-agent case, deferred, so a
    ``peers`` config raises rather than silently running peerless."""
    if config.transport and config.transport.get("kind") == _TRANSPORT_ARE:
        sim = _require_simulation(simulation, subject="transport 'are'")
        from sora.adapters.are_sim import AreTransport

        return AreTransport(sim)
    if config.transport and config.transport.get("peers"):
        raise NotImplementedError(
            "agent-to-agent transport (transport.peers) is not implemented yet; single-agent runs "
            "use the in-process transport"
        )
    return InProcessTransport()


def reason_strategy_for(config: AgentConfig) -> Any:
    """Construct the (required) Reason strategy named in ``strategies.reason``. Forwards
    ``max_subgoal_depth`` only when agent.yaml set it *and* the strategy's constructor declares it
    (the default does) — an omitted key falls through to the strategy's own default, so the numeric
    default lives in exactly one place. A custom ReasonStrategy that doesn't take the kwarg still
    constructs with no args, keeping the generic dotted-path wiring intact."""
    cls = import_object(config.strategies["reason"])
    override = config.max_subgoal_depth
    if override is not None and "max_subgoal_depth" in inspect.signature(cls).parameters:
        return cls(max_subgoal_depth=override)
    return cls()


def build_agent(config_path: str, *, simulation: Any | None = None) -> Agent:
    """What ``sora run`` calls before handing off to TerminalSession. The one place all the wiring
    happens — a developer implementing an agent never writes this. Constructs the single shared
    EnvironmentRegistry / WorkingMemory / memory modules / transport and hands the *same* instances
    to both DecisionCycle and Agent (ADR-0013). Stays synchronous: e.g. the async startup join runs
    in ``Agent.run()``.

    ``simulation`` is an opaque, runtime-provided object for adapters/transports that need one
    (currently only the ARE in-process integration's ``AreSimulation``, shared by an ``are-sim``
    workspace and the ``are`` transport). It keeps config generic — the per-run scenario is a CLI
    argument the runner turns into this object, not a key in agent.yaml. ``None`` for every other
    agent; passing it when config asks for ``are-sim``/``are`` is required (else a clear error)."""
    load_dotenv()  # convenience: pick up ANTHROPIC_API_KEY etc. from a local .env if present
    config = load_yaml(config_path)

    communication = transport_for(config, simulation)
    adapters = dict(adapter_for(entry, simulation) for entry in config.workspaces)
    # The runtime's own I/O channels (today: the user-reply tool) are always joined, wired to the
    # same transport handed to the cycle — so replying to the user is a uniform tool invoke, not a
    # special `send` action. Nothing to suppress under ARE: ARE already routes the user
    # channel through its transport, which this tool delegates to.
    rio_origin = WorkspaceOrigin(adapter=RUNTIME_IO_ADAPTER, address=RUNTIME_IO_ADDRESS)
    adapters[rio_origin] = RuntimeIOAdapter(origin=rio_origin, transport=communication)
    registry = EnvironmentRegistry(adapters=adapters)  # the single shared instance...
    working = WorkingMemory(registry=registry)  # ...held here read-only as an EnvironmentView
    semantic = SemanticMemory(backend_for(config.memory["semantic"]))
    procedural = ProceduralMemory(
        backend_for(config.memory["procedural"]),
        llm=llm_for(config),
        **procedural_prompts_for(config),
    )
    episodic = EpisodicMemory(backend_for(config.memory["episodic"]))

    strategies = Strategies(
        observe=import_object(config.strategies.get("observe", _DEFAULT_STRATEGIES["observe"]))(),
        reflect=import_object(config.strategies.get("reflect", _DEFAULT_STRATEGIES["reflect"]))(),
        situate=import_object(config.strategies.get("situate", _DEFAULT_STRATEGIES["situate"]))(),
        reason=reason_strategy_for(config),  # required — no default
        act=import_object(config.strategies.get("act", _DEFAULT_STRATEGIES["act"]))(),
    )

    cycle = DecisionCycle(
        strategies=strategies,
        communication=communication,
        actions=default_action_registry(),
        registry=registry,
        working=working,
        semantic=semantic,
        procedural=procedural,
        episodic=episodic,
        interrupt_handler=import_object(
            config.strategies.get("interrupt", _DEFAULT_STRATEGIES["interrupt"])
        )(),
        interrupt_policy=import_object(
            config.strategies.get("interrupt_policy", _DEFAULT_STRATEGIES["interrupt_policy"])
        )(),
        reconsideration=_reconsideration_for(
            config.strategies.get("context_adaptation", "before_writes")
        ),
        change_gate=import_object(
            config.strategies.get("change_gate", _DEFAULT_STRATEGIES["change_gate"])
        )(),
    )
    return Agent(
        cycle=cycle,
        registry=registry,
        working=working,
        semantic=semantic,
        procedural=procedural,
        episodic=episodic,
        communication=communication,
    )
