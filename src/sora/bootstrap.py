"""Internal; developers implement protocols, they don't call this directly.

This is the one place all the wiring happens: which memory backend, which transport, which adapters,
and — crucially — ``DecisionCycle`` and ``Agent`` sharing the *same* instances (ADR-0013).
A developer implementing an agent only ever writes ``agent.yaml`` plus, typically, one
``ReasonStrategy``; they never construct any of this by hand.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from sora.action import default_action_registry
from sora.cycle import Agent, DecisionCycle
from sora.environment import EnvironmentRegistry, WorkspaceOrigin
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from sora.strategies import Strategies
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
}
_DEFAULT_LLM_CLIENT = "sora.adapters.anthropic_llm.AnthropicLLMClient"


@dataclass(frozen=True)
class AgentConfig:
    """The parsed ``agent.yaml`` (its ``agent:`` block). Frozen so wiring can't mutate config
    mid-build. ``strategies``/``memory`` are dotted-path / URI maps resolved during ``build_agent``;
    ``workspaces`` is the raw list (each entry carries an ``origin`` plus adapter-specific keys like
    ``command``/``args``); ``llm`` is optional (absent -> no model, store/retrieve-only procedural
    memory)."""

    name: str
    strategies: dict[str, str]
    memory: dict[str, str]
    workspaces: list[dict[str, Any]]
    transport: dict[str, Any] | None = None
    llm: dict[str, Any] | None = None


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
    return AgentConfig(
        name=agent["name"],
        strategies=strategies,
        memory=dict(agent.get("memory", {})),
        workspaces=list(agent.get("workspaces", [])),
        transport=agent.get("transport"),
        llm=agent.get("llm"),
    )


def backend_for(spec: str) -> MemoryBackend:
    """Build a durable ``MemoryBackend`` from its config spec. ``file://<path>`` (or a bare path) ->
    ``FileMemoryBackend``; that's the only backend the core ships — a database/vector-store backend
    is a drop-in registered the same way once it exists."""
    path = spec.removeprefix("file://") if spec.startswith("file://") else spec
    if not path:
        raise ValueError(f"empty memory backend path in spec {spec!r}")
    return FileMemoryBackend(path)


def adapter_for(entry: dict[str, Any]) -> tuple[WorkspaceOrigin, WorkspaceAdapter]:
    """Build one workspace's ``(origin, adapter)`` from its agent.yaml entry. Dispatch is on
    ``origin.adapter``. Adapter SDKs are optional extras, so their imports are lazy (importing
    bootstrap must not require the ``mcp`` extra). The ``fake`` kind — plus any custom adapter — is
    an escape hatch: name a factory ``(origin) -> WorkspaceAdapter`` via ``factory:`` and it's
    resolved through ``import_object`` (the test/showcase seam)."""
    origin = WorkspaceOrigin(**entry["origin"])
    kind = origin.adapter
    if kind in ("mcp", "are-mcp"):
        if kind == "mcp":
            from sora.adapters.mcp import McpWorkspaceAdapter as _Adapter
        else:
            from sora.adapters.are_mcp import AreMcpWorkspaceAdapter as _Adapter
        return origin, _Adapter(
            command=entry["command"],
            args=list(entry["args"]),
            workspace_id=entry["workspace_id"],
            origin=origin,
            env=entry.get("env"),
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
    return client_cls(**settings)  # type: ignore[no-any-return]


def transport_for(config: AgentConfig) -> MessageTransport:
    """The single-agent default is an in-process inbox. An agent-to-agent transport (A2A/HTTP,
    driven by ``transport.peers``) is the multi-agent case and is deferred, so a ``peers`` config
    raises rather than silently running peerless."""
    if config.transport and config.transport.get("peers"):
        raise NotImplementedError(
            "agent-to-agent transport (transport.peers) is not implemented yet; single-agent runs "
            "use the in-process transport"
        )
    return InProcessTransport()


def build_agent(config_path: str) -> Agent:
    """What ``sora run`` calls before handing off to TerminalSession. The one place all the wiring
    happens — a developer implementing an agent never writes this. Constructs the single shared
    EnvironmentRegistry / WorkingMemory / memory modules / transport and hands the *same* instances
    to both DecisionCycle and Agent (ADR-0013). Stays synchronous: e.g. the async startup join runs
    in ``Agent.run()``."""
    load_dotenv()  # convenience: pick up ANTHROPIC_API_KEY etc. from a local .env if present
    config = load_yaml(config_path)

    adapters = dict(adapter_for(entry) for entry in config.workspaces)
    registry = EnvironmentRegistry(adapters=adapters)  # the single shared instance...
    working = WorkingMemory(registry=registry)  # ...held here read-only as an EnvironmentView
    semantic = SemanticMemory(backend_for(config.memory["semantic"]))
    procedural = ProceduralMemory(backend_for(config.memory["procedural"]), llm=llm_for(config))
    episodic = EpisodicMemory(backend_for(config.memory["episodic"]))
    communication = transport_for(config)

    strategies = Strategies(
        observe=import_object(config.strategies.get("observe", _DEFAULT_STRATEGIES["observe"]))(),
        reflect=import_object(config.strategies.get("reflect", _DEFAULT_STRATEGIES["reflect"]))(),
        situate=import_object(config.strategies.get("situate", _DEFAULT_STRATEGIES["situate"]))(),
        reason=import_object(config.strategies["reason"])(),  # required — no default
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
