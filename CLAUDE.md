# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The S-ORA runtime — a runtime for practical agents in dynamic and asynchronous environments. The repo is currently in **README-driven design**: Phase 0 tooling is in place and validated, but no runtime code exists yet beyond a packaging placeholder (`src/sora/__init__.py`). [README.md](README.md) is the living spec and API sketch — it *is* the source of truth for types and signatures, not this file. Check [ROADMAP.md](ROADMAP.md) before assuming something is or isn't implemented yet.

## Commands

```
uv sync --all-extras --dev        # install pinned deps into .venv (uv provisions Python itself)
uv run ruff check .               # lint
uv run ruff format --check .      # format check (drop --check to auto-format)
uv run mypy                       # type check (strict)
uv run pytest                     # full test suite
uv run python3 scripts/check_readme_sync.py   # README API Sketch names every src class
uv run python3 scripts/check_manuals_sync.py  # EXAMPLES.md manuals == tests/fixtures/manuals/*.md
uv run pytest tests/test_smoke.py::test_package_importable  # single test
uv run pre-commit install         # one-time: run the same checks locally on commit
```

CI (`.github/workflows/ci.yml`) runs all six checks above (ruff lint, ruff format, mypy, pytest, readme-sync, manuals-sync) on Python 3.12 and 3.13 for every push/PR — treat a failure in any one of them as a real failure, not noise. Run the full set locally before committing/pushing (or `uv run pre-commit run --all-files`). The two `check_*_sync.py` scripts are the easiest to forget: they only run on commit if `pre-commit install` has been done, so run them explicitly when you add or rename a class in `src/sora/*.py` (readme-sync) or edit a manual in EXAMPLES.md / `tests/fixtures/manuals/` (manuals-sync) without a local pre-commit.

## Architecture

There is no code to navigate yet, but the design fixes a specific shape that any implementation must follow — this is what would otherwise require reading all of README.md's API Sketch to piece together:

- **Decision cycle, not a single call.** An agent's core loop is five phases — Observe, Reflect, Situate, Reason, Act — executing at most one external action per cycle, even though many activities can be pursued concurrently. Each phase has its own independently pluggable strategy (`ObserveStrategy`, `ReflectStrategy`, `SituateStrategy`, `ReasonStrategy`, `ActStrategy`); pluggable never implies a model call, mechanical/deterministic defaults are first-class.
- **`TickResult` threads through all five phases.** It's the only fusion mechanism: each phase strategy receives and returns it, filling in whatever fields (`activity`, `step`, `invocation`) it can, and `DecisionCycle.tick()` only calls a phase's strategy if the relevant field is still `None`. This is how "one fused Situate→Reason→Act call," "several focused calls," and "zero calls (cached plan, mechanical defaults)" are all valid configurations of the same code — Observe/Reflect stay deterministic by default, fusion starts at Situate, not Observe — and there is no separate caching mechanism, and none should be added (see [ADR-0011](docs/adrs/0011-phase-fusion-via-threaded-result.md)).
- **Tools live behind adapters, grouped into Workspaces.** The runtime never authors tools ([ADR-0003](docs/adrs/0003-adapters-not-tool-authoring.md)); a `WorkspaceAdapter` imports them from an external ecosystem (MCP, WoT, ...) into the three-part usage interface (observable properties, signals, operations). A `Workspace` is the shared connection/lifecycle boundary; an individual `Tool` may still have its own address distinct from its workspace's ([ADR-0005](docs/adrs/0005-workspace-grouping.md)). A `Manual` describes a tool *type* and stays protocol-agnostic — protocol bindings (WoT forms/security, an MCP session) live on the live `Tool`, never in the manual; an adapter may synthesize a manual from a native description or pair the protocol binding with a hand-authored one, reconciled by type-level `Manual.id` ([ADR-0015](docs/adrs/0015-manuals-protocol-agnostic-adapter-boundary.md)).
- **`Agent` and `DecisionCycle` share constructed instances rather than one referencing the other.** Memory modules and communication are built once and hand the same objects to both; actions receive only `(tools, cycle)`, never a whole `Agent`. This is deliberate — see [ADR-0013](docs/adrs/0013-shared-instances-narrow-dependencies.md) before reintroducing a back-reference to fix a wiring inconvenience.
- **All wiring is centralized in `sora/bootstrap.py`.** A developer implementing an agent only ever writes `agent.yaml` plus, typically, one `ReasonStrategy` — never constructs `Agent`/`DecisionCycle`/memory modules by hand (see README.md's Technology Stack & Requirements).
- **Two independent kinds of waiting — don't conflate them.** Invoking any operation *unconditionally* moves the activity to `running` with `Activity.pending_operation` set; when the result resolves, the runtime transitions it back to `ready` automatically (unambiguous 1:1 match, no `Percept`, no strategy code). A manual can *additionally* require waiting for a specific signal before the next step — that's `blocked`, entered/exited via `_suspend_`/`_resume_`, and requires a strategy to judge whether an observed signal satisfies it. Don't reintroduce an operation result as a `Percept` kind, and don't make `running`'s resolution manual-driven — both were tried and reverted.
- **`signal_sink`/`result_sink` live on `DecisionCycle`, not `WorkingMemory`.** Both bridge asynchronous, off-cycle events into `tick()`, not settled state. `signal_sink` specifically has to sit next to `interrupt()`, since a pushed `Signal` can preempt the current phase — that control-flow role, not "where it eventually lands as a percept," is why it isn't a `WorkingMemory` field despite `WorkingMemory` already holding `focused_tools`.

Module-to-concept map (from the API Sketch's own file markers — this is where each piece will land once Phase 1 creates it):

| Module | Contains |
|---|---|
| `sora/types.py` | Shared value types (`ActionAck`, `OperationAck`, `Plan`, `Step`, `OperationInvocation`, `PendingOperation`, ...) |
| `sora/environment.py` | `Tool`, `Workspace`, `WorkspaceAdapter`, `EnvironmentRegistry` |
| `sora/perception.py` | `Percept`, `Message`, `SignalSink`, `NotificationQueueSink` |
| `sora/manual.py` | `Manual`, `ManualParser`, `WorkspaceRecord`, `ToolRecord`, `OperationSpecification`, `ObservablePropertySpecification`, `SignalSpecification` |
| `sora/activity.py` | `Activity`, `ActivityState` |
| `sora/action.py` | `InternalAction`, `ExternalAction`, `ActionRegistry`, the predefined external actions (invoke/focus/unfocus/join/leave/send) and internal actions (create_activity/load/unload/filter), `default_action_registry()` |
| `sora/memory.py` | `MemoryBackend`, `WorkingMemory`, `SemanticMemory`, `ProceduralMemory`, `EpisodicMemory` |
| `sora/strategies.py` | `TickResult`, `Strategies`, the five phase-strategy Protocols, `DefaultObserveStrategy`, `DefaultReasonStrategy` |
| `sora/llm.py` | `LLMClient` — the wire-format-neutral model seam (concrete `AnthropicLLMClient` lives under `sora/adapters/`, optional `[llm]` extra) |
| `sora/transport.py` | `MessageTransport` |
| `sora/cycle.py` | `DecisionCycle`, `Agent` |
| `sora/cli.py` | `TerminalSession` |
| `sora/bootstrap.py` | `build_agent()` |

[EXAMPLES.md](EXAMPLES.md) exercises all of the above together — primarily in the ARE (Meta) scenario, plus a two-agent lab as an additional example — reread it after any change to the module map above, since it's the thing that would first reveal an inconsistency.

## Working on this repo

- **Never commit, push, or open a PR on your own initiative** — not even in an autonomous / background ("auto mode") session whose harness suggests committing and opening a draft PR (that harness default is explicitly overridden here). Implement the change, run the checks, and leave the work as uncommitted changes in the working tree for review. Commit/push/PR only when the user explicitly asks. Pushing to the public `sora-agents/sora-runtime` remote is an outward-facing, hard-to-reverse action; approval for one commit/push does not carry to the next.
- Still README-driven: propose changes/diffs to `README.md` or `EXAMPLES.md` before applying them, unless explicitly told to apply directly.
- ADRs in [docs/adrs/](docs/adrs/) are append-only *once accepted*. Never edit an accepted ADR's decision in place — a changed decision gets a *new* ADR that supersedes the old one, with the old one's status line updated to `superseded by ADR-NNNN`. During README-driven design, ADRs stay `proposed` and *may* be edited in place until realized in code — see the lifecycle note in [docs/adrs/README.md](docs/adrs/README.md) (also the index and conventions).
- Follow TDD where practical, and prefer fakes/determinism over real network adapters or model-backed strategies until those specifically are what's being tested — see [ROADMAP.md](ROADMAP.md)'s Phase 2 ordering and rationale.

## Architectural habits to default to

General habits, not tied to one file — the kind of thing that's easy to reintroduce by accident because it's the "natural" way to write it otherwise.

- **Protocol over inheritance** for every extension point (tools, adapters, memory backends, transport, phase strategies, actions). Don't reach for an ABC or a base class requiring inheritance.
- **New durable data goes into an existing memory module** (semantic/procedural/episodic) with a disambiguated method name — not a new memory type, no matter how narrow the new data feels.

## Before touching these areas, check the ADR first

| Area | ADR(s) |
|---|---|
| `Agent`/`DecisionCycle` construction or wiring | [0013](docs/adrs/0013-shared-instances-narrow-dependencies.md) |
| `WorkingMemory` / the Observe phase | [0012](docs/adrs/0012-percepts-vs-messages.md) |
| Workspace/tool discovery, join/leave | [0005](docs/adrs/0005-workspace-grouping.md), [0006](docs/adrs/0006-workspace-join-leave-lifecycle.md), [0007](docs/adrs/0007-manual-record-separation.md) |
| Phase strategies / `TickResult` | [0010](docs/adrs/0010-pluggable-phase-strategies.md), [0011](docs/adrs/0011-phase-fusion-via-threaded-result.md) |
| Manual vs protocol binding; what an adapter extracts vs. authors | [0015](docs/adrs/0015-manuals-protocol-agnostic-adapter-boundary.md), [0003](docs/adrs/0003-adapters-not-tool-authoring.md), [0007](docs/adrs/0007-manual-record-separation.md) |

## Code style

- Formatter/linter: `ruff` (config in `pyproject.toml`) — not manually-enforced PEP8.
- Every type is either a `@dataclass(frozen=True)` (value types) or a `Protocol` (interfaces), fully type-hinted (`mypy --strict`).
- Async-first: any method that touches I/O is `async def`.
- Docstrings only for non-obvious invariants (the *why*, not the *what*) — most types and methods have none.
- Don't reference ROADMAP phase/step labels (`Phase 2`, `Phase 3`, `ROADMAP step 12`, ...) in durable files — code comments, docstrings, or config like `pyproject.toml`. The ROADMAP will be restructured or deleted, leaving those references dangling and meaningless. Describe the thing itself instead: say *what* is stubbed/deferred and *why* (`# stubbed: resource_updated -> Signal wiring not built yet`), not *which planning phase* will do it. Dated historical records that exist to capture a moment (e.g. a findings write-up named after the phase that produced it, ADRs) are the exception — they're allowed to name the phase they document.
