# Configuration Reference

The exact `agent.yaml` schema, as parsed by `sora.bootstrap.load_yaml`/`build_agent`
([source](https://github.com/sora-agents/sora-runtime/blob/main/src/sora/bootstrap.py)). For the
narrative walkthrough — what each block is *for* and why — see [Getting Started —
Configuration](../getting-started/configuration.md) and [Your First Agent — Anatomy of
agent.yaml](../getting-started/first-agent.md#anatomy-of-agentyaml).

Everything below lives under a single top-level `agent:` key.

## `agent.name`

`str`, required. The agent's identity.

## `agent.strategies`

`dict[str, str]`, mapping a phase/seam name to a dotted import path (resolved via
`sora.bootstrap.import_object`). Every key is optional *except* `reason`, which has no mechanical
default and fails loud (`ValueError`) if omitted.

| Key | Default (when omitted) | Notes |
| --- | --- | --- |
| `reason` | — (required) | The one phase with no mechanical default; planning needs a model. |
| `observe` | `sora.strategies.DefaultObserveStrategy` | |
| `reflect` | `sora.strategies.DefaultReflectStrategy` | |
| `situate` | `sora.strategies.DefaultSituateStrategy` | |
| `act` | `sora.strategies.DefaultActStrategy` | |
| `interrupt` | `sora.strategies.DefaultInterruptHandler` | Hard-interrupt follow-up handler. |
| `interrupt_policy` | `sora.strategies.NeverInterruptPolicy` | Which pushed signals preempt (none, by default). |
| `change_gate` | `sora.strategies.PerceptionSignatureGate` | The pre-revalidation change gate (ADR-0024). |
| `context_adaptation` | `before_writes` | A *level name* (`none` \| `before_writes` \| `before_each_op`) or a dotted path to a custom `ReconsiderationPolicy`. Not a dotted-path-only key like the others — see [Extension Protocols — Auxiliary seams](extension-protocols.md#auxiliary-seams). |

## `agent.memory`

`dict[str, str]`, required keys `semantic`, `procedural`, `episodic` (a `KeyError` at startup if any
is missing). Each value is a backend spec resolved by `sora.bootstrap.backend_for`: `file://<path>`
or a bare path selects the shipped `FileMemoryBackend`; a database/vector-store backend is a future
drop-in registered the same way.

```yaml
agent:
  memory:
    semantic: file://.sora/semantic
    procedural: file://.sora/procedural
    episodic: file://.sora/episodic
```

## `agent.workspaces`

`list[dict]`. Each entry is one `(WorkspaceOrigin, WorkspaceAdapter)` pair, built by
`sora.bootstrap.adapter_for`. Every entry needs:

| Key | Type | Notes |
| --- | --- | --- |
| `origin.adapter` | `str` | Built-in kinds: `mcp`, `are-mcp`, `are-sim`. Anything else requires `factory:`. |
| `origin.address` | `str` | An MCP server URI, a WoT directory base href, or a nominal label for a locally-spawned subprocess. |
| `workspace_id` | `str` | |

Adapter-specific keys:

| Key | Applies to | Notes |
| --- | --- | --- |
| `command`, `args`, `env` | `mcp`, `are-mcp` | Presence of `command` selects a locally-spawned **stdio** subprocess; absence means connect to the already-running server at `origin.address` (SSE by default, or `transport: streamable-http`). |
| `transport` | `mcp`, `are-mcp` (remote only) | `streamable-http`, otherwise SSE. |
| `manuals` | `mcp`, `are-mcp`, `are-sim` | Directory path wired into a `DirectoryManualSource`, paired with the adapter's synthesized manuals by `Manual.id` (ADR-0018). |
| `factory` | any custom adapter | Dotted path to a `(origin) -> WorkspaceAdapter` callable — the escape hatch for anything not built in. |

`are-sim` needs the runtime-injected `simulation` object (see `agent.transport` below and
`sora run --scenario`) — it is not a config key.

## `agent.transport`

`dict | None`, optional. Absent selects the default single-agent `InProcessTransport`.

| Value | Effect |
| --- | --- |
| absent | `InProcessTransport` (in-process inbox, no network). |
| `{kind: are}` | `AreTransport` — user messages flow through the running ARE scenario's `AgentUserInterface`; shares the injected `simulation` with an `are-sim` workspace. |
| `{peers: [...]}` | Raises `NotImplementedError` — agent-to-agent transport is not implemented yet. |

## `agent.llm`

`dict | None`, optional. Absent means no model — `ProceduralMemory` is store/retrieve-only (no
`infer`). Present, built by `sora.bootstrap.llm_for`:

| Key | Type | Notes |
| --- | --- | --- |
| `client` | `str`, optional | Dotted path to an `LLMClient` subclass; default `sora.adapters.anthropic_llm.AnthropicLLMClient`. |
| *(everything else)* | — | Passed through as constructor kwargs to the chosen client — e.g. `model:`. Never hardcoded; the API key comes from the environment, not this block (see [Getting Started — Configuring the LLM and its API key](../getting-started/configuration.md#configuring-the-llm-and-its-api-key)). |

Every configured client is wrapped in `MeteredLLMClient` automatically (timing/usage
instrumentation) — not a config option.

## `agent.procedural`

`dict[str, str] | None`, optional. Overrides `ProceduralMemory`'s built-in prompts, each a dotted
path to a callable satisfying the `PlanPrompt`/`GroundPrompt` Protocol:

| Key | Replaces |
| --- | --- |
| `plan_prompt` | `ProceduralMemory`'s built-in `default_plan_prompt` |
| `ground_prompt` | `ProceduralMemory`'s built-in `default_ground_prompt` |

A named callable fully replaces the built-in default — it does not patch pieces of it.

## `agent.max_subgoal_depth`

`int | None`, optional. Overrides the deliberative sub-goal recursion breaker's depth cap on
`DefaultReasonStrategy`. Omitted (the common case) means the strategy's own built-in default
applies — this key exists to raise it for a task with legitimately deep sub-goals, not to configure
routine behavior.

## `agent.max_replan_attempts`

`int | None`, optional. Overrides the coarse arm of the runaway-replanning breaker on
`DefaultReasonStrategy`: how many plans an activity may abandon *with a defect*, and without a
single operation having run, before the agent stops re-planning and blocks on an `InputWait` to ask
the user. Omitted means the strategy's own built-in default. The breaker's precise arm — the
replacement plan abandoned for the *same* defect as the plan it replaced — trips at two regardless
and is deliberately not configurable.

Only defect-bearing entries in `Activity.replan_trail` count toward this. A plan discarded because
the world moved under it ([ADR-0024](../architecture/adrs/0024-plan-reconsideration-context-adaptation.md))
is an honest attempt against a changed world rather than a repeat, so a fast-moving environment
never trips this cap — that pile-up is logged instead. Raise it for a domain where several
structurally different plans legitimately fail before one lands; lowering it mostly buys a faster
question to the user.

---

!!! info "Hand-authored framing pending"
    This page renders the exact, source-checked schema. The security-boundary and operational
    guidance the design calls for — e.g. why credentials never belong in `agent.llm`, or which
    `context_adaptation` level suits which failure mode — is still owed as hand-written prose; track
    it against the reference-authoring task.
