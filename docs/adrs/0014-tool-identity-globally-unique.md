# Tool identity is globally unique, guaranteed by the protocol adapter

* Status: proposed
* Date: 2026-07-12

## Context and Problem Statement

The live layer addresses tools by a bare `Tool.id` everywhere — `EnvironmentRegistry._tools` and `get()`, `WorkingMemory.focused_tools`, `InvokeAction`'s `registry.get(tool_id)`, `Step`/`OperationInvocation.tool_id`, and `Percept.source`. Nothing specifies that `Tool.id` is unique, yet a tool is, by design, a *shared* domain object: the Tool Model states tools "exist and evolve independently of any given agent and can be shared by multiple agents." Two agents that focus the same tool, or exchange messages about it, must name it identically — so identity has to be **global**, not local to one agent. At the same time `ToolRecord` scopes an instance by `workspace_id` only at the persistence layer, and a single registry can only ever *see* its own agent's joins. Who guarantees global uniqueness, and who enforces it? And should identity ride on `Tool.address` (URI-like), the way the Web identifies resources?

## Decision Drivers

* Tools are shared across agents (A&A meta-model): cross-agent shared focus and agent-to-agent messaging about a tool need *one* identifier per tool, i.e. global identity
* Different protocols supply identity differently: Web/WoT tools have their own URIs; MCP tools are only a *name* under one server, unique within it but not across servers (ARE proves this — its tools have no `address` at all; see docs/phase-2-findings.md)
* Ids must be stable across disconnect/reconnect: `ToolRecord.id` is persisted and `restore()`/`connect()` must reproduce the same id to re-resolve manuals and refer to the same instance
* A single registry cannot observe other agents, so it can only *enforce* the locally-visible slice; global uniqueness therefore has to be a by-construction property of the adapter, not something any one registry can verify end to end
* A collision the registry *can* see must fail loudly, never silently corrupt a sibling workspace

## Considered Options

* A — Adapter guarantees `Tool.id` is globally unique, derived from the tool's global address/origin; the registry enforces the slice it can see
* B — Scope ids to workspaces; address every live tool by a `(workspace_id, tool_id)` composite
* C — Registry mints a unique internal handle on join, disambiguating collisions itself

## Decision Outcome

Chosen option: "A — globally unique, adapter-guaranteed, registry-enforced", because it keeps the live layer a flat `id -> Tool` map while giving a shared tool one identity that every agent derives independently and agrees on. The adapter derives `Tool.id` as a **deterministic function of the tool's global identity** — its own URI where the protocol provides one (WoT gets this for free), or a value synthesized from the workspace's global origin/address where it does not (MCP). Because the derivation is anchored on a globally-meaningful address, two agents focusing the *same* addressable tool derive the *same* id, which is exactly what makes cross-agent focus and messaging about tools coherent.

Enforcement is layered: the adapter *guarantees* global uniqueness by construction; each registry *enforces* what it can see — `_register` raises on a duplicate id, and `leave` no longer pops a shared id — turning today's silent overwrite (and the cross-workspace deregistration `leave` can cause) into a fast, loud failure. A registry cannot detect a *global* collision it never observes, so global uniqueness ultimately rests on adapter correctness, with local enforcement as the backstop.

`address` is deliberately *not* the identity field: it is a locator (often absent, as in ARE/stdio, or workspace-level for MCP/SSE), whereas `id` is the stable, globally-unique handle derived *from* that address. Where a tool has no global address at all (e.g. a privately-spawned stdio subprocess), it is not a shared tool, so "same tool ⇒ same id" does not apply; its id is still globally-unique-by-construction from the origin the adapter knows and reproducible for that agent's `restore()`. This extends ADR-0007's instance identity from the persistence layer into the live layer.

### Positive Consequences

* One tool has one id across all agents, so shared focus and agent-to-agent messages that reference a tool stay coherent
* Live addressing stays a flat `id -> Tool` map; the in-flight operation-result path is already keyed by `op_id`, not `tool_id`, so it needs no change
* Matches the "identifiers are like Web identifiers" model: identity is derived from a global address (a URI where one exists)
* Deterministic derivation keeps `restore()` valid: the same records rebuild the same ids across runs
* A colliding adapter fails fast at join, rather than corrupting an already-joined workspace on a later `leave`

### Negative Consequences

* Adapters must derive ids from a *globally*-meaningful discriminator (the origin/address), not a local, config-assigned workspace label — otherwise "globally unique" degrades to merely locally unique
* Global ids are longer/namespaced (e.g. an origin-qualified app name); strategies should read ids from `all_tools()`/manuals rather than hardcode a bare guess
* No single registry can verify global uniqueness end to end; it rests on adapter correctness, with only local collisions catchable — so the contract must be stated and tested per adapter

## Pros and Cons of the Options

### A — Globally unique, adapter-guaranteed, registry-enforced

* Good, because a shared tool gets one identity every agent agrees on, and the guarantee sits with the code that understands the protocol's naming (ADR-0008)
* Good, because it composes with ADR-0007 (instance identity) and keeps the live layer a flat map
* Bad, because it obliges every adapter to anchor ids on a global address and exports namespaced ids to callers; global uniqueness can't be centrally verified

### B — Workspace-scoped composite key

* Good, because it makes instance identity explicitly `(workspace, tool)`
* Bad, because it is not even global — two agents can't name a shared tool identically — and the composite ripples through `invoke`, `focus`, `Percept.source`, `Step`/`OperationInvocation`, and every bare-`tool_id` call site in EXAMPLES.md

### C — Registry-minted handle

* Good, because it keeps friendly external ids and disambiguates locally
* Bad, because a per-registry handle is agent-local by definition — it cannot give a shared tool a stable cross-agent identity — and it hides the collision instead of making the adapter own naming

## Links

* Refines [ADR-0007](0007-manual-record-separation.md) (extends instance identity from persistence into the live layer)
* Depends on [ADR-0005](0005-workspace-grouping.md), [ADR-0006](0006-workspace-join-leave-lifecycle.md), and the adapter-as-seam principle in [ADR-0008](0008-protocol-based-extensibility.md)
* The `address`-is-not-identity observation and the registry-enforcement change are recorded in [docs/phase-2-findings.md](../phase-2-findings.md); live-layer enforcement is expected to land with the Phase 3 adapter hardening (ROADMAP step 12)
