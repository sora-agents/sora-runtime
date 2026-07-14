# Manuals stay protocol-agnostic; protocol bindings live on the tool instance

* Status: proposed
* Date: 2026-07-14

## Context and Problem Statement

A `Manual` describes a tool *type* and is meant to be reusable across instances and protocols (the same pump could be reached over MCP or CoAP). But the native descriptions adapters import from — an MCP tool list, a W3C WoT Thing Description (TD) — bundle *both* protocol-agnostic information (operation/property/signal names and their JSON-Schema data shapes) *and* protocol bindings (WoT TD `forms`/`securityDefinitions`; an MCP session plus the `<App>__op` name assembly). Where does each part land — what belongs in the `Manual`, what belongs elsewhere — and do we need a separate "instance description" abstraction alongside the `Manual`? The tension already exists in the design sketches: the ARE MCP adapter *synthesizes* a `Manual` from the MCP tool list (`_synth_manual`, `inputSchema → OperationSpecification.parameters`), while the WoT adapter sketch *loads a hand-authored Markdown manual* per Thing (`MarkdownManualParser().parse(load_manual(td.id))`) and uses the TD only for its protocol bindings.

## Decision Drivers

* A `Manual` must stay protocol-agnostic so it is reusable across instances/protocols and shareable across agents (the A&A shared-tool model).
* [ADR-0003](0003-adapters-not-tool-authoring.md): adapters extract whatever manual information the source provides — and no more; richness is capped by the source protocol.
* Real native descriptions are uneven: a typical WoT TD document is schema-rich but semantics-thin — it can be enriched with semantic tags defined in some ontology, but there are no other descriptions, preconditions/effects, behavior, or safety — while our hand-authored Markdown is the mirror image (semantics-rich, schema-loose).
* Avoid inventing a second model type where an existing boundary (the adapter) already splits the concern.

## Considered Options

* A separate `InstanceDescription`/protocol-binding model type held alongside each `Manual`.
* Fold the protocol binding into the `Manual` (a `Manual` per instance, carrying its bindings).
* Keep the protocol binding on the live `Tool` instance (adapter-owned); keep the `Manual` protocol-agnostic; let a `Manual` be produced by either adapter synthesis or Markdown authoring, reconciled by type-level `Manual.id`.

## Decision Outcome

Chosen: **protocol bindings live on the live `Tool` instance (created and owned by the `WorkspaceAdapter`) and never enter the `Manual`; the `Manual` stays protocol-agnostic.** Here *protocol binding* means the concrete, per-instance access mechanism for the tool's **application-layer** protocol (WoT `forms` over HTTP/CoAP; MCP's JSON-RPC session and name assembly) — distinct from the **transport** it runs over (stdio, SSE, TCP). The split is by *content*, not by call-graph:

* **Protocol-binding (application-layer) → `Tool` instance (adapter-owned):** WoT TD `forms` (href, `htv:methodName`, `contentType`) and `securityDefinitions`; an MCP `ClientSession` and the `<App>__op` name assembly. These implement `invoke`/`focus`/`observe` and are never serialized into a `Manual`.
* **Manual-level → `Manual`:** operation/property/signal names, natural-language semantics (functional description, preconditions, effects, behavior, usage protocols & safety), and **JSON-Schema data shapes** — which are protocol-agnostic and already have a home in the spec types' `parameters`/`schema` dicts.

A `Manual` has two interchangeable **provenance channels**, both producing the same protocol-agnostic type:

1. **Adapter synthesis** from a native description (MCP tool list, WoT TD document): reliably yields names + JSON Schema, plus whatever semantics the source carries (the ADR-0003 ceiling).
2. **Markdown authoring** parsed by `ManualParser`, retrieved by `manual_id` from a repository or semantic memory: yields the rich natural language semantics native formats lack.

The two reconcile by type-level `Manual.id` ([ADR-0007](0007-manual-record-separation.md)): an adapter may synthesize a thin manual and a curated repository may supply a richer one for the same id; or, as the WoT sketch shows, an adapter may itself load the authored Markdown for the Thing it is binding. `ManualParser` is therefore a shared `Manual` *producer*, not owned by either side — the invariant is what content goes where, not who calls whom.

**No separate instance-description model type is introduced.** The native description *is* the complete description; the `WorkspaceAdapter` is precisely the component that splits it into (protocol binding → `Tool`) + (data shape / semantics → `Manual`). A parallel model type would re-implement that boundary.

Because JSON Schema is manual-level, our Markdown format *may* carry it but need not: adapter-imported tools get their schema from the native format, and a hand-authored manual can rely on the clean format's light `(type, range)` hints (lifted into a minimal schema by the parser), with an optional inline-JSON-Schema escape hatch where full fidelity is required.

### Positive Consequences

* `Manual`s stay portable across instances and protocols and shareable across agents; the protocol binding is an instance concern.
* Reuses existing structure — `Manual.id` (type-level), the `parameters`/`schema` dicts, and the "retrieve manuals from external repositories" action — instead of a new abstraction.
* Explains the two existing adapters cleanly: ARE (synthesis) and WoT (authored Markdown) are two provenance channels of one boundary, not a contradiction.
* JSON Schema in Markdown becomes optional rather than mandatory, because the high-fidelity schema path is the adapter.

### Negative Consequences

* When both provenance channels fire for one `manual_id` (adapter-synthesized + repository-authored), a **merge policy** is required (which fields win). Deferred until a consumer needs it; no runtime path depends on merging today.
* Adapter-synthesized manuals remain semantics-thin (capped by the source per ADR-0003), so LLM reasoning still depends on someone authoring Markdown.
* The Markdown parser must optionally lift light type hints (and tolerate an optional inline schema) — a small added surface.

## Pros and Cons of the Options

### Separate `InstanceDescription` type

* Good, because protocol bindings and semantics become explicitly separate types.
* Bad, because it duplicates the split the adapter already performs, and forces every consumer to hold and correlate two objects where one protocol-agnostic `Manual` plus a live `Tool` already suffices.

### Fold the protocol binding into the `Manual` (per-instance manuals)

* Good, because a single object then carries everything needed to invoke a specific instance.
* Bad, because it destroys the `Manual`'s protocol-agnosticism and cross-instance reuse, re-coupling the shared, agent-agnostic description to one application-layer (tool-use) protocol — against the A&A shared-tool model and ADR-0007's type-level `Manual.id`.

### Protocol binding on the instance; protocol-agnostic `Manual` from either provenance (chosen)

* Good, because it keeps the `Manual` reusable, needs no new type, and subsumes both existing adapters.
* Bad, because it leaves a merge policy open and keeps adapter-synthesized manuals thin — both acceptable and deferrable.

## Links

* Depends on [ADR-0003](0003-adapters-not-tool-authoring.md), [ADR-0004](0004-tool-usage-interface.md)
* Refines [ADR-0007](0007-manual-record-separation.md) (type-level `Manual.id` is the reconciliation key); relates to [ADR-0014](0014-tool-identity-globally-unique.md) (instance identity vs. type-level manual id)
