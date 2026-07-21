# Manuals stay protocol-agnostic; protocol bindings live on the tool instance

* Status: proposed
* Date: 2026-07-14

## Context and Problem Statement

A `Manual` describes a tool *type*, reusable across instances and protocols (the same pump reachable over MCP or CoAP). But the native descriptions adapters import — an MCP tool list, a W3C WoT Thing Description (TD) — bundle *both* protocol-agnostic information (operation/property/signal names and their JSON-Schema data shapes) *and* protocol bindings (WoT TD `forms`/`securityDefinitions`; an MCP session plus `<App>__op` name assembly). Where does each part land, and do we need a separate "instance description" type alongside the `Manual`? The tension is already in the sketches: the ARE MCP adapter *synthesizes* a `Manual` from the tool list (`_synth_manual`, `inputSchema → OperationSpecification.parameters`), while the WoT sketch *loads a hand-authored Markdown manual* per Thing and uses the TD only for bindings.

## Decision Drivers

* A `Manual` must stay protocol-agnostic — reusable across instances/protocols, shareable across agents (the A&A shared-tool model).
* [ADR-0003](0003-adapters-not-tool-authoring.md): adapters extract whatever the source provides and no more; richness is capped by the source.
* Native descriptions are uneven: a WoT TD is schema-rich but semantics-thin; hand-authored Markdown is the mirror image (semantics-rich, schema-loose).
* Avoid inventing a second model type where the adapter boundary already splits the concern.

## Decision Outcome

**Protocol bindings live on the live `Tool` instance (adapter-owned) and never enter the `Manual`; the `Manual` stays protocol-agnostic.** *Protocol binding* means the per-instance access mechanism for the tool's **application-layer** protocol (WoT TD `forms` over HTTP/CoAP; MCP's JSON-RPC session and name assembly) — distinct from the **transport** it runs over (stdio, SSE, TCP). The split is by *content*:

* **Binding → `Tool`:** WoT TD `forms` (href, method, contentType) and `securityDefinitions`; the MCP `ClientSession` and `<App>__op` assembly. These implement `invoke`/`focus`/`observe` and are never serialized into a `Manual`.
* **Manual-level → `Manual`:** operation/property/signal names, natural-language semantics (functional description, preconditions, effects, behavior, usage protocols & safety), and JSON-Schema data shapes (the spec types' `parameters`/`schema` dicts).

A `Manual` has two interchangeable **provenance channels** producing the same type, reconciled by type-level `Manual.id` ([ADR-0007](0007-manual-record-separation.md)):

1. **Adapter synthesis** from a native description — reliably yields names + JSON Schema, plus whatever semantics the source carries.
2. **Markdown authoring** — yields the rich natural-language semantics native formats lack.

An adapter may synthesize a thin manual while a curated repository supplies a richer one for the same id; or, as the WoT sketch shows, an adapter may itself load the authored Markdown for the Thing it binds. `ManualParser` is thus a shared *producer*, owned by neither side. **No separate instance-description type is introduced** — the `WorkspaceAdapter` already splits the native description into (binding → `Tool`) + (data shape/semantics → `Manual`); a parallel type would re-implement that boundary.

### How much the hand-authored channel extracts

Regex-lifting loose Markdown into rigid fields proved brittle (mis-cased/mistyped headings, indentation, or lowercase sub-markers silently drop affordances; a missing hint crashes), and no consumer reads the fine-grained fields today — `DefaultActStrategy` ignores its `manual`, and `_synth_manual` already leaves properties/signals/usage empty. Since adapter sources are born structured and only hand-authored manuals use Markdown, the parsing tax is paid eagerly for structure nothing reads. Therefore:

* **Now:** the Markdown channel yields an **envelope** — `id`, `metadata`, `description`, `usage_protocols`, and verbatim `raw_text` — and does *not* lift bullet lists or `(type, range)` hints into `observable_properties`/`signals`/`operations`/schema. Those fields stay defined on `Manual` (the adapter channel fills them) but come back empty from Markdown until a consumer reads them. `raw_text` (not a reflowed rendering) is what feeds LLM context. This extends the "not lifted into discrete fields until a strategy consumes them" rule — already applied to operation sub-bullets — to properties and signals. Section-level slicing (e.g. only the Operations section for a binding, or Usage & Safety for a suspend judgment) is a lazy view over `raw_text` — split on `#` headings on demand — not stored chunks, so the whole manual is just `raw_text` with no reassembly to drift.
* **Later:** when a consumer needs machine-readable schemas *from hand-authored manuals* (e.g. a deterministic `ActStrategy` validating params, a `SituateStrategy` reading `signals`), move that content to a **structured header + prose body** — YAML/TOML front-matter parsed and schema-validated, prose kept free-text — not more regex, not full JSON. This aligns the hand-authored path with the JSON Schema the adapter channel already carries. *(Realized at names-level depth — an optional per-section ```yaml block declaring names + required keys — by [ADR-0018](0018-manual-merge-policy-and-authored-interface.md); full JSON-Schema authoring remains this "Later".)*

### Consequences

* `Manual`s stay portable and shareable; bindings are an instance concern; no new abstraction (reuses `Manual.id`, the `parameters`/`schema` dicts, and manual retrieval).
* The two adapters — ARE (synthesis) and WoT (authored Markdown) — are two provenance channels of one boundary, not a contradiction.
* The hand-authored channel has no fragile field extraction to fail silently; LLM context stays faithful via `raw_text`.
* Deferred: a **merge policy** when both channels fire for one id (realized MCP-scoped in [ADR-0018](0018-manual-merge-policy-and-authored-interface.md)); adapter-synthesized manuals stay semantics-thin (ADR-0003), so LLM reasoning still needs authored Markdown; `Manual` gains a `raw_text` field, and full schema fidelity for hand-authored tools waits on the later structured-header format.

## Considered Options

* **Separate `InstanceDescription` type** alongside each `Manual`. Rejected: duplicates the split the adapter already performs, forcing consumers to correlate two objects where a protocol-agnostic `Manual` plus a live `Tool` suffice.
* **Fold the binding into the `Manual`** (a per-instance manual). Rejected: destroys protocol-agnosticism and cross-instance reuse, re-coupling the shared description to one application-layer protocol — against the A&A model and ADR-0007.
* **Binding on the instance; protocol-agnostic `Manual` from either provenance (chosen).** Keeps the `Manual` reusable, needs no new type, subsumes both adapters; leaves a merge policy open and keeps synthesized manuals thin — both acceptable and deferrable.

## Links

* Depends on [ADR-0003](0003-adapters-not-tool-authoring.md), [ADR-0004](0004-tool-usage-interface.md)
* Refines [ADR-0007](0007-manual-record-separation.md) (type-level `Manual.id` is the reconciliation key); relates to [ADR-0014](0014-tool-identity-globally-unique.md) (instance identity vs. type-level manual id)
