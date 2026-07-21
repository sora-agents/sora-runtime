# Merge policy for a Manual's two provenance channels, and a names-level authored interface

* Status: proposed
* Date: 2026-07-20

## Context and Problem Statement

[ADR-0015](0015-manuals-protocol-agnostic-adapter-boundary.md) established that a `Manual` has two
interchangeable provenance channels reconciled by type-level `Manual.id` — **adapter synthesis** (a
native description → structured `operations`/`observable_properties`/`signals` with JSON Schema, no
`raw_text`) and **hand-authored Markdown** (`raw_text` + semantics, structured specs empty). It
deliberately deferred two things: the **merge policy** for when both channels fire for one id, and a
later **structured-header format** that would let hand-authored manuals carry machine-readable
schemas. Both now have a first consumer: an MCP tool should carry the adapter's operation schemas
(for grounding) *and* the author's Usage Protocols & Safety (for planning) as one manual. Which side
wins per field, and can the two even be cross-checked, or must we trust that a paired id is correct?

## Decision Drivers

* The structured usage interface (names + JSON Schema) is the *adapter's* to own — it is native and
  authoritative; re-authoring JSON Schema by hand duplicates it and drifts (ADR-0003/ADR-0015).
* The reasoning semantics (functional description, usage protocols & safety) are the *author's* to
  own — native descriptions are semantics-thin.
* A silent merge hides a real error: pairing the wrong authored manual to a tool, or documenting an
  operation the tool doesn't expose, should fail loud, not degrade reasoning invisibly.
* ADR-0015 warned against brittle regex-lifting of prose bullets into typed fields; any structure the
  author declares must be *explicit*, not extracted from free prose.
* Author burden must stay low: MCP already supplies the full JSON Schema, so the authored side needs
  only enough to *validate* the interface, not restate it.

## Considered Options

* **Id-only reconciliation** — trust a paired id, merge fields, validate nothing beyond `id`.
* **Full structured specs in authored manuals** — the author hand-writes complete
  `OperationSpecification`/schema; cross-validate names and schema.
* **Names-level authored interface + field-per-channel merge (chosen)** — the author optionally
  declares operation/property/signal *names* and their *required keys* in an explicit per-section
  block; merge cross-validates that against the adapter's specs and merges the rest by owner.

## Decision Outcome

Chosen: **names-level authored interface + field-per-channel merge**, realized as `merge_manuals`
plus an optional interface block the `MarkdownManualParser` lifts.

**Merge policy** (`merge_manuals(adapter, authored) -> Manual`), per field:

* `id` — must be equal; a mismatch raises `ManualMergeError` (the reconciliation key — a mismatch is
  almost always a wrong pairing).
* `operations` / `observable_properties` / `signals` — the **adapter's** structured specs (full,
  native JSON Schema). The authored side's specs are used only for validation, then discarded.
* `raw_text` — the **authored** manual's (the adapter has none).
* `description` — the authored manual's when non-empty, else the adapter's.
* `metadata` — a union, authored winning on conflict (the adapter's `source` survives, since an
  authored manual won't set it).

**Interface validation.** Where the authored manual *declares* a structured interface for an
affordance kind (see below), `merge_manuals` fails loud (`ManualMergeError`) if its names don't match
the adapter's for that kind, or if a required key it names is absent from the adapter's schema for
that affordance. Validation is *per affordance kind*: a kind the author leaves undeclared passes the
adapter's specs through unchecked (opt-in). Required-key checking is skipped for an affordance whose
adapter schema carries no `properties` key at all (the adapter models no fields there — e.g. ARE's
synthesized observable properties) — names still validate, fields simply can't.

**Authored interface block.** This realizes ADR-0015's deferred "structured header + prose body" at a
names-level depth. Inside `# Operations` / `# Observable Properties` / `# Signals`, an author may add
an optional fenced ```yaml block — a list of `{name: <str>, required: [<str>, ...]}` — which the
parser lifts into that section's structured spec field (name + a minimal JSON-Schema-shaped dict
carrying just the required keys); the prose stays verbatim in `raw_text`. No block → that spec list
stays empty (the ADR-0015 envelope, unchanged). Full JSON Schema stays the adapter's job.

**Loading channel.** A `ManualSource` Protocol (`async get(manual_id) -> Manual | None`) is the seam
by which an adapter pairs authored manuals with its synthesized ones; it returns a *parsed* `Manual`
(owning fetch+parse) so a remote manual catalogue is a drop-in for the shipped local
`DirectoryManualSource`. This is MCP-scoped for now (the `McpWorkspaceAdapter` takes an optional
`manual_source` and merges in `discover()`; the merged manual is what the tool carries and Join
persists, so `restore()`/`connect()` rebuild from it with no re-load). The WoT/Thing-Description
pairing reuses `merge_manuals` unchanged and is split to a later ADR.

### Positive Consequences

* The planner sees one manual with both the adapter's param schemas (grounding) and the author's
  usage protocols (planning) — neither channel had both before.
* A wrong pairing or a documented-but-absent operation fails loud at discover time, not as degraded
  reasoning later.
* Low author burden: names + required keys, not hand-written JSON Schema; and it's fully optional.
* No brittle prose extraction — the interface is an explicit, machine-readable block.

### Negative Consequences

* An interface block must declare an affordance kind's *full* name set to validate (partial
  declarations fail) — deliberate, but it means a showcase manual either declares the whole tool
  surface or omits the block (the gaia2 example omits it, staying prose-only).
* A second, lighter schema representation (names + required) now coexists with the adapter's full
  JSON Schema; a future full-schema authored format (ADR-0015's endpoint) would subsume it.

## Pros and Cons of the Options

### Id-only reconciliation

* Good, because trivial and zero author burden.
* Bad, because it can't catch a wrong pairing or a stale/incorrect authored interface — the exact
  failure this reconciliation is most likely to hit.

### Full structured specs in authored manuals

* Good, because it enables hand-authored-only tools (no adapter) and the richest validation.
* Bad, because authors re-write JSON Schema the adapter already provides (duplication, drift, high
  burden) for no v0.1.0 consumer — speculative generality against ADR-0003/ADR-0015's "cap richness
  at the source."

### Names-level authored interface + field-per-channel merge (chosen)

* Good, because it validates the interface (names + required keys) with minimal burden while letting
  the adapter own the full schema; explicit block, no prose extraction; optional and per-kind.
* Bad, because it introduces a second, partial schema shape and requires whole-name-set declaration
  to validate a kind.

## Links

* Refines [ADR-0015](0015-manuals-protocol-agnostic-adapter-boundary.md) (realizes its deferred merge
  policy and its "Later" structured-header format, at names-level depth, MCP-scoped)
* Depends on [ADR-0003](0003-adapters-not-tool-authoring.md) (richness capped at the source),
  [ADR-0007](0007-manual-record-separation.md) (type-level `Manual.id` as the reconciliation key)
* Relates to [ADR-0014](0014-tool-identity-globally-unique.md) (instance identity vs. type-level
  manual id)
