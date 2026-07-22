# Architecture Decision Records

This folder records the S-ORA runtime's architectural decisions using [MADR](https://adr.github.io/madr/) (Markdown Architectural Decision Records).

## Conventions

- One ADR per file, named `NNNN-kebab-case-title.md`. Numbers are 4-digit, zero-padded, assigned once, and never reused or reordered.
- New ADRs start from [`template.md`](template.md).
- Status is one of `proposed`, `accepted`, `rejected`, `deprecated`, or `superseded by ADR-NNNN`. Superseding a decision means writing a *new* ADR and updating the old one's status line — an accepted ADR's decision is never edited in place, so the log stays an accurate history of when and why things changed.
- **Lifecycle during design.** While the runtime is being built (README-driven design), ADRs stay `proposed`: they capture current design intent and *may be edited in place* as the design is refined. An ADR is promoted to `accepted` only once its decision is realized in code and has survived that realization (target: the first tagged release). The append-only / never-edit-in-place rule above applies from `accepted` onward, not to `proposed` drafts.
- Numbering starts roughly foundational → specific (see Planned below), but every new ADR simply takes the next available number regardless of theme — the numbering is chronological, not thematic.
- Plain markdown only; no ADR tooling (e.g. adr-tools, log4brains) required.

## Index

| ADR | Title | Status | Theme |
|-----|-------|--------|-------|
| [0001](0001-python-asyncio-runtime.md) | Python 3.12+ (asyncio) as the core runtime | proposed | Runtime & scope |
| [0002](0002-activity-as-sole-first-class-construct.md) | Activity as the sole first-class construct (no explicit BDI states) | proposed | Runtime & scope |
| [0003](0003-adapters-not-tool-authoring.md) | Adapters import tools; the runtime never authors them | proposed | Runtime & scope |
| [0004](0004-tool-usage-interface.md) | Tool usage interface: properties, signals, operations — async by design | proposed | Tool & workspace model |
| [0005](0005-workspace-grouping.md) | Workspace groups tools sharing a connection; per-tool address override | proposed | Tool & workspace model |
| [0006](0006-workspace-join-leave-lifecycle.md) | Join/leave as deliberate actions; discovery kept distinct from reconnection | proposed | Tool & workspace model |
| [0007](0007-manual-record-separation.md) | Manuals and tool/workspace records stored as separate entities | proposed | Memory Representations |
| [0008](0008-protocol-based-extensibility.md) | Every extension point is an open Protocol, never a closed enum or base class | proposed | Extensibility |
| [0009](0009-five-phase-decision-cycle.md) | Five fixed decision-cycle phases, one external action per cycle | proposed | Decision cycle |
| [0010](0010-pluggable-phase-strategies.md) | Every phase has an independently pluggable strategy | proposed | Decision cycle |
| [0011](0011-phase-fusion-via-threaded-result.md) | Phase fusion via a per-cycle threaded result | proposed | Decision cycle |
| [0012](0012-percepts-vs-messages.md) | Percepts and messages kept as two distinct channels | proposed | Decision cycle |
| [0013](0013-shared-instances-narrow-dependencies.md) | Agent/DecisionCycle share instances; narrow explicit dependencies everywhere | proposed | Composition & wiring |
| [0014](0014-tool-identity-globally-unique.md) | Tool identity is globally unique, guaranteed by the protocol adapter | proposed | Tool & workspace model |
| [0015](0015-manuals-protocol-agnostic-adapter-boundary.md) | Manuals stay protocol-agnostic; protocol bindings live on the tool instance | proposed | Tool & workspace model |
| [0016](0016-pluggable-activity-selection.md) | Activity selection is Situate's own pluggable sub-strategy, defaulting to round-robin | proposed | Decision cycle |
| [0017](0017-parameter-grounding-in-reason.md) | Parameter grounding is a Reason decision (references + escalation); Act stays mechanistic | proposed | Decision cycle |
| [0018](0018-manual-merge-policy-and-authored-interface.md) | Manual merge policy (adapter owns interface, author owns prose) + optional names-level authored interface | proposed | Tool & workspace model |
| [0019](0019-blocked-state-machinery-and-percept-storage.md) | Blocked-state machinery: mechanical Observe-hosted suspend/resume + split percept storage | proposed | Decision cycle |

## Planned

None currently pending — 16 architectural decisions written up above, all `proposed` during README-driven design (see the lifecycle note in Conventions; each is promoted to `accepted` once realized in code).
