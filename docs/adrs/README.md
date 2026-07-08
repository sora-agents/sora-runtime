# Architecture Decision Records

This folder records the S-ORA runtime's architectural decisions using [MADR](https://adr.github.io/madr/) (Markdown Architectural Decision Records).

## Conventions

- One ADR per file, named `NNNN-kebab-case-title.md`. Numbers are 4-digit, zero-padded, assigned once, and never reused or reordered.
- New ADRs start from [`template.md`](template.md).
- Status is one of `proposed`, `accepted`, `rejected`, `deprecated`, or `superseded by ADR-NNNN`. Superseding a decision means writing a *new* ADR and updating the old one's status line — an accepted ADR's decision is never edited in place, so the log stays an accurate history of when and why things changed.
- Numbering starts roughly foundational → specific (see Planned below), but every new ADR simply takes the next available number regardless of theme — the numbering is chronological, not thematic.
- Plain markdown only; no ADR tooling (e.g. adr-tools, log4brains) required.

## Index

| ADR | Title | Status | Theme |
|-----|-------|--------|-------|
| [0001](0001-python-asyncio-runtime.md) | Python 3.12+ (asyncio) as the core runtime | accepted | Runtime & scope |
| [0002](0002-activity-as-sole-first-class-construct.md) | Activity as the sole first-class construct (no explicit BDI states) | accepted | Runtime & scope |
| [0003](0003-adapters-not-tool-authoring.md) | Adapters import tools; the runtime never authors them | accepted | Runtime & scope |
| [0004](0004-tool-usage-interface.md) | Tool usage interface: properties, signals, operations — async by design | accepted | Tool & workspace model |
| [0005](0005-workspace-grouping.md) | Workspace groups tools sharing a connection; per-tool address override | accepted | Tool & workspace model |
| [0006](0006-workspace-join-leave-lifecycle.md) | Join/leave as deliberate actions; discovery kept distinct from reconnection | accepted | Tool & workspace model |
| [0007](0007-manual-record-separation.md) | Manuals and tool/workspace records stored as separate entities | accepted | Memory Representations |
| [0008](0008-protocol-based-extensibility.md) | Every extension point is an open Protocol, never a closed enum or base class | accepted | Extensibility |
| [0009](0009-five-phase-decision-cycle.md) | Five fixed decision-cycle phases, one external action per cycle | accepted | Decision cycle |
| [0010](0010-pluggable-phase-strategies.md) | Every phase has an independently pluggable strategy | accepted | Decision cycle |
| [0011](0011-phase-fusion-via-threaded-result.md) | Phase fusion via a per-cycle threaded result | accepted | Decision cycle |
| [0012](0012-percepts-vs-messages.md) | Percepts and messages kept as two distinct channels | accepted | Decision cycle |
| [0013](0013-shared-instances-narrow-dependencies.md) | Agent/DecisionCycle share instances; narrow explicit dependencies everywhere | accepted | Composition & wiring |

## Planned

None currently pending — 13 architectural decisions written up above.
