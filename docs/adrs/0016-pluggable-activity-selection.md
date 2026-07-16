# Activity selection is Situate's own pluggable sub-strategy, defaulting to round-robin

* Status: proposed
* Date: 2026-07-16

## Context and Problem Statement

"Which ready activity runs this cycle" *is* the agent's scheduler. The default Situate baked this
decision into `DefaultSituateStrategy.situate` as `ready[0]` — the oldest still-ready activity, since
`WorkingMemory.activities` is creation-ordered and never reordered. That is *static priority-by-age*:
an activity that stays `READY` across cycles (internal-only work, a wait step that never goes
`RUNNING`/`BLOCKED`) is reselected every cycle and **starves** younger activities, undercutting the
concurrency the five-phase cycle is meant to provide. How do we make selection fair by default while
keeping it swappable, *without* forcing anyone who wants a different scheduling policy to also
re-author Situate's mechanical activity-creation and working-memory adjustment?

## Decision Drivers

* Selection is the scheduler and deserves a fair default (anti-starvation), still with no model call.
* Selection deserves a dedicated extension point — richer policies (priority, aging, deadlines, an
  LLM-based scheduler) should plug in without re-implementing the rest of Situate.
* The selection policy must not leak into `cycle.py` — the pluggability stays inside Situate.
* [ADR-0010](0010-pluggable-phase-strategies.md) already lists "Situate (activity prioritization)" as
  a swappable judgment call; this refines *how* it's swapped, without a whole-`SituateStrategy` swap.
* [ADR-0008](0008-protocol-based-extensibility.md): extension points are `Protocol`s, not base classes.

## Considered Options

* **(a) Keep selection inline; swap the whole `SituateStrategy` for a custom policy.**
* **(b) A dedicated `ActivitySelectionStrategy` sub-strategy injected into `DefaultSituateStrategy`,
  with `RoundRobinActivitySelection` as the default.**

## Decision Outcome

Chosen option: **(b)**. Selection is intrinsically part of "Situate selects an activity," so it is a
sub-strategy *of* Situate rather than a phase of its own or a concern of `cycle.py`.
`DefaultSituateStrategy` gains a constructor taking an `ActivitySelectionStrategy` (defaulting to
`RoundRobinActivitySelection`) and delegates its pick to it; the `select` signature mirrors the
phase-strategy convention (`ready` set, `wm` read-model, `cycle` engine handle) and is `async` so a
future model-backed scheduler can consult memory or a model. `RoundRobinActivitySelection` is the
mechanical default: it carries a last-selected-id cursor across cycles and rotates over the ready
set, wrapping via modulo; a cold start or a last-pick that is no longer ready falls back to the
oldest — so behavior matches the old priority-by-age default until an activity lingers `READY`, at
which point selection rotates instead of pinning it. `select` returns only the scheduling decision
(`Activity | None`); the caller folds it into `TickResult`. Fusing step/invocation stays a full
`SituateStrategy` concern, not selection's, so this does not widen the sub-strategy's remit.

### Positive Consequences

* The default is fair (anti-starvation) with no model call, preserving the concurrency guarantee.
* Scheduling policy has its own seam: priority/aging/deadline/LLM schedulers are alternative
  `ActivitySelectionStrategy` implementations, none requiring Situate's other logic to be rewritten.
* `cycle.py` stays selection-agnostic; the boundary lives entirely inside Situate.
* A custom full `SituateStrategy` can still ignore the sub-strategy and select however it likes.

### Negative Consequences

* `DefaultSituateStrategy` grows a constructor, and the default now carries genuine cross-cycle cursor
  state (feasible because the strategy instance persists for the agent's lifetime — precedent:
  `DefaultReflectStrategy`'s task set). A fresh instance reproduces the old first-ready outcome, so
  existing tests that construct a new strategy per call are unaffected.

## Pros and Cons of the Options

### (a) Keep selection inline; swap the whole `SituateStrategy`

* Good, because it adds no new type — the existing `SituateStrategy` seam already exists.
* Bad, because changing *only* the scheduling policy forces re-authoring `_create_activities_from_messages`
  and `_adjust_working_memory`, which have nothing to do with selection — a large, error-prone surface
  to reproduce just to change one decision.

### (b) `ActivitySelectionStrategy` sub-strategy with a round-robin default

* Good, because it isolates the scheduler behind a small `Protocol`, ships a fair default, and keeps
  Situate's mechanical activity-creation/wm-adjustment reusable across selection policies.
* Bad, because it introduces one more extension point and cross-cycle state to reason about.

## Links

* Refines [ADR-0010](0010-pluggable-phase-strategies.md)
* Follows [ADR-0008](0008-protocol-based-extensibility.md)
* Concerns the Situate phase of [ADR-0009](0009-five-phase-decision-cycle.md)
