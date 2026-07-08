# Phase fusion via a per-cycle threaded result

* Status: accepted
* Date: 2026-07-05

## Context and Problem Statement

With every phase independently pluggable ([ADR-0010](0010-pluggable-phase-strategies.md)), how can one underlying computation (e.g., a single model call) serve multiple phases at once — down to one call per cycle — without hardcoding which phases are allowed to fuse, and without a combined-Protocol interface for every possible fusion grouping?

## Decision Drivers

* Avoid a combinatorial explosion of fusion-grouping interfaces, one per possible subset of phases
* Support fusion starting from any phase — including Observe or Reflect — not only the later phases

## Considered Options

* Hidden caching inside a strategy object that implements multiple phase Protocols: the object remembers its own prior output across calls, keyed and invalidated internally, so a later phase can reuse what an earlier one computed
* Dedicated combined-Protocol interfaces per common fusion grouping (e.g., a single "PlanningStrategy" spanning Situate+Reason+Act)
* One shared, per-cycle-scoped result value (`TickResult`) threaded through all five phases, with each phase's strategy filling in only what's still missing

## Decision Outcome

Chosen option: "One shared, per-cycle-scoped result value". Any fusion boundary (Observe-only, Reflect-through-Act, any subset) is representable by which fields a given phase happens to fill in, with no combinatorial set of interfaces required. `DecisionCycle.tick()` calls each phase's strategy only if the relevant field is still `None`. Because `TickResult`'s lifetime is exactly one `tick()` call, it also carries no risk of staleness across an interrupt — but that's a side effect of the value being scoped to a single call, not the reason for choosing this option over the combined-Protocol alternative.

### Positive Consequences

* Any fusion boundary is expressible without adding new types
* Immune to interrupt-related staleness, since nothing survives past the `tick()` call it was produced in

### Negative Consequences

* Every phase Protocol's signature includes the shared result type as both input and output, coupling all five phases to this one shared type
* The two rejected alternatives (hidden per-object caching; dedicated combined-Protocols) were both explicitly considered and rejected — noted here so they aren't re-litigated without new information

## Links

* Depends on [ADR-0010](0010-pluggable-phase-strategies.md)
* `Step` (README.md's API Sketch) is exactly what threads through Reason and Act via this mechanism
