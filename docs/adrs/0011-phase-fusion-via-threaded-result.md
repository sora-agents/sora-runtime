# Phase fusion via a per-cycle threaded result

* Status: proposed
* Date: 2026-07-12

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

This gate is sound only because, for Reason and Act, the field is the phase's *entire* output: Reason's whole job is to produce `step`, Act's to produce `invocation`. So finding that field already filled — by an earlier phase that computed it in the same call — means nothing is left to do, and skipping is safe. That "an earlier phase pre-fills a later phase's field" is the fusion the gate exists for (e.g. one Situate call that also sets `step` and `invocation`, so Reason and Act are skipped). Situate is the exception: its `activity` field records only the *selection*, but Situate also mutates working memory (re-focusing tools, loading/unloading manuals, filtering percepts), which no field captures. So a filled `activity` does not mean Situate's work is done — Situate is therefore never skipped: it always runs, selecting only when `activity` is unset. The gate governs skipping a *later* phase whose output was fused forward, never re-entry of the head-of-chain phase. See [docs/phase-2-findings.md](../phase-2-findings.md).

### Positive Consequences

* Any fusion boundary is expressible without adding new types
* Immune to interrupt-related staleness, since nothing survives past the `tick()` call it was produced in

### Negative Consequences

* Every phase Protocol's signature includes the shared result type as both input and output, coupling all five phases to this one shared type
* The two rejected alternatives (hidden per-object caching; dedicated combined-Protocols) were both explicitly considered and rejected — noted here so they aren't re-litigated without new information

## Links

* Depends on [ADR-0010](0010-pluggable-phase-strategies.md)
* `Step` (README.md's API Sketch) is exactly what threads through Reason and Act via this mechanism
