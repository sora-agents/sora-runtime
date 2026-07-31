# Per-cycle threaded result: field-gated phase short-circuiting

* Status: proposed
* Date: 2026-07-12

## Context and Problem Statement

With every phase independently pluggable ([ADR-0010](0010-pluggable-phase-strategies.md)), how does a later phase **reuse or skip** work an earlier phase already produced — a cached plan that makes Reason's inference unnecessary, params already resolved before Act — without hardcoding which phases may pre-fill which, without a combined-Protocol interface per grouping, and without bolting on a separate caching mechanism? (One computation *may* still pre-fill several phases at once — *fusion* — but that is a narrow, opt-in use, not what this mechanism is for; [ADR-0021](0021-llm-calls-as-async-internal-actions.md) records why a single call is not built to span phases.)

## Decision Drivers

* Avoid a combinatorial explosion of per-grouping interfaces, one per possible subset of phases
* No separate caching mechanism: skipping already-produced work must fall out of the same value that carries a phase's output, not a side store
* Keep any forward-fill (a phase pre-filling a later phase's field) expressible without new types, whichever phases are involved

## Considered Options

* Hidden caching inside a strategy object that implements multiple phase Protocols: the object remembers its own prior output across calls, keyed and invalidated internally, so a later phase can reuse what an earlier one computed
* Dedicated combined-Protocol interfaces per common fusion grouping (e.g., a single "PlanningStrategy" spanning Situate+Reason+Act)
* One shared, per-cycle-scoped result value (`TickResult`) threaded through all five phases, with each phase's strategy filling in only what's still missing

## Decision Outcome

Chosen option: "One shared, per-cycle-scoped result value". Which phases run is governed by which fields are already filled: `DecisionCycle.tick()` calls each phase's strategy only if the relevant field is still `None`. A field filled by an earlier phase (a cached plan set before Reason, a step already resolved) short-circuits the later phase — and any forward-fill boundary (Observe-only, Reflect-through-Act, any subset) is representable by which fields a phase fills, with no combinatorial set of interfaces and no separate cache. Because `TickResult`'s lifetime is exactly one `tick()` call, it also carries no risk of staleness across an interrupt — a side effect of the single-call scope, not the reason it was chosen over the combined-Protocol alternative.

This gate is sound only because, for Reason and Act, the field is the phase's *entire* output: Reason's whole job is to produce `step`, Act's to produce `invocation`. So finding that field already filled — by an earlier phase that computed it in the same call — means nothing is left to do, and skipping is safe. That "an earlier phase pre-fills a later phase's field" is the forward-fill the gate exists for — most often a cheap short-circuit (a cached plan set before Reason, so Reason is skipped), and in its fused form one Situate call that also sets `step` and `invocation` so Reason and Act are skipped (a narrow use; [ADR-0021](0021-llm-calls-as-async-internal-actions.md) explains why single-call fusion isn't pursued). Situate is the exception: its `activity` field records only the *selection*, but Situate also mutates working memory (re-focusing tools, loading/unloading manuals, filtering percepts), which no field captures. So a filled `activity` does not mean Situate's work is done — Situate is therefore never skipped: it always runs, selecting only when `activity` is unset. The gate governs skipping a *later* phase whose output was fused forward, never re-entry of the head-of-chain phase.

### Positive Consequences

* Any fusion boundary is expressible without adding new types
* Immune to interrupt-related staleness, since nothing survives past the `tick()` call it was produced in

### Negative Consequences

* Every phase Protocol's signature includes the shared result type as both input and output, coupling all five phases to this one shared type
* The two rejected alternatives (hidden per-object caching; dedicated combined-Protocols) were both explicitly considered and rejected — noted here so they aren't re-litigated without new information

## Links

* Depends on [ADR-0010](0010-pluggable-phase-strategies.md)
* Related: [ADR-0021](0021-llm-calls-as-async-internal-actions.md) records why a single call is not built to span multiple phases (fusion); this mechanism stands on its own as the field-gating / short-circuit seam (cached plan skips infer, resolved step skips Reason, present invocation skips Act's bind)
* `Step` (README.md's API Sketch) is exactly what threads through Reason and Act via this mechanism
