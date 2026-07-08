# Five fixed decision-cycle phases, one external action per cycle

* Status: accepted
* Date: 2026-07-05

## Context and Problem Statement

The runtime needs a core control-flow model for how an agent progresses its activities each cycle. What shape should this control flow take, and how much should an agent be allowed to commit to in a single cycle?

## Decision Drivers

* CoALA's decision-cycle framing
* Need bounded, auditable progress per cycle
* Need an explicit place for suspending and resuming an activity on a tool signal

## Considered Options

* A single, undifferentiated reasoning step per cycle
* Five phases — Observe, Reflect, Situate, Reason, Act — executing at most one external action per cycle

## Decision Outcome

Chosen option: "Five phases, one external action per cycle", because it gives each concern (perceiving, learning from outcomes, selecting an activity, planning, executing) its own seam — later made independently pluggable (see [ADR-0010](0010-pluggable-phase-strategies.md)) — while bounding what an agent commits to per cycle, which is what makes mid-cycle interruption meaningful (see README.md's Decision Cycle section).

### Positive Consequences

* Each phase is independently understandable, testable, and (later) pluggable
* Bounding one external action per cycle keeps the cycle auditable and interruptible

### Negative Consequences

* Fixes the runtime's control flow to this exact five-phase shape; changing the phase set or their order would ripple through every strategy Protocol and the dispatch logic in `DecisionCycle.tick()`

## Links

* Refined by [ADR-0010](0010-pluggable-phase-strategies.md), [ADR-0011](0011-phase-fusion-via-threaded-result.md)
