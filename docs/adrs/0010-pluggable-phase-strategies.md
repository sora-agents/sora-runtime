# Every phase has an independently pluggable strategy

* Status: accepted
* Date: 2026-07-05

## Context and Problem Statement

Only `ReasonStrategy` was pluggable at first. Operational experience — moving activity-completion detection between Observe and Reflect, and moving tool-parameter binding between Reason and Act to fight tool hallucination — showed that every phase involves a real judgment call worth making swappable, and that "pluggable" must not be conflated with "this phase always makes a model call."

## Decision Drivers

* Empirical history of moving specific judgment calls between phases to trade off cost against reliability
* Need default, non-model implementations so pluggability doesn't force a model-call tax on every phase
* Avoid privileging Reason as the only phase worth customizing, when Reflect, Situate, and Act carry comparable judgment calls

## Considered Options

* Keep only Reason pluggable
* Make all five phases independently pluggable, each with a mechanical/deterministic default, with no requirement that any strategy use a model call

## Decision Outcome

Chosen option: "All five phases independently pluggable", because Reflect (completion/failure judgment), Situate (activity prioritization), and Act (tool-parameter binding) are graded, ambiguous judgment calls in the same way Reason's planning is — there's no principled reason to hardcode them while Reason is swappable. Observe and Act default to mechanical/deterministic behavior; nothing in the runtime requires a phase's strategy to invoke a model.

### Positive Consequences

* Reflect, Situate, and Act's judgment calls can each be swapped, cached, or made model-backed independently of one another
* A fully deterministic, zero-model-call configuration remains a valid, first-class configuration, not a degraded one

### Negative Consequences

* Five Protocols to implement and default instead of one
* Nothing enforces that independently-configured strategies cooperate beyond the shared result contract (see [ADR-0011](0011-phase-fusion-via-threaded-result.md)) — a badly matched combination is possible

## Links

* Depends on [ADR-0009](0009-five-phase-decision-cycle.md)
* Enabled by [ADR-0011](0011-phase-fusion-via-threaded-result.md)
