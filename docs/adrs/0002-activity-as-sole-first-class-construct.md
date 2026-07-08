# Activity as the sole first-class construct (no explicit BDI states)

* Status: accepted
* Date: 2026-07-05

## Context and Problem Statement

S-ORA aligns with the CoALA conceptual framework and draws further inspiration from the Belief-Desire-Intention (BDI) model and practical implementations such as Jason. Given that lineage, should the runtime explicitly model beliefs, desires, and intentions as distinct types, or keep a single, simpler unit of work?

## Decision Drivers

* Avoid premature conceptual complexity in the core type system
* Keep the door open for a richer mental-state model without forcing it on every adopter

## Considered Options

* Model activities only, with no explicit belief/desire/intention types
* Explicitly map "activity" to "intention" and add first-class belief/desire types

## Decision Outcome

Chosen option: "Model activities only", because S-ORA sticks with activities by design — ascribing mental states like beliefs, desires, and intentions is left as a possible extension built on top of the core model, not baked into it.

### Positive Consequences

* Simpler core type system: one unit of work (goal + context + lifecycle state), not a triad of interacting types
* Doesn't force adopters into a BDI-style mental model to use the runtime

### Negative Consequences

* Forgoes directly-borrowed BDI machinery (e.g., explicit belief revision) that a more sophisticated agent might want
* Anyone wanting a BDI-faithful model must build the belief/desire/intention layer themselves on top of Activity

## Links

* Informs `ProceduralMemory` (README.md's Memory section) — procedural knowledge attaches to activities directly, not to a separate "intention" type
