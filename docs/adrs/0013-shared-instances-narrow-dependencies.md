# Agent/DecisionCycle share instances; narrow explicit dependencies everywhere

* Status: proposed
* Date: 2026-07-05

## Context and Problem Statement

Two related but separable questions arise once `Agent` holds `DecisionCycle` plus other shared state (`EnvironmentRegistry`, memory modules, transport):

1. **Ownership topology** — should these components form an ownership hierarchy, with `Agent` owning `DecisionCycle` and cross-cutting access flowing through that ownership (e.g. a back-reference from `DecisionCycle` to its owning `Agent`)? Or should they be constructed as peers sharing the same underlying instances, so neither needs to reach through the other?
2. **Call convention** — should `DecisionCycle.tick()` and every `ExternalAction`/`InternalAction` receive one bundled object carrying everything they might conceivably need, or only the specific, narrow dependencies each actually touches?

Concretely: memory modules and transport originally lived only on `DecisionCycle`, while `Agent` held `DecisionCycle` and `EnvironmentRegistry`. This made cross-cutting calls like `EnvironmentRegistry.restore()` awkward — it needs `SemanticMemory`, reachable only by reaching through `agent.cycle` from the `Agent` side — even though memory conceptually belongs to the agent as a whole, not to one iteration of its decision loop. A proposed fix to question 1 — a stored back-reference from `DecisionCycle` to its owning `Agent` — was evaluated and rejected: it requires two-phase construction, an "unbound until wired" state, and reintroduces the exact coupling `DecisionCycle`'s self-sufficiency was meant to avoid.

## Decision Drivers

* Avoid circular ownership between `Agent` and `DecisionCycle`
* Avoid any object having an "unbound until wired" state
* Keep `DecisionCycle.tick()` and `ExternalAction`/`InternalAction` implementations testable with minimal fixtures
* Keep every component declaring exactly what it depends on

## Considered Options

For the ownership topology (question 1):
* `DecisionCycle` holds a back-reference to its owning `Agent`, set post-construction
* `Agent` and `DecisionCycle` are constructed as peers from the same shared memory/transport instances, with no reference from either to the other

For the call convention (question 2):
* Every phase and action receives a single bundled object (e.g. the whole `Agent`) carrying every dependency it might need
* Every phase and action receives only the narrow, specific dependencies it actually needs (`tools`, `cycle`) as explicit parameters

## Decision Outcome

Chosen: peer instances with no back-reference (question 1), and narrow explicit dependencies on every signature (question 2) — two independent choices that happen to reinforce each other, not one inseparable package.

Sharing instances instead of a back-reference resolves question 1 on its own: `agent.semantic` and `agent.cycle.semantic` are the identical object by construction, so cross-cutting calls read naturally without reaching through `agent.cycle`, and there's no ownership cycle or two-phase construction to reason about. It doesn't by itself force any particular answer to question 2 — one could still pass a bundled `Agent` into every call. Narrow parameters were chosen on their own merits (testability with minimal fixtures, every action declaring exactly what it touches): `DecisionCycle.tick(tools)` takes only the one thing it doesn't already hold; `ExternalAction.execute(tools, cycle, **kwargs)` and `InternalAction.execute(cycle, **kwargs)` take only what each actually touches.

### Positive Consequences

* No circular references anywhere in the runtime
* Every action implementation declares exactly what it touches (`tools`, `cycle`) instead of an opaque `Agent`
* `DecisionCycle` remains fully valid the instant it's constructed — no unbound window

### Negative Consequences

* `DecisionCycle.tick()` and every action's `execute()` gained an explicit parameter instead of a single bundled object
* Whatever constructs an `Agent` (`sora/bootstrap.py` — see README.md's Technology Stack & Requirements) must build memory/transport once and pass the same instances to both `Agent` and `DecisionCycle` — getting this wrong (constructing separate instances) would silently break the "shared" invariant

## Links

* Depends on [ADR-0008](0008-protocol-based-extensibility.md)
