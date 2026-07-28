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

Sharing instances instead of a back-reference resolves question 1 on its own: `agent.semantic` and `agent.cycle.semantic` are the identical object by construction, so cross-cutting calls read naturally without reaching through `agent.cycle`, and there's no ownership cycle or two-phase construction to reason about. It doesn't by itself force any particular answer to question 2 — one could still pass a bundled `Agent` into every call. Narrow parameters were chosen on their own merits (testability with minimal fixtures, every action declaring exactly what it touches): `ExternalAction.execute(registry, cycle, **kwargs)` and `InternalAction.execute(cycle, **kwargs)` take only what each actually touches. (`DecisionCycle.tick()` originally took the registry as its single not-already-held parameter; the A2 refinement below folds the registry into the shared-instance set, so `tick()` now takes nothing — same principle, one fewer parameter.)

### Positive Consequences

* No circular references anywhere in the runtime
* Every action implementation declares exactly what it touches (`tools`, `cycle`) instead of an opaque `Agent`
* `DecisionCycle` remains fully valid the instant it's constructed — no unbound window

### Negative Consequences

* Every action's `execute()` gained an explicit parameter instead of a single bundled object (`DecisionCycle.tick()` briefly did too — see the A2 refinement, which returned it to zero-arg once the registry joined the shared-instance set)
* Whatever constructs an `Agent` (`sora/bootstrap.py` — see README.md's Technology Stack & Requirements) must build memory/transport once and pass the same instances to both `Agent` and `DecisionCycle` — getting this wrong (constructing separate instances) would silently break the "shared" invariant

## Registry access refined (A2)

`EnvironmentRegistry` is a single shared instance (built once in `sora/bootstrap.py`, per the model above) deliberately exposed at two different capabilities:

* `WorkingMemory.registry` advertises it through a **read-only `EnvironmentView`** Protocol (`get`/`get_workspace`/`all_tools`/`joined_workspaces`). Strategies receive working memory to *reason* over the live set of joined workspaces and tools — a legitimate part of the agent's current context — but `mypy --strict` forbids mutating connections through it.
* `DecisionCycle` holds the concrete, mutation-capable `EnvironmentRegistry` and passes it to `ExternalAction.execute(registry, cycle, ...)` at dispatch. `join`/`leave`/`restore` therefore live only in the action space, out of any strategy's reach.

This *extends* the narrow-dependencies choice rather than contradicting it: it splits one object's read and write capabilities across the boundary that matters — reasoning vs. acting — instead of handing every working-memory-holding strategy the full lifecycle API. "Currently-joined, live workspaces" is per-cycle contextual state and belongs in working memory; the durable `WorkspaceRecord`/`ToolRecord` knowledge stays in `SemanticMemory`. The two answer different questions (what am I connected to *now* vs. what have I ever discovered), so there is no duplication.

## Links

* Depends on [ADR-0008](0008-protocol-based-extensibility.md)
