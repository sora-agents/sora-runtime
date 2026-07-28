# Every extension point is an open Protocol, never a closed enum or base class

* Status: proposed
* Date: 2026-07-05

## Context and Problem Statement

The runtime needs a consistent pattern for pluggability across tools, adapters, memory backends, transport, phase strategies, and the action space. Should extension points require inheriting from runtime-provided base classes, or should the runtime rely on structural typing?

## Decision Drivers

* The runtime's own "flexible: highly customizable" goal
* Need adopters to plug in objects they don't own or can't modify (e.g., an existing SDK's client object) without inheriting from a runtime base class
* Avoid coupling adopter code to runtime internals

## Considered Options

* Abstract base classes (ABCs) requiring inheritance
* `typing.Protocol` structural typing
* A closed enum / fixed list for some extension points (e.g., the action space)

## Decision Outcome

Chosen option: "`typing.Protocol` structural typing", applied to every extension point without exception — including the action space itself, which is an open `ActionRegistry` rather than a closed enum, so third parties can register new actions alongside the predefined set without runtime changes.

### Positive Consequences

* Adopters can hand in objects they don't own (e.g., wrapping an existing SDK client) as long as the shape matches — no forced inheritance hierarchy
* Third parties can extend the action space without modifying the runtime's core

### Negative Consequences

* Relies on structural (duck) typing discipline — a static type checker enforces the shape, but there is no runtime enforcement unless a Protocol is explicitly marked `@runtime_checkable`

## Links

* Depends on [ADR-0001](0001-python-asyncio-runtime.md)
* Applies throughout — referenced by nearly every other ADR in this set
