# Tool usage interface: properties, signals, operations — async by design

* Status: proposed
* Date: 2026-07-05

## Context and Problem Statement

Following the Agents & Artifacts (A&A) meta-model, tools mediate agent-environment interaction (as well as agent-agent interaction). What shape should a tool's usage interface take, and should invoking a tool operation ever block the agent's decision cycle (e.g., if the operation is guaranteed to be short-lived)?

## Decision Drivers

* A&A's notion of a usage interface as the way agents interact with a domain object
* Need to support both stateless, function-call-like tools and stateful/observable ones (e.g., a physical device with meaningful ongoing state)
* The runtime's own reactiveness target — the decision cycle must not stall on a tool round-trip

## Considered Options

* Operations-only interface (function-calling style, matching most existing tool-calling ecosystems)
* Three-part interface: observable properties, signals, operations

## Decision Outcome

Chosen option: "Three-part interface", with operations invoked asynchronously — the decision cycle never blocks until an operation completes. Most existing ecosystems only provide the operations-only shape; adapters are responsible for approximating properties/signals where a richer model isn't natively available (see [ADR-0003](0003-adapters-not-tool-authoring.md)).

### Properties and signals must not duplicate each other

**A signal must never carry a snapshot of an observable property it accompanies.** Event-specific data is fine and expected — a `pressure_nominal` signal carrying the `psi` that tripped it is exactly what a signal payload is for. What is ruled out is a signal whose payload *is* the current value of a property the same tool already publishes.

The two halves of the interface are defined by opposite lifecycles (persistent re-observed state vs. transient event), and that distinction is what lets working memory store them differently: properties as a replace-by-`(source, name)` snapshot, signals as an append log ([ADR-0019](0019-blocked-state-machinery-and-percept-storage.md)). A signal that carries the property's value collapses the two — the append log then accumulates N copies of something whose whole point is that only the latest value matters, and every consumer that renders working memory pays for all N. Where those renderings are length-capped, the copies buy nothing: the signal rendering is capped exactly as the property rendering is, so each copy is at best the same truncated prefix the consumer already reads once off the property — paid for N times.

A consumer that needs the *value* reads the property. One that needs it at an instant when the property snapshot has not caught up — anything screening a signal at push time — reads it off the live tool, which is the artifact and therefore the authority; see [ADR-0020](0020-hard-interrupt-and-await-input.md). Correspondingly, an adapter that both refreshes a property and pushes a signal for the same change must refresh **first**, so the tool holds the new value before it announces it.

### Positive Consequences

* Supports the Suspension/Resumption pattern (wait for a signal or property change before proceeding), needed for physical or long-running operations
* Tools remain agent-agnostic and shareable across multiple agents, per the A&A model, which enables tool-mediated coordination

### Negative Consequences

* Adds conceptual overhead compared to a pure function-calling model
* Every adapter must decide how to approximate properties/signals when its source protocol doesn't have them natively
* Where a protocol's change notification naturally carries the new value (an MCP `resource_updated` whose resource the adapter then re-reads), the adapter has to deliberately drop it from the signal rather than pass it through — the convenient thing to write is the thing the rule forbids

## Links

* Depends on [ADR-0003](0003-adapters-not-tool-authoring.md)
* The non-blocking guarantee here is what README.md's Tool Model and Use / Activities sections implement (implicit `running` state, automatic resolution on the activity)
