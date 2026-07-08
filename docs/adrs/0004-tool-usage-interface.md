# Tool usage interface: properties, signals, operations — async by design

* Status: accepted
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

### Positive Consequences

* Supports the Suspension/Resumption pattern (wait for a signal or property change before proceeding), needed for physical or long-running operations
* Tools remain agent-agnostic and shareable across multiple agents, per the A&A model, which enables tool-mediated coordination

### Negative Consequences

* Adds conceptual overhead compared to a pure function-calling model
* Every adapter must decide how to approximate properties/signals when its source protocol doesn't have them natively

## Links

* Depends on [ADR-0003](0003-adapters-not-tool-authoring.md)
* The non-blocking guarantee here is what README.md's Tool Model and Use / Activities sections implement (implicit `running` state, automatic resolution on the activity)
