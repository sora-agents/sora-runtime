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

### A property that moved is a stimulus in its own right

**A change to an observable property is an event whether or not the tool announced one.** A signal is the only *announced* stimulus, so a change an adapter refreshed into a property without pushing a signal for — because its ecosystem has no signal facility, or because the announcement was lost — cannot be waited on at all, even though the new value is sitting in working memory. `properties` answers "what is true now" and never "what just moved"; that question had no answer here.

The runtime therefore derives the change itself, diffing each re-observed property against the value it last saw, as AgentSpeak's belief revision derives belief-change events by comparing percepts against the belief base. This does not weaken the rule above: the derived delta carries **identities only, never values**, so it is not a copy of the property in an append log — a consumer that needs the value still reads the property, exactly as it would for a signal.

It is a *third store*, not a third kind on an existing one. `signals` carries what the environment announced; a derived delta is an inference this runtime drew. Mixing them would put runtime-invented events in front of every consumer that renders observed signals, and would make a strategy unable to tell a claim the world made from one the runtime made on its behalf.

The adapter's signal stays the authoritative fast path: a change already announced for the same `(source, path)` in the same Observe is not derived a second time. This gives the refresh-before-announce ordering above a second job — an adapter that announces before it refreshes leaves the two producers describing different states of the world, and the deduplication has nothing stable to match on.

The scope is deliberately narrow. A derived change opens a pending condition's gate ([ADR-0022](0022-plan-representation-context-guard-and-subgoals.md)); it does **not** resume a `blocked_on` completion wait ([ADR-0019](0019-blocked-state-machinery-and-percept-storage.md)). The asymmetry is a cost argument, not a purity one: a spuriously opened condition costs one judge call answered "no", while a spurious resume would let an activity proceed as though its operation had finished.

### Positive Consequences

* Supports the Suspension/Resumption pattern (wait for a signal or property change before proceeding), needed for physical or long-running operations
* Tools remain agent-agnostic and shareable across multiple agents, per the A&A model, which enables tool-mediated coordination
* Suspension/Resumption no longer depends on the adapter having a signal facility at all — an operations-and-properties-only ecosystem still supports waiting on a change

### Negative Consequences

* Adds conceptual overhead compared to a pure function-calling model
* Change detection has two producers that must be reconciled every cycle. Deduplication is per `(source, path)` within one Observe, so an adapter that announces asynchronously rather than alongside its refresh can still land both a signal and a derived delta for one change
* The derived log needs its own retention bound, larger than the signal log's, because the two fill at different rates — signals per environment event, derived deltas per observation cycle in which anything moved. It is sized to outlast the longest an inference may occupy the watching activity before it can judge; a property that moves on nearly every tick still overruns it, and the answer there is to exclude that property from derivation rather than to grow the bound further
* Every adapter must decide how to approximate properties/signals when its source protocol doesn't have them natively
* Where a protocol's change notification naturally carries the new value (an MCP `resource_updated` whose resource the adapter then re-reads), the adapter has to deliberately drop it from the signal rather than pass it through — the convenient thing to write is the thing the rule forbids

## Links

* Depends on [ADR-0003](0003-adapters-not-tool-authoring.md)
* The non-blocking guarantee here is what README.md's Tool Model and Use / Activities sections implement (implicit `running` state, automatic resolution on the activity)
