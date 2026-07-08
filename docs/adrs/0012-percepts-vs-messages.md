# Percepts and messages kept as two distinct channels

* Status: accepted
* Date: 2026-07-05

## Context and Problem Statement

The initial working-memory design treated inbound agent-to-agent messages as just another `Percept` kind, merged into the same `perceptions` list as tool-originated stimuli. Some in the EMAS (engineering multi-agent systems) community take that view; the more common convention — and the one Jason (already cited as inspiration) follows — keeps environment perception and inter-agent communication on distinct interfaces.

## Decision Drivers

* Align with the more common EMAS convention and the runtime's own Jason lineage
* Messages are orthogonal to activity selection — they arrive independent of which activity is being situated, and may launch or advance an activity rather than being filtered to one, unlike perceptual input

## Considered Options

* Keep messages as a `Percept` kind alongside properties/signals/results (the initial design)
* Split messages into their own `WorkingMemory.messages` channel and a `Message` type, narrowing `Percept.kind` to environment stimuli only

## Decision Outcome

Chosen option: "Split into a separate channel", because it matches the more common EMAS/Jason convention and keeps Situate's "filtering the perceptual input" correctly scoped to perceptions only — messages were never meant to be filtered to one activity at Observe time, since receiving a message is orthogonal to which activity happens to be selected.

### Positive Consequences

* `WorkingMemory.perceptions` stays scoped to genuine environment stimuli (properties, signals, operation results)
* Matches established multi-agent-systems practice, avoiding an idiosyncratic conflation the runtime would otherwise have to justify indefinitely

### Negative Consequences

* `MessageTransport.receive()` and `WorkingMemory` each carry a second, parallel structure instead of one unified stream
* A `ReasonStrategy` must read two separate lists (`wm.perceptions`, `wm.messages`) instead of one

## Links

* Depends on [ADR-0009](0009-five-phase-decision-cycle.md) (both channels are populated during Observe)
