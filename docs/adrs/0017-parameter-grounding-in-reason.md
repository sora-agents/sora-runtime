# Parameter grounding is a Reason decision (references + escalation); Act stays mechanistic

* Status: proposed
* Date: 2026-07-19

## Context and Problem Statement

A plan is produced once and reused, but some of a step's parameters can only be known at *run time*:
`reply_to_email` needs the `email_id` that an earlier `list_emails`/`search_emails` returns. The
model can't know that value when it plans, so it either invents a literal (which fails against the
real tool) or leaves a placeholder. Two questions follow: **where** does the runtime decide such a
value, and **how**?

The initial approach parked this at `ActStrategy.bind` — "grounding an abstract Step into a concrete,
schema-conformant OperationInvocation (the tool-hallucination-prone step)." But *choosing* which
value to use ("reply to Alice" → email id 42) is a **decision**, not a mechanical schema mapping. And
until now the runtime didn't even *retain* prior results (`Activity.last_operation` is overwritten
each step), so there was nothing to ground against.

## Decision Drivers

* Deciding a parameter value is reasoning; Act should stay mechanistic (schema split / dispatch).
* A plan should stay a reusable **skeleton** — the same plan reused across runs, with values filled
  fresh each run from the current context (a *belief state*).
* Prefer *mechanistic* resolution where a value is unambiguous; only spend a model call when a
  genuine decision is required ("mechanistic where it can be, Reason where it must be").
* Reuse the existing model seam (`LLMClient` via `ProceduralMemory`), not a new injection mechanism.
* [ADR-0010](0010-pluggable-phase-strategies.md): phases are pluggable; the model path is Reason's.

## Considered Options

* **(a) Model-backed binding in Act** (the first attempt): a model call in `ActStrategy.bind` grounds
  params. Rejected — it puts a *decision* in Act, which should be mechanistic.
* **(b) Model-backed grounding in Reason**: every under-specified step triggers a model call in
  Reason to decide its params. Robust, but a model decision per such step even when a rule would do.
* **(c) References + deterministic resolution**: the planner emits structured references to prior
  results; a resolver fills them mechanically. Cleanest phase split, but can't resolve a genuinely
  ambiguous pick or a mis-guessed result shape.
* **(d) Hybrid**: (c) by default, escalating to (b) only when a reference can't resolve.

## Decision Outcome

Chosen option: **(d), grounding owned by Reason**. `DefaultActStrategy` is unchanged and stays the
mechanistic schema split. Grounding moves into `DefaultReasonStrategy`, on the advance path:

* The runtime retains an append-only execution trace: `CompletedOperation(invocation, ack)` entries
  on `Activity.history`, appended by `DefaultObserveStrategy` as each operation resolves (transient;
  not persisted).
* A param whose value depends on a prior result is a **reference**, emitted by the planner:
  * hard — `{"$from": "<operation_name>", "path": "<dotted path>"}` — resolved deterministically
    against the most recent matching history entry (`resolve_references`), or
  * soft — `{"$decide": "<description>"}` — always escalates.
* On advance, Reason grounds a **copy** of the step for this cycle (the stored plan keeps its
  references, so procedural reuse stays a skeleton): resolve references mechanically; if any remain
  unresolved (soft ref, missing step, bad path), **escalate** to one model call —
  `ProceduralMemory.ground(...)` — which decides the concrete params from the operation schema + the
  partially-resolved params + the rendered history. A step with no references is a pure no-op (the
  cheap advance path makes no model call), so typically ≤1 model call/cycle still holds.

The `ground` model call is packaged in `ProceduralMemory` (reusing its `LLMClient` and a pluggable
`GroundPrompt`, mirroring `infer`) only because procedural memory currently owns the model handle;
grounding is really an Act-adjacent reasoning act, and its eventual home is a client injected per
strategy (deferred — see Links).

### Positive Consequences

* Act stays purely mechanistic; the value *decision* lives in the reasoning phase, where it belongs.
* Plans are genuinely reusable skeletons — references make a stored plan replay against each run's
  own results.
* Cost-honest: unambiguous data flow resolves with no model call; the model is spent only on a real
  decision.
* No new model-call injection mechanism — reuses the `infer` machinery and the `LLMClient` seam.

### Negative Consequences

* The planner must emit references (a prompt contract) and, for a hard reference, guess the result
  *shape* (field path) — a mis-guess degrades to an escalation rather than a failure, but it's still
  a fragility.
* A second packaged model query (`ground`) lands in `ProceduralMemory`, which is a slight semantic
  stretch (procedural memory = skills/plans); accepted as a bridge until per-strategy client
  injection lands.
* `Activity` grows a transient `history` list (unbounded within a long activity — acceptable for
  v0.1.0's short plans; the fuller execution-trace design is deferred).

## Pros and Cons of the Options

### (a) Model-backed binding in Act

* Good, because Act's `bind` is already the documented grounding point.
* Bad, because it makes Act *decide* values — conflating the mechanical schema mapping with a
  reasoning decision, and it can't fall back cheaply for values a rule could resolve.

### (b) Model-backed grounding in Reason (always)

* Good, because it's robust to ambiguity and unknown result shapes.
* Bad, because it spends a model decision on every under-specified step, even ones a deterministic
  reference would resolve for free.

### (c) References + deterministic resolution (only)

* Good, because it's the cleanest phase split and cheapest (no grounding model call).
* Bad, because it can't resolve a genuinely ambiguous pick or recover from a mis-guessed result
  shape — some flows would dead-end.

### (d) Hybrid (chosen)

* Good, because it's mechanistic and free in the common case, and still correct when a decision is
  unavoidable — with Act kept mechanistic throughout.
* Bad, because it's the most machinery (references, a resolver, and an escalation path).

## Links

* Refines [ADR-0010](0010-pluggable-phase-strategies.md) (the model path is Reason's) and revises the
  README, which had placed grounding at `ActStrategy.bind`.
* Deferred: learning durable facts into `SemanticMemory` for cross-run reuse (a reference/belief
  could then resolve from semantic memory without a re-query), and per-strategy `LLMClient` injection
  (the eventual home for the `ground` call). Both are in the Phase-4 backlog.
