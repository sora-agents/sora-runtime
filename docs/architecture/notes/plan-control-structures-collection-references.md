# Plan control structures: data-bound collection references (map/filter/reduce/branch), expanded at grounding time

* Status: abandoned — kept as a historical record of a superseded approach
* Date: 2026-08-04

> **Abandoned (2026-08-05).** This was drafted as ADR-0022 before the AgentSpeak analysis in
> [`jason-plan-representation.md`](jason-plan-representation.md). Its functional-dataflow framing
> — first-class `$foreach`/`$if`/`$select`/`$reduce` collection references — was judged a
> conceptual misstep: it imports a computation-graph paradigm onto a practical-reasoning plan.
> It was demoted from an ADR to this note, and its ADR number was reissued to the replacement,
> [ADR-0022: Plan representation — retrieval-binding context guard + sub-goals](../adrs/0022-plan-representation-context-guard-and-subgoals.md),
> which handles iteration via sub-goals instead. Retained only so the reasoning trail is legible.

## Context and Problem Statement

A `Plan` is a flat `list[Step]`, inferred once (`_infer_`) before any tool has returned data, and
walked linearly — **one `Step` → one external action → one invocation**. [ADR-0017](../adrs/0017-parameter-grounding-in-reason.md)
added *scalar* run-time values to that skeleton: `{"$from": ...}` resolves a scalar from a prior
result mechanically, `{"$decide": ...}` escalates a scalar judgment to one model call. Neither
expresses an operation over a **collection**: "save *each* qualifying property", "remove *each*
already-saved one", "email *each* relative".

The first real Gaia2 run made the gap concrete (RentAFlat scenario, 2026-08-03): the oracle expected
`save_apartment` ×5, `remove_saved_apartment` ×2, `send_email` ×2; S-ORA did each **once** and failed
the judge's tool-call-count check. The planner *understood* the task — it wrote `$decide` descriptions
saying "repeat for each unique zip code" — but the runtime has no way to act on that prose: `_ground_`
resolves it to a single value and `step_index` advances. **The iteration count was frozen at
authoring time, when the count was unknowable.**

How should a plan express operations over collections — reliably (count bound to data, not a model
guess), while preserving one external action per cycle ([ADR-0009](../adrs/0009-five-phase-decision-cycle.md))
and the reusable-skeleton property of plans?

## Decision Drivers

* **Reliability = bind the iteration count to data, not prose.** A prose "repeat for each" is a model
  guess dressed as structure; the count must come from `len(collection)`, computed by code.
* **One external action per cycle must hold** ([ADR-0009](../adrs/0009-five-phase-decision-cycle.md)) — a
  step may not dispatch N invocations at once (that re-serializes multi-activity concurrency and
  breaks the cycle invariant).
* **Extend the existing reference family, don't add a foreign control-flow layer.** `$from`/`$decide`
  ([ADR-0017](../adrs/0017-parameter-grounding-in-reason.md)) already split *mechanical dataflow* from *model
  judgment*; collection handling should slot into that split, not sit beside it as an interpreter.
* **Plans stay reusable skeletons.** The stored plan keeps its references; expansion is per-run
  (grounds a copy), exactly as scalar grounding does today.
* **Act stays mechanistic** ([ADR-0017](../adrs/0017-parameter-grounding-in-reason.md)); deciding *what* to
  iterate over is a Reason act, so expansion lives on Reason's grounding path, not in Act.
* **A bounded language, not a Turing tarpit.** Admit a primitive only if it **(a)** binds a decision
  to data more reliably than prose *and* **(b)** isn't already owned by the decision cycle.

## Considered Options

* **(a) Prose keyword in `$decide`** ("repeat for each…") — the status-quo failure mode. The count is
  a model guess *and* the runtime can't even act on it (one `$decide` resolves one value).
* **(b) Dynamic re-planning.** On plan exhaustion, re-fire `_infer_` with the collected data in
  context; the model, now *seeing* the list, emits N concrete steps. No new step type.
* **(c) Data-bound collection references + grounding-time expansion.** Add a small, bounded family
  (`$foreach` map, `$if` branch, `$select`/`$reduce` fold) as the collection-valued counterpart to
  `$from`/`$decide`; a foreach/branch/fold **step** is expanded at grounding time into concrete steps
  spliced into the per-run plan, then dispatched one-per-cycle as usual.
* **(d) Batch/parallel dispatch** — one step emits N invocations at once. Rejected: violates one
  action per cycle.

## Decision Outcome

Chosen: **(c) as the fan-out backbone, with (b) retained as its complement.** They share the same
engine — both end in "concrete steps materialized into the plan, one action per cycle" — and differ
only in the **expansion source**, which *is* the reliability axis: (c) expands **mechanically** (count
= `len(collection)`), (b) expands via a **model call** (count = the model's emission). So (c) is the
default for the common Gaia2 shape ("do X to each qualifying Y"); (b) stays the right tool for a
genuinely *branching* continuation that isn't a uniform map (deferred, not built here).

Concretely, the first realized member is **`$foreach`**; `$if` and `$select`/`$reduce` are admitted
under the same principle but deferred (foreseen members, like ADR-0019's property-reaches-state
completion).

**`$foreach` shape.** It is a *step-level* reference (it multiplies the whole invocation), so — unlike
the *param-level* `$from`/`$decide` — it is a `Step` with a dedicated `next_action` sentinel
(`"foreach"`), the same way `WAIT` is a pseudo-action the cycle special-cases. Its `params` carry a
collection reference, an element binding, an optional predicate, and a step template:

```
Step(next_action="foreach", params={
    "in":   {"$from": "list_all_apartments", "path": "apartments"},  # the collection (mechanical)
    "as":   "apartment",                                             # element binding name
    "where": {"$decide": "violent_crime 5-10 and not already saved"}, # optional per-item predicate
    "do":   Step(next_action="invoke", params={..., "apartment_id": {"$from": "apartment.id"}}),
})
```

Inside `do`, params reference the bound element and prior results through the **unchanged** `$from`/
`$decide` machinery — the family composes with itself.

**Expansion (grounding-time, in `DefaultReasonStrategy`).** When Reason reaches a `foreach` step:

1. Resolve `in` mechanically against `Activity.history` → a list (a normal `$from` resolution;
   unresolved → escalate/return, exactly as today).
2. Apply `where` to filter: a mechanical comparison resolves for free; a `$decide` predicate escalates
   **per element** (the only model spend, and only for irreducible selection).
3. **Splice** `len(filtered)` concrete copies of `do` into the *per-run* step list at the current
   index, each with its element bound; the **stored** plan keeps the single `foreach` step
   (skeleton preserved for procedural reuse, mirroring ADR-0017's "ground a copy").
4. Advance normally — one invocation per cycle. **Count and dispatch are mechanical; the model touches
   only per-item selection.**

**Fan-out materializes as in-activity steps, not sub-activities (for now).** Sub-activity fan-out
(concurrent activities, one action each per cycle) is more S-ORA-native but needs a **join** before any
reduce ("two cheapest among the *newly saved*" must wait for all saves). In-activity step-splicing is
naturally sequential, so a later `$select`/`$reduce` step just reads the accumulated history. The
sub-activity variant + join is a foreseen alternative, deferred.

**The boundary that keeps this bounded.** Several "control structures" are deliberately **not** plan
primitives, because the decision cycle already owns them (driver (b)): **waiting/"until"** is the
`blocked_on`/`completion_signal` machinery ([ADR-0019](../adrs/0019-blocked-state-machinery-and-percept-storage.md));
**parallelism** is concurrent activities ([ADR-0009](../adrs/0009-five-phase-decision-cycle.md),
[ADR-0016](../adrs/0016-pluggable-activity-selection.md)); **reconsideration on change** is the interrupt seam
([ADR-0020](../adrs/0020-hard-interrupt-and-await-input.md)). The admissible family is therefore narrow:
map (`$foreach`), branch (`$if`), fold (`$select`/`$reduce`) — data-shaping, not process control.

### Positive Consequences

* Multi-item tasks become expressible; the iteration count is `len(data)`, not a model guess — the
  direct fix for the observed failure.
* One action per cycle is preserved: expansion produces steps, dispatch stays one-per-cycle.
* Reuses ADR-0017's split and machinery — `$foreach` is "a `$from` that iterates"; the model's only
  residual job is *selection* (`where`), which is irreducible judgment evaluated **per item against
  real data**, far more reliable than one upfront prose guess.
* Plans stay reusable skeletons; expansion is a transient per-run materialization.
* The boundary principle bounds the family and prevents the plan language from absorbing waiting,
  parallelism, and reconsideration, which the cycle already handles.

### Negative Consequences

* More grounding machinery: a list-valued resolver, per-element predicate evaluation/escalation, and
  step-splicing that mutates the per-run plan (the stored skeleton must be protected).
* A new prompt contract: the planner must author the collection reference (which list, which `path`,
  which `where`). A mis-authored ref degrades to an empty/wrong list rather than a hard failure, but
  it is still fragility — the same class as ADR-0017's result-shape guess.
* `$select`/`$reduce` partially overlaps `$decide`; the two need a clear boundary (structured fold when
  the ranking is expressible as sort-keys/take-N/tie-break; `$decide` only for genuinely fuzzy picks).
* Per-element `$decide` predicates can multiply model calls (one per item); acceptable because it is
  spent only on real selection, but it makes a large collection potentially expensive.

## Pros and Cons of the Options

### (a) Prose keyword in `$decide`

* Good, because it needs no new machinery.
* Bad, because the count is a model guess *and* unactionable — this is exactly the failure observed.

### (b) Dynamic re-planning

* Good, because it needs no new step type and handles heterogeneous/branching continuations, not just
  uniform maps.
* Bad, because the model still emits the count (data-*informed* guessing, not data-*bound*), and it
  adds a re-entrant planning loop foreign to the `$from`/`$decide` family. Kept as a complement, not
  the backbone.

### (c) Data-bound collection references (chosen)

* Good, because the count and dispatch are mechanical (`len(data)`), it extends the existing reference
  family, and it preserves one-action-per-cycle and the reusable skeleton.
* Bad, because it is the most machinery, and it adds a planner prompt contract for authoring
  collection references.

### (d) Batch/parallel dispatch

* Good, because it is the most direct "do N things".
* Bad, because it breaks the one-external-action-per-cycle invariant and re-serializes the concurrency
  the cycle is built to express.

## Links

* Refines [ADR-0017](../adrs/0017-parameter-grounding-in-reason.md) — extends the `$from`/`$decide` reference
  family with collection-valued members, expanded on the same Reason grounding path.
* Honors [ADR-0009](../adrs/0009-five-phase-decision-cycle.md) (one external action per cycle) — expansion
  produces steps, never a batch dispatch — and defers a sub-activity fan-out variant that would build
  on [ADR-0016](../adrs/0016-pluggable-activity-selection.md) plus a join.
* Consistent with the "no redundant mechanism" ethos of [ADR-0011](../adrs/0011-phase-fusion-via-threaded-result.md):
  waiting, parallelism, and reconsideration stay owned by the cycle
  ([ADR-0019](../adrs/0019-blocked-state-machinery-and-percept-storage.md),
  [ADR-0020](../adrs/0020-hard-interrupt-and-await-input.md)), not duplicated as plan keywords.
* Deferred members: `$if` (branch), `$select`/`$reduce` (fold), sub-activity fan-out + join, and the
  re-planning complement (option (b)).
