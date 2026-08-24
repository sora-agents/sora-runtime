# Structured-value data-ops: composable transforms as plan steps

* Status: proposed
* Date: 2026-08-06

## Context and Problem Statement

[ADR-0022](0022-plan-representation-context-guard-and-subgoals.md) gave a plan its recursion
primitive: a **mechanical sub-goal** maps a fixed template over a run-time collection, fanning out
`len(collection)` invocations so the count comes from the data, not a model guess. That closed the
"act once *per element*" under-count. But it can only *map* a collection as handed — it cannot
**narrow or reshape** one first. Goals rarely hand a plan exactly the collection to act on; they ask
it to act on a *derived* one — the elements that *qualify* (filter), the *distinct* keys, the
*top few* by some order (sort + take), or the values *gathered* from a prior per-element fan-out and
then filtered by a computed threshold. The inline `where` selection
[EXAMPLES.md](https://github.com/sora-agents/sora-runtime/blob/main/EXAMPLES.md) once sketched on the mechanical sub-goal was **documented but
never wired** (`_expand_mechanical` fanned the collection out unfiltered), and even a wired `where`
would still not cover dedupe, sort, gather, or aggregate. The general failing shape is a short
pipeline over a run-time collection: dedupe → map-invoke → gather results → filter by a threshold →
take.

So the question: how should a synthesized plan express structured-value data processing — filter,
dedupe, sort, limit, gather, aggregate — over collections discovered at run time, **without**
importing a declarative dataflow DSL (which re-serializes the multi-activity concurrency ADR-0009
protects and hides the computation inside a parameter) and without a foreign control-flow layer?

## Decision Drivers

* **Imperative composition, not declarative binding.** ADR-0022 already rejected (its option (a))
  encoding iteration as a declarative `$foreach`/`$select` *binding spec*; the same reasoning
  extends to filtering/aggregation. A sequence of steps is a Jason plan body / a computation graph;
  a lazy declarative DAG buried in a param reference is not.
* **Minimal but extensible vocabulary.** Coverage (how many ops) is a tunable dial; paradigm safety
  (steps, not bindings) must not depend on the vocabulary staying small. Developers must be able to
  add their own richer transforms.
* **Keep Act mechanistic** ([ADR-0017](0017-parameter-grounding-in-reason.md)): deciding a value is
  a Reason act; a data-op must never reach Act.
* **Bound what a (possibly hallucinated) plan step can drive.** A plan step must not be able to
  invoke a runtime-only lever (`_suspend_`, `_infer_`, `_create_activity_`).
* **Reuse the existing seams:** the hard/soft (`$from`/`$decide`) reference split (ADR-0017), the
  off-cycle model-call machinery ([ADR-0021](0021-llm-calls-as-async-internal-actions.md)),
  Protocol-based extension ([ADR-0008](0008-protocol-based-extensibility.md)), and — no new memory
  module for the intermediate values (a CLAUDE.md habit).

## Considered Options

* (a) **Declarative binding spec** — extend param references with `$foreach`/`$select`/`$where`
  operators the resolver evaluates lazily.
* (b) **One fused select+map sub-goal** — keep everything on the mechanical sub-goal: an inline
  `where` predicate (the original, unwired EXAMPLES sketch), plus a baked-in reduce.
* (c) **Data-op internal actions composed as plan steps** — small, single-purpose transforms the
  plan sequences, each reading a collection and writing a named binding.

## Decision Outcome

Chosen option: **(c) data-op internal actions**, because it is the only option that composes
arbitrary pipelines (distinct → map-invoke → collect → filter → take → reduce) while keeping the
computation *visible as ordinary plan steps* and the concurrency model intact.

Concretely:

* **Six built-in ops**: `filter`, `distinct`, `sort`, `take`, `collect`, `reduce`. Each reads an
  `in` collection — a `$from` reference (history), a `$bind` reference (a prior binding), or a
  literal — and writes a named result into a new **`Activity.bindings[out]`**, which a later step
  reads via `{"$bind": "<name>", "path": …}`. `$bind` is thereby **generalized** from ADR-0022's
  eager loop-element substitution to also read this binding store at ground time; the two coexist
  because the loop element is substituted at fan-out, before grounding runs.
* **A dedicated data-op bucket on `ActionRegistry`** (`register_data_op`/`data_op`/`is_data_op`),
  holding the *existing* `InternalAction` Protocol. Reason dispatches a plan-step data-op only from
  this bucket — so a hallucinated step can never drive a runtime lever, and the collection-`filter`
  never collides with the perception-prune `FilterPerceptionsAction` (also named `filter`, in
  `_internal`). Developers register their own transforms through the same seam.
* **`filter`'s predicate is hard-or-soft**, mirroring ADR-0017: a mechanical
  `{"path", "op", "value"}` comparison (eq/ne/lt/le/gt/ge/between/in) resolved for free, or a
  `{"$decide": …}` predicate escalated to one off-cycle `ProceduralMemory.select` **over the whole
  collection** (`PendingInference.kind="select"` carrying the target `out`; the result lands in
  `bindings[out]` via Observe, exactly like `_ground_`). The batched call is a deliberate
  simplification of a per-element judgment; per-element is a foreseen refinement.
* **Execution mirrors sub-goal handling in `reason()`**: a mechanical op runs inline and the loop
  continues to the next step (like a mechanical fan-out's splice); a `$decide` filter parks the
  activity RUNNING and resolves a later cycle. `step_index` advances **past** the op in both cases
  (its "action" is producing the binding, not dispatching an operation), unlike `_ground_` whose
  step still has to dispatch.
* **`map-invoke` stays the sole iteration primitive** and is *sequential* — each fanned-out
  invocation parks the activity RUNNING on its ack before the next fires; S-ORA concurrency is
  across activities, not within a fan-out. Data-ops are **transforms, not control structures**, so
  there is no pure-projection `map` data-op: projection folds into a consuming op's key-path
  (`sort by`, `distinct by`, `filter where.path`, a template's `{"$bind": …, "path": …}`).
* **`collect` is map-invoke's MapReduce gather**: a fan-out leaves N results scattered across
  `Activity.history`; `collect` materializes them (by operation name) into one binding that any
  downstream op (filter/sort/take/reduce) can consume. This is why `collect` is kept separate from
  `reduce` — fold it in and only `reduce` could see a fan-out's output.

Named bindings are transient run state (a sibling of `history`/`grounded_params`, **not** a new
memory module) and are cleared on replan, since they are coupled to the plan that produced them.

### Positive Consequences

* The "act on each *qualifying* element" and "map a tool over distinct keys, gather, then keep
  those past a threshold" shapes become expressible end-to-end, deterministically where possible and
  with a single model call only where a predicate genuinely needs judgment.
* The pipeline is legible as ordinary plan steps; procedural reuse keeps a reusable skeleton (the
  references, not this run's values).
* Vocabulary is open: a domain that needs `group_by`/`join` adds them via `register_data_op` with no
  runtime change.

### Negative Consequences

* A new step grammar the planner must learn (mitigated by `PLAN_SYSTEM_PROMPT` guidance).
* The `$decide` filter is batched, so a very large collection is judged in one prompt; splitting to
  per-element (or chunked) judgment is deferred.
* Two same-named-but-distinct `filter`s (collection vs perception) coexist; the separate bucket
  keeps them from colliding but a reader must know the two domains.

## Pros and Cons of the Options

### (a) Declarative binding spec

* Good, because the pipeline is compact — one enriched reference expresses the whole thing.
* Bad, because it re-serializes multi-activity concurrency into one lazy evaluation and hides the
  computation inside a parameter — the exact failure ADR-0022(a) rejected, now for data too.
* Bad, because error/escalation attribution (which stage needed the model?) is opaque.

### (b) One fused select+map sub-goal

* Good, because it needs no new step kind — just wire the `where` already sketched.
* Bad, because it couples selection to fan-out: a `distinct`, a post-map `collect`, or a `reduce`
  has nowhere to live, so the "distinct → map → collect → filter → take" shape is inexpressible.
* Bad, because a single inline `where` never generalized past one predicate; the real need is a
  pipeline, not a richer sub-goal.

### (c) Data-op internal actions (chosen)

* Good, because arbitrary pipelines compose from single-purpose steps; each stage is visible,
  attributable, and independently mechanical-or-escalated.
* Good, because it reuses every existing seam (InternalAction, hard/soft references, off-cycle model
  calls, Protocol extension) and adds no control-flow layer or memory module.
* Bad, because it is more surface than (b) — six classes and a registry bucket — and a longer plan
  grammar for the planner to use well.

## Extension (2026-08-07): deterministic collection-shape tiers and a single diagnostic site

The first end-to-end runs surfaced a gap the core decision left implicit: **what counts as "the
collection"** when a `$from`/`$bind` reference resolves to something that isn't already a bare list.
ARE's collection operations (`list_all_apartments` / `search_apartments` / `list_saved_apartments`)
return an `{id -> record}` mapping, and some tools wrap that payload under a lone key
(`{"apartments": {…}}`) or an envelope (`{"results": […]}`). The tool's own schema often synthesizes
to an opaque `{"type": "object"}`, so plan-time grounding guidance is blind to the shape — the
runtime must coerce **mechanically**, and a wrong guess silently fans a sub-goal out over a record's
*fields* (garbage) instead of its elements.

`_as_collection` therefore coerces in **deterministic tiers**, and it is the *only* place that
decides collection-hood (both the mechanical sub-goal and every data-op resolve through it):

1. a **list** is itself;
2. a **single-key envelope** (`{"apartments": {id -> rec}}`, `{"results": […]}`) is unwrapped and
   recursed into, *iff* the lone value is itself a collection — a single-element `{id -> record}`
   map whose record has *any* scalar field makes the recursion refuse, so it correctly falls
   through to the **mapping** tier. The residual ambiguity is any single-element map
   `{"a1": {record}}` whose lone record's fields are *all* mapping-valued — a one-field record (`{"a1": {"photos": […]}}`) **or** a
   many-field one (`{"a1": {"loc": {…}, "meta": {…}}}`): both recurse to a collection, so the record
   is unwrapped into its field-*values* instead of kept as one record. This is **undecidable** at
   this layer — `{K: {k1: {…}, k2: {…}}}` is structurally identical whether `K` is an id or a wrapper
   name — so no mechanical tie-break resolves it; any rule only shifts *which* shape misfires. The
   unwrap is chosen because ARE's records always carry scalar fields (so they never reach it) and its
   real envelopes are plural `{id -> record}` maps that must unwrap; the principled resolution for a
   tool returning all-mapping-field records is the deferred model-escalated extraction, not a further
   shape heuristic;
3. a **paginated envelope** — exactly one list-valued key, every sibling a *scalar drawn from a
   closed pagination vocabulary* (`total`, `range`, `offset`, `count`, `limit`, `has_more`, …) —
   yields that list. Added 2026-08-21 (see below); unlike the **single-key envelope** tier it
   deliberately is **not** purely structural;
4. an **`{id -> record}` mapping** (every value a mapping) iterates its *values* — lossless, since
   each record carries its own id;
5. anything else (a single record's fields, an `{id -> scalar}` map, a scalar) is **refused**
   (`None`) rather than guessed at.

An empty dict is an empty collection (`[]`), not a failure. Richer recovery for the shapes the
final tier still refuses — model-escalated *extraction* of a collection from an unrecognized
envelope — is deferred (the same escalation seam the `$decide` filter already uses).

**Why the paginated-envelope tier gives up on being structural (2026-08-21).** ARE's windowed list
operations return `{"events": […], "range": "(0, 1)", "total": 1}`, which the tiers above refused:
it is not a lone-key envelope and its values are not all mappings. Refusing was defensible in isolation but not in
context — those operations *declare* a bare return type, and the planning prompt tells the planner to
match the declared shape, so the plan wrote the empty path `""` exactly as instructed and the runtime
answered "add a `path`". A run lost a 220-second replanning round-trip to being punished for
believing the catalog. The obvious structural rule — one list-valued key, scalar siblings — is
**wrong**, and this is the point worth recording: an ARE calendar *event* is
`{"event_id": …, "title": …, "attendees": […]}`, the identical shape, and reading it as a collection
would fan a sub-goal out over one event's attendees instead of over events. So that tier additionally
requires every sibling key to come from a **closed vocabulary of pagination-metadata names**, which a
record's own fields are not drawn from. That is a vocabulary heuristic, not a structural proof, and
it is stated as such: it buys exactly the shape windowed list operations return, and every name added
to the vocabulary widens what gets read as a collection. The alternative — fixing the mis-declared
return types at the adapter — is worth doing too, but it cannot be relied on, since the runtime does
not control what an imported tool's schema claims about itself.

Observability is consolidated into a **single diagnostic site**, `_resolve_collection`: it is the
one place that logs *why* a resolution came up empty, distinguishing an **unresolved reference** (the
source op never ran / a bad path — usually a plan bug: nothing narrowed the collection first) from a
**resolved value of a shape the tiers refuse** (naming the offending type). A genuinely empty
collection stays silent (benign). Callers (`_expand_mechanical`, the data-op dispatch) no longer
re-warn — they just treat `None`/`[]` alike as "fan out over nothing", per the never-raise contract.
This keeps the diagnostic where the *reason* is known, instead of a generic "expanded to nothing" at
each call site.

**Cross-collection membership (reference-valued `in`/`not_in`).** A predicate often needs to test an
element against *another* run-time collection — "keep apartments **not** already saved", "keep the
zips we don't yet have a crime rate for". A `filter` predicate's `value` may therefore be a reference
(`$from`/`$bind`) to a second collection, with a new `not_in` operator alongside `in`. The reference
is resolved **once, in Reason** (`_resolve_predicate_value`, beside the `in`-collection
resolution in `_data_op`), projected by an optional `value_path` to a concrete list of keys, and
substituted into the predicate before the op runs — so `_matches` stays a pure literal comparison and
the batched/off-cycle machinery is untouched (membership is **mechanical, never a model call**). An
unresolvable membership reference degrades to an empty set: `in` matches nothing, `not_in` keeps
everything (fails open, so a missing exclusion list never silently drops the whole collection). This
lets a plan split a formerly-bundled `$decide` ("in 5..10 **and** not already saved") into its
mechanical half (`not_in` the saved list) and only the genuinely judgemental remainder.

**A reference-valued predicate is not only a membership set (2026-08-24).** The paragraph above was
written for `in`/`not_in`, and so was the code: resolution was gated on those two ops. But the need
it describes — an operand known only at run time — is not specific to membership. The canonical
`reduce` → "keep what beats the mean" pipeline this ADR proposes elsewhere is exactly that shape,
and it never worked: the raw reference dict reached `_matches`, every ordered comparison against it
raised `TypeError`, that was caught as a non-match, and the filter silently kept *nothing*. So
`value` resolves for every op, in two shapes, because the ops read it differently — membership
resolves as a *collection* and projects through `value_path`, while every other op resolves to the
operand *itself* (a scalar; for `between`, the `[lo, hi]` pair) with no projection, since there is
no collection to key on.

Two consequences follow from the failure mode rather than from symmetry. An unreadable operand is a
**defect that replans**, not a degrade-to-empty: the fail-open argument made above for membership
says a missing exclusion list must not silently drop the collection, and an unreadable threshold
drops it just as silently in the other direction. And a `$decide` in the *operand* position is
rejected with a message naming what to write instead — it is the "hide the computation inside a
parameter" shape this ADR rejected as option (a), it was never documented in any prompt, and it
previously resolved to an empty set or a raw dict, failing open either way.

**`collect` carries the fan-out key.** A map-invoke leaves N results scattered in history;
`collect` gathers them, and — because a tool's result often doesn't echo the input it was called
for (`get_crime_rate` returns a rate, not the zip) — each gathered item is enriched with its
invocation's params (the return wins on key collision; a non-dict result is wrapped as
`{**params, "result": …}`). This preserves the result↔input correlation a downstream
`filter`/membership needs, so the canonical `distinct zips → get_crime_rate each → keep 5–10`
pipeline is fully mechanical: `collect` → `filter between 5 10` on the rate → membership `in` join
on `zip_code`, with no `$decide`. Without it, correlating each rate back to its zip would force the
judgement escalation — which, being blind to other ops' history, cannot see the rates anyway.

## Links

* Refines [ADR-0022](0022-plan-representation-context-guard-and-subgoals.md) (Pass 2 item 1: the
  data-op layer the sub-goal work left open); reaffirms its option (a) rejection for data.
* Depends on [ADR-0017](0017-parameter-grounding-in-reason.md) (hard/soft reference split; Act stays
  mechanistic), [ADR-0021](0021-llm-calls-as-async-internal-actions.md) (off-cycle model call for
  the `$decide` filter), [ADR-0009](0009-five-phase-decision-cycle.md) (one external action per
  cycle — data-ops emit none), [ADR-0008](0008-protocol-based-extensibility.md) (the data-op bucket
  is an open Protocol seam).
