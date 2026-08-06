# Structured-value data-ops: composable transforms as plan steps

* Status: proposed
* Date: 2026-08-06

## Context and Problem Statement

[ADR-0022](0022-plan-representation-context-guard-and-subgoals.md) gave a plan its recursion
primitive: a **mechanical sub-goal** maps a fixed template over a run-time collection, fanning out
`len(collection)` invocations so the count comes from the data, not a model guess. That closed the
"save *each* apartment" under-count. But it can only *map* a collection it is handed — it cannot
**narrow or reshape** one first. The RentAFlat goalpost asks to "save each *qualifying* apartment";
the `where` selection [EXAMPLES.md](../../../EXAMPLES.md) showed inline on the mechanical sub-goal
was **documented but never wired** — `_expand_mechanical` fanned the collection out unfiltered. The
broader failing shape (Gaia2, 2026-08): "distinct zips → `get_crime_rate` each → keep the ones
scoring 5–10" — dedupe, map-invoke, gather the results, filter by a threshold, take.

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

* RentAFlat's "save each *qualifying* apartment" and the "distinct → crime-rate each → keep 5–10"
  pipeline become expressible end-to-end, deterministically where possible and with a single model
  call only where a predicate genuinely needs judgment.
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

## Links

* Refines [ADR-0022](0022-plan-representation-context-guard-and-subgoals.md) (Pass 2 item 1: the
  data-op layer the sub-goal work left open); reaffirms its option (a) rejection for data.
* Depends on [ADR-0017](0017-parameter-grounding-in-reason.md) (hard/soft reference split; Act stays
  mechanistic), [ADR-0021](0021-llm-calls-as-async-internal-actions.md) (off-cycle model call for
  the `$decide` filter), [ADR-0009](0009-five-phase-decision-cycle.md) (one external action per
  cycle — data-ops emit none), [ADR-0008](0008-protocol-based-extensibility.md) (the data-op bucket
  is an open Protocol seam).
