# Plan representation: retrieval-binding context guard + sub-goals (AgentSpeak-inspired)

* Status: proposed
* Date: 2026-08-05

## Context and Problem Statement

A `Plan` is a flat `list[Step]`, synthesized once by `_infer_` and walked linearly — one `Step`
→ at most one external action per cycle ([ADR-0009](0009-five-phase-decision-cycle.md)).
[ADR-0017](0017-parameter-grounding-in-reason.md) added run-time param binding to that skeleton:
`{"$from": ...}` resolves a scalar from a prior result mechanically, `{"$decide": ...}` escalates
a scalar judgment to one model call. Two gaps remain, both surfaced by real Gaia2 runs
(RentAFlat, 2026-08-03):

1. **A plan can only bind params from its own execution history.** `$from` resolves against
   `Activity.history`; there is no way to bind a param from long-term memory (the user's home
   address, a preferred vendor, a saved policy). The planner either invents a literal or
   mis-assumes the value is a prior output.
2. **A plan cannot express an operation over a collection discovered at run time.** "Save *each*
   qualifying apartment", "email *each* relative" — the iteration count is unknowable at
   synthesis time, so a linear plan collapses each "for each" to a single call and under-counts
   against the oracle.

The framing that resolves both: a `Plan` is **practical-reasoning-shaped**, not
data-computation-shaped. S-ORA **synthesizes** a plausible sequence of operators (action templates)
with an LLM — it does not search a state space, and params are not fully bound at synthesis time —
so it is a *non-planning* procedural system in the PRS/AgentSpeak lineage (the Procedural
Reasoning System was proposed precisely to *avoid* automated planning), differing only in that
plans are **synthesized** rather than developer-**authored**. That points at borrowing AgentSpeak's
plan *schema* — trigger + context guard + a body of sub-goals and actions — while keeping S-ORA's
own *substrate* (LLM-synthesized not authored; structured-value binding not logic unification;
`Activity` as the sole first-class unit per
[ADR-0002](0002-activity-as-sole-first-class-construct.md)). The BDI correspondence is deliberate
but bounded — the environment/artifact model and interleaved-intentions-as-activities are already
borrowed from JaCaMo/CArtAgO ([ADR-0004](0004-tool-usage-interface.md),
[ADR-0021](0021-llm-calls-as-async-internal-actions.md)) without importing a
belief/desire/intention triad. How should the plan representation realize that schema so that both
gaps close without importing that triad or a foreign control-flow layer?

## Decision Drivers

* **Bind from the whole knowledge state, not just recent history.** A param should resolve from
  long-term memory as readily as from a prior action's output — AgentSpeak's context binds from
  the whole belief base, not only fresh perception.
* **Retrieval, not unification.** Widening the binding reach must not drag in logic-based KR;
  keep the no-logic-KR boundary — structured-value binding, never logic-variable unification.
* **One external action per cycle must hold** ([ADR-0009](0009-five-phase-decision-cycle.md)) —
  no step may dispatch N invocations at once.
* **Reuse S-ORA's own selection mechanism.** Plan/sub-plan selection is *synthesis* (`_infer_`)
  and cached-plan `retrieve()`, not a reactive belief-base library lookup — so a sub-goal must
  not require machinery [ADR-0002](0002-activity-as-sole-first-class-construct.md) declined.
* **Plans stay reusable skeletons** ([ADR-0017](0017-parameter-grounding-in-reason.md)): the
  stored plan keeps its references and abstract sub-goals; per-run resolution grounds a copy.
* **Mechanical where it can be, model where it must be.** The same split `$from`/`$decide` draws
  at the param level should govern iteration: bind the count to `len(data)` when the shape is a
  uniform map; spend a model call only on genuinely open continuation.

## Considered Options

* **(a) A functional-dataflow family** — first-class `$foreach`/`$if`/`$select`/`$reduce`
  collection references, expanded at grounding time. Closes gap 2 but imports a
  computation-graph paradigm (LangGraph-shaped) alien to a practical-reasoning plan, and leaves
  gap 1 (memory binding) untouched.
* **(b) Widen `$from` into a multi-source resolver** — let a single reference reach into
  `{history, long-term memory, goal, literals}`. Closes gap 1 but muddies the one mechanism that
  was clean precisely because it was structural provenance over history.
* **(c) AgentSpeak schema, S-ORA substrate (chosen).** Give the plan a **retrieval-binding
  context guard** (closes gap 1) and a **sub-goal** body construct that re-fires `infer()`
  mid-plan (closes gap 2), keeping `$from` history-only and reusing existing synthesis/grounding
  machinery.

## Decision Outcome

Chosen option: **(c)**, because it closes both gaps with the plan's own reasoning shape rather
than a foreign paradigm, splits the two binding needs by origin and timing (each mechanism keeps
one job), and reuses `retrieve()`/`infer()`/`pending_inference` wholesale instead of adding a
bespoke expansion interpreter.

A plan is **trigger + context guard + body**, attached to the `Activity`
([ADR-0002](0002-activity-as-sole-first-class-construct.md)), never an intention type.

**Context guard — named retrieval, applicability, and `$decide`.** The guard is the
`Plan.context_guard` field: a list of clause dicts (reusing the `$from`/`$decide` dict-reference
convention — no new dataclass; the field is named `context_guard` rather than `context` to avoid
colliding with `Activity.context`). Each clause is one of:

* `{"bind": "<name>", "query": {...}}` — a mechanical retrieval from an existing memory module
  (semantic/episodic/procedural); binds the result under `<name>`. Applicability = the query
  returns something.
* a bare predicate clause — a pure applicability check that binds nothing (e.g. a required
  property/state must be present); exact shape left to the grounding implementation, a foreseen
  form.
* `{"$decide": "<description>"}` — escalation, only when applicability is genuine judgment.

A body param **reads** a bound value via `{"$bind": "<name>"}` (optional `path`) — the
named-binding sibling of `$from`. The two read tokens partition by source: `$from` reads
`Activity.history`, `$bind` reads the **named-binding namespace** (guard retrievals plus a
mechanical sub-goal's loop element).

Binding is **named retrieval, not unification** (keeps the no-logic-KR boundary): the guard is
AgentSpeak's context doing its dual job — applicability test *and* variable binding — over
S-ORA's memory modules. The guard is evaluated in **Reason, at plan applicability/entry** (right
after `retrieve()`/`infer()` yields a candidate, before the body advances); mechanical retrievals
resolve for free, a `$decide` clause escalates. This is the home for the **long-term-memory param
binding** gap 1 identified. `$from` is **unchanged** — history-only structural provenance,
resolved per step at grounding. The two mechanisms split by origin *and* timing: guard-memory
bound once at entry (stable knowledge), `$from`-history bound per step (fresh dataflow). Because
the guard *names* what it needs, a body param that references a guard name the guard failed to
bind makes the plan **inapplicable** — a mechanical unbindable flag instead of a hallucinated
literal.

**Trigger binding is implicit in synthesis.** `Plan.goal` is a string, not a structured term, so
goal-derived values are baked into params by the planner at synthesis time — no third runtime
binding source. Exactly two runtime mechanisms remain: guard-memory and `$from`-history.

**Sub-goal — the sole recursion primitive.** A sub-goal is a `Step` with a `next_action="subgoal"`
sentinel (the `WAIT`-style pattern), carrying a `goal` and a `mode`:

* **Deliberative** (`retrieve()` then `infer()`): reaching the sub-goal fires `_infer_`
  **mid-plan** — an off-cycle internal action exactly as at plan start
  ([ADR-0021](0021-llm-calls-as-async-internal-actions.md)): the activity goes to `running` with
  `pending_inference`, resolves via `inference_sink`. The synthesized sub-plan is **pushed as a
  new frame** onto the activity (see below); a cached sub-plan may be `retrieve()`d first (a real
  plan library keyed by the sub-goal). This is S-ORA's synthesis-as-selection reused, so it needs
  none of the reactive belief-base machinery
  [ADR-0002](0002-activity-as-sole-first-class-construct.md) declined.
* **Mechanical** (the uniform-map case): a sub-goal over a collection resolved at expansion from
  history (a `$from`) with a fixed step template fans out `len(collection)` concrete copies with
  the element bound — **no model call**. Count and dispatch are mechanical; the model touches only
  per-item selection when a predicate is a `$decide`.

The per-iteration loop element (e.g. `as: "apartment"`) resolves from the **named-binding
namespace** — the same namespace the context guard populates, read in the template via
`{"$bind": ...}` — *not* from `$from`, which stays history-only.

**Sub-plan execution = frame stack.** `Activity` generalizes its single `plan` + `step_index`
into a small stack — the active frame stays in `plan`/`step_index`, suspended parents in
`parent_frames` — AgentSpeak's intention stack, not a new first-class type. A deliberative sub-goal **pushes** the sub-plan; on completion the frame
**pops** and the parent resumes at its next step. Mechanical fan-out splices in-place within the
current frame (no push). One external action per cycle is preserved throughout: the active frame
advances one step per cycle exactly as a flat plan does; a mid-plan `_infer_` is an internal
action, not an external one.

**The functional-dataflow family is dropped.** `$if`/`$select`/`$reduce` are not plan primitives;
branch, fold, and heterogeneous continuation are just shapes a deliberative sub-plan's `infer()`
naturally produces. Waiting/"until", parallelism, and reconsideration remain owned by the cycle
([ADR-0019](0019-blocked-state-machinery-and-percept-storage.md),
[ADR-0009](0009-five-phase-decision-cycle.md),
[ADR-0020](0020-hard-interrupt-and-await-input.md)), never duplicated as plan keywords.

**Recursion is unbounded (like AgentSpeak).** Sub-goals may fire sub-goals; the cycle/activity
budget is the practical bound. This deliberately trades a bounded, non-Turing plan language for
generality and fidelity to the lineage. The intended overflow behavior for the pathological
("irrational") runaway case is a **last-resort circuit breaker at the budget ceiling**: synthesize
a summary of what the activity is attempting and transition to **await-input** via ADR-0020's
`InputWait`/`interrupt()` seam — distinct from normal sub-goal deliberation so deep tasks do not
pester the user. This reuses existing machinery and its wiring is **deferred**; the sub-goal
decision does not depend on it.

**Explicitly out of scope: operator preconditions.** A synthesized plan fails at the joints the LLM
silently got wrong — the RentAFlat cardinality bug, or a param assumed to be a prior output that
should have come from memory. Precondition reasoning does happen, but *implicitly in the LLM and
left unrepresented, hence unverifiable*. Making operator preconditions/postconditions **explicit**
would buy cheap, mechanical validation of a synthesized plan against operator contracts (no search)
— but that is deliberately **not** a step back toward the classical look-ahead planning PRS
rejected; it is a separate, labeled future decision, not part of this ADR.

### Positive Consequences

* Both real gaps close: params bind from long-term memory (guard), and multi-item tasks become
  expressible (sub-goals) with the mechanical case still `len(data)`-bound.
* Two clean binding mechanisms, one job each — guard-memory (bound once at entry) vs
  `$from`-history (bound per step) — instead of one overloaded resolver.
* The anti-hallucination flag falls out: a param naming an unbound guard value ⇒ inapplicable
  plan, not an invented literal.
* Near-zero new deliberation machinery: a sub-goal is "fire `_infer_` at `step_index > 0`", and
  sub-plan caching reuses `ProceduralMemory.retrieve()`.
* One recursion primitive subsumes branch/fold/re-planning; the plan language stays
  practical-reasoning-shaped, and the cycle keeps sole ownership of waiting/parallelism/interrupt.
* Plans stay (better) reusable skeletons: the stored plan holds abstract sub-goals; sub-plans are
  synthesized fresh per run.

### Negative Consequences

* The plan language becomes recursively expressive (Turing-capable via sub-goal recursion),
  abandoning a deliberately bounded family; correctness now leans on the cycle/activity budget
  and the deferred await-input circuit breaker.
* A deliberative sub-goal's count is **model-emitted**, not `len(data)`-bound — reliable when the
  model maps over a visible list, but weaker on large collections; the mechanical mode exists
  precisely to keep the common uniform-map shape data-bound.
* `Activity` grows a frame stack, and grounding grows a guard-evaluation step and a
  named-binding namespace distinct from `$from` — more state and more Reason-phase machinery.
* A new planner prompt contract: the model must author guard clauses and `subgoal` steps; a
  mis-authored guard degrades to an inapplicable/empty result rather than a hard failure, but it
  is still fragility of the same class as ADR-0017's result-shape guess.
* A sub-goal's `mode` (mechanical vs deliberative) is authored by the planner at synthesis, so a
  mis-classification — `mechanical` for a heterogeneous continuation, or `deliberative` for a
  uniform map (spending model calls a `len(data)` fan-out would have avoided) — is its own failure
  mode, the sub-goal-level analogue of the result-shape guess above.

## Pros and Cons of the Options

### (a) Functional-dataflow family (`$foreach`/`$if`/`$select`/`$reduce`)

* Good, because it closes the iteration gap with a mechanical, `len(data)`-bound count and
  extends the existing reference family directly.
* Bad, because it imports a computation-graph paradigm onto a practical-reasoning plan, admits a
  whole control-flow vocabulary, and does nothing for the memory-binding gap.

### (b) Widen `$from` into a multi-source resolver

* Good, because it closes the memory-binding gap with no new construct.
* Bad, because it overloads the one reference that was clean *because* it was history-only
  provenance, and conflates stable-knowledge retrieval (bound once) with fresh dataflow (bound
  per step).

### (c) AgentSpeak schema, S-ORA substrate (chosen)

* Good, because it closes both gaps in the plan's own reasoning shape, splits the two binding
  needs cleanly, reuses synthesis/grounding machinery, and collapses branch/fold/re-planning into
  one sub-goal primitive.
* Bad, because it accepts unbounded recursion and a model-emitted count for deliberative
  sub-goals, and it grows `Activity` state (frame stack) plus Reason-phase machinery (guard
  evaluation, named bindings).

## Links

* Refines [ADR-0017](0017-parameter-grounding-in-reason.md) — keeps its `$from`/`$decide` split
  and grounding path; adds the context-guard binding source (long-term memory) alongside
  history, and the sub-goal body construct.
* Extends [ADR-0021](0021-llm-calls-as-async-internal-actions.md) — a sub-goal fires `_infer_`
  **mid-plan** (`step_index > 0`), reusing `pending_inference`/`inference_sink` unchanged; on
  resolve the sub-plan is pushed as a frame rather than replacing `Activity.plan`.
* Consistent with [ADR-0002](0002-activity-as-sole-first-class-construct.md) — the frame stack
  generalizes `step_index`, not a belief/desire/intention triad; sub-goal selection is synthesis,
  not the reactive belief-base library ADR-0002 declined.
* Honors [ADR-0009](0009-five-phase-decision-cycle.md) — one external action per cycle; the
  active frame advances one step per cycle, mid-plan inference is an internal action.
* Defers to [ADR-0020](0020-hard-interrupt-and-await-input.md) — the recursion-overflow circuit
  breaker (budget ceiling → summary → await-input) is the intended behavior, wiring deferred.
