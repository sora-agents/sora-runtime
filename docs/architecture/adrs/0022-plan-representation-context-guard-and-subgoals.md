# Plan representation: retrieval-binding context guard, sub-goals, and signal-triggered pending conditions (AgentSpeak-inspired)

* Status: proposed
* Date: 2026-08-05

## Context and Problem Statement

A `Plan` is a flat `list[Step]`, synthesized once by `_infer_` and walked linearly — one `Step`
→ at most one external action per cycle ([ADR-0009](0009-five-phase-decision-cycle.md)).
[ADR-0017](0017-parameter-grounding-in-reason.md) added run-time param binding to that skeleton:
`{"$from": ...}` resolves a scalar from a prior result mechanically, `{"$decide": ...}` escalates
a scalar judgment to one model call. Four gaps remain, each surfaced by a real Gaia2 run:

1. **A plan can only bind params from its own execution history.** `$from` resolves against
   `Activity.history`; there is no way to bind a param from long-term memory (the user's home
   address, a preferred vendor, a saved policy). The planner either invents a literal or
   mis-assumes the value is a prior output. (RentAFlat, 2026-08-03.)
2. **A plan cannot express an operation over a collection discovered at run time.** "Save *each*
   qualifying apartment", "email *each* relative" — the iteration count is unknowable at
   synthesis time, so a linear plan collapses each "for each" to a single call and under-counts
   against the oracle. (RentAFlat, 2026-08-03.)
3. **A plan cannot bind a param from observed world state.** Observe keeps a live snapshot of every
   focused tool's observable properties in `WorkingMemory.properties`, refreshed each cycle, but no
   plan construct addresses it: `$from` reads `Activity.history`, `$bind` reads the named-binding
   namespace, and neither reaches the world *as currently perceived*. A Gaia2 adaptability run
   (2026-08-20) failed on exactly this — the contact the goal required sat in `Contacts.state` from
   the first cycle, while the only operation exposing the field it had to be matched on was
   paginated, so the agent told the user "I could not find a contact" about a record already in its
   own working memory.
4. **A plan cannot say what would make it relevant again.** A body is a finite sequence, so an
   activity terminates when it runs out — even where the goal it served was explicitly conditional.
   A Gaia2 adaptability run (2026-08-21) failed on exactly this: the synthesized plan's own prose
   stated three conditional clauses ("if he cannot make it, reschedule to the date he proposes"),
   the body encoded none of them, the activity terminated on a confirmation to the user, and the
   reply that arrived minutes later had nothing left to reach — 4 of 11 oracle actions. Nothing in
   the representation distinguishes *this goal is finished* from *this goal's body is finished*.

The framing that resolves all four: a `Plan` is **practical-reasoning-shaped**, not
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
belief/desire/intention triad. How should the plan representation realize that schema so that all
four gaps close without importing that triad or a foreign control-flow layer?

## Decision Drivers

* **Bind from the whole knowledge state, not just recent history.** A param should resolve from
  long-term memory as readily as from a prior action's output — AgentSpeak's context binds from
  the whole belief base, not only fresh perception.
* **The memory module, not the fact, fixes the binding time.** A binding source is characterized
  by *when* it may legitimately be read, not only by *where* the value comes from — and "when"
  follows from the memory module holding the value, not from how stable the fact itself is.
  Long-term memory holds a value until something rewrites it, so a plan may bind from it once at
  entry; working memory's property snapshot is replaced every cycle, so it may only be bound at the
  step that uses it. The same fact can legitimately sit in both. A mechanism that freezes a moving
  value is not a cheaper version of one that re-reads it — it is a different, and usually wrong,
  mechanism.
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
* **Staying relevant must be declared, and declaring it must be bounded.** An activity that outlives
  its body has to say *what* would revive it. "Keep it alive in case something happens" is not a
  representation: at one extreme nothing ever happens and the activity leaks forever, at the other
  every under-specified goal qualifies and nothing terminates. A survivable condition must name a
  mechanical gate narrow enough that most events never reach it, so the cost of watching is paid per
  *relevant* event rather than per event.

## Considered Options

* **(a) A functional-dataflow family** — first-class `$foreach`/`$if`/`$select`/`$reduce`
  collection references, expanded at grounding time. Closes gap 2 but imports a
  computation-graph paradigm (LangGraph-shaped) alien to a practical-reasoning plan, and leaves
  gap 1 (memory binding) untouched.
* **(b) Widen `$from` into a multi-source resolver** — let a single reference reach into
  `{history, long-term memory, goal, literals}`. Closes gap 1 but muddies the one mechanism that
  was clean precisely because it was structural provenance over history.
* **(c) AgentSpeak schema, S-ORA substrate (chosen).** Give the plan a **retrieval-binding
  context guard** (closes gap 1), a **sub-goal** body construct that re-fires `infer()`
  mid-plan (closes gap 2), a **`$prop` read token** over the observed-property snapshot (closes
  gap 3), and **declared pending conditions** that outlive the body (closes gap 4), keeping
  `$from` history-only and reusing existing synthesis/grounding machinery.

For gap 4 specifically:

* **Unconditional keep-alive** — an activity whose goal contains conditional language simply does
  not terminate when its body ends. Rejected: it is not a representation but the absence of one, and
  it makes termination undecidable in both directions (see the corresponding decision driver).
* **A trailing wait step in the body** — encode the wait as a final `Step`. Rejected: a body step is
  positional and single-shot, while the branch may fire *before* the body ends (a reply that beats
  the confirmation), fire more than once, or never fire at all. It also makes a waiting plan
  indistinguishable from a stalled one: a plan parked on such a step makes no progress by
  construction, so reconsideration checkpoints ([ADR-0024](0024-plan-reconsideration-context-adaptation.md))
  would keep firing replans into `Activity.replan_trail`, which counts exactly "replans with no
  progress between them" ([ADR-0025](0025-deliberation-breakers.md)) — a healthy wait would burn the
  replan budget and trip a breaker. `BLOCKED` avoids this for free, because Situate does not select
  a blocked activity at all, so no replans accrue while it waits.
* **Declared pending conditions (chosen)** — the plan names a mechanical gate, a semantic trigger, a
  continuation, and a bound; the activity blocks on the gate instead of terminating.

## Decision Outcome

Chosen option: **(c)**, because it closes all four gaps with the plan's own reasoning shape
rather than a foreign paradigm, splits the binding needs by origin and timing (each mechanism keeps
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
named-binding sibling of `$from`. The three read tokens partition by source: `$from` reads
`Activity.history`, `$bind` reads the **named-binding namespace** (guard retrievals plus a
mechanical sub-goal's loop element), and `$prop` reads `WorkingMemory.properties` — the observed
world-state snapshot (below).

Binding is **named retrieval, not unification** (keeps the no-logic-KR boundary): the guard is
AgentSpeak's context doing its dual job — applicability test *and* variable binding — over
S-ORA's memory modules. The guard is evaluated in **Reason, at plan applicability/entry** (right
after `retrieve()`/`infer()` yields a candidate, before the body advances); mechanical retrievals
resolve for free, a `$decide` clause escalates. This is the home for the **long-term-memory param
binding** gap 1 identified. `$from` is **unchanged** — history-only structural provenance,
resolved per step at grounding. These two split by origin *and* timing: guard-memory bound once
at entry (stable knowledge), `$from`-history bound per step (fresh dataflow) — and `$prop` below
adds a third cell to the same partition. Because
the guard *names* what it needs, a body param that references a guard name the guard failed to
bind makes the plan **inapplicable** — a mechanical unbindable flag instead of a hallucinated
literal.

**Observed world state — `$prop`, bound per step.** The third binding source is
`WorkingMemory.properties`: the replace-by-`(source, name)` snapshot Observe refreshes each cycle
for every focused tool ([ADR-0019](0019-blocked-state-machinery-and-percept-storage.md)). A body
param, a data-op's `in`, or a mechanical sub-goal's collection reads it as
`{"$prop": "<tool_id>.<property_name>"}` with an optional `path`, resolved **per step at
grounding** — the same point in the cycle where `$from` resolves.

*Why not the context guard.* Working memory is a memory module, and AgentSpeak's context binds from
a belief base that is largely perception, so the guard looks like the natural home for this. It is
the wrong one. A guard clause binds **once, at plan applicability**, whereas a property is
*re-observed* state whose entire value is being current. AgentSpeak can afford to conflate the two
because it re-checks context at every intention selection over an atomic belief base; an S-ORA plan
is a long-lived multi-cycle skeleton, so guard-binding a property would freeze a moving value for
the plan's whole life — hundreds of cycles in the run that motivated this. The guard therefore
reads long-term memory only. What separates the two mechanisms is the *memory module being read* and
the update discipline it guarantees — long-term memory is durable until rewritten, the property
snapshot is replaced every cycle — from which the binding time follows; provenance selects the
module rather than competing with freshness as the explanation.

*Why not widen `$from`.* That is option (b), rejected below for conflating binding *times*. A
distinct token keeps each mechanism's timing legible at the reference site rather than hiding it
inside one overloaded resolver.

*Addressing.* The property snapshot is keyed by `(source, name)`, so the canonical form is
qualified and split on the **last** dot — safe for a tool id that itself contains dots (`wot:lamp.local/Lamp.state`) because a
property name never contains one. A **bare** property name resolves *iff exactly one focused tool
exposes it*; where several do (ARE gives thirteen tools a `state` property) the reference is not
silently resolved to one of them — it fails carrying a defect that names the candidates, so the
replanned step can differ where it has to.

*Focus is a precondition, never an implicit effect.* A property is in working memory only while its
tool is focused, so a `$prop` naming an unfocused tool resolves to nothing and the plan must carry
its own `focus` step. The runtime must **not** auto-focus on that miss: focus is an *external*
action, and dispatching one from inside grounding would break one-external-action-per-cycle
([ADR-0009](0009-five-phase-decision-cycle.md)). The defect names the missing focus instead.

*Properties only.* `$prop` addresses the property snapshot, not `WorkingMemory.signals`. Signals are
an append log with their own role in the blocked-state machinery; a token addressing them is
foreseen, not decided here. A pending condition's `watch` (below) does not change this — it *matches*
a signal to open a gate, and never reads a value out of one to bind a param.

*Consolidation moves values between memory modules, not mechanisms.* Nothing makes an observed
property inherently transient — a contact's job or a car's color is stable knowledge that merely
arrives through perception — and a future process may consolidate such an observation into
**semantic memory**, whose remit is exactly the agent's durable knowledge about the world, so that
it outlives unfocusing. Consolidation moves the *value* between memory modules; it adds no binding
mechanism, because the consolidated copy is thereafter read by the guard like any other long-term
knowledge. It also makes the two-token choice *more* meaningful rather than less: once the same fact
can sit in both, `$prop` and `$bind` are how a plan says which it wants — the world as currently
perceived, or the knowledge the agent has committed to long-term memory.

**Trigger *binding* is implicit in synthesis.** `Plan.goal` is a string, not a structured term, so
goal-derived values are baked into params by the planner at synthesis time — the goal itself is
never a runtime binding source. Exactly three runtime mechanisms remain, partitioned by origin *and*
timing: guard-memory bound once at entry, `$from`-history bound per step, and `$prop`-world-state
bound per step. This says nothing about *triggering*: a pending condition's `watch` (below) is a
real, structured trigger — it simply yields no param values, so the three binding mechanisms are
untouched by it.

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

**Pending conditions — the trigger, pointed forward.** The schema this ADR borrows is *trigger +
context guard + body*, but only two thirds of it were ever load-bearing: the trigger was disposed of
as a binding source (above) and nothing took its place, which left a plan able to say what it is
doing and under what circumstances, never what would make it relevant *again*. `Plan` gains
`pending: tuple[PendingCondition, ...]`, four fields each:

* **`watch: SignalWait`** — the **mechanical gate**, reusing ADR-0019's wait type whole, `path`
  included. **Required**: a condition with no gate is rejected at synthesis, because it degenerates
  into evaluating every condition against every signal, which is the unbounded keep-alive this ADR
  rejected wearing a field name.
* **`when: str`** — the semantic trigger, evaluated only against changes that already passed the
  gate.
* **`then: str`** — the sub-goal to pursue when `when` holds.
* **`until: str | None`** — the bound that retires the condition.

Only the gate is typed. `when`/`then`/`until` are prose because their consumer is `_infer_`, which
already takes prose goals — so this adds no condition language, no predicate DSL, and no event
algebra to the plan. The generality comes from the split rather than from expressiveness: *where to
look* is the only part a protocol can answer, so it is the only part given a structure, and it is
also the part that has to be cheap.

The condition the run in gap 4 wrote in prose and dropped on the floor:

```json
{"pending": [{
  "watch": {"signal": "state_changed",
            "source": "insim:are/Emails",
            "path":   "folders.INBOX.emails"},
  "when":  "Åke replies that he cannot make the scheduled date, or proposes a different one",
  "then":  "Move the Film Production Day to the date Åke proposes: clear what is already scheduled that day, then re-add the full-day event",
  "until": "the Film Production Day has taken place"
}]}
```

*Skeleton vs. run state.* `Plan.pending` is part of the reusable skeleton, like the body and the
guard. The per-run state — which conditions are still unsatisfied, and how far each has evaluated —
lives on the `Activity` as `pending_conditions`, the same split `plan`/`step_index` already draws.
Any frame may declare conditions, and they are **lifted to the activity as soon as that frame is
live** — not when it pops. Lifting at entry satisfies both requirements at once: the condition
outlives the plan that noticed it (which is the point — a deliberative sub-goal is usually where the
agent first learns a branch exists, having just sent the mail, and its frame pops long before any
reply), *and* it is already watching while the body still runs, which is what the early-reply case
below needs. Lifting is idempotent, so it can simply run every cycle. Requiring the root planner to
foresee every branch before synthesizing a body would put the declaration in the one place least
likely to know.

A lifted condition's mark starts at the **current** signal count rather than at zero. A signal that
arrived before the condition was declared cannot be the event it waits for, and the retention log
holds hundreds — starting at zero would make every new condition re-judge the whole backlog on its
first cycle.

*State machine.* **Body exhausted:** with no unsatisfied conditions, `TERMINATED` as today; otherwise
`_suspend_` to `BLOCKED` with `blocked_on = ConditionWait(...)` carrying the union of the
unsatisfied conditions' watches — ADR-0019's third `blocked_on` variant. **A gate opens:** Observe's
existing resume pass matches an observed signal against those watches with the same name/source/path
equality it already applies, and resumes the activity to `READY`. No model, no new phase machinery,
no change to the pass's mechanical character. **Evaluation:** Situate selects the activity like any
other; Reason sees conditions whose gate opened past their mark and fires **one** `_infer_` covering
all of them — *given these `Change` records, which `when`s hold, is any `until` now satisfied, and
if a `when` holds, the plan for its `then`* — off-cycle on `pending_inference`/`inference_sink`
unchanged ([ADR-0021](0021-llm-calls-as-async-internal-actions.md)). **Resolution:** a holding
`when` pushes its `then` as a frame — the deliberative sub-goal path, triggered rather than
positional; a satisfied `until` drops the condition; if nothing holds and conditions remain,
`_suspend_` back to `BLOCKED` with marks advanced, so the same signal is never evaluated twice.

Conditions are live from plan entry, not only after the body, so a gate that opens mid-body is
evaluated at the activity's next Reason selection *before* the body advances — the early-reply case
needs no separate path, exactly as ADR-0019's early-completion-signal case does not. Firing does not
consume a condition either: `until` is what ends it, and the mark only suppresses re-evaluating a
signal already seen, so "whenever X" needs no extra field. Each condition carries **its own** mark
over ADR-0019's monotonic signal sequence — the per-waiter mark that ADR requires, since a shared
cursor would let the first condition to advance it blind the rest.

*What it costs.* The gate is what keeps this affordable, and the motivating run is the measurement:
of its four signals, two fail on `source` (Calendar), one is the agent's own `send_email` landing in
`folders.SENT.emails` and fails on `path`, and one — Åke's reply into `folders.INBOX.emails` —
opens the gate. **One** additional model call for the entire scenario, against seven the run already
spent. Batching per evaluation rather than per condition is what keeps that ratio as conditions
accumulate.

*Declared resumes; undeclared amends.* A signal that opens at least one declared gate is handled
here, and the activity **resumes** — the branch was always part of the goal it was pursuing, so it
is the same activity and the same episode. A signal that opens **no** gate is out of scope for this
ADR: recovering relevance for an activity that never declared the condition has a different shape
entirely — it stays terminated, the user is asked, and a *new* activity amends it — and is decided
in [ADR-0026](0026-undeclared-relevance-recovery.md). What this ADR leaves unclaimed is precisely
that ADR's input, so the cheaper declared path shrinks the expensive undeclared one as the planner
improves. This ADR owns what a plan declares it is waiting for;
[ADR-0019](0019-blocked-state-machinery-and-percept-storage.md) owns the signal's shape and how a
wait matches it.

**The functional-dataflow family is dropped.** `$if`/`$select`/`$reduce` are not plan primitives;
branch, fold, and heterogeneous continuation are just shapes a deliberative sub-plan's `infer()`
naturally produces. Waiting, parallelism, and reconsideration remain owned by the cycle
([ADR-0019](0019-blocked-state-machinery-and-percept-storage.md),
[ADR-0009](0009-five-phase-decision-cycle.md),
[ADR-0020](0020-hard-interrupt-and-await-input.md)), never duplicated as plan control flow.

A pending condition's `watch`/`until` does not breach that line, and the distinction is worth being
precise about, because the original wording of this decision drew it in the wrong place ("waiting /
`until` is cycle-owned, never a plan keyword"). What a plan must not contain is a construct that
*executes* a wait — a step that blocks, a keyword the body advances into. A `PendingCondition` is
not a step, never blocks the body, and executes nothing: it *declares what a wait would be for*,
and every state transition it implies — the suspend, the match, the resume — is the cycle's,
performed by machinery ADR-0019 already owns. Declaring is this ADR's job; waiting is the cycle's.

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

* All four gaps close: params bind from long-term memory (guard) and from observed world state
  (`$prop`), multi-item tasks become expressible (sub-goals) with the mechanical case still
  `len(data)`-bound, and a conditional goal can say what would revive it (pending conditions).
* The borrowed schema's trigger half becomes load-bearing instead of vestigial, and the plan gains a
  distinction it did not have: *this goal is finished* vs. *this goal's body is finished*. A
  conditional clause the planner previously stated in prose and silently dropped now has a field to
  land in — so an omission becomes measurable rather than invisible.
* A waiting activity is distinguishable from a stalled one, which the trailing-wait-step alternative
  would have destroyed — reconsideration and the breakers keep working unchanged
  ([ADR-0024](0024-plan-reconsideration-context-adaptation.md),
  [ADR-0025](0025-deliberation-breakers.md)).
* Again near-zero new machinery: gate matching is ADR-0019's existing resume pass, evaluation is
  `_infer_`, and the continuation is the frame stack this ADR already introduces. The genuinely new
  runtime concepts are `PendingCondition`, `ConditionWait`, and one Reason branch.
* Cost tracks *relevant* events rather than signal volume, because the gate is mechanical and
  narrow: one extra model call across the whole motivating run, with the three irrelevant signals
  rejected on `source` and `path` without reaching a model.
* Three clean binding mechanisms, one job each — guard-memory (bound once at entry) vs
  `$from`-history and `$prop`-world-state (both bound per step) — instead of one overloaded
  resolver.
* Bulk state an adapter already publishes becomes mechanically computable: `$prop` feeds a data-op's
  `in` directly ([ADR-0023](0023-structured-value-data-ops.md)), so a lookup over an observed
  collection costs one filter rather than draining a paginated operation. In the run that motivated
  this, the needed record was reachable in a single mechanical step with no model call at all.
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
* `$prop` costs a `focus` step — one extra cycle and one of the plan's external actions — before
  the first reference to a property, and a planner that omits it learns only from the defect.
* `$prop` hands *raw adapter values* to mechanical comparisons, which surfaces whatever an adapter
  failed to normalize. ARE builds app state with `asdict()`, which leaves `Enum` members in place,
  so a mechanical `eq` against `"Female"` silently fails against `<Gender.FEMALE: 'Female'>`, and
  the same value defeats JSON rendering in prompts. That is an adapter-normalization defect rather
  than a plan-language one, but `$prop` is what makes it reachable from a plan.
* It does **not** close the pagination gap. `$prop` helps only where an adapter publishes bulk state
  as a property; where data is reachable *only* through a paginated operation, a plan still cannot
  iterate to exhaustion — a mechanical sub-goal is `len(data)`-bound over a collection it already
  holds, and no data-op *produces* a collection. That gap remains open and is not addressed here.
* A consolidation process, if built, introduces staleness in the *other* direction: an observation
  consolidated into semantic memory is knowledge about a past perception, and a guard retrieval
  cannot tell that it has since gone stale. Whatever consolidates percepts must carry provenance,
  and a plan needing currency must keep using `$prop` even where a consolidated copy exists.
* A sub-goal's `mode` (mechanical vs deliberative) is authored by the planner at synthesis, so a
  mis-classification — `mechanical` for a heterogeneous continuation, or `deliberative` for a
  uniform map (spending model calls a `len(data)` fan-out would have avoided) — is its own failure
  mode, the sub-goal-level analogue of the result-shape guess above.
* **The planner has to emit conditions, and has already demonstrated it will not.** The run in gap 4
  stated all three conditional clauses in its own plan prose and encoded none of them. This decision
  supplies somewhere to put them; it cannot make the model put them there. That is the same
  prompt-contract fragility as guard clauses and sub-goal `mode`, but with the worst failure mode of
  the three: an omitted condition is silent and *looks like success* — the activity terminates with
  a confident confirmation to the user. This is exactly why a path for undeclared relevance is not
  redundant with this one, and the decision to keep both rests on this observation rather than on
  taste.
* **A gate that never opens has no bound.** `until` is evaluated in the call a gate opening triggers,
  so a condition whose gate never opens is never retired and its activity stays `BLOCKED`
  indefinitely. Triggering on *absence* needs a timer, which is deferred to separate work. Until
  then the only bound is external — the benchmark harness treats an activity blocked on a condition
  as finished once its simulation clock stops, since past the end of a timeline no signal can
  arrive. That is a harness-level bound, not a runtime one, and it does not generalize to a
  long-running agent.
* **The judgment stays semantic, so this is cheaper, not free.** Matching "he can't make it" against
  an email body is irreducibly a model call; the mechanical part is only which signals are allowed
  to reach it. Any reading of this decision as a call-free adaptation mechanism is wrong.
* **A false gate opening costs a full evaluation.** ADR-0019's bidirectional prefix matching and its
  coarse-form `Change` both deliberately widen the gate, and each spurious opening buys one cycle
  and one `_infer_`. Path scoping bounds this; it does not eliminate it.
* **`until` is itself a condition**, evaluated the same semantic way as `when` — so retiring a
  condition is exactly as fragile as firing one, and an `until` that never holds fails as silently
  as a `when` that never holds.
* **A signal claimed here is never offered to the undeclared path.** Accepted false negative: one
  signal may satisfy a declared condition on activity A *and* be the very thing that should have
  revived unrelated terminated activity B; B never sees it. Preferred over evaluating every signal
  against every terminated activity, which is the cost this layer exists to avoid.
* Lifting conditions out of popped frames means an activity's watch set grows with its sub-goal
  depth, and nothing prunes it but `until` — so a deep activity accumulates gates that each widen
  the chance of a false opening.

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

* Good, because it closes all four gaps in the plan's own reasoning shape, splits the binding
  needs cleanly by origin *and* timing, reuses synthesis/grounding machinery, and collapses
  branch/fold/re-planning into one sub-goal primitive — with the pending condition reusing that
  same primitive for a continuation triggered by a signal rather than by position.
* Bad, because it accepts unbounded recursion and a model-emitted count for deliberative
  sub-goals, it grows `Activity` state (frame stack, pending conditions) plus Reason-phase
  machinery (guard evaluation, named bindings, condition evaluation), and it puts three separate
  authoring obligations on the planner (guard clauses, sub-goal `mode`, pending conditions), each
  of which degrades quietly when the model gets it wrong.

## Links

* Refines [ADR-0017](0017-parameter-grounding-in-reason.md) — keeps its `$from`/`$decide` split
  and grounding path; adds the context-guard binding source (long-term memory) and the `$prop`
  world-state source alongside history, and the sub-goal body construct.
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
* Feeds [ADR-0023](0023-structured-value-data-ops.md) — a data-op's `in` accepts `$prop`, so an
  observed collection is filtered/sorted/reduced by the same mechanical pipeline that consumes
  `$from` and `$bind`, and the unreadable-collection defect defined there is what reports a `$prop`
  that named no readable collection.
* Reads the working-memory property snapshot defined by [ADR-0019](0019-blocked-state-machinery-and-percept-storage.md) —
  `$prop` addresses `WorkingMemory.properties` (the replace-by-key property snapshot) and
  deliberately not `WorkingMemory.signals`.
* Builds on [ADR-0019](0019-blocked-state-machinery-and-percept-storage.md) for pending conditions —
  a `watch` *is* a `SignalWait`, `path`-scoped by that ADR's located change summaries; the gate is
  matched by its existing Observe resume pass; `ConditionWait` is its third `blocked_on` variant;
  and each condition's evaluation mark is the per-waiter high-water mark it requires over the
  monotonic signal sequence. The boundary: that ADR owns the signal's shape and how a wait matches
  one, this ADR owns what a plan declares it is waiting *for*.
* Complemented by [ADR-0026](0026-undeclared-relevance-recovery.md) — a change that opens no declared
  gate here is that ADR's entire input, handled by amending a terminated activity rather than
  resuming one. Declared resumes; undeclared amends.
