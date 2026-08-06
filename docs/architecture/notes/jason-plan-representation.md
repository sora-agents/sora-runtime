# AgentSpeak/Jason as an inspiration for S-ORA's plan representation

* Date: 2026-08-04 (trimmed 2026-08-05)
* Status: design analysis — conceptual companion to [ADR-0022](docs/architecture/adrs/0022-plan-representation-context-guard-and-subgoals.md), which realizes the plan representation; also informs [ADR-0002](docs/architecture/adrs/0002-activity-as-sole-first-class-construct.md) and touches [ADR-0017](docs/architecture/adrs/0017-parameter-grounding-in-reason.md). The concrete plan-representation spec moved into ADR-0022; what remains here is the BDI-lineage framing, the synthesis-not-planning reframe, and the preconditions→plan-validation seed — the parts the ADR deliberately does not carry.

## The question

S-ORA's plan language binds operator params with a small dataflow vocabulary — `$from` (mechanical
reference to a prior action's output) and `$decide` (escalate to one model-judgment call). An
earlier proposal extended that with collection references (`$foreach`, plus a `map`/`filter`/
`reduce`/`branch` family) — which drifts toward a
**functional-dataflow** paradigm (LangGraph-style computation graphs). Is that the right paradigm
for representing *procedural knowledge*, or should the plan representation be inspired by
**AgentSpeak/Jason** (BDI), which was designed for exactly that — plans triggered by goals/events,
guarded by context conditions, with a body of subgoals and actions?

## Verdict

The instinct is right: a plan is **practical-reasoning-shaped**, not data-computation-shaped, and
piling `map`/`filter`/`reduce`/`branch` into it imports the wrong paradigm. The right move is to
borrow AgentSpeak's plan **schema** (trigger + context-guard + body) *loosely*, while keeping
S-ORA's own **substrate** (LLM-synthesized not developer-authored; structured-value binding not
logic unification; activities not a belief/desire/intention triad). "Adopt Jason" is the wrong
conclusion; "adopt its lineage, with a clear boundary" is the right one — and that boundary is
[ADR-0002](docs/architecture/adrs/0002-activity-as-sole-first-class-construct.md).

## What S-ORA already borrows (deliberately)

The BDI correspondence is not accidental — it was an inspiration source with a boundary drawn in
ADR-0002:

* **Environment / artifact model (from JaCaMo/CArtAgO).** Workspaces/tools with the three-part
  usage interface — observable properties, signals, operations ([ADR-0004](docs/architecture/adrs/0004-tool-usage-interface.md))
  — are CArtAgO artifacts; `invoke` is an artifact operation, invoked **asynchronously**. In
  JaCaMo, Jason actions invoke artifact operations asynchronously; that is part of the inspiration
  for S-ORA's off-cycle async operations ([ADR-0021](docs/architecture/adrs/0021-llm-calls-as-async-internal-actions.md)),
  not a place S-ORA is "ahead of" Jason.
* **Interleaved intentions ≈ activities.** Many concurrent activities, each advancing at most one
  external action per cycle ([ADR-0009](docs/architecture/adrs/0009-five-phase-decision-cycle.md)),
  mirrors BDI's interleaved intention execution — but modeled as `Activity`, the sole first-class
  unit, with no separate intention type (ADR-0002).

## The PRS reframe: S-ORA synthesizes, it does not plan

An earlier framing (that S-ORA's differentiator over ReAct is *classical look-ahead* via upfront
planning) is **withdrawn** as an overclaim. AgentSpeak's lack of look-ahead is not a deficiency —
its basis, the Procedural Reasoning System (PRS; Georgeff & Lansky), was proposed precisely to
*avoid* automated planning by having developers author plans. What S-ORA does is the same **kind**
of thing with authorship moved to an LLM: it **synthesizes a plausible sequence of operators**
(action templates), it does not search a state space.

The tells that this is synthesis, not planning:

* params are not fully bound at synthesis time; and
* there is no reasoning over operator **preconditions / postconditions**.

So there is no "look-ahead vs. reactive" tension between S-ORA and AgentSpeak: **both are
non-planning procedural systems.** The only substantive axis of difference is *synthesized* plans
(S-ORA) vs. *authored* plans (PRS/Jason).

A subtlety on preconditions: it is not that no precondition reasoning happens — it is **implicit in
the LLM and left unrepresented, hence unverifiable.** Synthesized sequences fail exactly at the
joints the LLM silently got wrong (the RentAFlat iteration/cardinality bug; a param that should come
from memory but was assumed to be a prior output). Making preconditions **explicit** is therefore
*not* a step toward classical planning — its non-planning payoff is **cheap, mechanical validation**
of a synthesized plan against operator contracts (no search). That is a labeled, separate future
decision, not part of "loosely AgentSpeak."

## Borrow the schema, decline the substrate

| AgentSpeak feature | S-ORA stance | Why |
|---|---|---|
| Plan = trigger + context + body | **Borrow** (loosely) | Practical-reasoning shape; attaches to `Activity`, not an intention type (ADR-0002) |
| Context condition (applicability) | **Borrow** as non-logic guard that also **binds** | Named retrieval over memory (binds values under names) + applicability, or `$decide` for genuine judgment — *applicability + binding*, not STRIPS preconditions |
| Iteration via recursion / per-item plan firing | **Borrow** as **sub-goals** | Reaching a sub-goal re-fires `infer()`/`retrieve()` — synthesis *is* S-ORA's selection mechanism, so no reactive belief-base machinery (which ADR-0002 declined) is needed; a uniform map still resolves mechanically (`len(collection)`), an open continuation re-infers |
| Logic-variable unification for binding | **Decline** | Drags in logic-based KR, a poor fit; `$from` (structural provenance binding over structured values) is the intended non-logic analogue |
| Belief / desire / intention types | **Decline** | ADR-0002: `Activity` is the sole first-class unit; procedural knowledge attaches to it |
| Effects / postconditions on operators | **Decline (for now)** | Their only real job is enabling the look-ahead PRS rejected; re-enter *only* for cheap plan validation, as a separate decision |

### The dataflow vocabulary, judged

* **`$from` — keep.** Structural provenance binding is mechanical (no LLM), and it is what
  distinguishes S-ORA from LangGraph (shared-state graph + reducers) rather than aligning it with
  that paradigm.
* **The `map`/`filter`/`reduce`/`branch` family (and `$foreach`/`$if`/`$select`/`$reduce`) —
  rejected as first-class.** Branch and fold are *shapes a synthesized sub-plan produces*;
  iteration is a **sub-goal**, not a functional-map primitive. Reaching a sub-goal re-fires
  `infer()`/`retrieve()`, which *is* S-ORA's selection mechanism — dissolving the "needs reactive
  selection over a belief base" objection to per-item plan firing. See
  [ADR-0022](docs/architecture/adrs/0022-plan-representation-context-guard-and-subgoals.md) for the
  realized form.

## The gap this surfaced, and where it landed

The clearest concrete gap this analysis surfaced: **param values bind only to outputs of previous
actions** (`$from` resolves into `Activity.history`); they **cannot be matched against long-term
memory.** In AgentSpeak terms, a plan body binds from the whole *belief base* (long-term beliefs
included), whereas `$from`-only-history can reference only recent perception. The second gap was
iteration over a run-time collection (the RentAFlat cardinality bug).

Both are resolved by the concrete representation this analysis pointed to — trigger + context-guard +
body, the two-mechanism binding split (guard-memory bound once at entry, `$from`-history bound per
step), and iteration via sub-goals (mechanical `len(collection)` fan-out or deliberative
re-`infer()`, run on a frame stack) — **specified in
[ADR-0022](docs/architecture/adrs/0022-plan-representation-context-guard-and-subgoals.md), not
repeated here.** The two substantive additions it identified: the context guard binds from long-term
memory (retrieval, not a widened `$from`), and sub-goals give the body recursive reach. The rest of
this note keeps the framing the ADR does not carry — the BDI/JaCaMo lineage, the
synthesis-not-planning reframe, and the preconditions→plan-validation seed — all above.

## Caveat (resolved)

The open question this note flagged — whether `ProceduralMemory` could carry parameterized,
context-guarded plan templates attached to an `Activity` without smuggling back an intention type —
was answered affirmatively in ADR-0022: the guard is dict clauses on `Plan` (no new type), and the
sub-goal frame stack generalizes `step_index` rather than adding an intention. The concept-level fit
still awaits verification against runtime code (not yet written).
