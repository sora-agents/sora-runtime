# Plan reconsideration via context-adaptation checkpoints (commitment as config)

* Status: proposed
* Date: 2026-08-17

## Context and Problem Statement

A synthesized `Plan` ([ADR-0022](0022-plan-representation-context-guard-and-subgoals.md)) is walked
linearly, and while it runs the world can move underneath it. The motivating failure is a real ARE
`email_calendar` run: the task is "schedule the team sync Alice asked for"; mid-run Alice sends a
follow-up changing the day (Monday → Tuesday); the agent, already committed to a Monday plan,
schedules Monday. The plan's *assumptions* were invalidated by a new percept and nothing forced a
reconsideration.

The context guard ([ADR-0022](0022-plan-representation-context-guard-and-subgoals.md)) is evaluated
**once at plan entry** — it answers *does this plan apply, and what does it bind*, not *is it still
valid three steps in*. And [ADR-0022](0022-plan-representation-context-guard-and-subgoals.md) is
explicit that "reconsideration remains owned by the cycle … never duplicated as plan keywords". So
the question is **how the cycle should decide, at run time, that an in-flight plan is no longer valid
and should be re-inferred** — without (a) blocking the cycle, (b) requiring domain knowledge to be
pre-authored where no author can foresee it, or (c) spending a model call every cycle and destroying
the amortization that makes a cached plan cheap.

This is the classic BDI **commitment** problem (Rao & Georgeff; Kinny & Georgeff; Schut &
Wooldridge): how strongly an agent commits to an intention/plan versus how eagerly it reconsiders in
light of new perception. The known trade-off is exactly the one S-ORA faces — stronger commitment
means more speed and less cost but less adaptiveness; the optimal reconsideration frequency tracks
the environment's rate of change, which the agent does not know a priori.

## Decision Drivers

* **No domain authoring of plan-invalidation semantics.** An `EmailClientApp` author does not, and
  should not have to, foresee "an incoming email could break some future plan's assumption." That is
  a property of the *plan* and *why the agent is reading mail*, not of the tool — cross-cutting, and
  unknowable at manual-authoring time. (An earlier draft that pushed a *maintenance predicate* into
  the tool `Manual` was rejected for exactly this; see Considered Options (a).)
* **Preserve the economy.** A static, unchanging world must cost **zero** reconsideration calls; the
  cached-plan amortization ([ADR-0022](0022-plan-representation-context-guard-and-subgoals.md)) must
  survive.
* **Reconsideration is cycle-owned.** It belongs in **Reason** (the phase that owns planning and
  invalidation), not as a plan/guard/manual keyword — honoring
  [ADR-0022](0022-plan-representation-context-guard-and-subgoals.md)'s boundary and
  [ADR-0009](0009-five-phase-decision-cycle.md)/[ADR-0019](0019-blocked-state-machinery-and-percept-storage.md)/[ADR-0020](0020-hard-interrupt-and-await-input.md).
* **General, not scenario-shaped.** The relevance decision ("does this change matter to this plan?")
  must not depend on efference tags, per-tool identity paths, or any device-specific bookkeeping that
  is unreliable to build in the general case.
* **A dial, not a fixed policy.** The right commitment strength is application- and
  environment-specific, so it is a **configuration choice**, pluggable and per-agent.

## Considered Options

* **(a) Mechanical maintenance predicates in the guard/manual.** Declare, per observable, a
  stability predicate (identity field-path + watch mode) and re-check it mechanically each cycle;
  synthesized best-effort or auto-attached from a manual flag. *Rejected.* It forces tool authors to
  reason about arbitrary future plans (the smell above), the `completion_signal` precedent does not
  transfer (a completion signal is intrinsic to an operation; "what invalidates a plan" is not
  intrinsic to a tool), and it needs per-tool identity scoping (or efference) to avoid firing on the
  agent's own writes — a general-case reliability problem we do not want to own.
* **(b) A model relevance-judgment every cycle.** General and needs no authoring, but a per-cycle
  model call destroys the amortization and, in a dense-write / embodied setting, is one call per
  cycle continuously. *Rejected as the default* (kept as the top of the frequency dial only).
* **(c) Commitment as configuration: context-adaptation checkpoints (chosen).** A per-agent
  `context_adaptation` setting decides **when** the cycle spends a general model **relevance
  judgment**, gated to *action commitment points* and fronted by a cheap mechanical change-gate, so a
  static world is free and a changing world pays only at the points where acting on staleness would
  do harm.

## Decision Outcome

Chosen option: **(c)**. Reconsideration stays in **Reason**, is driven by a per-agent
**configuration** dial, uses a **general model judgment** (no domain authoring, no efference), and is
**gated by action commitment points** so its cost tracks risk rather than time.

### The dial

```yaml
strategies:
  context_adaptation: before_writes   # none | before_writes | before_each_op
```

`context_adaptation` selects a pluggable **reconsideration policy** — a level name for the shipped
policies, or a dotted path to a custom one (the seam through which D5's author-declared checkpoints
and X2's adaptive policy will plug in later). The shipped levels:

* **`none`** — never reconsider on ambient percepts. (Blind commitment.) Correct for static tasks
  (most of Gaia2). Failure-driven re-planning is **orthogonal and still on**: a crashed/`not ok`
  step re-plans regardless — that is *impossibility*, not adaptation.
* **`before_writes`** (default) — before dispatching a **side-effecting** operation, run the
  reconsideration check. The risk-aligned default: a stale plan is only harmful when it *acts* on the
  staleness, and a write is the commitment point where that harm happens.

  A side-effecting step passes through Reason **twice** — once to escalate its params to the
  off-cycle `_ground_` model call, then again to dispatch the now-concrete invoke. The check runs on
  the **invoke pass only**, not the grounding pass: grounding is itself a side-effect-free model
  call, so guarding it would (a) spend a revalidation to maybe save a grounding — a model call to save a
  model call, net-negative in the common case where the plan is still valid — and (b) *miss* a change
  that lands during the grounding window (grounding read its world at dispatch, so late grounding
  cannot absorb a mid-flight change), which is exactly where a slow grounding call is most exposed.
  Checking after grounding catches that window. Because the check can fire a revalidation *between* grounding
  and the invoke, the resolved params are held (`Activity.grounded_params` peeked, not consumed, until
  the step actually commits) so the deferral doesn't force a second, costlier `_ground_` escalation.
* **`before_each_op`** — run the check before **every** external operation, read or write. Maximum
  caution for very dynamic worlds; still op-gated, so it skips planning/grounding/waiting cycles and
  is cheaper than "every cycle."

`before_writes` is framed as the **default checkpoint policy**: its implicit checkpoint is "every
side-effecting op." Author-declared checkpoints — a deferred generalization letting a manual name where
re-checking is worthwhile — generalize it for dense-write/embodied
tools where that implicit rule degenerates to a check almost every cycle — so this scale is
forward-compatible, not a retrofit.

### The check: a cheap mechanical gate, then a general model judgment

At a checkpoint the cycle runs a two-tier check:

1. **Mechanical change-gate (free).** Compare a baseline snapshot of perception (properties, signals,
   messages) against the current one. **Nothing changed → skip the judgment.** This is what keeps a
   static world at zero cost, even at `before_each_op`. It is generic — "did anything move," no
   per-property relevance, no authoring. It *over-fires* on the agent's own writes (a reply changes
   state), but that is harmless because the next tier decides.

   The baseline is the world the plan's **assumptions were formed against** — the perception snapshot
   at *infer time*, captured when the plan is fired and carried through to plan install (on
   `PendingInference`). It is deliberately **not** consulted at install; only at the policy's
   checkpoints. Capturing at infer time rather than at plan entry closes a real gap: a change that
   lands *during* inference (a slow model, a concurrent activity) is otherwise folded into a later
   baseline and never seen, so an inference-window invalidation (e.g. a "cancel the meeting" that
   arrives while the planner is thinking, then the world goes static) would slip past the first
   checkpoint. It costs at most one revalidation per plan, and only when the world actually moved during
   inference — exactly when a check is warranted; a static world still gates cold and pays nothing. A
   *reused* plan (retrieved, no fresh inference) has no infer-time snapshot and falls back to an
   entry-time baseline, anchored before the first checkpoint. The default signature is sensitive to
   *any* observable change, including non-semantic churn (a clock/heartbeat) and the agent's own
   writes; how that signature is computed is itself a pluggable seam — see *The change-gate is
   pluggable* below — so an application can project perception onto only its externally-meaningful
   part rather than exclude properties one at a time.
2. **Model relevance judgment (only when the gate is hot).** A single focused call: *given this plan
   (goal + the operations already executed + the intermediate values they bound + remaining steps) and
   these new percepts, is the plan still
   valid?* The executed half is load-bearing, not context padding: a checkpoint late in a plan leaves
   almost nothing in *remaining*, so a judgment made without history sees a goal whose work is nowhere
   in evidence and invalidates a plan that is in fact nearly done — and it is what makes the
   self-write reasoning below (*my own reply does not change the meeting day*) a judgment the model
   can actually make instead of a guess. This is **revalidation-first,
   then maybe re-infer**: on "still valid" the cycle proceeds; on "invalidated" it calls
   `reset_for_replan()` and re-infers against the current world. The revalidation is cheaper than a full
   re-plan, so in a mostly-stable world (most gate-hot checkpoints find nothing) it is the economical
   choice, and the expensive re-infer runs only on an actual invalidation.

   That "cheaper" has to be *maintained*, not assumed. Both prompts render the activity's executed
   history, which is append-only for the whole life of an activity — so an uncapped re-check drifts
   toward the size of the plan prompt it is meant to undercut, and it fires per write rather than per
   plan. Both are therefore windowed to their most recent N entries, the re-check's window
   deliberately the tighter: it answers one yes/no question about the remaining tail, where the recent
   results carry the signal and what is already done is implied by the tail's own contents. Grounding
   is left unwindowed — a `$from`/`$decide` reference may name any past result, and hiding the entry
   holding the referent fails the way truncating it mid-record does. Elision is marked with a count
   rather than being silent, since "nothing happened earlier" is a different claim from "not shown",
   and a planner told the former can legitimately decide to redo committed work.

   **The bindings are the other half of "already executed" (2026-08-24).** A data-op
   ([ADR-0023](0023-structured-value-data-ops.md)) appends nothing to `history` — its whole effect is
   a named entry in `Activity.bindings`. So the argument above, made for history, applies to bindings
   with more force: a plan whose early steps narrow a collection can reach a checkpoint having run
   most of itself while the re-check sees `(nothing executed yet)`. An observed run replanned at step
   9 for exactly this reason, and the replacement plan was the same plan with its bindings renamed —
   the expensive path taken to rediscover what was already computed. The re-check therefore renders
   `render_bindings` alongside the windowed history. The *plan* prompt deliberately does not: a
   replan discards bindings, so showing them there would describe state the new plan will not have.

**No efference, no manual relevance.** The self-write problem dissolves at tier 2: the revalidation reasons
"my own reply does not change the meeting day" and returns *still valid*, so a coarse gate no longer
loops. An agent's own action that *does* legitimately invalidate the plan is caught the same way —
nothing is excluded a priori.

### The change-gate is pluggable (efference on the cooperative path)

Tier 1's *how do I compute the signature* is a per-agent seam — `strategies.change_gate`, a
`ChangeGate` Protocol with one method, `signature(wm) -> object` — **orthogonal** to
`context_adaptation`: the policy decides *which* steps are checkpoints (WHEN), the gate decides
*whether* the world moved (WHETHER). Equal signatures across cycles mean nothing observable moved
since the plan was baselined, so the revalidation is skipped. The baseline is stored as `object`
(`PendingInference.baseline` / `Activity.reconsider_baseline`), so a gate may return any comparable
value; both the baseline capture and every later comparison route through the same gate, so they
share one signature space.

* **Default — `PerceptionSignatureGate`.** Domain-free: the sorted property reprs plus the
  signal/message log lengths (the original tier-1 snapshot). It over-fires on the agent's own writes,
  which is *correct but not free*: tier 2 still returns "still valid," so a self-write costs one
  revalidation rather than a loop. The right default when an app has authored nothing.
* **Domain gate — the efference filter.** An app that knows its external surface can project
  perception onto just that surface, so a self-caused change collapses to an unchanged signature and
  never even reaches the revalidation. This is the **same efference trick** a stateful `InterruptPolicy`
  applies on the hard-interrupt path (`MailDiffInterruptPolicy` diffs INBOX ids), now available on
  the *cooperative* path — and it is what removes the one-revalidation-per-self-write cost the default pays.

The ARE example ships `InboxChangeGate`: the union of INBOX email ids across observable `state`
properties, sharing the one `state → id-set` projection with `MailDiffInterruptPolicy`. The agent's
reply lands in SENT, read-flag flips and calendar adds don't touch the INBOX id-set — so none of the
agent's own writes move the signature, and only a genuine follow-up trips the checkpoint. It stays
example-level, email-shaped code (not manual-authored, not in the runtime); the general
read/write-tag efference that would make *any* self-caused change filterable without per-domain code
is still the foreseen, deferred follow-up.

### Side-effecting is intrinsic op metadata

`before_writes` needs to know which operations have side effects. Unlike plan-invalidation
semantics, **read-only-vs-write is a property of the operation itself** — exactly what a tool
author/adapter knows — and it is already standardized: MCP's `readOnlyHint`/`destructiveHint`/
`idempotentHint` tool annotations; ARE ops are plainly writes (`add_calendar_event`, `send_email`)
vs reads (`list`, `search`). Home: `OperationSpecification.side_effecting`, adapter-filled from those
hints or inferred, **unknown → treated as a write** (conservative: reconsider before anything that
might have a side effect). This is a sibling of `completion_signal`, not the cross-cutting
plan-knowledge the rejected option (a) demanded.

The question the flag answers is **would committing this step against a stale plan do damage**. Reporting
to the principal on the agent's own channel (runtime-io's `send_message_to_user`) moves nothing in the
environment and is nonetheless `side_effecting=True`, because delivery is irreversible and, for a plan whose
deliverable *is* the message — answering a question, reporting a decision — a follow-up that landed mid-run
makes the pending text wrong. That such a report is typically a plan's *last* step is the reason to check it
rather than a reason to skip it: nothing checks it afterwards, so skipping leaves that plan with no
checkpoint at all, which is the failure this ADR exists to prevent. The churn objection is real but
bounded by tier 1: the change gate skips the re-check for free whenever nothing observable moved, so a
status report in a static world still costs nothing. A spec cannot distinguish the trailing-status case
from the answer-shaped one, so the conservative reading is the only safe one.

### What an "invalidated" verdict discards: the whole activity, not the stale frame

`reset_for_replan()` clears `plan`, `step_index` **and `parent_frames`** — the entire intention stack
([ADR-0022](0022-plan-representation-context-guard-and-subgoals.md)) — then re-infers one plan against
`Activity.goal`. A discard taken three frames deep therefore throws away every parent's decomposition,
each of which cost its own `_infer_` call. That is deliberate, and the frame-local alternative
(pop only to the frame the drift invalidated, keep the parents, re-infer that frame against its own
sub-goal) is **rejected**:

* **Staleness is not a contiguous suffix of the stack.** A single "stale from depth N" answer assumes
  it is. Ambient drift more often invalidates *purpose* than *mechanics*: Alice moving the meeting
  makes the root's "create the event Monday" stale while the leaf's "find a free slot" is still good
  work. A root-stale/leaf-fine verdict collapses back to a full discard anyway, and
  root-stale/mid-fine/leaf-stale cannot be expressed at all.
* **Run state has no frame ownership.** `bindings` ([ADR-0023](0023-structured-value-data-ops.md)) and
  `history` are flat on the `Activity`. Keeping a parent frame means a surviving parent step can read
  a `{"$bind": ...}` value produced by the sub-plan just discarded, or ground against a history whose
  sub-goal-produced tail no longer matches what the re-inferred frame will produce. Clearing them
  instead breaks parent steps that legitimately depend on values bound before the sub-goal was
  entered. Sound frame-local replanning needs frame-scoped ownership of both — a change to
  [ADR-0023](0023-structured-value-data-ops.md), not a tweak here.
* **The frame's goal is itself suspect.** A sub-plan is re-inferred against the goal string in the
  parent's `subgoal` step — authored by pre-drift reasoning. "Find Alice's cheapest Monday option" is
  a stale question after Alice moves, however good the new sub-plan answering it is.
* **The index error is asymmetric.** Too shallow over-discards (harmless); too deep keeps a stale
  parent and *acts on it* — the silent wrong side effect this ADR exists to prevent — and a model
  biased toward minimal change errs that way. Blank-slate has no such failure mode.

**What recovers the cost instead: the replan sees the plan it replaces.** `reset_for_replan()` parks
the discard as a `SupersededPlan` (active frame + `step_index` + suspended parents) on
`Activity.superseded`, and the planning prompt renders its **un-run tail**, flattened across the stack
by the same `remaining_steps()` helper the revalidation uses. Framed as reusable material, not as a
negative example — told the old plan was *wrong*, a planner discards the parts that were fine too. The
planner then decides what still applies, which it does far better than a depth index could, and which
needs no frame ownership, no fallible frame index, and no partially-trusted stack. Observe clears the
bundle when the replacement installs, and a cached-plan reuse clears it too (it installs a plan without
spending an inference to consume it), so it is read by at most one inference. A *sub-goal* inference
never sees it at all — `_infer_` drops it from the sub-goal's activity copy, since the bundle describes
a plan for the activity's goal and the prompt introduces it as such, which would be a false statement
about a sub-goal that was never planned, let alone abandoned.

This is the same principle as rendering `history` into the revalidation prompt above, applied to the
other half of the loop: the re-check sees the work already done, and the re-infer that follows an
invalid verdict sees the intent it is replacing. Only the un-run tail is rendered — what already ran
reaches the same prompt as history, and repeating it as *intent* alongside its *results* is redundant
weight.

One asymmetry is left standing on purpose: entering a sub-goal re-anchors `reconsider_baseline` to the
sub-plan's infer-time world, but *popping* back to a parent does not restore that parent's baseline —
it resumes carrying the sub-goal's. That is conservative in the safe direction (drift during the
sub-goal makes the parent's next write revalidate rather than skip), so it stands.

### Placement, seam, and the reliability escape hatch

* **Reason** owns the checkpoint check (it already owns planning and invalidation), so
  [ADR-0022](0022-plan-representation-context-guard-and-subgoals.md)'s "reconsideration is
  cycle-owned, never a plan keyword" boundary is *restored*, not softened.
* The reconsideration policy is a **Protocol** (protocol-over-inheritance), wired via
  `agent.yaml`'s `strategies.context_adaptation`. **Per-agent now; per-activity later** is deferred
  (a delegated/background sub-task may commit differently from its parent).
* The **post-completion** "did the world change such that my finished plan is now wrong" check is
  **Reflect's** job — Reflect already owns completion judgment — not a `context_adaptation` tier.
* The model judgment is **best-effort, not a guarantee** (same reliability class as the planner). For
  high-stakes reliability, an **optional deterministic override** remains available: an *app-level*
  reconsideration trigger (the ARE example's `MailDiffInterruptPolicy`, relocated as scenario
  code/config — **not** manual-authored) can force a re-plan without waiting on the revalidation. Opt-in,
  not the default.

### The ARE scenario as a demonstrable knob

Same scenario, one setting: `context_adaptation: none` schedules Monday (fast, wrong);
`before_writes` runs the gate+revalidation at the `add_calendar_event` commitment point, catches Alice's
follow-up, and re-plans to Tuesday **before creating the wrong event** — one revalidation plus one re-infer.
That is the BDI speed-vs-adaptiveness trade-off shown directly, with the decision cycle (S-ORA's
thesis) at the centre rather than a bespoke signal handler.

### Positive Consequences

* No domain authoring of invalidation semantics; the general revalidation needs neither manuals, efference,
  nor identity paths. Restores [ADR-0022](0022-plan-representation-context-guard-and-subgoals.md)'s
  cycle-owned-reconsideration boundary.
* Economy preserved: the mechanical gate makes a static world free even at the most cautious level;
  the model spend tracks *risk* (commitment points), not time.
* Self-caused changes are handled correctly (revalidation decides) without a self-write loop and without an
  efference mechanism we could not build reliably in general.
* One config dial spans blind → cautious; it is the operationalization of BDI commitment strength,
  and the ARE scenario turns it into a visible knob.

### Negative Consequences

* The revalidation is best-effort — it can miss a subtle invalidation or over-fire; the deterministic
  app-level override exists precisely for cases where that is unacceptable.
* `before_writes` assumes side-effecting ops are sparse commitment points; a dense-write/embodied
  tool degenerates it toward a per-cycle revalidation (author-declared checkpoints, deferred, are the
  planned generalization).
* A static setting approximates an unknown, possibly time-varying world-change rate (a policy that
  adapts the commitment level to the observed world-change rate is the planned refinement, deferred).
* New machinery: a pluggable reconsideration-policy seam, a pluggable `ChangeGate` seam
  (`strategies.change_gate`, default `PerceptionSignatureGate`), an
  `OperationSpecification.side_effecting` flag, and a per-plan baseline snapshot on the
  `Activity`/`PendingInference` for the change-gate.
* Invalidation is coarse: a drift affecting one sub-goal still re-plans the whole activity. The
  superseded-plan context makes that cheap to recover from rather than free to avoid — the re-infer
  is one call, but it is still a call, and the planner may legitimately choose a different
  decomposition than the one it was shown.
* `Activity.superseded` is one more piece of transient, prompt-facing run state whose lifetime spans a
  reset (parked by `reset_for_replan()`, cleared by Observe on install) — a wider window than
  `grounded_params` or `reconsider_verdict`, and one a new invalidation path could forget to close.
* Making the user-reply channel a checkpoint costs a revalidation before the average plan's last step
  whenever the world moved during the run, including the many cases where the report was fine. That is
  paid to protect the answer-shaped plans, which an operation spec cannot tell apart from status ones.
* The history windows are fixed constants, not adaptive to model context or result size: a long plan of
  large results can still be heavy under the window, and a plan longer than the planning window loses
  its oldest entries. Each is a one-constant change, but neither is tuned per deployment.

## Pros and Cons of the Options

### (a) Mechanical maintenance predicates in the guard/manual

* Good, because the steady state is a mechanical comparison (no per-cycle model call), and a precise
  predicate is deterministic.
* Bad, because it demands tool authors foresee arbitrary plan-invalidation (unauthorable in
  practice), needs per-tool identity scoping or efference to avoid self-write loops, and pushes
  cross-cutting plan knowledge into tool manuals.

### (b) A model relevance-judgment every cycle

* Good, because it is fully general and needs no authoring.
* Bad, because a per-cycle model call destroys amortization; in a dense-write/embodied setting it is
  a continuous per-cycle cost.

### (c) Commitment as configuration: context-adaptation checkpoints (chosen)

* Good, because it is general (model revalidation, no authoring/efference), economical (mechanical gate →
  free when static; spend tracks risk), cycle-owned (restores the ADR-0022 boundary), and a tunable
  dial that directly realizes the BDI trade-off.
* Bad, because the revalidation is best-effort, `before_writes` weakens for dense-write tools (D5), and a
  static dial approximates an unknown change rate (X2).

## Links

* Honors [ADR-0022](0022-plan-representation-context-guard-and-subgoals.md) — reconsideration stays
  cycle-owned; the context guard remains **entry-only** (this ADR does not add a `maintain`
  clause). Supersedes an earlier, reverted in-place edit of ADR-0022 that had introduced a
  guard/manual maintenance predicate.
* Extends [ADR-0021](0021-llm-calls-as-async-internal-actions.md) — an invalidation re-infers
  off-cycle via `reset_for_replan()`/`pending_inference`, and an inference invalidated by a
  reconsideration is discarded on resolve (the stale-inference guard).
* Consistent with [ADR-0009](0009-five-phase-decision-cycle.md) — the check lives in Reason, one
  external action per cycle is unchanged, and the revalidation (like all model calls) runs off-cycle.
* Distinct from [ADR-0020](0020-hard-interrupt-and-await-input.md) — the hard-interrupt seam is for
  genuinely-urgent, mid-phase preemption (a user stop, a safety threshold); context-adaptation is
  non-urgent, checked at the Reason boundary. An app-level deterministic override, when used, may
  route through the interrupt seam.
* Reuses the [ADR-0023](0023-structured-value-data-ops.md) mechanical-vs-`$decide` split in spirit:
  the change-gate is the cheap mechanical tier; the relevance re-check is the escalation, throttled by
  the commitment dial rather than run every cycle.
* Deferred follow-on work: a per-activity policy override, author-declared checkpoints for
  dense-write tools, and adaptive commitment tuned to the observed world-change rate.
