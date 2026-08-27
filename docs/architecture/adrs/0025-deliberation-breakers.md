# Deliberation breakers: bounding replanning, recursion, and acting on a dead plan

* Status: proposed
* Date: 2026-08-21

## Context and Problem Statement

A plan is *synthesized* by a model rather than selected from an authored library
([ADR-0022](0022-plan-representation-context-guard-and-subgoals.md)), which buys generality at the
cost of every termination guarantee a plan library gives for free. An authored library cannot
recurse forever because a human wrote the reductions; a model can satisfy "plan for goal G" with a
plan whose body is another sub-goal for ~G, or answer a rejected plan with the same plan again, and
each turn of either loop costs a full inference. Three failure modes were observed in Gaia2 runs,
all of them the same shape — the runtime spending or *acting* on deliberation that could not
succeed:

1. **Runaway sub-goal recursion.** A deliberative sub-goal restating its parent instead of reducing
   it, deferring the work one level down each time.
2. **Runaway replanning.** A plan dropped as defective, its replacement dropped for the same defect,
   and so on, with nothing ever executed. In one run five consecutive plans each re-issued
   `get_contacts(offset=0)` — a call already in history — and got no further; on a local model each
   attempt cost minutes.
3. **Acting on a plan already known to be dead.** A filter chain wrote `friend_contact = []` at step
   3 (only the first ten of 125 contacts had been read, so nobody matched); the plan carried on and
   **deleted the user's real calendar appointment** at step 5, tripping over the empty binding only
   at step 7 where it finally needed the friend's address. The user lost an appointment and got
   nothing in return. Notably the plan was *correctly ordered* — it gathered before it destroyed —
   so this is not an ordering defect.
4. **A deliberation call that fails outright killing the activity.** A 2026-08-21 adaptability run
   ended when one replan's 2712-character output — eight good steps and a well-formed pending
   condition — carried a single stray brace. The parse failed, the activity was terminated, no
   episode was written, and the user was told nothing. Unlike the three above this is not the
   runtime *spending* on hopeless deliberation; it is the runtime destroying an activity that
   nothing had gone wrong with.

The question this ADR answers: **when does the runtime stop and ask a person, instead of spending
another inference, committing another act, or giving up on the activity entirely?**

## Decision Drivers

* **The detector cannot be a model call.** Each of these failure modes *is* the model failing;
  asking it whether it is stuck would spend an inference to decide whether to stop spending
  inferences, and would ask the unreliable component to grade itself.
* **Adapting freely is the design center, not the pathology.** An agent in a dynamic environment is
  *supposed* to replan without limit. A lifetime budget on adapting would cap the thing the runtime
  exists to do, so a breaker must be relative to progress, never absolute.
* **Irreversibility is asymmetric.** Abandoning a plan that might still have worked costs one
  inference. Committing a write on a plan that cannot work costs the user something no rollback
  recovers.
* **A person can redirect what the runtime cannot.** Every one of these states is one the user can
  resolve in a sentence ("he's in my contacts under his nickname"), so the terminus should be a
  question, not a death.
* **Tunable where it is a budget; fixed where it is correctness.**

## Considered Options

* **A single inference budget per activity** — the ceiling sketched in ADR-0022, tripped by total
  spend regardless of what went wrong.
* **Per-failure-mode mechanical detectors, escalating to await-input** — several cheap, specific
  tests, all ending in the same place.
* **A model-judged "am I making progress?" check** at each replan or sub-goal entry.
* **Terminating the activity when a breaker trips**, rather than pausing it.
* For the third failure mode specifically: **reordering plans so side-effecting steps run last**, or
  **planning-prompt guidance** to gather before destroying.

## Decision Outcome

Chosen option: **per-failure-mode mechanical detectors, escalating to await-input**, because a
single budget cannot tell a productive long task from a stuck short one — which is exactly the
distinction that matters — while each specific detector can, at essentially no cost.

Five detectors across three breakers, all evaluated **before** the spend or the act they guard, none
of them involving a model call — plus a fourth failure mode (below) that has no detector because it
needs none: it announces itself by raising, and is routed into breaker 2 rather than given a budget
of its own.

### 1. Sub-goal recursion (realizes ADR-0022's deferred overflow valve)

* **Depth cap** on the intention stack (`max_subgoal_depth`, default 4).
* **Goal token-overlap containment** against the ancestor sub-goals still on the stack (0.7): a new
  sub-goal whose tokens are largely contained in an ancestor's is re-stating it, not reducing it.
  Containment (`|A∩B| / min`), not Jaccard — the observed regress *elaborates* the same goal, piling
  on qualifiers, which grows the union and sinks Jaccard while the core token set stays contained.
  One class of sub-goal is **exempt**: a pending condition's `then`, which restates the goal that
  declared it by construction and whose reduction lives in the data rather than the wording
  ([ADR-0027](0027-achievement-and-maintenance-goals.md)). The depth cap still applies to it.

ADR-0022 recorded this behavior as intent and deferred the wiring to
[ADR-0020](0020-hard-interrupt-and-await-input.md). This ADR records what was actually built, which
differs from that sketch: two specific detectors rather than a budget ceiling, and a mechanically
rendered prompt rather than a synthesized summary. ADR-0022's decision is not reopened.

### 2. Runaway replanning

Counted against `Activity.replan_trail`, which holds **only replans with no progress between them**.

* **Repeated defect** — the planner was told what was wrong and wrote it again. No third attempt
  will differ, so this trips at two.
* **Plain count** (`max_replan_attempts`, default 5) as the coarse backstop for the case where every
  attempt fails differently.

**Progress is a call this activity has not already made** — same tool, same operation, same params.
The obvious alternative, "did `history` grow", is too generous and was observed failing: five plans
each re-issued one identical `get_contacts(offset=0)`, so every replan looked like progress, the
trail cleared each time, and the breaker never came near its cap while the agent went nowhere.
Re-running a call whose arguments already appear in history yields no fact the next plan did not
already have, so it cannot be what forgives a replan. The test deliberately errs toward *not*
forgiving — re-reading state that has since changed scores as no progress even though the result may
differ — because the trail only ever counts, and what it counts toward is asking the user.

### 3. Acting on a dead plan (the irreversibility guard)

Before committing a **side-effecting** step, and only then, Reason checks whether the plan can still
finish: if a later step reads a *value* out of data that provably is not there, that step cannot
work, the plan cannot finish as written, and it is dropped with that defect instead of committing
the write. Two references carry that proof, and **both are scanned** — a `$bind` naming a binding an
earlier step produced **empty**, and a `$from` naming an operation that already ran and came back
**empty**.

The second was added 2026-08-21, after a run walked through the guard by spelling the reference the
other way: the plan invoked `search_contacts`, got `[]`, and the runtime still committed
`add_calendar_event`, creating the event with `attendees: []`. Same evidence, same asymmetry, same
proof shape — it just sat in `Activity.history` rather than `Activity.bindings`. The dead reference
was nested two levels down, inside a `$decide` element of a list parameter under a plain `from` key,
so the scan walks to any depth rather than matching only whole-value references.

"Provably" is meant strictly, or working plans would break:

* A **collection position** — a data-op `in`, a membership `where` — is exempt for either reference.
  An empty collection there is a legitimate answer ("nothing to iterate", "exclude nothing"), the
  same line [ADR-0023](0023-structured-value-data-ops.md)'s data-ops already draw between an empty
  result and an unreadable one. This is not hypothetical: "cancel all Saturday appointments" when
  there are none is exactly this shape.
* A binding a later step **rewrites**, or an operation a later step **re-invokes**, stops being
  provably empty from that point on. The second half is what keeps the guard from eating the
  recoveries replanning exists to produce: a replan's whole strategy is often to re-run the search
  by another term, and the superseded attempt's `[]` is still sitting in the history when it does.
* An operation that has **not run yet** is not evidence. Reading at step 3 what step 1 invokes is
  how every plan looks before it runs.
* A **present but mis-pathed** `$from` is not evidence either. That is a real defect but a
  recoverable one — grounding reads the actual history and routinely resolves a value whose path was
  spelled wrong — so condemning would pre-empt a repair that works. An *empty* source admits no such
  repair, since no path finds a value in it; that is the line between the two.
* A mechanical sub-goal's loop-element name is excluded for free, since it is never a stored
  binding.
* Suspended parent frames are scanned too, in resume order: a sub-plan's caller runs later and reads
  the same flat `bindings` and the same history.

This guard runs **ahead of grounding**, unlike the ADR-0024 reconsideration checkpoint it sits
beside, because it reads only settled state and costs nothing — there is no sense buying a grounding
call for a dead plan. It is deliberately **not** routed through `cycle.reconsideration`: that policy
is configurable and may legitimately be switched off, whereas refusing to act on a plan that
provably cannot work is not a tuning knob. It lives in Reason, not Act, so Act stays mechanistic
([ADR-0017](0017-parameter-grounding-in-reason.md)).

Reordering plans was rejected because the observed plan was already ordered correctly; the defect was
viability, not order, and a runtime that reorders a plan changes semantics it does not understand
(some plans delete *in order to* free a slot). Prompt-only guidance was rejected for the same reason
— guidance for a mistake the planner is not making is prompt bloat, and the planner ordered correctly
in every plan that got that far.

### 4. A deliberation call that fails outright

The three above bound deliberation that *runs* and gets nowhere. The fourth failure mode is a
deliberation call that does not run at all: `infer`/`ground` raised — malformed model output, no LLM
configured, a connection reset — and resolves with an `error` instead of a value.

That used to terminate the activity, which was the most destructive disposition available and was
never actually chosen: it was what the branch did in the absence of anything better. It is wrong on
the same driver as the other three. **Nothing was attempted** — no operation ran, the world is
untouched, and the activity has nothing wrong with it beyond one unusable response. For a failed
sub-goal it was starker still: the parent plan was sitting there intact and got destroyed with it.

So a failed `plan`/`subgoal`/`ground` inference **replans carrying the defect**, exactly as an
unresolvable grounding already did (ADR-0024). `select`/`condition`/`revalidate` keep their existing
in-place degradations (empty shortlist, nothing fired, assume valid), which are cheaper still
because they keep the plan.

The retry is **not open-ended and needs no counter of its own** — breaker 2 already is that counter.
A permanently broken call (no LLM configured) fails identically twice and trips the repeated-defect
check at two; failures that differ each time are bounded by the plain count. This is why the trail
entry is **normalized to the error's cause** rather than carrying `repr(exc)` whole: two parse
failures quote different model output, would never compare equal, and would defeat the precise check
that makes a hopeless call cost two attempts instead of five. The full message is logged.

An inference kind with no degradation of its own still terminates — but no longer *silently*, which
is the second defect this fixes. The old branch wrote **no episode** (so the failure never reached
memory, and `DefaultReflectStrategy`'s "TERMINATED was already recorded" was untrue for this path)
and **told the user nothing** (so an activity born from an instruction ended without an answer).
Both are now done before the activity is handed back, and awaited rather than dispatched: it is the
activity's last cycle, so there is no later pass on which to finish.

### The shared terminus

All of them pause the activity to **await-input** (ADR-0020) rather than terminating it, carrying a
mechanically rendered prompt: the goal, why it stopped, and the specific evidence (each abandoned
plan's defect, oldest first). No model call renders it — summarizing why the model keeps failing is
the last place to spend another inference, and the trail is already the specific, quotable evidence a
person needs in order to answer.

Setting `blocked_on` is only half of asking, and originally the only half that happened: the prompt
was stored on the activity and never delivered, so the agent stopped on a question no one could
hear — `_resume_on_input` waited for a message the user had no reason to send. The two halves are
now one call (`_await_input`), which delivers the prompt on the agent's own user channel — the same
transport `runtime-io`'s `send_message_to_user` uses, called directly because at these points there
is no plan left to route through. The hard-interrupt pause deliberately does not report: the user
caused that one and does not need to be told they did.

They compose into one escalation path rather than four parallel ones: a failed inference or the
viability guard replans; replans that learn nothing new accumulate on the trail; the trail trips the
replan breaker; the breaker asks the user.

### Positive Consequences

* A stuck agent costs a bounded number of inferences and then asks a question a person can answer in
  a sentence, instead of burning a budget silently.
* An irreversible act is never committed on the strength of a plan the runtime can already prove
  will not finish.
* Every detector is deterministic and testable without a model, and none fires in a static, healthy
  run — the cost when nothing is wrong is a set-membership test and a short list scan.
* Long, genuinely productive tasks are unaffected: every count is relative to progress or to
  containment, never to elapsed spend.

### Negative Consequences

* Five constants (4, 0.7, 2, 5, and "empty") are calibrated against observed runs, not derived. They
  are coarse backstops, and the real fix for the sub-goal case is making the common
  map/filter/distinct shapes expressible without deliberation at all — which is what ADR-0023's
  data-ops began.
* The viability guard sees only *structured* references. A `$decide` that names a binding in prose
  ("the full name of the friend_contact") is not detected; in the observed run a later `$bind`
  caught the same plan, but a plan whose only use is a prose `$decide` still reaches the write.
* The guard fires only before a write. A plan that is dead but has no remaining side-effecting step
  runs to its dereference and fails there, as before.
* Token-overlap containment is a heuristic and will occasionally refuse a legitimate sub-goal that
  is genuinely a narrower restatement of its parent. Pausing rather than terminating is what makes
  that recoverable. One such class proved *structural* rather than occasional — a pending condition's
  `then` restates its declaring goal every time, scoring 0.94 on the run that found it — and is
  exempted above per [ADR-0027](0027-achievement-and-maintenance-goals.md).

## Pros and Cons of the Options

### A single inference budget per activity

* Good, because it is one number, trivially explained, and bounds cost absolutely.
* Bad, because it cannot distinguish a productive long task from a stuck short one — the only
  distinction that matters here — so it must be set high enough to be useless as a stuck-detector,
  or low enough to kill legitimate work.
* Bad, because it says nothing about *acting*: a budget still permits the irreversible delete.

### Per-failure-mode mechanical detectors, escalating to await-input

* Good, because each detector tests the actual pathology, so it can trip early (a repeated defect
  trips at two) without threatening healthy runs.
* Good, because it costs no model call, and works precisely when the model is the failing component.
* Bad, because it is several rules rather than one, each with a calibrated constant.

### A model-judged "am I making progress?" check

* Good, because it would catch semantic non-progress a token test misses (a different call that
  learns nothing new).
* Bad, because it spends an inference to decide whether to stop spending inferences, and asks the
  component that is failing to grade its own failure.

### Terminating the activity when a breaker trips

* Good, because it is simpler and bounds cost hard.
* Bad, because every one of these states is resolvable by a sentence from the user; terminating
  discards a recoverable task and, for the runaway-replan case, throws away the work already done.

### Reordering plans so side-effecting steps run last

* Good, because it needs no run-time evidence.
* Bad, because it fixes a defect that was not present — the observed plan gathered before it
  destroyed — while changing plan semantics the runtime cannot reason about (a delete may exist
  precisely to make room for what follows).

## Links

* Realizes the deferred overflow valve of
  [ADR-0022](0022-plan-representation-context-guard-and-subgoals.md) (sub-goal recursion), without
  reopening its decision.
* Depends on [ADR-0020](0020-hard-interrupt-and-await-input.md) — await-input is the terminus every
  breaker escalates to.
* Sits beside [ADR-0024](0024-plan-reconsideration-context-adaptation.md)'s reconsideration
  checkpoint but is deliberately outside its policy seam. ADR-0024 continues to own what an
  invalidated plan discards (the whole intention stack, not the stale frame) and why.
* Reads the bindings produced by [ADR-0023](0023-structured-value-data-ops.md)'s data-ops — and the
  operation history behind a `$from` — reusing their empty-is-an-answer / unreadable-is-a-question
  distinction.
* Keeps Act mechanistic per [ADR-0017](0017-parameter-grounding-in-reason.md): the guard is a Reason
  decision.
* Refined by [ADR-0027](0027-achievement-and-maintenance-goals.md) — a pending condition's `then` is
  exempt from goal token-overlap containment (the depth cap is unchanged), and a maintenance plan
  refused for want of a domain clock escalates to this ADR's shared terminus.
