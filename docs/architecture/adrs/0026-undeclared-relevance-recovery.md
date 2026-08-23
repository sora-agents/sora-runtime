# Undeclared-relevance recovery: an idle-scheduled judge over unclaimed signals, user-gated, amending rather than reopening

* Status: proposed
* Date: 2026-08-21

## Context and Problem Statement

[ADR-0022](0022-plan-representation-context-guard-and-subgoals.md) gives a plan a way to declare
what would make it relevant again, and [ADR-0019](0019-blocked-state-machinery-and-percept-storage.md)
gives the signal enough content and the wait enough scoping to make that declaration cheap to watch.
Both rest on the planner having declared the condition.

The run that motivated them declared nothing. Its synthesized plan stated three conditional clauses
in its own prose ("if he cannot make it, reschedule to the date he proposes"), encoded none of them
as structure, terminated on a confident confirmation to the user, and left the reply that arrived
minutes later with nothing to reach — 4 of 11 oracle actions. So the declared path covers the case
where the agent foresaw the branch, and the case that actually occurred is the one where it did not.
Prompt work can move that rate but cannot be relied on to reach zero, and the failure is silent: a
terminated activity with an unstated condition is indistinguishable, at termination, from one that
genuinely finished.

What — if anything — should the runtime do with an observed change that no plan is waiting for?

Two answers are already ruled out by earlier decisions. Signals do not carry the world with them
([ADR-0004](0004-tool-usage-interface.md), ADR-0019), so a change cannot be judged by reading the
signal alone. And keeping activities alive on the chance that something might happen was rejected in
ADR-0022 as the absence of a representation rather than one: at one extreme nothing ever happens and
the activity leaks, at the other every under-specified goal qualifies and nothing terminates.

## Decision Drivers

* **This path cannot assume the planner did its job**, because its entire reason to exist is the
  observation that the planner does not. A recovery mechanism premised on a declaration would
  inherit the failure it is meant to cover.
* **Cost must come out of slack, never out of the critical path.** The input here is a product of
  two unbounded sets — every observed change against every activity that ever terminated — so
  unlike a declared gate, it has no natural bound. It must never sit between a percept and an
  action.
* **Acting on an undeclared change is inventing a goal**, and inventing goals on the user's behalf
  is the user's call. The mechanism's own premise is that nobody said this mattered.
* **A closed record stays closed.** An episode is a historical claim about what was attempted and
  how it ended; editing one to make it current destroys the only account of what happened.
* **Reuse the existing seams.** No fourth memory module, no new kind of waiting, no new phase — the
  runtime already has episodic memory, an await-input path, and activity creation.

## Considered Options

* **(a) Do nothing; the declared path is the whole answer.** Good, because it costs nothing and
  keeps every judgment mechanical. Bad, because the one failure actually observed is precisely the
  undeclared case, so the runtime would ship a known hole whose only mitigation is prompt tuning —
  and the hole is silent, which is the worst property a known hole can have.
* **(b) Evaluate every change against every terminated activity, eagerly.** Good, because it misses
  nothing. Bad, because the cost is the product of two unbounded sets, paid on the critical path,
  for a hit rate that the declared path is deliberately designed to keep near zero.
* **(c) Reopen the terminated activity.** Good, because it is the smallest edit and keeps one
  activity per goal. Bad, because it rewrites a closed episode: `succeeded: true` becomes a lie in
  retrospect, the episode's step counts stop describing any real execution, and the record the
  agent learns from is no longer an account of what happened.
* **(d) An idle-scheduled judge producing a user-gated amending activity (chosen).**

## Decision Outcome

Chosen option: **(d)**, because it is the only option that covers the observed failure without
either paying for it on the critical path or letting the runtime act on a goal nobody stated.

**What it considers — the unclaimed set.** The candidate changes are exactly those that opened **no
declared gate**: ADR-0022's layer claims a signal that matched some `PendingCondition.watch`, and
what is left over is this layer's entire input. That subtraction is mechanical and free — it is the
same match already performed — and it has a property worth stating plainly: **this layer's cost
shrinks as the planner improves.** Every condition the planner learns to declare removes work from
here. The two are complements, not redundant paths, and the expensive one is the fallback.

Against that it reads episodes from `EpisodicMemory`, not live activities: episodes are durable and
survive a restart, whereas a terminated `Activity` is a run artifact.

**When it runs — scheduling, not triggering.** An unclaimed change makes the judge *eligible*; it
never makes it *run*. It runs on a tick where Situate selected nothing, or where everything
selectable is already `RUNNING` on `pending_inference` — the agent is waiting on a model either way,
so a parallel call costs latency nothing ([ADR-0021](0021-llm-calls-as-async-internal-actions.md):
inference is off-cycle, so this needs no new concurrency machinery). It never displaces an activity
that could otherwise advance. Eligibility persists — ADR-0019's retention cap is what eventually
bounds it, since an unclaimed signal survives in `wm.signals` exactly as an orphan one does, and the
judge keeps its own high-water mark over the monotonic sequence like any other waiter.

**What it produces.** One call per batch, covering all unclaimed changes and the candidate episodes
together, yielding **at most one** candidate — the episode most likely to have become relevant — and
a rationale. Not a ranked list: a second-best guess about an undeclared intention is not worth the
question it would cost.

**What happens then — the amending activity.** A candidate becomes a **new** `Activity` whose goal
amends the episode's ("*the Film Production Day this activity scheduled may need moving: Åke replied
he cannot make the 19th*"), carrying the prior episode's id. The original stays terminated and its
episode untouched, so the historical record remains true and the new work gets its own episode.

The new activity is **born `BLOCKED` on an `InputWait`** carrying the question. This is the single
construct that does both jobs — the ask and the amendment — and it reuses the await-input path
whole, following the precedent the interrupt handler and both breakers already set: assign
`blocked_on = InputWait(prompt=...)` directly rather than through `_suspend_`, which is for signal
waits. Consent then needs no bespoke evaluation, because `_resume_on_input` already clears the wait
on a user Message, calls `reset_for_replan()`, and re-infers with the reply *and* the executed
history visible — so a decline is answered by the same re-inference that a go-ahead is, and no
"interpret the user's yes/no" step is added anywhere.

**Bounding.** The candidate window is the N most recent episodes, not every episode ever stored. A
declined proposal is recorded against the `(change, episode)` pairing so the same suggestion is not
raised twice. And there is a cap on how often the layer may ask at all, because a mechanism that
interrupts the user on every stray change is worse than one that misses.

**What this needs from `EpisodicMemory`.** `consult()` retrieves by goal-equality, which is exactly
wrong here: the judge holds a change, not a goal. It needs a disambiguated sibling —
`consult_recent(limit)` — reading through `MemoryBackend.query()` with no filters. That in turn
needs something `learn()` does not store today: **an episode carries no timestamp**, so "recent" is
not currently expressible and the backend's stable key order is not recency. Both are additive (one
field, one method) and stay inside an existing memory module, per the rule that new durable data
never justifies a new memory type.

This ADR owns what to do about a change **no plan declared**. ADR-0022 owns what a plan declares it
waits for; ADR-0019 owns the signal's shape and how a wait matches one.

### Positive Consequences

* The failure actually observed has a path that does not depend on the planner having done the one
  thing it demonstrably fails to do — and the path degrades to a question rather than to silence.
* Cost is paid out of slack and bounded by a mechanical pre-filter that **shrinks as the declared
  layer improves**, so the two layers compose economically instead of duplicating each other.
* Episodes stay truthful: nothing is retroactively edited, `succeeded` keeps meaning what it said,
  and the amendment is legible as its own attempt with a pointer to what it amends.
* No new memory module, no new `blocked_on` variant, no new phase, no new concurrency machinery —
  one additive episodic field, one retrieval method, and one activity created the ordinary way.
* Consent falls out of `_resume_on_input` unchanged, so "the user said no" costs no dedicated
  mechanism and cannot drift out of sync with how every other `InputWait` resolves.
* A guess about intent is surfaced *as a guess*. The user sees the rationale and the episode it
  refers to, which is a reviewable claim rather than an action taken on their behalf.

### Negative Consequences

* **It runs least when it is needed most.** Idle-scheduling means a busy agent — precisely the one
  observing the most change — is the one that defers this longest. Accepted because the alternative
  displaces work that *is* declared, and because the declared layer is what covers the busy case;
  but it is a genuine inversion, not a neutral trade.
* **The judgment is made from a summary, not from history.** `Activity.history` is transient and
  never persisted, so an episode records the goal, outcome, plan, and last result — enough to judge
  relevance, but the amending activity starts **cold** and must rediscover state its predecessor
  already held. It will re-read what it already read once.
* **No principled setting exists for the ask-rate cap.** Too eager pesters the user until they stop
  reading; too shy reproduces the miss this layer exists to prevent. The number will be tuned by
  feel, and its cost is borne by a human rather than showing up in a metric.
* **An unrelated instruction can be captured by a pending question.** `_resume_on_input` clears
  `InputWait` on *every* blocked activity and claims the message batch as reconsideration input
  rather than letting Situate mint a new activity from it — so a user instruction arriving while an
  amending activity awaits confirmation both resumes that activity and is consumed by it, and the
  activity the instruction deserved is never created. Inherited rather than introduced here, but
  this layer makes it materially more likely by creating more waiters.
* **Nothing fires for absence.** A follow-up that never emits any signal — a reply that never comes,
  a deadline that passes quietly — produces no unclaimed change and so no candidate. Same gap the
  declared layer has, same deferred timer work closes it.
* **A change claimed by a declared gate never reaches here**, even when it is also the thing that
  should have revived some unrelated terminated activity. Accepted false negative, inherited from
  ADR-0022's subtraction.
* **The layer is inert exactly where a miss costs most.** With no user available — an unattended or
  autonomous run — the user gate has nobody to ask, and the safe degradation is to not act. So the
  setting with the least oversight is also the setting where this recovers nothing.
* **Relevance is not verifiable.** Unlike every other match in this runtime, there is no declared
  thing to compare against — the judge is asked whether an undeclared intention exists, and a wrong
  answer in either direction looks identical to a right one from the runtime's side. This is the
  least mechanically defensible of the three layers, and it is deliberately the last resort rather
  than the first.
* The candidate window (N most recent episodes) is arbitrary and has no principled value either; too
  small silently drops old-but-live commitments, too large grows the prompt and the error rate
  together.

## Pros and Cons of the Options

### (a) Do nothing

* Good, because every judgment stays mechanical and the cost is exactly zero.
* Bad, because it leaves the observed failure uncovered and silent, mitigable only by prompt tuning
  that cannot be shown to have worked.

### (b) Eager evaluation of every change against every terminated activity

* Good, because it has no false negatives and needs no scheduling policy.
* Bad, because it puts an unbounded product on the critical path to catch a case the declared layer
  is designed to make rare.

### (c) Reopen the terminated activity

* Good, because it is the smallest change and keeps one activity per goal.
* Bad, because it falsifies a closed episode, and episodic memory is what the agent learns from.

### (d) Idle-scheduled judge, user-gated, amending (chosen)

* Good, because cost comes out of slack and shrinks as the declared layer improves, the historical
  record stays intact, consent reuses the existing await-input path, and an unverifiable guess is
  surfaced as a question instead of an action.
* Bad, because it runs least when the agent is busiest, judges from a summary rather than history,
  is inert without a user to ask, and rests on a relevance judgment nothing can mechanically check.

## Links

* Complements [ADR-0022](0022-plan-representation-context-guard-and-subgoals.md) — that ADR's
  declared `PendingCondition` claims a matching change and *resumes* its activity; whatever it
  leaves unclaimed is this ADR's entire input, and this ADR *amends* rather than resumes. Declared
  resumes; undeclared amends.
* Builds on [ADR-0019](0019-blocked-state-machinery-and-percept-storage.md) — an unclaimed signal
  survives in the shared append-only log exactly as an orphan one does, bounded by the same
  retention cap, and the judge carries a per-waiter high-water mark over the same monotonic
  sequence.
* Reuses [ADR-0020](0020-hard-interrupt-and-await-input.md) — the amending activity is born
  `BLOCKED` on an `InputWait` and resolved by the existing user-Message path, following the same
  direct-assignment precedent as the interrupt handler and the deliberation breakers.
* Runs on [ADR-0021](0021-llm-calls-as-async-internal-actions.md) — the judge is an off-cycle
  inference, which is what lets it run in parallel with an activity already awaiting a model
  without any new concurrency machinery.
* Consistent with [ADR-0002](0002-activity-as-sole-first-class-construct.md) — an amendment is an
  ordinary `Activity` with an ordinary goal, not a new construct or a revived one.
* Bounded alongside [ADR-0025](0025-deliberation-breakers.md) — an amending activity is newly
  created rather than replanned, so it does not accrue to `Activity.replan_trail`; the ask-rate cap
  and the declined-pairing record are this layer's own bounds.
