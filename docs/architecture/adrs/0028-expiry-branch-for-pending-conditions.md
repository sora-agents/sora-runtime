# An expiry branch for pending conditions: the `otherwise` a window owes when nothing fires

* Status: proposed
* Date: 2026-10-04

## Context and Problem Statement

A `PendingCondition` is `watch` / `when` / `then` / `until`. `then` is reached by a **firing** — a
signal arrived and the judge agreed it was the awaited thing. `until` only stops the waiting:
retirement is documented as "the condition stops waiting", and nothing is attached to the ending. So
a plan can state *what to do when the awaited thing happens* and cannot state *what to do when it
does not*.

That second half is an ordinary reactive pattern, and a benchmark goal made it concrete: "send
individual messages to the colleagues I am meeting today asking who is supposed to order the cab.
**If after 3 minutes there is no response**, order a default cab." The agent sent the messages
correctly, then spent five restatements across a replan trying to express the rest — a maintenance
sub-goal to "wait up to 3 minutes", a condition watching for a reply, a window with a `pending`
clause — and the trace shows `condition fired: 0`, `blocked_on: 0`, `_suspend_: 0`. The window armed
and nothing ever fired, *correctly*, because no reply came. The cab was never ordered, and the two
follow-up messages confirming it were lost downstream of the same miss.

The restatements are not model confusion. They are a search for a construct that does not exist,
which is also why the attempt cost a replan instead of failing fast. How should a plan express the
branch taken when a declared window closes without the awaited event?

## Decision Drivers

* **"Act if no response arrives in time" is near the centre of what this runtime is for**, not an
  edge case. A runtime built for dynamic, asynchronous environments that can only react to things
  that *happen* is missing half of reacting.
* **The deadline is already declared and already resolved for free.** `Until.seconds` is the
  planner's declaration that a clause is a duration, and the retirement sweep closes the window
  every tick by comparison against the watched workspace's clock, at no model cost. Any solution
  that makes a deadline cost a judgement to notice is strictly worse than what already exists.
* **No new condition language.** `when`/`then` are prose precisely because their consumer is
  `_infer_`; structure appears only where a protocol can answer. A branch must not introduce a
  predicate DSL or an event algebra into the plan.
* **Act stays mechanistic, and the action space stays small.** A timer is not an external action.
* **A wrong trigger now *acts*.** Until this decision, an early retirement only stopped watching
  something. Attaching a goal to the ending means a false close takes a real action in the world, so
  the error direction has to be chosen deliberately rather than inherited.
* **The prompt baseline must not move implicitly.** A behavioural edit to the planning prompt
  invalidates comparison against every benchmark number recorded before it, so a runtime capability
  has to be able to land without teaching the planner in the same change.

## Considered Options

* **An expiry branch on `PendingCondition`** — one more prose field, `otherwise`, owed when the
  condition retires having never fired.
* **A clock-sourced condition** — synthesize clock advances as changes so a deadline can be
  `watch`ed like any other change, and the existing `then` carries the branch.
* **A timer or wait-for-duration action** — add sleep/timer to the action space and let the plan
  body express the window as steps.
* **Leave it to the planner** — a maintenance sub-goal that polls and decides for itself when the
  window is spent.

## Decision Outcome

Chosen option: **an expiry branch on `PendingCondition`**, because it is the only option that adds
the missing branch without adding a mechanism: the deadline it triggers on is already declared, is
already noticed every tick by comparison, and the branch itself is prose pursued down the path a
`then` already takes.

Concretely:

* `PendingCondition.otherwise: str | None` — prose, pursued as a deliberative sub-goal exactly as
  `then` is. Absent means today's behaviour in full: retirement stops the watching and attaches
  nothing.
* **Owed on retirement with zero firings.** A firing does not consume a condition (`until` is what
  ends it), so "the window closed" and "the awaited thing never happened" are the same observation
  until something records the difference; `PendingConditionState.ever_fired` is that record. A
  verdict that both fires and retires one condition — `{"fired": [0], "retired": [0]}`, the ordinary
  one-shot answer — owes only the `then`, which is why firings are marked *before* retirement is
  applied.
* **Owed from both retirement paths**, the clock's mechanical sweep and the batched judge's
  `retired` half. Restricting it to the clock reads safer, and is rejected below.
* **Queued onto the existing `condition_fired` queue**, carrying `on_expiry` to say which branch it
  owes. The queue already holds the declaring frame open as committed work, already keeps an
  activity whose body is finished off the termination path, already drains one at a time and only
  once the body is idle. An expiry needs all four and needs them to behave identically to a firing.
* **Decided once per waiter, and never off an unread judgement.** Three sites can close a
  window — the clock's sweep, the retirement judge's parked verdict, and the batched eligibility
  verdict — and the first two each act on a list of waiters captured *before* the call they were
  waiting on resolved. Dropping a condition deduplicates itself (removal is by identity, so a
  waiter another site already removed is simply not found), but deciding has no such natural no-op,
  so the outcome is marked on the waiter rather than inferred from the queue. What the mark records
  is the **decision, not the commitment** — a distinction that cost one defect on its own. A branch
  one site deliberately *refused* leaves nothing behind that a later site can read as a refusal:
  `ever_fired` is false because nothing fired, and the evaluation mark has advanced, so a stale
  sweep finds a quiet watch and every reason to queue. Refusals are therefore written down exactly
  as commitments are, and only deferral — the one outcome that is still pending — is left unmarked.
  A refusal is also recorded for a waiter that is **still live**, not only for one being retired:
  judging a waiter advances its evaluation cursor, and once advanced, a cursor past a change nobody
  could read is indistinguishable from a watch that nothing ever matched. A reply judged mid-window
  therefore has to leave its refusal behind for the sweep that closes that window two ticks later,
  or the sweep reads the cursor as the absence it was never told about.
  Checked against the waiter rather than against the activity's live `pending_conditions`, which
  looks equivalent but is not: the two callers sit on opposite sides of the removal, so a liveness
  filter would silently reject every waiter arriving from the verdict. Separately, a
  window that closes while a judgement over it is outstanding is not yet *known* to have never
  fired — `ever_fired` turns true only when a verdict is applied, and the clock sweep runs before
  Reason — so the branch decision is deferred to that verdict instead of guessed: it is already
  paid for, and it is the only thing that can tell "nobody replied" from "the reply is still being
  read". Both were reproduced against the first implementation of this decision; each let one
  verdict run work it had not authorized, in the near-deadline case this construct exists for.
* **Only an absence the condition can actually claim authorizes the branch.** `otherwise` is the one
  consumer in the runtime for which "no indices" is a positive authorization rather than a no-op, so
  every way of producing an empty answer becomes a way of acting. Several of them are not absences
  at all. A judgement that *errored* parks an empty verdict, and so does one whose output could not
  be read — unparseable text, a missing or non-list `fired`, a list left empty only because its
  entries were discarded. That degradation is correct fail-soft everywhere else, because an activity
  that was waiting keeps waiting, so the fix is not to stop degrading but to say what the degraded
  value *is*: `ConditionVerdict.failed` marks a verdict that is not an answer, set both by the error
  path and by the parser, and a non-answer decides nothing. A `fired` the judge answered and left
  empty stays an answer, and is the only one of these that authorizes anything. The mark has to
  suppress the branch on **every** route to it, including the one that does not look like a failure:
  `{"fired": "unknown", "retired": [0]}` has a readable `retired`, so the condition genuinely stops
  being watched — but a window that stopped being watched is only half of what the branch needs, and
  the half that failed is the other one. Valid retirement indices still retire; the branch they would
  otherwise owe does not follow from them. Only the negative branch is lost, and the waiter stays on
  watch: a judgement nobody could read has made an *absence* unavailable for good, which is the same
  asymmetry the declined-over-an-unjudged-change case takes, but it says nothing against a later
  change firing the condition outright. Separately, a window
  that closes while a matching change sits in memory **unjudged** declines outright: the gate opened,
  so an absence is the one reading no longer available. All were reproduced; each turned a flaky
  seam, a malformed reply, or a last-tick change into a cab order.
* **Seeds nothing.** A firing seeds the ids its change reported so the `then` sub-plan can name them
  mechanically. An expiry reports no changes, because nothing moved, so the reserved bindings are
  *cleared* rather than filled — handing the branch an authoritative empty set would silently make
  an `in` select nothing and a `not_in` exclude nothing.

**The branch is reachable from both retirement paths rather than the clock's alone.** The narrower
rule is tempting on the "a wrong trigger now acts" driver: a clock comparison cannot be wrong about
a declared duration, while a judge can be wrong about an event-shaped clause, and restricting the
branch would keep a false retirement as harmless as it is today. It is rejected because of what it
does when the planner writes prose instead of a number, which is the common case the bound-
declaration rule deliberately produces: a planner correctly told never to *guess* a `seconds` writes
an event-shaped string, and under the narrow rule that plan's branch becomes permanently
unreachable — the original defect, reproduced with no trace of itself. The judge's error direction
settles it: it is instructed to default to *keeping*, so its mistake is a window that stays open and
a branch that never fires, which is exactly the behaviour that preceded this decision.

### The lifecycle of an expiry decision

`otherwise` is the only consumer in the runtime for which an *empty* answer is an authorization
rather than a no-op. Before this decision, every way a condition judgement could degrade — a seam
error, a reply the parser could not read, a call that was abandoned — degraded to "nothing fired",
and nothing fired meant nothing happened. Afterwards each of those is a route to an external action,
and six review rounds each found a different one. What they had in common was not a missing guard
but a missing distinction, so it is stated here rather than left to be re-derived at each site:

> **Confirmed positive, confirmed negative, and unresolved are three different states.** Expiry must
> never turn unresolved evidence into a confirmed negative because a call finished, a cursor
> advanced, or a waiter disappeared.

Four concepts are involved and are deliberately kept apart, because every defect in this area came
from one standing in for another:

* **window closure** — has the `until` been reached (by clock comparison, or by the retirement
  judge)?
* **evidence resolution** — has every change this window *accepted* been readably judged?
* **retry allowance** — may a failed judgement be re-dispatched once?
* **terminal decision** — has this waiter's branch been decided, either way?

The decision a waiter is in, which governs the **expiry branch only**. `FIRED` and `REFUSED` are
terminal for `otherwise`; neither stops the condition watching, and a live condition may fire
repeatedly:

| State | Meaning | Terminal for `otherwise` |
|---|---|---|
| `UNDECIDED` | watching; no accepted evidence outstanding | no |
| `UNRESOLVED` | evidence accepted into the window, not yet readably judged | no |
| `FIRED` | a read verdict fired this condition | yes — no branch |
| `REFUSED` | evidence was accepted but is unresolvable | yes — no branch |
| `COMMITTED` | a read verdict established the absence; `otherwise` queued | yes — branch |

`UNRESOLVED` begins when evidence is **accepted**, not when a judgement is dispatched over it. The
interval in between is where the known last-tick race lives (see the negative consequences), so the
model has to name it rather than treat a waiter as untouched until a call goes out.

Acceptance is therefore **recorded** when a matching change is recognized, not re-derived later by
re-scanning working memory. The distinction is not bookkeeping: `WorkingMemory.signals` is a capped
log whose only eviction is that cap, so a question asked of retained memory answers "has this
evidence survived?" and not "did this window accept evidence?" — and those come apart exactly when
unrelated traffic fills the log. A reply that matched a watch and was then evicted before any
judgement ran read as a window that had been quiet, which is the authorization for `otherwise`. So
each waiter carries its own acceptance marks in the same coordinate space as its evaluation cursors,
and *unresolved* means the cursor has not reached the mark. An evicted change is still evidence;
what it is no longer is **judgeable**, and keeping those two facts separate is what turns that case
into a refusal instead of a wrong action.

Two properties of that recording are load-bearing, and each was a defect before it was a property.
It happens for **every live waiter**, in Observe, immediately before the trim — not inside the gate
checks, which look only at activities they already have business with (the eligibility sweep at ones
BLOCKED on a `ConditionWait`, the expiry sweep at windows it is closing). A READY activity is
neither, so acceptance recorded inside a gate check is conditional on scheduling, and a reply that
arrives while the agent has work in hand is recognized by nothing at all. And evidence **lost** to
the trim refuses the branch at the moment of the loss, rather than being left for the closing sweep
to infer: the evaluation cursor is a single scalar, so judging a *later* matching change advances it
past an evicted earlier one, and the acceptance mark it then meets reads as "everything accepted has
been answered". A mark can say that evidence is outstanding; only a decision can say that evidence
is gone.

Evidence that is **already under judgement** is the one case recorded rather than decided, and it
needs its own mark because the dispatch itself hides it: the cursor advances past a change when the
call goes *out*, so from that moment the entry sits below where a scan from the cursor would start.
Lose it there and a verdict that fails rewinds into a gate over a change that no longer exists —
re-judgeable in form only — and the next matching change to be judged advances the scalar cursor
past both. But the call is already paid for and may come back readable, and **a usable verdict
resolves its own evidence**: it answered the question it was asked about the change it was given,
whichever way it answered, so whether memory still holds that change decides nothing. Refusing the
branch on every eviction under judgement would therefore silence the construct's ordinary
near-deadline case. So the loss is recorded when it happens and converted into the refusal only
where a failure is resolved, which is also the only site that can tell a failure from an answer.

The invariants, which is what a review of this area should be checking rather than whether each
reported reproduction is fixed:

* **I1** `otherwise` runs at most once per waiter.
* **I2** `otherwise` never runs after a positive firing.
* **I3** A window that accepted no matching evidence expires freely and needs no judgement at all;
  every change it *did* accept must otherwise carry a usable negative answer before the branch is
  committed.
* **I4** An unresolved waiter never reaches a committed branch without an intervening usable answer.
* **I5** Only a usable answer restores the retry allowance.
* **I6** A valid reply and a genuine timeout each produce their intended action. **I1-I5 hold; I6
  does not, and that is accepted rather than outstanding** — a reply lost before anything could
  judge it (retired with its window, or evicted by retention) produces neither branch. A genuine
  timeout always produces its action; a valid reply does so unless it is lost. See the negative
  consequence below for the decision and what bounds it.

Every supported transition:

| Event | From | To |
|---|---|---|
| change matches the watch, accepted into the window | `UNDECIDED` | `UNRESOLVED` |
| judgement dispatched over accepted evidence | `UNRESOLVED` | `UNRESOLVED` (cursor advances past exactly that change) |
| verdict read, fired | `UNRESOLVED` | `FIRED` |
| verdict partly read: a firing claim is usable, another claim is not | `UNRESOLVED` | `FIRED` for the claims it answered; the rest as "verdict failed" |
| verdict read, empty, other accepted evidence still unjudged | `UNRESOLVED` | `UNRESOLVED` (allowance restored) |
| verdict read, empty, nothing unjudged, window open | `UNRESOLVED` | `UNDECIDED` (allowance restored) |
| verdict read, empty, nothing unjudged, window closed | `UNRESOLVED` | `COMMITTED` |
| verdict failed, waiter live, allowance available | `UNRESOLVED` | `UNRESOLVED` (cursor rolled back, allowance spent) |
| verdict failed, waiter's change evicted while the judgement was out | `UNRESOLVED` | `REFUSED` (ahead of liveness: a rollback cannot give back a log entry) |
| verdict failed, allowance spent or waiter already retired | `UNRESOLVED` | `REFUSED` |
| stall watchdog gives up on the call | `UNRESOLVED` | as "verdict failed" (a synthetic errored result) |
| late result after the watchdog, or a stale id | any | unchanged — discarded; the batch was already resolved by the path that preempted it |
| evaluation abandoned (an in-flight condition call discarded by a replan or an interrupt) | `UNRESOLVED` | as "verdict failed" |
| window closes with a judgement in flight | `UNRESOLVED` | deferred to that verdict |
| window closes, gate closed, nothing outstanding | `UNDECIDED` | `COMMITTED` |
| accepted evidence evicted by retention before anything judged it | `UNRESOLVED` | `REFUSED` (decided at the loss; no later judgement clears it) |
| evidence evicted by retention while a judgement over it was out | `UNRESOLVED` | `UNRESOLVED` + loss recorded (a usable verdict forgives it; a failed one refuses) |
| window closes over an unjudged change | `UNRESOLVED` | `REFUSED` |

Two rows carry most of the history. **"Verdict failed"** is one transition, applied at one site and
only after retirement has been applied, because liveness is what separates its two outcomes — a
waiter still on watch gets its change back and stays judgeable, one that can no longer be re-judged
gets a persistent refusal. It used to be two sites with two opinions: the seam-error path rolled the
cursor back under a one-retry bound, while the parse-degraded path left the cursor advanced *and*
refunded the allowance on the grounds that the seam had answered. A seam answering is not this
question being answered, so an unreadable reply consumed its change permanently and bought back a
retry it had never spent. **"Evaluation abandoned"** is distinct from the late-result discard
directly above it, and only the latter is a free no-op: a late result lands on a batch some other
path has already resolved, while an abandoned call removes the only evaluation that was ever going
to resolve one. Left unresolved, that waiter strands a window on a verdict that cannot arrive, and
the next dispatch leaves it outstanding nowhere with its cursor still advanced — at which point a
sweep reads the advance as the absence that authorizes the branch.

A failed verdict is not an *empty* one, which is the other half of that transition and was found
by reproduction rather than by reading: `{"fired": [0, 7]}` fails on an index that does not exist
while saying something perfectly usable about the one that does. The claims such an answer resolved
are resolved — giving one of them back produced a retry that asked the identical question about the
identical change, fired the same condition a second time, and dispatched the positive branch twice
off one reply. So the failure transition applies to the claims a verdict was **silent** about, never
to the ones it answered.

A rollback could not have landed without the same change to the terminal decision, which is why
they are one row and not two: Reason previously settled every judged waiter on the first failed
verdict, so giving the change back on its own would re-dispatch, receive a legitimate
answered-negative, and then find the branch already refused — suppressing a genuine timeout, the
I6 failure in the opposite direction.

### Positive Consequences

* The timeout-else pattern becomes writable, at no model cost in the common case where the planner
  declares the bound — the window still closes by comparison.
* No new state, no new wait kind, no new action, and no addition to the action space. `blocked` and
  `running` are untouched; this is a second exit from a mechanism that already exists.
* Because the parser reads named keys and ignores the rest, the runtime half lands **without
  moving any existing behaviour**: the pinned prompt digests are untouched, no built-in prompt emits
  `otherwise`, and benchmark numbers recorded before it stay comparable. That is an unchanged
  baseline, not a dormant capability — the field is live from the moment it lands, which the
  qualification below states precisely.
* An expiry and a firing are the same kind of commitment and are carried by the same queue, so the
  frame-lifetime and termination rules that govern one govern the other by construction rather than
  by a parallel implementation kept in step by hand.

### Negative Consequences

* **A retirement can now act.** The judged path inherits that, bounded only by the judge's
  keep-default. A retirement judge that is later tuned toward retiring would make this riskier, and
  that tuning is now a decision with a blast radius it did not have before.
* **One more field the planner has to be taught**, which is a prompt change, which re-baselines
  benchmark comparability when it lands. Until it does, no built-in prompt emits the field, so
  expected use is low — but low use is not the same as no effect, since the field is reachable
  without any prompt change (see the qualification below). The gap between this decision and its
  routine use is real; the gap between this decision and its being relied on is not.
* **`otherwise` is only mechanically timely when `until.seconds` is declared.** With an event-shaped
  clause the branch waits on the judge's idle sweep, so "if after 3 minutes" becomes "shortly after
  the judge next looks". The planner must declare the bound for the construct to mean what it reads
  like, and nothing in the runtime can enforce that.
* **A verdict that never lands, or lands unreadable twice, drops the branch.** Deferring the
  decision to an outstanding judgement means a verdict lost to the stale-id discard takes the
  deferred `otherwise` with it, and so does one the model returns malformed after its one retry. The degradation is to the
  behaviour that preceded this decision — a timeout that does not fire — rather than to a hang or a
  wrong action, and it is preferred to the alternative of holding the window open on a judgement
  that may never answer. Retrying is not available either: a deferred branch is reached only after
  the window closed and removed the waiter, so the marks a retry would roll back have nothing left
  to feed.
* **A change landing in a window's last tick costs the branch as well as itself.** Declining over
  an unjudged change is permanent — no evaluation is ever dispatched for a condition that has
  stopped existing — so that tick loses both branches. Making a retired condition still judgeable
  was considered and rejected here: it is a second lifecycle state for a condition, against the
  `retired_conditions` exclusion that stops a retired condition being lifted back onto watch, and
  the underlying race between retirement and eligibility predates this branch (a change arriving in
  the last tick of a window has never been judged). Deferring the *close* instead would make an
  unconditional clock exit conditional on evidence still arriving, which is the unclosable window
  this machinery refuses at plan time. So the clock keeps closing on time and the branch declines.
  Retention is a second route to the same outcome and is disclosed with it: a reply accepted into an
  open window whose log entry is evicted before the activity is free to judge it can no longer be
  judged at all, so its window is refused the moment that entry is dropped. Both are **known
  violations of I6**, and they are the only two: a reply that arrives inside its window — the case
  the construct most needs to get right — can produce neither branch. It is
  severable because it costs a no-op and never a wrong external action — which is the line the
  retention case crossed before acceptance was made durable, and the reason that one was a defect
  rather than a disclosed limitation — and they are named here rather than left implicit because
  naming a limitation is not the same as establishing the invariant. Closing the pair means
  **finite-evidence closure**: closing a window freezes the evidence it has already accepted while that judgement
  finishes, without holding the window open to new traffic — which is the second lifecycle state
  this bullet declines, scoped to evidence rather than to the condition. That is a separate decision
  and needs its own record.

  **That pair is accepted rather than closed, and this is the decision, not a deferral.** The
  guarantee this ADR makes is therefore I1-I5 plus a timeout that fires whenever the window was
  quiet — stated positively because it is what the construct is relied on for, and because the
  alternative reading ("I6 pending") invites a later reader to treat the gap as an unfinished
  task rather than a known shape. Three things decided it. The harm is bounded to the safe
  direction: both routes cost the branch and never a wrong external action, so the failure mode is
  the behaviour that preceded this construct. The reachability is measured rather than assumed —
  across a five-scenario benchmark suite the observed per-scenario volumes were 0-8 retained
  signals against a cap of 256, and 11-19 changed property receipts against a cap of 1024, so the
  retention route is one to two orders of magnitude out of reach at those volumes, leaving only the
  one-tick race between a change arriving and its window closing. And the cost of closing it falls
  on the shared condition lifecycle every `pending` condition runs through, including the ones that
  never declare a branch — a change whose defects in this family were found by review rather than
  by tests, eleven times.

  Accepted does not mean untested. Both routes are pinned as *intended* behaviour, so a later change
  cannot quietly convert them into the unsafe direction, and the two claims the acceptance itself
  makes are pinned alongside them: that unrelated traffic alone never suppresses a quiet window's
  timeout (the guarantee — an over-broad match in the retention reconciliation would silence every
  timeout in a busy environment, and passes every loss test), and that a refused window still
  releases its activity (the bound — a refusal that stranded a waiter would be a hang, which is
  strictly worse than the no-op this accepts, and is invisible to any test asserting only what was
  *not* dispatched). Reopening is a new ADR, and the evidence that would justify it is a run where
  a reply actually arrives inside a window and is lost — not the argument that it could.

  One qualification on what an unchanged prompt baseline means, since it is easy to over-read: no
  built-in prompt teaches `otherwise`, and the frozen prompt baseline is unchanged by this work —
  but neither fact disables the field. The plan parser accepts it, and an authored plan reaching the
  runtime through procedural memory can supply it without any prompt being involved. So low expected
  use is not a deactivation, and the disposition above is owed before the field can be relied on
  rather than before a prompt mentions it.
* **Still one window, one alternative.** Nothing here expresses "if no reply in 3 minutes do X, and
  if still nothing by 10 do Y" other than as two conditions.

## Pros and Cons of the Options

### An expiry branch on `PendingCondition`

* Good, because it triggers on a deadline the planner already declares and the runtime already
  evaluates every tick, so the common case costs nothing.
* Good, because the branch is prose pursued by the existing deliberative sub-goal path — no branch
  language, no predicate DSL, nothing added to the plan but a second string.
* Good, because it reuses the firing queue, inheriting frame lifetime, termination, ordering and
  idle-pursuit rules instead of re-deriving them.
* Good, because it can land without moving the prompt baseline, separating the capability from the
  prompt change that puts it into routine use and so from the re-baselining that change carries.
* Bad, because retirement acquires the power to act, which the judged path exposes to a model's
  judgement.
* Bad, because its timeliness depends on the planner declaring a bound it is otherwise free to
  write as prose.

### A clock-sourced condition

* Good, because it needs no new field: a deadline becomes a change, and the existing `then` is the
  branch.
* Good, because it generalizes — the foreseen property-reaches-state form of a wait is the same
  shape, so one mechanism would answer both.
* Bad, because it makes every deadline cost a condition judgement to notice, replacing a comparison
  that already runs free with a model call per window. That is backwards against the strongest
  driver here.
* Bad, because it requires a synthetic clock change source per workspace, which does not exist and
  whose cadence is a new question (how often does a clock "change"?) with no good answer.
* Bad, because a `when` would now be judged against the passage of time, so the judge is asked
  whether a deadline has passed — precisely the question the declared-bound rule exists to keep away
  from a model that is never told what domain time it is.

### A timer or wait-for-duration action

* Good, because it is the obvious shape and reads naturally in a plan body.
* Bad, because waiting in the body serializes the activity against a wall clock, which is what the
  decision cycle exists not to do: the agent would stop pursuing everything else for three minutes.
* Bad, because it puts a judgement about elapsed domain time into Act, which is mechanistic by
  decision, and adds an action whose effect is on the agent rather than the world.

### Leave it to the planner

* Good, because it needs no runtime change at all.
* Bad, because it is what already happens, and it does not work: the observed run produced five
  restatements, a replan, and no cab. There is no legal plan to converge on.
* Bad, because polling costs a model call per look at a window that is usually still open, which is
  the unbounded keep-alive this design rejected.

## Links

* Refines [ADR-0022](0022-plan-representation-context-guard-and-subgoals.md) — the condition's
  `watch`/`when`/`then` grammar and the rule that structure appears only where a protocol can answer
* Depends on [ADR-0027](0027-achievement-and-maintenance-goals.md) — the declared `until` bound, the
  domain clock it is resolved against, and the mechanical retirement sweep this branch hangs off
* Relates to [ADR-0019](0019-blocked-state-machinery-and-percept-storage.md) — the foreseen
  property-reaches-state form of a wait, which the clock-sourced alternative would have pulled
  forward
