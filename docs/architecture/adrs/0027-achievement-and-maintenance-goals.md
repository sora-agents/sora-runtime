# Achievement and maintenance sub-goals: completion criterion, frame lifetime, and the environment clock

* Status: proposed
* Date: 2026-08-27

## Context and Problem Statement

The runtime knows exactly one way for a sub-goal to be done: its plan ran out of steps. That is an
**achievement** goal — it names a state of the world, its plan is an attempt at reaching it, and
when the attempt is spent the frame pops and the parent resumes.

The Gaia2 "Time" scenario (run `aug26-run6`) planned something else with the same machinery: *watch
the calendar for the next four minutes and, whenever events are added, delete the preexisting events
they overlap.* That goal is not done when its steps run out — its steps were the **first iteration**.
It is done when its `until` says so. BDI has had the vocabulary for this since AgentSpeak: an
achievement goal (`!g`) versus a **maintenance** goal, which persists as long as its maintained
condition is relevant. The runtime was missing the concept, not in need of a new one.

Reading a maintenance goal with the achievement rule produced three coupled failures on that run:

* the sub-plan exhausted its four deletes and **popped**, resuming the parent at the step that told
  the user everything was done — while the monitoring window was still open;
* the condition's `then` ("delete every overlapping preexisting calendar event") restates the goal
  that declared it, so it scored 0.94 against
  [ADR-0025](0025-deliberation-breakers.md)'s 0.7 overlap breaker and was refused as a
  non-reducing sub-goal;
* the activity blocked permanently after 1 of the oracle's 5 deletes.

The planner itself signalled the gap: on that run it invented a `deadline` field on the sub-goal
step (`"deadline": {"$decide": "four minutes after the get_current_time result"}`) that is
documented nowhere and read by nothing in `src/sora/`. It reached for a sub-goal-level termination
criterion the plan language gave it no way to express.

A first fix shipped ahead of this ADR as a stopgap: *no frame pops while its conditions are live*.
That is a heuristic standing in for the missing distinction, and it is wrong in ADR-0022's own
motivating direction — a contingency condition (send the mail, watch for the reply) is explicitly
meant to outlive the frame that declared it.

## Decision Drivers

* The two goal shapes differ in **exactly one** place — what counts as done. Planning, frames,
  sub-goal recursion and pending conditions should stay shared, not fork.
* Which kind a sub-goal is must be **declared and mechanical**, never re-judged by a model per tick.
* [ADR-0002](0002-activity-as-sole-first-class-construct.md) rules out explicit BDI states on
  `Activity`. A goal *kind* must not smuggle one in.
* A maintenance goal that cannot terminate is worse than none: it holds its frame, and its parent's
  remaining steps, forever. Retirement is **required machinery**, not an optimization.
* An `until` must not cost a model call per tick — the reason `_eligible_conditions` exists.
* Time must not be bought out of [ADR-0009](0009-five-phase-decision-cycle.md)'s one-external-action
  budget.
* Domain time and infrastructure time are different clocks and must not merge (see *The clock*).

## Considered Options

* **(a) One goal kind; live conditions hold the frame** (the shipped stopgap).
* **(b) Infer the kind** from plan shape — a sub-goal that declares a `pending` with an `until` is
  maintenance.
* **(c) Declare the kind on the sub-goal step** (chosen).
* **(d) Model maintenance as an activity-level loop** — a plan construct that re-enters its own body.

## Decision Outcome

Chosen option: **(c) declare the kind on the sub-goal step**, because it puts a one-bit distinction
in the one place that already knows it — the planner, at the moment it writes the goal — and leaves
every other mechanism untouched. (b) is the same distinction read from a proxy that is wrong in both
directions: an achievement goal may legitimately declare a contingency `pending`, and a maintenance
goal's window may be bounded by something other than a condition. (d) reintroduces control flow into
the plan language, which the sub-goal mechanism exists precisely to avoid.

### 1. The declaration

`subgoal` steps gain `goal_kind: "achievement" | "maintenance"`, defaulting to **`achievement`**.
Every sub-goal written before this ADR keeps today's meaning and nothing migrates.

It is orthogonal to the existing `mode` field, and the two must not be conflated: `mode`
(`deliberative` | `mechanical`) says **how the sub-plan is produced**, `goal_kind` says **when the
sub-goal is finished**. A mechanical fan-out can serve a maintenance goal — on the motivating run it
did.

### 2. Completion and frame lifetime

* **Achievement** — unchanged. Steps exhausted ⇒ the frame pops, the parent resumes at the step
  after the sub-goal. [ADR-0022](0022-plan-representation-context-guard-and-subgoals.md)'s
  "its frame pops long before any reply" describes exactly this and remains true *for achievement
  goals*: a contingency condition it declared is already lifted onto the `Activity` and keeps
  watching after the pop. That is the design, not a leak.
* **Maintenance** — steps exhausted ⇒ the frame **stays**, and the activity `_suspend_`s to
  `BLOCKED` on the `ConditionWait` covering its conditions' watches. The body was one iteration. The
  sub-goal completes when **every condition it declared has retired**; then the frame pops and the
  parent resumes.

This replaces the stopgap. Under it, only a *maintenance* frame is held by its own conditions.

One part of the stopgap is **not** a goal-kind question and survives for both kinds: a frame is also
held while it owes committed work — a queued `condition_fired` or an unapplied `condition_verdict`.
That is "do not let the parent run ahead of work already accepted", and it applies regardless of
what kind of goal declared it.

### 3. A fired `then` runs at the declaring depth, and is exempt from the overlap breaker

Both rules already shipped; this is where they belong.

A condition's `then` restates the goal that declared it **by construction** — the planner is
instructed to phrase it like the original goal, and for a maintenance goal it *is* the goal, once.
Its reduction lives in the data (it is planned against a change that did not exist when the ancestor
was planned), not in the wording. So a `then` is exempt from ADR-0025's ancestor token-overlap
containment. The **depth cap still applies**: the exemption removes a heuristic that is
structurally wrong here, not the breaker that bounds real recursion.

And a `then` pushes **no frame**. A watch fires as many times as the world moves, so a stack that
grew once per firing would walk a healthy monitor into the depth cap. The body is idle before a
`then` starts, so there is nothing underneath it to return to.

### 4. Retirement (garbage collection) of pending conditions

ADR-0022 named this gap in its own consequences — *"an activity's watch set grows with its sub-goal
depth, and nothing prunes it but `until`"* — and today nothing prunes it via `until` either: `until`
is judged only inside the batched condition evaluation, which only runs when an observed change
makes a condition *eligible*. A condition whose watched collection goes quiet is never re-judged and
never retires. Reachable for a contingency condition; **load-bearing** for a maintenance goal, whose
frame lives exactly as long as its `until`.

* **All `until`s, not only time-bounded ones.** A time-bounded `until` resolves **mechanically**
  against the clock, at no model cost — that is the common case and the one this must not tax. An
  event-shaped `until` ("the Film Production Day has taken place") still needs the judge, and rides
  [ADR-0026](0026-undeclared-relevance-recovery.md)'s **idle-scheduled** cadence: eligible when the
  agent is otherwise idle or already waiting on a model, never displacing an activity that could
  advance.
* **Retire only; never fire.** A pass that could fire would duplicate the eligibility gate's job and
  reopen the cost question it settled. *(Foreseen, deliberately not now: firing on the idle cadence
  would also recover a watch whose signal was never matched. Revisit only with a measurement.)*
* **Observe retires, Reason pops.** Observe drops retired conditions; the next Reason finds the
  frame no longer held and pops it. Reason owns plan advancement everywhere else and keeps owning it
  here.

### 5. The clock

An `until` is a question about **domain** time. Every `time.time()` in the runtime today
(`Percept.observed_at`, `invoked_at`, `requested_at`, an episode's `ended_at`) is **host wall-clock**
and correct as infrastructure timing. Under a simulation these are different clocks — ARE's starts
at the scenario's `start_time` *and can run at a different rate* — so the two must never be merged
into one "now". `are_sim.py` already carries a comment recording the silent wrong-answer bug from
exactly that confusion (an agent told it was 1 Jan 1970 during a 2024-10-15 scenario).

Domain time is reached through a **`DomainClock` Protocol on the workspace**, defaulting to host
wall-clock, which the ARE adapter implements off the `Environment`'s `time_manager`.

* **Per workspace, not per agent.** A simulated workspace's clock is not merely offset from
  wall-clock, it can run at a different rate, and two workspaces may legitimately disagree. An
  agent-wide clock would be wrong by construction for the first environment we run against.
* **Which clock resolves an `until`**: the workspace owning the condition's `watch.source` —
  determinate whenever a watch names a source. A maintenance goal whose `until` would require
  comparing two workspaces' clocks is a **plan defect**, refused rather than silently resolved
  against one of them.
* **An instant, not a rendering.** The Protocol returns an absolute instant; comparisons happen on
  the instant, and a timezone is for display only, never in the comparison path.
* **The bound is declared, not inferred.** `until` gains a second, optional form:
  `{"text": "<the clause>", "seconds": <how long the window lasts>}` beside the plain string it has
  always been. A plain string means **event-shaped** and reaches the retirement judge, which is both
  the safe default and every plan written before this existed; `seconds` is the planner *saying*
  that the clause is a deadline and how long the window runs, counted from the moment the wait
  begins. This is the same move `watch` makes for the signal log — *where to look* and *when to
  stop* are the only parts of a condition a protocol can answer, so they are the only parts given a
  structure. It adds no predicate DSL and no event algebra, so ADR-0022's rationale is applied here,
  not excepted.

  The alternative — recognizing the bound in the prose downstream — was implemented first and
  rejected on its failure modes rather than on its polish. A recognizer has to guess whether a
  timestamp in a sentence is the bound or a noun in it ("until the 2024-10-15T09:00Z meeting has
  been rescheduled" retires while the meeting sits unrescheduled), and whether the word "deadline"
  names one ("until the submission deadline has passed" is event-shaped, but a recognizer reading it
  as timed refuses a sound plan under §6 below). Both errors are silent and both are expensive, in
  opposite directions. The planner wrote the clause and already knows which it meant; asking it is
  strictly better information, and a wrong `seconds` is then a visible plan defect rather than an
  emergent parser behaviour.
* **Relative only.** A `seconds` window needs no clock at plan time; an absolute instant would
  require *telling the planner what time it is*, which is a second clock seam and re-baselines every
  planning prompt. A clause naming an absolute moment therefore stays event-shaped and is judged.
  Adding an `instant` field later is a one-field change if a case demands it.
* **Not observed, and not polled.** ARE's `SystemApp` keeps no time in `get_state()`
  (`get_current_time` computes from `time_manager` on call), so there is nothing for poll-on-observe
  to publish — and if there were, a monotonically-advancing property would push `state_changed` on
  **every** Observe, and [ADR-0019](0019-blocked-state-machinery-and-percept-storage.md)'s retention
  cap would evict every real signal within one cap's worth of ticks (the motivating run took ~2400).
  An agent-side poll is worse: an invoke is an external action, and one-per-cycle means a
  maintenance goal reading a clock would starve the work it exists to do.

### 6. No clock is a plan defect, and ultimately a question for the user

A maintenance goal with a time-bounded `until` — one that *declares* a `seconds` window, per §5 —
in a workspace with no domain clock, cannot
terminate — it would hold its frame silently forever. It is **refused at plan validation** with a
named defect, feeding ADR-0025's replan path; repeated refusal escalates to that ADR's shared
terminus, await-input. The agent is missing a required capability, and the honest thing is to say
so: *I cannot tell time in this environment.*

### 7. Why this does not reopen ADR-0002

`goal_kind` is a field on a **step**, part of `Plan`'s reusable skeleton. It adds no state to
`Activity`, no member to `ActivityState`, and no construct beside the activity. ADR-0002 rejected
modelling desires/intentions as first-class runtime objects with their own lifecycle; it did not
reject naming a distinction the plan language already had to express. The BDI vocabulary is adopted
here for its precision, not its machinery.

### Positive Consequences

* The three run-6 failures have one cause and one fix, rather than three heuristics.
* ADR-0022's contingency case is restored to its intended behaviour, which the stopgap had broken:
  an achievement frame pops and its condition keeps watching.
* Long-running monitors become expressible at all, which is a prerequisite for the dynamic and
  asynchronous scenarios that are the runtime's design centre — and which the static Gaia2 runs
  under-exercise.
* Retirement bounds a growth ADR-0022 flagged and left open, for both goal kinds.
* The clock seam is narrow, defaults safely, and costs no action budget.

### Negative Consequences

* **The label is a model output and nothing verifies it.** A planner that writes `achievement` for a
  maintenance goal reproduces run 6 exactly. We chose declaration over inference deliberately, so a
  mismatch is a plan defect — a mechanical cross-check (a sub-goal declaring a non-trivially-bounded
  `pending` while labelled `achievement` is suspicious) is a foreseen refinement, not a decision here.
* **A maintenance frame holds its parent's remaining steps for the whole window.** That is correct —
  the parent's next step is by construction *what to do after monitoring* — but it means a long
  `until` stalls the parent by design, and a mis-planned `until` stalls it indefinitely until
  retirement or the user intervenes.
* **A declared bound is a model output and nothing verifies it**, the same exposure as `goal_kind`
  above and mitigated the same way: a planner that writes a `seconds` for "two weeks after the
  exhibition opens" retires a live window early. The judged sweep still runs over everything not
  mechanically retired, so a *missing* bound costs only a model call; only a *wrong* one is harmful,
  and it is at least visible in the plan.
* **A defaulting clock fails silently in the direction this ADR exists to prevent.** An adapter for
  a simulated environment that does not implement `DomainClock` gets host wall-clock, which is exactly the
  1970-vs-2024 bug. Plan-time rejection catches an *absent* clock, not a *wrong* one.
* Conditions still have no bound other than `until`. An activity blocked on a condition that has
  been quiet for a very long time is indistinguishable from a healthy monitor; periodically asking
  the user to validate such activities is logged as future work, not decided here.

## Pros and Cons of the Options

### (a) One goal kind; live conditions hold the frame

* Good, because it is one predicate and fixed the observed run immediately.
* Bad, because it is wrong for ADR-0022's own motivating case — a contingency condition is *meant*
  to outlive its frame, and this pins the frame open until the reply arrives.
* Bad, because it leaves the concept unnamed, so the next symptom gets its own heuristic.

### (b) Infer the kind from plan shape

* Good, because it needs no new field and no planner cooperation.
* Bad, because the proxy is wrong in both directions: achievement goals legitimately declare
  contingency conditions, and a maintenance window need not be expressed as one.
* Bad, because an inferred kind is invisible in the plan, so a mis-inference is undebuggable.

### (c) Declare the kind on the sub-goal step (chosen)

* Good, because the distinction is stated where it is known, once, and read mechanically thereafter.
* Good, because it defaults to today's behaviour — no migration, no reinterpretation of old plans.
* Good, because it is orthogonal to `mode`, so a mechanical fan-out can serve a maintenance goal.
* Bad, because it depends on the planner labelling correctly, with nothing verifying it.

### (d) An activity-level loop construct

* Good, because iteration would become explicit and reusable beyond conditions.
* Bad, because it puts control flow back into the plan language, which sub-goals exist to avoid, and
  it needs its own termination, re-entry and interruption semantics — a much larger decision than
  the one the evidence supports.

## Links

* Refines [ADR-0022](0022-plan-representation-context-guard-and-subgoals.md): its frame-completion
  rule and its "the frame pops long before any reply" rationale are hereby scoped to **achievement**
  sub-goals, and the watch-set growth it flagged in its consequences is closed by *Retirement* above.
  Its lifting, marking and batched-evaluation decisions are not reopened.
* Refines [ADR-0025](0025-deliberation-breakers.md): a fired `then` is exempt from goal token-overlap
  containment; the depth cap is unchanged, and await-input remains the terminus a rejected
  maintenance plan escalates to.
* Depends on [ADR-0026](0026-undeclared-relevance-recovery.md) for the idle-scheduled cadence that
  event-shaped `until` retirement rides.
* Depends on [ADR-0019](0019-blocked-state-machinery-and-percept-storage.md) — `ConditionWait`,
  signal retention, and the mechanical Observe-hosted suspend/resume the maintenance block reuses.
* Constrained by [ADR-0009](0009-five-phase-decision-cycle.md): one external action per cycle is why
  the clock cannot be polled by the agent.
* Compatible with [ADR-0002](0002-activity-as-sole-first-class-construct.md) — see *Why this does
  not reopen ADR-0002*.
* Keeps model calls off-cycle per [ADR-0021](0021-llm-calls-as-async-internal-actions.md): a
  judged `until` is an off-cycle inference like any other.
