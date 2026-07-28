# Hard-interrupt path: process-scheduling preemption, await-input, and a push-time InterruptPolicy seam

* Status: proposed
* Date: 2026-07-25

## Context and Problem Statement

`DecisionCycle.interrupt()` was a stub and nobody called it. Signals flowed only
*cooperatively*: a focused tool pushes to `signal_sink`, the once-per-cycle Observe drain lands it in
`WorkingMemory.signals`, and a `BLOCKED` activity waiting on it returns to `READY` — always at a cycle
boundary, never mid-phase. The README fixes one target this leaves open: reactiveness "backed by a hard
interrupt for high-priority events", "not a hard per-phase timeout, since an in-flight model call can't
be safely cut off mid-generation." Four coupled questions had to be answered together:

1. **When is a hard interrupt mandated, and by whom?** Matching any blocked wait is too broad (a robotic
   arm reaching position would preempt an unrelated running activity); letting every tool author declare
   an interrupting signal grants too much power without a trust model.
2. **How is the current phase preempted** without either losing committed external side effects or
   waiting out an unbounded model call?
3. **What happens to the interrupted activity** — does it need a new state?
4. **Where does the policy decision live** — the judgment that *this* pushed signal should preempt,
   versus flow cooperatively?

## Decision Drivers

* An interrupt is authoritative — it must be able to preempt the current phase regardless of where the
  cycle is mid-flight (the 10ms reactive target), not wait for the next boundary.
* Autonomy: interrupting the agent's own running activity is intrusive, so the authority to do it is
  narrow (a user stop now; manual-declared interrupting signals deferred behind a trust model).
* **No side effect is ever lost or double-applied.** A model call has no side effects and can be
  discarded; an already-dispatched external operation (a moving arm) must run to completion (unless an
  emergency hardware stop affordance is available).
* Keep the existing state machine and the cooperative signal path intact — the default must not change
  today's behavior or risk a self-write loop (the agent writes to the very tools it observes, so every
  write emits a signal — ADR-0019, and the ARE note's limitation 3).
* Reuse the phase seams already established (ADR-0010/0011): a new judgment is a *pluggable strategy*,
  not cycle-embedded logic.

## Considered Options

* **A new `PAUSED`/`INTERRUPTED` activity state.** Rejected: a process doesn't enter a new state when an
  interrupt arrives. By analogy, an OS *saves the execution context, runs a handler, and the scheduler
  picks what runs next*. The same shape fits here without growing the state enum (see Decision Outcome).
* **Per-phase timeout / cancel the model task outright.** Rejected: an LLM call can't be cut
  mid-generation cleanly, and cancelling a dispatched external op abandons a real side effect.
* **Interrupt authority in tool manuals (any tool declares an interrupting signal).** Deferred: too much
  power without a trust model; the wired authoritative source is a user stop.
* **Screen preemption inside Observe (or Reflect).** Rejected for the *policy* decision: it would only
  react at the next drain, defeating the point of preempting mid-phase; the screen must run at push time.
* **Process-scheduling model + push-time policy seam (chosen).**

## Decision Outcome

Chosen: **a hard interrupt modeled on process scheduling — save context, run a pluggable handler, let
the existing scheduler pick next — preempting via phase-boundary checkpoints plus true mid-flight
abandonment of the Reason model call, and screening which pushed signals preempt through a pluggable,
push-time `InterruptPolicy`. No new activity state.**

**Source & signature.** `interrupt(signal, *, target=None)` records an `InterruptRequest(signal, target)`
and sets an `asyncio.Event` (`_wake`). `signal` carries the *why* the handler reads; `target` names one
activity to preempt, `None` is agent-wide (a foreseen manual-signal interrupt could also be agent-wide or
span several activities). The one wired caller is a **user stop** — a reserved CLI control (`/stop`) that
calls `interrupt(Signal("user_stop", {}))` *directly* (a cooperative keyword would merely wait for the
next tick). This is distinct from `Agent.stop()` / Ctrl-C (graceful loop shutdown): a user stop means
"halt current work, stay alive, await instruction."

**Preemption mechanism.** `tick()` gains a **phase-boundary checkpoint** (`_preempted`) after each phase:
if an interrupt is pending, run the handler and abort the rest of the tick. The abandoned `TickResult`
carries no staleness — it never outlives one `tick()` (ADR-0011) — and Act (the cycle's single external
action) is never reached, so an interrupted tick commits nothing external. Reason is the one
unbounded-latency phase, so its **model calls** are **raced** against `_wake`
(`asyncio.wait(FIRST_COMPLETED)`) rather than merely checkpointed after: on an interrupt mid-inference the
in-flight call is **abandoned** — kept referenced in an `_abandoned` set so it finishes in the background
and its result is discarded (an LLM call can't be cut mid-generation; "let it finish, ignore the result").
The inter-tick idle wait becomes `wait_between_ticks(interval)` (a `wait_for` on `_wake`, then clear), so
an interrupt starts the next tick at once instead of sleeping out the interval.

**The race is a reusable primitive, and it races the *call* not the *phase*.** The mechanism is a
`DecisionCycle.abandon_on_interrupt(coro)` helper: it races one **side-effect-free** coroutine (the model
call) against `_wake`, returning the result normally or an `ABANDONED` sentinel on interrupt (detaching,
not cancelling — the HTTP finishes, result discarded). `DefaultReasonStrategy` wraps *just* its model
calls (`infer`, and the grounding escalation) in it and **guards every durable mutation
(`activity.plan`/`step_index`) behind a non-`ABANDONED` result**. This is deliberate on two counts.
*(1) Correctness:* an earlier design raced the whole `reason()` call, so an abandoned Reason kept running
detached and wrote `activity.plan` in the background *after* the interrupt handler had already re-routed
the activity — clobbering, e.g., a reconsideration handler that cleared the plan. Racing the pure call
and guarding the mutation closes that stale-write race. *(2) Generality:* any phase strategy that needs a
model call abandonable can call the same helper. It is **opt-in, not automatic**, because racing a phase
is only safe when its in-flight work has no durable side effects — true of Reason's model calls (pure
reads returning a value), but not of Observe/Situate/Act, which mutate working memory / activities /
dispatch external ops as they run. A phase that model-calls *without* the helper simply gets
checkpoint-after granularity (safe, just not the 10ms target); auto-wrapping every phase would trade that
latency gap for a torn-durable-state hazard.

**Invariant — abandon the model, never the external op.** A model result and the disposable `TickResult`
are the *only* things discarded. An already-dispatched external operation (side effects) is never
abandoned: a `RUNNING` activity is left `RUNNING`, its op runs to completion, and the interrupt is honored
at the next checkpoint *after* its ack resolves. The handler reports this by returning `False` (not yet
discharged) while any targeted activity is still `RUNNING`, so the request is revisited next checkpoint;
`True` once every targeted activity is routed.

**No new state — a pluggable `InterruptHandler` maps onto existing states.** The interrupted activity's
context is already saved (durable on the `Activity` dataclass — the PCB; the per-tick `TickResult` is the
disposable register state, already interrupt-immune per ADR-0011). The handler decides the follow-up,
mapping each targeted activity onto an *existing* state: `READY` (resume, or replan by clearing
`plan`/`step_index`), `BLOCKED` via a new **`InputWait`** `blocked_on` variant (await the user's next
instruction), or `TERMINATED` (drop). Then the existing `ActivitySelectionStrategy` (ADR-0016) picks
next. `DefaultInterruptHandler` is the user-stop default: pause each schedulable (`READY`) targeted
activity to `BLOCKED`/`InputWait` (halt but stay alive); a later user `Message` resumes it in Observe
(`_resume_on_input`, mirroring `_resume_on_signal` against `wm.messages`).

**`InputWait` — the second `blocked_on` variant SignalWait foresaw.** `blocked_on: SignalWait | InputWait
| None`. Same `BLOCKED` state, a different awaited stimulus: not a tool signal (nothing to name-match) but
inbound user input, so it carries only an optional human-facing prompt. Resumed by a user `Message`, not a
signal — so `_resume_on_signal` is guarded to a `SignalWait` and the resolve-loop auto-`READY` is guarded
on `state is RUNNING`, so a late ack for an activity a hard interrupt already routed away never resurrects
it.

**`InterruptPolicy` — signal preemption is *policy*, decided at push time.** `NotificationQueueSink` gains
an optional synchronous `on_push` hook, invoked before enqueue. The cycle wires
`signal_sink.on_push = self._screen_signal`, which calls `interrupt_policy.decide(source, signal, wm) ->
InterruptRequest | None`; a non-`None` result sets `_interrupt` + `_wake` the instant the signal arrives —
before the once-per-cycle Observe drain. The queue still enqueues for the cooperative drain (both paths
coexist); `result_sink` leaves `on_push` unset. The runtime default is **`NeverInterruptPolicy`** (returns
`None`): a pushed signal never preempts, so today's cooperative path is unchanged and there is no
self-write loop. Opting in is a deliberate, application-supplied policy — until read-write/efference
tagging exists, a policy can only tell an external event from the agent's own write by diffing observed
state (e.g. a set of inbox ids), which is application-shaped.

**Wiring.** `interrupt_handler` and `interrupt_policy` are new `DecisionCycle` constructor params
(defaults `DefaultInterruptHandler` / `NeverInterruptPolicy`), selected via `agent.yaml`
`strategies.interrupt` / `strategies.interrupt_policy` (mirroring the phase-strategy `import_object`
pattern — deliberately *not* folded into the five-field `Strategies` bundle, which is the decision chain).

**Example policy (ARE mail-diff).** The dynamic ARE showcase supplies `MailDiffInterruptPolicy` (diffs
INBOX email ids straight from the `state_changed` payload — fires on a genuine new inbound email, never on
the baseline, a SENT self-write, or a non-inbox signal) paired with a reconsidering `InterruptHandler`
(clears a live plan → the default Reason re-infers, or spawns one corrective activity if the goal already
completed). This centralizes reconsideration in *one* seam — the interrupt handler — replacing the
example's earlier in-Reason/in-Situate trigger strategies. Honest timing caveat: The ARE bridge emits
`state_changed` from `tool.observe()`, i.e. *during* the Observe phase (Observe-cadence, for determinism),
not off a background thread. So for the ARE sim today the policy fires inside the current tick's Observe
and aborts *that* tick's Reason/Act — there is no in-flight model call to abandon yet, and this is largely
a clean *relocation* of the INBOX-id logic into the seam rather than new timing capability. But it is the
exact architecture a genuinely asynchronous signal source (a user stop today; a future off-cycle ARE push)
reuses to abandon an in-flight inference, and it keeps the mechanism/policy split clean. The two future
unlocks are separately deferred: general **efference / read-write tagging** (retires the INBOX-id keying)
and an **off-cycle ARE push** from the Environment thread (retires this caveat → true mid-Reason
abandonment for the email scenario).

### Positive Consequences

* Authoritative preemption (10ms target) without a per-phase timeout and without a new activity state.
* No side effect is ever lost or double-applied: only a model result / disposable `TickResult` is
  discarded; a dispatched external op always runs to completion.
* The default changes nothing — `NeverInterruptPolicy` + `DefaultInterruptHandler` preserve the
  cooperative signal path and add only the user-stop capability; no self-write loop by default.
* Mechanism (interrupt/checkpoint/abandon) and policy (which signals preempt, what happens next) are
  cleanly split across two pluggable seams, reusing the ADR-0010/0011 posture.

### Negative Consequences

* For the ARE sim as-is the policy seam is mostly a relocation, not new timing benefit (Observe-cadence
  push) — the mid-generation-abandon payoff only materializes for a genuinely async producer (a user stop
  today).
* Distinguishing the agent's own writes from external events is still application-shaped (INBOX-id
  diffing) until efference tagging lands; `NeverInterruptPolicy` is the safe default precisely because no
  general test exists yet.
* A user stop pauses to `InputWait` but does not itself decide replan-vs-drop per activity — that nuance
  is the handler's, and the default keeps every paused activity resumable rather than judging intent.
* Mid-flight abandonment is **opt-in per model call** (`abandon_on_interrupt`), not a property of the
  cycle: a custom phase strategy that awaits a model call *without* the helper gets checkpoint-after
  granularity — the interrupt is honored only once that call returns, so a hung call blocks the tick. This
  is the accepted cost of not being able to safely auto-abandon phases that mutate durable state; the
  contract is documented (model-call-heavy work belongs in Reason, or must route through the helper).
* Manual-declared interrupting signals and a trust model for them remain deferred; the only wired source
  is the CLI user stop.
* `DefaultInterruptHandler` is a *user-stop* handler, not a general router: it recognizes only the
  `USER_STOP` signal and falls back to the same halt-to-await-input for any other interrupt it is handed
  (logged at warning level). That fallback is fail-safe (halt and ask a human, never barrel ahead), but a
  custom `InterruptPolicy` that raises its own signals must ship a paired handler — otherwise, headless
  (no user message to satisfy the `InputWait`), a targeted activity strands until quiescence. Dropping the
  interrupt instead was rejected as more surprising than the visible, logged fallback; a hard `USER_STOP`
  assertion was rejected as turning a recoverable misconfiguration into a crash.

## Links

* Builds on [ADR-0011](0011-phase-fusion-via-threaded-result.md) (the per-tick `TickResult` is disposable,
  so an abandoned tick carries no staleness — what makes checkpoint-and-abort safe).
* Extends [ADR-0019](0019-blocked-state-machinery-and-percept-storage.md): `InputWait` is the second
  `blocked_on` variant that ADR foresaw; the cooperative signal resume it defines is left intact, the hard
  interrupt is layered beside it.
* Uses [ADR-0016](0016-pluggable-activity-selection.md) (the scheduler picks next after the handler runs)
  and applies [ADR-0010](0010-pluggable-phase-strategies.md)/[ADR-0008](0008-protocol-based-extensibility.md)
  (both new seams are open Protocols with mechanical defaults).
* Preserves [ADR-0013](0013-shared-instances-narrow-dependencies.md) (the handler receives `(request, wm,
  cycle)`, not a whole Agent) and [ADR-0012](0012-percepts-vs-messages.md) (a user stop is a control
  signal, not a percept; the resuming user input stays a `Message` on the transport).
