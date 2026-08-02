# Hard-interrupt path: process-scheduling preemption, await-input, and a push-time InterruptPolicy seam

* Status: proposed
* Date: 2026-07-25

## Context and Problem Statement

`DecisionCycle.interrupt()` was a stub and nobody called it. Signals flowed only
*cooperatively*: a focused tool pushes to `signal_sink`, the once-per-cycle Observe drain lands it in
`WorkingMemory.signals`, and a `BLOCKED` activity waiting on it returns to `READY` — always at a cycle
boundary, never mid-phase. Reactiveness is "backed by a hard interrupt for high-priority events" —
preempting the current phase promptly, not only at the next cooperative boundary. Four coupled questions
had to be answered together:

1. **When is a hard interrupt mandated, and by whom?** Matching any blocked wait is too broad (a robotic
   arm reaching position would preempt an unrelated running activity); letting every tool author declare
   an interrupting signal grants too much power without a trust model.
2. **How is the current phase preempted** without losing committed external side effects?
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
the existing scheduler pick next — preempting via phase-boundary checkpoints, and screening which pushed
signals preempt through a pluggable, push-time `InterruptPolicy`. No new activity state.**

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
action) is never reached, so an interrupted tick commits nothing external. No phase blocks on an
unbounded-latency call: model calls run off-cycle as async internal actions
([ADR-0021](0021-llm-calls-as-async-internal-actions.md)), so the cycle is always within a bounded phase and
the checkpoint after it meets the reactive target — there is no in-phase model call to cut short. The
inter-tick idle wait is `wait_between_ticks(interval)` (a `wait_for` on `_wake`, then clear), so an interrupt
starts the next tick at once instead of sleeping out the interval.

**Invariant — never abandon a dispatched external op.** The disposable `TickResult` is the only thing an
interrupted tick discards. An already-dispatched external operation (side effects) is never abandoned: a
`RUNNING` activity is left `RUNNING`, its op runs to completion, and the interrupt is honored at the next
checkpoint *after* its ack resolves. The handler reports this by returning `False` (not yet discharged)
while any targeted activity is still `RUNNING`, so the request is revisited next checkpoint; `True` once
every targeted activity is routed. An activity `RUNNING` on an off-cycle *inference* is treated the same
way — left to resolve, its result discarded if the handler re-routed it in the meantime (the stale-result
reconciliation of [ADR-0021](0021-llm-calls-as-async-internal-actions.md), modeled on the late-ack guard
below).

**No new state — a pluggable `InterruptHandler` maps onto existing states.** The interrupted activity's
context is already saved (durable on the `Activity` dataclass — the PCB; the per-tick `TickResult` is the
disposable register state, already interrupt-immune per ADR-0011). The handler decides the follow-up,
mapping each targeted activity onto an *existing* state: `READY` (resume, or replan by clearing
`plan`/`step_index`), `BLOCKED` via a new **`InputWait`** `blocked_on` variant (await the user's next
instruction), or `TERMINATED` (drop). Then the existing `ActivitySelectionStrategy` (ADR-0016) picks
next. `DefaultInterruptHandler` is the user-stop default: pause each schedulable (`READY`) targeted
activity to `BLOCKED`/`InputWait` (halt but stay alive); a later user `Message` resumes it in Observe
(`_resume_on_input`, mirroring `_resume_on_signal` against `wm.messages`). The resume **clears the
plan** (`plan`/`step_index`) so Reason re-infers with the follow-up message and executed history
visible — a bare resume would advance the stale plan and never see the instruction. The follow-up
batch is **claimed** as reconsideration input (`messages_cursor`), so Situate does not also mint a
ghost activity from its text. Inbound messages are ambient inference context generally (rendered
into every plan prompt via `render_messages`), not only in the post-stop case.

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

**Example policy.** A concrete `InterruptPolicy` paired with a matching `InterruptHandler` exercises the seam
end-to-end: the dynamic ARE showcase supplies one that treats a genuine new inbound event as a
reconsideration trigger, centralizing reconsideration in this *one* seam rather than in bespoke
Reason/Situate strategies. The worked walkthrough — the policy's payload keying, the Observe-cadence timing
of that example's signal source, and how a reconsidering handler composes with off-cycle inference — lives
with the example (`EXAMPLES.md`, and the ARE dynamic-scenarios note
`docs/architecture/notes/are-dynamic-scenarios.md`); the requirement that such a handler invalidate an
in-flight inference is [ADR-0021](0021-llm-calls-as-async-internal-actions.md)'s.

### Positive Consequences

* Authoritative preemption (10ms target) without a per-phase timeout and without a new activity state.
* No side effect is ever lost or double-applied: only the disposable `TickResult` (and an off-cycle
  inference result the handler invalidated) is discarded; a dispatched external op always runs to completion.
* The default changes nothing — `NeverInterruptPolicy` + `DefaultInterruptHandler` preserve the
  cooperative signal path and add only the user-stop capability; no self-write loop by default.
* Mechanism (interrupt/checkpoint) and policy (which signals preempt, what happens next) are
  cleanly split across two pluggable seams, reusing the ADR-0010/0011 posture.

### Negative Consequences

* The push-time policy seam gives no mid-phase benefit over the cooperative path when the signal source only
  produces at Observe-cadence (as a synchronous, observe-driven bridge does): the interrupt is seen at the
  same drain the cooperative path would use. It earns its keep only for a genuinely off-cycle producer — and
  even then it changes only *when* the interrupt is seen, never whether an in-flight call is cut (it never is).
* Distinguishing the agent's own writes from external events is still application-shaped (INBOX-id
  diffing) until efference tagging lands; `NeverInterruptPolicy` is the safe default precisely because no
  general test exists yet.
* A user stop pauses to `InputWait` but does not itself decide replan-vs-drop per activity — that nuance
  is the handler's, and the default keeps every paused activity resumable rather than judging intent.
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
* Superseded in part by [ADR-0021](0021-llm-calls-as-async-internal-actions.md): the mid-flight abandonment
  of the Reason model call (`abandon_on_interrupt`, the `_abandoned` set) is retired now that model calls run
  off-cycle; what this ADR retains is the interrupt / checkpoint / await-input / policy machinery and the
  external-op invariant.
* Extends [ADR-0019](0019-blocked-state-machinery-and-percept-storage.md): `InputWait` is the second
  `blocked_on` variant that ADR foresaw; the cooperative signal resume it defines is left intact, the hard
  interrupt is layered beside it.
* Uses [ADR-0016](0016-pluggable-activity-selection.md) (the scheduler picks next after the handler runs)
  and applies [ADR-0010](0010-pluggable-phase-strategies.md)/[ADR-0008](0008-protocol-based-extensibility.md)
  (both new seams are open Protocols with mechanical defaults).
* Preserves [ADR-0013](0013-shared-instances-narrow-dependencies.md) (the handler receives `(request, wm,
  cycle)`, not a whole Agent) and [ADR-0012](0012-percepts-vs-messages.md) (a user stop is a control
  signal, not a percept; the resuming user input stays a `Message` on the transport).
