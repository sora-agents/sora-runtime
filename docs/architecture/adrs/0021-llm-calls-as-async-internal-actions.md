# LLM calls as off-cycle, deferred-result internal actions

* Status: proposed
* Date: 2026-07-31

## Context and Problem Statement

The runtime's only model calls — `ProceduralMemory.infer()` (produce a plan) and `ground()` (decide an
operation's concrete params when references can't be resolved mechanically) — ran **synchronously inside the
Reason phase**, blocking the whole `tick()` for the call's 2–20s latency. To stay reactive under that block,
[ADR-0020](0020-hard-interrupt-and-await-input.md) raced each call against the interrupt wake edge
(`abandon_on_interrupt`), abandoning it mid-flight on a hard interrupt.

This borrowed the interrupt-heavy shape of conventional LLM agent harnesses, and it fought two of S-ORA's
own commitments:

* **It contradicts the runtime's own treatment of long-latency I/O.** An external operation already runs
  *off-cycle*: `invoke` moves the activity to `RUNNING` with `Activity.pending_operation` set, the cycle
  moves on, and the result resolves later via `result_sink` (an unambiguous 1:1 match, no strategy code).
  A 20s *tool* call never blocks the cycle — but a 20s *model* call did. An LLM call is unbounded-latency
  I/O exactly like an external op; the synchronous treatment was the accident.
* **It violates the concurrency invariant.** [ADR-0009](0009-five-phase-decision-cycle.md) is built on
  "many activities pursued concurrently, at most one external action per cycle." But a blocking `infer`
  stalls the *entire* tick: while activity A spends 20s planning, activity B's ready operation waits behind
  it. The whole apparatus of racing the call against the interrupt (`abandon_on_interrupt`, the `_abandoned`
  set, the guarded mutation) existed only to soften a block that shouldn't exist.

The question: **how do we run unbounded-latency model calls without blocking the cycle** — while keeping the
one-action-per-cycle decision model, not inventing a new kind of waiting, and not making a model result a
`Percept` (which [ADR-0019](0019-blocked-state-machinery-and-percept-storage.md) forbids)?

## Decision Drivers

* Honor the multi-activity concurrency invariant (ADR-0009): one activity's deliberation must not stall
  another's action.
* Reuse the off-cycle machinery that already exists for external operations (the `RUNNING` + 1:1 resolve
  path), rather than adding a parallel waiting concept.
* A model result is neither observed environment state nor an event — it is *deliberation output*, not a
  `Percept` (ADR-0019); it must not travel the perception path or share `result_sink` with external acks.
* Keep mechanical, deterministic defaults first-class: the cheap paths (cached plan, references resolvable
  without a model) make **zero** model calls and must stay same-cycle.
* An in-flight LLM call cannot be cut mid-generation cleanly (relocated from ADR-0020's rejected
  per-phase-cancel option) — so the design must tolerate a call whose result is no longer wanted.

## Considered Options

* **Synchronous in-cycle call + mid-flight interrupt race** (the ADR-0020 status quo). Rejected: blocks the
  tick for the call's full latency, violates the concurrency invariant, and needs the whole
  `abandon_on_interrupt` apparatus just to stay reactive during a block that shouldn't exist.
* **Multi-phase fusion** — one model call serving Situate+Reason (or more) in a single round-trip. Rejected:
  a fused, activity-selecting call *re-serializes* the concurrency async buys (no activity is selected until
  it returns, so the cycle advances nothing meanwhile), and its "5 calls → 1" motivation presumes a model
  call in every phase — which the mechanical defaults make unnecessary, leaving only infer and ground (see
  [ADR-0011](0011-phase-fusion-via-threaded-result.md), whose threaded result is a field-gating seam, not a
  fusion mechanism). The residual optimization survives only as a fully-synchronous simple-mode
  configuration, mutually exclusive with the concurrent-async default.
* **Cancel the model task on interrupt.** Rejected: an LLM call can't be cut mid-generation cleanly;
  cancelling leaves a torn HTTP exchange. The runtime instead lets the call finish in the background and
  **discards its result on resolve**.
* **Model calls as off-cycle, deferred-result internal actions (chosen).**

## Decision Outcome

Chosen: **every model call runs as an asynchronous internal action that does not block the cycle.** A phase
strategy dispatches the call as a background task and returns immediately; the cycle keeps ticking (advancing
other activities, reacting to signals) and picks up the result in a later cycle.

**Two async patterns — kept distinct** (the same discipline ADR-0019 applies to "two kinds of waiting"):

1. **Deferred-result deliberation** — `infer`, `ground`, and (foreseen) a sensory `interpret`. The activity
   moves to `RUNNING` with a new `Activity.pending_inference` marker — **reusing** the `RUNNING` state and
   its unambiguous 1:1 resolve, *not* a new state. `pending_inference` and `pending_operation` are mutually
   exclusive per activity (a cycle emits either one external action or one internal action, so an activity is
   `RUNNING` on at most one of them). The result resolves in a later cycle's Observe through a **dedicated
   reasoning-result sink** on `DecisionCycle` (a second `NotificationQueueSink`, parallel to `result_sink`) —
   **distinct from** `result_sink` and **never** a `Percept`. On resolve, Observe attaches the `Plan`
   (`infer`) or grounded params (`ground`) and transitions `RUNNING → READY`.

2. **Background writes** — Reflect's episodic→procedural consolidation, and the procedural/episodic stores
   Reflect already dispatches. These have **no waiter**: fire-and-forget `create_task`, no `RUNNING`, no
   resolve. They are off the critical path to an action and must not be routed through `pending_inference`.

**Origination phase varies; what the call produces decides its landing zone.** A sensory `interpret` feeds
*selection*, so its output is genuine perception — a `Percept` (a property or signal) on the existing
perception path, needing no new `kind` (ADR-0019). `infer`/`ground` produce a plan/params — deliberation
output on the reasoning-result sink. Grounding stays a *Reason* act ([ADR-0017](0017-parameter-grounding-in-reason.md)),
now async; only the escalation (a reference the deterministic resolver can't bind) waits — the mechanical
resolve stays same-cycle. `interpret` is documented here as the pattern's natural extension but is not built
now; a plan-relevant image is better handled by a multimodal `infer` than a separate call.

**Stale-result reconciliation replaces mid-flight abandonment.** Because a deliberation now spans cycles,
observations can change under an in-flight call (a follow-up email, a resolved op). A result resolves **only**
against the activity's live `pending_inference`; if an `InterruptHandler` (or any strategy) re-routed or
invalidated the activity in the meantime, the resolve **discards** the result. This is modeled exactly on the
external-op late-ack guard retained from ADR-0020 (a late ack never resurrects an activity a hard interrupt
already routed away — the `state is RUNNING` reconciliation). One mechanism now covers both external acks and
inference results. The background call still runs to completion (it can't be cut mid-generation); only its
result is dropped.

**What this supersedes in ADR-0020.** The mid-flight-abandonment machinery — `abandon_on_interrupt`, the
`_abandoned` set, and "race the model call against the interrupt" — **retires**: with no synchronous model
call, the cycle never blocks on one, so there is nothing to race and the 10ms reactive target is met by the
phase-boundary checkpoints alone. **Retained from ADR-0020**, unchanged: the phase-boundary checkpoints,
`interrupt()` / the CLI user stop, the push-time `InterruptPolicy` seam, await-input / `InputWait`, and the
external-op invariant (a dispatched op is never abandoned; the interrupt is honored at the next checkpoint
after its ack resolves).

**Backstop.** This ADR is also the record that async-internal-actions were chosen *over* fusion and over the
synchronous+race design — so neither is re-adopted without new information (e.g. a workload where a single
fused round-trip demonstrably beats the concurrency loss).

### Positive Consequences

* The cycle stays responsive *by construction* — it never sits inside a model call — meeting ADR-0020's
  reactive target without racing anything.
* The concurrency invariant holds: activity B advances while activity A is `RUNNING` on an inference.
* `DecisionCycle` simplifies — the entire mid-flight-race apparatus is deleted, not relocated.
* A single reconciliation (the `RUNNING` resolve) protects both external ops and inferences against stale
  application after a re-route.

### Negative Consequences

* A plan or a grounded param lands a cycle (or more) later than the request — extra latency on the
  escalation path. The mechanical cheap paths are unaffected (still same-cycle, no model call).
* `RUNNING` now carries two mutually-exclusive pending kinds (`pending_operation`, `pending_inference`); the
  resolve path discriminates.
* A background model call whose result is later discarded still consumed its tokens and latency — the price
  of not being able to cancel mid-generation.
* Tests that asserted "the plan exists immediately after the Reason tick" must now pump cycles until the
  reasoning-result sink resolves (kept deterministic by a fake `LLMClient` that resolves on the next drain).

## Links

* Supersedes, in part, [ADR-0020](0020-hard-interrupt-and-await-input.md): the mid-flight-abandonment of the
  Reason model call (`abandon_on_interrupt`) retires; the interrupt/checkpoint/await-input/policy machinery
  and the external-op invariant are retained. ADR-0020's decision text is trimmed to its retained content.
* Refocuses [ADR-0011](0011-phase-fusion-via-threaded-result.md): the `TickResult` threading survives as the
  field-gating / short-circuit mechanism; multi-phase single-call fusion (which motivated async's rejection
  here) is retired as a goal.
* Extends [ADR-0019](0019-blocked-state-machinery-and-percept-storage.md): `pending_inference` is a second
  `RUNNING` pending kind reusing the 1:1 resolve; a model result is deliberation output, not a `Percept`.
* Keeps [ADR-0017](0017-parameter-grounding-in-reason.md) (grounding stays a Reason act, now async) and honors
  [ADR-0009](0009-five-phase-decision-cycle.md) (the concurrency invariant this restores).
* Applies [ADR-0016](0016-pluggable-activity-selection.md) (the scheduler picks next while an activity is
  `RUNNING` on an inference) and [ADR-0010](0010-pluggable-phase-strategies.md) /
  [ADR-0008](0008-protocol-based-extensibility.md) (dispatch and resolution stay pluggable with mechanical
  defaults).
