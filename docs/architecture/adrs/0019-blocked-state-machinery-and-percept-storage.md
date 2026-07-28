# Blocked-state machinery: mechanical Observe-hosted suspend/resume + split percept storage

* Status: proposed
* Date: 2026-07-22

## Context and Problem Statement

The runtime resolves `RUNNING -> READY` automatically in Observe (the `result_sink` 1:1 match), but
a tool manual can require a *second* wait: a long-running operation's own invoke ack means only
"accepted", and its real completion is marked by a domain signal (robotic-arm: `move_to` -> wait for
`target_reached` before actuating the gripper — a safety interlock). Three coupled questions had to
be answered together, because the third owns the signal lifecycle the first two depend on:

1. How is a `blocked` activity entered and left, and *where* in the cycle?
2. Is matching an observed signal against a manual's wait a mechanical step or a model/strategy
   judgment?
3. Properties and signals shared one flat `WorkingMemory.perceptions: list[Percept]` with opposite
   collection semantics (property = replace-by-`(source, name)` snapshot; signal = append log). A
   side index in `DefaultObserveStrategy` reconciled them by reading `p.payload.name` off a
   `payload: Any` — a smell surfaced during the Observe snapshot work. Signal retention/eviction was
   explicitly deferred to "the blocked-state machinery" by `_filter_`/`UnfocusAction`. Should signals
   be handled as percepts or not?

## Decision Drivers

* The wait in the motivating scenario is a **safety interlock**, preventing operation in invalid
  states — so it must be enforced deterministically, not depend on a planner remembering to emit it
  (an omission would silently drop the wait).
* Keep the automatic `RUNNING -> READY` resolve **manual-agnostic**: the two waits must compose, not
  fuse.
* Reflect evaluates outcomes / stores experience, not activity states in general; Act stays mechanistic
  ([ADR-0017](0017-parameter-grounding-in-reason.md)). Neither should grow manual-interpreting suspend logic.
* The property/signal flat-list smell (a side index over a `payload: Any`) should be removed at the
  source, not masked further.

## Considered Options

* **Suspend/resume as a Situate/Reflect judgment over prose** — a strategy reads the manual's usage
  prose and decides. Rejected: non-deterministic for a safety wait, and puts manual interpretation in phases that shouldn't own it.
* **Fused resolve** — the `result_sink` loop itself branches to `BLOCKED` when the op declares a
  completion signal. Rejected: makes `running`'s resolution manual-driven and breaks the orthogonality
  between invocation competion and signals the original design fixes.
* **Layered, mechanical, Observe-hosted (chosen)** — the resolve stays manual-agnostic
  (`RUNNING -> READY` always); a *separate* pass then blocks the activity if the just-completed op
  declares a completion signal; a signal-declared structurally in the manual makes both suspend and
  resume name-equality matches. Paired with splitting the percept store.

## Decision Outcome

Chosen: **layered, mechanical, Observe-hosted suspend/resume, driven by a structured
`OperationSpecification.completion_signal`, plus a split percept store.**

**State machine.** `Act` invokes -> `RUNNING` (unchanged). Observe drains `result_sink` ->
`RUNNING -> READY`, manual-agnostic (unchanged: `last_operation`/`history` set, `pending_operation`
cleared). Then a **suspend pass**: for each just-resolved *successful* op whose manual declares a
`completion_signal`, if that signal is not already observed, call the `_suspend_` internal action
(`READY -> BLOCKED`, recording `Activity.blocked_on: SignalWait`); if it *already* arrived (it beat
the ack), stay `READY` without blocking (the two waits compose, no deadlock). Then a **resume pass**:
for each `BLOCKED` activity whose `blocked_on` matches an observed signal (name equality, plus
`source` when scoped), call `_resume_` (`BLOCKED -> READY`, clear `blocked_on`). Neither pass evicts
the matched signal — `wm.signals` is a shared, append-only log any number of activities or strategies
may still read, so the same signal instance can satisfy more than one wait; only the fixed retention
cap ever removes an entry (see Signal lifecycle). The trim runs *after* both passes each tick, so a
signal that just arrived this tick is matched before it's ever a candidate for eviction. A failed op
does not suspend (Reflect terminates it). Both passes are in `DefaultObserveStrategy` and fully
mechanical — no LLM, no planner.

**Revision (2026-07-22):** the original version of this decision evicted the matched signal on both
the early-consume and resume paths (`wm.signals.remove(...)`), treating a completion signal as a
single-consumer token implicitly scoped to the one invocation that requested it — but `SignalWait`
carries no invocation/activity identity, only `(signal_name, source)`. That conflated two distinct
things: *resolution* (this one invocation's wait is satisfied — naturally single-consumer) and
*broadcast* (this event happened — naturally multi-reader, since `wm.signals` is already a shared
log any strategy may read directly). In practice this meant the retention trim (which ran *before*
matching) could evict a needed signal the same tick it arrived, and the suspend pass's early-consume
check (which runs before the resume pass) could delete a signal a *different*, already-`BLOCKED`
activity was waiting on before the resume pass ever saw it — silent starvation, order-dependent on
which activity happened to be processed first. Both are fixed by never deleting a matched signal and
moving the retention trim to run after both passes, which also uniformly fixes the eviction-ordering
risk called out below.

**Manual field.** `OperationSpecification` gains `completion_signal: str | None`. Authors declare it
as `completes_on:` in the optional operations interface block ([ADR-0018](0018-manual-merge-policy-and-authored-interface.md));
the parser lifts it. It is **author-owned** semantics a native description can't express, so
`merge_manuals` overlays the authored `completion_signal` onto the adapter's operations (a carve-out
from "adapter owns operations").

**Signal lifecycle.** Split `WorkingMemory.perceptions` into `properties: dict[(source, name),
Percept]` (replace-by-key snapshot) and `signals: list[Percept]` (append log). `Percept` loses its
`kind` field — the store discriminates. Signals are never evicted just for satisfying a wait — a
matched signal stays in `wm.signals`, a shared broadcast log, until the fixed cap (`_SIGNAL_RETENTION`,
newest-win) evicts it, whether it was ever matched or not (an early or orphan signal survives to a
later cycle's suspend/resume the same way an unmatched one does). `_filter_`/`UnfocusAction` prune
only `properties`.

This **supersedes**, in [ADR-0009](0009-five-phase-decision-cycle.md)'s "explicit place for
suspending/resuming" and the README/CLAUDE prose, the "resume is a judgment call left to
Situate/Reflect" framing (now: mechanical, in Observe). It **refines** [ADR-0012](0012-percepts-vs-messages.md)
(the percept store split) and [ADR-0018](0018-manual-merge-policy-and-authored-interface.md)
(the `completes_on` field + merge carve-out). The manual-agnostic resolve is *preserved*, not
reverted.

### Positive Consequences

* The safety wait is enforced mechanically and can't be forgotten by a strategy.
* The two waits compose cleanly; the early-signal race is handled by check-then-block.
* The `payload: Any` side-index smell is gone: the property store is keyed directly; signals are
  their own typed list with their own lifecycle.
* Reflect and Act are untouched; Act stays mechanistic (ADR-0017 preserved).

### Negative Consequences

* `Percept` losing `kind` and `WorkingMemory` gaining two fields ripples through README/EXAMPLES's
  single-percept-stream sketches and every reader/writer (mechanical, but broad).
* The retention bound is a fixed cap, not age- or ownership-based — a signal flood could still evict
  an unconsumed completion signal before its ack, or evict a signal a still-`BLOCKED` activity needs,
  once more than `_SIGNAL_RETENTION` signals land between the activity starting to wait and the real
  event recurring. Accepted as the deliberately-simple v0.1.0 policy, revisited if a real workload's
  signal volume needs richer (age- or ownership-aware) eviction.
* Because a matched signal is never deleted, it also survives to satisfy an activity that starts
  waiting on the identical `(signal_name, source)` *after* the physical event already happened and
  was already consumed by an earlier waiter — until it ages out of the retention window. This is the
  same trade the orphan case always accepted (a signal must survive to a later suspend); it is not
  new risk, but it is now also true of matched, not just never-matched, signals.
* `SignalWait`'s `source` is always the invoking op's own `invocation.tool_id` — there is no authoring
  path for a completion signal legitimately emitted by a *different* tool than the one invoked (e.g.
  an actuator/sensor split). Such a manual would suspend correctly but could never resume: the source
  scope would never match. Accepted as a deferred limitation, same posture as the multi-waiter
  eviction cap above — revisited if a real cross-tool completion signal is needed, at which point
  `completes_on:`/`OperationSpecification.completion_signal` would need to carry an authored source,
  not just a name (a README-facing authoring-format change, not just an internal one).
* Signal matching (`_match_signal`) is exact-string, case-sensitive name equality with no fuzzy
  fallback, timeout, or stuck-activity diagnostic beyond an info-level log at suspend time — this is
  consistent with every other name-equality match in the runtime (e.g. operation-name lookup), not a
  special weakness of this mechanism, but it means an authoring typo or case mismatch between a
  manual's `completes_on:` and a tool's actually-emitted signal name silently, permanently blocks the
  activity (Situate never reselects a `BLOCKED` activity) with no signal beyond that one log line.
  Accepted as the cost of staying mechanical rather than adding heuristic matching or new diagnostic
  machinery; revisited only if silent stuck activities prove to be a real operational problem.
* A property-reaches-state completion (README's "signal *or* property update") is not yet
  implemented — `blocked_on` is named generally so a `PropertyWait` variant slots in additively.

## Links

* Refines [ADR-0012](0012-percepts-vs-messages.md) (percept storage split) and
  [ADR-0018](0018-manual-merge-policy-and-authored-interface.md) (`completion_signal` + merge carve-out)
* Realizes [ADR-0009](0009-five-phase-decision-cycle.md)'s "explicit place for suspend/resume on a
  tool signal"; preserves [ADR-0017](0017-parameter-grounding-in-reason.md) (Act stays mechanistic)
* Builds on [ADR-0011](0011-phase-fusion-via-threaded-result.md) (Observe/Reflect deterministic by
  default; fusion starts at Situate)
