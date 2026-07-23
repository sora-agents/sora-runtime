# ARE dynamic scenario — limitations, what's example-specific, and future work

This example runs a **dynamic** ARE scenario end to end: the user asks the agent to schedule a
team sync and reply to Alice; mid-run, a follow-up email arrives that changes the answer
(Monday → Tuesday). The agent is made to reconsider through its own decision cycle rather than
finishing the stale plan.

Getting this to work end-to-end took a fair amount of **prompt tuning** and a few example-only
strategies. That effort is deliberate scaffolding around real, still-open runtime gaps — it is not
meant to look production-clean. This note records what is example-specific, what is a genuine
runtime limitation the example papers over, and which foundational extensions would replace the
scaffolding. The mechanics of *how* it works live in [`../../docs/are-dynamic-scenarios.md`](../../docs/are-dynamic-scenarios.md);
this note is the honest catalogue of seams.

Note: plan auto-caching is disabled runtime-wide (the default Reflect no longer stores completed
plans), so each run infers fresh — there is no stale plan cache to clear between runs.

---

## What is example-specific (not runtime)

Everything here lives under `examples/are_scenario/` and would not ship in the runtime:

- **`ReconcilingReasonStrategy`** — re-infers an in-flight plan when a *new inbound email* appears.
  The trigger keys on the set of **INBOX email ids** (`_inbound_email_ids`), which knows ARE's
  `EmailClientApp` state shape (`folders → INBOX → emails[*].email_id`). ARE-email-shaped.
- **`CorrectiveSituateStrategy`** — spawns a fresh corrective activity when a new inbound email
  lands *after* the goal already completed. Also keys on inbox ids, and uses a hard-coded
  `_CORRECTIVE_GOAL` string. Coordinates with Reason via a `_SEEN_INBOUND` set on
  `activity.context` so a given email is handled exactly once.
- **`reconciling_plan_prompt` / `_RECONCILE_INSTRUCTION`** — a `PlanPrompt` that appends
  dynamic-environment guidance to the default planning content: focus every tool the task changes
  (inbox *and* calendar), and reconcile against the *observed* current state (delete/update only a
  stale item you can currently see, since a step has no "skip if empty"). Written to read
  generally, but it is still hand-tuned prose selected to make this scenario behave.
- **`run.py` settle loop** — exits after a quiet window (`_SETTLE_S`) rather than on the first
  terminated activity, because a follow-up may spawn corrective work after the original completes,
  and the runner cannot tell a follow-up's `state_changed` from one the agent caused itself.

Two related changes were made in the **runtime** (not example-specific), because they are generally
correct:

- **Plan auto-caching is disabled.** The default Reflect no longer stores a completed plan to
  procedural memory, so nothing is replayed verbatim across runs (see limitation 2 for why the old
  behavior was unsound). `ProceduralMemory.store`/`retrieve` remain as latent capability for a future
  experience-distillation step.
- **Identifiers stay references.** The core planner prompt (`PLAN_SYSTEM_PROMPT` in
  `src/sora/memory.py`) tells the model to keep a volatile identifier (an email/event id) as a
  `$from` reference even when it is visible in observations, instead of hard-coding it — robust
  binding on its own merit (and it was what made a cached plan run-coupled, before caching was
  disabled).

---

## Current limitations

These are the seams the example works around. Each is a real runtime gap, not a bug in the example.

1. **No conditional execution / plan branching.** A planned step always runs; there is no
   "skip if empty" and no branch. This is why a blind `delete_calendar_event` crashed when no stale
   event existed, and why reconciliation had to be pushed into *observation-time judgement* in the
   prompt ("delete only what you can currently see"). That mitigation is **probabilistic** — it
   relies on the model omitting the delete when it observes nothing to delete.

2. **Auto-caching the corrected plan (resolved — plan caching is now disabled).** Previously, on
   completion `DefaultReflectStrategy` stored `activity.plan`; after a mid-flight re-inference that
   was the *corrected* plan (with a `delete`/reconcile step), stored under the original goal. Two
   problems: (a) it was the **uncommon** case — the common case (no follow-up: focus → search → add →
   reply) is abandoned before completing on a dynamic run, so it never got cached; (b) a corrected
   plan is replayed **verbatim** by the retrieve path, so from a clean slate its unconditional
   `delete {event_id: $from search_events}` hits an empty search → `None` → a crash. The correction
   is **experience, not a reusable procedure**. This is now avoided by **disabling plan auto-caching
   runtime-wide** — the default Reflect records only an episode (`episodic.learn`). Safely *reusing*
   a distilled common-case procedure is future work (see "episodic → procedural consolidation").

3. **Self-caused state changes are indistinguishable from external ones.** The agent writes to the
   very tool it observes; every write emits `state_changed`. A naive "a signal arrived → re-infer"
   trigger therefore loops forever (reply → signal → re-plan → reply → …). The example sidesteps
   this by keying on **INBOX ids** — the agent's reply lands in SENT, invisible to the trigger — but
   that is ARE-email-shaped, not general.

4. **Observation requires focus, and focus is model-driven.** Observable properties are snapshotted
   only for *focused* tools, so the plan must explicitly `focus` every tool it reconciles against.
   `focus` is an ordinary plan step the base planner treats as optional, so we lean on the prompt to
   request it. If the model omits a focus, the dynamic behavior silently never triggers.

5. **Observation-aware inference can bake run-specific literals.** A plan inferred while a tool is
   focused may hard-code a visible id; mitigated by the core-prompt "keep identifiers as references"
   change. This mattered most for cross-run reuse — now moot, since plans aren't cached — but it
   remains good within-run hygiene.

6. **Blind commitment; no reconsideration policy.** The agent has no BDI-style commitment strategy.
   "Re-infer on every new inbound email" is a blunt stand-in for intention reconsideration, with no
   notion of an intention being blocked vs. impossible vs. superseded.

---

## Foundational extensions (discussed, deferred)

Each of these would replace a chunk of the scaffolding above with a principled mechanism.

- **Conditional / guarded plan steps** — or, minimally, a runtime rule that *skips* an `invoke`
  whose required parameter resolves to `null` instead of calling the operation with `None`. This
  gives plans lightweight "act if it exists" conditionality and removes the blind-delete fragility
  **deterministically**, retiring limitation (1).

- **Episodic → procedural consolidation (procedural learning).** The near-term step is already
  done — plan auto-caching is disabled, so nothing unsound is reused, and every activity infers
  fresh. The consolidation pass is the *re-enable*: distil a reusable **common-case** procedure from
  accumulated episodes and store it deliberately (via `ProceduralMemory.store`, which Reason's
  dormant `retrieve` path will then serve) — rather than caching whatever plan happened to complete
  last. Restores safe reuse, the open half of limitation (2).

- **Efference / read-write tags.** Tag state changes the agent itself caused so *any* self-write is
  filtered from triggers regardless of tool, generalizing the INBOX-id trick. Retires limitation (3).

- **BDI-style commitment & reconsideration policies** (single-minded / open-minded), with a real
  intention lifecycle (blocked / impossible / superseded). Replaces "re-infer on every signal" with
  a principled decision about *when* to reconsider a plan. Retires limitation (6).

- **Hard-interrupt preemption** — a pushed signal preempting the current decision phase (via
  `DecisionCycle.interrupt()`), so the agent can react to a follow-up mid-phase rather than only at a
  cycle boundary. Complements, rather than replaces, the reconsideration policy above.
