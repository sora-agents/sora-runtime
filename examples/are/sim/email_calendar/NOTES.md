# ARE dynamic scenario — limitations, what's example-specific, and future work

This example runs a **dynamic** ARE scenario end to end: the user asks the agent to schedule a
team sync and reply to Alice; mid-run, a follow-up email arrives that changes the answer
(Monday → Tuesday). The agent is made to reconsider through its own decision cycle rather than
finishing the stale plan.

Getting this to work end-to-end took a fair amount of **prompt tuning** and a few example-only
strategies. That effort is deliberate scaffolding around real, still-open runtime gaps — it is not
meant to look production-clean. This note records what is example-specific, what is a genuine
runtime limitation the example papers over, and which foundational extensions would replace the
scaffolding. The mechanics of *how* it works live in
[`../../../../docs/architecture/notes/are-dynamic-scenarios.md`](../../../../docs/architecture/notes/are-dynamic-scenarios.md);
this note is the honest catalogue of seams.

Note: plan auto-caching is disabled runtime-wide (the default Reflect no longer stores completed
plans), so each run infers fresh — there is no stale plan cache to clear between runs.

---

## What is example-specific (not runtime)

Everything here lives under `examples/are/sim/email_calendar/` and would not ship in the runtime:

- **`MailDiffInterruptPolicy`** — an `InterruptPolicy` (screened at push time) that raises a hard
  interrupt on a *genuinely new inbound email*. It diffs the set of **INBOX email ids** carried in
  the `state_changed` signal payload against what it has already seen, which knows ARE's
  `EmailClientApp` state shape (`folders → INBOX → emails[*].email_id`). ARE-email-shaped. Stateful,
  so it also dedups — each new id fires once, the first non-empty inbox is the baseline, and the
  agent's own reply (landing in SENT) never fires. See [ADR-0020](../../../../docs/architecture/adrs/0020-hard-interrupt-and-await-input.md).
- **`ReconsiderInterruptHandler`** — the paired `InterruptHandler`. On that interrupt it clears the
  in-flight activity's plan so the default Reason re-infers, or — if the email landed *after* the goal
  already completed (no live activity) — spawns one fresh corrective activity (a hard-coded
  `_CORRECTIVE_GOAL` string). A user stop is delegated to `DefaultInterruptHandler`. Reconsideration
  thus lives in *one* seam, rather than being split across bespoke Reason/Situate strategies.
- **`reconciling_plan_prompt` / `_RECONCILE_INSTRUCTION`** — a `PlanPrompt` appending
  dynamic-environment guidance to the default planning content, **split into three fragments with
  different fates**:
  - `_OBSERVE_TO_NOTICE_CHANGE` — *scaffolding* for limitation 4 (perception gated on a model-driven
    `focus`). Domain-neutral rule, email/calendar only in its examples.
  - `_RECONCILE_AGAINST_OBSERVED` — *scaffolding* for limitation 1 (no guarded/skip-if-empty step).
    Domain-neutral rule, email/calendar only in its examples. **Narrowed** by the runtime required-
    null skip (`DefaultActStrategy.bind`) but not yet droppable — it still covers the stale-but-non-
    null id a true guarded step would catch (see limitation 1).
  - `_THREAD_READING` — **not scaffolding**: email-thread domain knowledge (a follow-up is usually a
    partial correction; read every relevant message, not just the top search hit). Its proper home is
    the email-client Manual (or semantic memory), so it travels with the tool; it lives in the example
    only because the are-sim adapter synthesizes manuals and has no hand-authored one to pair
    ([ADR-0015](../../../../docs/architecture/adrs/0015-manuals-protocol-agnostic-adapter-boundary.md)).
    Relocating it is tracked below.

  The first two are candidates to *promote* (example-free) into `PLAN_SYSTEM_PROMPT` or to *retire*
  once their gaps close; all three are still hand-tuned prose selected to make this scenario behave.
- **`--exit-when-idle` quiet-window settle** — headless runs exit after a quiet window
  (`sora run --exit-when-idle N`) rather than on the first terminated activity, because a follow-up
  may spawn corrective work after the original goal completes, and the runner cannot tell a
  follow-up's `state_changed` from one the agent caused itself. (There is no bespoke `run.py` here;
  `report.py` only appends the agent-outcome and `validate()` lines to the standard trace.)

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

1. **No guarded/conditional execution (a planned step always runs).** There is no "skip if empty"
   and no branch. The *observed* failure this once produced — a blind `delete_calendar_event` on a
   stale event that didn't exist — had **two** contributing causes, **both now mitigated**, which is
   why it no longer crashes:
   - **Cross-run plan replay (resolved).** A *corrected* plan (with a `delete`/reconcile step),
     stored on completion by the old plan auto-caching, was replayed verbatim from a clean slate
     where no stale event existed. Gone — plan auto-caching is disabled runtime-wide; every activity
     infers fresh (see limitation 2).
   - **`$from`-null reaching invoke (resolved).** Independently, an unconditional
     `delete {event_id: $from search_events}` whose search returned empty used to walk the path to
     `None` and call `delete(None)`. `resolve_references` now catches a bad path (an empty-list index
     is an `IndexError`) and **escalates the param to the off-cycle `_ground_` call** instead of
     dispatching `None`.

   The deterministic guard is now in place: `DefaultActStrategy.bind` **skips an `invoke` whose
   *required* param resolves to null** (an explicit `None` or a required key the step never supplied),
   consulting the operation's adapter-synthesized schema (`OperationSpecification.parameters`) to know
   which params are required. This also closes the caveat the old text named — a genuine `None` on an
   empty path (`_walk_path(None, "") -> None`, which resolves cleanly and never escalates) is now
   skipped rather than dispatched, *for a required param*.

   What **remains** — so `_RECONCILE_AGAINST_OBSERVED` is **narrowed, not dropped** — is the general
   gap: (a) a null on a param the schema marks *optional* still passes through by design (many
   operations take legitimately-optional params), and (b) with no true guarded/conditional step, a
   superfluous `delete`/`update` grounded to a **plausible-but-stale, non-null** id can still be
   planned and attempted — the guard keys on null, not on staleness. So the failure mode stays
   **degraded from a deterministic crash to a probabilistic mis-action**, now held off by (a) the
   guaranteed required-null skip, (b) the reconcile prompt's `_RECONCILE_AGAINST_OBSERVED` fragment,
   and (c) the model's ground-time judgement. Retiring that fragment awaits true guarded steps; until
   then it stays honest scaffolding. Also depends on structured specs being available: a hand-authored-
   only manual (no `required` schema) leaves required-ness unknowable, so the guard can't fire there.

2. **Auto-caching the corrected plan (resolved — plan caching is now disabled).** Previously, on
   completion `DefaultReflectStrategy` stored `activity.plan`; after a mid-flight re-inference that
   was the *corrected* plan (with a `delete`/reconcile step), stored under the original goal. Two
   problems: (a) it was the **uncommon** case — the common case (no follow-up: focus → search → add →
   reply) is abandoned before completing on a dynamic run, so it never got cached; (b) a corrected
   plan is replayed **verbatim** by the retrieve path, so from a clean slate it re-ran a `delete`
   with nothing to delete (the crash mechanics are in limitation 1). The correction is **experience,
   not a reusable procedure**. This is now avoided by **disabling plan auto-caching runtime-wide** —
   the default Reflect records only an episode (`episodic.learn`). Safely *reusing* a distilled
   common-case procedure is future work (see "episodic → procedural consolidation").

3. **Self-caused state changes are indistinguishable from external ones.** The agent writes to the
   very tool it observes; every write emits `state_changed`. A naive "a signal arrived → re-infer"
   trigger therefore loops forever (reply → signal → re-plan → reply → …). The example sidesteps
   this by keying on **INBOX ids** — the agent's reply lands in SENT, invisible to the trigger — but
   that is ARE-email-shaped, not general.

4. **Observation requires focus, and focus is model-driven.** Observable properties are snapshotted
   only for *focused* tools, so the plan must explicitly `focus` every tool it reconciles against.
   `focus` is an ordinary plan step the base planner treats as optional, so we lean on the prompt to
   request it. If the model omits a focus, the dynamic behavior silently never triggers. Note the
   base prompt motivates focus only to *read what you need now* (it even says unfocus when done), so
   the specific missing motivation is *holding* focus to catch a future change and to re-see your own
   writes across a replan — that is what `_OBSERVE_TO_NOTICE_CHANGE` supplies.

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

- **Hard-interrupt preemption (shipped — [ADR-0020](../../../../docs/architecture/adrs/0020-hard-interrupt-and-await-input.md)).**
  `DecisionCycle.interrupt()` now preempts the current phase (phase-boundary checkpoints + true
  mid-flight abandonment of the Reason model call), and this scenario's reconsideration runs through it:
  `MailDiffInterruptPolicy` screens the pushed `state_changed` signal at push time and raises the
  interrupt, `ReconsiderInterruptHandler` routes it. **What remains** is the timing payoff, not the
  mechanism: the ARE bridge emits `state_changed` from `tool.observe()` (Observe-cadence), so the
  interrupt fires inside the current tick's Observe — there is no in-flight model call to abandon yet, so
  for the ARE sim this is largely a clean *relocation* of the trigger into the seam. Making the ARE push
  genuinely **off-cycle** (from the Environment thread) is the deferred unlock that turns this into true
  mid-Reason abandonment for the email scenario; it complements, rather than replaces, the reconsideration
  policy (limitation 6). The `/stop` user stop already exercises the async-source path today.

- **Replan-by-amend on resume (show the un-executed remainder).** On a `/stop` resume the runtime
  clears the plan (`plan`/`step_index`) and Reason authors a fresh plan seeing goal + executed history
  + the follow-up message. For an unchanged-intent follow-up ("nothing, continue") that re-derives a
  plan the model effectively already had. A refinement: also render the *un-executed remainder*
  (`plan.steps[step_index:]`) so the resume inference is an *amend* ("here's the plan you were running,
  these steps remain, here's the new input") rather than an author-from-scratch. It does **not** save
  the inference call — seeing the message requires an infer, which requires clearing the plan (a bare
  resume never sees it) — so it is not a token-cost win; the payoff is steadier, less-divergent output,
  collapsing to "reproduce the remainder" for a no-op message. Needs the pre-resume `step_index`
  preserved (currently zeroed) and must render the remainder *as pending against history-as-done* to
  avoid re-running an already-executed side-effecting step (the re-send hazard, limitation 1). Gate on
  evidence that no-op resumes actually diverge or cost too much.

- **Cache-oriented plan-prompt layout.** The plan prompt renders sections in *attention* order today —
  goal, tools, observed properties/signals, executed history, then user messages **last** (the freshest
  instruction most salient, best for reconsideration). Prompt caching is not wired: the
  `LLMClient.complete` seam sets no `cache_control` (caching is a cycle/agent concern by design). When
  it lands, the real lever is a stable-prefix cache breakpoint after the *system prompt + tool catalog*
  (the large, slow-changing chunks), with the volatile block (properties/signals, history, messages)
  after it. A second-order tweak then: order that volatile block most-stable-first — messages and
  append-only history churn less than re-observed properties/signals, so they *could* precede observables
  — weighed against the recency cost of moving the fresh instruction earlier. Micro-ordering messages vs.
  observables is a rounding error next to caching the catalog, so revisit this *when caching exists*, not
  before.
