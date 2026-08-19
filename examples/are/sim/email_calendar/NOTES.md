# ARE dynamic scenario — limitations, what's example-specific, and future work

This example runs a **dynamic** ARE scenario end to end: the user asks the agent to schedule a
team sync and reply to Alice; mid-run, a follow-up email arrives that changes the answer
(Monday → Tuesday). The agent is made to reconsider through its own decision cycle rather than
finishing the stale plan.

**Primary reconsideration path: context-adaptation ([ADR-0024](../../../../docs/architecture/adrs/0024-plan-reconsideration-context-adaptation.md)).**
The `agent.yaml` sets `context_adaptation: before_writes`, so before each side-effecting step Reason
re-validates the in-flight plan against new perception: a cheap mechanical change-gate (did anything
observable move since the plan was inferred?) fronts a single model *revalidation* call (given the goal and
remaining steps, is the plan still valid?), and on "no" it re-plans against the current inbox. This
is **general runtime machinery** — no inbox-shape knowledge, no example code — and it is what handles
the mid-run follow-up. That re-plan is not blank: the invalidated plan's un-run steps are carried into
the replanning prompt (whole-activity, since invalidation clears the sub-goal stack too), so the
Tuesday plan is written against the Monday one it replaces rather than re-derived from nothing —
`reconciling_plan_prompt` inherits the section without being changed, since it composes the default's
user half. The `MailDiffInterruptPolicy` seam below is retained only as an **opt-in
deterministic override** (see it in the "example-specific" catalogue). This substantially addresses
old limitation 6 (blind commitment) at the *plan* level; intention-level commitment lifecycles remain
future work.

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

- **`MailDiffInterruptPolicy`** *(opt-in override, not the default path)* — an `InterruptPolicy`
  (screened at push time) that raises a hard interrupt on a *genuinely new inbound email*. On a
  `state_changed` event it diffs the set of **INBOX email ids** read off the emitting tool's own
  `state` observable against what it has already seen, which knows ARE's `EmailClientApp` state
  shape (`folders → INBOX → emails[*].email_id`). It reads the *tool*, not the signal (which is a
  bare event — ADR-0004) and not `wm.properties` (still a snapshot behind at push time — ADR-0020).
  ARE-email-shaped. Stateful, so it also dedups — each new id fires once, the first non-empty inbox is
  the baseline, and the agent's own reply (landing in SENT) never fires. See [ADR-0020](../../../../docs/architecture/adrs/0020-hard-interrupt-and-await-input.md).
  The default path is now the general context-adaptation mechanism (see the intro); this is wired only
  when `agent.yaml` uncomments the `interrupt_policy`/`interrupt` lines. Two things keep it worth
  retaining: it is **deterministic** (no model call) and **preemptive** (aborts the current phase),
  and — unlike a before-writes checkpoint — it also fires **after the goal completed** (see the handler).
- **`ReconsiderInterruptHandler`** *(opt-in override)* — the paired `InterruptHandler`. On that
  interrupt it clears the in-flight activity's plan so the default Reason re-infers, or — if the email
  landed *after* the goal already completed (no live activity) — spawns one fresh corrective activity
  (a hard-coded `_CORRECTIVE_GOAL` string). A user stop is delegated to `DefaultInterruptHandler`. That
  post-completion branch is the one thing context-adaptation structurally **can't** do: a terminated
  activity has no before-writes checkpoint to fire, so covering a follow-up that arrives after the goal
  is done still needs either this preemptive seam or a persistent monitoring intention (future work).
- **`reconciling_plan_prompt` / `_RECONCILE_INSTRUCTION`** — a `PlanPrompt` appending
  dynamic-environment guidance to the default planning content, **split into fragments with
  different fates**:
  - `_OBSERVE_TO_NOTICE_CHANGE` — *(retired)* was scaffolding for limitation 4 (perception gated on a
    model-driven `focus`). Dropped now that `JoinAction` auto-focuses joined tools — see below.
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
   very tool it observes; every write emits `state_changed`. The cost of this depends on the path:
   - Under the **opt-in interrupt override** (`MailDiffInterruptPolicy`), a naive "a signal arrived →
     re-infer" trigger would loop forever (reply → signal → re-plan → reply → …); the policy sidesteps
     it by keying on **INBOX ids** — the agent's reply lands in SENT, invisible to the trigger.
   - Under the **primary context-adaptation path** ([ADR-0024](../../../../docs/architecture/adrs/0024-plan-reconsideration-context-adaptation.md))
     there is no loop: the self-write does trip the mechanical change-gate at the next checkpoint, but
     the revalidation that fires returns *valid* (the agent's own SENT entry doesn't invalidate the plan),
     so the plan commits and execution proceeds. The residual cost is degraded from an infinite loop to
     **one wasted revalidation call per self-write**.

   Either way the underlying gap is the same — the runtime can't tell a self-caused change from an
   external one — and the INBOX-id trick that plugs it is ARE-email-shaped, not general. The
   principled fix is efference / read-write tagging (see the foundational extensions below), which
   would let the change-gate ignore self-writes directly and drop the wasted revalidation too.

4. **Observation requires focus (mitigated, not resolved).** Observable properties are snapshotted
   only for *focused* tools. This *was* model-driven — the plan had to explicitly `focus` every tool
   it reconciled against, and `focus` is an ordinary step the base planner treats as optional, so if
   the model omitted one the dynamic behavior silently never triggered. **Mitigated** by
   `JoinAction` now auto-focusing every tool of a joined workspace, so perception no longer hinges on
   a model focus step and `_OBSERVE_TO_NOTICE_CHANGE` is retired. This is a **temporary mechanical
   fallback**: focusing *everything* joined gives up the per-cycle observation-cost narrowing that
   intentional focus buys. The real fix — reliable, intentional model-driven focus/unfocus that
   attends to only the tools that matter (and *holds* that focus across a replan) — is still future
   work; `_focus_`/`_unfocus_` remain the seam for it.

5. **Observation-aware inference can bake run-specific literals.** Because the mailbox's `state`
   observable property is snapshotted into working memory in the same Observe that drains the thin
   `state_changed` signal — and that lands *before* the plan-inference call resolves — the planner
   sees the email contents in its context and extracts parameter values at plan time. The prompt sanctions this for *stable,
   meaningful* values (a title, a time, a name — memory.py's "fill parameters whose value is stable
   and meaningful") but forbids it for *volatile identifiers* (email/event ids → keep as `$from`
   references). Two consequences, both about what this leaves for grounding:

   - **Baking suppresses grounding entirely.** Grounding (`_ground_`) is demand-driven — it fires
     only for a `$decide` param or an unresolvable `$from`/`$bind`. A plan that bakes concrete
     literals leaves nothing to ground, so a data-dependent value (the meeting's attendees) never
     gets a focused, later extraction pass; it rides on whatever the one-shot planner produced.
   - This is fragile on a weak planner. Observed concretely on **qwen3-30b (local)**: Run #1 baked
     `attendees: ['Bob','Carol']` correctly and kept the email id as `$from search_emails` → PASS,
     with **zero grounding calls** (correct *and* cheapest). Run #2's re-inferred plan (a) baked the
     email id as a bare literal — a genuine violation of the "ids stay references" rule, harmless
     only because the id was stable within the run — and (b) *dropped* the attendees, so the
     calendar event scheduled with none → **FAIL**. Because those params were concrete-but-incomplete
     rather than `$decide`, there was no grounding pass to recover the missing attendees.

   The id-hardcoding half mattered most for cross-run reuse (now moot — plans aren't cached) and
   remains within-run hygiene. The deeper half is a genuine tension: eager literal-baking trades
   grounding's focused extraction for a cheaper single-shot plan — a win on a strong planner, a
   detail-dropping risk on a weak one — and the runtime can't force deferral (it can't tell which
   literals are data-dependent). The levers are prompt/config, and none is free; the real variable is
   planner capability.

6. **Commitment at the plan level, not yet the intention level (mostly addressed — [ADR-0024](../../../../docs/architecture/adrs/0024-plan-reconsideration-context-adaptation.md)).**
   The default path is no longer "re-infer on every new inbound email." `context_adaptation` gives a
   real BDI-flavoured commitment dial (`none | before_writes | before_each_op`): a mechanical
   change-gate plus a model revalidation decides *when* a plan is worth re-validating and whether new
   perception actually invalidates it. **What remains** is the intention lifecycle proper — an
   intention being blocked vs. impossible vs. superseded, per-*activity* commitment (this dial is
   per-agent), and a commitment posture that adapts to how fast the world is actually changing.

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

- **BDI-style commitment & reconsideration policies** — *plan-level shipped ([ADR-0024](../../../../docs/architecture/adrs/0024-plan-reconsideration-context-adaptation.md))*:
  `context_adaptation` replaces "re-infer on every signal" with a principled, config-selected decision
  about *when* to reconsider a plan and a revalidation of whether new perception invalidates it. What
  remains is the **intention lifecycle** proper (blocked / impossible / superseded), a per-activity
  (not just per-agent) commitment override, and a posture that adapts to the observed world-change
  rate. Substantially retires limitation (6).

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
