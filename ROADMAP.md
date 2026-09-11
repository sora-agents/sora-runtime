# S-ORA Roadmap

Supersedes the phase-structured roadmap, now archived verbatim as the historical record of Phases 0–4
at [docs/development/roadmap-phases-0-4.md](docs/development/roadmap-phases-0-4.md)
(tooling, skeleton, the ARE walking skeleton, the TDD rollout, and the advanced-features work that has
since landed). Nothing in this file repeats work that is already done; the old file is where to look for
*why* a shipped mechanism has the shape it does, and its per-task "Done:" notes remain the closest thing
to a design changelog.

This file tracks **only what remains**, organized around the release rather than around build phases —
the phase structure has outlived its usefulness now that the runtime is complete and the open work is a
release gate plus a deferred backlog.

**Status:** the core runtime is implemented and green — decision cycle, activities, memory modules,
action registry, data-ops, plan representation (sub-goals, conditions, maintenance windows), blocked
state, hard interrupt, off-cycle inference, attention policies, the ARE in-process and MCP adapters, the
Anthropic and OpenAI-compatible LLM adapters, per-call metering, and the terminal CLI. 1,226 tests
(1,209 collected, 17 live-gated) pass under `ruff` + `mypy --strict`.

---

## 1. Where we are

Landed since the phase roadmap was last meaningful (one line each — details in git history and the old file):

- **Runtime completeness.** Blocked state, hard interrupt, full ARE simulation-engine integration,
  attention reconciliation in Observe with a pluggable `FocusPolicy`, achievement vs. maintenance
  sub-goals, condition retirement, a domain clock on the workspace, `$prop`, declared conditions +
  located change signals + undeclared-relevance recovery, and the structured data-op layer.
- **Structure.** `strategies.py` (4.7k lines, mixed responsibilities) split into `sora/_strategies/`
  behind the unchanged public `sora/strategies.py` façade.
- **Model seam.** Multi-provider support behind the unchanged `LLMClient` Protocol: `AnthropicLLMClient`
  and `OpenAICompatLLMClient` (OpenAI direct and OpenRouter-hosted), plus a provider-neutral
  `CompletionRequest`, cached-input accounting, and per-inference metering (`LLMMeter`) attributing
  tokens, latency, round trips, finish reasons, and resolution outcome to a named semantic call.
- **Evaluation.** The offline-first `examples.gaia2.evaluation` harness, three ID-only Gaia2
  manifests (familiar / development / **locked** acceptance, five scenarios each, one per
  capability), a frozen seven-call prompt baseline, and the canonical report contract.
- **Documentation.** The MkDocs Material site under [docs/](docs/) with generated API reference; README
  is now a router into it.

---

## 2. The v0.1.0 release

### 2.1 What the tag claims — revised

The old tag marker gated v0.1.0 on "dynamic environments, blocked-state, hard-interrupt, full ARE
simulation-engine integration" **and** "multi-provider LLM support: OpenAI + Gemini". Everything in the
first clause has shipped. The second clause needs a correction rather than a milestone:

- **The Gemini claim holds — through the OpenAI-compatible surface, which is the whole point of the
  seam.** *(Decided 2026-09-04.)* `OpenAICompatLLMClient` is written for exactly this: OpenAI itself, Gemini's compat endpoint,
  hosted gateways like OpenRouter, and local runtimes (Ollama/vLLM/LM Studio) are all a `base_url` +
  `model` config change, not a new adapter class — stated in the module docstring, listed in
  `pyproject.toml`, and pinned by a test (`test_base_url_is_forwarded_to_the_sdk_so_gemini_and_local_route_by_config`).
  Reasoning-token extraction already accounts for Gemini's compat surface, and the deliberate omission
  of an explicit `reasoning` null is justified in code by Gemini's provider-default dynamic thinking.
  So the tag can say Gemini. **The honest qualifier is coverage, not capability:** live runs exist for
  OpenAI and OpenRouter only, so phrase the claim as *"OpenAI-compatible endpoints — OpenAI, Gemini,
  OpenRouter, local runtimes — are configuration, not code"*, and say which ones have been exercised.
  A **native** Gemini SDK client is a separate, unneeded thing; the compat surface is the supported
  path. Cheap way to upgrade the claim if wanted: one live-gated smoke run against the Gemini endpoint.
- **Per-strategy model injection (`B2`) is not a tag blocker.** Nothing in the v0.1.0 story needs a
  second model per agent; it moves below the line with its independent value intact.

Revised tag scope: *the reactive/deliberative decision cycle running real, dynamic, asynchronous
scenarios end-to-end against ARE, through a provider-neutral model seam, with its behaviour measured on
Gaia2 scenarios that never influenced its prompts.*

Gates **V1** (prompt consolidation) and **V2** (the Gaia2 evaluation sweep) are sequenced in a separate
tracker and are not detailed here; their subsection numbers are left in place so the identifiers used
across both do not shift. What follows are the gates this file owns.

### 2.4 Gate V3 — Correctness fixes worth taking before the tag

Small, bounded, and each one currently a silent wrong-answer path rather than a missing feature.

- [ ] **V3.1** **Stale `last_operation` survives an interrupt landing between Observe and Reflect.**
      A failed ack resolves the activity to `READY`; a hard interrupt at the `_preempted()` checkpoint
      aborts the tick before Reflect judges; the activity is parked on an `InputWait`; the user's next
      message resumes it — and Reflect's first judgment treats the *old* failure as a fresh plan
      defect, adding a redundant replan and a stale entry to the breaker trail before the new
      instruction is acted on. `last_operation` has exactly one writer and is cleared by neither
      `reset_for_replan` nor `_resume_on_input`. Narrow (needs a `/stop` in that window) but real, and
      exactly the cross-phase composition class that phase-isolated tests never catch.
- [x] **V3.2** **Decide the policy for a failed external operation** (may be documentation, not code).
      Today a not-ok ack terminates the activity in Reflect — never a replan, never a retry — so a plan
      that is otherwise right dies on one bad tool argument. The replan machinery already exists and
      already carries a defect string; it is simply not wired to tool-level execution failures, and a
      custom `ReflectStrategy` can opt in today without a runtime change. Either wire a bounded
      replan-on-failure (the `replan_trail` breaker already bounds retries) or state explicitly that
      terminate-on-failure is the default policy and the opt-in is the extension point. Distinct from
      `T4` (sub-plan *inference* failure) and from `A5` (guarded steps).

      **Done:** the default now performs bounded replan-on-failure. Reflect retains the failed
      invocation in execution history, resets the plan with a short operation defect, and clears
      the handled `last_operation` trigger so the replacement survives the next tick. Failed calls
      no longer count as progress that forgives `replan_trail`; with no other progress on the trail,
      identical failures stop at two and distinct failures stop at the configured backstop, both
      through the existing ask-the-user terminus — but a successful call the activity has not made
      before still clears it, and that hole is `V3.6`. Custom Reflect strategies retain the policy
      seam and may still terminate. The defect string carries a write-safety warning (see
      `ADR-0025` §5), so replan-prompt *content* moved even though no pinned prompt constant did:
      benchmark numbers that predate it are not strictly comparable across a run that replans on a
      failed write.

- [ ] **V3.6** **A failed operation's retry bound is forgiven by progress on a different step.**
      `Activity._progressed_since_replan` clears the whole `replan_trail` as soon as any successful
      call the activity has not made before follows the mark — a rule calibrated when a rejected
      operation *terminated* the activity, so the trail only ever had to bound plans that made no
      calls at all. `V3.2` routed execution failures onto the same trail, where the rule has a
      weakness the old role never exposed: the call that decides progress is not the call that is
      failing. A plan shaped `[read → ok, write → rejected]` can vary its read every round, clear
      the trail every round, and replan without bound — a path that terminated on the first
      rejection before `V3.2`. `ADR-0025` is self-consistent (it defines a novel successful call as
      progress), so this is a scope decision the ADR has to make explicitly, not a patch: decide
      whether progress forgives a defect on a *different* step, and say so where the progress rule
      is stated.

- [x] **V3.3** **A sub-goal that fans out to zero steps reports as success, and the final message
      then asserts the work happened.** Observed on the first gaia2-cli run: a mechanical filter
      (`matching_relatives`) kept 0 of 8, the sub-goal *"Send each matching relative an individual
      email..."* fanned out to **0 steps**, and the run finished by telling the user it had *"sent
      individual emails titled 'Properties List'"*. Nothing was sent, nothing failed, and the
      trajectory reads as a clean success. Two separable defects, and the second is the worse one:
      an empty fan-out is indistinguishable from a completed one, and `send_message_to_user` is
      grounded from the plan's *intent* rather than from what the activity actually committed — so
      the agent's own report is not evidence about its own behaviour. A judge scoring the transcript
      would be scoring a claim. Relates to the mechanical-filter-misses-everything family (an exact
      match against a snapshot dropping every candidate), but that filter is only how this one was
      *triggered*; the reporting gap stands on its own. Same class as V3.1 — a composition nothing
      in a phase-isolated test can see.

      **Done (the grounded path):** an empty fan-out is now recorded on the activity
      (`Activity.noop_subgoals`, appended at the splice site in `_dispatch_subgoal`) and rendered
      into the grounding prompt as its own section, beside — never inside — the execution history:
      history is what a `$from`/`$decide` resolves a reference against, so admitting an entry for an
      operation that never ran would corrupt the one thing history is trusted for.
      `GROUND_SYSTEM_PROMPT` gained the rule the record exists to support, written about
      natural-language TEXT generally rather than about one operation — an email body claiming work
      that did not happen is wrong in exactly the same way a user report is, and the grounder is
      never told which operation it is filling in for. The rule separates what the goal says was
      *intended* from what the record shows *happened*, and requires naming the work that did
      nothing rather than omitting it. Deliberately **not** put in the revalidation prompt: that
      call judges whether the remaining steps are still the right ones to run, and a fan-out that
      found nothing does not make them wrong — it makes the report's wording wrong, which is a
      grounding question. Routing it through revalidation would also have missed the observed run
      entirely, since both shipped Gaia2 configs set `context_adaptation: none`.

      **Done (the literal path):** a report whose `text` the planner wrote as a *literal* has no
      references, so grounding's cheap path returned it unexamined and the sentence reached the user
      having been checked against nothing. `DefaultReasonStrategy._resolve` now takes a `force`
      flag, and the invoke branch sets it for exactly one case: the operation is
      `SEND_MESSAGE_TO_USER` **and** the activity has a no-op sub-goal on record. That escalates the
      report to `_ground_` even with nothing to resolve, so it is re-phrased against the record by
      the rule above. The trigger is the *evidence of a gap*, not the step being a report — a run
      where every planned step did something keeps the cheap path and pays no model call, which
      matters because call count is a reported result axis. Scoped to the reply channel rather than
      to every step following a no-op deliberately: re-grounding a write would hand its concrete
      arguments back to a model with no reason to change them and every opportunity to, a worse
      trade than the report it protects. `SEND_MESSAGE_TO_USER` moved to `sora/types.py` to make
      that check possible without core importing an adapter — core already depended on the literal
      (`PLAN_SYSTEM_PROMPT` names it in prose), so this removes a duplicated string rather than
      adding coupling; `sora.adapters.runtime_io` re-exports it. Same precedent as `USER_STOP`.
      Costs **no** further re-baseline: it is a pure runtime change and moves no prompt row.

      **Not verified on a live scenario.** Both halves are pinned by unit tests, including one that
      reproduces the observed shape end to end (empty fan-out, then the report). Neither has been
      re-run against a real gaia2-cli scenario, and the run that produced the finding is gone.

      **Re-baselines the ground prompt.** `examples/gaia2/evaluation/campaigns/prompt/baseline.json`
      was regenerated (`python -m examples.gaia2.evaluation prompt snapshot --output ...`). Exactly
      one of the seven frozen rows moved — `ground`, its system and user text and both hashes; the
      other six are byte-identical, so the blast radius is provably confined to grounding. Taken now
      rather than deferred precisely because the freeze has not happened and no sweep result exists
      to invalidate — which is the ordering §3 describes, though note the claim there that the V3
      fixes cost no re-baseline was written about V3.1/V3.2 and does not hold for this one.

- [x] **V3.4** **Make the prompt modules tier-conditional.** The rendered prompts assume an
      environment with observable properties: they teach `$prop` and property-backed conditions to
      every agent, including ones whose environment has no properties at all. Where the environment
      offers only operations and signals, `$prop` is dead vocabulary; where it offers only
      operations, the signal-backed waits go with it. Select each call's modules from the
      environment's declared perception channels. Not cosmetic — vocabulary the environment cannot
      satisfy is a source of invalid plans. **Forces one decision to declare:** whether the
      environment's declared channels govern perception only, or the prompt as well. The latter is
      the honest configuration, since the alternative hands an agent instructions it cannot act on,
      but it makes prompt text co-vary with the environment, so state it rather than letting it be
      implicit. **Wants a prompt-module seam** — modularizing the seven prompts without changing
      rendered text is text-preserving by construction, so it is cheap to pull forward and is the
      enabler. Re-baselines the rows it touches, which is free before the prompt freeze and costly
      after it.

      **Done:** prompt fit now follows the independent `properties` / `signals` affordances
      declared by the manuals in each call's own tool scope. Operations-only calls omit observation,
      focus, `$prop`, and pending-condition vocabulary; signal-only calls keep event/focus/wait
      guidance without teaching properties; property-only calls keep property references and
      derived-change watches without claiming a signal channel. The seven built-ins assemble from
      named manifests and populate `CompletionRequest.sections`; custom plan/ground builders retain
      their tuple contract and empty section metadata. `agent.procedural.prompt_fit` defaults to
      `adaptive`, with `fixed-rich` available as the explicit sensitivity control. The prompt
      snapshot now freezes all 21 semantic-call/tier combinations and the rich tier remains
      byte-identical to its seven pre-change hashes.

### 2.5 Gate V4 — Release mechanics

- [ ] **V4.1** Write the real `CHANGELOG.md` entry — it still says *"No code has been released yet"*.
- [ ] **V4.2** Reconcile `pyproject.toml`'s `version = "0.1.0"` with the tag, and decide the
      post-tag versioning convention.
- [x] **V4.3** Loose root working documents relocated to an untracked `notes/` directory
      (gitignored): the prompt-consolidation lists, the release analysis, the failure write-ups, and
      the documentation-architecture proposals. They churn per session and their conclusions are
      distilled into tracked ADRs, `docs/architecture/notes/`, or this file when they land — so the
      standing rule is **delete a note once it has been distilled**.
- [x] **V4.4** ROADMAP citations removed from durable files. Per AGENTS.md, ADRs, design notes,
      source, and documentation pages must not cite roadmap task IDs or phase labels at all — they
      dangle the moment the roadmap is restructured, which is exactly what happened here. Those
      citations now describe the deferred work itself. A pointer to this file survives only in the
      project's own front matter (`README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `AGENTS.md`,
      `CLAUDE.md`), where a roadmap link is the point rather than a dangling cross-reference. `src/sora/__init__.py`'s
      docstring, which called the package a *"packaging placeholder"* with *"no public re-exports
      yet"*, was corrected at the same time.
- [ ] **V4.5** Docs sweep against the shipped surface — README claims, `docs/index.md` maturity
      boundaries, and the experimental/stable split for extension seams.
      `docs/architecture/status-and-stability.md` is still an unwritten scaffolding stub, and it is
      precisely the page a first release needs. `mkdocs build --strict` is
      already CI-enforced, so this is about accuracy, not links.
- [ ] **V4.6** State the known limitations honestly in the release notes rather than burying them.

### 2.6 Explicitly out of scope for v0.1.0

WoT and the two-agent lab, multimodal perception, per-strategy model injection, restore-drift
reconciliation, cross-workspace tool sharing, and every evidence-gated capability gap in §4. None of them is needed for the tag's claim, and several are deliberately waiting for a
concrete driver rather than a speculative build.

---

## 3. Active workstream — immediate order of work

Release-gate work runs in two tracks. V1 and V2 — prompt consolidation and the benchmark evaluation —
are sequenced in their own tracker. What this file orders is everything else, plus the one hard
constraint between the tracks.

1. **V3** — the correctness fixes, in any order, but **before the prompt freeze V2 depends on**.
   V3.1 and V3.2 touch no prompt text, so neither costs a re-baseline; **V3.3 did** — its whole
   subject is what the grounding prompt is shown about work that did not happen — and it moved the
   `ground` row of the frozen baseline. That is the argument for taking these early rather than
   against it: before the freeze a prompt-touching correctness fix is free, and after it the same fix
   forces a re-run. V3.1 is additionally a live wrong-answer path a benchmark run could otherwise hit.
   **V3.4 also moves a prompt baseline**, and for the same reason the freeze ordering matters to it.
2. **V4** — release mechanics, then the tag.

**The tag lands after the benchmark sweep has been run, not after it reaches a score.** The sweep's
value for a release is that a few hundred live scenarios surface minor issues nothing else does, and
fixing those is what the tag should carry.

Nothing in §4 starts before the tag.

---

## 4. Post-v0.1.0

Each item keeps its legacy identifier from the phase roadmap so historical cross-references in the archive and in git
notes still resolve.

### 4.1 Plan-language and data-op capability gaps *(evidence-gated)*

The standing rule for this group: **do not build against one scenario.** Prioritize by multi-scenario
Gaia2 results, and where an item was already checked against the benchmark and found net-negative, that
finding stands until new evidence arrives.

- [ ] **P1** *(T4)* Frame-local sub-goal replan and sub-plan failure propagation — keep the parent
      frames and re-infer only the active sub-plan; propagate instead of terminating when sub-plan
      inference fails; the grounding-escalation → re-infer-when-stale trigger; sub-plan `retrieve()`
      caching; and a synthesized (not templated) await-input prompt for the breaker.
- [ ] **P2** *(T5)* Data-op refinements — `select`/`$decide` cross-collection context, multi-key `sort`,
      a mechanical conditional data-op for count-dependent fallbacks, model-escalated envelope
      extraction, and per-element vs. batched `$decide`. Item (a) is the one real remaining capability
      gap but is unexercised today; build each on a concrete driver.
- [ ] **P3** *(T8)* Iterating a paginated operation. **Checked 2026-08-24 and found net-negative for
      Gaia2**: every ARE app publishes full state as a `state` property, so `$prop` already covers each
      case at zero tool calls, while a `range`-driven fan-out costs `ceil(total/limit)` sequential
      invocations and widens the planner vocabulary for every plan. Stays gated on a real
      paginated-only non-ARE tool; a fan-out **width** ceiling has to land with it or before it.
- [ ] **P4** *(T9)* Context guard — with its stated prerequisite, the disambiguated `SemanticMemory`
      world-knowledge methods, built as part of the task rather than after it (no new memory type).
      Percept consolidation into semantic memory must carry provenance if it follows.
- [ ] **P5** *(T12)* Reporting a run cut short by the wall clock. Still evidence-gated: it may be a
      symptom of replan churn since fixed, and a naive "here is what I got done" converts a clean
      incomplete into a possibly-wrong claim the judge can penalize. The one piece worth taking whenever
      this is opened: record the stop **reason** on `RunResult` instead of inferring a timeout from
      duration.

### 4.2 Dynamic environments and attention

- [ ] **P6** *(A6)* Relocate the `_THREAD_READING` domain knowledge into a hand-authored email-client
      manual, exercising the ADR-0015 pairing on the `are-sim` adapter (the MCP side shipped in E4).
      Retires the last non-scaffolding prompt fragment in the ARE example.
- [ ] **P7** *(A7)* Restore-drift reconciliation — a joined workspace whose live tool set moved since it
      was recorded. Leading candidate is an explicit agent-driven `refresh`/`resync` action, keeping
      `restore()` fast and pure; pin removed-tool `connect()` semantics in an adapter ADR. Analysis in
      [docs/architecture/notes/restore-drift-reconciliation.md](docs/architecture/notes/restore-drift-reconciliation.md).
- [ ] **P8** *(A11)* Ask the user about activities blocked on long-quiet conditions. Needs the domain
      clock (shipped), must measure age since the activity last *did* anything, and — being an
      unprompted outbound message that costs a benchmark turn — must be **off by default** and never
      enabled for benchmark runs.
- [ ] **P9** *(A12)* Root the derived-change path space so `_match_derived`'s path gate actually gates.
      A rootless whole-property diff matches **any** wait path, so an unrelated property movement buys a
      judge call. Decide first whether `SignalWait.path` is rooted at the property name or inside a
      property's value, then make both producers agree — rooting one side alone desyncs the announced
      and derived logs, and rooting both changes what the planner writes (a Gaia2 re-baseline). Carries
      a second, smaller half: the same-tick dedup in `_derive_property_changes`.

### 4.3 LLM provisioning and modality

- [ ] **P10** *(B2)* Surface an `LLMClient` to every phase strategy — a per-phase `model:` in
      `agent.yaml`, built in bootstrap, injected per strategy at construction. Multi-model then falls out
      for free. Prefer this over an ambient `cycle.llm`. Does **not** make `infer()` redundant; the risk
      to guard with guidance is hand-rolling planning against the raw client and losing the parse/reuse.
- [ ] **P11** *(C4)* Multimodal model support — a richer `LLMClient` Protocol carrying multimodal (and
      tool-calling) content, so an LLM-backed Observe can interpret raw perception. Orthogonal to P10:
      per-phase wiring picks *which* model, the Protocol shape decides *what modality*. Its real trigger
      is P12's reworked camera feed.
- [ ] **P12** Live coverage for the OpenAI-compatible endpoints that currently have only config and
      unit coverage (Gemini, local runtimes) — one skip-gated smoke run each, not new machinery. A
      *native* SDK client for any of them stays out of scope: the compat surface is the supported path.

### 4.4 WoT and the two-agent lab

- [ ] **P13** *(C1)* WoT adapter and the two-agent lab scenario, including reworking the `video-stream`
      device from an on-device text description into a genuine raw camera feed.
- [ ] **P14** *(C2)* Extend `Manual` reconciliation to WoT Thing Descriptions — the WoT half of the
      ADR-0015 merge policy.
- [ ] **P15** *(C3)* Cross-workspace tool sharing — decide whether one tool may belong to two
      workspaces. Leading candidate splits by connection ownership (connection-owned tools stay
      exclusive; self-addressed tools may be referenced from several workspaces, with refcounted
      deregistration), as an ADR refining ADR-0014. Analysis in
      [docs/architecture/notes/cross-workspace-tool-sharing.md](docs/architecture/notes/cross-workspace-tool-sharing.md).

### 4.5 Advanced reasoning and planning

- [ ] **P16** *(D2)* Multi-field `TickResult` fusion in practice; replanning-policy experiments.
- [ ] **P17** *(D3)* Log the full execution trace as an experience — an append-only `(step, invocation,
      ack)` log on `Activity` that Reflect serializes into the episode, turning an episode into a genuine
      step-by-step record. Also revisits which step failed and how replanning is recorded.
- [ ] **P18** *(D4)* Per-activity `context_adaptation` override, so a delegated or background sub-task
      can commit differently from its parent.
- [ ] **P19** *(D5)* Author-declared reconsideration checkpoints for dense-write tools, generalizing
      `before_writes` — a physical-safety boundary as much as an economic one.
- [ ] **P20** *(D6)* Purge stale **reads** from `Activity.history` on a reconsideration-driven replan
      while retaining the side-effecting acks (the re-send guard). Now unlocked by
      `OperationSpecification.side_effecting`.

### 4.6 Backlog / exploratory

- [ ] **X1** ARE-over-MCP — dynamic scenarios across a standard protocol wire. Not needed for any tag;
      valuable as protocol-interop input for the WebAgents CG. The A2 investigation established that
      polling is not an MCP limitation but that ARE's server has no off-request/cross-thread push path,
      and that `USER_MESSAGE` has no MCP push surface at all.
- [ ] **X2** Adaptive commitment — tune reconsideration density to the observed world-change rate
      (Kinny–Georgeff's γ) instead of a static `context_adaptation`. Speculative; needs multi-scenario
      evidence first.

---

## 5. Notes

- Keep this file current as items land or get reordered; it is the single place tracking implementation
  status now, and the phase roadmap is frozen.
- If an implementation step reveals that a design decision needs to change, write a new ADR superseding
  the old one (see [docs/architecture/adrs/README.md](docs/architecture/adrs/README.md)) rather than
  silently diverging from the design documents.
- Do not reference this file's item labels (`V3.1`, `P7`, …) from durable files — code comments,
  docstrings, config, **ADRs, design notes, or documentation pages**. Describe the thing itself;
  roadmaps get restructured and leave the reference dangling, which is exactly what happened to the
  phase roadmap's `D4`/`D5`/`X2` citations.
