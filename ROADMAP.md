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
- **Evaluation.** The offline-first `examples.gaia2.evaluation` harness, the `prompt` / `aamas2027`
  campaign boundary, three ID-only Gaia2 manifests (familiar / development / **locked** acceptance,
  five scenarios each, one per capability), three explicit model profiles, a frozen seven-call prompt
  baseline, a dated price sheet, budget policy, and the canonical report contract.
- **Documentation.** The MkDocs Material site under [docs/](docs/) with generated API reference; README
  is now a router into it.

Model profiles in use for prompt tuning: **GPT-5.4 medium** and **Kimi K2.5 with reasoning enabled**.
GPT-5.4 high is retained as a non-default transfer/paper declaration.

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

### 2.2 Gate V1 — Prompt consolidation

Tracked in full in the prompt-consolidation working note (a local, untracked task list with its own
commit-level sequencing; deliberately not duplicated here, and not a tracked file — its *outcome*, the
frozen prompt hash and the evaluation results, is what gets committed — see V2.6 on where).
Items 1–4 are done — cached-input accounting, the provider-neutral
completion request, call/section attribution, and the evaluation harness with frozen baselines.

- [ ] **V1.1** Items 5–6 — bound verdict-call output with explicit exhaustion handling, then tune
      verdict-call reasoning profiles. Blocked on the live baseline.
- [ ] **V1.2** Items 7–8 — modularize the seven prompts without changing rendered text, then move
      prompt provenance out of phrase pins.
- [ ] **V1.3** Items 9–10 — record contract ownership across the runtime/environment boundary, then
      declare and validate the plan action envelope.
- [ ] **V1.4** Items 11–12 — order dynamic planning context by volatility; scope small-call context by
      dependency. *(Items 11–12 move ahead of 7–10 if the item-4 cache-coverage measurement lands at
      ≥75% median fixed-prefix coverage — the priority gate is written into the consolidation plan.)*
- [ ] **V1.5** Item 13 — audit and ablate planner modules within the fixed six-variant budget.
- [ ] **V1.6** Item 14 — tool-catalog reduction. **Deferred past the tag** unless the expanded
      acceptance batch it requires is already being run for another reason; catalog omission has the
      highest silent-failure risk of anything in the plan.

### 2.3 Gate V2 — Gaia2 evaluation

This is the gate the release hinges on. **Decided 2026-09-04: the tag lands after the benchmark sweep
has been *run*, not after it reaches a score** — the sweep's real value for a release is that a few
hundred live scenarios surface minor issues nothing else does, and fixing those is what the tag should
carry. That places v0.1.0 **after** the `aamas2027` campaign's headline runs rather than before them.

The old `T10` ("Gaia2 all categories pass") is retired as a gate: a 100%-pass claim is stronger than
any suite here supports, and on the full set it is partly hostage to ARE's two documented
one-directional judge defects. The gate is `T11` — the run happening and its findings triaged.

- [ ] **V2.1** Debug the **familiar** (smoke) suite under GPT-5.4 — the five scenarios that have always
      been regression cases, on the primary tuning profile. *Immediate next step.*
- [ ] **V2.2** Improve prompt-eval harness logging so a failing run is diagnosable without re-running it.
      *(Being taken up in a separate session.)*
- [ ] **V2.3** Complete the consolidation campaign (V1) against familiar → **development** suites, one
      module or context section per commit with its quality, safety, latency, and token delta attached.
- [ ] **V2.4** Run the **locked acceptance** suite once, on the finalist, on both tuning profiles, and
      report the acceptance-set familiarity gap. The five acceptance ids are frozen and must stay
      uninspected until this run. This is the **finalist gate** — it decides which prompt candidate is
      frozen — not the release gate; the release gate is V2.6/V2.7 below.
- [ ] **V2.5** **Freeze the prompts before the sweep.** `PLAN_SYSTEM_PROMPT` edits break comparability
      across dates (threat 7 in the evaluation plan, and the standing re-baseline hazard), so V1 must be
      *finished*, not in flight, and the prompt hash recorded with every result. This is the hard
      ordering constraint between the two gates.
- [ ] **V2.6** **Run the benchmark sweep** — `aamas2027`, planned in the evaluation-plan working note
      (`notes/benchmarks/gaia2/`, untracked). **Decide before the sweep where the *results* live:** the
      plan can stay a local note, but a released benchmark claim needs its methodology and numbers
      committed somewhere citable — `docs/benchmarks/` is the natural home and is currently empty.
      Scope is a budget call, not a correctness one: **mini (160 scenarios, n=1) covers all five
      capabilities at ~$1.5k and is enough for the release gate**; the full 800 × n=3 (~$7k) buys
      statistical power for the paper, not release confidence. Its own prerequisites are already
      written down there — judge A/B, clock-semantics parity, judge selection, then the pilot.
- [ ] **V2.7** **Triage what the sweep surfaces, and split the fixes by comparability.** This is the
      step that justifies gating the tag on the run at all. A **runtime-only** fix is safe to take
      immediately — it cannot move a prompt baseline. A **prompt-touching** fix invalidates the numbers
      already produced, so it either waits until after the paper's results are locked or forces a
      re-run; decide which per finding rather than by reflex, and record the choice with the result.

### 2.4 Gate V3 — Correctness fixes worth taking before the tag

Small, bounded, and each one currently a silent wrong-answer path rather than a missing feature.

- [ ] **V3.1** **Stale `last_operation` survives an interrupt landing between Observe and Reflect.**
      A failed ack resolves the activity to `READY`; a hard interrupt at the `_preempted()` checkpoint
      aborts the tick before Reflect judges; the activity is parked on an `InputWait`; the user's next
      message resumes it — and Reflect's first judgment terminates it on the *old* failure before the
      new instruction is ever acted on. `last_operation` has exactly one writer and is cleared by
      neither `reset_for_replan` nor `_resume_on_input`. Narrow (needs a `/stop` in that window) but
      real, and exactly the cross-phase composition class that phase-isolated tests never catch.
- [ ] **V3.2** **Decide the policy for a failed external operation** (may be documentation, not code).
      Today a not-ok ack terminates the activity in Reflect — never a replan, never a retry — so a plan
      that is otherwise right dies on one bad tool argument. The replan machinery already exists and
      already carries a defect string; it is simply not wired to tool-level execution failures, and a
      custom `ReflectStrategy` can opt in today without a runtime change. Either wire a bounded
      replan-on-failure (the `replan_trail` breaker already bounds retries) or state explicitly that
      terminate-on-failure is the default policy and the opt-in is the extension point. Distinct from
      `T4` (sub-plan *inference* failure) and from `A5` (guarded steps).

### 2.5 Gate V4 — Release mechanics

- [ ] **V4.1** Write the real `CHANGELOG.md` entry — it still says *"No code has been released yet"*.
- [ ] **V4.2** Reconcile `pyproject.toml`'s `version = "0.1.0"` with the tag, and decide the
      post-tag versioning convention.
- [x] **V4.3** Loose root working documents relocated to an untracked `notes/` directory
      (gitignored): the prompt-consolidation lists, the release analysis, the failure write-ups, and
      the documentation-architecture proposals. They churn per session and their conclusions are
      distilled into tracked ADRs, `docs/architecture/notes/`, or this file when they land — so the
      standing rule is **delete a note once it has been distilled**.
- [x] **V4.4** ROADMAP citations removed from durable files. Per CLAUDE.md, ADRs, design notes,
      source, and documentation pages must not cite roadmap task IDs or phase labels at all — they
      dangle the moment the roadmap is restructured, which is exactly what happened here. Those
      citations now describe the deferred work itself. A pointer to this file survives only in the
      project's own front matter (`README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `CLAUDE.md`), where
      a roadmap link is the point rather than a dangling cross-reference. `src/sora/__init__.py`'s
      docstring, which called the package a *"packaging placeholder"* with *"no public re-exports
      yet"*, was corrected at the same time.
- [ ] **V4.5** Docs sweep against the shipped surface — README claims, `docs/index.md` maturity
      boundaries, and the experimental/stable split for extension seams.
      `docs/architecture/status-and-stability.md` is still an unwritten scaffolding stub, and it is
      precisely the page a first release needs. `mkdocs build --strict` is
      already CI-enforced, so this is about accuracy, not links.
- [ ] **V4.6** State the known limitations honestly in the release notes, citing the ARE judge defects
      and the clock-semantics convention rather than burying them.

### 2.6 Explicitly out of scope for v0.1.0

WoT and the two-agent lab, multimodal perception, per-strategy model injection, restore-drift
reconciliation, cross-workspace tool sharing, and every evidence-gated capability gap in §4. None of them is needed for the tag's claim, and several are deliberately waiting for a
concrete driver rather than a speculative build.

---

## 3. Active workstream — immediate order of work

1. **V2.1** — debug the familiar suite under GPT-5.4.
2. **V2.2** — prompt-eval harness logging (separate session).
3. **V1 / V2.3** — work the consolidation roadmap in its own numbered order, gated by the
   cache-coverage branch.
4. **V2.4** — the locked acceptance run, once, on the finalist.
5. **V2.5** — freeze the prompts and record the hash. Everything after this point is comparability-
   critical.
6. **V2.6** — the benchmark sweep (mini is sufficient for the gate).
7. **V2.7** — triage its findings; runtime-only fixes land, prompt-touching fixes wait or force a re-run.
8. **V4** — release mechanics, then tag.

**V3** (the two correctness fixes) can proceed in parallel with the campaign at any point *before*
step 5 — neither touches prompt text, so neither costs a re-baseline. Taking them early is preferable:
V3.1 is a live wrong-answer path the sweep could otherwise hit.

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
- Do not reference this file's item labels (`V2.4`, `P7`, …) from durable files — code comments,
  docstrings, config, **ADRs, design notes, or documentation pages**. Describe the thing itself;
  roadmaps get restructured and leave the reference dangling, which is exactly what happened to the
  phase roadmap's `D4`/`D5`/`X2` citations.
