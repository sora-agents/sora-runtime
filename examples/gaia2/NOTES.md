# Evaluating S-ORA on Gaia2 (ARE / Meta) — feasibility assessment

*Date: 2026-08-03. Source repos inspected: `sora-runtime` (this repo);
`meta-agents-research-environments` (ARE); `are-email-calendar-scenario`
(native port of the sim demo); the Gaia2 paper (ICLR 2026, `gaia2.pdf`).*

## Verdict

**Yes, an evaluation is sensible — and S-ORA is unusually well-positioned for
it — but there are concrete scoring-fidelity gaps to close before a
*comparable* number is possible, and the honest performance story is narrower
than "beats the ReAct baseline."**

Two things drive this. First, the plumbing is ~80% built already: S-ORA ships
real ARE integration (`sora/adapters/are_sim.py` in-process adapter +
`sora/adapters/are_mcp.py`), a scenario-loading seam that already accepts Gaia2
`.json`, a model-backed reasoning path, and skip-gated integration tests that
run an ARE `Environment` end-to-end. Second, S-ORA's design differentiators map
directly onto the three capabilities where every Gaia2 baseline collapses.

## What's already in place (not gaps)

- **Scenario loading.** `are_sim.load_scenario()` already branches on `.json`
  and delegates to ARE's `are.simulation.benchmark.scenario_loader.load_scenario`.
  Dotted `Scenario` classes and Gaia2 JSON both resolve.
- **Dynamic environment loop.** The in-process path runs ARE's `Environment`
  event loop on a background thread, so a scenario's timeline (mid-run email
  injections, follow-ups, user messages) actually fires. Exercised by
  `tests/test_are_sim_integration.py` (skip-gated on `are.simulation`).
- **Model-backed reasoning** (`DefaultReasonStrategy` + `AnthropicLLMClient`)
  and the **async / interrupt / blocked machinery** — the actual
  differentiators — exist and are tested.

## The gaps (ordered by how much they block a real run)

### 1. Agent actions logged as ARE `AGENT` events — already handled ✓

*(This was originally flagged as the non-negotiable blocker, on a static
reading of the code. Empirical testing against a running `Environment` shows it
is already satisfied — corrected here.)*

Gaia2's `GraphPerEventJudge` scores by matching **agent events against the
oracle event graph** — `extract_agent_events()` keeps only events where
`event_type == EventType.AGENT`. The concern was that S-ORA's `_AreTool.invoke`
calls the `AppTool` directly (`are_sim.py:392`), bypassing ARE's *`register_event`*
decorator, so no AGENT event would be logged.

That reading missed how ARE apps actually record events. Every app operation
method is decorated `@event_registered(event_type=EventType.AGENT)` (default),
and `Environment.register_apps` wires each app's `add_event` to `env.add_to_log`.
So calling an operation **inside a running environment self-registers exactly
one `CompletedEvent(event_type=AGENT)`** with the correct `tool_name`
(`EmailClientApp__list_emails`), `function_name`, args, and return value — no
`register_event` wrapper needed. Verified end-to-end through S-ORA's own invoke
path: `email_tool.invoke("list_emails")` appends one AGENT event to
`sim._env.event_log` (regression test
`tests/test_are_sim_integration.py::test_invoke_is_logged_as_are_agent_event`).

**Consequence:** no production change. Wrapping invokes in ARE's `register_event`
would be a *regression* — it double-logs (the inner method self-registers too).
The one thing to preserve is the invariant, which the new test guards.

### 2. No judge engine wired — the real remaining blocker

`load_scenario` never attaches a judge, so a JSON benchmark scenario's
`validate()` falls back to `ScenarioValidationResult(success=None)` — unscored
(`benchmark_scenario.py:259`). This is now the *first* real gap.

**Fix (one step, not two):** call ARE's `preprocess_scenario(scenario,
judge_config=GraphPerEventJudgeConfig(engine=create_judge_engine(...)),
offline_validation=False)`. It does the oracle work itself — runs the scenario
in **oracle mode** from the `OracleEvent`s already present in the Gaia2 JSON,
populating `scenario.oracle_run_event_log`, then sets `scenario.judge` and
`scenario.validate`. So there is **no separate `huggingface_loader` oracle
download** and `load_completed_events` stays `False` (that flag is only for
offline replay of a pre-recorded run). What this *does* require is provisioning
a **judge model + token budget** (Meta's reference judge is a ~70–120B open
model; the gaia2-cli quickstart uses Sonnet).

### 3. Multi-turn scoring alignment

Gaia2 scenarios are multi-turn and the judge validates **per turn**, using
`send_message_to_user` as the turn boundary. S-ORA's runner today runs "until
idle" then does a single final `validate()`. `AreTransport` already replies via
`send_message_to_user`, so it's partway there, but the run/validate loop needs
turn awareness to match the judge's per-turn structure.

### 4. Cannot ride ARE's official `gaia2-run` CLI (but don't need to)

The official CLI hardcodes `assert agent is None or agent == "default"`
(`benchmark/cli.py:451`), so S-ORA drives scenarios through its own runner. Two
routes exist, and — corrected from the first draft — **a leaderboard submission
does *not* require the HTTP container:**

- **(a)** S-ORA's own runner + self-judge, exporting each scenario with
  `JsonScenarioExporter.export_to_json_file(..., trace_dump_format="hf")` into
  `<out>/standard/<config>/` plus a per-config `output.jsonl`, then uploading
  with the standalone `are.simulation.benchmark.gaia2_upload_script` (no
  container, no default-agent CLI). This is **both** the controlled research
  comparison **and** a leaderboard-grade submission. (There is no `export_ln`
  API — the earlier note using that name was wrong; the format is HF-JSON.)
- **(b)** wrap S-ORA behind the **gaia2-cli container HTTP contract**
  (`gaia2-cli/shared/gaia2_adapter_base.py`: `/notify`, `/events` SSE,
  `/execute_action`, `/health`). Only needed for gaia2-cli's own trace
  viewer/repro flow or for Agent2Agent orchestration — **not** for a plain
  core-capability submission.

### 5. Batch / dataset plumbing

S-ORA runs one scenario per process. A benchmark run needs: pull the 800 (+320)
scenarios from HF (`gaia2-cli/scripts/export_hf_to_json.py` exists), iterate
with per-scenario env lifecycle, capture traces, and aggregate a per-capability
report.

### 6. Augmentation splits are extra scope

*Agent2Agent* needs ARE to spin up app-agents (a2a mode is default-agent-only)
and *Noise* needs tolerance to injected tool failures / API changes (S-ORA's
`invoke` would need retry / failure handling). Reasonable to defer these — the
headline "Gaia2" number is the core 5 capabilities (800 scenarios).

### Fair-comparison caveats (design, not code)

- Use the **same underlying model** as the baseline for an apples-to-apples
  claim.
- The paper's ReAct scaffold injects queued notifications *pre-LLM-call*; S-ORA
  surfaces them via poll-on-observe + hard interrupts. That's not a confound —
  it's precisely the thing being tested — but the writeup must be explicit
  about it.

## The Gaia2 scenarios (light summary)

- **800 core scenarios** = 5 capabilities × 160: **Execution, Search,
  Adaptability, Time, Ambiguity** — spread across **10 universes**
  (pre-populated simulated user environments) and **11 apps**
  (AgentUserInterface, EmailClient, MessagingApp, ChatsApp, Calendar, Contacts,
  Shopping, Cab, City, RentAFlat, FileSystem).
- **+320 augmentation** = 2 × 160 on the *Gaia2-mini* subset: **Agent2Agent**
  and **Noise**. Total validation set = **1120**; equal weight per capability
  in the final score.
- **Gaia2-mini** = 160 scenarios (32 per core capability) — use this to
  validate the harness cheaply before the full run.
- **Where to read them:** HF dataset `meta-agents-research-environments/gaia2`
  (each scenario is a JSON: config + event timeline + oracle events); ARE docs
  `docs/user_guide/gaia2_evaluation.rst` and `docs/foundations/scenarios.rst`;
  the paper (32pp). The eval doc quotes one worked example per capability.

## Predicted outcome

Baseline numbers from the paper's Table 2 (ARE's ReAct scaffold, pass@1).
Overall: **GPT-5(high) 42.1 · Claude-4-Sonnet-Thinking 37.8 ·
Claude-4-Sonnet 34.8 · GPT-5(low) 34.6 · Kimi-K2 ~21**. The revealing part is
the per-capability profile — the columns where everyone falls off a cliff:

| Capability   | Claude-4-Sonnet | GPT-5 (high) | Signal |
|--------------|-----------------|--------------|--------|
| Execution    | 57.9            | 69.2         | strong |
| Search       | 59.8            | 79.6         | strong |
| Adaptability | 24.2            | 51.9         | weak / mid |
| Time         | 38.1            | 40.4         | fragile (inverse scaling, "instant vs default time") |
| **Ambiguity**| **8.1**         | **0.0**      | **collapse** |
| Noise        | 27.7            | 35.4         | weak |
| Agent2Agent  | 27.9            | 17.9         | mid |

### The residual prior, properly scoped

It is tempting to read the paper as "scaffolds don't move performance, so
S-ORA won't either." That over-reads the evidence. The paper's broad claim is
that Gaia2's hardest failures are "intrinsic to model capabilities rather than
the scaffold," but the *evidence for the scaffold half* — the PTC ablation
(Table 6, App. B.3.2) — is narrow in three ways that matter here:

- It covers only **2 of 7 capabilities** (Execution and Time); S-ORA's target
  splits, **Adaptability and Ambiguity, were never in the ablation.**
- It tests a **single** alternative, **Parallel Tool Calling** — an *efficiency*
  variant that batches calls *within the same ReAct decision-making*, not a
  different control structure.
- PTC's own finding is that Time stays hard because the deficit is *sequential
  reasoning / temporal planning*, not a lack of parallelism.

The paper then closes the section with: *"these results confirm that our
qualitative conclusions are not artifact of the scaffold and that more research
on completely novel orchestration is needed."* So the ablation does **not**
bound S-ORA — S-ORA *is* a novel orchestration, of exactly the kind the paper
invites, aimed at capabilities the ablation never tested.

What the model-capability prior *does* legitimately bound is narrower and worth
stating precisely: S-ORA cannot make the model reason better, so it gains where
a task's deficit is **control structure** (a mechanism to pause and ask; not
acting on stale state; waiting for a timed event) and is bounded where the
deficit is **model reasoning** (recognizing a task is ambiguous in the first
place; constructing long sequential plans). S-ORA's differentiators sit
squarely on the control-structure side.

On the strong suits (Execution / Search), where the deficit is neither — the
baselines already do well and don't need S-ORA's machinery — expect
**near-parity plus efficiency wins** (off-cycle inference + not re-deriving
plans → fewer / cheaper model calls), the one place S-ORA's expected effect
does resemble PTC's "efficiency, not accuracy" result.

### The S-ORA-specific thesis — why this isn't just PTC

S-ORA's differentiators don't parallelize the same decisions; they change *when
and whether the agent acts, waits, or asks*. That maps precisely onto the three
capabilities where baselines collapse:

- **Ambiguity (0–8%):** S-ORA's `await-input` / `blocked` machinery makes
  "pause and ask for clarification" first-class — exactly what this capability
  rewards and what ReAct structurally lacks. Largest upside; because the
  overall score is equal-weighted, even modest absolute gains here move the
  headline number. *Caveat:* the machinery only *enables* asking — the Reason
  strategy still has to *recognize* the ambiguity, which is a model / prompt
  decision.
- **Adaptability (24–52%):** interrupt-driven replanning on new information
  (the Monday→Tuesday follow-up demo) is the canonical case — S-ORA discards
  stale in-flight inference and re-plans against updated state instead of
  acting on stale info. Plausible real gains.
- **Time (~30–42%, fragile):** blocked-on-signal + off-cycle timing composes
  cleanly with "wait N minutes, then act if no reply" — the exact pattern the
  Time split probes and where baselines show inverse scaling.

### Net prediction

With a faithful harness *and* a Reason strategy tuned to actually exploit
ask / wait / replan, expect **flat-to-slightly-up on Execution/Search, and the
real defensible story on Adaptability / Time / Ambiguity** — i.e. a *smoother
capability profile on the dynamic / async axes* rather than a dramatic jump in a
single overall number, delivered at **equal or lower model-call budget**. The
overall pass@1 could plausibly rise above a *same-model* ReAct baseline, driven
mostly by the Ambiguity / Adaptability / Time columns. An overall gain is fully
consistent with the paper — which explicitly calls for novel orchestration and
never ablated these capabilities — rather than in tension with it. The honest
bound is not "scaffolds don't help" but "S-ORA helps on the control-structure
share of each task, not the model-reasoning share," so the size of the gain is
what remains to be measured.

The most publishable framing is therefore **not** "S-ORA vs GPT-5" but a
**controlled, same-model comparison: S-ORA vs ARE's ReAct scaffold**, reported
per-capability — because that's where S-ORA's architecture is designed to show
up, and it isolates the scaffold contribution the paper says shouldn't exist
(making a genuine per-capability gain on the dynamic axes the interesting
result).

**Honest risk:** the Ambiguity upside is partly a test of S-ORA's *strategy
prompting*, not just its machinery — if `DefaultReasonStrategy` never decides to
ask, the `await-input` capability sits unused and that column stays near
baseline.

## Suggested sequencing

1. Gap #1 (AGENT-event logging) is already done — see the regression test.
   Close gap #2 (attach the judge via `preprocess_scenario`) and score a
   *single* known Gaia2 scenario end-to-end — the correctness gate.
2. Add gap #3 (per-turn alignment) and gap #5 (batch runner), then run
   **Gaia2-mini (160)** for a cheap full-profile read.
3. Run the **core 800** with a same-model ReAct baseline for the controlled
   comparison; report per-capability deltas.
4. Defer Agent2Agent / Noise (gap #6) and the official HTTP-container
   submission (gap #4b) unless a leaderboard entry is the goal.

## Update (2026-08-05): remaining steps to a full HF submission

The code plumbing from the sequencing above is now built and tested token-free
(scenario loading, judge attach, turn-aware run-to-completion, the batch runner,
and byte-compatible HF-format artifacts + upload-script compatibility — the last
locked by `tests/test_gaia2_upload_compat.py` against the *installed* uploader).
The single-scenario correctness gate (gap #2/#3) has been run against a real
model; it scores end-to-end. What remains before a *trustworthy, comparable*
full submission is the following ordered list. Steps 1 and 3–4 are the substance;
the HF upload itself (step 5) is a thin, optional publication step.

1. **Iteration / sub-goals (code — the one real blocker).** S-ORA's
   plan execution can't loop today, so a multi-item task ("for each of these
   emails…") collapses each iterated step to a *single* tool call and
   **under-counts against the oracle event graph**. Until ADR-0022's sub-goal
   iteration lands (mechanical `len(collection)` fan-out for uniform maps;
   re-`infer()` for open continuations), a full sweep would produce artificially
   low, non-comparable scores — so the full run is deliberately held on this. This
   is the only remaining engineering blocker; everything else below is running or
   reporting.

2. **Meter the smoke run (recommended, cheap).** Instrument the `--limit 3`
   batch run to record per-scenario **LLM calls + input/output tokens** for both
   the agent and the judge, then extrapolate a real cost from measured numbers
   instead of the a-priori estimate below. This turns "low thousands ±" into a
   budget you can actually approve before committing to the full sweep.

3. **Full S-ORA sweep.** Run all **five core capabilities** (Execution, Search,
   Adaptability, Time, Ambiguity) with `--num-runs 3` (pass@1 protocol) on
   `--split validation` (the test split is private), all into **one absolute**
   `--output-dir`. Emits `<out>/standard/<config>/output.jsonl` + HF trace files
   per scenario. Hold the **judge model constant** and record which one (it is a
   disclosed parameter of the submission, not a free knob — see below).

4. **Same-model ReAct baseline (M6 — the actual scientific result, *no HF*).**
   Run ARE's default scaffold on the **same agent model and same judge**
   (`uvx --from meta-agents-research-environments are-benchmark gaia2-run
   --hf-dataset meta-agents-research-environments/gaia2 --model <same> --agent
   default …`) and compute the **per-capability delta table, S-ORA − ReAct**.
   This is entirely local and needs no HuggingFace at all — it's the deliverable.
   Report **two metrics per capability: accuracy (pass@1) *and* cost-efficiency
   (LLM calls & tokens per task)** — S-ORA is expected to win on efficiency even
   where accuracy is at parity (on the ARE email example it used 4 LLM calls, one
   discarded to an interrupt, vs a ReAct implementation's 11, because plan
   amortization means most decision-cycle phases are mechanical, not model calls).

5. **HF upload — only if leaderboard visibility is wanted (optional, publication
   only).** HuggingFace is a public model/dataset host; Meta's Gaia2 leaderboard
   reads from *submitted datasets*. Note the judging already happened locally
   during step 3 — the upload publishes *already-scored traces*, it does not
   re-judge. Concretely:
   - Create a free HF account and a **write** access token (Settings → Access
     Tokens); `huggingface-cli login` (or set `HF_TOKEN`).
   - Run the standalone uploader over the step-3 output tree:
     `uv run python -m are.simulation.benchmark.gaia2_upload_script
     --input_dir <out> --output_dir <stats-dir>
     --model "S-ORA/<agent-model>" --split validation
     --hf_upload <org>/<dataset>`. It reconstructs results from the tree, prints a
     validation report, writes `computed_stats.json`, and `git push`es a
     **public** dataset repo the leaderboard consumes. (Omit `--hf_upload` to get
     the local validation report *without* publishing anything.)
   - Because the uploader resolves each row's `trace_id` via a bare
     `os.path.exists` from *its own* cwd, `--output-dir` in step 3 must be
     **absolute** (the batch runner now enforces this) — otherwise every trace is
     silently dropped from the submission.

**On the judge model (why it's a disclosed parameter, not a fixed one).** Gaia2
judging runs on the *submitter's* side before upload — you score locally, then
upload the already-scored traces plus `computed_stats.json`. There is no central
re-judging step at submit time, so the judge model isn't mechanically pinned; the
reference judge (Meta's ~70B open model, or Sonnet as the quickstart uses) is a
default/recommendation, and comparability is meant to be *recoverable* from the
full event-log traces that get uploaded (anyone can re-judge them). Practically:
disclose the judge you used, and — critically for *our* controlled comparison —
hold it **identical across the S-ORA and ReAct arms** so any judge bias cancels
in the delta. Whether the public leaderboard *disqualifies* a non-reference judge
is a policy question not verifiable from the tooling; it does not affect the local
delta table, which is the real deliverable.

## Update (2026-08-05): estimated cost of a full run

Order-of-magnitude only — "which power of ten," not a quote; step 2 above (meter
the smoke run) replaces it with a measured number. Each scenario run costs **two**
model bills: the agent doing the task and the judge scoring it. Assumptions:
Sonnet-class pricing (~$3 / M input, ~$15 / M output) for both; a Gaia2 scenario
is multi-turn with large tool/app context in each call.

| Component | Rough assumption | ~Cost / scenario run |
|---|---|---|
| Agent (S-ORA) | ~15–25 LLM calls, ~15k input each | ~$1.0 |
| Judge | per-oracle-event checks, ~15 calls | ~$0.5 |
| **Per run (either arm)** | | **~$1.5** (plausibly $0.5–$3) |

- **S-ORA full submission** (800 core × `--num-runs 3` = **2,400 runs**):
  ≈ **$3–4k**; order-of-magnitude **low thousands** ($1k–$10k depending on model
  and run verbosity).
- **ReAct baseline arm** (M6, another 800 × 3): similar order, but its **agent
  side runs higher per scenario** — ReAct pays ~1 call per action and re-sends
  cumulative history each step, where S-ORA amortizes (the 4-vs-11 call result
  above). So the controlled *study* (both arms + judge on each) is ≈ **$7k**,
  order-of-magnitude, range roughly **$2k–$15k+**.
- **Gaia2-mini harness validation** (160 scenarios): ≈ **$250** at one run per
  scenario, ≈ **$700** at `--num-runs 3`. Do this before the full sweep.

**Big levers:** the **agent model** (Opus multiplies this several-fold; Haiku
divides it), **`--num-runs`** (1 for a cheap profile vs 3 for a submission), and
**mini (160) vs full (800)**. The judge model is a third lever — a smaller judge
lowers cost but adds measurement noise (mitigated by the `--num-runs 3`
averaging). Meter the smoke run before approving the full budget.

## Update (2026-08-05): remaining TODOs for meaningful numbers

The integration is complete and *scores at single-scenario scale*, but the center
of gravity for a *meaningful* number has moved to the S-ORA runtime. Splitting the
remaining work:

**Runtime gaps** (the leverage — in `src/sora/`):

- [x] **Iteration / sub-goals (ADR-0022) — correctness gap.** Plan
  execution can't loop, so multi-item tasks collapse each iterated step to one
  call and under-count against the oracle graph. Numbers are *low/wrong* until
  fixed. The one confirmed blocker.
- [ ] **Reason strategy actually exploits ask / wait / replan — exploitation
  gap.** The `await-input` / `blocked` machinery only *enables* asking; if
  `DefaultReasonStrategy` never decides to ask/wait/replan, the Ambiguity /
  Adaptability / Time columns sit at baseline. Numbers are *valid but flat* until
  tuned. Co-equal with iteration/sub-goals — one makes the score correct, this
  makes it interesting.
- [ ] **Enumerate the gap list (n=1 today).** Only one capability's gate has run
  (iteration was what it found). Run the **five single-scenario gates** (one per
  capability, via `scripts/fetch_scenario.py` + `run_benchmark.py`) to surface the
  rest *before* committing the ~$3–4k full sweep.

**Integration gaps** (not runtime, but they gate trustworthy numbers — in
`examples/gaia2/`):

- [ ] **Per-scenario isolation at sweep scale.** S-ORA's model — one fresh
  `AreSimulation` per scenario — *matches ARE's own default* (`executor_type=
  "thread"`, one process; isolation by per-scenario app reconstruction +
  `deepcopy` per run, not by sandbox — `scenario_executor.py`). So the residual
  risk is narrow: any **S-ORA-side module-global** (runtime or `are_sim` adapter)
  a fresh `AreSimulation` fails to reset would bleed silently across 160–800 runs
  and corrupt scores without crashing. The fallback is ARE's opt-in
  `executor_type="process"` (subprocess-per-scenario) — a known path, not novel
  work. **De-risk cheaply:** run one scenario twice in a single process vs. once
  each in two fresh processes and assert identical scores/traces — any divergence
  *is* a bleed and points at the offending global. Audit import-scope state before
  building the subprocess path.
- [ ] **Real HF upload round-trip.** M5 is verified by reading the uploader +
  testing its parse half only — no trace has made the full trip to HF and back.

Status caveat: call the integration "complete in the small," not done — either
integration gap above can poison the numbers with no runtime gap in sight.

## Update (2026-08-26): ARE's case-sensitive verdict parse — patched, and disclose it

**Any Gaia2 number produced before 2026-08-26 with an OpenAI-family judge is not
interpretable.** ARE's soft checkers parse the judge model's verdict with a
case-sensitive substring test for `[[True]]`. OpenAI-family judges write `[[true]]`,
so `LLMChecker.__call__` records no vote, returns `None`, and `SoftToolJudge.compare`
reads that falsy `None` on the same code path as a genuine `False`. The judge's actual
answer — often an explicit pass, on the merits — is discarded, and nothing downstream
distinguishes it from a content failure.

It is worse than mis-scoring. `turn_condition_wrapper`
(`are/simulation/scenarios/utils/turn_conditions.py`) gates each turn's *release* on the
same verdict and calls `env.stop()` when it is falsy. The gate fires on the agent's
`send_message_to_user`, so one discarded verdict on turn 0 ends the scenario at the
agent's first reply. Every later turn's oracle writes are then reported by the
write-count gate as `missing (oracle did, agent did not)` — work the agent was never
given the chance to do, indistinguishable in the output from work it failed to do.

**The casing is not the model's** (corrected 2026-08-26 — an earlier version of this note
blamed OpenAI-family judges, and that was wrong). Both engines ARE ships end
`chat_completion` with `res.replace("False", "false").replace("True", "true")`
(`agents/llm/litellm/litellm_engine.py`, `agents/llm/hf/hf_engine.py`) — a JSON-shaped
normalization of *agent* output that also rewrites every judge response in transit. Since
`create_judge_engine` returns a `LiteLLMEngine` unconditionally, `"[[True]]" in response`
is **unsatisfiable on the shipped path, for every model**: the checker is handed
`[[true]]` no matter who answered. `[[False]]` is mangled the same way, so the failure
branch is dead too and the checker's only reachable return is `None`.

So the four `[[True]]`-family checkers — `signature_checker`, `tone_checker`,
`sanity_checker`, `cab_checker` — are structurally incapable of returning a verdict, while
the `[[Success]]` family is untouched by the replace and works normally. Pinned as a
regression test against the real package
(`tests/test_are_sim.py::test_ares_own_engine_mangles_a_true_verdict_before_the_checker_sees_it`,
using LiteLLM's `mock_response`, so no network and no tokens).

Why it survives in a high-profile benchmark, given that it is *not* model-specific:

- **It only fires off the fast path.** `SoftToolJudge.compare` returns `True` early when
  `equality_checker` (exact match after normalization) succeeds, and only then runs the
  soft checkers. So the dead checkers are reached exactly when the agent's free-text
  differs from the oracle's — which reads as "the paraphrase was judged and rejected".
- **It is silent.** `None` is falsy, so it rejects on the same code path a genuine `False`
  takes; the graph judge records only a boolean (`judge.py`, `TOOL_JUDGE_REJECT`) with no
  reasoning. On a benchmark where agents are *expected* to fail, that FAIL invites no
  investigation.
- **It is common-mode.** Every agent scored by the same judge is penalized identically, so
  aggregate leaderboard ordering still looks sane; it shows up only by replaying a single
  event pair offline.
- **It is glue.** A dozen lines of marker parsing plus one `.replace` in an engine written
  for a different purpose. Nothing tests the two together, and the unit tests for
  `LLMChecker` feed it strings directly, bypassing the engine that does the damage.

**Patched here** by `relax_judge_verdict_case()` (`sora/adapters/are_sim.py`), applied
by `attach_judge` by default and logged when it fires. It relaxes *only* that
comparison's case — prompts, checkers and vote tallying are untouched — so it restores
the parse ARE's own prompts and unit tests intend rather than lowering the bar.
Both `run_benchmark.py` and
`batch.py` print which parse a run used and take `--strict-verdict-case` to reproduce stock ARE;
a sweep also records it per row in `output.jsonl` (`metadata.verdict_parse`), on every scored
record including the default — a row that does not say which parse produced it cannot be compared
with one produced by stock ARE.

**When reporting numbers, say which parse was used.** A score obtained with the parse
relaxed is a different artifact from one obtained under stock ARE, and the distinction
has to survive being pasted somewhere without the log. Remove the patch once ARE fixes
this upstream.

**Still unexplained:** `aug25` and `aug26` ran the same two-turn scenario with
near-identical turn-0 work and got *opposite* gate verdicts. The mangling is
deterministic, so the earlier "nondeterministic casing / coin flip" reading of that pair
is dead — whatever separates the two runs is upstream of the parse (most likely which
side of `equality_checker`'s fast path each run's args landed on) and has not been
established. Either way, no runtime A/B on a multi-turn email scenario was meaningful
before this patch.


## Update (2026-08-26): a Gaia2 scenario has a ~1000-second **real-time** budget

**ARE's event loop is wall-clock paced, so `scenario.duration` is a real-time budget for the
agent, not a property of the scripted world.** `Environment._event_loop` runs
`while time_passed() <= duration`, and each iteration is `tick(); time.sleep(1)` with
`time_increment_in_seconds = 1` — one simulated second per real second. `ScenarioImportedFromJson`
defaults `duration = 1000`, and the Gaia2 JSON scenarios in `scenarios/` do not override it (their
metadata carries `duration: null`, which leaves the class default standing). So **every scenario
here gives the agent ~16 minutes of wall clock, total, across all turns.**

Nothing in the runner enforced or reported this. `--max-wall-seconds` defaults to 1200, *above*
the real limit, so it never binds; it is a backstop for a hung judge, not the budget.

### How it fails, and why it reads as an agent error

Discovered on `logs/aug26-run3-qwen3-30b-64k.log` (qwen3-30b served locally). Eight model calls
totalling **930.8s** — two plan inferences at **405s** and **329s** — against a 1000s budget. The
environment's loop exited mid-turn-0. What happened next is the trap:

- The agent's remaining steps still *executed*. App methods are called directly, and
  `@event_registered` logs the event whether or not the loop is alive, so `send_email` and the
  final reply both landed and both appear in the trace as ordinary successes.
- **No later turn was ever delivered.** A Gaia2 turn is released by a `ConditionCheckEvent` that
  only the live loop ticks (`turn_conditions.wrapped_condition` counts the agent's
  `send_message_to_user` events and calls the per-turn judge). With the loop gone, that check
  never runs, `BaseJudge.state.turn_idx` is never incremented, and `validate()` reports
  `Validation called at turn -1 but nb_turns is 2`.
- The write-count gate then reports **every** oracle call of the undelivered turn as
  `missing (oracle did, agent did not)` — here `Calendar__add_calendar_event: 1` and
  `Calendar__delete_calendar_event: 6`.

That output is indistinguishable from a competent agent that simply did nothing after turn 0. In
this run turn 0 itself was **correct** (`turn 0: ok`), and the two calendar deletions the agent was
"missing" belonged to a follow-up email it was never shown.

**This is the second distinct upstream cause with the same output signature.** The first is the
judge-gate rejection above (a falsy verdict calls `env.stop()`); this is the clock. Both produce
"a whole turn's oracle writes missing", and neither is the agent. Before reading a missing turn as
a failure, establish which one it was.

### What is instrumented now

- `AreSimulation.timeline_expired()` — mirrors ARE's own loop-exit test (`time_passed() > duration`,
  so equality is still running). Not on the `Simulation` Protocol; only eval code reads it, same as
  `is_paused()`.
- `RunResult.timeline_expired`, printed by `run_benchmark.py` **first**, before the verdict, because
  it reinterprets every line under it rather than adding to them. `batch.py` records it in
  `output.jsonl` metadata when true, so an aggregate can exclude a truncated run instead of
  averaging in a number about the host machine's speed.
- `--scenario-duration SECONDS` overrides the budget, and announces the override in the run's own
  output.

### On raising the budget

Raising the duration does not hand the agent more of the scripted world. **No event is pinned to an
absolute timestamp** — `event_time` is `0.0` for all 46 events across the five capabilities — and
every event but the opening user message hangs off a dependency. What an event carries is
`event_relative_time`, a delay in seconds *after its dependency fires*, so shifting the budget does
not shift the schedule.

| capability | events | dep-free | `event_relative_time` values | duration |
|---|---|---|---|---|
| adaptability | 14 | 1 | 0, 1 | 1000 |
| ambiguity | 7 | 1 | 0, 1 | 1000 |
| execution | 10 | 1 | 0, 1 | 1000 |
| search | 2 | 1 | 0, 1 | 1000 |
| **time** | 13 | 1 | 0, **2, 19, 31, 65, 91, 121, 183, 221** | 1000 |

Four of the five are effectively "as soon as the dependency fires" (`1.0`). In
`adaptability/scenario_universe_21_5e0gvz` that includes the two `Emails.create_and_add_email`
events carrying the friend's reply: they fire off the agent's own `send_message_to_user`, not at a
time.

**Time is the exception, and it matters for a slow model.** Its six
`Calendar.add_calendar_event_by_attendee` events are released 31, 65, 91, 121, 183 and 221 seconds
after the opening message — matching a goal that opens *"some events will be added to the calendar
in the next four minutes ... immediately remove all the conflicting preexisting events"*. That
cadence is measured on the same one-second-per-real-second clock, so it is 3.7 minutes of **real**
time. A model whose plan inference takes 405s is inside a single call for that entire window and
meets all six arrivals at once, well after "immediately". **`--scenario-duration` does not fix
this** — the events are relative to the user message, not to the budget, so more budget only buys
time *after* the window has already been missed. Time is the one capability where model latency
changes what the scenario means rather than merely whether it finishes.

The other residual is *simulated-clock drift*. Sim time advances one second per real second from
the scenario's `start_time` (`2024-10-15 07:00:00` for all five), so a 3000s run leaves
`get_current_time` ~50 minutes further along than a fast run would. The margin is usually enormous:
`adaptability` asks for "this upcoming Saturday" and the oracle resolves it to `2024-10-19`, so the
phrase only reinterprets once the clock passes Saturday — about **four days**, or a duration near
330,000s. Drift bites only where a goal is phrased on a scale comparable to the run's own length.
Either way a number produced under an overridden duration is not comparable to a published one,
which is why the override announces itself.

**The honest framing for a slow local model: it is not a Gaia2 result.** ARE's baselines assume a
hosted model answering in seconds. At ~400s per plan inference, two plan calls consume three
quarters of a scenario's stock lifetime, and no amount of correctness recovers turns that were
never delivered. Report local-model runs as trajectory/behaviour checks, and keep scored numbers on
a model fast enough to finish inside the stock budget.

**Not an S-ORA defect, and the replan was right.** The 329s second plan inference was a *correct*
recovery: the first plan's `search_contacts({"query": "Film Producer"})` returned `[]` (ARE's
contact search is name-based, not job-based), the plan-defect check caught the dependent step
before any write, and the replan filtered the `Contacts` property instead and found the right
person. The cost of being right was a third of the scenario's life — a statement about the model's
latency, not about the recovery.
