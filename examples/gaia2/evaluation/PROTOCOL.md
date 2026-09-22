# Gaia2 evaluation protocol

**Status: pre-sweep, not yet a pre-registration.** This document is the tracked, citable statement
of how the Gaia2 comparison is run. It becomes the frozen pre-registration only when the acceptance
parameters are added to it and committed *before* any locked-acceptance payload is opened — a
protocol committed after the results are known is not a pre-registration. Until then it records
settled methodology and names what is still open.

It supersedes the untracked working note the evaluation README used to point at. Development
history — why a rule reached its current shape, superseded attempts, dated gate reports — stays in
the working notes; what a later reader needs in order to check a published number is here.

The evaluation set is the complete Gaia2 `mini` selection: 32 scenarios from each of the five
capabilities, 160 scenarios total.

## 1. Timing conventions

A run is produced under exactly one of three conventions, chosen explicitly and recorded on every
row as `clock_mode`. They are not interchangeable, and results are not comparable across them.

| `clock_mode` | Flag | Environment clock during a model call |
|---|---|---|
| `generation_free` | `--generation-free` | Frozen, resumed by zero — generation costs the scenario nothing |
| `token_charged` | *(charge model attached)* | Frozen, resumed by a modelled token charge |
| `wall` | `--wall-clock` | Never frozen — generation is billed at measured elapsed time |

**`generation_free` is the primary convention for the campaign.** It is the **legacy ARE in-process**
convention: ARE's default `simulated_generation_time_mode="measured"` pauses the environment and
resumes it with `completion_duration`, and no shipped ARE engine ever writes that field, so the
resume offset is always zero and every row produced through that harness had generation costing
nothing. The freeze here is a strict generalization of it, with the charge set to zero.

Describe it as **compatible with ARE's in-process convention**, never as "leaderboard-comparable"
without qualification: the newer `gaia2-cli` leaderboard runs on a wall clock, which is a different
convention again, so not every published Gaia2 row was produced this way.

**`token_charged` is blocked.** Its coefficients failed their own range and drift gates on
2026-09-21 (§9). It remains implemented and selectable as a sensitivity arm, and the failure is
recorded against the frozen coefficients rather than fitted away.

Selecting a convention is explicit because it cannot be inferred safely. The harness once derived
freezing from whether a charge model was attached, so `charge=None` meant wall clock on both arms —
and on the ReAct side the engine explicitly billed each crossing at its measured elapsed time. A
sweep started with no charge would have been a wall-clock sweep wearing no label saying so.

**A contradictory configuration is refused, not ranked.** A charge model supplied *alongside*
generation-free once resolved to `token_charged`, on the reasoning that the caller got the stricter
of the two. The arms did not agree on that reading — S-ORA billed the crossing while ReAct's engine
charged zero — so the label was right for one arm and wrong for the other, on a row recording no
sign of the disagreement. `resolve_clock_mode` now raises, and both arms resolve their mode at
construction.

### 1.1 The token-charged policy, when it is used

The trajectory policy is `serialized_sum`. The environment clock is frozen while one or more
physical agent-model crossings are in flight. Each crossing — including parser repair, malformed
output retry, and a failed crossing without usage — receives its own non-negative frozen token
charge. The final outstanding crossing releases the environment after every crossing's charge has
been banked, so overlapping inference is charged as one lane. Tool operations receive no synthetic
duration: ARE's apps expose no authored execution latency, so local Python tool time is an
implementation artifact rather than a modelled resource.

This is deliberately conservative against S-ORA's asynchronous architecture, but it does not erase
the architecture's call-amortization effect: every physical crossing pays the same endpoint
coefficient model on both arms.

`llm_charged_union_seconds` is a declared sensitivity and never changes the trajectory. It is built
online on a settled parallel frontier: calls admitted before any sibling settles share a coordinate,
and a later admission starts at the furthest charged endpoint already settled in the current
uninterrupted freeze. This prevents causal backdating. It is not a reconstructed causal DAG — a
causally independent call admitted after an unrelated sibling settles is placed conservatively at
the settled frontier — so `charged_seconds - llm_charged_union_seconds` is a *lower bound* on what
ideal parallel inference could save.

It is reported as unavailable (null) for any run whose crossings did not all share one time axis; a
crossing admitted after the online judge stopped the environment carries no charged coordinate. An
aggregate containing one unavailable run is itself unavailable, never a sum over a shorter count.

## 2. Event delivery under a freeze

Scheduled scenario events are dispatched **synchronously at each final resume**, and not at all
during an individual frozen inference epoch. When the last outstanding crossing releases the
environment, the harness ticks ARE's loop before returning to the agent, so everything the frozen
interval skipped — due events and time-based notifications alike — is delivered before the agent can
begin its next crossing.

This is a blocking-inference execution model. It does **not** measure native mid-inference
responsiveness: an event authored inside an epoch is observed at the end of that epoch, late by up
to the epoch's charge, and the agent cannot interrupt, observe, or spawn an activity from it while
the crossing is in flight. Resume-time dispatch bounds that lateness at one epoch. Without it,
delivery would race ARE's one-second loop against the agent's next crossing, so lateness would
accumulate across a chain of calls and depend on host thread scheduling — making Time results
non-reproducible rather than merely conservative.

Because overlapping crossings are charged as one lane, the epoch is the union of everything in
flight: an event authored during the first of two overlapping crossings waits for the second too.

## 3. What the Time capability measures

The judge's `post_event_tolerance_seconds` is 25 seconds, measured from the oracle event's scheduled
time. Under a freeze the agent spends that budget on two things: delivery lateness (§2) plus the
cost of its own reaction.

The reaction is not a single crossing. Reacting to a scheduled event costs **two inferences** — a
condition evaluation, then a fresh sub-plan inference for the `then` branch, re-inferred per firing
— before the resulting call is dispatched mechanically. Measured on the `time` smoke logs at
`medium` reasoning effort against the frozen charge coefficients, that pair costs a median of
**≈5.3 charged seconds**, worst observed ≈11.9 s.

So a Time result is a measurement of reaction-plus-delivery against a 25-second budget, not of
reaction alone. When reporting, state the budget and both terms: an event missed by a margin smaller
than one reaction pair is a statement about the charge model's calibration as much as about the
agent.

Under `generation_free` both terms go to zero, which is the convention's main cost. Time is the only
split whose events fire mid-scenario and the only one the judge's timing checks gate. It does not
follow that nothing else is affected — freezing changes event scheduling, and removing generation
latency means **S-ORA's asynchronous responsiveness is not exercised by these numbers**. What is
reproducible here is the timing convention given a trajectory; model sampling remains stochastic.

If the charge model later passes its gates, charged Time is reported as a **separate sensitivity**
in a separate output root. It must never be spliced into a generation-free five-capability headline:
a result mixing clock modes has no coherent mean.

## 4. Watchdogs and truncation

Two watchdogs stop a run: the wall cap, and a scoring pass that sits paused past 180 seconds. Both
are recorded in one nullable field, `harness_truncation` ∈ {`wall_clock`, `judge_stall`, absent},
and **pass@1 excludes any value** — the score measures how well the agent did, not how far it got
before a cap.

One field rather than a boolean per watchdog is deliberate: the scoring rule needs the union, and a
pair of flags invites filtering on one of them. `terminal_cause` cannot carry it either, since it
reports both watchdogs *and* an expired timeline as `"timeout"`. A watchdog stops a run through
`Environment.stop()` — the same lever a timeline expiry and a rejected turn pull — so without a
distinct field a truncated run is indistinguishable from a completed one.

Both arms use one attribution rule, which prefers `judge_stall`, because the cap can elapse
*because* the judge stalled and not the reverse.

`--max-wall-seconds` defaults to **3600** under `--generation-free` and **1200** otherwise,
announced at startup and recorded on every row. Both campaign entry points resolve it from one
shared rule, so no CLI can offer the flag while keeping a cap that truncates the arm the raise
exists for. A frozen clock removes generation from the scenario's budget but not from the
operator's: real per-scenario time becomes the scenario timeline *plus* the whole of generation,
where a wall-clock run is bounded by the timeline alone. Projected from measured generation time on
the three pinned Time scenarios, five of six S-ORA runs exceed 1200 s and none of six ReAct runs do
— a cap firing on one arm only is not a shared condition. That projection assumes each environment
consumes its full 1000 simulated seconds; **the canary must establish actual runtime**, and the
value is revisited if it does not hold.

`--generation-free` is a campaign mode: `batch.py` and standalone `react_driver.py` offer it;
`run_benchmark.py` and the `evaluation` CLI deliberately do not, since their timing output is never
promoted into a campaign result.

## 5. The headline is a gated claim

`aggregate()` reports `overall` only when **every** one of these holds, and `headline_withheld`
names each that does not:

- all five core capabilities present, each with a comparable pass@1;
- one clock mode across them;
- one scenario-manifest digest across them, matching the manifest the report is run against;
- every scenario that manifest pins actually scored — "scored" meaning exactly what pass@1 means by
  it, so a capability whose every run was truncated cannot read as covered.

Coverage is the one condition the rows cannot answer alone: they record what ran, never what was
supposed to. The headline therefore **requires** `--scenario-manifest`, and without one it is
withheld as unverified rather than assumed. The ungated mean survives as `exploratory_mean` for
reading a sweep in progress; it prints as "not a reportable result" and must never be quoted as one.

The digest condition applies to rows recording *no* digest as well. Comparing only the digests that
are present leaves one shape unguarded: five fully scored capabilities, one clock, full coverage,
and nothing whatsoever tying the results to the manifest being reported against. Missing provenance
is not evidence of agreement.

## 6. A pair is audited before it is paid for

Two arms are a comparison only if they differ in the architecture and nothing else. Before an arm's
first scenario, its counterpart is checked for a matching scenario selection, `clock_mode`, and
`model_profile`. The endpoint matters specifically because decode rate is a property of the endpoint
rather than the model, so two internally homogeneous arms could otherwise be paired across timing
conventions or endpoints and report a difference that is the sum of the architecture and the
discrepancy.

A counterpart recording none of these fields is **refused, not assumed to match**: both arms are
written by this harness, so an absent field means the rows predate the provenance. The check runs
before the spend because the only repair after the fact is to re-run the pair.

A generation-free run reports no charge-accounting mismatch. The clamp-vs-raw-anomaly comparison
polices this harness's own cache bookkeeping; a generation-free run applies no clamp and computes no
charge, so a provider usage anomaly would be reported as an accounting mismatch naming the wrong
component. The check is restricted to `token_charged`.

## 7. Inputs, provenance, and frozen artifacts

The paid scenario set must be supplied through `batch --scenario-manifest`. The manifest pins the
dataset repository, immutable dataset revision, split, ordered exact scenario IDs, and its own
canonical SHA-256 digest. Batch validates the complete selection before opening paid artifacts,
records the digest on every result row, and refuses to pair with a counterpart arm whose digest or
scenario/run matrix differs. `--limit` remains smoke-only and is mutually exclusive with the
manifest.

The selection is [`campaigns/paper2027/mini-validation.json`](campaigns/paper2027/mini-validation.json):
all 32 `mini` IDs for each of `execution`, `search`, `adaptability`, `time`, and `ambiguity`, grouped
under their actual capability names. At the pinned revision every selected scenario appears in the
corresponding full capability configuration in the same relative order with an identical serialized
payload, so runs load each capability configuration and apply the shared manifest, preserving
capability labels and the five-way aggregate while evaluating exactly Gaia2 mini.

Frozen artifacts, verified 2026-09-21:

| Artifact | Digest | Kind |
|---|---|---|
| [`campaigns/paper2027/mini-validation.json`](campaigns/paper2027/mini-validation.json) | `cc6ebb08d388cc0e4dee635c0a031493c280972c033ba0712577413757515e3b` | canonical (parsed JSON) |
| [`charge_model.json`](charge_model.json) | `9065b1dc6bda1aa832b64867b0c0c68885930f7c74bb08b6c48ad140dcbca585` | raw bytes |
| [`campaigns/prompt/snapshots/pre-optimization-control.json`](campaigns/prompt/snapshots/pre-optimization-control.json) | `56c1cd60b049c73d12f258e03e59b31d3b00ca24cff31f647de38a5038276129` | canonical rendered prompt rows |

The manifest digest is canonical over parsed JSON, so reformatting does not change experiment
identity — renaming does, because the name is part of the canonical content. The two raw hashes
guard their complete files byte for byte. The charge-model identity and its semantic digest are also
written into every charged or frozen-profile wall-clock result.

**Prompt provenance is per record, not per report.** The two arms of a prompt comparison run
different prompts by construction, so a single report-level snapshot can only ever describe one of
them. Instead:

- A run declares a snapshot (`--prompt-snapshot`, defaulting to the control above), and the live
  renderer is checked against all 28 of its rows *before* any credential, provider, or scenario is
  opened. A mismatch names the rows that moved and refuses the run.
- The verified identity and digest are stamped on every record the run writes, and resuming a
  checkpoint written under other prompts is refused.
- Reports read those digests back off the rows. An arm with no digest, or an arm that mixes
  digests, withholds `mean_paired_score_delta`, its bootstrap interval, and the acceptance-expansion
  verdict, naming the reason in `paired_comparison_withheld`; the per-pair rows and an
  `exploratory_mean_paired_score_delta` remain for diagnosis.
- Paper sweeps declare no snapshot — they are not comparing prompt versions — but the S-ORA arm
  still records the digest it rendered, and the headline is withheld when capabilities within one
  sweep disagree about it. The ReAct arm records none, because it does not use these prompts.

**Operating point.** The sweep runs `gpt-5.4-high-paper`. Charge coefficients are keyed by model
(`gpt-5.4-2026-03-05`) and therefore shared with `gpt-5.4-medium-prompt`; sharing coefficients does
**not** make a result obtained at `medium` a validation of `high`. Range and drift audits must be run
at the profile the sweep will use, on both arms.

## 8. Declared diagnostics

Every scenario result carries:

- `llm_round_trips` — physical agent-model crossings, including repairs and retries;
- `llm_max_in_flight` — maximum simultaneous physical crossings on the host monotonic axis;
- `llm_overlapped_round_trips` — crossings sharing positive host-wall duration with another;
- wall-time sum and union;
- token-charge serialized sum and settled-frontier union;
- `clock_mode`, `harness_truncation`, `terminal_cause`, and the `max_wall_seconds` in force.

All seven agent-model entry points resolve through `cycle.procedural` and therefore the one
`ProceduralMemory._llm` instance the Gaia2 harness decorates. Online judge calls are a separate
scoring resource and are not counted as agent inference.

**Ex-ante overlap expectation.** In the prior 15 recorded Gaia2 runs, all of which created one
activity, `plan/v1` accounted for about 85% of inference wall time; the off-cycle `retirement/v1`
judge accounted for roughly 78 of 2,084 seconds (≈3.7%), and `relevance/v1` did not fire. Under those
conditions the serialized-minus-union difference should be small, and smaller on the charged axis
because retirement prompts are token-light. This is an expectation, not a bound. A materially larger
diagnostic is a finding that falsifies the single-activity premise — never a reason to revise frozen
policy or coefficients after seeing results.

## 9. Operational gates

Before a sweep, run paired real-call range and drift audits for both endpoints and a Time smoke. The
smoke must confirm delivery of all scheduled events, nonzero concurrency when the known
retirement/plan overlap is exercised, and negligible local tool execution time relative to simulated
elapsed. Repeat the drift audit over completed sweep logs.

**Record a failure against the frozen coefficients; never re-fit them inside the sweep.**

**Gate status, 2026-09-21: failed.** Both audits failed on both endpoints, blocking the
token-charged convention. The coefficients were not re-fitted. Compact evidence is preserved in
[`gates/2026-09-21/`](gates/2026-09-21/) — four paired `llm_calls.jsonl` row sets, both preflight
corner grids with their manifests, and the scenario selection — attested by its
[`SHA256SUMS`](gates/2026-09-21/SHA256SUMS). Kimi's drift failure is at the target operating point
and stands. The gpt-5.4 measurement was taken at `gpt-5.4-medium-prompt` while the coefficients were
fitted at `gpt-5.4-high-paper`, so reasoning effort is confounded with endpoint drift and that half
is diagnostic only — an upper bound of unknown tightness rather than a measured error.

The rows are tracked rather than left in a working tree because paid measurement does not reproduce:
decode is not deterministic, and the rate is a property of the endpoint rather than the model, so
re-running replaces this evidence instead of recreating it.

## 10. Comparability breaks

A result is comparable only with results produced on the same side of every line below.

| Date | Break |
|---|---|
| 2026-09-12 | Default billing sheet moved from `2026-09-02.json` to `2026-09-12.json` when the Kimi endpoint was repinned from DeepInfra to Venice. Changes cost accounting, not the frozen timing coefficients. |
| 2026-09-16 | Gaia timing moved from provider wall latency to the frozen token-charged clock. Timing-gated results before and after are not comparable, though the semantic prompts were unchanged. |
| 2026-09-19 | Commit `65876fc` showed the revalidation judge the armed conditions — a behavioural change that required re-freezing the then-combined `baseline.json`. A sweep run after it is not the same experiment as one before. |
| 2026-09-21 | `generation_free` adopted as the primary convention; `token_charged` demoted to a blocked sensitivity arm. Timing-gated results are not comparable across conventions. |
| 2026-09-22 | The scenario manifest was renamed `aamas2027-` to `paper2027-`, moving its canonical digest from `123b92db…` to `cc6ebb08…`. The dataset, revision, split, and every scenario ID are byte-identical, so no result is invalidated — but the digest is the identity key each row records and each paired-arm audit compares, so a row carrying the old one does not pair with a row carrying the new one. No paid rows exist under `123b92db…`. |

The former combined `baseline.json` was also re-cut on 2026-09-21 to add `paper2027` to
`kimi-k2.5-prompt`'s campaign list. That change survives in the separate `campaign.json` mirror and
is campaign-membership metadata only — no request parameter, operating point, prompt, judge profile,
or charge-model figure moved — so it costs no comparability.

## 11. Still open

Named here so that a reader can see what this protocol does *not* yet fix:

- **Acceptance parameters.** Repeats, decision thresholds, the expansion rule, and the selection rule
  are not yet stated. They must be added here and committed before the locked-acceptance payloads
  are opened, which is what turns this document into a pre-registration.
- **Judge model.** Selection and configuration are not frozen.
- **The `gpt-5.4-high-paper` re-audit**, which removes the reasoning-effort confound from the failed
  charge gate.
- **Freezing the campaign configuration.** Prompts have an immutable-snapshot mechanism;
  `campaigns/prompt/campaign.json` is only a mirror of the live profile, judge, and charge files and
  therefore records nothing historically. That is correct while the judge is still open, but
  pre-registration needs the configuration pinned the same way — either a snapshot of it, or its
  digest recorded here.
