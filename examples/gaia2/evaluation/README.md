# Gaia2 evaluation harness

This offline-first harness shares profiles, metering, price sheets, checkpoints, budget gates, and
report serialization across two campaigns:

- `prompt` evaluates frozen S-ORA prompt configurations on deterministic contract/neutral suites
  and the familiar, development, or locked-acceptance Gaia2 manifests.
- `aamas2027` holds the paper protocol freeze candidate, including the exact Gaia2 mini scenario
  selection. It remains uncommitted and must not produce paid results until its protocol checkpoint
  is reviewed and committed.

The dated charged-clock protocol is the freeze candidate for the Gaia2 timing comparison. It is a
working document, kept with the other untracked notes at
`.sora/notes/benchmarks/gaia2/charged-clock-protocol-2026-09-19.md`, alongside the gate report that
records running its operational gates. What it pins is tracked here regardless: the frozen
artifacts themselves, including the exact [Gaia2 mini manifest](campaigns/aamas2027/mini-validation.json),
and the hashes `campaigns/prompt/baseline.json` holds them to.

## The frozen charge coefficients are failing their audit

The compact evidence for the 2026-09-21 charge audit is preserved in [`gates/2026-09-21/`](gates/2026-09-21/)
— four paired `llm_calls.jsonl` row sets, both preflight corner grids with their manifests, and the
scenario selection — and is verified by its [`SHA256SUMS`](gates/2026-09-21/SHA256SUMS). Against
those rows the frozen coefficients in [`charge_model.json`](charge_model.json) failed the range and
drift gates as run. Kimi's drift failure is at the target operating point and stands; the gpt-5.4
measurement was taken at `gpt-5.4-medium-prompt` while the coefficients were fitted at
`gpt-5.4-high-paper`, so reasoning effort is confounded with endpoint drift and that half is
diagnostic only, an upper bound of unknown tightness rather than a measured error.

The coefficients are unchanged: a failure against them is recorded, never re-fitted to reach green.
The token-charged convention is therefore blocked, and the primary campaign convention is
`generation_free`. This section stays until an audit passes — the rows are kept because paid
measurement does not reproduce (decode is not deterministic, and the rate is a property of the
endpoint rather than the model), so re-running replaces the evidence rather than recreating it.

## Comparability breaks

- 2026-09-12: the default billing sheet moved from `2026-09-02.json` to `2026-09-12.json` when the
  Kimi endpoint was repinned from DeepInfra to Venice. This changes cost accounting, not the frozen
  timing coefficients; the selected sheet date and digest remain report provenance.
- 2026-09-16: Gaia timing moved from provider wall latency to the frozen token-charged clock.
  Timing-gated results from before and after this integration are not comparable, although the
  semantic prompts and their frozen snapshot did not change.

## Frozen prompt baseline

The current one-repeat baseline uses the `gpt-5.4-medium-prompt` profile
(`gpt-5.4-2026-03-05`, medium reasoning, 16,384 maximum output tokens) and the pinned
`gpt-5.1-2025-11-13` ARE graph-per-event judge. Both use `OPENAI_API_KEY`. The run covers the
contract suite, all 16 live-neutral cases, and five cases from each Gaia suite: familiar,
development, and acceptance.

Run every command below from the immutable source checkout or worktree that will produce the
baseline. The ignored Gaia scenario files may live in a different checkout, so set their absolute
path explicitly. Use a durable output directory that can be archived unchanged; do not use `/tmp`
for the real run.

```console
cd /absolute/path/to/sora-runtime-worktree
export PROMPT_SCENARIO_ROOT=/absolute/path/to/main-checkout/examples/gaia2/scenarios
export PROMPT_BASELINE_OUT=/absolute/path/to/baseline-archive/prompt-v2-gpt54-medium
export PROMPT_PRICE_SHEET=examples/gaia2/evaluation/price_sheets/2026-09-12.json
```

### 1. Freeze the source and prompt snapshot

Finish review and commit every runtime, harness, profile, judge, manifest, and prompt change before
capturing the snapshot. Confirm that the source tree is clean and record its commit:

```console
git status --short
git rev-parse HEAD
```

`git status --short` must print nothing. Then regenerate the seven canonical rendered prompt
inputs and their hashes:

```console
uv run python -m examples.gaia2.evaluation prompt snapshot \
  --output examples/gaia2/evaluation/campaigns/prompt/baseline.json
```

The snapshot records each of the seven semantic calls across all four perception profiles:
operations only (tier 1), operations plus signals (tier 2), properties only, and operations,
signals, and observable properties (tier 3). Each row includes the rendered system and user text,
hashes, declared channels, and per-module character counts from `CompletionRequest.sections`. Tier
3 stays byte-identical to the original rich prompt and is also the text used by the `fixed-rich`
sensitivity control; the default `adaptive` fit removes instructions for channels that the
environment does not declare.

Review and commit `baseline.json` before making any candidate prompt change or starting the paid
run. Capturing it from a clean source commit leaves unambiguous revision and dirty-state provenance;
the subsequent artifact-only commit does not change the runtime prompts represented by the
snapshot. The snapshot also freezes all evaluation profiles and settings, model identifiers, and
the pinned judge configuration. The dated price sheet and manifest digests are recorded by the run
report rather than embedded in the prompt snapshot.

### 2. Run the offline preflight

```console
uv run python -m examples.gaia2.evaluation prompt check \
  --scenario-root "$PROMPT_SCENARIO_ROOT" \
  --require-scenarios
```

The expected output is:

```text
check passed: 3 profiles, 3 Gaia manifests, 24 contract cases, 16 neutral cases, 15/15 ignored scenario files available
acceptance payloads remained locked and unopened
```

This command renders and compares the prompts, validates tracked profiles, judge, manifests, price
sheets, contracts, neutral cases, and report serialization, and checks that exactly one scenario
filename matches every Gaia manifest entry. For acceptance it verifies that resolution is rejected
without acknowledgement; it does not open an acceptance scenario payload. The preflight makes no
provider request and needs no credential.

### 3. Review the exact matrix without spending money

Start with an output directory that does not contain a checkpoint from another run. The dry-run
uses the same selection, limits, and budget authorization as the paid run:

```console
uv run python -m examples.gaia2.evaluation prompt run \
  --profile gpt-5.4-medium-prompt \
  --suite contract \
  --suite neutral \
  --suite familiar \
  --suite development \
  --suite acceptance \
  --arm baseline \
  --live-neutral \
  --gaia-repeats 1 \
  --output-dir "$PROMPT_BASELINE_OUT" \
  --scenario-root "$PROMPT_SCENARIO_ROOT" \
  --price-sheet "$PROMPT_PRICE_SHEET" \
  --max-gaia-runs 15 \
  --max-total-spend 60.50 \
  --confirm-budget 60.50 \
  --gaia-agent-reserve 3.00 \
  --judge-reserve 0.50 \
  --neutral-reserve 0.50 \
  --max-wall-seconds 1200 \
  --max-agent-llm-calls 200 \
  --dry-run
```

The fresh matrix contains:

- one aggregate contract record, covering all 24 contract cases;
- 16 live-neutral records;
- 15 Gaia records: five familiar, five development, and five acceptance;
- 15 cumulative Gaia runs; and
- a $60.50 conservative reserve: $53.00 for agent calls and $7.50 for judge calls.

Price rows, frozen latency coefficients, and decode-count conventions are bound to the profile's
declared `(provider, model, provider_routing)` endpoint identity. A profile repin therefore fails
the offline check and live-run preflight instead of silently reusing measurements from another
backend. This key cannot detect a provider moving weights or service behind an unchanged routing
pin; `python -m examples.gaia2.charge_drift` validates the frozen coefficients against real calls
from both arms once before the sweep and again at its end. After calls exist,
`python -m examples.gaia2.charge_range` also checks both arms against the measured charge box. Its
decode interval is `[16, 8192]` on both shipped endpoints, expressed as `completion_tokens` because
their frozen conventions include reasoning there; anything outside the box is reported as
unmeasured extrapolation, and the command never re-runs the fit. A logical row that sums multiple
provider round trips also fails closed: without per-round-trip usage, its aggregate cannot certify
that every crossing stayed inside the measured box.

Remove `--dry-run` only after reviewing the matrix, configuring the profile's credential variable,
and confirming that the scenario root contains the ignored Gaia2 payloads. Acceptance runs also
require `--ack-locked-acceptance` before any locked payload is opened. Contract and offline-neutral
cases run once even when `--gaia-repeats` is greater than one.

Dry-run exits before acceptance acknowledgement, credential resolution, provider construction, or
scenario loading. Review the complete printed matrix before authorizing the live run.

### 4. Execute or resume the baseline

Configure `OPENAI_API_KEY`, remove `--dry-run`, and explicitly acknowledge the locked acceptance
payloads:

```console
uv run python -m examples.gaia2.evaluation prompt run \
  --profile gpt-5.4-medium-prompt \
  --suite contract \
  --suite neutral \
  --suite familiar \
  --suite development \
  --suite acceptance \
  --arm baseline \
  --live-neutral \
  --gaia-repeats 1 \
  --output-dir "$PROMPT_BASELINE_OUT" \
  --scenario-root "$PROMPT_SCENARIO_ROOT" \
  --price-sheet "$PROMPT_PRICE_SHEET" \
  --max-gaia-runs 15 \
  --max-total-spend 60.50 \
  --confirm-budget 60.50 \
  --gaia-agent-reserve 3.00 \
  --judge-reserve 0.50 \
  --neutral-reserve 0.50 \
  --max-wall-seconds 1200 \
  --max-agent-llm-calls 200 \
  --ack-locked-acceptance
```

`--ack-locked-acceptance` only permits the selected acceptance files to be resolved and opened. It
does not alter scoring, reveal acceptance details in the final report, or bypass any other check.
The pinned judge is attached automatically; the optional `--judge-model`, `--judge-provider`, and
`--judge-endpoint` arguments are assertions and are rejected if they differ from the pin.

The checkpoint is append-only. Re-running the identical command skips completed matrix entries, so
an interruption between cases resumes safely. A Gaia attempt ending in a timeout, context overflow,
infrastructure error, LLM-call limit, or unscored completion is checkpointed but remains pending for
retry. That attempt still consumed a real run and budget: review `checkpoint.jsonl`, then explicitly
raise `--max-gaia-runs`, `--max-total-spend`, and `--confirm-budget` enough to cover the retry. Do not
delete the failed attempt to make a ceiling pass.

Live Gaia cases print the judge verdict and terminal cause as each case finishes. Consequently, an
attended run is not blind to acceptance outcomes. If acceptance must remain a holdout until a
candidate is finalized, defer the acceptance suite and run it later from this immutable baseline
checkout, or capture and seal the live output without inspecting it. Do not let an acceptance
result influence a prompt revision; a case that does influence development is no longer a holdout
and must be replaced.

Live Gaia runs admit at most 200 logical agent LLM calls per case. Every semantic call counts,
including plan, ground, select, revalidate, condition, retirement, and relevance; failed or
later-discarded calls still consume an admission. Parser repair and provider/SDK retries within one
logical call are billed as additional provider round trips but do not consume another admission.
External actions and S-ORA decision cycles are recorded separately and are not Gaia steps.

### 5. Verify completion and produce the report

After the live command finishes, rerun the dry-run command from step 3 against the same output
directory. Every matrix entry should show `checkpoint_status: "complete"`, `gaia_runs` should be
zero, and the command should schedule no paid work. Invalid attempts remain visible as additional
checkpoint rows even after a successful retry.

Create the canonical report without acceptance details:

```console
uv run python -m examples.gaia2.evaluation prompt report \
  --input "$PROMPT_BASELINE_OUT/checkpoint.jsonl" \
  --output "$PROMPT_BASELINE_OUT/report.json" \
  --price-sheet "$PROMPT_PRICE_SHEET"
```

Do not pass `--include-acceptance-details` for a normal baseline report. That flag only disables
report-time redaction of acceptance prompts, oracles, and trajectories already present in input
records; it neither unlocks nor reruns acceptance cases.

### 6. Archive the frozen baseline

Archive these outputs and inputs together:

- `checkpoint.jsonl`, `report.json`, and the generated `configs/` directory;
- the exact source commit ID and any recorded source dirty-diff hash;
- `campaigns/prompt/baseline.json`, `profiles.json`, and `campaigns/prompt/judge.json`;
- `price_sheets/2026-09-12.json`;
- all three prompt manifest files and their digests from `report.json`; and
- any captured process output or external diagnostic artifacts associated with the run.

Store checksums outside the archive or put the archive in read-only/immutable storage. Candidate
runs must use the same scenarios, profile, judge, price sheet, harness behavior, limits, reserves,
and repeat count; only the intended prompt changes and `--arm candidate` should differ. Use a new
candidate output directory rather than mixing baseline and candidate checkpoints during execution;
the `prompt report` command accepts multiple `--input` arguments when producing the paired report.

Normal reports redact locked acceptance prompts, oracles, and detailed trajectories. The harness
makes no paid call unless `prompt run` is invoked without `--dry-run` and has pending live entries.
