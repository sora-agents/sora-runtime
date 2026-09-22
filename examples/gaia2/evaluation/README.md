# Gaia2 evaluation harness

This offline-first harness shares profiles, metering, price sheets, checkpoints, budget gates, and
report serialization across two campaigns:

- `prompt` evaluates frozen S-ORA prompt configurations on deterministic contract/neutral suites
  and the familiar, development, or locked-acceptance Gaia2 manifests.
- `paper2027` holds the paper protocol artifacts, including the exact Gaia2 mini scenario selection.
  Its methodology is tracked in [`PROTOCOL.md`](PROTOCOL.md); the acceptance parameters and the
  judge are not frozen yet, so no locked-acceptance payload may be opened until they are added there
  and committed.

[`PROTOCOL.md`](PROTOCOL.md) is the tracked statement of how the comparison is run: the three clock
conventions and which one is primary, event delivery under a freeze, what the Time capability
actually measures, the watchdog and truncation rules, the gated headline, the paired-arm audit, the
frozen artifact hashes, the declared diagnostics, and the operational gates. It is not yet a
pre-registration — the acceptance parameters and the judge are still open, and it says so.

The dated working notes behind it, including the gate report that records running the operational
gates, stay untracked at `.sora/notes/benchmarks/gaia2/`. What the protocol pins is tracked here:
the frozen artifacts themselves, including the exact
[Gaia2 mini manifest](campaigns/paper2027/mini-validation.json), the immutable prompt snapshots
under `campaigns/prompt/snapshots/`, and the separately checked campaign configuration.

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
path explicitly. Keep local generated output under the repository's ignored `.sora/` namespace;
do not use `/tmp` for the real run, and archive the complete output directory before removing its
checkout or worktree.

```console
cd /absolute/path/to/sora-runtime-worktree
export PROMPT_SCENARIO_ROOT=/absolute/path/to/main-checkout/examples/gaia2/scenarios
export PROMPT_BASELINE_OUT="$PWD/.sora/gaia2/evaluations/prompt-v2-gpt54-medium"
export PROMPT_PRICE_SHEET=examples/gaia2/evaluation/price_sheets/2026-09-12.json
```

### 1. Freeze the source and prompt snapshot

Finish review and commit every runtime, harness, profile, judge, manifest, and prompt change before
capturing the snapshot. Confirm that the source tree is clean and record its commit:

```console
git status --short
git rev-parse HEAD
```

`git status --short` must print nothing. Then regenerate the current prompt matrix:

```console
uv run python -m examples.gaia2.evaluation prompt snapshot \
  --identity pre-optimization-control \
  --reason "Pre-optimization control captured after the byte-identical prompt source cleanup." \
  --output examples/gaia2/evaluation/campaigns/prompt/snapshots/pre-optimization-control.json
```

The snapshot records each of the seven semantic calls across all four perception profiles:
operations only (tier 1), operations plus signals (tier 2), properties only, and operations,
signals, and observable properties (tier 3). Each row includes the rendered system and user text,
hashes, declared channels, and per-module character counts from `CompletionRequest.sections`. Tier
3 stays byte-identical to the original rich prompt and is also the text used by the `fixed-rich`
sensitivity control; the default `adaptive` fit removes instructions for channels that the
environment does not declare.

Review and commit the named snapshot before making any candidate prompt change or starting the paid
run. Its filename must equal its identity, and the command refuses to overwrite that identity with
different bytes. Capturing it from a clean prompt-source commit records that revision, a null prompt
source dirty-diff hash, why the identity was created, and a canonical digest over all 28 rendered
rows. The test names this file directly; there is no movable `current` pointer to re-point.

Evaluation profiles, settings, judge, and charge coefficients have a different lifecycle. They are
mirrored in `campaigns/prompt/campaign.json`, regenerated with `prompt configuration`, and checked
deeply against `profiles.json`, `judge.json`, and `charge_model.json`. That file may move until the
campaign itself is frozen without rewriting the immutable prompt control. The dated price sheet and
manifest digests remain report provenance rather than prompt-snapshot content.

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

Each live Gaia attempt also records replayable diagnostics by default. Use `--no-diagnostics` only
when explicitly accepting that the attempt cannot be reconstructed or re-scored from its own
evidence. Diagnostics are buffered in memory while the scenario runs, then exported after ARE has
stopped, so filesystem latency cannot move the simulator's wall-clock trajectory. Here, replay
means auditing the ordered execution and reapplying scoring to the stored judge exchanges; it does
not mean deterministic re-execution of the agent or simulator.

Ordinary attempts land under `artifacts/<filesystem-safe-entry-key>/attempt-N/`; acceptance
attempts land under `artifacts/acceptance/<filesystem-safe-entry-key>/attempt-N/`. Attempt numbers
never overwrite an existing directory, including an orphan left by an interrupted exporter. Each
bundle contains:

- `run.json`, with outcome, terminal cause, timing, configuration, profile, version, revision, and
  seed provenance;
- `trajectory.jsonl`, with ordered cycle/phase context, activity transitions, actions and results,
  pending work, and environment-boundary observations;
- `judge_recording.json`, in the existing `examples.gaia2.rescore` schema, plus `verdict.json` and
  `write_counts.json`;
- `llm_calls.json`, `llm/exchanges.jsonl`, and deduplicated exact prompts under
  `llm/prompts/<sha256>.txt`;
- `session.log` and ARE's Hugging Face trace when available; and
- `index.json`, with sorted paths, sizes, SHA-256 hashes, and any component export errors.

The checkpoint keeps only a compact artifact reference for live Gaia call details after verifying
that `llm_calls.json` is readable and exactly matches them. Otherwise it retains the call rows
inline, so a partial export cannot erase paid-run accounting. Report creation hydrates externalized
details from `llm_calls.json`; legacy checkpoints with inline calls remain readable. An export
failure does not change or erase the attempt's score: its checkpoint reference records the failure
and the report increments `aggregates.diagnostics.incomplete`.

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
- `campaigns/prompt/snapshots/pre-optimization-control.json`, `campaigns/prompt/campaign.json`,
  `profiles.json`, and `campaigns/prompt/judge.json`;
- `price_sheets/2026-09-12.json`;
- all three prompt manifest files and their digests from `report.json`; and
- the complete `artifacts/` tree, including every failed, retried, or orphan attempt directory.

Treat the entire `artifacts/acceptance/` subtree as sealed acceptance material, not only the judge
recording or ARE trace inside it. It is ignored by the repository and must not be copied into an
ordinary report archive, logs, tickets, or review attachments. Normal reports omit acceptance
diagnostic references and payloads; `--include-acceptance-details` reveals them only for an
explicitly authorized detailed report.

Store checksums outside the archive or put the archive in read-only/immutable storage. Candidate
runs must use the same scenarios, profile, judge, price sheet, harness behavior, limits, reserves,
and repeat count; only the intended prompt changes and `--arm candidate` should differ. Use a new
candidate output directory rather than mixing baseline and candidate checkpoints during execution;
the `prompt report` command accepts multiple `--input` arguments when producing the paired report.

Normal reports redact locked acceptance prompts, oracles, and detailed trajectories. The harness
makes no paid call unless `prompt run` is invoked without `--dry-run` and has pending live entries.
