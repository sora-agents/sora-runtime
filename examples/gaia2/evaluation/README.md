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

Run commands from the repository root. The offline check opens no provider credential or
acceptance payload:

```console
uv run python -m examples.gaia2.evaluation prompt check
```

Render the current prompt matrix without changing the tracked baseline:

```console
uv run python -m examples.gaia2.evaluation prompt snapshot
```

The snapshot records each of the seven semantic calls across all four perception profiles:
operations only (tier 1), operations plus signals (tier 2), properties only, and operations,
signals, and observable properties (tier 3). Each row includes the rendered system and user text,
hashes, declared channels, and per-module character counts from `CompletionRequest.sections`. Tier
3 stays byte-identical to the original rich prompt and is also the text used by the `fixed-rich`
sensitivity control; the default `adaptive` fit removes instructions for channels that the
environment does not declare.

Inspect an exact three-repeat Gaia matrix and its cumulative reserve without running it:

```console
uv run python -m examples.gaia2.evaluation prompt run \
  --profile gpt-5.4-medium-prompt \
  --suite development \
  --arm baseline \
  --gaia-repeats 3 \
  --output-dir /tmp/sora-gaia2-prompt \
  --price-sheet examples/gaia2/evaluation/price_sheets/2026-09-12.json \
  --confirm-budget 180 \
  --dry-run
```

Live Gaia runs admit at most 200 logical agent LLM calls by default. Every semantic call made by
the agent counts, including plan, ground, select, revalidate, condition, retirement, and relevance;
failed or later-discarded calls still consume an admission. Parser repair and provider/SDK retries
inside one logical call are reported and billed as round trips but do not consume another
admission. Override
the guard explicitly with `--max-agent-llm-calls`. External actions and S-ORA decision cycles are
reported separately as architectural diagnostics and do not define Gaia2 steps. The actual limit
is stored with every live Gaia case and summarized in report provenance.

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

Combine one or more checkpoint files into the canonical report:

```console
uv run python -m examples.gaia2.evaluation prompt report \
  --input /tmp/sora-gaia2-prompt/checkpoint.jsonl \
  --output /tmp/sora-gaia2-prompt/report.json \
  --price-sheet examples/gaia2/evaluation/price_sheets/2026-09-12.json
```

Normal reports redact locked acceptance prompts, oracles, and detailed trajectories. The harness
does not make a paid call unless `prompt run` is invoked without `--dry-run` and with an explicit
live suite/profile selection.
