# Gaia2 evaluation harness

This offline-first harness shares profiles, metering, price sheets, checkpoints, budget gates, and
report serialization across two campaigns:

- `prompt` evaluates frozen S-ORA prompt configurations on deterministic contract/neutral suites
  and the familiar, development, or locked-acceptance Gaia2 manifests.
- `aamas2027` is reserved for the later paper protocol. Its dataset, scaffold arms, judge, clock
  policy, and ablations are not frozen yet, so it cannot run experiments today.

Run commands from the repository root. The offline check opens no provider credential or
acceptance payload:

```console
uv run python -m examples.gaia2.evaluation prompt check
```

Render the current seven canonical prompt inputs without changing the tracked baseline:

```console
uv run python -m examples.gaia2.evaluation prompt snapshot
```

Inspect an exact three-repeat Gaia matrix and its cumulative reserve without running it:

```console
uv run python -m examples.gaia2.evaluation prompt run \
  --profile gpt-5.4-medium-prompt \
  --suite development \
  --arm baseline \
  --gaia-repeats 3 \
  --output-dir /tmp/sora-gaia2-prompt \
  --price-sheet examples/gaia2/evaluation/price_sheets/2026-09-02.json \
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

Remove `--dry-run` only after reviewing the matrix, configuring the profile's credential variable,
and confirming that the scenario root contains the ignored Gaia2 payloads. Acceptance runs also
require `--ack-locked-acceptance` before any locked payload is opened. Contract and offline-neutral
cases run once even when `--gaia-repeats` is greater than one.

Combine one or more checkpoint files into the canonical report:

```console
uv run python -m examples.gaia2.evaluation prompt report \
  --input /tmp/sora-gaia2-prompt/checkpoint.jsonl \
  --output /tmp/sora-gaia2-prompt/report.json \
  --price-sheet examples/gaia2/evaluation/price_sheets/2026-09-02.json
```

Normal reports redact locked acceptance prompts, oracles, and detailed trajectories. The harness
does not make a paid call unless `prompt run` is invoked without `--dry-run` and with an explicit
live suite/profile selection.
