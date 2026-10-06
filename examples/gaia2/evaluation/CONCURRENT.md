# Concurrent Gaia2 runs and infrastructure canary

`python -m examples.gaia2.concurrent` supervises at most **four batch processes total**, across
all selected models and arms. Each job runs one capability at one model operating point for one
arm. Scenarios within that job remain sequential. ReAct finishes a capability before the paired
S-ORA capability starts, so the existing counterpart check always reads a complete matrix.
Each model has a separate output root; every S-ORA scenario/repeat/attempt gets fresh file-backed memory.
The launcher passes the original whole manifest to every job, preserving its canonical digest.

Run from a dedicated worktree with the ARE dependencies installed:

```console
uv sync --all-extras --group are
```

Keep real artifacts in a durable absolute directory, such as the main checkout's ignored
`.sora/runs/`, and keep the source worktree available for the duration of the run. Do not edit its
source or update dependencies while it is running. The launcher pins the source commit, a digest
of tracked and nonignored source files outside the selected output subtree, installed dependency
versions, configs, judge, manifest, charge sheet, and the selected prompt snapshot. It checks these
before each new job starts.
Nonignored output directories inside the source worktree are supported; their artifacts have separate
checksum attestations. Output roots that contain the worktree or overlap tracked source are refused
before execution. Other nonignored files remain part of the source digest, including siblings of
the output root.

## Canary

The default [canary manifest](campaigns/paper2027/canary-not-evidence.json) reuses two already
inspected familiar cases: Search `28_4sn4lc` and Time `27_daamy9`. With both default profiles
(`gpt-5.4-high-paper` and `kimi-k2.5-prompt`) and both arms, this is **eight scenario runs**.
These are disposable infrastructure measurements, excluded from paper results.

First review the generated commands. This loads no scenario payloads, requires no credentials,
does not stage/download assets, creates no output directory, and spends no tokens:

```console
uv run python -m examples.gaia2.concurrent \
    --output-dir /absolute/path/to/main-checkout/.sora/runs/canary-01
```

After the debugging fixes are incorporated into the source worktree, provide a named snapshot
that matches its live prompts. Use the campaign's snapshot command to create that snapshot;
do not overwrite a historical identity to make a guard pass. A disposable canary may use a
snapshot explicitly identified as development. Both the launcher and each S-ORA worker verify
it before paid execution. The unchanged batch CLI can still record an undeclared live digest;
the concurrent launcher requires the declared snapshot for S-ORA execution.

Export the credential variables named by the pinned profiles and judge (`OPENAI_API_KEY` and
`OPENROUTER_API_KEY` for the defaults). The launcher checks names/presence without printing values.
Only adding `--execute` launches paid jobs:

```console
uv run python -m examples.gaia2.concurrent \
    --output-dir /absolute/path/to/main-checkout/.sora/runs/canary-01 \
    --prompt-snapshot /absolute/path/to/matching-snapshot.json \
    --execute
```

On macOS, prefix the command with `caffeinate -i` to keep the machine awake. Ordinary desktop use
can continue. The parent stages the local ARE filesystem once before launching workers and refuses
remote fallback if staging fails; workers inherit `DEMO_FS_PATH` before importing ARE. The selected
path and requested filesystem revision are recorded in the launch plan. An explicitly overridden
filesystem path is an operator-supplied input; its contents are not checksummed by this launcher.

Inspect `launcher-summary.json`, the per-job logs, and `resources.jsonl`. The latter samples
process-group RSS, `ps` CPU percentage, and host load about every five seconds. CPU percentage is
`ps`'s lifetime average; RSS sums can count shared pages more than once. Compare resource pressure
with responsiveness during normal desktop use. If necessary, run a new canary with `--workers 2`.
Do not infer provider throughput solely from core count.

The canary is ready when both profiles/arms finish, the expected matrices and checksums verify,
and there are no infrastructure exceptions, missing scored artifacts, or harness truncations.
**It does not require 8/8 scenario passes.** A zero score from an ordinary agent failure is retained
and does not stop admissions. Output-limit calls and missing usage coverage are reported separately
for review. Time-event delivery still needs inspection of its trace; launcher completion alone
cannot establish that every scheduled event was handled correctly.

Job elapsed times include loading, initialization, agent execution, judging, and export. The summary
also reports agent-call token totals and round trips for accepted attempts. Interrupted/infra-failed
attempts retain their own call logs and incur additional spend, as does judge billing. Use
provider billing for the total dollar spend. There is no dollar-budget enforcement here. Agree an
appropriate spending limit before a paid launch; scenario selection, repeats, and the per-scenario
call limits in the agent configs bound the work but do not guarantee a dollar ceiling.

## Larger runs

Use the same launcher with the [mini manifest](campaigns/paper2027/mini-validation.json) after
campaign readiness and the final source/prompt freeze:

```console
uv run python -m examples.gaia2.concurrent \
    --scenario-manifest examples/gaia2/evaluation/campaigns/paper2027/mini-validation.json \
    --output-dir /absolute/path/to/main-checkout/.sora/runs/paper-final-01 \
    --prompt-snapshot /absolute/path/to/final-snapshot.json
```

Review this dry plan before adding `--execute`: the defaults select 160 scenarios × two models ×
two arms, or **640 runs**, with one repeat each. `--profiles`, `--arms`, and `--num-runs` allow
smaller explicitly selected matrices. `--arms react` requires no S-ORA snapshot and can measure
the ReAct arm alone. A run root's plan is immutable; adding an arm or changing source/settings
requires a new plan/root. Reusing ReAct evidence across later S-ORA versions requires a separate
provenance review rather than automatically adopting another root's artifacts.

All runs use `generation_free`, with the same per-scenario watchdog on both arms (3600 seconds by
default). A full-job watchdog additionally allows 600 seconds startup plus 600 seconds per scenario
for work outside that bracket. Infrastructure failure stops new admissions and asks other workers to finish their current
scenario. There are no automatic retries or resamples. An explicit resume restarts interrupted or
infrastructure-failed scenarios while keeping completed scored failures. Ctrl-C terminates active
process groups immediately and preserves partial artifacts and interrupted job status.

## Pause and resume

To pause an active sweep, run this in a second terminal:

```console
uv run python -m examples.gaia2.concurrent --output-dir /absolute/path/to/run-root --pause
```

This stops new jobs and asks each worker to finish its **current scenario**, checkpoint the result,
and exit before starting the next. Wait for the launcher to exit and `launcher-status.json` to show
`state: paused` with no active jobs before closing the laptop or disconnecting. Pause can take as
long as the slowest active scenario (up to the scenario watchdog plus judging/export), so request
it ahead of a move. An API failure while draining is preserved and still needs recovery.

To continue a paused sweep, or recover after a crash/credit failure:

```console
uv run python -m examples.gaia2.concurrent --output-dir /absolute/path/to/run-root --execute --resume
```

The launcher reads the original manifest, profiles, arms, snapshot, worker count, and watchdog
from `launch-plan.json`. It verifies the same source, dependencies, and input hashes, checks every
completed checkpoint, and skips those scenario/repeat pairs. Both passes and ordinary agent/judge
failures are kept. Explicit resume restarts only scenarios interrupted before checkpoint completion
or recorded as infrastructure failures (including recognized off-cycle provider/credit errors).
The runtime's synthetic `inference stalled: no result after ...s` watchdog error also stops admissions
and makes that scenario eligible for an explicit restart.
Model parser failures with a scored result remain agent outcomes, even when their terminal label
says infrastructure error. There are no automatic retries in the background.

A resumed scenario starts with a fresh ARE simulation and fresh S-ORA memory. There is no recovery
of the exact agent/world state midway through a scenario. An in-flight request may have been billed
before its reply/usage reached the machine; restarting can therefore spend more money. Recovery
attempts remain auditable and are excluded from the canonical accepted trajectory/call log.

An unplanned sleep may break an active connection or trigger a watchdog after wake. Resume follows
the same checkpoint rules. Use the pause command and wait for draining when sleep is planned.
`caffeinate -i` helps prevent idle sleep; closing the lid is not a graceful pause command.

If the source, model settings, judge, manifest, or snapshot need to change, use a new run root.
A resume refuses changed provenance or tampered checkpoints before launching more paid work.
An OS file lock prevents a second supervisor from entering the root. Workers inherit it, so if
the supervisor crashes, surviving workers still hold the lock while finishing their current
scenario. They then detect the missing supervisor and stop. The OS releases the lock when those
processes exit; there is no stale lock to delete. The `.launcher-lock` file remains on disk and
its presence alone does not mean the run is active. Do not delete it to bypass a surviving worker.

## Preserved artifacts

The root contains `launch-plan.json`, `launcher-status.json`, `launcher-summary.json`,
`launcher-invocations.jsonl`, `resources.jsonl`, and logs beneath `logs/`. Every relaunch writes a
new log filename rather than truncating the old one; invocation summaries and resource samples
append with distinct invocation IDs.

The existing batch layout is preserved beneath each profile: `standard/<capability>/` for S-ORA
and `react/standard/<capability>/` for ReAct. Each scenario attempt lives under
`attempts/<scenario-key>.run<repeat>/attempt-<n>/`, with its own call log, trace, judge responses,
S-ORA memory/config, and result when those were produced. Call logs are flushed and fsynced per
completed model call. A fully exported attempt is checksummed before becoming a reusable checkpoint.
Partial and failed attempts stay in place when a new attempt starts; nothing paid is overwritten.

The capability's `output.jsonl` and `llm_calls.jsonl` are derived views of accepted attempts,
rebuilt atomically from verified checkpoints. `recovery.json` lists the preserved attempts excluded
from those views, whose token usage/billing must be accounted separately. Once the complete capability
matrix finishes, its whole directory receives the existing final checksum attestation and becomes
immutable. A truncated partial result or missing checkpoint commit is never promoted to a completed
scenario. A crash while writing a checkpoint can require repeating that scenario.

These guarantees concern data that reached disk. A hard interruption may leave no final ARE trace
for the active scenario, and cannot preserve a provider response that never reached the process.
The existing log prefix/call records remain available. Keep the entire durable output root backed
up; no launcher can protect it against disk failure or manual deletion.

Launcher summaries remain operational diagnostics. Use the existing batch aggregation and campaign
reporting to determine reportability and headline exclusions, and disclose infrastructure restarts
rather than treating them as extra scored trials.
