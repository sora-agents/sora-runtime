# Gaia2 benchmark harness

Running S-ORA against Meta's [Gaia2](https://huggingface.co/datasets/meta-agents-research-environments/gaia2)
scenarios. There are two harnesses, and they are not interchangeable.

**This page is the in-process one**: S-ORA and the ARE simulator in one Python process, driven by
three entry points. It is the fast path for development — no container, a scenario in seconds, the
whole trajectory in your own log.

| Script | Use it to |
|---|---|
| `scripts/fetch_scenario.py` | Pull one scenario JSON down to inspect or replay locally |
| `run_benchmark.py` | Run **one** scenario, print the judge verdict |
| `batch.py` | Run a whole capability, emit leaderboard artifacts, report pass@1 |
| `rescore.py` | Re-score a **stored** run offline, under either judge-verdict parse |

**[`cli/`](cli/README.md) is the container one** — Meta's own `gaia2-cli` harness, one container per
scenario, the agent driving ten command-line apps behind a setuid wrapper. Slower to iterate on, and
the one *reported* numbers should come from: the in-process path diverges from stock ARE in ways a
published Gaia2 result cannot absorb, most consequentially on the clock. Keep both.

## Setup

```bash
uv sync --all-extras --group are        # ARE package + the llm extra
export ANTHROPIC_API_KEY=sk-ant-...     # agent model (or put it in a .gitignored .env)
export HF_TOKEN=hf_...                  # gated dataset + an HF-hosted judge model
```

The agent config is [`agent.yaml`](agent.yaml) — deliberately scenario-agnostic; the scenario is
always a per-run CLI argument. [`agent.dev.yaml`](agent.dev.yaml) points the same agent at a local
Ollama model for cheap iteration; for that one, build the model first:

```bash
ollama pull qwen3:30b
ollama create qwen3:30b-64k -f examples/gaia2/qwen3-30b-64k.Modelfile   # instant, shares blobs
```

The `-64k` step is not optional — a plan prompt overruns Ollama's default context window and is
silently truncated. See [`qwen3-30b-64k.Modelfile`](qwen3-30b-64k.Modelfile) for why the window
cannot be set per request.

### The file-system fallback is staged locally

Both drivers call `ensure_local_fallback_fs()` before importing ARE, which downloads ARE's
`demo_filesystem` (294 files, ~247 MiB) once into the standard Hugging Face cache and points
`DEMO_FS_PATH` at it. This is not just a speed-up. Stock ARE answers every placeholder's file size
with its own `paths-info` request, so each `Files.get_state()` costs ~294 round-trips and ~66s —
twice per run. The second one lands *inside* the agent's focus baseline while the scenario clock is
already running, so the agent never perceives anything the scenario injects in that window. A run
log shows it as a first plan prompt whose `Calendar.state` holds more events than the scenario's
initial state, beside "(none observed yet)". See [`_local_fs.py`](_local_fs.py).

The first run pays ~30s to download; after that it is free and offline. Same bytes and same tree, so
scores are unaffected — but timings are, so runs from before this are not comparable.

- `SORA_GAIA2_LOCAL_FS=0` — opt out, use the Hub (slow, and blind at the head of the timeline).
- `DEMO_FS_PATH=/some/path` — preset wins; point it at your own tree.
- `GAIA2_FS_REVISION=<sha>` — track a different upstream commit than the pinned one.

Once the snapshot is cached, `HF_HUB_OFFLINE=1` runs fine *on top of* this. Do not reach for it
*instead*: with ARE still pointed at the Hub, its stat-failure path *deletes* the file's registry
entry, permanently un-backing it, so a later read returns the empty placeholder rather than an
error — a broken run that looks like a working one.

> Gaia2 data is Meta's and gated by the dataset terms — fetch it on demand, never redistribute it.
> `*.scenario.json` and the whole `scenarios/` directory are gitignored; keep fetched scenarios in
> one of those two, since a scenario fetched by id keeps an upstream filename that the suffix
> pattern alone does not catch.

## `scripts/fetch_scenario.py` — grab one scenario

```bash
# list the scenario ids for a capability + split
python -m examples.gaia2.scripts.fetch_scenario --capability execution --list

# fetch the first one -> ./execution-validation-0.scenario.json
python -m examples.gaia2.scripts.fetch_scenario --capability execution

# a specific one, to a chosen path
python -m examples.gaia2.scripts.fetch_scenario --capability ambiguity --index 2 --out amb.json
```

`--capability` names the dataset config (`execution`, `search`, `adaptability`, `time`, `ambiguity`, `mini`) — the same spelling `batch.py` uses,
`--split` defaults to `validation` (the test split is private), `--dataset` overrides the HF repo.

## `run_benchmark.py` — one scenario

```bash
python -m examples.gaia2.run_benchmark \
    --scenario ./execution-validation-0.scenario.json \
    --judge-model claude-sonnet-5 --judge-provider anthropic \
    --verbose
```

| Flag | Meaning |
|---|---|
| `--scenario PATH_OR_DOTTED` | **Required.** Scenario `.json`, or a dotted path to a `Scenario` subclass |
| `--config AGENT_YAML` | Agent config (default `examples/gaia2/agent.yaml`) |
| `--judge-model` / `--judge-provider` / `--judge-endpoint` | Attach ARE's oracle-graph judge so the run is scored. Omit for an unscored trajectory check |
| `--judge-recording PATH` | Keep this run's raw judge responses as JSON, so it can be re-scored later |
| `--no-judge-recording` | Skip recording them (and the both-parse comparison) |
| `--init-turns` | Deliver every turn of a multi-turn scenario **without** a judge. Excludes `--judge-model` |
| `--max-wall-seconds` | Safety cap, default 1200 |
| `--exit-when-idle SECONDS` | Old single-turn quiet-window stop. Only correct for single-turn scenarios |
| `--verbose` / `--log-file PATH` | Stream / mirror the full trajectory |
| `--llm-calls PATH` | Append one JSON line per model call — tokens, cache reads, measured latency |

The run stops **timeline-aware** by default: it rides through the idle gaps between turns and ends
once ARE's event loop has completed the scenario, then validates once.

### Multi-turn scenarios need a judge *or* `--init-turns`

A multi-turn scenario's later turns hang off `OracleEvent`s that an agent-mode environment ignores,
so a plain unscored run **silently stops after turn 1**. Either attach a judge (which also becomes
the per-turn release gate — a failed verdict on turn 1 ends the run) or pass `--init-turns` to
release every turn unconditionally and stay unscored. Use `--init-turns` for development, when the
behaviour under test only shows up in a later turn.

## `batch.py` — a whole capability

```bash
# smoke: three scenarios
python -m examples.gaia2.batch --capability ambiguity --limit 3 \
    --judge-model claude-sonnet-5 --judge-provider anthropic \
    --output-dir .sora/gaia2/out

# aggregate whatever has been run
python -m examples.gaia2.batch --report-only .sora/gaia2/out
```

Writes, under `{output-dir}/standard/{capability}/`, one HF-format trace per (scenario, run),
`output.jsonl` in ARE's own benchmark-result shape, `llm_calls.jsonl` (below), and — for a scored
sweep — `judge_responses/{scenario}.run{n}.json` per run, named in that row's
`metadata.judge_recording`.
`--report-only` prints per-capability pass@1 and the equal-weight overall across the five core
capabilities.

Flags mirror `run_benchmark.py`, plus: `--capability` (the dataset config to run), `--split`,
`--hf-dataset`, `--output-dir`, `--num-runs` (Gaia2 uses 3), `--limit`, and `--model` (the label
recorded in the trace — match it to `agent.yaml`'s `llm.model`).

### `--arm` — which agent runs the sweep

`--arm sora` (the default) runs S-ORA, configured by `--config`. `--arm react` runs ARE's own
published ReAct agent as the baseline, configured by `--profile` — the same model, reasoning
setting and output cap the S-ORA arm is pointed at. That last part is checked, not assumed: a react
sweep compares the profile's operating point against `--config`'s `llm:` block and refuses before
spending anything if they disagree, since two arms at different reasoning efforts produce two
pass@1 columns that look comparable, carry the same model label, and are not. The comparison runs
over the union of what the two sides declare, minus a named few that are not the operating point
(`stall_timeout`, `max_retries`, `instrument`, `max_logical_calls`) — so routing counts too: the
same model name at a different `base_url`, or behind a different `provider_routing`, is a different
serving path, and a setting added to either side is compared from the day it exists.

```bash
# the pair: same capability, same scenarios, same judge, one root
python -m examples.gaia2.batch --capability execution --limit 5 \
    --judge-model claude-sonnet-5 --judge-provider anthropic --output-dir .sora/gaia2/pair
python -m examples.gaia2.batch --capability execution --limit 5 --arm react \
    --profile gpt-5.4-medium-prompt \
    --judge-model claude-sonnet-5 --judge-provider anthropic --output-dir .sora/gaia2/pair

python -m examples.gaia2.batch --report-only .sora/gaia2/pair              # S-ORA
python -m examples.gaia2.batch --report-only .sora/gaia2/pair --arm react  # baseline
```

Everything except the agent is shared: the judge and its verdict parse, the oracle replay, the
tool-call gate, the record shape, pass@1, and `llm_calls.jsonl`. The point of routing both arms
through this one file is that a per-arm discrepancy in how a row is *scored* — rather than in how
the agent performed — has nowhere to hide.

`gpt-5.4-medium-prompt` is the profile matching the shipped `agent.yaml`; to run the pair at
another operating point, move both sides.

Four things that follow from that, each of which would otherwise tilt the comparison in a
direction nothing in the artifacts would reveal:

- **A react sweep lands under `{output-dir}/react/standard/{capability}/`.** Both arms write
  `output.jsonl` and `llm_calls.jsonl` per capability, and re-running a capability truncates them
  on purpose — so sharing a root would have the second arm destroy the first arm's results,
  including the per-call rows the cost comparison is computed from. The default arm keeps the bare
  root, and with it the layout ARE's upload script walks.
- **An unjudged react run stays unscored.** `ScenarioRunner` always calls `scenario.validate()`,
  and with no judge attached that falls through to ARE's base implementation, which returns
  `env.state != FAILED` — True for any run that merely did not crash. The S-ORA arm already guards
  on this; without the same guard, every unjudged baseline run would score a free pass.
- **A crash is an `exception` row, not a failure.** ARE's `run()` collapses `success=None` plus an
  exception into `success=False`. pass@1 *excludes* an exception and *counts* a failure, so left
  collapsed an errored react run would be a genuine miss while the equivalent S-ORA run was dropped
  from the denominator.

- **Timeline expiry is latched when the agent stops, not when the row is written.** ARE pauses
  nothing at shutdown — `Environment.stop()` leaves `TimeManager` running — and the first thing it
  does after the agent returns is `scenario.validate()`, whose judge pass can take minutes. Read
  off the clock afterwards, a run that finished inside its budget would report as expired and be
  dropped from pass@1. The S-ORA arm latches at its own shutdown for the same reason.

`--init-turns` means the same thing on both arms — without a judge it decides whether turns 2..n
are delivered at all — so a paired sweep compares equal work. `--max-wall-seconds` applies to the
S-ORA arm; the react arm is bounded by ARE's own event loop, which exits at `scenario.duration`.

### `llm_calls.jsonl` — one row per model call

Written for every sweep, scored or not, and costing no tokens: `call_id`, `arm`, `model`,
`semantic_label`, `input_tokens`, `cached_input_tokens`, `output_tokens`, `reasoning_tokens`,
`seconds`, `round_trips`, `finish_reason`, plus `scenario_id` / `run_number` so one file holds a
whole capability. `round_trips > 1` is one decision that crossed the wire more than once (a
parser-repair pass on the S-ORA side), with the token fields summed across them.

The ReAct arm groups differently, because ARE calls the engine again on a malformed output and
there is no logical-call id to reuse: each crossing is its own row, and `bracket_id` is what says
they were one charged step. It has to be on the row — the extra keys the engine returns in its
metadata dict are dropped by ARE's `LLMOutputThoughtActionLog`, so after the run the file is the
only place a retried decision can still be told apart from two decisions.

A `null` token field means the provider did not report it — never a measured zero. `usage_captured:
false` says the tokens are not a complete account of the call: the usage block was unreachable, or
only some of a repaired call's round-trips reported one (a retry that dies before reporting still
counts as a crossing, since the timing record comes from a `finally`). Most providers ship no cache
or reasoning detail block, so `null` there is the ordinary case and `cached_input_tokens: 0` is a
real, measured miss. `cached_input_tokens` in particular exists only here: the human-readable trace
drops it.

A failed round-trip is recorded with an `error:` `finish_reason` and is *charged* — the agent
emitted the call and waited for it to fail. "Failed" does not mean "unbilled": ARE validates the
response after LiteLLM returned it, so a content-filtered answer raises with a full usage block
already in hand, and the row reports what that crossing actually cost. Only a crossing that
produced no response at all bills the charge model's fixed per-call term alone. The fit excludes
these rows; the bill does not.

Rows for a scenario are written when that scenario's recording ends, not as each call returns: a
repaired call is two round-trips on one `call_id` and nothing in the stream marks the last of them,
so the row can only be settled once the scenario's stream has. `batch.py` truncates the file once at
the start of a capability sweep, alongside the `output.jsonl` it replaces — otherwise a re-run's
rows would sit next to the previous sweep's under the same `scenario_id` and `run_number`, and the
fit would count the stale ones a second time. `run_benchmark.py --llm-calls` appends, so several
single-scenario runs can be pointed at one file deliberately.

`examples.gaia2.react_engine.MeteredLiteLLMEngine` writes the same schema with `arm: "react"` for
the ARE baseline, whose stock engine reports nothing at all — see that module for why ARE's own
agent is charged zero for thinking.

It also puts the baseline on the same request as the other arm. ARE's `LiteLLMEngine` ignores its
own `**kwargs` and hands `litellm.completion` five fixed arguments, so an uninstrumented baseline
runs at the provider's defaults — no reasoning setting, no output cap, no provider routing, and
never streamed — while S-ORA runs at the profile's. Build it with
`MeteredLiteLLMEngine.from_profile(profile)` and both arms send what
`ModelProfile.request_kwargs()` says, streaming when the profile streams. Streaming matters beyond
tidiness: S-ORA's client streams by default because its stall timeout means "the provider went
quiet", which is only observable on a streamed call, so a non-streaming baseline differs from it in
transport on precisely the per-arm comparison the charge model exists to make. Both changes are
applied where the response is already intercepted; ARE's own `chat_completion` body still runs
untouched, including the `True`/`False` lowercasing its checkers depend on.

One streaming trap is worth knowing about, because nothing downstream could detect it: LiteLLM's
`stream_chunk_builder` fills a *missing* usage block by re-tokenizing the prompt and completion
locally, so a provider that ignores `include_usage` produces token counts that are plausible,
wrong, and indistinguishable from reported ones. Rows therefore take their tokens from the
provider's own trailing usage chunk and never from the rebuilt response — a stream that reported no
usage is written `usage_captured: false` with null tokens and charged the fixed per-call term,
which undercounts visibly instead of mis-fitting silently.

## `react_driver.py` — one scenario on the ReAct arm

The counterpart to `run_benchmark.py`: the same scenario, ARE's own published agent, one
`LLMCallRecord` per model round-trip.

```
python -m examples.gaia2.react_driver \
    --scenario examples/gaia2/scenarios/execution/acc-scenario_universe_23_1hu54e.json \
    --profile gpt-5.4-high-paper \
    --judge-model claude-sonnet-5 --judge-provider anthropic \
    --llm-calls react_calls.jsonl --run-number 0
```

It deliberately does not re-implement ARE's run. `ScenarioRunner` still owns the environment, the
oracle mode, the turn wiring and the trace export — a baseline is only a baseline if that stays
ARE's code — and everything is injected through seams ARE already exposes. `LLMEngineBuilder.
_create_concrete_engine`, documented upstream as overridable, returns a `MeteredLiteLLMEngine` built
from the profile instead of a stock engine; `AgentBuilder.build` is subclassed only to wrap the
agent's `pause_env` afterwards, which is the one signal an engine gets that a *step* rather than a
round-trip has begun. A runner pointed at a model the profile does not name raises before it spends
anything, because a mis-wired experiment produces ordinary-looking rows against the wrong model.

The judge is attached here rather than by ARE, using the same `attach_judge` the S-ORA arm calls —
`relax_judge_verdict_case` included, since ARE's engines lowercase `True`/`False` on the way out of
every `chat_completion` and an unparsed verdict both mis-scores the event and withholds the
remaining turns. Two arms scored by different judge wiring are a comparison of scorers.
`max_turns` is left unset on purpose — ARE's config defaults it to 1, but `run_scenario` overrides
it from `scenario.nb_turns`, which every Gaia2 scenario carries.

Initialization is not optional and not visible: `ScenarioRunner` refuses an uninitialized scenario,
`load_scenario` does not initialize, and `attach_judge` / `populate_oracle_events` both do it as a
side effect — so whether a caller has already satisfied it cannot be told from outside. A run that
skipped it returns `success=None` with zero recorded calls, which reads exactly like a model that
did nothing. `run_react_on_scenario` therefore initializes the scenario itself (idempotently); it is
that function's precondition, not a rule for every caller to remember. Preprocessing proper is a
separate matter, and is what decides whether turns 2..n are delivered.

`run_react_on_scenario(scenario, profile, ...)` is the entry point `batch.py` uses for `--arm
react`: it takes an already-loaded scenario and returns the *S-ORA arm's* `RunResult` type, so one
record shape covers both arms. The S-ORA-only fields (`llm_report`, `replan_count`, `prop_reads`,
...) are left at their defaults rather than filled with a plausible-looking ReAct analogue —
`agent_llm_calls` counts *logical* calls on that arm and would silently become a round-trip count
here. `llm_calls.jsonl` is the comparable per-call record.

## `latency_grid.py` — the designed grid the charge model is fitted on

**This one spends money.** Everything else in this directory is free to re-run; this is a token
purchase of roughly $10 per model, so start with `--dry-run`, which prints the plan and the target
volume and calls nothing.

```
# what it would do, and what it would cost in tokens
python3 -m examples.gaia2.latency_grid --profile gpt-5.4-high-paper --dry-run

# run it
python3 -m examples.gaia2.latency_grid --profile gpt-5.4-high-paper \
    --out grid/gpt-5.4-high.jsonl
```

The charge model is `a0 + uncached_in/R_in + cached_in/R_cache + out/R_out`, frozen before the
sweep and applied identically to both arms. Its coefficients cannot be fitted from the agents'
own calls: neither arm varies prompt length independently of answer length, so an ordinary
least-squares fit on stored trajectories returns a **negative** input coefficient — a longer prompt
served faster. The grid exists to move the two axes independently, which is the only thing that
identifies `R_in`.

Calls go through one plain `AsyncOpenAI` built from the named profile — deliberately neither arm's
client. `a0` has to be the model's own per-call cost; measuring it through one arm's stack would
fold that arm's SDK overhead into a coefficient later charged to both. The client is built once and
reused, so `a0` is not inflated by handshakes a real run amortizes.

Two files come out, joined on `call_id`: `llm_calls.jsonl` rows with `arm: "grid"`, and a
`.manifest` sidecar carrying what makes a row an *experiment* rather than a call — the cell's
targets, cache condition, block, warm-up flag, prompt and profile hashes, and `fit_eligible`. Read
`fit_eligible` first: warm-ups, failures and cells that missed their target are all kept in the
files, because every one of them was paid for, and all are excluded from the fit. A cached cell
additionally needs the provider to have *reported* its cached token count — a usage block with no
`prompt_tokens_details` leaves `uncached = input - cached` undefined on that arm, which is an
unknown regressor rather than a small one. The uncached arm needs no such count: its prompts are
unique per call, so an unreported cache is the zero it is. `cached_on_target` is reported
separately and deliberately does *not* gate eligibility — a reported cache miss is still a valid
observation of the input term, but an arm that missed everywhere means `R_cache` was never
measured, and that has to be legible without recomputing it row by row.

Because the two files join on `call_id`, and `call_id` is deterministic from the cell and the
block, a second run into existing outputs would duplicate join keys. The CLI refuses that: pass
`--overwrite` to truncate both files together, or `--resume` to continue. Resume skips at the
granularity of a *unit*, not a cell — a cached group is a warm-up plus the measurements it warms,
so a group left incomplete is re-run whole rather than re-entered against a prefix the provider
evicted hours ago, and the repeated cells take suffixed call ids so nothing collides.

The run opens with two cheap calibration calls that measure tokens per filler word and then
**freeze** the ratio. This is not a warm-up nicety: word counts are derived from that ratio, so a
ratio still refining itself during the grid would make a cell's prompt depend on which calls
happened to run before it, break the claim that the stored seed regenerates the run's bytes, and —
worst — shift a cached prefix's word count between a warm-up and the measurement it warms, quietly
handing the provider a different head from the one it cached. The frozen ratio and both word counts
land on every manifest row, so any prompt in the run can be rebuilt exactly and checked against its
recorded `prompt_sha256`.

Prompts are deterministic from the stored `--seed` and that frozen ratio, so a published
coefficient can be traced back to the exact bytes that produced it. Everything except the two token axes comes from the profile —
reasoning setting, temperature, provider routing, streaming — because latency does not transfer
across them. `max_completion_tokens` is the single deliberate exception: it is the only way to
force an output length, so it is the grid's independent variable rather than the profile's 16,384.

### Submitting

The artifacts are exactly what ARE's uploader consumes — no container, no re-export.

```bash
# 1. run each core capability at NUM_RUNS=3, same --output-dir
for c in execution search adaptability time ambiguity; do
  python -m examples.gaia2.batch --capability "$c" --num-runs 3 \
      --judge-model claude-sonnet-5 --judge-provider anthropic \
      --model S-ORA/claude-... --output-dir "$PWD/.sora/gaia2/out"
done

# 2. hand the tree to ARE's uploader
uv run python -m are.simulation.benchmark.gaia2_upload_script \
    --input_dir "$PWD/.sora/gaia2/out" --output_dir "$PWD/.sora/gaia2/stats" \
    --model S-ORA/claude-... --split validation --hf_upload <org>/<dataset>
```

Use an **absolute** `--output-dir`: each `output.jsonl` row's `trace_id` is a path the uploader has
to resolve from its own cwd.

## `rescore.py` — re-score a stored run

ARE keeps **one boolean per judged event** and throws the judge's actual answer away, so a sweep run
without recording can never be re-scored afterwards, at any price. A scored run
therefore stores, per judged event, the tool name, the agent and oracle arguments as the judge
selected them, the `equality_checker` outcome, and every raw model response with its `[[…]]`
markers. `rescore.py` reads those back and re-applies ARE's own rule — equality fast path, else a
strict conjunction over the soft checkers — **once per verdict parse**, with no model call and
without ARE installed.

```bash
# one scenario: run_benchmark prints both parses inline; --judge-recording keeps the file
python -m examples.gaia2.run_benchmark --scenario ./amb.json \
    --judge-model claude-sonnet-5 --judge-provider anthropic \
    --judge-recording ./run.judge.json

# a whole sweep: walk the artifact tree (files or directories, any mix)
python -m examples.gaia2.rescore .sora/gaia2/out

# the acceptance gate on the recording pipeline itself
python -m examples.gaia2.rescore .sora/gaia2/out --require-divergence
```

| Flag | Meaning |
|---|---|
| `PATH...` | **Required.** Recording files, or directories to walk — an artifact `--output-dir` works |
| `--require-divergence` | Exit non-zero unless the two parses differ on an event the equality checker *missed* |
| `--output PATH` | Also write the summary as JSON |

**Why two scores rather than one.** ARE's engines lowercase `True`/`False` on the way out of every
model call, while its `[[True]]`-family checkers compare case-sensitively — so those checkers cannot
return a verdict at all, and the unparsed answer rejects on the same falsy path a genuine rejection
takes. A single number cannot separate *the agent got it wrong* from *the scorer could not say yes*;
the two together can. Runs relax the parse by default and record which one they used;
`--strict-verdict-case` (on either driver) scores under stock ARE instead.

**The re-scorer audits itself.** It re-implements ARE's rule rather than calling into it, so every
recording is also re-scored under the parse the run *actually* used, where it must reproduce ARE's
boolean event for event. A mismatch prints a loud `⚠` beside the scores it invalidates and exits
non-zero — read that line before reading the numbers above it.

**Two things are mechanically enforced**, because the failure mode here is deprioritization rather
than difficulty:

- A **scored** `batch.py` sweep records by default and *refuses to start* if it cannot arm the
  recorder. `--no-judge-recording` is the deliberate opt-out and is never the default.
- `--require-divergence` is strict about *where* divergence falls: it passes only when the parses
  differ on an event the equality checker missed. A pipeline can record faithfully and still never
  exercise the checker path — every event settled by the fast path, no model consulted — and that
  is indistinguishable from genuine agreement in any aggregate. Verify it on a scenario ending in a
  paraphrased message to the user; those are the events that reach the soft checkers at all.

The recorder is pinned against ARE's own `SoftToolJudge` driven by a fake engine, after an early
live run caught it storing verdicts with no checker answers behind them. A model-backed run has
since stored a complete recording end to end. A recording whose events carry an empty `checkers`
list is still the signature of that failure rather than of an easy verdict, so it is worth a glance
before reading any scores. `send_email` is the sharpest case: its checkers are the `[[True]]` family
that stock ARE cannot parse at all, per *Why two scores rather than one* above.
