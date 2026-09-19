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
| `--wall-clock` | Robustness mode: use elapsed wall time instead of the frozen token charge. Not timing-comparable to the main sweep |
| `--charge-profile NAME` | Resolve an ambiguous frozen-profile match explicitly. The named profile must still match the config; excludes `--wall-clock` |
| `--allow-unfrozen-config` | Permit a local/unlisted config and necessarily use wall time. Development-only; excludes `--charge-profile` |
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
`--hf-dataset`, `--hf-revision`, `--scenario-manifest`, `--output-dir`, `--num-runs` (Gaia2 uses
3), `--limit`, and `--model` (the label recorded in the trace — match it to `agent.yaml`'s
`llm.model`). `--limit` is for loader-order smoke tests. A comparable paid subset uses
`--scenario-manifest`, which is mutually exclusive with `--limit` and pins an ordered exact-ID set
at an immutable dataset revision.

The sweep manifest reuses the evaluation manifests' envelope without their five-capability
restriction:

```json
{
  "schema_version": 1,
  "name": "time-shared-set",
  "dataset": "meta-agents-research-environments/gaia2",
  "revision": "<immutable dataset commit>",
  "split": "validation",
  "cases": [
    {"capability": "time", "id": "scenario_universe_..."}
  ]
}
```

Batch loads and validates the entire selection before opening artifacts, runs in manifest order,
records the canonical manifest digest on every row, and refuses an already-written counterpart arm
whose digest or `(scenario_id, run_number)` matrix differs. The manifest can contain several
capabilities, but one batch invocation still runs the requested `--capability` only.

The paper freeze candidate uses
[`evaluation/campaigns/aamas2027/mini-validation.json`](evaluation/campaigns/aamas2027/mini-validation.json):
all 32 scenarios selected by Gaia2 `mini` for each of the five capabilities. The cases are recorded
under `execution`, `search`, `adaptability`, `time`, and `ambiguity`, rather than under a synthetic
`mini` result bucket. Each invocation therefore loads the corresponding full capability config and
selects the mini IDs, preserving the normal per-capability reports and equal-weight five-way
aggregate while evaluating exactly the mini suite.

Run each capability separately with the same manifest; repeat the loop for each arm and operating
point in the frozen run matrix:

```bash
for capability in execution search adaptability time ambiguity; do
  python -m examples.gaia2.batch \
    --capability "$capability" \
    --scenario-manifest examples/gaia2/evaluation/campaigns/aamas2027/mini-validation.json \
    --num-runs 3 \
    --judge-model "$JUDGE_MODEL" --judge-provider "$JUDGE_PROVIDER" \
    --output-dir .sora/gaia2/aamas2027
done
```

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

`--wall-clock` changes only the clock policy. The S-ORA arm still derives and validates the frozen
profile, and records its charge-model identity/digest, so a robustness row can be joined to the
charged operating point it checks; batch rows also name the exact `model_profile`.
`--charge-profile NAME` resolves an ambiguous match but does
not bypass validation: the named profile must describe the config exactly. For a local/Ollama or
otherwise unlisted config, pass `--allow-unfrozen-config`; because no frozen coefficients exist for
that operating point, the flag necessarily selects wall time. A missing config is always an error.

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

- **Timeline expiry is latched when the agent stops, not when the row is written.** The charged
  clock resumes outside model calls, and the first thing ARE does after the agent returns is
  `scenario.validate()`, whose judge pass can take minutes. Read off the clock afterwards, a run
  that finished inside its budget could report as expired and be dropped from pass@1. The S-ORA
  arm latches at its own shutdown for the same reason.

`--init-turns` means the same thing on both arms — without a judge it decides whether turns 2..n
are delivered at all — so a paired sweep compares equal work. `--max-wall-seconds` is the same
1200-second scenario safety cap on both arms, over the same *span*: agent execution plus scoring,
excluding artifact serialization. The span matters as much as the number — ARE's `ScenarioRunner`
validates inside the call the ReAct watchdog wraps, so capping only S-ORA's agent loop would leave
its judge pass, which can run for minutes, effectively uncapped on one arm alone. The separate 180-second watchdog applies only while
the online judge holds the environment paused; an agent generation may legitimately approach its
profile's 600-second transport timeout without being mistaken for a stalled judge.

### `llm_calls.jsonl` — one row per model call

Written for every sweep, scored or not, and costing no tokens: `call_id`, `arm`, `model`,
`semantic_label`, `input_tokens`, `cached_input_tokens`, `output_tokens`, `reasoning_tokens`,
`seconds`, `charged_seconds`, `round_trip_windows`, `round_trips`, `finish_reason`, plus
`scenario_id` / `run_number` so one file holds a whole capability. `seconds` remains the measured
provider latency used by the drift audit; `charged_seconds` is what moved the simulated clock.
Each `round_trip_windows` entry preserves one physical crossing's monotonic start/end, charge, and
`charged_started_at`: the scenario elapsed time on a separate parallel-charge sensitivity axis
when that crossing was admitted. Calls admitted before any sibling settles share a coordinate;
after a completion, a newly admitted follow-up starts at the furthest settled parallel endpoint.
The trajectory's serialized accumulator is deliberately not used for this coordinate. That makes
two distinct overlap diagnostics recoverable offline without another paid run: provider wall
intervals `[started_at, finished_at]`, and policy-relevant charged intervals
`[charged_started_at, charged_started_at + charged_seconds]`. `round_trips > 1`
is one decision that crossed the wire more than once (a parser-repair pass on the S-ORA side), with
the token fields summed across them and one fixed charge term paid per crossing.

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

Every scenario constructs one `TotalInputCharge` from the frozen `charge_model.json` and reuses it
for all of that scenario's calls. The environment clock is frozen while a physical model crossing
is in flight, then advanced by its token charge. The fixed primary trajectory policy is
`serialized_sum`: concurrent S-ORA calls are depth-counted so an early completion cannot resume the
world underneath another call, and every overlapping call's charge is summed as though inference
used one lane. This is deliberately conservative against S-ORA's asynchronous architecture. Run
results report both wall sum/union and charged sum/union. The scenario row and reports name the
concurrency diagnostics exactly as `llm_round_trips`, `llm_max_in_flight`, and
`llm_overlapped_round_trips`; these mean all physical crossings, maximum simultaneous crossings on
the host monotonic axis, and crossings sharing positive wall duration with another, respectively.
The charged union is an online settled-frontier sensitivity, not a reconstructed causal DAG. An
independent call admitted after an unrelated completion is conservatively placed at the settled
frontier, so union elapsed is conservative-high and inferred parallel savings are a lower bound.
Neither union retroactively changes the `serialized_sum` trajectory. Concurrency
does not require multiple activities: background condition-retirement judgement deliberately runs
without occupying an activity's `pending_inference`, and can overlap that activity's plan call.
ReAct's malformed-output retries remain inside ARE's pause bracket, but the harness deliberately
discards ARE's `completion_duration` offset and resumes with the independently computed bracket
charge so measured provider latency cannot leak into the frozen clock. Missing cache
detail is charged as uncached input; a call with no usage still pays the fixed term. A provider
report with cached input greater than total input clamps only the negative uncached residual. Each
run records the charge-model identity and digest, the clamp count, and a raw-usage anomaly count
computed independently. A mismatch is recorded without aborting a paid batch; offline report
verification rejects it.

The dated [charged-clock experiment protocol](evaluation/charged-clock-protocol-2026-09-19.md)
records the primary policy, sensitivity semantics, frozen artifact hashes, ex-ante overlap
expectation, and operational gates. It becomes the pre-sweep checkpoint when committed together
with the exact paid-scenario manifest before results are collected.

ARE does not model tool-execution duration: its apps execute locally and expose no authored
latency. Tool calls therefore acquire no generation-pause token. Their Python execution time is an
implementation artifact rather than a quantity either arm is charged for; a Time smoke should
still measure it once against simulated elapsed before the sweep. `--wall-clock` on the scenario,
batch, ReAct, and evaluation drivers disables token charging for a separately labeled robustness
run. The charged environment remains installed for its depth-counted judge pause and safe teardown,
so this mode means wall-timed agent inference, not byte-for-byte stock ARE internals.

Rows for a scenario are written when that scenario's recording ends, not as each call returns: a
repaired call is two round-trips on one `call_id` and nothing in the stream marks the last of them,
so the row can only be settled once the scenario's stream has. `batch.py` truncates the file once at
the start of a capability sweep, alongside the `output.jsonl` it replaces — otherwise a re-run's
rows would sit next to the previous sweep's under the same `scenario_id` and `run_number`, and the
fit would count the stale ones a second time. `run_benchmark.py --llm-calls` appends, so several
single-scenario runs can be pointed at one file deliberately. Monotonic timestamps are comparable
only within one process invocation; split an appended file by invocation before computing interval
unions across it.

`examples.gaia2.react_engine.MeteredLiteLLMEngine` writes the same schema with `arm: "react"` for
the ARE baseline, whose stock engine reports nothing at all — see that module for why ARE's own
agent is charged zero for thinking.

It also puts the baseline on the same request as the other arm. ARE's `LiteLLMEngine` ignores its
own `**kwargs` and hands `litellm.completion` five fixed arguments, so an uninstrumented baseline
runs at the provider's defaults — no reasoning setting, no output cap, no provider routing — while
S-ORA runs at the profile's. Build it with `MeteredLiteLLMEngine.from_profile(profile)` and both
arms send what `ModelProfile.request_kwargs()` says. Both changes are applied where the response is
already intercepted; ARE's own `chat_completion` body still runs untouched, including the
`True`/`False` lowercasing its checkers depend on.

That path reaches the provider through LiteLLM rather than through an SDK client, so it resolves
`provider`/`model`/`endpoint` by its own rules — rules no test that fakes the wire exercises, and
that a repin can invalidate silently. `react_driver.py --preflight` is the guard, the same contract
the grid's has for the other arm's client:

```bash
python3 -m examples.gaia2.react_driver --profile kimi-k2.5-prompt --preflight
```

It needs no scenario, sends one cheap call, and exits non-zero so a script chaining it into a sweep
stops rather than discovering the problem partway through a paid run. All three profiles passed on
2026-09-13 — resolved and billed, non-streamed, both usage detail blocks present, the
`kimi-k2.5-prompt` pin served by Venice.

A returning call is not by itself evidence that the arm runs at the profile's operating point,
which is why the route check is only half of it. LiteLLM's failure mode here is to *drop* a
parameter rather than raise — `drop_params` is what it recommends — and a dropped `reasoning_effort`
still comes back with a plausible answer at the provider's default, which is precisely the arm-to-arm
drift the sweep's own guard refuses to start with. So each probe sends a value only the far end can
refuse, and being refused is the pass: `reasoning_effort: supreme` and `service_tier: supreme` each
return OpenAI's own enumeration of accepted values, and a nonexistent name in the provider pin
returns OpenRouter's list of who really serves the model. Neither reply is producible by a request that never left the process.
`UnsupportedParamsError` is the one refusal that means the opposite — LiteLLM's own table rejected
the call before sending it, the failure `allowed_openai_params` exists to prevent — so it is
reported as a drop, not a crossing.

Crossing is not honoring, though, and one setting can be checked for both. Every shipped profile
asks the model to think — two by `reasoning_effort`, `kimi-k2.5-prompt` by OpenRouter's `reasoning`
block inside `extra_body` — so the route check also fails a call that reports zero reasoning tokens
against a profile that asked for some. That block declares no level, so this is the only honoring
question it can be asked. A zero has two readings and the next step differs between them: the
endpoint ignored the request, or a repin landed on one that does not itemize reasoning separately.
Both shipped endpoints itemize it today.

Only probes whose refusal has actually been observed are shipped, because a provider that clamps an
out-of-range value instead of refusing it would be reported as having dropped the setting — a false
alarm on the one check whose job is to be trusted. Everything else the profile sends is printed as
unprobed rather than passed over, so the coverage gap is visible: today that is
`max_completion_tokens`, `temperature` and `extra_headers`.

### What the profiles pin about *how* the call is served

A latency number is only reproducible if the conditions it was measured under are recorded, and two
of those live outside the request the profile describes.

`service_tier` is the first. Left unset OpenAI resolves it to `auto`, which picks a tier from
whatever credits the account happens to hold — so the serving condition would be defined outside
this repo and free to move between two runs the profile digest calls identical. Both OpenAI
profiles now pin it to `default`. Measured on 2026-09-13, that is a *recording rather than a
re-baseline*: unset, `auto` and `default` all resolved to `default` on this account, so nothing
about the operating point moved and numbers recorded before the pin stay comparable with numbers
recorded after it.

**`default` is pinned rather than any other tier because it is the only one observed to resolve to
itself.** OpenAI validates this parameter — an invented value is refused with the enumeration — but
validating is not honoring: `fast` is *accepted* and served as `priority`, which the response
reports back. So the wire probe proves the parameter crossed and still cannot prove the arm ran at
the named tier, and a profile pinning `fast` would record a serving condition that was never in
force. `tests/test_gaia2_evaluation.py` holds the set of tiers whose resolution has actually been
observed and reddens on a pin outside it; widening that set means re-measuring, not editing it.
OpenRouter has no such parameter at all — it answers 200 and echoes `service_tier: null` — so
`kimi-k2.5-prompt` leaves it omitted rather than claiming a pin that does nothing.

The model id is the second, and it cannot be pinned. OpenRouter ids name a *family*:
`moonshotai/kimi-k2.5` is served today from the snapshot `moonshotai/kimi-k2.5-0127`, and the dated
form is not an addressable alternative — the API accepts it, routes it identically, and normalizes
it straight back to the alias in the response, so there is no request a profile can send that names
a snapshot. What is available is noticing the swap: the models listing reports each id's
`canonical_slug`, and a repointed alias reports a different one. Both the ReAct preflight and the
latency grid read it (a free GET, no tokens) and refuse on a mismatch, because every latency
recorded under the old slug was measured on a different model. The grid checks it in its own right
rather than inheriting the preflight's: the preflight guards one arm's *routing*, while the grid is
where the seconds the charge model is fitted from are actually bought, and nothing obliges an
operator to run the one before the other. A listing that will not load is reported as unverified
rather than failed — it says nothing either way, and reddening the gate on a network blip is what a
check whose job is to be trusted cannot afford.

Because the id names a family, the profile digest cannot see a repoint at all: the profile is
byte-identical on either side of one. So the resolved slug is recorded on every manifest row, and
resume compares it alongside the digest — the digest catches a repin, a changed thinking budget or
a transport flip, and only the slug catches the model moving underneath an unchanged profile.

Transport follows the profile too, through a separate field. `stream` is read by the latency grid
as well as by both arms, so all three always sit on one transport — a per-call latency coefficient
fitted on one does not price an arm running on another. The shipped profiles are **non-streamed**,
which is also ARE's native behaviour. The cost is that `stall_timeout` stops meaning "the provider
went quiet" — observable only between chunks — and becomes a total-duration cap, so it is sized as
one (600s). Outside a measured comparison, streaming is the better default: it is what lets that
timeout tell a stalled connection from a slow one.

A streaming trap is what forced the choice, and nothing downstream could detect it: LiteLLM's
`stream_chunk_builder` fills a *missing* usage block by re-tokenizing the prompt and completion
locally, so a provider that ignores `include_usage` produces token counts that are plausible,
wrong, and indistinguishable from reported ones. Rows therefore take their tokens from the
provider's own trailing usage chunk and never from the rebuilt response — necessary but not
sufficient: on OpenRouter the usage chunk is *itself* assembled that way, measured there as
content-only completion tokens with no cache detail at all. A stream that reported no usage is
written `usage_captured: false` with null tokens and charged the fixed per-call term, which
undercounts visibly instead of mis-fitting silently.

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

# send the four corner cells once, for real, and report what came back
python3 -m examples.gaia2.latency_grid --profile gpt-5.4-high-paper --preflight \
    --out grid/preflight-gpt-5.4-high.jsonl

# run it
python3 -m examples.gaia2.latency_grid --profile gpt-5.4-high-paper \
    --out grid/gpt-5.4-high.jsonl
```

`--preflight` comes between the two, and its exit status is the point: a corner that was refused,
or that came back with no output tokens, exits non-zero, so a script chaining preflight into the
paid grid stops instead of proceeding. Both ends fail for reasons that appear nowhere in the middle
of the design — a reasoning model may reject an output cap as small as the bottom level, and the
top cell is the longest prompt the profile will ever send — and either refusal otherwise surfaces
partway through a $10 run. It caught a live one: a profile pinned to an OpenRouter provider that
had stopped serving its model refused all four corners with `NotFoundError`, which is a routing
failure the grid itself would have reported as four dead cells. Repinning that profile then
exposed a second refusal hiding behind the first, which is the better argument for the step: with
`require_parameters: true` OpenRouter keeps only providers declaring support for every parameter
sent, and *no* provider serving that model declares `max_completion_tokens` — they all declare the
older `max_tokens` — so the modern field the OpenAI SDK emits emptied the candidate set on a name,
not a capability. The flag was never the guard it looked like either. Asked which of the six
providers serving that model honour `reasoning`, all six say they do; asked to answer a question,
one returns `reasoning_tokens: 0` and folds its thinking into `content`, which for an agent whose
planner parses JSON is a defect and not a preference. A serving condition is established by sending
a request and reading the usage block back, which is what these four calls are — and they read more
than the exit status. `max_completion_tokens` means two different things in the wild: on some
endpoints it bounds the whole completion, so reasoning spends the budget and the smallest caps come
back with empty content; on others it bounds the content alone and reasoning runs past the cap.
Forcing an output length is how this grid measures a decode rate, so the second kind reports counts
that miss their target, and at a 10% tolerance a couple of hundred reasoning tokens is enough to
disqualify every cell at the three lowest output levels — taking `a0`, which only the lowest level
pins, and half the cached arm with them. One endpoint was observed alternating between both meanings
on identical requests, which is worse than either alone: what survives is then a self-selected
subsample rather than an honestly missing row. The preflight exits zero on all of this, because an
off-target count is not a refusal — it is something these four calls let you read before the grid
runs, not something they gate. The corner calls are recorded like every other paid crossing, under
`phase: "preflight"`, which `fit_eligible` already excludes.

`--range-check FILE` is the other half and costs nothing: it reads a file, calls nothing, and so
takes no `--profile` at all —

```console
python3 -m examples.gaia2.latency_grid --range-check runs/react-pilot/llm_calls.jsonl
```

It bands an arm's recorded `llm_calls.jsonl` against the axes and applies the extension rule fixed
before the first pilot ran — **extend an axis by one level when the band beyond its top holds 5% or
more of that axis's token mass** — because a threshold chosen after seeing the numbers is not a
check. It extends only, never retracts: a level already in the design is what separates the
coefficients from each other. An axis no usable row reported is called out as
`INCONCLUSIVE` and exits non-zero rather than reading as in range: a pilot whose rows all failed
produces the same 0% beyond-top share as one that stayed neatly inside the design, and only one of
the two has actually been checked.

The ReAct pilot has since run: 50 calls over two scenarios, max prompt 22.9k and max completion
1.7k, nothing in the top input band, so neither axis moves. The number worth carrying out of it is
that **91% of the ReAct arm's input tokens came back cached**, against an S-ORA arm whose prompts
are unique per call — `R_cache` is nearly the whole of the baseline's input cost, not a refinement
on it.

After the coefficients are frozen, `charge_range.py` performs the stricter paired-arm audit against
the regressors the charge actually multiplies:

```console
python3 -m examples.gaia2.charge_range \
    --profile kimi-k2.5-prompt \
    --sora runs/sora/llm_calls.jsonl \
    --react runs/react/llm_calls.jsonl
```

The measured box is `[1000, 64000]` uncached input tokens, `[0, 64000]` cached input tokens, and
`[16, 8192]` decode tokens on both endpoints. Decode is stated in its physical terms and obtained
through the convention frozen in `ChargeModelSheet`; for both shipped endpoints it is exactly
`completion_tokens`, with reasoning already contained, rather than completion plus reasoning.
The checker reports, per arm and axis, both the fraction of calls and the fraction of token mass
outside the box. It fails closed on missing coverage or any value outside: applying the frozen
coefficient there is unmeasured extrapolation, not merely a more uncertain estimate. Repeat
`--sora` and `--react` to combine per-capability files. The script reads only existing artifacts
and never re-runs the untracked fit. A S-ORA parser repair can produce one logical row whose token
fields sum multiple provider round trips. The measured box applies to each crossing, so the checker
excludes that aggregate and fails it as uncertifiable rather than treating its sum as one request or
scaling the bounds and potentially hiding an individual outlier.

`charge_drift.py` is the other post-hoc check. An endpoint/profile digest catches a declared
repin, and the OpenRouter catalogue guard catches an alias resolving to a different canonical
model, but neither can see service move behind an unchanged identity. Run a paired pilot before
the sweep and repeat the command over the full call logs afterwards:

```console
python3 -m examples.gaia2.charge_drift \
    --profile kimi-k2.5-prompt \
    --sora runs/sora/llm_calls.jsonl \
    --react runs/react/llm_calls.jsonl
```

The audit applies the frozen coefficients to each real call, including one intercept per reported
round trip, and reports MAPE plus aggregate signed bias for each arm. It passes only when each
arm's absolute signed bias is at most 15% and the between-arm bias gap is at most 10 percentage
points. Those are drift gates, not an invitation to re-fit: a failure records that live service no
longer resembles the frozen accounting unit closely enough to validate its neutrality. Rows with
incomplete usage, missing cache detail, invalid token splits, or non-positive latency are reported
as excluded; an arm with no usable rows fails closed. The file arguments are arm-specific and a
wrong arm or model is an error rather than a filtered row. Repeat `--sora` and `--react` to combine
the per-capability files in a whole sweep without rewriting them.

Run the paired range and drift audits before starting a paid sweep for both the OpenAI profile and
the Venice-pinned Kimi profile, plus a short Time-capability smoke in which all scheduled events
must arrive. Repeat the drift audit over the completed sweep logs. A failure is evidence that the
live endpoint no longer matches the frozen accounting unit; record it and stop rather than
re-fitting the coefficients inside the sweep.

The charge model is `a0 + uncached_in/R_in + cached_in/R_cache + out/R_out`, frozen before the
sweep and applied identically to both arms. Its coefficients cannot be fitted from the agents'
own calls: neither arm varies prompt length independently of answer length, so an ordinary
least-squares fit on stored trajectories returns a **negative** input coefficient — a longer prompt
served faster. The grid exists to move the two axes independently, which is the only thing that
identifies `R_in`.

Because this clock changes the trajectory itself, timing results produced before the charged-clock
integration are not comparable with results produced after it, even though the semantic prompts
and their frozen snapshot did not change.

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

Resume also refuses a manifest whose recorded `profile_sha256` is not the profile now being run.
`call_id` is derived from the cell and the block alone, so it is unchanged by a repin, a different
thinking budget or a transport flip — each of which moves the seconds this grid exists to measure.
Without the check a resumed run fills the remaining cells at a new operating point and hands the
fit one dataset that is really two, with nothing in the seconds marking where the seam is. A row
that records no digest at all counts as foreign for the same reason: it cannot be shown to match —
and so does a row that will not parse, which is the half-written last line an interrupted run
leaves behind. That line is read twice, to opposite ends: it is *not* a completed call, so its cell
is re-run, but it *is* an unknown for provenance, because dropping it would let a manifest of
nothing but bad lines pass a check it was never subjected to.

The same interrupt can also land one write earlier. A cell is written to the calls file first and
the manifest second, so a run killed between them leaves a call row with no manifest row at all —
invisible to every check above, since provenance lives on the manifest side. Its cell re-runs, which
is correct: a measurement with no experiment record next to it is not a completed cell. But the
uniqueness of `call_id` spans both files, so the ids the repeat must avoid are read from both, and
the repeat takes a suffixed id rather than the orphan's. Resume reports how many such rows it found
and continues — an orphan is the interrupted run `--resume` exists for, and refusing it would leave
re-measuring the whole grid as the only way out.

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
