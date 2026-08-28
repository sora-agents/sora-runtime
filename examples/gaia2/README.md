# Gaia2 benchmark harness

Running S-ORA against Meta's [Gaia2](https://huggingface.co/datasets/meta-agents-research-environments/gaia2)
scenarios, on top of the ARE simulator. Three entry points:

| Script | Use it to |
|---|---|
| `scripts/fetch_scenario.py` | Pull one scenario JSON down to inspect or replay locally |
| `run_benchmark.py` | Run **one** scenario, print the judge verdict |
| `batch.py` | Run a whole capability, emit leaderboard artifacts, report pass@1 |

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
| `--init-turns` | Deliver every turn of a multi-turn scenario **without** a judge. Excludes `--judge-model` |
| `--max-wall-seconds` | Safety cap, default 1200 |
| `--exit-when-idle SECONDS` | Old single-turn quiet-window stop. Only correct for single-turn scenarios |
| `--verbose` / `--log-file PATH` | Stream / mirror the full trajectory |

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

Writes, under `{output-dir}/standard/{capability}/`, one HF-format trace per (scenario, run) plus
`output.jsonl` in ARE's own benchmark-result shape. `--report-only` prints per-capability pass@1 and
the equal-weight overall across the five core capabilities.

Flags mirror `run_benchmark.py`, plus: `--capability` (the dataset config to run), `--split`,
`--hf-dataset`, `--output-dir`, `--num-runs` (Gaia2 uses 3), `--limit`, and `--model` (the label
recorded in the trace — match it to `agent.yaml`'s `llm.model`).

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

## Notes

Known gaps, scenario-level findings, and benchmark caveats live in [NOTES.md](NOTES.md).
