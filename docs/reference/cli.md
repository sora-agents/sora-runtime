# CLI Reference

The exact flag set for `sora`, as defined in `sora.cli.main`
([source](https://github.com/sora-agents/sora-runtime/blob/main/src/sora/cli.py)). For a guided
walkthrough — what to run first, what the output means — see [Quickstart](../getting-started/quickstart.md)
and [Your First Agent](../getting-started/first-agent.md).

## `sora run [config]`

Starts a persistent terminal session.

| Argument/Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `config` | positional, optional | `agent.yaml` | Path to the agent config file. |
| `--verbose` | flag | off | Print the per-phase decision-cycle trace to the terminal. |
| `--log-file PATH` | str | none | Write the complete execution trace (per-phase cycle log, model prompts, operation results, plans) to this file — always full detail, independent of `--verbose`, which only controls what the terminal shows. |
| `--color` | flag | auto | Force ANSI color output. Mutually exclusive with `--no-color`. Default is auto: on only for a TTY, off if `NO_COLOR` is set. |
| `--no-color` | flag | — | Disable ANSI color output. Mutually exclusive with `--color`. |
| `--task TEXT` | str | none | Submit this text as the initial user message at startup. Mutually exclusive with `--task-file`/`--scenario`. |
| `--task-file PATH` | str | none | Read the initial user message from this file at startup. Mutually exclusive with `--task`/`--scenario`. |
| `--scenario REF` | str | none | An ARE scenario reference (dotted `Scenario` subclass path, or a Gaia2 `.json` file) — injected as the runtime `simulation` object for an `are-sim` workspace / `are` transport in `agent.yaml`. Mutually exclusive with `--task`/`--task-file`: the scenario delivers its own task through the `AgentUserInterface` timeline. |
| `--report DOTTED.PATH` | str | none | Call this `(agent, simulation) -> None` after the session ends — e.g. to print custom scoring/checks. |
| `--exit-when-idle SECONDS` | float | none (wait for stdin) | Auto-exit once every activity has stayed `TERMINATED` for this many seconds, instead of waiting for stdin — useful for scripted/headless runs. |

`--color`/`--no-color` form one mutually exclusive group; `--task`/`--task-file`/`--scenario` form
another.

## `sora init <dir>`

Scaffolds a minimal example agent.

| Argument | Type | Description |
| --- | --- | --- |
| `dir` | positional, required | Directory to create — must not already exist. |

---

!!! info "Hand-authored framing pending"
    This table is the exact, source-checked flag set. The operational guidance the design calls for
    — when to reach for `--exit-when-idle` vs. an external timeout, how `--report` fits a CI harness
    — is still owed as hand-written prose; track it against the reference-authoring task.
