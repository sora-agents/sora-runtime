# Your First Agent

Building on the [Quickstart](quickstart.md)'s `sora init`/`sora run` walkthrough: what `--verbose`/`--log-file` show you, how `agent.yaml` wires an agent together, and a fuller example agent to run next.

## Verbose and debug output

Use `--verbose` to print each decision-cycle phase instead of just the conversational output:

    [cycle 1] Observe  - message from user: "what time is it?"
    [cycle 1] Situate  - created activity=ask-time from message; loaded manual: clock
    [cycle 1] Reason   - plan: invoke clock.get_time
    [cycle 1] Act      - invoked clock.get_time -> ack
    [cycle 2] Observe  - perceived signal: clock.time_reported
    [cycle 2] Reflect  - activity ask-time completed; stored to episodic memory

Type `exit` or `quit` for a clean shutdown (leaves joined workspaces, closes any MCP subprocess);
Ctrl-D (EOF) does the same. `--task "..."` (or `--task-file path`) submits an initial goal at
startup, before you'd type anything yourself — useful for scripting a run non-interactively, e.g.
`sora run agent.yaml --task-file task.txt`, without needing to type it in by hand.

`--verbose` is a *display* setting, not a log level: it selects the one-line-per-event terminal view,
while the runtime separately emits finer `DEBUG` detail — the prompt behind each model call, each
operation's result, and the plan bodies themselves (as inferred, as reused from procedural memory, as
entered for a sub-goal, as re-spliced when a mechanical sub-goal fans out, and as discarded when
context-adaptation invalidates one, paired with the replacement inferred against the moved world).
`--log-file PATH` mirrors that complete trace to a file, always at full detail and independent of
what the terminal is showing:

    [cycle 4] Reason   - plan invalidated by context-adaptation for 'book the cheapest flight'
    [cycle 4] Reason   - discarded plan for activity trip (was at step 1)
    0: invoke {"tool_id": "Flights", "operation_name": "search"}
    1: invoke {"tool_id": "Flights", "operation_name": "book"}
    [cycle 6] Observe  - plan for activity trip
    0: invoke {"tool_id": "Flights", "operation_name": "search"}
    1: invoke {"tool_id": "Hotels", "operation_name": "search"}

There is no `--log-level` flag: `sora run` holds the `sora` logger at `DEBUG` and lets each handler
decide what to show, so a run recorded with `--log-file` never has to be repeated at a higher
verbosity to recover the detail.

Three more optional flags, mainly for driving an ARE scenario without a bespoke runner script (see
[Running the ARE examples](#running-the-are-examples)): `--scenario <ref>` injects a runtime
`AreSimulation` for an `are-sim` workspace/`are` transport (mutually exclusive with `--task`/
`--task-file` — the scenario delivers its own task through the `AgentUserInterface`); `--report
dotted.path` calls a `(agent, simulation) -> None` hook after the session ends, e.g. to print
custom scoring/checks; `--exit-when-idle SECONDS` auto-exits once every activity has stayed
`TERMINATED` for that long, instead of waiting on stdin — useful for a scripted/headless run.

## Anatomy of `agent.yaml`

`uv sync` installs pinned dependencies into a project-local `.venv` per `uv.lock` — commit the lockfile
so runs are reproducible. `uv run` executes inside that environment without manual activation.

`agent.yaml` wires the pluggable pieces:

    agent:
      name: my-agent
      strategies:
        reason: sora.reason.default   # observe/reflect/situate/act default to sora's built-in mechanical strategies
      memory:
        working: in_process
        semantic: file://./.sora/memory/semantic
        procedural: file://./.sora/memory/procedural
        episodic: file://./.sora/memory/episodic
      procedural:                     # optional: override ProceduralMemory's built-in prompts
        plan_prompt: my_agent.prompts.plan       # default: sora.memory.default_plan_prompt
        ground_prompt: my_agent.prompts.ground   # default: sora.memory.default_ground_prompt
      transport: http://localhost:8765
      workspaces:
        - origin: {adapter: mcp, address: "mcp://localhost:6000"}   # clock tool, imported via the MCP adapter

Workspaces declared here are joined automatically at startup, before the first cycle runs — which is
why the `[cycle 1]` trace above already has the clock manual loaded, with no explicit `_join_` shown.
The joined workspaces *are* the toolset the default Situate works from: each cycle it loads their
tools' manuals into working memory (`_load_`), unloads any no longer backed by a joined workspace
(`_unload_`), and filters *observable-property* percepts down to the joined workspaces' tools
(`_filter_`). `_filter_` prunes only properties — a re-observed snapshot, safe to drop and reproduced
next cycle. Signals are **fire-and-forget** and are never dropped by `_filter_`: a signal may still
matter to another (or a `blocked`) activity, so its retention/eviction is owned by the blocked-state
machinery's fixed retention cap, not a per-cycle prune — satisfying a wait never evicts a signal
early. As a temporary fallback the runtime auto-focuses a workspace's tools on `_join_`, so an agent
perceives its joined tools without a `focus` step. The intended path is still intentional focusing —
an external action (one per cycle, dispatched at Act) that a richer strategy emits as a `_focus_`
plan step (and `_unfocus_` to narrow observation cost); that override is unaffected.

## Running the ARE examples

Two runnable showcases drive S-ORA against Meta's [Agents Research Environments](https://github.com/facebookresearch/meta-agents-research-environments) (ARE). Both need a live model and the ARE package, so they live outside the pytest suite and share the same one-time setup — install the `are` dependency group and provide a key:

    $ uv sync --all-extras --group are
    $ export ANTHROPIC_API_KEY=sk-ant-...     # or a .gitignored .env, as above

**MCP path — static snapshot** (`examples/are/mcp/email_calendar/`). S-ORA's `AreMcpWorkspaceAdapter` connects over ARE's MCP server, which serves a scenario's *initial* app state; this fits the single-shot plan→ground→act loop:

    $ uv run python -m examples.are.mcp.email_calendar.run
    # or the same scenario through the CLI:
    $ uv run sora run examples/are/mcp/email_calendar/agent.yaml \
        --task-file examples/are/mcp/email_calendar/task.txt --verbose

**In-process path — dynamic timeline** (`examples/are/sim/email_calendar/`). To exercise a scenario's *event timeline* — mid-run email injections, follow-ups, task delivery — S-ORA runs the ARE `Environment` directly. The scenario is a **per-run input on the command line**, not config: a dotted `Scenario` subclass or a Gaia2 `.json` file, passed via `sora run`'s `--scenario` flag, which turns it into the `simulation` object `build_agent(config, simulation=...)` injects. This runs through the same `sora run`/`TerminalSession` trace as any other agent — `TerminalSession` only needs a transport with an outbound `.sent` log, not specifically `InProcessTransport`:

    $ uv run sora run examples/are/sim/email_calendar/agent.yaml \
        --scenario examples.are.sim.email_calendar.scenario.EmailScheduleScenario --verbose    # watch it interactively

    # headless + scored: auto-exit once quiescent, then print the agent's outcome + ARE's own
    # scenario.validate() score via a plain (agent, simulation) -> None hook (examples/are/sim/email_calendar/report.py)
    $ uv run sora run examples/are/sim/email_calendar/agent.yaml \
        --scenario examples.are.sim.email_calendar.scenario.EmailScheduleScenario \
        --report examples.are.sim.email_calendar.report.report --exit-when-idle 8

**Scenarios you can run.** The in-process path is scenario-agnostic — `--scenario` is required (there is no default) and accepts any of three forms:

- **`examples.are.sim.email_calendar.scenario.EmailScheduleScenario`** — the bundled illustrative scenario. A *dynamic* scenario: Alice emails to schedule a Monday team sync, then a follow-up email mid-run moves it to Tuesday, surfacing as a `state_changed` signal that drives a replan.
- **A Gaia2 `.json` benchmark scenario** — any scenario file from Meta's Gaia2 benchmark (distributed with [ARE](https://github.com/facebookresearch/meta-agents-research-environments)), loaded through ARE's benchmark scenario loader. These are not vendored in this repo; point at your own copy.
- **A dotted `Scenario` subclass you author** — subclass ARE's `Scenario` (see `examples/are/sim/email_calendar/scenario.py` for the template) and pass its dotted path.

Separately, the **MCP path** runs one seeded static scenario, `examples/are/mcp/email_calendar` (a 30-minute team sync from an inbox email); it is fixed by that example's `agent.yaml`/`task.txt` rather than selected with `--scenario`.

The MCP path's standalone `examples/are/mcp/email_calendar/run.py` drives the decision cycle until the activity terminates, prints the trajectory, and prints the runtime's own INFO trace via plain `logging.basicConfig`; raise or lower it with `LOGLEVEL` (e.g. `LOGLEVEL=WARNING`) — it exists primarily as the reference example for [driving an agent programmatically](../guides/cli-and-programmatic-runs.md#driving-an-agent-programmatically), without `TerminalSession`. The in-process dynamic path runs entirely through `sora run` (above), so its trace/trajectory/footer are the same colored `[cycle N] Phase - ...` output (or terse `[invoking ...]` cues) as any other `sora run` session, controlled by `--verbose`/`--color`/`--no-color` (and `--log-file` for the full-detail file mirror), not `LOGLEVEL`. For how the two adapter paths differ and why the dynamic path exists, see [EXAMPLES.md](https://github.com/sora-agents/sora-runtime/blob/main/EXAMPLES.md#running-dynamic-scenarios-in-process) and the [ARE dynamic scenarios design note](../architecture/notes/are-dynamic-scenarios.md).

## See also

- [Configuration](configuration.md) — LLM provider setup, API keys
- [CLI & Programmatic Runs](../guides/cli-and-programmatic-runs.md) — driving an `Agent` from your own code
- [Workspace & Tool Integration](../guides/workspace-and-tool-integration.md) — connecting to a remote vs. local MCP server
- [Custom Strategies](../guides/custom-strategies.md) — customizing the planning/grounding prompts
