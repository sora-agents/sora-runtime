# Quickstart

    $ git clone https://github.com/sora-agents/sora-runtime.git && cd sora-runtime
    $ uv sync --all-extras --dev
    $ uv run sora init ~/path/to/my-agent
    Created ~/path/to/my-agent/
      agent.yaml
      clock_tool.py
      manuals/clock.md
      pyproject.toml

    $ cd ~/path/to/my-agent
    $ uv sync                                 # the llm extra is already pinned into the dependency
    $ export ANTHROPIC_API_KEY=sk-ant-...     # credentials via the environment (see Configuring the LLM)
    $ uv run sora run
    +----------------------------------------------+
    | S-ORA -- minimal terminal interface          |
    |                                              |
    | Type a goal in plain English to delegate it. |
    | Type '/exit' or '/quit' to quit.             |
    +----------------------------------------------+
    model: claude-opus-4-8
    what time is it?
    [invoking clock.get_time...]
    It's 14:32.

`sora init <dir>` scaffolds a minimal, immediately-runnable example agent — `agent.yaml`, a
hand-authored `manuals/clock.md`, and `clock_tool.py` (a small `WorkspaceAdapter`/`Workspace`/`Tool`
trio implementing the clock tool directly, not via a real external server — there isn't one to
depend on for this). Real integrations import tools through an adapter (MCP, WoT, ...) instead of
hand-writing one — see [ADR-0003](../architecture/adrs/0003-adapters-not-tool-authoring.md) — `clock_tool.py` is
a deliberate, self-contained exception so the example needs nothing beyond an LLM key to run.

`sora run [config]` starts a persistent terminal session: it drives the decision cycle continuously, streams
external actions and messages as they happen, and reads terminal input as a `Message` (sender `"user"`)
for the next Observe phase — goals can be typed in at any point, not just at startup; Situate turns an
unhandled one into a new activity via _create_activity_. There's deliberately no `"> "` prompt — in a
plain line-buffered terminal it can't survive asynchronous output landing mid-line, so it would just be
misleading; the startup banner explains how to interact instead. `config` is an optional path to a
different `agent.yaml` (defaults to `agent.yaml` in the current directory, e.g. `sora run other-agent.yaml`).

## See also

- [Your First Agent](first-agent.md) — verbose/debug output, `agent.yaml` anatomy, and running a fuller example agent
- [Configuration](configuration.md) — the LLM provider extra and API key setup referenced above
