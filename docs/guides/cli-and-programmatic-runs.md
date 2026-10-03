# CLI & Programmatic Runs

## Driving an agent programmatically

`sora run` is one way to run an `Agent` — the terminal CLI. Embedding S-ORA in your own program
(a test harness, an evaluation runner, a service) instead means calling `build_agent()` and
`Agent.run()`/`stop()` directly, without `TerminalSession` at all — `examples/are/mcp/email_calendar/run.py`
is a runnable reference for that shape: build the agent, `transport.submit()` an initial `Message`
(what `sora run --task` does for you at the CLI), drive `agent.run()` as a background task, poll for
the condition you care about (an activity reaching `TERMINATED`, a timeout), then `await agent.stop()`
and cancel/await the task in a `finally` for teardown.

Awaiting that task is not just tidiness — it is the teardown. `Agent.run()` unwinds its own
`finally`: it leaves every workspace it joined (closing MCP sessions and their subprocesses) and
releases the model client's HTTP connection pool. Cancelling without awaiting skips all of it, and
a pool left open is finalized by the garbage collector after the event loop is gone, which asyncio
reports as a bare `Task exception was never retrieved ... RuntimeError('Event loop is closed')`
with no stack into your code. Teardown lives after the loop rather than in `stop()` so that it
cannot race a tick still in flight.

## See also

- [Quickstart](../getting-started/quickstart.md) — the `sora run` CLI basics
- [Your First Agent](../getting-started/first-agent.md) — `--verbose`/`--log-file`/`--task`/`--scenario`/`--report`/`--exit-when-idle` flags
