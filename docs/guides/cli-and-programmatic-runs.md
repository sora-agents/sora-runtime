# CLI & Programmatic Runs

## Driving an agent programmatically

`sora run` is one way to run an `Agent` — the terminal CLI. Embedding S-ORA in your own program
(a test harness, an evaluation runner, a service) instead means calling `build_agent()` and
`Agent.run()`/`stop()` directly, without `TerminalSession` at all — `examples/are/mcp/email_calendar/run.py`
is a runnable reference for that shape: build the agent, `transport.submit()` an initial `Message`
(what `sora run --task` does for you at the CLI), drive `agent.run()` as a background task, poll for
the condition you care about (an activity reaching `TERMINATED`, a timeout), then `await agent.stop()`
and cancel/await the task in a `finally` for teardown.

## See also

- [Quickstart](../getting-started/quickstart.md) — the `sora run` CLI basics
- [Your First Agent](../getting-started/first-agent.md) — `--verbose`/`--log-file`/`--task`/`--scenario`/`--report`/`--exit-when-idle` flags
