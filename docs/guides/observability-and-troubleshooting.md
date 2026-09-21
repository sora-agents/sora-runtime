# Observability & Troubleshooting

S-ORA emits conventional Python logs for interactive inspection and can additionally send
structured runtime events to a context-local diagnostic sink. The structured observer is optional,
has no `Agent` or `DecisionCycle` back-reference, and is fail-soft: serialization, sink, and export
failures do not alter runtime control flow.

## Runtime event collection

Install a sink only around the work that needs tracing:

```python
from sora.diagnostics import RuntimeEventCollector, collect_runtime_events

collector = RuntimeEventCollector()
with collect_runtime_events(collector):
    await agent.run()

events = collector.snapshot()
```

Every collected row is JSON-safe and carries a monotonically assigned sequence number, elapsed
time, decision-cycle and phase context when available, activity ID, cause, and payload. Events cover
activity state and plan changes, pending operations and inferences, action dispatch/results,
cycle/phase entry and exit, and values crossing the environment boundary. Source timestamps are
retained when the underlying signal, message, or property observation supplies one.

Action payloads are deliberately bounded on the decision-cycle thread. Internal calls summarize
non-scalar runtime objects; external calls retain a capped structural preview with explicit
truncation markers. Data-ops preserve control fields such as `where`, `source`, and `out` within
that cap while summarizing the input collection, so an empty binding can be diagnosed without
walking the whole collection during the live run. Plan lifecycle and activity-context events use
the same bounded preview. Raw LLM exchanges and environment-specific traces remain the exact
evidence for their respective boundaries.

The decision cycle is correlation context rather than a second state machine. The environment is
likewise treated as a boundary: protocol- or simulator-specific traces remain authoritative for
their own semantics.

## Raw LLM exchanges

`capture_llm_exchanges()` installs the equivalent fail-soft, context-local observer around a
`MeteredLLMClient`. It sees the exact request and raw provider response (or exception), the logical
call ID, activity inference ID when present, and round-trip duration. Token counts remain owned by
provider-native `LLMUsage` records; capture does not estimate or duplicate accounting.

## Gaia evaluation bundles

Live runs from `python -m examples.gaia2.evaluation prompt run` buffer runtime events, LLM
exchanges, and the session log until ARE stops, then export a non-overwriting attempt bundle.
Diagnostics are enabled by default and can be disabled explicitly with `--no-diagnostics`.

An attempt's `index.json` lists hashes, byte sizes, and component export failures. The existing
`judge_recording.json` can be re-scored offline without a model call:

```console
python -m examples.gaia2.rescore .sora/gaia2/evaluations/prompt-baseline/artifacts/entry/attempt-0
```

This replay reconstructs ordered evidence and scoring; it does not promise deterministic agent or
simulator re-execution. Acceptance bundles are stored under a separate `artifacts/acceptance/`
subtree and are redacted from ordinary reports.
