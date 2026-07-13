# Phase 3 test triage — `tests/test_cycle_wiring.py`

This is the A4 decision (see [ROADMAP.md](../ROADMAP.md), Track A). The walking skeleton left one
fast, ARE-free wiring file — [`tests/test_cycle_wiring.py`](../tests/test_cycle_wiring.py) — whose
own docstring called *all* of it "throwaway spike code." That blanket framing is now wrong: the
Phase-3 reorg turned most of these assertions into the natural first tests for the permanent build-out.
A4 records, per test group, **which assertions are promoted to permanent TDD tests, which scaffolding
is replaced, and which task owns the promotion** — so the "throwaway" framing is dropped *incrementally*
as each layer below is re-driven, not deleted wholesale up front.

## How the framing gets dropped (the mechanism)

The spike file stays and keeps running until each layer is re-driven — it is the only fast, deterministic
coverage of the tick path, and deleting it before its replacements exist would leave a gap. As each
Track B/C/D task lands, it **lifts its rows out of this file into the named permanent module** (adding
the assertions that the spike didn't cover) and deletes them here. When the last group leaves,
`test_cycle_wiring.py` is deleted outright. No task should carry the "throwaway spike" wording forward:
promoted assertions are permanent tests from the moment they land in their new home.

## Triage table

| Spike group (tests) | Fate | Owning task | Permanent home |
|---|---|---|---|
| **NotificationQueueSink** — `push_then_drain_yields_in_order`, `drain_is_empty_after_draining`, `drain_snapshots_current_depth` | **Promote (verbatim)** | independent (no gate); lands with D1, its first real consumer | `tests/test_perception.py` |
| **EnvironmentRegistry** — `join_registers_workspace_and_tools`, `leave_closes_and_deregisters` | **Promote + extend** | **C2** | `tests/test_environment.py` |
| **InvokeAction** — `invoke_action_sets_running_then_pushes_result` | **Promote** | **C4** | `tests/test_action.py` |
| **ActionRegistry** — `action_registry_lookup` | **Promote** | **C3/C4** | `tests/test_action.py` |
| **DefaultObserveStrategy** — `observe_resolves_running_activity` | **Promote** | **D1** | `tests/test_cycle.py` (or `test_strategies.py`) |
| **DecisionCycle.tick end-to-end** — `tick_end_to_end_invoke_then_resolve` | **Re-drive** (rewrite harness, keep outcome assertions) | **D4** | `tests/test_cycle.py` |

## Per-group rationale

### NotificationQueueSink — promote verbatim
Pure primitive tests with no cycle dependency and no contract change ahead of them. Keep all three.
`drain_snapshots_current_depth` in particular pins a real invariant — an item pushed *during* a drain
waits for the *next* drain — which is the Observe phase's no-starvation guarantee for `result_sink`/
`signal_sink` within a single cycle. It's worth a permanent test in its own right. No gate: the sink is
foundational, so these can move to `tests/test_perception.py` any time; they naturally land alongside
**D1**, which is the first phase to actually consume `result_sink.drain()`.

### EnvironmentRegistry join/get/leave — promote and extend under C2
The join/get/leave assertions are the permanent registry contract and stay. **C2 must add what the
spike does not cover:** [ADR-0014](adrs/0014-tool-identity-globally-unique.md) id-uniqueness — fail
loud on a duplicate `Tool.id` at `join`, and guarantee `leave` never pops a *shared* id (the
cross-workspace deregistration hazard in [phase-2-findings §6](phase-2-findings.md)) — plus `restore()`
(needs B2). Note the coupling to **A2**: once the registry moves off `WorkingMemory` onto
`DecisionCycle` and `WorkingMemory` exposes only a read-only `EnvironmentView`, the construction in
these tests changes; the *assertions* are unaffected.

### InvokeAction + ActionRegistry — promote under C4/C3
`invoke_action_sets_running_then_pushes_result` is the permanent `InvokeAction` contract: it transitions
the activity to `RUNNING` with `pending_operation` set, fires the tool round-trip off-cycle, and lands
the result on `result_sink` keyed by `op_id` (never as a `Percept` — the unconditional-wait path from
[CLAUDE.md](../CLAUDE.md)'s "two kinds of waiting"). Keep it; **C4** re-drives it as permanent.
`action_registry_lookup` is trivial but permanent — with A3 landed it should look up via
`InvokeAction.name`, not the `"invoke"` literal it still uses here.

### DefaultObserveStrategy — promote under D1
`observe_resolves_running_activity` is the permanent Observe contract for the running-resolution path:
a pushed `OperationAck` whose `op_id` matches `pending_operation` transitions the activity `RUNNING →
READY`, clears `pending_operation`, and sets `last_operation` — the automatic, unambiguous 1:1 match
with no strategy judgment and no `Percept`. **D1** re-drives it. (Note: `DefaultReflectStrategy`,
`DefaultSituateStrategy`, `DefaultActStrategy` are wired into the spike's `_cycle` helper but only
`DefaultObserveStrategy` is asserted here — the others get their own tests under D2/D3/D4.)

### DecisionCycle.tick end-to-end — re-drive under D4
The observable outcome — invoke `list_emails` exactly once, the off-cycle result lands, `last_operation`
is set — is permanent and its assertions survive. But the *harness* around them must be rewritten,
because the spike depends on three things Phase 3 changes:
- the §7a Situate `activity`-gate — fixed in **D3** (Situate always runs);
- the §7b `next_action == "invoke"` string branch in `tick()` — replaced in **D4** by the `_act()`
  bind-then-dispatch boundary, with binding declared by the action rather than a hardcoded cycle branch;
- the string-typed `ListEmailsReasonStrategy` fixture — superseded by **D4**'s deterministic
  `ReasonStrategy` plus A3's named step constants.

So D4 keeps the end-state assertions and the drive-until-resolved shape, but rebuilds the fixture and
drops the string special-casing.

## Scaffolding (not assertions) — replace, don't promote

- **`_cycle()` wiring helper** — constructs `WorkingMemory(registry=registry)` and threads the registry
  through both. **A2/C2 replace this**: the mutable registry handle moves onto `DecisionCycle`, and
  `WorkingMemory` exposes only a read-only `EnvironmentView`. The helper is rebuilt to match, not lifted
  verbatim.
- **In-file fakes** — promote into shared fixtures rather than copy-pasting: `FakeTool`/`FakeWorkspace`/
  `FakeAdapter` → the reusable fake adapter of **C1**; `DictBackend` → the in-memory double for **B1**'s
  file-backed backend; `NullTransport` → transport tests under **C3**.
- **`ListEmailsReasonStrategy`** — its *role* (advance to invoking once) is superseded by D4's
  deterministic strategy; it is not lifted verbatim.
- **The module docstring's "throwaway spike" wording** — dropped now (A4); replaced with a pointer to
  this triage.

## Summary

Of the nine spike tests, eight are promoted (two of them — the registry pair — extended with new C2
assertions) and one (the end-to-end tick) is re-driven with a rewritten harness but surviving
assertions. **No test's assertions are discarded.** Only spike scaffolding — the wiring helper, the
in-file fakes, the string-typed reason fixture, and the blanket framing — is replaced.
