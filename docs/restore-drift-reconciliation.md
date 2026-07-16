# `restore()` and environment drift — analysis and open gap

Design note behind the dynamic-environments / restore-drift task. Records why `restore()` does **not**
handle a workspace whose live tool set has changed since it was last joined, what actually happens
when it drifts, and the options for closing the gap. No decision is made here — this is the analysis
that motivates the task; the decision belongs in an adapter ADR once the shape is chosen (it depends
on the MCP-adapter hardening).

## What `restore()` does today

`restore()` is the **fast, deterministic reconnect-from-records path**, deliberately distinct from
`join()`:

- It iterates only the persisted `ToolRecord`s handed to it and passes exactly that set to
  `adapter.connect()`. It **never calls `discover()`** — the docstring says *"Skips discovery
  entirely."*
- `WorkspaceAdapter.connect()`'s contract is *"all its tools rebuilt"* **from the records**, *"no
  re-fetching manuals"* — manuals are resolved from `SemanticMemory`, not the live server.
- It performs **no write-back**: `last_seen_at` is untouched and nothing is re-persisted. It is a pure
  read-side reconstruction of the last-known snapshot.

This is intentional. [ADR-0014](adrs/0014-tool-identity-globally-unique.md) wants *"the same records
rebuild the same ids across runs"* — `restore()` trades freshness for speed and reproducibility.
Freshness is supposed to come from the *other* path: a deliberate `join()`, which calls `discover()`
and re-persists via `JoinAction` (`store_workspace_record` / `store_tool_record` / `store_manual`).
`restore()` reconstitutes what the agent knew; `join()` re-learns what is there.

## What happens when the workspace has drifted

| Drift since last join | `restore()` behavior |
|---|---|
| **Tool added** | Silently absent — no `ToolRecord`, so `connect()` never rebuilds it. Not registered, not focusable, not in `all_tools()` until a fresh `join()`. May be present on the wire but ignored. |
| **Tool removed** | Its record still exists → `connect()` is asked to rebuild a tool that is gone. **Adapter-dependent** (see gap below): either it fails/skips, or it builds a **stale handle** whose failure is deferred to `invoke()`/`focus()` time. |
| **Manual / schema changed** (same id, new ops) | Manual is resolved from `SemanticMemory` — the **stale** persisted manual, by design. The agent reasons over an outdated interface. |
| **Id / address changed** | `restore()` uses the persisted `id`+`address` as-is (no re-derivation), so it builds a handle under the old id, possibly pointing at the wrong endpoint. |
| **Duplicate produced** | `_register` still fails loud — the ADR-0014 backstop is intact on this path too. |

The headline case: **a newly-added tool is invisible after `restore()`**, silently, until a re-join.

## The underspecified spot

For a **removed** tool, the runtime contract does not say whether `connect()` should eagerly validate
against the live session (raise or skip the dead tool) or lazily build a handle and let the failure
surface later at invoke time. The current in-process fake (`tests/fakes.py`) does the lazy thing
unconditionally. "connect eager-validates vs. lazy-rebuilds" changes whether a stale record is a fast
failure or a deferred one, and belongs in the adapter ADR written during MCP-adapter hardening.

## The gap this task addresses

`restore()` is doing exactly what it says, but "what it says" assumes the world did not move. There is
currently **no reconciliation / refresh action and no drift detection**: an agent that restores at
startup into a changed environment silently runs on a stale world model, with nothing prompting a
re-join. Whether that is acceptable depends on the target — fine for short-lived or static
environments, a real problem for long-lived agents against mutable MCP servers.

Options to weigh (not yet decided):

1. **A `refresh` / `resync` external action** — re-`discover()` a joined workspace, diff against the
   registry + records, register/deregister the delta, and re-persist. Keeps `restore()` pure and
   fast; makes reconciliation an explicit, agent-driven decision (consistent with join/leave being
   deliberate, per README's Tool Model).
2. **A reconciling `restore()` variant** — restore-then-`discover()`-and-diff at startup, behind a
   flag, for agents that always want a fresh view. Costs the discovery round-trip `restore()` exists
   to avoid, so it is opt-in.
3. **Leave `restore()` as-is and document the contract loudly** — freshness is *only* ever obtained
   via `join()`; `restore()` is snapshot-only by definition. Cheapest, but pushes the drift burden
   onto every agent author.

Cross-references: builds on the ADR-0014 identity model and the join/leave lifecycle
([ADR-0006](adrs/0006-workspace-join-leave-lifecycle.md)); the removed-tool `connect()` semantics
couple to the MCP-adapter hardening in Track E. The manual-drift row also touches the two-channel
`Manual` reconciliation deferred to the Manual-reconciliation task.
