# Join/leave as deliberate actions; discovery kept distinct from reconnection

* Status: accepted
* Date: 2026-07-05

## Context and Problem Statement

The original design had a single `_discover_` external action that eagerly scanned every configured adapter and registered everything found. This contradicts the runtime's own philosophy that an activity should get a filtered view of the environment relevant to it — eagerly connecting to every configured workspace regardless of need doesn't fit that model, and conflates "finding a workspace for the first time" with "reconnecting to one already known."

## Decision Drivers

* Avoid unnecessary, eager connections at agent startup
* Keep environment access agent-driven and activity-scoped, matching the runtime's general philosophy
* Anticipate — without building yet — open environments where not every workspace is known in advance

## Considered Options

* Keep a single bulk `_discover_` action, run at startup across all configured adapters
* Replace it with `_join_`/`_leave_` external actions operating on one workspace at a time, config-scoped for now: agents decide when to `_join_`/`_leave_` a workspace for a selected activity, but for now can only choose among pre-configured workspaces (for security concerns, i.e. an agent cannot autonomously connect to any workspace discovered at run time)
* Additionally support dynamic discovery of previously-unknown workspaces now

## Decision Outcome

Chosen option: "`_join_`/`_leave_`, config-scoped", with dynamic discovery of unknown workspaces explicitly deferred as a foreseen future extension rather than built now. `WorkspaceAdapter` correspondingly exposes two distinct operations: `discover()` (a real scan/enumeration, used the first time a workspace is joined) and `connect()` (reconnects from cached `WorkspaceRecord`/`ToolRecord`s, skipping the scan and manual re-fetch). `EnvironmentRegistry` is keyed by the full `WorkspaceOrigin` (adapter + address), not just the adapter name, so an agent can join multiple workspaces that share a protocol without ambiguity.

### Positive Consequences

* Connections are established only when an activity actually needs them, not eagerly at startup
* Symmetric, explicit teardown via `_leave_`, calling `workspace.close()` once for all its tools
* Restoring from a previous session reuses cached manuals/records instead of re-scanning or re-fetching

### Negative Consequences

* `_join_` needs an addressable target; today that target can only be something declared in `agent.yaml` — there is no way yet to discover a workspace nobody configured in advance

## Links

* Depends on [ADR-0005](0005-workspace-grouping.md)
* Refines [ADR-0007](0007-manual-record-separation.md)
