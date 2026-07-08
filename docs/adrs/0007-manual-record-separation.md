# Manuals and tool/workspace records stored as separate entities

* Status: accepted
* Date: 2026-07-05

## Context and Problem Statement

The initial persistence design keyed `SemanticMemory` storage by tool instance id, storing a `Manual` per instance. This conflates "knowledge about a tool type" (its manual — shared by design across every instance of that type) with "knowledge about one specific discovered instance" (its address, discovery timestamps). It also duplicated workspace-level connection info (adapter, address) onto every tool record from the same connection.

## Decision Drivers

* Avoid storing the same manual content once per instance when many instances share one type (e.g., many identical AC units)
* Avoid duplicating workspace connection info onto every tool record that shares a connection
* Support reconnection from cached records after a restart, without needing a live connection or re-discovery

## Considered Options

* Key everything by tool instance id (original design)
* Split `Manual` (type-level, own `id`) from `ToolRecord` (instance-level, references `manual_id` and `workspace_id`) and `WorkspaceRecord` (connection info, referenced by `ToolRecord`, not duplicated onto it)

## Decision Outcome

Chosen option: "Split Manual / ToolRecord / WorkspaceRecord", because it lets many tool instances share one manual and lets many tool records from the same connection share one `WorkspaceRecord`, rather than duplicating either.

### Positive Consequences

* Manuals are stored and retrieved once regardless of how many instances share them
* Workspace connection info (adapter, address) lives in exactly one place, referenced by id
* `EnvironmentRegistry.restore()` can reconnect from cached records without a live connection or a fresh scan

### Negative Consequences

* Adds several types (`Manual.id`, `ToolRecord`, `WorkspaceRecord`, `WorkspaceOrigin`) and a small amount of indirection (e.g., `record.address or workspace_record.origin.address` fallback) compared to a flat, single-table design

## Links

* Depends on [ADR-0005](0005-workspace-grouping.md), [ADR-0006](0006-workspace-join-leave-lifecycle.md)
* These records live in `SemanticMemory`, alongside manuals — see README.md's Memory section
