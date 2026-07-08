# Workspace groups tools sharing a connection; per-tool address override

* Status: accepted
* Date: 2026-07-05

## Context and Problem Statement

It was initially unclear whether one external server/session (e.g., one MCP server) should map to one S-ORA `Tool` (with many operations) or many `Tool`s (one per operation) — and neither framing cleanly supports a WoT-style environment --- or, more precisely, a Hypermedia Multi-Agent System style environment --- where a workspace can host virtual tools directly while grouping physical devices that live at their own, different network addresses.

## Decision Drivers

* Support heterogeneous addressing (a lab environment mixing co-located virtual tools and remote physical devices)
* Need a shared connection/lifecycle boundary distinct from individual tool identity
* Inspiration from the A&A "workspace" concept for grouping related artifacts, and the more recent evolution of A&A workspaces towards standalone, deployable bounded contexts in CArtAgO

## Considered Options

* One `Tool` per external server/connection, with operations aggregated
* One `Tool` per external operation, with adapter-config-driven granularity and no shared grouping concept
* Introduce `Workspace` as an explicit grouping construct, with an optional per-tool address override

## Decision Outcome

Chosen option: "Introduce `Workspace`", because it supports both aggregation styles as adapter policy (nothing forces one granularity) while adding a real capability neither alternative had: a shared connection/lifecycle boundary whose individual tools can still be addressed independently. A workspace's adapter fixes the tool-use protocol for everything inside it (e.g., all-MCP or all-WoT), while `Tool.address` overrides the workspace's own address only when a specific tool isn't reachable through it.

### Positive Consequences

* Supports mixed environments (e.g., a lab with virtual tools on a hub server and physical devices on their own addresses) without special-casing
* The underlying connection/session is established and torn down once per workspace, not once per tool
* Tool granularity (one Tool per server vs. one per operation) remains an adapter policy choice, not a runtime constraint

### Negative Consequences

* Adds a new top-level concept beyond the runtime's original six Main Concepts groupings
* Requires `WorkspaceOrigin`/`WorkspaceRecord` types and a registry that tracks live workspaces, not just a flat list of tools

## Links

* Depends on [ADR-0003](0003-adapters-not-tool-authoring.md), [ADR-0004](0004-tool-usage-interface.md)
* Refined by [ADR-0006](0006-workspace-join-leave-lifecycle.md), [ADR-0007](0007-manual-record-separation.md)
