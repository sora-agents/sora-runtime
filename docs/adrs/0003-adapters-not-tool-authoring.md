# Adapters import tools; the runtime never authors them

* Status: accepted
* Date: 2026-07-05

## Context and Problem Statement

Should S-ORA define its own tool-authoring framework (a way to declare new tools from scratch), or should it consume tools already defined by external ecosystems (MCP, OpenAPI, WoT, etc.)? The conundrum: most existing ecosystems do not define tools exactly per S-ORA's usage-interface model (observable properties + signals + operations), so some gap-filling is unavoidable either way. One noticeable exception is [Yggdrasil](https://github.com/interactions-HSG/yggdrasil) and the work around hMAS, which is also inspired by A&A and comes closest.

## Decision Drivers

* Avoid duplicating effort against mature, widely-adopted tool ecosystems
* Keep the runtime's scope focused on the agent decision cycle, not tool definition
* Preserve the option to extract the usage-interface model into its own spec later, without a rewrite

## Considered Options

* Build a native S-ORA tool-authoring framework
* Consume tools from external ecosystems via adapters, approximating missing pieces where the source ecosystem lacks them
* Spin off a separate "tool model" framework/spec immediately

## Decision Outcome

Chosen option: "Consume tools from external ecosystems via adapters", because it leverages ecosystems that already exist (MCP, WoT) instead of asking adopters to learn a new tool-definition standard, and because validating the usage-interface model inside S-ORA first is cheaper than committing to a separate project before the model is proven. The usage-interface spec is kept as a module with no dependency on the decision-cycle/activity internals specifically so it *could* be extracted later — see Consequences.

### Positive Consequences

* Immediate access to existing tool ecosystems (MCP, WoT, OpenAPI) with no new authoring standard to design or promote
* Avoids the overhead of maintaining a second project/spec before the core model is validated

### Negative Consequences

* Most existing ecosystems expose only operations, so adapters must approximate observable properties and signals (e.g., via polling) where no richer model is available
* Tool richness is capped by what each source protocol actually exposes

## Links

* Depends on [ADR-0001](0001-python-asyncio-runtime.md)
* Informs [ADR-0004](0004-tool-usage-interface.md), [ADR-0005](0005-workspace-grouping.md)
