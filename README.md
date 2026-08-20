# S-ORA Agent Runtime

A runtime for practical agents in dynamic and asynchronous environments.

> **Status:** this project is under active development. The core runtime — the five-phase decision cycle, activities, memory modules, the action registry, and the terminal CLI — is implemented and covered by an extensive test suite (see [ROADMAP.md](ROADMAP.md) for the phase-by-phase build-out); advanced features (richer dynamic-environment handling, additional protocol adapters, multimodal perception, full benchmark coverage) are still in progress. [EXAMPLES.md](EXAMPLES.md) remains a worked spec scenario, [docs/reference/python-api.md](docs/reference/python-api.md) is the exact type/signature reference, and [docs/architecture/adrs/](docs/architecture/adrs/) records why specific decisions were made.

Key features of a S-ORA agent:
- asynchronous at all levels: uses tools and communicates asynchronously
- concurrent: prioritizes and handles multiple activities at the same time
- reactive: targets never blocking more than 10ms, backed by a hard interrupt for high-priority events (see [Decision Cycle](docs/concepts/decision-cycle.md))

Key features of the S-ORA runtime:
- lightweight: minimal runtime, focused on the decision cycle
- efficient: minimizes overhead during agent execution
- flexible: highly customizable, choose your own trade-offs

## Documentation

Full documentation: **[docs/index.md](docs/index.md)**.

- **New here?** [Quickstart](docs/getting-started/quickstart.md) · [Your First Agent](docs/getting-started/first-agent.md) · [Configuration](docs/getting-started/configuration.md)
- **Understand the design:** [Concepts](docs/concepts/runtime-model.md) — the CoALA/BDI/Jason lineage, activities, the tool model and use, tool manuals, memory modules, and the [decision cycle](docs/concepts/decision-cycle.md) that ties them together.
- **Extend it:** [Guides](docs/guides/cli-and-programmatic-runs.md) and [Extensions](docs/extensions/adapters.md).
- **Why it's built this way:** [Architecture Decision Records](docs/architecture/adrs/README.md).
- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md).

The exact Python API — every type and signature — is generated straight from source:
[docs/reference/python-api.md](docs/reference/python-api.md). It supersedes this file's former
hand-maintained API sketch and can't drift, since it's rebuilt from `src/sora/**/*.py` on every
`mkdocs build`. A verbatim snapshot of the old sketch (with its narrative inline comments) is kept
for reference at [docs/reference/api-sketch-notes.md](docs/reference/api-sketch-notes.md).
