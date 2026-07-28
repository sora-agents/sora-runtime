# Python 3.12+ (asyncio) as the core runtime

* Status: proposed
* Date: 2026-07-05

## Context and Problem Statement

S-ORA needs a core execution language and an asynchronous I/O concurrency model — cooperative, non-blocking concurrency for many activities waiting on tool/model round-trips, not CPU-bound parallelism. Key requirements: async at all levels, a reactive target of never blocking more than 10ms, support for many concurrent activities, and close proximity to the LLM/agent tooling ecosystem (MCP, A2A, provider SDKs) since S-ORA is being built through rapid, README-driven iteration.

## Decision Drivers

* Proximity to the LLM/agent tooling ecosystem (MCP, A2A, provider SDKs)
* Developer base
* Async I/O concurrency without a blocking scheduler
* Iteration speed while the design is still actively changing

## Considered Options

* Python (asyncio)
* Elixir/Erlang (BEAM)
* Rust (tokio)
* TypeScript/Node

## Decision Outcome

Chosen option: "Python (asyncio)", because the LLM/tool ecosystem is overwhelmingly Python-first and this is a research/prototyping-heavy project where velocity and ecosystem leverage matter more than the theoretical elegance of an actor-model runtime.

### Positive Consequences

* Fastest access to the MCP/A2A/provider-SDK ecosystem, minimizing glue code
* Large contributor pool
* Fast iteration fits the README-driven development approach

### Negative Consequences

* The GIL limits true parallelism for CPU-bound work; CPU-heavy logic must be offloaded (thread/process pool) to protect the reactiveness target
* Weaker raw throughput/latency guarantees than Rust or BEAM

## Pros and Cons of the Options

### Python (asyncio)

* Good, because it has the most mature LLM/agent tooling ecosystem
* Good, because of the large developer/hiring pool
* Bad, because the GIL blocks true parallelism for CPU-bound work in the same process

### Elixir/Erlang (BEAM)

* Good, because lightweight actor-style processes map almost 1:1 onto activities
* Good, because the preemptive scheduler gives a real non-blocking guarantee, not just "don't write blocking code"
* Good, because mailboxes and supervision trees already solve inter-agent messaging and failure handling
* Bad, because the developer pool is tiny and the LLM/agent ecosystem is thin — most AI-specific work would end up shelling out to Python anyway, undermining the point of switching

### Rust (tokio)

* Good, because there is no GIL — genuine parallelism and strong throughput/latency guarantees
* Good, because its type system fits a Protocol/trait-heavy design well
* Bad, because borrow-checker friction and compile times cut against fast iteration while the design is still unstable
* Bad, because the LLM/agent tooling ecosystem is less mature than Python's

### TypeScript/Node

* Good, because it has a similarly non-blocking, single-threaded async model to Python's
* Bad, because its LLM/agent tooling ecosystem is less mature than Python's; not seriously pursued for the core runtime

## Links

* Enables [ADR-0008](0008-protocol-based-extensibility.md) (`typing.Protocol` structural typing is a Python-specific mechanism)
