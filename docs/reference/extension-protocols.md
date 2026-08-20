# Extension Protocols Reference

Every extension seam is a `Protocol`, not a base class to subclass (see `CLAUDE.md`'s "Protocol
over inheritance" habit) — implement the methods below with matching signatures and your type
satisfies the seam structurally, no inheritance required. This page collects the exact contracts
for the five kinds this design calls out as extension points: adapters, phase strategies, memory
backends, transports, and LLM clients. For the full type/API surface, including the ones not
covered here, see the [Python API Reference](python-api.md); for what each *concept* is for, see
[Concepts](../concepts/runtime-model.md).

## Adapters

`WorkspaceAdapter` imports externally-defined tools (MCP, WoT, ...) into S-ORA's usage interface —
see [ADR-0003](../architecture/adrs/0003-adapters-not-tool-authoring.md). An adapter is also
responsible for the `Tool`/`Workspace` instances it hands back.

```python
class WorkspaceAdapter(Protocol):
    name: str  # e.g. "mcp" — matches WorkspaceOrigin.adapter

    async def discover(self) -> list[Workspace]: ...

    async def connect(
        self,
        workspace_record: WorkspaceRecord,
        tool_records: list[ToolRecord],
        manuals: dict[str, Manual],
    ) -> Workspace: ...
```

`discover()` enumerates workspaces this adapter can reach (today, one configured adapter instance is
scoped to exactly one workspace). `connect()` re-establishes a workspace from its known records —
one connection, all its tools rebuilt, no re-fetching manuals — using each `tool_record.address`
when set, else falling back to `workspace_record.origin.address`. The adapter must ensure tools have
globally unique ids (ADR-0014).

```python
class Workspace(Protocol):
    id: str
    origin: WorkspaceOrigin

    def tools(self) -> list[Tool]: ...

    async def close(self) -> None: ...


class Tool(Protocol):
    id: str
    manual: Manual
    address: str | None  # overrides the workspace's address when this tool has its own endpoint

    async def invoke(self, operation_name: str, **params: Any) -> OperationAck: ...

    async def focus(self, sink: SignalSink) -> None: ...

    async def unfocus(self) -> None: ...

    def observe(self) -> list[ObservableProperty]: ...
```

A `Workspace` is the shared connection/lifecycle boundary (one MCP session, one WoT-described
environment); a `Tool` may still have its own `address` distinct from its workspace's
([ADR-0005](../architecture/adrs/0005-workspace-grouping.md)).

## Phase strategies

Every phase of the decision cycle is independently pluggable
([ADR-0010](../architecture/adrs/0010-pluggable-phase-strategies.md)), threaded through the shared
`TickResult`
([ADR-0011](../architecture/adrs/0011-phase-fusion-via-threaded-result.md)). Configured per-agent
via `agent.yaml`'s `strategies:` block — see [Configuration Reference](configuration.md).

```python
class ObserveStrategy(Protocol):
    async def observe(self, cycle: DecisionCycle) -> TickResult: ...


class ReflectStrategy(Protocol):
    async def reflect(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult: ...

    def failed(self, activity: Activity) -> bool: ...


class SituateStrategy(Protocol):
    async def situate(
        self,
        activities: list[Activity],
        wm: WorkingMemory,
        cycle: DecisionCycle,
        result: TickResult,
    ) -> TickResult: ...


class ReasonStrategy(Protocol):  # the one phase with no mechanical default
    async def reason(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult: ...


class ActStrategy(Protocol):
    async def bind(
        self, step: Step, manual: Manual | None, cycle: DecisionCycle, result: TickResult
    ) -> TickResult: ...
```

`DecisionCycle.tick()` only calls a phase's strategy if the relevant `TickResult` field
(`activity`/`step`/`invocation`) is still `None` — an earlier phase filling it short-circuits the
later one. `Situate` is the intended fusion entry point (it may fill `step`/`invocation` too);
`Observe`/`Reflect` are not fusion targets.

### Auxiliary seams

Not part of the five-phase `Strategies` bundle, but selected the same way (a name or dotted path in
`agent.yaml`'s `strategies:` block):

```python
class ActivitySelectionStrategy(Protocol):
    async def select(
        self, ready: list[Activity], wm: WorkingMemory, cycle: DecisionCycle
    ) -> Activity | None: ...


class InterruptPolicy(Protocol):
    def decide(self, source: str, signal: Signal, wm: WorkingMemory) -> InterruptRequest | None: ...


class InterruptHandler(Protocol):
    async def handle(
        self, request: InterruptRequest, wm: WorkingMemory, cycle: DecisionCycle
    ) -> bool: ...


class ReconsiderationPolicy(Protocol):
    def should_check(self, side_effecting: bool | None) -> bool: ...


class ChangeGate(Protocol):
    def signature(self, wm: WorkingMemory) -> object: ...
```

`ActivitySelectionStrategy` is `SituateStrategy`'s own scheduler sub-strategy (default: fair
round-robin). `InterruptPolicy.decide` is consulted **synchronously**, the instant a signal is
pushed to `signal_sink` — before the once-per-cycle Observe drain — since push is itself synchronous;
returning an `InterruptRequest` preempts the current phase (a hard interrupt), `None` lets the
signal flow cooperatively. `InterruptHandler.handle` runs the interrupt follow-up: mapping each
targeted activity onto `READY`/`BLOCKED`/`TERMINATED`
([ADR-0020](../architecture/adrs/0020-hard-interrupt-and-await-input.md)). `ReconsiderationPolicy`
and `ChangeGate` are the two independent knobs behind the pre-revalidation checkpoint
([ADR-0024](../architecture/adrs/0024-plan-reconsideration-context-adaptation.md)):
`ReconsiderationPolicy` decides *when* a checkpoint runs; `ChangeGate` decides *whether* the world
moved since the plan was baselined.

## Memory backends

```python
class MemoryBackend(Protocol):  # pluggable: file, DB, vector store
    async def get(self, key: str) -> Any: ...

    async def put(self, key: str, value: Any) -> None: ...

    async def query(self, **filters: Any) -> list[Any]: ...
```

`query()` returns every stored value matching all `filters`, ordered most-relevant-first with ties
broken deterministically — callers may treat `result[0]` as the single best match. A ranking
backend (a vector store) orders by relevance; a non-ranking one (the shipped `FileMemoryBackend`,
exact-match JSON-per-key) treats all matches as equally relevant and falls back to a stable key
order. The semantic/procedural/episodic memory modules convert their dataclasses to/from plain
JSON-serializable values before touching the backend — implement `MemoryBackend` against that
generic contract, never against a specific dataclass shape, and a database/vector-store backend
becomes a true drop-in.

## Transports

```python
class MessageTransport(Protocol):  # pluggable: A2A, HTTP, in-process
    async def send(self, to: str, content: dict[str, Any]) -> None: ...

    def receive(self) -> AsyncIterator[Message]: ...
```

`receive()` is a non-`async def` **returning** an `AsyncIterator` (not itself a coroutine), so
`async for m in transport.receive()` reads exactly what is queued *now* and stops — it must never
block the cycle waiting on a future `send()`/`submit()`. The runtime drains it once per Observe.
The shipped `InProcessTransport` is the single-agent default (an in-process inbox, no network);
agent-to-agent transport (A2A/HTTP, `agent.yaml`'s `transport.peers`) is the multi-agent case and is
not implemented yet.

## LLM clients

```python
class LLMClient(Protocol):
    async def complete(self, *, system: str, prompt: str) -> str: ...
```

Deliberately narrow and wire-format-neutral: a system instruction plus a prompt in, text out —
commits to no provider shape (not OpenAI `chat/completions`, not Anthropic `messages`), so
`ProceduralMemory.infer` stays independent of any one SDK. **Non-ownership contract**: an
`LLMClient` owns *only* the round-trip. Retries, streaming, credential refresh, prompt caching, and
interrupt handling belong to the cycle/agent, never to the client — that boundary is what lets a
second provider slot in without touching the decision cycle. Configured via `agent.yaml`'s `llm:`
block (see [Configuration Reference](configuration.md#agentllm)); every configured client is wrapped
in the shipped `MeteredLLMClient` automatically for timing/usage instrumentation.

---

!!! info "Hand-authored framing pending"
    These are the exact contracts. The invariants and trade-offs each seam expects an implementer to
    respect — e.g. what a custom `InterruptPolicy` must guard against to avoid a self-write loop, or
    when a domain `ChangeGate` earns its keep over the default — are partly captured in the ADRs
    linked above and partly still owed as dedicated extension guides (`docs/extensions/*.md`, not
    yet written); this page doesn't restate that framing.
