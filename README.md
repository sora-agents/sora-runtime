# S-ORA Agent Runtime

A runtime for practical agents in dynamic and asynchronous environments.

> **Status:** this project is currently in README-driven design — this file and [EXAMPLES.md](EXAMPLES.md) are the spec, and no runtime code exists yet beyond a packaging placeholder. See [ROADMAP.md](ROADMAP.md) for implementation status and [docs/adrs/](docs/adrs/) for why specific decisions were made.

Key features of a S-ORA agent:
- asynchronous at all levels: uses tools and communicates asynchronously
- concurrent: prioritizes and handles multiple activities at the same time
- reactive: targets never blocking more than 10ms, backed by a hard interrupt for high-priority events (see Decision Cycle)

Key features of the S-ORA runtime:
- lightweight: deals with the decision cycle, lets you do the rest
- efficient: blazing fast, maximizes throughput
- flexible: highly customizable, choose your own trade-offs

## Main concepts

S-ORA is aligned with the [CoALA conceptual framework](https://arxiv.org/abs/2309.02427) for cognitive language agents, and draws further insight from classical agent architectures, specifically the Belief-Desire-Intention (BDI) model and practical implementations such as [Jason](https://github.com/jason-lang/jason).

Main concepts: [activities](#activities), [tool model and use](#tool-model-and-use), [tool manuals](#tool-manuals), [memory modules](#memory), the [action space](#action-space), and the [S-ORA decision cycle](#the-s-ora-decision-cycle).

### Activities

An activity is the central unit of work for a S-ORA agent: it is a means to achieve a goal and has a context that represents a filtered view of the environment relevant to the activity. An agent can pursue multiple activities concurrently, but only one activity is executed in each decision cycle. It can also drop an activity if it is no
longer desirable or achievable, or it can suspend the activity while waiting for external events and conditions.

An activity can be in one of four states:

- running: the activity has an invoked operation in flight — invoked but not yet resolved; the agent won't reselect it until the operation resolves, though other activities may still be picked and progressed meanwhile;
- blocked: the agent is waiting for external events (e.g., signals from tools) to proceed with the activity;
- ready: the agent can pick and pursue the activity;
- terminated: the activity was completed or dropped.

An activity is eligible for selection only when ready; running and blocked activities are skipped until something transitions them back. Invoking an operation always, implicitly, moves an activity to running until that operation's own result comes back — this is unconditional, independent of anything the tool's manual says. A manual can additionally require blocking on a specific signal before the next step, layered on top of (not instead of) that implicit wait — the two are orthogonal: a signal may have nothing to do with the operation just invoked, or depend on other conditions entirely. Because resolving a running activity is an unambiguous, one-to-one match between an invoked operation and its result, the runtime does this automatically, with no strategy code involved; resolving a blocked activity is not automatic, since it requires matching a signal against whatever the manual says to wait for — a genuine judgment call left to Situate/Reflect.

### Tool Model and Use

S-ORA is inspired by the Agents and Artifacts (A&A) meta-model, which has its roots in activity theory: the agents' activities are mediated via tools. A tool is a domain object with its own control flow and internal state, with which agents can interact through a usage interface. Tools exist and evolve independently of any given agent and can be shared by multiple agents.

A tool's _usage interface_ is defined by:

- _observable properties_, which expose a persistent observable state; if an agent is observing the tool, the state is reflected in the agent's working memory
- _signals_, which represent transient events that occur within the tool and carry information that may be relevant to agents
- _operations_, which represent external actions provided by a tool

The usage interface is inherently asynchronous: when an agent invokes a tool operation, the agent's decision cycle does not block until the operation completes.

This distinction — an agent's action versus a tool's operation — mirrors the action/operation split in agent meta-models built on Agents & Artifacts, such as JaCaMo (Jason combined with the CArtAgO artifact-based environment). Concretely, invoking an operation produces two acknowledgments, not one: an immediate `ActionAck` confirming the action itself was dispatched — the same generic outcome every external action returns — and a separate `OperationAck` carrying the tool's own eventual result, made available later on the activity itself (see Activities) once the operation actually completes.

S-ORA does not define its own tool-authoring framework. Tools are expected to be defined elsewhere (e.g., via MCP, OpenAPI, or plain function signatures) and adapted into this usage interface; since most existing ecosystems expose only operations, adapters may need to approximate observable properties and signals (e.g., via polling) where no richer model is available. Adapters should only import primitives that are model-controlled — e.g., MCP Tools, not MCP Resources, which are application-controlled and belong outside the agent's own focus/observe reasoning. Resource subscriptions (e.g., MCP's `resources/subscribe`) are one valid mechanism for the approximation above, on the same footing as polling — but only when the adapter author documents the resulting event as a specific Signal in the tool's manual. MCP Resources carry no structural guarantee, at the protocol level, of corresponding to any coherent tool's actual state or events — that guarantee comes from the adapter author's own curation, not from the mechanism used to implement it. A raw, undocumented pass-through of whatever a resource happens to contain is excluded, whether read once or subscribed to.

Tools that share a connection or session — e.g., multiple operations exposed by one MCP server — are grouped into a workspace: a shared lifecycle boundary whose tools remain individually focusable, but whose underlying connection is established and torn down once, not per tool. A workspace's adapter fixes the tool-use protocol for everything inside it (e.g., all-MCP, all-WoT), but individual tools may still have their own connection address distinct from the workspace's — e.g., a hypermedia workspace for a lab could group virtual tools hosted on the workspace's own server alongside physical devices reachable at their own addresses in the same room.

Joining and leaving a workspace are deliberate, agent-driven actions (_join_/_leave_), not the result of an eager, upfront scan of every configured target. Today, join targets are limited to workspaces declared in the agent's own configuration; open, dynamic discovery of previously-unknown workspaces (e.g., for open environments where not every tool is known in advance) is foreseen but deliberately deferred.

We break down the process of using a tool into five phases:

- Discovery: the agent discovers the tool at run time — for example, through MCP or another tool calling protocol that supports tool discovery
- Learning: the agent retrieves the tool's manual and loads it into its context; thus, the agent learns how to use the tool by reading its manual
- Focus: the agent decides whether to subscribe to the tool's observable properties and signals to perceive relevant state changes and domain events
- Operation: the agent invokes operations that return an immediate acknowledgment (but not necessarily the final result or outcome)
- Suspension and Resumption: the agent may decide to suspend the operation of a tool if the manual specifies waiting for a signal or observable property update before resuming

### Tool Manuals

Tools can be described by manuals. Any manual format can be used. S-ORA currently uses Markdown, though more structured formats (e.g., XML) may prove better suited for parsing and validation. Regardless of format, a manual is structured into six parts:

1. Tool Metadata: includes general metadata about the tool, such as category information (e.g., "Critical Infrastructure / Fluid Dynamics"), to facilitate dynamic loading into the agent's context window;
2. Functional Description: a short natural language description of the tool as a domain object and its intended purpose;
3. Observable Properties: definitions of observable properties that may populate the agent's working memory, such as the current state of an air conditioner (AC);
4. Signals: definitions of domain events that may be emitted by the tool, such as the AC reaching a target temperature;
5. Operations: definitions of commands to interact with the tool, including the commands' intended purposes, preconditions, and effects;
6. Usage Protocols & Safety: operating instructions, including safety constraints (if any) or conditions under which an activity must be suspended (e.g., to wait for specific signals).

### Memory

The [CoALA framework](https://arxiv.org/abs/2309.02427) distinguishes between short and long-term memory.

Short-term memory, or **working memory**, maintains the agent's ongoing activities, perceptual input, and other contextual knowledge relevant to the current decision cycle. It is transient and optimized for speed (in-process).

Long-term memory modules are optional, persistent, and pluggable (e.g., a file-backed implementation to start, with database or vector-store backends as drop-in alternatives). They have their well-defined place in the S-ORA decision cycle:

- semantic memory: captures the agent's long-term knowledge about the world and itself, including tool manuals and discovered tool/workspace records — new kinds of durable "world knowledge" belong here, not in a new memory module
- procedural memory: captures the procedural knowledge the agent can query to derive or revise a plan for its current activity; this includes implicit knowledge encoded in LLM weights, and explicit knowledge captured as skills or plans — a plan is a multi-step, goal-indexed artifact, deliberately reusable across activities with similar goals, not something regenerated every cycle
- episodic memory: stores relevant experiences, such as successful activity completions, which may be retrieved for guidance in future activities

### Action space

Two types of actions: **internal actions**, for interacting with memory modules; and **external actions**, for interacting with the external world. The action space is extensible — agents and downstream frameworks can register additional internal or external actions beyond the predefined set below.

Predefined internal actions:

- **semantic memory**: _retrieve_ and _store_ tool manuals
- **working memory**: _load_ and _unload_ tool manuals from semantic memory; _filter_ perceptual input relevant to the current activity; _create_ a new activity from an unhandled message; _suspend_ and _resume_ an activity
- **procedural memory**: _retrieve_ a plan of action for the current activity, _infer_ one if a suitable one is not already known, or _store_ one that was actually followed to a successful completion
- **episodic memory**: _learn_ from experience by saving a summary of an activity completion, or _consult_ previous
experiences

Predefined external actions:

- _invoke_ a tool operation
- _join_ a configured workspace (connects and registers its tools) and _leave_ one (closes the connection)
- _retrieve_ manuals from external repositories
- _focus_ on and _unfocus_ from tools to perceive observable properties and signals
- _send_ messages to other agents, via a pluggable protocol (e.g., A2A, plain HTTP)

### The S-ORA Decision Cycle

The S-ORA decision cycle manages concurrent activities by selecting one activity to progress and executing at most one external action per cycle.

Observe => Reflect (optional) => Situate => Reason => Act

The decision cycle follows 5 steps:

- Observe: the agent receives perceptual input and messages asynchronously, which are reflected in the agent's working memory
- Reflect: for each activity, decides whether it has completed successfully or failed — and if so, executes an internal action to summarize and store the experience in episodic memory; "optional" means this decision itself is cheap by default and made fresh every cycle, not that the cycle is externally told when to check; the judgment is synchronous — it must land before Situate selects, so a just-completed activity is never re-selected the same cycle — while summarizing and storing run asynchronously and never block the cycle; several activities may terminate in the same cycle
- Situate: the agent selects an activity and adjusts its working memory for that activity — for example, by selecting tools, loading required manuals, unloading obsolete ones, and filtering the perceptual input; if an unhandled message in working memory doesn't correspond to any existing activity, Situate creates one via the internal _create_activity_ action before selecting
- Reason: the agent retrieves or infers a plan for the current activity — a multi-step, reusable artifact, not regenerated every cycle — and selects the next step to advance it; if the activity already has a valid plan, this is as cheap as reading its next step, no replanning involved; the Situate phase may suggest prerequisite external actions for situated reasoning, such as to retrieve tool manuals from an external repository, focus on or unfocus from tools; these prerequisite actions should take priority unless a more urgent action is needed — for example, to respond to a critical signal; if no prerequisite or urgent actions are required, the agent selects the next external action that advances the plan, which is either to send a message to another agent or invoke a tool operation
- Act: binds the step to a concrete invocation and executes the external action; for external actions that invoke tool operations, if the tool's manual specifies waiting for a signal or an observable property change before completion, the agent invokes the suspend internal action to suspend the current activity; the activity can resume once the expected external event is received

The five phases are a ceiling, not a quota: every cycle runs the pipeline, but a given cycle may conclude with one external action, with internal work only (e.g., storing experiences), or with nothing to do — at most one external action per cycle, never a mandatory one.

How many model calls a cycle costs is a configuration choice, not a property of the runtime. Observe and Reflect are deterministic by default: Observe mechanically ingests percepts and messages (an LLM-backed Observe is possible where perception itself needs interpretation — e.g., describing a camera snapshot — but that is an addition, not a fusion entry point), and Reflect's completion judgment may be deterministic or model-backed, with summarizing and storing dispatched asynchronously so they never block the cycle. Situate → Reason → Act form the decision chain proper — select an activity, advance its plan, bind a concrete invocation — and are the natural unit to fuse into a single model call, made after this cycle's percepts and messages are already in working memory. In the common case — a valid cached plan, mechanical defaults — a cycle costs zero model calls. A hard interrupt can preempt the current phase for high-priority signals, independent of where the cycle is mid-flight — this is what backs the 10ms reactiveness target, deliberately not a hard per-phase timeout, since an in-flight model call can't be safely cut off mid-generation.

Every phase has a pluggable strategy. A strategy may short-circuit later phases by producing their answer directly — e.g., Situate deciding the step and the concrete invocation in the same call that selects the activity — so that a single underlying computation can serve multiple phases. The shared decision value lives only for the duration of one cycle.

## Technology Stack & Requirements

- **Runtime — Python 3.12+ (asyncio)**: the decision cycle, activities, memory modules, and action
  registry live here. Chosen for async I/O concurrency without a blocking scheduler, and because the
  LLM/tool ecosystem (MCP, A2A, provider SDKs) is Python-first.
- **CLI**: the runtime ships a minimal terminal interface — output is streamed as the decision cycle
  runs, and the user can type input at any point, which is queued as a `Message` (sender `"user"`) for
  the next Observe phase — terminal input is user communication, not environment stimuli, so it's never
  a `Percept`. Similar in spirit to existing coding-agent CLIs: a persistent terminal session, not a
  one-shot command. Richer UIs are out of scope for the runtime and belong to whatever agent consumes it.
- **LLM access**: no hard dependency on a single provider SDK; the default `ReasonStrategy` targets any
  async HTTP chat-completions endpoint. Provider SDKs are optional extras.
- **Protocol adapters** (MCP, A2A, OpenAPI) ship as optional extras (e.g. `pip install sora-runtime[mcp]`)
  so the core package stays dependency-light.
- **Manual parsing**: pluggable `ManualParser` per format — Markdown by default, XML as an alternative.
- **Memory backends**: pluggable `MemoryBackend` — file-based by default, database/vector-store as
  drop-in alternatives.
- **Tooling**: `pytest` + `mypy` for testing and type-checking; `uv` for packaging.
- **Zero manual wiring for the common case**: implementing an agent means writing `agent.yaml` and,
  typically, one `ReasonStrategy` — never constructing `Agent`/`DecisionCycle`/memory modules by hand.
  All wiring is centralized in `sora/bootstrap.py` (see API Sketch).

## Running S-ORA

    $ uvx --from sora-runtime sora init my-agent --template minimal
    Created my-agent/
      pyproject.toml
      agent.yaml
      manuals/clock.md

    $ cd my-agent
    $ uv sync --extra mcp
    $ uv run sora run
    > what time is it?
    [invoking clock.get_time...]
    It's 14:32.
    >

`sora run` starts a persistent terminal session: it drives the decision cycle continuously, streams
external actions and messages as they happen, and reads terminal input as a `Message` (sender `"user"`)
for the next Observe phase — goals can be typed in at any point, not just at startup; Situate turns an
unhandled one into a new activity via _create_activity_. Use `--verbose` to print each decision-cycle
phase instead of just the conversational output:

    [cycle 1] Observe  - message from user: "what time is it?"
    [cycle 1] Situate  - created activity=ask-time from message; loaded manual: clock
    [cycle 1] Reason   - plan: invoke clock.get_time
    [cycle 1] Act      - invoked clock.get_time -> ack
    [cycle 2] Observe  - perceived signal: clock.time_reported
    [cycle 2] Reflect  - activity ask-time completed; stored to episodic memory

`uv sync` installs pinned dependencies into a project-local `.venv` per `uv.lock` — commit the lockfile
so runs are reproducible. `uv run` executes inside that environment without manual activation.

`agent.yaml` wires the pluggable pieces:

    agent:
      name: my-agent
      strategies:
        reason: sora.reason.default   # observe/reflect/situate/act default to sora's built-in mechanical strategies
      memory:
        working: in_process
        semantic: file://./.sora/memory/semantic
        procedural: file://./.sora/memory/procedural
        episodic: file://./.sora/memory/episodic
      transport: http://localhost:8765
      workspaces:
        - origin: {adapter: mcp, address: "mcp://localhost:6000"}   # clock tool, imported via the MCP adapter

Workspaces declared here are joined automatically at startup, before the first cycle runs — which is
why the `[cycle 1]` trace above already has the clock manual loaded, with no explicit `_join_` shown.

## API Sketch

```python
    # sora/types.py — primitives referenced throughout; kept minimal on purpose
    @dataclass(frozen=True)
    class ObservableProperty:
        name: str
        value: Any

    @dataclass(frozen=True)
    class Signal:
        name: str
        payload: dict

    @dataclass(frozen=True)
    class ActionAck:          # returned by ExternalAction.execute() — dispatch, not outcome (see EXAMPLES.md)
        ok: bool
        result: Any = None

    @dataclass(frozen=True)
    class OperationAck:       # returned by Tool.invoke() — the tool's own ack, arrives async via result_sink
        ok: bool
        result: Any = None

    @dataclass(frozen=True)
    class Step:
        next_action: str      # e.g. "invoke", "send", "focus", "wait"
        params: dict

    @dataclass(frozen=True)
    class Plan:                # multi-step, goal-indexed, reusable — the thing ProceduralMemory stores
        id: str                  # stable identity for storage/reuse
        goal: str                  # matched against future activities' goals — the retrieval key
        steps: list[Step]

    @dataclass(frozen=True)
    class OperationInvocation:  # was Invocation — the concrete, schema-bound call, distinct from a Step's more abstract decision
        tool_id: str
        operation_name: str     # correlates to OperationSpecification.name, same way tool_id correlates to Tool.id
        params: dict            # bound, ready to pass to Tool.invoke() — this is the tool-hallucination-prone step

    @dataclass(frozen=True)
    class PendingOperation:   # tracks one in-flight invoke — lives on Activity, not on WorkingMemory or Percept
        id: str                 # correlates to what InvokeAction pushed into result_sink
        invocation: OperationInvocation
        invoked_at: float

    # sora/environment.py — usage interface + adapters (S-ORA does not author tools, only consumes them)
    class Tool(Protocol):
        id: str
        manual: Manual
        address: str | None   # overrides the workspace's address when this tool has its own endpoint
        async def invoke(self, operation_name: str, **params) -> OperationAck: ...
        async def focus(self, sink: SignalSink) -> None: ...
        async def unfocus(self) -> None: ...
        def observe(self) -> list[ObservableProperty]: ...
    
    @dataclass(frozen=True)
    class WorkspaceOrigin:
        """The part of a WorkspaceRecord only the adapter can know: how to (re)connect."""
        adapter: str    # e.g. "mcp", "wot" — matches WorkspaceAdapter.name
        address: str      # e.g. an MCP server URI, or a WoT directory's base href

    class Workspace(Protocol):
        """A shared connection/lifecycle and tool-use-protocol boundary: e.g. one MCP session, or one
        WoT-described environment, however many tools it exposes. Tools within a workspace stay
        individually focusable, and may have their own address; the workspace's own connection —
        however many of its tools actually use it — is (re)established once."""
        id: str                              # matches WorkspaceRecord.id / ToolRecord.workspace_id
        origin: WorkspaceOrigin
        def tools(self) -> list[Tool]: ...
        async def close(self) -> None: ...   # contained tools go stale together

    class WorkspaceAdapter(Protocol):     # was ToolAdapter — it always operated at workspace granularity
        """Imports externally-defined tools (MCP, OpenAPI, WoT, ...) into the S-ORA usage interface.
        The tool-use protocol is fixed once per workspace (e.g. all-MCP, all-WoT); per-tool addressing
        within that protocol (see Tool.address) is a separate, orthogonal concern."""
        name: str    # e.g. "mcp" — matches WorkspaceOrigin.adapter
        async def discover(self) -> list[Workspace]:
            """Enumerates workspaces this adapter can reach. Today, each configured adapter instance is
            scoped to exactly one workspace (config-driven join — see Tool Model and Use); the same
            method is what open, dynamic discovery would call too, once that's in scope."""
        async def connect(self, workspace_record: WorkspaceRecord, tool_records: list[ToolRecord],
                           manuals: dict[str, Manual]) -> Workspace:
            """Re-establishes a workspace from its known records — one connection, all its tools rebuilt,
            no re-fetching manuals. Per tool_record: uses tool_record.address if set, else falls back
            to workspace_record.origin.address."""

    class EnvironmentRegistry:        # was ToolRegistry — now tracks workspaces, not just flattened tools
        """Live, in-process handles for workspaces (and their tools) the agent currently has a connection
        to. Populated by join()/restore() — never persisted directly (see WorkspaceRecord/ToolRecord)."""
        def __init__(self, adapters: dict[WorkspaceOrigin, WorkspaceAdapter] | None = None):
            """Keyed by the full origin (adapter + address), not just adapter name — an agent can join
            multiple workspaces that share a protocol (e.g. two separate MCP servers) without ambiguity."""
        def get(self, tool_id: str) -> Tool: ...
        def get_workspace(self, workspace_id: str) -> Workspace: ...
        def all_tools(self) -> list[Tool]: ...
        async def join(self, origin: WorkspaceOrigin) -> Workspace:
            """Predefined external action _join_: looks up the adapter registered for this exact origin,
            calls its discover() (config-scoped to just this target today), registers the workspace."""
        async def leave(self, workspace_id: str) -> None:
            """Predefined external action _leave_: closes the workspace's connection, deregisters it
            and all its tools."""
        async def restore(self, workspace_records: list[WorkspaceRecord], tool_records: list[ToolRecord],
                           semantic: SemanticMemory) -> list[Workspace]:
            """Reconnects to already-known workspaces via adapter.connect() — one call per workspace,
            looking up each one's adapter by workspace_record.origin, resolving each tool's manual from
            SemanticMemory first. Skips discovery entirely."""

    # sora/perception.py
    @dataclass(frozen=True)
    class Percept:
        source: str            # tool id
        kind: str               # "property" | "signal" — genuine environment stimuli only; an invoked
        payload: Any             # operation's own result is not a Percept (see Activity.pending_operation/
        observed_at: float        # last_operation) and neither are agent messages (see WorkingMemory.messages)

    @dataclass(frozen=True)
    class Message:
        sender: str
        content: dict
        received_at: float

    class SignalSink(Protocol):
        """Narrow, write-only interface: tools push here, they never see WorkingMemory or DecisionCycle."""
        def push(self, source: str, signal: Signal) -> None: ...

    class NotificationQueueSink(Generic[T]):     # was QueueSink — too generic a name to keep
        """Generic FIFO sink: producers push, _observe() drains once per cycle. Concrete backing for
        SignalSink (tool-facing) and for the runtime-internal channel that carries invoke() results —
        both are, structurally, queues of asynchronous notifications awaiting delivery as percepts."""
        def __init__(self) -> None:
            self._queue: asyncio.Queue[tuple[str, T]] = asyncio.Queue()
        def push(self, source: str, item: T) -> None: ...
        async def drain(self) -> AsyncIterator[tuple[str, T]]: ...

    # sora/manual.py
    @dataclass(frozen=True)
    class OperationSpecification:   # was Operation — renamed for symmetry with the two specs below
        name: str
        description: str
        parameters: dict     # JSON-Schema-shaped

    @dataclass(frozen=True)
    class ObservablePropertySpecification:
        name: str
        description: str
        schema: dict          # JSON-Schema-shaped, matching e.g. a WoT property affordance

    @dataclass(frozen=True)
    class SignalSpecification:
        name: str
        description: str
        schema: dict          # JSON-Schema-shaped, matching e.g. a WoT event affordance

    @dataclass(frozen=True)
    class Manual:
        id: str            # type identifier — NOT a tool instance id; shared across instances
        metadata: dict; description: str
        observable_properties: list[ObservablePropertySpecification]
        signals: list[SignalSpecification]
        operations: list[OperationSpecification]
        usage_protocols: str

    class ManualParser(Protocol):     # Markdown by default, XML pluggable
        def parse(self, raw: str) -> Manual: ...

    @dataclass(frozen=True)
    class WorkspaceRecord:
        """A WorkspaceOrigin that's actually been connected to, plus the identity/bookkeeping only
        assigned once that connection exists. Not duplicated onto every ToolRecord that references it;
        individual tools may still override the address (see ToolRecord.address)."""
        id: str            # matches Workspace.id once live
        origin: WorkspaceOrigin
        discovered_at: float
        last_seen_at: float

    @dataclass(frozen=True)
    class ToolRecord:
        """Durable record of a discovered tool instance — many records can share one manual_id,
        and every record from the same connection shares one workspace_id."""
        id: str            # instance id, matches Tool.id once live
        manual_id: str
        workspace_id: str   # references WorkspaceRecord.id
        address: str | None  # overrides WorkspaceRecord.origin.address; e.g. a physical device's own endpoint
        discovered_at: float
        last_seen_at: float

    # sora/activity.py
    class ActivityState(Enum):
        RUNNING = "running"; BLOCKED = "blocked"; READY = "ready"; TERMINATED = "terminated"

    @dataclass
    class Activity:
        id: str; goal: str; context: dict
        state: ActivityState = ActivityState.READY
        plan: Plan | None = None    # once set, Reason can just advance it instead of (re)planning
        step_index: int = 0
        pending_operation: PendingOperation | None = None  # set while RUNNING; runtime clears it on resolve
        last_operation: OperationAck | None = None          # most recently resolved result, for Reason to read
        # context is exclusively for strategy-author data — the runtime itself never writes into it,
        # which is what keeps pending_operation/last_operation as dedicated fields instead of context
        # keys with a naming convention: no shared namespace means no collision to avoid in the first place

    # sora/action.py — extensible action space
    class InternalAction(Protocol):
        name: str
        async def execute(self, cycle: DecisionCycle, **kwargs) -> Any:
            """No EnvironmentRegistry access — internal actions only ever touch memory."""

    class ExternalAction(Protocol):
        name: str
        async def execute(self, registry: EnvironmentRegistry, cycle: DecisionCycle, *,
                           activity_id: str, **kwargs) -> ActionAck:
            """Narrower than passing a whole Agent: tools (Agent-owned) + cycle (memory/transport/sinks),
            nothing else — see the tick() signature below for why. `activity_id` is always passed by
            tick()'s dispatch, absorbed harmlessly by actions that don't need it (all but _invoke_)."""

    class ActionRegistry:
        def register_internal(self, action: InternalAction) -> None: ...
        def register_external(self, action: ExternalAction) -> None: ...

    class InvokeAction:                # predefined external action: _invoke_
        name = "invoke"
        async def execute(self, registry: EnvironmentRegistry, cycle: DecisionCycle, *,
                           activity_id: str, tool_id: str, operation_name: str, **params) -> ActionAck:
            tool = tools.get(tool_id)
            invocation = OperationInvocation(tool_id=tool_id, operation_name=operation_name, params=params)
            op_id = new_id()
            activity = cycle.working.activities[activity_id]
            activity.pending_operation = PendingOperation(id=op_id, invocation=invocation, invoked_at=now())
            activity.state = ActivityState.RUNNING   # implicit, unconditional — see Activities
            asyncio.create_task(self._call(cycle, tool, operation_name, params, op_id))
            return ActionAck(ok=True)     # immediate — the round-trip runs off-cycle, cycle never blocks
        async def _call(self, cycle: DecisionCycle, tool: Tool, operation_name: str, params: dict, op_id: str) -> None:
            ack = await tool.invoke(operation_name, **params)
            cycle.result_sink.push(op_id, ack)   # keyed by op_id, not tool_id — see DefaultObserveStrategy

    class FocusAction:                # predefined external action: _focus_
        name = "focus"
        async def execute(self, registry: EnvironmentRegistry, cycle: DecisionCycle, *,
                           tool_id: str, **kwargs) -> ActionAck:
            tool = tools.get(tool_id)
            await tool.focus(cycle.signal_sink)
            cycle.working.focused_tools[tool_id] = tool
            return ActionAck(ok=True)

    class UnfocusAction:              # predefined external action: _unfocus_
        name = "unfocus"
        async def execute(self, registry: EnvironmentRegistry, cycle: DecisionCycle, *,
                           tool_id: str, **kwargs) -> ActionAck:
            tool = cycle.working.focused_tools.pop(tool_id, None)
            if tool is not None:
                await tool.unfocus()
            return ActionAck(ok=True)

    class JoinAction:                  # predefined external action: _join_ — implies discover/connect
        name = "join"
        async def execute(self, registry: EnvironmentRegistry, cycle: DecisionCycle, *,
                           origin: WorkspaceOrigin, **kwargs) -> ActionAck:
            workspace = await tools.join(origin)
            await cycle.semantic.store_workspace_record(WorkspaceRecord(
                id=workspace.id, origin=origin,
                discovered_at=now(), last_seen_at=now(),
            ))
            for tool in workspace.tools():
                await cycle.semantic.store_manual(tool.manual)
                await cycle.semantic.store_tool_record(ToolRecord(
                    id=tool.id, manual_id=tool.manual.id, workspace_id=workspace.id,
                    address=tool.address,   # None unless this tool overrides the workspace's address
                    discovered_at=now(), last_seen_at=now(),
                ))
            return ActionAck(ok=True, result=[tool.id for tool in workspace.tools()])

    class LeaveAction:                 # predefined external action: _leave_ — implies close
        name = "leave"
        async def execute(self, registry: EnvironmentRegistry, cycle: DecisionCycle, *,
                           workspace_id: str, **kwargs) -> ActionAck:
            await tools.leave(workspace_id)
            return ActionAck(ok=True)

    class SendAction:                  # predefined external action: _send_
        name = "send"
        async def execute(self, registry: EnvironmentRegistry, cycle: DecisionCycle, *,
                           to: str, content: dict, **kwargs) -> ActionAck:
            await cycle.communication.send(to, content)   # tools unused here — every ExternalAction still
            return ActionAck(ok=True)                  # gets the same uniform (tools, cycle) signature

    # sora/memory.py
    class MemoryBackend(Protocol):    # pluggable: file, DB, vector store
        async def get(self, key: str) -> Any: ...
        async def put(self, key: str, value: Any) -> None: ...
        async def query(self, **filters) -> list[Any]: ...

    class WorkingMemory:              # transient, in-process, fast
        activities: dict[str, Activity]
        perceptions: list[Percept]    # stimuli from the environment: properties and signals only
        messages: list[Message]        # inbound agent-to-agent communication — kept distinct
        focused_tools: dict[str, Tool]

    class SemanticMemory:              # knowledge about the world: tool types, workspaces, instances
        def __init__(self, backend: MemoryBackend): ...
        async def retrieve_manual(self, manual_id: str) -> Manual | None: ...
        async def store_manual(self, manual: Manual) -> None: ...
        async def retrieve_workspace_record(self, workspace_id: str) -> WorkspaceRecord | None: ...
        async def store_workspace_record(self, record: WorkspaceRecord) -> None: ...
        async def list_workspace_records(self) -> list[WorkspaceRecord]: ...
        async def retrieve_tool_record(self, tool_id: str) -> ToolRecord | None: ...
        async def store_tool_record(self, record: ToolRecord) -> None: ...
        async def list_tool_records(self) -> list[ToolRecord]: ...   # reconstitute known instances at startup

    class ProceduralMemory:
        def __init__(self, backend: MemoryBackend): ...
        async def retrieve(self, activity: Activity) -> Plan | None:
            """Looks up a cached Plan matching this activity's goal — e.g. exact match or embedding
            similarity, backend-dependent. The cheap path: skips infer() entirely when it hits."""
        async def infer(self, activity: Activity) -> Plan:
            """Produces a new multi-step Plan when no cached one fits — the expensive path, potentially
            an LLM call producing a whole sequence of Steps at once, not just the next one."""
        async def store(self, plan: Plan) -> None:
            """Persists a Plan that was actually followed to completion, so future retrieve() calls for
            similar goals can reuse it. Called by ReflectStrategy on success only — a failed plan isn't
            something future activities should retrieve by default."""

    class EpisodicMemory:
        def __init__(self, backend: MemoryBackend): ...
        async def learn(self, activity: Activity, summary: str) -> None: ...
        async def consult(self, activity: Activity) -> list[Any]: ...

    # sora/strategies.py — one pluggable strategy per phase, threaded through a shared TickResult
    @dataclass(frozen=True)
    class TickResult:
        """The decision surface for one cycle. Every phase strategy receives and returns one of these.
        Whatever's still None, DecisionCycle fills in by calling the next phase's own strategy — so a
        fully-decomposed configuration produces one field at a time, and a fully-fused one can fill in
        everything from a single Observe (or Reflect) call. Lives only for the duration of one tick()
        call — nothing persists across cycles, so there's no cache to key or invalidate."""
        activity: Activity | None = None
        step: Step | None = None      # this cycle's concrete decision — not the whole (possibly multi-step) Plan
        invocation: OperationInvocation | None = None

    class ObserveStrategy(Protocol):
        async def observe(self, cycle: DecisionCycle) -> TickResult:
            """Mutates cycle.working (perceptions, messages) as a side effect — same as the default
            below. Default: mechanical, no model call, returns an empty TickResult(). An LLM-backed
            Observe is for interpreting raw perception itself (e.g., describing a camera snapshot),
            not for deciding the cycle — decision-chain fusion starts at Situate, not here."""

    class ReflectStrategy(Protocol):
        async def reflect(self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle,
                           result: TickResult) -> TickResult:
            """Decides whether this activity just completed or failed — deterministic or model-backed,
            depending on the application — and if so, summarizes and stores to episodic memory. On
            success, also stores activity.plan via cycle.procedural.store() so future activities with
            a similar goal can reuse it; on failure, it isn't stored. The completion judgment is
            synchronous — it must land before Situate selects, so a just-completed activity is never
            re-selected the same cycle — while the summarize/store side effects are dispatched
            asynchronously and never block the cycle; several activities may terminate in the same
            cycle. Passes `result` through, optionally adding to it. Default: performs the completion
            check and the store-on-success, leaves TickResult's other fields untouched. `cycle` is
            what makes these memory calls possible at all — previously missing from this Protocol
            despite the calls it was already documented as making."""

    class SituateStrategy(Protocol):
        async def situate(self, activities: list[Activity], wm: WorkingMemory, cycle: DecisionCycle,
                           result: TickResult) -> TickResult:
            """Selects the next activity and adjusts wm for it. Only called if result.activity is still
            None. Also responsible for activity creation: if wm.messages has one that doesn't correspond
            to any existing activity, invokes the internal _create_activity_ action (via cycle) before
            selecting. Head of the decision chain (Situate -> Reason -> Act) and the intended entry
            point for fusing the remaining phases into one model call — it runs after this cycle's
            percepts and messages are already in working memory. May additionally fill in
            step/invocation, short-circuiting Reason/Act."""

    class ReasonStrategy(Protocol):   # pluggable; default targets 1 LLM call/cycle
        async def reason(self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle,
                          result: TickResult) -> TickResult:
            """Only called if result.step is still None. Typical implementation: if activity.plan is
            already set and still valid, just read activity.plan.steps[activity.step_index] and advance
            the index — no model call. Otherwise, retrieve a cached Plan via cycle.procedural.retrieve()
            or infer a new one (the expensive path), reset step_index to 0, and use its first Step.
            Deciding when a plan counts as invalidated is entirely up to the implementation. May
            additionally fill in invocation, short-circuiting Act — this is where the historical 'tool
            hallucination' risk lives if it does."""

    class ActStrategy(Protocol):
        async def bind(self, step: Step, manual: Manual | None, cycle: DecisionCycle,
                        result: TickResult) -> TickResult:
            """Only called if result.invocation is still None. Binds an abstract Step to a concrete,
            schema-conformant OperationInvocation. `cycle` is available for implementations that
            cache bindings (e.g. belief-state -> params) rather than re-deriving one every time."""

    @dataclass(frozen=True)
    class Strategies:          # bundles the five, so DecisionCycle.__init__ doesn't take five loose params
        observe: ObserveStrategy
        reflect: ReflectStrategy
        situate: SituateStrategy
        reason: ReasonStrategy
        act: ActStrategy

    class DefaultObserveStrategy:
        """The runtime's built-in default — purely mechanical, no LLM. This is the exact logic
        previously inlined in DecisionCycle._observe()."""
        async def observe(self, cycle: DecisionCycle) -> TickResult:
            for tool in cycle.working.focused_tools.values():
                for prop in tool.observe():
                    cycle.working.perceptions.append(Percept(tool.id, "property", prop, now()))
            async for source, signal in cycle.signal_sink.drain():
                cycle.working.perceptions.append(Percept(source, "signal", signal, now()))
            async for op_id, ack in cycle.result_sink.drain():
                # unambiguous 1:1 match — resolved automatically, never a Percept, no strategy involved
                activity = next((a for a in cycle.working.activities.values()
                                  if a.pending_operation and a.pending_operation.id == op_id), None)
                if activity is not None:
                    activity.last_operation = ack
                    activity.pending_operation = None
                    activity.state = ActivityState.READY
            async for message in cycle.communication.receive():
                cycle.working.messages.append(message)
            return TickResult()

    # sora/transport.py
    class MessageTransport(Protocol): # pluggable: A2A, HTTP, in-process
        async def send(self, to: str, content: dict) -> None: ...
        async def receive(self) -> AsyncIterator[Message]: ...

    # sora/cycle.py
    class DecisionCycle:
        def __init__(self, strategies: Strategies, communication: MessageTransport,
                     actions: ActionRegistry, working: WorkingMemory,
                     semantic: SemanticMemory, procedural: ProceduralMemory,
                     episodic: EpisodicMemory):
            # Both sinks live here rather than on WorkingMemory: they're the bridge from
            # asynchronous, off-cycle events into this engine's tick()/interrupt() — not settled
            # state. signal_sink specifically has to be co-located with interrupt() below, since
            # a pushed Signal can preempt the current phase; that control-flow role, not "where
            # it eventually lands as a percept," is why it isn't a WorkingMemory field.
            self.signal_sink: NotificationQueueSink[Signal] = NotificationQueueSink()        # tools push here via focus()
            self.result_sink: NotificationQueueSink[OperationAck] = NotificationQueueSink()  # InvokeAction pushes here — internal only
            ...
        async def tick(self, registry: EnvironmentRegistry) -> None:
            """One Observe -> Reflect -> Situate -> Reason -> Act pass, threading a TickResult through
            all five phases and calling each phase's own strategy only for whatever's still missing —
            so a fully-fused Observe (or Reflect) call can skip the rest of the cycle entirely. `tools`
            is the only thing this doesn't already hold (working/semantic/procedural/episodic/
            communication are shared with Agent, constructed once and passed to both — see
            sora/bootstrap.py)."""
            result = await self.strategies.observe.observe(self)
            for activity in self.working.activities.values():
                result = await self.strategies.reflect.reflect(activity, self.working, self, result)
            if result.activity is None:
                ready = [a for a in self.working.activities.values() if a.state is ActivityState.READY]
                result = await self.strategies.situate.situate(ready, self.working, self, result)
            if result.step is None:
                result = await self.strategies.reason.reason(result.activity, self.working, self, result)
            if result.invocation is None and result.step.next_action == "invoke":
                manual = self.working.focused_tools[result.step.params["tool_id"]].manual
                result = await self.strategies.act.bind(result.step, manual, self, result)
            # dispatch result.invocation (if set) or result.step.params to the matching registered
            # ExternalAction via self.actions, always passing activity_id=result.activity.id — elided,
            # same as the rest of Act's action-lookup today
            ...
        async def interrupt(self, signal: Signal) -> None:
            """Preempts the current phase for a high-priority event (10ms target)."""

    class Agent:
        """Owns the pieces that are conceptually the agent's own — tools, memory, transport — built
        from the same shared instances as DecisionCycle, so e.g. agent.registry.restore(records,
        agent.semantic) never needs to reach through agent.cycle."""
        def __init__(self, cycle: DecisionCycle, registry: EnvironmentRegistry,
                     working: WorkingMemory, semantic: SemanticMemory,
                     procedural: ProceduralMemory, episodic: EpisodicMemory,
                     communication: MessageTransport): ...
        async def run(self) -> None:
            """Loop: await self.cycle.tick(self.tools) — passes only what tick() doesn't already have."""
        async def stop(self) -> None: ...

    # sora/cli.py — the runtime's minimal terminal interface
    class TerminalSession:
        """Streams cycle output to stdout; queues stdin as Message(sender="user", ...) — not a Percept,
        since terminal input is user communication, not environment stimuli. No UI beyond this."""
        def __init__(self, agent: Agent, verbose: bool = False): ...
        async def run(self) -> None: ...

    # sora/bootstrap.py — internal; developers implement protocols, they don't call this directly
    def build_agent(config_path: str) -> Agent:
        """What `sora run` calls before handing off to TerminalSession. This is the one place all the
        wiring (which memory backend, which transport, which adapters, DecisionCycle <-> Agent sharing
        the same instances) actually happens — a developer implementing an agent never writes this."""
        config = load_yaml(config_path)
        working = WorkingMemory()
        semantic = SemanticMemory(backend_for(config.memory.semantic))
        procedural = ProceduralMemory(backend_for(config.memory.procedural))
        episodic = EpisodicMemory(backend_for(config.memory.episodic))
        communication = HttpTransport(self=config.transport.self, peers=config.transport.peers)
        strategies = Strategies(
            observe=import_object(config.strategies.get("observe", "sora.observe.default"))(),
            reflect=import_object(config.strategies.get("reflect", "sora.reflect.default"))(),
            situate=import_object(config.strategies.get("situate", "sora.situate.default"))(),
            reason=import_object(config.strategies["reason"])(),   # the one most agents actually override
            act=import_object(config.strategies.get("act", "sora.act.default"))(),
        )

        cycle = DecisionCycle(strategies=strategies, communication=communication, actions=default_action_registry(),
                               working=working, semantic=semantic, procedural=procedural, episodic=episodic)
        adapters = {WorkspaceOrigin(**w["origin"]): adapter_for(w["origin"]) for w in config.workspaces}
        tools = EnvironmentRegistry(adapters=adapters)
        return Agent(cycle=cycle, tools=tools, working=working, semantic=semantic,
                     procedural=procedural, episodic=episodic, communication=communication)
```
