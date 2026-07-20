# S-ORA Agent Runtime

A runtime for practical agents in dynamic and asynchronous environments.

> **Status:** this project is currently under development and follows README-driven design — this file and [EXAMPLES.md](EXAMPLES.md) are the spec. See [ROADMAP.md](ROADMAP.md) for implementation status and [docs/adrs/](docs/adrs/) for why specific decisions were made.

Key features of a S-ORA agent:
- asynchronous at all levels: uses tools and communicates asynchronously
- concurrent: prioritizes and handles multiple activities at the same time
- reactive: targets never blocking more than 10ms, backed by a hard interrupt for high-priority events (see Decision Cycle)

Key features of the S-ORA runtime:
- lightweight: minimal runtime, focused on the decision cycle
- efficient: minimizes overhead during agent execution
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

How finely a server's primitives map to tools is the adapter's call. A plain MCP adapter maps each MCP tool to one S-ORA tool with a single operation and no observable properties or signals (its resources being application-controlled, per the preceding paragraph); a _curating_ adapter can lift a richer abstraction on top — e.g., the ARE adapter groups a server's `<App>__<operation>` tools into one tool per app and surfaces that app's state resource as a curated observable/signal. The `<App>__` convention is that adapter's own curation, not canonical MCP.

A tool's `address` is a _locator_ and may be absent — e.g., tools multiplexed over one MCP stdio connection have none — whereas its `id` is the stable _handle_ the agent uses to focus and invoke it, and is **globally unique**: because a tool is a shared object, two agents focusing the same tool, or messaging about it, must name it identically. The per-protocol adapter guarantees this by deriving the id from the tool's global identity — its URI where the protocol provides one, or a value synthesized from the workspace's global origin/address otherwise — deterministically, so a later `restore()` reproduces the same id. A single registry can only enforce the ids it sees (it rejects a collision within its own joined set rather than letting one workspace's tool shadow another's); global uniqueness itself rests on the adapter. See [ADR-0014](docs/adrs/0014-tool-identity-globally-unique.md).

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

In the Markdown rendering, Observable Properties, Signals, and Operations are `-` bullet lists. An operation bullet may additionally carry optional labeled sub-bullets — `Preconditions:`, `Effects:`, and `Behavior:` (whether the operation completes synchronously or is long-running, and which signal, if any, indicates completion) — expressing the operation semantics part 5 calls for. For now these are folded into the operation's single `description` (see `OperationSpecification` in the API Sketch): fully available to a reasoning strategy as text, but deliberately not lifted into discrete model fields until a strategy actually consumes them — the labels are the seams where that structure would later attach.

Property, signal, and operation entries carry their data shapes as JSON Schema in the spec types' `schema`/`parameters` fields (see the API Sketch). A manual describes a tool *type* and stays protocol-agnostic: JSON Schema is data shape, not a protocol binding, so it is filled either by an adapter from a native description (an MCP tool schema, a WoT TD affordance schema) or, for a hand-authored manual, lifted from the light `(type, range)` hints above — with an optional inline JSON Schema where full fidelity is needed. The protocol binding — how to actually reach one instance — lives on the live `Tool`, never in the manual. See [ADR-0015](docs/adrs/0015-manuals-protocol-agnostic-adapter-boundary.md).

#### The clean Markdown format

`MarkdownManualParser` (the default `ManualParser`) parses this format; malformed input raises `ManualParseError`. The document is a flat sequence of `# `-level sections whose headings are the six parts above (`# Tool Metadata`, `# Functional Description`, `# Observable Properties`, `# Signals`, `# Operations`, `# Usage Protocols & Safety`). `# Tool Metadata` is `key: value` lines — `id:` is **required** (it becomes `Manual.id`; a manual with no `id` is rejected), every other key lands in `metadata`; the remaining sections are free prose, with the observable-property / signal / operation lists written as `-` bullets (or the literal `(none)` when empty).

The parser yields a `Manual` **envelope**: it fills `id`, `metadata`, `description` (from Functional Description), and the verbatim `raw_text`, and leaves the structured `observable_properties` / `signals` / `operations` fields empty — those are the *adapter* channel's to fill from a native description's schemas (see [ADR-0015](docs/adrs/0015-manuals-protocol-agnostic-adapter-boundary.md)). Hand-authored prose is not lifted into typed fields (that extraction was brittle and unread); a consumer that wants one section — the operations for a binding, usage & safety for a suspend judgment — reads `manual.section(ManualSection.OPERATIONS)` (the six canonical section titles are the `ManualSection` StrEnum — one source of truth, no literals to mistype), a lazy slice of `raw_text` on its `#` headings, and the whole manual is just `raw_text`. When a consumer eventually needs machine-readable schemas *from* hand-authored manuals, that content moves to a structured header (front-matter) rather than being regex-lifted from prose.

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
- Situate: the agent selects an activity and adjusts its working memory for that activity — for example, by loading required manuals, unloading obsolete ones, and filtering the perceptual input; if an unhandled message in working memory doesn't correspond to any existing activity, Situate creates one via the internal _create_activity_ action before selecting; which ready activity to select — the agent's scheduler — is its own pluggable sub-strategy, defaulting to fair round-robin rotation over the ready set (anti-starvation, still no model call) so richer policies (priority, aging, deadlines, an LLM-based scheduler) can replace just the pick without re-authoring the rest of Situate
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
- **LLM access**: the runtime's one seam onto a model is the wire-format-neutral `LLMClient`
  Protocol (`sora/llm.py`) — system + prompt in, text out, committing to no provider shape — so no
  hard dependency on a single provider SDK. The shipped default client targets the Anthropic
  Messages API via the optional `[llm]` extra (model id is a config value, never hardcoded; the
  provider API key is supplied through the environment — e.g. `ANTHROPIC_API_KEY` — never committed
  to `agent.yaml`, see [Configuring the LLM](#configuring-the-llm-and-its-api-key)). Provider SDKs
  are optional extras. The model call itself lives in `ProceduralMemory.infer`, behind the default
  `ReasonStrategy`; the text→`Plan` conversion there is the anti-corruption boundary.
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
    $ uv sync --extra mcp --extra llm        # llm extra: the model-backed default Reason strategy
    $ export ANTHROPIC_API_KEY=sk-ant-...     # credentials via the environment (see Configuring the LLM)
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
The joined workspaces *are* the toolset the default Situate works from: each cycle it loads their
tools' manuals into working memory (`_load_`), unloads any no longer backed by a joined workspace
(`_unload_`), and filters *observable-property* percepts down to the joined workspaces' tools
(`_filter_`). `_filter_` prunes only properties — a re-observed snapshot, safe to drop and reproduced
next cycle. Signals are **fire-and-forget** and are never dropped by `_filter_`: a signal may still
matter to another (or a `blocked`) activity, so its retention/eviction is consumption-driven and
owned by the blocked-state machinery, not a per-cycle prune. The default does not auto-*focus* —
focusing is an external action (one per cycle, dispatched at Act), so an agent that needs to perceive
a tool's properties/signals emits `_focus_` as a plan step.

#### Connecting to an MCP server: remote vs. local

An `mcp` (or `are-mcp`) workspace connects over whichever transport its entry describes — the runtime
does **not** have to deploy the server itself:

    workspaces:
      # Remote: connect to an already-running server (nothing is spawned). `address` is the URL;
      # SSE is the default, or add `transport: streamable-http`.
      - origin: {adapter: mcp, address: "http://localhost:8080/sse"}
        workspace_id: remote-tools

      # Local: the adapter spawns and owns a stdio subprocess. Give it a `command` (+ `args`);
      # `address` is then just a nominal label.
      - origin: {adapter: mcp, address: "stdio:clock"}
        workspace_id: clock
        command: uvx
        args: ["mcp-server-clock"]

The rule is simply: an entry with a `command` runs a local stdio subprocess; otherwise `address` is
treated as the URL of an existing server to connect to. Either way `discover()` enumerates the
server's tools and `restore()` reconnects the same way — the transport is the only thing that differs.

### Configuring the LLM and its API key

The default `ReasonStrategy` is **model-backed** — Reason is the one phase with no mechanical default,
since planning needs a model — so running an agent with it requires an LLM. Install the provider extra
and supply credentials through the **environment**, never through `agent.yaml`:

    $ uv sync --extra llm                    # the default AnthropicLLMClient (official Anthropic SDK)
    $ export ANTHROPIC_API_KEY=sk-ant-...     # the secret lives in the environment, not in any file
    $ uv run sora run

The API key is a **secret — keep it out of version control.** `agent.yaml` is committed, so it names
the *model* (a config value you can swap freely, e.g. `claude-opus-4-8`) but never the key. The
shipped `AnthropicLLMClient` reads the key from the environment via the Anthropic SDK's standard
resolution — `ANTHROPIC_API_KEY`, or an `ant auth login` profile for local dev — so the client needs
no key in code or config. Only pass one explicitly (`AnthropicLLMClient(api_key=...)`) when you must
inject a specific key programmatically. In production, load the key from a secrets manager or a
`.gitignore`d `.env` at start-up; never paste keys into `agent.yaml`, manuals, prompts, or source.

For local development, **copy `.env.example` to `.env`** and set `ANTHROPIC_API_KEY` there — `sora run`
loads a local `.env` automatically when present, so you don't need to `export` it each time. `.env`
is gitignored, and **real environment variables still take precedence** (a `.env` value is used only
when the variable isn't already set), so it never silently overrides a key you exported deliberately.

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
        next_action: str      # an ExternalAction.name ("invoke", "send", "focus", ...) or the WAIT sentinel
        params: dict          # the action's own argument bag, passed through opaquely and destructured by
        #                       the action — shape is per-action (send -> {to, content}, focus -> {tool_id}).
        #                       `invoke` mixes routing (tool_id/operation_name, under the TOOL_ID/OPERATION_NAME
        #                       keys) with the operation's args; Act's bind splits them. Build one via invoke_step().

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

    @dataclass(frozen=True)
    class CompletedOperation:   # one resolved invocation + its ack — an entry in Activity.history
        invocation: OperationInvocation   # a later step can ground its params against it: a
        ack: OperationAck                 # $from reference reads an earlier operation's result here

    # sora/environment.py — usage interface + adapters (S-ORA does not author tools, only consumes them)
    class Tool(Protocol):
        id: str               # globally unique, derived from the tool's global address/origin — see ADR-0014
        manual: Manual
        address: str | None   # a locator (may be absent), not identity; overrides the workspace's address when this tool has its own endpoint
        async def invoke(self, operation_name: str, **params) -> OperationAck: ...
        async def focus(self, sink: SignalSink) -> None: ...
        async def unfocus(self) -> None: ...
        def observe(self) -> list[ObservableProperty]: ...
    
    @dataclass(frozen=True)
    class WorkspaceOrigin:
        """The part of a WorkspaceRecord only the adapter can know: how to (re)connect."""
        adapter: str    # e.g. "mcp", "wot" — matches WorkspaceAdapter.name
        address: str      # e.g. an MCP server URI, or a WoT directory's base href; for a stdio-spawned
                          # server it's a stable nominal label, not a locator — the adapter holds the
                          # command/args and is keyed by origin, so restore() reconnects without them

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
        within that protocol (see Tool.address) is a separate, orthogonal concern. Each adapter assigns
        globally-unique tool ids, derived deterministically from the tool's global address/origin so
        restore() reproduces them (see ADR-0014)."""
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

    class EnvironmentView(Protocol):
        """Read-only projection of the live environment that WorkingMemory exposes to strategies: they
        reason over the currently-joined workspaces and tools — a legitimate part of the agent's current
        context — but cannot mutate connections through it (join/leave/restore live only in the action
        space; mypy --strict enforces the read-only boundary). EnvironmentRegistry satisfies this
        structurally and adds the mutators. See ADR-0013."""
        def get(self, tool_id: str) -> Tool: ...
        def get_workspace(self, workspace_id: str) -> Workspace: ...
        def all_tools(self) -> list[Tool]: ...
        def joined_workspaces(self) -> list[Workspace]: ...   # the live joined set, for reasoning

    class EnvironmentRegistry:        # was ToolRegistry — now tracks workspaces, not just flattened tools
        """Live, in-process handles for workspaces (and their tools) the agent currently has a connection
        to. Populated by join()/restore() — never persisted directly (see WorkspaceRecord/ToolRecord).
        The single shared instance (built in bootstrap): DecisionCycle holds it mutation-capable for
        action dispatch, and WorkingMemory mirrors the same object read-only as an EnvironmentView."""
        def __init__(self, adapters: dict[WorkspaceOrigin, WorkspaceAdapter] | None = None):
            """Keyed by the full origin (adapter + address), not just adapter name — an agent can join
            multiple workspaces that share a protocol (e.g. two separate MCP servers) without ambiguity."""
        def get(self, tool_id: str) -> Tool: ...   # tool_id is globally unique — ADR-0014
        def get_workspace(self, workspace_id: str) -> Workspace: ...
        def all_tools(self) -> list[Tool]: ...
        def joined_workspaces(self) -> list[Workspace]: ...   # satisfies EnvironmentView
        async def join(self, origin: WorkspaceOrigin) -> Workspace:
            """Predefined external action _join_: looks up the adapter registered for this exact origin,
            calls its discover() (config-scoped to just this target today), registers the workspace.
            Raises if a discovered tool id collides with one already registered — the adapter must
            guarantee globally-unique ids (ADR-0014), so a collision the registry can see is a bug,
            not a silent overwrite (it can only see its own agent's joins)."""
        async def leave(self, workspace_id: str) -> None:
            """Predefined external action _leave_: closes the workspace's connection, deregisters it
            and all its tools."""
        async def restore(self, workspace_records: list[WorkspaceRecord], tool_records: list[ToolRecord],
                           semantic: SemanticMemory) -> list[Workspace]:
            """Reconnects to already-known workspaces via adapter.connect() — one call per workspace,
            looking up each one's adapter by workspace_record.origin, resolving each tool's manual from
            SemanticMemory first. Skips discovery entirely."""

    # sora/perception.py
    class PerceptKind(StrEnum):   # closed set — one source of truth; each member == its str value
        PROPERTY = "property"
        SIGNAL = "signal"

    @dataclass(frozen=True)
    class Percept:
        source: str            # tool id
        kind: PerceptKind       # genuine environment stimuli only; an invoked operation's own result
        payload: Any             # is not a Percept (see Activity.pending_operation/last_operation) and
        observed_at: float        # neither are agent messages (see WorkingMemory.messages)

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
    class ManualSection(StrEnum):   # the six canonical `#`-headed manual sections — one source of truth
        METADATA = "Tool Metadata"; DESCRIPTION = "Functional Description"
        OBSERVABLE_PROPERTIES = "Observable Properties"; SIGNALS = "Signals"
        OPERATIONS = "Operations"; USAGE_AND_SAFETY = "Usage Protocols & Safety"

    @dataclass(frozen=True)
    class OperationSpecification:   # was Operation — renamed for symmetry with the two specs below
        name: str
        description: str     # folds in any Preconditions/Effects/Behavior sub-bullets as prose (see
                             #   Tool Manuals); discrete fields deferred until a strategy consumes them
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
        # structured specs: the adapter channel fills these from a native description; the
        # hand-authored Markdown channel leaves them empty and carries content in raw_text
        observable_properties: list[ObservablePropertySpecification]
        signals: list[SignalSpecification]
        operations: list[OperationSpecification]
        raw_text: str | None = None    # verbatim authored source (Markdown channel); None if synthesized
        def section(self, name: str) -> str | None: ...   # lazy `#`-section slice of raw_text

    class ManualParser(Protocol):     # Markdown by default, XML pluggable
        def parse(self, raw: str) -> Manual: ...

    class ManualParseError(ValueError): ...   # e.g. a manual with no derivable id
    class MarkdownManualParser:               # the default ManualParser (clean Markdown format)
        def parse(self, raw: str) -> Manual: ...   # yields a Manual envelope (raw_text; specs empty)

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
        id: str            # instance id, matches Tool.id once live; globally unique + stable across reconnect (ADR-0014)
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
        history: list[CompletedOperation] = []              # append-only trace of resolved ops — a later step
        #                                                     grounds param references against it (see Reason
        #                                                     grounding); last_operation keeps only the newest
        # context and is exclusively for strategy-author data — the runtime itself never writes into it,
        # which is what keeps pending_operation/last_operation as dedicated fields instead of context keys
        # with a naming convention (no shared namespace means no collision to avoid in the first place)

    # sora/action.py — extensible action space
    class InternalAction(Protocol):
        name: str
        async def execute(self, cycle: DecisionCycle, **kwargs) -> Any:
            """No EnvironmentRegistry access — internal actions only ever touch memory."""

    class ExternalAction(Protocol):
        name: str
        requires_binding: bool     # whether Act must do *parameter binding* on this step — grounding its
        #                            abstract Step into a concrete OperationInvocation (not a *protocol
        #                            binding*, the adapter's Tool concern) — before dispatch. Only _invoke_
        #                            does; every other action dispatches straight from its Step params.
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
        requires_binding = True        # abstract Step -> a concrete, schema-conformant OperationInvocation
        async def execute(self, registry: EnvironmentRegistry, cycle: DecisionCycle, *,
                           activity_id: str, tool_id: str, operation_name: str, **params) -> ActionAck:
            tool = registry.get(tool_id)
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

    def invoke_step(tool_id: str, operation_name: str, **op_args) -> Step:
        """Constructor for an `invoke` Step: packs the routing keys (tool_id, operation_name) alongside
        the operation arguments in Step.params under the TOOL_ID/OPERATION_NAME constants — the one Step
        whose params bag mixes routing with arguments (DefaultActStrategy.bind splits them). Use this
        rather than hand-writing that magic-keyed dict."""
        return Step(next_action=InvokeAction.name,
                    params={TOOL_ID: tool_id, OPERATION_NAME: operation_name, **op_args})

    class FocusAction:                # predefined external action: _focus_
        name = "focus"
        async def execute(self, registry: EnvironmentRegistry, cycle: DecisionCycle, *,
                           tool_id: str, **kwargs) -> ActionAck:
            tool = registry.get(tool_id)
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
            # drop the tool's now-stale property snapshot (signals stay — fire-and-forget)
            cycle.working.perceptions[:] = [p for p in cycle.working.perceptions
                                            if not (p.kind == "property" and p.source == tool_id)]
            return ActionAck(ok=True)

    class JoinAction:                  # predefined external action: _join_ — implies discover/connect
        name = "join"
        async def execute(self, registry: EnvironmentRegistry, cycle: DecisionCycle, *,
                           origin: WorkspaceOrigin, **kwargs) -> ActionAck:
            workspace = await registry.join(origin)
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
            # workspace_id addresses it (for a later _leave_); tool_ids are a self-contained
            # snapshot of what was gained, legible after leave / across an agent boundary.
            # The snapshot is useful for logging (e.g., saving an episode to memory).
            return ActionAck(ok=True, result={
                "workspace_id": workspace.id,
                "tool_ids": [tool.id for tool in workspace.tools()],
            })

    class LeaveAction:                 # predefined external action: _leave_ — implies close
        name = "leave"
        async def execute(self, registry: EnvironmentRegistry, cycle: DecisionCycle, *,
                           workspace_id: str, **kwargs) -> ActionAck:
            for tool in registry.get_workspace(workspace_id).tools():   # unfocus first: leaving
                focused = cycle.working.focused_tools.pop(tool.id, None) # deregisters these tools,
                if focused is not None:                                  # so no stale focus (live
                    await focused.unfocus()                             # subscription) is left behind
            await registry.leave(workspace_id)
            return ActionAck(ok=True)

    class SendAction:                  # predefined external action: _send_
        name = "send"
        async def execute(self, registry: EnvironmentRegistry, cycle: DecisionCycle, *,
                           to: str, content: dict, **kwargs) -> ActionAck:
            await cycle.communication.send(to, content)   # registry unused here — every ExternalAction still
            return ActionAck(ok=True)                  # gets the same uniform (registry, cycle) signature

    # Predefined internal actions — the (cycle, **kwargs) signature, memory-only (no registry). These
    # are the *mechanism* half of the working-memory levers Situate drives; the *policy* (which goal,
    # which manuals) lives in the SituateStrategy.
    class CreateActivityAction:        # predefined internal action: _create_activity_
        name = "create_activity"
        async def execute(self, cycle: DecisionCycle, **kwargs) -> Activity:
            activity = Activity(id=kwargs.get("activity_id") or new_id(),
                                goal=kwargs["goal"], context=kwargs.get("context") or {})
            cycle.working.activities[activity.id] = activity  # goal from an unhandled message
            return activity

    class LoadManualAction:            # predefined internal action: _load_
        name = "load"
        async def execute(self, cycle: DecisionCycle, **kwargs) -> None:
            manual = await cycle.semantic.retrieve_manual(kwargs["manual_id"])
            if manual is not None:     # unknown id -> no-op (a stale reference can't crash the cycle)
                cycle.working.loaded_manuals[kwargs["manual_id"]] = manual

    class UnloadManualAction:          # predefined internal action: _unload_
        name = "unload"
        async def execute(self, cycle: DecisionCycle, **kwargs) -> None:
            cycle.working.loaded_manuals.pop(kwargs["manual_id"], None)   # absent id -> no-op

    class FilterPerceptionsAction:     # predefined internal action: _filter_
        name = "filter"
        async def execute(self, cycle: DecisionCycle, **kwargs) -> None:
            tool_ids = kwargs["tool_ids"]        # prune observable-property percepts to relevant tools;
            cycle.working.perceptions[:] = [     # signals are fire-and-forget -> always retained (their
                p for p in cycle.working.perceptions   # eviction is consumption-driven, owned by the
                if p.kind is PerceptKind.SIGNAL or p.source in tool_ids]  # blocked-state machinery)

    def default_action_registry() -> ActionRegistry:   # the six external + four internal, assembled once
        ...                                            # what bootstrap and test harnesses register through

    # sora/llm.py — the one seam onto a language model; wire-format-neutral on purpose
    class LLMClient(Protocol):
        """A single completion round-trip: a system instruction + a prompt in, text out. Commits to
        no provider shape (not OpenAI chat/completions, not Anthropic messages), so the reasoning
        path stays SDK-independent and the concrete client (an optional extra under sora/adapters —
        AnthropicLLMClient, model id from config) is the only place a wire format appears. Owns
        *only* the round-trip: retries, streaming, credential refresh, prompt caching, and interrupt
        are the cycle/agent's. The text -> Plan anti-corruption boundary is ProceduralMemory.infer,
        not here."""
        async def complete(self, *, system: str, prompt: str) -> str: ...

    # sora/memory.py
    class MemoryBackend(Protocol):    # pluggable: file, DB, vector store
        async def get(self, key: str) -> Any: ...
        async def put(self, key: str, value: Any) -> None: ...
        async def query(self, **filters) -> list[Any]:
            """Every stored value matching all `filters`, ordered most-relevant-first with ties
            broken deterministically: a caller may treat `result[0]` as the single best/canonical
            match and the order as stable across identical calls. Backends with a relevance notion
            (a vector store) rank by it; backends without one (exact-match file storage) treat all
            matches as equally relevant and fall back to a stable key order. This guarantee is what
            lets ProceduralMemory.retrieve() take the top match without knowing the backend."""

    class FileMemoryBackend:          # the default: one JSON file per key under a root directory
        """Deals only in JSON-serializable values — the memory modules serialize their dataclasses
        to/from dict/list/scalar, so the backend stays generic (a DB/vector-store backend is a true
        drop-in). Reads re-parse from disk, so returned values are fresh copies, never live refs.
        Writes are atomic (temp file + os.replace). Keys are quoted into safe filenames, so URI /
        <App>__<op> tool ids work as keys."""
        def __init__(self, root: str | Path): ...

    class WorkingMemory:              # transient, in-process, fast
        registry: EnvironmentView     # read-only view of the live joined workspaces/tools: the agent
                                       # reasons over what it's currently connected to; the durable
                                       # WorkspaceRecord/ToolRecord knowledge stays in SemanticMemory
                                       # (what am I connected to now vs. what have I ever discovered)
        activities: dict[str, Activity]
        perceptions: list[Percept]    # stimuli from the environment: properties and signals only
        messages: list[Message]        # inbound agent-to-agent communication — kept distinct
        focused_tools: dict[str, Tool]
        loaded_manuals: dict[str, Manual]  # manuals pulled from SemanticMemory by _load_ (removed by
                                            # _unload_) — distinct from focused_tools: focusing a tool
                                            # is I/O (an external action), loading its manual is memory

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

    class PlanPrompt(Protocol):   # builds infer()'s (system, user) prompt from (activity, tools)
        def __call__(self, activity: Activity, tools: dict[str, Manual]) -> tuple[str, str]: ...
        #   default_plan_prompt is the built-in one; PLAN_SYSTEM_PROMPT / render_tools are reusable
        #   pieces a custom PlanPrompt can lean on. The response contract ({"steps":[...]}) stays
        #   fixed — customize the *prompt*, not the parse. PLAN_SYSTEM_PROMPT also tells the model to
        #   emit a *reference* — {"$from": "<op>", "path": "<dotted path>"} or {"$decide": "..."} —
        #   for a param whose value depends on an earlier step's result, never a made-up literal.

    class GroundPrompt(Protocol):   # builds ground()'s (system, user) prompt — grounding's counterpart
        def __call__(self, activity: Activity, operation_name: str, manual: Manual | None,
                     partial_params: dict) -> tuple[str, str]: ...
        #   default_ground_prompt is the built-in one; GROUND_SYSTEM_PROMPT / render_history are the
        #   reusable pieces. Response contract is fixed ({"params": {...}}).

    class ProceduralMemory:
        def __init__(self, backend: MemoryBackend, llm: LLMClient | None = None,
                     prompt: PlanPrompt = default_plan_prompt,
                     ground_prompt: GroundPrompt = default_ground_prompt): ...
        #   llm is the model behind infer()/ground(); None keeps store/retrieve usable with no LLM.
        #   prompt / ground_prompt are the knobs for planning / grounding *content*.
        async def retrieve(self, activity: Activity) -> Plan | None:
            """Looks up a cached Plan matching this activity's goal — e.g. exact match or embedding
            similarity, backend-dependent. Returns the backend's top-ranked match (query() orders
            most-relevant-first — see MemoryBackend), so this stays one line regardless of backend.
            The cheap path: skips infer() entirely when it hits."""
        async def infer(self, activity: Activity, tools: dict[str, Manual]) -> Plan:
            """Produces a new multi-step Plan when no cached one fits — the model path: one LLMClient
            call producing a whole sequence of Steps at once. This is procedural memory querying its
            'implicit knowledge encoded in LLM weights'. `tools` (id -> its Manual) is the planning
            catalog, passed in by the caller that holds the live registry (a memory module never
            reaches into the environment). Converts the model's JSON answer into Plan/Step (the
            anti-corruption boundary); malformed output raises ValueError. No llm -> raises."""
        async def ground(self, activity: Activity, operation_name: str, manual: Manual | None,
                         partial_params: dict) -> dict:
            """The Reason-phase grounding *escalation*: decide an operation's concrete params from the
            execution context when a reference can't be resolved mechanically. One LLMClient call over
            the operation schema + partial params + the activity's history; parses {"params": {...}}
            (anti-corruption); no llm -> raises. Packaged here (like infer) because procedural memory
            owns the model handle; grounding a step is really an Act-adjacent reasoning act — see
            ADR-0017. (The mechanical reference resolver lives in Reason, not here.)"""
        async def store(self, plan: Plan) -> None:
            """Persists a Plan that was actually followed to completion, so future retrieve() calls for
            similar goals can reuse it. Called by ReflectStrategy on success only — a failed plan isn't
            something future activities should retrieve by default."""

    class EpisodicMemory:
        def __init__(self, backend: MemoryBackend): ...
        async def learn(self, activity: Activity, summary: str, *, succeeded: bool) -> None:
            """Records one episode per activity (keyed by its id). Beyond the prose summary, the
            stored record is a self-contained experience — outcome, the plan snapshot, step progress
            (step_index/step_count), and the last operation result — capturing as much as survives
            on the activity. `succeeded` is passed in because ActivityState.TERMINATED can't tell a
            completed activity from a failed one; only the judging ReflectStrategy knows. The plan is
            kept in full even on success (procedural memory holds it too): on failure it's the only 
            copy, since procedural memory does not store failed plans."""
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
            """Selects the next activity and adjusts wm for it. Always runs — unlike Reason/Act it is
            not gated on its own output field, because adjusting wm (selecting tools, loading/unloading
            manuals, filtering percepts) must reflect this cycle's fresh percepts even for an
            already-selected activity. Selects only if result.activity is still None; a pre-set
            selection (uncommon — e.g. an Observe that pins the activity handling a critical signal) is
            respected and situated, not overridden. Also responsible for activity creation: if
            wm.messages has one that doesn't correspond to any existing activity, invokes the internal
            _create_activity_ action (via cycle) before selecting. Head of the decision chain (Situate
            -> Reason -> Act) and the intended entry point for fusing the remaining phases into one
            model call — it runs after this cycle's percepts and messages are already in working memory.
            May additionally fill in step/invocation, short-circuiting Reason/Act (those forward-fusion
            gates remain; only Situate's own activity gate is removed)."""

    class ActivitySelectionStrategy(Protocol):   # Situate's scheduler; own pluggable sub-strategy
        async def select(self, ready: list[Activity], wm: WorkingMemory,
                          cycle: DecisionCycle) -> Activity | None:
            """Picks which ready activity progresses this cycle (empty -> None) — a scheduling
            policy, not a phase. DefaultSituateStrategy delegates its pick here so a richer scheduler
            (priority, aging, deadlines, an LLM-based one) swaps in without re-authoring Situate's
            activity-creation and wm-adjustment. `async` + `cycle` let such a policy consult memory
            or a model; the default (RoundRobinActivitySelection) consults neither."""

    class ReasonStrategy(Protocol):   # pluggable; default targets 1 LLM call/cycle
        async def reason(self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle,
                          result: TickResult) -> TickResult:
            """Only called if result.step is still None. Typical implementation: if activity.plan is
            already set and still valid, just read activity.plan.steps[activity.step_index] and advance
            the index — no model call. Otherwise, retrieve a cached Plan via cycle.procedural.retrieve()
            or infer a new one (the expensive path), reset step_index to 0, and use its first Step.
            Deciding when a plan counts as invalidated is entirely up to the implementation. Also
            *grounds* the step: a param whose value depends on an earlier result is a reference the
            default resolves against activity.history, escalating to cycle.procedural.ground() only
            when it can't be resolved mechanically — deciding a value is reasoning, so it lives here
            (ADR-0017). May additionally fill in invocation, short-circuiting Act — this is where the
            historical 'tool hallucination' risk lives if it does."""

    class ActStrategy(Protocol):
        async def bind(self, step: Step, manual: Manual | None, cycle: DecisionCycle,
                        result: TickResult) -> TickResult:
            """Only called if result.invocation is still None. *Parameter binding*: split an invoke
            Step's routing keys from its (by now already-grounded) params into a concrete
            OperationInvocation. Mechanistic — deciding param *values* is Reason's grounding, not
            Act's (ADR-0017). Distinct from a *protocol binding* (WoT forms/security, an MCP session)
            — how the adapter's Tool reaches the instance, never surfaced here (ADR-0015). `cycle` is
            available for implementations that cache bindings rather than re-deriving one each time."""

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
            # Properties are persistent, re-observed state: one percept per (source, name),
            # last value wins — a replaced snapshot, not a growing append log.
            index = {(p.source, p.payload.name): i
                     for i, p in enumerate(cycle.working.perceptions) if p.kind == "property"}
            for tool in cycle.working.focused_tools.values():
                for prop in tool.observe():
                    percept = Percept(tool.id, "property", prop, now())
                    key = (tool.id, prop.name)
                    if key in index:
                        cycle.working.perceptions[index[key]] = percept   # replace in place
                    else:
                        index[key] = len(cycle.working.perceptions)
                        cycle.working.perceptions.append(percept)
            async for source, signal in cycle.signal_sink.drain():
                cycle.working.perceptions.append(Percept(source, "signal", signal, now()))  # append
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

    # DefaultReflectStrategy / DefaultSituateStrategy / DefaultActStrategy: the mechanical, no-LLM
    # defaults for the other decision-chain phases — same role as DefaultObserveStrategy (bodies
    # provisional). Named here so the sketch matches the code's default set; wired in by bootstrap
    # as sora.reflect.default / sora.situate.default / sora.act.default.

    class DefaultReasonStrategy:
        """Reason's default — the effective default Reason strategy (Reason has no *mechanical*
        default; planning is inherently the model path). Deterministic orchestration around the one
        model call, which is isolated in ProceduralMemory.infer: cheap path advances an existing
        plan's step_index (no model, no lookup); else reuse a cached plan (procedural.retrieve) or
        infer a fresh one (procedural.infer, passing the joined tools id->Manual as the catalog); an
        exhausted plan yields no step. Wired in by bootstrap as sora.reason.default."""
        async def reason(self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle,
                         result: TickResult) -> TickResult: ...

    class RoundRobinActivitySelection:
        """DefaultSituateStrategy's default selection sub-strategy: fair rotation over the ready set,
        carrying a last-selected-id cursor across cycles (cold start / last-pick-gone -> oldest).
        Deterministic, no LLM. Anti-starvation replacement for a static priority-by-age pick, which
        reselects an activity that lingers READY every cycle and starves younger ones.
        DefaultSituateStrategy(selection=RoundRobinActivitySelection()) delegates the pick to it."""

    # sora/transport.py
    class MessageTransport(Protocol): # pluggable: A2A, HTTP, in-process
        async def send(self, to: str, content: dict) -> None: ...
        def receive(self) -> AsyncIterator[Message]: ...   # non-async: returns an async generator

    class InProcessTransport:
        """The single-agent default: an in-process inbox, no network. receive() drains what's queued
        now; whoever holds the agent (CLI/showcase/test) delivers inbound goals via submit(). 
        send() records outbound content. A peer-to-peer transport (A2A/HTTP, transport.peers) is the 
        multi-agent case."""
        def submit(self, message: Message) -> None: ...

    # sora/cycle.py
    class DecisionCycle:
        def __init__(self, strategies: Strategies, communication: MessageTransport,
                     actions: ActionRegistry, registry: EnvironmentRegistry,
                     working: WorkingMemory, semantic: SemanticMemory,
                     procedural: ProceduralMemory, episodic: EpisodicMemory):
            self.registry = registry   # the shared, mutation-capable handle, passed to external
            #                            actions at dispatch; WorkingMemory holds the same instance
            #                            read-only (as EnvironmentView) for strategies to reason over.
            # Both sinks live here rather than on WorkingMemory: they're the bridge from
            # asynchronous, off-cycle events into this engine's tick()/interrupt() — not settled
            # state. signal_sink specifically has to be co-located with interrupt() below, since
            # a pushed Signal can preempt the current phase; that control-flow role, not "where
            # it eventually lands as a percept," is why it isn't a WorkingMemory field.
            self.signal_sink: NotificationQueueSink[Signal] = NotificationQueueSink()        # tools push here via focus()
            self.result_sink: NotificationQueueSink[OperationAck] = NotificationQueueSink()  # InvokeAction pushes here — internal only
            ...
        async def tick(self) -> None:
            """One Observe -> Reflect -> Situate -> Reason -> Act pass, threading a TickResult through
            all five phases and calling each phase's own strategy only for whatever's still missing —
            so a fully-fused Observe (or Reflect) call can skip the rest of the cycle entirely. Takes
            no arguments: registry/working/semantic/procedural/episodic/communication are all shared
            with Agent, constructed once and passed to both — see sora/bootstrap.py. (Dispatch uses
            self.registry — the mutation-capable handle — not working.registry, which is read-only.)"""
            result = await self.strategies.observe.observe(self)
            for activity in self.working.activities.values():
                result = await self.strategies.reflect.reflect(activity, self.working, self, result)
            # Situate always runs: it re-situates wm for the (possibly already-selected) activity every
            # cycle, and selects only if result.activity is still None. Unlike the step/invocation gates
            # below — genuine forward-fusion short-circuits — Situate is not gated on its own field.
            ready = [a for a in self.working.activities.values() if a.state is ActivityState.READY]
            result = await self.strategies.situate.situate(ready, self.working, self, result)
            if result.activity is None:
                return               # nothing selectable this cycle — at most one action, never a mandatory one
            if result.step is None:
                result = await self.strategies.reason.reason(result.activity, self.working, self, result)
            if result.step is not None:
                await self._act(result.activity, result.step, result)   # bind-then-dispatch boundary

        async def _act(self, selected: Activity, step: Step, result: TickResult) -> None:
            """WAIT is the cycle's no-op sentinel — guarded first, before the registry lookup that
            would otherwise KeyError on it. Otherwise resolve the step's ExternalAction and let *it*
            declare whether the step needs binding (requires_binding) — only _invoke_ does, so the
            generic cycle stays uncoupled from any one action's name and a custom binding action binds
            too — then dispatch exactly one external action: the bound invocation's routing keys +
            params when present, else the raw step params (invoke resolves its tool through the
            registry, not the focus set)."""
            if step.next_action == "wait":
                return
            action = self.actions.external(step.next_action)
            if result.invocation is None and action.requires_binding:
                tool = self.registry.get(step.params["tool_id"])
                result = await self.strategies.act.bind(step, tool.manual, self, result)
            # dispatch result.invocation (if set) or step.params to `action` via action.execute,
            # always passing activity_id=selected.id — elided, same as the rest of Act's dispatch today
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
                     communication: MessageTransport, *, tick_interval: float = 0.05): ...
        async def run(self) -> None:
            """Join the configured workspaces once at startup (through the _join_ action, so records/
            manuals persist and the tools are already available on the first cycle — README's
            'joined automatically at startup'), then loop await self.cycle.tick() until stop(),
            leaving the workspaces finally. The join lives here, not in the synchronous
            bootstrap, because it is async I/O."""
        async def stop(self) -> None: ...

    # sora/cli.py — the runtime's minimal terminal interface
    class TerminalSession:
        """Streams cycle output to stdout; queues stdin as Message(sender="user", ...) — not a Percept,
        since terminal input is user communication, not environment stimuli. No UI beyond this."""
        def __init__(self, agent: Agent, verbose: bool = False): ...
        async def run(self) -> None: ...

    # sora/bootstrap.py — internal; developers implement protocols, they don't call this directly
    @dataclass(frozen=True)
    class AgentConfig:
        """The parsed agent.yaml `agent:` block. strategies/memory are dotted-path / URI maps
        resolved during build_agent; workspaces is the raw list (each entry: an `origin` plus
        adapter-specific keys like command/args); llm is optional (absent -> no model)."""
        name: str
        strategies: dict[str, str]
        memory: dict[str, str]
        workspaces: list[dict]
        transport: dict | None = None
        llm: dict | None = None

    def import_object(path: str) -> Any: ...        # resolve a dotted (pkg.mod.Attr) / module:attr path
    def load_yaml(config_path: str) -> AgentConfig: ...  # parse agent.yaml; require strategies.reason
    def backend_for(spec: str) -> MemoryBackend: ...     # file://<path> (or bare path) -> FileMemoryBackend
    def adapter_for(entry: dict) -> tuple[WorkspaceOrigin, WorkspaceAdapter]: ...  # dispatch on origin.adapter
    def llm_for(config: AgentConfig) -> LLMClient | None: ...  # the llm: block -> a client, else None
    def transport_for(config: AgentConfig) -> MessageTransport: ...  # InProcessTransport (peers -> raise)

    def build_agent(config_path: str) -> Agent:
        """What `sora run` calls before handing off to TerminalSession. This is the one place all the
        wiring (which memory backend, which transport, which adapters, DecisionCycle <-> Agent sharing
        the same instances) actually happens — a developer implementing an agent never writes this.
        Stays synchronous: the async startup join runs in Agent.run()."""
        load_dotenv()   # convenience for development
        config = load_yaml(config_path)
        adapters = dict(adapter_for(entry) for entry in config.workspaces)
        registry = EnvironmentRegistry(adapters=adapters)   # the single shared instance...
        working = WorkingMemory(registry=registry)          # ...held here read-only as EnvironmentView
        semantic = SemanticMemory(backend_for(config.memory["semantic"]))
        procedural = ProceduralMemory(backend_for(config.memory["procedural"]), llm=llm_for(config))
        episodic = EpisodicMemory(backend_for(config.memory["episodic"]))
        communication = transport_for(config)
        strategies = Strategies(
            observe=import_object(config.strategies.get("observe", "sora.strategies.DefaultObserveStrategy"))(),
            reflect=import_object(config.strategies.get("reflect", "sora.strategies.DefaultReflectStrategy"))(),
            situate=import_object(config.strategies.get("situate", "sora.strategies.DefaultSituateStrategy"))(),
            reason=import_object(config.strategies["reason"])(),   # required — Reason has no default
            act=import_object(config.strategies.get("act", "sora.strategies.DefaultActStrategy"))(),
        )

        cycle = DecisionCycle(strategies=strategies, communication=communication,
                               actions=default_action_registry(), registry=registry, working=working,
                               semantic=semantic, procedural=procedural, episodic=episodic)
        return Agent(cycle=cycle, registry=registry, working=working, semantic=semantic,
                     procedural=procedural, episodic=episodic, communication=communication)
```
