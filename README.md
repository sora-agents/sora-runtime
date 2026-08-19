# S-ORA Agent Runtime

A runtime for practical agents in dynamic and asynchronous environments.

> **Status:** this project is currently under development and follows README-driven design — this file and [EXAMPLES.md](EXAMPLES.md) are the spec. See [ROADMAP.md](ROADMAP.md) for implementation status and [docs/architecture/adrs/](docs/architecture/adrs/) for why specific decisions were made.

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

An activity is eligible for selection only when ready; running and blocked activities are skipped until something transitions them back. Invoking an operation always, implicitly, moves an activity to running until that operation's own result comes back — this is unconditional, independent of anything the tool's manual says, and resolving it back to ready is an unambiguous one-to-one match the runtime does automatically, with no strategy code involved. A manual can additionally require blocking on a specific signal before the next step, layered on top of (not instead of) that implicit wait — the two are orthogonal: the operation resolves to ready first, and a *separate* step then blocks the activity until the signal arrives. That block is likewise mechanical: an operation declares its completion signal in its manual (`OperationSpecification.completion_signal`), so entering `blocked` (the `_suspend_` action) and leaving it once the signal is observed (the `_resume_` action — the matched signal itself is left in `wm.signals`, not evicted, so it can also satisfy another activity waiting on the same signal, or a strategy reading it directly; only a fixed retention cap ever removes it) are both name-equality matches the Observe phase performs deterministically — no model call, no judgment. (A completion driven by an observable property reaching a state, rather than a signal, is a foreseen second form — deferred.)

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

A tool's `address` is a _locator_ and may be absent — e.g., tools multiplexed over one MCP stdio connection have none — whereas its `id` is the stable _handle_ the agent uses to focus and invoke it, and is **globally unique**: because a tool is a shared object, two agents focusing the same tool, or messaging about it, must name it identically. The per-protocol adapter guarantees this by deriving the id from the tool's global identity — its URI where the protocol provides one, or a value synthesized from the workspace's global origin/address otherwise — deterministically, so a later `restore()` reproduces the same id. A single registry can only enforce the ids it sees (it rejects a collision within its own joined set rather than letting one workspace's tool shadow another's); global uniqueness itself rests on the adapter. See [ADR-0014](docs/architecture/adrs/0014-tool-identity-globally-unique.md).

Joining and leaving a workspace are deliberate, agent-driven actions (_join_/_leave_), not the result of an eager, upfront scan of every configured target. Today, join targets are limited to workspaces declared in the agent's own configuration; open, dynamic discovery of previously-unknown workspaces (e.g., for open environments where not every tool is known in advance) is foreseen but deliberately deferred.

We break down the process of using a tool into five phases:

- Discovery: the agent discovers the tool at run time — for example, through MCP or another tool calling protocol that supports tool discovery
- Learning: the agent retrieves the tool's manual and loads it into its context; thus, the agent learns how to use the tool by reading its manual
- Focus: the agent decides whether to subscribe to the tool's observable properties and signals to perceive relevant state changes and domain events
- Operation: the agent invokes operations that return an immediate acknowledgment (but not necessarily the final result or outcome)
- Suspension and Resumption: if a tool's manual declares that a long-running operation's completion is marked by a specific signal (or, as a deferred second form, an observable property update), the runtime suspends the activity after invoking that operation and resumes it once the signal is observed — a mechanical match, not a per-operation decision

### Tool Manuals

Tools can be described by manuals. Any manual format can be used. S-ORA currently uses Markdown, though more structured formats (e.g., XML) may prove better suited for parsing and validation. Regardless of format, a manual is structured into six parts:

1. Tool Metadata: includes general metadata about the tool, such as category information (e.g., "Critical Infrastructure / Fluid Dynamics"), to facilitate dynamic loading into the agent's context window;
2. Functional Description: a short natural language description of the tool as a domain object and its intended purpose;
3. Observable Properties: definitions of observable properties that may populate the agent's working memory, such as the current state of an air conditioner (AC);
4. Signals: definitions of domain events that may be emitted by the tool, such as the AC reaching a target temperature;
5. Operations: definitions of commands to interact with the tool, including the commands' intended purposes, preconditions, and effects;
6. Usage Protocols & Safety: operating instructions, including safety constraints (if any) or conditions under which an activity must be suspended (e.g., to wait for specific signals).

In the Markdown rendering, Observable Properties, Signals, and Operations are `-` bullet lists. An operation bullet may additionally carry optional labeled sub-bullets — `Preconditions:`, `Effects:`, and `Behavior:` (whether the operation completes synchronously or is long-running, and which signal, if any, indicates completion) — expressing the operation semantics part 5 calls for. These are folded into the operation's single `description`: fully available to a reasoning strategy as text, but not lifted into discrete model fields until a strategy actually consumes them — the labels are the seams where that structure would later attach. The one such field a consumer now needs, the completion signal, is the exception: an author may declare it explicitly as `completes_on:` in the optional operations interface block (see The clean Markdown format / ADR-0018), which the parser lifts into `OperationSpecification.completion_signal` for the blocked-state machinery to match against — the mechanical wait the Activities section describes.

Property, signal, and operation entries carry their data shapes as JSON Schema in the spec types' `schema`/`parameters` fields (see the API Sketch). A manual describes a tool *type* and stays protocol-agnostic: JSON Schema is data shape, not a protocol binding, so it is filled either by an adapter from a native description (an MCP tool schema, a WoT TD affordance schema) or, for a hand-authored manual, lifted from the light `(type, range)` hints above — with an optional inline JSON Schema where full fidelity is needed. The protocol binding — how to actually reach one instance — lives on the live `Tool`, never in the manual. See [ADR-0015](docs/architecture/adrs/0015-manuals-protocol-agnostic-adapter-boundary.md).

#### The clean Markdown format

`MarkdownManualParser` (the default `ManualParser`) parses this format; malformed input raises `ManualParseError`. The document is a flat sequence of `# `-level sections whose headings are the six parts above (`# Tool Metadata`, `# Functional Description`, `# Observable Properties`, `# Signals`, `# Operations`, `# Usage Protocols & Safety`). `# Tool Metadata` is `key: value` lines — `id:` is **required** (it becomes `Manual.id`; a manual with no `id` is rejected), every other key lands in `metadata`; the remaining sections are free prose, with the observable-property / signal / operation lists written as `-` bullets (or the literal `(none)` when empty).

The parser yields a `Manual` **envelope**: it fills `id`, `metadata`, `description` (from Functional Description), and the verbatim `raw_text`, and leaves the structured `observable_properties` / `signals` / `operations` fields empty — those are the *adapter* channel's to fill from a native description's schemas (see [ADR-0015](docs/architecture/adrs/0015-manuals-protocol-agnostic-adapter-boundary.md)). Hand-authored prose is not lifted into typed fields (that extraction was brittle and unread); a consumer that wants one section — the operations for a binding, usage & safety for a suspend judgment — reads `manual.section(ManualSection.OPERATIONS)` (the six canonical section titles are the `ManualSection` StrEnum — one source of truth, no literals to mistype), a lazy slice of `raw_text` on its `#` headings, and the whole manual is just `raw_text`. When a consumer eventually needs machine-readable schemas *from* hand-authored manuals, that content moves to a structured header (front-matter) rather than being regex-lifted from prose.

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
- **procedural memory**: _retrieve_ a plan of action for the current activity, _infer_ one if a suitable one is not already known, or _store_ one that was actually followed to a successful completion (auto-store/reuse is currently disabled — the default cycle infers a fresh plan each activity, since replaying a stored plan verbatim is unsound; the operations remain for distilling reusable procedures from episodes)
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
- Reason: the agent infers a plan for the current activity (or retrieves a stored one, once procedure reuse is enabled — auto-caching is currently disabled, so each activity infers fresh) — a multi-step artifact, advanced across cycles rather than regenerated every cycle — and selects the next step to advance it; if the activity already has a valid plan, this is as cheap as reading its next step, no replanning involved; the Situate phase may suggest prerequisite external actions for situated reasoning, such as to retrieve tool manuals from an external repository, focus on or unfocus from tools; these prerequisite actions should take priority unless a more urgent action is needed — for example, to respond to a critical signal; if no prerequisite or urgent actions are required, the agent selects the next external action that advances the plan, which is either to send a message to another agent or invoke a tool operation
- Act: binds the step to a concrete invocation and executes the external action — mechanically, with no manual interpretation of its own. The suspend/resume that layers a signal-wait on top of a long-running operation is *not* done here: once the operation resolves, the Observe phase mechanically suspends the activity if the operation's manual declares a completion signal, and resumes it once that signal is observed (see Activities)

The five phases are a ceiling, not a quota: every cycle runs the pipeline, but a given cycle may conclude with one external action, with internal work only (e.g., storing experiences), or with nothing to do — at most one external action per cycle, never a mandatory one.

How many model calls a cycle costs is a configuration choice, not a property of the runtime. Observe and Reflect are deterministic by default: Observe mechanically ingests percepts and messages (an LLM-backed Observe is possible where perception itself needs interpretation — e.g., describing a camera snapshot — which runs off-cycle as an async internal action whose result lands as a percept a later cycle, not a fusion entry point), and Reflect's completion judgment may be deterministic or model-backed, with summarizing and storing dispatched asynchronously so they never block the cycle. Situate → Reason → Act form the decision chain proper — select an activity, advance its plan, bind a concrete invocation. The model calls this can need — infer a plan, ground a param — run off-cycle as internal actions: the activity waits in RUNNING and the result lands a later cycle, so no phase ever blocks the cycle. Fusing selection and planning into a single model call is a narrow synchronous-mode option, not the default, since it re-serializes that concurrency (ADR-0021). In the common case — an already-inferred plan being advanced, mechanical defaults — a cycle costs zero model calls. A hard interrupt can preempt the current phase for high-priority signals, independent of where the cycle is mid-flight — the 10ms reactiveness target, met by phase-boundary checkpoints; because model calls run off-cycle, no phase blocks on one, so there is no in-flight model call to cut short.

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

    $ git clone https://github.com/sora-agents/sora-runtime.git && cd sora-runtime
    $ uv sync --all-extras --dev
    $ uv run sora init ~/path/to/my-agent
    Created ~/path/to/my-agent/
      agent.yaml
      clock_tool.py
      manuals/clock.md
      pyproject.toml

    $ cd ~/path/to/my-agent
    $ uv sync                                 # the llm extra is already pinned into the dependency
    $ export ANTHROPIC_API_KEY=sk-ant-...     # credentials via the environment (see Configuring the LLM)
    $ uv run sora run
    +----------------------------------------------+
    | S-ORA -- minimal terminal interface          |
    |                                              |
    | Type a goal in plain English to delegate it. |
    | Type '/exit' or '/quit' to quit.             |
    +----------------------------------------------+
    what time is it?
    [invoking clock.get_time...]
    It's 14:32.

`sora init <dir>` scaffolds a minimal, immediately-runnable example agent — `agent.yaml`, a
hand-authored `manuals/clock.md`, and `clock_tool.py` (a small `WorkspaceAdapter`/`Workspace`/`Tool`
trio implementing the clock tool directly, not via a real external server — there isn't one to
depend on for this). Real integrations import tools through an adapter (MCP, WoT, ...) instead of
hand-writing one — see [ADR-0003](docs/architecture/adrs/0003-adapters-not-tool-authoring.md) — `clock_tool.py` is
a deliberate, self-contained exception so the example needs nothing beyond an LLM key to run.

`sora run [config]` starts a persistent terminal session: it drives the decision cycle continuously, streams
external actions and messages as they happen, and reads terminal input as a `Message` (sender `"user"`)
for the next Observe phase — goals can be typed in at any point, not just at startup; Situate turns an
unhandled one into a new activity via _create_activity_. There's deliberately no `"> "` prompt — in a
plain line-buffered terminal it can't survive asynchronous output landing mid-line, so it would just be
misleading; the startup banner explains how to interact instead. `config` is an optional path to a
different `agent.yaml` (defaults to `agent.yaml` in the current directory, e.g. `sora run other-agent.yaml`).
Use `--verbose` to print each decision-cycle phase instead of just the conversational output:

    [cycle 1] Observe  - message from user: "what time is it?"
    [cycle 1] Situate  - created activity=ask-time from message; loaded manual: clock
    [cycle 1] Reason   - plan: invoke clock.get_time
    [cycle 1] Act      - invoked clock.get_time -> ack
    [cycle 2] Observe  - perceived signal: clock.time_reported
    [cycle 2] Reflect  - activity ask-time completed; stored to episodic memory

Type `exit` or `quit` for a clean shutdown (leaves joined workspaces, closes any MCP subprocess);
Ctrl-D (EOF) does the same. `--task "..."` (or `--task-file path`) submits an initial goal at
startup, before you'd type anything yourself — useful for scripting a run non-interactively, e.g.
`sora run agent.yaml --task-file task.txt`, without needing to type it in by hand.

`--verbose` is a *display* setting, not a log level: it selects the one-line-per-event terminal view,
while the runtime separately emits finer `DEBUG` detail — the prompt behind each model call, each
operation's result, and the plan bodies themselves (as inferred, as reused from procedural memory, as
entered for a sub-goal, as re-spliced when a mechanical sub-goal fans out, and as discarded when
context-adaptation invalidates one, paired with the replacement inferred against the moved world).
`--log-file PATH` mirrors that complete trace to a file, always at full detail and independent of
what the terminal is showing:

    [cycle 4] Reason   - plan invalidated by context-adaptation for 'book the cheapest flight'
    [cycle 4] Reason   - discarded plan for activity trip (was at step 1)
    0: invoke {"tool_id": "Flights", "operation_name": "search"}
    1: invoke {"tool_id": "Flights", "operation_name": "book"}
    [cycle 6] Observe  - plan for activity trip
    0: invoke {"tool_id": "Flights", "operation_name": "search"}
    1: invoke {"tool_id": "Hotels", "operation_name": "search"}

There is no `--log-level` flag: `sora run` holds the `sora` logger at `DEBUG` and lets each handler
decide what to show, so a run recorded with `--log-file` never has to be repeated at a higher
verbosity to recover the detail.

Three more optional flags, mainly for driving an ARE scenario without a bespoke runner script (see
[Running the ARE examples](#running-the-are-examples)): `--scenario <ref>` injects a runtime
`AreSimulation` for an `are-sim` workspace/`are` transport (mutually exclusive with `--task`/
`--task-file` — the scenario delivers its own task through the `AgentUserInterface`); `--report
dotted.path` calls a `(agent, simulation) -> None` hook after the session ends, e.g. to print
custom scoring/checks; `--exit-when-idle SECONDS` auto-exits once every activity has stayed
`TERMINATED` for that long, instead of waiting on stdin — useful for a scripted/headless run.

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
      procedural:                     # optional: override ProceduralMemory's built-in prompts
        plan_prompt: my_agent.prompts.plan       # default: sora.memory.default_plan_prompt
        ground_prompt: my_agent.prompts.ground   # default: sora.memory.default_ground_prompt
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
matter to another (or a `blocked`) activity, so its retention/eviction is owned by the blocked-state
machinery's fixed retention cap, not a per-cycle prune — satisfying a wait never evicts a signal
early. As a temporary fallback the runtime auto-focuses a workspace's tools on `_join_`, so an agent
perceives its joined tools without a `focus` step. The intended path is still intentional focusing —
an external action (one per cycle, dispatched at Act) that a richer strategy emits as a `_focus_`
plan step (and `_unfocus_` to narrow observation cost); that override is unaffected.

#### Customizing the planning/grounding prompts

`agent.yaml`'s `procedural:` block (above) points at your own `PlanPrompt`/`GroundPrompt`
callables — each fully *replaces* the corresponding built-in (`default_plan_prompt` /
`default_ground_prompt`), it doesn't patch pieces of it; write one to change wording, tone, or the
cost/quality tradeoff of planning and grounding. For example, the built-in prompt's default
guidance for a `send` step reporting a not-yet-known result is a `$decide` reference — a natural
sentence, but it costs one extra `ProceduralMemory.ground()` model call at run time (see
`PlanPrompt` in the API Sketch). A stricter prompt can trade that phrasing for a free, mechanical
`$from` copy:

    # my_agent/prompts.py
    from sora.memory import PLAN_SYSTEM_PROMPT, render_tools

    def cheap_plan_prompt(activity, tools, observed, messages):
        system = PLAN_SYSTEM_PROMPT + (
            "\nPrefer a bare $from copy over $decide phrasing for send content, even if it reads "
            "less like a sentence — minimizing model calls matters more than prose here."
        )
        user = f"Goal: {activity.goal}\n\nAvailable tools:\n{render_tools(tools)}"
        return system, user

    # my_agent/agent.yaml
      procedural:
        plan_prompt: my_agent.prompts.cheap_plan_prompt

`ground_prompt` follows the same shape (`GroundPrompt` in the API Sketch), for customizing the
grounding escalation itself rather than what the plan asks it to do.

#### Driving an agent programmatically

`sora run` is one way to run an `Agent` — the terminal CLI. Embedding S-ORA in your own program
(a test harness, an evaluation runner, a service) instead means calling `build_agent()` and
`Agent.run()`/`stop()` directly, without `TerminalSession` at all — `examples/are/mcp/email_calendar/run.py`
is a runnable reference for that shape: build the agent, `transport.submit()` an initial `Message`
(what `sora run --task` does for you at the CLI), drive `agent.run()` as a background task, poll for
the condition you care about (an activity reaching `TERMINATED`, a timeout), then `await agent.stop()`
and cancel/await the task in a `finally` for teardown.

#### Connecting to an MCP server: remote vs. local

An `mcp` (or `are-mcp`) workspace connects over whichever transport its entry describes — the runtime
does **not** have to deploy the server itself:

    workspaces:
      # Remote: connect to an already-running server (nothing is spawned). `address` is the URL;
      # SSE is the default, or add `transport: streamable-http`.
      - origin: {adapter: mcp, address: "http://localhost:8080/sse"}
        workspace_id: remote-tools

      # Local: the adapter spawns and owns a stdio subprocess. Give it a `command` (+ `args`);
      # `address` is then just a nominal label. `mcp-server-time` is the official MCP project's
      # reference time server (github.com/modelcontextprotocol/servers/tree/main/src/time).
      - origin: {adapter: mcp, address: "stdio:time"}
        workspace_id: time
        command: uvx
        args: ["mcp-server-time"]

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

### Running the ARE examples

Two runnable showcases drive S-ORA against Meta's [Agents Research Environments](https://github.com/facebookresearch/meta-agents-research-environments) (ARE). Both need a live model and the ARE package, so they live outside the pytest suite and share the same one-time setup — install the `are` dependency group and provide a key:

    $ uv sync --all-extras --group are
    $ export ANTHROPIC_API_KEY=sk-ant-...     # or a .gitignored .env, as above

**MCP path — static snapshot** (`examples/are/mcp/email_calendar/`). S-ORA's `AreMcpWorkspaceAdapter` connects over ARE's MCP server, which serves a scenario's *initial* app state; this fits the single-shot plan→ground→act loop:

    $ uv run python -m examples.are.mcp.email_calendar.run
    # or the same scenario through the CLI:
    $ uv run sora run examples/are/mcp/email_calendar/agent.yaml \
        --task-file examples/are/mcp/email_calendar/task.txt --verbose

**In-process path — dynamic timeline** (`examples/are/sim/email_calendar/`). To exercise a scenario's *event timeline* — mid-run email injections, follow-ups, task delivery — S-ORA runs the ARE `Environment` directly. The scenario is a **per-run input on the command line**, not config: a dotted `Scenario` subclass or a Gaia2 `.json` file, passed via `sora run`'s `--scenario` flag, which turns it into the `simulation` object `build_agent(config, simulation=...)` injects. This runs through the same `sora run`/`TerminalSession` trace as any other agent — `TerminalSession` only needs a transport with an outbound `.sent` log, not specifically `InProcessTransport`:

    $ uv run sora run examples/are/sim/email_calendar/agent.yaml \
        --scenario examples.are.sim.email_calendar.scenario.EmailScheduleScenario --verbose    # watch it interactively

    # headless + scored: auto-exit once quiescent, then print the agent's outcome + ARE's own
    # scenario.validate() score via a plain (agent, simulation) -> None hook (examples/are/sim/email_calendar/report.py)
    $ uv run sora run examples/are/sim/email_calendar/agent.yaml \
        --scenario examples.are.sim.email_calendar.scenario.EmailScheduleScenario \
        --report examples.are.sim.email_calendar.report.report --exit-when-idle 8

**Scenarios you can run.** The in-process path is scenario-agnostic — `--scenario` is required (there is no default) and accepts any of three forms:

- **`examples.are.sim.email_calendar.scenario.EmailScheduleScenario`** — the bundled illustrative scenario. A *dynamic* scenario: Alice emails to schedule a Monday team sync, then a follow-up email mid-run moves it to Tuesday, surfacing as a `state_changed` signal that drives a replan.
- **A Gaia2 `.json` benchmark scenario** — any scenario file from Meta's Gaia2 benchmark (distributed with [ARE](https://github.com/facebookresearch/meta-agents-research-environments)), loaded through ARE's benchmark scenario loader. These are not vendored in this repo; point at your own copy.
- **A dotted `Scenario` subclass you author** — subclass ARE's `Scenario` (see `examples/are/sim/email_calendar/scenario.py` for the template) and pass its dotted path.

Separately, the **MCP path** runs one seeded static scenario, `examples/are/mcp/email_calendar` (a 30-minute team sync from an inbox email); it is fixed by that example's `agent.yaml`/`task.txt` rather than selected with `--scenario`.

The MCP path's standalone `examples/are/mcp/email_calendar/run.py` drives the decision cycle until the activity terminates, prints the trajectory, and prints the runtime's own INFO trace via plain `logging.basicConfig`; raise or lower it with `LOGLEVEL` (e.g. `LOGLEVEL=WARNING`) — it exists primarily as the reference example for [driving an agent programmatically](#driving-an-agent-programmatically), without `TerminalSession`. The in-process dynamic path runs entirely through `sora run` (above), so its trace/trajectory/footer are the same colored `[cycle N] Phase - ...` output (or terse `[invoking ...]` cues) as any other `sora run` session, controlled by `--verbose`/`--color`/`--no-color` (and `--log-file` for the full-detail file mirror), not `LOGLEVEL`. For how the two adapter paths differ and why the dynamic path exists, see [EXAMPLES.md](EXAMPLES.md#running-dynamic-scenarios-in-process) and the [ARE dynamic scenarios design note](docs/architecture/notes/are-dynamic-scenarios.md).

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
    class SignalWait:         # what a `blocked` activity waits for — see Activity.blocked_on
        signal_name: str      # matched mechanically in Observe (name equality, plus source when scoped)
        source: str | None = None  # tool id the signal must come from; None matches any source
        # A future variant will wait on an observable property reaching a state (README's "signal *or*
        # property update") — deferred; hence Activity.blocked_on is named generally, not blocked_on_signal.

    @dataclass(frozen=True)
    class InputWait:          # the second blocked_on variant SignalWait foresaw — see Activity.blocked_on
        prompt: str | None = None  # optional human-facing note on what's awaited
        # A `blocked` activity awaiting the user's next instruction, set by the interrupt handler when a
        # hard interrupt (a user stop) pauses it; cleared in Observe when a user Message arrives (not a
        # tool signal to match — the awaited stimulus is inbound user input). See ADR-0020.

    @dataclass(frozen=True)
    class InterruptRequest:   # a pending hard interrupt, recorded on DecisionCycle by interrupt()
        signal: Signal        # the "why" the interrupt handler reads to route each targeted activity
        target: str | None = None  # activity id to preempt; None = agent-wide (every schedulable activity)
        # A pushed signal only becomes an InterruptRequest through an InterruptPolicy; an ordinary signal
        # that merely matches a wait resumes cooperatively in Observe, never here. See ADR-0020.

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
        next_action: str      # an ExternalAction.name ("invoke", "send", "focus", ...) or a WAIT / SUBGOAL sentinel
        params: dict          # the action's own argument bag, passed through opaquely and destructured by
        #                       the action — shape is per-action (send -> {to, content}, focus -> {tool_id}).
        #                       `invoke` mixes routing (tool_id/operation_name, under the TOOL_ID/OPERATION_NAME
        #                       keys) with the operation's args; Act's bind splits them. Build one via invoke_step().
        #                       next_action="subgoal" is a sentinel (like WAIT): params = {"goal": <str>, "mode":
        #                       "mechanical"|"deliberative"}. "mechanical" also carries {"in": <collection ref>,
        #                       "as": <element name>, "template": <Step>} — Reason fans out len(collection) copies
        #                       of the template, each with the element bound in the named-binding namespace
        #                       (read via {"$bind": <as>}, optional "path"), no model call — count = len(collection),
        #                       not a model guess. "deliberative" re-fires _infer_ (or retrieve) mid-plan and runs the
        #                       resulting sub-plan as a pushed frame. Not an external action. ADR-0022.

    @dataclass(frozen=True)
    class Plan:                # multi-step, goal-indexed, reusable — the thing ProceduralMemory stores
        id: str                  # stable identity for storage/reuse
        goal: str                  # matched against future activities' goals — the retrieval key
        context_guard: list[dict] = []  # AgentSpeak-style guard clauses, evaluated once at plan entry (Reason), before
        #                            the body advances. Each is {"bind": <name>, "query": {...}} (a memory retrieval
        #                            binding <name> in the named-binding namespace; applicability = query non-empty),
        #                            a bare predicate dict (a pure check, binds nothing), or {"$decide": "..."} (an
        #                            escalation, only for genuine judgment). Retrieval, not unification. A body
        #                            param reads a bound value via {"$bind": <name>} (optional "path") — the
        #                            named-binding sibling of $from; that is how a param binds from long-term
        #                            memory, while $from stays history-only. Naming an unbound guard name -> plan inapplicable
        #                            (a mechanical unbindable flag, not a hallucinated literal). ADR-0022.
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
    class PendingInference:   # tracks one in-flight infer()/ground() — lives on Activity, mutually exclusive
        id: str                 #   with pending_operation (a cycle emits one action). Correlates to what
        kind: str               #   _infer_/_ground_ pushed into inference_sink. kind: "plan" (infer ->
        requested_at: float     #   Activity.plan), "subgoal" (a mid-plan infer -> a sub-plan PUSHED as a frame,
        #                         parent kept — ADR-0022), "ground" (ground -> the pending step's params), or
        #                         "revalidate" (a plan-validity verdict -> reconsider_verdict, ADR-0024). An
        #                         interrupt handler that re-routes the activity clears/replaces this; a result
        #                         whose id no longer matches the live pending_inference is discarded on resolve
        #                         (the stale-inference guard, mirroring pending_operation's late-ack). ADR-0021.
        out: str | None = None  # target binding for kind="select" ($decide filter — ADR-0023); None otherwise
        baseline: object | None = None  # the fire-time perception signature (plan/subgoal: the infer-time
        #                         world; revalidate: the world it was checked against), moved onto
        #                         Activity.reconsider_baseline on resolve so the context-adaptation gate
        #                         baselines against that fire-time world — a change landing mid-flight then
        #                         earns its own reconsideration rather than being absorbed (ADR-0024)

    @dataclass(frozen=True)
    class InferenceResult:    # what infer()/ground() resolve to — arrives async via inference_sink, never a
        id: str                 #   Percept (deliberation output, not observed state — ADR-0019/0021).
        value: Plan | dict | bool  # correlates to PendingInference.id; a Plan (kind="plan"), grounded params
        #                         (kind="ground"), or a bool verdict (kind="revalidate" — ADR-0024).
        #                         DefaultObserveStrategy applies it on resolve.

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
    @dataclass(frozen=True)
    class Percept:
        source: str            # tool id
        payload: Any            # an ObservableProperty (in WorkingMemory.properties) or a Signal (in
        observed_at: float      #   .signals) — the store discriminates, so there is no `kind` field.
        # genuine environment stimuli only: an invoked operation's own result is not a Percept (see
        # Activity.pending_operation/last_operation), and neither are agent messages (see .messages).

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
            # Optional synchronous screen, invoked on every push *before* enqueue. The cycle wires this on
            # its signal_sink so an InterruptPolicy can turn a just-pushed signal into a hard interrupt the
            # instant it arrives (before the once-per-cycle drain). Left None on result_sink and elsewhere.
            self.on_push: Callable[[str, T], None] | None = None
        def push(self, source: str, item: T) -> None: ...   # calls on_push (if set), then enqueues
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
        completion_signal: str | None = None  # the signal marking a long-running op's real completion
                             #   (its ack means only "accepted") — author-owned (a native description
                             #   can't express it), so merge_manuals keeps it; drives the blocked wait
        returns: dict | None = None    # JSON-Schema-shaped result shape (array/object/leaf), the
                             #   counterpart to `parameters` — adapter-synthesized from a native return
                             #   type/description. Lets a planner author a resolvable `$from` path into
                             #   a prior result instead of guessing its shape; None if undeterminable
        side_effecting: bool | None = None  # does invoking this MUTATE THE ENVIRONMENT the plan reasons
                             #   about (vs leaving it unchanged)? Outward-facing is a different question: a
                             #   report to the principal on the agent's own channel (runtime-io's
                             #   send_message_to_user) is False. Fills the before_writes reconsideration
                             #   checkpoint (ADR-0024). Adapter-owned like `returns`: from MCP readOnlyHint /
                             #   ARE write_operation / inference. None = UNKNOWN -> treated as a write
                             #   (conservative: reconsider before it)

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
        def parse(self, raw: str) -> Manual: ...   # Manual envelope; also lifts an optional per-section
                                                   #   ```yaml interface block (names + required; plus an
                                                   #   operation's completes_on completion signal) if present

    class ManualParseError(ValueError): ...   # e.g. a manual with no derivable id
    class MarkdownManualParser:               # the default ManualParser (clean Markdown format)
        def parse(self, raw: str) -> Manual: ...   # yields a Manual envelope (raw_text; specs empty)

    class ManualMergeError(ValueError): ...   # ids mismatch, or authored interface diverges from adapter's
    # Reconcile a Manual's two provenance channels by id (ADR-0018): adapter owns the structured specs
    # + JSON Schema, authored Markdown owns raw_text/description; validates a declared authored interface.
    def merge_manuals(adapter: Manual, authored: Manual) -> Manual: ...

    class ManualSource(Protocol):     # resolve a Manual.id to a hand-authored Manual (adapter pairing seam)
        async def get(self, manual_id: str) -> Manual | None: ...
    class DirectoryManualSource:      # the default ManualSource: *.md in a dir, indexed by parsed Manual.id
        def __init__(self, root, parser: ManualParser | None = None) -> None: ...
        async def get(self, manual_id: str) -> Manual | None: ...

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
        plan: Plan | None = None    # the ACTIVE frame's plan — once set, Reason advances it instead of (re)planning
        step_index: int = 0         # the active frame's cursor
        parent_frames: list[tuple[Plan, int]] = []  # suspended parent frames — the intention stack (ADR-0022);
        #                             empty for a flat plan. A deliberative subgoal pushes (plan, step_index) here
        #                             and makes the sub-plan the active (plan, step_index); when the active frame
        #                             is exhausted and this is non-empty, Reason pops the top back into
        #                             (plan, step_index) and the parent resumes. Generalizes plan/step_index into a
        #                             stack rather than adding an intention type (ADR-0002).
        pending_operation: PendingOperation | None = None  # set while RUNNING on an invoke; cleared on resolve
        pending_inference: PendingInference | None = None  # set while RUNNING on an off-cycle infer()/ground();
        #                                                    mutually exclusive with pending_operation; resolved
        #                                                    in Observe via inference_sink (ADR-0021)
        grounded_params: dict | None = None                # a resolved _ground_ escalation's concrete params,
        #                                                    consumed by Reason's next pass to emit the step
        last_operation: OperationAck | None = None          # most recently resolved result, for Reason to read
        reconsider_baseline: object | None = None           # ADR-0024 context-adaptation: a compact perception
        #                                                     signature captured when the plan started executing;
        #                                                     the before_writes checkpoint diffs it vs the live world
        reconsider_verdict: bool | None = None              # a resolved revalidation verdict parked for Reason's next
        #                                                     pass (True -> proceed + re-baseline; False -> replan)
        blocked_on: SignalWait | InputWait | None = None    # set while BLOCKED; what's awaited before READY —
        #                                                     a SignalWait (tool completion signal; set by
        #                                                     _suspend_, cleared by _resume_ — see below) or an
        #                                                     InputWait (user's next instruction after a hard
        #                                                     interrupt; cleared in Observe on a user Message)
        history: list[CompletedOperation] = []              # append-only trace of resolved ops — a later step
        #                                                     grounds param references against it (see Reason
        #                                                     grounding); last_operation keeps only the newest
        # context is exclusively for strategy-author data — the runtime itself never writes into it,
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
        def register_data_op(self, action: InternalAction) -> None: ...   # plan-composable data-ops (ADR-0023)
        def data_op(self, name: str) -> InternalAction: ...               #   dispatched by Reason, not Act
        def is_data_op(self, name: str) -> bool: ...

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
            # drop the tool's now-stale property snapshot (signals stay — their own store, untouched)
            for key in [k for k in cycle.working.properties if k[0] == tool_id]:
                del cycle.working.properties[key]
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
                # Temporary fallback: auto-focus every joined tool so perception doesn't hinge on a
                # model `focus` step (an unfocused tool's state isn't observed). Goal is intentional,
                # model-driven focus; `_focus_`/`_unfocus_` stay as that path / a manual override.
                await tool.focus(cycle.signal_sink)
                cycle.working.focused_tools[tool.id] = tool
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
            tool_ids = kwargs["tool_ids"]         # prune observable-property percepts to relevant tools;
            cycle.working.drop_properties(lambda source: source in tool_ids)  # signals: their own
            # store, never touched here — owned by the blocked-state machinery's retention cap instead.

    class SuspendAction:               # predefined internal action: _suspend_
        name = "suspend"
        async def execute(self, cycle: DecisionCycle, **kwargs) -> None:
            activity = cycle.working.activities[kwargs["activity_id"]]  # READY -> BLOCKED, recording the
            activity.state = ActivityState.BLOCKED                      # signal it waits for. The decision
            activity.blocked_on = kwargs["wait"]  # (a SignalWait)      # to suspend is Observe's (a long-
            #  running op declared a completion signal not yet observed); this action is just the flip.

    class ResumeAction:                # predefined internal action: _resume_
        name = "resume"
        async def execute(self, cycle: DecisionCycle, **kwargs) -> None:
            activity = cycle.working.activities[kwargs["activity_id"]]  # BLOCKED -> READY once the awaited
            activity.state = ActivityState.READY                       # signal was observed (the caller
            activity.blocked_on = None       # matched it — the signal itself stays in working memory)

    # The two LLM calls are internal actions too — dispatched off-cycle exactly like _invoke_ (set a
    # pending marker, go RUNNING, create_task, return at once; the cycle never blocks). Reason fires them
    # inline when it needs a plan/param it can't produce mechanically; the result resolves a cycle or more
    # later via inference_sink (ADR-0021). ProceduralMemory still owns the model handle and the prompt/parse.
    class InferAction:                 # predefined internal action: _infer_ — the async plan model call
        name = "infer"
        async def execute(self, cycle: DecisionCycle, **kwargs) -> None:
            activity = cycle.working.activities[kwargs["activity_id"]]
            inf_id = new_id()
            activity.pending_inference = PendingInference(id=inf_id, kind="plan", requested_at=now())
            activity.state = ActivityState.RUNNING   # off-cycle, like _invoke_ — immediate, never blocks
            asyncio.create_task(self._call(cycle, activity, inf_id,
                                           kwargs["tools"], kwargs.get("observed")))  # tools: id->Manual
        async def _call(self, cycle, activity, inf_id, catalog, observed) -> None:
            plan = await cycle.procedural.infer(activity, catalog, observed)          # the LLMClient call
            cycle.inference_sink.push(inf_id, InferenceResult(id=inf_id, value=plan))

    class GroundAction:                # predefined internal action: _ground_ — the async param-grounding escalation
        name = "ground"
        async def execute(self, cycle: DecisionCycle, **kwargs) -> None:
            activity = cycle.working.activities[kwargs["activity_id"]]
            inf_id = new_id()
            activity.pending_inference = PendingInference(id=inf_id, kind="ground", requested_at=now())
            activity.state = ActivityState.RUNNING
            asyncio.create_task(self._call(cycle, activity, inf_id, kwargs))
        async def _call(self, cycle, activity, inf_id, kw) -> None:
            params = await cycle.procedural.ground(activity, kw["operation_name"], kw.get("manual"),
                                                   kw["partial_params"], kw.get("observed"))
            cycle.inference_sink.push(inf_id, InferenceResult(id=inf_id, value=params))

    class RevalidateAction:                 # predefined internal action: _revalidate_ — async plan-validity re-check
        name = "revalidate"            #   the context-adaptation reconsideration model call (ADR-0024)
        async def execute(self, cycle: DecisionCycle, **kwargs) -> None:
            activity = cycle.working.activities[kwargs["activity_id"]]
            inf_id = new_id()
            activity.pending_inference = PendingInference(id=inf_id, kind="revalidate", requested_at=now())
            activity.state = ActivityState.RUNNING
            asyncio.create_task(self._call(cycle, activity, inf_id, kwargs))
        async def _call(self, cycle, activity, inf_id, kw) -> None:
            valid = await cycle.procedural.revalidate(activity, kw.get("observed"), kw.get("messages"))
            cycle.inference_sink.push(inf_id, InferenceResult(id=inf_id, value=valid))

    # Data-ops (ADR-0023): the plan's composable data-processing layer. Each is an InternalAction in
    # ActionRegistry's dedicated data-op bucket (not _internal), so only these — never a runtime-only
    # lever — are dispatchable from a plan step, and the collection-`filter` never collides with the
    # perception-prune `FilterPerceptionsAction`. A data-op reads a run-time collection Reason already
    # resolved (from history via $from, a prior binding via $bind, or a literal) and writes a named
    # binding into Activity.bindings[out], which a later step reads via {"$bind": "<name>"}. The
    # pipeline is imperative — one op per step (a declarative $foreach/$select binding spec stays
    # rejected, ADR-0022 (a)). Mechanical ops run inline; only FilterAction's $decide predicate
    # escalates, to one off-cycle ProceduralMemory.select over the whole collection (kind="select",
    # landing in bindings[out] via Observe — like _ground_). The vocabulary is a tunable coverage
    # decision; developers register their own richer transforms via register_data_op.
    class FilterAction:                # data-op: _filter_ — keep matching elements
        name = "filter"                # where: {"path","op","value"} (eq/ne/lt/le/gt/ge/between/in) or {"$decide": ...}
        async def execute(self, cycle: DecisionCycle, **kwargs) -> None: ...
    class DistinctAction:              # data-op: _distinct_ — dedupe (optionally by a key path)
        name = "distinct"
        async def execute(self, cycle: DecisionCycle, **kwargs) -> None: ...
    class SortAction:                  # data-op: _sort_ — order by a key path (desc?)
        name = "sort"
        async def execute(self, cycle: DecisionCycle, **kwargs) -> None: ...
    class TakeAction:                  # data-op: _take_ — first n elements (Spark/FP take; = SQL LIMIT)
        name = "take"
        async def execute(self, cycle: DecisionCycle, **kwargs) -> None: ...
    class CollectAction:               # data-op: _collect_ — gather a fan-out's per-op history results (MapReduce gather)
        name = "collect"
        async def execute(self, cycle: DecisionCycle, **kwargs) -> None: ...
    class ReduceAction:                # data-op: _reduce_ — aggregate to a scalar (sum/min/max/count/mean)
        name = "reduce"
        async def execute(self, cycle: DecisionCycle, **kwargs) -> None: ...

    def default_action_registry() -> ActionRegistry:   # six external + eight internal + six data-ops
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

    class MeteredLLMClient:            # transparent decorator bootstrap wraps every client in
        """Times each round-trip and logs a `sora.llm` cue carrying the elapsed seconds. Not a
        breach of the non-ownership contract — that forbids the *client itself* growing timing;
        this instruments one from the outside, so the concrete client stays a bare round-trip. Each
        `done`/`usage` cue is tagged with `current_inference_id` (a ContextVar the _infer_/_ground_
        action sets, task-local per background call) so the meter can attribute a round-trip to the
        off-cycle inference that drove it."""
        def __init__(self, inner: LLMClient) -> None: ...
        async def complete(self, *, system: str, prompt: str) -> str: ...

    @dataclass(frozen=True)
    class LLMUsage:                    # provider-native token accounting for one round-trip
        """Surfaced by the *concrete* client (opt-in via its `instrument:` config), not the outer
        timing decorator: token counts live only in the provider's response, where no wrapper can
        see them — the one thing that can't be metered from outside. `output_tokens` folds in
        thinking, and adaptive thinking doesn't return the thinking as countable text, so the
        deliberation share is *estimated* as output the answer doesn't explain: output_tokens minus
        the answer's tokens (from the measured `answer_chars`). `log_llm_usage(usage)` emits it as a
        `sora.llm` usage record, paired with (distinct from) the timing `done` cue."""
        input_tokens: int; output_tokens: int; answer_chars: int
        @property
        def thinking_tokens(self) -> int: ...   # est: output the answer doesn't account for
        @property
        def thinking_share(self) -> float: ...

    class LLMMeter(logging.Handler):   # tallies the `sora.llm` cues: calls + in-model seconds, and
        """(when the client is instrumented) token totals + thinking share. A run surface
        (TerminalSession, an example runner) attaches it to the `sora` logger and calls summary() at
        the end — no reference to the client (which bootstrap hands off) needed. Token tally is
        opt-in: an uninstrumented client emits no usage record and summary() reports timing only.
        A `discarded` cue (from `log_llm_discarded`, emitted by Observe when an off-cycle inference's
        result is invalidated/superseded — ADR-0021) folds that call's already-metered cost into a
        `wasted_*` bucket: the call ran to completion and is real, billed cost (kept in the grand
        totals), but did no useful work, so summary() shows used vs. discarded side by side. With no
        discards the summary is byte-for-byte the terse pre-existing line."""
        def summary(self, wall_seconds: float | None = None) -> str: ...

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
        # Environment stimuli, stored by their opposite lifecycles: properties are a replace-by-
        # (source, name) snapshot (one entry, last value wins); signals are an append log — a matched
        # signal is never evicted just for satisfying a wait, only a fixed retention cap bounds it.
        properties: dict[tuple[str, str], Percept]
        signals: list[Percept]
        messages: list[Message]        # inbound agent-to-agent communication — kept distinct
        messages_cursor: int           # count already routed (goal) or claimed (resume); a consumed-
                                        # cursor over the append-only log so each message is handled once
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

    @dataclass(frozen=True)
    class PerceptSnapshot:   # the agent's currently-observed world state, bundled for a planning/
        #                      grounding prompt. An empty snapshot (PerceptSnapshot()) means nothing
        #                      observed yet.
        properties: list[Percept] = field(default_factory=list)
        signals: list[Percept] = field(default_factory=list)

    class PlanPrompt(Protocol):   # builds infer()'s (system, user) prompt from (activity, tools, observed, messages)
        def __call__(self, activity: Activity, tools: dict[str, Manual],
                     observed: PerceptSnapshot, messages: list[Message]) -> tuple[str, str]: ...
        #   default_plan_prompt is the built-in one; PLAN_SYSTEM_PROMPT / render_tools /
        #   render_properties / render_signals / render_history / render_messages are reusable pieces.
        #   messages are recent user instructions (a follow-up after a stop, a mid-task correction),
        #   ambient context distinct from the goal string; history lets a replan skip a done step.
        #   The response contract ({"context_guard":[...], "steps":[...]}) stays fixed — customize the
        #   *prompt*, not the parse. PLAN_SYSTEM_PROMPT also tells the model to emit a *reference* —
        #   {"$from": "<op>", "path": "<dotted path>"} or {"$decide": "..."} — for a param whose
        #   value depends on an earlier step's result, never a made-up literal, and to reuse an
        #   already-observed property/signal value directly instead of re-discovering it. It further
        #   tells the model to author `context_guard` clauses for values that come from long-term
        #   memory (bound by name, read via {"$bind": name}, not $from), and to emit a "subgoal" step — mechanical for a uniform
        #   map over a collection, deliberative for an open continuation — instead of guessing an
        #   iteration count inline (ADR-0022).

    class GroundPrompt(Protocol):   # builds ground()'s (system, user) prompt — grounding's counterpart
        def __call__(self, activity: Activity, operation_name: str, manual: Manual | None,
                     partial_params: dict, observed: PerceptSnapshot) -> tuple[str, str]: ...
        #   default_ground_prompt is the built-in one; GROUND_SYSTEM_PROMPT / render_history /
        #   render_properties / render_signals are the reusable pieces. Response contract is fixed
        #   ({"params": {...}}).

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
            The cheap path: skips infer() entirely when it hits. A deliberative sub-goal calls this
            first too, keyed by the sub-goal's goal — a sub-plan library, not just a top-level one
            (ADR-0022)."""
        async def infer(self, activity: Activity, tools: dict[str, Manual],
                        observed: PerceptSnapshot | None = None,
                        messages: list[Message] | None = None) -> Plan:
            """Produces a new multi-step Plan when no cached one fits — the model path: one LLMClient
            call producing a whole sequence of Steps at once. This is procedural memory querying its
            'implicit knowledge encoded in LLM weights'. `tools` (id -> its Manual) is the planning
            catalog, passed in by the caller that holds the live registry (a memory module never
            reaches into the environment); `observed` is the caller's current properties/signals
            snapshot (omittable — defaults to none observed), so planning isn't blind to already-
            known world state. Converts the model's JSON answer into Plan/Step — including the
            plan's `context_guard` clauses and any `subgoal` steps — (the anti-corruption boundary);
            malformed output raises ValueError. No llm -> raises. Also fired mid-plan for a
            deliberative sub-goal (the sub-goal's goal as the planning target); the resulting
            sub-plan is pushed as a frame, not a replacement for the parent (ADR-0022)."""
        async def ground(self, activity: Activity, operation_name: str, manual: Manual | None,
                         partial_params: dict, observed: PerceptSnapshot | None = None) -> dict:
            """The Reason-phase grounding *escalation*: decide an operation's concrete params from the
            execution context when a reference can't be resolved mechanically. One LLMClient call over
            the operation schema + partial params + the currently observed properties/signals
            (`observed`, omittable) + the activity's history; parses {"params": {...}}
            (anti-corruption); no llm -> raises. Packaged here (like infer) because procedural memory
            owns the model handle; grounding a step is really an Act-adjacent reasoning act — see
            ADR-0017. (The mechanical reference resolver lives in Reason, not here.)"""
        async def revalidate(self, activity: Activity, observed: PerceptSnapshot | None = None,
                        messages: list[Message] | None = None) -> bool:
            """The context-adaptation relevance judgment (ADR-0024): is the activity's in-progress plan
            still VALID given the current world? One LLMClient call over the goal + the operations already
            executed + the plan's remaining steps + observed properties/signals + recent messages; parses
            {"valid": bool} (fail-soft to True). Same model seam as infer/ground; no llm -> raises. A False
            verdict tells Reason to re-infer — general (no domain-authored predicate), so the agent's own
            writes don't spuriously invalidate the plan. History is what makes that last part decidable:
            at a late checkpoint the remaining tail is one step and the goal's work is all behind it."""
        async def store(self, plan: Plan) -> None:
            """Persists a Plan so future retrieve() calls for similar goals can reuse it. NOT called by
            the default ReflectStrategy: auto-caching a completed plan and replaying it verbatim is
            unsound (a corrected/observation-coupled plan isn't reusable), so plan caching is disabled
            until reusable procedures are distilled from episodes — this stays available for that."""

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
        field an earlier phase already filled short-circuits the later phase (a cached plan skips Reason,
        a resolved step skips Act's bind). One model call may fill several fields at once (fusion) — a
        narrow, opt-in use, not the goal (ADR-0011/0021). Lives only for the duration of one tick() call —
        nothing persists across cycles, so there's no cache to key or invalidate."""
        activity: Activity | None = None
        step: Step | None = None      # this cycle's concrete decision — not the whole (possibly multi-step) Plan
        invocation: OperationInvocation | None = None

    class ObserveStrategy(Protocol):
        async def observe(self, cycle: DecisionCycle) -> TickResult:
            """Mutates cycle.working (properties, signals, messages) as a side effect — same as the
            default below. Default: mechanical, no model call, returns an empty TickResult(). An LLM-backed
            Observe is for interpreting raw perception itself (e.g., describing a camera snapshot) — and,
            like every model call, runs off-cycle as an async internal action whose result lands as a
            percept a later cycle (ADR-0021), never blocking Observe or deciding the cycle."""

    class ReflectStrategy(Protocol):
        async def reflect(self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle,
                           result: TickResult) -> TickResult:
            """Decides whether this activity just completed or failed — deterministic or model-backed,
            depending on the application — and if so, summarizes and stores to episodic memory. (The
            default does NOT auto-cache the completed plan to procedural memory — replaying a stored
            plan verbatim is unsound; distilling reusable procedures from episodes is future work.)
            The completion judgment is
            synchronous — it must land before Situate selects, so a just-completed activity is never
            re-selected the same cycle — while the summarize/store side effects are dispatched
            asynchronously and never block the cycle; several activities may terminate in the same
            cycle. Passes `result` through, optionally adding to it. Default: performs the completion
            check and the store-on-success, leaves TickResult's other fields untouched. `cycle` is
            what makes these memory calls possible at all — previously missing from this Protocol
            despite the calls it was already documented as making."""

        def failed(self, activity: Activity) -> bool:
            """This strategy's own judgment of whether `activity` has failed — a judgment call, not
            a fact recorded on Activity itself, since a different ReflectStrategy may define failure
            differently than the default's "resolved operation, not ok" rule. Callers outside the
            cycle (a reporting hook, a test) go through whichever strategy is actually configured
            (e.g. `agent.cycle.strategies.reflect.failed(activity)`) rather than re-deriving the rule."""

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
            -> Reason -> Act), running after this cycle's percepts and messages are in working memory. May
            additionally fill in step/invocation, short-circuiting Reason/Act (those forward-fill gates
            remain; only Situate's own activity gate is removed). Fusing selection *and* planning into one
            model call is possible but re-serializes multi-activity concurrency (no activity is selected
            until it returns), so it belongs to a synchronous simple-mode configuration, not the async
            default — see ADR-0021."""

    class ActivitySelectionStrategy(Protocol):   # Situate's scheduler; own pluggable sub-strategy
        async def select(self, ready: list[Activity], wm: WorkingMemory,
                          cycle: DecisionCycle) -> Activity | None:
            """Picks which ready activity progresses this cycle (empty -> None) — a scheduling
            policy, not a phase. DefaultSituateStrategy delegates its pick here so a richer scheduler
            (priority, aging, deadlines, an LLM-based one) swaps in without re-authoring Situate's
            activity-creation and wm-adjustment. `async` + `cycle` let such a policy consult memory
            or a model; the default (RoundRobinActivitySelection) consults neither."""

    class ReasonStrategy(Protocol):   # pluggable; default makes at most one (off-cycle) model call/cycle
        async def reason(self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle,
                          result: TickResult) -> TickResult:
            """Only called if result.step is still None. Typical implementation: if activity.plan is
            already set and still valid, just read activity.plan.steps[activity.step_index] and advance
            the index — no model call. Otherwise, retrieve a cached Plan via cycle.procedural.retrieve()
            or fire the _infer_ internal action — an off-cycle model call that moves the activity to
            RUNNING and yields no step this cycle; the plan lands a later cycle via inference_sink.
            Deciding when a plan counts as invalidated is entirely up to the implementation. Also
            *grounds* the step: a param whose value depends on an earlier result is a reference the
            default resolves against activity.history, escalating via the _ground_ internal action (same
            off-cycle shape) only when it can't be resolved mechanically — deciding a value is reasoning,
            so it lives here (ADR-0017/0021). On plan entry it evaluates the plan's `context_guard` —
            mechanical retrievals bind for free into the named-binding namespace, a `$decide` clause
            escalates — and a body param naming an unbound guard value makes the plan inapplicable. A
            `subgoal` step it either expands mechanically (fan out len(collection) template copies, one
            per cycle) or dispatches as a mid-plan _infer_/retrieve that pushes the sub-plan as a frame;
            an exhausted active frame with suspended parent_frames pops back to the parent (ADR-0022).
            May additionally fill in invocation, short-circuiting Act — this is where the historical
            'tool hallucination' risk lives if it does."""

    class ActStrategy(Protocol):
        async def bind(self, step: Step, manual: Manual | None, cycle: DecisionCycle,
                        result: TickResult) -> TickResult:
            """Only called if result.invocation is still None. *Parameter binding*: split an invoke
            Step's routing keys from its (by now already-grounded) params into a concrete
            OperationInvocation. Mechanistic — deciding param *values* is Reason's grounding, not
            Act's (ADR-0017). One mechanical guard still lives here (no judgment, so Act stays
            mechanistic): a *required* param that resolves to null (per the manual's
            OperationSpecification schema) is a schema violation, so the default emits no invocation
            and the invoke is skipped; without a schema, required-ness is unknowable and binding
            proceeds. Distinct from a *protocol binding* (WoT forms/security, an MCP session) — how
            the adapter's Tool reaches the instance, never surfaced here (ADR-0015). `cycle` is
            available for implementations that cache bindings rather than re-deriving one each time."""

    @dataclass(frozen=True)
    class Strategies:          # bundles the five, so DecisionCycle.__init__ doesn't take five loose params
        observe: ObserveStrategy
        reflect: ReflectStrategy
        situate: SituateStrategy
        reason: ReasonStrategy
        act: ActStrategy
    # The two hard-interrupt seams are deliberately NOT in this bundle (which is the decision chain).
    # They're separate DecisionCycle params, selected via agent.yaml strategies.interrupt /
    # strategies.interrupt_policy — mechanism and policy for preemption, not a per-phase strategy. See ADR-0020.

    class InterruptPolicy(Protocol):   # decides which pushed signals preempt — consulted at push time
        def decide(self, source: str, signal: Signal, wm: WorkingMemory) -> InterruptRequest | None:
            """Consulted synchronously the instant a signal is pushed to signal_sink (via on_push), before
            the once-per-cycle Observe drain. Return an InterruptRequest to preempt the current phase, or
            None to let the signal flow cooperatively (reacted to at the next cycle boundary). Sync because
            push is sync. A stateful policy may diff the signal against remembered state (e.g. a set of
            inbox ids) to fire only on a genuine external event and filter the agent's own writes — the
            only distinguishable-external test until read-write/efference tagging lands."""

    class NeverInterruptPolicy:        # the runtime default: no pushed signal ever preempts
        """Preserves today's cooperative signal path unchanged (drained in Observe, resumes a BLOCKED
        activity). With no runtime way yet to tell the agent's own writes from external events, preempting
        on a signal would risk a self-write loop; opting in is a deliberate, application-supplied policy."""
        def decide(self, source: str, signal: Signal, wm: WorkingMemory) -> InterruptRequest | None: ...

    # strategies.context_adaptation — how eagerly Reason re-validates an in-progress plan against new
    # perception before committing a side-effecting step (ADR-0024): a pluggable policy gating an
    # off-cycle revalidation, per-agent, selected by a level name or a dotted path. Reconsideration stays
    # cycle-owned (ADR-0022) — the policy only decides WHEN, not the act. Default before_writes.
    class ReconsiderationPolicy(Protocol):   # decides whether to run the validity check before a step
        def should_check(self, side_effecting: bool | None) -> bool: ...   # True write / False read / None unknown
    class NoneReconsideration:   # none — never reconsider on ambient percepts (blind); should_check -> False
        def should_check(self, side_effecting: bool | None) -> bool: ...
    class BeforeWrites:          # before_writes (default) — check before a side-effecting step, skip known reads
        def should_check(self, side_effecting: bool | None) -> bool: ...   # side_effecting is not False
    class BeforeEachOp:          # before_each_op — check before EVERY external step; should_check -> True
        def should_check(self, side_effecting: bool | None) -> bool: ...

    # strategies.change_gate — the cheap mechanical test the reconsideration checkpoint runs BEFORE it
    # spends a revalidation: has anything observable moved since the plan was baselined? Orthogonal to
    # context_adaptation, which decides WHICH steps are checkpoints (WHEN); the gate decides WHETHER the
    # world moved (a signature compared to Activity.reconsider_baseline; equal -> skip it, free
    # when static). A domain gate that projects perception onto only its externally-meaningful part
    # filters the agent's OWN writes here — the same efference trick a stateful InterruptPolicy uses,
    # applied to the cooperative path. Per-agent, selected by a dotted path. Default: the domain-free
    # PerceptionSignatureGate. The baseline is stored as `object` (PendingInference.baseline /
    # Activity.reconsider_baseline), so a gate may return any comparable signature.
    class ChangeGate(Protocol):     # produces a comparable signature of perception for the pre-revalidation gate
        def signature(self, wm: WorkingMemory) -> object: ...
    class PerceptionSignatureGate:  # the runtime default: domain-free (sorted property reprs + log lengths)
        """No domain knowledge — the replace-by-key property snapshot (by repr) plus the signal/message
        append-log lengths. Equal signatures mean nothing observable moved since the baseline. A self-
        caused write still moves it (a new state_changed signal, a changed property), so under this
        default the checkpoint spends one revalidation on the agent's own writes; a domain ChangeGate that
        projects to only the external surface is how an application removes that (e.g. the ARE example's
        INBOX-id gate, which self-writes to SENT / read-flags / calendar don't move). See ADR-0024."""
        def signature(self, wm: WorkingMemory) -> object: ...

    class InterruptHandler(Protocol):  # decides an interrupted activity's follow-up — the "interrupt handler"
        async def handle(self, request: InterruptRequest, wm: WorkingMemory, cycle: DecisionCycle) -> bool:
            """Runs after tick() aborts on a pending interrupt (the process-scheduling 'interrupt handler').
            Context is already saved (durable on Activity; the per-tick TickResult was discarded, immune to
            interrupt staleness per ADR-0011), so this only decides the follow-up: map each targeted activity
            onto an existing state — READY (resume, or replan by clearing plan/step_index), BLOCKED via
            InputWait (await the user's next instruction), or TERMINATED (drop) — then the
            ActivitySelectionStrategy picks next. Never abandons an in-flight *external* op: a RUNNING
            activity is left RUNNING and revisited at the next checkpoint once its ack resolves. Returns True
            once every targeted activity is routed (request discharged), False while some are still RUNNING."""

    class DefaultInterruptHandler:     # the runtime default: a user stop pauses to await input
        """Pauses each targeted, schedulable (READY) activity to a resumable point via an InputWait, so the
        agent halts current work but stays alive; a later user Message resumes it (DefaultObserveStrategy's
        _resume_on_input). A RUNNING activity (mid external op) is left to finish and routed on a later
        checkpoint, so a physical side effect always runs to completion. target=None is agent-wide. It is a
        user-stop handler, not a general router: it recognizes only the USER_STOP signal and treats any
        other interrupt as unrouted — same halt-to-await-input fallback, logged at warning level, since a
        custom InterruptPolicy is expected to ship a paired handler for its own signals (ADR-0020)."""
        async def handle(self, request: InterruptRequest, wm: WorkingMemory, cycle: DecisionCycle) -> bool: ...

    class DefaultObserveStrategy:
        """The runtime's built-in default — purely mechanical, no LLM. This is the exact logic
        previously inlined in DecisionCycle._observe()."""
        async def observe(self, cycle: DecisionCycle) -> TickResult:
            # Properties are persistent, re-observed state: one entry per (source, name), last value
            # wins — the keyed store *is* the snapshot (no side index, no growing append log).
            for tool in cycle.working.focused_tools.values():
                for prop in tool.observe():
                    cycle.working.properties[(tool.id, prop.name)] = Percept(tool.id, prop, now())
            async for source, signal in cycle.signal_sink.drain():
                cycle.working.signals.append(Percept(source, signal, now()))          # append log
            just_resolved = []
            async for op_id, ack in cycle.result_sink.drain():
                # unambiguous 1:1 match — resolved automatically to READY (manual-agnostic), never a
                # Percept, no strategy involved. The *blocked* wait is layered on top below, not fused.
                activity = next((a for a in cycle.working.activities.values()
                                  if a.pending_operation and a.pending_operation.id == op_id), None)
                if activity is not None and activity.state is ActivityState.RUNNING:  # RUNNING guard: a late
                    #  ack for an activity a hard interrupt already routed away (paused/dropped) must not
                    #  resurrect it — the op finishes, but its resolve doesn't force a spurious READY
                    invocation = activity.pending_operation.invocation
                    activity.last_operation = ack
                    activity.pending_operation = None
                    activity.state = ActivityState.READY
                    just_resolved.append((activity, invocation))
            async for inf_id, res in cycle.inference_sink.drain():
                # Off-cycle infer()/ground() result: 1:1 match to the live pending_inference, never a Percept.
                # Discarded if the activity was re-routed (id no longer matches the live one) — the same
                # stale-guard as a late ack. On match: plan -> Activity.plan; subgoal -> push a frame;
                # ground -> grounded_params for the pending step (consumed by Reason's next pass). Then
                # RUNNING -> READY.
                activity = next((a for a in cycle.working.activities.values()
                                  if a.pending_inference and a.pending_inference.id == inf_id), None)
                if activity is not None and activity.state is ActivityState.RUNNING:
                    if activity.pending_inference.kind == "plan":
                        activity.plan = res.value; activity.step_index = 0
                    elif activity.pending_inference.kind == "subgoal":
                        activity.parent_frames.append((activity.plan, activity.step_index))
                        activity.plan = res.value; activity.step_index = 0   # sub-plan runs as a frame (ADR-0022)
                    else:
                        activity.grounded_params = res.value
                    activity.pending_inference = None
                    activity.state = ActivityState.READY
            # Suspend pass: a resolved, successful op whose manual declares a completion_signal blocks
            # until that signal is observed — unless it already arrived (early signal -> stay READY,
            # don't block). Resume pass: a BLOCKED activity whose blocked_on matches an observed signal
            # returns to READY; the matched signal is left in place, not evicted. Both mechanical (name
            # equality), via the _suspend_ / _resume_ internal actions — no judgment needed.
            self._suspend_on_completion_signal(cycle, just_resolved)
            self._resume_on_signal(cycle)          # only a SignalWait — matched against observed signals
            if len(cycle.working.signals) > _SIGNAL_RETENTION:     # trim last: today's signal must
                del cycle.working.signals[:-_SIGNAL_RETENTION]     # survive to be matched above first
            async for message in cycle.communication.receive():
                cycle.working.messages.append(message)
            self._resume_on_input(cycle.working)   # an InputWait (hard-interrupt pause) is satisfied by a
            #                                        user Message, not a signal — resumed here, not above
            return TickResult()

    # DefaultReflectStrategy / DefaultSituateStrategy / DefaultActStrategy: the mechanical, no-LLM
    # defaults for the other decision-chain phases — same role as DefaultObserveStrategy (bodies
    # provisional). Named here so the sketch matches the code's default set; wired in by bootstrap
    # as sora.reflect.default / sora.situate.default / sora.act.default.

    class DefaultReasonStrategy:
        """Reason's default — the effective default Reason strategy (Reason has no *mechanical*
        default; planning is inherently the model path). Deterministic orchestration around two
        off-cycle model calls, isolated in ProceduralMemory.infer/ground and dispatched as the _infer_
        / _ground_ internal actions: the cheap path advances an existing plan's step_index (no model, no
        lookup) and grounds its params against activity.history; else, when a plan is needed, reuse a
        cached plan (procedural.retrieve) or fire _infer_ (passing the joined tools id->Manual as the
        catalog) — which moves the activity to RUNNING and returns no step this cycle; the plan lands a
        later cycle via inference_sink, and Reason then advances it. A param that can't be resolved
        mechanically fires _ground_ the same way (RUNNING, no step; the resolved params land as
        activity.grounded_params, consumed next pass to emit the concrete step). While an activity is
        RUNNING on an inference Reason simply yields no step for it — no model call blocks the cycle, so
        there is nothing to race or abandon (ADR-0021, superseding ADR-0020's mid-flight abandonment).
        Reuse is currently always a miss — the default Reflect no longer stores completed plans (verbatim
        replay is unsound), so every activity infers until reusable procedures are distilled from
        episodes; an exhausted active frame pops back to a suspended parent_frame (ADR-0022) or, with
        none, yields no step. Wired in by bootstrap as sora.reason.default."""
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
    # AreTransport (sora/adapters/are_sim.py) is a second MessageTransport: over a running ARE
    # scenario's AgentUserInterface — receive() drains unread USER messages, send() -> send_message_to_user,
    # submit() -> send_message_to_agent (an ad hoc user line surfaces on the next receive() drain, so a
    # scenario-driven session accepts typed input and a /stop resume like the in-process one).
    # Selected by transport.kind: are; shares the AreSimulation with the are-sim workspace. See EXAMPLES.md.

    # sora/cycle.py
    class DecisionCycle:
        def __init__(self, strategies: Strategies, communication: MessageTransport,
                     actions: ActionRegistry, registry: EnvironmentRegistry,
                     working: WorkingMemory, semantic: SemanticMemory,
                     procedural: ProceduralMemory, episodic: EpisodicMemory,
                     interrupt_handler: InterruptHandler | None = None,      # default DefaultInterruptHandler
                     interrupt_policy: InterruptPolicy | None = None,        # default NeverInterruptPolicy
                     reconsideration: ReconsiderationPolicy | None = None,   # default None (bootstrap: before_writes)
                     change_gate: ChangeGate | None = None):                 # default PerceptionSignatureGate
            self.registry = registry   # the shared, mutation-capable handle, passed to external
            #                            actions at dispatch; WorkingMemory holds the same instance
            #                            read-only (as EnvironmentView) for strategies to reason over.
            # The two hard-interrupt seams (ADR-0020): interrupt_handler decides an interrupted activity's
            # next state (default: a user stop pauses it to await input); interrupt_policy screens pushed
            # signals for ones that should preempt (default: none do — the cooperative path is unchanged).
            self.interrupt_handler = interrupt_handler or DefaultInterruptHandler()
            self.interrupt_policy = interrupt_policy or NeverInterruptPolicy()
            self.reconsideration = reconsideration or NoneReconsideration()  # ADR-0024; bootstrap: before_writes
            self.change_gate = change_gate or PerceptionSignatureGate()      # ADR-0024 pre-revalidation change-gate
            # These sinks live here rather than on WorkingMemory: they bridge asynchronous, off-cycle
            # events into this engine's tick()/interrupt() — not settled state. signal_sink specifically
            # has to be co-located with interrupt() below, since a pushed Signal can preempt the current
            # phase; that control-flow role, not "where it eventually lands as a percept," is why it isn't
            # a WorkingMemory field. result_sink carries invoke() acks; inference_sink carries off-cycle
            # infer()/ground() results (InferenceResult, never a Percept — ADR-0021).
            self.signal_sink: NotificationQueueSink[Signal] = NotificationQueueSink()        # tools push here via focus()
            self.result_sink: NotificationQueueSink[OperationAck] = NotificationQueueSink()  # InvokeAction pushes here — internal only
            self.inference_sink: NotificationQueueSink[InferenceResult] = NotificationQueueSink()  # _infer_/_ground_ push here — internal only
            self.signal_sink.on_push = self._screen_signal   # screen every signal at push time (below)
            self._interrupt: InterruptRequest | None = None  # a pending hard interrupt (None when idle)
            self._wake = asyncio.Event()                     # edge that wakes a waiting cycle; set with _interrupt
            ...
        async def tick(self) -> None:
            """One Observe -> Reflect -> Situate -> Reason -> Act pass, threading a TickResult through
            all five phases and calling each phase's own strategy only for whatever's still missing —
            so a field an earlier phase already filled short-circuits the later phase. Takes
            no arguments: registry/working/semantic/procedural/episodic/communication are all shared
            with Agent, constructed once and passed to both — see sora/bootstrap.py. (Dispatch uses
            self.registry — the mutation-capable handle — not working.registry, which is read-only.)
            A phase-boundary checkpoint (_preempted) after each phase aborts the tick on a pending hard
            interrupt: the disposable TickResult is dropped (no staleness, ADR-0011) and Act — the cycle's
            single external action — is never reached, so an interrupted tick commits nothing external."""
            result = await self.strategies.observe.observe(self)
            if await self._preempted(): return
            for activity in self.working.activities.values():
                result = await self.strategies.reflect.reflect(activity, self.working, self, result)
            if await self._preempted(): return
            # Situate always runs: it re-situates wm for the (possibly already-selected) activity every
            # cycle, and selects only if result.activity is still None. Unlike the step/invocation gates
            # below — genuine forward-fill short-circuits — Situate is not gated on its own field.
            ready = [a for a in self.working.activities.values() if a.state is ActivityState.READY]
            result = await self.strategies.situate.situate(ready, self.working, self, result)
            if await self._preempted(): return
            if result.activity is None:
                return               # nothing selectable this cycle — at most one action, never a mandatory one
            if result.step is None:
                # Reason fires its model calls off-cycle as _infer_/_ground_ internal actions, so it never
                # blocks here — an activity RUNNING on an inference simply yields no step this cycle (ADR-0021).
                result = await self.strategies.reason.reason(result.activity, self.working, self, result)
                if await self._preempted(): return
            if result.step is not None:
                await self._act(result.activity, result.step, result)   # bind-then-dispatch boundary

        async def _preempted(self) -> bool:
            """Phase-boundary checkpoint. No interrupt pending -> False, tick continues. Otherwise run the
            handler (routes each targeted activity onto an existing state) and return True to abort the tick.
            The request clears only once the handler reports it discharged; while a targeted activity is still
            RUNNING (external op in flight, left to finish) it stays pending, revisited next checkpoint."""
            if self._interrupt is None: return False
            if await self.interrupt_handler.handle(self._interrupt, self.working, self):
                self._interrupt = None
            return True

        def _screen_signal(self, source: str, signal: Signal) -> None:
            """signal_sink.on_push: consulted synchronously as each signal is pushed, before the cooperative
            Observe drain. If the InterruptPolicy elects to preempt, record the request and wake the cycle;
            otherwise the signal just flows to the drain as before (both paths coexist)."""
            request = self.interrupt_policy.decide(source, signal, self.working)
            if request is not None:
                self._interrupt = request; self._wake.set()

        async def wait_between_ticks(self, interval: float) -> None:
            """Interruptible idle wait between ticks: sleep up to `interval` but wake immediately if
            interrupt() (or a signal policy) fired — so a hard interrupt starts the next tick without
            waiting out the interval — instead of a bare asyncio.sleep. The edge is consumed (cleared) here."""

        async def _act(self, selected: Activity, step: Step, result: TickResult) -> None:
            """WAIT is the cycle's no-op sentinel — guarded first, before the registry lookup that
            would otherwise KeyError on it. Otherwise resolve the step's ExternalAction and let *it*
            declare whether the step needs binding (requires_binding) — only _invoke_ does, so the
            generic cycle stays uncoupled from any one action's name and a custom binding action binds
            too — then dispatch exactly one external action: the bound invocation's routing keys +
            params when present; a *non*-binding action with no invocation dispatches its raw step
            params (invoke resolves its tool through the registry, not the focus set); a *binding*
            action that produced no invocation is a deliberate skip (e.g. bind's required-null guard)
            and dispatches nothing this step — Reason has already advanced step_index, so the activity
            continues."""
            if step.next_action == "wait":
                return
            action = self.actions.external(step.next_action)
            if result.invocation is None and action.requires_binding:
                tool = self.registry.get(step.params["tool_id"])
                result = await self.strategies.act.bind(step, tool.manual, self, result)
            # dispatch result.invocation (if set) or step.params to `action` via action.execute,
            # always passing activity_id=selected.id — elided, same as the rest of Act's dispatch today
            ...
        async def interrupt(self, signal: Signal, *, target: str | None = None) -> None:
            """Raise a hard interrupt: preempt the current phase for an authoritative event (10ms target).
            Records an InterruptRequest(signal, target) and wakes the loop; the next phase-boundary
            checkpoint runs the handler and aborts the tick.
            `signal` is the "why" the handler reads; `target` names one activity, None = agent-wide. The
            one wired caller is a user stop from the CLI (/stop) — distinct from Agent.stop()/Ctrl-C
            (graceful shutdown). A cooperative signal that merely matches a wait resumes in Observe and
            never comes here; an InterruptPolicy promotes a pushed signal to this path. See ADR-0020."""

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
            each iteration ending in await self.cycle.wait_between_ticks(tick_interval) — an
            interruptible idle wait, so a hard interrupt (user stop, or a signal a policy preempts on)
            starts the next tick at once — leaving the workspaces finally. The join lives here, not in the
            synchronous bootstrap, because it is async I/O."""
        async def stop(self) -> None: ...

    # sora/cli.py — the runtime's minimal terminal interface
    class TerminalSession:
        """Streams cycle output to stdout; queues stdin as Message(sender="user", ...) — not a Percept,
        since terminal input is user communication, not environment stimuli. The reserved line `/stop` is
        the exception: it calls cycle.interrupt(Signal("user_stop", {})) directly (a hard interrupt —
        halt current work, stay alive, resume on the next instruction), distinct from Ctrl-C
        (Agent.stop(), graceful shutdown). No UI beyond this."""
        def __init__(self, agent: Agent, verbose: bool = False): ...
        async def run(self) -> None: ...

    # _Presenter (private, sora/cli.py): a logging.Handler that formats the runtime's existing
    # sora.* log records into TerminalSession's --verbose `[cycle N] Phase - ...` trace, adding no
    # new log call sites — not part of the public API.

    # _Console (private, sora/cli.py): tracks whether the terminal cursor sits at the start of a
    # line, so lines printed through it are always cleanly newline-separated — not part of the
    # public API.

    # _PresentableTransport (private, sora/cli.py): a runtime_checkable Protocol requiring a `.sent`
    # outbound log — the structural capability TerminalSession needs to stream a transport's replies,
    # satisfied by both InProcessTransport and the ARE in-process AreTransport. Not part of the
    # public API.

    # _SubmittableTransport (private, sora/cli.py): a runtime_checkable Protocol requiring a
    # `submit(Message)` method — the narrower capability of accepting ad hoc input (--task/
    # --task-file, typed stdin lines, a /stop resume), which both InProcessTransport and AreTransport
    # have. A custom presentable-but-not-submittable transport still degrades gracefully (no goal
    # prompt, /stop can't promise a resume). Not part of the public API.

    # sora/scaffold.py — `sora init`'s file generator
    def write_project(project_dir: Path) -> None:
        """Scaffolds a minimal, immediately-runnable example agent into project_dir: pyproject.toml,
        agent.yaml, manuals/clock.md, and clock_tool.py. Refuses to touch an already-existing
        target."""

    # sora/bootstrap.py — internal; developers implement protocols, they don't call this directly
    @dataclass(frozen=True)
    class AgentConfig:
        """The parsed agent.yaml `agent:` block. strategies/memory are dotted-path / URI maps
        resolved during build_agent; workspaces is the raw list (each entry: an `origin` plus
        adapter-specific keys like command/args); llm is optional (absent -> no model); procedural
        optionally names dotted-path plan_prompt/ground_prompt overrides for ProceduralMemory
        (absent -> its own built-in default_plan_prompt/default_ground_prompt)."""
        name: str
        strategies: dict[str, str]
        memory: dict[str, str]
        workspaces: list[dict]
        transport: dict | None = None
        llm: dict | None = None
        procedural: dict[str, str] | None = None

    def import_object(path: str) -> Any: ...        # resolve a dotted (pkg.mod.Attr) / module:attr path
    def load_yaml(config_path: str) -> AgentConfig: ...  # parse agent.yaml; require strategies.reason
    def backend_for(spec: str) -> MemoryBackend: ...     # file://<path> (or bare path) -> FileMemoryBackend
    def adapter_for(entry: dict, simulation: Any = None) -> tuple[WorkspaceOrigin, WorkspaceAdapter]: ...
        #   dispatch on origin.adapter; the are-sim (in-process ARE) kind receives the injected simulation
    def llm_for(config: AgentConfig) -> LLMClient | None: ...  # the llm: block -> a client, else None
    def procedural_prompts_for(config: AgentConfig) -> dict[str, Any]: ...
        #   resolves procedural.plan_prompt/ground_prompt dotted paths into ProceduralMemory kwargs
        #   ({} if the block is absent); a custom callable fully replaces the built-in default, it
        #   doesn't patch pieces of it.
    def transport_for(config: AgentConfig, simulation: Any = None) -> MessageTransport: ...
        #   InProcessTransport by default (peers -> raise); transport.kind: are -> AreTransport, which
        #   shares `simulation` with the are-sim workspace (user messages via the scenario's AUI)

    def build_agent(config_path: str, *, simulation: Any = None) -> Agent:
        """What `sora run` calls before handing off to TerminalSession. This is the one place all the
        wiring (which memory backend, which transport, which adapters, DecisionCycle <-> Agent sharing
        the same instances) actually happens — a developer implementing an agent never writes this.
        Stays synchronous: the async startup join runs in Agent.run().

        `simulation` is an opaque, runtime-provided shared object for adapters/transports that need one
        (currently only the ARE in-process integration's AreSimulation, shared by an are-sim workspace
        and the are transport). It keeps config generic — the per-run scenario is a CLI argument the
        runner turns into this object, not a key in agent.yaml. None for every other agent."""
        load_dotenv()   # convenience for development
        config = load_yaml(config_path)
        adapters = dict(adapter_for(entry, simulation) for entry in config.workspaces)
        registry = EnvironmentRegistry(adapters=adapters)   # the single shared instance...
        working = WorkingMemory(registry=registry)          # ...held here read-only as EnvironmentView
        semantic = SemanticMemory(backend_for(config.memory["semantic"]))
        procedural = ProceduralMemory(backend_for(config.memory["procedural"]), llm=llm_for(config),
                                       **procedural_prompts_for(config))
        episodic = EpisodicMemory(backend_for(config.memory["episodic"]))
        communication = transport_for(config, simulation)
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
