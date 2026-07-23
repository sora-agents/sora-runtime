# ARE dynamic scenarios via an in-process simulation bridge

Design note behind the in-process ARE integration. Records how S-ORA runs a *dynamic* ARE (Meta
Agents Research Environments) scenario — a live, ticking `Environment` whose event timeline actually
fires — why that path is **in-process** rather than over MCP, and what validating it against the
installed ARE source turned up. This is an integration note, not an architectural decision: the
durable seams it relies on are already fixed by [ADR-0013](adrs/0013-shared-instances-narrow-dependencies.md)
(centralized wiring), [ADR-0012](adrs/0012-percepts-vs-messages.md) (user messages on the transport),
and [ADR-0003](adrs/0003-adapters-not-tool-authoring.md) (adapters import tools, never author them).
The whole integration lives behind one file (`sora/adapters/are_sim.py`, optional `are` dependency
group) and one opaque bootstrap kwarg, so it is deletable without touching the runtime core — which is
exactly why it is a note and not an ADR. The MCP alternative below stays an open exploratory task.

## The problem

The runtime also reaches ARE over its MCP tool surface. That server exposes a *static snapshot* of app
initial state and never runs `Environment.run`, so a scenario's **event timeline** — mid-run email
injections, task/follow-up delivery through the `AgentUserInterface` (AUI) — never fires. To reproduce
a *dynamic* Gaia2 scenario (signal-driven replanning against a live, ticking world), the ARE
`Environment` event loop must run and its two off-cycle event channels must bridge into the runtime:
**app state changes** → a `Signal` for a focused tool, and **AUI USER messages** → the
`MessageTransport`. The question was which shape carries these.

Constraints that shaped the answer:

* The bridge must surface **off-cycle** changes: a timeline-injected email happens with no agent
  action and no in-flight request.
* Keep the runtime's seams intact — user messages are a `MessageTransport` concern (ADR-0012), tools
  and signals are the `WorkspaceAdapter`/`signal_sink` concern; don't smuggle one across the other.
* Config stays generic and domain-agnostic (a scenario is a *per-run* input, like a task), and
  bootstrap must not hard-depend on ARE.
* Reuse existing lifecycle patterns (a workspace owning its connection) and existing S-ORA-side
  logic, testable without the optional ARE dependency.

## Why in-process, not over MCP

Two shapes were considered:

* **(a) In-process** — a `WorkspaceAdapter` + `MessageTransport` over the live ARE app objects, with
  `Environment.run(..., wait_for_end=False)` on a background thread in the same process.
* **(b) Over MCP** — keep the `are_mcp` adapter; a launcher runs `Environment.run` in a thread and
  points `ARESimulationMCPServer` at the live apps.

The deciding investigation: ARE's MCP server emits `resource_updated` **only from inside a write-tool
request** (`are_simulation_mcp_server.py`), using that request's session — it keeps no subscription
registry and runs no state watcher. MCP the *protocol* supports async `notifications/resources/updated`,
so this is an ARE **implementation** gap, not a protocol limit; but closing it for timeline changes
needs a durable client session pushed from the Environment's *thread* into the server's asyncio loop
(off-request, cross-thread) — plumbing that lives in ARE, not S-ORA. And the AUI USER message has no
MCP push surface at all, and belongs on the transport seam, not the adapter's resource surface.

So **in-process won**, because the two channels MCP makes hard become direct method calls on shared
objects, with no ARE changes and no framework-fighting. **ARE-over-MCP (option b) is not discarded** —
it is retained as a backlog/exploratory item (a launcher + poll-on-observe spike whose findings feed
protocol-interop discussion in the WebAgents CG); it does not gate the release. See ROADMAP.md.

Trade-offs accepted with (a): it gives up exercising a real MCP wire for the dynamic path (the static
seeded MCP demo keeps that), and it carries the ARE thread-safety burden documented under Findings.

## The three components (`sora/adapters/are_sim.py`)

Lazy ARE imports throughout (the optional `are` dependency-group):

* **`AreSimulation`** owns the `Environment`/scenario lifecycle and is the single object both seams
  share. A `threading.Lock` serializes S-ORA's *own* concurrent app calls (an `invoke` on a worker
  thread vs an `observe` on the cycle thread); it does **not** — and cannot — serialize against ARE's
  event-loop thread, which mutates app state with no lock we can share (see Findings).
* **`AreInProcessWorkspaceAdapter`** imports each live app as a tool (ops from `app.get_tools()`,
  plus a `state` observable + `state_changed` signal). Off-cycle changes surface by **poll-on-observe**:
  the tool re-reads `app.get_state()` each Observe and pushes `state_changed` on a diff — the
  in-process analogue of the MCP resource-update push, tied to the cycle's own Observe cadence (so it
  is deterministic, no extra timer). The AUI app is excluded from the tool surface.
* **`AreTransport`** (`MessageTransport`) drains the AUI's unread USER messages in `receive()` and
  replies via `send_message_to_user()` in `send()` — resolving the "USER_MESSAGE not wired" gap.

## Wiring

* **The workspace owns the Environment lifecycle** — `discover()` starts the simulation, the
  workspace's `close()` stops it — exactly as `_McpWorkspace` owns its stdio subprocess. So
  `Agent.run()` needs no change: startup `_join_` starts it, teardown `leave` stops it.
* **One opaque injection seam in bootstrap** — `build_agent(config, *, simulation=None)` threads the
  runtime-built `AreSimulation` into the `are-sim` adapter and the `are` transport. Config names the
  *kinds* only (generic); the scenario is a CLI argument the runner turns into the simulation. This
  keeps ADR-0013 intact (one wiring place; the runner supplies one shared object) and bootstrap
  ARE-agnostic (the object is opaque; ARE is imported lazily in the dispatch branch, like `mcp`).

## Handling the dynamic change (example strategies, not runtime)

The bridge only *surfaces* the mid-run change (a follow-up email); making the agent act on it is the
application's job, done entirely through the pluggable phase strategies (no runtime change). The
showcase ships two, in `examples/are_scenario/strategies.py`, so the follow-up is handled regardless
of when it lands relative to the original activity's life:

* **`ReconcilingReasonStrategy`** — while the scheduling activity is still in flight, a **new inbound
  email** invalidates its plan and **re-infers** from the now-updated observations. It calls
  `procedural.infer` *directly* rather than nulling the plan and delegating: the activity already has
  an in-flight plan, so `DefaultReasonStrategy` would take its cheap path and merely advance that
  plan — re-inferring directly is what forces a fresh plan mid-flight.
* **`CorrectiveSituateStrategy`** — if the email instead lands *after* the goal already completed
  (an email is only observed state, and never spawns an activity the way a USER message does),
  it spawns one fresh corrective activity so the agent still reconciles. Situate is already the
  activity-creation phase; this just triggers on observed inbox state rather than a message.

**Why the trigger is inbound-email *content*, not "a signal arrived".** The obvious trigger — re-plan
whenever a `state_changed` signal appears — **loops forever**: the agent writes to the very tool it
watches (its reply), every write emits a `state_changed`, so it re-plans on its own action, replies
again, and so on. (Signals are also never consumed from `WorkingMemory` — an ADR-0019 decision so one
signal can satisfy multiple waiters — so a count-based trigger can't drain them either.) The fix keys
on the set of **INBOX email ids**: a follow-up grows the inbox, while the agent's reply lands in SENT,
so a self-write is structurally invisible to the trigger. This is example-level, ARE-email-shaped
logic (`_inbound_email_ids` knows the `folders/INBOX/emails` shape). The general fix — efference /
read-write tags so *any* self-caused change is filtered regardless of tool — is deferred alongside
BDI-style commitment policies and hard-interrupt preemption.

**Precondition: the plan must focus the tools it reconciles against** — the inbox *and* every tool
whose state it changes (here the calendar). Observable properties are only snapshotted for a
*focused* tool (`DefaultObserveStrategy`), so without a `focus` step the tool's state never reaches
working memory. An unfocused inbox means the agent runs blind to the follow-up (observed directly: it
scheduled the wrong day and never reacted); an unfocused calendar means it can't see what it already
booked, so it can't tell a stale event to delete from none — and since a step has no "skip if empty",
a blindly-planned delete of a non-existent event fails outright. Reconciliation is therefore
observation-driven: the plan deletes/updates only a stale item it can *currently see*. Focus is a
plan step the base planner treats as optional, so `reconciling_plan_prompt` asks for it explicitly.
(Plan auto-caching is disabled runtime-wide — the default Reflect no longer stores completed plans —
so each run infers fresh; there is no stale cached plan to clear between runs.)

Exactly one path handles a given email (a shared `_SEEN_INBOUND` set on `activity.context` hands off
between them without double-fixing). The "you may have already acted — inspect current state and
undo/modify rather than duplicating" instruction is a commitment-aware
**plan prompt** (`reconciling_plan_prompt`, wired via `agent.yaml`'s `procedural.plan_prompt`), *not*
BDI-style commitment machinery (single-minded/open-minded reconsideration as a first-class pluggable
policy) — that, and hard-interrupt preemption so the reaction is immediate rather than next-tick
(the `DecisionCycle.interrupt()` item in ROADMAP.md), are deferred. The strategies are pinned by
deterministic fakes in `tests/test_are_dynamic_strategies.py`.

## What this buys

* The dynamic story is real: a timeline change → `state_changed` signal → the agent replans (in
  flight) or spawns corrective work (after completion) through its own cycle; the task and
  follow-ups flow through the transport; a run is scoreable via `scenario.validate(env)`.
* The poll-on-observe value-diff is sound: ARE's `get_state()` returns a freshly built tree of
  primitives each call (recursive `serialize_field`/`make_serializable`), so `state != last_state`
  is a real value comparison — no identity aliasing (silently never firing) and no fresh-identity
  churn (spuriously firing every cycle). See Findings.
* No ARE changes and no FastMCP off-request/cross-thread plumbing.
* Runs against **any** ARE scenario (dotted `Scenario` subclass or Gaia2 `.json`), config unchanged.
* The S-ORA-side logic depends only on a small duck-typed app/AUI interface, so it is unit-tested
  with fakes (CI-green), independent of the optional ARE dependency (ADR-0003).

## Accepted costs

* Gives up exercising a real MCP wire for the dynamic path (the static seeded MCP demo keeps that).
* Thread-safety burden: the Environment thread mutates apps concurrently, and — as the finding below
  establishes — **ARE exposes no lock we can share**, so this race cannot be closed from S-ORA. It is
  mitigated (bounded retry on the transient concurrent-modification error), not eliminated; a real,
  ongoing constraint we accept for a showcase.
* Poll-on-observe reads whole app state each Observe for focused tools (fine at showcase scale;
  a background-poll variant is the fallback if the Observe cadence proves too coarse).

## Findings from validating against the installed ARE source (2026-07-23)

Four claims were checked directly against ARE's source (`environment.py`, `apps/*.py`,
`utils/serialization.py`); three corrected the implementation, one was a false alarm:

* **Concurrent app mutation is unguarded (confirmed; mitigated, not closed).** `Environment` runs
  `process_event` → `event.execute()` (the app mutation, e.g. `email.add_email`) on its `"EventLoop"`
  thread, and `environment.py` holds **no `threading.Lock`** (only `threading.Event` for stop/pause,
  which are signals, not mutexes; apps are plain dataclasses with no internal lock). So a focused
  tool's `get_state()` on the cycle thread can iterate a container the event loop is concurrently
  growing → a transient `RuntimeError: ... changed size during iteration`. The `AreSimulation` lock
  serializes only S-ORA's own calls; it never covers the event-loop thread. Because mutation is
  bursty (`tick()` then `time.sleep(1)`), the window is small and the failure is *intermittent* — the
  worst profile for a demo (green in tests, occasional live corruption). Mitigation: `_AreTool._read_state`
  retries the snapshot (`_STATE_READ_ATTEMPTS`) and re-raises only if it never settles. `env.pause()`
  is **not** a fix — it stalls between ticks, not mid-tick, so it gives no mutual exclusion.
* **The `state_changed` value-diff is sound (false alarm, now documented).** A concern that
  `get_state()` might return a live/aliased object (breaking `!=`) was dismissed: `get_state()` fully
  materializes a fresh primitive tree every call, so the diff is a correct value comparison. The
  `invoke`-serializes-but-`observe`-doesn't asymmetry is therefore harmless — `get_state()` already
  serializes.
* **The AUI task message carries a `0.0` timestamp (confirmed; fixed).** `AUIMessage.timestamp` is
  *simulation-relative* time (`time_manager.time()`), and the default scenario delivers the task via
  the AUI at `delay_seconds=0` (`start_time = 0`) → timestamp `0.0`. `AreTransport.receive` derived
  `received_at` with `... or time.time()`, which discards a legitimate `0.0` for wall-clock time and
  corrupts ordering; fixed to distinguish absent (`None`) from `0.0`. (Aside: `get_last_unread_messages`
  already filters to `Sender.USER` and marks them read, so the transport's hardcoded `sender="user"`
  and drain-once assumption both hold — a second dismissed concern.)
* **Container/union arg types were mis-mapped, so grounding produced ill-typed values (confirmed;
  fixed).** ARE `AppTool` args declare their type as a *string* (`list[str] | None`,
  `int | float | None`, `dict[str, Any]`, ...), and each is fed to the grounding model through a
  synthesized JSON-Schema parameter. The first cut knew only the four scalars and collapsed
  everything else to JSON `string`, so the model was told `attendees` (a `list[str]`) was a string,
  filled it `"Alice, Bob"`, and ARE's own runtime type-check rejected it (`must be of type
  list[str] | None, got str`) — the failure only shows against a live model, since grounding is
  where the value is invented. A survey of **all 44 ARE apps found only 10 distinct arg-type
  strings**; `_json_type` now maps every one faithfully — recursive on `list[...]` (so the item type
  is right too), `dict`/`dict[...]` → object, and unions split on `|` (`None` dropped) with an
  all-numeric union (`int | float`) → JSON `number` (admits both). Two residual limits the schema
  *can't* close, by nature: a free-form `dict[str, Any]` maps to an untyped object with no key
  guidance (that's an authored manual's job, [ADR-0018](adrs/0018-manual-merge-policy-and-authored-interface.md)), and any remaining model deviation surfaces
  as a failed `OperationAck` that terminates the activity — a graceful failure, not a crash. Related
  reporting fix in the showcase runner: ARE's base `Scenario.validate` only checks the environment
  didn't enter a `FAILED` state and runs any oracle validators, so a scenario with no oracle events
  (like the default one) reports `PASS` even when the agent failed the task; `examples/are_scenario/run.py`
  now reports the agent's own outcome separately and labels the ARE check as vacuous without oracle
  events.

## Links

* Depends on [ADR-0019](adrs/0019-blocked-state-machinery-and-percept-storage.md) (signal storage; a
  `send_message_to_user` ask-user maps onto `_suspend_`/`_resume_`).
* Applies [ADR-0013](adrs/0013-shared-instances-narrow-dependencies.md) (the opaque `simulation`
  injection keeps wiring centralized in bootstrap).
* Applies [ADR-0012](adrs/0012-percepts-vs-messages.md) (USER messages stay on the transport, not the
  signal/percept path) and [ADR-0003](adrs/0003-adapters-not-tool-authoring.md) (the adapter imports
  ARE's tools, it does not author them).
* The ARE-over-MCP alternative (shape b) is tracked as a backlog/exploratory item in ROADMAP.md.
