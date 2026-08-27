# S-ORA Examples

Two worked scenarios against the S-ORA API (see [docs/reference/python-api.md](docs/reference/python-api.md) for the exact types). The ARE (Meta) scenario below is the primary implementation target (see [ROADMAP.md](ROADMAP.md)); the two-agent lab afterward is an additional example that exercises the hypermedia (WoT) tool/workspace model and cross-agent messaging.

# Example: Evaluating a S-ORA Agent on ARE (Meta)

**ARE** (Agents Research Environments, [arxiv](https://arxiv.org/abs/2509.17158)) is a benchmarking platform for dynamic, multi-step reasoning tasks. Unlike a single-tool ecosystem, ARE provides a full simulated environment: a set of **apps** (email client, calendar, file system, shopping, ...) backed by a discrete-event simulation engine, **scenarios** that evolve over time via scheduled events, and a validation harness for scoring agent trajectories. The Gaia2 benchmark — 800 scenarios across 10 domains — runs on top of ARE.

A S-ORA agent fits naturally into ARE because both share the same structural view of an agent operating in a tool-mediated environment:

| S-ORA concept | ARE concept |
|---|---|
| `Workspace` | ARE `Environment` (one per scenario run) |
| `Tool` | ARE `App` (one S-ORA tool per app; operations = `@app_tool` methods) |
| `WorkspaceAdapter` | ARE's built-in MCP server, consumed via S-ORA's MCP adapter |
| `Observable Property` | App state (polled from the MCP resource `app://{name}/state` each `observe()`) |
| `Signal` | MCP `resource_updated` notification — emitted by the ARE server after every write operation |
| `Message` | ARE `USER_MESSAGE` from the notification system (the scenario's initial task and follow-ups) |
| `Activity` | ARE scenario task (one or more activities, depending on task complexity) |

**Two adapter paths.** The table above and the walkthrough that follows describe the **MCP path** — S-ORA's `AreMcpWorkspaceAdapter` over ARE's MCP server. That server exposes a *static snapshot* of a scenario's initial app state and does not run the simulation engine, so it fits the single-shot plan→ground→act loop (the seeded `examples/are/mcp/email_calendar` showcase). To run a scenario's **event timeline** — mid-run email injections, follow-ups, task delivery — S-ORA also ships an **in-process path** that runs the ARE `Environment` directly; see [Running dynamic scenarios in-process](#running-dynamic-scenarios-in-process) below and the [ARE dynamic scenarios design note](docs/architecture/notes/are-dynamic-scenarios.md).

## Scenario: scheduling a meeting from email

A Gaia2-style task, `scenario_email_calendar`: the scenario injects an email from Alice ("Can you set up a 30-minute team sync with Bob and Carol next Monday?"), then validates that the agent creates the correct calendar event and replies. (`scenario_email_calendar` is an *illustrative* id — no such scenario ships in ARE 1.2.0; the real seeded scenario must be pinned against the installed ARE version. See the launch note below.)

The scenario's initial user message arrives in `working_memory.messages`. The agent creates an activity:

```python
Activity(id="schedule-sync",
         goal="schedule 30-min sync with Bob and Carol next Monday and reply to Alice")
```

`ScheduleFromEmailStrategy.reason()` retrieves a plan from procedural memory (if a similar goal was completed before) or derives one:

```python
Plan(
    id="plan-schedule-from-email",
    goal="schedule meeting and reply to requester",
    context_guard=[
        # Evaluated once at plan entry (Reason), before the body advances. A retrieval from
        # long-term memory, bound by name — not a prior step's output — so it can't be a $from.
        # Applicable only if the value is known; body params read it via {"$bind": ...}.
        {"bind": "default_duration_min",
         "query": {"memory": "semantic", "key": "default_meeting_minutes"}},
    ],
    steps=[
        Step(next_action="invoke",
             params={"tool_id": "EmailClientApp", "operation_name": "list_emails",
                     "folder": "inbox", "limit": 5}),
        Step(next_action="invoke",
             params={"tool_id": "CalendarApp", "operation_name": "get_calendar_events_from_to"}),
        Step(next_action="invoke",
             params={"tool_id": "CalendarApp", "operation_name": "add_calendar_event",
                     "duration_min": {"$bind": "default_duration_min"}}),  # from the guard, not history
        Step(next_action="invoke",
             params={"tool_id": "EmailClientApp", "operation_name": "reply_to_email"}),
    ]
)
```

Each step executes in a separate decision cycle. Two of the three binding mechanisms feed the params here, split by origin *and* timing: the **context guard** binds `default_duration_min` once at plan entry from semantic memory (read in the body via `{"$bind": ...}`), while **`$from`** resolves per-step from prior results in the activity's history — the `list_emails` result feeds the `email_id` for `reply_to_email`, and the `get_calendar_events_from_to` result determines which Monday slot is free. The third, **`$prop`**, also resolves per step but reads the observed property snapshot rather than history (`{"$prop": "CalendarApp.state"}`); this plan needs no world-state binding, since every value it grounds is either stable knowledge or a prior result. `add_calendar_event` is a write operation, so the ARE MCP server immediately sends `resource_updated` for `CalendarApp/state` — S-ORA delivers this as a signal `Percept` (appended to `wm.signals`) on the next `observe()`, which the `ReflectStrategy` uses to confirm the operation succeeded before advancing the plan.

If the request instead named *several* people to reply to — "reply to everyone who asked" — the plan would not hard-code a reply count; it would emit a **sub-goal** step (`next_action="subgoal"`), which Reason expands into one `reply_to_email` per requester. That mechanism, mechanical and deliberative, is illustrated on the multi-item RentAFlat scenario below.

### Sub-goals: multi-item work (RentAFlat)

A multi-item task — the RentAFlat scenario asks to *save each qualifying apartment*, *remove the ones already saved*, and *email each relative* — cannot hard-code its counts at synthesis time, because the counts are unknown until a search returns. A flat plan collapses each "for each" to a single call. Sub-goals fix this in two modes.

**Mechanical sub-goal** — a uniform map over a collection that a prior step produced. It fans out `len(collection)` copies of a fixed template, one invocation per cycle; the count is `len(data)`, never a model guess. *Narrowing* the collection to the qualifying items is not the sub-goal's job — a separate **`filter` data-op** step ([ADR-0023](docs/architecture/adrs/0023-structured-value-data-ops.md)) does it first, writing a named binding the sub-goal then iterates:

```python
# The plan body reaches these after a search step whose result is on the activity's history.
# 1) narrow the search result to the qualifying apartments -> a named binding.
Step(next_action="filter",
     params={
         "in":    {"$from": "search_apartments", "path": "apartments"},  # collection from history ($from)
         "out":   "qualifying",                                          # writes Activity.bindings["qualifying"]
         "where": {"$decide": "violent-crime index in 5..10 and not already saved"},  # soft predicate -> model
     })
# 2) map save_apartment over that binding — the count comes from len(qualifying), not the model.
Step(next_action="subgoal",
     params={
         "goal": "save each qualifying apartment",
         "mode": "mechanical",
         "in":   {"$bind": "qualifying"},                               # the filter's output, not a fresh $from
         "as":   "apt",                                                 # element name (named-binding namespace)
         "template": Step(next_action="invoke",
                          params={"tool_id": "RentAFlat", "operation_name": "save_apartment",
                                  "apartment_id": {"$bind": "apt", "path": "id"}}),  # element, not $from
     })
```

The `filter` step runs first: a mechanical `{"path", "op", "value"}` predicate is free, while the `{"$decide": …}` shown here escalates to one off-cycle model call over the whole collection (`ProceduralMemory.select`, resolving into the binding a later cycle like a grounding escalation). The sub-goal then resolves `in` from that binding and splices one `save_apartment` per surviving element into the active frame. Five qualifying apartments ⇒ exactly `save_apartment` ×5 — the count the oracle event graph checks, bound to the data rather than frozen at authoring time. (This composes further: dedupe with `distinct`, gather a fan-out's per-item results with `collect`, then `filter`/`take`/`reduce` them — see ADR-0023. `collect` keeps each result's *input* arguments alongside it, so a per-zip crime rate stays joinable to the `zip_code` it was fetched for even when the tool's return doesn't echo it — turning `distinct zips → get_crime_rate each → keep 5–10` into a mechanical `between` + `in` join with no `$decide`.)

The `$decide` above bundles two conditions — *"violent-crime index in 5..10 **and** not already saved"*. The "not already saved" half no longer needs the model: a mechanical `not_in` against another collection expresses it deterministically — `{"path": "apartment_id", "op": "not_in", "value": {"$from": "list_saved_apartments"}, "value_path": "apartment_id"}` keeps only the candidates whose id is absent from the saved list — leaving `$decide` for the genuinely judgemental crime threshold (or making the whole predicate mechanical if a `between` covers it). This cross-collection membership (`in`/`not_in` with a reference `value`, resolved once in Reason and projected by `value_path`) is the [ADR-0023](docs/architecture/adrs/0023-structured-value-data-ops.md) extension.

**Deliberative sub-goal** — an open, heterogeneous continuation that is *not* a uniform map (a mix of removals and tailored emails, whose shape depends on what actually got saved). Reason fires `_infer_` mid-plan (`kind="subgoal"`), so the model synthesizes the sub-plan while *seeing* the real post-save state; the sub-plan is pushed as a frame, and the parent resumes when it completes:

```python
Step(next_action="subgoal",
     params={
         "goal": "reconcile the saved shortlist against the family's requests and notify each relative",
         "mode": "deliberative",
     })
```

Reaching it moves the activity to `RUNNING` with `pending_inference` (`kind="subgoal"`), exactly like the initial plan infer — but at `step_index > 0`. When it resolves, `(plan, step_index)` is pushed onto `parent_frames` and the synthesized sub-plan becomes the active frame: e.g. `remove_saved_apartment` for the two duplicates it now sees, then a `send_email` to each of the two relatives with a summary tailored to what was saved. This is where a uniform mechanical fan-out would be wrong — the removals and emails are heterogeneous and their contents depend on run-time state — so the count and shape come from the model reading real data, not from a template. (A cached sub-plan for this sub-goal can also be retrieved first, a sub-plan library keyed by the sub-goal's goal.)

**Maintenance sub-goal** — the same `subgoal` step with `"goal_kind": "maintenance"` ([ADR-0027](docs/architecture/adrs/0027-achievement-and-maintenance-goals.md)). `mode` says how the sub-plan is produced; `goal_kind` says when the sub-goal is *finished*. An achievement sub-goal (the default, and what the two examples above are) is done when its steps run out. A maintenance one — *"for the next four minutes, whenever events are added, delete the preexisting events they overlap"* — treats those steps as the first iteration: its frame stays, the activity blocks on the `ConditionWait` covering its declared conditions, and the parent resumes only once every condition that sub-plan declared has retired via its `until`.

## Connecting via the ARE MCP server

ARE ships an MCP server that exposes any scenario's app tools as standard MCP tools. S-ORA connects using its built-in MCP adapter — no custom adapter code is needed for operations:

```bash
# The walking skeleton drives apps directly with the deterministic, seed-free --apps form over
# stdio — no port to bind and the adapter owns the subprocess lifecycle, which is far more robust
# for a gated integration test. SSE (below) remains a valid transport.
python -m are.simulation.apps.mcp.server.are_simulation_mcp_server \
    --apps are.simulation.apps.email_client.EmailClientApp are.simulation.apps.calendar.CalendarApp \
    --transport stdio

# The full four-step reproduction instead needs a *seeded* multi-app scenario (bare --apps yields an
# empty inbox). The exact scenario id and seeding are version-dependent — pin them to the installed
# ARE version rather than to this sketch.
```

`agent.yaml`:

```yaml
agent:
  name: gaia2-agent
  strategies:
    reason: examples.are.mcp.email_calendar.ScheduleFromEmailStrategy
  memory:
    working: in_process
    semantic: file://./.sora/memory/semantic
    procedural: file://./.sora/memory/procedural
    episodic: file://./.sora/memory/episodic
  transport:
    self: http://localhost:8766
  workspaces:
    - origin: {adapter: mcp, address: "http://localhost:8080/sse"}
```

The `mcp` origin above uses an SSE URL as its `address`, which locates the one MCP *server* — a workspace-level locator, not any per-app endpoint. A stdio origin instead carries the server's `command`/`args`, with `address` a nominal label (e.g. `stdio:are-email`), since stdio has no URL. Either way, the MCP adapter's `discover()` connects to the server, enumerates all app tools across all apps as S-ORA `OperationSpecification` objects, and returns a single `Workspace`. Each app becomes a separate `Tool` within that workspace — `EmailClientApp`, `CalendarApp`, `SandboxFileSystem` — each with its own manual derived from the MCP tool descriptions.

A single agent could join two ARE servers (two workspaces), each exposing an app of the same name — so the adapter derives each tool's **globally-unique** id from its server's origin (not the bare app name), keeping the flat `EnvironmentRegistry` collision-free and letting any agent that reaches the same server name the tool identically. See [ADR-0014](docs/architecture/adrs/0014-tool-identity-globally-unique.md). (The exact tool-name mapping is an adapter detail: ARE's real class is `EmailClientApp`, and the ARE MCP server exposes its operations namespaced as `EmailClientApp__list_emails`.)

## Mapping ARE's app state to S-ORA observable properties and signals

ARE's own MCP server (`ARESimulationMCPServer`) decides to expose each app's internal state as an MCP **resource** at a fixed, per-app-*type* URI — `app://{app_name}/state`, where `app_name` is the app's class-level name (`EmailClientApp.app_name()` returns `"EmailClientApp"` for every instance, not per-instance) — and fires an MCP `resource_updated` notification on that same URI whenever a write operation changes it. This is entirely ARE's design, not S-ORA's: MCP's `Resource` type carries no JSON Schema (unlike `Tool.inputSchema`/`outputSchema`), so the resource's content shape is invisible at the protocol level — only a `mimeType`/`description` are available, and S-ORA's adapter doesn't even fetch those (it reads the resource by URI directly and never calls `list_resources()`).

S-ORA's `AreMcpWorkspaceAdapter` curates this one resource per app into exactly one `ObservableProperty` (`state`) and one `Signal` (`state_changed`) via its `_observable_bindings` hook — the base `McpWorkspaceAdapter` synthesizes both as empty lists for a vanilla MCP tool (a raw MCP tool alone carries no observable state; see [ADR-0004](docs/architecture/adrs/0004-tool-usage-interface.md)), and curation is exactly the act of lifting ARE's resource convention into that shape. Because the URI is scoped to the app *type* and only meaningful within one MCP session, it's never treated as a global identifier: each workspace gets its own session and its own resource-routing table, and every `Signal` a tool pushes travels alongside its own globally-unique, origin-qualified tool id (`SignalSink.push(source, signal)` → `Percept.source`) rather than encoding identity in the `Signal` itself — so two workspaces each running an `EmailClientApp` never collide, even though both fire an identically-named `state_changed` signal.

Because no schema exists anywhere along this path, any structure an agent needs about `state`'s actual content has to live in a hand-authored manual's prose, paired in via `ManualSource`/`merge_manuals` ([ADR-0018](docs/architecture/adrs/0018-manual-merge-policy-and-authored-interface.md)) — see `examples/are/mcp/email_calendar/manuals/email-client.md` for a worked example describing the shape (folders, emails, and their fields) in full.

The two halves divide the labour strictly: the **property** carries the state, the **signal** carries only the fact that it moved, plus *where*. A signal never duplicates an observable property it accompanies ([ADR-0004](docs/architecture/adrs/0004-tool-usage-interface.md)): carrying the snapshot in both would put a copy of the whole mailbox into `wm.signals` on *every* write, where the append-log semantics keep all of them and each prompt renders a length-capped — hence unusable — prefix of one.

But thin must not collapse into **contentless**. A payload of `{"app": "Emails"}` tells a waiter that something moved and leaves it to re-derive what — against `properties`, which is a replace-by-key snapshot and by construction holds *no previous value to diff against*. That derivation isn't merely wasteful, it's impossible unless the waiter keeps its own shadow copy. So the payload also carries `changes: list[Change]` — a dotted `path` into the property plus the **identities** that appeared, vanished, or were updated, never the values behind them ([ADR-0019](docs/architecture/adrs/0019-blocked-state-machinery-and-percept-storage.md)). The snapshot stays in `properties` and the summary says where to look inside it, so the property becomes *dereferenceable* rather than re-scannable — reading one named record where a re-scan walks 127. Nothing is duplicated, because a delta is exactly what a snapshot cannot express.

The in-process adapter gets this nearly free: `observe()` already diffs the previous and current snapshots to decide whether anything changed at all, and today throws that delta away.

## Signals from ARE write operations

ARE's MCP server sends an MCP `resource_updated` notification (`app://{app_name}/state`) whenever a write operation completes — for example, when a scheduled event injects a new email into the inbox. The S-ORA MCP adapter translates these into `Signal` objects and pushes them to the `signal_sink` of the tool the agent is focused on:

```python
# Inside the MCP WorkspaceAdapter — wired in when the agent calls FocusAction
async def focus(self, sink: SignalSink) -> None:
    self._session.set_resource_updated_handler(
        lambda uri: sink.push(
            source=self._tool_id_for(uri),   # the globally-unique id discover() assigned, not the bare app name — ADR-0014
            # The event, not the state. MCP's resources/updated carries a URI and nothing else, so this
            # adapter DEGRADES to the coarse Change — "something under here moved" — rather than failing
            # or inventing a delta it cannot compute. Consumers must accept this form (ADR-0019).
            signal=Signal(name="state_changed",
                          payload={"uri": str(uri), "changes": [Change(path=str(uri))]}),
        )
    )
```

On the next `observe()`, `DefaultObserveStrategy` drains the `signal_sink` and appends a `Percept(source="EmailClientApp", payload=Signal("state_changed", {"uri": ...}), ...)` to `wm.signals`. The reasoning strategy checks for that percept and decides to re-invoke `list_emails` to discover what changed — the signal tells it *that* something moved; the refreshed `state` property and a read operation tell it *what*.

For `USER_MESSAGE` entries from the ARE notification system — the task prompt the scenario delivers to the agent, plus follow-ups — the target is S-ORA's `MessageTransport`, so they arrive in `working_memory.messages` just like any agent-to-agent message. Over the **MCP path** this has no push surface (the `AgentUserInterface` is a tool, not a resource the server notifies on), so the seeded showcase submits the task directly; it is genuinely wired in the **in-process path** below, where `AreTransport` drains the scenario's `AgentUserInterface` for the agent.

## Plan reuse across scenarios

Procedural memory is *designed* to pay off across ARE benchmark runs: a `schedule-from-email` plan — the four-step shape ("read email → check calendar → create event → reply") — could be stored once and reused whenever the same goal pattern recurs, turning hundreds of similar scheduling scenarios into one LLM call per step rather than one to derive the plan plus one per step.

That auto-caching is **currently disabled**, though: `ReflectStrategy` no longer calls `cycle.procedural.store(activity.plan)`, because replaying a completed plan verbatim is unsound — a plan corrected mid-run, or coupled to one run's observed ids, is not a reusable template. So every activity infers a fresh plan. The `store`/`retrieve` operations remain; the payoff returns once a consolidation step distils a reusable *common-case* procedure from accumulated episodes and stores that deliberately, rather than caching whatever plan happened to complete last.

## ARE's dynamic events: reconsidering a plan mid-run

ARE scenarios can inject mid-scenario events — a follow-up email ("actually, can we push it to Tuesday?") arriving while the agent is mid-plan. In ARE's default ReAct agent this restarts the turn from scratch. S-ORA reconsiders through its own decision cycle, two ways: the general **context-adaptation** mechanism (the primary path), and a **hard interrupt** (an opt-in override).

### Primary — context-adaptation ([ADR-0024](docs/architecture/adrs/0024-plan-reconsideration-context-adaptation.md))

The showcase sets `context_adaptation: before_writes`. Before committing a side-effecting step, Reason re-validates the in-flight plan against new perception:

1. A scheduled ARE event injects a new email; the runtime surfaces that state change as a `state_changed` signal — in the **in-process path** the focused tool's poll-on-observe diff catches it (the MCP path can only push `resource_updated` for the agent's *own* writes, not a background timeline injection — which is why the dynamic story runs in-process; see below).
2. A **cheap mechanical change-gate** notices that perception moved since the plan was *inferred* — the baseline is the world the plan's assumptions were formed against, captured at infer time. This is free when the world is static, so a run with no follow-up spends zero model calls, even at `before_each_op`.
3. When the gate is hot, a single off-cycle **revalidation** call asks *given the goal and the remaining steps, is the plan still valid?* On "still valid" the write proceeds and the baseline advances; on "invalidated" the activity `reset_for_replan()`s and the (default, model-backed) Reason re-infers a fresh plan against the now-updated inbox — the corrected date, same shape — with execution resuming from the corrected step.

This is **general runtime machinery**: no inbox-shape knowledge, no example code, no domain authoring. The revalidation reasons about the agent's *own* writes itself, so a self-caused `state_changed` (the reply landing in SENT) is judged "still valid" rather than triggering the reply→signal→re-plan→reply loop a coarse trigger would. The commitment is a config dial — `none | before_writes | before_each_op` — a plan-level BDI commitment strategy, not bespoke per-scenario logic.

That default is *correct but not free* — a self-caused `state_changed` still spends one revalidation to rule itself out. Because *how* the gate computes its signature is itself a pluggable seam (`strategies.change_gate`, default `PerceptionSignatureGate`), this showcase wires a domain **`InboxChangeGate`** that projects perception onto just the INBOX email ids: the reply→SENT, read-flag, and calendar self-writes leave that id-set unchanged, so they don't move the signature and never reach the revalidation — the *same* efference filter the hard-interrupt path's `MailDiffInterruptPolicy` applies, now on the cooperative path (they share one `state → id-set` projection). Dropping that one `agent.yaml` line falls back to the domain-free default: still correct, just one wasted revalidation per self-write.

### Opt-in override — a hard interrupt

For a *preemptive*, no-model alternative — and to cover a follow-up that lands **after** the goal already completed, which a before-writes checkpoint structurally can't (a terminated activity has no checkpoint) — the showcase also ships `MailDiffInterruptPolicy`/`ReconsiderInterruptHandler`, wired by uncommenting two `agent.yaml` lines. Reconsideration is then driven by a **hard interrupt** (`DecisionCycle.interrupt()`), screened at push time by a pluggable `InterruptPolicy` and routed by an `InterruptHandler` — see [ADR-0020](docs/architecture/adrs/0020-hard-interrupt-and-await-input.md):

1. The same `state_changed` signal (step 1 above) is screened at push time by the configured `InterruptPolicy`. `MailDiffInterruptPolicy` diffs the **INBOX email ids** read off the emitting tool's own `state` observable against what it has already seen — the tool, not `wm.properties`, because a policy is screened at push time, upstream of the once-per-cycle property snapshot, so working memory still holds the pre-change world at that instant ([ADR-0020](docs/architecture/adrs/0020-hard-interrupt-and-await-input.md)): a genuinely new inbound email raises a hard interrupt (`interrupt(Signal("new_inbound_email", ...))`), while the agent's own reply — which lands in SENT, not INBOX — never does (the structural self-write filter, deterministic but ARE-email-shaped). The runtime default, `NeverInterruptPolicy`, would let the signal flow cooperatively instead.
2. The pending interrupt preempts the current phase at the next checkpoint, and the `InterruptHandler` runs. `ReconsiderInterruptHandler` **clears the in-flight activity's plan** — and, because plan inference now runs off-cycle ([ADR-0021](docs/architecture/adrs/0021-llm-calls-as-async-internal-actions.md)), **invalidates any inference already in flight** on that activity (clearing its `pending_inference`), so a plan being inferred against the *pre-email* observations is discarded on resolve instead of overwriting the re-planned goal — so Reason re-infers a fresh plan against the now-updated observations; if the change landed *after* the goal already completed (no live activity), it spawns one corrective activity. Reconsideration thus lives in *one* seam (the interrupt handler), not split across bespoke Reason/Situate strategies.

No work already in flight is lost or misapplied: an interrupt never abandons a dispatched external op (it runs to completion; the interrupt is honored at the next checkpoint after its ack resolves), and an in-flight *inference* likewise finishes in the background — its result simply discarded on resolve if the activity was re-routed (the stale-inference guard). The `_suspend_` / `_resume_` mechanism from the robotic-arm example (below) applies here too, if a long-running ARE operation (e.g., waiting for a user to reply) needs to block the activity until the expected event arrives.

A user `/stop` follows the same reconsideration-by-replan model, driven by user input rather than an inbound email: the stop pauses the activity to `BLOCKED`/`InputWait`, and the follow-up line the user then types is treated as **reconsideration input seen at the next inference, not a new task** — it resumes the paused activity (clearing its plan so Reason re-infers with the message and executed history visible) and is *not* turned into a second activity. So typing "nothing, continue" resumes the original work rather than spawning a do-nothing ghost activity alongside it.

Timing caveat (override path): the ARE bridge emits `state_changed` from `tool.observe()`, i.e. *during* the Observe phase (Observe-cadence, for determinism), not off a background thread — so the policy fires inside the current tick's Observe and the checkpoint after Observe aborts the rest of that tick before Act. For the ARE sim as-is this is a clean *relocation* of the reconsideration trigger into the interrupt seam rather than new timing capability; a genuinely off-cycle signal source (a `/stop` user stop today, a future off-cycle ARE push) would let the policy fire mid-tick instead. Either way, because inference runs off-cycle, no in-flight model call is ever cut short — a stale inference is discarded when it resolves, never abandoned mid-generation.

### Staying relevant after the activity ends

Both paths above handle a change that arrives **while** the activity is live: the checkpoint re-validates an in-flight plan, the interrupt preempts one. Neither generalizes to a change that arrives *after* the goal already completed, which is why the hard-interrupt showcase carries the ad-hoc clause "if the change landed after the goal already completed (no live activity), it spawns one corrective activity" — an ARE-shaped answer wired into an example handler. Three pieces of general runtime machinery replace it.

An adaptability run makes the gap concrete. The agent cleared a conflicting appointment, scheduled a full-day event, emailed the attendee, confirmed to the user, and terminated the activity. Minutes later the attendee replied that he could not make it and proposed Thursday instead. Nothing was left alive to read it: **4 of 11** oracle actions. Three independent defects, one per layer — an unlimited wall clock would not have helped with any of them.

**1. The plan could not say what would revive it.** The synthesized plan's own prose stated three conditional clauses; its body encoded none, because a body is a finite sequence and nothing in the representation distinguished *this goal is finished* from *this goal's body is finished*. `Plan.pending` is that missing declaration — the trigger half of the plan schema, pointed forward:

```python
Plan(
    goal="Schedule a Film Production Day with Åke and let him know",
    steps=[...],                     # delete the conflict, add the event, email Åke, confirm
    pending=(PendingCondition(
        # The mechanical GATE — required. Typed, because "where to look" is the only part a protocol
        # can answer, and the only part that has to be cheap.
        watch=SignalWait(signal_name="state_changed",
                         source="insim:are/Emails",
                         path="folders.INBOX.emails"),
        when="Åke replies that he cannot make the scheduled date, or proposes a different one",
        then="Move the Film Production Day to the date Åke proposes: clear what is already "
             "scheduled that day, then re-add the full-day event",
        until="the Film Production Day has taken place",
    )),
)
```

With that unsatisfied condition the exhausted activity **blocks on a `ConditionWait`** instead of terminating. It is not a step and never blocks the body — it declares what a wait would be *for*, while every transition stays the cycle's.

**2. The signal could not say what changed.** `{"app": "Emails"}` would have told a revived activity only that the mailbox moved. With `changes` (above), the reply is a `Change(path="folders.INBOX.emails", added=("email-…",))`.

That `path` is also what makes the gate discriminate without any reasoning about self-causation. Of the run's four signals, two came from Calendar and fail on `source`; the agent's **own** `send_email` moved `folders.SENT.emails` and fails on `path`; only Åke's reply lands in `folders.INBOX.emails` and opens the gate. This is the same discrimination the showcase's `InboxChangeGate` and `MailDiffInterruptPolicy` hand-code as an ARE-email-shaped efference filter — now falling out of general machinery, because *where a change landed* is protocol-level information rather than a domain judgment.

Observe's existing resume pass matches the gate and returns the activity to `READY` — mechanically, no model. Being eligible is not the same as holding, so Reason then spends **one** batched `_infer_` over every eligible condition (*which `when`s hold, is any `until` satisfied, and if so the plan for its `then`*), and pushes `then` as a frame. One extra model call for the whole scenario, against the seven the failing run already spent.

**3. The planner might not declare anything at all** — as it demonstrably did not here, having written all three clauses in prose. So a change that opens **no** declared gate falls to undeclared-relevance recovery ([ADR-0026](docs/architecture/adrs/0026-undeclared-relevance-recovery.md)): a judge over recent episodes, **idle-scheduled** so it never displaces work that can actually advance, producing at most one candidate. That candidate becomes a **new** activity amending the terminated one — born `BLOCKED` on an `InputWait`, so the user is asked before the agent acts on a goal nobody stated, and the closed episode is never rewritten into a lie.

The three compose by subtraction, cheapest first: layer 1 claims a change that matched a declared gate, and only what is left over reaches the expensive judge. **Every condition the planner learns to declare removes work from layer 3** — they are complements, and the guess is the fallback.

## Running dynamic scenarios in-process

The MCP path above serves a *static* snapshot: ARE's MCP server never runs `Environment.run`, and it can only push `resource_updated` from inside a write-tool request — so a timeline-injected email (or an `AgentUserInterface` `USER_MESSAGE`) is never pushed to the client off-request. To make the *dynamic* story real, S-ORA ships an **in-process** path (`sora/adapters/are_sim.py`, [ARE dynamic scenarios design note](docs/architecture/notes/are-dynamic-scenarios.md)) that talks to the live ARE app objects directly and runs the `Environment` event loop on a background thread:

- **`AreSimulation`** owns the `Environment`/scenario lifecycle (started when the workspace is joined, stopped when it's left) and scores the run via `scenario.validate(env)`.
- **`AreInProcessWorkspaceAdapter`** imports each app as a tool (its ops from `app.get_tools()`, minus the `AgentUserInterface`); app-state changes surface as a `state_changed` `Signal` by **poll-on-observe** — the focused tool re-reads `get_state()` each `observe()` and diffs, the in-process analogue of the MCP resource push but driven by the cycle's own cadence, so a background timeline change is caught even though nothing pushed it.
- **`AreTransport`** (a `MessageTransport`) drains the scenario's `AgentUserInterface` unread USER messages in `receive()` and replies via `send_message_to_user()` in `send()` — this is where `USER_MESSAGE` routing is actually wired.

The scenario is a **per-run input**, not config: `agent.yaml` names the `are-sim` workspace and the `are` transport generically (no scenario key), and the runner passes the scenario — a dotted `Scenario` subclass or a Gaia2 `.json` — on the command line (`sora run`'s `--scenario` flag), which `build_agent(config, simulation=...)` injects (the workspace owns the Environment lifecycle, so `Agent.run()` starts it on the startup join and stops it on teardown). See `examples/are/sim/email_calendar/` — `uv run sora run examples/are/sim/email_calendar/agent.yaml --scenario <ref>` (add `--report examples.are.sim.email_calendar.report.report --exit-when-idle <n>` for a headless, scored run) — which reproduces the mid-run Monday→Tuesday follow-up email and the signal-driven replan against a live ARE `Environment`. (Bringing this dynamic story back onto the MCP wire — a launcher that runs the Environment plus poll-on-observe — is a backlog/exploratory item; see [ROADMAP.md](ROADMAP.md).)

---

# Example: A Two-Agent Lab (additional example)

This walks through a complete, two-agent scenario against the S-ORA API (see [docs/reference/python-api.md](docs/reference/python-api.md) for the exact types), exercising every concept end to end: workspaces, manuals, focus/observe, invoke/suspend/resume on a signal, and cross-agent messaging kept distinct from perception.

## Scenario

A lab contains three devices, described in one hypermedia (WoT) workspace:

- **`video-stream`** — a ceiling camera that does its own scene understanding and publishes a text description; observation-only, no operations.
- **`blinds`** — motorized blinds; one operation, one observable property.
- **`robotic-arm`** — a 6-axis arm with a gripper; opens/closes the gripper and moves its tool-center point to a 6-DOF pose (position + orientation); movement is physical and takes real time, so it emits a signal on completion.

Two agents share this one workspace, each focusing a different subset of its tools:

- **`arm-agent`** focuses `robotic-arm` only. Its goal is to pick up a block, but it has no way to see the workbench itself.
- **`room-agent`** focuses `video-stream` and `blinds`. It can see the workbench but doesn't control the arm.

Because neither agent has everything it needs on its own, `arm-agent` asks `room-agent` what it sees — via a **message** — before it can plan where to move.

This example uses `ObservableProperty`, `Signal`, `ActionAck`, `OperationAck`, `Step`, `Plan`, `OperationInvocation`, `TickResult`, and `SendAction` as defined in the [Python API Reference](docs/reference/python-api.md#sora.types) — no redefinitions needed here.

## The `lab` workspace

One WoT-described environment, `id="lab"`, reachable via a Thing Directory. `video-stream` and `blinds` are virtual Things hosted on that same directory server; `robotic-arm` is a physical device on its own address elsewhere in the room — exactly the mixed-addressing case a workspace is meant to support.

### Tool manuals

`manuals/video-stream.md`:

```markdown
# Tool Metadata
category: Lab / Perception
id: video-stream

# Functional Description
A ceiling-mounted camera over the workbench that performs on-device scene understanding and
publishes a symbolic description of what's currently in view — not raw video.

# Observable Properties
- scene (string): natural-language description of the objects currently visible on the workbench,
  updated whenever the scene changes.

# Signals
(none)

# Operations
(none — this tool is observation-only)

# Usage Protocols & Safety
Focus on this tool to keep `scene` current in working memory. No operations to invoke.
```

`manuals/blinds.md`:

```markdown
# Tool Metadata
category: Lab / Environment Control
id: blinds

# Functional Description
Motorized blinds covering the workbench's window, controlling ambient light.

# Observable Properties
- position (integer, 0-100): current blind position; 0 is fully closed, 100 is fully open.

# Signals
(none)

# Operations
- set_position(level: integer 0-100): moves the blinds to the given position.

# Usage Protocols & Safety
set_position completes synchronously; no suspension needed. Check `position` to confirm the move.
```

`manuals/robotic-arm.md`:

```markdown
# Tool Metadata
category: Lab / Manipulation
id: robotic-arm
wot_td: urn:cherrybot

# Functional Description
A 6-axis robotic arm (the cherryBot) with a parallel gripper, mounted at the edge of the workbench.
Its tool-center point (TCP) is a full 6-DOF pose: position plus orientation. This manual is the
protocol-agnostic, semantic half of the tool's description; the matching WoT Thing Description
(urn:cherrybot) carries the protocol binding — HTTP forms and API-key security. The two are
complementary, reconciled by tool type — an adapter maps urn:cherrybot to this manual's id
robotic-arm. See ADR-0015.

# Observable Properties
- tcp (object): current TCP pose — coordinate [x, y, z] in millimetres (x, y in -720..720; z in
  -178.3..1010) and rotation [roll, pitch, yaw] in degrees (each in -180..180).
- gripper (integer, 0-800): gripper aperture; 0 is fully closed, 800 is fully open.

# Signals
- target_reached: emitted when a move_to operation's target pose is physically reached. This is a
  semantic affordance the manual adds: the cherryBot reports completion via a webhook the WoT TD
  does not yet model, and the runtime surfaces it as this signal.

# Operations
- move_to(speed, target): moves the TCP to the given 6-DOF target pose at the given speed. speed is
  an integer in 10..400; target is a pose object with the same coordinate + rotation shape as the
  tcp property.
  - Behavior: long-running — physical motion that takes real time; completion is signalled by
    target_reached.
  - Effects: repositions the TCP to target and updates the tcp property.
- open_gripper(): opens the gripper fully (aperture 800).
  - Effects: sets gripper to 800.
- close_gripper(): closes the gripper fully (aperture 0).
  - Effects: sets gripper to 0.

# Usage Protocols & Safety
An operator must be registered before the arm accepts motion commands (the cherryBot TD exposes
registerOperator / removeOperator for this). move_to is a physical motion that takes real time:
after invoking it, suspend the activity and wait for the target_reached signal before invoking
close_gripper, open_gripper, or another move_to.
```

## The adapter: `WoTWorkspaceAdapter`

Implements `WorkspaceAdapter`. `discover()` builds the workspace fresh from the directory; `connect()` rebuilds it from cached records, using each tool's own address when it has one.

Note the boundary (see [ADR-0015](docs/architecture/adrs/0015-manuals-protocol-agnostic-adapter-boundary.md)): the adapter takes the TD's *protocol bindings* (`wot_client_for(td)`) and pairs them with a protocol-agnostic `Manual`. Here that `Manual` is loaded from a hand-authored Markdown file keyed by `td.id` — the reasoning semantics a TD lacks — while the TD itself could equally feed the manual's JSON-Schema data shapes; the two provenance channels reconcile by `Manual.id`. Either way the protocol binding stays on the `Tool` and never enters the `Manual`:

```python
class WoTWorkspaceAdapter:
    name = "wot"

    def __init__(self, directory_uri: str):
        self._directory_uri = directory_uri

    async def discover(self) -> list[Workspace]:
        tds = await wot_fetch_directory(self._directory_uri)      # 3 Thing Descriptions
        manuals = {td.id: MarkdownManualParser().parse(load_manual(td.id)) for td in tds}
        tools = [self._build_tool(td, manuals[td.id]) for td in tds]
        origin = WorkspaceOrigin(adapter=self.name, address=self._directory_uri)
        return [_WoTWorkspace(id="lab", origin=origin, tools=tools)]

    async def connect(self, workspace_record: WorkspaceRecord, tool_records: list[ToolRecord],
                       manuals: dict[str, Manual]) -> Workspace:
        tools = []
        for record in tool_records:
            address = record.address or workspace_record.origin.address   # per-tool override, else fall back
            td = await wot_fetch_thing(address)
            tools.append(self._build_tool(td, manuals[record.manual_id]))
        return _WoTWorkspace(id=workspace_record.id, origin=workspace_record.origin, tools=tools)

    def _build_tool(self, td, manual: Manual) -> Tool:
        client = wot_client_for(td)
        directory_uri = self._directory_uri
        class _WoTTool:
            id = td.id
            def __init__(self):
                self.manual = manual
                self.address = td.base if td.base != directory_uri else None   # None => rides the workspace's connection
            async def invoke(self, operation_name: str, **params) -> OperationAck:
                result = await client.invoke_action(operation_name, params)
                return OperationAck(ok=True, result=result)
            async def focus(self, sink: SignalSink) -> None:
                await client.subscribe_all(lambda name, data: sink.push(td.id, Signal(name, data)))
            async def unfocus(self) -> None:
                await client.unsubscribe_all()
            def observe(self) -> list[ObservableProperty]:
                return [ObservableProperty(name, client.cached_property(name)) for name in td.properties]
        return _WoTTool()

class _WoTWorkspace:
    def __init__(self, id: str, origin: WorkspaceOrigin, tools: list[Tool]):
        self.id, self.origin, self._tools = id, origin, tools
    def tools(self) -> list[Tool]:
        return self._tools
    async def close(self) -> None:
        for tool in self._tools:
            await tool.unfocus()
```

`video-stream`'s and `blinds`' Thing Descriptions have `base == directory_uri`, so their `Tool.address` comes out `None` — they ride the workspace's own connection. `robotic-arm`'s TD has its own `base`, so its `Tool.address` is set, matching the mixed-addressing case from the README.

`EnvironmentRegistry` is keyed by the full `WorkspaceOrigin`, so each agent registers its `WoTWorkspaceAdapter` instance against the exact `{adapter: wot, address: "http://lab.local/things"}` origin it serves:

```python
tools = EnvironmentRegistry(adapters={
    WorkspaceOrigin(adapter="wot", address="http://lab.local/things"): WoTWorkspaceAdapter("http://lab.local/things"),
})
```

> Note: this example writes tool ids as bare names (`robotic-arm`, `blinds`, `video-stream`), which reads cleanly because they come from one shared WoT workspace whose Thing URIs are already global. Both agents naming `robotic-arm` identically is exactly the globally-unique-identity property from [ADR-0014](docs/architecture/adrs/0014-tool-identity-globally-unique.md) — here the "namespacing" is just the Thing's own URI.

## Agent configuration

Both agents join the same workspace; only their focus differs. `transport.peers` is how `_send_`'s `to` parameter resolves to an address — deliberately minimal, no directory service.

`arm-agent/agent.yaml`:

```yaml
agent:
  name: arm-agent
  strategies:
    reason: examples.arm_agent.PickUpBlockStrategy   # observe/reflect/situate/act default to sora's built-ins
  memory:
    working: in_process
    semantic: file://./.sora/memory/semantic
    procedural: file://./.sora/memory/procedural
    episodic: file://./.sora/memory/episodic
  transport:
    self: http://localhost:8766
    peers:
      room-agent: http://localhost:8767
  workspaces:
    - origin: {adapter: wot, address: "http://lab.local/things"}
```

`room-agent/agent.yaml` is identical except `name: room-agent`, `transport.self: http://localhost:8767`, `peers.arm-agent: http://localhost:8766`, and its own `strategies.reason`.

Note: this supersedes the flat `tools: [mcp://localhost:6000]` form used before `_join_`/workspaces existed — `workspaces:` (a list of `WorkspaceOrigin`s) is the current shape.

## Startup: joining the workspace

Identical on both agents — only the focus step afterward differs:

```python
lab = WorkspaceOrigin(adapter="wot", address="http://lab.local/things")
ack = await JoinAction().execute(agent.registry, agent.cycle, origin=lab)
# ack.result == {"workspace_id": "lab", "tool_ids": ["video-stream", "blinds", "robotic-arm"]}
```

`JoinAction` connects via `EnvironmentRegistry.join()`, registers all three tools in `agent.registry`, and persists the `WorkspaceRecord` plus each tool's `Manual`/`ToolRecord` to `agent.semantic` — so a restart can `restore()` instead of rejoining from scratch. Its `ActionAck.result` carries `{"workspace_id", "tool_ids"}`: the `workspace_id` addresses the workspace for a later `_leave_`, while the `tool_ids` are a self-contained snapshot of what was gained — legible from an episodic trace after the workspace is left, or across an agent boundary, without dereferencing the live registry. Every action takes `(registry, cycle)` rather than the whole `Agent` — narrower than `Agent`, and it's what lets `DecisionCycle.tick()` avoid storing a back-reference to its own `Agent` (see the README's Agent/DecisionCycle wiring).

`room-agent` then focuses what it can see:

```python
await FocusAction().execute(agent.registry, agent.cycle, tool_id="video-stream")
await FocusAction().execute(agent.registry, agent.cycle, tool_id="blinds")
```

`arm-agent` focuses what it controls:

```python
await FocusAction().execute(agent.registry, agent.cycle, tool_id="robotic-arm")
```

## Perceiving the room

Once focused, `room-agent`'s `_observe()` polls `video-stream.observe()` every cycle and reflects the result into working memory as a percept — this is what `focus()` without any operations is for:

```python
Percept(source="video-stream",
        payload=ObservableProperty("scene",
            "Two piles: a blue block is on top of a red block; a green block is on top of a yellow block."),
        observed_at=1751629200.0)
```

This lands in `room_agent.working.properties` under the key `("video-stream", "scene")` — nobody asked for it, it's just there because `room-agent` is focused on the tool that produces it (a re-observed snapshot, so a later cycle overwrites it in place).

## Coordinating across agents

`arm-agent` has no `video-stream` in its `EnvironmentRegistry` at all — it never joined that tool's focus — so the only way to find out what's on the workbench is to ask. This is a message, not a percept: it doesn't originate from a focused tool, and it's addressed to a specific agent rather than broadcast as environment state.

`arm-agent` sends the query as part of its `pick-up-block` activity:

```python
await SendAction().execute(agent.registry, agent.cycle, to="room-agent",
    content={"type": "query", "question": "what's in front of the robot?"})
```

`room-agent`'s next `_observe()` drains this off `MessageTransport.receive()` straight into `working.messages` — never wrapped as a `Percept`:

```python
Message(sender="arm-agent",
        content={"type": "query", "question": "what's in front of the robot?"},
        received_at=1751629201.0)
```

`room-agent`'s reasoning strategy sees the message in `wm.messages`, reads the latest `scene` percept out of `wm.properties`, and answers:

```python
await SendAction().execute(agent.registry, agent.cycle, to="arm-agent",
    content={"type": "reply",
             "answer": "There are two piles: in the first, a blue block is on top of a red block. "
                       "In the second, a green block is on top of a yellow block."})
```

`arm-agent` receives this the same way — a `Message` in `wm.messages`, sender `"room-agent"` — and now has what it needs to plan a target position.

## Controlling the arm: two independent kinds of waiting

`arm-agent`'s plan resolves to a target position over the blue block and issues the move:

```python
await InvokeAction().execute(agent.registry, agent.cycle, activity_id="pick-up-block",
                              tool_id="robotic-arm", operation="move_to", speed=200,
                              target={"coordinate": {"x": 120.0, "y": 45.0, "z": 30.0},
                                      "rotation": {"roll": 0.0, "pitch": 90.0, "yaw": 0.0}})
```

`InvokeAction` fires this as a background task — the cycle doesn't block for the seconds a physical move takes — and, unconditionally, transitions `pick-up-block` to `running` with `pending_operation` set to this call. This isn't manual-specific: *any* invoke does this, regardless of what the tool's manual says.

A few cycles later, `move_to`'s own `OperationAck` comes back. This resolves automatically — an unambiguous match between the pending operation and its result, so the runtime clears `pending_operation`, sets `last_operation`, and returns the activity straight to `ready`, with no strategy code involved and no `Percept` produced:

```python
activity.last_operation = OperationAck(ok=True, result={"tcp": {
    "coordinate": {"x": 120.0, "y": 45.0, "z": 30.0},
    "rotation": {"roll": 0.0, "pitch": 90.0, "yaw": 0.0}}})
activity.pending_operation = None
activity.state = ActivityState.READY
```

This is where the *second*, independent kind of waiting comes in: `robotic-arm`'s manual declares that `move_to`'s completion is marked by the `target_reached` signal (`OperationSpecification.completion_signal`, from the operation's `completes_on:` interface block) — a condition about the arm's physical state, separate from whether `move_to`'s own ack has returned. This is handled *mechanically, in Observe, layered on top of the automatic resolve above* — not by a strategy and not in `reason()`. In the same `observe()` that resolved `move_to` to `ready`, a suspend pass notices the completed op declares a completion signal that hasn't arrived yet and calls the internal `_suspend_` action, moving `pick-up-block` from `ready` to `blocked` and recording `blocked_on=SignalWait("target_reached", source="robotic-arm")`. The two waits compose — implicit-and-automatic, then a separate mechanical block — rather than being the same mechanism. A few cycles later the signal arrives:

```python
Percept(source="robotic-arm",
        payload=Signal(name="target_reached", payload={}),
        observed_at=1751629210.0)
```

Observe's resume pass matches it against `blocked_on` by name (no judgment — the wait is structurally declared) and calls `_resume_` to return `pick-up-block` to `ready` — the signal itself stays in `wm.signals` (only the fixed retention cap evicts it, not the match). The plan advances to closing the gripper, which goes through the exact same implicit `running` → automatic-resolve cycle as `move_to` did — no block this time, since `close_gripper` declares no completion signal (it's synchronous):

```python
await InvokeAction().execute(agent.registry, agent.cycle, activity_id="pick-up-block",
                              tool_id="robotic-arm", operation="close_gripper")
# ... a few cycles later, resolved automatically:
activity.last_operation = OperationAck(ok=True, result={"gripper": 0})
```

## A minimal reasoning strategy

`PickUpBlockStrategy` shows the seam `ReasonStrategy` provides — a small, deterministic strategy is enough to demonstrate the pipeline without a real model call. It's only called when `result.step` is still `None`, and it hands back the accumulated `TickResult`, not a bare `Step`. This particular activity genuinely can't be fully planned upfront — the coordinates depend on room-agent's reply, which hasn't arrived yet the first few cycles — so it decides one step at a time from `activity.context`, rather than building a multi-step `Plan`:

Note there is no `BLOCKED` case to handle: a blocked activity is never selected by Situate (it's not `ready`), so `reason()` is never called on one — the suspend/resume is entirely mechanical, in Observe (above), not the strategy's concern.

```python
class PickUpBlockStrategy:
    async def reason(self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle,
                      result: TickResult) -> TickResult:
        if "target" not in activity.context:
            reply = next((m for m in wm.messages
                           if m.sender == "room-agent" and m.content.get("type") == "reply"), None)
            if reply:
                activity.context["target"] = locate_blue_block(reply.content["answer"])
                return TickResult(activity=activity, step=Step(
                    next_action="invoke",
                    params={"tool_id": "robotic-arm", "operation_name": "move_to",
                            "speed": 200, "target": activity.context["target"]}))
            return TickResult(activity=activity, step=Step(
                next_action="send",
                params={"to": "room-agent", "content": {"type": "query", "question": "what's in front of the robot?"}}))
        if activity.context.get("gripper") != 0:
            return TickResult(activity=activity, step=Step(
                next_action="invoke", params={"tool_id": "robotic-arm", "operation_name": "close_gripper"}))
        return TickResult(activity=activity, step=Step(next_action="wait", params={}))
```

`locate_blue_block` (a stand-in for whatever turns "a blue block is on top of a red block" into coordinates) is out of scope here — the point is that `wm.messages` and the perceptual stores (`wm.properties`, `wm.signals`) are all plain, readable inputs to `reason()`, kept separate but equally available. Note that marking the activity `TERMINATED` once the gripper is closed isn't this strategy's job anymore — that judgment now belongs to a `ReflectStrategy` (here, as simple as checking `activity.context.get("gripper") == 0`), not shown in full to keep this example focused on Reason and Act.

## Fusing Reason into Act

Once `activity.context["target"]` holds real coordinates, there's nothing left for a separate `ActStrategy` call to bind — `PickUpBlockStrategy` already has the exact `x`/`y`/`z`. It can fill `invocation` directly in the same return:

```python
                return TickResult(activity=activity,
                    step=Step(next_action="invoke", params={"tool_id": "robotic-arm", "operation_name": "move_to"}),
                    invocation=OperationInvocation(tool_id="robotic-arm", operation_name="move_to",
                        params={"speed": 200, "target": activity.context["target"]}))
```

`DecisionCycle.tick()`'s `if result.invocation is None` guard sees this already set and never calls `act_strategy.bind()` that cycle — one call did Reason's and Act's jobs together. This is the concrete version of the runtime's general point: pluggability doesn't force any particular number of calls, and a strategy fuses forward only when it actually has the answer already — for a tool whose params need a lookup or unit conversion the reasoning strategy doesn't have handy, leaving `invocation=None` still routes to a separate, more constrained `ActStrategy` call instead.

## Reusing a plan across activities

This illustrates the reuse *capability* (the default `ReflectStrategy` no longer auto-stores completed plans — see [Plan reuse across scenarios](#plan-reuse-across-scenarios) — so realizing it needs a deliberate store: a custom Reflect, or the future episode-consolidation step). Given such a store, once `pick-up-block` (the first stack) completes its `Plan` — the sequence of *action types* ("send", "invoke move_to", "invoke close_gripper"), independent of the specific coordinates — is kept via `cycle.procedural.store(activity.plan)`, keyed by a goal like `"pick up the top block of a stack"`.

When a second activity starts — `pick-up-second-block`, same goal, different stack — its first `reason()` call has no `activity.plan` yet, so it calls `cycle.procedural.retrieve(activity)` before falling back to deriving one from scratch:

```python
class PickUpBlockStrategy:
    async def reason(self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle,
                      result: TickResult) -> TickResult:
        if activity.plan is None:
            activity.plan = await cycle.procedural.retrieve(activity) or Plan(
                id=f"plan-{activity.id}", goal=activity.goal,
                steps=[Step("send", {...}), Step("invoke", {"operation_name": "move_to"}), Step("invoke", {"operation_name": "close_gripper"})])
            activity.step_index = 0
        step = activity.plan.steps[activity.step_index]
        activity.step_index += 1
        return TickResult(activity=activity, step=step)
```

For the second stack, `cycle.procedural.retrieve()` hits — the *shape* of the plan (ask, move, grip) is identical even though the coordinates differ — so `reason()` never re-derives that shape, only fills in per-step params (still its own job, same as before) as each step comes up. The saving isn't "zero work per cycle," it's "no re-deriving the *sequence* every time," which is exactly what made a single `next_action` field worth promoting to a real `Plan`.

## Shutting down

```python
await UnfocusAction().execute(agent.registry, agent.cycle, tool_id="robotic-arm")
await LeaveAction().execute(agent.registry, agent.cycle, workspace_id="lab")
```

`LeaveAction` calls `workspace.close()`, tearing down the WoT client's subscriptions in one call rather than per tool, and deregisters every tool that came from `lab` in one step.
