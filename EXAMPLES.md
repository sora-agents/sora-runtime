# S-ORA Examples

Two worked scenarios against the API sketched in [README.md](README.md). The ARE (Meta) scenario below is the primary implementation target (see [ROADMAP.md](ROADMAP.md)); the two-agent lab afterward is an additional example that exercises the hypermedia (WoT) tool/workspace model and cross-agent messaging.

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

## Scenario: scheduling a meeting from email

A Gaia2-style task, `scenario_email_calendar`: the scenario injects an email from Alice ("Can you set up a 30-minute team sync with Bob and Carol next Monday?"), then validates that the agent creates the correct calendar event and replies.

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
    steps=[
        Step(next_action="invoke",
             params={"tool_id": "EmailApp", "operation_name": "list_emails",
                     "folder": "inbox", "limit": 5}),
        Step(next_action="invoke",
             params={"tool_id": "CalendarApp", "operation_name": "get_calendar_events_from_to"}),
        Step(next_action="invoke",
             params={"tool_id": "CalendarApp", "operation_name": "add_calendar_event"}),
        Step(next_action="invoke",
             params={"tool_id": "EmailApp", "operation_name": "reply_to_email"}),
    ]
)
```

Each step executes in a separate decision cycle. Concrete parameters (target date, attendees, email ID) are bound from the most recent percepts in working memory each cycle: the `list_emails` result feeds the `email_id` for `reply_to_email`, and the `get_calendar_events_from_to` result determines which Monday slot is free. `add_calendar_event` is a write operation, so the ARE MCP server immediately sends `resource_updated` for `CalendarApp/state` — S-ORA delivers this as a `Percept(kind="signal")` on the next `observe()`, which the `ReflectStrategy` uses to confirm the operation succeeded before advancing the plan.

## Connecting via the ARE MCP server

ARE ships an MCP server that exposes any scenario's app tools as standard MCP tools. S-ORA connects using its built-in MCP adapter — no custom adapter code is needed for operations:

```bash
# Start the ARE MCP server for a scenario
python are_simulation_mcp_server.py --scenario scenario_email_calendar --transport sse
```

`agent.yaml`:

```yaml
agent:
  name: gaia2-agent
  strategies:
    reason: examples.gaia2.ScheduleFromEmailStrategy
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

The MCP adapter's `discover()` connects to the server, enumerates all app tools across all apps as S-ORA `OperationSpecification` objects, and returns a single `Workspace`. Each app becomes a separate `Tool` within that workspace — `EmailApp`, `CalendarApp`, `SandboxFileSystem` — each with its own manual derived from the MCP tool descriptions.

A single agent could join two ARE servers (two workspaces), each exposing an app of the same name — so the adapter derives each tool's **globally-unique** id from its server's origin (not the bare app name), keeping the flat `EnvironmentRegistry` collision-free and letting any agent that reaches the same server name the tool identically. See [ADR-0014](docs/adrs/0014-tool-identity-globally-unique.md). (The exact tool-name mapping is an adapter detail: ARE's real class is `EmailClientApp`, and the ARE MCP server exposes its operations namespaced as `EmailClientApp__list_emails`.)

## Signals from ARE write operations

ARE's MCP server sends an MCP `resource_updated` notification (`app://{app_name}/state`) whenever a write operation completes — for example, when a scheduled event injects a new email into the inbox. The S-ORA MCP adapter translates these into `Signal` objects and pushes them to the `signal_sink` of the tool the agent is focused on:

```python
# Inside the MCP WorkspaceAdapter — wired in when the agent calls FocusAction
async def focus(self, sink: SignalSink) -> None:
    self._session.set_resource_updated_handler(
        lambda uri: sink.push(
            source=self._tool_id_for(uri),   # the globally-unique id discover() assigned, not the bare app name — ADR-0014
            signal=Signal(name="state_updated", payload={"uri": str(uri)}),
        )
    )
```

On the next `observe()`, `DefaultObserveStrategy` drains the `signal_sink` and appends a `Percept(source="EmailApp", kind="signal", payload=Signal("state_updated", ...), ...)` to working memory. The reasoning strategy checks for that percept and decides to re-invoke `list_emails` to discover what changed.

For `USER_MESSAGE` entries from the ARE notification system — the initial task prompt the scenario delivers to the agent — the adapter routes these through S-ORA's `MessageTransport`, so they arrive in `working_memory.messages` just like any agent-to-agent message.

## Plan reuse across scenarios

The most concrete payoff of S-ORA's procedural memory is across ARE benchmark runs. After the first `schedule-from-email` activity completes successfully, `ReflectStrategy` calls `cycle.procedural.store(activity.plan)` — persisting the four-step shape ("read email → check calendar → create event → reply") keyed by the goal.

The next scenario with the same goal pattern hits `cycle.procedural.retrieve(activity)` on its first `reason()` call, skips plan derivation entirely, and goes straight to step execution. In a Gaia2 run with hundreds of similar scheduling scenarios, this means one LLM call per step rather than one to derive the plan plus one per step — the exact throughput trade-off the runtime is designed to let you dial.

## ARE's dynamic events as reactive interrupts

ARE scenarios can inject mid-scenario events — a follow-up email from Bob ("actually, can we push it to Tuesday?") arriving while the agent is mid-plan. In ARE's default ReAct agent this restarts the turn from scratch. In S-ORA:

1. A scheduled ARE event injects a new email into the inbox.
2. The MCP server sends `resource_updated` for `EmailApp/state`.
3. `DefaultObserveStrategy` delivers it as `Percept(source="EmailApp", kind="signal", ...)`.
4. `ReflectStrategy` sees the signal and marks the current activity's plan stale.
5. The next `reason()` call re-derives a plan from the updated working memory — new target date, same shape — and execution resumes from step 2 (`get_calendar_events_from_to` with the corrected date).

No tool call already in flight is lost: the `_suspend_` / `_resume_` mechanism from the robotic-arm example (below) applies here too, if a long-running ARE operation (e.g., waiting for a user to reply) needs to block the activity until the expected event arrives.

---

# Example: A Two-Agent Lab (additional example)

This walks through a complete, two-agent scenario against the API sketched in [README.md](README.md), exercising every concept end to end: workspaces, manuals, focus/observe, invoke/suspend/resume on a signal, and cross-agent messaging kept distinct from perception.

## Scenario

A lab contains three devices, described in one hypermedia (WoT) workspace:

- **`video-stream`** — a ceiling camera that does its own scene understanding and publishes a text description; observation-only, no operations.
- **`blinds`** — motorized blinds; one operation, one observable property.
- **`robotic-arm`** — a 6-axis arm with a gripper; opens/closes the gripper and moves to 3D coordinates; movement is physical and takes real time, so it emits a signal on completion.

Two agents share this one workspace, each focusing a different subset of its tools:

- **`arm-agent`** focuses `robotic-arm` only. Its goal is to pick up a block, but it has no way to see the workbench itself.
- **`room-agent`** focuses `video-stream` and `blinds`. It can see the workbench but doesn't control the arm.

Because neither agent has everything it needs on its own, `arm-agent` asks `room-agent` what it sees — via a **message** — before it can plan where to move.

This example uses `ObservableProperty`, `Signal`, `ActionAck`, `OperationAck`, `Step`, `Plan`, `OperationInvocation`, `TickResult`, and `SendAction` as defined in the README's [API Sketch](README.md#api-sketch) — no redefinitions needed here.

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

# Functional Description
A 6-axis robotic arm with a parallel gripper, mounted at the edge of the workbench.

# Observable Properties
- gripper_state (string): "open" or "closed"
- position (3 floats): current end-effector coordinates [x, y, z], in millimeters

# Signals
- target_reached: emitted when a move_to operation's target position is physically reached

# Operations
- open_gripper(): opens the gripper
- close_gripper(): closes the gripper
- move_to(x: float, y: float, z: float): moves the end-effector to the given coordinates

# Usage Protocols & Safety
move_to is a physical motion that takes real time: after invoking it, suspend the activity and wait
for the target_reached signal before invoking close_gripper, open_gripper, or another move_to.
```

## The adapter: `WoTWorkspaceAdapter`

Implements `WorkspaceAdapter`. `discover()` builds the workspace fresh from the directory; `connect()` rebuilds it from cached records, using each tool's own address when it has one:

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

> Note: this example writes tool ids as bare names (`robotic-arm`, `blinds`, `video-stream`), which reads cleanly because they come from one shared WoT workspace whose Thing URIs are already global. Both agents naming `robotic-arm` identically is exactly the globally-unique-identity property from [ADR-0014](docs/adrs/0014-tool-identity-globally-unique.md) — here the "namespacing" is just the Thing's own URI.

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

Note: this supersedes the flat `tools: [mcp://localhost:6000]` form shown in the main README's [Running S-ORA](README.md#running-s-ora) section, which predates `_join_`/workspaces — `workspaces:` (a list of `WorkspaceOrigin`s) is the current shape.

## Startup: joining the workspace

Identical on both agents — only the focus step afterward differs:

```python
lab = WorkspaceOrigin(adapter="wot", address="http://lab.local/things")
await JoinAction().execute(agent.registry, agent.cycle, origin=lab)
```

`JoinAction` connects via `EnvironmentRegistry.join()`, registers all three tools in `agent.registry`, and persists the `WorkspaceRecord` plus each tool's `Manual`/`ToolRecord` to `agent.semantic` — so a restart can `restore()` instead of rejoining from scratch. Every action takes `(tools, cycle)` rather than the whole `Agent` — narrower than `Agent`, and it's what lets `DecisionCycle.tick()` avoid storing a back-reference to its own `Agent` (see the README's Agent/DecisionCycle wiring).

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
Percept(source="video-stream", kind="property",
        payload=ObservableProperty("scene",
            "Two piles: a blue block is on top of a red block; a green block is on top of a yellow block."),
        observed_at=1751629200.0)
```

This lands in `room_agent.working.perceptions` — nobody asked for it, it's just there because `room-agent` is focused on the tool that produces it.

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

`room-agent`'s reasoning strategy sees the message in `wm.messages`, reads the latest `scene` percept out of `wm.perceptions`, and answers:

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
                              tool_id="robotic-arm", operation="move_to", x=120.0, y=45.0, z=30.0)
```

`InvokeAction` fires this as a background task — the cycle doesn't block for the seconds a physical move takes — and, unconditionally, transitions `pick-up-block` to `running` with `pending_operation` set to this call. This isn't manual-specific: *any* invoke does this, regardless of what the tool's manual says.

A few cycles later, `move_to`'s own `OperationAck` comes back. This resolves automatically — an unambiguous match between the pending operation and its result, so the runtime clears `pending_operation`, sets `last_operation`, and returns the activity straight to `ready`, with no strategy code involved and no `Percept` produced:

```python
activity.last_operation = OperationAck(ok=True, result={"position": [120.0, 45.0, 30.0]})
activity.pending_operation = None
activity.state = ActivityState.READY
```

This is where the *second*, independent kind of waiting comes in: `robotic-arm`'s manual additionally says to wait for the `target_reached` signal before doing anything else — a condition about the arm's physical state, unrelated to whether `move_to`'s own ack has returned. Back in `reason()`, seeing `last_operation` set but knowing the manual requires this extra wait, the strategy's next decision is the internal `_suspend_` action, moving `pick-up-block` from `ready` to `blocked`. The two waits compose — implicit-and-automatic, then explicit-and-manual-driven — rather than being the same mechanism:

```python
Percept(source="robotic-arm", kind="signal",
        payload=Signal(name="target_reached", payload={}),
        observed_at=1751629210.0)
```

Reflect/Situate notices this signal (a genuine judgment call — matching it against what the manual said to wait for — which is why *this* resume isn't automatic the way the operation-completion one was), and the activity becomes `ready` again. The plan advances to closing the gripper, which goes through the exact same implicit `running` → automatic-resolve cycle as `move_to` did — no percept, no suspend, since this tool's manual doesn't require waiting for anything beyond the operation's own result:

```python
await InvokeAction().execute(agent.registry, agent.cycle, activity_id="pick-up-block",
                              tool_id="robotic-arm", operation="close_gripper")
# ... a few cycles later, resolved automatically:
activity.last_operation = OperationAck(ok=True, result={"gripper_state": "closed"})
```

## A minimal reasoning strategy

`PickUpBlockStrategy` shows the seam `ReasonStrategy` provides — a small, deterministic strategy is enough to demonstrate the pipeline without a real model call. It's only called when `result.step` is still `None`, and it hands back the accumulated `TickResult`, not a bare `Step`. This particular activity genuinely can't be fully planned upfront — the coordinates depend on room-agent's reply, which hasn't arrived yet the first few cycles — so it decides one step at a time from `activity.context`, rather than building a multi-step `Plan`:

```python
class PickUpBlockStrategy:
    async def reason(self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle,
                      result: TickResult) -> TickResult:
        if activity.state is ActivityState.BLOCKED:
            return TickResult(activity=activity, step=Step(next_action="wait", params={}))
        if "target" not in activity.context:
            reply = next((m for m in wm.messages
                           if m.sender == "room-agent" and m.content.get("type") == "reply"), None)
            if reply:
                activity.context["target"] = locate_blue_block(reply.content["answer"])
                return TickResult(activity=activity, step=Step(
                    next_action="invoke",
                    params={"tool_id": "robotic-arm", "operation_name": "move_to", **activity.context["target"]}))
            return TickResult(activity=activity, step=Step(
                next_action="send",
                params={"to": "room-agent", "content": {"type": "query", "question": "what's in front of the robot?"}}))
        if activity.context.get("gripper_state") != "closed":
            return TickResult(activity=activity, step=Step(
                next_action="invoke", params={"tool_id": "robotic-arm", "operation_name": "close_gripper"}))
        return TickResult(activity=activity, step=Step(next_action="wait", params={}))
```

`locate_blue_block` (a stand-in for whatever turns "a blue block is on top of a red block" into coordinates) is out of scope here — the point is that `wm.messages` and `wm.perceptions` are both plain, readable inputs to `reason()`, kept separate but equally available. Note that marking the activity `TERMINATED` once the gripper is closed isn't this strategy's job anymore — that judgment now belongs to a `ReflectStrategy` (here, as simple as checking `activity.context.get("gripper_state") == "closed"`), not shown in full to keep this example focused on Reason and Act.

## Fusing Reason into Act

Once `activity.context["target"]` holds real coordinates, there's nothing left for a separate `ActStrategy` call to bind — `PickUpBlockStrategy` already has the exact `x`/`y`/`z`. It can fill `invocation` directly in the same return:

```python
                return TickResult(activity=activity,
                    step=Step(next_action="invoke", params={"tool_id": "robotic-arm", "operation_name": "move_to"}),
                    invocation=OperationInvocation(tool_id="robotic-arm", operation_name="move_to", params=activity.context["target"]))
```

`DecisionCycle.tick()`'s `if result.invocation is None` guard sees this already set and never calls `act_strategy.bind()` that cycle — one call did Reason's and Act's jobs together. This is the concrete version of the runtime's general point: pluggability doesn't force any particular number of calls, and a strategy fuses forward only when it actually has the answer already — for a tool whose params need a lookup or unit conversion the reasoning strategy doesn't have handy, leaving `invocation=None` still routes to a separate, more constrained `ActStrategy` call instead.

## Reusing a plan across activities

Once `pick-up-block` (the first stack) completes, `ReflectStrategy` stores its `Plan` — the sequence of *action types* ("send", "invoke move_to", "invoke close_gripper"), independent of the specific coordinates — via `cycle.procedural.store(activity.plan)`, keyed by a goal like `"pick up the top block of a stack"`.

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
