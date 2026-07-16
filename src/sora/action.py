"""Extensible action space: internal actions (memory) and external actions (the world)."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any, Protocol

from sora.activity import Activity, ActivityState
from sora.manual import ToolRecord, WorkspaceRecord
from sora.perception import PerceptKind
from sora.types import (
    OPERATION_NAME,
    TOOL_ID,
    ActionAck,
    OperationInvocation,
    PendingOperation,
    Step,
)

if TYPE_CHECKING:
    from sora.cycle import DecisionCycle
    from sora.environment import EnvironmentRegistry, Tool, WorkspaceOrigin


class InternalAction(Protocol):
    name: str

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> Any:
        """No EnvironmentRegistry access — internal actions only ever touch memory."""
        ...


class ExternalAction(Protocol):
    name: str
    # Whether the cycle's Act phase must do *parameter binding* on this step before dispatch —
    # grounding its abstract Step into a concrete, schema-conformant OperationInvocation (not a
    # *protocol binding*, which is the adapter's Tool concern — see ADR-0015). Only _invoke_ does;
    # every other external action dispatches straight from its Step params.
    requires_binding: bool

    async def execute(
        self,
        registry: EnvironmentRegistry,
        cycle: DecisionCycle,
        *,
        activity_id: str,
        **kwargs: Any,
    ) -> ActionAck:
        """Narrower than passing a whole Agent: registry (from working memory) + cycle
        (memory/transport/sinks), nothing else. `activity_id` is always passed by tick()'s
        dispatch, absorbed harmlessly by actions that don't need it (all but _invoke_)."""
        ...


class ActionRegistry:
    def __init__(self) -> None:
        self._internal: dict[str, InternalAction] = {}
        self._external: dict[str, ExternalAction] = {}

    def register_internal(self, action: InternalAction) -> None:
        self._internal[action.name] = action

    def register_external(self, action: ExternalAction) -> None:
        self._external[action.name] = action

    def internal(self, name: str) -> InternalAction:
        return self._internal[name]

    def external(self, name: str) -> ExternalAction:
        return self._external[name]


class InvokeAction:  # predefined external action: _invoke_
    name = "invoke"
    requires_binding = True  # abstract Step -> a concrete, schema-conformant OperationInvocation

    def __init__(self) -> None:
        # Hold strong refs to in-flight background invokes so they aren't GC'd mid-flight.
        self._tasks: set[asyncio.Task[None]] = set()

    async def execute(
        self,
        registry: EnvironmentRegistry,
        cycle: DecisionCycle,
        *,
        activity_id: str,
        **kwargs: Any,
    ) -> ActionAck:
        # tool_id/operation_name ride in via **kwargs, keeping this a structural ExternalAction
        # (the README sketch's explicit-param form isn't Protocol-compatible under mypy --strict —
        # see docs/phase-2-findings.md).
        tool_id = kwargs.pop(TOOL_ID)
        operation_name = kwargs.pop(OPERATION_NAME)
        params = kwargs
        tool = registry.get(tool_id)
        invocation = OperationInvocation(
            tool_id=tool_id, operation_name=operation_name, params=params
        )
        invocation_id = uuid.uuid4().hex
        activity = cycle.working.activities[activity_id]
        activity.pending_operation = PendingOperation(
            id=invocation_id, invocation=invocation, invoked_at=time.time()
        )
        activity.state = ActivityState.RUNNING  # implicit, unconditional — see Activities
        task = asyncio.create_task(self._call(cycle, tool, operation_name, params, invocation_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return ActionAck(ok=True)  # immediate — the round-trip runs off-cycle, cycle never blocks

    async def _call(
        self,
        cycle: DecisionCycle,
        tool: Tool,
        operation_name: str,
        params: dict[str, Any],
        invocation_id: str,
    ) -> None:
        ack = await tool.invoke(operation_name, **params)
        cycle.result_sink.push(invocation_id, ack)  # keyed by invocation_id, not tool_id


def invoke_step(tool_id: str, operation_name: str, **op_args: Any) -> Step:
    """Assemble an `invoke` Step. tool_id/operation_name are *routing* (decided at Reason time);
    they ride in Step.params under the TOOL_ID/OPERATION_NAME keys alongside the operation's own
    arguments, which Act binds. `invoke` is the one Step whose params bag mixes routing with
    arguments (DefaultActStrategy.bind splits them apart) — use this factory instead of hand-writing
    that magic-keyed dict at every call site."""
    return Step(
        next_action=InvokeAction.name,
        params={TOOL_ID: tool_id, OPERATION_NAME: operation_name, **op_args},
    )


# Every predefined external action takes the uniform (registry, cycle, **kwargs) signature and reads
# its own params out of **kwargs, rather than declaring them as explicit keyword-only args. An extra
# required keyword-only param would break the structural-subtype relation to ExternalAction under
# mypy --strict, so ActionRegistry.register_external() wouldn't type-check. The README's explicit-
# param form is illustration only (see InvokeAction, docs/phase-2-findings.md); activity_id, passed
# by tick()'s dispatch, lands harmlessly in **kwargs for all but invoke.


class FocusAction:  # predefined external action: _focus_
    name = "focus"
    requires_binding = False

    async def execute(
        self, registry: EnvironmentRegistry, cycle: DecisionCycle, **kwargs: Any
    ) -> ActionAck:
        tool_id = kwargs[TOOL_ID]
        tool = registry.get(tool_id)
        await tool.focus(cycle.signal_sink)
        cycle.working.focused_tools[tool_id] = tool
        return ActionAck(ok=True)


class UnfocusAction:  # predefined external action: _unfocus_
    name = "unfocus"
    requires_binding = False

    async def execute(
        self, registry: EnvironmentRegistry, cycle: DecisionCycle, **kwargs: Any
    ) -> ActionAck:
        tool_id = kwargs[TOOL_ID]
        tool = cycle.working.focused_tools.pop(tool_id, None)
        if tool is not None:
            await tool.unfocus()
        # Unfocusing stops re-observing this tool, so its observable-property snapshot is
        # permanently stale — drop it. Signals from the same source stay: they're fire-and-forget
        # and may still matter elsewhere (same rationale as _filter_).
        perceptions = cycle.working.perceptions
        perceptions[:] = [
            p for p in perceptions if not (p.kind is PerceptKind.PROPERTY and p.source == tool_id)
        ]
        return ActionAck(ok=True)


class JoinAction:  # predefined external action: _join_ — implies discover/connect
    name = "join"
    requires_binding = False

    async def execute(
        self, registry: EnvironmentRegistry, cycle: DecisionCycle, **kwargs: Any
    ) -> ActionAck:
        origin: WorkspaceOrigin = kwargs["origin"]
        workspace = await registry.join(origin)
        now = time.time()
        await cycle.semantic.store_workspace_record(
            WorkspaceRecord(id=workspace.id, origin=origin, discovered_at=now, last_seen_at=now)
        )
        for tool in workspace.tools():
            await cycle.semantic.store_manual(tool.manual)
            await cycle.semantic.store_tool_record(
                ToolRecord(
                    id=tool.id,
                    manual_id=tool.manual.id,
                    workspace_id=workspace.id,
                    address=tool.address,  # None unless this tool overrides the workspace's address
                    discovered_at=now,
                    last_seen_at=now,
                )
            )
        # workspace_id addresses it (for a later _leave_); tool_ids are a self-contained snapshot
        # of what was gained, legible after leave / across an agent boundary (see EXAMPLES.md).
        # the snapshot is useful for logging, e.g. saving an episode to memory
        return ActionAck(
            ok=True,
            result={
                "workspace_id": workspace.id,
                "tool_ids": [tool.id for tool in workspace.tools()],
            },
        )


class LeaveAction:  # predefined external action: _leave_ — implies close
    name = "leave"
    requires_binding = False

    async def execute(
        self, registry: EnvironmentRegistry, cycle: DecisionCycle, **kwargs: Any
    ) -> ActionAck:
        workspace_id = kwargs["workspace_id"]
        # Unfocus any of this workspace's tools first: leaving deregisters them, and a tool can only
        # be focused via its workspace (focused_tools ⊆ joined tools per A&A), so a departing
        # workspace must not leave a stale focus (a live signal subscription + a dangling handle)
        # behind. Read the tools before registry.leave() pops the workspace.
        for tool in registry.get_workspace(workspace_id).tools():
            focused = cycle.working.focused_tools.pop(tool.id, None)
            if focused is not None:
                await focused.unfocus()
        await registry.leave(workspace_id)
        return ActionAck(ok=True)


class SendAction:  # predefined external action: _send_
    name = "send"
    requires_binding = False

    async def execute(
        self, registry: EnvironmentRegistry, cycle: DecisionCycle, **kwargs: Any
    ) -> ActionAck:
        # registry unused here — every ExternalAction still gets the same uniform signature.
        await cycle.communication.send(kwargs["to"], kwargs["content"])
        return ActionAck(ok=True)


# Predefined internal actions. Each takes the (cycle, **kwargs) InternalAction signature and only
# ever touches memory (no EnvironmentRegistry) — the mechanism half of the working-memory levers
# Situate drives (create/load/unload/filter); the *policy* (which goal, which manuals) lives in the
# SituateStrategy. Params ride in via **kwargs, same reason as the external actions above.


class CreateActivityAction:  # predefined internal action: _create_activity_
    name = "create_activity"

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> Activity:
        # goal is the only required input; the strategy derives it from an unhandled message.
        # context defaults empty and activity_id is generated unless the caller pins one.
        activity = Activity(
            id=kwargs.get("activity_id") or uuid.uuid4().hex,
            goal=kwargs["goal"],
            context=kwargs.get("context") or {},
        )
        cycle.working.activities[activity.id] = activity
        return activity


class LoadManualAction:  # predefined internal action: _load_
    name = "load"

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> None:
        manual_id = kwargs["manual_id"]
        manual = await cycle.semantic.retrieve_manual(manual_id)
        # unknown id -> no-op, so a stale reference doesn't blow up the cycle
        if manual is not None:
            cycle.working.loaded_manuals[manual_id] = manual


class UnloadManualAction:  # predefined internal action: _unload_
    name = "unload"

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> None:
        cycle.working.loaded_manuals.pop(kwargs["manual_id"], None)  # absent id -> no-op


class FilterPerceptionsAction:  # predefined internal action: _filter_
    name = "filter"

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> None:
        # Prune observable-property percepts to the relevant tools, in place (`tool_ids` is the
        # relevant set — the default passes the joined workspaces' tools). Signals are retained
        # regardless of source: they're fire-and-forget, so dropping one is unrecoverable, and it
        # may still matter to another (or a blocked) activity. Signal retention/eviction is
        # consumption-driven and owned by the blocked-state machinery, not this per-cycle prune.
        tool_ids = kwargs["tool_ids"]
        perceptions = cycle.working.perceptions
        perceptions[:] = [
            p for p in perceptions if p.kind is PerceptKind.SIGNAL or p.source in tool_ids
        ]


def default_action_registry() -> ActionRegistry:
    """The predefined action space, assembled once: the six external actions plus the four internal
    working-memory actions. bootstrap and test harnesses register everything through this rather
    than naming each action inline."""
    registry = ActionRegistry()
    for external in (
        InvokeAction(),
        FocusAction(),
        UnfocusAction(),
        JoinAction(),
        LeaveAction(),
        SendAction(),
    ):
        registry.register_external(external)
    for internal in (
        CreateActivityAction(),
        LoadManualAction(),
        UnloadManualAction(),
        FilterPerceptionsAction(),
    ):
        registry.register_internal(internal)
    return registry
