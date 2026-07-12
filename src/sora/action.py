"""Extensible action space: internal actions (memory) and external actions (the world)."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any, Protocol

from sora.activity import ActivityState
from sora.types import ActionAck, OperationInvocation, PendingOperation

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
        tool_id = kwargs.pop("tool_id")
        operation_name = kwargs.pop("operation_name")
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


class FocusAction:  # predefined external action: _focus_
    name = "focus"

    async def execute(
        self, registry: EnvironmentRegistry, cycle: DecisionCycle, *, tool_id: str, **kwargs: Any
    ) -> ActionAck:
        raise NotImplementedError


class UnfocusAction:  # predefined external action: _unfocus_
    name = "unfocus"

    async def execute(
        self, registry: EnvironmentRegistry, cycle: DecisionCycle, *, tool_id: str, **kwargs: Any
    ) -> ActionAck:
        raise NotImplementedError


class JoinAction:  # predefined external action: _join_ — implies discover/connect
    name = "join"

    async def execute(
        self,
        registry: EnvironmentRegistry,
        cycle: DecisionCycle,
        *,
        origin: WorkspaceOrigin,
        **kwargs: Any,
    ) -> ActionAck:
        raise NotImplementedError


class LeaveAction:  # predefined external action: _leave_ — implies close
    name = "leave"

    async def execute(
        self,
        registry: EnvironmentRegistry,
        cycle: DecisionCycle,
        *,
        workspace_id: str,
        **kwargs: Any,
    ) -> ActionAck:
        raise NotImplementedError


class SendAction:  # predefined external action: _send_
    name = "send"

    async def execute(
        self,
        registry: EnvironmentRegistry,
        cycle: DecisionCycle,
        *,
        to: str,
        content: dict[str, Any],
        **kwargs: Any,
    ) -> ActionAck:
        raise NotImplementedError
