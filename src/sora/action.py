"""Extensible action space: internal actions (memory) and external actions (the world)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from sora.cycle import DecisionCycle
    from sora.environment import EnvironmentRegistry, WorkspaceOrigin
    from sora.types import ActionAck


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
    def register_internal(self, action: InternalAction) -> None: ...

    def register_external(self, action: ExternalAction) -> None: ...


class InvokeAction:  # predefined external action: _invoke_
    name = "invoke"

    async def execute(
        self,
        registry: EnvironmentRegistry,
        cycle: DecisionCycle,
        *,
        activity_id: str,
        tool_id: str,
        operation_name: str,
        **params: Any,
    ) -> ActionAck:
        raise NotImplementedError


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
