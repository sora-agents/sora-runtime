"""Shared value types referenced throughout; kept minimal on purpose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObservableProperty:
    name: str
    value: Any


@dataclass(frozen=True)
class Signal:
    name: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class OperationInvocation:  # the concrete call, different from Step's abstract decision
    tool_id: str
    operation_name: str
    params: dict[str, Any]  # bound, ready for Tool.invoke() — the tool-hallucination-prone step


@dataclass(frozen=True)
class PendingOperation:  # tracks one in-flight invoke — lives on Activity, not WorkingMemory
    id: str  # correlates to what InvokeAction pushed into result_sink
    invocation: OperationInvocation
    invoked_at: float


@dataclass(frozen=True)
class ActionAck:  # returned by ExternalAction.execute() — dispatch, not outcome (see EXAMPLES.md)
    ok: bool
    result: Any = None


@dataclass(frozen=True)
class OperationAck:  # returned by Tool.invoke() — the tool's ack, arrives async via result_sink
    ok: bool
    result: Any = None


@dataclass(frozen=True)
class CompletedOperation:  # one resolved invocation + its ack, retained on Activity.history
    # The per-step execution trace a later step grounds against: e.g., a `reply_to_email`'s
    # email_id is resolved from an earlier `list_emails`/`search_emails` result. Kept because
    # last_operation holds only the *most recent* result (overwritten each step), which loses
    # earlier ones.
    invocation: OperationInvocation
    ack: OperationAck


@dataclass(frozen=True)
class Step:
    # next_action names the external action to dispatch (an ExternalAction.name — "invoke", "send",
    # "focus", ...) or the WAIT sentinel. params is that action's own argument bag: the cycle passes
    # it through opaquely and each action destructures it, so its shape is per-action — send ->
    # {to, content}, focus -> {tool_id}, join -> {origin}, and so on. `invoke` is the one whose bag
    # mixes *routing* (tool_id/operation_name, under the TOOL_ID/OPERATION_NAME keys) with the
    # operation's own arguments; DefaultActStrategy.bind splits the routing back out into an
    # OperationInvocation. Build an invoke Step with action.invoke_step() rather than hand-writing
    # those keys.
    next_action: str
    params: dict[str, Any]


@dataclass(frozen=True)
class Plan:  # multi-step, goal-indexed, reusable — the thing ProceduralMemory stores
    id: str  # stable identity for storage/reuse
    goal: str  # matched against future activities' goals — the retrieval key
    steps: list[Step]


# Named constants for Step.next_action values and invoke routing keys — one source of truth instead
# of bare string literals scattered across the cycle/actions/strategies (typos there are invisible
# to mypy). Registered ExternalActions are addressed by their own `.name` (e.g. InvokeAction.name);
# WAIT is the one pseudo-action the cycle special-cases (it dispatches no ExternalAction).
WAIT = "wait"

# Keys under which an `invoke` Step carries its routing in Step.params (and in InvokeAction's
# kwargs), before Act binds them into an OperationInvocation.
TOOL_ID = "tool_id"
OPERATION_NAME = "operation_name"
