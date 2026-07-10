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
class Step:
    next_action: str  # e.g. "invoke", "send", "focus", "wait"
    params: dict[str, Any]


@dataclass(frozen=True)
class Plan:  # multi-step, goal-indexed, reusable — the thing ProceduralMemory stores
    id: str  # stable identity for storage/reuse
    goal: str  # matched against future activities' goals — the retrieval key
    steps: list[Step]
