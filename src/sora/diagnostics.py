"""Fail-soft, context-local runtime event diagnostics."""

from __future__ import annotations

import contextvars
import dataclasses
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime
from enum import Enum
from itertools import islice
from typing import Any, Protocol


class RuntimeEventSink(Protocol):
    """A write-only observer of already-committed runtime events."""

    def emit(self, event: dict[str, Any]) -> None: ...


_sink: contextvars.ContextVar[RuntimeEventSink | None] = contextvars.ContextVar(
    "sora_runtime_event_sink", default=None
)
_cycle: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "sora_runtime_event_cycle", default=None
)
_phase: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sora_runtime_event_phase", default=None
)
_activity: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sora_runtime_event_activity", default=None
)
_cause: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sora_runtime_event_cause", default=None
)


def json_safe(value: Any) -> Any:
    """Return a deterministic JSON-compatible projection of arbitrary runtime values."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: json_safe(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [json_safe(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((json_safe(item) for item in value), key=repr)
    return {"type": type(value).__name__, "repr": repr(value)}


_PREVIEW_MAX_DEPTH = 4
_PREVIEW_MAX_ITEMS = 32
_PREVIEW_MAX_NODES = 128
_PREVIEW_MAX_STRING = 2048


def _diagnostic_summary(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(value).__name__}
    if isinstance(value, dict | list | tuple | set | frozenset):
        summary["length"] = len(value)
    return summary


def diagnostic_preview(value: Any, *, budget: list[int] | None = None, depth: int = 0) -> Any:
    """Return a JSON-safe structural preview with bounded cycle-thread traversal."""
    remaining = budget if budget is not None else [_PREVIEW_MAX_NODES]
    if remaining[0] <= 0 or depth > _PREVIEW_MAX_DEPTH:
        return {**_diagnostic_summary(value), "truncated": True}
    remaining[0] -= 1
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        if len(value) <= _PREVIEW_MAX_STRING:
            return value
        return {
            "type": "str",
            "length": len(value),
            "prefix": value[:_PREVIEW_MAX_STRING],
            "truncated": True,
        }
    if isinstance(value, Enum):
        return diagnostic_preview(value.value, budget=remaining, depth=depth + 1)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": diagnostic_preview(str(value), budget=remaining, depth=depth + 1),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        dataclass_fields = list(islice(dataclasses.fields(value), _PREVIEW_MAX_ITEMS + 1))
        projected_fields = {
            field.name: diagnostic_preview(
                getattr(value, field.name), budget=remaining, depth=depth + 1
            )
            for field in dataclass_fields[:_PREVIEW_MAX_ITEMS]
        }
        if len(dataclass_fields) > _PREVIEW_MAX_ITEMS:
            projected_fields["__diagnostic_truncated__"] = True
        return projected_fields
    if isinstance(value, Mapping):
        mapping_items = list(islice(value.items(), _PREVIEW_MAX_ITEMS + 1))
        projected_mapping = {
            str(key): diagnostic_preview(item, budget=remaining, depth=depth + 1)
            for key, item in mapping_items[:_PREVIEW_MAX_ITEMS]
        }
        if len(mapping_items) > _PREVIEW_MAX_ITEMS:
            projected_mapping["__diagnostic_truncated__"] = True
        return projected_mapping
    if isinstance(value, list | tuple):
        sequence_items = value[: _PREVIEW_MAX_ITEMS + 1]
        projected_sequence = [
            diagnostic_preview(item, budget=remaining, depth=depth + 1)
            for item in sequence_items[:_PREVIEW_MAX_ITEMS]
        ]
        if len(sequence_items) > _PREVIEW_MAX_ITEMS:
            projected_sequence.append({"__diagnostic_truncated__": True, "length": len(value)})
        return projected_sequence
    return _diagnostic_summary(value)


class RuntimeEventCollector:
    """Thread-safe in-memory sink for one attempt.

    Sequence assignment and elapsed-time sampling happen at emission, so events from concurrent
    inference tasks retain the order in which they crossed this observer.
    """

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []

    def emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            row = {
                "sequence": len(self._events),
                "elapsed_seconds": time.perf_counter() - self._started,
                **event,
            }
            self._events.append(row)

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(dict(row) for row in self._events)


@contextmanager
def collect_runtime_events(sink: RuntimeEventSink) -> Iterator[RuntimeEventSink]:
    """Install ``sink`` for this context and tasks created inside it."""
    token = _sink.set(sink)
    try:
        yield sink
    finally:
        _sink.reset(token)


@contextmanager
def runtime_event_context(
    *,
    cycle: int | None = None,
    phase: str | None = None,
    activity_id: str | None = None,
    cause: str | None = None,
) -> Iterator[None]:
    """Add correlation context without coupling the observer to ``DecisionCycle`` or ``Agent``."""
    tokens: list[tuple[contextvars.ContextVar[Any], contextvars.Token[Any]]] = []
    if cycle is not None:
        tokens.append((_cycle, _cycle.set(cycle)))
    if phase is not None:
        tokens.append((_phase, _phase.set(phase)))
    if activity_id is not None:
        tokens.append((_activity, _activity.set(activity_id)))
    if cause is not None:
        tokens.append((_cause, _cause.set(cause)))
    try:
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


def emit_runtime_event(
    event: str,
    *,
    payload: Any = None,
    activity_id: str | None = None,
    cause: str | None = None,
    source_timestamp: float | None = None,
) -> None:
    """Best-effort event emission; observer failures are deliberately invisible to the runtime."""
    sink = _sink.get()
    if sink is None:
        return
    try:
        row = {
            "event": event,
            "cycle": _cycle.get(),
            "phase": _phase.get(),
            "activity_id": activity_id if activity_id is not None else _activity.get(),
            "cause": cause if cause is not None else _cause.get(),
            "payload": json_safe(payload if payload is not None else {}),
        }
        if source_timestamp is not None:
            row["source_timestamp"] = source_timestamp
        sink.emit(row)
    except BaseException:
        # A diagnostic is never allowed to change runtime control flow or a paid score.
        return


def runtime_events_enabled() -> bool:
    return _sink.get() is not None


@contextmanager
def runtime_phase(cycle: int, phase: str, *, activity_id: str | None = None) -> Iterator[None]:
    with runtime_event_context(cycle=cycle, phase=phase, activity_id=activity_id):
        emit_runtime_event("phase.entry", payload={"phase": phase})
        try:
            yield
        except BaseException as error:
            emit_runtime_event("phase.exit", payload={"phase": phase, "ok": False, "error": error})
            raise
        emit_runtime_event("phase.exit", payload={"phase": phase, "ok": True})
