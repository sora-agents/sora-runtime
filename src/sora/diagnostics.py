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
# What a stream has already been sent, keyed by (source, name) and holding the last property value
# *successfully delivered* to it. The elision of an unchanged property's value (see
# DefaultObserveStrategy._snapshot_properties) is only lossless if the value it defers to is in the
# *reader's* stream, and working memory is not that in two separate ways: it persists across
# attempts and is written before any sink is installed, so a collector started mid-run would be
# handed a name-only marker for a property whose value it had never seen; and it is shared, so a
# value observed under one collector would leave another calling its own older value unchanged.
# Keeping the value rather than just the key answers both, since "unchanged" then means unchanged
# relative to what the reader was actually told.
#
# A stream is a *sink*, not a collection context. Nesting `collect_runtime_events` on one sink —
# or entering it from two tasks at once — produces several contexts appending to a single log, and
# per-context ledgers would then disagree with the log they share: a value recorded under the inner
# context is in the stream the outer context's reader walks, so an outer ledger that had not seen
# it would elide against the wrong row and carry a superseded value forward. Ownership therefore
# follows the sink, keyed by identity and refcounted by the number of open contexts below.
_stream_ledgers: dict[int, dict[tuple[str, str], Any]] = {}
_stream_contexts: dict[int, int] = {}
_stream_lock = threading.Lock()


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
        """Every row collected so far, as one tuple. Convenient for assertions; for export use
        ``stream`` instead, which yields the same rows without a second copy of the whole log."""
        with self._lock:
            return tuple(dict(row) for row in self._events)

    def stream(self) -> Iterator[dict[str, Any]]:
        """The same rows ``snapshot`` returns, one at a time, holding only one of them at once.

        Why this is safe to do without the lock held across the whole walk: the event list is
        append-only — ``emit`` appends and nothing ever removes, reorders, or rewrites a row — so a
        row's index never changes once assigned. Reading the length once under the lock therefore
        fixes a prefix that is already complete and will stay so, which is the same point-in-time
        view ``snapshot`` takes, reached without materializing it. Rows emitted by a concurrent
        inference task *during* the walk are simply not in that prefix, exactly as they would not be
        in a snapshot taken at the same moment.

        The per-row copy matches ``snapshot``'s: shallow, so a nested payload is shared either way.
        """
        with self._lock:
            length = len(self._events)
        for index in range(length):
            with self._lock:
                row = self._events[index]
            yield dict(row)


@contextmanager
def collect_runtime_events(sink: RuntimeEventSink) -> Iterator[RuntimeEventSink]:
    """Install ``sink`` for this context and tasks created inside it.

    The per-property value ledger the unchanged-property elision defers to belongs to ``sink``, and
    every context open on that sink shares it: they are all appending to one log, so they have to
    agree about what that log already carries. The ledger lives as long as at least one of those
    contexts does. Arming the same sink again after the last one closes starts a fresh ledger and
    so restates each value once more, which is the recoverable direction — redundancy, never a
    marker deferring to a value the reader cannot find.

    ``id(sink)`` is a safe key precisely because of that lifetime: while any context is open it
    holds a strong reference to the sink, so the identity cannot be recycled underneath the entry,
    and once the last one closes the entry is gone rather than left for an unrelated object that
    later lands on the same id. Refcounting rather than weak references keeps this working for a
    sink that is unhashable or carries ``__slots__``, which a one-method Protocol is free to be.
    """
    token = _sink.set(sink)
    key = id(sink)
    with _stream_lock:
        _stream_contexts[key] = _stream_contexts.get(key, 0) + 1
        _stream_ledgers.setdefault(key, {})
    try:
        yield sink
    finally:
        with _stream_lock:
            remaining = _stream_contexts.get(key, 1) - 1
            if remaining > 0:
                _stream_contexts[key] = remaining
            else:
                _stream_contexts.pop(key, None)
                _stream_ledgers.pop(key, None)
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
) -> bool:
    """Best-effort event emission; observer failures are deliberately invisible to the runtime.

    Returns whether the row reached the sink. No runtime decision may branch on that — swallowing a
    sink failure is the whole point, so that a diagnostic can never alter control flow or a paid
    score — but state kept *about the stream* has to stay consistent with what the stream actually
    received, which is what the property-value ledger above uses it for. Every other caller ignores
    it, and none may start reading it to decide anything the agent does.
    """
    sink = _sink.get()
    if sink is None:
        return False
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
        return False
    return True


def runtime_events_enabled() -> bool:
    return _sink.get() is not None


def _current_ledger() -> dict[tuple[str, str], Any] | None:
    """The ledger of the installed sink, or ``None`` when nothing is listening.

    Derived from ``_sink`` rather than tracked separately so that the ledger cannot drift from the
    stream it describes. Read without ``_stream_lock``: the entry exists for as long as this
    context is open, and the single-key accesses below are each atomic, so the lock is only needed
    where entries appear and disappear. Holding it per property would put it on a hot path — the
    snapshot runs over every attended tool, every cycle.
    """
    sink = _sink.get()
    return None if sink is None else _stream_ledgers.get(id(sink))


def streamed_property_value(source: str, name: str) -> Any:
    """The last value for ``(source, name)`` this stream actually received, or ``None`` if none.

    This is the precondition for eliding one: a name-only marker is readable only by someone who
    can carry a value forward from earlier in the same stream, and correct only if the value they
    would carry is the current one. ``None`` covers every case with nothing to defer to — no stream
    installed, a property this stream has not been sent, a delivery the sink dropped — and all of
    them want the value emitted, so comparing against ``None`` fails safe by construction.
    """
    ledger = _current_ledger()
    return None if ledger is None else ledger.get((source, name))


def record_streamed_property_value(source: str, name: str, value: Any) -> None:
    """Note the value for ``(source, name)`` that this stream has now been sent in full.

    Call only after a *successful* delivery. Recording one the sink rejected would turn every later
    re-observation into a marker deferring to a row that is not in the bundle, and a sink that
    recovers would never repair it: the ledger would already claim the value had been sent.
    """
    ledger = _current_ledger()
    if ledger is not None:
        ledger[(source, name)] = value


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
