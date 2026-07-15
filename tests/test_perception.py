"""Permanent tests for the perception primitives feeding the Observe phase.

Covers ``NotificationQueueSink`` — the FIFO backing both ``signal_sink`` and the runtime-internal
``result_sink`` — and the ``PerceptKind`` StrEnum invariant. The three sink tests are promoted
verbatim from the walking-skeleton spike (``tests/test_cycle_wiring.py``);
``drain_snapshots_current_depth`` in particular pins the within-cycle no-starvation guarantee (an
item pushed *during* a drain waits for the next drain). The Observe phase is the sink's first real
consumer, so the primitive lands its permanent home here.
"""

from __future__ import annotations

from sora.perception import NotificationQueueSink, PerceptKind

# --------------------------------------------------------------------------------------------------
# NotificationQueueSink
# --------------------------------------------------------------------------------------------------


async def test_sink_push_then_drain_yields_in_order() -> None:
    # FIFO order is deliberately *not* any sorted order, and a source repeats — so this fails a
    # LIFO stack, a sort-by-source, a sort-by-value, and a dict-keyed-by-source implementation
    # (a realistic case: one source pushing several items), not just a reversed queue.
    sink: NotificationQueueSink[int] = NotificationQueueSink()
    sink.push("b", 2)
    sink.push("a", 1)
    sink.push("b", 3)
    drained = [item async for item in sink.drain()]
    assert drained == [("b", 2), ("a", 1), ("b", 3)]


async def test_sink_drain_is_empty_after_draining() -> None:
    sink: NotificationQueueSink[int] = NotificationQueueSink()
    sink.push("a", 1)
    assert [item async for item in sink.drain()] == [("a", 1)]
    assert [item async for item in sink.drain()] == []


async def test_sink_drain_snapshots_current_depth() -> None:
    # An item pushed *during* a drain waits for the next drain (no starvation within a cycle).
    sink: NotificationQueueSink[int] = NotificationQueueSink()
    sink.push("a", 1)
    seen = []
    async for item in sink.drain():
        seen.append(item)
        sink.push("late", 99)  # must not be yielded by this same drain
    assert seen == [("a", 1)]
    assert [i async for i in sink.drain()] == [("late", 99)]


# --------------------------------------------------------------------------------------------------
# PerceptKind
# --------------------------------------------------------------------------------------------------


def test_percept_kind_members_compare_equal_to_their_string_value() -> None:
    # Documented StrEnum guarantee: each member *is* its string value, so existing payloads and
    # serialisations that used the bare strings are unaffected. Binding to `str` demonstrates the
    # substitutability directly (and keeps mypy's strict-equality check happy).
    assert isinstance(PerceptKind.PROPERTY, str)
    prop: str = PerceptKind.PROPERTY
    signal: str = PerceptKind.SIGNAL
    assert prop == "property"
    assert signal == "signal"
