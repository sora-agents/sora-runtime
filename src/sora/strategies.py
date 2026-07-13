"""One pluggable strategy per phase, threaded through a shared TickResult."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from sora.activity import ActivityState
from sora.perception import Percept, PerceptKind
from sora.types import OPERATION_NAME, TOOL_ID, OperationInvocation, Step

if TYPE_CHECKING:
    from sora.activity import Activity
    from sora.cycle import DecisionCycle
    from sora.manual import Manual
    from sora.memory import WorkingMemory


@dataclass(frozen=True)
class TickResult:
    """The decision surface for one cycle. Every phase strategy receives and returns one of these.
    Whatever's still None, DecisionCycle fills in by calling the next phase's own strategy — so a
    fully-decomposed configuration produces one field at a time, and a fused Situate can fill in
    step/invocation too, deciding the rest of the cycle in one call. Lives only for the duration of
    one tick() call — nothing persists across cycles, so there's no cache to key or invalidate.

    A freeform per-tick scratchpad for multi-call strategy configurations (e.g. a fused Situate
    passing notes to a separate, focused Act) is a foreseen addition, deferred until the first such
    configuration actually exists."""

    activity: Activity | None = None
    step: Step | None = None  # this cycle's decision — not the whole (possibly multi-step) Plan
    invocation: OperationInvocation | None = None


class ObserveStrategy(Protocol):
    async def observe(self, cycle: DecisionCycle) -> TickResult:
        """Mutates cycle.working (perceptions, messages) as a side effect — same as the default
        below. Default: mechanical, no model call, returns an empty TickResult(). An LLM-backed
        Observe is for interpreting raw perception itself (e.g. describing a camera snapshot), not
        for deciding the cycle — decision-chain fusion starts at Situate, not here."""
        ...


class ReflectStrategy(Protocol):
    async def reflect(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        """Decides whether this activity just completed or failed — deterministic or model-backed,
        depending on the application — and if so, summarizes and stores to episodic memory. On
        success, also stores activity.plan via cycle.procedural.store() so future activities with a
        similar goal can reuse it; on failure, it isn't stored. The completion judgment is
        synchronous — it must land before Situate selects, so a just-completed activity is never
        re-selected the same cycle — while the summarize/store side effects are dispatched
        asynchronously and never block the cycle; several activities may terminate in the same
        cycle. Passes `result` through, optionally adding to it. Default: performs the completion
        check and the store-on-success, leaves TickResult's other fields untouched. `cycle` is what
        makes these memory calls possible at all — previously missing from this Protocol despite
        the calls it was already documented as making."""
        ...


class SituateStrategy(Protocol):
    async def situate(
        self,
        activities: list[Activity],
        wm: WorkingMemory,
        cycle: DecisionCycle,
        result: TickResult,
    ) -> TickResult:
        """Selects the next activity and adjusts wm for it. Only called if result.activity is still
        None. Also responsible for activity creation: if wm.messages has one that doesn't correspond
        to any existing activity, invokes the internal _create_activity_ action (via cycle) before
        selecting. Head of the decision chain (Situate -> Reason -> Act) and the intended entry
        point for fusing the remaining phases into one model call — it runs after this cycle's
        percepts and messages are already in working memory. May additionally fill in
        step/invocation, short-circuiting Reason/Act."""
        ...


class ReasonStrategy(Protocol):  # pluggable; default targets 1 LLM call/cycle
    async def reason(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        """Only called if result.step is still None. Typical implementation: if activity.plan is
        already set and still valid, just read activity.plan.steps[activity.step_index] and
        advance the index — no model call. Otherwise, retrieve a cached Plan via
        cycle.procedural.retrieve() or infer a new one (the expensive path), reset step_index to
        0, and use its first Step. Deciding when a plan counts as invalidated is entirely up to
        the implementation. May additionally fill in invocation, short-circuiting Act — this is
        where the historical 'tool hallucination' risk lives if it does."""
        ...


class ActStrategy(Protocol):
    async def bind(
        self, step: Step, manual: Manual | None, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        """Only called if result.invocation is still None. Binds an abstract Step to a concrete,
        schema-conformant OperationInvocation. `cycle` is available for implementations that cache
        bindings (e.g. belief-state -> params) rather than re-deriving one every time."""
        ...


@dataclass(frozen=True)
class Strategies:  # bundles the five, so DecisionCycle.__init__ doesn't take five loose params
    observe: ObserveStrategy
    reflect: ReflectStrategy
    situate: SituateStrategy
    reason: ReasonStrategy
    act: ActStrategy


class DefaultObserveStrategy:
    """The runtime's built-in default — purely mechanical, no LLM."""

    async def observe(self, cycle: DecisionCycle) -> TickResult:
        wm = cycle.working
        for tool in wm.focused_tools.values():
            for prop in tool.observe():
                wm.perceptions.append(Percept(tool.id, PerceptKind.PROPERTY, prop, time.time()))
        async for source, signal in cycle.signal_sink.drain():
            wm.perceptions.append(Percept(source, PerceptKind.SIGNAL, signal, time.time()))
        async for invocation_id, ack in cycle.result_sink.drain():
            # Unambiguous 1:1 match: the invoke's own result resolves its activity automatically,
            # never as a Percept, no strategy involved (see Activities in README).
            for activity in wm.activities.values():
                if activity.pending_operation and activity.pending_operation.id == invocation_id:
                    activity.last_operation = ack
                    activity.pending_operation = None
                    activity.state = ActivityState.READY
                    break
        async for message in cycle.communication.receive():
            wm.messages.append(message)
        return TickResult()


class DefaultReflectStrategy:
    """Spike default: no completion judgment yet — a just-selected activity stays selectable.
    The real episodic/procedural store-on-success logic isn't built yet."""

    async def reflect(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        return result


class DefaultSituateStrategy:
    """Spike default: pick the first ready activity; no activity-creation-from-messages yet."""

    async def situate(
        self,
        activities: list[Activity],
        wm: WorkingMemory,
        cycle: DecisionCycle,
        result: TickResult,
    ) -> TickResult:
        if result.activity is not None or not activities:
            return result
        return replace(result, activity=activities[0])


class DefaultActStrategy:
    """Spike default: bind an ``invoke`` Step straight to an OperationInvocation, splitting the
    tool_id/operation_name routing keys out of the bound params."""

    async def bind(
        self, step: Step, manual: Manual | None, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        params = {k: v for k, v in step.params.items() if k not in (TOOL_ID, OPERATION_NAME)}
        invocation = OperationInvocation(
            tool_id=step.params[TOOL_ID],
            operation_name=step.params[OPERATION_NAME],
            params=params,
        )
        return replace(result, invocation=invocation)
