"""Strategy protocols and the decision-cycle result contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from sora.activity import Activity
from sora.types import (
    OperationInvocation,
    Step,
)

if TYPE_CHECKING:
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
        depending on the application — and if so, summarizes and stores to episodic memory. (The
        default does NOT auto-cache the completed plan to procedural memory — replaying a stored
        plan verbatim is unsound; distilling reusable procedures from episodes is future work.) The
        completion judgment is
        synchronous — it must land before Situate selects, so a just-completed activity is never
        re-selected the same cycle — while the summarize/store side effects are dispatched
        asynchronously and never block the cycle; several activities may terminate in the same
        cycle. Passes `result` through, optionally adding to it. Default: performs the completion
        check and the store-on-success, leaves TickResult's other fields untouched. `cycle` is what
        makes these memory calls possible at all — previously missing from this Protocol despite
        the calls it was already documented as making."""
        ...

    def failed(self, activity: Activity) -> bool:
        """This strategy's own judgment of whether `activity` has failed — a *judgment call*, not
        a fact recorded on `Activity` itself, since a different ReflectStrategy may define failure
        differently (e.g. from a signal, or a partial-success rule) than the default's "resolved
        operation, not ok" rule. Exposed on the Protocol (not just the default) so callers outside
        the decision cycle — a reporting hook, a test assertion — go through whichever strategy is
        actually configured rather than re-deriving the rule themselves."""
        ...


class SituateStrategy(Protocol):
    async def situate(
        self,
        activities: list[Activity],
        wm: WorkingMemory,
        cycle: DecisionCycle,
        result: TickResult,
    ) -> TickResult:
        """Selects the next activity and adjusts wm for it. Always runs — unlike Reason/Act it is
        not gated on its own output field, because adjusting wm (selecting tools, loading/unloading
        manuals, filtering percepts) must reflect this cycle's fresh percepts even for an
        already-selected activity. Selects only if result.activity is still None; a pre-set
        selection (uncommon — e.g. an Observe that pins the activity handling a critical signal) is
        respected and situated, not overridden. Also responsible for activity creation: if
        wm.messages includes a new goal delegation, invokes the internal _create_activity_ action
        (via cycle) before selecting. Head of the decision chain (Situate -> Reason -> Act) and the
        intended entry point for fusing the remaining phases into one model call — it runs after
        this cycle's percepts and messages are already in working memory. May additionally fill in
        step/invocation, short-circuiting Reason/Act (those forward-fusion gates remain; only
        Situate's own activity gate is removed)."""
        ...


class ActivitySelectionStrategy(Protocol):
    async def select(
        self, ready: list[Activity], wm: WorkingMemory, cycle: DecisionCycle
    ) -> Activity | None:
        """Picks the activity to progress this cycle from the ready set (empty -> None). A
        scheduling policy, not a phase: it decides *which* ready activity runs, nothing else — the
        caller (Situate) folds the pick into TickResult; fusing step/invocation stays a full
        SituateStrategy concern. `async` + the `cycle` handle are for a richer policy (priority,
        aging, deadlines, or an LLM-based scheduler) that consults memory or a model; the mechanical
        default consults neither."""
        ...


class FocusPolicy(Protocol):
    def attend(self, wm: WorkingMemory) -> set[str]:
        """Tool ids the agent should be attending to this cycle.

        Pure and set-valued: the caller owns the diff against `wm.focused_tools` and performs the
        focus/unfocus, so a policy never touches the environment and can be unit-tested as a
        function. A scheduling-style sub-strategy, not a phase — the same shape as
        `ActivitySelectionStrategy` (ADR-0016), and injected the same way (a constructor argument
        to the default Observe strategy, not an `agent.yaml` key)."""
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
        """Only called if result.invocation is still None. This is *parameter binding*: grounding
        an abstract Step into a concrete, schema-conformant OperationInvocation (the tool-
        hallucination-prone step — where "email the boss" becomes validated `{to, subject, ...}`).
        Distinct from a *protocol binding* (WoT forms/security, an MCP session), which is how the
        adapter's Tool actually reaches the instance and never surfaces here — see ADR-0015. `cycle`
        is available for implementations that cache bindings (e.g. belief-state -> params) rather
        than re-deriving one every time."""
        ...


@dataclass(frozen=True)
class Strategies:  # bundles the five, so DecisionCycle.__init__ doesn't take five loose params
    observe: ObserveStrategy
    reflect: ReflectStrategy
    situate: SituateStrategy
    reason: ReasonStrategy
    act: ActStrategy
