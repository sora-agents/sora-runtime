"""Undeclared-relevance judgement and amendment handling."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Protocol

from sora._strategies.interaction import (
    _await_input,
)
from sora.action import (
    _spawn_tracked,
)
from sora.activity import Activity
from sora.memory import (
    PerceptSnapshot,
)
from sora.perception import Percept
from sora.types import (
    Change,
    RelevanceCandidate,
    changes_of,
    path_matches,
)

if TYPE_CHECKING:
    from sora.cycle import DecisionCycle
    from sora.memory import WorkingMemory

log = logging.getLogger("sora.strategies")


class RelevanceJudge(Protocol):
    def consider(self, cycle: DecisionCycle) -> Awaitable[None]:
        """Called on an IDLE tick — one where Situate selected nothing, so either there is no
        schedulable activity or everything schedulable is already awaiting a model (ADR-0026).

        Scheduling, not triggering: an unclaimed change makes this eligible the moment it lands, but
        it never runs in preference to an activity that could actually advance. Implementations must
        return promptly — fire any model call off-cycle and apply the result on a later call — so an
        idle tick stays responsive to arriving signals.
        """
        ...


class DefaultRelevanceJudge:
    """Undeclared-relevance recovery (ADR-0026): notice that a change bears on work that already
    finished, ask the user, and amend rather than reopen.

    Deliberately **opt-in**. It spends a model call on an unverifiable judgement and, when it fires,
    interrupts a person — and its own ADR records that with nobody available to ask, the safe
    degradation is to not act. An unattended run should therefore get the declared-condition layer
    and nothing else unless someone chose otherwise.

    Its input is only what the declared gates left unclaimed, so every condition the planner learns
    to declare removes work from here.
    """

    def __init__(self, *, window: int = 10, max_asks: int = 3) -> None:
        # Neither number is principled, which is why both are settings rather than constants.
        # `window` too small silently drops old-but-live commitments and too large grows the prompt
        # and the error rate together; `max_asks` too high pesters the user until they stop reading
        # and too low reproduces the miss this exists to prevent.
        self._window = window
        self._max_asks = max_asks
        self._mark = 0  # high-water over wm.signals_appended — this judge's own, like any waiter's
        self._asks = 0
        self._in_flight = False
        self._result: RelevanceCandidate | None = None
        self._declined: set[tuple[str, str]] = set()
        self._tasks: set[asyncio.Task[None]] = set()

    async def consider(self, cycle: DecisionCycle) -> None:
        # Apply first: a parked result is applied INSIDE the tick, so every mutation of working
        # memory happens on-cycle. The background call only ever sets a field — the same discipline
        # the sinks enforce for infer/ground/invoke (ADR-0021), with a one-slot mailbox instead of
        # a queue because at most one judgement is ever in flight.
        if self._result is not None:
            candidate, self._result = self._result, None
            await self._amend(cycle, candidate)
            return
        if self._in_flight or self._asks >= self._max_asks:
            return
        unclaimed = self._unclaimed(cycle.working)
        # Advance the mark whether or not anything is judged: a change that opened no gate and did
        # not reach a call has still been considered, and re-considering it every idle tick would
        # turn an idle agent into a spend loop.
        self._mark = cycle.working.signals_appended
        if not unclaimed:
            return
        episodes = await cycle.episodic.consult_recent(self._window)
        episodes = [e for e in episodes if isinstance(e, dict) and e.get("activity_id")]
        if not episodes:
            return
        # Paired with the source that reported them, exactly as the declared-condition gate does:
        # a `Change` names the path that moved but not the tool it moved on, and the judgement
        # needs both to dereference the ids back into records (ProceduralMemory.render_changes).
        changes: list[tuple[str, Change]] = []
        for percept in unclaimed:
            changes.extend((percept.source, change) for change in changes_of(percept.payload))
        observed = PerceptSnapshot(
            list(cycle.working.properties.values()), list(cycle.working.signals)
        )
        self._in_flight = True
        _spawn_tracked(self._tasks, self._call(cycle, episodes, changes, observed))

    async def _call(
        self,
        cycle: DecisionCycle,
        episodes: list[Any],
        changes: list[tuple[str, Change]],
        observed: PerceptSnapshot,
    ) -> None:
        try:
            candidate = await cycle.procedural.judge_relevance(episodes, changes, observed)
        except Exception:  # noqa: BLE001 — a failed judgement means "nothing follows up", not a crash
            log.exception("relevance: judgement failed")
            candidate = None
        finally:
            self._in_flight = False
        self._result = candidate

    def _unclaimed(self, wm: WorkingMemory) -> list[Percept]:
        """Signals past this judge's mark that opened NO declared gate.

        The subtraction that defines this layer's input. A change claimed by some activity's
        pending condition is layer 1's business and is never offered here — an accepted false
        negative, since that same change might also have borne on an unrelated finished activity,
        but the alternative is judging every signal against every terminated activity.
        """
        watches = [
            state.condition.watch
            for activity in wm.activities.values()
            for state in activity.pending_conditions
        ]
        first_seq = wm.signals_appended - len(wm.signals)
        out: list[Percept] = []
        for offset, percept in enumerate(wm.signals):
            if first_seq + offset < self._mark:
                continue
            signal = percept.payload
            claimed = any(
                w.signal_name == signal.name
                and (w.source is None or percept.source == w.source)
                # `path`, deliberately NOT `kind`, unlike the eligibility gate: `kind` says what may
                # *open* a gate, not what a gate is answerable for. Narrowing here inverts the
                # purpose it was added for — a watch declared `added` on a collection the agent also
                # deletes from is exactly the shape `kind` exists to spare a judge call, and reading
                # it here hands that same delete to *this* judge instead, which additionally
                # interrupts a person. The wider test keeps the saving where it was won.
                and path_matches(w.path, changes_of(signal))
                for w in watches
            )
            if not claimed:
                out.append(percept)
        return out

    async def _amend(self, cycle: DecisionCycle, candidate: RelevanceCandidate) -> None:
        """Create the amending activity — born BLOCKED on an InputWait, so the user is asked before
        the agent acts on a goal nobody stated.

        A NEW activity, never the terminated one revived: an episode is a historical claim about
        what was attempted and how it ended, and editing one to make it current would turn
        `succeeded` into a retrospective lie in the very record the agent learns from.

        Through `_await_input` like every other breaker, because the wait and the *asking* are two
        halves of one act: parking on a question that was never delivered would leave this layer
        silently inert, and would let the user's next unrelated message be read by
        `_resume_on_input` as consent to an amendment they were never shown.

        Consent needs no mechanism of its own. `_resume_on_input` already clears an InputWait on a
        user Message, drops the (empty) plan, and re-infers with the reply and history visible — so
        a decline is answered by the same path a go-ahead is.
        """
        if (candidate.episode_id, candidate.goal) in self._declined:
            return
        self._declined.add((candidate.episode_id, candidate.goal))
        self._asks += 1
        activity = Activity(
            id=uuid.uuid4().hex,
            goal=candidate.goal,
            # The amendment points back at what it amends; the original stays terminated and its
            # episode untouched.
            context={"amends": candidate.episode_id},
        )
        # Ask first, register after: nothing else runs between the two, and an activity is never
        # visible in working memory in any state but the blocked one it is born in.
        await _await_input(cycle, activity, candidate.question)
        cycle.working.activities[activity.id] = activity
        log.info(
            "relevance: proposing amendment to episode %s -> %r",
            candidate.episode_id,
            candidate.goal,
        )
