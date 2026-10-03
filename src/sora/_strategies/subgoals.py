"""Mechanical subgoal expansion and recursion safeguards."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from sora.activity import Activity
from sora.data_ops import (
    _resolve_collection,
)
from sora.memory import (
    step_from_raw,
)
from sora.perception import Percept
from sora.references import (
    _REF_BIND,
    _REF_PATH,
    _walk_path,
)
from sora.types import (
    CompletedOperation,
    PropertyReadMeter,
    Step,
)

if TYPE_CHECKING:
    pass

log = logging.getLogger("sora.strategies")

# --- sub-goals: mechanical fan-out over a collection (ADR-0022) -----------------------------------
# A `subgoal` Step with mode="mechanical" is expanded in Reason into one concrete step per element
# of a run-time collection — the count is len(data), not a model guess (the RentAFlat "for each"
# fix). _SUBGOAL_RUNNING / _SUBGOAL_SPLICED are the two outcomes _subgoal reports to reason():
# a deliberative sub-goal fired _infer_ and is RUNNING (return, no step); a mechanical one spliced
# its expansion into the plan in place (re-loop and read the first expanded step); a deliberative
# one the loop-guard refused pauses the activity to await input (no step) -> _SUBGOAL_HALTED;
# a mechanical one whose collection could not be read dropped the plan (no step) -> _SUBGOAL_DEFECT.
_SUBGOAL_RUNNING = object()

_SUBGOAL_SPLICED = object()

_SUBGOAL_HALTED = object()

_SUBGOAL_DEFECT = object()


# Circuit breaker for runaway deliberative sub-goal recursion (ADR-0022's deferred overflow valve,
# pulled forward). Synthesis-as-selection has no termination guarantee an *authored* plan library
# has: the model can satisfy "plan for goal G" by emitting a plan whose body is another deliberative
# sub-goal for ~G, deferring instead of reducing, and recurse until a budget (or credit) runs out.
# Two mechanical detectors, tripped before the _infer_ spend: a depth cap on the intention stack
# (configurable per DefaultReasonStrategy, wired from agent.yaml's `max_subgoal_depth`), and
# token-overlap against the ancestor sub-goals — a new sub-goal whose tokens are largely contained
# in one still on the stack is re-stating it, not reducing. Overlap (|A&B| / min), not Jaccard: the
# observed regress *elaborates* the same goal (piling on qualifiers), which grows the union and
# sinks Jaccard while the core token set stays contained, so containment is what catches the reword.
# Tripping pauses to await-input (ADR-0020) rather than terminating, so a deep-but-legitimate task
# can be redirected, not killed. Both are coarse backstops; the real fix is making the common
# map/filter/distinct shapes expressible without deliberation at all.
_DEFAULT_MAX_SUBGOAL_DEPTH = 4

_SUBGOAL_GOAL_OVERLAP = 0.7


def _goal_token_overlap(a: str, b: str) -> float:
    """Token overlap coefficient over two goal strings — ``|A&B| / min(|A|, |B|)``, 1.0 when the
    smaller token set is contained in the larger, 0.0 disjoint. Cheap and deterministic (no model
    call). Overlap, not Jaccard: the non-reducing recursion re-states an ancestor's goal with extra
    qualifiers, growing the union (which sinks Jaccard) while the core stays contained — containment
    is the signal that survives the reword."""
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _ancestor_subgoal_goals(activity: Activity) -> list[str]:
    """The goals of the deliberative sub-goals still suspended on the intention stack — each parent
    frame's ``(plan, idx)`` points back at the ``subgoal`` step that pushed it. The root
    ``activity.goal`` is deliberately excluded: the first decomposition legitimately shares its
    vocabulary, so comparing against it would false-trip a single, valid refinement."""
    goals: list[str] = []
    for plan, idx, _ in activity.parent_frames:
        if 0 <= idx < len(plan.steps):
            goal = plan.steps[idx].params.get("goal")
            if isinstance(goal, str):
                goals.append(goal)
    return goals


def _pursued_goal(activity: Activity) -> str | None:
    """The goal a fired condition's ``then`` plan serves, or ``None`` if the active plan is not one.

    ``Activity.pursued_goals`` keys each goal by the frame depth it was installed at, because a
    ``then`` pushes no frame and so cannot be told apart by the stack alone. Matching on depth is
    what keeps a record from answering for a plan it does not belong to: a deeper entry means an
    authored sub-goal was entered from inside the ``then`` and its own frame records its goal, a
    shallower one means a ``then`` is still owed further out — either beneath an authored sub-goal
    or beneath a nested ``then`` — and is not what the active plan serves."""
    return activity.pursued_goals.get(len(activity.parent_frames))


def _within_condition_body(activity: Activity) -> bool:
    """Whether the activity is executing a fired condition's ``then`` body — either at the depth
    the ``then`` was installed at, or inside an authored sub-goal entered from within it.

    Deliberately broader than ``_pursued_goal``, and the difference is the whole point: that one
    answers "is the *active plan* the ``then``", which goes false the moment the body enters a
    sub-goal of its own, while this answers "is condition-driven work still on the stack at or
    above us". A zero-step fan-out nested two levels inside a watch body is no more a gap in the
    user's work than one sitting directly in it, so the caller that gates on this needs the wider
    question.

    ``<=`` rather than ``==`` is what buys that: an entry deeper than us was installed at a depth a
    discard has since popped out past — which ``pursued_goals`` deliberately does not clear, so this
    has to exclude it rather than trust the field's mere non-emptiness — while one at or above our
    depth is a ``then`` still owed here or beneath us. Any single such entry answers yes, so a
    nested ``then`` whose own entry is discharged on exhaustion is still inside its outer one."""
    return any(depth <= len(activity.parent_frames) for depth in activity.pursued_goals)


def _active_frame_goal(activity: Activity) -> str | None:
    """The sub-goal the activity is currently executing *inside*, or ``None`` at the top level.

    Read off the innermost suspended frame — the ``subgoal`` step that pushed it — the same place
    ``_ancestor_subgoal_goals`` reads from. It answers "what is the goal of the plan that would
    replace the active one", which is the activity's own goal only when the stack is empty: a
    frame's sub-plan is inferred against the sub-goal, never against the user's request, and a
    replan that forgot the distinction would hand the planner the user's words under a prompt
    asserting they are not the user's (see ``_render_goal_provenance``). ``None`` also for a frame
    whose sub-goal step is unreadable, so the caller falls back to the activity's goal rather than
    silently planning for some other frame's."""
    if not activity.parent_frames:
        return None
    plan, index, _mark = activity.parent_frames[-1]
    if not 0 <= index < len(plan.steps):  # defensive: a frame whose parent was replanned under it
        return None
    goal = plan.steps[index].params.get("goal")
    return goal if isinstance(goal, str) and goal else None


def _substitute_bindings(obj: Any, name: str, element: Any) -> Any:
    """Replace every ``{"$bind": name, "path": ...}`` in a template with the value at that path of
    the current loop ``element``, recursively. Only the named binding is substituted;
    ``$from``/``$decide`` references (and a ``$bind`` for a different name) pass through untouched,
    to be grounded later by the ordinary Reason path. A path that doesn't resolve substitutes a
    ``None`` — which the Act required-param guard skips, not a literal ``$bind`` dict reaching the
    tool."""
    if isinstance(obj, dict):
        if obj.get(_REF_BIND) == name:
            try:
                return _walk_path(element, obj.get(_REF_PATH, ""))
            except (KeyError, IndexError, TypeError, ValueError):
                log.warning("subgoal: $bind path %r did not resolve against %r", obj, element)
                return None
        return {k: _substitute_bindings(v, name, element) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_bindings(v, name, element) for v in obj]
    return obj


def _expand_mechanical(
    step: Step,
    history: list[CompletedOperation],
    bindings: dict[str, Any] | None = None,
    properties: dict[tuple[str, str], Percept] | None = None,
    meter: PropertyReadMeter | None = None,
) -> tuple[list[Step], str | None]:
    """Fan a mechanical sub-goal out to one concrete ``Step`` per element of its ``in`` collection,
    the element substituted for ``{"$bind": "<as>"}`` in its ``template``. The ``in`` collection may
    be a ``$from`` (history), a ``$bind`` (a data-op output binding, e.g. a filtered shortlist), or
    a ``$prop`` (bulk state an adapter publishes as an observable property).

    Returns the expansion and a ``defect``. An empty collection expands to no steps and *is* the
    answer — the sub-goal had nothing to do and the plan should continue. A collection that could
    not be read expands to no steps too, but means the opposite, so it comes back as a defect for
    the caller to replan on. Collapsing the two is how "cancel each event on Saturday" quietly
    became a no-op in a real run while the event sat in history, correctly fetched, all along."""
    elements, defect = _resolve_collection(
        step.params.get("in"), history, bindings, properties, meter
    )
    if defect is not None:
        return [], defect
    if not elements:
        return [], None
    loop_var = step.params.get("as", "")
    template = step.params.get("template", {})
    return [
        step_from_raw(_substitute_bindings(template, loop_var, element)) for element in elements
    ], None
