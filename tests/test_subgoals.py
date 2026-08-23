"""Sub-goals — the plan's sole recursion primitive (ADR-0022).

A sub-goal is a ``Step`` with ``next_action == "subgoal"``. Reaching one either:

* **mechanical** — fans out ``len(collection)`` concrete copies of a ``template`` step, one per
  element of a ``$from`` collection, the loop element substituted for ``{"$bind": "<as>"}``. No
  model call; the count is ``len(data)``, not a model guess. This is the fix for the RentAFlat
  "for each" collapse (a multi-item step that ran exactly once).
* **deliberative** — fires ``_infer_`` mid-plan for the sub-goal's own goal; the synthesized
  sub-plan runs as a pushed frame on the ``Activity`` (``parent_frames`` — the intention stack
  generalizing ``plan``+``step_index``); the parent resumes at the step after the sub-goal when the
  sub-plan exhausts.

See ADR-0022. Grounding of an expanded step's remaining references is the ordinary Reason path
(test_grounding.py); this file pins the fan-out, the mid-plan infer, and the frame stack.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from fakes import FakeAdapter, FakeLLMClient, FakeTool, FakeWorkspace, plan_json
from sora.action import InferAction, default_action_registry, invoke_step
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, Tool, WorkspaceOrigin
from sora.memory import (
    PLAN_SYSTEM_PROMPT,
    EpisodicMemory,
    FileMemoryBackend,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
    step_from_raw,
)
from sora.perception import Message
from sora.strategies import (
    _DEFAULT_MAX_SUBGOAL_DEPTH,
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReasonStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    Strategies,
    TickResult,
)
from sora.transport import MessageTransport
from sora.types import (
    CompletedOperation,
    InputWait,
    OperationAck,
    OperationInvocation,
    Plan,
    Step,
    SupersededPlan,
)

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")


def _history(operation_name: str, result: object, *, ok: bool = True) -> CompletedOperation:
    return CompletedOperation(
        OperationInvocation("realestate", operation_name, {}), OperationAck(ok=ok, result=result)
    )


class _NullTransport:
    async def send(self, to: str, content: dict[str, object]) -> None: ...

    def receive(self) -> AsyncIterator[Message]:
        async def _drain() -> AsyncIterator[Message]:
            return
            yield  # pragma: no cover — never-yielding async generator

        return _drain()


def _cycle(
    tmp_path: Path, procedural: ProceduralMemory, tool: Tool
) -> tuple[DecisionCycle, WorkingMemory, EnvironmentRegistry]:
    registry = EnvironmentRegistry(
        adapters={_ORIGIN: FakeAdapter("fake", FakeWorkspace("ws", _ORIGIN, [tool]))}
    )
    working = WorkingMemory(registry=registry)
    transport: MessageTransport = _NullTransport()
    strategies = Strategies(
        observe=DefaultObserveStrategy(),
        reflect=DefaultReflectStrategy(),
        situate=DefaultSituateStrategy(),
        reason=DefaultReasonStrategy(),
        act=DefaultActStrategy(),
    )
    cycle = DecisionCycle(
        strategies=strategies,
        communication=transport,
        actions=default_action_registry(),
        registry=registry,
        working=working,
        semantic=SemanticMemory(FileMemoryBackend(tmp_path / "semantic")),
        procedural=procedural,
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "episodic")),
    )
    return cycle, working, registry


def _no_llm_procedural(tmp_path: Path) -> ProceduralMemory:
    # Mechanical fan-out makes no model call, so a procedural with no LLM is enough (and would raise
    # if the fan-out ever escalated, proving it didn't).
    return ProceduralMemory(FileMemoryBackend(tmp_path / "proc"))


def _mechanical_subgoal(collection_path: str = "") -> Step:
    return Step(
        next_action="subgoal",
        params={
            "goal": "save each apartment",
            "mode": "mechanical",
            "in": {"$from": "search_apartments", "path": collection_path},
            "as": "apt",
            "template": {
                "action": "invoke",
                "tool_id": "realestate",
                "operation_name": "save_apartment",
                "params": {"apartment_id": {"$bind": "apt", "path": "id"}},
            },
        },
    )


# --------------------------------------------------------------------------------------------------
# Mechanical fan-out — len(collection) concrete steps, no model call
# --------------------------------------------------------------------------------------------------


async def test_mechanical_subgoal_fans_out_one_step_per_element(tmp_path: Path) -> None:
    tool = FakeTool("realestate", invoke_results={"save_apartment": {"saved": True}})
    cycle, working, registry = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    await registry.join(_ORIGIN)
    skeleton = Plan(id="p", goal="shortlist", steps=[_mechanical_subgoal()])
    activity = Activity(
        id="a",
        goal="shortlist",
        context={},
        plan=skeleton,
        step_index=0,
        history=[_history("search_apartments", [{"id": "a1"}, {"id": "a2"}, {"id": "a3"}])],
    )
    working.activities["a"] = activity

    # Each reason() call yields exactly one concrete invoke — the sub-goal fanned out to three.
    ids: list[str] = []
    for _ in range(3):
        result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
        assert result.step is not None
        assert result.step.next_action == "invoke"
        ids.append(result.step.params["apartment_id"])
    assert ids == ["a1", "a2", "a3"]  # len(collection) invokes, each element's id bound

    # Plan exhausted after the three expanded steps — no fourth step, no leftover sub-goal.
    exhausted = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert exhausted.step is None
    assert activity.plan is not None
    assert all(s.next_action == "invoke" for s in activity.plan.steps)

    # The fan-out is a per-run splice: the original skeleton object is never mutated.
    assert skeleton.steps[0].next_action == "subgoal"
    assert activity.plan is not skeleton


async def test_empty_collection_expands_to_no_steps(tmp_path: Path) -> None:
    tool = FakeTool("realestate", invoke_results={"save_apartment": {"saved": True}})
    cycle, working, registry = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    await registry.join(_ORIGIN)
    activity = Activity(
        id="a",
        goal="shortlist",
        context={},
        plan=Plan(id="p", goal="shortlist", steps=[_mechanical_subgoal()]),
        step_index=0,
        history=[_history("search_apartments", [])],  # nothing to iterate
    )
    working.activities["a"] = activity

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert result.step is None  # zero elements -> the sub-goal simply vanishes
    assert activity.plan is not None
    assert activity.plan.steps == []


async def test_fan_out_resolves_a_qualified_operation_reference(tmp_path: Path) -> None:
    """The planner wrote ``insim:are/Calendar.get_calendar_events_from_to`` — the form its own tool
    catalog uses — and the fan-out silently produced nothing. The qualified spelling resolves."""
    tool = FakeTool("realestate", invoke_results={"save_apartment": {"saved": True}})
    cycle, working, registry = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    await registry.join(_ORIGIN)
    subgoal = _mechanical_subgoal()
    subgoal.params["in"] = {"$from": "realestate.search_apartments", "path": ""}
    activity = Activity(
        id="a",
        goal="shortlist",
        context={},
        plan=Plan(id="p", goal="shortlist", steps=[subgoal]),
        step_index=0,
        history=[_history("search_apartments", [{"id": "a1"}, {"id": "a2"}])],
    )
    working.activities["a"] = activity

    ids = []
    for _ in range(2):
        result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
        assert result.step is not None
        ids.append(result.step.params["apartment_id"])
    assert ids == ["a1", "a2"]
    assert activity.plan is not None  # no defect: the reference was readable all along


async def test_fan_out_reads_through_a_paginated_envelope(tmp_path: Path) -> None:
    """The exact shape that cost a run a 220-second replan: ARE's windowed list operations return
    ``{'events': [...], 'range': ..., 'total': N}`` while declaring a bare return type, so a
    ``$from`` with no ``path`` — which is what the declared shape tells the planner to write —
    landed on the envelope. Demanding a ``path`` was punishing the planner for believing the
    catalog. The payload beside pagination metadata is unambiguous enough to just take."""
    tool = FakeTool("realestate", invoke_results={"save_apartment": {"saved": True}})
    cycle, working, registry = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    await registry.join(_ORIGIN)
    wrapped = {"apartments": [{"id": "a1"}], "range": "(0, 1)", "total": 1}
    activity = Activity(
        id="a",
        goal="shortlist",
        context={},
        plan=Plan(id="p", goal="shortlist", steps=[_mechanical_subgoal()]),
        step_index=0,
        history=[_history("search_apartments", wrapped)],
    )
    working.activities["a"] = activity

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step is not None
    assert result.step.params["apartment_id"] == "a1"
    assert activity.plan is not None  # no defect: the envelope was readable
    assert activity.replan_trail == []


async def test_fan_out_over_a_record_with_a_list_field_still_replans_naming_the_path(
    tmp_path: Path,
) -> None:
    """The boundary the envelope tier must not cross. A record carrying one list field has the same
    shape as the envelope — one list, scalar siblings — and reading it as a collection would fan
    out over an event's *attendees* instead of over events. Only pagination-metadata siblings make
    it an envelope; anything else is still refused, with the defect naming the path to add."""
    tool = FakeTool("realestate", invoke_results={"save_apartment": {"saved": True}})
    cycle, working, registry = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    await registry.join(_ORIGIN)
    record = {"event_id": "e1", "title": "Standup", "attendees": [{"id": "a1"}]}
    activity = Activity(
        id="a",
        goal="shortlist",
        context={},
        plan=Plan(id="p", goal="shortlist", steps=[_mechanical_subgoal()]),
        step_index=0,
        history=[_history("search_apartments", record)],
    )
    working.activities["a"] = activity

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step is None
    assert activity.plan is None  # dropped, rather than "this sub-goal had nothing to do"
    defect = activity.replan_trail[-1]
    assert defect is not None
    assert "'attendees'" in defect and "path" in defect


async def test_unresolvable_collection_replans_rather_than_vanishing(tmp_path: Path) -> None:
    """The `in` reference names an operation that never ran, so the runtime cannot tell how many
    elements there are — which is *not* the same as knowing there are none. Splicing in zero steps
    would silently assert the sub-goal had nothing to do; a real run lost three calendar
    cancellations that way. The plan is dropped instead, carrying a defect the planner can use."""
    tool = FakeTool("realestate", invoke_results={"save_apartment": {"saved": True}})
    cycle, working, registry = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    await registry.join(_ORIGIN)
    activity = Activity(
        id="a",
        goal="shortlist",
        context={},
        plan=Plan(id="p", goal="shortlist", steps=[_mechanical_subgoal()]),
        step_index=0,
        history=[],  # search_apartments never ran -> the `in` $from cannot resolve
    )
    working.activities["a"] = activity

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step is None  # nothing is committed off an expansion the runtime could not build
    assert activity.plan is None  # dropped -> Reason re-infers next cycle
    defect = activity.replan_trail[-1]
    assert defect is not None
    # Names the sub-goal (which one failed) and what to do (the reference produced nothing yet).
    assert "save each apartment" in defect
    assert "no operation has run" in defect


async def test_bind_substitutes_a_nested_field_of_the_loop_element(tmp_path: Path) -> None:
    tool = FakeTool("realestate", invoke_results={"save_apartment": {"saved": True}})
    cycle, working, registry = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    await registry.join(_ORIGIN)
    subgoal = Step(
        next_action="subgoal",
        params={
            "goal": "save each",
            "mode": "mechanical",
            "in": {"$from": "search_apartments", "path": ""},
            "as": "apt",
            "template": {
                "action": "invoke",
                "tool_id": "realestate",
                "operation_name": "save_apartment",
                "params": {"apartment_id": {"$bind": "apt", "path": "listing.ref"}},
            },
        },
    )
    activity = Activity(
        id="a",
        goal="shortlist",
        context={},
        plan=Plan(id="p", goal="shortlist", steps=[subgoal]),
        step_index=0,
        history=[_history("search_apartments", [{"listing": {"ref": "R1"}}])],
    )
    working.activities["a"] = activity

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert result.step is not None
    assert result.step.params["apartment_id"] == "R1"  # $bind walked the nested path


# --------------------------------------------------------------------------------------------------
# Deliberative sub-goal — mid-plan infer, sub-plan on a pushed frame
# --------------------------------------------------------------------------------------------------


async def test_deliberative_subgoal_infers_against_the_subgoal_goal(tmp_path: Path) -> None:
    subplan = plan_json({"action": "send", "to": "user", "content": {"text": "done"}})
    llm = FakeLLMClient(subplan)
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm)
    tool = FakeTool("realestate", invoke_results={"save_apartment": {"saved": True}})
    cycle, working, registry = _cycle(tmp_path, procedural, tool)
    await registry.join(_ORIGIN)
    subgoal = Step(
        next_action="subgoal",
        params={"goal": "notify each relative", "mode": "deliberative"},
    )
    parent = Plan(id="p", goal="reconcile the shortlist", steps=[subgoal])
    activity = Activity(id="a", goal="reconcile the shortlist", context={}, plan=parent)
    working.activities["a"] = activity

    # Reaching the sub-goal fires _infer_ off-cycle: RUNNING, no step, index not advanced.
    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert result.step is None
    state = activity.state
    assert state is ActivityState.RUNNING
    assert activity.pending_inference is not None
    assert activity.pending_inference.kind == "subgoal"
    assert activity.step_index == 0

    await asyncio.sleep(0)  # let the background _infer_ task run and push its result
    # It planned for the SUB-goal's goal, not the parent activity's goal.
    assert any("notify each relative" in prompt for _system, prompt in llm.calls)

    # Observe resolves the sub-plan: push a frame and enter it, parent saved at the sub-goal index.
    await DefaultObserveStrategy().observe(cycle)
    assert len(activity.parent_frames) == 1
    assert activity.parent_frames[0] == (parent, 0)
    assert activity.plan is not None
    assert activity.plan is not parent
    assert activity.plan.goal == "notify each relative"
    assert activity.step_index == 0
    state = activity.state
    assert state is ActivityState.READY


async def test_a_parked_superseded_plan_never_reaches_a_subgoal_prompt(tmp_path: Path) -> None:
    # A superseded bundle is context for re-planning the *activity's* goal, and the prompt section
    # introduces it as "a previous plan for this goal". A sub-goal was never planned, let alone
    # abandoned, so rendering it there would describe a plan that does not exist (ADR-0024). The
    # bundle can still be parked here because a cached-plan install consumes no inference.
    subplan = plan_json({"action": "send", "to": "user", "content": {"text": "done"}})
    llm = FakeLLMClient(subplan)
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm)
    tool = FakeTool("realestate", invoke_results={"save_apartment": {"saved": True}})
    cycle, working, registry = _cycle(tmp_path, procedural, tool)
    await registry.join(_ORIGIN)
    subgoal = Step(
        next_action="subgoal",
        params={"goal": "notify each relative", "mode": "deliberative"},
    )
    parent = Plan(id="p", goal="reconcile the shortlist", steps=[subgoal])
    activity = Activity(id="a", goal="reconcile the shortlist", context={}, plan=parent)
    dropped = Plan(id="old", goal="reconcile the shortlist", steps=[invoke_step("t", "stale_op")])
    activity.superseded = SupersededPlan(plan=dropped, step_index=0, parent_frames=[])
    working.activities["a"] = activity

    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    await asyncio.sleep(0)  # let the background _infer_ task run

    _system, prompt = llm.calls[-1]
    assert "notify each relative" in prompt  # it is the sub-goal's prompt
    assert "previous plan for this goal was abandoned" not in prompt
    assert "stale_op" not in prompt
    # Still parked on the real activity: only the sub-goal's copy drops it, so the replacement
    # top-level plan (if one is ever inferred) has not been robbed of its context.
    assert activity.superseded is not None


async def test_entered_sub_plan_is_traced_at_debug(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A sub-plan is a plan too: entering one logs its body and the depth it entered at, so a trace
    of a decomposed run shows the sub-plan's steps and not just that a frame was pushed."""
    subplan = plan_json({"action": "send", "to": "user", "content": {"text": "done"}})
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=FakeLLMClient(subplan))
    cycle, working, registry = _cycle(tmp_path, procedural, FakeTool("realestate"))
    await registry.join(_ORIGIN)
    subgoal = Step(
        next_action="subgoal", params={"goal": "notify each relative", "mode": "deliberative"}
    )
    parent = Plan(id="p", goal="reconcile the shortlist", steps=[subgoal])
    activity = Activity(id="a", goal="reconcile the shortlist", context={}, plan=parent)
    working.activities["a"] = activity

    with caplog.at_level(logging.DEBUG, logger="sora.strategies"):
        await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
        await asyncio.sleep(0)  # let the off-cycle _infer_ push its result
        await DefaultObserveStrategy().observe(cycle)

    entered = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.DEBUG and r.getMessage().startswith("observe: sub-plan")
    ]
    assert len(entered) == 1
    assert "nested under 1 frame(s)" in entered[0]  # frames suspended below, not the recursion cap
    assert "0: send" in entered[0]


async def test_fanned_out_plan_body_is_traced_at_debug(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A mechanical fan-out rewrites the plan in place, so logging only the expansion's size would
    leave the trace's last plan body the pre-splice one — and every step index printed afterwards
    would refer to a plan the log never showed."""
    tool = FakeTool("realestate", invoke_results={"save_apartment": {"saved": True}})
    cycle, working, registry = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    await registry.join(_ORIGIN)
    activity = Activity(
        id="a",
        goal="shortlist",
        context={},
        plan=Plan(id="p", goal="shortlist", steps=[_mechanical_subgoal()]),
        step_index=0,
        history=[_history("search_apartments", [{"id": "a1"}, {"id": "a2"}])],
    )
    working.activities["a"] = activity

    with caplog.at_level(logging.DEBUG, logger="sora.strategies"):
        await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    spliced = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.DEBUG and r.getMessage().startswith("reason: plan for activity a")
    ]
    assert len(spliced) == 1
    assert "at step 0" in spliced[0]  # where the expansion starts, i.e. what runs next
    assert "0: invoke" in spliced[0] and "1: invoke" in spliced[0]  # every expanded step
    assert "a1" in spliced[0] and "a2" in spliced[0]  # each element's binding already substituted


async def test_frame_pops_and_parent_resumes_after_the_subgoal(tmp_path: Path) -> None:
    tool = FakeTool("realestate", invoke_results={"save_apartment": {"saved": True}})
    cycle, working, registry = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    await registry.join(_ORIGIN)
    subgoal = Step(next_action="subgoal", params={"goal": "sub", "mode": "deliberative"})
    report = Step(next_action="send", params={"to": "user", "content": "all done"})
    parent = Plan(id="p", goal="top", steps=[subgoal, report])
    # Simulate a sub-plan already active on a pushed frame (one step, about to exhaust).
    subplan = Plan(id="s", goal="sub", steps=[Step(next_action="send", params={"to": "x"})])
    activity = Activity(
        id="a",
        goal="top",
        context={},
        plan=subplan,
        step_index=0,
        parent_frames=[(parent, 0)],
    )
    working.activities["a"] = activity

    # The sub-plan's one step is emitted, exhausting it (index -> 1).
    first = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert first.step is not None
    assert first.step.params == {"to": "x"}
    assert activity.step_index == 1

    # Next reason pops the frame and resumes the parent at the step *after* the sub-goal.
    resumed = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert resumed.step is report
    assert activity.plan is parent
    assert activity.step_index == 2
    assert activity.parent_frames == []


async def test_reflect_does_not_complete_a_parent_while_a_frame_is_pending(tmp_path: Path) -> None:
    # A just-exhausted sub-plan must not be judged "completed": the parent still has work, waiting
    # on the frame pop Reason does next. Completion needs the top plan exhausted AND no frames.
    tool = FakeTool("realestate")
    cycle, _, _ = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    parent = Plan(id="p", goal="top", steps=[Step(next_action="send", params={})])
    subplan = Plan(id="s", goal="sub", steps=[Step(next_action="send", params={})])
    activity = Activity(
        id="a",
        goal="top",
        context={},
        plan=subplan,
        step_index=1,  # exhausted sub-plan
        parent_frames=[(parent, 0)],
    )

    result = await DefaultReflectStrategy().reflect(activity, cycle.working, cycle, TickResult())

    state = activity.state
    assert state is ActivityState.READY  # NOT terminated — a frame is still pending
    assert result.step is None


async def test_nested_subgoals_pop_the_frame_stack_in_order(tmp_path: Path) -> None:
    tool = FakeTool("realestate")
    cycle, working, registry = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    await registry.join(_ORIGIN)
    # Each parent has a real step *after* the sub-goal it pushed (at index 0), so its frame —
    # recorded as (plan, 0) — resumes it at index 1 with work still to do.
    grandparent = Plan(
        id="gp",
        goal="gp",
        steps=[
            Step(next_action="subgoal", params={"goal": "p", "mode": "deliberative"}),
            Step(next_action="send", params={"n": "gp-after"}),
        ],
    )
    parent = Plan(
        id="p",
        goal="p",
        steps=[
            Step(next_action="subgoal", params={"goal": "l", "mode": "deliberative"}),
            Step(next_action="send", params={"n": "p-after"}),
        ],
    )
    leaf = Plan(id="l", goal="l", steps=[Step(next_action="send", params={})])
    activity = Activity(
        id="a",
        goal="gp",
        context={},
        plan=leaf,
        step_index=1,  # leaf exhausted
        parent_frames=[(grandparent, 0), (parent, 0)],  # deepest first
    )
    working.activities["a"] = activity

    # Leaf exhausted -> pop parent; parent resumes at the step after its sub-goal (index 1).
    step = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert step.step is not None
    assert step.step.params == {"n": "p-after"}
    assert activity.plan is parent
    assert activity.parent_frames == [(grandparent, 0)]
    assert activity.step_index == 2  # parent's post-sub-goal step consumed

    # Parent exhausted -> pop grandparent, resuming it at the step after ITS sub-goal (index 1).
    step = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert step.step is not None
    assert step.step.params == {"n": "gp-after"}
    assert activity.plan is grandparent
    assert activity.parent_frames == []


# --------------------------------------------------------------------------------------------------
# Deliberative recursion guard — refuse a non-reducing sub-goal, await input (ADR-0022 valve)
# --------------------------------------------------------------------------------------------------


async def test_deliberative_subgoal_halts_at_the_depth_cap(tmp_path: Path) -> None:
    tool = FakeTool("realestate")
    cycle, working, registry = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    await registry.join(_ORIGIN)
    # A stack already _DEFAULT_MAX_SUBGOAL_DEPTH deep, ancestor goals all distinct so the *depth*
    # cap (not the overlap check) is what trips. _no_llm_procedural would raise if _infer_ fired.
    frames: list[tuple[Plan, int]] = [
        (
            Plan(
                id=f"f{i}",
                goal=f"g{i}",
                steps=[Step(next_action="subgoal", params={"goal": f"ancestor task number {i}"})],
            ),
            0,
        )
        for i in range(_DEFAULT_MAX_SUBGOAL_DEPTH)
    ]
    active = Plan(
        id="active",
        goal="active",
        steps=[
            Step(
                next_action="subgoal",
                params={"goal": "some wholly unrelated final errand", "mode": "deliberative"},
            )
        ],
    )
    activity = Activity(id="a", goal="root", context={}, plan=active, parent_frames=frames)
    working.activities["a"] = activity

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert result.step is None
    state = activity.state
    assert state is ActivityState.BLOCKED
    assert isinstance(activity.blocked_on, InputWait)
    assert activity.pending_inference is None  # no _infer_ was spent


async def test_deliberative_subgoal_halts_when_goal_repeats_an_ancestor(tmp_path: Path) -> None:
    tool = FakeTool("realestate")
    cycle, working, registry = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    await registry.join(_ORIGIN)
    # The observed runaway shape: each level re-states the same zip-filtering task in reworded form.
    ancestor_goal = (
        "From the list_all_apartments result, compute the distinct set of zip_code values, then "
        "invoke get_crime_rate for each and keep the zip codes with violent_crime between 5 and 10."
    )
    repeated_goal = (
        "From the list_all_apartments result, extract the distinct set of zip_code values, then "
        "invoke get_crime_rate for each distinct zip and keep those with violent_crime "
        "between 5 and 10."
    )
    parent = Plan(
        id="p", goal="p", steps=[Step(next_action="subgoal", params={"goal": ancestor_goal})]
    )
    active = Plan(
        id="active",
        goal="active",
        steps=[Step(next_action="subgoal", params={"goal": repeated_goal, "mode": "deliberative"})],
    )
    activity = Activity(id="a", goal="root", context={}, plan=active, parent_frames=[(parent, 0)])
    working.activities["a"] = activity

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert result.step is None
    state = activity.state
    assert state is ActivityState.BLOCKED
    assert isinstance(activity.blocked_on, InputWait)
    assert activity.pending_inference is None


async def test_deliberative_subgoal_halts_on_reworded_elaboration(tmp_path: Path) -> None:
    tool = FakeTool("realestate")
    cycle, working, registry = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    await registry.join(_ORIGIN)
    # The verbatim goals from the RentAFlat run that ran away: the sub-goal restates its parent with
    # extra qualifiers ("exactly once", "relative to the current call count", ...). Token Jaccard is
    # ~0.51 (below the 0.70 threshold — the old metric let this recurse); the overlap coefficient is
    # ~0.72 because the parent's tokens stay contained, so the breaker now trips at the 2nd call.
    # Depth is only 1 here (< the cap), so it is the overlap check, not the depth cap, that fires.
    ancestor_goal = (
        "Using the apartments from list_all_apartments, collect every distinct zip_code and call "
        "insim:are/City.get_crime_rate for each distinct zip_code. Identify the set of zip codes "
        "whose violent_crime value is between 5 and 10 inclusive. Respect the City API call limit."
    )
    reworded_goal = (
        "From the list_all_apartments result, compute the set of distinct zip_code values, "
        "then for each distinct zip_code call insim:are/City.get_crime_rate exactly once "
        "(never repeating a zip_code so as to respect the City API call limit relative to "
        "the current call count), and collect the violent_crime value for each zip_code."
    )
    parent = Plan(
        id="p", goal="p", steps=[Step(next_action="subgoal", params={"goal": ancestor_goal})]
    )
    active = Plan(
        id="active",
        goal="active",
        steps=[Step(next_action="subgoal", params={"goal": reworded_goal, "mode": "deliberative"})],
    )
    activity = Activity(id="a", goal="root", context={}, plan=active, parent_frames=[(parent, 0)])
    working.activities["a"] = activity

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert result.step is None
    state = activity.state
    assert state is ActivityState.BLOCKED
    assert isinstance(activity.blocked_on, InputWait)
    assert activity.pending_inference is None  # no _infer_ was spent


async def test_deliberative_subgoal_depth_cap_is_configurable(tmp_path: Path) -> None:
    tool = FakeTool("realestate")
    cycle, working, registry = _cycle(tmp_path, _no_llm_procedural(tmp_path), tool)
    await registry.join(_ORIGIN)
    # Two frames deep with distinct goals: the default cap (4) would let this infer, but a strategy
    # built with max_subgoal_depth=2 trips the depth cap instead. Proves the agent.yaml knob reaches
    # the guard. _no_llm_procedural would raise if _infer_ fired.
    frames: list[tuple[Plan, int]] = [
        (
            Plan(
                id=f"f{i}",
                goal=f"g{i}",
                steps=[Step(next_action="subgoal", params={"goal": f"ancestor task number {i}"})],
            ),
            0,
        )
        for i in range(2)
    ]
    active = Plan(
        id="active",
        goal="active",
        steps=[
            Step(
                next_action="subgoal",
                params={"goal": "some wholly unrelated final errand", "mode": "deliberative"},
            )
        ],
    )
    activity = Activity(id="a", goal="root", context={}, plan=active, parent_frames=frames)
    working.activities["a"] = activity

    strategy = DefaultReasonStrategy(max_subgoal_depth=2)
    result = await strategy.reason(activity, working, cycle, TickResult())
    assert result.step is None
    assert activity.state is ActivityState.BLOCKED
    assert isinstance(activity.blocked_on, InputWait)
    assert activity.pending_inference is None


async def test_deliberative_subgoal_infers_when_distinct_and_shallow(tmp_path: Path) -> None:
    # An ancestor is present but its goal is unrelated, and the stack is shallow: neither detector
    # should fire — a single, legitimate decomposition still gets its _infer_ (no false positive).
    subplan = plan_json({"action": "send", "to": "user", "content": {"text": "done"}})
    llm = FakeLLMClient(subplan)
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm)
    tool = FakeTool("realestate")
    cycle, working, registry = _cycle(tmp_path, procedural, tool)
    await registry.join(_ORIGIN)
    parent = Plan(
        id="p",
        goal="p",
        steps=[Step(next_action="subgoal", params={"goal": "compute the qualifying zip codes"})],
    )
    active = Plan(
        id="active",
        goal="active",
        steps=[
            Step(
                next_action="subgoal",
                params={
                    "goal": "email each relative the two cheapest properties",
                    "mode": "deliberative",
                },
            )
        ],
    )
    activity = Activity(id="a", goal="root", context={}, plan=active, parent_frames=[(parent, 0)])
    working.activities["a"] = activity

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    assert result.step is None
    state = activity.state
    assert state is ActivityState.RUNNING
    assert activity.pending_inference is not None
    assert activity.pending_inference.kind == "subgoal"
    assert activity.blocked_on is None
    await asyncio.sleep(0)  # let the spawned _infer_ task finish so it doesn't dangle


# --------------------------------------------------------------------------------------------------
# InferAction — kind + goal override
# --------------------------------------------------------------------------------------------------


async def test_infer_action_kind_subgoal_plans_against_the_overridden_goal(tmp_path: Path) -> None:
    llm = FakeLLMClient(plan_json({"action": "send", "to": "user", "content": {"text": "x"}}))
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm)
    tool = FakeTool("realestate")
    cycle, working, registry = _cycle(tmp_path, procedural, tool)
    await registry.join(_ORIGIN)
    activity = Activity(id="a", goal="parent goal", context={})
    working.activities["a"] = activity

    await InferAction().execute(
        cycle,
        activity_id="a",
        tools={t.id: t.manual for t in registry.all_tools()},
        kind="subgoal",
        goal="the sub goal",
    )
    assert activity.pending_inference is not None
    assert activity.pending_inference.kind == "subgoal"

    await asyncio.sleep(0)
    _system, prompt = llm.calls[-1]
    assert "the sub goal" in prompt  # planned against the override
    assert "parent goal" not in prompt  # not the activity's own goal


async def test_infer_action_defaults_to_kind_plan_and_activity_goal(tmp_path: Path) -> None:
    llm = FakeLLMClient(plan_json({"action": "send", "to": "user", "content": {"text": "x"}}))
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm)
    tool = FakeTool("realestate")
    cycle, working, registry = _cycle(tmp_path, procedural, tool)
    await registry.join(_ORIGIN)
    activity = Activity(id="a", goal="parent goal", context={})
    working.activities["a"] = activity

    await InferAction().execute(
        cycle,
        activity_id="a",
        tools={t.id: t.manual for t in registry.all_tools()},
    )
    assert activity.pending_inference is not None
    assert activity.pending_inference.kind == "plan"  # unchanged default

    await asyncio.sleep(0)
    _system, prompt = llm.calls[-1]
    assert "parent goal" in prompt  # the activity's own goal


# --------------------------------------------------------------------------------------------------
# Parsing + prompt
# --------------------------------------------------------------------------------------------------


def test_step_from_raw_parses_a_subgoal_template() -> None:
    raw = {
        "action": "subgoal",
        "goal": "save each apartment",
        "mode": "mechanical",
        "in": {"$from": "search_apartments", "path": ""},
        "as": "apt",
        "template": {
            "action": "invoke",
            "tool_id": "realestate",
            "operation_name": "save_apartment",
            "params": {"apartment_id": {"$bind": "apt", "path": "id"}},
        },
    }
    step = step_from_raw(raw)
    assert step.next_action == "subgoal"
    assert step.params["goal"] == "save each apartment"
    assert step.params["mode"] == "mechanical"
    assert step.params["as"] == "apt"
    assert step.params["template"]["operation_name"] == "save_apartment"  # nested dict preserved


def test_step_from_raw_defaults_missing_action_to_invoke() -> None:
    step = step_from_raw({"tool_id": "realestate", "operation_name": "search_apartments"})
    assert step.next_action == "invoke"
    assert step.params["tool_id"] == "realestate"


def test_plan_prompt_documents_subgoal_steps() -> None:
    assert "subgoal" in PLAN_SYSTEM_PROMPT


async def test_subgoal_prompt_is_told_it_is_a_subgoal_without_mutating_the_activity(
    tmp_path: Path,
) -> None:
    """The provenance clause has to reach the *sub-goal's* prompt, and the frame that carries it is
    seeded on a throwaway copy — pushing it onto the real activity here would double up when Observe
    pushes it for real at install time."""
    llm = FakeLLMClient(plan_json({"action": "send", "to": "user", "content": {"text": "done"}}))
    procedural = ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=llm)
    cycle, working, registry = _cycle(tmp_path, procedural, FakeTool("realestate"))
    await registry.join(_ORIGIN)
    subgoal = Step(
        next_action="subgoal",
        params={"goal": "notify each relative", "mode": "deliberative"},
    )
    parent = Plan(id="p", goal="reconcile the shortlist", steps=[subgoal])
    activity = Activity(id="a", goal="reconcile the shortlist", context={}, plan=parent)
    working.activities["a"] = activity

    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())
    await asyncio.sleep(0)  # let the background _infer_ task build its prompt and call the LLM

    prompt = next(p for _s, p in llm.calls if "notify each relative" in p)
    assert "NOT a request from the user" in prompt
    # the fire itself must not have pushed the frame — Observe owns that, on the real activity
    assert activity.parent_frames == []

    await DefaultObserveStrategy().observe(cycle)
    assert activity.parent_frames == [(parent, 0)]  # pushed exactly once, by Observe
