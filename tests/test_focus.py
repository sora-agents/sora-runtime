"""Attention scoped to live intentions.

Design note: `docs/architecture/notes/attention-scoped-to-live-intentions.md`.

`IntentionScopedFocus` is a pure set-valued function of working memory, so most of this file drives
it directly — that is the whole point of making the policy pure rather than letting Observe compute
the set inline. The reconciliation tests at the bottom cover the half that isn't pure: the diff
against `focused_tools`, and its placement *before* the property snapshot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fakes import FakeAdapter, FakeTool, FakeWorkspace, ScriptedTransport
from sora.action import default_action_registry, invoke_step
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, WorkspaceOrigin
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from sora.perception import Percept
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    FocusAllJoined,
    IntentionScopedFocus,
    Strategies,
    TickResult,
    scoped_snapshot,
)
from sora.types import (
    ConditionWait,
    ObservableProperty,
    PendingCondition,
    PendingConditionState,
    Plan,
    Signal,
    SignalWait,
    Step,
)

pytestmark = pytest.mark.asyncio

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")


def _registry(*tools: FakeTool) -> EnvironmentRegistry:
    return EnvironmentRegistry(
        adapters={_ORIGIN: FakeAdapter("fake", FakeWorkspace("ws", _ORIGIN, list(tools)))}
    )


async def _joined_wm(*tools: FakeTool) -> tuple[WorkingMemory, EnvironmentRegistry]:
    registry = _registry(*tools)
    await registry.join(_ORIGIN)
    return WorkingMemory(registry=registry), registry


def _planned(plan: Plan | None, **kwargs: Any) -> Activity:
    return Activity(id=kwargs.pop("id", "a1"), goal="g", context={}, plan=plan, **kwargs)


def _step(tool_id: str) -> Step:
    return invoke_step(tool_id, "op")


# --------------------------------------------------------------------------------------------------
# IntentionScopedFocus — the broad cases
# --------------------------------------------------------------------------------------------------


async def test_an_activity_with_no_plan_broadens_to_every_joined_tool() -> None:
    # Planning reads property SHAPES to choose $prop over paginated scanning, so an activity about
    # to be planned needs the whole world. reset_for_replan() clears `plan`, so this same clause is
    # what covers the replan window — which is why attention needs no grace period.
    wm, _ = await _joined_wm(FakeTool("a"), FakeTool("b"), FakeTool("c"))
    wm.activities["a1"] = _planned(None)

    assert IntentionScopedFocus().attend(wm) == {"a", "b", "c"}


async def test_no_activities_at_all_broadens_to_every_joined_tool() -> None:
    # An idle agent stays observant: the ADR-0026 relevance judge fires from the idle branch and
    # reads the world, so going blind between tasks would silence it.
    wm, _ = await _joined_wm(FakeTool("a"), FakeTool("b"))

    assert IntentionScopedFocus().attend(wm) == {"a", "b"}


async def test_one_unplanned_activity_broadens_even_beside_a_planned_one() -> None:
    wm, _ = await _joined_wm(FakeTool("a"), FakeTool("b"), FakeTool("c"))
    wm.activities["a1"] = _planned(Plan(id="p", goal="g", steps=[_step("a")]))
    wm.activities["a2"] = _planned(None, id="a2")

    assert IntentionScopedFocus().attend(wm) == {"a", "b", "c"}


# --------------------------------------------------------------------------------------------------
# IntentionScopedFocus — the narrow cases
# --------------------------------------------------------------------------------------------------


async def test_a_plan_attends_only_the_tools_its_steps_invoke() -> None:
    wm, _ = await _joined_wm(*[FakeTool(t) for t in "abcde"])
    wm.activities["a1"] = _planned(
        Plan(id="p", goal="g", steps=[_step("a"), _step("c"), _step("a")])
    )

    assert IntentionScopedFocus().attend(wm) == {"a", "c"}


async def test_a_focus_step_is_counted_and_an_unfocus_step_is_not() -> None:
    # An `unfocus` step names a tool in order to STOP attending to it. Counting its tool_id would
    # attend the tool right back on the very next Observe and make the step a permanent no-op.
    wm, _ = await _joined_wm(FakeTool("a"), FakeTool("watched"), FakeTool("done"))
    wm.activities["a1"] = _planned(
        Plan(
            id="p",
            goal="g",
            steps=[
                Step(next_action="focus", params={"tool_id": "watched"}),
                _step("a"),
                Step(next_action="unfocus", params={"tool_id": "done"}),
            ],
        )
    )

    assert IntentionScopedFocus().attend(wm) == {"a", "watched"}


async def test_an_unfocus_step_is_overridden_when_another_step_names_the_same_tool() -> None:
    # A known rough edge, pinned rather than fixed. The scan covers EVERY step of a live plan, not
    # just the un-run tail — that is what removes the need for history retention — so an earlier
    # `invoke` re-attends the tool the `unfocus` just released, on the next Observe. The derived
    # floor overrides the explicit act, and the tool is re-baselined at the adapter in between.
    wm, _ = await _joined_wm(FakeTool("mail"), FakeTool("cal"))
    wm.activities["a1"] = _planned(
        Plan(
            id="p",
            goal="g",
            steps=[
                _step("mail"),
                Step(next_action="unfocus", params={"tool_id": "mail"}),
                _step("cal"),
            ],
        )
    )

    assert IntentionScopedFocus().attend(wm) == {"mail", "cal"}


async def test_a_nested_prop_reference_keeps_its_tool_attended() -> None:
    # The $prop trap: a property reference lives inside a step's PARAMS (here under a data-op's
    # `in`), not under tool_id. A scan that only reads tool_id releases the tool mid-plan and the
    # reference then resolves to nothing — silent blindness, the failure intention-scoped
    # focus exists to remove.
    wm, _ = await _joined_wm(FakeTool("realestate"), FakeTool("Contacts"))
    wm.activities["a1"] = _planned(
        Plan(
            id="p",
            goal="g",
            steps=[
                Step(
                    next_action="filter",
                    params={
                        "in": {"$prop": "Contacts.state", "path": "contacts"},
                        "out": "shortlist",
                        "where": {"$decide": "anyone in sales"},
                    },
                )
            ],
        )
    )

    assert IntentionScopedFocus().attend(wm) == {"Contacts"}


async def test_a_prop_reference_folded_into_one_dotted_token_still_names_its_tool() -> None:
    # The tool id is matched against the joined id set, never found by splitting on the first dot:
    # neither half of a property key is dot-free (a WoT tool id contains dots, and nothing forbids
    # a property called `sensor.temp`). Longest match wins.
    wm, _ = await _joined_wm(FakeTool("wot:lamp.local/Lamp"))
    wm.activities["a1"] = _planned(
        Plan(
            id="p",
            goal="g",
            steps=[
                Step(next_action="take", params={"in": {"$prop": "wot:lamp.local/Lamp.state.on"}})
            ],
        )
    )

    assert IntentionScopedFocus().attend(wm) == {"wot:lamp.local/Lamp"}


async def test_a_bare_prop_name_names_no_tool_and_attends_nothing() -> None:
    wm, _ = await _joined_wm(FakeTool("a"), FakeTool("b"))
    wm.activities["a1"] = _planned(
        Plan(id="p", goal="g", steps=[Step(next_action="take", params={"in": {"$prop": "state"}})])
    )

    assert IntentionScopedFocus().attend(wm) == set()


async def test_suspended_parent_frames_keep_their_tools_attended() -> None:
    # A sub-goal push suspends the parent frame rather than discarding it; its steps still run
    # after the sub-plan lands, so its tools are still live intentions.
    wm, _ = await _joined_wm(FakeTool("parent"), FakeTool("child"), FakeTool("other"))
    activity = _planned(Plan(id="sub", goal="s", steps=[_step("child")]))
    activity.parent_frames = [(Plan(id="p", goal="g", steps=[_step("parent")]), 1, 0)]
    wm.activities["a1"] = activity

    assert IntentionScopedFocus().attend(wm) == {"parent", "child"}


async def test_condition_watches_and_blocked_on_sources_are_attended() -> None:
    # Not redundant with the step scan: a plan routinely watches a messaging app for a reply while
    # every one of its steps touches email. Releasing the watched tool stops its signals at the
    # source, so the condition could never fire.
    wm, _ = await _joined_wm(
        FakeTool("Email"), FakeTool("Messenger"), FakeTool("Cab"), FakeTool("x")
    )
    condition = PendingCondition(
        watch=SignalWait(signal_name="state_changed", source="Messenger"),
        when="a reply arrives",
        then="answer it",
    )
    activity = _planned(Plan(id="p", goal="g", steps=[_step("Email")], pending=(condition,)))
    activity.pending_conditions = [PendingConditionState(condition=condition)]
    activity.state = ActivityState.BLOCKED
    activity.blocked_on = ConditionWait(
        watches=(SignalWait(signal_name="ride_updated", source="Cab"),)
    )
    wm.activities["a1"] = activity

    assert IntentionScopedFocus().attend(wm) == {"Email", "Messenger", "Cab"}


async def test_a_terminated_activity_contributes_nothing() -> None:
    wm, _ = await _joined_wm(FakeTool("a"), FakeTool("b"))
    done = _planned(Plan(id="p", goal="g", steps=[_step("a")]))
    done.state = ActivityState.TERMINATED
    live = _planned(Plan(id="q", goal="h", steps=[_step("b")]), id="a2")
    wm.activities = {"a1": done, "a2": live}

    assert IntentionScopedFocus().attend(wm) == {"b"}


async def test_the_last_activity_terminating_falls_back_to_broad() -> None:
    wm, _ = await _joined_wm(FakeTool("a"), FakeTool("b"))
    done = _planned(Plan(id="p", goal="g", steps=[_step("a")]))
    done.state = ActivityState.TERMINATED
    wm.activities["a1"] = done

    assert IntentionScopedFocus().attend(wm) == {"a", "b"}


async def test_a_tool_id_no_longer_joined_is_never_re_attended() -> None:
    # Intersected with the registry last, so a plan carried over from a departed workspace cannot
    # resurrect a focus on a tool that no longer exists.
    wm, _ = await _joined_wm(FakeTool("a"))
    wm.activities["a1"] = _planned(Plan(id="p", goal="g", steps=[_step("a"), _step("gone")]))

    assert IntentionScopedFocus().attend(wm) == {"a"}


async def test_focus_all_joined_ignores_activities_entirely() -> None:
    wm, _ = await _joined_wm(FakeTool("a"), FakeTool("b"), FakeTool("c"))
    wm.activities["a1"] = _planned(Plan(id="p", goal="g", steps=[_step("a")]))

    assert FocusAllJoined().attend(wm) == {"a", "b", "c"}


# --------------------------------------------------------------------------------------------------
# Reconciliation in Observe — the impure half
# --------------------------------------------------------------------------------------------------


def _cycle(tmp_path: Path, registry: EnvironmentRegistry, wm: WorkingMemory) -> DecisionCycle:
    return DecisionCycle(
        strategies=Strategies(
            observe=DefaultObserveStrategy(),
            reflect=DefaultReflectStrategy(),
            situate=DefaultSituateStrategy(),
            reason=_InertReason(),
            act=DefaultActStrategy(),
        ),
        communication=ScriptedTransport(),
        actions=default_action_registry(),
        registry=registry,
        working=wm,
        semantic=SemanticMemory(FileMemoryBackend(tmp_path / "sem")),
        procedural=ProceduralMemory(FileMemoryBackend(tmp_path / "proc")),
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "epi")),
    )


class _InertReason:
    async def reason(
        self, activity: Activity, wm: WorkingMemory, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        return result


async def test_observe_narrows_attention_within_the_same_ticks_snapshot(tmp_path: Path) -> None:
    # Reconciliation runs BEFORE _snapshot_properties, so a plan that just landed narrows the world
    # on this tick rather than one tick late — and the released tool's stale snapshot goes with it.
    kept = FakeTool("kept", properties=[ObservableProperty("state", 1)])
    dropped = FakeTool("dropped", properties=[ObservableProperty("state", 2)])
    wm, registry = await _joined_wm(kept, dropped)
    cycle = _cycle(tmp_path, registry, wm)
    strategy = DefaultObserveStrategy(focus=IntentionScopedFocus())

    await strategy.observe(cycle)  # no activities -> broad
    assert set(wm.focused_tools) == {"kept", "dropped"}
    assert set(wm.properties) == {("kept", "state"), ("dropped", "state")}

    wm.activities["a1"] = _planned(Plan(id="p", goal="g", steps=[_step("kept")]))
    await strategy.observe(cycle)

    assert set(wm.focused_tools) == {"kept"}
    assert dropped.focused is False
    assert set(wm.properties) == {("kept", "state")}


async def test_the_default_policy_attends_everything_joined_even_with_a_narrow_plan(
    tmp_path: Path,
) -> None:
    # The shipped default is FocusAllJoined, not IntentionScopedFocus. Narrowing buys ~825 prompt
    # tokens per model call and zero judge calls, and costs a re-baselining risk that fails
    # silently — so it is opt-in. Attention is still RECONCILED here, which is what keeps
    # perception from hinging on the model emitting a `focus` step.
    kept = FakeTool("kept", properties=[ObservableProperty("state", 1)])
    other = FakeTool("other", properties=[ObservableProperty("state", 2)])
    wm, registry = await _joined_wm(kept, other)
    cycle = _cycle(tmp_path, registry, wm)
    wm.activities["a1"] = _planned(Plan(id="p", goal="g", steps=[_step("kept")]))

    await DefaultObserveStrategy().observe(cycle)

    assert set(wm.focused_tools) == {"kept", "other"}
    assert set(wm.properties) == {("kept", "state"), ("other", "state")}


async def test_attaching_a_tool_emits_no_spurious_change_on_that_tick(tmp_path: Path) -> None:
    # tool.focus() establishes the adapter's change baseline, so the observe() immediately below
    # compares equal. If reconciliation ran after the snapshot this would be a false wake every
    # time attention widened.
    tool = FakeTool("a", signals_on_focus=[])
    wm, registry = await _joined_wm(tool)
    cycle = _cycle(tmp_path, registry, wm)

    await DefaultObserveStrategy().observe(cycle)

    assert set(wm.focused_tools) == {"a"}
    assert wm.signals == []


async def test_reset_for_replan_re_broadens_attention_on_the_next_tick(tmp_path: Path) -> None:
    # The reason attention needs neither a grace period nor history retention: reset_for_replan()
    # clears `plan`, and an activity with no plan is broad by rule.
    wm, registry = await _joined_wm(FakeTool("a"), FakeTool("b"), FakeTool("c"))
    cycle = _cycle(tmp_path, registry, wm)
    strategy = DefaultObserveStrategy(focus=IntentionScopedFocus())
    activity = _planned(Plan(id="p", goal="g", steps=[_step("a")]))
    wm.activities["a1"] = activity

    await strategy.observe(cycle)
    assert set(wm.focused_tools) == {"a"}

    activity.reset_for_replan()
    await strategy.observe(cycle)

    assert set(wm.focused_tools) == {"a", "b", "c"}


async def test_an_explicit_focus_step_survives_reconciliation(tmp_path: Path) -> None:
    # The plan-level override still works: a `focus` step's tool is a live intention for as long as
    # the plan is, so the next Observe does not immediately release what Act just focused.
    watched = FakeTool("watched", properties=[ObservableProperty("state", 1)])
    wm, registry = await _joined_wm(FakeTool("a"), watched)
    cycle = _cycle(tmp_path, registry, wm)
    wm.activities["a1"] = _planned(
        Plan(
            id="p",
            goal="g",
            steps=[Step(next_action="focus", params={"tool_id": "watched"}), _step("a")],
        )
    )

    await DefaultObserveStrategy().observe(cycle)

    assert set(wm.focused_tools) == {"a", "watched"}
    assert ("watched", "state") in wm.properties


# --------------------------------------------------------------------------------------------------
# scoped_snapshot — the per-activity prompt view (non-destructive)
# --------------------------------------------------------------------------------------------------


async def test_scoped_snapshot_keeps_only_the_activitys_own_tools() -> None:
    wm, _ = await _joined_wm(FakeTool("mine"), FakeTool("theirs"))
    wm.properties[("mine", "state")] = Percept("mine", ObservableProperty("state", 1), 0.0)
    wm.properties[("theirs", "state")] = Percept("theirs", ObservableProperty("state", 2), 0.0)
    wm.signals.append(Percept("mine", Signal("changed", {}), 0.0))
    wm.signals.append(Percept("theirs", Signal("changed", {}), 0.0))
    activity = _planned(Plan(id="p", goal="g", steps=[_step("mine")]))

    view = scoped_snapshot(wm, activity)

    assert [p.source for p in view.properties] == ["mine"]
    assert [p.source for p in view.signals] == ["mine"]


async def test_scoped_snapshot_is_non_destructive() -> None:
    # The invariant that lets this layer exist at all: the shared store must not depend on which
    # activity the scheduler picked, or the ADR-0024 change signature moves when nothing moved.
    wm, _ = await _joined_wm(FakeTool("mine"), FakeTool("theirs"))
    wm.properties[("mine", "state")] = Percept("mine", ObservableProperty("state", 1), 0.0)
    wm.properties[("theirs", "state")] = Percept("theirs", ObservableProperty("state", 2), 0.0)
    before = dict(wm.properties)
    activity = _planned(Plan(id="p", goal="g", steps=[_step("mine")]))

    scoped_snapshot(wm, activity)

    assert wm.properties == before


async def test_scoped_snapshot_of_an_unplanned_activity_is_the_whole_world() -> None:
    wm, _ = await _joined_wm(FakeTool("a"), FakeTool("b"))
    wm.properties[("a", "state")] = Percept("a", ObservableProperty("state", 1), 0.0)
    wm.properties[("b", "state")] = Percept("b", ObservableProperty("state", 2), 0.0)

    view = scoped_snapshot(wm, _planned(None))

    assert {p.source for p in view.properties} == {"a", "b"}
