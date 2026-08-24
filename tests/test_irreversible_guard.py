"""The irreversibility guard: no write commits on a plan already known to be unfinishable.

Reconstructed from the run that motivated it. A filter chain wrote ``friend_contact = []`` at step 3
(the planner had paged only the first ten of 125 contacts, so nobody matched); the plan carried
on, read the calendar at step 4, DELETED the user's real appointment at step 5, and only tripped
over the empty binding at step 7, where it finally needed the friend's address. The user lost an
appointment and got nothing in return. The evidence that the plan could not finish was sitting in
``Activity.bindings`` two steps before the irreversible act — so Reason now reads it there, before
committing any side-effecting step, and re-plans instead.

Not an ordering rule: that plan gathered before it destroyed, which is the correct order. A
viability rule.
"""

from __future__ import annotations

from pathlib import Path

from fakes import FakeAdapter, FakeLLMClient, FakeTool, FakeWorkspace, ScriptedTransport
from sora.action import default_action_registry, invoke_step
from sora.activity import Activity
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry, WorkspaceOrigin
from sora.manual import Manual, OperationSpecification
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReasonStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    NoneReconsideration,
    Strategies,
    TickResult,
    _dereferenced_bindings,
    _dereferenced_operations,
    _is_empty,
    _spent_operation_read,
    _unsatisfiable_reference,
)
from sora.types import (
    CompletedOperation,
    OperationAck,
    OperationInvocation,
    Plan,
    Step,
)

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")

# delete_op stands in for the delete_calendar_event of the run; read_op for the calendar read that
# preceded it. The guard tells them apart only through OperationSpecification.side_effecting, which
# ARE's adapter lifts from the app's own write_operation flag — no name heuristic anywhere.
_MANUAL = Manual(
    id="t",
    metadata={},
    description="",
    observable_properties=[],
    signals=[],
    operations=[
        OperationSpecification(
            name="delete_op", description="", parameters={}, side_effecting=True
        ),
        OperationSpecification(name="read_op", description="", parameters={}, side_effecting=False),
        OperationSpecification(name="mystery_op", description="", parameters={}),  # unknown
    ],
)


async def _cycle(tmp_path: Path) -> tuple[DecisionCycle, WorkingMemory]:
    tool = FakeTool("t", manual=_MANUAL, invoke_results={"delete_op": "ok", "read_op": "ok"})
    registry = EnvironmentRegistry(
        adapters={_ORIGIN: FakeAdapter("fake", FakeWorkspace("ws", _ORIGIN, [tool]))}
    )
    await registry.join(_ORIGIN)
    working = WorkingMemory(registry=registry)
    cycle = DecisionCycle(
        strategies=Strategies(
            observe=DefaultObserveStrategy(),
            reflect=DefaultReflectStrategy(),
            situate=DefaultSituateStrategy(),
            reason=DefaultReasonStrategy(),
            act=DefaultActStrategy(),
        ),
        communication=ScriptedTransport(),
        actions=default_action_registry(),
        registry=registry,
        working=working,
        semantic=SemanticMemory(FileMemoryBackend(tmp_path / "sem")),
        procedural=ProceduralMemory(FileMemoryBackend(tmp_path / "proc"), llm=FakeLLMClient("{}")),
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "epi")),
        reconsideration=NoneReconsideration(),  # isolate the guard from the ADR-0024 checkpoint
    )
    return cycle, working


def _use(name: str, path: str = "email") -> dict[str, object]:
    return {"$bind": name, "path": path}


def _activity(
    working: WorkingMemory, steps: list[Step], *, bindings: dict[str, object]
) -> Activity:
    activity = Activity(id="a", goal="book a day with my friend", context={})
    activity.plan = Plan(id="p", goal=activity.goal, steps=steps)
    activity.bindings.update(bindings)
    working.activities[activity.id] = activity
    return activity


# ── the observed failure ─────────────────────────────────────────────────────────────────────────


async def test_a_write_does_not_commit_when_a_later_step_needs_an_empty_binding(
    tmp_path: Path,
) -> None:
    """The run, reduced: the delete is next, and a later step reads a binding already produced
    empty. The delete must not go out."""
    cycle, working = await _cycle(tmp_path)
    activity = _activity(
        working,
        [
            invoke_step("t", "delete_op", event_id="evt-1"),
            invoke_step("t", "delete_op", recipients=[_use("friend_contact")]),
        ],
        bindings={"filtered_contacts": [], "friend_contact": []},
    )

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step is None  # nothing committed
    assert activity.plan is None  # dropped -> Reason re-infers next cycle
    assert activity.replan_trail and "friend_contact" in str(activity.replan_trail[0])


async def test_the_defect_tells_the_planner_what_was_empty_and_what_to_do(tmp_path: Path) -> None:
    """The defect string is the whole payload of the abort — it is what the replanning prompt shows,
    so a replacement plan that pages the same ten contacts again is a failure of this message."""
    cycle, working = await _cycle(tmp_path)
    activity = _activity(
        working,
        [invoke_step("t", "delete_op"), invoke_step("t", "delete_op", to=_use("friend_contact"))],
        bindings={"friend_contact": []},
    )

    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    defect = str(activity.replan_trail[0])
    assert "EMPTY" in defect  # names the condition
    assert "step 1" in defect  # names where it will bite
    assert "whole collection" in defect  # names the way out


async def test_a_read_step_is_not_gated_by_the_guard(tmp_path: Path) -> None:
    """Only writes. Continuing a doomed plan through a read costs a cycle; nothing is lost that
    cannot be redone, and the plan may still be rewritten before it reaches a write."""
    cycle, working = await _cycle(tmp_path)
    activity = _activity(
        working,
        [invoke_step("t", "read_op"), invoke_step("t", "delete_op", to=_use("friend_contact"))],
        bindings={"friend_contact": []},
    )

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step == invoke_step("t", "read_op")  # committed
    assert activity.plan is not None


async def test_an_operation_of_unknown_side_effect_is_gated_like_a_write(tmp_path: Path) -> None:
    """`side_effecting` is None when a manual doesn't say. Treating the unknown as a write matches
    what the reconsideration checkpoint already does, and the asymmetry is the same: the cost of
    being wrong is one inference in one direction and an unrecoverable act in the other."""
    cycle, working = await _cycle(tmp_path)
    activity = _activity(
        working,
        [invoke_step("t", "mystery_op"), invoke_step("t", "delete_op", to=_use("friend_contact"))],
        bindings={"friend_contact": []},
    )

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step is None
    assert activity.plan is None


async def test_a_write_commits_when_nothing_is_provably_dead(tmp_path: Path) -> None:
    cycle, working = await _cycle(tmp_path)
    activity = _activity(
        working,
        [invoke_step("t", "delete_op"), invoke_step("t", "delete_op", to=_use("friend_contact"))],
        bindings={"friend_contact": [{"email": "ake@x.com"}]},
    )

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step == invoke_step("t", "delete_op")
    assert activity.plan is not None


async def test_the_guard_scans_a_suspended_parent_frame_too(tmp_path: Path) -> None:
    """A sub-plan's caller runs later and reads the same flat bindings, so a dead reference waiting
    in the parent is just as fatal — and the write in hand is inside the sub-plan."""
    cycle, working = await _cycle(tmp_path)
    parent = Plan(
        id="parent",
        goal="g",
        steps=[
            Step("subgoal", {"goal": "sub", "mode": "deliberative"}),
            invoke_step("t", "delete_op", to=_use("friend_contact")),
        ],
    )
    activity = _activity(working, [invoke_step("t", "delete_op")], bindings={"friend_contact": []})
    activity.parent_frames.append((parent, 0, 0))

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step is None
    assert activity.plan is None


# ── what does NOT count as dead ──────────────────────────────────────────────────────────────────


async def test_an_empty_binding_in_a_collection_position_is_an_answer_not_a_defect(
    tmp_path: Path,
) -> None:
    """`in` is where a collection belongs, and an empty one there means "nothing to iterate" — the
    same line _data_op already draws between an empty result and an unreadable one. Gating on it
    would make "no matches" indistinguishable from "the plan is broken"."""
    cycle, working = await _cycle(tmp_path)
    activity = _activity(
        working,
        [
            invoke_step("t", "delete_op"),
            Step("filter", {"in": {"$bind": "shortlist"}, "out": "kept", "where": {}}),
        ],
        bindings={"shortlist": []},
    )

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step == invoke_step("t", "delete_op")  # committed


async def test_a_binding_a_later_step_rewrites_is_not_provably_empty(tmp_path: Path) -> None:
    """Empty now, but a step between here and the dereference writes the same name — so nothing is
    proven and the guard must stay out of the way."""
    cycle, working = await _cycle(tmp_path)
    activity = _activity(
        working,
        [
            invoke_step("t", "delete_op"),
            Step("filter", {"in": [{"email": "x@y.z"}], "out": "friend_contact", "where": {}}),
            invoke_step("t", "delete_op", to=_use("friend_contact")),
        ],
        bindings={"friend_contact": []},
    )

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step == invoke_step("t", "delete_op")


async def test_a_step_already_behind_the_cursor_does_not_count(tmp_path: Path) -> None:
    """Only the plan's remaining steps can still bite; one that already ran had its chance."""
    cycle, working = await _cycle(tmp_path)
    activity = _activity(
        working,
        [invoke_step("t", "delete_op", to=_use("friend_contact")), invoke_step("t", "delete_op")],
        bindings={"friend_contact": []},
    )
    activity.step_index = 1

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step == invoke_step("t", "delete_op")


# ── the predicates ───────────────────────────────────────────────────────────────────────────────


def test_is_empty_is_emptiness_not_falsiness() -> None:
    empty: tuple[object, ...] = ([], {}, "", None, set())
    assert all(_is_empty(v) for v in empty)
    # 0 and False are values a step can act on — an hour count, a "confirmed" flag.
    usable: tuple[object, ...] = (0, False, [0], {"a": None}, "0")
    assert not any(_is_empty(v) for v in usable)


def test_dereferenced_bindings_finds_nested_references_and_skips_collection_keys() -> None:
    step = Step(
        "invoke",
        {
            "recipients": [{"$bind": "friend", "path": "email"}],  # inside a list
            "body": {"greeting": {"$bind": "salutation"}},  # inside a dict
            "in": {"$bind": "shortlist"},  # a collection position
            "where": {"path": "id", "op": "not_in", "value": {"$bind": "already_saved"}},
        },
    )
    assert _dereferenced_bindings(step) == {"friend", "salutation"}


def test_a_loop_element_name_that_shadows_an_empty_binding_is_still_the_loop_element() -> None:
    """Nothing stops a plan from naming its loop element after the binding it iterates — writing
    `filter(out: "contacts")` then `subgoal(in: {"$bind": "contacts"}, as: "contacts")` is natural.
    When that binding is empty the fan-out is legitimately zero steps ("nothing matched, nothing to
    do"), but the template's `{"$bind": "contacts"}` looked like a read of the dead binding and
    condemned the plan at the next write. The element name is excluded explicitly, not assumed
    distinct."""
    activity = Activity(id="a", goal="g", context={})
    activity.plan = Plan(
        id="p",
        goal="g",
        steps=[
            Step(
                "subgoal",
                {
                    "in": {"$bind": "contacts"},
                    "as": "contacts",  # shadows the binding it iterates
                    "template": {"action": "invoke", "params": {"id": {"$bind": "contacts"}}},
                },
            )
        ],
    )
    activity.bindings["contacts"] = []

    assert _unsatisfiable_reference(activity) is None


# ── the same proof, spelled `$from` ──────────────────────────────────────────────────────────────
# A later run got past the guard through the other reference token. The plan invoked
# `search_contacts` (a job title against an operation that matches only names — it returned `[]`),
# and the runtime still committed `add_calendar_event`, creating the event with `attendees: []`.
# The evidence sat in `Activity.history` rather than `Activity.bindings`; everything else about the
# situation — provably no value, a write next, nothing to roll it back — was identical.


def _from(name: str, path: str = "0.email") -> dict[str, object]:
    return {"$from": name, "path": path}


def _ran(operation_name: str, result: object) -> CompletedOperation:
    return CompletedOperation(
        OperationInvocation("t", operation_name, {}), OperationAck(ok=True, result=result)
    )


async def test_a_write_does_not_commit_when_a_step_reads_an_operation_that_returned_empty(
    tmp_path: Path,
) -> None:
    cycle, working = await _cycle(tmp_path)
    activity = _activity(
        working, [invoke_step("t", "delete_op", to=_from("search_contacts"))], bindings={}
    )
    activity.history.append(_ran("search_contacts", []))

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step is None
    assert activity.plan is None
    assert "search_contacts" in str(activity.replan_trail[0])
    assert "empty result" in str(activity.replan_trail[0])


async def test_the_decide_element_shape_that_actually_slipped_through(tmp_path: Path) -> None:
    """The write that got out carried its dead reference nested two levels down, inside a `$decide`
    element of a list parameter, under a plain `from` key. A scan matching only a param whose whole
    value is a reference would still miss it, so this pins the real shape verbatim."""
    cycle, working = await _cycle(tmp_path)
    attendees = [
        {
            "$decide": "contact.first_name + ' ' + contact.last_name",
            "from": {"$from": "search_contacts", "path": "0"},
        }
    ]
    activity = _activity(
        working,
        [invoke_step("t", "delete_op", title="Film Production Day", attendees=attendees)],
        bindings={},
    )
    activity.history.append(_ran("search_contacts", []))

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step is None
    assert activity.plan is None
    assert "search_contacts" in str(activity.replan_trail[0])


# ── what must NOT be treated as proof ────────────────────────────────────────────────────────────


async def test_an_operation_that_has_not_run_yet_is_not_evidence(tmp_path: Path) -> None:
    """The ordinary plan shape — invoke at one step, read the result at the next. Absence from the
    history is how every plan looks before it runs; treating it as death would condemn them all."""
    cycle, working = await _cycle(tmp_path)
    activity = _activity(
        working,
        [
            invoke_step("t", "delete_op"),
            invoke_step("t", "delete_op", to=_from("search_contacts")),
        ],
        bindings={},
    )

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step == invoke_step("t", "delete_op")  # the write commits
    assert activity.plan is not None


async def test_an_operation_the_plan_re_invokes_first_is_not_provably_empty(tmp_path: Path) -> None:
    """The `out`-rewrite rule, for operations — and the case that decides whether this guard is
    safe at all. It is exactly a replan's second attempt: the first `search_contacts` returned
    nothing, so the new plan searches again by a different term before reading it. Condemning that
    on the strength of the superseded attempt would make the guard eat the very recoveries the
    replan machinery exists to produce."""
    cycle, working = await _cycle(tmp_path)
    activity = _activity(
        working,
        [
            invoke_step("t", "delete_op"),
            invoke_step("t", "search_contacts", query="Lindstrom"),
            invoke_step("t", "delete_op", to=_from("search_contacts")),
        ],
        bindings={},
    )
    activity.history.append(_ran("search_contacts", []))  # the earlier attempt, still in history

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step == invoke_step("t", "delete_op")
    assert activity.plan is not None


async def test_a_present_source_read_by_a_wrong_path_is_left_to_grounding(tmp_path: Path) -> None:
    """A mis-pathed reference into a source that DID return data is a recoverable defect — grounding
    reads the real history and routinely resolves the value anyway. Only an empty source admits no
    repair, and that is the line between condemning a plan and letting it proceed."""
    cycle, working = await _cycle(tmp_path)
    activity = _activity(
        working,
        [invoke_step("t", "delete_op", to=_from("search_contacts", "0.nonesuch"))],
        bindings={},
    )
    activity.history.append(_ran("search_contacts", [{"email": "ake@x.se"}]))

    assert _unsatisfiable_reference(activity) is None  # the guard declines to condemn it
    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    # The step escalates to grounding rather than committing — which is the point: the plan is
    # still alive and the bad path is grounding's problem, not the guard's.
    assert activity.plan is not None
    assert activity.replan_trail == []


async def test_an_empty_operation_result_in_a_collection_position_is_an_answer(
    tmp_path: Path,
) -> None:
    """The same exemption the binding scan makes: an empty collection to iterate is "nothing to do",
    not a dead plan."""
    cycle, working = await _cycle(tmp_path)
    activity = _activity(working, [invoke_step("t", "delete_op")], bindings={})
    activity.plan = Plan(
        id="p",
        goal="g",
        steps=[
            invoke_step("t", "delete_op"),
            Step(
                "subgoal",
                {
                    "in": {"$from": "search_contacts"},
                    "as": "item",
                    "template": {"action": "invoke", "params": {"id": {"$bind": "item"}}},
                },
            ),
        ],
    )
    activity.history.append(_ran("search_contacts", []))

    result = await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert result.step == invoke_step("t", "delete_op")
    assert activity.plan is not None


# ── the unit-level shape of the new scan ─────────────────────────────────────────────────────────


def test_dereferenced_operations_finds_nested_refs_and_skips_collection_keys() -> None:
    step = Step(
        "invoke",
        {
            "to": {"$from": "search_contacts", "path": "0.email"},
            "body": {"$decide": "a note", "from": {"$from": "get_time"}},
            "in": {"$from": "list_all"},  # a collection position: exempt
        },
    )
    names = sorted(str(ref["$from"]) for ref in _dereferenced_operations(step))
    assert names == ["get_time", "search_contacts"]


def test_a_qualified_reference_is_matched_against_a_re_invocation_by_bare_name() -> None:
    """`_latest_result` accepts three spellings of an operation, so the re-invocation check has to
    recognise all three — otherwise a plan that re-runs `search_contacts` but reads it back as
    `Contacts.search_contacts` is condemned by the attempt it is about to supersede."""
    history = [_ran("search_contacts", [])]
    assert _spent_operation_read(_from("search_contacts"), history, set()) is not None
    assert _spent_operation_read(_from("search_contacts"), history, {"search_contacts"}) is None
    qualified = _from("insim:are/Contacts.search_contacts")
    assert _spent_operation_read(qualified, history, {"search_contacts"}) is None


def test_a_void_result_is_empty_for_the_purpose_of_reading_a_value_out_of_it() -> None:
    # An operation that returns nothing (an ARE delete) has no value to read, whatever the path.
    assert _spent_operation_read(_from("delete_op"), [_ran("delete_op", None)], set()) is not None
    # A real value at the path is not a defect.
    got = [_ran("search_contacts", [{"email": "ake@x.se"}])]
    assert _spent_operation_read(_from("search_contacts"), got, set()) is None
