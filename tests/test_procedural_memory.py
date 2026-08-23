"""Tests for the deterministic, file-backed ``ProceduralMemory`` (store/retrieve).

``ProceduralMemory`` caches ``Plan``s a completed activity actually followed, so a later activity
with the same goal reuses one instead of re-planning. The default is deterministic: a ``Plan`` is
stored under its own stable ``id`` and retrieved by an exact match on its ``goal`` (the retrieval
key). ``infer()`` — the model path that synthesizes a fresh plan by querying the pluggable ``llm``
(an ``LLMClient``) — is covered here too, exercised deterministically with a ``FakeLLMClient``.

Store/retrieve are backed by a real ``FileMemoryBackend`` (the deterministic default, already fast
and covered by ``test_memory_backend.py``) rather than a fake, so these tests pin the actual
serialization round-trip the module owns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fakes import FakeLLMClient, fake_manual, plan_json
from sora.action import invoke_step
from sora.activity import Activity
from sora.manual import (
    Manual,
    ObservablePropertySpecification,
    OperationSpecification,
    SignalSpecification,
)
from sora.memory import (
    _HISTORY_RENDER_PLAN,
    _HISTORY_RENDER_REVALIDATE,
    GROUND_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    FileMemoryBackend,
    PerceptSnapshot,
    ProceduralMemory,
    default_ground_prompt,
    default_plan_prompt,
    remaining_steps,
    render_bindings,
    render_history,
    render_messages,
    render_properties,
    render_signals,
    render_tools,
)
from sora.perception import Message, Percept
from sora.types import (
    CompletedOperation,
    ObservableProperty,
    OperationAck,
    OperationInvocation,
    Plan,
    Signal,
    Step,
    SupersededPlan,
)


def _activity(goal: str) -> Activity:
    return Activity(id=f"act-{goal}", goal=goal, context={})


def _property_percept(source: str, name: str, value: object) -> Percept:
    return Percept(source, ObservableProperty(name, value), 0.0)


def _signal_percept(source: str, name: str, payload: dict[str, object]) -> Percept:
    return Percept(source, Signal(name, payload), 0.0)


def _plan(plan_id: str, goal: str, steps: list[Step] | None = None) -> Plan:
    if steps is None:
        steps = [Step(next_action="invoke", params={"tool_id": "EmailClientApp", "n": 1})]
    return Plan(id=plan_id, goal=goal, steps=steps)


def _memory(tmp_path: Path) -> ProceduralMemory:
    return ProceduralMemory(FileMemoryBackend(tmp_path))


# --------------------------------------------------------------------------------------------------
# store -> retrieve round-trip
# --------------------------------------------------------------------------------------------------


async def test_store_then_retrieve_by_goal_returns_equal_plan(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    plan = _plan("p1", "schedule from email")
    await mem.store(plan)
    got = await mem.retrieve(_activity("schedule from email"))
    assert got == plan


async def test_retrieve_unknown_goal_returns_none(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    await mem.store(_plan("p1", "schedule from email"))
    assert await mem.retrieve(_activity("book a flight")) is None


async def test_retrieve_on_empty_store_returns_none(tmp_path: Path) -> None:
    mem = _memory(tmp_path / "never_written")
    assert await mem.retrieve(_activity("anything")) is None


async def test_store_same_id_overwrites(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    await mem.store(_plan("p1", "schedule from email", [Step(next_action="wait", params={})]))
    updated = _plan("p1", "schedule from email", [Step(next_action="invoke", params={"n": 2})])
    await mem.store(updated)
    got = await mem.retrieve(_activity("schedule from email"))
    assert got == updated


async def test_retrieve_matches_on_goal_not_id(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    await mem.store(_plan("p-a", "goal A"))
    await mem.store(_plan("p-b", "goal B"))
    got = await mem.retrieve(_activity("goal B"))
    assert got is not None
    assert got.id == "p-b"


async def test_retrieve_is_exact_match_not_fuzzy(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    await mem.store(_plan("p1", "schedule from email"))
    # A similar-but-different goal must not resolve to the stored plan.
    assert await mem.retrieve(_activity("schedule email")) is None


# --------------------------------------------------------------------------------------------------
# Serialization fidelity of the Plan/Step shape
# --------------------------------------------------------------------------------------------------


async def test_multi_step_plan_round_trips_with_order_and_types(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    steps = [
        Step(next_action="invoke", params={"tool_id": "EmailClientApp", "n": 1}),
        Step(next_action="wait", params={}),
        Step(next_action="send", params={"to": "peer", "content": {"nested": [1, 2, 3]}}),
    ]
    plan = _plan("p-multi", "compose reply", steps)
    await mem.store(plan)
    got = await mem.retrieve(_activity("compose reply"))
    assert got == plan
    assert got is not None
    assert [s.next_action for s in got.steps] == ["invoke", "wait", "send"]


async def test_empty_steps_plan_round_trips(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    plan = _plan("p-empty", "noop goal", [])
    await mem.store(plan)
    got = await mem.retrieve(_activity("noop goal"))
    assert got == plan
    assert got is not None
    assert got.steps == []


# --------------------------------------------------------------------------------------------------
# Durability + isolation (the file backend re-reads from disk)
# --------------------------------------------------------------------------------------------------


async def test_plan_persists_across_memory_instances(tmp_path: Path) -> None:
    await _memory(tmp_path).store(_plan("p1", "durable goal"))
    # A fresh module over the same root — i.e. a process restart — sees the prior store.
    got = await _memory(tmp_path).retrieve(_activity("durable goal"))
    assert got is not None
    assert got.id == "p1"


async def test_retrieved_plan_is_isolated_from_store(tmp_path: Path) -> None:
    mem = _memory(tmp_path)
    await mem.store(_plan("p1", "schedule from email"))
    first = await mem.retrieve(_activity("schedule from email"))
    assert first is not None
    first.steps[0].params["mutated"] = True  # mutate the returned plan's step params
    second = await mem.retrieve(_activity("schedule from email"))
    assert second is not None
    assert "mutated" not in second.steps[0].params  # store untouched


# --------------------------------------------------------------------------------------------------
# infer(): the model path — behind a pluggable LLMClient, exercised here with a FakeLLMClient
# --------------------------------------------------------------------------------------------------


def _llm_memory(tmp_path: Path, llm: FakeLLMClient) -> ProceduralMemory:
    return ProceduralMemory(FileMemoryBackend(tmp_path), llm=llm)


async def test_infer_without_llm_raises(tmp_path: Path) -> None:
    # A store/retrieve-only ProceduralMemory (no LLM) cannot infer — it raises rather than
    # silently returning an empty plan.
    mem = _memory(tmp_path)  # constructed without an LLM
    with pytest.raises(RuntimeError):
        await mem.infer(_activity("anything"), {})


async def test_infer_builds_plan_from_model_json(tmp_path: Path) -> None:
    llm = FakeLLMClient(
        plan_json(
            {"action": "focus", "tool_id": "EmailClientApp"},
            {"action": "invoke", "tool_id": "EmailClientApp", "operation_name": "list_emails"},
            {"action": "unfocus", "tool_id": "EmailClientApp"},
        )
    )
    mem = _llm_memory(tmp_path, llm)
    tools: dict[str, Manual] = {"EmailClientApp": fake_manual("EmailClientApp", ["list_emails"])}

    plan = await mem.infer(_activity("triage inbox"), tools)

    assert plan.goal == "triage inbox"
    # The advertised vocabulary round-trips: focus/unfocus (generic non-invoke actions carrying
    # tool_id) and invoke (routing keys packed under TOOL_ID/OPERATION_NAME via invoke_step).
    assert plan.steps == [
        Step(next_action="focus", params={"tool_id": "EmailClientApp"}),
        invoke_step("EmailClientApp", "list_emails"),
        Step(next_action="unfocus", params={"tool_id": "EmailClientApp"}),
    ]


async def test_infer_defaults_missing_action_to_invoke(tmp_path: Path) -> None:
    # A step with no explicit "action" is treated as invoke (the common case).
    llm = FakeLLMClient(
        plan_json({"tool_id": "clock", "operation_name": "get_time", "params": {"tz": "UTC"}})
    )
    mem = _llm_memory(tmp_path, llm)

    plan = await mem.infer(_activity("what time"), {})

    assert plan.steps == [invoke_step("clock", "get_time", tz="UTC")]


async def test_infer_prompt_carries_goal_and_tool_operations(tmp_path: Path) -> None:
    llm = FakeLLMClient(plan_json({"action": "wait"}))
    mem = _llm_memory(tmp_path, llm)
    tools = {"EmailClientApp": fake_manual("EmailClientApp", ["list_emails", "send_email"])}

    await mem.infer(_activity("schedule from email"), tools)

    assert len(llm.calls) == 1
    _system, prompt = llm.calls[0]
    assert "schedule from email" in prompt  # the goal
    assert "EmailClientApp" in prompt  # the tool id (needed to emit invoke steps)
    assert "list_emails" in prompt and "send_email" in prompt  # its operations


async def test_infer_prompt_carries_observed_properties_and_signals(tmp_path: Path) -> None:
    # infer() shouldn't reason blind — the agent's currently-known world state (not just past
    # action results) is available to the planner too.
    llm = FakeLLMClient(plan_json({"action": "wait"}))
    mem = _llm_memory(tmp_path, llm)
    properties = [_property_percept("thermostat", "temperature", 72)]
    signals = [_signal_percept("clock", "tick", {"n": 1})]

    await mem.infer(_activity("g"), {}, PerceptSnapshot(properties, signals))

    _system, prompt = llm.calls[0]
    assert "thermostat.temperature = 72" in prompt
    assert "clock.tick" in prompt and '"n": 1' in prompt


async def test_infer_with_no_percepts_reports_none_observed(tmp_path: Path) -> None:
    # No properties/signals passed -> the default empty rendering, not an omitted section.
    llm = FakeLLMClient(plan_json({"action": "wait"}))
    mem = _llm_memory(tmp_path, llm)

    await mem.infer(_activity("g"), {})

    _system, prompt = llm.calls[0]
    assert prompt.count("(none observed yet)") == 2  # properties section + signals section


async def test_infer_tolerates_code_fences(tmp_path: Path) -> None:
    fenced = "```json\n" + plan_json({"action": "wait"}) + "\n```"
    mem = _llm_memory(tmp_path, FakeLLMClient(fenced))

    plan = await mem.infer(_activity("g"), {})

    assert plan.steps == [Step(next_action="wait", params={})]


async def test_infer_tolerates_prose_wrapped_json(tmp_path: Path) -> None:
    # Adaptive thinking sometimes emits a lead-in/trailing sentence around the JSON despite the ask;
    # the first balanced {...} is still recovered. Shared _load_json_object, so ground/select too.
    prose = "Sure, here is the plan:\n" + plan_json({"action": "wait"}) + "\nHope that helps!"
    mem = _llm_memory(tmp_path, FakeLLMClient(prose))

    plan = await mem.infer(_activity("g"), {})

    assert plan.steps == [Step(next_action="wait", params={})]


async def test_infer_tolerates_a_prose_lead_in_that_itself_contains_braces(tmp_path: Path) -> None:
    # Sharper than the above: the prose lead-in ITSELF contains a balanced brace-group before the
    # real JSON. Betting on the first balanced {...} would extract "{action}" and fail; the loader
    # tries each object in order until one parses, so the real plan object still wins.
    prose = "Use the {action} template, then: " + plan_json({"action": "wait"})
    mem = _llm_memory(tmp_path, FakeLLMClient(prose))

    plan = await mem.infer(_activity("g"), {})

    assert plan.steps == [Step(next_action="wait", params={})]


async def test_infer_rejects_non_json_output(tmp_path: Path) -> None:
    mem = _llm_memory(tmp_path, FakeLLMClient("sorry, I can't help with that"))
    with pytest.raises(ValueError):
        await mem.infer(_activity("g"), {})


async def test_infer_rejects_output_without_steps(tmp_path: Path) -> None:
    mem = _llm_memory(tmp_path, FakeLLMClient('{"plan": "no steps key here"}'))
    with pytest.raises(ValueError):
        await mem.infer(_activity("g"), {})


async def test_infer_result_is_storable_and_reusable(tmp_path: Path) -> None:
    # An inferred plan is an ordinary Plan: store it, and a later same-goal retrieve reuses it.
    llm = FakeLLMClient(
        plan_json({"action": "invoke", "tool_id": "clock", "operation_name": "get_time"})
    )
    mem = _llm_memory(tmp_path, llm)
    plan = await mem.infer(_activity("what time"), {})
    await mem.store(plan)

    got = await mem.retrieve(_activity("what time"))
    assert got == plan


# --------------------------------------------------------------------------------------------------
# PlanPrompt: planning *content* is customizable; the response contract stays fixed
# --------------------------------------------------------------------------------------------------


async def test_infer_uses_injected_prompt(tmp_path: Path) -> None:
    # A custom PlanPrompt fully controls what the LLM is asked (system + user), without subclassing
    # ProceduralMemory or re-implementing planning in a ReasonStrategy. The Protocol hands every
    # PlanPrompt the current observed world state too, not just the defaults.
    def custom_prompt(
        activity: Activity,
        tools: dict[str, Manual],
        observed: PerceptSnapshot,
        messages: list[Message],
    ) -> tuple[str, str]:
        return "SYS: plan tersely", (
            f"CUSTOM goal={activity.goal} tools={sorted(tools)} "
            f"properties={len(observed.properties)} signals={len(observed.signals)}"
        )

    llm = FakeLLMClient(plan_json({"action": "wait"}))
    mem = ProceduralMemory(FileMemoryBackend(tmp_path), llm=llm, prompt=custom_prompt)

    await mem.infer(_activity("g"), {"clock": fake_manual("clock", ["get_time"])})

    assert llm.calls == [
        ("SYS: plan tersely", "CUSTOM goal=g tools=['clock'] properties=0 signals=0")
    ]


async def test_infer_default_prompt_is_used_when_none_injected(tmp_path: Path) -> None:
    # No prompt argument -> the default PlanPrompt (goal + tool catalog under PLAN_SYSTEM_PROMPT).
    llm = FakeLLMClient(plan_json({"action": "wait"}))
    mem = ProceduralMemory(FileMemoryBackend(tmp_path), llm=llm)  # prompt defaulted

    await mem.infer(
        _activity("triage"), {"EmailClientApp": fake_manual("EmailClientApp", ["list"])}
    )

    system, user = llm.calls[0]
    assert system == PLAN_SYSTEM_PROMPT
    assert "triage" in user and "EmailClientApp" in user and "list" in user


def test_default_plan_prompt_exposes_reusable_pieces() -> None:
    # The default builder and its parts are public, so a custom PlanPrompt can reuse them.
    # observed is omittable — an unrelated caller isn't forced to supply one.
    tools = {"EmailClientApp": fake_manual("EmailClientApp", ["list_emails"])}
    system, user = default_plan_prompt(_activity("triage"), tools)
    assert system == PLAN_SYSTEM_PROMPT
    assert "triage" in user
    assert render_tools(tools) in user  # the tool rendering is a reusable public helper


def test_default_plan_prompt_includes_percept_rendering() -> None:
    tools = {"EmailClientApp": fake_manual("EmailClientApp", ["list_emails"])}
    properties = [_property_percept("EmailClientApp", "unread_count", 3)]
    signals = [_signal_percept("EmailClientApp", "new_email", {"id": 1})]
    system, user = default_plan_prompt(
        _activity("triage"), tools, PerceptSnapshot(properties, signals)
    )
    assert system == PLAN_SYSTEM_PROMPT
    assert render_properties(properties) in user
    assert render_signals(signals) in user


def test_render_messages_empty_and_populated() -> None:
    # The user-instruction channel: a bounded recent window of messages, "(none)" when empty.
    assert render_messages([]) == "(none)"
    msgs = [
        Message(sender="user", content={"text": "schedule the sync"}, received_at=0.0),
        Message(sender="user", content={"text": "actually make it Tuesday"}, received_at=1.0),
    ]
    rendered = render_messages(msgs)
    assert "user: schedule the sync" in rendered
    assert "user: actually make it Tuesday" in rendered


def test_default_plan_prompt_renders_history_and_messages() -> None:
    # A replan sees what has already been executed (so it doesn't repeat a side-effecting step) and
    # the recent user instructions (so a follow-up reaches inference, not just the goal string).
    tools = {"EmailClientApp": fake_manual("EmailClientApp", ["send_email"])}
    activity = _activity("reply to Alice")
    activity.history.append(
        CompletedOperation(
            OperationInvocation(tool_id="EmailClientApp", operation_name="send_email", params={}),
            OperationAck(ok=True, result={"sent": True}),
        )
    )
    messages = [Message(sender="user", content={"text": "nothing, continue"}, received_at=0.0)]
    _, user = default_plan_prompt(activity, tools, PerceptSnapshot(), messages)
    assert render_history(activity.history) in user
    assert render_messages(messages) in user


def _completed(operation_name: str) -> CompletedOperation:
    return CompletedOperation(
        OperationInvocation(tool_id="t", operation_name=operation_name, params={}),
        OperationAck(ok=True, result={"ok": True}),
    )


def test_render_history_keeps_the_most_recent_entries_and_marks_the_elision() -> None:
    # A silent drop reads as "nothing happened earlier", which is a different claim from "not
    # shown" — a planner told the former can legitimately decide to redo work already committed.
    history = [_completed(f"op{i}") for i in range(25)]

    rendered = render_history(history, 10)

    assert "(… 15 earlier operation(s) not shown)" in rendered
    assert "op24" in rendered and "op15" in rendered  # the recent tail is what survives
    assert "op14" not in rendered and "op0(" not in rendered


def test_render_history_is_unbounded_by_default() -> None:
    # The default is the grounding caller's: a $from/$decide reference may name any past result, so
    # hiding the entry that holds the referent fails the way truncating it mid-record does.
    history = [_completed(f"op{i}") for i in range(25)]

    rendered = render_history(history)

    assert "not shown" not in rendered
    assert "op0(" in rendered and "op24" in rendered


def test_render_history_says_nothing_about_elision_when_it_fits() -> None:
    assert "not shown" not in render_history([_completed("only_op")], 10)


def test_revalidation_carries_a_tighter_history_window_than_planning() -> None:
    # ADR-0024 justifies the re-check as much cheaper than the re-plan it guards, and it fires per
    # write rather than per plan. Rendering the same history in both is what erodes that, so these
    # two must actually differ — equal windows would leave the cost objection standing.
    assert _HISTORY_RENDER_REVALIDATE < _HISTORY_RENDER_PLAN


def test_default_plan_prompt_omits_superseded_section_when_none_parked() -> None:
    # The common case is a first plan, not a replan: no bundle parked -> no section, no framing
    # sentence telling the planner something was abandoned when nothing was.
    _, user = default_plan_prompt(_activity("schedule the sync"), {})
    assert "previous plan" not in user.lower()


def test_default_plan_prompt_renders_superseded_tail_across_the_subgoal_stack() -> None:
    # A discard taken inside a sub-plan drops the suspended parents too (ADR-0024), so the
    # replacement inference must see their un-run steps as well — a view of the active frame alone
    # would understate what it has to re-derive.
    activity = _activity("schedule the sync")
    sub = _plan("p1", "find a slot", [invoke_step("Cal", "read_slots"), invoke_step("Cal", "book")])
    parent = _plan(
        "p0", "schedule the sync", [invoke_step("Cal", "sub"), invoke_step("Cal", "tell")]
    )
    activity.superseded = SupersededPlan(plan=sub, step_index=1, parent_frames=[(parent, 0)])

    _, user = default_plan_prompt(activity, {})

    assert "previous plan for this goal was abandoned" in user
    assert "already executed 1 step(s) of that plan, inside a sub-plan nested under 1" in user
    assert "book" in user  # the active frame's un-run tail
    assert "tell" in user  # the suspended parent's un-run tail, dropped with it


def test_default_plan_prompt_superseded_section_omits_the_executed_prefix() -> None:
    # What already ran is rendered as *history* by this same prompt; repeating those steps as
    # *intent* would be redundant weight, so only the un-run tail is shown (ADR-0024).
    activity = _activity("schedule the sync")
    plan = _plan(
        "p1",
        "schedule the sync",
        [invoke_step("Cal", "already_ran"), invoke_step("Cal", "not_yet")],
    )
    activity.superseded = SupersededPlan(plan=plan, step_index=1, parent_frames=[])

    _, user = default_plan_prompt(activity, {})

    assert "not_yet" in user
    assert "already_ran" not in user


def test_default_plan_prompt_frames_the_superseded_plan_as_reusable_not_wrong() -> None:
    # Framing matters more than presence: told the old plan was *wrong*, a planner discards the
    # parts that were fine too — which is the entire benefit of showing it. It is stale, not wrong.
    activity = _activity("schedule the sync")
    activity.superseded = SupersededPlan(plan=_plan("p1", "g"), step_index=0, parent_frames=[])

    _, user = default_plan_prompt(activity, {})

    assert "Reuse whatever still applies" in user
    assert "stale, not wrong throughout" in user


def test_remaining_steps_is_what_revalidation_and_the_superseded_rendering_share() -> None:
    # One definition of "remaining" for both prompts: the active frame from step_index, then each
    # suspended parent's steps after the sub-goal that pushed it, innermost frame first.
    sub = _plan("p1", "sub", [invoke_step("t", "a"), invoke_step("t", "b")])
    mid = _plan("p0a", "mid", [invoke_step("t", "m0"), invoke_step("t", "m1")])
    root = _plan("p0", "root", [invoke_step("t", "r0"), invoke_step("t", "r1")])

    tail = remaining_steps(sub, 1, [(root, 0), (mid, 0)])

    assert [step.params["operation_name"] for step in tail] == ["b", "m1", "r1"]


async def test_infer_forwards_messages_to_prompt(tmp_path: Path) -> None:
    # The recent user messages reach the injected PlanPrompt (and thus the model), off the goal.
    captured: dict[str, object] = {}

    def capturing_prompt(
        activity: Activity,
        tools: dict[str, Manual],
        observed: PerceptSnapshot,
        messages: list[Message],
    ) -> tuple[str, str]:
        captured["messages"] = messages
        return "SYS", "USER"

    llm = FakeLLMClient(plan_json({"action": "wait"}))
    mem = ProceduralMemory(FileMemoryBackend(tmp_path), llm=llm, prompt=capturing_prompt)
    messages = [Message(sender="user", content={"text": "make it Tuesday"}, received_at=0.0)]

    await mem.infer(_activity("g"), {}, PerceptSnapshot(), messages)

    assert captured["messages"] == messages


# --------------------------------------------------------------------------------------------------
# render_tools: the full A&A usage interface (operations + observable properties + signals),
# so the planner can see what focusing a tool would perceive (what motivates a focus/unfocus step)
# --------------------------------------------------------------------------------------------------


def test_render_tools_surfaces_properties_and_signals_to_motivate_focus() -> None:
    manual = Manual(
        id="thermostat",
        metadata={},
        description="a thermostat",
        observable_properties=[
            ObservablePropertySpecification("temperature", "current reading", {})
        ],
        signals=[SignalSpecification("target_reached", "fires at the setpoint", {})],
        operations=[OperationSpecification("set_target", "set the setpoint", {})],
    )
    rendered = render_tools({"thermostat": manual})
    assert "operation `set_target`" in rendered
    assert "property `temperature`" in rendered  # a focusable observable — motivates focus
    assert "signal `target_reached`" in rendered  # a focusable event — motivates focus
    assert "focus to perceive" in rendered and "focus to receive" in rendered


def test_render_tools_surfaces_operation_parameter_schema() -> None:
    # Without the parameter schema in the prompt the planner guesses param names/formats and the
    # invoke fails against the real tool (e.g. ARE wants `start_datetime` in YYYY-MM-DD HH:MM:SS,
    # not a made-up `start`). The schema lives on OperationSpecification.parameters — render it.
    manual = Manual(
        id="CalendarApp",
        metadata={},
        description="a calendar",
        observable_properties=[],
        signals=[],
        operations=[
            OperationSpecification(
                "get_events",
                "list events in a range",
                {
                    "type": "object",
                    "properties": {
                        "start_datetime": {
                            "type": "string",
                            "description": "range start in YYYY-MM-DD HH:MM:SS",
                        },
                        "limit": {"type": "int", "description": "max events, default 10"},
                    },
                    "required": ["start_datetime"],
                },
            )
        ],
    )
    rendered = render_tools({"CalendarApp": manual})
    assert "start_datetime (string, required): range start in YYYY-MM-DD HH:MM:SS" in rendered
    assert "limit (int): max events, default 10" in rendered  # optional param -> no "required"
    assert "required" not in rendered.split("limit (int)")[1].split("\n")[0]


def test_render_tools_surfaces_operation_return_shape() -> None:
    # Without the result shape the planner guesses the $from path (e.g. `emails.0.id` against a bare
    # list keyed by `email_id`), it doesn't resolve mechanically, and grounding escalates to the
    # model. Rendering `returns:` lets it author `0.email_id` and resolve deterministically.
    manual = Manual(
        id="EmailClientApp",
        metadata={},
        description="mail",
        observable_properties=[],
        signals=[],
        operations=[
            OperationSpecification(
                "search_emails",
                "search across folders",
                {"type": "object", "properties": {"query": {"type": "string"}}},
                returns={
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sender": {"type": "string"},
                            "email_id": {"type": "string"},
                        },
                    },
                    "description": "A list of emails that match the query.",
                },
            )
        ],
    )
    rendered = render_tools({"EmailClientApp": manual})
    assert "returns: array of object with fields: sender, email_id" in rendered
    assert "(A list of emails that match the query.)" in rendered


def test_render_tools_surfaces_nested_record_fields_in_a_wrapped_result() -> None:
    # ARE's list_emails returns ReturnedEmails: an object whose `emails` field is a *list of
    # records* keyed by `email_id`. Flattening that field to a bare `object` hides the very field a
    # $from path (`emails.0.email_id`) must bind against, forcing the model escalation the returns
    # rendering exists to remove — so the nested record's field names have to surface.
    manual = Manual(
        id="EmailClientApp",
        metadata={},
        description="mail",
        observable_properties=[],
        signals=[],
        operations=[
            OperationSpecification(
                "list_emails",
                "list a folder",
                {"type": "object", "properties": {"folder": {"type": "string"}}},
                returns={
                    "type": "object",
                    "properties": {
                        "emails": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sender": {"type": "string"},
                                    "email_id": {"type": "string"},
                                },
                            },
                        },
                        "total_emails": {"type": "integer"},
                    },
                },
            )
        ],
    )
    rendered = render_tools({"EmailClientApp": manual})
    assert (
        "returns: object with fields: emails (array of object with fields: sender, email_id), "
        "total_emails" in rendered
    )


def test_render_tools_omits_absent_return_shape() -> None:
    # An operation with no declared returns (None) renders no `returns:` line at all.
    manual = Manual(
        id="clock",
        metadata={},
        description="",
        observable_properties=[],
        signals=[],
        operations=[OperationSpecification("get_time", "", {})],
    )
    assert "returns:" not in render_tools({"clock": manual})


def test_render_tools_omits_absent_affordance_groups() -> None:
    # An invoke-only tool (no observables/signals) shows operations only — nothing to focus.
    manual = Manual(
        id="clock",
        metadata={},
        description="",
        observable_properties=[],
        signals=[],
        operations=[OperationSpecification("get_time", "", {})],
    )
    rendered = render_tools({"clock": manual})
    assert "operation `get_time`" in rendered
    assert "focus" not in rendered  # no properties/signals => no focus framing for this tool


def test_render_tools_falls_back_to_authored_markdown_sections() -> None:
    # Hand-authored Markdown channel (structured specs empty): serve the prose sections, still under
    # the focus-framed group labels so the affordance is legible.
    raw = (
        "# Observable Properties\n- temperature: the current reading\n\n"
        "# Signals\n- target_reached\n\n"
        "# Operations\n- set_target: set the setpoint\n"
    )
    manual = Manual(
        id="thermostat",
        metadata={},
        description="",
        observable_properties=[],
        signals=[],
        operations=[],
        raw_text=raw,
    )
    rendered = render_tools({"thermostat": manual})
    assert "temperature" in rendered and "target_reached" in rendered and "set_target" in rendered
    assert "focus to perceive" in rendered  # the prose section keeps the focus-framed label


def test_render_tools_surfaces_usage_protocols_and_safety() -> None:
    # The constraints a plan must respect (part 6) reach the planner. Prose-only (authored
    # Markdown), and the planning system prompt tells the model to honor them.
    raw = (
        "# Operations\n- start: begin pumping\n\n"
        "# Usage Protocols & Safety\nNever start the pump while the intake valve is closed.\n"
    )
    manual = Manual(
        id="pump",
        metadata={},
        description="",
        observable_properties=[],
        signals=[],
        operations=[],
        raw_text=raw,
    )
    rendered = render_tools({"pump": manual})
    assert "usage protocols & safety" in rendered.lower()
    assert "intake valve is closed" in rendered  # the safety constraint reaches the planner
    assert "safety" in PLAN_SYSTEM_PROMPT.lower()  # and the model is told to respect it


def test_plan_prompt_instructs_references_for_runtime_dependent_params() -> None:
    # The planner must emit references (not invented literals) for values only known at runtime.
    assert "$from" in PLAN_SYSTEM_PROMPT and "$decide" in PLAN_SYSTEM_PROMPT


def test_plan_prompt_steers_conditional_selection_to_a_decide_filter() -> None:
    # A selection with a CONDITIONAL tie-break (only applies among items tied on the primary key,
    # or count-dependent) must NOT be resolved by `sort` + `take` — taking first collapses the tie
    # by incidental order and drops the candidates the rule needs. The planner is told to sort to
    # gather the candidates, then apply the whole rule in one $decide filter over that collection.
    lowered = PLAN_SYSTEM_PROMPT.lower()
    assert "tie-break" in lowered
    assert "sort" in lowered and "take" in lowered and "$decide" in lowered
    # The specific trap named: don't `take` before the conditional rule sees the tied candidates.
    assert "discards" in lowered or "before the rule" in lowered
    # But it must NOT over-route: a plain top-N stays sort+take even when a tie is merely possible —
    # only a goal-SPECIFIED tie-break rule escalates to $decide (guards against the $decide path
    # swallowing ordinary top-N selections and paying a needless, nondeterministic model call).
    assert "plain top-n" in lowered


def test_plan_prompt_keeps_visible_identifiers_as_references() -> None:
    # An observed identifier (an email/event id currently visible in properties) must still be
    # referenced, not hardcoded — otherwise a goal-keyed cached plan bakes in one run's ids and
    # fails against the next run's data. Guards the "use a visible value directly" guidance from
    # swallowing volatile ids again.
    lowered = PLAN_SYSTEM_PROMPT.lower()
    assert "identifier" in lowered and "reusable" in lowered


# --------------------------------------------------------------------------------------------------
# render_properties / render_signals: rendering the agent's currently-known world state (not just
# past action results) for a planning/grounding prompt — see WorkingMemory's replace-by-key
# properties snapshot vs. append-log signals split (ADR-0012/ADR-0019).
# --------------------------------------------------------------------------------------------------


def test_render_properties_formats_source_name_value() -> None:
    rendered = render_properties([_property_percept("thermostat", "temperature", 72)])
    assert rendered == "- thermostat.temperature = 72"


def test_render_properties_empty_reports_none_observed() -> None:
    assert render_properties([]) == "(none observed yet)"


def test_render_properties_renders_json_literals_not_python_repr() -> None:
    # A model is told to reuse an already-observed value verbatim in its own strict-JSON answer —
    # rendering with repr() would show it Python's False/None/True, not JSON's false/null/true.
    rendered = render_properties([_property_percept("door", "locked", False)])
    assert rendered == "- door.locked = false"
    rendered = render_properties([_property_percept("door", "code", None)])
    assert rendered == "- door.code = null"


def test_render_properties_truncates_long_values() -> None:
    # Unbounded like render_history would be — a large property value must not grow every
    # planning/grounding prompt without bound.
    rendered = render_properties([_property_percept("sensor", "buffer", "x" * 1000)])
    assert len(rendered) < 500
    assert rendered.endswith("…")


def test_render_signals_formats_source_name_payload() -> None:
    rendered = render_signals([_signal_percept("clock", "tick", {"n": 1})])
    assert "clock.tick" in rendered and '"n": 1' in rendered


def test_render_signals_empty_reports_none_observed() -> None:
    assert render_signals([]) == "(none observed yet)"


def test_render_signals_keeps_duplicate_occurrences() -> None:
    # Signals are an append log — the same name from the same source twice is two distinct
    # occurrences, not a repeat to collapse.
    percepts = [
        _signal_percept("EmailClientApp", "new_email", {"id": 1}),
        _signal_percept("EmailClientApp", "new_email", {"id": 2}),
    ]
    rendered = render_signals(percepts)
    assert rendered.count("EmailClientApp.new_email") == 2
    assert '"id": 1' in rendered and '"id": 2' in rendered


def test_render_signals_falls_back_to_str_for_non_json_serializable_payload() -> None:
    # A signal payload isn't guaranteed JSON-safe (e.g. an adapter pushing a datetime) — that must
    # degrade to a str() rendering, not crash the whole infer()/ground() call.
    percept = _signal_percept("sensor", "reading", {"at": object()})
    rendered = render_signals([percept])
    assert "sensor.reading" in rendered  # renders instead of raising TypeError


def test_render_signals_truncates_long_payloads() -> None:
    rendered = render_signals([_signal_percept("clock", "tick", {"data": "x" * 1000})])
    assert len(rendered) < 500
    assert rendered.endswith("…")


# --------------------------------------------------------------------------------------------------
# ground(): the escalation model call — decide an operation's params from execution history
# --------------------------------------------------------------------------------------------------


def _params_json(**params: object) -> str:
    import json

    return json.dumps({"params": params})


def _activity_with_history(goal: str, operation_name: str, result: object) -> Activity:
    inv = OperationInvocation("EmailClientApp", operation_name, {"query": "Alice"})
    return Activity(
        id=f"act-{goal}",
        goal=goal,
        context={},
        history=[CompletedOperation(inv, OperationAck(ok=True, result=result))],
    )


async def test_ground_without_llm_raises(tmp_path: Path) -> None:
    mem = ProceduralMemory(FileMemoryBackend(tmp_path))  # store/retrieve only, no model
    with pytest.raises(RuntimeError, match="no LLM configured"):
        await mem.ground(_activity_with_history("g", "search", {}), "reply", None, {})


async def test_ground_returns_concrete_params(tmp_path: Path) -> None:
    llm = FakeLLMClient(_params_json(email_id=42, body="hi Alice"))
    mem = ProceduralMemory(FileMemoryBackend(tmp_path), llm=llm)
    activity = _activity_with_history("reply", "search_emails", {"emails": [{"id": 42}]})

    params = await mem.ground(
        activity, "reply_to_email", None, {"email_id": {"$decide": "Alice's id"}}
    )

    assert params == {"email_id": 42, "body": "hi Alice"}


async def test_ground_prompt_carries_operation_schema_and_history(tmp_path: Path) -> None:
    llm = FakeLLMClient(_params_json(email_id=42))
    mem = ProceduralMemory(FileMemoryBackend(tmp_path), llm=llm)
    manual = Manual(
        id="EmailClientApp",
        metadata={},
        description="email",
        observable_properties=[],
        signals=[],
        operations=[
            OperationSpecification(
                "reply_to_email",
                "reply to a message",
                {
                    "type": "object",
                    "properties": {"email_id": {"type": "int", "description": "id to reply to"}},
                    "required": ["email_id"],
                },
            )
        ],
    )
    activity = _activity_with_history("reply", "search_emails", {"emails": [{"id": 42}]})

    await mem.ground(activity, "reply_to_email", manual, {"email_id": {"$decide": "x"}})

    system, user = llm.calls[0]
    assert system == GROUND_SYSTEM_PROMPT
    assert "email_id" in user and "id to reply to" in user  # the operation schema
    assert "search_emails" in user and "42" in user  # the prior result (history)


async def test_ground_prompt_carries_observed_properties_and_signals(tmp_path: Path) -> None:
    # Grounding shouldn't reason blind either — the current world state can settle a param a prior
    # operation result alone can't (e.g. a value already visible as an observed property).
    llm = FakeLLMClient(_params_json(email_id=42))
    mem = ProceduralMemory(FileMemoryBackend(tmp_path), llm=llm)
    activity = _activity_with_history("reply", "search_emails", {"emails": [{"id": 42}]})
    properties = [_property_percept("EmailClientApp", "unread_count", 3)]
    signals = [_signal_percept("EmailClientApp", "new_email", {"id": 42})]

    await mem.ground(
        activity,
        "reply_to_email",
        None,
        {"email_id": {"$decide": "x"}},
        PerceptSnapshot(properties, signals),
    )

    _system, prompt = llm.calls[0]
    assert "EmailClientApp.unread_count = 3" in prompt
    assert "EmailClientApp.new_email" in prompt


def test_default_ground_prompt_includes_percept_rendering() -> None:
    activity = _activity_with_history("reply", "search_emails", {"emails": [{"id": 42}]})
    properties = [_property_percept("EmailClientApp", "unread_count", 3)]
    signals = [_signal_percept("EmailClientApp", "new_email", {"id": 42})]

    system, user = default_ground_prompt(
        activity, "reply_to_email", None, {}, PerceptSnapshot(properties, signals)
    )

    assert system == GROUND_SYSTEM_PROMPT
    assert render_properties(properties) in user
    assert render_signals(signals) in user


def test_render_bindings_names_each_binding_and_its_value() -> None:
    rendered = render_bindings({"newly_sorted": [{"name": "Mechelen Mansion", "price": 2000}]})
    assert "newly_sorted" in rendered
    assert "Mechelen Mansion" in rendered and "2000" in rendered


def test_render_bindings_empty_is_a_placeholder_not_blank() -> None:
    assert render_bindings({}) == "(none)"


async def test_ground_prompt_carries_named_data_op_bindings(tmp_path: Path) -> None:
    # A $decide that names a binding an earlier data-op step produced (a filtered/sorted collection)
    # must see that binding's ACTUAL data — otherwise the grounder can only emit a placeholder. This
    # is the RentAFlat email regression: content = "$decide: names+prices of the two cheapest from
    # newly_sorted" grounded to boilerplate because the binding wasn't in the prompt.
    llm = FakeLLMClient(_params_json(content="Mechelen Mansion ($2,000)"))
    mem = ProceduralMemory(FileMemoryBackend(tmp_path), llm=llm)
    activity = _activity_with_history("email", "list_saved_apartments", {"saved": []})
    activity.bindings["newly_sorted"] = [{"name": "Mechelen Mansion", "price": 2000}]

    await mem.ground(
        activity,
        "send_email",
        None,
        {"content": {"$decide": "names and prices of the two cheapest from newly_sorted"}},
    )

    _system, prompt = llm.calls[0]
    assert "newly_sorted" in prompt
    assert "Mechelen Mansion" in prompt and "2000" in prompt


async def test_select_tolerates_prose_wrapped_keep(tmp_path: Path) -> None:
    # The regression seen in a run log: a prose-wrapped {"keep": [...]} used to die on a bare
    # json.loads, failing the whole $decide filter. Now recovered via the shared extractor.
    llm = FakeLLMClient('Looking at the items, I would keep: {"keep": [1]}')
    mem = ProceduralMemory(FileMemoryBackend(tmp_path), llm=llm)

    kept = await mem.select(_activity_with_history("f", "s", {}), ["a", "b", "c"], "the middle one")

    assert kept == ["b"]


async def test_ground_rejects_non_json(tmp_path: Path) -> None:
    mem = ProceduralMemory(FileMemoryBackend(tmp_path), llm=FakeLLMClient("sorry, no"))
    with pytest.raises(ValueError, match="could not parse grounded params"):
        await mem.ground(_activity_with_history("g", "s", {}), "op", None, {})


async def test_ground_rejects_output_without_params_key(tmp_path: Path) -> None:
    import json

    mem = ProceduralMemory(FileMemoryBackend(tmp_path), llm=FakeLLMClient(json.dumps({"x": 1})))
    with pytest.raises(ValueError, match="could not parse grounded params"):
        await mem.ground(_activity_with_history("g", "s", {}), "op", None, {})


# -- history/bindings rendering is all-or-nothing under a total budget ----------------------------
#
# The bug these pin: a paged tool result (ARE's `get_contacts` -> 10 records + a
# `metadata: {range, total: 125}` envelope, ~7.6k chars) was rendered through a 4000-char per-value
# cap, so the model saw ~6.5 records and *no* sign that 115 more existed. The paging affordance
# lives in the tail, which is exactly what a prefix cut removes — and a record cut mid-field
# yields a truncated email address that still looks like one. Whole-or-absent is the invariant; the
# budget bounds the prompt instead.


def _completed_with(operation_name: str, result: object) -> CompletedOperation:
    return CompletedOperation(
        OperationInvocation(tool_id="t", operation_name=operation_name, params={}),
        OperationAck(ok=True, result=result),
    )


def test_render_history_keeps_a_large_result_whole() -> None:
    # The regression: 7637 chars is the measured size of a real get_contacts page, comfortably over
    # the 4000-char cap this replaces. The envelope at the very end must survive.
    body = "x" * 7600 + "metadata:{'total':125}"
    rendered = render_history([_completed_with("get_contacts", body)])
    assert "metadata:{'total':125}" in rendered
    assert "…" not in rendered


def test_render_history_elides_a_whole_entry_rather_than_cutting_it() -> None:
    big, small = "A" * 5000, "B" * 10
    rendered = render_history(
        [_completed_with("big_op", big), _completed_with("small_op", small)], budget=1000
    )
    # the over-budget entry is named and counted, but no prefix of its result leaks through
    assert "big_op" in rendered
    assert "elided" in rendered and "5000 chars" in rendered
    assert "A" * 20 not in rendered
    assert small in rendered  # the newest entry is still whole


def test_render_history_never_emits_a_prefix_of_a_result() -> None:
    # The invariant, stated directly: every result is present in full or not at all. Each body is
    # given a unique head *and* tail so "a prefix of this result leaked" is actually detectable —
    # bodies sharing a long common prefix would make the check vacuous.
    results = [f"head{i}-" + "R" * 900 + f"-tail{i}" for i in range(6)]
    rendered = render_history(
        [_completed_with(f"op{i}", r) for i, r in enumerate(results)], budget=2000
    )
    for r in results:
        assert (r in rendered) or (r[:50] not in rendered)


def test_render_history_always_renders_the_newest_entry_in_full() -> None:
    # A $from/$decide resolves against the newest result in the common case, so a budget that could
    # elide it would reintroduce the same silent loss under a new name.
    body = "Z" * 9000
    rendered = render_history([_completed_with("only_op", body)], budget=100)
    assert body in rendered


def test_render_history_preserves_oldest_to_newest_order_after_budget_selection() -> None:
    entries = [_completed_with(f"op{i}", "q" * 400) for i in range(6)]
    rendered = render_history(entries, budget=1200)
    positions = [rendered.index(f"op{i}") for i in range(6)]
    assert positions == sorted(positions)


def test_render_history_budget_composes_with_the_entry_window() -> None:
    history = [_completed_with(f"op{i}", "w" * 800) for i in range(20)]
    rendered = render_history(history, 5, budget=1000)
    assert "15 earlier operation(s) not shown" in rendered  # window marker still there
    assert "elided" in rendered  # and the budget still trims within the window
    assert "op19" in rendered


def test_render_bindings_is_all_or_nothing_and_keeps_the_last_binding() -> None:
    rendered = render_bindings(
        {"wide": [{"pad": "p" * 4000}], "cheapest": [{"name": "Mechelen Mansion"}]}, budget=500
    )
    assert "Mechelen Mansion" in rendered  # newest binding survives whole
    assert "elided" in rendered and "p" * 20 not in rendered
    assert rendered.index("wide") < rendered.index("cheapest")  # insertion order preserved


# -- goal provenance: only the plan carrying the user's goal reports back --------------------------
#
# PLAN_SYSTEM_PROMPT says to end with `send_message_to_user` "if the goal came from the user", but a
# sub-goal is inferred against its own goal string with the parent nowhere in the prompt, so the
# model could not evaluate the condition and every plan in the tree read as the user's. One turn
# then produced one report per plan — observed as three messages to the user in a single Gaia2 turn,
# where the benchmark's turn gate counts exactly one.


def test_plan_prompt_tells_a_subgoal_plan_not_to_report_to_the_user() -> None:
    activity = _activity("notify each relative")
    activity.parent_frames = [(Plan(id="p", goal="reconcile the shortlist", steps=[]), 0)]
    _, user = default_plan_prompt(activity, {}, PerceptSnapshot(), [])
    assert "NOT a request from the user" in user
    assert "do NOT end this plan by invoking `send_message_to_user`" in user


def test_plan_prompt_says_nothing_about_provenance_for_a_top_level_goal() -> None:
    # The empty stack is the top-level case, which must read exactly as before: the system prompt's
    # own condition applies, unqualified.
    _, user = default_plan_prompt(_activity("reconcile the shortlist"), {}, PerceptSnapshot(), [])
    assert "NOT a request from the user" not in user
    assert "send_message_to_user" not in user
