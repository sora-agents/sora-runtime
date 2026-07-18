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
    PLAN_SYSTEM_PROMPT,
    FileMemoryBackend,
    ProceduralMemory,
    default_plan_prompt,
    render_tools,
)
from sora.types import Plan, Step


def _activity(goal: str) -> Activity:
    return Activity(id=f"act-{goal}", goal=goal, context={})


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


async def test_infer_tolerates_code_fences(tmp_path: Path) -> None:
    fenced = "```json\n" + plan_json({"action": "wait"}) + "\n```"
    mem = _llm_memory(tmp_path, FakeLLMClient(fenced))

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
    # ProceduralMemory or re-implementing planning in a ReasonStrategy.
    def custom_prompt(activity: Activity, tools: dict[str, Manual]) -> tuple[str, str]:
        return "SYS: plan tersely", f"CUSTOM goal={activity.goal} tools={sorted(tools)}"

    llm = FakeLLMClient(plan_json({"action": "wait"}))
    mem = ProceduralMemory(FileMemoryBackend(tmp_path), llm=llm, prompt=custom_prompt)

    await mem.infer(_activity("g"), {"clock": fake_manual("clock", ["get_time"])})

    assert llm.calls == [("SYS: plan tersely", "CUSTOM goal=g tools=['clock']")]


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
    tools = {"EmailClientApp": fake_manual("EmailClientApp", ["list_emails"])}
    system, user = default_plan_prompt(_activity("triage"), tools)
    assert system == PLAN_SYSTEM_PROMPT
    assert "triage" in user
    assert render_tools(tools) in user  # the tool rendering is a reusable public helper


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
