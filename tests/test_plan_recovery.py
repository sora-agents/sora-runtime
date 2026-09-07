"""Recovering a plan whose JSON is malformed — the two fallbacks under ``ProceduralMemory.infer``.

An inferred plan is one model call, and on a local reasoning model that call is minutes and tens of
thousands of tokens. Losing all of it to a syntax slip — and, because a failed inference terminates
the activity, losing the activity with it — is the failure these two layers exist to stop. They are
ordered cheapest-first and neither one invents content:

1. ``_drop_surplus_closers`` deletes closers that have no valid reading at all, and the result is
   still handed to ``json.loads`` rather than trusted. Free.
2. one re-inference that shows the model its own parse error. A second round trip, so it runs only
   when the free repair could not help.

The motivating case is the 2026-08-21 adaptability run: a 2712-character plan — eight good steps and
a well-formed pending condition — discarded over one stray brace at the tail of its `pending` block.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from fakes import FakeLLMClient
from sora.activity import Activity
from sora.llm import LLMMeter, MeteredLLMClient, current_inference_id
from sora.memory import (
    FileMemoryBackend,
    ProceduralMemory,
    _drop_surplus_closers,
    _load_json_object,
)

# --------------------------------------------------------------------------------------------------
# the free repair
# --------------------------------------------------------------------------------------------------


def test_a_surplus_closing_brace_is_dropped() -> None:
    assert _load_json_object('{"a": 1}}') == {"a": 1}


def test_a_surplus_bracket_is_dropped() -> None:
    assert _load_json_object('{"a": [1, 2]]}') == {"a": [1, 2]}


def test_a_mismatched_closer_is_dropped() -> None:
    """A ``]`` where an object is open closes nothing — it cannot be what was meant."""
    assert _load_json_object('{"a": 1]}') == {"a": 1}


def test_the_real_failing_plan_is_recovered() -> None:
    """The shape of the model output from the failing run, surplus brace and all."""
    raw = (
        '{"steps": [{"action": "invoke", "tool_id": "insim:are/Emails", '
        '"operation_name": "send_email", "params": {"recipients": ["a@b.c"]}}], '
        '"pending": [{"watch": {"signal": "state_changed", "source": "insim:are/Emails", '
        '"path": "folders.INBOX"}, "when": "he replies", "then": "rebook", '
        '"until": "the day has passed"}}]}'
    )
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)
    recovered = _load_json_object(raw)
    assert len(recovered["steps"]) == 1
    assert recovered["pending"][0]["watch"]["path"] == "folders.INBOX"
    assert recovered["pending"][0]["until"] == "the day has passed"


def test_a_surplus_closer_mid_document_is_repaired_not_mined_for_a_fragment() -> None:
    """A surplus closer *inside* the document balances it early, so everything after that point
    scans as a top-level object of its own — and the first of those that parses is a single step:
    a perfectly valid object that is not the plan. The caller then dies on a missing ``steps`` key,
    handed a "repair" that discarded the plan it was repairing.

    Recorded from the 2026-09-07 gaia2-cli time-scenario run: a three-step plan, its maintenance
    sub-goal and pending condition intact, came back as its own last step and the run terminated on
    ``KeyError('steps')`` — with the one retry reproducing the same output verbatim, so the free
    repair was the only thing that could have saved it. It could: reading this document is what it
    is for, and it never got the chance. Ordering is the whole defect."""
    raw = (
        '{"steps":[{"action":"focus","tool_id":"/home/agent/bin/calendar"},'
        '{"action":"subgoal","goal":"delete events overlapping the added ones",'
        '"mode":"mechanical","goal_kind":"maintenance",'
        '"pending":[{"watch":{"signal":"env_notification","source":"/home/agent/bin/calendar"},'
        '"when":"events were added","then":"delete the conflicting ones",'
        '"until":{"text":"four minutes have passed","seconds":240}}}]},'  # the surplus brace
        '{"action":"invoke","tool_id":"runtime/UserChannel",'
        '"operation_name":"send_message_to_user","params":{"text":"done"}}]}'
    )
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)
    recovered = _load_json_object(raw)
    assert [step["action"] for step in recovered["steps"]] == ["focus", "subgoal", "invoke"]
    condition = recovered["steps"][1]["pending"][0]
    assert condition["until"] == {"text": "four minutes have passed", "seconds": 240}


def test_a_closer_inside_a_string_is_never_dropped() -> None:
    assert _load_json_object('{"a": "}]}"}') == {"a": "}]}"}


def test_an_escaped_quote_does_not_end_the_string() -> None:
    assert _load_json_object('{"a": "say \\"}\\" now"}}') == {"a": 'say "}" now'}


def test_valid_json_is_left_alone() -> None:
    assert _drop_surplus_closers('{"a": [1]}') is None


def test_an_unclosed_tail_is_not_completed() -> None:
    """The repair never closes what is open. Completing a truncated response would turn it into a
    shorter-but-plausible plan, and a plan silently missing its last steps is worse than one that
    failed to parse — it looks like it succeeded."""
    truncated = '{"steps": [{"action": "focus", "tool_id": "t1"}'
    assert _drop_surplus_closers(truncated) is None
    with pytest.raises(json.JSONDecodeError):
        _load_json_object(truncated)


def test_repair_does_not_shadow_prose_wrapped_json() -> None:
    """The repair runs first, but it declines on anything it has no reading for: prose has no
    surplus closer, so it returns None and the scan over balanced groups still wins here."""
    assert _load_json_object('Here is the plan: {"steps": []}') == {"steps": []}


def test_a_stray_closer_in_the_prose_does_not_cost_the_embedded_object() -> None:
    """Both fallbacks in sequence: the repair removes the stray closer, the remaining text is still
    prose rather than JSON, and the scan then finds the object that was embedded in it."""
    assert _load_json_object('oops } — here is the plan: {"steps": []}') == {"steps": []}


# --------------------------------------------------------------------------------------------------
# the one re-inference
# --------------------------------------------------------------------------------------------------


def _activity() -> Activity:
    return Activity(id="a1", goal="do the thing", context={})


@pytest.mark.asyncio
async def test_unrepairable_output_is_retried_once_with_the_error(tmp_path: Path) -> None:
    llm = FakeLLMClient(["not json at all", '{"steps": [{"action": "focus", "tool_id": "t1"}]}'])
    memory = ProceduralMemory(FileMemoryBackend(tmp_path), llm=llm)
    plan = await memory.infer(_activity(), {})
    assert len(plan.steps) == 1
    assert len(llm.calls) == 2
    assert [(request.semantic_label, request.prompt_version) for request in llm.requests] == [
        ("plan", "1"),
        ("plan", "1"),
    ]
    retry_prompt = llm.calls[1][1]
    assert "could not be parsed" in retry_prompt
    assert "not json at all" in retry_prompt  # it is shown its own output, to fix in place


@pytest.mark.asyncio
async def test_a_successful_parse_retry_is_counted_as_a_repair(tmp_path: Path) -> None:
    llm = FakeLLMClient(["not json at all", '{"steps": []}'])
    memory = ProceduralMemory(FileMemoryBackend(tmp_path), llm=MeteredLLMClient(llm))
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    previous = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(meter)
    token = current_inference_id.set("inf-retry")
    try:
        await memory.infer(_activity(), {})
    finally:
        current_inference_id.reset(token)
        logger.removeHandler(meter)
        logger.setLevel(previous)

    (inference,) = meter.report().inferences
    assert inference.round_trips == 2
    assert inference.malformed_fields_repaired == 1


@pytest.mark.asyncio
async def test_a_repairable_plan_costs_no_second_call(tmp_path: Path) -> None:
    """The free repair runs first, so a stray brace never buys a round trip."""
    llm = FakeLLMClient(['{"steps": [{"action": "focus", "tool_id": "t1"}]}}'])
    memory = ProceduralMemory(FileMemoryBackend(tmp_path), llm=llm)
    plan = await memory.infer(_activity(), {})
    assert len(plan.steps) == 1
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_report_counts_repaired_json_and_dropped_malformed_plan_fields(
    tmp_path: Path,
) -> None:
    raw = (
        json.dumps(
            {
                "steps": [
                    {
                        "action": "subgoal",
                        "goal": "child",
                        "goal_kind": ["not", "a", "label"],
                    }
                ],
                "pending": [
                    {"watch": {}, "when": "reply arrives", "then": "follow up"},
                    "not an object",
                    {
                        "watch": {"signal": "changed", "kind": "moved"},
                        "when": "it changes",
                        "then": "follow up",
                        "until": {"text": "tomorrow", "seconds": "soon"},
                    },
                ],
            }
        )
        + "}"
    )
    llm = FakeLLMClient(raw)
    memory = ProceduralMemory(FileMemoryBackend(tmp_path), llm=MeteredLLMClient(llm))
    meter = LLMMeter()
    logger = logging.getLogger("sora.llm")
    previous = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(meter)
    token = current_inference_id.set("inf-malformed")
    try:
        plan = await memory.infer(_activity(), {})
    finally:
        current_inference_id.reset(token)
        logger.removeHandler(meter)
        logger.setLevel(previous)

    assert len(plan.steps) == 1
    assert len(plan.pending) == 1
    assert plan.pending[0].watch.kind is None
    assert plan.pending[0].until is not None
    assert plan.pending[0].until.seconds is None
    (inference,) = meter.report().inferences
    assert inference.malformed_fields_dropped == 5
    assert inference.malformed_fields_repaired == 1


@pytest.mark.asyncio
async def test_a_clean_plan_costs_no_second_call(tmp_path: Path) -> None:
    llm = FakeLLMClient(['{"steps": [{"action": "focus", "tool_id": "t1"}]}'])
    memory = ProceduralMemory(FileMemoryBackend(tmp_path), llm=llm)
    await memory.infer(_activity(), {})
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_the_retry_is_not_a_loop(tmp_path: Path) -> None:
    """Twice-unparseable raises, exactly as one failed parse did before — no third attempt."""
    llm = FakeLLMClient(["nonsense", "still nonsense"])
    memory = ProceduralMemory(FileMemoryBackend(tmp_path), llm=llm)
    with pytest.raises(ValueError):
        await memory.infer(_activity(), {})
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_the_retry_keeps_the_declared_conditions(tmp_path: Path) -> None:
    """Steps and pending parse from the same object and share the one retry, so a recovered plan
    never comes back complete-looking but silently stripped of its gate."""
    good = json.dumps(
        {
            "steps": [{"action": "focus", "tool_id": "t1"}],
            "pending": [
                {
                    "watch": {"signal": "state_changed", "source": "t1"},
                    "when": "he replies",
                    "then": "rebook",
                }
            ],
        }
    )
    llm = FakeLLMClient(["{{{", good])
    memory = ProceduralMemory(FileMemoryBackend(tmp_path), llm=llm)
    plan = await memory.infer(_activity(), {})
    assert len(plan.pending) == 1
    assert plan.pending[0].then == "rebook"
