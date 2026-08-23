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
from pathlib import Path

import pytest

from fakes import FakeLLMClient
from sora.activity import Activity
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
    """The repair runs last, so the existing prose fallback still wins where it applies."""
    assert _load_json_object('Here is the plan: {"steps": []}') == {"steps": []}


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
    retry_prompt = llm.calls[1][1]
    assert "could not be parsed" in retry_prompt
    assert "not json at all" in retry_prompt  # it is shown its own output, to fix in place


@pytest.mark.asyncio
async def test_a_repairable_plan_costs_no_second_call(tmp_path: Path) -> None:
    """The free repair runs first, so a stray brace never buys a round trip."""
    llm = FakeLLMClient(['{"steps": [{"action": "focus", "tool_id": "t1"}]}}'])
    memory = ProceduralMemory(FileMemoryBackend(tmp_path), llm=llm)
    plan = await memory.infer(_activity(), {})
    assert len(plan.steps) == 1
    assert len(llm.calls) == 1


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
