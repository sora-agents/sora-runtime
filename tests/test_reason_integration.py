"""Skip-gated integration test: the real, model-backed planning path.

Drives the shipped ``AnthropicLLMClient`` through ``ProceduralMemory.infer()`` against real Claude,
proving E2's "first real, model-backed reasoning" end-to-end. **Opt-in and skip-gated** (same shape
as ``test_are_walking_skeleton.py``): marked ``integration`` — excluded from the default ``pytest``
run (see pyproject ``addopts``) — and it needs both the ``llm`` extra (``uv sync --extra llm``) and
a live ``ANTHROPIC_API_KEY``. CI stays deterministic; the deterministic per-layer contract lives in
``test_procedural_memory.py`` (over a ``FakeLLMClient``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fakes import fake_manual
from sora.activity import Activity
from sora.memory import FileMemoryBackend, ProceduralMemory
from sora.types import Step


@pytest.mark.integration
async def test_infer_produces_a_plan_against_real_claude(tmp_path: Path) -> None:
    pytest.importorskip("anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    from sora.adapters.anthropic_llm import AnthropicLLMClient

    llm = AnthropicLLMClient()  # model id defaults, but is a config value (ctor arg)
    mem = ProceduralMemory(FileMemoryBackend(tmp_path), llm=llm)
    tools = {
        "EmailClientApp": fake_manual("EmailClientApp", ["list_emails", "send_email"]),
        "CalendarApp": fake_manual("CalendarApp", ["list_events", "create_event"]),
    }
    activity = Activity(
        id="a",
        goal="find the meeting proposed in my inbox and add it to my calendar",
        context={},
    )

    try:
        plan = await mem.infer(activity, tools)
        assert plan.goal == activity.goal
        assert plan.steps, "the model returned an empty plan"
        assert all(isinstance(step, Step) for step in plan.steps)
        # No hallucinated tools: every invoke step references a tool id from the given catalog.
        for step in plan.steps:
            if step.next_action == "invoke":
                assert step.params["tool_id"] in tools
    finally:
        await llm.aclose()
