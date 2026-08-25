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
from sora.bootstrap import load_dotenv
from sora.manual import Manual
from sora.memory import FileMemoryBackend, ProceduralMemory
from sora.types import Step


async def _user_channel() -> dict[str, Manual]:
    """The agent's own reply channel, exactly as bootstrap always joins it.

    ``PLAN_SYSTEM_PROMPT`` requires a plan for a user-given goal to end by invoking this tool, so a
    catalog without it asks the model to obey an instruction it cannot satisfy — and an obedient
    model then invents an id, tripping the no-hallucinated-tools assertion below on a plan that was
    actually correct. Built through the real adapter rather than a literal id so it cannot drift
    from what bootstrap wires."""
    from sora.adapters.runtime_io import RUNTIME_IO_ADAPTER, RUNTIME_IO_ADDRESS, RuntimeIOAdapter
    from sora.environment import WorkspaceOrigin
    from sora.transport import InProcessTransport

    adapter = RuntimeIOAdapter(
        origin=WorkspaceOrigin(adapter=RUNTIME_IO_ADAPTER, address=RUNTIME_IO_ADDRESS),
        transport=InProcessTransport(),
    )
    (workspace,) = await adapter.discover()
    return {tool.id: tool.manual for tool in workspace.tools()}


@pytest.mark.integration
async def test_infer_produces_a_plan_against_real_claude(tmp_path: Path) -> None:
    pytest.importorskip("anthropic")
    # The same convenience `build_agent` applies, for the same reason: without it this test skips
    # even when a local .env has the key, and reports "not set" rather than "not run".
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    from sora.adapters.anthropic_llm import AnthropicLLMClient

    llm = AnthropicLLMClient()  # model id defaults, but is a config value (ctor arg)
    mem = ProceduralMemory(FileMemoryBackend(tmp_path), llm=llm)
    tools = {
        "EmailClientApp": fake_manual("EmailClientApp", ["list_emails", "send_email"]),
        "CalendarApp": fake_manual("CalendarApp", ["list_events", "create_event"]),
        **(await _user_channel()),
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
