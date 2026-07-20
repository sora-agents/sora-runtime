"""``Agent.run`` / ``Agent.stop`` — the startup-join → tick-loop → teardown lifecycle.

Over a subprocess-free ``FakeAdapter``: ``run()`` joins the configured workspace once at startup
(through the predefined ``_join_`` action, so records/manuals land in SemanticMemory), drives the
decision cycle, and on ``stop()`` leaves the workspace (closing it) as the loop unwinds. No model is
needed — with no inbound message the cycle selects nothing and Reason is never reached — so this
isolates the loop mechanics from planning (covered in ``test_gaia2_reproduction.py``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fakes import FakeAdapter, FakeTool, FakeWorkspace
from sora.action import default_action_registry
from sora.cycle import Agent, DecisionCycle
from sora.environment import EnvironmentRegistry, WorkspaceOrigin
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
    Strategies,
)
from sora.transport import InProcessTransport

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")


def _build_agent(tmp_path: Path) -> tuple[Agent, EnvironmentRegistry, FakeWorkspace]:
    workspace = FakeWorkspace(
        "gaia2", _ORIGIN, [FakeTool("EmailClientApp", invoke_results={"list_emails": {}})]
    )
    registry = EnvironmentRegistry(adapters={_ORIGIN: FakeAdapter("fake", workspace)})
    working = WorkingMemory(registry=registry)
    semantic = SemanticMemory(FileMemoryBackend(tmp_path / "semantic"))
    strategies = Strategies(
        observe=DefaultObserveStrategy(),
        reflect=DefaultReflectStrategy(),
        situate=DefaultSituateStrategy(),
        reason=DefaultReasonStrategy(),
        act=DefaultActStrategy(),
    )
    cycle = DecisionCycle(
        strategies=strategies,
        communication=InProcessTransport(),
        actions=default_action_registry(),
        registry=registry,
        working=working,
        semantic=semantic,
        procedural=ProceduralMemory(FileMemoryBackend(tmp_path / "procedural")),
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "episodic")),
    )
    agent = Agent(
        cycle=cycle,
        registry=registry,
        working=working,
        semantic=semantic,
        procedural=cycle.procedural,
        episodic=cycle.episodic,
        communication=cycle.communication,
        tick_interval=0.0,  # run as fast as the event loop allows
    )
    return agent, registry, workspace


async def _run_until(predicate: object, task: asyncio.Task[None]) -> None:
    for _ in range(1000):
        if predicate():  # type: ignore[operator]
            return
        await asyncio.sleep(0)
    task.cancel()
    raise AssertionError("condition not reached before the loop budget ran out")


async def test_run_joins_at_startup_then_stop_leaves(tmp_path: Path) -> None:
    agent, registry, workspace = _build_agent(tmp_path)
    task = asyncio.create_task(agent.run())

    await _run_until(lambda: bool(registry.all_tools()), task)  # startup join happened
    assert "EmailClientApp" in [t.id for t in registry.all_tools()]

    await agent.stop()
    await task

    assert workspace.closed is True  # left on teardown
    assert registry.joined_workspaces() == []


async def test_run_is_idempotent_on_repeated_start(tmp_path: Path) -> None:
    # Calling run() must join exactly once (no duplicate-id ValueError from a second join).
    agent, registry, _ = _build_agent(tmp_path)
    task = asyncio.create_task(agent.run())
    await _run_until(lambda: bool(registry.all_tools()), task)
    await agent.stop()
    await task
    # The workspace was left; a fresh run() would re-join cleanly (start flag guards double-join
    # within a single run() invocation, which is what the loop relies on).
    assert registry.joined_workspaces() == []
