"""``Agent.run`` / ``Agent.stop`` — the startup-join → tick-loop → teardown lifecycle.

Over a subprocess-free ``FakeAdapter``: ``run()`` joins the configured workspace once at startup
(through the predefined ``_join_`` action, so records/manuals land in SemanticMemory), drives the
decision cycle, and on ``stop()`` leaves the workspace (closing it) as the loop unwinds. No model is
needed — with no inbound message the cycle selects nothing and Reason is never reached — so this
isolates the loop mechanics from planning (covered in ``test_are_mcp_email_calendar.py``).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from fakes import FakeAdapter, FakeLLMClient, FakeTool, FakeWorkspace
from sora.action import default_action_registry
from sora.cycle import Agent, DecisionCycle
from sora.environment import EnvironmentRegistry, Workspace, WorkspaceOrigin
from sora.llm import CompletionRequest, LLMClient, MeteredLLMClient
from sora.manual import Manual, ToolRecord, WorkspaceRecord
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


def _build_agent(
    tmp_path: Path, llm: LLMClient | None = None
) -> tuple[Agent, EnvironmentRegistry, FakeWorkspace]:
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
        procedural=ProceduralMemory(FileMemoryBackend(tmp_path / "procedural"), llm=llm),
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


class _FailingAdapter:
    """A configured adapter whose join fails — proving a partial startup join is cleaned up."""

    name = "fake"

    async def discover(self) -> list[Workspace]:
        raise RuntimeError("second workspace is unavailable")

    async def connect(
        self,
        workspace_record: WorkspaceRecord,
        tool_records: list[ToolRecord],
        manuals: dict[str, Manual],
    ) -> Workspace:
        raise RuntimeError("unused")  # pragma: no cover


async def test_run_partial_startup_join_failure_still_closes_joined_workspaces(
    tmp_path: Path,
) -> None:
    # A second configured workspace whose join raises: _start() joins the first, then fails on the
    # second. run()'s finally must still leave (close) the first — otherwise its live MCP subprocess
    # would leak, since the exception escapes run() before the loop's teardown could run.
    agent, registry, workspace = _build_agent(tmp_path)
    bad_origin = WorkspaceOrigin(adapter="fake", address="fake://bad")
    registry._adapters[bad_origin] = _FailingAdapter()  # configured after the good one -> joins 2nd

    with pytest.raises(RuntimeError, match="second workspace is unavailable"):
        await agent.run()

    assert workspace.closed is True  # the already-joined workspace was closed despite the failure
    assert registry.joined_workspaces() == []


async def test_startup_trace_names_the_model(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A trace that doesn't name its model can't be read back later: the same odd trajectory is an
    expected small-model artifact or a real runtime defect depending on what produced it — and the
    log is often read weeks after the run, by someone who no longer knows which config was used.
    Logged (not printed with the CLI banner) so it reaches a --log-file capture too."""
    agent, registry, _ = _build_agent(
        tmp_path, llm=MeteredLLMClient(FakeLLMClient(), model="qwen3:30b-64k")
    )
    with caplog.at_level(logging.INFO, logger="sora.cycle"):
        task = asyncio.create_task(agent.run())
        await _run_until(lambda: bool(registry.all_tools()), task)
        await agent.stop()
        await task

    startup = [r.getMessage() for r in caplog.records if r.getMessage().startswith("startup:")]
    assert startup[0] == "startup: model qwen3:30b-64k"  # before the workspace joins it explains


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


class _ClosableLLMClient:
    """A client that offers the optional ``aclose`` courtesy, so teardown becomes observable.

    Counts rather than flags: closing a connection pool twice is its own defect (the second call
    lands on an already-released transport), so the count is what the assertions below check.
    """

    model = "fake-model"

    def __init__(self, *, fail: bool = False) -> None:
        self.closed = 0
        self._fail = fail

    async def complete(self, request: CompletionRequest) -> str:
        raise AssertionError("these lifecycle tests never reach a model call")

    async def aclose(self) -> None:
        self.closed += 1
        if self._fail:
            raise RuntimeError("the connection pool refused to close")


class _UnclosableWorkspace(FakeWorkspace):
    """A workspace whose close fails — the other half of run()'s shared teardown going wrong."""

    async def close(self) -> None:
        raise RuntimeError("workspace close failed")


async def test_run_closes_the_model_client_on_teardown(tmp_path: Path) -> None:
    """A model client holding an HTTP connection pool has to be released while the loop it was
    created on is still running. Left open, it is finalized by the garbage collector after
    ``asyncio.run`` has already closed the loop, and asyncio reports that as a bare
    ``Task exception was never retrieved ... RuntimeError('Event loop is closed')`` — a traceback
    per run, in exactly the logs a failing run is read back from, plus a leaked pool per run.

    run()'s finally owns it for the same reason it owns leaving workspaces: it is *after* the loop,
    so it cannot race a tick that is still in flight — which closing from stop() would.
    """
    client = _ClosableLLMClient()
    agent, registry, workspace = _build_agent(tmp_path, llm=client)
    task = asyncio.create_task(agent.run())

    await _run_until(lambda: bool(registry.all_tools()), task)
    await agent.stop()
    await task

    assert client.closed == 1
    assert workspace.closed is True  # the teardown it shares the finally with still happened


async def test_run_closes_the_model_client_when_the_loop_task_is_cancelled(tmp_path: Path) -> None:
    """The shutdown path every run surface actually takes: ``TerminalSession`` cancels the runner
    task rather than only setting the stop flag, so the close is reached while the task is already
    unwinding a CancelledError. If it did not survive that, the close would be dead code on the one
    path a benchmark run uses.
    """
    client = _ClosableLLMClient()
    agent, registry, workspace = _build_agent(tmp_path, llm=client)
    task = asyncio.create_task(agent.run())

    await _run_until(lambda: bool(registry.all_tools()), task)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert client.closed == 1
    assert workspace.closed is True


async def test_run_closes_the_model_client_even_when_leaving_a_workspace_fails(
    tmp_path: Path,
) -> None:
    """The two teardowns are independent, so a workspace whose close raises must not strand the
    HTTP pool — and the leave failure must still surface rather than being swallowed by the close
    that follows it."""
    client = _ClosableLLMClient()
    agent, registry, _ = _build_agent(tmp_path, llm=client)
    bad_origin = WorkspaceOrigin(adapter="fake", address="fake://unclosable")
    registry._adapters[bad_origin] = FakeAdapter(
        "fake", _UnclosableWorkspace("unclosable", bad_origin, [])
    )

    task = asyncio.create_task(agent.run())
    await _run_until(lambda: len(registry.joined_workspaces()) == 2, task)
    await agent.stop()
    with pytest.raises(RuntimeError, match="workspace close failed"):
        await task

    assert client.closed == 1


async def test_a_failing_close_does_not_mask_what_the_run_was_unwinding(tmp_path: Path) -> None:
    """Printing a teardown failure over the real one is the harm being fixed here, so a close that
    raises must not become the exception that escapes run(). The startup-join failure is what the
    caller needs to see."""
    client = _ClosableLLMClient(fail=True)
    agent, registry, workspace = _build_agent(tmp_path, llm=client)
    bad_origin = WorkspaceOrigin(adapter="fake", address="fake://bad")
    registry._adapters[bad_origin] = _FailingAdapter()

    with pytest.raises(RuntimeError, match="second workspace is unavailable"):
        await agent.run()

    assert client.closed == 1  # attempted, and its own failure absorbed
    assert workspace.closed is True


@pytest.mark.parametrize("llm", [FakeLLMClient(), None], ids=["no-aclose", "no-model-at-all"])
async def test_a_model_client_offering_no_aclose_is_left_alone(
    tmp_path: Path, llm: LLMClient | None
) -> None:
    """``LLMClient`` is one method wide, so teardown is an optional courtesy a client may offer and
    not a requirement — the same duck-typing ``ProceduralMemory.model`` already uses. A client
    without it (and an agent with no model at all) must not turn run()'s teardown into an
    AttributeError, which would strand the workspace leave sharing that finally."""
    agent, registry, workspace = _build_agent(tmp_path, llm=llm)
    task = asyncio.create_task(agent.run())

    await _run_until(lambda: bool(registry.all_tools()), task)
    await agent.stop()
    await task

    assert workspace.closed is True
