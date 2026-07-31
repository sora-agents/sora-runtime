"""``examples/are/sim/email_calendar/report.py`` — the ``sora run --report`` hook for the ARE
showcase.

Pure formatting over ``agent.working.activities`` + an opaque ``simulation.validate()`` call, so
it's tested directly rather than through a real ``sora run`` invocation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from examples.are.sim.email_calendar.report import report

from fakes import FakeAdapter, FakeTool, FakeWorkspace
from sora.action import default_action_registry
from sora.activity import Activity, ActivityState
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
from sora.types import OperationAck

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")


def _build_agent(tmp_path: Path) -> Agent:
    workspace = FakeWorkspace("clock", _ORIGIN, [FakeTool("Clock", invoke_results={})])
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
    return Agent(
        cycle=cycle,
        registry=registry,
        working=working,
        semantic=semantic,
        procedural=cycle.procedural,
        episodic=cycle.episodic,
        communication=cycle.communication,
    )


class _FakeOutcome:
    def __init__(self, *, success: bool, rationale: str | None = None) -> None:
        self.success = success
        self.rationale = rationale


class _FakeSimulation:
    def __init__(
        self, *, ok: bool = True, rationale: str | None = None, raises: Exception | None = None
    ) -> None:
        self._ok = ok
        self._rationale = rationale
        self._raises = raises

    def validate(self) -> _FakeOutcome:
        if self._raises is not None:
            raise self._raises
        return _FakeOutcome(success=self._ok, rationale=self._rationale)


def test_report_prints_completed_when_no_activity_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agent = _build_agent(tmp_path)
    agent.working.activities["a1"] = Activity(
        id="a1", goal="schedule it", context={}, state=ActivityState.TERMINATED
    )

    report(agent, None)

    out = capsys.readouterr().out
    assert "agent outcome: completed" in out
    assert "ARE validation" not in out  # no simulation given -> no ARE-specific line


def test_report_prints_failed_when_an_activity_s_last_operation_was_not_ok(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agent = _build_agent(tmp_path)
    agent.working.activities["a1"] = Activity(
        id="a1",
        goal="schedule it",
        context={},
        state=ActivityState.TERMINATED,
        last_operation=OperationAck(ok=False, result=None),
    )

    report(agent, None)

    assert "agent outcome: ❌ FAILED" in capsys.readouterr().out


def test_report_prints_are_validation_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agent = _build_agent(tmp_path)
    report(agent, _FakeSimulation(ok=True))

    out = capsys.readouterr().out
    assert "ARE validation: ✅ PASS" in out


def test_report_prints_are_validation_fail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agent = _build_agent(tmp_path)
    report(agent, _FakeSimulation(ok=False))

    out = capsys.readouterr().out
    assert "ARE validation: FAIL" in out


def test_report_prints_are_validation_rationale_when_present(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agent = _build_agent(tmp_path)
    report(agent, _FakeSimulation(ok=False, rationale="scheduled_on_tuesday=False"))

    out = capsys.readouterr().out
    assert "ARE validation: FAIL" in out
    assert "scheduled_on_tuesday=False" in out


def test_report_prints_are_validation_n_a_when_validate_raises(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agent = _build_agent(tmp_path)
    report(agent, _FakeSimulation(raises=RuntimeError("no oracle events")))

    out = capsys.readouterr().out
    assert "ARE validation: n/a (no oracle events)" in out
