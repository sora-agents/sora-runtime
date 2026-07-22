"""``build_agent`` — the centralized wiring (ADR-0013 shared instances).

Drives the real ``build_agent`` from a written ``agent.yaml`` over a ``fake`` adapter (via the
``factory:`` seam) and ``fakes.FakeLLMClient``, so no network, no model, no subprocess. The
invariants that matter: every layer shares the *one* ``EnvironmentRegistry`` (mutable on the cycle,
read-only on working memory), the four defaulted phase strategies resolve to the ``Default*``
classes while the required ``reason`` resolves to the configured one, and the ``llm:`` block is
wired into procedural memory — or absent, leaving it store/retrieve-only.
"""

from __future__ import annotations

from pathlib import Path

from fakes import FakeAdapter, FakeTool, FakeWorkspace
from sora.bootstrap import build_agent
from sora.environment import WorkspaceOrigin
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReasonStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
)
from sora.transport import InProcessTransport


def make_gaia2_adapter(origin: WorkspaceOrigin) -> FakeAdapter:
    """Factory named by the test agent.yaml's ``factory:`` key — resolved through import_object."""
    workspace = FakeWorkspace(
        "gaia2",
        origin,
        [
            FakeTool("EmailClientApp", invoke_results={"list_emails": {"emails": []}}),
            FakeTool("CalendarApp", invoke_results={"get_calendar_events_from_to": {"events": []}}),
        ],
    )
    return FakeAdapter("fake", workspace)


def _write_config(tmp_path: Path, *, with_llm: bool) -> Path:
    llm_block = "  llm:\n    client: fakes.FakeLLMClient\n" if with_llm else ""
    text = (
        "agent:\n"
        "  name: gaia2-test\n"
        "  strategies:\n"
        "    reason: sora.strategies.DefaultReasonStrategy\n"
        "  memory:\n"
        f"    semantic: file://{tmp_path}/semantic\n"
        f"    procedural: file://{tmp_path}/procedural\n"
        f"    episodic: file://{tmp_path}/episodic\n"
        f"{llm_block}"
        "  workspaces:\n"
        '    - origin: {adapter: fake, address: "fake://ws"}\n'
        "      factory: test_build_agent.make_gaia2_adapter\n"
    )
    path = tmp_path / "agent.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_build_agent_shares_one_registry_across_layers(tmp_path: Path) -> None:
    agent = build_agent(str(_write_config(tmp_path, with_llm=True)))
    # ADR-0013: one EnvironmentRegistry, held mutable on the cycle and read-only on working memory.
    assert agent.registry is agent.cycle.registry
    assert agent.working.registry is agent.registry
    assert agent.cycle.working is agent.working
    # Memory modules and transport are the same instances on both Agent and DecisionCycle.
    assert agent.cycle.semantic is agent.semantic
    assert agent.cycle.procedural is agent.procedural
    assert agent.cycle.episodic is agent.episodic
    assert agent.cycle.communication is agent.communication


def test_build_agent_resolves_default_and_required_strategies(tmp_path: Path) -> None:
    agent = build_agent(str(_write_config(tmp_path, with_llm=True)))
    strategies = agent.cycle.strategies
    assert isinstance(strategies.observe, DefaultObserveStrategy)
    assert isinstance(strategies.reflect, DefaultReflectStrategy)
    assert isinstance(strategies.situate, DefaultSituateStrategy)
    assert isinstance(strategies.act, DefaultActStrategy)
    assert isinstance(strategies.reason, DefaultReasonStrategy)  # the configured (required) one


def test_build_agent_wires_llm_into_procedural_memory(tmp_path: Path) -> None:
    from fakes import FakeLLMClient

    agent = build_agent(str(_write_config(tmp_path, with_llm=True)))
    assert isinstance(agent.procedural._llm, FakeLLMClient)  # the E2 model seam


def test_build_agent_without_llm_leaves_procedural_model_less(tmp_path: Path) -> None:
    agent = build_agent(str(_write_config(tmp_path, with_llm=False)))
    assert agent.procedural._llm is None  # store/retrieve-only, no model


def test_build_agent_uses_in_process_transport_by_default(tmp_path: Path) -> None:
    agent = build_agent(str(_write_config(tmp_path, with_llm=False)))
    assert isinstance(agent.communication, InProcessTransport)


def _fake_plan_prompt(activity: object, tools: object) -> tuple[str, str]:
    return "fake plan system", "fake plan user"


def test_build_agent_wires_custom_plan_prompt_into_procedural_memory(tmp_path: Path) -> None:
    path = _write_config(tmp_path, with_llm=False)
    path.write_text(
        path.read_text(encoding="utf-8")
        + "  procedural:\n"
        + "    plan_prompt: test_build_agent._fake_plan_prompt\n",
        encoding="utf-8",
    )
    agent = build_agent(str(path))
    assert agent.procedural._prompt is _fake_plan_prompt


def test_build_agent_without_procedural_block_uses_default_prompt(tmp_path: Path) -> None:
    from sora.memory import default_plan_prompt

    agent = build_agent(str(_write_config(tmp_path, with_llm=False)))
    assert agent.procedural._prompt is default_plan_prompt
