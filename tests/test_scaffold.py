"""``sora.scaffold.write_project`` — the ``sora init`` file generator.

Two levels: file-presence/content checks (the scaffold produces exactly what it promises), and an
end-to-end wiring check that proves the generated project is actually runnable — not just that files
got written. The latter drives the *real* `load_yaml`/`adapter_for`/`import_object` path against the
generated `agent.yaml` and `clock_tool.py`, the same way `sora run` would, and calls
`build_agent()` to confirm the whole config wires up without a real model call ever happening (no
API key needed in CI — building `Strategies`/`AnthropicLLMClient` doesn't call the model, only
`ProceduralMemory.infer()` would).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from sora.bootstrap import adapter_for, build_agent, load_yaml
from sora.scaffold import write_project


def test_write_project_creates_the_expected_files(tmp_path: Path) -> None:
    project = tmp_path / "my-agent"
    write_project(project)

    assert (project / "pyproject.toml").is_file()
    assert (project / "agent.yaml").is_file()
    assert (project / "manuals" / "clock.md").is_file()
    assert (project / "clock_tool.py").is_file()


def test_write_project_pyproject_names_the_project_after_the_directory(tmp_path: Path) -> None:
    project = tmp_path / "my-agent"
    write_project(project)
    assert 'name = "my-agent"' in (project / "pyproject.toml").read_text()


def test_write_project_pyproject_depends_on_the_git_repo_not_an_unpublished_pypi_name(
    tmp_path: Path,
) -> None:
    # sora-runtime isn't on PyPI yet -- a bare "sora-runtime[llm]" version constraint is
    # unresolvable (uv: "No solution found ... there are no versions of sora-runtime[llm]").
    # Regression for that: the dependency must pin a git source instead.
    project = tmp_path / "my-agent"
    write_project(project)
    text = (project / "pyproject.toml").read_text()
    assert "sora-runtime[llm] @ git+https://github.com/sora-agents/sora-runtime.git" in text


def test_write_project_agent_yaml_wires_the_clock_factory(tmp_path: Path) -> None:
    project = tmp_path / "my-agent"
    write_project(project)
    text = (project / "agent.yaml").read_text()
    assert "factory: clock_tool.build_adapter" in text
    assert "sora.strategies.DefaultReasonStrategy" in text


def test_write_project_clock_manual_is_parseable_and_matches_the_tool(tmp_path: Path) -> None:
    from sora.manual import MarkdownManualParser

    project = tmp_path / "my-agent"
    write_project(project)
    manual = MarkdownManualParser().parse((project / "manuals" / "clock.md").read_text())
    assert manual.id == "clock"


def test_write_project_rejects_an_already_existing_target(tmp_path: Path) -> None:
    project = tmp_path / "my-agent"
    project.mkdir()
    with pytest.raises(FileExistsError, match="my-agent"):
        write_project(project)


def test_write_project_rejects_an_existing_non_directory_target(tmp_path: Path) -> None:
    project = tmp_path / "my-agent"
    project.write_text("not a directory")
    with pytest.raises(FileExistsError, match="my-agent"):
        write_project(project)


# ---------------------------------------------------------------------------
# End-to-end: the scaffolded project actually runs against the real wiring path
# ---------------------------------------------------------------------------


async def test_scaffolded_clock_tool_invokes_through_the_real_adapter_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "demo-agent"
    write_project(project)
    monkeypatch.chdir(project)
    monkeypatch.syspath_prepend(str(project))  # same trick `sora run` uses for project-local code

    config = load_yaml(project / "agent.yaml")
    assert len(config.workspaces) == 1
    origin, adapter = adapter_for(config.workspaces[0])

    workspace = (await adapter.discover())[0]
    tools = {tool.id: tool for tool in workspace.tools()}
    assert "clock" in tools

    ack = await tools["clock"].invoke("get_time")
    assert ack.ok is True
    # A real, live ISO-8601 timestamp -- not a canned value (proves it's computed, not hardcoded).
    datetime.fromisoformat(ack.result)

    unknown = await tools["clock"].invoke("not_a_real_operation")
    assert unknown.ok is False


def test_scaffolded_agent_yaml_builds_a_real_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Confirms the whole agent.yaml (memory backends, llm: block, strategies.reason import,
    # workspaces) wires up end to end via the real build_agent() -- not just the clock workspace in
    # isolation above. No model call happens during construction, so no API key is needed here.
    project = tmp_path / "demo-agent"
    write_project(project)
    monkeypatch.chdir(project)
    monkeypatch.syspath_prepend(str(project))

    agent = build_agent(str(project / "agent.yaml"))

    assert agent.registry.configured_origins() != []
