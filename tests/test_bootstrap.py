"""Tests for bootstrap's ``.env`` convenience loader and config-wiring helpers.

``load_dotenv`` picks up a local ``.env`` (e.g. ``ANTHROPIC_API_KEY``) into the environment without
an explicit ``export`` — called first by ``build_agent`` so a model-backed agent finds credentials.
The one invariant that matters for correctness: **real environment variables always win** over the
file. Each test isolates ``os.environ`` (a per-test copy via monkeypatch) so the loader's writes
don't leak across tests.

The rest of this module pins the small pure helpers ``build_agent`` is assembled from —
``import_object``, ``load_yaml``, ``backend_for``, ``adapter_for``, ``llm_for``, ``transport_for`` —
independently of the full wiring (which ``test_build_agent.py`` covers).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sora.bootstrap import (
    AgentConfig,
    adapter_for,
    backend_for,
    import_object,
    llm_for,
    load_dotenv,
    load_yaml,
    procedural_prompts_for,
    transport_for,
)
from sora.environment import WorkspaceOrigin
from sora.memory import FileMemoryBackend
from sora.strategies import DefaultReasonStrategy
from sora.transport import InProcessTransport


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    # Swap os.environ for a copy so load_dotenv's setdefault writes are undone after the test.
    env = dict(os.environ)
    monkeypatch.setattr(os, "environ", env)
    return env


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / ".env"
    p.write_text(content, encoding="utf-8")
    return p


def test_load_dotenv_sets_absent_keys(tmp_path: Path, isolated_env: dict[str, str]) -> None:
    isolated_env.pop("SORA_TEST_KEY", None)
    load_dotenv(_write(tmp_path, "SORA_TEST_KEY=abc123\n"))
    assert os.environ["SORA_TEST_KEY"] == "abc123"


def test_load_dotenv_does_not_override_real_env(
    tmp_path: Path, isolated_env: dict[str, str]
) -> None:
    isolated_env["SORA_TEST_KEY"] = "from-real-env"
    load_dotenv(_write(tmp_path, "SORA_TEST_KEY=from-dotenv\n"))
    assert os.environ["SORA_TEST_KEY"] == "from-real-env"  # the process environment always wins


def test_load_dotenv_missing_file_is_noop(tmp_path: Path, isolated_env: dict[str, str]) -> None:
    load_dotenv(tmp_path / "does-not-exist.env")  # must not raise


def test_load_dotenv_ignores_comments_blanks_and_normalizes(
    tmp_path: Path, isolated_env: dict[str, str]
) -> None:
    for key in ("SORA_A", "SORA_B", "SORA_C", "SORA_D"):
        isolated_env.pop(key, None)
    content = (
        "# a comment\n"
        "\n"
        '  SORA_A = "double-quoted"  \n'  # surrounding whitespace + double quotes stripped
        "export SORA_B=exported\n"  # optional `export` prefix tolerated
        "SORA_C='single-quoted'\n"
        "SORA_D=has=equals=signs\n"  # only the first `=` splits key/value
    )
    load_dotenv(_write(tmp_path, content))
    assert os.environ["SORA_A"] == "double-quoted"
    assert os.environ["SORA_B"] == "exported"
    assert os.environ["SORA_C"] == "single-quoted"
    assert os.environ["SORA_D"] == "has=equals=signs"


# --------------------------------------------------------------------------------------------------
# import_object — both dotted-path forms
# --------------------------------------------------------------------------------------------------


def test_import_object_resolves_dotted_and_colon_forms() -> None:
    assert import_object("sora.strategies.DefaultReasonStrategy") is DefaultReasonStrategy
    assert import_object("sora.strategies:DefaultReasonStrategy") is DefaultReasonStrategy


def test_import_object_rejects_bare_name() -> None:
    with pytest.raises(ValueError, match="dotted import path"):
        import_object("DefaultReasonStrategy")


def test_import_object_missing_attribute_raises() -> None:
    with pytest.raises(AttributeError):
        import_object("sora.strategies.NoSuchStrategy")


# --------------------------------------------------------------------------------------------------
# load_yaml — parse the `agent:` block, require `reason`
# --------------------------------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "agent.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_yaml_parses_agent_block(tmp_path: Path) -> None:
    config = load_yaml(
        _write_yaml(
            tmp_path,
            "agent:\n"
            "  name: demo\n"
            "  strategies:\n"
            "    reason: sora.strategies.DefaultReasonStrategy\n"
            "  memory:\n"
            "    semantic: file://./s\n"
            "  workspaces: []\n"
            "  llm:\n"
            "    model: claude-opus-4-8\n",
        )
    )
    assert config.name == "demo"
    assert config.strategies["reason"] == "sora.strategies.DefaultReasonStrategy"
    assert config.memory["semantic"] == "file://./s"
    assert config.workspaces == []
    assert config.llm == {"model": "claude-opus-4-8"}


def test_load_yaml_parses_procedural_block(tmp_path: Path) -> None:
    config = load_yaml(
        _write_yaml(
            tmp_path,
            "agent:\n"
            "  name: demo\n"
            "  strategies:\n"
            "    reason: sora.strategies.DefaultReasonStrategy\n"
            "  memory:\n"
            "    semantic: file://./s\n"
            "  workspaces: []\n"
            "  procedural:\n"
            "    plan_prompt: test_bootstrap._fake_plan_prompt\n"
            "    ground_prompt: test_bootstrap._fake_ground_prompt\n",
        )
    )
    assert config.procedural == {
        "plan_prompt": "test_bootstrap._fake_plan_prompt",
        "ground_prompt": "test_bootstrap._fake_ground_prompt",
    }


def test_load_yaml_absent_procedural_block_is_none(tmp_path: Path) -> None:
    config = load_yaml(
        _write_yaml(
            tmp_path,
            "agent:\n"
            "  name: demo\n"
            "  strategies:\n"
            "    reason: sora.strategies.DefaultReasonStrategy\n"
            "  memory: {}\n"
            "  workspaces: []\n",
        )
    )
    assert config.procedural is None


def test_load_yaml_requires_reason_strategy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strategies.reason is required"):
        load_yaml(_write_yaml(tmp_path, "agent:\n  name: demo\n  strategies: {}\n"))


# --------------------------------------------------------------------------------------------------
# backend_for
# --------------------------------------------------------------------------------------------------


def test_backend_for_file_uri_and_bare_path() -> None:
    assert isinstance(backend_for("file:///tmp/x"), FileMemoryBackend)
    assert isinstance(backend_for("/tmp/x"), FileMemoryBackend)


def test_backend_for_empty_path_raises() -> None:
    with pytest.raises(ValueError, match="empty memory backend path"):
        backend_for("file://")


# --------------------------------------------------------------------------------------------------
# adapter_for — dispatch on origin.adapter, factory escape hatch
# --------------------------------------------------------------------------------------------------


def _fake_adapter_factory(origin: WorkspaceOrigin) -> object:
    from fakes import FakeAdapter, FakeWorkspace

    return FakeAdapter("fake", FakeWorkspace("ws", origin, []))


def test_adapter_for_resolves_factory() -> None:
    origin, adapter = adapter_for(
        {
            "origin": {"adapter": "fake", "address": "fake://ws"},
            "factory": "test_bootstrap._fake_adapter_factory",
        }
    )
    assert origin == WorkspaceOrigin(adapter="fake", address="fake://ws")
    assert adapter.name == "fake"


def test_adapter_for_unknown_adapter_without_factory_raises() -> None:
    with pytest.raises(ValueError, match="no adapter for"):
        adapter_for({"origin": {"adapter": "carrier-pigeon", "address": "x"}})


def test_adapter_for_wires_manual_source_from_manuals_dir(tmp_path: Path) -> None:
    from sora.manual import DirectoryManualSource

    _origin_, adapter = adapter_for(
        {
            "origin": {"adapter": "mcp", "address": "stdio:local"},
            "workspace_id": "local",
            "command": "python",
            "args": ["-m", "srv"],
            "manuals": str(tmp_path),
        }
    )
    assert isinstance(adapter._manual_source, DirectoryManualSource)  # type: ignore[attr-defined]


def test_adapter_for_without_manuals_key_has_no_manual_source() -> None:
    _origin_, adapter = adapter_for(
        {
            "origin": {"adapter": "mcp", "address": "stdio:local"},
            "workspace_id": "local",
            "command": "python",
            "args": ["-m", "srv"],
        }
    )
    assert adapter._manual_source is None  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------------------
# llm_for / transport_for
# --------------------------------------------------------------------------------------------------


def _config(
    *,
    llm: dict[str, object] | None = None,
    transport: dict[str, object] | None = None,
    procedural: dict[str, str] | None = None,
) -> AgentConfig:
    return AgentConfig(
        name="demo",
        strategies={"reason": "sora.strategies.DefaultReasonStrategy"},
        memory={},
        workspaces=[],
        transport=transport,
        llm=llm,
        procedural=procedural,
    )


def test_llm_for_absent_block_is_none() -> None:
    assert llm_for(_config(llm=None)) is None


def test_llm_for_builds_named_client_with_kwargs() -> None:
    from fakes import FakeLLMClient
    from sora.llm import MeteredLLMClient

    client = llm_for(_config(llm={"client": "fakes.FakeLLMClient", "response": "hi"}))
    # Every built client is wrapped for per-call timing/logging; the named client is underneath.
    assert isinstance(client, MeteredLLMClient)
    assert isinstance(client._inner, FakeLLMClient)


def test_transport_for_defaults_to_in_process() -> None:
    assert isinstance(transport_for(_config()), InProcessTransport)


def test_transport_for_rejects_peers() -> None:
    with pytest.raises(NotImplementedError, match="agent-to-agent"):
        transport_for(_config(transport={"peers": {"other": "http://x"}}))


# --------------------------------------------------------------------------------------------------
# ARE in-process injection seam — the opaque `simulation` shared by an are-sim workspace + are
# transport. Config stays generic (no scenario key); the scenario is a runtime (CLI) input. These
# assert the *wiring* only (a stub stands in for AreSimulation), so no ARE dependency is needed.
# --------------------------------------------------------------------------------------------------


def _are_entry() -> dict[str, object]:
    return {"origin": {"adapter": "are-sim", "address": "insim:are"}, "workspace_id": "are"}


def test_adapter_for_are_sim_injects_the_shared_simulation() -> None:
    from sora.adapters.are_sim import AreInProcessWorkspaceAdapter

    sim = object()
    _origin_, adapter = adapter_for(_are_entry(), sim)
    assert isinstance(adapter, AreInProcessWorkspaceAdapter)
    assert adapter._sim is sim


def test_adapter_for_are_sim_without_simulation_raises() -> None:
    with pytest.raises(ValueError, match="simulation"):
        adapter_for(_are_entry())


def test_transport_for_are_kind_injects_the_same_simulation() -> None:
    from sora.adapters.are_sim import AreTransport

    sim = object()
    transport = transport_for(_config(transport={"kind": "are"}), sim)
    assert isinstance(transport, AreTransport)
    assert transport._sim is sim


def test_transport_for_are_kind_without_simulation_raises() -> None:
    with pytest.raises(ValueError, match="simulation"):
        transport_for(_config(transport={"kind": "are"}))


def test_shipped_adapter_names_match_bootstrap_dispatch_kinds() -> None:
    # Each shipped adapter declares its kind as `.name`; bootstrap dispatches on the same literal
    # but can't read `.name` (lazy imports), so the two are synced only by convention. This guard
    # makes a rename on either side fail loudly instead of silently missing dispatch.
    from sora.adapters.are_mcp import AreMcpWorkspaceAdapter
    from sora.adapters.are_sim import AreInProcessWorkspaceAdapter
    from sora.adapters.mcp import McpWorkspaceAdapter
    from sora.bootstrap import _ADAPTER_ARE_MCP, _ADAPTER_ARE_SIM, _ADAPTER_MCP

    assert McpWorkspaceAdapter.name == _ADAPTER_MCP
    assert AreMcpWorkspaceAdapter.name == _ADAPTER_ARE_MCP
    assert AreInProcessWorkspaceAdapter.name == _ADAPTER_ARE_SIM


# --------------------------------------------------------------------------------------------------
# procedural_prompts_for — optional PlanPrompt/GroundPrompt overrides
# --------------------------------------------------------------------------------------------------


def _fake_plan_prompt(activity: object, tools: object) -> tuple[str, str]:
    return "fake plan system", "fake plan user"


def _fake_ground_prompt(
    activity: object, operation_name: str, manual: object, partial_params: object
) -> tuple[str, str]:
    return "fake ground system", "fake ground user"


def test_procedural_prompts_for_absent_block_is_empty() -> None:
    assert procedural_prompts_for(_config(procedural=None)) == {}


def test_procedural_prompts_for_resolves_plan_prompt_only() -> None:
    kwargs = procedural_prompts_for(
        _config(procedural={"plan_prompt": "test_bootstrap._fake_plan_prompt"})
    )
    assert kwargs == {"prompt": _fake_plan_prompt}


def test_procedural_prompts_for_resolves_both_prompts() -> None:
    kwargs = procedural_prompts_for(
        _config(
            procedural={
                "plan_prompt": "test_bootstrap._fake_plan_prompt",
                "ground_prompt": "test_bootstrap._fake_ground_prompt",
            }
        )
    )
    assert kwargs == {"prompt": _fake_plan_prompt, "ground_prompt": _fake_ground_prompt}
