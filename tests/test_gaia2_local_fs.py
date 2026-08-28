"""``examples/gaia2/_local_fs.py`` — the decision of *whether* to stage ARE's file-system fallback
locally, tested without a download. The download itself is `huggingface_hub`'s, and the effect of
the resulting `DEMO_FS_PATH` is ARE's; what is ours is the ordering guard and the four ways this
declines to act, each of which fails silently in production if it regresses.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest
from examples.gaia2 import _local_fs
from examples.gaia2._local_fs import ensure_local_fallback_fs


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither the developer's shell nor a prior test may decide these."""
    for name in ("DEMO_FS_PATH", "SORA_GAIA2_LOCAL_FS", "GAIA2_FS_REVISION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(sys.modules, _local_fs._ARE_CONFIG_MODULE, raising=False)


def _install_hub(monkeypatch: pytest.MonkeyPatch, snapshot_download: Any) -> None:
    """Stand in for `huggingface_hub`, which `_local_fs` imports inside the function."""
    stub = type(sys)("huggingface_hub")
    stub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)


def _fake_hub(monkeypatch: pytest.MonkeyPatch, root: Any, calls: list[dict[str, Any]]) -> None:
    def snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(root)

    _install_hub(monkeypatch, snapshot_download)


def test_stages_snapshot_and_sets_demo_fs_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    (tmp_path / _local_fs._SUBDIR).mkdir()
    calls: list[dict[str, Any]] = []
    _fake_hub(monkeypatch, tmp_path, calls)

    path = ensure_local_fallback_fs(announce=False)

    assert path == str(tmp_path / _local_fs._SUBDIR)
    assert os.environ["DEMO_FS_PATH"] == path
    # Pinned, not `main`: a benchmark's inputs must not move under it between runs.
    assert calls[0]["revision"] == _local_fs._REVISION
    assert calls[0]["repo_type"] == "dataset"


def test_revision_is_overridable(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    (tmp_path / _local_fs._SUBDIR).mkdir()
    calls: list[dict[str, Any]] = []
    _fake_hub(monkeypatch, tmp_path, calls)
    monkeypatch.setenv("GAIA2_FS_REVISION", "deadbeef")

    ensure_local_fallback_fs(announce=False)

    assert calls[0]["revision"] == "deadbeef"


def test_opt_out_leaves_are_on_the_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SORA_GAIA2_LOCAL_FS", "0")

    def boom(**kwargs: Any) -> str:
        raise AssertionError("must not download when opted out")

    _install_hub(monkeypatch, boom)

    assert ensure_local_fallback_fs(announce=False) is None
    assert "DEMO_FS_PATH" not in os.environ


def test_preset_demo_fs_path_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit choice outranks ours — including someone pointing ARE at their own tree."""
    monkeypatch.setenv("DEMO_FS_PATH", "/somewhere/of/my/own")

    def boom(**kwargs: Any) -> str:
        raise AssertionError("must not download over a preset DEMO_FS_PATH")

    _install_hub(monkeypatch, boom)

    assert ensure_local_fallback_fs(announce=False) == "/somewhere/of/my/own"


def test_declines_once_are_is_already_imported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole point of the guard: ARE binds DEMO_FS_PATH as a default argument at import time,
    so setting it afterwards is a no-op that *looks* like a fix. It must say so, not pretend."""
    (tmp_path / _local_fs._SUBDIR).mkdir()
    _fake_hub(monkeypatch, tmp_path, [])
    monkeypatch.setitem(sys.modules, _local_fs._ARE_CONFIG_MODULE, type(sys)("are_config_stub"))

    assert ensure_local_fallback_fs(announce=False) is None
    assert "DEMO_FS_PATH" not in os.environ
    assert "imported before" in capsys.readouterr().err


def test_download_failure_degrades_to_the_hub(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing snapshot is a slow run, not a failed one."""

    def boom(**kwargs: Any) -> str:
        raise OSError("no network")

    _install_hub(monkeypatch, boom)

    assert ensure_local_fallback_fs(announce=False) is None
    assert "DEMO_FS_PATH" not in os.environ
    assert "falling back to" in capsys.readouterr().err


def test_snapshot_without_the_subdir_is_not_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """A partial/renamed snapshot must not become a DEMO_FS_PATH pointing at nothing — ARE would
    then restore an empty tree and every file read would return a placeholder."""
    _fake_hub(monkeypatch, tmp_path, [])  # tmp_path deliberately has no demo_filesystem/

    assert ensure_local_fallback_fs(announce=False) is None
    assert "DEMO_FS_PATH" not in os.environ
    assert "no demo_filesystem/ directory" in capsys.readouterr().err
