"""Tests for bootstrap's ``.env`` convenience loader.

``load_dotenv`` picks up a local ``.env`` (e.g. ``ANTHROPIC_API_KEY``) into the environment without
an explicit ``export`` — called first by ``build_agent`` so a model-backed agent finds credentials.
The one invariant that matters for correctness: **real environment variables always win** over the
file. Each test isolates ``os.environ`` (a per-test copy via monkeypatch) so the loader's writes
don't leak across tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sora.bootstrap import load_dotenv


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
