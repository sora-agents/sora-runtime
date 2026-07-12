"""Smoke test validating the toolchain (uv/ruff/mypy/pytest) is wired up correctly.

Real tests start once the API sketch has a package to test.
"""

import sora


def test_package_importable() -> None:
    assert sora.__version__ == "0.1.0"
