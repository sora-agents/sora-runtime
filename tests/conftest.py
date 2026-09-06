"""Test-suite-wide setup.

Loads the repo-root ``.env`` before collection, using the runtime's own dependency-free loader, so
the skip gates on the opt-in tests read the same local configuration the runtime does. Without this
the suite is the one entry point that ignores ``.env``: ``build_agent`` loads it, ``.env.example``
documents it as "loaded automatically when present", and a developer who has filled it in
reasonably expects a gate keyed on ``SORA_GAIA2_CLI_DIR`` to open rather than skip silently.

``load_dotenv`` uses ``os.environ.setdefault``, so a real environment variable still wins — CI,
which sets nothing and has no ``.env``, is unaffected, and an explicit ``FOO=... uv run pytest``
still overrides the file.

This cannot silently switch on network-backed tests: every gate that would reach a real provider is
*also* marked ``integration`` and excluded by the default ``addopts``, so a populated ``.env`` opens
those only for someone who has already opted in with ``-m integration``.
"""

from __future__ import annotations

from pathlib import Path

from sora.bootstrap import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
