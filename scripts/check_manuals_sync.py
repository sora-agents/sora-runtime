#!/usr/bin/env python3
"""Pre-commit/CI check: EXAMPLES.md's inline tool manuals must match the extracted test fixtures.

EXAMPLES.md documents several tool manuals inline (fenced ```markdown blocks labelled
`manuals/<name>.md`), while tests/fixtures/manuals/ holds the same manuals as standalone files the
ManualParser test suite parses. Nothing otherwise keeps the two copies identical — edit one, forget
the other, and the spec and the parsed fixtures silently diverge. This asserts every manual block in
EXAMPLES.md has a byte-for-byte identical fixture.

Enforced one-directionally in the way that matters: the fixtures directory MAY hold extra manuals
with no EXAMPLES.md counterpart — a minimal clock.md, or deliberately-malformed manuals under
invalid/ that exist only to exercise the parser's error path — and those are ignored here. What's
enforced is that nothing documented inline in EXAMPLES.md lacks a matching fixture, and no matching
fixture has drifted from it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "EXAMPLES.md"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "manuals"

# Matches:  `manuals/<name>.md`:\n\n```markdown\n<body>```
_BLOCK = re.compile(r"`manuals/(?P<name>[\w-]+\.md)`:\n\n```markdown\n(?P<body>.*?)```", re.S)


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def manual_blocks(text: str) -> dict[str, str]:
    return {m.group("name"): m.group("body") for m in _BLOCK.finditer(text)}


def main() -> int:
    blocks = manual_blocks(EXAMPLES.read_text())
    if not blocks:
        print(
            f"No inline manual blocks found in {rel(EXAMPLES)} — expected fenced "
            "```markdown blocks labelled `manuals/<name>.md`. Did the doc format change? "
            "Update this check if so."
        )
        return 1

    problems: list[str] = []
    for name, body in sorted(blocks.items()):
        fixture = FIXTURES_DIR / name
        if not fixture.exists():
            problems.append(f"  {name}: in EXAMPLES.md but missing from {rel(FIXTURES_DIR)}/")
        elif fixture.read_text() != body:
            problems.append(f"  {name}: fixture differs from its EXAMPLES.md block (copy verbatim)")

    if not problems:
        return 0

    print("EXAMPLES.md manuals and tests/fixtures/manuals/ have drifted:\n")
    print("\n".join(problems))
    print(
        f"\nChecked {len(blocks)} manual block(s) in {rel(EXAMPLES)} against {rel(FIXTURES_DIR)}/"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
