#!/usr/bin/env python3
"""Pre-commit/CI check: every class defined in src/sora/*.py must be named in README.md.

README.md's API Sketch is a hand-maintained mirror of the real code (see CLAUDE.md's "source of
truth" note) — nothing enforces that they stay in sync. This catches the specific drift an IDE
rename produces: code changes cleanly, prose doesn't follow, and nobody notices until someone reads
both side by side.

Deliberately one-directional: README.md is allowed to describe things not yet in code (that's what
"README-driven design" means, see ROADMAP.md) — it just can't fall behind what already exists.
Checks class names only (not fields or methods), which keeps false positives low: dataclass field
names like `id` or `name` are common English words that would otherwise trigger noisy, meaningless
failures against prose.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src" / "sora"
README = REPO_ROOT / "README.md"


def class_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def main() -> int:
    readme_text = README.read_text()
    missing: dict[str, list[str]] = {}

    for py_file in sorted(SRC_DIR.glob("*.py")):
        absent = [name for name in sorted(class_names(py_file)) if name not in readme_text]
        if absent:
            missing[py_file.name] = absent

    if not missing:
        return 0

    print("README.md is missing classes that exist in code (update the doc, or undo the rename):\n")
    for filename, names in missing.items():
        print(f"  {filename}: {', '.join(names)}")
    print(f"\nChecked class names against {README.relative_to(REPO_ROOT)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
