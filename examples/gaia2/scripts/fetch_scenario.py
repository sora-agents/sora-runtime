"""Fetch a single Gaia2 scenario JSON from HuggingFace — a dev/inspection convenience.

Not part of the benchmark run path: ``batch.py`` / ``run_benchmark.py`` pull scenarios directly via
ARE's ``setup_scenarios_iterator``. This standalone script is for grabbing *one* scenario per
capability to eyeball locally, or to materialize a fixture for a skip-gated test — the
licensing-safe alternative to committing dataset JSON into the repo (the Gaia2 data is Meta's, gated
by the HuggingFace dataset terms; fetch on demand, don't redistribute — hence ``*.scenario.json`` is
gitignored).

Needs the ``are`` dependency group (``uv sync --all-extras --group are``) and, for a gated/private
dataset, an ``HF_TOKEN``.

    # list the scenario ids for one capability + split
    python -m examples.gaia2.scripts.fetch_scenario --config execution --list

    # fetch the first execution/validation scenario to ./execution-validation-0.scenario.json
    python -m examples.gaia2.scripts.fetch_scenario --config execution

    # a specific one, to a chosen path
    python -m examples.gaia2.scripts.fetch_scenario --config ambiguity --index 2 --out amb.json

    # one per core capability (shell loop)
    for c in execution search adaptability time ambiguity; do
        python -m examples.gaia2.scripts.fetch_scenario --config "$c"
    done
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DEFAULT_DATASET = "meta-agents-research-environments/gaia2"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fetch_scenario",
        description="Fetch one Gaia2 scenario JSON from HuggingFace (dev/inspection convenience).",
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="CAPABILITY",
        help="Dataset config / capability, e.g. execution, search, adaptability, time, ambiguity.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        metavar="SPLIT",
        help="Dataset split (default: validation; the test split is private).",
    )
    parser.add_argument(
        "--dataset",
        default=_DEFAULT_DATASET,
        metavar="REPO",
        help=f"HuggingFace dataset repo (default: {_DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        metavar="N",
        help="Which scenario in the listing to fetch (default: 0, the first).",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="Where to write the JSON (default: <config>-<split>-<index>.scenario.json).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the scenario ids for --config/--split and exit (no fetch).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Lazy: ARE (with its HuggingFace loaders) is an optional dependency group, so --help works
    # without it installed. Mirrors run_benchmark.py.
    from are.simulation.scenarios.utils.load_utils import (
        get_scenario_from_huggingface,
        list_huggingface_scenarios,
    )

    ids = list_huggingface_scenarios(args.dataset, args.config, args.split)
    if not ids:
        print(
            f"no scenarios for {args.dataset} config={args.config!r} split={args.split!r} "
            "(check the capability/split names, and HF_TOKEN for a gated dataset)",
            file=sys.stderr,
        )
        return 1

    if args.list:
        for sid in ids:
            print(sid)
        print(f"\n{len(ids)} scenario(s).", file=sys.stderr)
        return 0

    if not 0 <= args.index < len(ids):
        print(
            f"--index {args.index} out of range: {len(ids)} scenario(s) (0..{len(ids) - 1})",
            file=sys.stderr,
        )
        return 1

    sid = ids[args.index]
    data = get_scenario_from_huggingface(args.dataset, args.config, args.split, sid)
    if data is None:
        print(f"failed to fetch scenario {sid!r}", file=sys.stderr)
        return 1

    out = args.out or f"{args.config}-{args.split}-{args.index}.scenario.json"
    Path(out).write_text(data, encoding="utf-8")
    print(f"wrote {out}  (scenario {sid!r}, {len(data)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
