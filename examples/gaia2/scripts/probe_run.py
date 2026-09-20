"""Run ``run_benchmark`` with the per-event timing probe installed.

Usage::

    python -m examples.gaia2.scripts.probe_run --probe-out PATH -- <run_benchmark args>

Everything after ``--`` is handed to ``run_benchmark`` untouched, so the run under measurement is
the shipped one.  The probe is installed *after* the local file-system fallback is staged and
*before* the runner imports ARE, because ARE binds ``DEMO_FS_PATH`` as a default argument at import
time -- staging it afterwards is a silent no-op, and patching before the import has nothing to
patch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--" not in raw:
        raise SystemExit("separate probe options from run_benchmark options with --")
    split = raw.index("--")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-out", required=True, metavar="PATH")
    args = parser.parse_args(raw[:split])
    forwarded = raw[split + 1 :]

    if "" not in sys.path:
        sys.path.insert(0, "")

    from examples.gaia2._local_fs import ensure_local_fallback_fs

    ensure_local_fallback_fs()

    from examples.gaia2.scripts import timing_probe

    out = Path(args.probe_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    timing_probe.install(out)
    timing_probe.record_meta(command=forwarded)
    print(f"timing probe -> {out}", flush=True)

    from examples.gaia2 import run_benchmark

    try:
        run_benchmark.main(forwarded)
    finally:
        timing_probe.close()


if __name__ == "__main__":
    main()
