"""Re-score stored Gaia2 runs offline, under either of ARE's two judge-verdict parses.

A scored run writes the judge's raw answers beside its trace (see ``batch.py``). This reads them
back and re-applies ARE's own rule — the ``equality_checker`` fast path, else a strict conjunction
over the soft checkers — once per parse, with no model, no ARE, and no tokens:

    python -m examples.gaia2.rescore .sora/gaia2/out
    python -m examples.gaia2.rescore .sora/gaia2/out --require-divergence

Why both parses. ARE's engines lowercase ``True``/``False`` in transit, so its ``[[True]]``-family
soft checkers cannot return a verdict and an *unparsed* answer rejects the event on exactly the
falsy path a genuine rejection takes (``relax_judge_verdict_case`` has the full account). A score
therefore carries an unresolvable ambiguity unless the same events are also read under the other
parse: only the difference separates "the agent got it wrong" from "the scorer could not say yes".

``--require-divergence`` is the acceptance gate on the recording pipeline itself, and it is
deliberately strict about *where* the divergence falls. A pipeline can record faithfully and still
never exercise the patched checker path — every event settled by the equality fast path, no model
consulted — and that failure is indistinguishable from genuine agreement in any aggregate. So the
gate passes only when at least one event that ``equality_checker`` MISSED is scored differently by
the two parses. Point it at a run of a scenario that ends in a paraphrased message to the user;
those are the events that reach the soft checkers at all.

``disagreements``, printed per recording, is the re-scorer auditing itself: re-scored under the
parse the run actually used, it must reproduce the boolean ARE returned for every event. A non-zero
count is a defect in this re-implementation, not evidence about the run, and invalidates the scores
printed next to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sora.adapters.are_judge import (
    VERDICT_PARSES,
    JudgeRecording,
    compare_parses,
    recording_from_dict,
    rescore,
)


def _load(path: Path) -> JudgeRecording | None:
    """Read one recording, or None if the file is some other JSON in the artifact tree.

    An artifact directory also holds exported traces and ``output.jsonl``; walking it must not turn
    an unrelated file into an error, and must not silently read one as an empty recording either —
    hence the shape check rather than a bare ``try``.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        return None
    if payload.get("verdict_parse") is None and not payload["events"]:
        return None  # an empty file with no provenance is not a recording of anything
    return recording_from_dict(payload)


def _discover(paths: list[Path]) -> list[tuple[Path, JudgeRecording]]:
    found: list[tuple[Path, JudgeRecording]] = []
    for path in paths:
        candidates = sorted(path.rglob("*.json")) if path.is_dir() else [path]
        for candidate in candidates:
            recording = _load(candidate)
            if recording is not None:
                found.append((candidate, recording))
    return found


def summarize(paths: list[Path]) -> dict[str, Any]:
    """Re-score every recording under ``paths`` (files or directories) under both parses."""
    rows: list[dict[str, Any]] = []
    for path, recording in _discover(paths):
        comparison = compare_parses(recording)
        live = recording.verdict_parse
        # Only meaningful against the parse the run was actually scored under; a recording that
        # does not say which that was cannot be audited, and says so rather than guessing one.
        disagreements = (
            list(rescore(recording, parse=live).disagreements_with_recorded)
            if live in VERDICT_PARSES
            else None
        )
        rows.append(
            {
                "path": str(path),
                "scenario_id": recording.scenario_id,
                "run_number": recording.run_number,
                "verdict_parse": live,
                "events": len(recording.events),
                "scores": {parse: comparison.score(parse) for parse in VERDICT_PARSES},
                "divergent_events": list(comparison.divergent_events),
                "divergent_off_equality_fast_path": comparison.divergent_off_equality_fast_path,
                "disagreements_with_recorded": disagreements,
            }
        )
    return {
        "recordings": rows,
        "events": sum(row["events"] for row in rows),
        "divergent_events": sum(len(row["divergent_events"]) for row in rows),
        "divergence_off_equality_fast_path": any(
            row["divergent_off_equality_fast_path"] for row in rows
        ),
        "unreproduced_recordings": sum(1 for row in rows if row["disagreements_with_recorded"]),
    }


def _print(summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = summary["recordings"]
    if not rows:
        print("no judge recordings found")
        return
    # The parse names are spelled out rather than abbreviated: this table gets pasted into a
    # write-up without its command line, and a score means nothing without the parse that produced
    # it (see relax_judge_verdict_case).
    print(
        f"\n{'scenario':<28} {'run':>3} {'events':>6} {'stock':>8} {'case-insensitive':>17}  "
        f"diverged"
    )
    for row in rows:
        stock = row["scores"]["stock"]
        relaxed = row["scores"]["case-insensitive"]
        print(
            f"  {str(row['scenario_id']):<26} {row['run_number']!s:>3} {row['events']:>6} "
            f"{'n/a' if stock is None else f'{stock:6.1%}':>8} "
            f"{'n/a' if relaxed is None else f'{relaxed:6.1%}':>17}  "
            f"{len(row['divergent_events'])}"
        )
        if row["disagreements_with_recorded"]:
            # Loud, and next to the numbers it invalidates: this says the re-implementation below
            # does not reproduce the judge it is re-reading, so its scores mean nothing yet.
            print(
                f"      ⚠ re-score under the live parse ({row['verdict_parse']}) disagrees with "
                f"ARE on events {row['disagreements_with_recorded']} — fix before reading the "
                f"scores above"
            )
    print(
        f"\n{summary['events']} judged events, {summary['divergent_events']} scored differently "
        f"by the two parses"
    )
    print(
        "  at least one divergence off the equality fast path: "
        f"{'yes' if summary['divergence_off_equality_fast_path'] else 'no'}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rescore",
        description="Re-score stored Gaia2 judge recordings under both verdict parses.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="Recording files, or directories to walk (an artifact --output-dir works).",
    )
    parser.add_argument(
        "--require-divergence",
        action="store_true",
        help=(
            "Exit non-zero unless the two parses differ on at least one event the equality checker "
            "missed. The acceptance gate on the recording pipeline: divergence only on the fast "
            "path means the patched checker path was never exercised, which is indistinguishable "
            "from agreement."
        ),
    )
    parser.add_argument("--output", metavar="PATH", help="Also write the summary as JSON here.")
    args = parser.parse_args(argv)

    summary = summarize([Path(p) for p in args.paths])
    _print(summary)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")

    if summary["unreproduced_recordings"]:
        print(
            "\nFAIL: the offline re-score does not reproduce ARE's own verdicts under the parse "
            "the run used.",
            file=sys.stderr,
        )
        return 1
    if args.require_divergence and not summary["divergence_off_equality_fast_path"]:
        print(
            "\nFAIL: no event that the equality checker missed is scored differently by the two "
            "parses. The recording may be faithful and still never have exercised the patched "
            "checker path. Re-run on a scenario that ends in a paraphrased message to the user.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
