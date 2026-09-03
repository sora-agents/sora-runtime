"""Gaia2 batch harness — run a whole capability, emit leaderboard-grade artifacts, report pass@1.

One invocation runs S-ORA over every scenario of *one* capability (dataset config) and writes, under
``{output_dir}/standard/{capability}/``:

  * one HF-format trace file per (scenario, run), via ARE's ``JsonScenarioExporter`` — the exact
    artifact ``gaia2_upload_script.py`` consumes, so a run doubles as a leaderboard submission; and
  * ``output.jsonl`` — one line per (scenario, run) in ARE's own
    ``_export_benchmark_result_jsonl`` shape (``task_id``/``trace_id``/``score``/``metadata``).

Run each of the five core capabilities once to populate ``{output_dir}/standard/*``, then
``--report-only {output_dir}`` prints a per-capability pass@1 table plus the equal-weight overall
(Gaia2's headline metric). The formatting/aggregation helpers are pure and ARE-free (unit-tested
without the ``are`` extra); everything that touches ARE or spends model tokens is lazy and lives in
``main``/``_run_capability``.

    uv sync --all-extras --group are
    export ANTHROPIC_API_KEY=sk-ant-...   HF_TOKEN=hf_...     # HF gated dataset + judge model
    # smoke test — three scenarios, no leaderboard intent:
    python -m examples.gaia2.batch --capability ambiguity --split validation --limit 3 \
        --judge-model claude-sonnet-5 --judge-provider anthropic --output-dir .sora/gaia2/out
    # aggregate whatever configs have been run:
    python -m examples.gaia2.batch --report-only .sora/gaia2/out

Submission (leaderboard-grade). The artifacts this writes are exactly what ARE's standalone
``gaia2_upload_script`` consumes — no container, no re-export. To submit a capability:

    # 1. run each core capability with Gaia2's NUM_RUNS=3, on the validation split (test is
    #    private); --output-dir is absolutized so the traces stay findable from any cwd:
    python -m examples.gaia2.batch --capability execution --split validation --num-runs 3 \
        --judge-model claude-sonnet-5 --judge-provider anthropic \
        --model S-ORA/claude-... --output-dir .sora/gaia2/out
    #    ... repeat for search, adaptability, time, ambiguity (same --output-dir).
    # 2. hand the whole tree to ARE's uploader (its own --model label names the submission):
    uv run python -m are.simulation.benchmark.gaia2_upload_script \
        --input_dir .sora/gaia2/out --output_dir .sora/gaia2/stats \
        --model S-ORA/claude-... --split validation --hf_upload <org>/<dataset>

The uploader walks ``{input_dir}/standard/{config}/output.jsonl`` (our exact layout), keys each row
by ``metadata.{scenario_id, run_number}``, maps ``metadata.status`` → pass/fail, and reads the file
at each row's ``trace_id`` for the trace payload — so all three must be present and the ``trace_id``
path must resolve (hence the absolute ``--output-dir``). ``tests/test_gaia2_upload_compat.py`` locks
this round-trip against the installed uploader.

Per-scenario isolation is per fresh ``AreSimulation``; app/global-state bleed across scenarios in
one process is a known risk (a subprocess-per-scenario runner is the fallback if it bites) — fine
for the ``--limit`` smoke runs this is scoped to. The full 160/800 sweep is intentionally held until
the plan iteration primitive lands, since multi-item tasks currently under-count tool calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

_DEFAULT_CONFIG = "examples/gaia2/agent.yaml"
_DEFAULT_HF_DATASET = "meta-agents-research-environments/gaia2"

# The five capabilities Gaia2's headline (equal-weight) score averages over. A run may target any
# dataset config (incl. `mini`); the report weights only these five when they're present.
_CORE_CAPABILITIES = ("execution", "search", "adaptability", "time", "ambiguity")


# -- pure helpers (no ARE import; unit-tested without the `are` extra) -----------------------------


def _score_status(
    success: bool | None, exception: BaseException | None
) -> tuple[float | None, str]:
    """Mirror ARE's ``get_scenario_result_info``: (1.0,"success") / (0.0,"failed") /
    (None,"exception") / (None,"no_validation"). A score of None (unscored or errored) is excluded
    from pass@1 rather than counted as a miss."""
    if success is True:
        return 1.0, "success"
    if success is False:
        return 0.0, "failed"
    if exception is not None:
        return None, "exception"
    return None, "no_validation"


def _resolve_model_label(label: str | None, config_path: str) -> str | None:
    """The ``model_id`` stamped into every exported trace — and read straight off the leaderboard —
    names which model produced the run, so it must not be able to disagree with the model the run
    actually used. ``--model`` is only a *label* (a submission is org-prefixed, e.g.
    ``S-ORA/claude-opus-4-8``): nothing threads a model id into ``build_agent``, so agent.yaml's
    ``llm.model`` is the sole thing that selects one. Hence: omit the label and the configured model
    id becomes it; pass one and it has to name the configured model, or the run is refused up front
    rather than mislabeled after the tokens are spent."""
    from sora.bootstrap import load_yaml

    configured = (load_yaml(config_path).llm or {}).get("model")
    if configured is None:
        return label  # no `llm:` block, so nothing for the label to contradict
    configured = str(configured)
    if label is None:
        return configured
    if configured not in label:
        raise SystemExit(
            f"--model {label!r} does not name the model {config_path} actually uses "
            f"({configured!r}). The label is only recorded in the trace — it cannot override the "
            f"config. Fix the label, or change llm.model in the config."
        )
    return label


def _verdict_parse(args: argparse.Namespace) -> str | None:
    """How this scored record's judge verdicts were parsed, or None when nothing scored it."""
    if not args.judge_model:
        return None
    return "stock" if args.strict_verdict_case else "case-insensitive"


def _jsonl_record(
    *,
    scenario_id: str,
    run_number: int,
    success: bool | None,
    rationale: str | None,
    exception: BaseException | None,
    trace_id: str | None,
    awaiting_input: list[str] | None = None,
    write_counts: Any = None,
    timeline_expired: bool = False,
    verdict_parse: str | None = None,
) -> dict[str, Any]:
    """One ``output.jsonl`` line, matching ARE's ``_export_benchmark_result_jsonl`` exactly:
    ``task_id``/``trace_id``/``score`` at top level, and a ``metadata`` dict with all-None values
    stripped (but a False ``has_exception`` kept)."""
    score, status = _score_status(success, exception)
    metadata: dict[str, Any] = {
        "scenario_id": scenario_id,
        "run_number": run_number,
        "status": status,
        "has_exception": exception is not None,
        "exception_type": type(exception).__name__ if exception is not None else None,
        "exception_message": str(exception) if exception is not None else None,
        "rationale": rationale,
        # Why the run stopped short, when it stopped on a question (a replan or sub-goal breaker
        # tripping) rather than on the timeline. A distinct failure mode from scoring badly, and
        # invisible otherwise. None when there was none, so the strip below keeps every ordinary
        # run's record byte-identical to ARE's own shape.
        "awaiting_input": awaiting_input or None,
        # ARE's clock, not the agent, ended this run: the scenario's duration is a real-time budget
        # (its event loop is wall-clock paced), so past it no later turn is delivered. Recorded only
        # when True, because it invalidates this record's own score and mismatch fields rather than
        # qualifying them — an aggregate that averages these in is measuring the host, not the
        # agent. None otherwise, so an ordinary record stays byte-identical to ARE's own shape.
        "timeline_expired": timeline_expired or None,
        # Scoring provenance, recorded on every scored record — including the default. Unlike the
        # diagnostics around it this is not "extra information about an ordinary run": the default
        # relaxes ARE's verdict parse, so a sweep's scores are obtained under a patched judge, and
        # a record that does not say so cannot be compared with one produced by stock ARE. Same
        # reasoning as run_benchmark printing it: which of the two this is must survive the record
        # being read without the log beside it. None on an unscored run — there were no verdicts.
        "verdict_parse": verdict_parse,
        # ARE's tool-call-count gate, recomputed offline (no judge model). Recorded only when it
        # FAILS: a failure is conclusive — the judge applies this gate before any per-event
        # matching — so it explains a zero that the rationale otherwise attributes to the
        # trajectory. None when it passed or could not be computed, so the strip below keeps an
        # ordinary record byte-identical to ARE's own shape.
        # `user_replies` appears only when that is a failing dimension, but it has to appear then:
        # replies to the user are counted apart from the domain tools (the judge tolerates a few
        # extra), so a turn that made exactly the right tool calls and one reply too many has an
        # empty `surplus` AND an empty `missing` — a recorded mismatch with nothing in it to say
        # what mismatched, which is indistinguishable from a bug in this check.
        "write_count_mismatch": None
        if write_counts is None or write_counts.passed
        else [
            {
                "turn": t.turn,
                "surplus": t.surplus,
                "missing": t.missing,
                **(
                    {}
                    if t.replies_within_band
                    else {
                        "user_replies": {
                            "agent": t.agent_user_replies,
                            "oracle": t.oracle_user_replies,
                            "extra_allowed": t.extra_user_replies_allowed,
                        }
                    }
                ),
            }
            for t in write_counts.turns
            if not t.passed
        ],
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}
    return {"task_id": scenario_id, "trace_id": trace_id, "score": score, "metadata": metadata}


def _write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f)
            f.write("\n")


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _pass_at_1(records: list[dict[str, Any]]) -> tuple[float | None, int, int]:
    """Pass@1 over one config's records = mean of the non-None scores (each record is one run;
    unscored, errored, and timeline-expired records are excluded). Returns (pass@1 or None if
    nothing scored, scored_count, total_count)."""
    scores = [
        record["score"]
        for record in records
        if record.get("score") is not None
        and not record.get("metadata", {}).get("timeline_expired", False)
    ]
    total = len(records)
    if not scores:
        return None, 0, total
    return sum(scores) / len(scores), len(scores), total


def aggregate(output_dir: str) -> dict[str, Any]:
    """Read every ``{output_dir}/standard/{config}/output.jsonl`` and summarize. Returns
    ``{"configs": {config: {"pass_at_1", "scored", "total"}}, "overall": float | None}`` where
    ``overall`` is the equal-weight mean of pass@1 across the core capabilities that were actually
    run (Gaia2's headline metric)."""
    standard = os.path.join(output_dir, "standard")
    configs: dict[str, dict[str, Any]] = {}
    if os.path.isdir(standard):
        for name in sorted(os.listdir(standard)):
            path = os.path.join(standard, name, "output.jsonl")
            if not os.path.isfile(path):
                continue
            p, scored, total = _pass_at_1(_read_jsonl(path))
            configs[name] = {"pass_at_1": p, "scored": scored, "total": total}
    core = [
        configs[c]["pass_at_1"]
        for c in _CORE_CAPABILITIES
        if c in configs and configs[c]["pass_at_1"] is not None
    ]
    overall = sum(core) / len(core) if core else None
    return {"configs": configs, "overall": overall}


def _print_report(summary: dict[str, Any]) -> None:
    configs: dict[str, dict[str, Any]] = summary["configs"]
    if not configs:
        print("no results found (nothing under {output_dir}/standard/*/output.jsonl)")
        return
    print("\nGaia2 pass@1 by capability")
    print(f"  {'capability':<14} {'pass@1':>8}   scored/total")
    for name in sorted(configs):
        row = configs[name]
        p = row["pass_at_1"]
        cell = "n/a" if p is None else f"{p:6.1%}"
        print(f"  {name:<14} {cell:>8}   {row['scored']}/{row['total']}")
    overall = summary["overall"]
    if overall is not None:
        core = [
            c for c in _CORE_CAPABILITIES if c in configs and configs[c]["pass_at_1"] is not None
        ]
        print(f"  {'overall':<14} {overall:6.1%}   (equal-weight over {', '.join(core)})")


# -- run (lazy ARE imports) -----------------------------------------------------------------------


def _run_capability(args: argparse.Namespace) -> list[dict[str, Any]]:
    from are.simulation.benchmark.scenario_loader import setup_scenarios_iterator

    config_dir = os.path.join(args.output_dir, "standard", args.capability)
    os.makedirs(config_dir, exist_ok=True)

    if args.judge_model:
        # Said once at the top of the sweep as well as per record: an operator watching the run
        # should not have to open output.jsonl to learn the scores are being produced under a
        # patched ARE.
        print(f"judge verdict parse: {_verdict_parse(args)}")

    records: list[dict[str, Any]] = []
    # Stream each record to output.jsonl as it's produced (and flush): a long sweep spends real
    # model tokens, so an abort partway through (a bad scenario, Ctrl-C) must leave a valid partial
    # file of the scenarios already completed rather than discarding all of them — records were
    # previously buffered in memory and written only once at the very end.
    with open(os.path.join(config_dir, "output.jsonl"), "w", encoding="utf-8") as out:
        for run_number in range(args.num_runs):
            # Re-create the iterator each run so every run gets fresh, un-run scenario objects (a
            # scenario is stateful once played); HF caches locally, so re-iteration is cheap.
            scenarios = setup_scenarios_iterator(
                dataset_path=None,
                dataset_config=args.capability,
                dataset_split=args.split,
                hf=args.hf_dataset,
                hf_revision=None,
                load_completed_events=False,
                limit=args.limit,
            )
            for scenario, _events in scenarios:
                # Disambiguate this run's trace file: ARE's get_run_id keys the hf trace filename on
                # scenario.run_number, so without this every run of a scenario writes
                # {scenario_id}.json and later runs silently overwrite earlier ones (leaving their
                # trace_ids pointing at the wrong trace at upload).
                scenario.run_number = run_number
                rec = _run_one_scenario(scenario, run_number, args, config_dir)
                records.append(rec)
                json.dump(rec, out)
                out.write("\n")
                out.flush()
                print(
                    f"[{args.capability}] {scenario.scenario_id} run {run_number}: "
                    f"{rec['metadata']['status']} score={rec['score']}",
                    flush=True,
                )
    return records


def _run_one_scenario(
    scenario: Any,
    run_number: int,
    args: argparse.Namespace,
    config_dir: str,
) -> dict[str, Any]:
    """Run + score + export one scenario into a jsonl record. Any error *for this scenario* (an
    attach_judge/oracle-preprocess failure, or an unexpected export error) becomes an ``exception``
    record so the sweep continues instead of aborting every remaining scenario. A
    ``KeyboardInterrupt`` still propagates so an operator can abort (the streamed output.jsonl keeps
    what's done)."""
    from are.simulation.data_handler.exporter import JsonScenarioExporter
    from are.simulation.scenarios.scenario import ScenarioStatus

    from examples.gaia2._runner import run_scenario
    from sora.adapters.are_sim import (
        attach_judge,
        initialize_turns,
        populate_oracle_events,
    )

    try:
        if args.judge_model:
            attach_judge(
                scenario,
                model=args.judge_model,
                provider=args.judge_provider,
                endpoint=args.judge_endpoint,
                relax_verdict_case=not args.strict_verdict_case,
            )
        else:
            # Replays the oracle so an unscored sweep still reports ARE's tool-call-count gate;
            # deterministic and modelless, and must precede initialize_turns (it soft_resets).
            try:
                populate_oracle_events(scenario)
            except Exception as exc:  # noqa: BLE001 — a diagnostic must never cost the run
                # Caught here rather than by the outer handler, which would record the scenario as
                # an *errored run* and export no trace: the gate is optional information, so
                # failing to compute it must not turn a runnable scenario into a hole in the
                # sweep. The replay restores the scenario before it raises, so the run below still
                # starts from a clean environment — just without a gate.
                print(f"  {scenario.scenario_id}: oracle replay failed ({exc}) — no gate")
            if args.init_turns:
                initialize_turns(scenario)
        result = run_scenario(
            scenario,
            config=args.config,
            verbose=args.verbose,
            max_wall_seconds=args.max_wall_seconds,
            read_stdin=False,
        )
    except Exception as e:  # this scenario's judge/preprocess failed — record it, keep sweeping
        return _jsonl_record(
            scenario_id=scenario.scenario_id,
            run_number=run_number,
            success=None,
            rationale=None,
            exception=e,
            trace_id=None,
        )

    trace_id: str | None = None
    if result.environment is not None:
        # Mirror ARE's own truthy collapse (scenario_runner): None/False -> Invalid.
        decision = (
            ScenarioStatus.Valid.value if result.outcome.success else ScenarioStatus.Invalid.value
        )
        _ok, trace_id = JsonScenarioExporter().export_to_json_file(
            result.environment,
            scenario,
            model_id=args.model,
            agent_id="sora",
            validation_decision=decision,
            validation_rationale=result.outcome.rationale,
            run_duration=result.duration,
            output_dir=config_dir,
            trace_dump_format="hf",
            scenario_exception=result.exception,
        )

    return _jsonl_record(
        scenario_id=scenario.scenario_id,
        run_number=run_number,
        success=result.outcome.success,
        rationale=result.outcome.rationale,
        exception=result.exception,
        trace_id=trace_id,
        awaiting_input=result.awaiting_input,
        write_counts=result.write_counts,
        timeline_expired=result.timeline_expired,
        verdict_parse=_verdict_parse(args),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="batch",
        description="Run S-ORA over a Gaia2 capability, emit HF traces + output.jsonl, report.",
    )
    parser.add_argument(
        "--report-only",
        metavar="OUTPUT_DIR",
        help="Skip running; just aggregate an existing OUTPUT_DIR and print the pass@1 table.",
    )
    parser.add_argument(
        "--capability",
        metavar="CONFIG",
        help="Dataset config to run (e.g. execution, search, adaptability, time, ambiguity, mini).",
    )
    parser.add_argument(
        "--split", default="validation", help="Dataset split (default: validation)."
    )
    parser.add_argument(
        "--hf-dataset",
        default=_DEFAULT_HF_DATASET,
        metavar="REPO",
        help=f"HuggingFace dataset repo (default: {_DEFAULT_HF_DATASET}).",
    )
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        metavar="AGENT_YAML",
        help=f"Agent config (default: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--output-dir",
        default=".sora/gaia2/out",
        metavar="DIR",
        help="Artifact root; traces + output.jsonl land under DIR/standard/<capability>/.",
    )
    parser.add_argument(
        "--model",
        metavar="LABEL",
        help=(
            "Agent-model label recorded in the trace, org-prefixed for a submission (e.g. "
            "S-ORA/claude-opus-4-8). It must name agent.yaml's llm.model — which is the only thing "
            "that selects the model — or the run is refused. Omit to label with llm.model itself."
        ),
    )
    parser.add_argument("--num-runs", type=int, default=1, help="Runs per scenario (default: 1).")
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N", help="Only the first N scenarios (smoke)."
    )
    parser.add_argument(
        "--judge-model", metavar="MODEL", help="Judge model; omit for unscored runs."
    )
    parser.add_argument(
        "--judge-provider", metavar="PROVIDER", help="LiteLLM provider for the judge."
    )
    parser.add_argument("--judge-endpoint", metavar="URL", help="Custom endpoint for the judge.")
    parser.add_argument(
        "--strict-verdict-case",
        action="store_true",
        help=(
            "Do NOT relax ARE's case-sensitive judge-verdict parse (see run_benchmark for the "
            "defect). The default relaxes it, and every scored record says which of the two it "
            "was; pass this to score a sweep under stock ARE."
        ),
    )
    parser.add_argument(
        "--init-turns",
        action="store_true",
        help=(
            "Deliver every turn of a multi-turn scenario without a judge (runs stay unscored). "
            "Without it, an unscored multi-turn scenario stops after turn 1. Excludes "
            "--judge-model."
        ),
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=1200.0,
        metavar="SECONDS",
        help="Per-scenario wall-clock safety cap (default 1200).",
    )
    parser.add_argument("--verbose", action="store_true", help="Stream each scenario's trajectory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.report_only:
        _print_report(aggregate(args.report_only))
        return

    if not args.capability:
        raise SystemExit("--capability is required unless --report-only is given")

    if args.judge_model and args.init_turns:
        # Only the first of the two takes effect (ARE's initialize_turns is idempotent), leaving the
        # judge as the turn gate — the opposite of what --init-turns asks for. Refuse, don't ignore.
        raise SystemExit("--init-turns and --judge-model are mutually exclusive")

    # Before a single token is spent: the trace label has to agree with the model the config selects
    # (and absent a label, becomes it), so an exported trace can't attribute the run to a model that
    # never ran it.
    args.model = _resolve_model_label(args.model, args.config)

    # Absolutize the artifact root before anything writes under it: the HF trace path ARE returns
    # (and stores as each record's `trace_id`) is `os.path.join(output_dir, "hf", <file>)`, and the
    # standalone upload script resolves that `trace_id` with a bare `os.path.exists` from *its* cwd.
    # A relative default (`.sora/gaia2/out`) would make every trace unfindable — and silently
    # dropped from the submission — unless the upload ran from this same directory. Absolute is
    # cwd-independent for that separate step.
    args.output_dir = os.path.abspath(args.output_dir)

    # A `python -m` entry point does not put cwd on sys.path, but agent.yaml's dotted refs resolve
    # project-local code from cwd — match what `sora run` / run_benchmark do.
    if "" not in sys.path:
        sys.path.insert(0, "")

    # Once for the sweep, not per scenario, and before any ARE import: ARE reads DEMO_FS_PATH at
    # import time and binds it as a default argument, so staging it later is a silent no-op.
    from examples.gaia2._local_fs import ensure_local_fallback_fs

    ensure_local_fallback_fs()

    _run_capability(args)
    _print_report(aggregate(args.output_dir))


if __name__ == "__main__":
    main()
