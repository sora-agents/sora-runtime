"""Gaia2 benchmark driver — the specialized command that keeps benchmark concerns out of core.

Scoring a Gaia2 scenario needs one thing the generic ``sora run`` does not do: attach ARE's
oracle-graph judge *before* the run, so ``AreSimulation.validate()`` returns a real
``ScenarioValidationResult`` instead of the ``success=None`` no-op. Rather than teach the core CLI
about judges (it deliberately stays benchmark-agnostic — see its ``--report`` seam), that wiring
lives here, over the same public seams ``sora run`` uses: ``load_scenario`` → ``attach_judge`` →
``build_agent`` → ``TerminalSession`` → ``validate``.

Correctness gate (one scenario):

    uv sync --all-extras --group are
    export ANTHROPIC_API_KEY=sk-ant-...
    export HF_TOKEN=hf_...                       # judge model, if using a HF-hosted one
    python -m examples.gaia2.run_benchmark \
        --scenario /path/to/gaia2_scenario.json \
        --judge-model claude-sonnet-5 --judge-provider anthropic --verbose

The run stops **timeline-aware**: it rides through the idle gaps between a scenario's turns and ends
once ARE's own event loop has completed the scenario (all turns delivered, all per-turn judge checks
fired), then calls ``validate()`` once — so a multi-turn scenario is scored fully. A wall-clock cap
(``--max-wall-seconds``) is the safety valve; ``--exit-when-idle`` opts back into the old
single-turn quiet-window heuristic.

The binding budget is usually neither of those: ARE's event loop sleeps one real second per tick, so
``scenario.duration`` (1000s by default) is a **real-time** allowance for the whole run. Overrun it
and the environment stops mid-run — later turns are never delivered, and the result then looks
exactly like an agent that did nothing, so the run reports ``timeline_expired`` above its own
verdict. ``--scenario-duration`` raises it; see NOTES.md for why that is safe and what it costs.

Without ``--judge-model`` the run is unscored (the judge no-op), useful for a quick trajectory
check — but on a *multi-turn* scenario it also silently stops after turn 1, because the later turns'
events hang off ``OracleEvent``s that an agent-mode environment ignores. ``--init-turns`` wires
those turns up without a judge, so every turn is delivered unconditionally and an unscored run still
exercises the later-turn behaviour. Scenario iteration, ``--num-runs``, and HF trace export (the
full batch harness) build on this same file.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

_DEFAULT_CONFIG = "examples/gaia2/agent.yaml"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_benchmark",
        description="Run S-ORA on a Gaia2 scenario and print ARE's judge score.",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        metavar="PATH_OR_DOTTED",
        help="A Gaia2 `.json` scenario file (or a dotted path to a Scenario subclass/instance).",
    )
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        metavar="AGENT_YAML",
        help=f"Agent config (default: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--judge-model",
        metavar="MODEL",
        help=(
            "Attach ARE's GraphPerEvent judge so validate() scores against the oracle event "
            "graph. The judge model is contacted only at validate() time. Omit for an unscored run."
        ),
    )
    parser.add_argument(
        "--judge-provider",
        metavar="PROVIDER",
        help="LiteLLM provider for --judge-model (e.g. anthropic, huggingface). Optional.",
    )
    parser.add_argument(
        "--judge-endpoint",
        metavar="URL",
        help="Custom endpoint URL for --judge-model. Optional.",
    )
    parser.add_argument(
        "--strict-verdict-case",
        action="store_true",
        help=(
            "Do NOT relax ARE's case-sensitive judge-verdict parse. ARE reads the judge's verdict "
            "with a case-sensitive substring test for [[True]]; an OpenAI-family judge writes "
            "[[true]], so no vote is recorded, the checker returns None, and the falsy None is "
            "read as a rejection -- which also stops the environment at the turn gate and "
            "withholds every later turn. The default relaxes only that comparison's case, matching "
            "what ARE's own Llama reference judge already produces. Pass this to reproduce stock "
            "ARE behavior."
        ),
    )
    parser.add_argument(
        "--init-turns",
        action="store_true",
        help=(
            "Deliver every turn of a multi-turn scenario without attaching a judge: the run stays "
            "unscored, but the later turns fire unconditionally instead of being gated on a judge "
            "verdict about the earlier ones. Without this (and without --judge-model) a multi-turn "
            "scenario silently stops after turn 1. Mutually exclusive with --judge-model."
        ),
    )
    parser.add_argument(
        "--scenario-duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Override the scenario's own duration (default 1000 for a JSON benchmark scenario). "
            "ARE's event loop is wall-clock paced -- one real second per tick -- so this is the "
            "real-time budget the agent has to finish EVERY turn, and a slow model has the "
            "environment expire mid-run rather than merely scoring badly. Raising it hands the "
            "agent more time, not more of the scripted world: no event is pinned to an absolute "
            "timestamp, and the scheduled ones (the Time capability releases calendar events "
            "31-221s after the opening message) are relative to a dependency, so the override "
            "does not move them. What it does change is how far the simulated clock drifts from "
            "the scenario's start_time. A score obtained under an override is not comparable to "
            "a published one."
        ),
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=1200.0,
        metavar="SECONDS",
        help=(
            "Safety cap: stop after this much wall-clock even if the timeline has not ended "
            "(default 1200). The normal stop is the scenario timeline completing (see below)."
        ),
    )
    parser.add_argument(
        "--exit-when-idle",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Override the turn-aware stop with the old quiet-window heuristic: stop once every "
            "activity has stayed TERMINATED this long. Only correct for single-turn scenarios; a "
            "multi-turn scenario would exit in the gap before a later turn arrives. Omit for the "
            "timeline-aware default."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Stream the full trajectory.")
    parser.add_argument("--log-file", metavar="PATH", help="Mirror the full trace to this file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.judge_model and args.init_turns:
        # Both route through preprocess_scenario and only the first takes effect (ARE's
        # initialize_turns is idempotent), so the combination cannot mean what it looks like: the
        # judge would still be the gate. Refuse rather than silently ignore --init-turns.
        raise SystemExit("--init-turns and --judge-model are mutually exclusive")

    # A console-script/`python -m` entry point does not put the invocation dir on sys.path, but the
    # agent.yaml's dotted refs (and a dotted --scenario) resolve project-local code from cwd — match
    # what `sora run` does.
    if "" not in sys.path:
        sys.path.insert(0, "")

    # Before any ARE import: ARE reads DEMO_FS_PATH at import time and binds it as a default
    # argument, so staging it afterwards is a silent no-op. See _local_fs for what it costs.
    from examples.gaia2._local_fs import ensure_local_fallback_fs

    ensure_local_fallback_fs()

    # Lazy: ARE and the LLM client are optional dependency groups, only needed for an actual run.
    from examples.gaia2._runner import run_scenario
    from sora.adapters.are_sim import (
        attach_judge,
        initialize_turns,
        load_scenario,
        populate_oracle_events,
    )

    print(f"loading scenario {args.scenario!r} ...", flush=True)
    scenario: Any = load_scenario(args.scenario)

    if args.scenario_duration is not None:
        # Set before attach_judge/initialize_turns: both preprocess the scenario, and
        # Environment.run copies `duration` off it at start. Announced because it makes the run
        # incomparable to one obtained under the stock budget, and that fact has to survive the
        # output being pasted somewhere without the command line.
        print(f"scenario duration: {scenario.duration}s -> {args.scenario_duration}s (overridden)")
        scenario.duration = args.scenario_duration

    if args.judge_model:
        # Attach before the run: preprocess_scenario runs the scenario's OracleEvents in oracle mode
        # (deterministic, no model) to build the graph validate() later scores against.
        attach_judge(
            scenario,
            model=args.judge_model,
            provider=args.judge_provider,
            endpoint=args.judge_endpoint,
            relax_verdict_case=not args.strict_verdict_case,
        )
        # Disclosed in the run's own output, not just the log: a score obtained with the verdict
        # parse relaxed is not the same artifact as one obtained under stock ARE, and which of the
        # two it is must survive being pasted somewhere without the log.
        print(
            "judge verdict parse: "
            + ("stock ARE (case-sensitive)" if args.strict_verdict_case else "case-insensitive")
        )
    else:
        # No judge, so nothing else would replay the oracle — do it here (deterministic, no model)
        # purely so the run can still be told whether it cleared ARE's tool-call-count gate. Must
        # precede initialize_turns: it soft_resets the apps, which is ARE's own ordering.
        try:
            populate_oracle_events(scenario)
        except Exception as exc:  # noqa: BLE001 — a diagnostic must never cost the run
            # The gate is extra information *about* a run that is otherwise perfectly runnable, and
            # this is the unscored path — nobody asked to be scored here. Aborting would trade the
            # whole run for a check that was optional to begin with, so say what was lost and go
            # on. The replay restores the scenario before it raises, so the agent still starts
            # from a clean environment; it just starts without a gate to be judged against.
            print(f"warning: oracle replay failed ({exc}) — running without the write-count gate")
        if args.init_turns:
            # Same turn wiring, no judge and no gate: every turn is released regardless of how the
            # earlier ones went, which is what exercising a later turn's behaviour needs.
            initialize_turns(scenario)

    # run_scenario owns the turn-aware done condition (ride through the idle gaps between a
    # scenario's turns; stop once the timeline has completed and the agent is idle; a wall-clock cap
    # is the safety valve). --exit-when-idle opts back into the old single-turn heuristic.
    try:
        result = run_scenario(
            scenario,
            config=args.config,
            verbose=args.verbose,
            log_file=args.log_file,
            max_wall_seconds=args.max_wall_seconds,
            exit_when_idle=args.exit_when_idle,
        )
    except KeyboardInterrupt:
        print("\nrun aborted (Ctrl-C) — skipping validation")
        return

    _print_score(result, scored=bool(args.judge_model))


def _print_score(result: Any, *, scored: bool) -> None:
    if result.exception is not None:  # a judge/oracle misconfig or a run-time crash
        print(f"\nGaia2 validation: n/a ({result.exception})")
        return

    # Printed FIRST and unconditionally, because it reinterprets every line that follows rather
    # than adding to them. When ARE's clock ran out mid-run, the later turns were never delivered
    # at all — the judge reports the turn index never advanced and the gate reports the whole
    # turn's oracle calls as missing, which is exactly what a capable agent that did nothing would
    # produce. Nothing else in the output separates the two.
    if getattr(result, "timeline_expired", False):
        print(
            "\n⏱  ARE timeline EXPIRED mid-run (scenario.duration reached in "
            f"{result.duration:.0f}s of wall clock).\n"
            "    ARE's event loop is wall-clock paced, so a scenario's duration is a real-time\n"
            "    budget for the agent; past it the environment stops and no later turn is ever\n"
            "    delivered. Read the verdict and gate below as a truncated run, not as a wrong\n"
            "    one — a turn the agent never saw cannot have been failed. Re-run with a larger\n"
            "    --scenario-duration, or with a faster model, before drawing any conclusion."
        )

    outcome = result.outcome
    if outcome.success is None:
        note = "no judge attached (--judge-model omitted)" if not scored else "no verdict produced"
        print(f"\nGaia2 validation: unscored ({note})")
    else:
        print(f"\nGaia2 validation: {'✅ PASS' if outcome.success else '❌ FAIL'}")
    if getattr(outcome, "rationale", None):
        print(f"    {outcome.rationale}")
    # Printed for scored and unscored runs alike. On an unscored run it is the only pass/fail
    # signal available; on a scored one it says whether a FAIL was decided before the judge ever
    # looked at the trajectory, which is the difference between "did the wrong thing" and "did an
    # extra thing".
    counts = getattr(result, "write_counts", None)
    if counts is not None:
        print(f"\n{counts.summary()}")
    # A run that stopped to ask something looks identical to a run that merely did badly, unless
    # the question is printed. It is the most actionable line in the output when it appears.
    for prompt in getattr(result, "awaiting_input", []):
        print(f"\nAgent stopped to ask:\n{prompt}")


if __name__ == "__main__":
    main()
