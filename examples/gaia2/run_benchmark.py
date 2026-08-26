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

    if args.judge_model:
        # Attach before the run: preprocess_scenario runs the scenario's OracleEvents in oracle mode
        # (deterministic, no model) to build the graph validate() later scores against.
        attach_judge(
            scenario,
            model=args.judge_model,
            provider=args.judge_provider,
            endpoint=args.judge_endpoint,
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
