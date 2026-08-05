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

Only ``validate()`` contacts the judge model; without ``--judge-model`` the run is unscored (the
judge no-op), useful for a quick trajectory check. Scenario iteration, ``--num-runs``, and HF trace
export (the full batch harness) build on this same file.
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

    # A console-script/`python -m` entry point does not put the invocation dir on sys.path, but the
    # agent.yaml's dotted refs (and a dotted --scenario) resolve project-local code from cwd — match
    # what `sora run` does.
    if "" not in sys.path:
        sys.path.insert(0, "")

    # Lazy: ARE and the LLM client are optional dependency groups, only needed for an actual run.
    from examples.gaia2._runner import run_scenario
    from sora.adapters.are_sim import attach_judge, load_scenario

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


if __name__ == "__main__":
    main()
