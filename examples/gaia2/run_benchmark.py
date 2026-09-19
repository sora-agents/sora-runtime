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

A scored run also records the judge's raw answers — which ARE itself discards — and prints the same
events re-scored under *both* verdict parses, because one score cannot separate an agent that got it
wrong from a scorer that could not say yes (``relax_judge_verdict_case`` has the account). That
costs no model call: the answers were already paid for. ``--judge-recording PATH`` keeps the file so
``python -m examples.gaia2.rescore`` can read it later.

The run stops **turn-aware**: it rides through the idle gaps between a scenario's turns, then ends
as soon as the agent's final reply has settled and calls ``validate()`` once. ARE installs online
judge gates between turns but not after the final one, so waiting for its event loop to stop would
otherwise consume the scenario's full duration after successful work. A wall-clock cap
(``--max-wall-seconds``) is the safety valve; ``--exit-when-idle`` opts back into the old
single-turn quiet-window heuristic.

The binding budget is usually neither of those: ``scenario.duration`` (1000s by default) is a
**simulated-time** allowance for the whole run. Idle time remains wall-clock paced, but the harness
freezes the environment around each physical model call and advances it by the call's frozen token
charge. The pre-registered rule sums concurrent calls as a conservative single inference lane and
reports their observed wall-time union separately. ``--wall-clock`` disables the generation freeze
for a separately labeled robustness run. Overrun the allowance and the environment stops mid-run —
later turns are never delivered, and the result then looks exactly like an agent that did nothing,
so the run reports ``timeline_expired`` above its own verdict. ``--scenario-duration`` raises that
allowance. It does not shift the scripted schedule: Gaia2 delays are relative to the dependency
that fires them. A score obtained under an override is still not comparable to the stock duration.

Without ``--judge-model`` the run is unscored (the judge no-op), useful for a quick trajectory
check — but on a *multi-turn* scenario it also silently stops after turn 1, because the later turns'
events hang off ``OracleEvent``s that an agent-mode environment ignores. ``--init-turns`` wires
those turns up without a judge, so every turn is delivered unconditionally and an unscored run still
exercises the later-turn behaviour. Scenario iteration, ``--num-runs``, and HF trace export (the
full batch harness) build on this same file.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from examples.gaia2.llm_calls import LLMCallWriter

_DEFAULT_CONFIG = "examples/gaia2/agent.yaml"
_DEFAULT_PROFILES = Path(__file__).resolve().parent / "evaluation" / "profiles.json"
_CHARGE_MODEL = Path(__file__).resolve().parent / "evaluation" / "charge_model.json"


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
        "--judge-recording",
        metavar="PATH",
        help=(
            "Write this run's raw judge responses here as JSON. A scored run records them either "
            "way and prints both parses' scores; this keeps the file, so "
            "`python -m examples.gaia2.rescore PATH` can re-score it later. ARE keeps only a "
            "boolean per judged event, so a run whose responses were not kept is never re-scorable."
        ),
    )
    parser.add_argument(
        "--no-judge-recording",
        action="store_true",
        help="Skip recording the judge's raw responses (and the both-parse comparison below).",
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
            "This is the simulated-time budget for every turn. Idle time is wall-clock paced, "
            "while physical model calls advance it by their frozen token charge. Raising it hands "
            "the agent more time, not more of the scripted world: scheduled events are relative "
            "to a dependency, so the override does not move them. A score obtained under an "
            "override is not comparable to one using the stock duration."
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
        "--wall-clock",
        action="store_true",
        help=(
            "Run the scenario on elapsed wall time instead of freezing model calls and applying "
            "the frozen token charge. This is a robustness mode; its timing result is not "
            "comparable to a token-charged run."
        ),
    )
    parser.add_argument(
        "--charge-profile",
        metavar="NAME",
        help=(
            "Explicit frozen profile to use for token charging when automatic config matching is "
            "ambiguous. It must still match --config exactly and cannot be combined with "
            "--wall-clock or --allow-unfrozen-config."
        ),
    )
    parser.add_argument(
        "--allow-unfrozen-config",
        action="store_true",
        help=(
            "Allow a config outside the frozen profile set for a local/development run. This "
            "necessarily uses wall time because no frozen coefficients exist for the config."
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
    parser.add_argument(
        "--llm-calls",
        metavar="PATH",
        help=(
            "Append one JSON line per model call (tokens, cache reads, measured latency, frozen "
            "charge) to this file. This is what the charge model is fitted and validated against."
        ),
    )
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
    from examples.gaia2.batch import _check_operating_point, _profile_for_config
    from examples.gaia2.evaluation.core import ChargeModelSheet, load_profiles
    from sora.adapters.are_sim import (
        attach_judge,
        initialize_turns,
        load_scenario,
        populate_oracle_events,
    )

    profiles = load_profiles(_DEFAULT_PROFILES)
    profile = None
    if args.charge_profile and args.wall_clock:
        raise SystemExit("--charge-profile and --wall-clock are mutually exclusive")
    if args.charge_profile and args.allow_unfrozen_config:
        raise SystemExit("--charge-profile and --allow-unfrozen-config are mutually exclusive")
    if args.allow_unfrozen_config:
        if not os.path.isfile(args.config):
            raise SystemExit(f"--config does not exist: {args.config}")
        args.wall_clock = True
        print("unfrozen config: using wall clock (development mode; not sweep-comparable)")
    elif args.charge_profile:
        if args.charge_profile not in profiles:
            raise SystemExit(
                f"unknown charge profile {args.charge_profile!r}; have {sorted(profiles)}"
            )
        profile = profiles[args.charge_profile]
        if not os.path.isfile(args.config):
            raise SystemExit(f"--config does not exist: {args.config}")
        _check_operating_point(profile, args.config)
        print(f"explicit charge profile {profile.name} -> {profile.model} at {profile.endpoint}")
    else:
        # Wall-clock is a timing policy, not permission to drift off the frozen operating point.
        profile = _profile_for_config(profiles, args.config)
    charge_sheet = ChargeModelSheet.load(_CHARGE_MODEL)
    profile_charge = None if profile is None else charge_sheet.charge_for(profile)
    charge = None if args.wall_clock else profile_charge

    print(f"loading scenario {args.scenario!r} ...", flush=True)
    scenario: Any = load_scenario(args.scenario)
    record_judge = bool(args.judge_model) and not args.no_judge_recording

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
        if record_judge:
            from sora.adapters.are_judge import arm_judge_recording, judge_recording_armed

            arm_judge_recording()
            record_judge = judge_recording_armed()
            if not record_judge:
                # Not fatal here — this is the single-scenario gate, not a sweep whose tokens are
                # unrecoverable. Say so, because the both-parse comparison below will be missing.
                print(
                    "warning: judge-response recording could not be armed — scores are single-parse"
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
    llm_calls = LLMCallWriter(args.llm_calls) if args.llm_calls else None
    try:
        result = run_scenario(
            scenario,
            config=args.config,
            verbose=args.verbose,
            log_file=args.log_file,
            max_wall_seconds=args.max_wall_seconds,
            exit_when_idle=args.exit_when_idle,
            record_judge=record_judge,
            verdict_parse="stock" if args.strict_verdict_case else "case-insensitive",
            llm_calls=llm_calls,
            scenario_id=getattr(scenario, "scenario_id", None),
            charge=charge,
            charge_model_identity=(profile_charge.identity if profile_charge is not None else None),
            charge_model_digest=charge_sheet.digest if profile_charge is not None else None,
        )
    except KeyboardInterrupt:
        print("\nrun aborted (Ctrl-C) — skipping validation")
        return
    finally:
        if llm_calls is not None:
            llm_calls.close()
            print(f"    wrote {llm_calls.written} model calls to {llm_calls.path}")

    raw_anomalies = int(getattr(result, "raw_cached_input_anomalies", 0))
    if charge is not None and charge.cached_input_clamps != raw_anomalies:
        print(
            "warning: charge clamp count disagrees with independently counted raw usage "
            f"anomalies ({charge.cached_input_clamps} != {raw_anomalies})"
        )
    if charge is None:
        print("scenario clock: wall (robustness mode; not comparable to token-charged runs)")
    else:
        assert profile is not None
        print(
            f"simulated clock: {float(getattr(result, 'charged_seconds', 0.0)):.2f}s charged "
            f"({profile.name}, {charge_sheet.digest})"
        )
        print(
            "inference charge sensitivity: "
            f"sum={float(getattr(result, 'charged_seconds', 0.0)):.2f}s, "
            f"parallel-union={float(getattr(result, 'llm_charged_union_seconds', 0.0)):.2f}s "
            "(trajectory policy: serialized_sum); "
            f"wall sum/union={float(getattr(result, 'llm_wall_seconds', 0.0)):.2f}/"
            f"{float(getattr(result, 'llm_wall_union_seconds', 0.0)):.2f}s"
        )
        print(
            "inference concurrency: "
            f"max={int(getattr(result, 'llm_max_in_flight', 0))}, "
            f"overlapped={int(getattr(result, 'llm_overlapped_round_trips', 0))}/"
            f"{int(getattr(result, 'llm_round_trips', 0))} physical crossings"
        )

    _print_score(result, scored=bool(args.judge_model))
    _report_judge_recording(result.judge_recording, args.judge_recording)


def _report_judge_recording(recording: Any, output_path: str | None) -> None:
    """Re-score this run's judged events under both verdict parses, and optionally keep them.

    Printed next to the verdict because a single score cannot distinguish an agent that got it
    wrong from a scorer that could not say yes — ARE's engines make its ``[[True]]``-family checkers
    unable to return a verdict, and an unparsed answer rejects on the same falsy path a real
    rejection takes (see ``relax_judge_verdict_case``). The two numbers together do distinguish
    them, and cost nothing: no model, no ARE, pure re-parsing of answers already paid for."""
    if recording is None:
        return
    import json

    from sora.adapters.are_judge import VERDICT_PARSES, compare_parses

    if not recording.events:
        print("\njudge recording: no judged events (nothing to re-score)")
        return
    comparison = compare_parses(recording)
    print(f"\njudge recording: {len(recording.events)} judged events")
    for parse in VERDICT_PARSES:
        score = comparison.score(parse)
        print(f"    {parse:<18} {'n/a' if score is None else f'{score:.1%}'} of events pass")
    if comparison.divergent_off_equality_fast_path:
        print(
            f"    the two parses disagree on events {list(comparison.divergent_events)}, at least "
            "one of which the equality checker missed — so this run's score depends on the parse"
        )
    elif comparison.divergent_events:
        print("    the two parses disagree only on equality-fast-path events (no model consulted)")
    else:
        print("    the two parses agree on every event")
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(recording.to_dict(), f)
        print(f"    wrote {output_path}")


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
