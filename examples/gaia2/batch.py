"""Gaia2 batch harness — run a whole capability, emit leaderboard-grade artifacts, report pass@1.

One invocation runs *one* arm over every scenario of *one* capability (dataset config) and writes,
under ``{output_dir}/standard/{capability}/``:

  * one HF-format trace file per (scenario, run), via ARE's ``JsonScenarioExporter`` — the exact
    artifact ``gaia2_upload_script.py`` consumes, so a run doubles as a leaderboard submission; and
  * ``output.jsonl`` — one line per (scenario, run) in ARE's own
    ``_export_benchmark_result_jsonl`` shape (``task_id``/``trace_id``/``score``/``metadata``); and
  * ``judge_responses/{scenario}.run{n}.json`` — the judge's raw answers per judged event, which
    ARE itself discards. A scored sweep writes them by default and **refuses to start if it
    cannot**: the graph judge keeps a bare boolean, so a run swept without them can never be
    re-scored afterwards at any price, and that is the only failure here with no recovery. Read
    them back with ``python -m examples.gaia2.rescore``.

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

Arms. ``--arm sora`` (the default) runs S-ORA, configured by ``--config`` and matched to exactly
one frozen profile for charging; ``--arm react`` runs ARE's own published ReAct agent as the
baseline, configured by ``--profile`` — the same model,
reasoning setting and output cap the S-ORA arm is pointed at, which is what makes the two columns
comparable, and which a react sweep checks against ``--config`` before spending anything rather
than leaving to whoever wrote the command. Everything else — the judge and its verdict parse, the
oracle replay, the tool-call gate, the record shape, pass@1 — is shared, so the two arms differ in
the agent and nothing else.
A react sweep lands under ``{output_dir}/react/standard/{capability}/`` so both can share one
``--output-dir``; ``--report-only DIR --arm react`` reads it back. The default arm keeps the bare
root, and with it the layout the upload script walks.

    # the pair, same capability, same judge, one root:
    python -m examples.gaia2.batch --capability execution --limit 5 \
        --judge-model claude-sonnet-5 --judge-provider anthropic --output-dir .sora/gaia2/pair
    python -m examples.gaia2.batch --capability execution --limit 5 --arm react \
        --profile gpt-5.4-medium-prompt \
        --judge-model claude-sonnet-5 --judge-provider anthropic --output-dir .sora/gaia2/pair

The profile named there is the one whose operating point matches the shipped ``agent.yaml``; the
S-ORA command derives that same profile from the config and refuses an ambiguous or absent match.
Run the pair at a different operating point by moving *both* sides, not one.

Per-scenario isolation is per fresh ``AreSimulation``; app/global-state bleed across scenarios in
one process is a known risk (a subprocess-per-scenario runner is the fallback if it bites) — fine
for the ``--limit`` smoke runs this is scoped to. The full 160/800 sweep is intentionally held until
the plan iteration primitive lands, since multi-item tasks currently under-count tool calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

from examples.gaia2.evaluation.core import (
    LEGACY_CLOCK_MODE,
    resolve_clock_mode,
    resolve_max_wall_seconds,
)
from examples.gaia2.llm_calls import LLMCallWriter

_DEFAULT_CONFIG = "examples/gaia2/agent.yaml"
_DEFAULT_HF_DATASET = "meta-agents-research-environments/gaia2"
_DEFAULT_PROFILES = "examples/gaia2/evaluation/profiles.json"

# The two arms of the paired comparison: S-ORA's decision cycle, and ARE's own published ReAct
# agent as the baseline. The arm names match the `arm` field on every per-call row in
# llm_calls.jsonl, which is what joins a sweep's records to its cost.
_ARMS = ("sora", "react")

# The five capabilities Gaia2's headline (equal-weight) score averages over. A run may target any
# dataset config (incl. `mini`); the report weights only these five when they're present.
_CORE_CAPABILITIES = ("execution", "search", "adaptability", "time", "ambiguity")


# -- pure helpers (no ARE import; unit-tested without the `are` extra) -----------------------------


@dataclass(frozen=True)
class SweepManifest:
    """Exact, ordered Gaia2 inputs shared by every arm and model in a paid sweep."""

    name: str
    dataset: str
    revision: str
    split: str
    cases: tuple[tuple[str, str], ...]
    digest: str

    def scenario_ids(self, capability: str) -> tuple[str, ...]:
        return tuple(
            case_id for case_capability, case_id in self.cases if case_capability == capability
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_sweep_manifest(path: str | Path) -> SweepManifest:
    """Load a dataset-revision-pinned scenario selection for the batch runner.

    Prompt-evaluation manifests deliberately require five cases, one per capability. A batch sweep
    can instead pin any non-empty ordered subset of one or more capabilities, so it shares the
    envelope but has its own validation rather than weakening that campaign's locked schema.
    """

    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"--scenario-manifest does not exist: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in --scenario-manifest {manifest_path}: {exc}") from exc

    required = {"schema_version", "name", "dataset", "revision", "split", "cases"}
    if not isinstance(raw, dict) or set(raw) != required or raw.get("schema_version") != 1:
        raise SystemExit(
            f"invalid sweep manifest {manifest_path}: expected exactly {sorted(required)} with "
            "schema_version 1"
        )
    name = raw["name"]
    dataset = raw["dataset"]
    revision = raw["revision"]
    split = raw["split"]
    rows = raw["cases"]
    if not all(isinstance(value, str) and value for value in (name, dataset, revision, split)):
        raise SystemExit(
            f"invalid sweep manifest {manifest_path}: name/dataset/revision/split must be "
            "non-empty strings"
        )
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"invalid sweep manifest {manifest_path}: cases must be a non-empty list")

    cases: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"capability", "id"}:
            raise SystemExit(
                f"invalid sweep manifest {manifest_path}: every case must contain only "
                "capability/id"
            )
        capability = row["capability"]
        case_id = row["id"]
        if not isinstance(capability, str) or not capability:
            raise SystemExit(f"invalid sweep manifest capability in {manifest_path}: {row!r}")
        if not isinstance(case_id, str) or not case_id:
            raise SystemExit(f"invalid sweep manifest scenario id in {manifest_path}: {row!r}")
        cases.append((capability, case_id))
    ids = [case_id for _capability, case_id in cases]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"invalid sweep manifest {manifest_path}: scenario ids must be unique")

    return SweepManifest(
        name=name,
        dataset=dataset,
        revision=revision,
        split=split,
        cases=tuple(cases),
        digest=hashlib.sha256(_canonical_json(raw).encode()).hexdigest(),
    )


def _select_manifest_scenarios(
    loaded: list[tuple[Any, Any]], manifest: SweepManifest, capability: str
) -> list[tuple[Any, Any]]:
    """Return fresh scenario objects in manifest order, failing before any paid artifact opens."""

    wanted = manifest.scenario_ids(capability)
    if not wanted:
        raise RuntimeError(
            f"sweep manifest {manifest.name!r} selects no scenarios for capability {capability!r}"
        )
    by_id: dict[str, tuple[Any, Any]] = {}
    duplicates: set[str] = set()
    for item in loaded:
        scenario_id = str(item[0].scenario_id)
        if scenario_id in by_id:
            duplicates.add(scenario_id)
        by_id[scenario_id] = item
    if duplicates:
        raise RuntimeError(f"Gaia2 loader returned duplicate scenario ids: {sorted(duplicates)}")
    missing = [scenario_id for scenario_id in wanted if scenario_id not in by_id]
    if missing:
        raise RuntimeError(
            f"sweep manifest {manifest.name!r} scenarios missing from "
            f"{manifest.dataset}@{manifest.revision}/{capability}/{manifest.split}: {missing}"
        )
    return [by_id[scenario_id] for scenario_id in wanted]


def _pinned_hf_scenarios(
    *,
    dataset: str,
    capability: str,
    split: str,
    revision: str,
    limit: int | None,
) -> Iterator[tuple[Any, Any]]:
    """Load one immutable HF revision without ARE's misnamed datasets keyword.

    The installed ARE loader passes ``dataset_revision=`` to ``datasets.load_dataset``. That is a
    builder-config parameter, not Hugging Face's repository selector (``revision=``), so it both
    misses the normal cache and fails to pin the source commit. Keep the correction local to the
    benchmark harness until ARE exposes the repository revision correctly.
    """

    from are.simulation.benchmark.scenario_loader import load_scenario
    from are.simulation.data_handler.models import ExportedHuggingFaceMetadata
    from datasets import load_dataset  # type: ignore[import-untyped]

    loaded = load_dataset(
        dataset,
        name=capability,
        revision=revision,
        streaming=True,
    )
    if split not in loaded:
        raise RuntimeError(f"split {split!r} not found in {dataset}@{revision}/{capability}")

    for index, row in enumerate(loaded[split]):
        if limit is not None and index >= limit:
            break
        scenario_id = row.get("scenario_id")
        if not scenario_id:
            continue
        scenario, completed_events = load_scenario(
            row["data"],
            str(scenario_id),
            False,
            hf_metadata=ExportedHuggingFaceMetadata(
                dataset=dataset,
                split=split,
                revision=revision,
            ),
        )
        if scenario is None or completed_events is None:
            continue
        scenario.run_number = row.get("run_number")
        yield scenario, completed_events


def _resolved_charge(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    """``(charge, profile_charge, charge_model)`` for this run.

    One derivation rather than one per caller, because the pre-flight pairing check has to predict
    the clock mode the rows will carry: a check deriving it separately could agree with itself and
    still disagree with what the run went on to record."""
    charge_model = getattr(args, "charge_model", None)
    if charge_model is None:
        from examples.gaia2.evaluation.core import ChargeModelSheet

        charge_model = ChargeModelSheet.load(
            Path(__file__).resolve().parent / "evaluation" / "charge_model.json"
        )
    profile = getattr(args, "model_profile", None)
    profile_charge = charge_model.charge_for(profile) if profile is not None else None
    charge = (
        None
        if getattr(args, "wall_clock", False) or getattr(args, "generation_free", False)
        else profile_charge
    )
    return charge, profile_charge, charge_model


def _intended_clock_mode(args: argparse.Namespace) -> str:
    """The clock mode this arm's rows will carry, known before the first scenario runs."""
    charge, _profile_charge, _sheet = _resolved_charge(args)
    return resolve_clock_mode(charge, bool(getattr(args, "generation_free", False)))


def _verify_counterpart_pairing(args: argparse.Namespace, manifest: SweepManifest) -> None:
    """Refuse a paid pair whose already-written arm is not comparable to the one about to run.

    Two arms are a comparison only if they differ in the architecture and nothing else. The
    scenario selection was checked here from the start; the clock convention and the operating
    point were not, so two individually valid arms — each internally homogeneous, each passing
    every per-file guard — could be paired across different timing conventions or different
    endpoints and produce a difference that is not the one being measured. Checked here, before
    tokens are spent, because after the fact the only repair is to re-run the pair."""

    counterpart_root = (
        os.path.join(args.output_dir, "react") if args.arm == "sora" else args.output_dir
    )
    path = os.path.join(counterpart_root, "standard", args.capability, "output.jsonl")
    if not os.path.isfile(path):
        return
    rows = _read_jsonl(path)
    recorded_digests = [row.get("metadata", {}).get("scenario_manifest_digest") for row in rows]
    if not recorded_digests or any(digest != manifest.digest for digest in recorded_digests):
        raise RuntimeError(
            f"counterpart arm {path} does not carry this sweep manifest digest "
            f"{manifest.digest}: found "
            f"{sorted({str(digest) for digest in recorded_digests}) or ['missing']}"
        )
    _verify_counterpart_field(
        rows,
        path,
        field="clock_mode",
        expected=_intended_clock_mode(args),
        what="clock convention",
    )
    profile_name = getattr(getattr(args, "model_profile", None), "name", None)
    if profile_name is not None:
        _verify_counterpart_field(
            rows,
            path,
            field="model_profile",
            expected=str(profile_name),
            what="operating point",
        )
    expected = {
        (scenario_id, run_number)
        for scenario_id in manifest.scenario_ids(args.capability)
        for run_number in range(args.num_runs)
    }
    actual = {
        (
            str(row.get("metadata", {}).get("scenario_id")),
            int(row.get("metadata", {}).get("run_number", -1)),
        )
        for row in rows
    }
    if actual != expected or len(rows) != len(expected):
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            f"counterpart arm {path} is not the same complete scenario/run matrix; "
            f"rows={len(rows)}, expected_rows={len(expected)}, missing={missing}, "
            f"unexpected={unexpected}"
        )


def _verify_counterpart_field(
    rows: list[dict[str, Any]], path: str, *, field: str, expected: str, what: str
) -> None:
    """Refuse when the written arm disagrees with this one about ``field``.

    A missing value is a disagreement too: these rows were written by the same harness, so an
    absent field means the counterpart predates this provenance rather than that it happens to
    match."""
    found = sorted({str(row.get("metadata", {}).get(field)) for row in rows})
    if found != [expected]:
        raise RuntimeError(
            f"counterpart arm {path} ran under a different {what}: this arm records "
            f"{field}={expected}, the counterpart records {', '.join(found)}"
        )


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


def _arm_root(output_dir: str, arm: str) -> str:
    """Where one arm's artifacts live under a shared ``--output-dir``.

    The S-ORA arm keeps the bare root, so its layout — and the standalone uploader's walk of
    ``{root}/standard/{config}/output.jsonl``, which ``tests/test_gaia2_upload_compat.py`` locks —
    is byte-for-byte what it was. Every other arm gets its own subtree. Without this, sweeping both
    arms into one ``--output-dir`` would have the second silently truncate the first's
    ``output.jsonl`` *and* ``llm_calls.jsonl``: same capability, same filenames, and the truncation
    is deliberate (re-running a capability replaces its artifacts), so nothing would flag it."""
    return output_dir if arm == "sora" else os.path.join(output_dir, arm)


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


def _resolve_react_label(label: str | None, profile: Any) -> str:
    """The same contract as :func:`_resolve_model_label`, for the arm whose model comes from a
    profile instead of from agent.yaml. The profile is the only thing that selects the model on this
    arm — ARE's runner config is built from it — so a label that does not name it is refused before
    any tokens are spent rather than mislabeling the trace afterwards."""
    if label is None:
        return str(profile.model)
    if str(profile.model) not in label:
        raise SystemExit(
            f"--model {label!r} does not name the model profile {profile.name!r} actually uses "
            f"({profile.model!r}). The label is only recorded in the trace — it cannot override "
            f"the profile. Fix the label, or pass a different --profile."
        )
    return label


# Which keys are *not* part of the operating point. Everything else either side declares is, so a
# setting added to a profile or to agent.yaml is compared from the day it exists rather than the day
# someone remembers to widen a list — the failure of an allowlist here is silent and lands on a
# comparison that cannot be redone. Transport decides how a call is carried, not what was asked for;
# `instrument` decides whether the call is recorded, and its failure mode is a loudly empty
# llm_calls.jsonl; `max_logical_calls` budgets S-ORA's own loop and has no per-request meaning at
# all (ARE's loop cap is the react arm's analogue, and it is not a profile field).
_NOT_THE_OPERATING_POINT = frozenset(
    {"stall_timeout", "max_retries", "instrument", "max_logical_calls"}
)


def _configured_operating_point(config_path: str) -> dict[str, Any] | None:
    if not os.path.isfile(config_path):
        return None
    from sora.bootstrap import load_yaml

    return load_yaml(config_path).llm or {}


def _operating_point_diffs(
    profile: Any, configured: dict[str, Any] | None
) -> list[tuple[str, Any, Any]]:
    if configured is None:
        return []
    settings = profile.client_settings()
    unset = "<unset>"
    compared = (set(settings) | set(configured)) - _NOT_THE_OPERATING_POINT
    return [
        (name, settings.get(name, unset), configured.get(name, unset))
        for name in sorted(compared)
        if settings.get(name, unset) != configured.get(name, unset)
    ]


def _check_operating_point(profile: Any, config_path: str) -> None:
    """Refuse a react sweep whose profile does not run at the same operating point as the config.

    The point of the baseline is that the two arms differ in the agent and nothing else, and the
    operating point is the one difference nothing downstream can correct: a row records which
    profile produced it, so a ReAct arm run at ``high`` against an S-ORA config at ``medium``
    produces two pass@1 columns that look comparable, are labelled with the same model, and are
    not. It is also the difference easiest to introduce by accident, since the two arms are two
    separate commands reading two separate files. ``ModelProfile.client_settings`` already speaks
    agent.yaml's ``llm:`` vocabulary, so the comparison is the same keys on both sides, and it is
    taken over the union of what the two declare rather than a list of interesting ones — routing
    is as much the experiment as reasoning effort is, since the same model name at a different
    ``base_url`` or behind a different ``provider_routing`` is a different serving path, and
    ``stream`` decides whether the token counts in the cost column were reported by the provider
    or reconstructed locally. A missing config is not an error — a react-only run has nothing to
    be compared against."""
    diffs = _operating_point_diffs(profile, _configured_operating_point(config_path))
    if not diffs:
        return
    detail = "\n".join(
        f"  {name}: profile {here!r} vs config {there!r}" for name, here, there in diffs
    )
    raise SystemExit(
        f"profile {profile.name!r} and {config_path} do not describe the same operating point:\n"
        f"{detail}\n"
        "The two arms are comparable only at the same one. Pick a profile that matches the "
        "config, change the config's llm: block to match the profile, or point --config at the "
        "config this profile pairs with."
    )


def _profile_for_config(profiles: dict[str, Any], config_path: str) -> Any:
    """Return the one frozen profile describing ``config_path``'s operating point."""
    configured = _configured_operating_point(config_path)
    if configured is None:
        raise SystemExit(f"--config does not exist: {config_path}")
    matching = [
        profile for profile in profiles.values() if not _operating_point_diffs(profile, configured)
    ]
    if len(matching) != 1:
        raise SystemExit(
            f"--config must match exactly one frozen model profile; matched "
            f"{[profile.name for profile in matching]}. Use --charge-profile to select between "
            "matching frozen profiles, or --allow-unfrozen-config for an explicitly wall-clock "
            "development run."
        )
    return matching[0]


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
    harness_truncation: str | None = None,
    terminal_cause: str | None = None,
    max_wall_seconds: float | None = None,
    write_counts: Any = None,
    timeline_expired: bool = False,
    verdict_parse: str | None = None,
    judge_recording_path: str | None = None,
    charged_seconds: float | None = None,
    model_profile: str | None = None,
    charge_model_identity: dict[str, Any] | None = None,
    charge_model_digest: str | None = None,
    cached_input_clamps: int | None = None,
    raw_cached_input_anomalies: int | None = None,
    charge_accounting_consistent: bool | None = None,
    clock_mode: str | None = None,
    inference_charge_policy: str | None = None,
    llm_wall_seconds: float | None = None,
    llm_wall_union_seconds: float | None = None,
    llm_charged_union_seconds: float | None = None,
    llm_round_trips: int | None = None,
    llm_max_in_flight: int | None = None,
    llm_overlapped_round_trips: int | None = None,
    scenario_manifest_digest: str | None = None,
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
        # Which harness watchdog ended this run, when one did: `"wall_clock"` (`--max-wall-seconds`
        # elapsed) or `"judge_stall"` (the scoring pass sat paused past its own cap). Either way the
        # environment was stopped mid-trajectory. Recorded separately from `timeline_expired`
        # because both stop the world the same way and are otherwise indistinguishable in the
        # artifact, and because this one is the harness cutting a run short rather than the run
        # reaching the end of its schedule. Excluded from pass@1 for that reason.
        #
        # One nullable name rather than a boolean per watchdog: the exclusion needs the union, and
        # a pair of flags invites filtering on just one — which is exactly how a stalled-judge run
        # kept scoring on one arm while the other excluded it under a label that misdiagnosed it.
        # None when neither fired, so an ordinary record stays byte-identical to ARE's own shape.
        "harness_truncation": harness_truncation,
        # Why the run finally stopped, in the run's own taxonomy. Computed on both arms and, until
        # now, dropped on the way into this record — which left a watchdog-truncated run looking
        # exactly like one that finished.
        "terminal_cause": terminal_cause,
        # The watchdog value this run was given. Part of the timing configuration, not a harness
        # detail: under a frozen clock the cap is the only bound on a trajectory's length, so a row
        # that does not carry it cannot say whether its own truncation was plausible.
        "max_wall_seconds": max_wall_seconds,
        # Scoring provenance, recorded on every scored record — including the default. Unlike the
        # diagnostics around it this is not "extra information about an ordinary run": the default
        # relaxes ARE's verdict parse, so a sweep's scores are obtained under a patched judge, and
        # a record that does not say so cannot be compared with one produced by stock ARE. Same
        # reasoning as run_benchmark printing it: which of the two this is must survive the record
        # being read without the log beside it. None on an unscored run — there were no verdicts.
        "verdict_parse": verdict_parse,
        # Where this run's judge responses were stored. Recorded because the score above is not
        # re-derivable without them and ARE keeps nothing: a row that scores a run but cannot say
        # where its judge answers went is a row nobody can ever re-score. Absolute, for the same
        # reason `trace_id` is — the re-scorer is a separate step, run from a different cwd.
        "judge_recording": judge_recording_path,
        "charged_seconds": charged_seconds,
        "model_profile": model_profile,
        "charge_model_identity": charge_model_identity,
        "charge_model_digest": charge_model_digest,
        "cached_input_clamps": cached_input_clamps,
        "raw_cached_input_anomalies": raw_cached_input_anomalies,
        "charge_accounting_consistent": charge_accounting_consistent,
        "clock_mode": clock_mode,
        "inference_charge_policy": inference_charge_policy,
        "llm_wall_seconds": llm_wall_seconds,
        "llm_wall_union_seconds": llm_wall_union_seconds,
        "llm_charged_union_seconds": llm_charged_union_seconds,
        "llm_round_trips": llm_round_trips,
        "llm_max_in_flight": llm_max_in_flight,
        "llm_overlapped_round_trips": llm_overlapped_round_trips,
        "scenario_manifest_digest": scenario_manifest_digest,
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


def _arm_judge_recording() -> bool:
    """Install the judge-response patch, and report whether it is in force.

    Named as its own indirection so the batch gate can be tested without ARE: arming is idempotent
    and returns False on a second call, which is not the same answer as "not armed"."""
    from sora.adapters.are_judge import arm_judge_recording, judge_recording_armed

    arm_judge_recording()
    return judge_recording_armed()


def _require_judge_recording(args: argparse.Namespace) -> bool:
    """Whether this sweep records judge responses — refusing to start a scored one that cannot.

    ARE's graph judge keeps a bare boolean, so a scored sweep run without recording can never be
    re-scored afterwards, at any price: the tokens are spent and the judge's reasoning is gone. That
    makes it the one failure in the harness with no recovery, and the reason this is a mechanical
    refusal rather than a documented recommendation. An unscored sweep judges nothing and is
    unaffected; ``--no-judge-recording`` is the deliberate opt-out, and it is never the default."""
    if not args.judge_model or args.no_judge_recording:
        return False
    if not _arm_judge_recording():
        raise SystemExit(
            "judge-response recording could not be armed (is the `are` extra installed?). "
            "A scored sweep without it cannot be re-scored afterwards — ARE keeps only a boolean "
            "per judged event. Fix the environment, or pass --no-judge-recording to accept that."
        )
    return True


def _write_judge_recording(
    config_dir: str, recording: Any, scenario_id: str, run_number: int
) -> str | None:
    """Store one run's judge responses beside its trace. None when there was nothing to store."""
    if recording is None:
        return None
    directory = os.path.join(config_dir, "judge_responses")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{scenario_id}.run{run_number}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recording.to_dict(), f)
    return path


def _write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f)
            f.write("\n")


def _optional_sum(values: Iterable[Any]) -> float | None:
    """Sum that propagates an unavailable parallel-charge counterfactual (see llm_calls)."""
    total = 0.0
    for value in values:
        if value is None:
            return None
        total += float(value)
    return total


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _is_scored(record: dict[str, Any]) -> bool:
    """Whether this run's score counts toward pass@1.

    One predicate rather than a filter inlined at each consumer, because the headline's scenario
    coverage check has to mean exactly the same thing by "scored" as pass@1 does — a coverage check
    running on a looser rule would report a capability as complete on the strength of runs the
    score itself threw away."""
    return (
        record.get("score") is not None
        and not record.get("metadata", {}).get("timeline_expired", False)
        # A watchdog-truncated run was stopped by this harness mid-trajectory, so its score
        # measures how far the agent got before the cap, not how well it did. Excluded for the
        # same reason an expired timeline is, and named separately because the two have different
        # causes and different fixes. Any watchdog counts — the field names which one fired, and
        # testing it for truth rather than for a particular value is what keeps a second watchdog
        # from being added later without being excluded here.
        and not record.get("metadata", {}).get("harness_truncation")
    )


def _pass_at_1(records: list[dict[str, Any]]) -> tuple[float | None, int, int]:
    """Pass@1 over one config's records = mean of the non-None scores (each record is one run;
    unscored, errored, timeline-expired and watchdog-truncated records are excluded). Returns
    (pass@1 or None if nothing scored, scored_count, total_count)."""
    scores = [record["score"] for record in records if _is_scored(record)]
    total = len(records)
    if not scores:
        return None, 0, total
    return sum(scores) / len(scores), len(scores), total


def aggregate(output_dir: str, manifest: SweepManifest | None = None) -> dict[str, Any]:
    """Read every ``{output_dir}/standard/{config}/output.jsonl`` and summarize.

    ``overall`` is Gaia2's headline metric — the equal-weight mean of pass@1 across the five core
    capabilities — and is ``None`` unless the results can carry it: all five present and scored, one
    clock mode, one scenario manifest, and (against ``manifest``) every pinned scenario actually
    scored. ``headline_withheld`` names each unmet condition. ``exploratory_mean`` is the mean over
    whatever did score, for reading a sweep in progress; it is not a result."""
    standard = os.path.join(output_dir, "standard")
    configs: dict[str, dict[str, Any]] = {}
    if os.path.isdir(standard):
        for name in sorted(os.listdir(standard)):
            path = os.path.join(standard, name, "output.jsonl")
            if not os.path.isfile(path):
                continue
            rows = _read_jsonl(path)
            p, scored, total = _pass_at_1(rows)
            # Same rule as the per-record report: a row without a mode is a legacy row and is
            # named as one, so it participates in the mixing check instead of being filtered out
            # of it. Excluding it let a legacy run sit beside a charged one and still look
            # homogeneous, which is the one mixture pass@1 must refuse.
            clock_modes = sorted(
                {
                    str(row.get("metadata", {}).get("clock_mode") or LEGACY_CLOCK_MODE)
                    for row in rows
                }
            )
            mixed_clock_modes = len(clock_modes) > 1
            manifest_markers = {
                row.get("metadata", {}).get("scenario_manifest_digest") for row in rows
            }
            manifest_digests = sorted(
                {str(digest) for digest in manifest_markers if digest is not None}
            )
            clamps = sum(int(row.get("metadata", {}).get("cached_input_clamps", 0)) for row in rows)
            anomalies = sum(
                int(row.get("metadata", {}).get("raw_cached_input_anomalies", 0)) for row in rows
            )
            accounting_mismatches = sum(
                (
                    row.get("metadata", {}).get("charge_accounting_consistent") is False
                    or (
                        row.get("metadata", {}).get("charge_accounting_consistent") is None
                        # Only a charged run has clamps to disagree about. A generation-free run
                        # applies no cache clamp and computes no charge, so comparing its raw
                        # usage anomalies against a structurally zero clamp count reports a
                        # mismatch that describes the provider's usage figures, not this harness's
                        # accounting — which is what this counter exists to police.
                        and row.get("metadata", {}).get("clock_mode") == "token_charged"
                        and "cached_input_clamps" in row.get("metadata", {})
                        and "raw_cached_input_anomalies" in row.get("metadata", {})
                        and int(row["metadata"]["cached_input_clamps"])
                        != int(row["metadata"]["raw_cached_input_anomalies"])
                    )
                )
                for row in rows
            )
            mixed_scenario_manifests = len(manifest_markers) > 1
            configs[name] = {
                # Timing policy changes trajectories, so a mixed file has no coherent aggregate
                # score. The same is true of two different scenario selections. Keep the rows
                # readable but refuse to promote either mixture as pass@1.
                "pass_at_1": None if mixed_clock_modes or mixed_scenario_manifests else p,
                "scored": scored,
                "total": total,
                "charged_seconds": sum(
                    float(row.get("metadata", {}).get("charged_seconds", 0.0)) for row in rows
                ),
                "llm_wall_seconds": sum(
                    float(row.get("metadata", {}).get("llm_wall_seconds", 0.0)) for row in rows
                ),
                "llm_wall_union_seconds": sum(
                    float(row.get("metadata", {}).get("llm_wall_union_seconds", 0.0))
                    for row in rows
                ),
                "llm_wall_overlap_seconds": sum(
                    max(
                        0.0,
                        float(row.get("metadata", {}).get("llm_wall_seconds", 0.0))
                        - float(row.get("metadata", {}).get("llm_wall_union_seconds", 0.0)),
                    )
                    for row in rows
                ),
                # A null union means that run mixed time axes and has no counterfactual; the
                # group inherits the unavailability rather than averaging over a short count.
                "llm_charged_union_seconds": _optional_sum(
                    row.get("metadata", {}).get("llm_charged_union_seconds", 0.0) for row in rows
                ),
                "llm_charged_overlap_seconds": _optional_sum(
                    None
                    if row.get("metadata", {}).get("llm_charged_union_seconds", 0.0) is None
                    else max(
                        0.0,
                        float(row.get("metadata", {}).get("charged_seconds", 0.0))
                        - float(row.get("metadata", {}).get("llm_charged_union_seconds", 0.0)),
                    )
                    for row in rows
                ),
                "llm_round_trips": sum(
                    int(row.get("metadata", {}).get("llm_round_trips", 0)) for row in rows
                ),
                "llm_max_in_flight": max(
                    (int(row.get("metadata", {}).get("llm_max_in_flight", 0)) for row in rows),
                    default=0,
                ),
                "llm_overlapped_round_trips": sum(
                    int(row.get("metadata", {}).get("llm_overlapped_round_trips", 0))
                    for row in rows
                ),
                "scored_scenario_ids": sorted(
                    {
                        str(row.get("metadata", {}).get("scenario_id"))
                        for row in rows
                        if _is_scored(row)
                        and row.get("metadata", {}).get("scenario_id") is not None
                    }
                ),
                "cached_input_clamps": clamps,
                "raw_cached_input_anomalies": anomalies,
                "charge_accounting_mismatches": accounting_mismatches,
                "clock_modes": clock_modes,
                "mixed_clock_modes": mixed_clock_modes,
                "scenario_manifest_digests": manifest_digests,
                "mixed_scenario_manifests": mixed_scenario_manifests,
                "inference_charge_policies": sorted(
                    {
                        str(row.get("metadata", {}).get("inference_charge_policy"))
                        for row in rows
                        if row.get("metadata", {}).get("inference_charge_policy") is not None
                    }
                ),
                "charge_models": [
                    json.loads(encoded)
                    for encoded in sorted(
                        {
                            json.dumps(
                                {
                                    "identity": row.get("metadata", {}).get(
                                        "charge_model_identity"
                                    ),
                                    "digest": row.get("metadata", {}).get("charge_model_digest"),
                                },
                                sort_keys=True,
                            )
                            for row in rows
                            if row.get("metadata", {}).get("charge_model_digest") is not None
                        }
                    )
                ],
                "model_profiles": sorted(
                    {
                        str(row.get("metadata", {}).get("model_profile"))
                        for row in rows
                        if row.get("metadata", {}).get("model_profile") is not None
                    }
                ),
            }
    scored_core = [
        configs[c]["pass_at_1"]
        for c in _CORE_CAPABILITIES
        if c in configs and configs[c]["pass_at_1"] is not None
    ]
    exploratory_mean = sum(scored_core) / len(scored_core) if scored_core else None
    withheld = _headline_withheld(configs, manifest)
    return {
        "configs": configs,
        # The paper number, or nothing. Every earlier version of this key was a mean over whatever
        # happened to be on disk, which is a useful development readout and a dangerous headline:
        # a capability that failed to run, or ran under a second clock, or ran a different scenario
        # selection, simply left the average — quietly raising or lowering it with no trace in the
        # number itself. An incomplete result is now unavailable rather than approximate.
        "overall": None if withheld else exploratory_mean,
        # The same mean, named for what it is, so partial sweeps stay readable while they are in
        # progress. Never promote this into a report.
        "exploratory_mean": exploratory_mean,
        "headline_withheld": withheld,
        "overall_clock_modes": _promoted_values(configs, "clock_modes"),
        "overall_scenario_manifest_digests": _promoted_values(configs, "scenario_manifest_digests"),
    }


def _promoted_values(configs: dict[str, dict[str, Any]], key: str) -> list[str]:
    """The distinct values of a per-capability list field, across the core capabilities present."""
    return sorted(
        {
            str(value)
            for name in _CORE_CAPABILITIES
            if name in configs
            for value in configs[name].get(key, ())
        }
    )


def _headline_withheld(
    configs: dict[str, dict[str, Any]], manifest: SweepManifest | None
) -> tuple[str, ...]:
    """Every reason this set of results cannot be reported as one Gaia2 headline.

    Each capability file already refuses to promote its own pass@1 when its rows mix clock modes or
    scenario manifests, but the headline is a mean *across* files, and a per-file check cannot see
    any of the ways that mean goes wrong: a capability that ran under a second clock, a capability
    that ran a different scenario selection, a capability that is simply absent, and a capability
    whose own pass@1 was suppressed — which used to drop out of the average silently, so suppressing
    it made the headline *more* available rather than less.

    Returned as reasons rather than a bool because "no headline" is not actionable on its own, and
    because the operator needs to see which of these a sweep is one step away from fixing."""
    reasons: list[str] = []

    missing = [name for name in _CORE_CAPABILITIES if name not in configs]
    if missing:
        reasons.append(f"capabilities not run: {', '.join(missing)}")
    unscorable = [
        name
        for name in _CORE_CAPABILITIES
        if name in configs and configs[name]["pass_at_1"] is None
    ]
    if unscorable:
        reasons.append(f"capabilities with no comparable pass@1: {', '.join(unscorable)}")

    clock_modes = _promoted_values(configs, "clock_modes")
    if len(clock_modes) > 1:
        reasons.append(f"capabilities ran under different clock modes: {', '.join(clock_modes)}")

    digests = _promoted_values(configs, "scenario_manifest_digests")
    if len(digests) > 1:
        reasons.append(
            "capabilities ran under different scenario manifests: "
            + ", ".join(digest[:12] for digest in digests)
        )

    # Coverage is the one gate that cannot be checked from the rows alone: they say what ran, never
    # what was supposed to. Withheld rather than assumed when no manifest is given — a headline
    # computed over an unknown denominator is the failure this gate exists for.
    if manifest is None:
        reasons.append("scenario coverage unverified: no --scenario-manifest given")
    else:
        # Not `if digests and ...`: rows that recorded no digest at all are the case this check
        # most needs to catch. Missing provenance is not evidence of agreement with the manifest,
        # and treating it as one let a fully scored five-capability sweep carrying no digest
        # whatsoever earn the headline — the one shape where nothing else would have objected.
        if digests != [manifest.digest]:
            found = ", ".join(digest[:12] for digest in digests) or "no digest recorded"
            reasons.append(
                f"results do not carry this manifest's digest {manifest.digest[:12]}: {found}"
            )
        uncovered: list[str] = []
        for name in _CORE_CAPABILITIES:
            scored = set(configs.get(name, {}).get("scored_scenario_ids", ()))
            uncovered.extend(
                f"{name}/{scenario_id}"
                for scenario_id in manifest.scenario_ids(name)
                if scenario_id not in scored
            )
        if uncovered:
            shown = ", ".join(uncovered[:5]) + (" ..." if len(uncovered) > 5 else "")
            reasons.append(f"{len(uncovered)} manifest scenarios not scored: {shown}")

    return tuple(reasons)


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
        if row.get("clock_modes"):
            if row["mixed_clock_modes"]:
                print("    WARNING: mixed clock modes; pass@1 suppressed as incomparable")
            print(
                f"    clock={','.join(row['clock_modes'])}  "
                + (
                    f"charged-sum/union={row['charged_seconds']:.2f}/"
                    + (
                        "unavailable  "
                        if row["llm_charged_union_seconds"] is None
                        else f"{row['llm_charged_union_seconds']:.2f}s  "
                    )
                    if "token_charged" in row["clock_modes"]
                    else ""
                )
                + f"wall-sum/union={row['llm_wall_seconds']:.2f}/"
                f"{row['llm_wall_union_seconds']:.2f}s  "
                f"inference-concurrency=max {row['llm_max_in_flight']}, "
                f"overlapped {row['llm_overlapped_round_trips']}/{row['llm_round_trips']}  "
                f"cache-clamps={row['cached_input_clamps']}  "
                f"accounting-mismatches={row['charge_accounting_mismatches']}"
            )
        if row.get("scenario_manifest_digests"):
            print(
                "    scenario-manifest="
                + ",".join(digest[:12] for digest in row["scenario_manifest_digests"])
                + ("  WARNING: mixed manifests" if row["mixed_scenario_manifests"] else "")
            )
    overall = summary.get("overall")
    if overall is not None:
        print(
            f"  {'overall':<14} {overall:6.1%}   "
            f"(equal-weight over {', '.join(_CORE_CAPABILITIES)})"
        )
        return
    print("  overall        withheld — these results are not one Gaia2 headline:")
    for reason in summary.get("headline_withheld", ()):
        print(f"                 - {reason}")
    exploratory = summary.get("exploratory_mean")
    if exploratory is not None:
        scored = [
            c for c in _CORE_CAPABILITIES if c in configs and configs[c]["pass_at_1"] is not None
        ]
        print(
            f"                 exploratory mean over {len(scored)}/{len(_CORE_CAPABILITIES)} "
            f"capabilities ({', '.join(scored)}) = {exploratory:.1%} — not a reportable result"
        )


# -- run (lazy ARE imports) -----------------------------------------------------------------------


def _run_capability(args: argparse.Namespace) -> list[dict[str, Any]]:
    from are.simulation.benchmark.scenario_loader import setup_scenarios_iterator

    sweep_manifest: SweepManifest | None = getattr(args, "sweep_manifest", None)
    if sweep_manifest is not None:
        _verify_counterpart_pairing(args, sweep_manifest)
    config_dir = os.path.join(_arm_root(args.output_dir, args.arm), "standard", args.capability)
    os.makedirs(config_dir, exist_ok=True)
    print(f"arm: {args.arm}  ->  {config_dir}")

    if args.judge_model:
        # Said once at the top of the sweep as well as per record: an operator watching the run
        # should not have to open output.jsonl to learn the scores are being produced under a
        # patched ARE.
        print(f"judge verdict parse: {_verdict_parse(args)}")

    # Armed once for the sweep, before the first scenario is loaded, and refuses a scored sweep it
    # cannot record: the window on this closes with the run rather than slipping.
    record_judge = _require_judge_recording(args)
    if args.judge_model:
        print(f"judge-response recording: {'on' if record_judge else 'OFF (--no-judge-recording)'}")

    def scenarios_for_run() -> Any:
        revision = getattr(args, "hf_revision", None)
        limit = None if sweep_manifest is not None else args.limit
        if revision is None:
            loaded = iter(
                setup_scenarios_iterator(
                    dataset_path=None,
                    dataset_config=args.capability,
                    dataset_split=args.split,
                    hf=args.hf_dataset,
                    hf_revision=None,
                    load_completed_events=False,
                    limit=limit,
                )
            )
        else:
            loaded = iter(
                _pinned_hf_scenarios(
                    dataset=args.hf_dataset,
                    capability=args.capability,
                    split=args.split,
                    revision=revision,
                    limit=limit,
                )
            )
        if sweep_manifest is not None:
            return _select_manifest_scenarios(list(loaded), sweep_manifest, args.capability)
        try:
            first = next(loaded)
        except StopIteration as exc:
            raise RuntimeError(
                f"Gaia2 loader returned zero scenarios for capability={args.capability!r}, "
                f"split={args.split!r}; refusing to emit a clean 0/0 sweep"
            ) from exc
        return chain((first,), loaded)

    # Probe before opening either artifact in truncate mode. A cache/revision miss in ARE's loader
    # can otherwise replace a paid capability with empty files and print a deceptively clean 0/0.
    first_run_scenarios = scenarios_for_run()

    records: list[dict[str, Any]] = []
    # Stream each record to output.jsonl as it's produced (and flush): a long sweep spends real
    # model tokens, so an abort partway through (a bad scenario, Ctrl-C) must leave a valid partial
    # file of the scenarios already completed rather than discarding all of them — records were
    # previously buffered in memory and written only once at the very end.
    # One per-call model record file for the whole capability, accumulated across scenarios and
    # runs and separated by each row's scenario_id: the charge model is fitted and validated
    # against these rows, and a per-scenario file would make that a directory walk instead of a
    # read. Written for unscored sweeps too — it costs no tokens. Truncated once here, like
    # output.jsonl above: re-running a capability into this directory replaces every other artifact
    # in it, and rows left over from the previous sweep would carry the same scenario_id and
    # run_number as the new ones, so the fit would count them a second time.
    llm_calls = LLMCallWriter(os.path.join(config_dir, "llm_calls.jsonl"), reset=True)
    with (
        open(os.path.join(config_dir, "output.jsonl"), "w", encoding="utf-8") as out,
        llm_calls,
    ):
        for run_number in range(args.num_runs):
            # Re-create the iterator each run so every run gets fresh, un-run scenario objects (a
            # scenario is stateful once played); HF caches locally, so re-iteration is cheap.
            scenarios = first_run_scenarios if run_number == 0 else scenarios_for_run()
            for scenario, _events in scenarios:
                # Disambiguate this run's trace file: ARE's get_run_id keys the hf trace filename on
                # scenario.run_number, so without this every run of a scenario writes
                # {scenario_id}.json and later runs silently overwrite earlier ones (leaving their
                # trace_ids pointing at the wrong trace at upload).
                scenario.run_number = run_number
                rec = _run_one_scenario(
                    scenario,
                    run_number,
                    args,
                    config_dir,
                    record_judge=record_judge,
                    llm_calls=llm_calls,
                )
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
    *,
    record_judge: bool = False,
    llm_calls: LLMCallWriter | None = None,
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

    charge, profile_charge, charge_model = _resolved_charge(args)
    profile_name = getattr(args.model_profile, "name", None)
    # A frozen-profile wall-clock run is the robustness counterpart of that operating point. Keep
    # the identity/digest even though it is not applied; an explicitly unfrozen dev config has no
    # such identity and leaves both absent.
    charge_identity = profile_charge.identity if profile_charge is not None else None
    charge_digest = charge_model.digest if profile_charge is not None else None

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
            # Deliberately not forced on for the react arm, even though its runner has a harder
            # initialization requirement (see `run_react_on_scenario`, which meets it itself):
            # without a judge this flag decides whether turns 2..n are delivered at all, so an arm
            # that set it while the other did not would be running a longer scenario, and the two
            # pass@1 columns would not be measuring the same work.
            if args.init_turns:
                initialize_turns(scenario)
        if args.arm == "react":
            from examples.gaia2.react_driver import run_react_on_scenario

            result = run_react_on_scenario(
                scenario,
                args.model_profile,
                writer=llm_calls,
                run_number=run_number,
                record_judge=record_judge,
                verdict_parse=_verdict_parse(args),
                log_fn=lambda msg: print(msg, flush=True),
                charge=charge,
                generation_free=getattr(args, "generation_free", False),
                max_wall_seconds=args.max_wall_seconds,
                charge_model_identity=charge_identity,
                charge_model_digest=charge_digest,
            )
        else:
            result = run_scenario(
                scenario,
                config=args.config,
                verbose=args.verbose,
                max_wall_seconds=args.max_wall_seconds,
                read_stdin=False,
                record_judge=record_judge,
                verdict_parse=_verdict_parse(args),
                llm_calls=llm_calls,
                scenario_id=scenario.scenario_id,
                run_number=run_number,
                charge=charge,
                generation_free=getattr(args, "generation_free", False),
                charge_model_identity=charge_identity,
                charge_model_digest=charge_digest,
            )
    except Exception as e:  # this scenario's judge/preprocess failed — record it, keep sweeping
        return _jsonl_record(
            scenario_id=scenario.scenario_id,
            run_number=run_number,
            success=None,
            rationale=None,
            exception=e,
            trace_id=None,
            charge_model_identity=charge_identity,
            charge_model_digest=charge_digest,
            model_profile=str(profile_name) if profile_name is not None else None,
            cached_input_clamps=(charge.cached_input_clamps if charge is not None else None),
            # The run never returned its independently observed count. Do not invent agreement or
            # disagreement: the row remains readable and explicitly lacks the second measurement.
            raw_cached_input_anomalies=None,
            charge_accounting_consistent=None,
            clock_mode=resolve_clock_mode(charge, getattr(args, "generation_free", False)),
            max_wall_seconds=args.max_wall_seconds,
            scenario_manifest_digest=getattr(getattr(args, "sweep_manifest", None), "digest", None),
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
            agent_id=args.arm,
            validation_decision=decision,
            validation_rationale=result.outcome.rationale,
            run_duration=result.duration,
            output_dir=config_dir,
            trace_dump_format="hf",
            scenario_exception=result.exception,
        )

    recording_path = _write_judge_recording(
        config_dir, result.judge_recording, scenario.scenario_id, run_number
    )
    if record_judge and result.judge_recording is not None and not result.judge_recording.events:
        # A scored run that judged nothing is the shape the recording exists to catch: the pipeline
        # looks healthy, the file is written, and there is nothing in it to re-score later.
        print(f"  {scenario.scenario_id}: warning — scored run recorded no judged events")

    return _jsonl_record(
        scenario_id=scenario.scenario_id,
        run_number=run_number,
        success=result.outcome.success,
        rationale=result.outcome.rationale,
        exception=result.exception,
        trace_id=trace_id,
        awaiting_input=result.awaiting_input,
        write_counts=result.write_counts,
        judge_recording_path=recording_path,
        timeline_expired=result.timeline_expired,
        harness_truncation=result.harness_truncation,
        terminal_cause=result.terminal_cause,
        max_wall_seconds=args.max_wall_seconds,
        verdict_parse=_verdict_parse(args),
        charged_seconds=result.charged_seconds,
        model_profile=str(profile_name) if profile_name is not None else None,
        charge_model_identity=result.charge_model_identity,
        charge_model_digest=result.charge_model_digest,
        cached_input_clamps=result.cached_input_clamps,
        raw_cached_input_anomalies=result.raw_cached_input_anomalies,
        charge_accounting_consistent=result.charge_accounting_consistent,
        clock_mode=result.clock_mode,
        inference_charge_policy=result.inference_charge_policy,
        llm_wall_seconds=result.llm_wall_seconds,
        llm_wall_union_seconds=result.llm_wall_union_seconds,
        llm_charged_union_seconds=result.llm_charged_union_seconds,
        llm_round_trips=int(getattr(result, "llm_round_trips", 0)),
        llm_max_in_flight=int(getattr(result, "llm_max_in_flight", 0)),
        llm_overlapped_round_trips=int(getattr(result, "llm_overlapped_round_trips", 0)),
        scenario_manifest_digest=getattr(getattr(args, "sweep_manifest", None), "digest", None),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="batch",
        description="Run one arm over a Gaia2 capability, emit HF traces + output.jsonl, report.",
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
        "--hf-revision",
        metavar="REVISION",
        help="Pinned Hugging Face dataset revision. Set automatically by --scenario-manifest.",
    )
    parser.add_argument(
        "--scenario-manifest",
        type=Path,
        metavar="JSON",
        help=(
            "Exact ordered scenario IDs for a comparable sweep. Pins dataset/revision/split, "
            "records its digest on every row, and excludes --limit."
        ),
    )
    parser.add_argument(
        "--arm",
        default="sora",
        choices=_ARMS,
        help=(
            "Which agent runs the sweep: S-ORA's decision cycle (default) or ARE's own published "
            "ReAct agent as the baseline. `react` requires --profile and ignores --config; its "
            "artifacts land under DIR/react/standard/<capability>/ so both arms can share one "
            "--output-dir without overwriting each other."
        ),
    )
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        metavar="AGENT_YAML",
        help=f"Agent config for --arm sora (default: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--profile",
        metavar="NAME",
        help=(
            "Model profile for --arm react (from --profiles-path). It is the only thing that "
            "selects the model, the reasoning setting and the output cap on that arm — the same "
            "operating point the S-ORA arm is configured at, which is what makes the two "
            "comparable."
        ),
    )
    parser.add_argument(
        "--charge-profile",
        metavar="NAME",
        help=(
            "Explicit frozen charge profile for --arm sora when automatic config matching is "
            "ambiguous. The selected profile must still match --config exactly. Cannot be used "
            "with --wall-clock or --allow-unfrozen-config."
        ),
    )
    parser.add_argument(
        "--allow-unfrozen-config",
        action="store_true",
        help=(
            "Allow an S-ORA config outside the frozen profile set for local/development runs. "
            "This necessarily uses wall time because no frozen coefficients exist for the config."
        ),
    )
    parser.add_argument(
        "--profiles-path",
        type=Path,
        default=Path(_DEFAULT_PROFILES),
        metavar="JSON",
        help=f"Profile definitions (default: {_DEFAULT_PROFILES}).",
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
            "S-ORA/claude-opus-4-8). It must name the model the run actually uses — agent.yaml's "
            "llm.model, or the profile's model on --arm react — or the run is refused. Omit to "
            "label with that model id itself."
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
        "--no-judge-recording",
        action="store_true",
        help=(
            "Do NOT record the judge's raw responses. A scored sweep records them by default and "
            "refuses to start if it cannot: ARE keeps only a boolean per judged event, so a run "
            "swept without them can never be re-scored afterwards, at any price."
        ),
    )
    parser.add_argument(
        "--init-turns",
        action="store_true",
        help=(
            "Deliver every turn of a multi-turn scenario without a judge (runs stay unscored). "
            "Without it, an unscored multi-turn scenario stops after turn 1. Excludes "
            "--judge-model. Means the same thing on both arms, so a paired sweep compares equal "
            "work."
        ),
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Per-scenario wall-clock safety cap for both arms. Defaults to 1200, or to 3600 under "
            "--generation-free, where a frozen clock makes real per-scenario time the scenario "
            "timeline plus the whole of generation rather than the timeline alone. The value used "
            "is announced and recorded on every row."
        ),
    )
    parser.add_argument(
        "--wall-clock",
        action="store_true",
        help=(
            "Use elapsed wall time instead of the frozen token-charged clock. Intended only for "
            "a separately labeled robustness sweep; results are not timing-comparable."
        ),
    )
    parser.add_argument(
        "--generation-free",
        action="store_true",
        help=(
            "Freeze the scenario across every model call and resume it by zero, so generation "
            "costs the environment nothing. This is the legacy ARE in-process convention — that "
            "pause/resume path reads a completion_duration that no shipped engine writes — so it "
            "needs no calibration and has no coefficient that can drift. Excludes --wall-clock. "
            "Recorded as clock_mode=generation_free, and not timing-comparable to token-charged "
            "runs."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Stream each scenario's trajectory.")
    return parser.parse_args(argv)


# A frozen clock removes generation from the scenario's own budget but not from the operator's:
# real per-scenario time becomes the scenario timeline *plus* the whole of generation, where a
# wall-clock run is bounded by the timeline alone. Measured S-ORA Time scenarios project past 1200s
# under that arithmetic while every ReAct one stays under it, so leaving the cap where it is would
# truncate one arm and not the other — which is not a shared condition, and so not a comparison.


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.wall_clock and args.generation_free:
        raise SystemExit(
            "--wall-clock and --generation-free are different clock conventions; pick one"
        )
    if args.max_wall_seconds is None:
        args.max_wall_seconds = resolve_max_wall_seconds(None, args.generation_free)
        print(
            f"watchdog: --max-wall-seconds not given; using {args.max_wall_seconds:g}s "
            f"({'generation-free' if args.generation_free else 'default'})"
        )

    if args.report_only:
        # Loaded here rather than inherited from the sweep args, which are only assembled on the
        # path that actually runs scenarios: without it a report of a finished sweep could never
        # verify its own coverage, and would withhold the headline it was asked to produce.
        _print_report(
            aggregate(
                _arm_root(args.report_only, args.arm),
                manifest=(
                    _load_sweep_manifest(args.scenario_manifest) if args.scenario_manifest else None
                ),
            )
        )
        return

    if not args.capability:
        raise SystemExit("--capability is required unless --report-only is given")

    args.sweep_manifest = None
    if args.scenario_manifest is not None:
        if args.limit is not None:
            raise SystemExit("--scenario-manifest and --limit are mutually exclusive")
        manifest = _load_sweep_manifest(args.scenario_manifest)
        if args.hf_dataset != manifest.dataset:
            raise SystemExit(
                f"--hf-dataset {args.hf_dataset!r} does not match sweep manifest dataset "
                f"{manifest.dataset!r}"
            )
        if args.split != manifest.split:
            raise SystemExit(
                f"--split {args.split!r} does not match sweep manifest split {manifest.split!r}"
            )
        if args.hf_revision is not None and args.hf_revision != manifest.revision:
            raise SystemExit(
                f"--hf-revision {args.hf_revision!r} does not match sweep manifest revision "
                f"{manifest.revision!r}"
            )
        if not manifest.scenario_ids(args.capability):
            raise SystemExit(
                f"sweep manifest {manifest.name!r} has no {args.capability!r} scenarios"
            )
        args.hf_revision = manifest.revision
        args.sweep_manifest = manifest
        print(
            f"scenario manifest {manifest.name}: "
            f"{len(manifest.scenario_ids(args.capability))} {args.capability} cases, "
            f"digest={manifest.digest}"
        )

    if args.judge_model and args.init_turns:
        # Only the first of the two takes effect (ARE's initialize_turns is idempotent), leaving the
        # judge as the turn gate — the opposite of what --init-turns asks for. Refuse, don't ignore.
        raise SystemExit("--init-turns and --judge-model are mutually exclusive")

    # Before a single token is spent: the trace label has to agree with the model the run actually
    # selects, so an exported trace can't attribute the run to a model that never ran it. Which
    # thing selects it differs by arm — agent.yaml's llm.model for S-ORA, the profile for ReAct —
    # and nothing downstream can tell a mislabeled trace from a correct one.
    args.model_profile = None
    from examples.gaia2.evaluation.core import ChargeModelSheet, load_profiles

    profiles = load_profiles(args.profiles_path)
    args.charge_model = ChargeModelSheet.load(
        Path(__file__).resolve().parent / "evaluation" / "charge_model.json"
    )
    if args.arm == "react":
        if args.charge_profile:
            raise SystemExit("--charge-profile applies only to --arm sora")
        if args.allow_unfrozen_config:
            raise SystemExit("--allow-unfrozen-config applies only to --arm sora")
        if not args.profile:
            raise SystemExit("--arm react requires --profile (it selects the model for that arm)")
        if args.profile not in profiles:
            raise SystemExit(f"unknown profile {args.profile!r}; have {sorted(profiles)}")
        args.model_profile = profiles[args.profile]
        args.charge_model.for_profile(args.model_profile)
        _check_operating_point(args.model_profile, args.config)
        args.model = _resolve_react_label(args.model, args.model_profile)
        print(
            f"profile {args.model_profile.name} -> {args.model_profile.model} "
            f"at {args.model_profile.endpoint}"
        )
    else:
        if args.profile:
            raise SystemExit(
                "--profile applies to --arm react; --arm sora derives its frozen charge profile "
                "from --config"
            )
        if args.charge_profile and args.wall_clock:
            raise SystemExit("--charge-profile and --wall-clock are mutually exclusive")
        if args.charge_profile and args.generation_free:
            raise SystemExit("--charge-profile and --generation-free are mutually exclusive")
        if args.charge_profile and args.allow_unfrozen_config:
            raise SystemExit("--charge-profile and --allow-unfrozen-config are mutually exclusive")
        if args.allow_unfrozen_config:
            if not os.path.isfile(args.config):
                raise SystemExit(f"--config does not exist: {args.config}")
            # Checked here and not beside the --wall-clock/--generation-free conflict above,
            # because this is the branch that *creates* the conflict: it selects the wall clock
            # itself, after that check has already run. Without this the pair is accepted, the
            # run announces wall-clock development mode, and then freezes anyway.
            if args.generation_free:
                raise SystemExit(
                    "--allow-unfrozen-config selects the wall clock; it cannot be combined with "
                    "--generation-free"
                )
            args.wall_clock = True
            print("unfrozen config: using wall clock (development mode; not sweep-comparable)")
        elif args.charge_profile:
            if args.charge_profile not in profiles:
                raise SystemExit(
                    f"unknown charge profile {args.charge_profile!r}; have {sorted(profiles)}"
                )
            args.model_profile = profiles[args.charge_profile]
            if not os.path.isfile(args.config):
                raise SystemExit(f"--config does not exist: {args.config}")
            _check_operating_point(args.model_profile, args.config)
            print(
                f"explicit charge profile {args.model_profile.name} -> "
                f"{args.model_profile.model} at {args.model_profile.endpoint}"
            )
        else:
            # Clock policy is independent of operating-point validation. A wall-clock robustness
            # run still identifies the frozen profile it is a robustness check of.
            args.model_profile = _profile_for_config(profiles, args.config)
        if args.model_profile is not None:
            args.charge_model.for_profile(args.model_profile)
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
    _print_report(
        aggregate(
            _arm_root(args.output_dir, args.arm),
            manifest=getattr(args, "sweep_manifest", None),
        )
    )


if __name__ == "__main__":
    main()
