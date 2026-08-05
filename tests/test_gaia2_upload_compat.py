"""The batch harness's artifacts feed ARE's real submission uploader unchanged.

Proves the ``output.jsonl`` that ``examples/gaia2/batch.py`` writes is byte-compatible with the
standalone ``gaia2_upload_script``: its ``reconstruct_results_from_output_dir`` parses our rows into
the ``MultiScenarioValidationResult`` structure the HuggingFace upload consumes — with the
pass/fail/exception mapping and per-``(scenario_id, run_number)`` keying intact, and each row's
``trace_id`` resolving to a trace file on disk. This locks the run-doubles-as-a-leaderboard-
submission guarantee against the *installed* uploader, so a signature/shape drift breaks here
instead of at submit time.

Opt-in (``-m integration``, needs ``uv sync --all-extras --group are``); no model tokens and no
network — only the uploader's local parse half runs, never ``upload_consolidated_results_to_hf``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("are.simulation.benchmark.gaia2_upload_script")

from are.simulation.benchmark.gaia2_upload_script import (  # noqa: E402
    reconstruct_results_from_output_dir,
)
from examples.gaia2.batch import _jsonl_record, _write_jsonl  # noqa: E402

pytestmark = pytest.mark.integration


def test_output_jsonl_reconstructs_into_are_upload_structure(tmp_path: Path) -> None:
    # The exact tree batch.py emits: {root}/standard/{config}/output.jsonl, traces under .../hf/.
    config_dir = tmp_path / "standard" / "execution"
    hf_dir = config_dir / "hf"
    hf_dir.mkdir(parents=True)

    # One real trace file, referenced by the passing row's trace_id as an absolute path (what
    # batch.py now emits) — the uploader reads it back via a bare os.path.exists(trace_id).
    trace_path = hf_dir / "s-pass.json"
    trace_path.write_text('{"trace": "ok"}', encoding="utf-8")

    # Build the rows with our own writer (dogfood the exact record shape), covering every status.
    records = [
        _jsonl_record(
            scenario_id="s-pass",
            run_number=0,
            success=True,
            rationale="matched oracle graph",
            exception=None,
            trace_id=str(trace_path),
        ),
        _jsonl_record(
            scenario_id="s-fail",
            run_number=0,
            success=False,
            rationale="wrong tool",
            exception=None,
            trace_id=None,
        ),
        _jsonl_record(
            scenario_id="s-err",
            run_number=0,
            success=None,
            rationale=None,
            exception=ValueError("boom"),
            trace_id=None,
        ),
        # same scenario, a second run — the (scenario_id, run_number) key must stay distinct.
        _jsonl_record(
            scenario_id="s-pass",
            run_number=1,
            success=True,
            rationale=None,
            exception=None,
            trace_id=None,
        ),
    ]
    _write_jsonl(str(config_dir / "output.jsonl"), records)

    results = reconstruct_results_from_output_dir(str(tmp_path), "S-ORA/test-model")

    # Keyed by (phase, config, a2a_prop, has_tool_aug, has_env_events) for the `standard` phase.
    key = ("standard", "execution", 0.0, False, False)
    assert key in results
    scenario_results = results[key].scenario_results

    # Per-(scenario_id, run_number) keying; s-pass appears twice (run 0 and run 1), un-collapsed.
    assert set(scenario_results) == {
        ("s-pass", 0),
        ("s-fail", 0),
        ("s-err", 0),
        ("s-pass", 1),
    }

    passed = scenario_results[("s-pass", 0)]
    assert passed.success is True
    assert passed.rationale == "matched oracle graph"
    assert passed.export_path == str(trace_path)  # trace_id resolved to a real file on disk

    failed = scenario_results[("s-fail", 0)]
    assert failed.success is False
    assert failed.rationale == "wrong tool"

    # An errored run reconstructs as success=None with an exception — not a false miss.
    errored = scenario_results[("s-err", 0)]
    assert errored.success is None
    assert errored.exception is not None
