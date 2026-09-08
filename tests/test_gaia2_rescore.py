"""``examples/gaia2/rescore.py`` — the offline re-score of a stored run, and the mechanical gate
that keeps a scored sweep from being launched without recording.

No ARE, no model tokens: a recording is a JSON file, and re-scoring it is pure arithmetic. The gate
is here rather than left to an operator's discipline because the failure it guards against is
deprioritization, not difficulty — an unrecorded sweep cannot be re-scored afterwards at any price.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest
from examples.gaia2 import batch
from examples.gaia2.rescore import main, summarize

from sora.adapters.are_judge import (
    CheckerCall,
    JudgedEvent,
    JudgeRecording,
    recording_from_dict,
)


def _tone(response: str) -> CheckerCall:
    return CheckerCall("tone_checker", "[[True]]", "[[False]]", (response,), {})


def _recording(*, scenario_id: str = "s1") -> JudgeRecording:
    """One fast-path event and one the stock parse discards — the shape a scenario ending in a
    paraphrased message to the user produces."""
    return JudgeRecording(
        scenario_id=scenario_id,
        run_number=0,
        verdict_parse="stock",
        events=(
            JudgedEvent(
                tool_name="CalendarApp__add_event",
                agent_args={"title": "Film Production Day"},
                oracle_args={"title": "Film Production Day"},
                equality=True,
                verdict=True,
            ),
            JudgedEvent(
                tool_name="EmailClientApp__send_email",
                agent_args={"content": "Hi Åke, ..."},
                oracle_args={"content": "Dear Åke, ..."},
                equality=False,
                verdict=False,
                checkers=(_tone("The tone is fine. Evaluation: [[true]]"),),
            ),
        ),
    )


def _write(path: Path, recording: JudgeRecording) -> Path:
    path.write_text(json.dumps(recording.to_dict()), encoding="utf-8")
    return path


# -- summarize -----------------------------------------------------------------------------------


def test_summarize_scores_a_stored_run_under_both_parses(tmp_path: Path) -> None:
    _write(tmp_path / "s1.json", _recording())
    summary = summarize([tmp_path])
    assert summary["recordings"][0]["scores"] == {"stock": 0.5, "case-insensitive": 1.0}
    assert summary["recordings"][0]["divergent_events"] == [1]
    assert summary["events"] == 2


def test_summarize_reports_the_exit_criterion_separately_from_mere_divergence(
    tmp_path: Path,
) -> None:
    # Divergence that only ever lands on the equality fast path would mean the patched checker path
    # was never exercised — recorded faithfully, and still unable to tell a scorer artefact from an
    # agent failure. So the criterion asks about the events equality MISSED, not about any change.
    _write(tmp_path / "s1.json", _recording())
    assert summarize([tmp_path])["divergence_off_equality_fast_path"] is True

    agreeing = JudgeRecording(
        events=(JudgedEvent("t", {}, {}, equality=True, verdict=True),), verdict_parse="stock"
    )
    _write(tmp_path / "s2.json", agreeing)
    only_agreeing = summarize([tmp_path / "s2.json"])
    assert only_agreeing["divergence_off_equality_fast_path"] is False


def test_summarize_flags_a_recording_whose_live_parse_does_not_reproduce(tmp_path: Path) -> None:
    # The re-scorer re-implements ARE's rule; on the parse the run actually used it must agree with
    # what ARE returned. A disagreement is a defect here, and has to be visible rather than folded
    # into the score it corrupts.
    wrong = JudgeRecording(
        verdict_parse="stock",
        events=(
            JudgedEvent(
                "t", {}, {}, equality=False, verdict=True, checkers=(_tone("Evaluation: [[true]]"),)
            ),
        ),
    )
    _write(tmp_path / "s1.json", wrong)
    summary = summarize([tmp_path])
    assert summary["recordings"][0]["disagreements_with_recorded"] == [0]
    assert summary["unreproduced_recordings"] == 1


def test_summarize_reads_a_directory_tree_of_recordings(tmp_path: Path) -> None:
    nested = tmp_path / "standard" / "time" / "judge_responses"
    nested.mkdir(parents=True)
    _write(nested / "a.json", _recording(scenario_id="a"))
    _write(nested / "b.json", _recording(scenario_id="b"))
    summary = summarize([tmp_path])
    assert [r["scenario_id"] for r in summary["recordings"]] == ["a", "b"]
    assert summary["events"] == 4


def test_summarize_skips_json_that_is_not_a_recording(tmp_path: Path) -> None:
    # An artifact tree also holds output.jsonl and exported traces; walking it must not choke.
    (tmp_path / "trace.json").write_text('{"world": []}', encoding="utf-8")
    _write(tmp_path / "s1.json", _recording())
    assert len(summarize([tmp_path])["recordings"]) == 1


# -- the CLI --------------------------------------------------------------------------------------


def test_cli_prints_both_scores(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tmp_path / "s1.json", _recording())
    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "stock" in out and "case-insensitive" in out


def test_cli_require_divergence_passes_when_the_patched_path_was_exercised(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "s1.json", _recording())
    assert main([str(tmp_path), "--require-divergence"]) == 0


def test_cli_require_divergence_fails_on_a_recording_that_never_diverges(tmp_path: Path) -> None:
    _write(
        tmp_path / "s1.json",
        JudgeRecording(
            events=(JudgedEvent("t", {}, {}, equality=True, verdict=True),), verdict_parse="stock"
        ),
    )
    assert main([str(tmp_path), "--require-divergence"]) == 1


def test_cli_reports_nothing_to_read_rather_than_claiming_agreement(tmp_path: Path) -> None:
    # An empty tree re-scores to "no divergence" arithmetically, which is exactly the reading the
    # exit criterion must not accept.
    assert main([str(tmp_path), "--require-divergence"]) == 1


def test_cli_writes_a_json_summary(tmp_path: Path) -> None:
    _write(tmp_path / "s1.json", _recording())
    out = tmp_path / "summary.json"
    assert main([str(tmp_path), "--output", str(out)]) == 0
    assert json.loads(out.read_text())["events"] == 2


# -- the batch gate -------------------------------------------------------------------------------


def _args(**overrides: Any) -> Namespace:
    base: dict[str, Any] = {
        "judge_model": "gpt-5.5",
        "no_judge_recording": False,
        "strict_verdict_case": False,
    }
    base.update(overrides)
    return Namespace(**base)


def test_batch_refuses_a_scored_sweep_it_could_never_re_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ARE absent (or the patch points moved): arming reports False. Spending a sweep's tokens on a
    # run that cannot be re-scored is the one outcome with no recovery, so refuse it up front.
    monkeypatch.setattr(batch, "_arm_judge_recording", lambda: False)
    with pytest.raises(SystemExit, match="judge-response recording"):
        batch._require_judge_recording(_args())


def test_batch_allows_an_unscored_sweep_without_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nothing judged, so nothing to record and nothing to re-score later.
    monkeypatch.setattr(batch, "_arm_judge_recording", lambda: False)
    assert batch._require_judge_recording(_args(judge_model=None)) is False


def test_batch_honours_an_explicit_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(batch, "_arm_judge_recording", lambda: False)
    assert batch._require_judge_recording(_args(no_judge_recording=True)) is False


def test_batch_arms_recording_for_a_scored_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(batch, "_arm_judge_recording", lambda: True)
    assert batch._require_judge_recording(_args()) is True


def test_batch_record_names_a_written_recording(tmp_path: Path) -> None:
    path = tmp_path / "s1.json"
    record = batch._jsonl_record(
        scenario_id="s1",
        run_number=0,
        success=True,
        rationale=None,
        exception=None,
        trace_id="t",
        verdict_parse="case-insensitive",
        judge_recording_path=str(path),
    )
    assert record["metadata"]["judge_recording"] == str(path)


def test_batch_record_omits_the_pointer_when_nothing_was_recorded() -> None:
    record = batch._jsonl_record(
        scenario_id="s1",
        run_number=0,
        success=None,
        rationale=None,
        exception=None,
        trace_id=None,
    )
    assert "judge_recording" not in record["metadata"]


def test_writing_a_recording_round_trips(tmp_path: Path) -> None:
    recording = _recording()
    path = batch._write_judge_recording(str(tmp_path), recording, "s1", 0)
    assert path is not None
    assert recording_from_dict(json.loads(Path(path).read_text())) == recording


def test_writing_no_recording_is_not_a_file(tmp_path: Path) -> None:
    assert batch._write_judge_recording(str(tmp_path), None, "s1", 0) is None
