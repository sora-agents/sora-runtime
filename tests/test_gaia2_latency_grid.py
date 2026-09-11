"""The designed latency grid: shape, prompt determinism, and what reaches the two files.

Nothing here touches a network. The client is faked at the ``chat.completions.create`` seam, which
is the whole surface ``LatencyGrid`` uses, so the streaming path, the usage join and the failure
path are all exercised without a key.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from examples.gaia2.evaluation.core import load_profiles
from examples.gaia2.latency_grid import (
    CALIBRATION_CALLS,
    EVAL_ROOT,
    INPUT_LEVELS,
    OUTPUT_LEVELS,
    GridCell,
    GridSpec,
    LatencyGrid,
    _Calibration,
    _main,
    build_prompt,
    completed_call_ids,
    format_range_check,
    range_check,
)
from examples.gaia2.llm_calls import LLMCallWriter


class _FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def __aiter__(self) -> Any:
        for chunk in self._chunks:
            yield chunk


class _FakeClient:
    """Stands in for ``AsyncOpenAI``. ``chat`` and ``completions`` resolve back to self, which is
    all the attribute chain the grid walks."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    @property
    def chat(self) -> _FakeClient:
        return self

    @property
    def completions(self) -> _FakeClient:
        return self

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        nxt = self.responses.pop(0) if self.responses else _stream(prompt=10, completion=10)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _stream(
    *,
    prompt: int,
    completion: int,
    cached: int | None = None,
    reasoning: int | None = None,
    finish_reason: str = "length",
    model: str = "gpt-5.4-2026-03-05",
) -> _FakeStream:
    content = SimpleNamespace(
        model=model,
        usage=None,
        choices=[
            SimpleNamespace(finish_reason=finish_reason, delta=SimpleNamespace(content="1\n2\n3\n"))
        ],
    )
    usage = SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=(
                SimpleNamespace(cached_tokens=cached) if cached is not None else None
            ),
            completion_tokens_details=(
                SimpleNamespace(reasoning_tokens=reasoning) if reasoning is not None else None
            ),
        ),
        choices=[],
    )
    return _FakeStream([content, usage])


@pytest.fixture
def profile() -> Any:
    return load_profiles(EVAL_ROOT / "profiles.json")["gpt-5.4-high-paper"]


@pytest.fixture
def grid(tmp_path: Path, profile: Any) -> tuple[LatencyGrid, _FakeClient, LLMCallWriter]:
    client = _FakeClient([])
    writer = LLMCallWriter(tmp_path / "grid.jsonl")
    spec = GridSpec(blocks=1)
    runner = LatencyGrid(
        profile,
        spec=spec,
        writer=writer,
        manifest_path=tmp_path / "grid.manifest",
        client=client,
    )
    # Pre-frozen: `run()` would otherwise open with the calibration calls, and every count in
    # every other test would be a count of those too. The phase has its own tests below.
    runner.calibration.freeze()
    return runner, client, writer


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# -- the design ------------------------------------------------------------------------------


def test_every_block_holds_every_cell_exactly_once() -> None:
    """A complete randomized block, not a sample: the median of a cell is only meaningful if each
    block contributes exactly one observation of it."""
    spec = GridSpec(blocks=5)
    per_block: dict[int, Counter[str]] = {}
    for cell in spec.cells():
        per_block.setdefault(cell.block, Counter())[cell.cell_id] += 1
    assert sorted(per_block) == [0, 1, 2, 3, 4]
    first = per_block[0]
    assert set(first.values()) == {1}
    for block in per_block.values():
        assert block == first
    # Globally unique, because call_id is the join key between the two files: a duplicate would
    # silently attach one row's cell metadata to another row's tokens.
    call_ids = [cell.call_id() for cell in spec.cells()]
    assert len(call_ids) == len(set(call_ids))


def test_a_cached_prefix_is_warmed_then_measured_without_interruption() -> None:
    """A scattered cached arm would fit R_cache against prefills that had gone cold — the provider
    TTL is minutes, and a full block takes longer than that."""
    cells = GridSpec(blocks=2).cells()
    for prefix in GridSpec().cached_prefix_levels:
        for block in (0, 1):
            positions = [
                index
                for index, cell in enumerate(cells)
                if cell.cached_prefix_tokens == prefix and cell.block == block
            ]
            assert positions == list(range(positions[0], positions[0] + len(positions)))
            assert cells[positions[0]].warmup
            assert not any(cells[i].warmup for i in positions[1:])


def test_the_cached_arm_varies_the_cached_token_count() -> None:
    """A single fixed prefix size is identifiable only as an intercept shift, which cannot be told
    apart from any systematic offset the cached condition carries."""
    sizes = {c.cached_prefix_tokens for c in GridSpec().cells() if c.cached}
    assert len(sizes) >= 4


def test_cell_order_is_shuffled_but_reproducible_from_the_seed() -> None:
    order = [c.call_id() for c in GridSpec(blocks=2, seed=7).cells()]
    assert order == [c.call_id() for c in GridSpec(blocks=2, seed=7).cells()]
    assert order != [c.call_id() for c in GridSpec(blocks=2, seed=8).cells()]
    first_block = [c for c in GridSpec(blocks=2, seed=7).cells() if c.block == 0]
    second_block = [c for c in GridSpec(blocks=2, seed=7).cells() if c.block == 1]
    assert [c.cell_id for c in first_block] != [c.cell_id for c in second_block]


# -- the prompts -----------------------------------------------------------------------------


def test_the_same_seed_rebuilds_the_same_prompt() -> None:
    """A published grid whose prompts cannot be regenerated is not reproducible; the seed and the
    call id are the whole record of what was sent."""
    cell = GridCell(input_tokens=4_000, output_tokens=256, block=0)
    spec = GridSpec(seed=11)
    assert build_prompt(cell, spec, _Calibration()) == build_prompt(cell, spec, _Calibration())
    other = build_prompt(cell, GridSpec(seed=12), _Calibration())
    assert other != build_prompt(cell, spec, _Calibration())


def test_two_uncached_cells_never_share_filler() -> None:
    """Uncached means uncached: repeated filler would let the provider's prefix cache serve a cell
    the fit is treating as a cold prefill."""
    spec = GridSpec()
    a = build_prompt(GridCell(input_tokens=4_000, output_tokens=64), spec, _Calibration()).text
    b = build_prompt(GridCell(input_tokens=4_000, output_tokens=256), spec, _Calibration()).text
    assert a != b


def test_a_cached_level_sends_a_byte_identical_head_and_a_unique_tail() -> None:
    spec = GridSpec()
    calibration = _Calibration()
    first = build_prompt(
        GridCell(input_tokens=17_000, output_tokens=256, cached_prefix_tokens=16_000),
        spec,
        calibration,
    ).text
    second = build_prompt(
        GridCell(input_tokens=17_000, output_tokens=4_096, cached_prefix_tokens=16_000),
        spec,
        calibration,
    ).text
    shared = 0
    while shared < min(len(first), len(second)) and first[shared] == second[shared]:
        shared += 1
    # The identical head is the cacheable part and has to dominate; the tails must still differ.
    assert shared > len(first) * 0.8
    assert first != second
    other_level = build_prompt(
        GridCell(input_tokens=5_000, output_tokens=256, cached_prefix_tokens=4_000),
        spec,
        calibration,
    ).text
    assert not other_level.startswith(first[:200])


def test_calibration_moves_toward_what_the_provider_counted() -> None:
    calibration = _Calibration()
    before = calibration.tokens_per_word
    for _ in range(20):
        calibration.observe(words=1_000, reported_tokens=3_000)
    assert calibration.tokens_per_word < before
    assert calibration.tokens_per_word == pytest.approx(3.0, abs=0.2)


# -- the request -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_only_the_output_cap_deviates_from_the_profile(
    grid: tuple[LatencyGrid, _FakeClient, LLMCallWriter],
) -> None:
    """The grid has to sit at the sweep's operating point — latency does not transfer across
    reasoning settings — except for the cap, which is the only way to force an output length."""
    runner, client, _ = grid
    client.responses = [_stream(prompt=4_000, completion=256)]
    await runner.run_cell(GridCell(input_tokens=4_000, output_tokens=256))
    (sent,) = client.calls
    assert sent["model"] == "gpt-5.4-2026-03-05"
    assert sent["reasoning_effort"] == "high"
    assert sent["max_completion_tokens"] == 256  # not the profile's 16384
    assert "temperature" not in sent  # intentionally_omitted in the profile
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_provider_routing_rides_where_the_arm_puts_it(tmp_path: Path) -> None:
    """Kimi's pinned single-provider route is part of the operating point: fitted on one route and
    swept on another, every coefficient is wrong and nothing says so."""
    kimi = load_profiles(EVAL_ROOT / "profiles.json")["kimi-k2.5-prompt"]
    client = _FakeClient([_stream(prompt=1_000, completion=64, model="moonshotai/kimi-k2.5")])
    with LLMCallWriter(tmp_path / "grid.jsonl") as writer:
        runner = LatencyGrid(
            kimi,
            spec=GridSpec(blocks=1),
            writer=writer,
            manifest_path=tmp_path / "m.jsonl",
            client=client,
        )
        await runner.run_cell(GridCell(input_tokens=1_000, output_tokens=64))
    (sent,) = client.calls
    assert sent["extra_body"]["provider"]["only"] == ["venice"]
    assert sent["extra_body"]["provider"]["allow_fallbacks"] is False
    assert sent["extra_body"]["reasoning"] == {"enabled": True}
    assert sent["extra_headers"] == {"X-OpenRouter-Metadata": "enabled"}
    assert sent["temperature"] == 0.5


# -- what lands on disk ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_measured_call_lands_in_both_files_joined_by_call_id(
    grid: tuple[LatencyGrid, _FakeClient, LLMCallWriter], tmp_path: Path
) -> None:
    runner, client, writer = grid
    client.responses = [_stream(prompt=4_100, completion=256, cached=0, reasoning=200)]
    written = await runner.run(limit=1, log=lambda _msg: None)
    assert written == 1
    (row,) = _rows(writer.path)
    (entry,) = _rows(tmp_path / "grid.manifest")
    assert row["call_id"] == entry["call_id"]
    assert row["arm"] == "grid"
    assert row["input_tokens"] == 4_100 and row["output_tokens"] == 256
    assert row["reasoning_tokens"] == 200
    assert row["cached_input_tokens"] == 0  # a measured miss, not an unreported field
    assert row["usage_captured"] is True
    assert row["seconds"] > 0
    assert entry["profile"] == "gpt-5.4-high-paper"
    assert len(entry["profile_sha256"]) == 64 and len(entry["prompt_sha256"]) == 64
    assert entry["seed"] == GridSpec.seed


@pytest.mark.asyncio
async def test_a_warmup_is_paid_for_recorded_and_kept_out_of_the_fit(
    grid: tuple[LatencyGrid, _FakeClient, LLMCallWriter], tmp_path: Path
) -> None:
    """It populates the cache rather than measuring a hit. Dropping it would hide a real cost;
    fitting it would price a cold prefill as a cached one."""
    runner, client, writer = grid
    client.responses = [_stream(prompt=17_000, completion=256, cached=0)]
    await runner.run_cell(
        GridCell(input_tokens=17_000, output_tokens=256, cached_prefix_tokens=16_000, warmup=True)
    )
    outcome = await runner.run_cell(
        GridCell(input_tokens=17_000, output_tokens=256, cached_prefix_tokens=16_000, warmup=True)
    )
    assert outcome.manifest["warmup"] is True
    assert outcome.manifest["fit_eligible"] is False
    assert outcome.manifest["cache_condition"] == "cached"


@pytest.mark.asyncio
async def test_a_failed_call_is_still_written_and_excluded(
    grid: tuple[LatencyGrid, _FakeClient, LLMCallWriter], tmp_path: Path
) -> None:
    """A crossing that failed was still emitted and still waited on. Never discard a paid call —
    exclude it from the fit and keep it in the audit."""
    runner, client, writer = grid
    client.responses = [RuntimeError("upstream said no")]
    outcome = await runner.run_cell(GridCell(input_tokens=1_000, output_tokens=64))
    runner.writer.write(outcome.record)
    (row,) = _rows(writer.path)
    assert row["finish_reason"] == "error:RuntimeError"
    assert row["usage_captured"] is False
    assert row["input_tokens"] is None and row["output_tokens"] is None
    assert row["seconds"] > 0
    assert outcome.manifest["error"] == "RuntimeError"
    assert outcome.manifest["fit_eligible"] is False


@pytest.mark.asyncio
async def test_a_cell_that_missed_its_target_is_kept_but_not_fitted(
    grid: tuple[LatencyGrid, _FakeClient, LLMCallWriter],
) -> None:
    """The provider stopped early — a refusal, a stop sequence, a shorter cap than asked for. The
    row is real and billed; it is just not an observation of the cell it was aimed at."""
    runner, client, _ = grid
    client.responses = [_stream(prompt=4_000, completion=12, finish_reason="stop")]
    outcome = await runner.run_cell(GridCell(input_tokens=4_000, output_tokens=4_096))
    assert outcome.manifest["output_on_target"] is False
    assert outcome.manifest["fit_eligible"] is False
    assert outcome.record.output_tokens == 12  # kept, not nulled


@pytest.mark.asyncio
async def test_calls_are_issued_one_at_a_time(
    grid: tuple[LatencyGrid, _FakeClient, LLMCallWriter],
) -> None:
    """Concurrency here would measure our own queueing and call it the provider's latency."""
    runner, client, _ = grid
    client.responses = [_stream(prompt=1_000, completion=64) for _ in range(4)]
    await runner.run(limit=4, log=lambda _msg: None)
    assert len(client.calls) == 4


# -- the frozen ratio ------------------------------------------------------------------------


def test_a_frozen_ratio_stops_moving() -> None:
    """Every word count in the run is derived from this number, so it has to stop moving before
    the first measured call — see the next two tests for what moves if it does not."""
    calibration = _Calibration()
    calibration.observe(words=1_000, reported_tokens=3_000)
    frozen = calibration.freeze().tokens_per_word
    calibration.observe(words=1_000, reported_tokens=9_000)
    assert calibration.tokens_per_word == frozen


def test_a_moving_ratio_would_shift_a_cached_head() -> None:
    """The failure the freeze exists to prevent, stated as the property that has to hold.

    A cached group is a warm-up followed by the measurements it warms. The head is byte-identical
    only if its *word count* is, and the word count comes from the ratio — so a ratio that refined
    itself on the warm-up's own reported tokens would hand the provider a different prefix from the
    one it just stored, and the arm whose only purpose is measuring cache would measure less of it.
    """
    spec = GridSpec()
    warmup = GridCell(
        input_tokens=17_000, output_tokens=256, cached_prefix_tokens=16_000, warmup=True
    )
    measured = GridCell(input_tokens=17_000, output_tokens=4_096, cached_prefix_tokens=16_000)
    calibration = _Calibration().freeze()
    head = build_prompt(warmup, spec, calibration)
    calibration.observe(words=head.words, reported_tokens=20_000)  # ignored: frozen
    assert build_prompt(measured, spec, calibration).prefix_words == head.prefix_words

    # And unfrozen, the same sequence moves it — the guard is load-bearing, not decorative.
    drifting = _Calibration()
    drifting_head = build_prompt(warmup, spec, drifting)
    drifting.observe(words=drifting_head.words, reported_tokens=20_000)
    assert build_prompt(measured, spec, drifting).prefix_words != drifting_head.prefix_words


@pytest.mark.asyncio
async def test_calibration_runs_first_is_recorded_and_is_never_fitted(
    tmp_path: Path, profile: Any
) -> None:
    client = _FakeClient([_stream(prompt=2_000, completion=16) for _ in range(CALIBRATION_CALLS)])
    client.responses.append(_stream(prompt=1_000, completion=64))
    writer = LLMCallWriter(tmp_path / "grid.jsonl")
    runner = LatencyGrid(
        profile,
        spec=GridSpec(blocks=1),
        writer=writer,
        manifest_path=tmp_path / "grid.manifest",
        client=client,
    )
    await runner.run(limit=1, log=lambda _msg: None)
    writer.close()
    assert runner.calibration.frozen
    rows = _rows(tmp_path / "grid.manifest")
    assert [row["phase"] for row in rows] == ["calibration"] * CALIBRATION_CALLS + ["grid"]
    assert all(row["fit_eligible"] is False for row in rows[:CALIBRATION_CALLS])
    # The third input to `build_prompt`, without which the recorded seed does not regenerate bytes.
    assert rows[-1]["tokens_per_word"] == runner.calibration.tokens_per_word
    assert rows[-1]["prefix_words"] + rows[-1]["unique_words"] == rows[-1]["filler_words"]


# -- eligibility -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_cached_cell_without_a_reported_cache_count_is_not_fitted(
    grid: tuple[LatencyGrid, _FakeClient, LLMCallWriter],
) -> None:
    """A provider can ship a usage block with no ``prompt_tokens_details`` at all. The call
    succeeded and its total is real, but on the cached arm ``uncached = input - cached`` is then
    undefined, and a fit that trusts ``fit_eligible`` would take an unknown regressor for a known
    one — the one failure mode here that produces a plausible number rather than a missing one."""
    runner, client, _ = grid
    client.responses = [_stream(prompt=17_000, completion=256, cached=None)]
    outcome = await runner.run_cell(
        GridCell(input_tokens=17_000, output_tokens=256, cached_prefix_tokens=16_000)
    )
    assert outcome.record.usage_captured is True  # the call itself was fine
    assert outcome.manifest["reported_cached_input_tokens"] is None
    assert outcome.manifest["output_on_target"] is True
    assert outcome.manifest["fit_eligible"] is False


@pytest.mark.asyncio
async def test_an_uncached_cell_needs_no_cache_count(
    grid: tuple[LatencyGrid, _FakeClient, LLMCallWriter],
) -> None:
    """Its prompt is unique per call, so an unreported cached count is the zero it is. Requiring
    one here would throw away the whole grid on every provider that reports no cache detail."""
    runner, client, _ = grid
    client.responses = [_stream(prompt=4_000, completion=256, cached=None)]
    outcome = await runner.run_cell(GridCell(input_tokens=4_000, output_tokens=256))
    assert outcome.manifest["fit_eligible"] is True
    assert outcome.manifest["cached_on_target"] is None


@pytest.mark.asyncio
async def test_a_cached_cell_that_missed_its_prefix_is_visible_but_still_fitted(
    grid: tuple[LatencyGrid, _FakeClient, LLMCallWriter],
) -> None:
    """A reported miss is a *measurement*: the row is a valid observation of the input term at
    zero cached tokens. It still has to be visible, because a cached arm that missed everywhere
    means ``R_cache`` was never measured at all."""
    runner, client, _ = grid
    client.responses = [_stream(prompt=17_000, completion=256, cached=0)]
    outcome = await runner.run_cell(
        GridCell(input_tokens=17_000, output_tokens=256, cached_prefix_tokens=16_000)
    )
    assert outcome.manifest["cached_on_target"] is False
    assert outcome.manifest["fit_eligible"] is True


# -- resuming --------------------------------------------------------------------------------


def test_resume_skips_finished_units_and_repeats_a_broken_group(
    tmp_path: Path, profile: Any
) -> None:
    """Unit granularity, not cell: re-entering a cached group partway would measure a prefix this
    run never sent and the provider has long since evicted."""
    spec = GridSpec(blocks=1)
    units = spec.units()
    finished = next(unit for unit in units if len(unit) > 1)
    broken = next(unit for unit in units if len(unit) > 1 and unit is not finished)
    recorded = [cell.call_id() for cell in finished] + [broken[0].call_id()]
    runner = LatencyGrid(
        profile,
        spec=spec,
        writer=LLMCallWriter(tmp_path / "grid.jsonl"),
        manifest_path=tmp_path / "grid.manifest",
        client=_FakeClient([]),
        completed=recorded,
    )
    pending = runner.pending_units()
    assert finished not in [[*unit] for unit in pending]
    # The group with one cell recorded comes back whole, warm-up included.
    assert [cell.call_id() for cell in broken] in [[c.call_id() for c in u] for u in pending]


@pytest.mark.asyncio
async def test_a_repeated_cell_never_reuses_a_call_id(tmp_path: Path, profile: Any) -> None:
    """The two files join on ``call_id``. A resumed run that re-ran a cell would otherwise put two
    rows under one key and leave the join ambiguous — for a $10 measurement, silently."""
    cell = GridCell(input_tokens=1_000, output_tokens=64)
    writer = LLMCallWriter(tmp_path / "grid.jsonl")
    runner = LatencyGrid(
        profile,
        spec=GridSpec(blocks=1),
        writer=writer,
        manifest_path=tmp_path / "grid.manifest",
        client=_FakeClient([_stream(prompt=1_000, completion=64) for _ in range(2)]),
        completed=[cell.call_id()],
    )
    runner.calibration.freeze()
    first = await runner.run_cell(cell)
    second = await runner.run_cell(cell)
    assert first.record.call_id == f"{cell.call_id()}-r2"
    assert second.record.call_id == f"{cell.call_id()}-r3"
    # The design point is untouched, so grouping by cell still collects the repeats.
    assert first.manifest["cell_id"] == second.manifest["cell_id"] == cell.cell_id


def test_completed_ids_survive_a_half_written_last_line(tmp_path: Path) -> None:
    """Exactly the state an interrupted run leaves behind, and the case resume is for."""
    manifest = tmp_path / "grid.manifest"
    manifest.write_text('{"call_id": "uncached-in1000-out64-b0"}\n{"call_id": "uncach')
    assert completed_call_ids(manifest) == {"uncached-in1000-out64-b0"}
    assert completed_call_ids(tmp_path / "missing.manifest") == set()


@pytest.mark.asyncio
async def test_the_cli_refuses_to_append_into_an_existing_grid(tmp_path: Path) -> None:
    """Appending duplicates deterministic call ids; truncating throws away a paid-for grid. The
    default does neither."""
    out = tmp_path / "grid.jsonl"
    out.write_text("")
    argv = ["--profile", "gpt-5.4-high-paper", "--out", str(out)]
    with pytest.raises(SystemExit) as caught:
        await _main(argv)
    assert "--resume" in str(caught.value) and "--overwrite" in str(caught.value)


# -- the pilot: are the axes wide enough, and are the corners accepted? --------------------------


def _call_row(**over: Any) -> dict[str, Any]:
    row = {
        "call_id": "c",
        "arm": "react",
        "input_tokens": 1_000,
        "output_tokens": 100,
        "usage_captured": True,
        "finish_reason": "stop",
    }
    row.update(over)
    return row


def test_an_axis_the_pilot_stays_inside_is_left_as_designed() -> None:
    ranges = {a.axis: a for a in range_check([_call_row() for _ in range(5)])}
    assert ranges["input"].extend_to is None
    assert ranges["output"].extend_to is None
    assert ranges["input"].exceeded is False


def test_an_axis_the_pilot_overruns_by_mass_is_extended() -> None:
    """The ReAct arm re-feeds a growing prefix, so the case the grid's own axes were never fitted
    against is a prompt past the top input level. One doubling, per the declared rule."""
    rows = [_call_row(input_tokens=90_000)] + [_call_row(input_tokens=1_000) for _ in range(5)]
    ranges = {a.axis: a for a in range_check(rows)}
    assert ranges["input"].beyond_share > 0.05
    assert ranges["input"].extend_to == 128_000


def test_one_outlier_below_the_threshold_is_reported_and_not_acted_on() -> None:
    """The half of the rule that is easy to drop: exceeding the top level is *not* the trigger.
    A single long call cannot say whether extrapolating there is wrong, and the table the axes came
    from granted a level to nothing smaller than 3.3% of the mass."""
    rows = [_call_row(input_tokens=70_000)] + [_call_row(input_tokens=64_000) for _ in range(60)]
    axis = {a.axis: a for a in range_check(rows)}["input"]
    assert axis.exceeded is True
    assert axis.beyond_share < 0.05
    assert axis.extend_to is None
    assert axis.maximum == 70_000


def test_a_call_at_exactly_the_top_level_has_reached_it_not_overrun_it() -> None:
    axis = {a.axis: a for a in range_check([_call_row(input_tokens=64_000)])}["input"]
    assert axis.exceeded is False
    assert axis.extend_to is None


def test_rows_that_measured_nothing_are_not_read_as_small_calls() -> None:
    """An uncaptured or failed row carries fields, and counting them would pull the mass down
    exactly where the check is looking."""
    rows = [
        _call_row(input_tokens=90_000),
        _call_row(input_tokens=10, usage_captured=False),
        _call_row(input_tokens=10, finish_reason="error:APIError"),
    ]
    axis = {a.axis: a for a in range_check(rows)}["input"]
    assert axis.calls == 1
    assert axis.total_tokens == 90_000


def test_the_range_check_reads_a_file_and_calls_nothing(tmp_path: Path) -> None:
    """No --profile, deliberately: the check runs on whatever machine holds the rows, which need
    not be one with a profile's credentials, and naming one to satisfy the parser would be a
    dummy argument standing in for something the mode never reads."""
    path = tmp_path / "llm_calls.jsonl"
    path.write_text("\n".join(json.dumps(_call_row()) for _ in range(3)) + "\n")
    assert asyncio.run(_main(["--range-check", str(path)])) == 0


def test_an_axis_with_no_usable_sample_is_inconclusive_and_fails(tmp_path: Path) -> None:
    """Absence of evidence is not a passed check. Every row here measured nothing, so the axis has
    zero mass and a `beyond_share` of 0.0 — arithmetically identical to a pilot that stayed neatly
    inside the design, and the one case where exiting zero would be a lie."""
    rows = [
        _call_row(usage_captured=False),
        _call_row(finish_reason="error:APIError"),
    ]
    axis = {a.axis: a for a in range_check(rows)}["input"]
    assert axis.calls == 0
    assert axis.inconclusive is True
    assert axis.extend_to is None
    assert "INCONCLUSIVE" in format_range_check([axis])
    path = tmp_path / "llm_calls.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert asyncio.run(_main(["--range-check", str(path)])) == 1


def test_a_populated_axis_is_not_dragged_down_by_an_empty_one(tmp_path: Path) -> None:
    """The two axes are counted independently, so a row reporting input but no output leaves the
    output axis unchecked while the input axis still reads normally."""
    rows = [_call_row(output_tokens=None) for _ in range(3)]
    ranges = {a.axis: a for a in range_check(rows)}
    assert ranges["input"].inconclusive is False
    assert ranges["input"].extend_to is None
    assert ranges["output"].inconclusive is True
    path = tmp_path / "llm_calls.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert asyncio.run(_main(["--range-check", str(path)])) == 1


def test_every_mode_that_reaches_the_network_still_demands_a_profile() -> None:
    """Making --profile optional is scoped to the offline branch; dropping it anywhere else has to
    stay an error rather than becoming a run against some default."""
    with pytest.raises(SystemExit):
        asyncio.run(_main(["--dry-run"]))


def test_the_corners_are_both_ends_of_both_axes(grid: Any) -> None:
    runner, _, _ = grid
    corners = {(c.input_tokens, c.output_tokens) for c in runner.corner_cells()}
    assert corners == {
        (INPUT_LEVELS[0], OUTPUT_LEVELS[0]),
        (INPUT_LEVELS[0], OUTPUT_LEVELS[-1]),
        (INPUT_LEVELS[-1], OUTPUT_LEVELS[0]),
        (INPUT_LEVELS[-1], OUTPUT_LEVELS[-1]),
    }


def test_a_preflight_call_is_paid_for_recorded_and_kept_out_of_the_fit(
    grid: Any, tmp_path: Path
) -> None:
    """The small-end check's whole point: the bottom cell is sent for real, so a provider that
    refuses that output cap is found before the grid is paid for — and the attempt is on disk
    either way, without ever being eligible for the fit."""
    runner, client, writer = grid
    with writer:
        outcomes = asyncio.run(runner.preflight(log=lambda *_: None))
    assert len(outcomes) == 4
    rows = _rows(tmp_path / "grid.manifest")
    assert {row["phase"] for row in rows} == {"preflight"}
    assert not any(row["fit_eligible"] for row in rows)
    assert all(row["call_id"].startswith("preflight-") for row in rows)


def test_a_refused_corner_is_recorded_and_exits_non_zero(
    grid: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An endpoint can reject a `max_completion_tokens` too small for a reasoning model, and that
    rejection reaches only the bottom cell — so it would otherwise surface partway through a paid
    grid rather than in the call made to look for it."""
    runner, client, writer = grid
    client.responses = [RuntimeError("max_completion_tokens too small")] + [
        _stream(prompt=10, completion=10) for _ in range(3)
    ]
    with writer:
        outcomes = asyncio.run(runner.preflight(log=lambda *_: None))
    refused = [o for o in outcomes if o.manifest["error"]]
    assert [o.record.finish_reason for o in refused] == ["error:RuntimeError"]
    assert _rows(tmp_path / "grid.jsonl")[0]["finish_reason"] == "error:RuntimeError"
