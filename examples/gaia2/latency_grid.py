"""The designed latency grid that produces the charge model's coefficients.

The charge model is ``a0 + uncached_in/R_in + cached_in/R_cache + out/R_out``, frozen before the
sweep and applied identically to both arms. It cannot be fitted from either arm's own calls —
neither varies prompt length independently of answer length, which returns a *negative* input
coefficient — so the coefficients come from this grid, which moves the two axes independently.
``examples/gaia2/README.md`` is the operator-facing description: what each mode costs, what
``fit_eligible`` excludes and why, and how the two output files join.

Three invariants that are not visible from the code that enforces them:

- **The client is neither arm's.** Calls go through one plain ``AsyncOpenAI`` per run, built once
  and reused. ``a0`` has to be the *model's* fixed per-call cost, and routing the grid through one
  arm's stack would fold that arm's SDK overhead into a coefficient later charged to both.
- **Everything except the two token axes comes from the profile** — reasoning setting, temperature,
  provider routing, streaming — because latency does not transfer across them.
  ``max_completion_tokens`` is the single deliberate exception: it is the only mechanism that forces
  an output length, so it is the grid's independent variable rather than the profile's 16,384.
- **The axis levels were fitted to observed token mass, not to a context window**, and the top of
  the input axis is provisional: the calls they were derived from are S-ORA's, and an arm that
  re-feeds a growing prefix could exceed them. ``range_check`` is what tests that, under a rule
  fixed before the first pilot ran.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, Timeout

from examples.gaia2.evaluation.core import ModelProfile, load_profiles
from examples.gaia2.llm_calls import LLMCallRecord, LLMCallWriter, read_usage

EVAL_ROOT = Path(__file__).resolve().parent / "evaluation"

# The frozen grid. Constants rather than CLI options: the levels are part of what gets published
# alongside the coefficients, and a coefficient fitted at a shape nobody can reconstruct is not a
# frozen charge model. `--blocks` and `--seed` are runtime knobs; the shape is not.
INPUT_LEVELS = (1_000, 4_000, 8_000, 16_000, 32_000, 64_000)
OUTPUT_LEVELS = (64, 256, 1_024, 4_096, 8_192)
# The cached arm varies the *cached token count* rather than reusing one fixed prefix. A constant
# cached size is identifiable only through an intercept shift, which is weak and cannot be told
# apart from any systematic offset the cached condition happens to carry.
CACHED_PREFIX_LEVELS = (4_000, 16_000, 32_000, 64_000)
CACHED_OUTPUT_LEVELS = (256, 4_096)
CACHED_SUFFIX_TOKENS = 1_000
BLOCKS = 5
# Calls made before the grid, to measure the tokens-per-word ratio the prompts are built from and
# then freeze it. Small and cheap on both axes — the ratio is a property of the tokenizer, so it
# does not need a large prompt to measure, and these calls are paid for without being measurements.
CALIBRATION_CALLS = 2
CALIBRATION_INPUT_TOKENS = 2_000
CALIBRATION_OUTPUT_TOKENS = 16

# Held constant across every cell so it can never differentiate them.
SYSTEM_PROMPT = "You are a precise assistant. Follow the final instruction exactly."
PADDING_PREAMBLE = "Reference data, do not summarize:"
# Hex words are deliberately tokenizer-hostile: they resist merging, so the tokens-per-word ratio
# stays close across BPE vocabularies instead of drifting with the natural-language a model was
# trained on. The exact ratio still gets calibrated per model at run time.
_ASSUMED_TOKENS_PER_WORD = 5.0
# Targets are approached, never asserted. Local tokenization is at best targeting assistance and is
# unavailable for some hosted models, so the fit uses `prompt_tokens`/`completion_tokens` as
# reported and this bound only flags a cell that missed badly enough to be worth knowing about.
ON_TARGET_TOLERANCE = 0.10


@dataclass(frozen=True)
class GridCell:
    """One point of the design. ``input_tokens`` is the whole prompt's target, of which
    ``cached_prefix_tokens`` is the shared, deliberately cacheable head (0 on the uncached arm)."""

    input_tokens: int
    output_tokens: int
    cached_prefix_tokens: int = 0
    block: int = 0
    warmup: bool = False
    phase: str = "grid"

    @property
    def cached(self) -> bool:
        return self.cached_prefix_tokens > 0

    @property
    def cell_id(self) -> str:
        """Identifies a *design point*, which a warm-up is not — hence the suffix. Without it a
        warm-up collides with the measured cell at the same prefix and output, and anyone grouping
        by cell to take a median folds a cold prefill into a cached cell unless they also happened
        to filter on ``fit_eligible``."""
        condition = f"cached{self.cached_prefix_tokens}" if self.cached else "uncached"
        cell = f"{condition}-in{self.input_tokens}-out{self.output_tokens}"
        if self.phase != "grid":
            cell = f"{self.phase}-{cell}"
        return f"{cell}-warmup" if self.warmup else cell

    def call_id(self) -> str:
        return f"{self.cell_id}-b{self.block}"


@dataclass(frozen=True)
class GridSpec:
    input_levels: tuple[int, ...] = INPUT_LEVELS
    output_levels: tuple[int, ...] = OUTPUT_LEVELS
    cached_prefix_levels: tuple[int, ...] = CACHED_PREFIX_LEVELS
    cached_output_levels: tuple[int, ...] = CACHED_OUTPUT_LEVELS
    cached_suffix_tokens: int = CACHED_SUFFIX_TOKENS
    blocks: int = BLOCKS
    seed: int = 20260910

    def cells(self) -> list[GridCell]:
        """Every cell in execution order — :meth:`units` flattened."""
        return [cell for unit in self.units() for cell in unit]

    def units(self) -> list[list[GridCell]]:
        """Execution order, in the groups that must not be broken up.

        Each block holds every cell exactly once, so provider load — which drifts over the hours a
        grid takes — is spread across cells rather than aligned with one. Five blocks rather than
        four because the median of an odd sample is an observation instead of an interpolation
        between the two middle draws, and one congested call moves an interpolated median directly.

        Cached cells are shuffled as *contiguous groups*, not individually: a prefix has to be
        warmed and then measured back-to-back inside the provider's cache TTL, and a run that
        scattered them would fit ``R_cache`` against prefills that had gone cold."""
        ordered: list[list[GridCell]] = []
        for block in range(self.blocks):
            rng = random.Random(f"{self.seed}:block:{block}")
            units: list[list[GridCell]] = [
                [GridCell(input_tokens=in_tok, output_tokens=out_tok, block=block)]
                for in_tok in self.input_levels
                for out_tok in self.output_levels
            ]
            for prefix in self.cached_prefix_levels:
                total = prefix + self.cached_suffix_tokens
                group = [
                    GridCell(
                        input_tokens=total,
                        # Warm the prefix with the cheapest output on the arm; it is paid for and
                        # recorded, but it is the call that *populates* the cache rather than one
                        # that measures a hit, so the fit must not see it.
                        output_tokens=min(self.cached_output_levels),
                        cached_prefix_tokens=prefix,
                        block=block,
                        warmup=True,
                    )
                ]
                group += [
                    GridCell(
                        input_tokens=total,
                        output_tokens=out_tok,
                        cached_prefix_tokens=prefix,
                        block=block,
                    )
                    for out_tok in self.cached_output_levels
                ]
                units.append(group)
            rng.shuffle(units)
            ordered.extend(units)
        return ordered

    def volume(self) -> dict[str, int]:
        """Target token totals, for a dry run. Approximate by construction — the request is built
        to a target and billed at what the provider counted."""
        cells = self.cells()
        return {
            "calls": len(cells),
            "uncached_input_tokens": sum(c.input_tokens - c.cached_prefix_tokens for c in cells),
            "cached_input_tokens": sum(c.cached_prefix_tokens for c in cells),
            "output_tokens": sum(c.output_tokens for c in cells),
            "warmup_calls": sum(1 for c in cells if c.warmup),
        }


@dataclass
class _Calibration:
    """Tokens per filler word, measured against what the provider actually counted, then frozen.

    The ratio has to stop moving before the first measured call, and this is not a stylistic
    preference. Word counts are derived from it, so a ratio that kept refining would make the
    prompt for a given cell depend on which calls happened to run — and succeed — before it. Two
    concrete failures follow. The stored seed would no longer regenerate the run's bytes, which is
    the whole basis on which a published coefficient can be traced to the prompt that produced it.
    And a cached prefix is byte-identical across a group only if its *word count* is identical, so
    a ratio that moved between a warm-up and the measurement it warms shifts the boundary and
    hands the provider's prefix cache a shorter head than the one it stored — a silently degraded
    cache level in the arm whose only purpose is measuring cache.

    The ratio is taken over the whole prompt, so the fixed framing (a system line and a preamble,
    about twenty tokens) is folded in — under 2% at the smallest cell and negligible above it."""

    tokens_per_word: float = _ASSUMED_TOKENS_PER_WORD
    samples: int = 0
    frozen: bool = False

    def words_for(self, target_tokens: int) -> int:
        return max(1, round(target_tokens / self.tokens_per_word))

    def observe(self, words: int, reported_tokens: int) -> None:
        if self.frozen or words <= 0 or reported_tokens <= 0:
            return
        ratio = reported_tokens / words
        # Plain running mean: the quantity is a property of the tokenizer, not a drifting one, so
        # there is nothing for a decay factor to track.
        self.tokens_per_word = (self.tokens_per_word * self.samples + ratio) / (self.samples + 1)
        self.samples += 1

    def freeze(self) -> _Calibration:
        self.frozen = True
        return self


def _filler(words: int, rng: random.Random) -> str:
    return " ".join(f"{rng.getrandbits(32):08x}" for _ in range(words))


@dataclass(frozen=True)
class _Prompt:
    """One cell's user message and the word counts it was built from.

    The counts are carried, not recomputed, because they are what makes the prompt reconstructable:
    together with the seed and the cell they are the complete input to :func:`build_prompt`, and
    both land on the manifest row. ``prompt_sha256`` lets a reader *check* a rebuilt prompt; these
    are what let them build it in the first place."""

    text: str
    prefix_words: int
    unique_words: int

    @property
    def words(self) -> int:
        return self.prefix_words + self.unique_words


def build_prompt(cell: GridCell, spec: GridSpec, calibration: _Calibration) -> _Prompt:
    """The user message for one cell, and the filler word counts it is built from.

    Filler is deterministic, from a seed that is recorded — never unrecorded randomness. A grid is
    a published measurement, and "the prompts were random" makes it unreproducible. Reproducing a
    prompt takes the seed, the cell, *and* the frozen tokens-per-word ratio, since that ratio sets
    the word counts; all three are on every manifest row for that reason.

    On the cached arm the shared prefix is derived from the prefix *size* alone, so every call at
    that level sends a byte-identical head and the provider's prefix cache can hit, while the tail
    stays unique per call so nothing is served from a whole-response cache."""
    parts: list[str] = []
    prefix_words = 0
    if cell.cached:
        prefix_words = calibration.words_for(cell.cached_prefix_tokens)
        prefix_rng = random.Random(f"{spec.seed}:prefix:{cell.cached_prefix_tokens}")
        parts.append(f"{PADDING_PREAMBLE}\n{_filler(prefix_words, prefix_rng)}")
        unique_tokens = cell.input_tokens - cell.cached_prefix_tokens
    else:
        unique_tokens = cell.input_tokens
    unique_words = calibration.words_for(unique_tokens)
    unique_rng = random.Random(f"{spec.seed}:{cell.call_id()}")
    parts.append(f"{PADDING_PREAMBLE}\n{_filler(unique_words, unique_rng)}")
    # Forces the output length together with `max_completion_tokens`: enumerating this far always
    # overruns the cap, so the cell ends by truncation at its target rather than by the model
    # deciding it is finished. With reasoning enabled the budget is spent on thinking first, which
    # is the same decode loop at the same rate and so measures the same `R_out`.
    parts.append(
        f"Output the integers from 1 to {max(16, cell.output_tokens)}, one per line, "
        "with no other text."
    )
    return _Prompt("\n\n".join(parts), prefix_words, unique_words)


def prompt_digest(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\x00{user}".encode()).hexdigest()


def profile_digest(profile: ModelProfile) -> str:
    return hashlib.sha256(json.dumps(profile.to_dict(), sort_keys=True).encode()).hexdigest()


@dataclass
class _Outcome:
    record: LLMCallRecord
    manifest: dict[str, Any]


class LatencyGrid:
    """Runs the grid against one profile, writing an ``LLMCallRecord`` per call plus a manifest.

    Two files rather than a wider record. ``LLMCallRecord`` describes *a model call* and is shared
    with both scaffold arms; a cell's targets, cache condition, block index, warm-up flag and
    prompt identity describe *an experiment*, and pushing them into the shared row would put six
    permanently-null columns on every row the benchmark writes. They join on ``call_id``.

    Calls run strictly one at a time. Concurrency would be faster and would also make every
    measurement a measurement of our own queueing."""

    def __init__(
        self,
        profile: ModelProfile,
        *,
        spec: GridSpec | None = None,
        writer: LLMCallWriter,
        manifest_path: Path,
        api_key: str | None = None,
        client: Any = None,
        completed: Iterable[str] = (),
    ) -> None:
        self.profile = profile
        self.spec = spec or GridSpec()
        self.writer = writer
        self.manifest_path = manifest_path
        self.calibration = _Calibration()
        self._profile_sha256 = profile_digest(profile)
        self._client = client if client is not None else self._build_client(profile, api_key)
        self._manifest: Any = None
        # Call ids already on disk from an interrupted run. They decide what is skipped, and they
        # seed the uniqueness check so a re-run cell cannot land on an id the file already holds.
        self._completed = set(completed)
        self._seen = set(self._completed)

    @staticmethod
    def _build_client(profile: ModelProfile, api_key: str | None) -> AsyncOpenAI:
        key = api_key or os.environ.get(profile.credential_env)
        if not key:
            raise RuntimeError(f"{profile.credential_env} is not set")
        kwargs: dict[str, Any] = {"api_key": key, "base_url": profile.endpoint}
        if profile.sdk_max_retries is not None:
            # Retries stay off by profile: a silently retried call would be billed once and
            # measured as the sum of both attempts.
            kwargs["max_retries"] = profile.sdk_max_retries
        if profile.stall_timeout is not None:
            kwargs["timeout"] = Timeout(profile.stall_timeout, connect=10.0)
        return AsyncOpenAI(**kwargs)

    def _unique_call_id(self, cell: GridCell) -> str:
        """The cell's id, suffixed if the file already holds one.

        The two files join on ``call_id``, so it has to be unique across everything in them —
        including a cell a resumed run deliberately repeats. ``cell_id`` is untouched, so grouping
        by design point still collects the repeat, which is what a reader wants: it was paid for
        and it measured the same cell."""
        call_id = cell.call_id()
        if call_id not in self._seen:
            self._seen.add(call_id)
            return call_id
        attempt = 2
        while f"{call_id}-r{attempt}" in self._seen:
            attempt += 1
        self._seen.add(f"{call_id}-r{attempt}")
        return f"{call_id}-r{attempt}"

    def _request_kwargs(self, cell: GridCell, user: str) -> dict[str, Any]:
        return {
            "model": self.profile.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            **self.profile.request_kwargs(),
            # The one setting the grid overrides, and it has to come last: the profile sends its
            # own output cap, and forcing an output length is what the grid is for. See the module
            # docstring.
            "max_completion_tokens": cell.output_tokens,
        }

    async def _call(self, kwargs: dict[str, Any]) -> tuple[Any, str | None, str | None]:
        """(usage-carrying object, finish_reason, observed model). Streamed when the profile
        streams, because streaming is not a free choice on the S-ORA side — its stall timeout means
        "went quiet" only on a streamed call — and a transport difference between the grid and the
        arms would land straight on the per-arm bias P4 is built to compare."""
        if not self.profile.stream:
            response = await self._client.chat.completions.create(**kwargs)
            return response, None, getattr(response, "model", None)
        usage_chunk: Any = None
        finish_reason: str | None = None
        observed_model: str | None = None
        stream = await self._client.chat.completions.create(
            **kwargs, stream=True, stream_options={"include_usage": True}
        )
        async with stream:
            async for chunk in stream:
                candidate = getattr(chunk, "model", None)
                if isinstance(candidate, str):
                    observed_model = candidate
                if getattr(chunk, "usage", None) is not None:
                    usage_chunk = chunk
                for choice in getattr(chunk, "choices", None) or []:
                    reason = getattr(choice, "finish_reason", None)
                    if isinstance(reason, str):
                        finish_reason = reason
        return usage_chunk, finish_reason, observed_model

    async def run_cell(self, cell: GridCell, *, phase: str = "grid") -> _Outcome:
        prompt = build_prompt(cell, self.spec, self.calibration)
        kwargs = self._request_kwargs(cell, prompt.text)
        started = time.perf_counter()
        error: str | None = None
        try:
            response, stream_reason, observed_model = await self._call(kwargs)
        except Exception as exc:  # a paid crossing that failed is still a paid crossing
            response, stream_reason, observed_model = None, None, None
            error = type(exc).__name__
        seconds = time.perf_counter() - started
        input_tokens, cached, output_tokens, reasoning, reason, captured = read_usage(response)
        finish_reason = stream_reason or reason
        if captured:
            # A no-op once frozen, which is before the first measured call — see `_Calibration`.
            self.calibration.observe(prompt.words, input_tokens)
        record = LLMCallRecord(
            call_id=self._unique_call_id(cell),
            arm="grid",
            model=observed_model or self.profile.model,
            semantic_label=cell.cell_id,
            input_tokens=input_tokens if captured else None,
            cached_input_tokens=cached,
            output_tokens=output_tokens if captured else None,
            reasoning_tokens=reasoning,
            seconds=seconds,
            round_trips=1,
            finish_reason=f"error:{error}" if error else finish_reason,
            usage_captured=captured,
        )
        manifest = {
            "call_id": record.call_id,
            "cell_id": cell.cell_id,
            "phase": phase,
            "profile": self.profile.name,
            "profile_sha256": self._profile_sha256,
            "seed": self.spec.seed,
            # The third input to `build_prompt`, and the one that is not a constant of the design:
            # without it the seed alone does not regenerate the run's bytes.
            "tokens_per_word": self.calibration.tokens_per_word,
            "block": cell.block,
            "warmup": cell.warmup,
            "cache_condition": "cached" if cell.cached else "uncached",
            "target_input_tokens": cell.input_tokens,
            "target_cached_prefix_tokens": cell.cached_prefix_tokens,
            "target_output_tokens": cell.output_tokens,
            "filler_words": prompt.words,
            "prefix_words": prompt.prefix_words,
            "unique_words": prompt.unique_words,
            "prompt_sha256": prompt_digest(SYSTEM_PROMPT, prompt.text),
            "reported_input_tokens": input_tokens if captured else None,
            "reported_output_tokens": output_tokens if captured else None,
            "reported_cached_input_tokens": cached,
            "input_on_target": _on_target(input_tokens if captured else None, cell.input_tokens),
            "output_on_target": _on_target(output_tokens if captured else None, cell.output_tokens),
            # Reported against the prefix the cell asked to have cached. Diagnostic, not a filter:
            # a cached cell whose prefix missed is still a valid observation of the input term at
            # `cached_input_tokens=0`, but a cached arm that missed *systematically* means `R_cache`
            # was never measured, and that has to be visible without recomputing it per row.
            "cached_on_target": (
                _on_target(cached, cell.cached_prefix_tokens) if cell.cached else None
            ),
            # The one flag the fit reads first. A warm-up, a failure, or a cell that missed its
            # target is excluded from the fit and kept in the file: every one of them was paid for,
            # and an audit that cannot see what was paid for is not an audit.
            #
            # A cached cell additionally needs the provider to have *reported* its cached count.
            # `read_usage` returns captured=True with cached=None when a provider ships a usage
            # block but no `prompt_tokens_details`, and on the cached arm that leaves the split
            # `uncached = input - cached` undefined — an unknown regressor, not a small one. The
            # uncached arm keeps a None here: its prompts are unique per call, so the fit reads it
            # as the zero it is, and requiring a count there would discard the whole grid on every
            # provider that reports no cache detail at all.
            "fit_eligible": (
                not cell.warmup
                and phase == "grid"
                and error is None
                and captured
                and (cached is not None or not cell.cached)
                and _on_target(output_tokens, cell.output_tokens) is True
            ),
            "error": error,
        }
        return _Outcome(record=record, manifest=manifest)

    def pending_units(self, limit: int | None = None) -> list[list[GridCell]]:
        """The units still to run, skipping only those already recorded *in full*.

        Granularity is the unit, not the cell, and that is the point of resuming this way: a cached
        group is a warm-up plus the measurements it warms, so re-entering it partway would measure
        a prefix this run never sent and the provider has long since evicted. A group missing any
        cell is therefore re-run whole, and the warm-up's repeat takes a suffixed call id rather
        than colliding with the one already on disk."""
        units = [
            unit
            for unit in self.spec.units()
            if not all(cell.call_id() in self._completed for cell in unit)
        ]
        if limit is None:
            return units
        kept: list[list[GridCell]] = []
        remaining = limit
        for unit in units:
            if remaining <= 0:
                break
            # Whole units, so `--limit` cannot itself sever a warm-up from its measurements.
            kept.append(unit)
            remaining -= len(unit)
        return kept

    async def calibrate(self, *, log: Any = print) -> None:
        """Measure tokens-per-filler-word against the provider, then freeze it for the run.

        Runs before the first measured call and never during one. These calls are recorded like
        any other — they are paid for — under ``phase: "calibration"`` and excluded from the fit."""
        if self.calibration.frozen:
            return
        for index in range(CALIBRATION_CALLS):
            cell = GridCell(
                input_tokens=CALIBRATION_INPUT_TOKENS,
                output_tokens=CALIBRATION_OUTPUT_TOKENS,
                block=index,
                phase="calibration",
            )
            outcome = await self.run_cell(cell, phase="calibration")
            self._write(outcome)
            log(
                f"[calibration {index + 1}/{CALIBRATION_CALLS}] "
                f"{outcome.manifest['call_id']} in={outcome.record.input_tokens}"
            )
        self.calibration.freeze()
        log(f"  tokens per word frozen at {self.calibration.tokens_per_word:.3f}")

    def _write(self, outcome: _Outcome) -> None:
        self.writer.write(outcome.record)
        if self._manifest is None:
            raise RuntimeError("manifest is not open")
        self._manifest.write(json.dumps(outcome.manifest, sort_keys=True) + "\n")
        self._manifest.flush()

    async def run(self, *, limit: int | None = None, log: Any = print) -> int:
        units = self.pending_units(limit)
        cells = [cell for unit in units for cell in unit]
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with self.manifest_path.open("a", encoding="utf-8") as manifest:
            self._manifest = manifest
            try:
                await self.calibrate(log=log)
                for index, cell in enumerate(cells, start=1):
                    outcome = await self.run_cell(cell)
                    self._write(outcome)
                    written += 1
                    log(
                        f"[{index}/{len(cells)}] {outcome.manifest['call_id']} "
                        f"{outcome.record.seconds:.2f}s "
                        f"in={outcome.record.input_tokens} out={outcome.record.output_tokens}"
                        + ("" if outcome.manifest["fit_eligible"] else "  (excluded from fit)")
                    )
            finally:
                self._manifest = None
        return written

    def corner_cells(self) -> list[GridCell]:
        """The four extremes of the design: each axis's lowest and highest level, crossed.

        Corners rather than every cell because the two refusals worth finding early live at the
        ends and nowhere in between — an output cap too small for a reasoning model to answer
        under, and a prompt at the top of the input axis. Crossed rather than taken one axis at a
        time so the longest prompt is also seen with the longest decode, which is the cell most
        likely to hit a wall-clock or context limit."""
        return [
            GridCell(input_tokens=i, output_tokens=o, block=0, phase="preflight")
            for i in (self.spec.input_levels[0], self.spec.input_levels[-1])
            for o in (self.spec.output_levels[0], self.spec.output_levels[-1])
        ]

    async def preflight(self, *, log: Any = print) -> list[_Outcome]:
        """Send each corner once and record it, so the paid grid is not where a corner is refused.

        Calibration runs first, exactly as it would for the grid: the prompts are built from the
        frozen tokens-per-word ratio, so a corner sent before freezing it would not be the prompt
        the grid will send. Nothing here writes into the grid's own files — the caller points the
        writer elsewhere — but everything here is written *somewhere*, because these calls are paid
        for like any other."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        outcomes: list[_Outcome] = []
        with self.manifest_path.open("a", encoding="utf-8") as manifest:
            self._manifest = manifest
            try:
                await self.calibrate(log=log)
                cells = self.corner_cells()
                for index, cell in enumerate(cells, start=1):
                    outcome = await self.run_cell(cell, phase="preflight")
                    self._write(outcome)
                    outcomes.append(outcome)
                    record = outcome.record
                    log(
                        f"[corner {index}/{len(cells)}] {record.call_id} "
                        f"in={record.input_tokens} out={record.output_tokens} "
                        f"reasoning={record.reasoning_tokens} "
                        f"finish={record.finish_reason} {(record.seconds or 0.0):.2f}s"
                    )
            finally:
                self._manifest = None
        return outcomes


def completed_call_ids(manifest_path: Path) -> set[str]:
    """Call ids already recorded, for resuming. A line that will not parse is skipped rather than
    fatal: a run killed mid-write leaves a partial last line, and that is exactly the case resume
    exists for. The cell it belonged to is then re-run, which is the safe direction."""
    if not manifest_path.exists():
        return set()
    ids: set[str] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        call_id = row.get("call_id")
        if isinstance(call_id, str):
            ids.add(call_id)
    return ids


def _on_target(reported: int | None, target: int) -> bool | None:
    if reported is None or target <= 0:
        return None
    return abs(reported - target) / target <= ON_TARGET_TOLERANCE


# Fixed before the first pilot ran, which is the only thing that makes it a check rather than a
# reading of the numbers. Changing it changes what that pilot was allowed to conclude, so it moves
# with a note saying why and not as a tuning knob. The README states the rule it implements.
AXIS_EXTENSION_THRESHOLD = 0.05


@dataclass(frozen=True)
class AxisRange:
    """One axis of the design, measured against calls that actually happened."""

    axis: str
    levels: tuple[int, ...]
    bands: tuple[tuple[str, int], ...]
    beyond_tokens: int
    total_tokens: int
    maximum: int | None
    calls: int

    @property
    def beyond_share(self) -> float:
        return self.beyond_tokens / self.total_tokens if self.total_tokens else 0.0

    @property
    def inconclusive(self) -> bool:
        """No usable sample on this axis, which is not the same as nothing landing beyond it.

        A pilot whose rows all failed, went uncaptured, or never reported this axis's token count
        yields `beyond_share` 0.0 by the same arithmetic as a well-behaved pilot that stayed inside
        the design. Reading that as "in range" would let absence of evidence pass the check."""
        return self.calls == 0 or self.total_tokens == 0

    @property
    def extend_to(self) -> int | None:
        """The level to add, or None to leave the axis as designed — the rule, and only the rule."""
        if self.inconclusive or self.beyond_share < AXIS_EXTENSION_THRESHOLD:
            return None
        return self.levels[-1] * 2

    @property
    def exceeded(self) -> bool:
        """Whether anything at all landed past the top level, extension-worthy or not."""
        return self.maximum is not None and self.maximum > self.levels[-1]


def _axis_range(axis: str, levels: tuple[int, ...], values: list[int]) -> AxisRange:
    edges = list(levels)
    counted: list[tuple[str, int]] = []
    for index, (low, high) in enumerate(zip(edges, edges[1:], strict=False)):
        # The last band closes on the right, so a call landing exactly on the top level is one the
        # axis *reaches*. Counting it as beyond would extend the axis on a call the design already
        # measures, which is the opposite of what the rule is for.
        last = index == len(edges) - 2
        inside = sum(v for v in values if low <= v <= high) if last else 0
        bounded = sum(v for v in values if low <= v < high)
        counted.append((f"{low}-{high}", inside or bounded))
    below = sum(v for v in values if v < edges[0])
    beyond = sum(v for v in values if v > edges[-1])
    bands = [(f"<{edges[0]}", below), *counted, (f">{edges[-1]}", beyond)]
    return AxisRange(
        axis=axis,
        levels=levels,
        bands=tuple(bands),
        beyond_tokens=beyond,
        total_tokens=sum(values),
        maximum=max(values) if values else None,
        calls=len(values),
    )


def range_check(rows: Iterable[dict[str, Any]], spec: GridSpec | None = None) -> list[AxisRange]:
    """Band a pilot's recorded calls against the grid's axes, under the rule fixed before the pilot.

    Extend an axis by one level, doubling its top, when the band beyond that top holds at least
    `AXIS_EXTENSION_THRESHOLD` of the axis's token mass. Mass rather than call count, because a
    coefficient is fitted against mass and the two disagree sharply; and extend only, never
    retract, since a level already in the design is what separates the coefficients from each
    other.

    Reads the same ``llm_calls.jsonl`` the arms write, and takes only rows whose usage is a
    complete account of the call: a row with ``usage_captured`` false, or one whose call failed, is
    not a measurement of how large that arm's prompts get — it is a measurement of nothing, and
    counting it as a small call would bias the mass downwards exactly where the check is trying to
    look. The band beyond the top level is open (``>``): a call sitting exactly at the top level is
    one the design measures, so it counts as reaching the axis rather than as overrunning it."""
    spec = spec or GridSpec()
    inputs: list[int] = []
    outputs: list[int] = []
    for row in rows:
        if not row.get("usage_captured", True):
            continue
        if str(row.get("finish_reason") or "").startswith("error:"):
            continue
        if isinstance(row.get("input_tokens"), int):
            inputs.append(int(row["input_tokens"]))
        if isinstance(row.get("output_tokens"), int):
            outputs.append(int(row["output_tokens"]))
    return [
        _axis_range("input", spec.input_levels, inputs),
        _axis_range("output", spec.output_levels, outputs),
    ]


def format_range_check(ranges: list[AxisRange]) -> str:
    lines: list[str] = []
    for axis in ranges:
        lines.append(
            f"{axis.axis} axis {axis.levels}  ({axis.calls} calls, "
            f"{axis.total_tokens / 1e3:.1f}k tokens, max {axis.maximum})"
        )
        for label, tokens in axis.bands:
            share = tokens / axis.total_tokens if axis.total_tokens else 0.0
            lines.append(f"    {label:>14}  {tokens / 1e3:8.1f}k  {share:6.1%}")
        if axis.inconclusive:
            lines.append("    INCONCLUSIVE: no usable call reported tokens on this axis")
            continue
        if axis.extend_to:
            verdict = f"EXTEND to {axis.extend_to}"
        elif axis.exceeded:
            verdict = "exceeded, below threshold: keep as designed"
        else:
            verdict = "in range"
        lines.append(
            f"    beyond top level: {axis.beyond_share:.1%} "
            f"(rule: extend at {AXIS_EXTENSION_THRESHOLD:.0%})  ->  {verdict}"
        )
    return "\n".join(lines)


def read_calls(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Not `required`, because --range-check runs offline and ahead of profile loading; the modes
    # that do reach the network check for it themselves, below.
    parser.add_argument("--profile", default=None, help="profile name from profiles.json")
    parser.add_argument("--profiles-path", type=Path, default=EVAL_ROOT / "profiles.json")
    parser.add_argument("--out", type=Path, default=Path("latency_grid.jsonl"))
    parser.add_argument("--manifest", type=Path, default=None, help="default: --out + .manifest")
    parser.add_argument("--blocks", type=int, default=BLOCKS)
    parser.add_argument("--seed", type=int, default=GridSpec.seed)
    parser.add_argument("--limit", type=int, default=None, help="run only the first N cells")
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument(
        "--resume",
        action="store_true",
        help="continue into existing outputs, re-running any cached group left incomplete",
    )
    existing.add_argument(
        "--overwrite",
        action="store_true",
        help="truncate both output files and run the whole grid again",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the target token volume, then exit without calling anything",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="send the four corner cells once and report what came back, instead of the grid",
    )
    parser.add_argument(
        "--range-check",
        type=Path,
        metavar="LLM_CALLS_JSONL",
        help="band a pilot's recorded calls against the axes and apply the declared extension "
        "rule; reads a file and calls nothing",
    )
    return parser


async def _main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = GridSpec(blocks=args.blocks, seed=args.seed)
    if args.range_check is not None:
        # Offline, and deliberately ahead of everything else: reading a pilot's rows needs no
        # profile, no credential and no network, and a check that refused to run without them
        # would be unusable on the machine holding the rows.
        ranges = range_check(read_calls(args.range_check), spec)
        print(format_range_check(ranges))
        # An axis with no usable sample has not passed the check, it has gone unchecked — so a
        # script chaining this into a decision about the axes stops here rather than reading the
        # empty result as a confirmation that the design holds.
        return 1 if any(axis.inconclusive for axis in ranges) else 0
    if args.profile is None:
        raise SystemExit("--profile is required for every mode except --range-check")
    profiles = load_profiles(args.profiles_path)
    if args.profile not in profiles:
        raise SystemExit(f"unknown profile {args.profile!r}; have {sorted(profiles)}")
    profile = profiles[args.profile]
    volume = spec.volume()
    print(f"profile {profile.name} -> {profile.model} at {profile.endpoint}")
    print(f"  stream={profile.stream}  sha256={profile_digest(profile)[:12]}")
    print(
        f"  {volume['calls']} calls ({volume['warmup_calls']} warm-ups) over {spec.blocks} blocks, "
        f"seed {spec.seed}"
    )
    print(
        f"  + {CALIBRATION_CALLS} calibration calls at {CALIBRATION_INPUT_TOKENS} input tokens, "
        "which freeze the tokens-per-word ratio the prompts are built from"
    )
    print(
        f"  targets: {volume['uncached_input_tokens'] / 1e6:.2f}M uncached input, "
        f"{volume['cached_input_tokens'] / 1e6:.2f}M cached input, "
        f"{volume['output_tokens'] / 1e3:.0f}k output"
    )
    if args.dry_run:
        return 0
    manifest_path = args.manifest or args.out.with_suffix(args.out.suffix + ".manifest")
    # The two files join on `call_id`, which is deterministic from the cell and the block — so
    # appending into an existing pair silently duplicates join keys, and a reader has no way to
    # tell a repeat from the original. Blindly truncating is the other way to lose a grid, and this
    # one costs about $10 to re-run, so neither happens without being asked for.
    existing = [path for path in (args.out, manifest_path) if path.exists()]
    completed: set[str] = set()
    if existing and not (args.resume or args.overwrite):
        listed = ", ".join(str(path) for path in existing)
        raise SystemExit(
            f"{listed} already exists; pass --resume to continue it or --overwrite to replace it"
        )
    if args.resume:
        completed = completed_call_ids(manifest_path)
        print(f"  resuming: {len(completed)} calls already recorded")
    with LLMCallWriter(args.out, reset=args.overwrite) as writer:
        if args.overwrite:
            # Truncated together with the calls file: one file reset and the other appended is the
            # same broken join, arrived at from the other side.
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text("", encoding="utf-8")
        grid = LatencyGrid(
            profile,
            spec=spec,
            writer=writer,
            manifest_path=manifest_path,
            completed=completed,
        )
        if args.preflight:
            outcomes = await grid.preflight()
            refused = [o for o in outcomes if o.manifest["error"]]
            empty = [
                o for o in outcomes if not o.manifest["error"] and not (o.record.output_tokens or 0)
            ]
            for outcome in refused:
                print(f"  REFUSED {outcome.record.call_id}: {outcome.manifest['error']}")
            for outcome in empty:
                print(f"  EMPTY   {outcome.record.call_id}: accepted, no output tokens reported")
            print(f"wrote {len(outcomes)} preflight calls to {args.out} and {manifest_path}")
            # A corner that was refused is a cell the grid cannot run; a corner that came back
            # empty is one it can run and learn nothing from. Both are worth a non-zero exit, so a
            # script chaining preflight into the paid grid stops here rather than proceeding.
            return 1 if refused or empty else 0
        written = await grid.run(limit=args.limit)
    print(f"wrote {written} calls to {args.out} and {manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(_main()))
