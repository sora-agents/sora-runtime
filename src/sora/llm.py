"""Provider-agnostic LLM access for the reasoning path."""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, cast

# A dedicated child of the `sora` tree so instrumentation records are addressable on their own —
# the CLI presenter surfaces them as a per-call cue, `LLMMeter` tallies them, and neither has to
# reach into the reasoning path to do it.
_llm_log = logging.getLogger("sora.llm")


# Correlates each metered round-trip to the off-cycle inference (the _infer_/_ground_ internal
# action) that drove it, so the meter can later fold a *discarded* inference's cost — interrupted or
# superseded, but still run to completion (ADR-0021) — into a separate "wasted" bucket instead of
# conflating it with useful work. The action sets it around the model call; it stays None for any
# other LLM use (which is therefore never attributable to a discard). ContextVars are task-local and
# copied per task at creation, so a background inference sets its own without touching siblings.
current_inference_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_inference_id", default=None
)

# A logical model-call id is broader than an off-cycle activity inference. Retirement and
# relevance judgements deliberately do not move an activity to RUNNING (ADR-0021), but their log
# records still need a stable identity. A semantic caller may hold this scope across parsing; the
# metered transport supplies a one-round-trip scope when the caller has none.
current_llm_call_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_llm_call_id", default=None
)


@contextmanager
def llm_call_scope() -> Iterator[str]:
    """Correlate one semantic LLM call without changing activity inference state."""
    existing = current_llm_call_id.get()
    call_id = existing or current_inference_id.get() or uuid.uuid4().hex
    token = current_llm_call_id.set(call_id)
    try:
        yield call_id
    finally:
        current_llm_call_id.reset(token)


def _logical_call_id() -> str:
    return current_llm_call_id.get() or current_inference_id.get() or uuid.uuid4().hex


def _id_tag(call_id: str | None, request: CompletionRequest | None = None) -> str:
    """A short, eyeballable logical-call prefix for per-call cues
    (``[32a3ad56] plan/v1 ``), so a later ``discarded`` cue is visibly the *same* call as its
    usage/timing lines — otherwise unrecoverable once those lines have already printed."""
    if call_id is None:
        return ""
    label = f"{request.semantic_label}/v{request.prompt_version} " if request is not None else ""
    return f"[{call_id[:8]}] {label}"


# A response's answer text averages ~this many characters per token (English/JSON prose). Only used
# to turn the *measured* answer length into an estimated answer-token count, so the thinking
# estimate below can subtract it from output_tokens. Rough on purpose — the discriminating signal
# (a thinking-bound call runs ~90%+, an answer-bound one near 0) survives any sane value of it.
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class CompletionProfile:
    """Optional provider-neutral transport hints for one completion.

    ``reasoning`` exposes only the portable intersection: ``low``, ``medium``, and ``high``.
    Providers add incompatible extremes such as ``minimal``, ``xhigh``, or ``max``; ``None`` keeps
    the provider/model default rather than disabling reasoning. Adapters may map a band to a
    measured native setting or ignore it. ``max_output_tokens`` is likewise a hint, not a response
    contract.
    """

    max_output_tokens: int | None = None
    reasoning: Literal["low", "medium", "high"] | None = None


@dataclass(frozen=True)
class PromptSection:
    """Provider-neutral size metadata for one named section of a rendered prompt."""

    name: str
    characters: int
    dynamic: bool


@dataclass(frozen=True)
class CompletionRequest:
    """One provider-neutral text completion request.

    The request describes the call and carries transport hints only. It does not own response
    validation, repair, caching policy, or retry; those stay with the reasoning/cycle layer that
    understands the runtime contract. ``sections`` describes already-rendered text for metering —
    it does not ask a provider to cache, reorder, or otherwise reinterpret that text.
    """

    system: str
    user: str
    semantic_label: str
    prompt_version: str
    profile: CompletionProfile | None = None
    sections: tuple[PromptSection, ...] = ()


def _request_metadata(request: CompletionRequest | None) -> dict[str, object]:
    if request is None:
        return {
            "llm_semantic_label": None,
            "llm_prompt_version": None,
            "llm_section_characters": None,
            "llm_dynamic_section_characters": None,
        }
    section_characters = sum(section.characters for section in request.sections)
    dynamic_characters = sum(section.characters for section in request.sections if section.dynamic)
    return {
        "llm_semantic_label": request.semantic_label,
        "llm_prompt_version": request.prompt_version,
        # No declarations means unavailable, not a measured zero-size prompt.
        "llm_section_characters": section_characters if request.sections else None,
        "llm_dynamic_section_characters": dynamic_characters if request.sections else None,
    }


@dataclass(frozen=True)
class LLMUsage:
    """Provider-native token accounting for one round-trip. Surfaced by the *concrete* client
    (``AnthropicLLMClient``) rather than the outer timing decorator, because — unlike wall-clock,
    which any wrapper can measure — token counts live in the provider's response and no wrapper can
    see them.

    ``output_tokens`` *includes* thinking tokens: the API bills deliberation as output and does not
    break it out — and, under adaptive thinking, does **not** return the thinking as countable text
    either (the blocks are summarized/omitted), so counting thinking-block characters reports ~0 no
    matter how much thinking happened. The reliable estimate instead subtracts the *answer* (whose
    text we do see — ``answer_chars``, converted to tokens) from ``output_tokens``: whatever output
    the answer doesn't explain is deliberation. Approximate, but it correctly separates a
    thinking-bound call (a tiny answer behind a large output) from an answer-bound one.

    ``reasoning_tokens`` is the *exact* deliberation count when the provider reports one directly
    (OpenAI ``completion_tokens_details.reasoning_tokens``, Gemini ``thoughts_token_count``). When
    present it supersedes the estimate above; when ``None`` (Anthropic adaptive thinking, which does
    not break thinking out) the estimate stands. Keeping both means one metric reads uniformly
    across providers — exact where available, estimated where not.

    ``input_tokens`` is normalized to the total input processed, including cache reads and writes;
    ``cached_input_tokens`` is the subset served from a cache. It is ``None`` when the provider did
    not report cache usage, distinct from an explicit zero cache hit. OpenAI reports total input as
    one inclusive figure, while Anthropic reports ordinary input, cache writes, and cache reads as
    separate fields, so its adapter sums those fields at the provider boundary."""

    input_tokens: int
    output_tokens: int
    answer_chars: int
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None

    @property
    def answer_tokens(self) -> int:
        return round(self.answer_chars / _CHARS_PER_TOKEN)

    @property
    def thinking_tokens(self) -> int:
        """Deliberation tokens: the provider's exact ``reasoning_tokens`` when it reports one, else
        the estimate — the billed output the returned answer doesn't account for (clamped at 0,
        since the answer-token estimate can slightly exceed a short output)."""
        if self.reasoning_tokens is not None:
            return self.reasoning_tokens
        return max(0, self.output_tokens - self.answer_tokens)

    @property
    def thinking_share(self) -> float:
        """Estimated fraction of ``output_tokens`` spent thinking rather than answering (0.0 when
        the call produced no output at all)."""
        return self.thinking_tokens / self.output_tokens if self.output_tokens else 0.0


def log_llm_usage(
    usage: LLMUsage,
    request: CompletionRequest | None = None,
    *,
    finish_reason: str | None = None,
) -> None:
    """Emit one ``sora.llm`` usage record for a single round-trip. Kept here, not in the concrete
    client, so the *record shape* (event name, field names) stays owned by this instrumentation
    module — the client only supplies the provider-native numbers. Paired with, and distinct from,
    ``MeteredLLMClient``'s timing record: one call emits at most one ``done`` (seconds) and, when
    the client is instrumented, one ``usage`` (tokens). ``LLMMeter`` tallies both."""
    inference_id = current_inference_id.get()
    call_id = _logical_call_id()
    _llm_log.info(
        "~ llm %susage: %d in / %d out tok (~%d answer, ~%.0f%% thinking)",
        _id_tag(call_id, request),
        usage.input_tokens,
        usage.output_tokens,
        usage.answer_tokens,
        usage.thinking_share * 100,
        extra={
            "llm_event": "usage",
            "llm_input_tokens": usage.input_tokens,
            "llm_output_tokens": usage.output_tokens,
            "llm_answer_chars": usage.answer_chars,
            # None on the estimate path (Anthropic); an exact count when the provider reports one.
            "llm_reasoning_tokens": usage.reasoning_tokens,
            "llm_cached_input_tokens": usage.cached_input_tokens,
            "llm_call_id": call_id,
            "llm_inference_id": inference_id,
            "llm_finish_reason": finish_reason,
            **_request_metadata(request),
        },
    )


def log_llm_discarded(inference_id: str) -> None:
    """Emit a ``sora.llm`` record marking an off-cycle inference's result as discarded — interrupted
    or superseded (ADR-0021). The model call ran to completion and was already metered (its cost is
    real, provider-billed), but it did no useful work, so ``LLMMeter`` moves that id's metered cost
    into its wasted bucket. Emitted from Observe, the one place a discard is decided; a no-op for
    the tally if the id was never metered (an uninstrumented run still logs the cue). The ``[id]``
    tag matches the one on this call's usage/timing cues, so the reader can see which lines it
    voids."""
    _llm_log.info(
        "~ llm %sdiscarded (result superseded)",
        _id_tag(inference_id),
        extra={
            "llm_event": "discarded",
            "llm_call_id": inference_id,
            "llm_inference_id": inference_id,
        },
    )


LLMOutcome = Literal["success", "unresolvable", "error"]


def log_llm_outcome(inference_id: str, outcome: LLMOutcome) -> None:
    """Record how a completed inference resolved, independently from whether it became stale."""
    _llm_log.info(
        "~ llm %soutcome: %s",
        _id_tag(inference_id),
        outcome,
        extra={
            "llm_event": "outcome",
            "llm_call_id": inference_id,
            "llm_inference_id": inference_id,
            "llm_outcome": outcome,
        },
    )


def log_llm_late_completion(inference_id: str, outcome: LLMOutcome) -> None:
    """Record a provider result that arrived after its runtime inference stopped being live."""
    _llm_log.info(
        "~ llm %slate completion: %s (result not applied)",
        _id_tag(inference_id),
        outcome,
        extra={
            "llm_event": "late_completion",
            "llm_call_id": inference_id,
            "llm_inference_id": inference_id,
            "llm_late_outcome": outcome,
        },
    )


def log_llm_malformed(*, dropped: int = 0, repaired: int = 0) -> None:
    """Count malformed response fields conservatively dropped or mechanically repaired."""
    if not dropped and not repaired:
        return
    inference_id = current_inference_id.get()
    call_id = _logical_call_id()
    _llm_log.info(
        "~ llm %smalformed: %d dropped / %d repaired",
        _id_tag(call_id),
        dropped,
        repaired,
        extra={
            "llm_event": "malformed",
            "llm_call_id": call_id,
            "llm_inference_id": inference_id,
            "llm_malformed_fields_dropped": dropped,
            "llm_malformed_fields_repaired": repaired,
        },
    )


class LLMClient(Protocol):
    """A single completion round-trip — the runtime's one seam onto a language model.

    Deliberately narrow and wire-format-neutral: a ``CompletionRequest`` in, text out. It commits
    to *no* provider shape — not OpenAI ``chat/completions``, not Anthropic ``messages`` — so the
    reasoning path stays independent of any one SDK, and a concrete adapter is the only place a
    wire format appears. Adapters may ignore unsupported profile hints.

    Non-ownership contract: the request owns call description and transport hints; an
    ``LLMClient`` owns only the round-trip. Validation, repair, caching policy, retry, credential
    refresh, and interrupt handling remain outside the client. The text-to-domain anti-corruption
    boundary therefore stays in procedural memory, never here.
    """

    async def complete(self, request: CompletionRequest) -> str: ...


def _model_name(client: object) -> str | None:
    """A client's own model id, if it names one. Duck-typed on purpose: `LLMClient` is deliberately
    a single method, so this is an optional courtesy a client may offer, never a requirement."""
    model = getattr(client, "model", None)
    return model if isinstance(model, str) else None


class MeteredLLMClient:
    """A transparent ``LLMClient`` decorator that times each round-trip and logs a ``sora.llm`` cue.

    It does *not* violate the ``LLMClient`` non-ownership contract: the contract forbids the
    *client itself* from growing timing/retry responsibilities, keeping every concrete provider
    thin. This wraps one from the outside — an instrumentation layer bootstrap slips in front of the
    real client — so the client stays a bare round-trip while the run gains observability. Each call
    emits one record carrying the elapsed seconds as a structured ``llm_seconds`` field, so a reader
    (`LLMMeter`, the CLI presenter) never has to parse it back out of the message text.
    """

    def __init__(self, inner: LLMClient, *, model: str | None = None) -> None:
        self._inner = inner
        # The model id, purely descriptive — a run surface reads it to record *which* model produced
        # a trace (an unexpected trajectory is a different question for a small local model than for
        # a frontier one). It lives here rather than on `LLMClient` because the Protocol is one
        # method wide by design and a concrete client is free not to name its model at all.
        # Bootstrap passes what config says; falling back to the client's own name matters, because
        # a config that omits `model:` still runs a model — the client's default — and reporting
        # "none" there would be a plain lie rather than a missing detail.
        self.model = model if model is not None else _model_name(inner)

    async def complete(self, request: CompletionRequest) -> str:
        with llm_call_scope() as call_id:
            start = time.perf_counter()
            try:
                return await self._inner.complete(request)
            finally:
                elapsed = time.perf_counter() - start
                inference_id = current_inference_id.get()
                _llm_log.info(
                    "~ llm %s(%.2fs)",
                    _id_tag(call_id, request),
                    elapsed,
                    extra={
                        "llm_event": "done",
                        "llm_seconds": elapsed,
                        "llm_call_id": call_id,
                        "llm_inference_id": inference_id,
                        **_request_metadata(request),
                    },
                )

    async def aclose(self) -> None:
        # Forward lifecycle to the wrapped client if it has any — keeps the decorator drop-in for a
        # client whose teardown someone calls (the Anthropic one holds an HTTP client).
        aclose = getattr(self._inner, "aclose", None)
        if aclose is not None:
            await aclose()


@dataclass(frozen=True)
class LLMInferenceReport:
    call_id: str | None
    inference_id: str | None
    semantic_label: str | None
    prompt_version: str | None
    round_trips: int
    latency_seconds: float
    input_tokens: int
    cached_input_tokens: int
    cache_observed_input_tokens: int
    cache_unknown_input_tokens: int
    output_tokens: int
    section_characters: int | None
    dynamic_section_characters: int | None
    finish_reasons: tuple[str, ...]
    outcome: LLMOutcome | None
    discarded: bool
    malformed_fields_dropped: int
    malformed_fields_repaired: int

    @property
    def dynamic_section_share(self) -> float | None:
        if self.section_characters is None or not self.section_characters:
            return None
        return (self.dynamic_section_characters or 0) / self.section_characters


@dataclass(frozen=True)
class LLMReport:
    calls: int
    usage_calls: int
    latency_seconds: float
    input_tokens: int
    cached_input_tokens: int
    cache_observed_input_tokens: int
    cache_unknown_input_tokens: int
    output_tokens: int
    thinking_tokens: int
    section_characters: int | None
    dynamic_section_characters: int | None
    malformed_fields_dropped: int
    malformed_fields_repaired: int
    inferences: tuple[LLMInferenceReport, ...]


@dataclass(frozen=True)
class _InferenceAccumulator:
    call_id: str | None
    inference_id: str | None
    semantic_label: str | None = None
    prompt_version: str | None = None
    round_trips: int = 0
    latency_seconds: float = 0.0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_observed_input_tokens: int = 0
    cache_unknown_input_tokens: int = 0
    output_tokens: int = 0
    section_characters: int | None = None
    dynamic_section_characters: int | None = None
    finish_reasons: tuple[str, ...] = ()
    outcome: LLMOutcome | None = None
    discarded: bool = False
    malformed_fields_dropped: int = 0
    malformed_fields_repaired: int = 0


class LLMMeter(logging.Handler):
    """Tallies the ``sora.llm`` per-call records the instrumentation emits — call count and summed
    in-model seconds (from ``MeteredLLMClient``), plus token totals, cache hit rate, and thinking
    share (from ``log_llm_usage``, when the concrete client is instrumented) — so a run surface can
    report them at the end without holding a reference to the client (which bootstrap builds and
    hands off).
    Attach it to the ``sora`` logger for the run, then call ``summary()``. Mirrors how the CLI's
    ``_Presenter`` reads the same log stream. The token tally is opt-in: with an uninstrumented
    client no ``usage`` record arrives and ``summary()`` reports timing only, exactly as before."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.calls = 0
        self.total_seconds = 0.0
        self.usage_calls = 0
        self.input_tokens = 0
        # Cache totals retain observation coverage: an absent provider field is unknown, not a
        # measured miss. `cached_input_tokens` sums the known cache reads; the two input buckets
        # make the denominator and its coverage explicit for mixed-provider/compatibility runs.
        self.cached_input_tokens = 0
        self.cache_observed_input_tokens = 0
        self.cache_unknown_input_tokens = 0
        self.output_tokens = 0
        self.answer_chars = 0
        # Summed per-call deliberation tokens: each call contributes its OWN figure — the provider's
        # exact reasoning count when it reports one (OpenAI/Gemini), else that call's answer-
        # subtraction estimate. Summing per call rather than pooling one estimate over the totals is
        # what keeps a mixed-provider run correct: an exact-reporting model in one phase and an
        # estimate-only one (Anthropic) in another both land, each measured its own way.
        self.thinking_tokens = 0
        # The wasted subset of the grand totals above: an inference whose result was discarded
        # (interrupted/superseded) still ran to completion, so its cost is counted in the totals —
        # these break out how much of that cost did no useful work, so a run surface can show used
        # vs. wasted side by side without hiding real (billed) cost.
        self.wasted_calls = 0
        self.wasted_seconds = 0.0
        self.wasted_input_tokens = 0
        self.wasted_output_tokens = 0
        self.section_characters: int | None = None
        self.dynamic_section_characters: int | None = None
        self.malformed_fields_dropped = 0
        self.malformed_fields_repaired = 0
        # Per-inference-id partials, retained so a later `discarded` cue can fold that call's
        # already-metered cost into the wasted buckets (one round-trip per id; popped on discard).
        # An id never discarded lingers here for the run — bounded by call count, negligible.
        self._seconds_by_id: dict[str, float] = {}
        self._calls_by_id: dict[str, int] = {}
        self._tokens_by_id: dict[str, tuple[int, int]] = {}  # id -> (input, output)
        self._inferences: dict[
            tuple[str | None, str | None, str | None, str | None], _InferenceAccumulator
        ] = {}

    def _inference_for(
        self, record: logging.LogRecord
    ) -> tuple[tuple[str | None, str | None, str | None, str | None], _InferenceAccumulator]:
        call_id = getattr(record, "llm_call_id", None)
        inference_id = getattr(record, "llm_inference_id", None)
        semantic_label = getattr(record, "llm_semantic_label", None)
        prompt_version = getattr(record, "llm_prompt_version", None)
        identity = call_id or inference_id
        if identity is not None:
            existing_key = next((key for key in self._inferences if key[0] == identity), None)
            if existing_key is not None:
                accumulator = self._inferences[existing_key]
                resolved_label = semantic_label or accumulator.semantic_label
                resolved_version = prompt_version or accumulator.prompt_version
                key = (
                    identity,
                    inference_id or accumulator.inference_id,
                    resolved_label,
                    resolved_version,
                )
                if key != existing_key:
                    del self._inferences[existing_key]
                    accumulator = replace(
                        accumulator,
                        call_id=call_id or accumulator.call_id,
                        inference_id=inference_id or accumulator.inference_id,
                        semantic_label=resolved_label,
                        prompt_version=resolved_version,
                    )
                    self._inferences[key] = accumulator
                return key, accumulator
        key = (identity, inference_id, semantic_label, prompt_version)
        candidate = self._inferences.get(key)
        if candidate is None:
            candidate = _InferenceAccumulator(
                call_id=call_id,
                inference_id=inference_id,
                semantic_label=semantic_label,
                prompt_version=prompt_version,
            )
            self._inferences[key] = candidate
        return key, candidate

    def _replace_inference(
        self,
        record: logging.LogRecord,
        **changes: object,
    ) -> None:
        key, accumulator = self._inference_for(record)
        self._inferences[key] = replace(accumulator, **cast(Any, changes))

    def emit(self, record: logging.LogRecord) -> None:
        event = getattr(record, "llm_event", None)
        call_id = getattr(record, "llm_call_id", None)
        inference_id = getattr(record, "llm_inference_id", None)
        if event == "done":
            self.calls += 1
            seconds = getattr(record, "llm_seconds", 0.0)
            self.total_seconds += seconds
            section_characters = getattr(record, "llm_section_characters", None)
            dynamic_characters = getattr(record, "llm_dynamic_section_characters", None)
            counted_sections = section_characters if isinstance(section_characters, int) else None
            counted_dynamic = dynamic_characters if isinstance(dynamic_characters, int) else None
            if counted_sections is not None:
                self.section_characters = (self.section_characters or 0) + counted_sections
                self.dynamic_section_characters = (self.dynamic_section_characters or 0) + (
                    counted_dynamic or 0
                )
            _, accumulator = self._inference_for(record)
            self._replace_inference(
                record,
                round_trips=accumulator.round_trips + 1,
                latency_seconds=accumulator.latency_seconds + seconds,
                section_characters=(
                    (accumulator.section_characters or 0) + counted_sections
                    if counted_sections is not None
                    else accumulator.section_characters
                ),
                dynamic_section_characters=(
                    (accumulator.dynamic_section_characters or 0) + (counted_dynamic or 0)
                    if counted_sections is not None
                    else accumulator.dynamic_section_characters
                ),
            )
            if inference_id is not None:
                self._seconds_by_id[inference_id] = (
                    self._seconds_by_id.get(inference_id, 0.0) + seconds
                )
                self._calls_by_id[inference_id] = self._calls_by_id.get(inference_id, 0) + 1
        elif event == "usage":
            self.usage_calls += 1
            input_tokens = getattr(record, "llm_input_tokens", 0)
            cached_input_tokens = getattr(record, "llm_cached_input_tokens", None)
            output_tokens = getattr(record, "llm_output_tokens", 0)
            answer_chars = getattr(record, "llm_answer_chars", 0)
            reasoning_tokens = getattr(record, "llm_reasoning_tokens", None)
            self.input_tokens += input_tokens
            if cached_input_tokens is None:
                self.cache_unknown_input_tokens += input_tokens
            else:
                self.cached_input_tokens += cached_input_tokens
                self.cache_observed_input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.answer_chars += answer_chars
            # Add this call's own deliberation figure — exact when the provider reported one, else
            # its estimate — reusing LLMUsage.thinking_tokens so the exact/estimate choice and the
            # clamp are decided per call. A single exact-reporting call no longer discards the
            # estimated thinking of every other call, which pooling one estimate would have done.
            self.thinking_tokens += LLMUsage(
                input_tokens, output_tokens, answer_chars, reasoning_tokens=reasoning_tokens
            ).thinking_tokens
            finish_reason = getattr(record, "llm_finish_reason", None)
            _, accumulator = self._inference_for(record)
            self._replace_inference(
                record,
                input_tokens=accumulator.input_tokens + input_tokens,
                cached_input_tokens=(accumulator.cached_input_tokens + (cached_input_tokens or 0)),
                cache_observed_input_tokens=(
                    accumulator.cache_observed_input_tokens
                    + (input_tokens if cached_input_tokens is not None else 0)
                ),
                cache_unknown_input_tokens=(
                    accumulator.cache_unknown_input_tokens
                    + (input_tokens if cached_input_tokens is None else 0)
                ),
                output_tokens=accumulator.output_tokens + output_tokens,
                finish_reasons=(
                    accumulator.finish_reasons + (finish_reason,)
                    if isinstance(finish_reason, str)
                    else accumulator.finish_reasons
                ),
            )
            if inference_id is not None:
                previous = self._tokens_by_id.get(inference_id, (0, 0))
                self._tokens_by_id[inference_id] = (
                    previous[0] + input_tokens,
                    previous[1] + output_tokens,
                )
        elif event == "discarded" and inference_id is not None:
            _, accumulator = self._inference_for(record)
            self._replace_inference(record, discarded=True)
            seconds = self._seconds_by_id.pop(inference_id, None)
            if seconds is not None:  # the discarded call was metered (timing always is)
                self.wasted_calls += self._calls_by_id.pop(inference_id, 1)
                self.wasted_seconds += seconds
            tokens = self._tokens_by_id.pop(inference_id, None)
            if tokens is not None:  # ...and, when the client is instrumented, its tokens too
                self.wasted_input_tokens += tokens[0]
                self.wasted_output_tokens += tokens[1]
        elif event == "outcome" and inference_id is not None:
            outcome = getattr(record, "llm_outcome", None)
            if outcome in ("success", "unresolvable", "error"):
                _, accumulator = self._inference_for(record)
                # A watchdog result and the provider result can race under the same id. Whichever
                # result resolved the activity is terminal; a later stale cue must not rewrite it.
                if accumulator.outcome is None:
                    self._replace_inference(record, outcome=outcome)
        elif event == "malformed":
            dropped = getattr(record, "llm_malformed_fields_dropped", 0)
            repaired = getattr(record, "llm_malformed_fields_repaired", 0)
            self.malformed_fields_dropped += dropped
            self.malformed_fields_repaired += repaired
            # A semantic call scope correlates parser recovery even for background judgements,
            # which intentionally have no activity inference id.
            if call_id is not None or inference_id is not None:
                _, accumulator = self._inference_for(record)
                self._replace_inference(
                    record,
                    malformed_fields_dropped=(accumulator.malformed_fields_dropped + dropped),
                    malformed_fields_repaired=(accumulator.malformed_fields_repaired + repaired),
                )

    def report(self) -> LLMReport:
        """Return the machine-readable aggregate and per-inference instrumentation."""
        inferences = tuple(
            LLMInferenceReport(
                call_id=row.call_id,
                inference_id=row.inference_id,
                semantic_label=row.semantic_label,
                prompt_version=row.prompt_version,
                round_trips=row.round_trips,
                latency_seconds=row.latency_seconds,
                input_tokens=row.input_tokens,
                cached_input_tokens=row.cached_input_tokens,
                cache_observed_input_tokens=row.cache_observed_input_tokens,
                cache_unknown_input_tokens=row.cache_unknown_input_tokens,
                output_tokens=row.output_tokens,
                section_characters=row.section_characters,
                dynamic_section_characters=row.dynamic_section_characters,
                finish_reasons=row.finish_reasons,
                outcome=row.outcome,
                discarded=row.discarded,
                malformed_fields_dropped=row.malformed_fields_dropped,
                malformed_fields_repaired=row.malformed_fields_repaired,
            )
            for row in self._inferences.values()
        )
        return LLMReport(
            calls=self.calls,
            usage_calls=self.usage_calls,
            latency_seconds=self.total_seconds,
            input_tokens=self.input_tokens,
            cached_input_tokens=self.cached_input_tokens,
            cache_observed_input_tokens=self.cache_observed_input_tokens,
            cache_unknown_input_tokens=self.cache_unknown_input_tokens,
            output_tokens=self.output_tokens,
            thinking_tokens=self.thinking_tokens,
            section_characters=self.section_characters,
            dynamic_section_characters=self.dynamic_section_characters,
            malformed_fields_dropped=self.malformed_fields_dropped,
            malformed_fields_repaired=self.malformed_fields_repaired,
            inferences=inferences,
        )

    def _pooled(self) -> LLMUsage:
        """This run's pooled input/output/answer totals as one ``LLMUsage`` (0/0 when nothing was
        metered), used only for the answer-token display. Thinking is *not* read off this — see
        ``thinking_share``, which sums each call's own figure so a mixed exact/estimated run stays
        right (pooling one estimate here would drop the estimate for every call once one reported an
        exact count)."""
        return LLMUsage(self.input_tokens, self.output_tokens, self.answer_chars)

    @property
    def thinking_share(self) -> float:
        """Deliberation share of total output across every instrumented call: the summed per-call
        thinking (each call exact where its provider reported one, estimated where not) over pooled
        output (0.0 when nothing was metered)."""
        return self.thinking_tokens / self.output_tokens if self.output_tokens else 0.0

    @property
    def cache_hit_rate(self) -> float | None:
        """Share of cache-observed input served from provider prompt caches, or ``None`` when no
        input carried cache accounting."""
        if not self.cache_observed_input_tokens:
            return None
        return self.cached_input_tokens / self.cache_observed_input_tokens

    @property
    def cache_coverage(self) -> float:
        """Share of total input for which the provider reported cache usage."""
        return self.cache_observed_input_tokens / self.input_tokens if self.input_tokens else 0.0

    def summary(self, wall_seconds: float | None = None) -> str:
        plural = "" if self.calls == 1 else "s"
        text = f"{self.calls} LLM call{plural}"
        if self.wasted_calls:  # break out used vs. discarded only when something was discarded
            text += f" ({self.calls - self.wasted_calls} used, {self.wasted_calls} discarded)"
        text += f", {self.total_seconds:.1f}s in-model"
        if self.wasted_seconds:
            used_seconds = self.total_seconds - self.wasted_seconds
            text += f" ({used_seconds:.1f}s used, {self.wasted_seconds:.1f}s wasted)"
        if wall_seconds is not None:
            text += f", {wall_seconds:.1f}s wall"
        if self.usage_calls:
            pooled = self._pooled()
            text += (
                f"; {self.input_tokens} in / {self.output_tokens} out tokens "
                f"(~{pooled.answer_tokens} answer, ~{self.thinking_share * 100:.0f}% thinking)"
            )
            cache_hit_rate = self.cache_hit_rate
            if cache_hit_rate is None:
                text += "; cache usage unavailable"
            else:
                text += (
                    f"; {self.cached_input_tokens} cached, {cache_hit_rate * 100:.0f}% hit "
                    f"({self.cache_coverage * 100:.0f}% cache coverage)"
                )
            if self.wasted_output_tokens:
                text += f"; {self.wasted_output_tokens} out discarded"
        return text
