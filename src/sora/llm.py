"""Provider-agnostic LLM access for the reasoning path."""

from __future__ import annotations

import contextvars
import logging
import time
from dataclasses import dataclass
from typing import Protocol

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


def _id_tag(inference_id: str | None) -> str:
    """A short, eyeballable inference-id prefix for the per-call cues (``[32a3ad56] ``), so a later
    ``discarded`` cue is visibly the *same* call as its usage/timing lines — which is otherwise
    unrecoverable, since the discard is only known a cycle after those lines already printed. Empty
    for a call not driven by an off-cycle inference (id ``None``), leaving those lines unchanged."""
    return f"[{inference_id[:8]}] " if inference_id else ""


# A response's answer text averages ~this many characters per token (English/JSON prose). Only used
# to turn the *measured* answer length into an estimated answer-token count, so the thinking
# estimate below can subtract it from output_tokens. Rough on purpose — the discriminating signal
# (a thinking-bound call runs ~90%+, an answer-bound one near 0) survives any sane value of it.
_CHARS_PER_TOKEN = 4


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
    across providers — exact where available, estimated where not."""

    input_tokens: int
    output_tokens: int
    answer_chars: int
    reasoning_tokens: int | None = None

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


def log_llm_usage(usage: LLMUsage) -> None:
    """Emit one ``sora.llm`` usage record for a single round-trip. Kept here, not in the concrete
    client, so the *record shape* (event name, field names) stays owned by this instrumentation
    module — the client only supplies the provider-native numbers. Paired with, and distinct from,
    ``MeteredLLMClient``'s timing record: one call emits at most one ``done`` (seconds) and, when
    the client is instrumented, one ``usage`` (tokens). ``LLMMeter`` tallies both."""
    inference_id = current_inference_id.get()
    _llm_log.info(
        "~ llm %susage: %d in / %d out tok (~%d answer, ~%.0f%% thinking)",
        _id_tag(inference_id),
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
            "llm_inference_id": inference_id,
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
        extra={"llm_event": "discarded", "llm_inference_id": inference_id},
    )


class LLMClient(Protocol):
    """A single completion round-trip — the runtime's one seam onto a language model.

    Deliberately narrow and wire-format-neutral: a system instruction plus a prompt in, text out.
    It commits to *no* provider shape — not OpenAI ``chat/completions``, not Anthropic ``messages``
    — so the reasoning path (``ProceduralMemory.infer``) stays independent of any one SDK, and the
    concrete client (an optional extra under ``sora/adapters/``) is the only place a wire format
    appears. The canonical format the runtime converts *to* is its own domain (``Plan``/``Step``),
    not a borrowed message schema; that conversion (the anti-corruption boundary) lives in
    ``infer``, never here.

    Non-ownership contract: an ``LLMClient`` owns *only* the round-trip. Retries, streaming,
    credential refresh, prompt caching, and interrupt handling belong to the cycle/agent, never to
    the client. Keeping that boundary explicit is what lets a second provider slot in without
    touching the decision cycle.
    """

    async def complete(self, *, system: str, prompt: str) -> str: ...


class MeteredLLMClient:
    """A transparent ``LLMClient`` decorator that times each round-trip and logs a ``sora.llm`` cue.

    It does *not* violate the ``LLMClient`` non-ownership contract: the contract forbids the
    *client itself* from growing timing/retry responsibilities, keeping every concrete provider
    thin. This wraps one from the outside — an instrumentation layer bootstrap slips in front of the
    real client — so the client stays a bare round-trip while the run gains observability. Each call
    emits one record carrying the elapsed seconds as a structured ``llm_seconds`` field, so a reader
    (`LLMMeter`, the CLI presenter) never has to parse it back out of the message text.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    async def complete(self, *, system: str, prompt: str) -> str:
        start = time.perf_counter()
        try:
            return await self._inner.complete(system=system, prompt=prompt)
        finally:
            elapsed = time.perf_counter() - start
            inference_id = current_inference_id.get()
            _llm_log.info(
                "~ llm %s(%.2fs)",
                _id_tag(inference_id),
                elapsed,
                extra={
                    "llm_event": "done",
                    "llm_seconds": elapsed,
                    "llm_inference_id": inference_id,
                },
            )

    async def aclose(self) -> None:
        # Forward lifecycle to the wrapped client if it has any — keeps the decorator drop-in for a
        # client whose teardown someone calls (the Anthropic one holds an HTTP client).
        aclose = getattr(self._inner, "aclose", None)
        if aclose is not None:
            await aclose()


class LLMMeter(logging.Handler):
    """Tallies the ``sora.llm`` per-call records the instrumentation emits — call count and summed
    in-model seconds (from ``MeteredLLMClient``), plus token totals and thinking share (from
    ``log_llm_usage``, when the concrete client is instrumented) — so a run surface can report them
    at the end without holding a reference to the client (which bootstrap builds and hands off).
    Attach it to the ``sora`` logger for the run, then call ``summary()``. Mirrors how the CLI's
    ``_Presenter`` reads the same log stream. The token tally is opt-in: with an uninstrumented
    client no ``usage`` record arrives and ``summary()`` reports timing only, exactly as before."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.calls = 0
        self.total_seconds = 0.0
        self.usage_calls = 0
        self.input_tokens = 0
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
        # Per-inference-id partials, retained so a later `discarded` cue can fold that call's
        # already-metered cost into the wasted buckets (one round-trip per id; popped on discard).
        # An id never discarded lingers here for the run — bounded by call count, negligible.
        self._seconds_by_id: dict[str, float] = {}
        self._tokens_by_id: dict[str, tuple[int, int]] = {}  # id -> (input, output)

    def emit(self, record: logging.LogRecord) -> None:
        event = getattr(record, "llm_event", None)
        inference_id = getattr(record, "llm_inference_id", None)
        if event == "done":
            self.calls += 1
            seconds = getattr(record, "llm_seconds", 0.0)
            self.total_seconds += seconds
            if inference_id is not None:
                self._seconds_by_id[inference_id] = seconds
        elif event == "usage":
            self.usage_calls += 1
            input_tokens = getattr(record, "llm_input_tokens", 0)
            output_tokens = getattr(record, "llm_output_tokens", 0)
            answer_chars = getattr(record, "llm_answer_chars", 0)
            reasoning_tokens = getattr(record, "llm_reasoning_tokens", None)
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.answer_chars += answer_chars
            # Add this call's own deliberation figure — exact when the provider reported one, else
            # its estimate — reusing LLMUsage.thinking_tokens so the exact/estimate choice and the
            # clamp are decided per call. A single exact-reporting call no longer discards the
            # estimated thinking of every other call, which pooling one estimate would have done.
            self.thinking_tokens += LLMUsage(
                input_tokens, output_tokens, answer_chars, reasoning_tokens=reasoning_tokens
            ).thinking_tokens
            if inference_id is not None:
                self._tokens_by_id[inference_id] = (input_tokens, output_tokens)
        elif event == "discarded" and inference_id is not None:
            seconds = self._seconds_by_id.pop(inference_id, None)
            if seconds is not None:  # the discarded call was metered (timing always is)
                self.wasted_calls += 1
                self.wasted_seconds += seconds
            tokens = self._tokens_by_id.pop(inference_id, None)
            if tokens is not None:  # ...and, when the client is instrumented, its tokens too
                self.wasted_input_tokens += tokens[0]
                self.wasted_output_tokens += tokens[1]

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
            if self.wasted_output_tokens:
                text += f"; {self.wasted_output_tokens} out discarded"
        return text
