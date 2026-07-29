"""Provider-agnostic LLM access for the reasoning path."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

# A dedicated child of the `sora` tree so instrumentation records are addressable on their own —
# the CLI presenter surfaces them as a per-call cue, `LLMMeter` tallies them, and neither has to
# reach into the reasoning path to do it.
_llm_log = logging.getLogger("sora.llm")


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
    thinking-bound call (a tiny answer behind a large output) from an answer-bound one."""

    input_tokens: int
    output_tokens: int
    answer_chars: int

    @property
    def answer_tokens(self) -> int:
        return round(self.answer_chars / _CHARS_PER_TOKEN)

    @property
    def thinking_tokens(self) -> int:
        """Estimated deliberation tokens: the billed output the returned answer doesn't account for
        (clamped at 0, since the answer-token estimate can slightly exceed a short output)."""
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
    _llm_log.info(
        "~ llm usage: %d in / %d out tok (~%d answer, ~%.0f%% thinking)",
        usage.input_tokens,
        usage.output_tokens,
        usage.answer_tokens,
        usage.thinking_share * 100,
        extra={
            "llm_event": "usage",
            "llm_input_tokens": usage.input_tokens,
            "llm_output_tokens": usage.output_tokens,
            "llm_answer_chars": usage.answer_chars,
        },
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
            _llm_log.info(
                "~ llm (%.2fs)", elapsed, extra={"llm_event": "done", "llm_seconds": elapsed}
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

    def emit(self, record: logging.LogRecord) -> None:
        event = getattr(record, "llm_event", None)
        if event == "done":
            self.calls += 1
            self.total_seconds += getattr(record, "llm_seconds", 0.0)
        elif event == "usage":
            self.usage_calls += 1
            self.input_tokens += getattr(record, "llm_input_tokens", 0)
            self.output_tokens += getattr(record, "llm_output_tokens", 0)
            self.answer_chars += getattr(record, "llm_answer_chars", 0)

    def _pooled(self) -> LLMUsage:
        """This run's usage as one ``LLMUsage`` over the pooled totals, so the answer/thinking
        estimates are the per-call ones applied to the sums (0.0/0 when nothing was metered)."""
        return LLMUsage(self.input_tokens, self.output_tokens, self.answer_chars)

    @property
    def thinking_share(self) -> float:
        """Estimated deliberation share of total output tokens across every instrumented call — the
        pooled counterpart to ``LLMUsage.thinking_share`` (0.0 when nothing was metered)."""
        return self._pooled().thinking_share

    def summary(self, wall_seconds: float | None = None) -> str:
        plural = "" if self.calls == 1 else "s"
        text = f"{self.calls} LLM call{plural}, {self.total_seconds:.1f}s in-model"
        if wall_seconds is not None:
            text += f", {wall_seconds:.1f}s wall"
        if self.usage_calls:
            pooled = self._pooled()
            text += (
                f"; {self.input_tokens} in / {self.output_tokens} out tokens "
                f"(~{pooled.answer_tokens} answer, ~{pooled.thinking_share * 100:.0f}% thinking)"
            )
        return text
