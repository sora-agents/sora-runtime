"""Anthropic-backed default ``LLMClient`` — the shipped concrete implementation of the reasoning
seam (``sora.llm.LLMClient``).

Lives under ``adapters/`` because it is a concrete integration with an external ecosystem (the same
place the MCP adapters live) and depends on an optional extra: ``pip install sora-runtime[llm]``.
The core never imports it — only ``bootstrap``/an application wires it in — so the provider SDK
stays out of the dependency-free core. Keeping it a thin, flat client is deliberate: the wider
provider decoupling (declarative profiles x behavioral transports over a normalized response) is
deferred until a second provider or an LLM-based Reason phase needs it.
"""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic

from sora.llm import LLMUsage, log_llm_usage

# Only a fallback: the model id is a configuration value (a ctor arg, wired from agent.yaml), never
# baked in — swapping Opus/Sonnet/versions must not require a code change.
DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicLLMClient:
    """A single-round-trip ``LLMClient`` over the official Anthropic SDK (``AsyncAnthropic``).

    Owns only the completion call (see the ``LLMClient`` non-ownership contract). ``model`` is a
    config value — passed in, defaulting to ``DEFAULT_MODEL`` only as a fallback. Adaptive thinking
    is enabled explicitly (Opus runs without it otherwise); the response's text blocks are joined
    into the plain string the reasoning path parses.

    ``instrument`` (default off, wired from the ``llm:`` config block) opts a run into per-call
    token accounting: the client emits a ``sora.llm`` usage record (via ``log_llm_usage``) alongside
    ``MeteredLLMClient``'s timing record. This *does* live in the concrete client rather than an
    outer wrapper, but it is not the timing/retry ownership the contract forbids: token counts and
    the thinking/answer split are provider-native — they exist only in the response object here, and
    no transparent decorator could observe them. The client still merely *surfaces* the datum (it
    neither tallies nor formats a run summary); with the flag off it stays a bare round-trip.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 8192,
        instrument: bool = False,
    ) -> None:
        # api_key=None lets the SDK resolve credentials from the environment / an `ant` profile.
        self._client = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()
        self._model = model
        self._max_tokens = max_tokens
        self._instrument = instrument

    async def complete(self, *, system: str, prompt: str) -> str:
        # Stream rather than a single create(): the SDK refuses a non-streaming call whose
        # worst-case duration (estimated from max_tokens) could exceed 10 minutes, which a large
        # plan-inference budget behind adaptive thinking crosses. Streaming lifts that ceiling;
        # get_final_message() consumes the whole stream and returns the same assembled Message.
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            message = await stream.get_final_message()
        # Join the answer's text blocks (skip thinking blocks); getattr keeps this robust to the
        # content-block union under strict typing without depending on the SDK's block class names.
        parts: list[str] = []
        for block in message.content:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        text = "".join(parts)
        if self._instrument:
            # answer_chars is the returned answer's length — the thinking estimate subtracts it from
            # output_tokens. Thinking-block *text* is not counted: adaptive thinking doesn't return
            # it, so it can't be measured directly (see LLMUsage).
            log_llm_usage(_usage_of(message, answer_chars=len(text)))
        return text

    async def aclose(self) -> None:
        """Release the underlying HTTP client. Optional — the cycle/agent owns lifecycle."""
        await self._client.close()


def _usage_of(message: Any, *, answer_chars: int) -> LLMUsage:
    """Build a provider-neutral ``LLMUsage`` from an Anthropic message's ``usage`` block and the
    already-measured answer length. Tolerant of a missing/partial ``usage`` (getattr + ``or 0``) so
    instrumentation can never break a call — a metering gap degrades to zeros, never raises."""
    usage = getattr(message, "usage", None)
    return LLMUsage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        answer_chars=answer_chars,
    )
