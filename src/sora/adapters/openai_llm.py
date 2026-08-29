"""OpenAI-compatible ``LLMClient`` — the OpenAI SDK pointed at any OpenAI-compatible endpoint.

One client covers three provider families through ``base_url``/``model`` config alone, so a new
provider is a config change, not a new adapter:

* OpenAI itself (no ``base_url``),
* hosted gateways and Gemini's OpenAI-compatible surface (selected with ``base_url``),
* local runtimes — Ollama, LM Studio, vLLM, llama.cpp — which all serve ``/v1/chat/completions``.

Lives under ``adapters/`` because it is a concrete integration on an optional extra
(``pip install sora-runtime[openai]``); the core never imports it — only ``bootstrap`` / an
application wires it in, keeping the provider SDK out of the dependency-free core. Native Gemini
(the ``google-genai`` SDK) is deliberately *not* here: its edge is multimodal / thinking-config,
which the string-in/string-out seam has no use for yet — it belongs with multimodal Observe.
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI, Timeout

from sora.llm import LLMUsage, log_llm_usage

# Only a fallback: the model id is a configuration value (a ctor arg, wired from agent.yaml), never
# baked in — swapping models/providers must not require a code change.
DEFAULT_MODEL = "gpt-5.1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

# Seconds of SILENCE tolerated on a streaming completion, not a cap on how long it may take: with
# `stream=True` the SDK's read timeout applies between chunks, so a legitimately long call keeps
# resetting it while a stalled connection does not. That distinction is why streaming has to come
# first — on a single-shot call the same setting would be a duration cap, and capping duration is
# what makes a thinking model's slow-but-healthy answer indistinguishable from a dead socket.
#
# The SDK's own default is 600s read with two retries, so a stalled request costs up to ~30 minutes
# of silence. Observed: a first plan inference that took ~14 minutes to return a 3,853-token
# completion, spending a whole benchmark scenario's real-time budget before the agent had a plan.
#
# The one gap in the "silence, not duration" reading is the FIRST chunk: nothing has streamed yet,
# so this does bound time-to-first-token, and two targets can legitimately exceed it there — a
# reasoning model that stays quiet while it reasons, and a local runtime doing prompt eval over a
# long context. Both are configuration, not code: raise `stall_timeout` (or pass None to disable it)
# for such a target. It is left at 90s because the shipped configs do not hit it, and because the
# cost of raising it is paid on the failure this exists to bound — a genuinely dead socket holds the
# activity for the whole value, against a Gaia2 scenario's ~1000s real-time budget for everything.
DEFAULT_STREAM_STALL_TIMEOUT = 90.0
DEFAULT_CONNECT_TIMEOUT = 10.0


class OpenAICompatLLMClient:
    """A single-round-trip ``LLMClient`` over the OpenAI SDK (``AsyncOpenAI``), usable against any
    OpenAI-compatible endpoint.

    Owns only the completion call (see the ``LLMClient`` non-ownership contract). ``model`` and
    ``base_url`` are config values — passed in — so OpenAI, Gemini's compat endpoint, and a local
    runtime differ only by configuration. The response's message content is returned as the plain
    string the reasoning path parses.

    The call streams by default and carries an explicit inter-chunk timeout, so a connection that
    goes quiet surfaces as an error in tens of seconds instead of consuming the SDK's 600s read
    timeout twice over. Streaming is what makes that timeout mean "stalled" rather than "slow" —
    see ``DEFAULT_STREAM_STALL_TIMEOUT``. Retries stay the SDK's to own (the ``LLMClient``
    non-ownership contract); this only bounds how long each attempt may say nothing.

    ``instrument`` (default off, wired from the ``llm:`` config block) opts a run into per-call
    token accounting: the client emits a ``sora.llm`` usage record (via ``log_llm_usage``). Where
    the provider reports reasoning tokens (OpenAI ``completion_tokens_details.reasoning_tokens``,
    Gemini via its compat surface), that exact count rides through as ``LLMUsage.reasoning_tokens``;
    otherwise the meter falls back to its answer-subtraction estimate. As with the Anthropic client,
    this surfacing lives here because token counts exist only in the response object.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
        instrument: bool = False,
        stream: bool = True,
        stall_timeout: float | None = DEFAULT_STREAM_STALL_TIMEOUT,
    ) -> None:
        # With no explicit base_url, api_key=None lets the SDK resolve OPENAI_API_KEY while an
        # explicit official URL prevents OPENAI_BASE_URL from redirecting that credential. A
        # configured base_url is a separate trust boundary: it never inherits OPENAI_API_KEY and
        # receives either an explicit key or a placeholder.
        # max_tokens defaults to unset: reasoning models reject a cap; when set it is sent as
        # `max_completion_tokens` (the current field; the legacy `max_tokens` is rejected by
        # reasoning models).
        client_kwargs: dict[str, Any] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        elif base_url is not None:
            # A custom endpoint (a local runtime — Ollama/LM Studio/vLLM/llama.cpp) generally
            # ignores the key, but AsyncOpenAI still refuses to construct without a non-empty one.
            # Supply a placeholder so pointing `base_url` at a local runtime is config-only, with no
            # dummy `api_key:` required. An authenticated custom endpoint receives its selected
            # environment credential from bootstrap as the explicit api_key argument.
            client_kwargs["api_key"] = "not-needed"
        client_kwargs["base_url"] = base_url if base_url is not None else DEFAULT_OPENAI_BASE_URL
        if stall_timeout is not None:
            # Explicit per-phase timeout rather than a scalar: a scalar would apply the same budget
            # to connect, and a slow TLS handshake is not the failure being bounded here.
            client_kwargs["timeout"] = Timeout(
                stall_timeout,
                connect=DEFAULT_CONNECT_TIMEOUT,
            )
        self._client = AsyncOpenAI(**client_kwargs)
        self.model = model  # public: bootstrap's metered wrapper reports it in the run trace
        self._max_tokens = max_tokens
        self._instrument = instrument
        # Streaming is the default because it is what makes the timeout above a stall detector.
        # `stream: false` is the escape hatch for an OpenAI-compatible endpoint whose streaming is
        # broken or absent (some local runtimes); on that path the timeout reverts to a duration
        # cap, so a target that needs it usually wants `stall_timeout: null` as well.
        self._stream = stream

    async def complete(self, *, system: str, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if self._max_tokens is not None:
            kwargs["max_completion_tokens"] = self._max_tokens
        if not self._stream:
            response = await self._client.chat.completions.create(**kwargs)
            text = _text_of(response)
            if self._instrument:
                log_llm_usage(_usage_of(response, answer_chars=len(text)))
            return text
        if self._instrument:
            # Only asked for when it will be used: a chunked response carries no usage block unless
            # this is set, and an OpenAI-compatible server that does not know the field would reject
            # the request outright. Uninstrumented runs never send it.
            kwargs["stream_options"] = {"include_usage": True}
        parts: list[str] = []
        usage_chunk: Any = None
        stream = await self._client.chat.completions.create(**kwargs, stream=True)
        # `async with`, not a bare `async for`: the stall timeout above raises *mid-iteration* —
        # that is the path it exists for — and an unclosed stream leaves the httpx response open,
        # so its connection never returns to the pool. On a long run against a flaky endpoint that
        # leaks until the pool is exhausted, and since `Timeout(stall_timeout, ...)` sets the pool
        # budget to the same value, healthy calls then start timing out waiting for a connection.
        async with stream:
            async for chunk in stream:
                # The usage block rides a final chunk that carries no choices, so both are collected
                # independently rather than assuming they arrive together.
                if getattr(chunk, "usage", None) is not None:
                    usage_chunk = chunk
                for choice in getattr(chunk, "choices", None) or []:
                    content = getattr(getattr(choice, "delta", None), "content", None)
                    if content:
                        parts.append(content)
        text = "".join(parts)
        if self._instrument:
            log_llm_usage(_usage_of(usage_chunk, answer_chars=len(text)))
        return text

    async def aclose(self) -> None:
        """Release the underlying HTTP client. Optional — the cycle/agent owns lifecycle."""
        await self._client.close()


def _text_of(response: Any) -> str:
    """Join/return the assistant message text from a chat-completions response. Tolerant of an empty
    ``choices``, a missing ``message`` (some local OpenAI-compatible servers omit it on certain
    finish reasons), or a null ``content`` (a refusal or tool-only turn), so a text-less turn
    degrades to ``""`` rather than raising."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    return content or ""


def _usage_of(response: Any, *, answer_chars: int) -> LLMUsage:
    """Build a provider-neutral ``LLMUsage`` from a chat-completions ``usage`` block and the
    already-measured answer length. Reads the exact ``reasoning_tokens`` and cached-input subset
    when the provider reports them; ``prompt_tokens`` already includes that cached subset.
    Tolerant of a missing/partial ``usage`` (getattr + ``or 0``) so instrumentation never breaks —
    a metering gap degrades to zeros, never raises."""
    usage = getattr(response, "usage", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(completion_details, "reasoning_tokens", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cached_input = getattr(prompt_details, "cached_tokens", None)
    return LLMUsage(
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        answer_chars=answer_chars,
        reasoning_tokens=int(reasoning) if reasoning is not None else None,
        cached_input_tokens=int(cached_input) if cached_input is not None else None,
    )
