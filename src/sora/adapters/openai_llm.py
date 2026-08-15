"""OpenAI-compatible ``LLMClient`` — the OpenAI SDK pointed at any OpenAI-compatible endpoint.

One client covers three provider families through ``base_url``/``model`` config alone, so a new
provider is a config change, not a new adapter:

* OpenAI itself (no ``base_url``),
* Gemini's OpenAI-compatible surface (``base_url`` = the ``.../v1beta/openai/`` endpoint),
* local runtimes — Ollama, LM Studio, vLLM, llama.cpp — which all serve ``/v1/chat/completions``.

Lives under ``adapters/`` because it is a concrete integration on an optional extra
(``pip install sora-runtime[openai]``); the core never imports it — only ``bootstrap`` / an
application wires it in, keeping the provider SDK out of the dependency-free core. Native Gemini
(the ``google-genai`` SDK) is deliberately *not* here: its edge is multimodal / thinking-config,
which the string-in/string-out seam has no use for yet — it belongs with multimodal Observe.
"""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI

from sora.llm import LLMUsage, log_llm_usage

# Only a fallback: the model id is a configuration value (a ctor arg, wired from agent.yaml), never
# baked in — swapping models/providers must not require a code change.
DEFAULT_MODEL = "gpt-5.1"


class OpenAICompatLLMClient:
    """A single-round-trip ``LLMClient`` over the OpenAI SDK (``AsyncOpenAI``), usable against any
    OpenAI-compatible endpoint.

    Owns only the completion call (see the ``LLMClient`` non-ownership contract). ``model`` and
    ``base_url`` are config values — passed in — so OpenAI, Gemini's compat endpoint, and a local
    runtime differ only by configuration. The response's message content is returned as the plain
    string the reasoning path parses.

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
    ) -> None:
        # api_key/base_url=None let the SDK resolve them from the environment (OPENAI_API_KEY /
        # OPENAI_BASE_URL) — only pass them to override. max_tokens defaults to unset: reasoning
        # models reject a cap and every target runs fine without one; when set it is sent as
        # `max_completion_tokens` (the current field; the legacy `max_tokens` is rejected by
        # reasoning models).
        client_kwargs: dict[str, Any] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        elif base_url is not None and not os.environ.get("OPENAI_API_KEY"):
            # A custom endpoint (a local runtime — Ollama/LM Studio/vLLM/llama.cpp) generally
            # ignores the key, but AsyncOpenAI still refuses to construct without a non-empty one.
            # Supply a placeholder so pointing `base_url` at a local runtime is config-only, with no
            # dummy `api_key:` required. OpenAI proper (no base_url) still needs a real key, and an
            # endpoint that *does* require auth is served by setting api_key/OPENAI_API_KEY.
            client_kwargs["api_key"] = "not-needed"
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**client_kwargs)
        self._model = model
        self._max_tokens = max_tokens
        self._instrument = instrument

    async def complete(self, *, system: str, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if self._max_tokens is not None:
            kwargs["max_completion_tokens"] = self._max_tokens
        response = await self._client.chat.completions.create(**kwargs)
        text = _text_of(response)
        if self._instrument:
            log_llm_usage(_usage_of(response, answer_chars=len(text)))
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
    already-measured answer length. Reads the exact ``reasoning_tokens`` when the provider reports
    one (o-series / Gemini thinking models), else leaves it ``None`` so the meter estimates.
    Tolerant of a missing/partial ``usage`` (getattr + ``or 0``) so instrumentation never breaks —
    a metering gap degrades to zeros, never raises."""
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None)
    return LLMUsage(
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        answer_chars=answer_chars,
        reasoning_tokens=int(reasoning) if reasoning is not None else None,
    )
