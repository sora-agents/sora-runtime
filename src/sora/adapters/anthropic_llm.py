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

from anthropic import AsyncAnthropic

# Only a fallback: the model id is a configuration value (a ctor arg, wired from agent.yaml), never
# baked in — swapping Opus/Sonnet/versions must not require a code change.
DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicLLMClient:
    """A single-round-trip ``LLMClient`` over the official Anthropic SDK (``AsyncAnthropic``).

    Owns only the completion call (see the ``LLMClient`` non-ownership contract). ``model`` is a
    config value — passed in, defaulting to ``DEFAULT_MODEL`` only as a fallback. Adaptive thinking
    is enabled explicitly (Opus runs without it otherwise); the response's text blocks are joined
    into the plain string the reasoning path parses.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 8192,
    ) -> None:
        # api_key=None lets the SDK resolve credentials from the environment / an `ant` profile.
        self._client = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()
        self._model = model
        self._max_tokens = max_tokens

    async def complete(self, *, system: str, prompt: str) -> str:
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        # Join text blocks (skip thinking blocks); getattr keeps this robust to the content-block
        # union under strict typing without depending on the SDK's block class names.
        parts: list[str] = []
        for block in message.content:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        return "".join(parts)

    async def aclose(self) -> None:
        """Release the underlying HTTP client. Optional — the cycle/agent owns lifecycle."""
        await self._client.close()
