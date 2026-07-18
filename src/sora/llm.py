"""Provider-agnostic LLM access for the reasoning path."""

from __future__ import annotations

from typing import Protocol


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
